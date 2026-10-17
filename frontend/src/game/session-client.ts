import { flyMotion } from './fly-motion.ts'
import type { SaberAction } from './fly-motion.ts'

export interface Score {
  score: number
  combo: number
  max_combo: number
  hits: number
  misses: number
  accuracy: number
}
export type SessionStatus =
  'idle' | 'connecting' | 'ready' | 'playing' | 'paused' | 'ended' | 'error'
export interface SessionSnapshot {
  status: SessionStatus
  error: string
  score: Score | null
  sabers: SaberAction[]
  removed: ReadonlySet<number>
  lastEvent: { kind: string; note_id: number } | null
}
export const emptySession: SessionSnapshot = {
  status: 'idle',
  error: '',
  score: null,
  sabers: [],
  removed: new Set(),
  lastEvent: null,
}

interface ClientOptions {
  url: () => string
  getTime: () => number
  onChange: (state: Partial<SessionSnapshot>) => void
  onFailure: () => void
  makeSocket?: (url: string) => WebSocket
}

function validScore(value: unknown): value is Score {
  if (!value || typeof value !== 'object') return false
  const s = value as Score
  return (
    ['score', 'combo', 'max_combo', 'hits', 'misses'].every((k) => {
      const n = s[k as keyof Score]
      return Number.isInteger(n) && n >= 0
    }) &&
    s.score <= 100 &&
    Number.isFinite(s.accuracy) &&
    s.accuracy >= 0 &&
    s.accuracy <= 1
  )
}

/** Owns one single-player socket; audio remains the source of song time. */
export class SessionClient {
  private socket: WebSocket | null = null
  private key = ''
  private status: SessionStatus = 'idle'
  private clock: ReturnType<typeof setInterval> | undefined
  private deadline: ReturnType<typeof setTimeout> | undefined
  private rejectReady: ((error: Error) => void) | undefined
  private removed = new Set<number>()
  private started = false
  private readonly options: ClientOptions

  constructor(options: ClientOptions) {
    this.options = options
  }

  private update(state: Partial<SessionSnapshot>) {
    if (state.status) this.status = state.status
    this.options.onChange(state)
  }
  private send(message: object) {
    if (this.socket?.readyState === 1) this.socket.send(JSON.stringify(message))
  }
  private stopClock() {
    clearInterval(this.clock)
    this.clock = undefined
  }
  private clearRun() {
    this.removed = new Set()
    flyMotion.reset()
    this.update({ score: null, sabers: [], removed: this.removed, lastEvent: null })
  }
  disconnect() {
    const socket = this.socket
    this.socket = null
    this.key = ''
    this.started = false
    this.stopClock()
    clearTimeout(this.deadline)
    this.rejectReady?.(new Error('Session cancelled.'))
    this.rejectReady = undefined
    socket?.close()
    flyMotion.reset()
    this.update({ ...emptySession })
  }
  private fail(message: string) {
    this.rejectReady?.(new Error(message))
    this.rejectReady = undefined
    this.disconnect()
    this.update({ status: 'error', error: message })
    this.options.onFailure()
  }

  async prepare(mapId: string, brain: string) {
    if (this.key === `${mapId}/${brain}` && ['ready', 'playing', 'paused'].includes(this.status))
      return
    this.disconnect()
    this.key = `${mapId}/${brain}`
    this.clearRun()
    this.update({ status: 'connecting', error: '' })
    await new Promise<void>((resolve, reject) => {
      this.rejectReady = reject
      let socket: WebSocket
      try {
        socket = (this.options.makeSocket ?? ((url) => new WebSocket(url)))(this.options.url())
      } catch {
        this.fail('Could not open a live session. Check the backend and press play to retry.')
        return
      }
      this.socket = socket
      this.deadline = setTimeout(() => {
        if (this.socket === socket) this.fail('The live session timed out. Press play to retry.')
      }, 15000)
      socket.addEventListener('open', () => {
        if (this.socket === socket)
          this.send({ type: 'start', game: 'saber', brains: [brain], map_id: mapId, seed: 0 })
      })
      socket.addEventListener('message', (event) => {
        if (this.socket !== socket || typeof event.data !== 'string') return
        try {
          const message = JSON.parse(event.data)
          if (!message || typeof message !== 'object') return
          if (message.type === 'error' && typeof message.message === 'string') {
            this.fail(`Live session: ${message.message}. Press play to retry.`)
          } else if (message.type === 'ready' && this.status === 'connecting') {
            if (
              message.map_id !== mapId ||
              !Array.isArray(message.slots) ||
              message.slots.length !== 1 ||
              message.slots[0].slot !== 0 ||
              message.slots[0].brain !== brain
            ) {
              this.fail('The server prepared a different song or brain. Press play to retry.')
              return
            }
            clearTimeout(this.deadline)
            this.rejectReady = undefined
            this.update({ status: 'ready' })
            resolve()
          } else if (message.slot === 0 && this.status === 'playing') {
            if (
              message.type === 'frame' &&
              validScore(message.render_state?.score) &&
              flyMotion.acceptMessage(message)
            ) {
              this.update({
                score: message.render_state.score,
                sabers: message.render_state.sabers,
              })
            } else if (
              message.type === 'event' &&
              ['hit', 'miss', 'bad_cut', 'bomb'].includes(message.kind) &&
              Number.isInteger(message.note_id) &&
              message.note_id >= 0
            ) {
              this.removed.add(message.note_id)
              this.update({
                removed: this.removed,
                lastEvent: { kind: message.kind, note_id: message.note_id },
              })
            } else if (message.type === 'ended' && validScore(message.score)) {
              this.stopClock()
              flyMotion.release()
              this.update({ status: 'ended', score: message.score, sabers: [] })
            }
          }
        } catch {
          /* Ignore unrelated or malformed messages; binary activity has a separate contract. */
        }
      })
      socket.addEventListener('close', () => {
        if (this.socket === socket) {
          if (this.status === 'ended') {
            this.socket = null
            this.key = ''
            this.stopClock()
          } else this.fail('The live session disconnected. Press play to reconnect.')
        }
      })
      socket.addEventListener('error', () => {
        if (this.socket === socket)
          this.fail('Could not reach the live session. Check the backend and press play to retry.')
      })
    })
  }

  play() {
    if (!this.socket || !['ready', 'paused'].includes(this.status)) return
    if (this.started) {
      this.send({ type: 'clock', song_t: this.options.getTime() })
      this.send({ type: 'pause', paused: false })
    } else {
      this.clearRun()
      this.send({ type: 'play', song_t: this.options.getTime() })
      this.started = true
    }
    flyMotion.setPaused(false)
    this.update({ status: 'playing' })
    this.stopClock()
    this.clock = setInterval(
      () => this.send({ type: 'clock', song_t: this.options.getTime() }),
      1000,
    )
  }
  pause() {
    if (this.status !== 'playing') return
    this.send({ type: 'clock', song_t: this.options.getTime() })
    this.send({ type: 'pause', paused: true })
    this.stopClock()
    flyMotion.setPaused(true)
    this.update({ status: 'paused' })
  }
  seek() {
    if (!this.socket || !['playing', 'paused', 'ready', 'ended'].includes(this.status)) return
    const playing = this.status === 'playing'
    this.stopClock()
    this.clearRun()
    this.send({ type: 'play', song_t: this.options.getTime() })
    this.started = true
    this.send({ type: 'pause', paused: !playing })
    flyMotion.setPaused(!playing)
    this.update({ status: playing ? 'ready' : 'paused' })
    if (playing) this.play()
  }
}
