import { decodeActivity } from './activity.ts'
import type { ActivityFrame } from './activity.ts'
import { flyMotion } from '../game/fly-motion.ts'
import type { SaberAction } from '../game/fly-motion.ts'
import { interventionsMessage, isUntouched, noControls } from './interventions.ts'
import type { BrainControls } from './interventions.ts'

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
  /** A rejected message once the session is ready; the session keeps running. */
  controlsError: string
  score: Score | null
  sabers: SaberAction[]
  removed: ReadonlySet<number>
  lastEvent: { kind: string; note_id: number } | null
}
export const emptySession: SessionSnapshot = {
  status: 'idle',
  error: '',
  controlsError: '',
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
  /** Brain activity while playing, in step order; never called for a brain-less player. */
  onActivity?: (frame: ActivityFrame) => void
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
  private activitySizes = { neurons: 0, types: 0 }
  private activitySteps = new Map<number, number>()
  private controls = noControls
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
  private sendControls() {
    if (this.activitySizes.neurons > 0 && ['ready', 'playing', 'paused'].includes(this.status))
      this.send(interventionsMessage(this.controls))
  }
  /** Replace the brain and eye controls now, and on every later start until changed. */
  setInterventions(controls: BrainControls) {
    this.controls = controls
    this.update({ controlsError: '' })
    this.sendControls()
  }
  private stopClock() {
    clearInterval(this.clock)
    this.clock = undefined
  }
  private clearRun() {
    this.removed = new Set()
    this.activitySteps.clear()
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
      socket.binaryType = 'arraybuffer'
      this.deadline = setTimeout(() => {
        if (this.socket === socket) this.fail('The live session timed out. Press play to retry.')
      }, 15000)
      socket.addEventListener('open', () => {
        if (this.socket === socket)
          this.send({ type: 'start', game: 'saber', brains: [brain], map_id: mapId, seed: 0 })
      })
      socket.addEventListener('message', (event) => {
        if (this.socket !== socket) return
        if (event.data instanceof ArrayBuffer) {
          this.receiveActivity(event.data)
          return
        }
        if (typeof event.data !== 'string') return
        try {
          const message = JSON.parse(event.data)
          if (!message || typeof message !== 'object') return
          if (message.type === 'error' && typeof message.message === 'string') {
            if (this.status === 'connecting')
              this.fail(`Live session: ${message.message}. Press play to retry.`)
            else this.update({ controlsError: message.message })
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
            this.activitySizes = {
              neurons: Number.isInteger(message.n_neurons) ? message.n_neurons : 0,
              types: Array.isArray(message.cell_types) ? message.cell_types.length : 0,
            }
            this.update({ status: 'ready' })
            if (!isUntouched(this.controls)) this.sendControls()
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
          /* Ignore unrelated or malformed messages. */
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

  private receiveActivity(data: ArrayBuffer) {
    if (this.status !== 'playing' || !this.options.onActivity) return
    const frame = decodeActivity(data, this.activitySizes)
    if (!frame || frame.slot !== 0) return
    if (frame.step <= (this.activitySteps.get(frame.kind) ?? -1)) return
    this.activitySteps.set(frame.kind, frame.step)
    this.options.onActivity(frame)
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
