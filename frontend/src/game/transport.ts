export interface PlaybackSnapshot {
  time: number
  duration: number
  playing: boolean
  rate: number
  volume: number
}

// Rendering reads getTime() at frame rate. React receives small UI snapshots at 15 Hz.
// A future inference connection can consume the same song time without owning the camera.
export class AudioTransport {
  private context: AudioContext | null = null
  private gain: GainNode | null = null
  private buffer: AudioBuffer | null = null
  private source: AudioBufferSourceNode | null = null
  private offset = 0
  private startedAt = 0
  private playing = false
  private rate = 1
  private volume = 0.7
  private generation = 0
  private animation = 0
  private lastUpdate = 0
  private onChange: (snapshot: PlaybackSnapshot) => void
  private makeContext: () => AudioContext

  constructor(onChange: (snapshot: PlaybackSnapshot) => void,
    makeContext: () => AudioContext = () => new AudioContext()) {
    this.onChange = onChange
    this.makeContext = makeContext
  }

  private audioContext() {
    if (!this.context) {
      this.context = this.makeContext()
      this.gain = this.context.createGain()
      this.gain.gain.value = this.volume
      this.gain.connect(this.context.destination)
      this.context.addEventListener('statechange', () => {
        if (this.playing && this.context?.state !== 'running') this.pause()
      })
    }
    return this.context
  }

  async decode(audio: Blob) {
    try {
      const buffer = await this.audioContext().decodeAudioData(await audio.arrayBuffer())
      if (!Number.isFinite(buffer.duration) || buffer.duration <= 0 || buffer.duration > 1800)
        throw new Error('Duration must be between 0 and 30 minutes.')
      return buffer
    } catch (cause) {
      throw new Error(`Could not decode this map’s audio. ${cause instanceof Error ? cause.message : 'Try another map or audio format.'}`)
    }
  }

  load(buffer: AudioBuffer) {
    this.pause()
    this.buffer = buffer
    this.offset = 0
    this.emit()
  }

  getTime() {
    const time = this.offset + (this.playing && this.context ?
      (this.context.currentTime - this.startedAt) * this.rate : 0)
    return Math.max(0, Math.min(this.buffer?.duration ?? 0, time))
  }

  snapshot(): PlaybackSnapshot {
    return { time: this.getTime(), duration: this.buffer?.duration ?? 0,
      playing: this.playing, rate: this.rate, volume: this.volume }
  }

  private emit() { this.onChange(this.snapshot()) }

  private stopSource() {
    if (this.source) {
      this.source.onended = null
      this.source.stop()
      this.source.disconnect()
      this.source = null
    }
    cancelAnimationFrame(this.animation)
  }

  private startSource() {
    const context = this.audioContext()
    if (!this.buffer || !this.gain) return
    const source = context.createBufferSource()
    source.buffer = this.buffer
    source.playbackRate.value = this.rate
    source.connect(this.gain)
    source.onended = () => {
      if (this.source !== source) return
      source.disconnect()
      this.source = null
      this.offset = this.buffer?.duration ?? 0
      this.playing = false
      cancelAnimationFrame(this.animation)
      this.emit()
    }
    this.startedAt = context.currentTime
    this.source = source
    source.start(0, this.offset)
    this.playing = true
    this.lastUpdate = 0
    const tick = (now: number) => {
      if (!this.playing) return
      if (now - this.lastUpdate >= 65) { this.emit(); this.lastUpdate = now }
      this.animation = requestAnimationFrame(tick)
    }
    this.animation = requestAnimationFrame(tick)
  }

  async play() {
    if (!this.buffer || this.playing) return
    const generation = ++this.generation
    await this.audioContext().resume()
    if (generation !== this.generation || !this.buffer) return
    if (this.offset >= this.buffer.duration) this.offset = 0
    this.startSource()
    this.emit()
  }

  pause() {
    this.generation++
    this.offset = this.getTime()
    this.playing = false
    this.stopSource()
    this.emit()
  }

  seek(time: number) {
    if (!this.buffer || !Number.isFinite(time)) return
    const wasPlaying = this.playing
    this.generation++
    this.stopSource()
    this.offset = Math.max(0, Math.min(this.buffer.duration, time))
    this.playing = false
    if (wasPlaying && this.offset < this.buffer.duration) this.startSource()
    this.emit()
  }

  setRate(rate: number) {
    if (!Number.isFinite(rate) || rate < 0.25 || rate > 2) return
    const time = this.getTime()
    this.rate = rate
    this.seek(time)
    this.emit()
  }

  setVolume(volume: number) {
    if (!Number.isFinite(volume)) return
    this.volume = Math.max(0, Math.min(1, volume))
    if (this.gain && this.context)
      this.gain.gain.setTargetAtTime(this.volume, this.context.currentTime, 0.015)
    this.emit()
  }
}
