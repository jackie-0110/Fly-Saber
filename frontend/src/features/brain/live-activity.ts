import type { ActivityFrame } from '../session/activity.ts'

/** Activity older than this is shown as paused, never held as if live. */
export const STALE_MS = 1000

/**
 * The latest live brain activity, kept outside React: the session writes it, the neuron
 * cloud reads it each animation frame and redraws when a step changes. Values are change
 * from rest (docs/reference/PROTOCOL.md section 6), indexed like `neurons.json` and `ready.cell_types`.
 */
export const liveActivity = {
  neurons: null as Int8Array | null,
  neuronStep: -1,
  types: null as Int8Array | null,
  typeStep: -1,
  /** `performance.now()` of the latest update, for showing activity as stale. */
  receivedAt: 0,
  listeners: new Set<() => void>(),
  staleTimer: undefined as ReturnType<typeof setTimeout> | undefined,

  receive(frame: ActivityFrame) {
    if (frame.kind === 1) {
      this.neurons = frame.values
      this.neuronStep = frame.step
    } else {
      this.types = frame.values
      this.typeStep = frame.step
    }
    this.receivedAt = performance.now()
    clearTimeout(this.staleTimer)
    this.staleTimer = setTimeout(() => this.notify(), STALE_MS)
    this.notify()
  },
  clear() {
    this.neurons = this.types = null
    this.neuronStep = this.typeStep = -1
    this.receivedAt = 0
    clearTimeout(this.staleTimer)
    this.notify()
  },
  /** True while neuron activity is arriving. */
  live(now = performance.now()) {
    return this.neurons !== null && now < this.receivedAt + STALE_MS
  },
  /** Calls `listener` after every update, clear, and when activity goes stale. */
  subscribe(listener: () => void) {
    this.listeners.add(listener)
    return () => {
      this.listeners.delete(listener)
    }
  },
  notify() {
    this.listeners.forEach((listener) => listener())
  },
}
