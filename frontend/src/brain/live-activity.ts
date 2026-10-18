import type { ActivityFrame } from '../game/activity.ts'

/**
 * The latest live brain activity, kept outside React: the session writes it, the neuron
 * cloud reads it each animation frame and redraws when a step changes. Values are change
 * from rest (docs/PROTOCOL.md section 6), indexed like `neurons.json` and `ready.cell_types`.
 */
export const liveActivity = {
  neurons: null as Int8Array | null,
  neuronStep: -1,
  types: null as Int8Array | null,
  typeStep: -1,
  /** `performance.now()` of the latest update, for showing activity as stale. */
  receivedAt: 0,

  receive(frame: ActivityFrame) {
    if (frame.kind === 1) {
      this.neurons = frame.values
      this.neuronStep = frame.step
    } else {
      this.types = frame.values
      this.typeStep = frame.step
    }
    this.receivedAt = performance.now()
  },
  clear() {
    this.neurons = this.types = null
    this.neuronStep = this.typeStep = -1
    this.receivedAt = 0
  },
}
