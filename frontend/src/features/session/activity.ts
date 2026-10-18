/** Binary activity messages from a live session (docs/reference/PROTOCOL.md sections 3 and 6). */

export const NEURON_ACTIVITY = 1
export const TYPE_ACTIVITY = 2
const HEADER_BYTES = 8

export interface ActivityFrame {
  kind: typeof NEURON_ACTIVITY | typeof TYPE_ACTIVITY
  slot: number
  step: number
  /**
   * Change from the untouched brain's rest, +-127 at +-0.25 rate units: one value per neuron
   * in `neurons.json` order (kind 1) or per cell type in `ready.cell_types` order (kind 2).
   */
  values: Int8Array
}

/** Decodes one message; null if its header or length does not match this session. */
export function decodeActivity(
  data: ArrayBuffer,
  sizes: { neurons: number; types: number },
): ActivityFrame | null {
  if (data.byteLength < HEADER_BYTES) return null
  const header = new DataView(data, 0, HEADER_BYTES)
  const kind = header.getUint8(0)
  if (header.getUint16(2, true) !== 0) return null
  const expected =
    kind === NEURON_ACTIVITY ? sizes.neurons : kind === TYPE_ACTIVITY ? sizes.types : -1
  if (data.byteLength - HEADER_BYTES !== expected) return null
  return {
    kind: kind as ActivityFrame['kind'],
    slot: header.getUint8(1),
    step: header.getUint32(4, true),
    values: new Int8Array(data, HEADER_BYTES),
  }
}
