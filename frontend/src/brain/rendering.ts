interface ConnectionGraph {
  asset: { n_neurons: number }
  edges: { pre: number[]; post: number[] }
}

export const MOVING_EDGE_LIMIT = 20_000
// The default "Sample" detail draws this many evenly spaced edges, moving or not;
// "All" draws every filtered edge at rest. Software WebGL stalls on the full graph.
export const SAMPLE_EDGE_LIMIT = MOVING_EDGE_LIMIT
export type ConnectionDetail = 'off' | 'sample' | 'all'

export function sampleEdges(edges: number[], limit: number): number[] {
  if (edges.length <= limit) return edges
  return Array.from({ length: limit }, (_, i) => edges[Math.floor((i * edges.length) / limit)])
}
export const CAMERA_SETTLE_MS = 160

// Reuse the N neuron positions rather than uploading 2E duplicate positions.
// Put an evenly spaced sample first so movement only changes the draw range;
// the complete index buffer stays on the GPU and contains every edge once.
export function connectionIndex(data: ConnectionGraph, edges: number[], selected: number | null) {
  const IndexArray = data.asset.n_neurons <= 65_536 ? Uint16Array : Uint32Array
  const index = new IndexArray(edges.length * 2)
  let offset = 0
  const append = (edge: number) => {
    index[offset++] = data.edges.pre[edge]
    index[offset++] = data.edges.post[edge]
  }
  if (selected !== null) {
    // Two material groups keep reciprocal connections correctly colored without
    // duplicating positions or assigning ambiguous per-neuron direction colors.
    for (const edge of edges) if (data.edges.pre[edge] === selected) append(edge)
    const outgoing = offset
    for (const edge of edges) if (data.edges.pre[edge] !== selected) append(edge)
    return { index, outgoing }
  }
  const sampleCount = Math.min(MOVING_EDGE_LIMIT, edges.length)
  for (let sample = 0; sample < sampleCount; sample++) {
    append(edges[Math.floor((sample * edges.length) / sampleCount)])
  }
  let sample = 0
  for (let position = 0; position < edges.length; position++) {
    if (sample < sampleCount && position === Math.floor((sample * edges.length) / sampleCount)) {
      sample++
    } else append(edges[position])
  }
  return { index, outgoing: 0 }
}
