export interface BrainAsset {
  schema_version: 1
  source: string
  n_neurons: number
  cell_types: string[]
  superclasses: string[]
  roles: string[]
  sides: string[]
  nts: string[]
  position_sources: string[]
  bounds_um: { min: number[]; max: number[] }
  body_id: number[]
  type: number[]
  superclass: number[]
  role: number[]
  side: number[]
  nt: number[]
  sign: number[]
  position_um: number[]
  position_source: number[]
}

export interface BrainData {
  asset: BrainAsset
  edges: BrainEdges
  groups: Record<string, number[]>
  positions: Float32Array
  radius: number
  distinctPositionCount: number
}

export interface BrainEdges {
  schema_version: 1
  source: string
  n_neurons: number
  n_edges: number
  n_synapses: number
  body_id: number[]
  pre: number[]
  post: number[]
  // Anatomical synapse counts, not learned model efficacies.
  weight: number[]
  sign: number[]
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isFiniteArray(value: unknown, length: number): value is number[] {
  return Array.isArray(value) && value.length === length && value.every(Number.isFinite)
}

// Keep array index i unchanged: the backend activity stream and groups use that index.
export function parseBrainData(raw: unknown, rawGroups: unknown, rawEdges: unknown): BrainData {
  if (!isRecord(raw) || raw.schema_version !== 1 || typeof raw.source !== 'string') {
    throw new Error('Unsupported brain export. Expected neuron schema version 1.')
  }
  const count = raw.n_neurons
  if (typeof count !== 'number' || !Number.isInteger(count) || count < 1 || count > 200_000) {
    throw new Error('Invalid neuron count in the brain export.')
  }
  const tables = {
    type: 'cell_types',
    superclass: 'superclasses',
    role: 'roles',
    side: 'sides',
    nt: 'nts',
    position_source: 'position_sources',
  }
  for (const [field, lookup] of Object.entries(tables)) {
    const labels = raw[lookup]
    const codes = raw[field]
    if (
      !Array.isArray(labels) ||
      !labels.length ||
      !labels.every((v) => typeof v === 'string') ||
      !isFiniteArray(codes, count) ||
      !codes.every((v) => Number.isInteger(v) && v >= 0 && v < labels.length)
    )
      throw new Error(`Invalid ${field} metadata in the brain export.`)
  }
  if (
    !isFiniteArray(raw.body_id, count) ||
    !raw.body_id.every(Number.isSafeInteger) ||
    !isFiniteArray(raw.sign, count) ||
    !raw.sign.every((v) => [-1, 0, 1].includes(v)) ||
    !isFiniteArray(raw.position_um, count * 3)
  )
    throw new Error('Invalid neuron IDs or coordinates in the brain export.')
  if (
    !isRecord(raw.bounds_um) ||
    !isFiniteArray(raw.bounds_um.min, 3) ||
    !isFiniteArray(raw.bounds_um.max, 3)
  )
    throw new Error('Invalid coordinate bounds in the brain export.')
  const { min, max } = raw.bounds_um
  if (max.some((v, axis) => v < min[axis])) throw new Error('Invalid coordinate bounds.')
  const extent = Math.max(...max.map((v, axis) => v - min[axis]))
  if (extent <= 0) throw new Error('The brain export has no spatial extent.')
  if (!isRecord(rawGroups)) throw new Error('Invalid neuron groups export.')
  for (const [name, indices] of Object.entries(rawGroups)) {
    if (
      !Array.isArray(indices) ||
      !indices.every((v) => Number.isInteger(v) && v >= 0 && v < count)
    ) {
      throw new Error(`Invalid neuron indices in group ${name}.`)
    }
  }
  const asset = raw as unknown as BrainAsset
  const edges = parseBrainEdges(rawEdges, asset)
  const center = min.map((v, axis) => (v + max[axis]) / 2)
  const scale = 2 / extent
  const positions = new Float32Array(count * 3)
  let radius = 0
  const distinctPositions = new Set<string>()
  for (let i = 0; i < count; i++) {
    distinctPositions.add(asset.position_um.slice(i * 3, i * 3 + 3).join(','))
    const x = (asset.position_um[i * 3] - center[0]) * scale
    const y = (asset.position_um[i * 3 + 1] - center[1]) * scale
    const z = (asset.position_um[i * 3 + 2] - center[2]) * scale
    // Uniform scale and a rigid rotation only: source (x,y,z) -> display (x,-z,y).
    positions.set([x, -z, y], i * 3)
    radius = Math.max(radius, Math.hypot(x, y, z))
  }
  return {
    asset,
    edges,
    groups: rawGroups as Record<string, number[]>,
    positions,
    radius,
    distinctPositionCount: distinctPositions.size,
  }
}

function parseBrainEdges(raw: unknown, asset: BrainAsset): BrainEdges {
  if (!isRecord(raw) || raw.schema_version !== 1 || typeof raw.source !== 'string')
    throw new Error('Unsupported brain connections export. Expected edge schema version 1.')
  const count = raw.n_edges
  if (
    raw.n_neurons !== asset.n_neurons ||
    !isFiniteArray(raw.body_id, asset.n_neurons) ||
    !raw.body_id.every((id, i) => id === asset.body_id[i])
  )
    throw new Error('Brain connections do not match the neuron ordering.')
  if (
    typeof count !== 'number' ||
    !Number.isSafeInteger(count) ||
    count < 0 ||
    count > 5_000_000 ||
    !isFiniteArray(raw.pre, count) ||
    !isFiniteArray(raw.post, count) ||
    !isFiniteArray(raw.weight, count) ||
    !isFiniteArray(raw.sign, count)
  )
    throw new Error('Invalid connection arrays in the brain export.')
  let synapses = 0
  for (let edge = 0; edge < count; edge++) {
    const pre = raw.pre[edge]
    const post = raw.post[edge]
    const weight = raw.weight[edge]
    if (
      !Number.isInteger(pre) ||
      pre < 0 ||
      pre >= asset.n_neurons ||
      !Number.isInteger(post) ||
      post < 0 ||
      post >= asset.n_neurons ||
      !Number.isSafeInteger(weight) ||
      weight < 1 ||
      raw.sign[edge] !== asset.sign[pre]
    )
      throw new Error(`Invalid endpoints, synapse count, or sign for connection ${edge}.`)
    synapses += weight
  }
  if (!Number.isSafeInteger(synapses) || raw.n_synapses !== synapses)
    throw new Error('Invalid total synapse count in the brain connections export.')
  return raw as unknown as BrainEdges
}

export async function loadBrainData(signal: AbortSignal): Promise<BrainData> {
  const base = `${import.meta.env.BASE_URL}data/brain/`
  const responses = await Promise.all(
    ['neurons.json', 'groups.json', 'edges.json'].map((file) =>
      fetch(`${base}${file}`, { signal }),
    ),
  )
  if (responses.some((response) => !response.ok))
    throw new Error('Could not load the brain data files.')
  const [asset, groups, edges] = await Promise.all(responses.map((response) => response.json()))
  return parseBrainData(asset, groups, edges)
}

// Both endpoints must survive the filters. Selection reveals only incident edges.
export function filterEdgeIndices(
  data: BrainData,
  indices: number[],
  selected: number | null = null,
): number[] {
  const visible = new Uint8Array(data.asset.n_neurons)
  for (const index of indices) visible[index] = 1
  const result: number[] = []
  const { pre, post, n_edges } = data.edges
  for (let edge = 0; edge < n_edges; edge++) {
    if (
      visible[pre[edge]] &&
      visible[post[edge]] &&
      (selected === null || pre[edge] === selected || post[edge] === selected)
    )
      result.push(edge)
  }
  return result
}

export function neuronConnectionStats(data: BrainData, index: number) {
  let incoming = 0,
    outgoing = 0,
    incomingSynapses = 0,
    outgoingSynapses = 0
  const { pre, post, weight, n_edges } = data.edges
  for (let edge = 0; edge < n_edges; edge++) {
    if (pre[edge] === index) {
      outgoing++
      outgoingSynapses += weight[edge]
    }
    if (post[edge] === index) {
      incoming++
      incomingSynapses += weight[edge]
    }
  }
  return { incoming, outgoing, incomingSynapses, outgoingSynapses }
}

export function filterNeuronIndices(data: BrainData, group: string, cellType: string): number[] {
  const indices = group
    ? (data.groups[group] ?? [])
    : Array.from({ length: data.asset.n_neurons }, (_, i) => i)
  return indices.filter((i) => cellType === '' || data.asset.type[i] === Number(cellType))
}

export function cellTypeColor(type: number): string {
  return `hsl(${(type * 137.508 + 165) % 360}, 48%, 69%)`
}
