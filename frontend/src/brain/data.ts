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
  groups: Record<string, number[]>
  positions: Float32Array
  radius: number
  distinctPositionCount: number
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isFiniteArray(value: unknown, length: number): value is number[] {
  return Array.isArray(value) && value.length === length && value.every(Number.isFinite)
}

// Keep array index i unchanged: the backend activity stream and groups use that index.
export function parseBrainData(raw: unknown, rawGroups: unknown): BrainData {
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
    groups: rawGroups as Record<string, number[]>,
    positions,
    radius,
    distinctPositionCount: distinctPositions.size,
  }
}

export async function loadBrainData(signal: AbortSignal): Promise<BrainData> {
  const base = `${import.meta.env.BASE_URL}data/brain/`
  const responses = await Promise.all(
    ['neurons.json', 'groups.json'].map((file) => fetch(`${base}${file}`, { signal })),
  )
  if (responses.some((response) => !response.ok))
    throw new Error('Could not load the brain data files.')
  const [asset, groups] = await Promise.all(responses.map((response) => response.json()))
  return parseBrainData(asset, groups)
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
