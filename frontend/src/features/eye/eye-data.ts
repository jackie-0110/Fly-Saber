/** Processed luminance, docs/reference/PROTOCOL.md section 3. Values are unsigned, unlike activity. */
export interface EyeFrame {
  slot: number
  step: number
  left: Uint8Array
  right: Uint8Array
}

export type EyeSide = 'left' | 'right'
export type EyeLattice = Record<EyeSide, [number, number][]>

export function decodeEyeImage(data: ArrayBuffer): EyeFrame | null {
  if (data.byteLength < 12) return null
  const view = new DataView(data)
  if (view.getUint8(0) !== 3 || view.getUint8(1) > 1 || view.getUint16(2, true) !== 0) return null
  const left = view.getUint16(8, true)
  const right = view.getUint16(10, true)
  if (!left || !right || data.byteLength !== 12 + left + right) return null
  return {
    slot: view.getUint8(1),
    step: view.getUint32(4, true),
    left: new Uint8Array(data, 12, left),
    right: new Uint8Array(data, 12 + left, right),
  }
}

export function parseEyeLattice(value: unknown): EyeLattice {
  if (!value || typeof value !== 'object') throw new Error('Invalid eye lattice')
  for (const side of ['left', 'right'] as const) {
    const points = (value as EyeLattice)[side]
    if (
      !Array.isArray(points) ||
      !points.length ||
      points.length > 65535 ||
      !points.every((p) => Array.isArray(p) && p.length === 2 && p.every((n) => Number.isFinite(n)))
    )
      throw new Error('Invalid eye lattice')
  }
  return value as EyeLattice
}

export function matchesLattice(frame: EyeFrame | null, lattice: EyeLattice | null) {
  return (
    !!frame &&
    !!lattice &&
    frame.left.length === lattice.left.length &&
    frame.right.length === lattice.right.length
  )
}

/** Equal angular scale in both eyes; elevation points up, azimuth points right. */
export function mosaicGeometry(lattice: EyeLattice) {
  const all = [...lattice.left, ...lattice.right]
  // Real columns have some holes. Use the median nearest neighbor distance so holes
  // don't enlarge individual facets, and don't assume the synthetic config's spacing.
  const distances = lattice.left
    .map(([x, y], i, points) => {
      let distance = Infinity
      points.forEach(([a, b], j) => {
        if (i !== j) distance = Math.min(distance, Math.hypot(x - a, y - b))
      })
      return distance
    })
    .filter((n) => Number.isFinite(n) && n > 0)
    .sort((a, b) => a - b)
  const radius = ((distances[Math.floor(distances.length / 2)] ?? 5) / Math.sqrt(3)) * 0.94
  const minY = Math.min(...all.map((p) => -p[1])) - radius
  const height = Math.max(...all.map((p) => -p[1])) + radius - minY
  const width = Math.max(
    ...(['left', 'right'] as const).map((side) => {
      const xs = lattice[side].map((p) => p[0])
      return Math.max(...xs) - Math.min(...xs) + radius * 2
    }),
  )
  const hex = Array.from({ length: 6 }, (_, i) => {
    const angle = Math.PI / 6 + (i * Math.PI) / 3
    return `${Math.cos(angle) * radius},${Math.sin(angle) * radius}`
  }).join(' ')
  return { radius, minY, height, width, hex }
}
