import { useMemo } from 'react'
import { matchesLattice, mosaicGeometry } from './eye-data.ts'
import type { EyeSide } from './eye-data.ts'
import { useEyeView } from './eye-state.ts'

const GHOST = Array.from({ length: 13 }, (_, row) =>
  Array.from({ length: 11 }, (_, col) => [col * 5 + (row % 2) * 2.5, row * 4.33]),
)
  .flat()
  .filter(([x, y]) => ((x - 26) / 26) ** 2 + ((y - 26) / 29) ** 2 < 1)

/** Person A can render this directly beside eye controls. Blind labels state only;
 * pixels always show what the server actually gave the brain, including gray blindness. */
export function EyeMosaic({
  side,
  blind = false,
  slot = 0,
}: {
  side: EyeSide
  blind?: boolean
  slot?: number
}) {
  const lattice = useEyeView((s) => s.lattice)
  const frame = useEyeView((s) => s.frames[slot] ?? null)
  const geometry = useMemo(() => (lattice ? mosaicGeometry(lattice) : null), [lattice])
  const pixels = matchesLattice(frame, lattice) ? frame![side] : null
  const points = lattice?.[side]
  const name = `${side === 'left' ? 'Left' : 'Right'} eye${blind ? ', blindfolded' : ''}`
  if (!points || !geometry)
    return (
      <svg
        className="eye-mosaic is-waiting"
        viewBox="-3 -3 58 60"
        role="img"
        aria-label={`${name}: awaiting eye image`}
      >
        {GHOST.map(([x, y], i) => (
          <circle key={i} cx={x} cy={y} r="1.2" />
        ))}
      </svg>
    )
  const minX = Math.min(...points.map((p) => p[0])) - geometry.radius
  return (
    <svg
      className={`eye-mosaic${pixels ? '' : ' is-waiting'}`}
      viewBox={`${minX} ${geometry.minY} ${geometry.width} ${geometry.height}`}
      role="img"
      aria-label={`${name}: ${pixels ? `${pixels.length} live facets` : 'awaiting eye image'}`}
    >
      {points.map(([azimuth, elevation], i) => (
        <polygon
          key={i}
          points={geometry.hex}
          transform={`translate(${azimuth},${-elevation})`}
          fill={pixels ? `rgb(${pixels[i]}, ${pixels[i]}, ${pixels[i]})` : undefined}
        />
      ))}
    </svg>
  )
}
