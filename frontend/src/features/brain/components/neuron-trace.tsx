import { useEffect, useState } from 'react'
import { liveActivity } from '@/features/brain/live-activity'

// Neuron activity arrives at 15 Hz (docs/PROTOCOL.md section 6), so 60 samples is 4 s.
const SAMPLES = 60
const WIDTH = 240
const HEIGHT = 36

// Rates move by hundredths while the stream spans +-0.25, so a fixed scale draws a flat
// line. The trace scales to its own recent peak instead, never tighter than this (of 127),
// so a neuron at rest stays flat rather than magnifying single-step jitter.
const MIN_PEAK = 8

/**
 * The selected neuron's live activity as a sparkline: change from rest above and below a
 * rest line, scaled to the last 4 s; the caption gives the actual change in rate units.
 */
export function NeuronTrace({ index }: { index: number }) {
  const [samples, setSamples] = useState<number[]>([])
  useEffect(() => {
    let step = -1
    // An empty trace for a new neuron, filled from its next frame on.
    const reset = setTimeout(() => setSamples([]))
    const unsubscribe = liveActivity.subscribe(() => {
      const values = liveActivity.neurons
      if (!values || liveActivity.neuronStep === step || index >= values.length) return
      step = liveActivity.neuronStep
      const value = values[index]
      setSamples((previous) => [...previous.slice(-(SAMPLES - 1)), value])
    })
    return () => {
      clearTimeout(reset)
      unsubscribe()
    }
  }, [index])

  if (!samples.length)
    return <p className="neuron-trace-empty">Its activity traces here during live play.</p>
  const x = (i: number) => (i / (SAMPLES - 1)) * WIDTH
  const peak = Math.max(MIN_PEAK, ...samples.map(Math.abs))
  const y = (v: number) => HEIGHT / 2 - (v / peak) * (HEIGHT / 2 - 2)
  const latest = samples.at(-1) ?? 0
  const offset = SAMPLES - samples.length
  const points = samples.map((v, i) => `${x(i + offset).toFixed(1)},${y(v).toFixed(1)}`).join(' ')
  return (
    <figure className="neuron-trace" data-sign={latest > 0 ? 'up' : latest < 0 ? 'down' : 'rest'}>
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        preserveAspectRatio="none"
        role="img"
        aria-label={`Live activity, last 4 seconds: ${describe(latest)}`}
      >
        <line className="neuron-trace-rest" x1="0" x2={WIDTH} y1={HEIGHT / 2} y2={HEIGHT / 2} />
        <polyline points={points} />
      </svg>
      <figcaption>
        <span>Live activity, last 4 s</span>
        <span>{describe(latest)}</span>
      </figcaption>
    </figure>
  )
}

function describe(value: number) {
  if (value === 0) return 'at rest'
  const rate = ((Math.abs(value) / 127) * 0.25).toFixed(3)
  return `${value > 0 ? '+' : '−'}${rate} from rest`
}
