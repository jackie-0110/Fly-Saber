import { ArrowDown, ArrowLeft, ArrowRight, ArrowUp } from 'lucide-react'
import type { SaberAction } from '@/features/game/fly-motion'
import { BrainStage, EyeToggles } from '@/features/brain/components/brain-controls'
import { useSession } from '@/features/session/session'

// Cut directions as the protocol numbers them: 0 up, 1 down, 2 left, 3 right.
const CUTS = [ArrowUp, ArrowDown, ArrowLeft, ArrowRight]
const SABERS = [
  { color: 0, name: 'Left', tone: 'magenta' },
  { color: 1, name: 'Right', tone: 'green' },
] as const

const plural = (count: number, one: string, many = `${one}s`) =>
  `${count} ${count === 1 ? one : many}`

export function SessionReadouts() {
  const score = useSession((s) => s.score)
  const sabers = useSession((s) => s.sabers)
  const status = useSession((s) => s.status)
  return (
    <section className="session-readouts" aria-label="Session readouts">
      <div className="readout-block">
        <h2>Visual input</h2>
        <EyeToggles />
        <p className="readout-status">
          <span className="status-dot" />
          Awaiting fly's-eye stream
        </p>
      </div>
      <BrainStage />
      <div className="readout-block">
        <h2>Motor output</h2>
        {sabers.length ? (
          <div className="motor-grid" aria-label="Saber targets">
            {SABERS.map(({ color, name, tone }) => (
              <SaberRow
                key={color}
                name={name}
                tone={tone}
                action={sabers.find((s) => s.color === color)}
              />
            ))}
          </div>
        ) : (
          <p className="readout-status">
            <span className="status-dot" />
            {status === 'ended' ? 'Session complete' : 'Awaiting simulation'}
          </p>
        )}
        {score && (
          <span className="readout-caption">
            {plural(score.hits, 'hit')} · {plural(score.misses, 'miss', 'misses')}
          </span>
        )}
      </div>
      <div className="readout-block">
        <h2>Performance</h2>
        <div className="performance-values">
          <div>
            <strong>{score ? `${(score.accuracy * 100).toFixed(1)}%` : '--'}</strong>
            <span>Accuracy</span>
          </div>
          <div>
            <strong>{score?.combo ?? '--'}</strong>
            <span>Combo</span>
          </div>
          <div>
            <strong>{score?.score ?? '--'}</strong>
            <span>Score</span>
          </div>
        </div>
      </div>
    </section>
  )
}

/** One saber's target on the 4 × 3 note grid (cell = layer * 4 + lane) and its cut. */
function SaberRow({
  name,
  tone,
  action,
}: {
  name: string
  tone: 'magenta' | 'green'
  action: SaberAction | undefined
}) {
  const cell = action?.cell ?? null
  const Cut = action?.cut === null || action?.cut === undefined ? null : CUTS[action.cut]
  return (
    <div className={`motor-row ${tone}`}>
      <span className="motor-name">{name}</span>
      <span
        className="cell-grid"
        role="img"
        aria-label={`${name} saber: ${cell === null ? 'idle' : `cell ${cell}`}`}
      >
        {[2, 1, 0].flatMap((layer) =>
          [0, 1, 2, 3].map((lane) => (
            <i key={`${layer}-${lane}`} data-active={cell === layer * 4 + lane} />
          )),
        )}
      </span>
      <span className="motor-cut" aria-hidden="true">
        {Cut ? <Cut /> : '·'}
      </span>
    </div>
  )
}
