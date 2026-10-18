import { ArrowDown, ArrowLeft, ArrowRight, ArrowUp } from 'lucide-react'
import type { SaberAction } from '@/game/fly-motion'
import { BrainStage, EyeToggles } from '@/components/brain-controls'
import { useBrainControls } from '@/game/controls-state'
import { useSession } from '@/game/session'

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
        <EyeStatus />
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

/*
 * What the eyes are doing now. The server renders the fly's view from the notes every
 * frame of live play; the picture itself is not streamed to the page yet, so this says
 * whether the fly is seeing rather than promising an image.
 */
function EyeStatus() {
  const status = useSession((s) => s.status)
  const left = useBrainControls((s) => s.controls.blindLeft)
  const right = useBrainControls((s) => s.controls.blindRight)
  let tone = ''
  let text =
    status === 'ready' || status === 'paused'
      ? 'Sees the track once the song plays'
      : 'Sees the track during live play'
  if (left && right) {
    tone = 'red'
    text = 'Both eyes blindfolded: the fly sees nothing'
  } else if (status === 'playing') {
    tone = 'green'
    text = left
      ? 'Seeing the track, left eye blindfolded'
      : right
        ? 'Seeing the track, right eye blindfolded'
        : 'Seeing the track live'
  }
  return (
    <p className="readout-status">
      <span className={`status-dot ${tone}`} />
      {text}
    </p>
  )
}
