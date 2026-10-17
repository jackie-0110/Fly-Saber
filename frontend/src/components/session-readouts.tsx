import { Crosshair, Eye, Gauge } from 'lucide-react'
import { useSession } from '@/game/session'

export function SessionReadouts() {
  const score = useSession((s) => s.score)
  const sabers = useSession((s) => s.sabers)
  const lastEvent = useSession((s) => s.lastEvent)
  const status = useSession((s) => s.status)
  return (
    <section className="session-readouts" aria-label="Session readouts">
      <div className="readout-block">
        <h2>
          <Gauge aria-hidden="true" />
          Performance
        </h2>
        <div className="performance-values">
          <div>
            <strong>{score ? `${(score.accuracy * 100).toFixed(1)}%` : '--'}</strong>
            <span>ACCURACY</span>
          </div>
          <div>
            <strong>{score?.combo ?? '--'}</strong>
            <span>COMBO</span>
          </div>
          <div>
            <strong>{score?.score ?? '--'}</strong>
            <span>SCORE</span>
          </div>
        </div>
      </div>
      <div className="readout-block">
        <h2>
          <Eye aria-hidden="true" />
          Visual input
        </h2>
        <p className="readout-status">
          <span className="status-dot" />
          Awaiting fly's-eye stream
        </p>
        <span className="readout-caption">Two eyes. One view of the world.</span>
      </div>
      <div className="readout-block">
        <h2>
          <Crosshair aria-hidden="true" />
          Motor output
        </h2>
        <p className="readout-status">
          <span className="status-dot" />
          {sabers.length
            ? sabers
                .map(
                  (s) =>
                    `${s.color === 0 ? 'Left' : 'Right'}: ${s.cell === null ? 'idle' : `cell ${s.cell}`}`,
                )
                .join(' · ')
            : status === 'ended'
              ? 'Session complete'
              : 'Awaiting simulation'}
        </p>
        <span className="readout-caption">
          {score
            ? `${score.hits} hits · ${score.misses} misses${lastEvent ? ` · ${lastEvent.kind.replace('_', ' ')} #${lastEvent.note_id}` : ''}`
            : 'Saber actions will appear here.'}
        </span>
      </div>
    </section>
  )
}
