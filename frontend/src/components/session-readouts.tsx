import { useSession } from '@/game/session'

export function SessionReadouts() {
  const score = useSession((s) => s.score)
  const sabers = useSession((s) => s.sabers)
  const lastEvent = useSession((s) => s.lastEvent)
  const status = useSession((s) => s.status)
  return (
    <section className="session-readouts" aria-label="Session readouts">
      <div className="readout-block">
        <h2>Visual input</h2>
        <p className="readout-status">
          <span className="status-dot" />
          Awaiting fly's-eye stream
        </p>
      </div>
      <div className="readout-block">
        <h2>Motor output</h2>
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
