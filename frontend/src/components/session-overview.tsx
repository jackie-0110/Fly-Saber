import { Activity, SlidersHorizontal } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { usePlayer } from '@/game/store'
import { useSession } from '@/game/session'

export function SessionOverview() {
  const song = usePlayer((s) => s.song)
  const chart = usePlayer((s) => s.chart)
  const brain = usePlayer((s) => s.brain)
  const status = useSession((s) => s.status)
  const error = useSession((s) => s.error)
  return (
    <>
      <section className="session-overview" aria-labelledby="overview-heading">
        <div className="inspector-section-heading">
          <h2 id="overview-heading">Session overview</h2>
          <Activity aria-hidden="true" />
        </div>
        <dl className="overview-list">
          <div>
            <dt>Game</dt>
            <dd>Fly Saber</dd>
          </div>
          <div>
            <dt>Brain variant</dt>
            <dd className="muted-value">{status === 'idle' ? 'Not connected' : brain}</dd>
          </div>
          <div>
            <dt>Dataset</dt>
            <dd className="muted-value">MaleCNS v1.0 subgraph</dd>
          </div>
          <div>
            <dt>Map / audio</dt>
            <dd className="muted-value">
              {song && chart ? `${song.title} · ${chart.difficulty.name}` : 'No map loaded'}
            </dd>
          </div>
          <div>
            <dt>Session status</dt>
            <dd>
              <span className="status-dot amber" />
              {status === 'idle' ? 'No live session' : `Live session: ${status}`}
            </dd>
          </div>
        </dl>
      </section>
      {error && <p role="alert">{error}</p>}
      <div className="intervention-placeholder">
        <SlidersHorizontal aria-hidden="true" />
        <span>Brain controls</span>
        <Badge variant="outline">Coming later</Badge>
      </div>
      <p className="inspector-footnote">
        Live songs use server actions and scores. Community maps run as local previews. Neuron
        firing awaits the activity stream.
      </p>
    </>
  )
}
