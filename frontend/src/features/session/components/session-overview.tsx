import { Button } from '@/components/ui/button'
import {
  describeControls,
  useBrainControls,
  useControlsConfigState,
} from '@/features/session/controls-state'
import { difficultyLabel } from '@/features/songs/chart'
import { BaselineCompare } from '@/features/session/components/baseline-compare'
import { liveMapId, usePlayer } from '@/features/session/store'
import { useSession } from '@/features/session/session'

/** What is running now: live status first, then the run's setup and score. */
export function SessionOverview() {
  const song = usePlayer((s) => s.song)
  const chart = usePlayer((s) => s.chart)
  const brain = usePlayer((s) => s.brains.find((b) => b.id === s.brain)?.label ?? s.brain)
  const live = usePlayer((s) => !!liveMapId(s))
  const setInspectorTab = usePlayer((s) => s.setInspectorTab)
  const status = useSession((s) => s.status)
  const error = useSession((s) => s.error)
  const score = useSession((s) => s.score)
  const controls = useBrainControls((s) => s.controls)
  const { config } = useControlsConfigState()
  const interventions = describeControls(controls, config)
  const idle = status === 'idle'

  return (
    <>
      <section className="session-status" data-live={!idle} aria-label="Live session">
        <p className="session-state">
          <span className={`status-dot ${idle ? 'amber' : 'green'}`} />
          {idle ? 'No live session' : `Live session: ${status}`}
        </p>
        <p className="session-hint">
          {idle
            ? song
              ? live
                ? 'Press play to start this song live: the brain drives the sabers.'
                : 'This map plays as a preview. Pick a ready-to-play song to run it live.'
              : 'Pick a song to begin. Ready-to-play songs run live on the brain.'
            : 'The brain drives the sabers; judged notes and score arrive from the server.'}
        </p>
        {idle && !live && (
          <Button variant="outline" size="xs" onClick={() => setInspectorTab('songs')}>
            Browse songs
          </Button>
        )}
      </section>
      {error && <p role="alert">{error}</p>}

      {score && (
        <dl className="session-score">
          <div>
            <dt>Accuracy</dt>
            <dd>{(score.accuracy * 100).toFixed(1)}%</dd>
          </div>
          <div>
            <dt>Hits</dt>
            <dd>{score.hits}</dd>
          </div>
          <div>
            <dt>Misses</dt>
            <dd>{score.misses}</dd>
          </div>
          <div>
            <dt>Best combo</dt>
            <dd>{score.max_combo}</dd>
          </div>
        </dl>
      )}

      <BaselineCompare />

      <section className="session-overview" aria-labelledby="overview-heading">
        <div className="inspector-section-heading">
          <h2 id="overview-heading">This run</h2>
        </div>
        <dl className="overview-list">
          <div>
            <dt>Song</dt>
            <dd className={song ? undefined : 'muted-value'}>
              {song && chart
                ? `${song.title} · ${difficultyLabel(chart.difficulty.name)}`
                : 'None loaded'}
            </dd>
          </div>
          <div>
            <dt>Mode</dt>
            <dd className={song ? undefined : 'muted-value'}>
              {!song ? '—' : live ? 'Live on the brain' : 'Local preview'}
            </dd>
          </div>
          <div>
            <dt>Brain</dt>
            <dd className={brain ? undefined : 'muted-value'}>{brain || 'Loading…'}</dd>
          </div>
          <div>
            <dt>Interventions</dt>
            <dd className={interventions ? undefined : 'muted-value'}>{interventions ?? 'None'}</dd>
          </div>
          <div>
            <dt>Connectome</dt>
            <dd className="muted-value">MaleCNS v1.0 visual-to-motor subgraph</dd>
          </div>
        </dl>
      </section>
    </>
  )
}
