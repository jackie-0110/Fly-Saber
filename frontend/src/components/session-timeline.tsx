import { Pause, Play, RotateCcw } from 'lucide-react'

const CONTROLS = [
  { label: 'Play session', Icon: Play },
  { label: 'Pause session', Icon: Pause },
  { label: 'Restart session', Icon: RotateCcw },
]

export function SessionTimeline() {
  return (
    <section className="timeline" aria-labelledby="timeline-heading">
      <div className="section-head">
        <h2 id="timeline-heading">Timeline</h2>
        <span className="section-meta">No session loaded</span>
      </div>
      <div className="timeline-row">
        <div className="playback" role="group" aria-label="Playback">
          {CONTROLS.map(({ label, Icon }) => (
            <button key={label} type="button" className="tool-button" disabled aria-label={label}>
              <Icon aria-hidden="true" />
            </button>
          ))}
          <span className="playback-speed" aria-label="Playback speed 1×">
            1×
          </span>
        </div>
        <div className="timeline-track" aria-hidden="true">
          <div className="timeline-rail">
            <span className="timeline-head" />
          </div>
          <div className="timeline-ends">
            <span>00:00</span>
            <span>--:--</span>
          </div>
        </div>
        <p className="timeline-clock">
          00:00 <span>/ --:--</span>
        </p>
      </div>
    </section>
  )
}
