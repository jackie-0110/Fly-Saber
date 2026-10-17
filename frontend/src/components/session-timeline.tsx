import { Clock3, Pause, Play, RotateCcw } from 'lucide-react'
import { Button } from '@/components/ui/button'

export function SessionTimeline() {
  return (
    <section className="session-timeline" aria-labelledby="timeline-heading">
      <div className="timeline-heading">
        <h2 id="timeline-heading">
          <Clock3 aria-hidden="true" />
          Session timeline
        </h2>
        <span>No session loaded</span>
      </div>
      <div className="timeline-controls">
        <div className="playback-buttons">
          <Button variant="outline" size="icon-sm" disabled aria-label="Play session">
            <Play aria-hidden="true" />
          </Button>
          <Button variant="outline" size="icon-sm" disabled aria-label="Pause session">
            <Pause aria-hidden="true" />
          </Button>
          <Button variant="outline" size="icon-sm" disabled aria-label="Restart session">
            <RotateCcw aria-hidden="true" />
          </Button>
          <span className="playback-speed">1×</span>
        </div>
        <div className="timeline-track" aria-label="Timeline unavailable until a session is loaded">
          <div className="timeline-rail">
            <span />
          </div>
          <div className="timeline-ticks">
            <span>00:00</span>
            <span>--:--</span>
            <span>--:--</span>
            <span>--:--</span>
            <span>--:--</span>
            <span>--:--</span>
          </div>
        </div>
        <div className="timeline-time">
          00:00<span>/ --:--</span>
        </div>
      </div>
    </section>
  )
}
