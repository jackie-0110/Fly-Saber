import { Crosshair, Eye, Gauge } from 'lucide-react'

export function SessionReadouts() {
  return (
    <section className="session-readouts" aria-label="Session readouts">
      <div className="readout-block">
        <h2>
          <Gauge aria-hidden="true" />
          Performance
        </h2>
        <div className="performance-values">
          <div>
            <strong>--</strong>
            <span>ACCURACY</span>
          </div>
          <div>
            <strong>--</strong>
            <span>COMBO</span>
          </div>
          <div>
            <strong>--</strong>
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
          Awaiting simulation
        </p>
        <span className="readout-caption">Saber actions will appear here.</span>
      </div>
    </section>
  )
}
