import { Activity, CircleHelp, Gamepad2, SlidersHorizontal } from 'lucide-react'
import { BrainPreview } from '@/components/brain-preview'
import { FlyMark } from '@/components/fly-mark'
import { GamePreview } from '@/components/game-preview'
import { SessionTimeline } from '@/components/session-timeline'
import { SessionReadouts } from '@/components/session-readouts'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import './App.css'

export default function App() {
  return (
    <div className="app-shell">
      <header className="app-header">
        <a className="brand" href="/" aria-label="FlyArcade home">
          <span className="brand-mark">
            <FlyMark />
          </span>
          Fly<span>Arcade</span>
        </a>
        <Separator orientation="vertical" className="header-divider" />
        <span className="workspace-name">Neural simulation workspace</span>
        <Separator orientation="vertical" className="header-divider secondary-divider" />
        <div className="game-name">
          <Gamepad2 aria-hidden="true" />
          <h1>Fly Saber</h1>
        </div>
        <div className="header-actions">
          <Badge variant="outline" className="preview-badge">
            <span className="status-dot amber" />
            Static preview
          </Badge>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="ghost" size="icon-sm" aria-label="About this preview">
                <CircleHelp aria-hidden="true" />
              </Button>
            </TooltipTrigger>
            <TooltipContent className="help-tooltip">
              Original static illustrations. Live gameplay, movable 3D cameras, and neuron activity
              will be connected in a future build.
            </TooltipContent>
          </Tooltip>
        </div>
      </header>
      <main className="workspace">
        <section className="observation-workspace" aria-label="Fly Saber observation workspace">
          <GamePreview />
          <SessionTimeline />
          <SessionReadouts />
        </section>
        <aside className="inspector" aria-label="Brain and session inspector">
          <BrainPreview />
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
                <dd className="muted-value">Not connected</dd>
              </div>
              <div>
                <dt>Dataset</dt>
                <dd className="muted-value">Awaiting connectome</dd>
              </div>
              <div>
                <dt>Map / audio</dt>
                <dd className="muted-value">No map loaded</dd>
              </div>
              <div>
                <dt>Session status</dt>
                <dd>
                  <span className="status-dot amber" />
                  Preview only
                </dd>
              </div>
            </dl>
          </section>
          <div className="intervention-placeholder">
            <SlidersHorizontal aria-hidden="true" />
            <span>Brain controls</span>
            <Badge variant="outline">Coming later</Badge>
          </div>
          <p className="inspector-footnote">
            A visual preview of the workspace. Game state and neural activity will appear when a
            session is connected.
          </p>
        </aside>
      </main>
      <footer className="app-footer">
        <span>
          <span className="status-dot" />
          Backend not connected
        </span>
        <span>
          FlyArcade <span className="footer-slash">/</span> Visual input → neural activity → motor
          output
        </span>
        <span>
          STATIC BUILD <span className="footer-slash">/</span> 001
        </span>
      </footer>
    </div>
  )
}
