import { ChevronDown } from 'lucide-react'
import { BrainPanel } from '@/components/brain-panel'
import { FlyMark } from '@/components/fly-mark'
import { GamePreview } from '@/components/game-preview'
import { SessionTimeline } from '@/components/session-timeline'
import { SignalPath } from '@/components/signal-path'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import './App.css'

export default function App() {
  return (
    <div className="app-shell">
      <header className="app-header">
        <h1>
          <a className="brand" href="/" aria-label="FlySaber home">
            <FlyMark />
            FlySaber
          </a>
        </h1>
        <span className="product-line">Neural simulation workspace</span>
        <Popover>
          <PopoverTrigger className="connection-status">
            <span className="status-dot warn" aria-hidden="true" />
            Offline preview
            <ChevronDown aria-hidden="true" />
          </PopoverTrigger>
          <PopoverContent align="end" className="status-popover">
            <p className="status-popover-title">Simulation server not connected</p>
            <p>
              The game scene and brain are static illustrations. Gameplay, the 3D brain, and live
              neural activity appear once a session streams from the server.
            </p>
          </PopoverContent>
        </Popover>
      </header>
      <main className="workspace">
        <GamePreview />
        <SessionTimeline />
        <SignalPath />
        <BrainPanel />
      </main>
    </div>
  )
}
