import { BackendStatus } from '@/components/backend-status'
import { BrainPreview } from '@/components/brain-preview'
import { FlyMark } from '@/components/fly-mark'
import { GamePreview } from '@/components/game-preview'
import { SessionTimeline } from '@/components/session-timeline'
import { SessionReadouts } from '@/components/session-readouts'
import { Separator } from '@/components/ui/separator'
import './App.css'

export default function App() {
  return (
    <div className="app-shell">
      <header className="app-header">
        <a className="brand" href="/" aria-label="Fly Saber home">
          <span className="brand-mark">
            <FlyMark />
          </span>
          Fly<span className="brand-word">Saber</span>
        </a>
        <Separator orientation="vertical" className="header-divider" />
        <span className="workspace-name">Neural simulation workspace</span>
        <div className="header-actions">
          <BackendStatus />
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
        </aside>
      </main>
    </div>
  )
}
