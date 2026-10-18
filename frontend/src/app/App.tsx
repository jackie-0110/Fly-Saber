import { BackendStatus } from '@/features/session/components/backend-status'
import {
  BrainControlsButton,
  BrainControlsPanel,
} from '@/features/brain/components/brain-controls-panel'
import { BrainPreview } from '@/features/brain/components/brain-preview'
import { FlyMark } from '@/components/fly-mark'
import { GamePreview } from '@/features/game/components/game-preview'
import { SessionTimeline } from '@/features/session/components/session-timeline'
import { SessionReadouts } from '@/features/session/components/session-readouts'
import { SongHeader } from '@/features/songs/components/song-header'
import { ThemeToggle } from '@/components/theme-toggle'
import './plate.css'

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
        <span className="workspace-name">Connectome-constrained visuomotor control</span>
        <div className="header-actions">
          <BrainControlsButton />
          <BackendStatus />
          <ThemeToggle />
        </div>
      </header>
      <main className="workspace">
        <section className="observation-workspace" aria-label="Fly Saber observation workspace">
          <SongHeader />
          <GamePreview />
          <div className="transport">
            <SessionTimeline />
            <SessionReadouts />
          </div>
        </section>
        <aside className="inspector" aria-label="Brain and session inspector">
          <BrainPreview />
        </aside>
      </main>
      <BrainControlsPanel />
    </div>
  )
}
