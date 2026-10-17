import { useRef, useState } from 'react'
import { Maximize2 } from 'lucide-react'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'

const SCORES = ['Accuracy', 'Combo', 'Score']

export function GamePreview() {
  const panelRef = useRef<HTMLElement>(null)
  const [fullscreenError, setFullscreenError] = useState('')

  async function toggleFullscreen() {
    try {
      if (document.fullscreenElement) await document.exitFullscreen()
      else await panelRef.current?.requestFullscreen()
      setFullscreenError('')
    } catch {
      setFullscreenError('Fullscreen is unavailable in this browser.')
    }
  }

  return (
    <section className="game-scene" ref={panelRef} aria-label="Game scene">
      <img
        className="game-art"
        src="/previews/fly-saber.svg"
        alt="Static illustration of a fruit fly holding coral and teal sabers on a four-lane track with approaching patterned rhythm blocks."
      />
      <dl className="scene-hud" aria-label="Performance">
        {SCORES.map((label) => (
          <div key={label}>
            <dt>{label}</dt>
            <dd aria-label="No data">–</dd>
          </div>
        ))}
      </dl>
      <div className="scene-footer">
        <ul className="saber-legend" aria-label="Saber colors">
          <li>
            <i className="swatch coral" aria-hidden="true" />
            Left saber
          </li>
          <li>
            <i className="swatch mint" aria-hidden="true" />
            Right saber
          </li>
        </ul>
        <Tooltip>
          <TooltipTrigger asChild>
            <button
              type="button"
              className="scene-button"
              onClick={toggleFullscreen}
              aria-label="Toggle fullscreen"
            >
              <Maximize2 aria-hidden="true" />
            </button>
          </TooltipTrigger>
          <TooltipContent>Fullscreen</TooltipContent>
        </Tooltip>
      </div>
      <p className="scene-error" role="status">
        {fullscreenError}
      </p>
    </section>
  )
}
