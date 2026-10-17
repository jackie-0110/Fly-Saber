import { useRef, useState } from 'react'
import { Expand, Move, Orbit, ScanLine, Video } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'

export function GamePreview() {
  const panelRef = useRef<HTMLDivElement>(null)
  const [fullscreenError, setFullscreenError] = useState('')

  async function expandScene() {
    try {
      if (document.fullscreenElement) await document.exitFullscreen()
      else await panelRef.current?.requestFullscreen()
      setFullscreenError('')
    } catch {
      setFullscreenError('Fullscreen is unavailable in this browser.')
    }
  }

  return (
    <Card className="game-viewport" ref={panelRef} role="region" aria-labelledby="game-heading">
      <img
        className="game-art"
        src="/previews/fly-saber.svg"
        alt="Static illustration of a fruit fly holding coral and teal sabers on a four-lane track with approaching patterned rhythm blocks."
      />
      <div className="scene-topline">
        <div className="scene-title">
          <ScanLine aria-hidden="true" />
          <h2 id="game-heading">Game viewport</h2>
          <span className="scene-divider" />
          <span>Fly Saber</span>
        </div>
        <Badge variant="outline" className="scene-label">
          <span className="status-dot" />
          STATIC SCENE
        </Badge>
      </div>
      <div className="orientation-marker" aria-hidden="true">
        <span>FORWARD</span>
        <div>↑</div>
        <small>Z</small>
      </div>
      <div className="scene-bottomline">
        <div className="saber-legend">
          <span>
            <i className="coral" />
            Left saber
          </span>
          <span>
            <i className="mint" />
            Right saber
          </span>
        </div>
        <div className="camera-toolbar" aria-label="Game view controls">
          <span className="camera-view-label">
            <Video aria-hidden="true" />
            Third person
          </span>
          <Tooltip>
            <TooltipTrigger asChild>
              <span
                className="disabled-control"
                tabIndex={0}
                aria-label="Game camera orbit and pan will be available with the 3D scene"
              >
                <Button variant="ghost" size="icon-sm" disabled aria-label="Orbit game camera">
                  <Orbit aria-hidden="true" />
                </Button>
              </span>
            </TooltipTrigger>
            <TooltipContent>Orbit, pan, and zoom are planned for the 3D game scene.</TooltipContent>
          </Tooltip>
          <Button variant="ghost" size="icon-sm" disabled aria-label="Pan game camera">
            <Move aria-hidden="true" />
          </Button>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="icon-sm"
                onClick={expandScene}
                aria-label="Toggle game view fullscreen"
              >
                <Expand aria-hidden="true" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Expand game view</TooltipContent>
          </Tooltip>
        </div>
      </div>
      <p className="fullscreen-status" role="status">
        {fullscreenError}
      </p>
    </Card>
  )
}
