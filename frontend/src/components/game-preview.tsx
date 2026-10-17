import { Component, lazy, Suspense, useCallback, useRef, useState } from 'react'
import type { KeyboardEvent, ReactNode } from 'react'
import {
  Expand,
  Eye,
  EyeOff,
  LoaderCircle,
  Maximize,
  Orbit,
  RotateCcw,
  ScanLine,
  Video,
  X,
  ZoomIn,
  ZoomOut,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { openSongLibrary, usePlayer } from '@/game/store'
import type { GameCameraAction } from '@/game/game-scene'
import { FlyMotionPreview } from '@/game/fly-motion-preview'
import { useSession } from '@/game/session'

const GameScene = lazy(() => import('@/game/game-scene'))

class GameBoundary extends Component<
  { children: ReactNode; fallback: ReactNode },
  { failed: boolean }
> {
  state = { failed: false }
  static getDerivedStateFromError() {
    return { failed: true }
  }
  render() {
    return this.state.failed ? this.props.fallback : this.props.children
  }
}

function CameraButton({
  label,
  children,
  onClick,
  disabled = false,
}: {
  label: string
  children: ReactNode
  onClick: () => void
  disabled?: boolean
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          variant="ghost"
          size="icon-sm"
          aria-label={label}
          onClick={onClick}
          disabled={disabled}
        >
          {children}
        </Button>
      </TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  )
}

export function GamePreview() {
  const [motionPreview] = useState(
    () => new URLSearchParams(window.location.search).get('flyMotion') === 'preview',
  )
  const panelRef = useRef<HTMLDivElement>(null)
  const [fullscreenError, setFullscreenError] = useState('')
  const [sceneAttempt, setSceneAttempt] = useState(0)
  const [contextLost, setContextLost] = useState(false)
  const [cameraReady, setCameraReady] = useState(false)
  const [showFly, setShowFly] = useState(true)
  const chart = usePlayer((s) => s.chart)
  const loading = usePlayer((s) => s.loading)
  const error = usePlayer((s) => s.error)
  const sessionStatus = useSession((s) => s.status)
  const navigate = useRef<((action: GameCameraAction) => void) | null>(null)
  const onNavigateReady = useCallback((handler: ((action: GameCameraAction) => void) | null) => {
    navigate.current = handler
    setCameraReady(handler !== null)
  }, [])
  const onFailure = useCallback(() => {
    setContextLost(true)
    setCameraReady(false)
  }, [])
  const retry = () => {
    setContextLost(false)
    setSceneAttempt((value) => value + 1)
  }

  async function expandScene() {
    try {
      if (document.fullscreenElement) await document.exitFullscreen()
      else await panelRef.current?.requestFullscreen()
      setFullscreenError('')
    } catch {
      setFullscreenError('Fullscreen is unavailable in this browser.')
    }
  }

  function cameraKey(event: KeyboardEvent<HTMLDivElement>) {
    if (event.target !== event.currentTarget) return
    const action: Record<string, GameCameraAction> = {
      ArrowLeft: 'left',
      ArrowRight: 'right',
      ArrowUp: 'up',
      ArrowDown: 'down',
      '+': 'in',
      '=': 'in',
      '-': 'out',
      Home: 'reset',
    }
    if (action[event.key]) {
      event.preventDefault()
      navigate.current?.(action[event.key])
    } else if (event.code === 'Space' && chart) {
      event.preventDefault()
      const player = usePlayer.getState()
      if (player.playing) player.pause()
      else void player.play()
    }
  }

  const fallback = (
    <div className="game-scene-fallback" role="status">
      <Orbit aria-hidden="true" />
      <h3>The 3D track is unavailable.</h3>
      <p>This view needs WebGL 2. Song selection, audio, and the timeline still work.</p>
      <Button variant="outline" size="sm" onClick={retry}>
        Retry 3D view
      </Button>
    </div>
  )

  return (
    <Card className="game-viewport" ref={panelRef} role="region" aria-labelledby="game-heading">
      <div
        className="game-stage"
        tabIndex={0}
        role="region"
        aria-label="Interactive 3D game view"
        aria-describedby="game-camera-help"
        onKeyDown={cameraKey}
      >
        {contextLost ? (
          fallback
        ) : (
          <GameBoundary key={sceneAttempt} fallback={fallback}>
            <Suspense
              fallback={
                <div className="game-scene-fallback" role="status">
                  <LoaderCircle className="spin" />
                  <p>Preparing the 3D track…</p>
                </div>
              }
            >
              <GameScene
                chart={chart}
                showFly={showFly}
                motionPreview={motionPreview}
                onNavigateReady={onNavigateReady}
                onFailure={onFailure}
              />
            </Suspense>
          </GameBoundary>
        )}
      </div>
      <div className="scene-topline">
        <div className="scene-title">
          <ScanLine aria-hidden="true" />
          <h2 id="game-heading">Game viewport</h2>
          {sessionStatus !== 'idle' && (
            <span className="live-session-status" role="status">
              {sessionStatus === 'connecting'
                ? 'Connecting live session…'
                : `Live · ${sessionStatus}`}
            </span>
          )}
        </div>
      </div>
      {motionPreview && <FlyMotionPreview />}
      {!motionPreview && !chart && !loading && cameraReady && (
        <div className="scene-empty">
          <h2>Choose a song to begin</h2>
          <p>Watch the fly play a live song or explore a community chart.</p>
          <Button size="sm" onClick={openSongLibrary}>
            Choose a song
          </Button>
        </div>
      )}
      {(loading || error) && (
        <div className={`song-notice${error ? ' is-error' : ''}`} role="status">
          {loading ? (
            <LoaderCircle className="spin" aria-hidden="true" />
          ) : (
            <ScanLine aria-hidden="true" />
          )}
          <p>{loading || error}</p>
          <Button
            variant="ghost"
            size="icon-xs"
            aria-label={loading ? 'Cancel song loading' : 'Dismiss song error'}
            onClick={() =>
              loading ? usePlayer.getState().cancelLoad() : usePlayer.getState().clearError()
            }
          >
            <X aria-hidden="true" />
          </Button>
        </div>
      )}
      {chart && chart.warnings.length > 0 && (
        <details className="chart-warnings">
          <summary>Map compatibility notes ({chart.warnings.length})</summary>
          {chart.warnings.map((warning) => (
            <p key={warning}>{warning}</p>
          ))}
        </details>
      )}
      <div className="scene-bottomline">
        <div className="saber-legend">
          <span>
            <i className="coral" />
            Left notes
          </span>
          <span>
            <i className="mint" />
            Right notes
          </span>
        </div>
        <div className="camera-toolbar" aria-label="Game camera controls">
          <CameraButton
            label="Reset game camera"
            disabled={!cameraReady}
            onClick={() => navigate.current?.('reset')}
          >
            <RotateCcw />
          </CameraButton>
          <CameraButton
            label="Track overview"
            disabled={!cameraReady}
            onClick={() => navigate.current?.('overview')}
          >
            <Maximize />
          </CameraButton>
          <CameraButton
            label="Player view"
            disabled={!cameraReady}
            onClick={() => navigate.current?.('player')}
          >
            <Video />
          </CameraButton>
          <span className="camera-tool-divider" />
          <CameraButton
            label="Zoom game camera in"
            disabled={!cameraReady}
            onClick={() => navigate.current?.('in')}
          >
            <ZoomIn />
          </CameraButton>
          <CameraButton
            label="Zoom game camera out"
            disabled={!cameraReady}
            onClick={() => navigate.current?.('out')}
          >
            <ZoomOut />
          </CameraButton>
          <CameraButton
            label={showFly ? 'Hide fly model' : 'Show fly model'}
            disabled={!cameraReady}
            onClick={() => setShowFly((value) => !value)}
          >
            {showFly ? <Eye /> : <EyeOff />}
          </CameraButton>
          <CameraButton label="Toggle game view fullscreen" onClick={expandScene}>
            <Expand />
          </CameraButton>
        </div>
      </div>
      <p className="game-camera-help" id="game-camera-help">
        Drag to orbit · Scroll to zoom · Right-drag to pan
        <span>Focus view: arrows rotate, +/− zoom, Home resets, Space plays.</span>
      </p>
      <p className="fullscreen-status" role="status">
        {fullscreenError}
      </p>
    </Card>
  )
}
