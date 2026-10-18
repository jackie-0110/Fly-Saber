import { useEffect, useRef, useState } from 'react'
import { Maximize2, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { EyeToggles } from '@/components/brain-controls'
import { usePlayer } from '@/game/store'
import { useSession } from '@/game/session'
import { EyeMosaic } from './eye-mosaic'
import { matchesLattice } from './eye-data'
import type { EyeFrame } from './eye-data'
import { loadEyeLattice, useEyeView } from './eye-state'

function EyePair({ labels = false }: { labels?: boolean }) {
  return (
    <div className="eye-pair">
      {(['left', 'right'] as const).map((side) => (
        <div key={side}>
          <EyeMosaic side={side} />
          {labels && <span>{side === 'left' ? 'Left eye' : 'Right eye'}</span>}
        </div>
      ))}
    </div>
  )
}

function EyeStatus() {
  const status = useSession((s) => s.status)
  const lattice = useEyeView((s) => s.lattice)
  const frame = useEyeView((s) => s.frames[0] ?? null)
  const error = useEyeView((s) => s.error)
  const brain = usePlayer((s) => s.brain)
  const [staleFrame, setStaleFrame] = useState<EyeFrame | null>(null)
  useEffect(() => {
    const timer = setTimeout(() => setStaleFrame(frame), 1000)
    return () => clearTimeout(timer)
  }, [frame])
  const stale = frame !== null && frame === staleFrame
  const mismatch = frame && lattice && !matchesLattice(frame, lattice)
  const live = !!frame && status === 'playing' && !stale && !mismatch
  const label = error
    ? 'Eye layout unavailable'
    : mismatch
      ? 'Eye image does not match layout'
      : brain === 'expert'
        ? 'Expert has no eye stream'
        : frame
          ? status === 'paused'
            ? 'Paused eye image'
            : status === 'ended'
              ? 'Final eye image'
              : stale
                ? 'Waiting for eye images'
                : "Live fly's-eye view"
          : "Awaiting fly's-eye stream"
  return (
    <p className="readout-status" role="status">
      <span className={`status-dot${live ? ' green' : ''}`} />
      {label}
    </p>
  )
}

/** Keeps the lattice loaded for both the readout and the game inset. */
export function EyeReadout() {
  const brain = usePlayer((s) => s.brain)
  const error = useEyeView((s) => s.error)
  const expanded = useEyeView((s) => s.expanded)
  const [attempt, setAttempt] = useState(0)
  useEffect(() => {
    const controller = new AbortController()
    void loadEyeLattice(brain, controller.signal)
    return () => controller.abort()
  }, [brain, attempt])
  return (
    <>
      <h2>Visual input</h2>
      <div className="eye-readout-controls">
        <EyeToggles />
        <button
          type="button"
          className="eye-expand"
          aria-label="Enlarge fly's-eye view"
          aria-expanded={expanded}
          aria-controls="fly-eye-inset"
          onClick={() => useEyeView.setState({ expanded: !expanded })}
        >
          <Maximize2 aria-hidden="true" />
        </button>
      </div>
      <div className="eye-readout-copy">
        <EyeStatus />
        {error && (
          <button type="button" className="eye-retry" onClick={() => setAttempt((n) => n + 1)}>
            Retry eye layout
          </button>
        )}
      </div>
    </>
  )
}

/** A nonmodal inset leaves the song, timeline and brain controls usable. */
export function EyeInset() {
  const expanded = useEyeView((s) => s.expanded)
  const close = useRef<HTMLButtonElement>(null)
  useEffect(() => {
    if (!expanded) return
    close.current?.focus({ preventScroll: true })
    const escape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        dismiss()
      }
    }
    document.addEventListener('keydown', escape)
    return () => document.removeEventListener('keydown', escape)
  }, [expanded])
  function dismiss() {
    useEyeView.setState({ expanded: false })
    document.querySelector<HTMLButtonElement>('.eye-expand')?.focus({ preventScroll: true })
  }
  if (!expanded) return null
  return (
    <section className="eye-inset" id="fly-eye-inset" aria-label="Fly's-eye view">
      <div className="eye-inset-header">
        <h2>Fly's-eye view</h2>
        <Button
          ref={close}
          variant="ghost"
          size="icon-xs"
          aria-label="Close fly's-eye view"
          onClick={dismiss}
        >
          <X />
        </Button>
      </div>
      <EyePair labels />
      <EyeStatus />
      <p className="eye-inset-caption">
        Luminance after eye controls. Lost contrast appears as gray.
      </p>
    </section>
  )
}
