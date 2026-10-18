import { useCallback, useEffect, useRef, useState } from 'react'
import type { KeyboardEvent, PointerEvent, ReactNode } from 'react'

/*
 * A window that floats over the page: drag its title bar to move it, drag any edge or
 * corner to resize it, collapse it to the title bar. It snaps to the screen edges, stays
 * on screen when the browser resizes, and remembers where it was in localStorage.
 */

export interface Frame {
  x: number
  y: number
  w: number
  h: number
}

const MIN_W = 300
const MIN_H = 220
const MARGIN = 8
// Within this distance of a screen edge, a moved window lands on the edge.
const SNAP = 16
// Arrow keys move (or with Shift, resize) a focused title bar by this much.
const KEY_STEP = 16
const EDGES = ['n', 's', 'e', 'w', 'ne', 'nw', 'se', 'sw'] as const
type Edge = (typeof EDGES)[number]

function readFrame(key: string): Frame | null {
  try {
    const raw = JSON.parse(localStorage.getItem(key) ?? 'null')
    if (raw && ['x', 'y', 'w', 'h'].every((k) => Number.isFinite(raw[k]))) return raw as Frame
  } catch {
    // Unreadable or blocked storage: start from the default frame.
  }
  return null
}

function writeFrame(key: string, frame: Frame) {
  try {
    localStorage.setItem(key, JSON.stringify(frame))
  } catch {
    // Storage can be blocked; the window then forgets its place on reload.
  }
}

/** Keep the window on screen: size within the viewport, and its title bar always reachable. */
function clamp(frame: Frame): Frame {
  const vw = window.innerWidth
  const vh = window.innerHeight
  const w = Math.min(Math.max(frame.w, MIN_W), vw - 2 * MARGIN)
  const h = Math.min(Math.max(frame.h, MIN_H), vh - 2 * MARGIN)
  const x = Math.min(Math.max(frame.x, MARGIN), vw - w - MARGIN)
  const y = Math.min(Math.max(frame.y, MARGIN), vh - h - MARGIN)
  return { x, y, w, h }
}

function snap(frame: Frame): Frame {
  const vw = window.innerWidth
  const vh = window.innerHeight
  let { x, y } = frame
  if (x - MARGIN < SNAP) x = MARGIN
  if (vw - MARGIN - (x + frame.w) < SNAP) x = vw - MARGIN - frame.w
  if (y - MARGIN < SNAP) y = MARGIN
  if (vh - MARGIN - (y + frame.h) < SNAP) y = vh - MARGIN - frame.h
  return { ...frame, x, y }
}

function resized(start: Frame, edge: Edge, dx: number, dy: number): Frame {
  let { x, y, w, h } = start
  if (edge.includes('e')) w = start.w + dx
  if (edge.includes('s')) h = start.h + dy
  if (edge.includes('w')) {
    w = Math.max(MIN_W, start.w - dx)
    x = start.x + start.w - w
  }
  if (edge.includes('n')) {
    h = Math.max(MIN_H, start.h - dy)
    y = start.y + start.h - h
  }
  return { x, y, w, h }
}

export function FloatingWindow({
  storageKey,
  defaultFrame,
  label,
  title,
  collapsed,
  children,
}: {
  /** localStorage key for the window's last position and size. */
  storageKey: string
  /** Where the window first opens, computed when it mounts. */
  defaultFrame: () => Frame
  label: string
  /** The title bar's contents; dragging anywhere on it but its buttons moves the window. */
  title: ReactNode
  /** Collapsed, only the title bar shows. */
  collapsed: boolean
  children: ReactNode
}) {
  const [frame, setFrame] = useState<Frame>(() => clamp(readFrame(storageKey) ?? defaultFrame()))
  const gesture = useRef<{
    edge: Edge | 'move'
    px: number
    py: number
    start: Frame
  } | null>(null)
  const root = useRef<HTMLElement>(null)

  const commit = useCallback(
    (next: Frame) => {
      setFrame(next)
      writeFrame(storageKey, next)
    },
    [storageKey],
  )

  useEffect(() => {
    root.current?.focus()
    const onResize = () => setFrame((current) => clamp(current))
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [])

  const begin = (edge: Edge | 'move') => (event: PointerEvent<HTMLElement>) => {
    if (event.button !== 0) return
    // Buttons in the title bar stay buttons.
    if (edge === 'move' && (event.target as HTMLElement).closest('button')) return
    event.preventDefault()
    event.currentTarget.setPointerCapture(event.pointerId)
    gesture.current = { edge, px: event.clientX, py: event.clientY, start: frame }
  }
  const move = (event: PointerEvent<HTMLElement>) => {
    const g = gesture.current
    if (!g) return
    const dx = event.clientX - g.px
    const dy = event.clientY - g.py
    setFrame(
      g.edge === 'move'
        ? snap(clamp({ ...g.start, x: g.start.x + dx, y: g.start.y + dy }))
        : clamp(resized(g.start, g.edge, dx, dy)),
    )
  }
  const end = () => {
    if (!gesture.current) return
    gesture.current = null
    setFrame((current) => {
      writeFrame(storageKey, current)
      return current
    })
  }
  const onTitleKey = (event: KeyboardEvent<HTMLElement>) => {
    if ((event.target as HTMLElement).closest('button')) return
    const step = {
      ArrowLeft: [-KEY_STEP, 0],
      ArrowRight: [KEY_STEP, 0],
      ArrowUp: [0, -KEY_STEP],
      ArrowDown: [0, KEY_STEP],
    }[event.key]
    if (!step) return
    event.preventDefault()
    const [dx, dy] = step
    commit(
      clamp(
        event.shiftKey
          ? { ...frame, w: frame.w + dx, h: frame.h + dy }
          : { ...frame, x: frame.x + dx, y: frame.y + dy },
      ),
    )
  }

  return (
    <section
      ref={root}
      className="floating-window"
      data-collapsed={collapsed || undefined}
      role="dialog"
      aria-modal="false"
      aria-label={label}
      tabIndex={-1}
      style={{
        left: frame.x,
        top: frame.y,
        width: frame.w,
        height: collapsed ? undefined : frame.h,
      }}
    >
      <header
        className="floating-window-title"
        tabIndex={0}
        aria-label={`${label}: drag to move, arrow keys move, Shift and arrows resize`}
        onPointerDown={begin('move')}
        onPointerMove={move}
        onPointerUp={end}
        onPointerCancel={end}
        onKeyDown={onTitleKey}
      >
        {title}
      </header>
      {!collapsed && children}
      {!collapsed &&
        EDGES.map((edge) => (
          <span
            key={edge}
            className="resize-handle"
            data-edge={edge}
            aria-hidden="true"
            onPointerDown={begin(edge)}
            onPointerMove={move}
            onPointerUp={end}
            onPointerCancel={end}
          />
        ))}
    </section>
  )
}
