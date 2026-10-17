import { useEffect, useRef, useState } from 'react'
import { previewMotion } from './fly-motion'
import type { CutDirection, MotorFrame, SaberAction } from './fly-motion'

const moves = ['Idle', 'Reach', 'Cut up', 'Cut down', 'Cut left', 'Cut right', 'All movements']

/** Explicit opt-in art review, never a source of session actions or scores. */
export function FlyMotionPreview() {
  const [move, setMove] = useState(6)
  const [cell, setCell] = useState(5)
  const [hand, setHand] = useState('both')
  const [paused, setPaused] = useState(false)
  const pausedRef = useRef(false)
  useEffect(() => {
    previewMotion.reset()
    let last = performance.now() / 1000
    let elapsed = 0
    let step = 0
    const tick = () => {
      const now = performance.now() / 1000
      const delta = now - last
      last = now
      if (pausedRef.current) return
      elapsed += delta
      const age = elapsed
      const cycle = Math.floor(age / 1.1)
      const phase = age % 1.1
      const all = move === 6
      const idle = move === 0 || (all && cycle % 7 === 0)
      const target = all ? cycle % 12 : cell
      const cut =
        idle || move === 1 || phase < 0.4 || phase > 0.7
          ? null
          : ((all ? cycle % 4 : move - 2) as CutDirection)
      const sabers: SaberAction[] = [0, 1].map((color) => ({
        color: color as 0 | 1,
        cell:
          idle || (hand !== 'both' && hand !== String(color))
            ? null
            : color === 1 && all
              ? 11 - target
              : target,
        cut: hand !== 'both' && hand !== String(color) ? null : cut,
      }))
      const frame: MotorFrame = {
        type: 'frame',
        slot: 0,
        render_state: { t: age, step: step++, sabers },
      }
      previewMotion.acceptMessage(frame)
    }
    tick()
    const timer = window.setInterval(tick, 1000 / 30)
    return () => {
      clearInterval(timer)
      previewMotion.reset()
    }
  }, [move, cell, hand])
  useEffect(() => {
    pausedRef.current = paused
    previewMotion.setPaused(paused)
  }, [paused, move, cell, hand])
  return (
    <section className="fly-motion-preview" aria-label="Fly animation preview">
      <strong>Animation preview</strong>
      <span>Sample movements · no backend policy</span>
      <label>
        Movement
        <select
          aria-label="Preview movement"
          value={move}
          onChange={(e) => setMove(Number(e.target.value))}
        >
          {moves.map((label, index) => (
            <option key={label} value={index}>
              {label}
            </option>
          ))}
        </select>
      </label>
      <label>
        Saber
        <select aria-label="Preview saber" value={hand} onChange={(e) => setHand(e.target.value)}>
          <option value="both">Both</option>
          <option value="0">Left · red</option>
          <option value="1">Right · blue</option>
        </select>
      </label>
      <label>
        Target cell
        <select
          aria-label="Preview target cell"
          value={cell}
          disabled={move === 6 || move === 0}
          onChange={(e) => setCell(Number(e.target.value))}
        >
          {Array.from({ length: 12 }, (_, index) => (
            <option key={index} value={index}>
              {index} · lane {(index % 4) + 1}, row {Math.floor(index / 4) + 1}
            </option>
          ))}
        </select>
      </label>
      <button type="button" onClick={() => setPaused((value) => !value)}>
        {paused ? 'Resume preview' : 'Pause preview'}
      </button>
    </section>
  )
}
