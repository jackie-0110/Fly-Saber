import { useEffect, useRef, useState } from 'react'

// Each brain or eye change re-centres the brain's readout on the server, a few frames'
// work, so a dragged slider sends once it settles instead of on every step.
const SLIDER_SETTLE_MS = 250

/** A labelled slider for one control; commits after the drag settles. */
export function RangeControl({
  label,
  min,
  max,
  step,
  value,
  disabled,
  onCommit,
  lowLabel,
  highLabel,
  format = (v) => v.toFixed(2),
  title,
  className = 'range-control',
  note,
}: {
  label: string
  min: number
  max: number
  step: number
  value: number
  disabled?: boolean
  onCommit: (value: number) => void
  lowLabel?: string
  highLabel?: string
  format?: (value: number) => string
  title?: string
  className?: string
  /** A line under the slider, such as why it has no effect right now. */
  note?: string
}) {
  const [draft, setDraft] = useState<number | null>(null)
  const timer = useRef<ReturnType<typeof setTimeout>>(undefined)
  useEffect(() => () => clearTimeout(timer.current), [])
  const shown = draft ?? value
  // A preset can set a level past the slider's ends; say so rather than clamp silently.
  const outside = shown < min || shown > max
  return (
    <label className={className} title={title}>
      <span className="range-name">{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={Math.min(max, Math.max(min, shown))}
        disabled={disabled}
        onChange={(event) => {
          const next = Number((Math.round(Number(event.target.value) / step) * step).toFixed(3))
          setDraft(next)
          clearTimeout(timer.current)
          timer.current = setTimeout(() => {
            onCommit(next)
            setDraft(null)
          }, SLIDER_SETTLE_MS)
        }}
      />
      <span className="range-value" data-outside={outside || undefined}>
        {format(shown)}
      </span>
      {(lowLabel || highLabel) && (
        <span className="range-ends">
          <span>{lowLabel}</span>
          <span>{highLabel}</span>
        </span>
      )}
      {(note || outside) && (
        <span className="range-note">{note ?? 'Set by a preset, past the slider’s range'}</span>
      )}
    </label>
  )
}
