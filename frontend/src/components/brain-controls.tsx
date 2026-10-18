import { useEffect, useRef, useState } from 'react'
import { create } from 'zustand'
import { RotateCcw } from 'lucide-react'
import { AROUSAL, LOOMING, useBrainControls } from '@/game/controls-state'
import { Button } from '@/components/ui/button'
import { loadControls } from '@/brain/data'
import type { Controls } from '@/brain/data'
import { isUntouched, noControls } from '@/game/interventions'
import { useSession } from '@/game/session'
import { usePlayer } from '@/game/store'
import { EyeMosaic } from '@/eyeview/eye-mosaic'

/*
 * Brain controls sit in the signal path where they act: the blindfolds on the two eyes in
 * Visual input, the lesion and mood in the Brain stage between input and motor output.
 */

// Each change re-centres the brain's readout on the server, a few frames' work, so a
// dragged slider sends once it settles instead of on every step.
const SLIDER_SETTLE_MS = 250
// Every mood range in controls.json is a multiple of this, so the slider can land on 0.
const MOOD_STEP = 0.05

// controls.json, loaded once and shared by both halves of the controls.
const useControlsConfig = create<{ config: Controls | null; failed: boolean }>(() => ({
  config: null,
  failed: false,
}))
let configRequest: Promise<void> | null = null
function requestConfig() {
  configRequest ??= loadControls(new AbortController().signal)
    .then((config) => useControlsConfig.setState({ config, failed: false }))
    .catch(() => {
      configRequest = null
      useControlsConfig.setState({ failed: true })
    })
}

function useControlsState() {
  const { config, failed } = useControlsConfig()
  useEffect(requestConfig, [])
  const controls = useBrainControls((s) => s.controls)
  const change = useBrainControls((s) => s.change)
  const brain = usePlayer((s) => s.brain)
  return { config, failed, controls, change, brain, disabled: !config || brain === 'expert' }
}

/** The fly's two eyes; pressing one blindfolds it. */
export function EyeToggles() {
  const { controls, change, disabled } = useControlsState()
  const eyes = [
    { side: 'left', label: 'L', blind: controls.blindLeft, key: 'blindLeft' },
    { side: 'right', label: 'R', blind: controls.blindRight, key: 'blindRight' },
  ] as const
  return (
    <div className="eye-toggles">
      {eyes.map((eye) => (
        <button
          key={eye.side}
          type="button"
          className="eye-toggle"
          aria-label={`Blindfold ${eye.side} eye`}
          aria-pressed={eye.blind}
          disabled={disabled}
          title={eye.blind ? `Remove the ${eye.side} blindfold` : `Blindfold the ${eye.side} eye`}
          onClick={() => change({ [eye.key]: !eye.blind })}
        >
          <EyeMosaic side={eye.side} blind={eye.blind} />
          <span>{eye.label}</span>
        </button>
      ))}
    </div>
  )
}

/** The Brain stage of the signal path: lesions and mood, between eyes and sabers. */
export function BrainStage() {
  const { config, failed, controls, change, brain, disabled } = useControlsState()
  const controlsError = useSession((s) => s.controlsError)
  const looming = config?.sections.find((s) => s.group === LOOMING)
  const arousal = config?.moods.find((m) => m.name === AROUSAL)
  const lesioned = controls.lesion.includes(LOOMING)

  const message = failed
    ? 'Brain controls could not load.'
    : brain === 'expert'
      ? 'The scripted expert has no brain to change.'
      : controlsError
        ? `The server rejected that change: ${controlsError}.`
        : ''

  return (
    <section className="readout-block brain-stage-block" aria-label="Brain controls">
      <div className="brain-stage-heading">
        <h2>Brain</h2>
        <Button
          variant="ghost"
          size="xs"
          className="brain-reset"
          title="Controls carry over to the next song until reset."
          disabled={disabled || isUntouched(controls)}
          onClick={() => change(noControls)}
        >
          <RotateCcw aria-hidden="true" />
          Reset controls
        </Button>
      </div>
      <button
        type="button"
        className="lesion-toggle"
        aria-label="Lesion looming detectors"
        aria-pressed={lesioned}
        disabled={disabled || !looming}
        title={looming?.description}
        onClick={() =>
          change({
            lesion: lesioned
              ? controls.lesion.filter((g) => g !== LOOMING)
              : [...controls.lesion, LOOMING],
          })
        }
      >
        <span className="lesion-box" aria-hidden="true" />
        Lesion looming detectors
      </button>
      {arousal && (
        <MoodSlider
          mood={arousal}
          level={controls.mood[AROUSAL] ?? 0}
          disabled={disabled}
          onCommit={(level) =>
            change({ mood: { ...useBrainControls.getState().controls.mood, [AROUSAL]: level } })
          }
        />
      )}
      <p className="brain-controls-status" role="status">
        {message}
      </p>
    </section>
  )
}

function MoodSlider({
  mood,
  level,
  disabled,
  onCommit,
}: {
  mood: Controls['moods'][number]
  level: number
  disabled: boolean
  onCommit: (level: number) => void
}) {
  const [draft, setDraft] = useState<number | null>(null)
  const timer = useRef<ReturnType<typeof setTimeout>>(undefined)
  useEffect(() => () => clearTimeout(timer.current), [])
  return (
    <label className="mood-slider" title={mood.description}>
      <span className="mood-name">{mood.label}</span>
      <input
        type="range"
        min={mood.min}
        max={mood.max}
        step={MOOD_STEP}
        value={draft ?? level}
        disabled={disabled}
        onChange={(event) => {
          const next = Number(
            (Math.round(Number(event.target.value) / MOOD_STEP) * MOOD_STEP).toFixed(2),
          )
          setDraft(next)
          clearTimeout(timer.current)
          timer.current = setTimeout(() => {
            onCommit(next)
            setDraft(null)
          }, SLIDER_SETTLE_MS)
        }}
      />
      <span className="mood-ends">
        <span>{mood.low_label}</span>
        <span>{mood.high_label}</span>
      </span>
    </label>
  )
}
