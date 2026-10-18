import { RotateCcw } from 'lucide-react'
import {
  AROUSAL,
  LOOMING,
  useBrainControls,
  useControlsConfigState,
} from '@/features/session/controls-state'
import { Button } from '@/components/ui/button'
import { RangeControl } from '@/components/range-control'
import { isUntouched, noControls } from '@/features/session/interventions'
import { useSession } from '@/features/session/session'
import { usePlayer } from '@/features/session/store'
import { EyeMosaic } from '@/features/eye/eye-mosaic'

/*
 * Brain controls sit in the signal path where they act: the blindfolds on the two eyes in
 * Visual input, the lesion and mood in the Brain stage between input and motor output.
 * The Brain stage keeps the two demo favourites; the header's Brain controls button opens
 * the full panel.
 */

// Every mood range in controls.json is a multiple of this, so the slider can land on 0.
const MOOD_STEP = 0.05

function useControlsState() {
  const { config, failed } = useControlsConfigState()
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
          size="icon-xs"
          className="brain-reset"
          aria-label="Reset controls"
          title="Reset every brain and eye control. They carry over to the next song until reset."
          disabled={disabled || isUntouched(controls)}
          onClick={() => change(noControls)}
        >
          <RotateCcw aria-hidden="true" />
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
        <RangeControl
          className="mood-slider"
          label={arousal.label}
          min={arousal.min}
          max={arousal.max}
          step={MOOD_STEP}
          value={controls.mood[AROUSAL] ?? 0}
          disabled={disabled}
          lowLabel={arousal.low_label}
          highLabel={arousal.high_label}
          format={(v) => (v > 0 ? `+${v.toFixed(2)}` : v.toFixed(2))}
          title={arousal.description}
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
