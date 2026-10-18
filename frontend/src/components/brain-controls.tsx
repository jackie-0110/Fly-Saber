import { useEffect, useRef, useState } from 'react'
import { create } from 'zustand'
import { Button } from '@/components/ui/button'
import { loadControls } from '@/brain/data'
import type { Controls } from '@/brain/data'
import { isUntouched, noControls } from '@/game/interventions'
import type { BrainControls } from '@/game/interventions'
import { useSession } from '@/game/session'
import { liveSession, usePlayer } from '@/game/store'

const LOOMING = 'looming_detectors'
const AROUSAL = 'arousal'
// Each change re-centres the brain's readout on the server, a few frames' work, so a
// dragged slider sends once it settles instead of on every step.
const SLIDER_SETTLE_MS = 250
// Every mood range in controls.json is a multiple of this, so the slider can land on 0.
const MOOD_STEP = 0.05

const useBrainControls = create<{
  controls: BrainControls
  change: (next: Partial<BrainControls>) => void
}>((set, get) => ({
  controls: noControls,
  change: (next) => {
    const controls = { ...get().controls, ...next }
    set({ controls })
    liveSession.setInterventions(controls)
  },
}))

export function BrainControlsPanel() {
  const [config, setConfig] = useState<Controls | null>(null)
  const [failed, setFailed] = useState(false)
  useEffect(() => {
    const controller = new AbortController()
    loadControls(controller.signal)
      .then(setConfig)
      .catch(() => {
        if (!controller.signal.aborted) setFailed(true)
      })
    return () => controller.abort()
  }, [])
  const controls = useBrainControls((s) => s.controls)
  const change = useBrainControls((s) => s.change)
  const brain = usePlayer((s) => s.brain)
  const controlsError = useSession((s) => s.controlsError)
  const looming = config?.sections.find((s) => s.group === LOOMING)
  const arousal = config?.moods.find((m) => m.name === AROUSAL)
  const disabled = !config || brain === 'expert'
  const lesioned = controls.lesion.includes(LOOMING)

  const message = failed
    ? 'Brain controls could not load.'
    : brain === 'expert'
      ? 'The scripted expert has no brain to change.'
      : controlsError
        ? `The server rejected that change: ${controlsError}.`
        : ''

  return (
    <section className="brain-controls" aria-label="Brain controls">
      <h2>Brain controls</h2>
      <Button
        variant="outline"
        size="xs"
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
        Lesion looming
      </Button>
      <Button
        variant="outline"
        size="xs"
        aria-label="Blindfold left eye"
        aria-pressed={controls.blindLeft}
        disabled={disabled}
        onClick={() => change({ blindLeft: !controls.blindLeft })}
      >
        Blind left
      </Button>
      <Button
        variant="outline"
        size="xs"
        aria-label="Blindfold right eye"
        aria-pressed={controls.blindRight}
        disabled={disabled}
        onClick={() => change({ blindRight: !controls.blindRight })}
      >
        Blind right
      </Button>
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
      <Button
        variant="ghost"
        size="xs"
        title="Controls carry over to the next song until reset."
        disabled={disabled || isUntouched(controls)}
        onClick={() => change(noControls)}
      >
        Reset controls
      </Button>
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
      <span>{mood.label}</span>
      <span className="mood-end">{mood.low_label}</span>
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
      <span className="mood-end">{mood.high_label}</span>
    </label>
  )
}
