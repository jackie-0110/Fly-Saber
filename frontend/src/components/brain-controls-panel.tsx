import { useEffect, useRef } from 'react'
import { RotateCcw, X, Zap } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { RangeControl } from '@/components/range-control'
import type { ControlSection, Controls, MoodPreset } from '@/brain/data'
import {
  describeControls,
  GIANT_FIBER,
  useBrainControls,
  useControlsConfigState,
} from '@/game/controls-state'
import { isUntouched, noControls } from '@/game/interventions'
import type { BrainControls } from '@/game/interventions'
import { useSession } from '@/game/session'
import { usePlayer } from '@/game/store'

/*
 * The full set of brain and eye controls (spec features 3 to 6), opened from the Brain
 * stage over the inspector so the game view stays in sight. Every label, description,
 * range and preset comes from controls.json; only the pathway grouping lives here.
 */

// Sections in signal order, eye to motor, then the modulators the moods drive.
const PATHWAYS: { name: string; groups: string[] }[] = [
  { name: 'Eyes', groups: ['eye_left', 'eye_right'] },
  {
    name: 'Visual detectors',
    groups: ['motion_detectors', 'looming_detectors', 'LC_all', 'lobula_plate_tangential'],
  },
  { name: 'Central brain', groups: ['intermediates'] },
  { name: 'Motor', groups: ['giant_fiber', 'DN_all'] },
  { name: 'Modulators', groups: ['octopamine', 'dopamine', 'serotonin'] },
]

// Eye-control ranges, measured live on the real brain (docs/RESULTS.md, eye controls). The
// trained fly is very sensitive to degraded vision: untouched it hit 34.5% of a window, blur
// 2 / 4 / 6 deg gave 20 / 13 / 6%, 10% of facets lost 11% and 20% lost 0%, visual noise
// 0.02 / 0.05 / 0.1 gave 19 / 13 / 7%. Each slider ends where the fly stops playing, so its
// whole travel shows a gradient instead of an instant collapse.
const EYE = {
  blurDeg: { max: 8, step: 0.5 },
  facetDropout: { max: 0.25, step: 0.01 },
  eyeNoise: { max: 0.15, step: 0.005 },
}
const STIMULATION_STEP = 0.5
const MOOD_STEP = 0.05
// Neural noise is a switch, not a slider: measured live, any level from 0.002 to 0.5 took
// the real brain from 34.5% to about 2% of notes hit, because the motor readout scales
// descending-neuron rates by their tiny resting spread and amplifies the noise with them.
const NOISE_ON = 0.05

function groupedSections(config: Controls) {
  const known = new Set(PATHWAYS.flatMap((p) => p.groups))
  const byGroup = new Map(config.sections.map((s) => [s.group, s]))
  const groups = PATHWAYS.map((p) => ({
    name: p.name,
    sections: p.groups.flatMap((g) => byGroup.get(g) ?? []),
  }))
  const other = config.sections.filter((s) => !known.has(s.group))
  if (other.length) groups.push({ name: 'Other', sections: other })
  return groups.filter((g) => g.sections.length)
}

/** A preset is active when every mood matches it, moods it leaves out at 0. */
function presetActive(preset: MoodPreset, controls: BrainControls, config: Controls) {
  return config.moods.every(
    (m) => Math.abs((controls.mood[m.name] ?? 0) - (preset.mood[m.name] ?? 0)) < 1e-6,
  )
}

export function BrainControlsPanel() {
  const open = useBrainControls((s) => s.panelOpen)
  const setOpen = useBrainControls((s) => s.setPanelOpen)
  const controls = useBrainControls((s) => s.controls)
  const change = useBrainControls((s) => s.change)
  const { config, failed } = useControlsConfigState()
  const brain = usePlayer((s) => s.brain)
  const controlsError = useSession((s) => s.controlsError)
  const panel = useRef<HTMLElement>(null)

  useEffect(() => {
    if (!open) return
    panel.current?.focus()
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setOpen(false)
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [open, setOpen])

  if (!open) return null
  const disabled = !config || brain === 'expert'
  const summary = describeControls(controls, config)
  const message = failed
    ? 'Brain controls could not load.'
    : brain === 'expert'
      ? 'The scripted expert has no brain to change. Pick a fly brain in Songs.'
      : controlsError
        ? `The server rejected that change: ${controlsError}.`
        : ''
  const lesioned = (group: string) => controls.lesion.includes(group)
  const toggleLesion = (group: string) =>
    change({
      lesion: lesioned(group)
        ? controls.lesion.filter((g) => g !== group)
        : [...controls.lesion, group],
    })
  const setStimulation = (group: string, current: number) =>
    change({ stimulate: { ...useBrainControls.getState().controls.stimulate, [group]: current } })
  const setMood = (name: string, level: number) =>
    change({ mood: { ...useBrainControls.getState().controls.mood, [name]: level } })

  return (
    <section
      ref={panel}
      className="controls-sheet"
      role="dialog"
      aria-modal="false"
      aria-label="Brain controls panel"
      tabIndex={-1}
    >
      <header className="controls-sheet-header">
        <div>
          <h2>Brain controls</h2>
          <p>{summary ?? 'The untouched brain. Every change applies live, mid-song too.'}</p>
        </div>
        <div className="controls-sheet-actions">
          <Button
            variant="ghost"
            size="xs"
            disabled={disabled || isUntouched(controls)}
            onClick={() => change(noControls)}
          >
            <RotateCcw aria-hidden="true" />
            Reset all
          </Button>
          <Button
            variant="ghost"
            size="icon-xs"
            aria-label="Close brain controls"
            title="Close (Esc)"
            onClick={() => setOpen(false)}
          >
            <X aria-hidden="true" />
          </Button>
        </div>
      </header>
      {message && (
        <p className="controls-sheet-message" role="status">
          {message}
        </p>
      )}

      {config && (
        <div className="controls-sheet-body">
          <TriggerEscape config={config} disabled={disabled} onSet={setStimulation} />

          <section className="controls-group" aria-labelledby="moods-heading">
            <h3 id="moods-heading">Mood</h3>
            {config.presets.length > 0 && (
              <div className="preset-row" role="group" aria-label="Mood presets">
                {config.presets.map((preset) => (
                  <button
                    key={preset.name}
                    type="button"
                    className="preset-button"
                    aria-pressed={presetActive(preset, controls, config)}
                    disabled={disabled}
                    title={preset.description}
                    onClick={() =>
                      change({
                        mood: presetActive(preset, controls, config) ? {} : { ...preset.mood },
                      })
                    }
                  >
                    {preset.label}
                  </button>
                ))}
              </div>
            )}
            {config.moods.map((mood) => (
              <RangeControl
                key={mood.name}
                label={mood.label}
                min={mood.min}
                max={mood.max}
                step={MOOD_STEP}
                value={controls.mood[mood.name] ?? 0}
                disabled={disabled}
                lowLabel={mood.low_label}
                highLabel={mood.high_label}
                format={(v) => (v > 0 ? `+${v.toFixed(2)}` : v.toFixed(2))}
                title={mood.description}
                note={
                  mood.group && lesioned(mood.group)
                    ? `No effect while ${sectionLabel(config, mood.group)} are lesioned`
                    : undefined
                }
                onCommit={(level) => setMood(mood.name, level)}
              />
            ))}
            <div className="noise-row">
              <button
                type="button"
                className="preset-button"
                aria-pressed={controls.noise > 0}
                disabled={disabled}
                onClick={() =>
                  change({
                    noise: controls.noise > 0 ? 0 : Math.min(NOISE_ON, config.noise.max),
                  })
                }
              >
                Neural noise
              </button>
              <p>
                Random noise in every neuron. Any amount swamps this fly&apos;s motor signal, so it
                is on or off.
              </p>
            </div>
          </section>

          <section className="controls-group" aria-labelledby="sections-heading">
            <h3 id="sections-heading">Brain sections</h3>
            <p className="controls-hint">
              Silence a section (a lesion) or drive it with light (optogenetics). A lesion wins over
              stimulation.
            </p>
            {groupedSections(config).map((pathway) => (
              <div key={pathway.name} className="pathway">
                <h4>{pathway.name}</h4>
                {pathway.sections.map((section) => (
                  <SectionRow
                    key={section.group}
                    section={section}
                    lesioned={lesioned(section.group)}
                    current={controls.stimulate[section.group] ?? 0}
                    max={config.stimulation.max}
                    disabled={disabled}
                    onToggle={() => toggleLesion(section.group)}
                    onStimulate={(current) => setStimulation(section.group, current)}
                  />
                ))}
              </div>
            ))}
          </section>

          <section className="controls-group" aria-labelledby="eyes-heading">
            <h3 id="eyes-heading">Eyes</h3>
            <div className="blindfold-row" role="group" aria-label="Blindfolds">
              {(
                [
                  ['left', 'blindLeft'],
                  ['right', 'blindRight'],
                ] as const
              ).map(([side, key]) => (
                <button
                  key={side}
                  type="button"
                  className="preset-button"
                  aria-pressed={controls[key]}
                  disabled={disabled}
                  onClick={() => change({ [key]: !controls[key] })}
                >
                  {side === 'left' ? 'Left eye' : 'Right eye'} blindfold
                </button>
              ))}
            </div>
            <RangeControl
              label="Blur"
              min={0}
              max={EYE.blurDeg.max}
              step={EYE.blurDeg.step}
              value={controls.blurDeg}
              disabled={disabled}
              format={(v) => `${v.toFixed(1)}°`}
              title="Widens each facet's view, in degrees, so edges smear."
              onCommit={(blurDeg) => change({ blurDeg })}
            />
            <RangeControl
              label="Facets lost"
              min={0}
              max={EYE.facetDropout.max}
              step={EYE.facetDropout.step}
              value={controls.facetDropout}
              disabled={disabled}
              format={(v) => `${Math.round(v * 100)}%`}
              title="Share of facets that see nothing but empty sky."
              onCommit={(facetDropout) => change({ facetDropout })}
            />
            <RangeControl
              label="Visual noise"
              min={0}
              max={EYE.eyeNoise.max}
              step={EYE.eyeNoise.step}
              value={controls.eyeNoise}
              disabled={disabled}
              title="Random flicker added to what each facet sees."
              format={(v) => v.toFixed(3)}
              onCommit={(eyeNoise) => change({ eyeNoise })}
            />
          </section>
        </div>
      )}
    </section>
  )
}

function sectionLabel(config: Controls, group: string) {
  const label = config.sections.find((s) => s.group === group)?.label ?? group
  return label.replace(/ \(.*\)$/, '').toLowerCase()
}

function SectionRow({
  section,
  lesioned,
  current,
  max,
  disabled,
  onToggle,
  onStimulate,
}: {
  section: ControlSection
  lesioned: boolean
  current: number
  max: number
  disabled: boolean
  onToggle: () => void
  onStimulate: (current: number) => void
}) {
  return (
    <div className="section-row" data-lesioned={lesioned || undefined} title={section.description}>
      <div className="section-name">
        <span>{section.label}</span>
        <small>{section.n_neurons.toLocaleString()} neurons</small>
      </div>
      <button
        type="button"
        className="lesion-toggle"
        aria-label={`Lesion ${section.label}`}
        aria-pressed={lesioned}
        disabled={disabled}
        onClick={onToggle}
      >
        <span className="lesion-box" aria-hidden="true" />
        Silence
      </button>
      <RangeControl
        label={`Stimulate ${section.label}`}
        className="range-control stimulate-control"
        min={0}
        max={max}
        step={STIMULATION_STEP}
        value={current}
        disabled={disabled || lesioned}
        format={(v) => (v ? v.toFixed(1) : 'off')}
        onCommit={onStimulate}
      />
    </div>
  )
}

/**
 * Drives the giant fiber, the escape command neuron, at full strength while held (spec
 * demo step 5). Release restores whatever stimulation it had before.
 */
function TriggerEscape({
  config,
  disabled,
  onSet,
}: {
  config: Controls
  disabled: boolean
  onSet: (group: string, current: number) => void
}) {
  const held = useRef<number | null>(null)
  const available = config.sections.some((s) => s.group === GIANT_FIBER)
  if (!available) return null
  const press = () => {
    if (disabled || held.current !== null) return
    held.current = useBrainControls.getState().controls.stimulate[GIANT_FIBER] ?? 0
    onSet(GIANT_FIBER, config.stimulation.max)
  }
  const release = () => {
    if (held.current === null) return
    onSet(GIANT_FIBER, held.current)
    held.current = null
  }
  return (
    <div className="trigger-escape">
      <button
        type="button"
        className="trigger-button"
        disabled={disabled}
        onPointerDown={(event) => {
          event.currentTarget.setPointerCapture(event.pointerId)
          press()
        }}
        onPointerUp={release}
        onPointerCancel={release}
        onBlur={release}
        onKeyDown={(event) => {
          if ((event.key === ' ' || event.key === 'Enter') && !event.repeat) {
            event.preventDefault()
            press()
          }
        }}
        onKeyUp={(event) => {
          if (event.key === ' ' || event.key === 'Enter') release()
        }}
      >
        <Zap aria-hidden="true" />
        Hold to trigger escape
      </button>
      <p>Drives the giant fiber, the fly&apos;s escape command neuron, while held.</p>
    </div>
  )
}
