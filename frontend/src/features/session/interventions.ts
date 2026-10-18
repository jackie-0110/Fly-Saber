/** Live brain and eye controls for one player (docs/reference/PROTOCOL.md sections 3 and 4). */
export interface BrainControls {
  /** Brain sections silenced, by group name from controls.json. */
  lesion: string[]
  /** Group name to opto current, 0 to controls.json `stimulation.max`. */
  stimulate: Record<string, number>
  /** Mood name to level, within that mood's range in controls.json. */
  mood: Record<string, number>
  /** Neural noise std, 0 to controls.json `noise.max` (the "jitter" slider). */
  noise: number
  blindLeft: boolean
  blindRight: boolean
  /** Extra Gaussian acceptance, degrees added to the eye's FWHM. */
  blurDeg: number
  /** Share of ommatidia dropped, 0 to 1. */
  facetDropout: number
  /** Luminance noise std on the eye image. */
  eyeNoise: number
}

export const noControls: BrainControls = {
  lesion: [],
  stimulate: {},
  mood: {},
  noise: 0,
  blindLeft: false,
  blindRight: false,
  blurDeg: 0,
  facetDropout: 0,
  eyeNoise: 0,
}

const allZero = (values: Record<string, number>) => Object.values(values).every((v) => v === 0)

export function isUntouched(controls: BrainControls) {
  return (
    !controls.lesion.length &&
    allZero(controls.stimulate) &&
    allZero(controls.mood) &&
    controls.noise === 0 &&
    !controls.blindLeft &&
    !controls.blindRight &&
    controls.blurDeg === 0 &&
    controls.facetDropout === 0 &&
    controls.eyeNoise === 0
  )
}

/** Drop zero entries, so the message lists only what is actually changed. */
const nonZero = (values: Record<string, number>) =>
  Object.fromEntries(Object.entries(values).filter(([, v]) => v !== 0))

/** One slot's whole control state; each message replaces that slot's previous one. */
export function interventionsMessage(controls: BrainControls, slot = 0) {
  return {
    type: 'set_interventions',
    slot,
    interventions: {
      lesion: controls.lesion,
      stimulate: nonZero(controls.stimulate),
      mood: nonZero(controls.mood),
      noise: controls.noise,
    },
    eye: {
      blind_left: controls.blindLeft ? 1 : 0,
      blind_right: controls.blindRight ? 1 : 0,
      blur_deg: controls.blurDeg,
      facet_dropout: controls.facetDropout,
      noise: controls.eyeNoise,
    },
  }
}
