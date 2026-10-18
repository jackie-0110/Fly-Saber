/** Live brain and eye controls for one player (docs/PROTOCOL.md sections 3 and 4). */
export interface BrainControls {
  /** Brain sections silenced, by group name from controls.json. */
  lesion: string[]
  /** Mood name to level, within that mood's range in controls.json. */
  mood: Record<string, number>
  blindLeft: boolean
  blindRight: boolean
}

export const noControls: BrainControls = {
  lesion: [],
  mood: {},
  blindLeft: false,
  blindRight: false,
}

export function isUntouched(controls: BrainControls) {
  return (
    !controls.lesion.length &&
    Object.values(controls.mood).every((level) => level === 0) &&
    !controls.blindLeft &&
    !controls.blindRight
  )
}

/** The whole control state for slot 0; each message replaces the previous one. */
export function interventionsMessage(controls: BrainControls) {
  return {
    type: 'set_interventions',
    slot: 0,
    interventions: { lesion: controls.lesion, mood: controls.mood },
    eye: { blind_left: controls.blindLeft ? 1 : 0, blind_right: controls.blindRight ? 1 : 0 },
  }
}
