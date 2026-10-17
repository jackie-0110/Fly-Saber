// Chart coordinates follow docs/PROTOCOL.md. Time is always seconds in the audio file.
export interface Note {
  id: number
  time_sec: number
  lane: number
  layer: number
  color: 0 | 1 | null
  cut_direction: number | null
  is_bomb: boolean
}

export interface Wall {
  time_sec: number
  duration_sec: number
  lane: number
  layer: number
  width: number
  height: number
}

export interface Chart {
  schema_version: 1
  map_id: string
  characteristic: string
  difficulty: { name: string; note_speed_mps: number; spawn_distance_m: number }
  notes: Note[]
  walls: Wall[]
  warnings: string[]
}

export interface Song {
  id: string
  title: string
  subtitle: string
  artist: string
  mapper: string
  bpm: number
  sourceUrl: string | null
  audio: Blob
  cover: Blob | null
  charts: Chart[]
}

export type JsonObject = Record<string, unknown>

export function object(value: unknown, label = 'Map'): JsonObject {
  if (!value || typeof value !== 'object' || Array.isArray(value))
    throw new Error(`${label} must be a JSON object.`)
  return value as JsonObject
}

export function list(value: unknown, label: string): JsonObject[] {
  if (value === undefined) return []
  if (!Array.isArray(value)) throw new Error(`${label} must be an array.`)
  return value.map((item) => object(item, label))
}

export function numeric(value: unknown, fallback = 0): number {
  if (value === undefined) return fallback
  if (typeof value !== 'number' || !Number.isFinite(value))
    throw new Error('Map contains an invalid number.')
  return value
}

export function string(value: unknown, fallback = ''): string {
  return typeof value === 'string' ? value : fallback
}

interface BpmChange {
  beat: number
  bpm: number
}

// Precompute segments once, then binary-search them for each note and wall.
export function beatClock(bpm: number, changes: BpmChange[] = [], offset = 0) {
  if (!(bpm > 0 && bpm <= 10000)) throw new Error('The song has an invalid BPM.')
  let previousBeat = 0
  let seconds = offset
  let currentBpm = bpm
  const segments = [{ beat: 0, seconds, bpm }]
  for (const change of [...changes].sort((a, b) => a.beat - b.beat)) {
    if (
      !Number.isFinite(change.beat) ||
      change.beat < 0 ||
      !(change.bpm > 0 && change.bpm <= 10000)
    )
      throw new Error('The chart contains an invalid BPM change.')
    seconds += ((change.beat - previousBeat) * 60) / currentBpm
    segments.push({ beat: change.beat, seconds, bpm: change.bpm })
    previousBeat = change.beat
    currentBpm = change.bpm
  }
  return (beat: number) => {
    let lo = 0
    let hi = segments.length
    while (lo + 1 < hi) {
      const mid = (lo + hi) >>> 1
      if (segments[mid].beat <= beat) lo = mid
      else hi = mid
    }
    const segment = segments[lo]
    return segment.seconds + ((beat - segment.beat) * 60) / segment.bpm
  }
}

export function sampleClock(data: JsonObject): (beat: number) => number {
  const frequency = numeric(data.songFrequency)
  const regions = list(data.bpmData, 'Audio BPM regions')
    .map((region) => ({
      startBeat: numeric(region.sb),
      endBeat: numeric(region.eb),
      start: numeric(region.si) / frequency,
      end: numeric(region.ei) / frequency,
    }))
    .sort((a, b) => a.startBeat - b.startBeat)
  if (
    !(frequency > 0) ||
    !regions.length ||
    regions.some(
      (r, i) =>
        r.endBeat <= r.startBeat ||
        r.end <= r.start ||
        (i > 0 &&
          (Math.abs(r.startBeat - regions[i - 1].endBeat) > 0.001 ||
            Math.abs(r.start - regions[i - 1].end) > 0.001)),
    )
  )
    throw new Error('The map has invalid audio timing regions.')
  return (beat) => {
    let lo = 0
    let hi = regions.length
    while (lo + 1 < hi) {
      const mid = (lo + hi) >>> 1
      if (regions[mid].startBeat <= beat) lo = mid
      else hi = mid
    }
    const r = regions[lo]
    return r.start + ((beat - r.startBeat) / (r.endBeat - r.startBeat)) * (r.end - r.start)
  }
}

export function spawnDistance(speed: number, bpm: number, offset: number) {
  let halfJump = 4
  while ((speed * 60 * halfJump) / bpm > 18) halfJump /= 2
  return (speed * 60 * Math.max(halfJump + offset, 0.25)) / bpm
}

export interface ChartOptions {
  id: string
  name: string
  characteristic: string
  bpm: number
  speed: number
  jumpOffset: number
  offset: number
  toSeconds?: (beat: number) => number
}

export function parseChart(value: unknown, options: ChartOptions): Chart {
  const data = object(value)
  const version = Number(string(data.version ?? data._version).split('.')[0])
  if (![2, 3, 4].includes(version))
    throw new Error(
      `Chart format ${string(data.version ?? data._version, 'unknown')} is unsupported.`,
    )
  if (
    version === 2 &&
    (list(data._BPMChanges, 'BPM changes').length ||
      list(object(data._customData ?? {})._BPMChanges, 'BPM changes').length)
  )
    throw new Error(
      'This legacy map uses editor-specific BPM changes. Export it as a v3 map first.',
    )
  const toSeconds =
    options.toSeconds ??
    beatClock(
      options.bpm,
      version >= 3
        ? list(data.bpmEvents, 'BPM events').map((e) => ({
            beat: numeric(e.b),
            bpm: numeric(e.m),
          }))
        : [],
      options.offset,
    )
  const notes: Note[] = []
  const walls: Wall[] = []
  let skipped = 0
  function addNote(
    beat: unknown,
    x: unknown,
    y: unknown,
    color: unknown,
    direction: unknown,
    bomb: boolean,
  ) {
    const lane = numeric(x)
    const layer = numeric(y)
    const c = bomb ? null : numeric(color)
    const cut = bomb ? null : numeric(direction)
    if (
      !Number.isInteger(lane) ||
      lane < 0 ||
      lane > 3 ||
      !Number.isInteger(layer) ||
      layer < 0 ||
      layer > 2 ||
      (!bomb && ((c !== 0 && c !== 1) || !Number.isInteger(cut) || cut! < 0 || cut! > 8))
    ) {
      skipped++
      return
    }
    notes.push({
      id: 0,
      time_sec: toSeconds(numeric(beat)),
      lane,
      layer,
      color: c as 0 | 1 | null,
      cut_direction: cut,
      is_bomb: bomb,
    })
  }
  function addWall(
    beat: unknown,
    duration: unknown,
    x: unknown,
    y: unknown,
    width: unknown,
    height: unknown,
  ) {
    const start = numeric(beat)
    const length = numeric(duration)
    const lane = numeric(x)
    const w = numeric(width)
    const h = numeric(height)
    if (length <= 0 || w <= 0 || h <= 0 || lane < 0 || lane >= 4) {
      skipped++
      return
    }
    walls.push({
      time_sec: toSeconds(start),
      duration_sec: toSeconds(start + length) - toSeconds(start),
      lane,
      layer: numeric(y),
      width: w,
      height: h,
    })
  }
  if (version === 2) {
    for (const n of list(data._notes, 'Notes')) {
      if (![0, 1, 3].includes(numeric(n._type))) continue
      addNote(n._time, n._lineIndex, n._lineLayer, n._type, n._cutDirection, n._type === 3)
    }
    for (const w of list(data._obstacles, 'Walls')) {
      if (w._type !== 0 && w._type !== 1) {
        skipped++
        continue
      }
      addWall(
        w._time,
        w._duration,
        w._lineIndex,
        w._type === 1 ? 2 : 0,
        w._width,
        w._type === 1 ? 3 : 5,
      )
    }
  } else {
    function metadata(item: JsonObject, key: string): JsonObject {
      if (version === 3) return item
      const index = numeric(item.i)
      const values = data[key]
      if (!Number.isInteger(index) || !Array.isArray(values) || !values[index])
        throw new Error(`The chart references missing ${key} metadata.`)
      if (numeric(item.r) !== 0)
        throw new Error('Rotating charts are not supported in the four-lane view.')
      return object(values[index])
    }
    for (const n of list(data.colorNotes, 'Notes')) {
      const m = metadata(n, 'colorNotesData')
      addNote(n.b, m.x, m.y, m.c, m.d, false)
    }
    for (const n of list(data.bombNotes, 'Bombs')) {
      const m = metadata(n, 'bombNotesData')
      addNote(n.b, m.x, m.y, null, null, true)
    }
    for (const w of list(data.obstacles, 'Walls')) {
      const m = metadata(w, 'obstaclesData')
      addWall(w.b, m.d, m.x, m.y, m.w, m.h)
    }
  }
  if (notes.length > 100000 || walls.length > 20000)
    throw new Error('This chart is too large to preview.')
  notes.sort((a, b) => a.time_sec - b.time_sec || a.lane - b.lane || a.layer - b.layer)
  notes.forEach((n, i) => {
    n.id = i
  })
  walls.sort((a, b) => a.time_sec - b.time_sec)
  const warnings: string[] = []
  if (skipped) warnings.push(`${skipped} objects outside the supported grid were omitted.`)
  if (['sliders', 'burstSliders', 'arcs', 'chains'].some((k) => list(data[k], k).length))
    warnings.push('Arc guides and chain slices are not rendered; their chart notes remain visible.')
  const speed = options.speed > 0 && options.speed <= 100 ? options.speed : 10
  return {
    schema_version: 1,
    map_id: options.id,
    characteristic: options.characteristic,
    difficulty: {
      name: options.name,
      note_speed_mps: speed,
      spawn_distance_m: spawnDistance(speed, options.bpm, options.jumpOffset),
    },
    notes,
    walls,
    warnings,
  }
}

export function lowerBound(notes: readonly { time_sec: number }[], time: number): number {
  let lo = 0
  let hi = notes.length
  while (lo < hi) {
    const mid = (lo + hi) >>> 1
    if (notes[mid].time_sec < time) lo = mid + 1
    else hi = mid
  }
  return lo
}

export function formatTime(seconds: number) {
  const safe = Math.max(0, Number.isFinite(seconds) ? seconds : 0)
  return `${Math.floor(safe / 60)
    .toString()
    .padStart(2, '0')}:${Math.floor(safe % 60)
    .toString()
    .padStart(2, '0')}`
}

export const difficultyOrder = ['Easy', 'Normal', 'Hard', 'Expert', 'ExpertPlus']
export const difficultyLabel = (name: string) => (name === 'ExpertPlus' ? 'Expert+' : name)
