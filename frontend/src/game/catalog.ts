import { backendJson, backendUrl } from './backend.ts'
import type { Song, Chart } from './chart.ts'

export interface BrainOption {
  id: string
  label: string
  description: string
}
export interface ServerMap {
  map_id: string
  split: 'train' | 'heldout'
  song: { title: string; artist: string; mapper: string; license: string | null }
  difficulty: { name: string }
  note_count: number
}

export async function loadCatalog(signal: AbortSignal) {
  const [maps, brains] = await Promise.all([
    backendJson('/api/maps', signal),
    backendJson('/api/brains', signal),
  ])
  if (!Array.isArray(maps) || !Array.isArray(brains))
    throw new Error('The server catalog is unavailable. Retry when the backend is running.')
  if (
    !maps.every(
      (m) =>
        m &&
        typeof m.map_id === 'string' &&
        /^[a-z0-9]+_[a-z0-9]+$/.test(m.map_id) &&
        ['train', 'heldout'].includes(m.split) &&
        typeof m.song?.title === 'string' &&
        typeof m.song?.artist === 'string' &&
        typeof m.song?.mapper === 'string' &&
        typeof m.difficulty?.name === 'string' &&
        Number.isInteger(m.note_count),
    ) ||
    !brains.every(
      (b) =>
        b &&
        typeof b.id === 'string' &&
        typeof b.label === 'string' &&
        typeof b.description === 'string',
    )
  )
    throw new Error('The server returned an invalid catalog. Retry after updating the backend.')
  return { maps: maps as ServerMap[], brains: brains as BrainOption[] }
}

type ServerChart = Chart & {
  split: 'train' | 'heldout' | 'community'
  song: {
    title: string
    artist: string
    mapper: string
    bpm: number
    attribution: string | null
    license: string | null
    source_url: string | null
  }
}

/** Fetch and validate one normalized map; the chart a brain plays on the server. */
async function loadServerChart(mapId: string, signal: AbortSignal): Promise<ServerChart> {
  const raw = (await backendJson(`/api/maps/${encodeURIComponent(mapId)}`, signal)) as ServerChart
  if (
    !raw ||
    raw.schema_version !== 1 ||
    raw.map_id !== mapId ||
    !raw.song ||
    !raw.difficulty ||
    !Array.isArray(raw.notes) ||
    !Array.isArray(raw.walls) ||
    !['train', 'heldout', 'community'].includes(raw.split) ||
    ![raw.song.title, raw.song.artist, raw.song.mapper, raw.difficulty.name].every(
      (s) => typeof s === 'string',
    ) ||
    !Number.isFinite(raw.song.bpm) ||
    raw.song.bpm <= 0 ||
    raw.notes.length > 100000 ||
    raw.walls.length > 20000 ||
    !Number.isFinite(raw.difficulty.note_speed_mps) ||
    raw.difficulty.note_speed_mps <= 0 ||
    !Number.isFinite(raw.difficulty.spawn_distance_m) ||
    raw.difficulty.spawn_distance_m <= 0
  )
    throw new Error('The server returned an invalid map. Choose another song.')
  if (
    !raw.notes.every(
      (n, i) =>
        n &&
        n.id === i &&
        Number.isFinite(n.time_sec) &&
        (i === 0 || n.time_sec >= raw.notes[i - 1].time_sec) &&
        Number.isInteger(n.lane) &&
        n.lane >= 0 &&
        n.lane <= 3 &&
        Number.isInteger(n.layer) &&
        n.layer >= 0 &&
        n.layer <= 2 &&
        typeof n.is_bomb === 'boolean' &&
        (n.is_bomb
          ? n.color === null && n.cut_direction === null
          : (n.color === 0 || n.color === 1) &&
            Number.isInteger(n.cut_direction) &&
            n.cut_direction! >= 0 &&
            n.cut_direction! <= 8),
    ) ||
    !raw.walls.every(
      (w, i) =>
        w &&
        [w.time_sec, w.duration_sec, w.lane, w.layer, w.width, w.height].every(Number.isFinite) &&
        w.duration_sec > 0 &&
        w.width > 0 &&
        w.height > 0 &&
        (i === 0 || w.time_sec >= raw.walls[i - 1].time_sec),
    )
  )
    throw new Error('The server map has invalid notes or walls. Choose another song.')
  return raw
}

export async function loadServerSong(mapId: string, signal: AbortSignal): Promise<Song> {
  const raw = await loadServerChart(mapId, signal)
  const response = await fetch(backendUrl(`/api/maps/${encodeURIComponent(mapId)}/audio`), {
    signal: AbortSignal.any([signal, AbortSignal.timeout(30000)]),
  })
  if (!response.ok)
    throw new Error('This song’s allowlisted audio is unavailable. Choose another song.')
  return {
    id: mapId,
    title: raw.song.title,
    subtitle: '',
    artist: raw.song.artist,
    mapper: raw.song.mapper,
    bpm: raw.song.bpm,
    sourceUrl:
      typeof raw.song.source_url === 'string' && /^https?:\/\//.test(raw.song.source_url)
        ? raw.song.source_url
        : null,
    audio: await response.blob(),
    cover: null,
    charts: [{ ...raw, characteristic: 'Standard', warnings: [] }],
    serverMapId: mapId,
    split: raw.split,
    attribution: raw.song.attribution ?? raw.song.license ?? undefined,
  }
}

/**
 * Ask the server to fetch a community map's charts so a brain can play them, and swap in
 * the server's Standard charts: the brain is judged on those, so the scene must show them.
 * The audio stays the copy the browser downloaded. If the server cannot play the map, the
 * song stays a preview and each chart says why.
 */
export async function withLiveCharts(song: Song, key: string, signal: AbortSignal): Promise<Song> {
  let charts: ServerChart[]
  try {
    const response = await fetch(backendUrl(`/api/community/${encodeURIComponent(key)}`), {
      method: 'POST',
      signal: AbortSignal.any([signal, AbortSignal.timeout(60000)]),
    })
    const body: unknown = await response.json().catch(() => null)
    if (!response.ok)
      throw new Error(
        body && typeof body === 'object' && 'detail' in body && typeof body.detail === 'string'
          ? body.detail
          : `the server returned ${response.status}`,
      )
    if (
      !Array.isArray(body) ||
      !body.every((id) => typeof id === 'string' && /^[a-z0-9]+_[a-z0-9]+$/.test(id))
    )
      throw new Error('the server returned an invalid map list')
    charts = await Promise.all(body.map((id: string) => loadServerChart(id, signal)))
  } catch (cause) {
    signal.throwIfAborted()
    // The server words its refusals as "the fly cannot play this map yet: <why>"; keep the why.
    const reason = (cause instanceof Error ? cause.message : 'the server is unavailable').replace(
      /^the fly cannot play this map yet:\s*/i,
      '',
    )
    for (const chart of song.charts)
      chart.warnings.push(
        `Preview only, so the fly stays still. The server can’t play it: ${reason}.`,
      )
    return song
  }
  const live = new Map(charts.map((c) => [c.difficulty.name, c]))
  return {
    ...song,
    split: 'community',
    charts: song.charts.map((chart) => {
      const server = chart.characteristic === 'Standard' && live.get(chart.difficulty.name)
      if (!server) return chart
      return {
        schema_version: 1,
        map_id: server.map_id,
        characteristic: 'Standard',
        difficulty: server.difficulty,
        notes: server.notes,
        walls: server.walls,
        warnings: chart.warnings,
        live: true,
      }
    }),
  }
}
