import { unzip, strFromU8 } from 'fflate'
import { difficultyOrder, list, numeric, object, parseChart, sampleClock, string } from './chart.ts'
import type { Chart, JsonObject, Song } from './chart.ts'

const API = 'https://api.beatsaver.com'
const MAX_ARCHIVE = 64 * 1024 * 1024
const MAX_EXPANDED = 192 * 1024 * 1024
const supportedModes = ['Standard', 'OneSaber', 'NoArrows', 'Lawless']

export interface SongResult {
  id: string
  title: string
  artist: string
  mapper: string
  bpm: number
  duration: number
  coverUrl: string
  downloadUrl: string
  difficulties: { name: string; characteristic: string; notes: number; nps: number }[]
}

function cdnUrl(value: unknown, extension: string): string {
  if (typeof value !== 'string') throw new Error('The song is missing a download URL.')
  const url = new URL(value)
  if (url.protocol !== 'https:' || !/^(?:[a-z0-9-]+\.)?beatsaver\.com$/.test(url.hostname) ||
    !url.pathname.toLowerCase().endsWith(extension))
    throw new Error('The song has an unsupported asset URL.')
  return url.href
}

export function parseSongResult(value: unknown): SongResult {
  const data = object(value, 'Song')
  const metadata = object(data.metadata, 'Song metadata')
  const versions = list(data.versions, 'Map versions')
  const version = versions.find((v) => v.state === 'Published') ?? versions[0]
  if (!version || !/^[a-f0-9]+$/i.test(string(data.id))) throw new Error('Invalid song metadata.')
  const difficulties = list(version.diffs, 'Difficulties').filter((d) =>
    supportedModes.includes(string(d.characteristic))).map((d) => ({
      name: string(d.difficulty), characteristic: string(d.characteristic),
      notes: numeric(d.notes), nps: numeric(d.nps),
    }))
  return {
    id: string(data.id), title: string(metadata.songName, string(data.name, 'Untitled')),
    artist: string(metadata.songAuthorName, 'Unknown artist'),
    mapper: string(metadata.levelAuthorName, 'Unknown mapper'), bpm: numeric(metadata.bpm),
    duration: numeric(metadata.duration), coverUrl: cdnUrl(version.coverURL, '.jpg'),
    downloadUrl: cdnUrl(version.downloadURL, '.zip'), difficulties,
  }
}

export function mapKey(query: string): string | null {
  const trimmed = query.trim()
  if (/^[a-f0-9]{1,8}$/i.test(trimmed)) return trimmed.toLowerCase()
  try {
    const url = new URL(trimmed)
    if (['beatsaver.com', 'www.beatsaver.com'].includes(url.hostname))
      return /^\/maps\/([a-f0-9]+)\/?$/i.exec(url.pathname)?.[1].toLowerCase() ?? null
  } catch { /* A search phrase is not a URL. */ }
  return null
}

async function request(url: string, signal: AbortSignal, timeout = 20000) {
  const response = await fetch(url, { signal: AbortSignal.any([signal, AbortSignal.timeout(timeout)]),
    credentials: 'omit', referrerPolicy: 'no-referrer' })
  if (response.status === 429) throw new Error('Song search is busy. Wait a moment and try again.')
  if (response.status === 404) throw new Error('That map was not found. Try a song title or another map link.')
  if (!response.ok) throw new Error(`The song service returned ${response.status}. Try again shortly.`)
  return response
}

export async function searchSongs(query: string, page: number, sort: string, signal: AbortSignal) {
  const key = mapKey(query)
  if (key) return { songs: [parseSongResult(await (await request(`${API}/maps/id/${key}`, signal)).json())], more: false }
  const params = new URLSearchParams({ q: query.trim(), sortOrder: sort })
  const response = object(await (await request(`${API}/search/text/${page}?${params}`, signal)).json())
  if (!Array.isArray(response.docs)) throw new Error('The song service returned an invalid catalog.')
  const songs: SongResult[] = []
  for (const item of response.docs) {
    try { songs.push(parseSongResult(item)) } catch { /* One removed or malformed map cannot hide the rest. */ }
  }
  return { songs, more: response.docs.length >= 20 }
}

export async function downloadSong(result: SongResult, signal: AbortSignal): Promise<Song> {
  const response = await request(result.downloadUrl, signal, 90000)
  const length = Number(response.headers.get('content-length'))
  if (length > MAX_ARCHIVE) throw new Error('This map ZIP exceeds the 64 MB import limit.')
  const reader = response.body?.getReader()
  if (!reader) throw new Error('Your browser could not download this map. Try importing its ZIP.')
  const chunks: Uint8Array[] = []
  let size = 0
  try {
    while (true) {
      const chunk = await reader.read()
      if (chunk.done) break
      size += chunk.value.length
      if (size > MAX_ARCHIVE) { await reader.cancel(); throw new Error('This map ZIP exceeds the 64 MB import limit.') }
      chunks.push(chunk.value)
    }
  } finally { reader.releaseLock() }
  signal.throwIfAborted()
  const archive = new Uint8Array(size)
  let offset = 0
  for (const chunk of chunks) { archive.set(chunk, offset); offset += chunk.length }
  return importSong(archive, result.id, `https://beatsaver.com/maps/${result.id}`, signal)
}

export async function importSongFile(file: File, signal: AbortSignal): Promise<Song> {
  if (!file.name.toLowerCase().endsWith('.zip'))
    throw new Error('Choose a map ZIP containing Info.dat, chart files, and audio. Plain audio has no block chart.')
  if (file.size > MAX_ARCHIVE) throw new Error('This map ZIP exceeds the 64 MB import limit.')
  return importSong(new Uint8Array(await file.arrayBuffer()), `local-${crypto.randomUUID()}`, null, signal)
}

async function unpack(bytes: Uint8Array, signal: AbortSignal): Promise<Record<string, Uint8Array>> {
  signal.throwIfAborted()
  if (bytes.length > MAX_ARCHIVE) throw new Error('This map ZIP exceeds the 64 MB import limit.')
  return new Promise((resolve, reject) => {
    let expanded = 0
    let files = 0
    let limitError = ''
    const terminate = unzip(bytes, {
      filter: (entry) => {
        files++
        expanded += entry.originalSize
        if (expanded > MAX_EXPANDED || entry.originalSize > MAX_ARCHIVE || files > 5000)
          limitError = 'This map archive expands beyond the import limit.'
        return !limitError && /\.(dat|json|ogg|egg|mp3|wav|jpg|jpeg|png|webp)$/i.test(entry.name) &&
          !entry.name.split('/').includes('__MACOSX')
      },
    }, (error, data) => {
      signal.removeEventListener('abort', abort)
      if (signal.aborted) reject(signal.reason)
      else if (limitError) reject(new Error(limitError))
      else if (error) reject(new Error('Could not open that ZIP. Choose a complete, unmodified map archive.'))
      else resolve(data)
    })
    function abort() { terminate(); reject(signal.reason) }
    signal.addEventListener('abort', abort, { once: true })
  })
}

function json(bytes: Uint8Array, name: string): JsonObject {
  try { return object(JSON.parse(strFromU8(bytes).replace(/^\uFEFF/, '')), name) }
  catch { throw new Error(`Could not read ${name}. It must contain valid JSON.`) }
}

export async function importSong(bytes: Uint8Array, id: string, sourceUrl: string | null,
  signal: AbortSignal): Promise<Song> {
  const files = await unpack(bytes, signal)
  const infos = Object.keys(files).filter((name) => name.split('/').at(-1)?.toLowerCase() === 'info.dat')
  if (infos.length !== 1) throw new Error('Choose a single map ZIP containing one Info.dat file, its charts, and audio.')
  const infoPath = infos[0]
  const prefix = infoPath.slice(0, infoPath.length - 'info.dat'.length)
  const names = new Map(Object.keys(files).map((name) => [name.toLowerCase(), name]))
  function file(filename: string): Uint8Array {
    const name = names.get(`${prefix}${filename.replaceAll('\\', '/')}`.toLowerCase())
    if (!name) throw new Error(`The map ZIP is missing ${filename || 'a required file'}.`)
    return files[name]
  }
  const info = json(files[infoPath], 'Info.dat')
  const version = Number(string(info.version ?? info._version).split('.')[0])
  if (version !== 2 && version !== 4) throw new Error(`Info.dat version ${version} is unsupported.`)
  const modern = version === 4
  const songMeta = modern ? object(info.song) : info
  const audioMeta = modern ? object(info.audio) : info
  const bpm = numeric(modern ? audioMeta.bpm : info._beatsPerMinute)
  if (!(bpm > 0 && bpm <= 10000)) throw new Error('The song has an invalid BPM.')
  const audioName = string(modern ? audioMeta.songFilename : info._songFilename)
  const audioBytes = file(audioName)
  if (!audioBytes.length) throw new Error('The map audio file is empty.')
  const audioType = /\.(egg|ogg)$/i.test(audioName) ? 'audio/ogg' : /\.mp3$/i.test(audioName) ? 'audio/mpeg' : 'audio/wav'
  const coverName = string(modern ? info.coverImageFilename : info._coverImageFilename)
  let cover: Blob | null = null
  if (coverName) {
    try {
      const bytes = file(coverName)
      if (bytes.length < 8 * 1024 * 1024) cover = new Blob([new Uint8Array(bytes)], {
        type: /\.png$/i.test(coverName) ? 'image/png' : /\.webp$/i.test(coverName) ? 'image/webp' : 'image/jpeg',
      })
    } catch { /* Cover art is optional. */ }
  }
  const offset = numeric(info._songTimeOffset)
  // Match the Python normalizer rather than guessing the sign of a legacy offset.
  if (offset !== 0) throw new Error('This map uses a legacy song offset. Re-export it with the offset applied to its chart.')
  const audioData = modern ? json(file(string(audioMeta.audioDataFilename)), 'Audio timing') : null
  const timing = audioData ? sampleClock(audioData) : undefined
  const entries: { characteristic: string; data: JsonObject }[] = modern
    ? list(info.difficultyBeatmaps, 'Difficulties').map((data) => ({ characteristic: string(data.characteristic), data }))
    : list(info._difficultyBeatmapSets, 'Difficulty sets').flatMap((set) =>
      list(set._difficultyBeatmaps, 'Difficulties').map((data) => ({
        characteristic: string(set._beatmapCharacteristicName), data,
      })))
  const charts: Chart[] = []
  const failures: string[] = []
  for (const entry of entries) {
    if (!supportedModes.includes(entry.characteristic)) continue
    const d = entry.data
    const name = string(modern ? d.difficulty : d._difficulty)
    const filename = string(modern ? d.beatmapDataFilename : d._beatmapFilename)
    try {
      const data = json(file(filename), filename)
      const chart = parseChart(data, {
        id: `${id}_${entry.characteristic.toLowerCase()}_${name.toLowerCase()}`, name,
        characteristic: entry.characteristic, bpm, offset,
        speed: numeric(modern ? d.noteJumpMovementSpeed : d._noteJumpMovementSpeed, 10),
        jumpOffset: numeric(modern ? d.noteJumpStartBeatOffset : d._noteJumpStartBeatOffset),
        toSeconds: timing,
      })
      const custom = object(d.customData ?? d._customData ?? {})
      const required = custom.requirements ?? custom._requirements
      if (Array.isArray(required) && required.length)
        chart.warnings.push(`Map extensions (${required.filter((v) => typeof v === 'string').join(', ')}) are not applied.`)
      charts.push(chart)
    } catch (cause) { failures.push(`${name}: ${cause instanceof Error ? cause.message : 'Could not read chart.'}`) }
  }
  if (!charts.length) throw new Error(failures[0] ?? 'This map has no supported four-lane charts. Choose Standard, One Saber, No Arrows, or Lawless.')
  charts.sort((a, b) => supportedModes.indexOf(a.characteristic) - supportedModes.indexOf(b.characteristic) ||
    difficultyOrder.indexOf(a.difficulty.name) - difficultyOrder.indexOf(b.difficulty.name))
  if (failures.length) for (const chart of charts) chart.warnings.push(`Some difficulties could not be loaded: ${failures.join(' ')}`)
  const authors = modern ? object(entries[0]?.data.beatmapAuthors ?? {}) : null
  return {
    id, title: string(modern ? songMeta.title : info._songName, 'Untitled song'),
    subtitle: string(modern ? songMeta.subTitle : info._songSubName),
    artist: string(modern ? songMeta.author : info._songAuthorName, 'Unknown artist'),
    mapper: modern ? (Array.isArray(authors?.mappers) ? authors.mappers.filter((v) => typeof v === 'string').join(', ') : 'Unknown mapper') : string(info._levelAuthorName, 'Unknown mapper'),
    bpm, sourceUrl, audio: new Blob([new Uint8Array(audioBytes)], { type: audioType }), cover, charts,
  }
}
