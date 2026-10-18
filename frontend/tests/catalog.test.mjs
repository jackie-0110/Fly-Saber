import assert from 'node:assert/strict'
import { test } from 'node:test'
import { loadCatalog, loadServerSong, withLiveCharts } from '../src/features/songs/catalog.ts'

const map = {
  schema_version: 1, map_id: 'abc12_hard', split: 'heldout',
  song: { title: 'Test', artist: 'Tester', mapper: 'Mapper', bpm: 120, source_url: 'https://example.com/map', license: 'CC0', attribution: 'Original fixture' },
  difficulty: { name: 'Hard', note_speed_mps: 7, spawn_distance_m: 12 },
  notes: [{ id: 0, time_sec: 1, lane: 0, layer: 0, color: 0, cut_direction: 8, is_bomb: false }], walls: [],
}

test('server songs retain exact IDs, note timing, mapper credit and attribution; audio uses the server endpoint', async (t) => {
  const paths = []
  t.mock.method(globalThis, 'fetch', async (path) => {
    paths.push(path)
    return String(path).endsWith('/audio') ? new Response('audio') : Response.json(map)
  })
  const song = await loadServerSong('abc12_hard', new AbortController().signal)
  assert.deepEqual(paths, ['/api/maps/abc12_hard', '/api/maps/abc12_hard/audio'])
  assert.equal(song.serverMapId, map.map_id)
  assert.equal(song.mapper, 'Mapper')
  assert.equal(song.attribution, 'Original fixture')
  assert.equal(song.split, 'heldout')
  assert.deepEqual(song.charts[0].notes, map.notes)
})

test('bad map identity, note indexes/coordinates and unsorted timing fail before audio downloads', async (t) => {
  let value = map
  t.mock.method(globalThis, 'fetch', async (path) => {
    assert.equal(String(path).endsWith('/audio'), false)
    return Response.json(value)
  })
  for (const bad of [null, { ...map, map_id: 'other_easy' },
    { ...map, notes: [{ ...map.notes[0], lane: 8 }] },
    { ...map, notes: [{ ...map.notes[0], id: 5 }] },
    { ...map, notes: [...map.notes, { ...map.notes[0], id: 1, time_sec: 0 }] },
  ]) {
    value = bad
    await assert.rejects(loadServerSong('abc12_hard', new AbortController().signal), /invalid/)
  }
})

test('catalog failures, HTML fallbacks and malformed entries are recoverable errors', async (t) => {
  let mode = 'html'
  t.mock.method(globalThis, 'fetch', async () => mode === 'html'
    ? new Response('<html>fallback</html>')
    : mode === 'error' ? new Response('', { status: 503 }) : Response.json([{}]))
  for (const next of ['html', 'error', 'bad']) {
    mode = next
    await assert.rejects(loadCatalog(new AbortController().signal))
  }
})

function previewSong() {
  const chart = (name, characteristic = 'Standard') => ({
    schema_version: 1, map_id: `c0ffee_${name.toLowerCase()}`, characteristic,
    difficulty: { name, note_speed_mps: 10, spawn_distance_m: 12 }, notes: [], walls: [], warnings: [],
  })
  return {
    id: 'c0ffee', title: 'Community', subtitle: '', artist: 'A', mapper: 'M', bpm: 120,
    sourceUrl: null, audio: new Blob(['audio']), cover: null,
    charts: [chart('Hard'), chart('Hard', 'OneSaber'), chart('Expert')],
  }
}

test('community songs swap in the server charts the brain plays and keep their own audio', async (t) => {
  const calls = []
  t.mock.method(globalThis, 'fetch', async (path, init) => {
    calls.push(`${init?.method ?? 'GET'} ${path}`)
    if (String(path).startsWith('/api/community/')) return Response.json(['c0ffee_hard'])
    return Response.json({ ...map, map_id: 'c0ffee_hard', split: 'community' })
  })
  const before = previewSong()
  const song = await withLiveCharts(before, 'c0ffee', new AbortController().signal)
  assert.deepEqual(calls, ['POST /api/community/c0ffee', 'GET /api/maps/c0ffee_hard'])
  assert.equal(song.audio, before.audio)
  assert.equal(song.split, 'community')
  const [hard, oneSaber, expert] = song.charts
  assert.equal(hard.live, true)
  assert.deepEqual(hard.notes, map.notes)
  assert.equal(oneSaber.live, undefined)
  assert.equal(expert.live, undefined)
})

test('a map the server cannot play stays a preview and says why', async (t) => {
  t.mock.method(globalThis, 'fetch', async () =>
    Response.json({ detail: 'the fly cannot play this map yet: Info.dat version 4.0.0' }, { status: 422 }))
  const song = await withLiveCharts(previewSong(), 'c0ffee', new AbortController().signal)
  assert.ok(song.charts.every((c) => !c.live))
  assert.match(song.charts[0].warnings[0], /Preview only.*Info.dat version 4/)
})
