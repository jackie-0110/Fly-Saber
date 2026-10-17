import assert from 'node:assert/strict'
import { test } from 'node:test'
import { loadCatalog, loadServerSong } from '../src/game/catalog.ts'

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
