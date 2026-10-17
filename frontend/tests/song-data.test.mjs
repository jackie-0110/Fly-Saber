import assert from 'node:assert/strict'
import { test } from 'node:test'
import { strToU8, zipSync } from 'fflate'
import { beatClock, formatTime, lowerBound, parseChart, sampleClock } from '../src/game/chart.ts'
import { importSong, mapKey, parseSongResult } from '../src/game/songs.ts'
import { info, mapZip, result, v2, v3, wav } from './song-fixtures.mjs'

const options = { id: 'fixture_normal', name: 'Normal', characteristic: 'Standard', bpm: 120, speed: 10, jumpOffset: 0, offset: 0 }
const signal = () => new AbortController().signal

test('legacy note timings, bomb identity, nine directions, and wall duration survive normalization', () => {
  const chart = parseChart(v2, options)
  assert.deepEqual(chart.notes.map((n) => n.time_sec), [2, 2.5, 4, 8])
  assert.deepEqual(chart.notes.map((n) => n.id), [0, 1, 2, 3])
  assert.equal(chart.notes[0].cut_direction, 8)
  assert.equal(chart.notes[2].color, null)
  assert.equal(chart.notes[2].is_bomb, true)
  assert.equal(chart.notes[3].cut_direction, 5)
  assert.deepEqual(chart.walls[0], { time_sec: 5, duration_sec: 1, lane: 0, layer: 2, width: 1, height: 3 })
})

test('v3 defaulted zero fields and BPM changes produce audio timestamps', () => {
  const chart = parseChart(v3, options)
  assert.deepEqual(chart.notes.map((n) => n.time_sec), [2, 3, 6, 14])
  assert.equal(chart.notes[1].lane, 0)
  assert.equal(chart.notes[1].cut_direction, 0)
  assert.equal(chart.walls[0].duration_sec, 2)
})

test('tempo clock handles exact boundaries, changes at zero, and out-of-order events', () => {
  const time = beatClock(120, [{ beat: 8, bpm: 240 }, { beat: 0, bpm: 60 }, { beat: 4, bpm: 120 }])
  assert.equal(time(0), 0)
  assert.equal(time(4), 4)
  assert.equal(time(8), 6)
  assert.equal(time(12), 7)
  assert.throws(() => beatClock(0), /BPM/)
  assert.throws(() => beatClock(120, [{ beat: 3, bpm: 0 }]), /BPM/)
})

test('v4 indexed metadata resolves notes and audio sample regions without using the initial BPM', () => {
  const toSeconds = sampleClock({ songFrequency: 100, bpmData: [
    { sb: 0, eb: 4, si: 0, ei: 200 }, { sb: 4, eb: 8, si: 200, ei: 600 },
  ] })
  const chart = parseChart({ version: '4.0.0', colorNotes: [{ b: 6, i: 1 }, { b: 2 }],
    colorNotesData: [{ x: 1, y: 1, c: 0, d: 6 }, { x: 3, y: 2, c: 1, d: 4 }],
    bombNotes: [{ b: 4 }], bombNotesData: [{}],
    obstacles: [{ b: 3 }], obstaclesData: [{ d: 2, w: 1, h: 3 }],
  }, { ...options, toSeconds })
  assert.deepEqual(chart.notes.map((n) => n.time_sec), [1, 2, 4])
  assert.equal(chart.notes[2].lane, 3)
  assert.equal(chart.notes[2].cut_direction, 4)
  assert.equal(chart.walls[0].duration_sec, 1.5)
  assert.throws(() => parseChart({ version: '4.0.0', colorNotes: [{ i: 5 }] }, options), /metadata/)
})

test('malformed numbers and versions are rejected and off-grid notes are explicitly reported', () => {
  assert.throws(() => parseChart({ version: '9.0.0' }, options), /unsupported/)
  assert.throws(() => parseChart({ version: '3.0.0', colorNotes: [{ b: '2' }] }, options), /invalid number/)
  const chart = parseChart({ version: '3.0.0', colorNotes: [{ x: 10 }, {}] }, options)
  assert.equal(chart.notes.length, 1)
  assert.match(chart.warnings[0], /1 objects/)
})

test('ZIP imports are case-insensitive, handle nested roots and BOMs, and preserve all difficulties', async () => {
  const zip = zipSync({ 'A/INFO.DAT': strToU8('\uFEFF' + JSON.stringify(info)),
    'A/normal.DAT': strToU8(JSON.stringify(v2)), 'A/HARD.dat': strToU8(JSON.stringify(v3)), 'A/song.wav': wav() })
  const song = await importSong(zip, 'fixture', null, signal())
  assert.equal(song.title, 'Orbit Test')
  assert.equal(song.mapper, 'Test Mapper')
  assert.equal(song.audio.type, 'audio/wav')
  assert.deepEqual(song.charts.map((c) => c.difficulty.name), ['Normal', 'Hard'])
})

test('ZIP import rejects missing audio, ambiguous archives, invalid JSON, and cancellation', async () => {
  await assert.rejects(importSong(zipSync({ 'Info.dat': strToU8(JSON.stringify(info)) }), 'x', null, signal()), /missing song.wav/)
  await assert.rejects(importSong(zipSync({ 'Info.dat': strToU8('{}'), 'other/Info.dat': strToU8('{}') }), 'x', null, signal()), /single map/)
  await assert.rejects(importSong(mapZip({ 'Info.dat': strToU8('not json') }), 'x', null, signal()), /valid JSON/)
  const controller = new AbortController(); controller.abort()
  await assert.rejects(importSong(mapZip(), 'x', null, controller.signal), { name: 'AbortError' })
})

test('one malformed difficulty cannot prevent other supported difficulties from loading', async () => {
  const song = await importSong(mapZip({ 'Hard.dat': strToU8('{"version":"9.0.0"}') }), 'x', null, signal())
  assert.equal(song.charts.length, 1)
  assert.match(song.charts[0].warnings[0], /Some difficulties/)
})

test('map IDs and links are recognized, catalog assets stay on the community CDN', () => {
  assert.equal(mapKey('https://beatsaver.com/maps/ABC12'), 'abc12')
  assert.equal(mapKey('abc12'), 'abc12')
  assert.equal(mapKey('https://example.org/maps/abc12'), null)
  assert.equal(mapKey('Daft Punk'), null)
  assert.equal(parseSongResult(result).title, 'Orbit Test')
  assert.throws(() => parseSongResult({ ...result, versions: [{ ...result.versions[0], downloadURL: 'https://example.com/x.zip' }] }), /asset URL/)
})

test('seeking finds the same note window in either direction, including empty charts', () => {
  const notes = parseChart(v2, options).notes
  assert.equal(lowerBound(notes, 4), 2)
  assert.equal(lowerBound(notes, 2), 0)
  assert.equal(lowerBound(notes, 100), 4)
  assert.equal(lowerBound([], 1), 0)
  assert.equal(formatTime(125.8), '02:05')
  assert.equal(formatTime(NaN), '00:00')
})
