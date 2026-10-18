import assert from 'node:assert/strict'
import { test } from 'node:test'
import { emptySession, SessionClient } from '../src/game/session-client.ts'
import { flyMotion } from '../src/game/fly-motion.ts'
import { interventionsMessage, noControls } from '../src/game/interventions.ts'

class Socket extends EventTarget {
  readyState = 0
  sent = []
  send(data) { this.sent.push(JSON.parse(data)) }
  close() { this.readyState = 3; this.dispatchEvent(new Event('close')) }
  open() { this.readyState = 1; this.dispatchEvent(new Event('open')) }
  receive(data) { this.dispatchEvent(new MessageEvent('message', { data: JSON.stringify(data) })) }
  receiveActivity(kind, step, values) {
    const data = new ArrayBuffer(8 + values.length)
    const view = new DataView(data)
    view.setUint8(0, kind)
    view.setUint32(4, step, true)
    new Int8Array(data, 8).set(values)
    this.dispatchEvent(new MessageEvent('message', { data }))
  }
}
const score = { score: 75, combo: 3, max_combo: 3, hits: 3, misses: 1, accuracy: 0.75 }
const ready = { type: 'ready', map_id: 'abc12_hard', slots: [{ slot: 0, brain: 'expert' }] }
const brainReady = { ...ready, slots: [{ slot: 0, brain: 'real' }], n_neurons: 3, cell_types: ['a', 'b'] }
const frame = { type: 'frame', slot: 0, render_state: { t: 2, step: 60, score, sabers: [{ color: 0, cell: 3, cut: 1 }] } }
function setup() {
  const sockets = []
  const state = { ...emptySession }
  const activity = []
  let time = 0, failures = 0
  const client = new SessionClient({
    url: () => 'ws://test/ws/session', getTime: () => time,
    onChange: (next) => Object.assign(state, next), onFailure: () => failures++,
    onActivity: (frame) => activity.push([frame.kind, frame.step, [...frame.values]]),
    makeSocket: () => { const socket = new Socket(); sockets.push(socket); return socket },
  })
  return { client, state, sockets, activity, time: (t) => time = t, failures: () => failures }
}

test('start waits for ready, drives real motor frames, handles score/events and preserves pause state', async (t) => {
  const s = setup()
  t.after(() => s.client.disconnect())
  const pending = s.client.prepare('abc12_hard', 'expert')
  const socket = s.sockets[0]
  socket.open()
  assert.deepEqual(socket.sent, [{ type: 'start', game: 'saber', brains: ['expert'], map_id: 'abc12_hard', seed: 0 }])
  assert.equal(s.state.status, 'connecting')
  socket.receive(ready)
  await pending
  s.time(1.5)
  s.client.play()
  assert.deepEqual(socket.sent.at(-1), { type: 'play', song_t: 1.5 })
  socket.receive(frame)
  assert.deepEqual(s.state.score, score)
  assert.equal(flyMotion.sample().active, true)
  socket.receive({ type: 'event', slot: 0, note_id: 4, kind: 'hit', t: 2 })
  assert.equal(s.state.removed.has(4), true)
  s.client.pause()
  socket.receive({ ...frame, render_state: { ...frame.render_state, score: { ...score, score: 99 } } })
  assert.equal(s.state.score.score, 75)
  s.client.play()
  assert.equal(socket.sent.filter((m) => m.type === 'play').length, 1, 'resume must preserve accumulated score')
  assert.deepEqual(socket.sent.at(-1), { type: 'pause', paused: false })
  s.time(0.2)
  s.client.seek()
  assert.equal(s.state.removed.size, 0)
  assert.equal(s.state.score, null)
  assert.equal(socket.sent.filter((m) => m.type === 'play').at(-1).song_t, 0.2)
  socket.receive({ type: 'ended', slot: 0, score })
  assert.equal(s.state.status, 'ended')
  socket.close()
  assert.deepEqual(s.state.score, score)
})

test('disconnect rejects pending starts; late messages cannot mutate a replacement session', async (t) => {
  const s = setup()
  t.after(() => s.client.disconnect())
  const pending = s.client.prepare('abc12_hard', 'expert')
  const rejection = assert.rejects(pending, /cancelled/)
  const old = s.sockets[0]
  s.client.disconnect()
  await rejection
  const next = s.client.prepare('abc12_hard', 'expert')
  const socket = s.sockets[1]
  old.receive(ready)
  assert.equal(s.state.status, 'connecting')
  socket.open(); socket.receive(ready); await next
  s.client.play()
  socket.close()
  assert.equal(s.state.status, 'error')
  assert.equal(s.failures(), 1)
  assert.equal(flyMotion.sample().active, false)
})

test('server errors reject readiness with the actual error, and another play can reconnect', async (t) => {
  const s = setup()
  t.after(() => s.client.disconnect())
  const pending = s.client.prepare('abc12_hard', 'expert')
  const rejection = assert.rejects(pending, /unknown brain/)
  s.sockets[0].open()
  s.sockets[0].receive({ type: 'error', message: 'unknown brain' })
  await rejection
  assert.equal(s.state.status, 'error')
  const retry = s.client.prepare('abc12_hard', 'expert')
  s.sockets[1].open(); s.sockets[1].receive(ready); await retry
  assert.equal(s.state.status, 'ready')
})

test('ignores other slots, malformed scores, duplicate frames and invalid events', async (t) => {
  const s = setup()
  t.after(() => s.client.disconnect())
  const pending = s.client.prepare('abc12_hard', 'expert')
  const socket = s.sockets[0]
  socket.open(); socket.receive(ready); await pending; s.client.play()
  socket.receive({ ...frame, slot: 1 })
  socket.receive({ ...frame, render_state: { ...frame.render_state, score: { ...score, accuracy: 2 } } })
  socket.receive({ type: 'event', slot: 0, kind: 'hit', note_id: -1 })
  assert.equal(s.state.score, null)
  assert.equal(s.state.removed.size, 0)
  socket.receive(frame)
  socket.receive({ ...frame, render_state: { ...frame.render_state, score: { ...score, combo: 100 } } })
  assert.equal(s.state.score.combo, 3)
})

test('passes brain activity on while playing, in step order, and again after a seek', async (t) => {
  const s = setup()
  t.after(() => s.client.disconnect())
  const pending = s.client.prepare('abc12_hard', 'real')
  const socket = s.sockets[0]
  socket.open()
  socket.receive({ ...ready, slots: [{ slot: 0, brain: 'real' }], n_neurons: 3, cell_types: ['a', 'b'] })
  await pending
  socket.receiveActivity(2, 1, [1, 1])
  assert.deepEqual(s.activity, [], 'not playing yet')
  s.client.play()
  socket.receiveActivity(1, 60, [-5, 0, 7])
  socket.receiveActivity(2, 60, [1, -1])
  socket.receiveActivity(2, 59, [9, 9])
  socket.receiveActivity(2, 61, [9, 9, 9])
  assert.deepEqual(s.activity, [[1, 60, [-5, 0, 7]], [2, 60, [1, -1]]], 'stale and malformed dropped')
  s.time(0.1)
  s.client.seek()
  socket.receiveActivity(2, 3, [2, 2])
  assert.deepEqual(s.activity.at(-1), [2, 3, [2, 2]])
})

test('brain controls go out live and again after every ready, for brains only', async (t) => {
  const s = setup()
  t.after(() => s.client.disconnect())
  const lesioned = { ...noControls, lesion: ['looming_detectors'], blindLeft: true }
  s.client.setInterventions(lesioned)
  const pending = s.client.prepare('abc12_hard', 'real')
  const socket = s.sockets[0]
  socket.open(); socket.receive(brainReady); await pending
  assert.deepEqual(socket.sent.at(-1), interventionsMessage(lesioned), 'a new start clears them on the server')
  s.client.play()
  const aroused = { ...lesioned, mood: { arousal: 0.2 } }
  s.client.setInterventions(aroused)
  assert.deepEqual(socket.sent.at(-1), interventionsMessage(aroused))

  const expert = s.client.prepare('abc12_hard', 'expert')
  s.sockets[1].open(); s.sockets[1].receive(ready); await expert
  s.client.setInterventions(lesioned)
  assert.equal(s.sockets[1].sent.some((m) => m.type === 'set_interventions'), false, 'the expert has no brain')
})

test('a server error after ready is shown without ending the session', async (t) => {
  const s = setup()
  t.after(() => s.client.disconnect())
  const pending = s.client.prepare('abc12_hard', 'real')
  const socket = s.sockets[0]
  socket.open(); socket.receive(brainReady); await pending; s.client.play()
  socket.receive({ type: 'error', message: 'unknown group' })
  assert.equal(s.state.status, 'playing')
  assert.match(s.state.controlsError, /unknown group/)
  assert.equal(s.failures(), 0)
  socket.receive(frame)
  assert.deepEqual(s.state.score, score)
  s.client.setInterventions(noControls)
  assert.equal(s.state.controlsError, '')
})
