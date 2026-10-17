import assert from 'node:assert/strict'
import { test } from 'node:test'
import { emptySession, SessionClient } from '../src/game/session-client.ts'
import { flyMotion } from '../src/game/fly-motion.ts'

class Socket extends EventTarget {
  readyState = 0
  sent = []
  send(data) { this.sent.push(JSON.parse(data)) }
  close() { this.readyState = 3; this.dispatchEvent(new Event('close')) }
  open() { this.readyState = 1; this.dispatchEvent(new Event('open')) }
  receive(data) { this.dispatchEvent(new MessageEvent('message', { data: JSON.stringify(data) })) }
}
const score = { score: 75, combo: 3, max_combo: 3, hits: 3, misses: 1, accuracy: 0.75 }
const ready = { type: 'ready', map_id: 'abc12_hard', slots: [{ slot: 0, brain: 'expert' }] }
const frame = { type: 'frame', slot: 0, render_state: { t: 2, step: 60, score, sabers: [{ color: 0, cell: 3, cut: 1 }] } }
function setup() {
  const sockets = []
  const state = { ...emptySession }
  let time = 0, failures = 0
  const client = new SessionClient({
    url: () => 'ws://test/ws/session', getTime: () => time,
    onChange: (next) => Object.assign(state, next), onFailure: () => failures++,
    makeSocket: () => { const socket = new Socket(); sockets.push(socket); return socket },
  })
  return { client, state, sockets, time: (t) => time = t, failures: () => failures }
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
