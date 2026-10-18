import assert from 'node:assert/strict'
import { test } from 'node:test'
import { AudioTransport } from '../src/features/session/transport.ts'

globalThis.requestAnimationFrame = () => 1
globalThis.cancelAnimationFrame = () => {}

function setup() {
  const sources = []
  const snapshots = []
  const context = { currentTime: 0, state: 'running', destination: {}, addEventListener() {},
    resume: async () => {},
    createGain: () => ({ gain: { value: 1, setTargetAtTime() {} }, connect() {} }),
    createBufferSource() {
      const source = { playbackRate: { value: 1 }, connect() {}, disconnect() {},
        start(when, offset) { this.offset = offset }, stop() { this.stopped = true }, onended: null }
      sources.push(source)
      return source
    },
  }
  const player = new AudioTransport((snapshot) => snapshots.push(snapshot), () => context)
  player.load({ duration: 120 })
  return { player, context, sources, snapshots }
}

test('play, pause, scrub backward, and resume use audio time without drift or stale sources', async () => {
  const { player, context, sources } = setup()
  await player.play()
  context.currentTime = 20
  assert.equal(player.getTime(), 20)
  player.pause()
  context.currentTime = 40
  assert.equal(player.getTime(), 20)
  assert.equal(sources[0].stopped, true)
  player.seek(5)
  await player.play()
  assert.equal(sources[1].offset, 5)
  context.currentTime = 43
  assert.equal(player.getTime(), 8)
  player.seek(1)
  assert.equal(sources[2].offset, 1)
  context.currentTime = 44
  assert.equal(player.getTime(), 2)
})

test('rate changes preserve the song position and apply only to future clock movement', async () => {
  const { player, context, sources } = setup()
  await player.play()
  context.currentTime = 10
  player.setRate(0.5)
  assert.equal(player.getTime(), 10)
  assert.equal(sources.at(-1).playbackRate.value, 0.5)
  context.currentTime = 20
  assert.equal(player.getTime(), 15)
})

test('end-of-track seek stops playback and play starts from zero; out-of-bounds seeks clamp', async () => {
  const { player, sources } = setup()
  await player.play()
  player.seek(999)
  assert.equal(player.snapshot().playing, false)
  assert.equal(player.getTime(), 120)
  await player.play()
  assert.equal(sources.at(-1).offset, 0)
  player.seek(-50)
  assert.equal(player.getTime(), 0)
})

test('pause or load during asynchronous audio resume cancels the pending play', async () => {
  const { player, context, sources } = setup()
  let resume
  context.resume = () => new Promise((resolve) => { resume = resolve })
  const playing = player.play()
  player.pause()
  resume()
  await playing
  assert.equal(sources.length, 0)
  assert.equal(player.snapshot().playing, false)
})

test('ended callbacks are detached when seeking and real completion updates the transport', async () => {
  const { player, sources } = setup()
  await player.play()
  player.seek(30)
  assert.equal(sources[0].onended, null)
  sources[1].onended()
  assert.equal(player.snapshot().playing, false)
  assert.equal(player.getTime(), 120)
})
