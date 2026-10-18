import assert from 'node:assert/strict'
import { test } from 'node:test'
import { attachFlyMotionStream, cellCenter, CUT_VECTORS, FlyMotionController, STROKE_SECONDS } from '../src/features/game/fly-motion.ts'

const frame = (step, sabers, slot = 0, t = step / 30) => ({ type: 'frame', slot, render_state: { step, t, sabers } })
const hand = (color, cell, cut = null) => ({ color, cell, cut })
const near = (a, b) => assert.ok(Math.abs(a - b) < 1e-8, `${a} != ${b}`)

const close = (actual, expected) => {
  if (typeof actual === 'number' && typeof expected === 'number') near(actual, expected)
  else if (actual && typeof actual === 'object') {
    assert.deepEqual(Object.keys(actual), Object.keys(expected))
    for (const key of Object.keys(actual)) close(actual[key], expected[key])
  } else assert.equal(actual, expected)
}

test('each saber reaches all twelve grid cells at the chart hit plane', () => {
  for (const color of [0, 1]) for (let cell = 0; cell < 12; cell++) {
    const motion = new FlyMotionController()
    assert.equal(motion.acceptMessage(frame(1, [hand(color, cell)]), 0), true)
    close(motion.sample(0.1).hands[color].center, cellCenter(cell))
  }
})

test('every directional stroke crosses the selected cell in the backend direction for both colors', () => {
  for (const color of [0, 1]) for (let cut = 0; cut < 4; cut++) {
    const motion = new FlyMotionController()
    motion.acceptMessage(frame(1, [hand(color, 6, cut)]), 0)
    const before = motion.sample(STROKE_SECONDS * 0.3).hands[color]
    const middle = motion.sample(STROKE_SECONDS / 2).hands[color]
    const after = motion.sample(STROKE_SECONDS * 0.8).hands[color]
    close(middle.center, cellCenter(6))
    const [dx, dy] = CUT_VECTORS[cut]
    assert.ok((after.center[0] - before.center[0]) * dx + (after.center[1] - before.center[1]) * dy > 0)
    assert.ok(middle.energy > 0.9)
  }
})

test('both sabers move independently and an omitted saber lowers in single-saber mode', () => {
  const motion = new FlyMotionController()
  const rest = motion.sample(0).hands[1]
  motion.acceptMessage(frame(1, [hand(0, 0, 0), hand(1, 11, 1)]), 0)
  close(motion.sample(0.07).hands.map((h) => h.center), [cellCenter(0), cellCenter(11)])
  motion.acceptMessage(frame(2, [hand(0, 4)]), 0.1)
  close(motion.sample(0.4).hands[1], rest)
})

test('30 Hz repeated commands do not restart the same stroke on each frame', () => {
  const motion = new FlyMotionController()
  for (let step = 0; step <= 2; step++) motion.acceptMessage(frame(step, [hand(0, 5, 3)]), step / 30)
  close(motion.sample(0.07).hands[0].center, cellCenter(5))
  const before = motion.sample(0.08).hands[0]
  motion.acceptMessage(frame(3, [hand(0, 9, 1)]), 0.08)
  close(motion.sample(0.08).hands[0], before)
})

test('no-cut holds position, idle lowers, and missing streams return to rest', () => {
  const motion = new FlyMotionController()
  const rest = motion.sample(0).hands
  motion.acceptMessage(frame(1, [hand(0, 11)]), 0)
  close(motion.sample(0.2).hands[0].center, cellCenter(11))
  close(motion.sample(1).hands, rest)
  motion.acceptMessage(frame(2, [hand(0, null, 2)]), 1)
  close(motion.sample(1.3).hands, rest)
})

test('paused poses freeze exactly and resume without a time jump', () => {
  const motion = new FlyMotionController()
  motion.acceptMessage(frame(1, [hand(0, 4, 2)]), 0)
  motion.setPaused(true, 0.05)
  const frozen = motion.sample(0.05)
  close(motion.sample(99), frozen)
  assert.equal(motion.acceptMessage(frame(2, [hand(0, 8, 3)]), 90), false)
  motion.setPaused(false, 100)
  close(motion.sample(100), frozen)
  assert.notDeepEqual(motion.sample(100.1), frozen)
})

test('slot filtering, malformed data, stale steps and backwards timestamps cannot corrupt a pose', () => {
  const motion = new FlyMotionController(1)
  const good = frame(4, [hand(1, 6, 2)], 1)
  assert.equal(motion.acceptMessage(frame(4, [hand(0, 3)]), 0), false)
  assert.equal(motion.acceptMessage(good, 0), true)
  const pose = motion.sample(0.07)
  for (const value of [null, {}, good, frame(3, [hand(1, 6)], 1), frame(5, [hand(1, 6)], 1, 0),
    frame(5, [hand(1, 12)], 1), frame(5, [hand(1, 3, 4)], 1), frame(5, [hand(1, NaN)], 1),
    frame(5, [hand(0, 3), hand(0, 4)], 1), frame(5, [], 1), frame(5, [hand(1, 3)], 1, Infinity)]) {
    assert.equal(motion.acceptMessage(value, 0.07), false)
    close(motion.sample(0.07), pose)
  }
})

test('reset permits a fresh timeline; disconnect while paused still settles to rest', () => {
  const motion = new FlyMotionController()
  const rest = motion.sample(0).hands
  motion.acceptMessage(frame(99, [hand(0, 8, 0)]), 0)
  motion.setPaused(true, 0.07)
  motion.release(10)
  close(motion.sample(10.3).hands, rest)
  motion.acceptMessage({ type: 'ready' }, 11)
  assert.equal(motion.acceptMessage(frame(0, [hand(1, 1)]), 11), true)
  motion.reset()
  close(motion.sample(12).hands, rest)
})

test('stream attachment consumes protocol JSON, ignores binary, and removes all listeners', () => {
  const socket = new EventTarget()
  const motion = new FlyMotionController()
  const detach = attachFlyMotionStream(socket, motion)
  socket.dispatchEvent(new MessageEvent('message', { data: '{bad' }))
  socket.dispatchEvent(new MessageEvent('message', { data: new ArrayBuffer(8) }))
  socket.dispatchEvent(new MessageEvent('message', { data: JSON.stringify(frame(1, [hand(0, 2)])) }))
  assert.equal(motion.sample().active, true)
  detach()
  motion.reset()
  socket.dispatchEvent(new MessageEvent('message', { data: JSON.stringify(frame(2, [hand(0, 4)])) }))
  assert.equal(motion.sample().active, false)
})


test('a one-tick cut completes through its cell before following a subsequent no-cut target', () => {
  const motion = new FlyMotionController()
  motion.acceptMessage(frame(0, [hand(0, 5, 0)]), 0)
  motion.acceptMessage(frame(1, [hand(0, 10)]), 1 / 30)
  close(motion.sample(0.07).hands[0].center, cellCenter(5))
  motion.acceptMessage(frame(2, [hand(0, 11)]), 2 / 30)
  close(motion.sample(0.07).hands[0].center, cellCenter(5))
  close(motion.sample(0.25).hands[0].center, cellCenter(11))
  motion.acceptMessage(frame(8, [hand(0, 11)]), 0.26)
  close(motion.sample(0.3).hands[0].center, cellCenter(11))
})


test('low-cell reaches and swings keep the saber grip and blade midpoint above the track', () => {
  for (const color of [0, 1]) for (let cell = 0; cell < 4; cell++) for (const cut of [null, 0, 1, 2, 3]) {
    const motion = new FlyMotionController()
    motion.acceptMessage(frame(0, [hand(color, cell, cut)]), 0)
    for (let t = 0; t < 0.35; t += 0.01) {
      const pose = motion.sample(t).hands[color]
      assert.ok(pose.center[1] >= 0.12 - 1e-8)
      assert.ok(pose.center[1] - Math.cos(pose.angle) * 0.88 >= 0.12 - 1e-8)
    }
  }
})
