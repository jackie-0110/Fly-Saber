import assert from 'node:assert/strict'
import { test } from 'node:test'
import { liveActivity, STALE_MS } from '../src/features/brain/live-activity.ts'

test('keeps the latest neuron and cell-type activity apart, each with its own step', () => {
  const neurons = Int8Array.of(-3, 0, 9)
  const types = Int8Array.of(4, -1)
  liveActivity.receive({ kind: 1, slot: 0, step: 60, values: neurons })
  liveActivity.receive({ kind: 2, slot: 0, step: 61, values: types })
  assert.equal(liveActivity.neurons, neurons)
  assert.equal(liveActivity.neuronStep, 60, 'a cell-type update must not trigger a neuron redraw')
  assert.equal(liveActivity.types, types)
  assert.equal(liveActivity.typeStep, 61)
  assert.ok(liveActivity.receivedAt > 0)
  liveActivity.clear()
  assert.deepEqual(
    [liveActivity.neurons, liveActivity.types, liveActivity.neuronStep, liveActivity.receivedAt],
    [null, null, -1, 0],
  )
})

test('reads as live until activity stops arriving, then tells listeners it went stale', (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] })
  let notified = 0
  const unsubscribe = liveActivity.subscribe(() => notified++)
  t.after(() => { unsubscribe(); liveActivity.clear() })
  liveActivity.receive({ kind: 1, slot: 0, step: 1, values: Int8Array.of(1) })
  assert.equal(liveActivity.live(liveActivity.receivedAt), true)
  assert.equal(liveActivity.live(liveActivity.receivedAt + STALE_MS), false)
  assert.equal(notified, 1)
  t.mock.timers.tick(STALE_MS)
  assert.equal(notified, 2, 'the stale fade needs a redraw without a new message')
})
