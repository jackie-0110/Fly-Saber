import assert from 'node:assert/strict'
import { test } from 'node:test'
import { decodeActivity } from '../src/game/activity.ts'

const sizes = { neurons: 4, types: 2 }
function message(kind, slot, step, values, reserved = 0) {
  const data = new ArrayBuffer(8 + values.length)
  const view = new DataView(data)
  view.setUint8(0, kind)
  view.setUint8(1, slot)
  view.setUint16(2, reserved, true)
  view.setUint32(4, step, true)
  new Int8Array(data, 8).set(values)
  return data
}

test('decodes neuron and cell-type activity with signed change from rest', () => {
  const neurons = decodeActivity(message(1, 1, 70000, [-127, 0, 5, 127]), sizes)
  assert.deepEqual({ ...neurons, values: [...neurons.values] }, {
    kind: 1, slot: 1, step: 70000, values: [-127, 0, 5, 127],
  })
  assert.deepEqual([...decodeActivity(message(2, 0, 3, [-1, 2]), sizes).values], [-1, 2])
})

test('rejects messages that do not match the session', () => {
  assert.equal(decodeActivity(message(1, 0, 3, [1, 2, 3]), sizes), null, 'wrong neuron count')
  assert.equal(decodeActivity(message(2, 0, 3, [1, 2, 3, 4]), sizes), null, 'wrong type count')
  assert.equal(decodeActivity(message(3, 0, 3, [1, 2]), sizes), null, 'eye images are not activity')
  assert.equal(decodeActivity(message(2, 0, 3, [1, 2], 1), sizes), null, 'reserved bytes set')
  assert.equal(decodeActivity(new ArrayBuffer(5), sizes), null, 'shorter than the header')
})
