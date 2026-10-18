import assert from 'node:assert/strict'
import { test } from 'node:test'
import { decodeEyeImage, matchesLattice, mosaicGeometry, parseEyeLattice } from '../src/eyeview/eye-data.ts'

function packet() {
  const buffer = new ArrayBuffer(17)
  const view = new DataView(buffer)
  view.setUint8(0, 3)
  view.setUint8(1, 1)
  view.setUint32(4, 0x12345678, true)
  view.setUint16(8, 2, true)
  view.setUint16(10, 3, true)
  new Uint8Array(buffer, 12).set([0, 128, 255, 42, 99])
  return buffer
}

test('eye packets use unsigned luminance and little endian counts and step', () => {
  const frame = decodeEyeImage(packet())
  assert.equal(frame.slot, 1)
  assert.equal(frame.step, 0x12345678)
  assert.deepEqual([...frame.left], [0, 128])
  assert.deepEqual([...frame.right], [255, 42, 99])
})

test('invalid kind, slot, reserved bytes, counts and truncated payloads are rejected', () => {
  for (const [offset, value] of [[0, 2], [1, 2], [2, 1], [8, 0], [10, 4]]) {
    const data = packet()
    new DataView(data).setUint8(offset, value)
    assert.equal(decodeEyeImage(data), null)
  }
  for (let length = 0; length < 17; length++)
    assert.equal(decodeEyeImage(packet().slice(0, length)), null)
})

test('lattice validation rejects malformed coordinates and mismatching image counts', () => {
  const lattice = { left: [[-10, 0], [-5, 4.33]], right: [[10, 0], [5, 4.33], [0, 0]] }
  assert.equal(parseEyeLattice(lattice), lattice)
  assert.equal(matchesLattice(decodeEyeImage(packet()), lattice), true)
  assert.equal(matchesLattice(decodeEyeImage(packet()), { ...lattice, left: [[0, 0]] }), false)
  for (const value of [null, {}, { left: [], right: [] }, { left: [[0, NaN]], right: [[0, 0]] }])
    assert.throws(() => parseEyeLattice(value))
  const geometry = mosaicGeometry(lattice)
  assert.ok(geometry.width > 10 && geometry.height > 4.33)
  assert.equal(geometry.hex.split(' ').length, 6)
})
