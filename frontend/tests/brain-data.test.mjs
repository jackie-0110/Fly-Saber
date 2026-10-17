import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'
import { filterNeuronIndices, parseBrainData } from '../src/brain/data.ts'

const asset = JSON.parse(
  readFileSync(new URL('../public/data/brain/neurons.json', import.meta.url)),
)
const groups = JSON.parse(
  readFileSync(new URL('../public/data/brain/groups.json', import.meta.url)),
)

test('the actual brain export keeps every index, ID, and position provenance', () => {
  const data = parseBrainData(asset, groups)
  assert.equal(data.positions.length, asset.n_neurons * 3)
  assert.deepEqual(
    filterNeuronIndices(data, '', ''),
    Array.from({ length: asset.n_neurons }, (_, i) => i),
  )
  assert.equal(data.asset, asset)
  assert.ok(data.radius > 0)
  assert.ok(asset.position_source.includes(asset.position_sources.indexOf('type_mean')))
})

test('centering and rigid rotation preserve relative distances with one uniform scale', () => {
  const data = parseBrainData(asset, groups)
  const span = Math.max(...asset.bounds_um.max.map((v, axis) => v - asset.bounds_um.min[axis]))
  const scale = 2 / span
  for (const i of [0, 50, 1000, asset.n_neurons - 1]) {
    const rawDelta = asset.position_um
      .slice(i * 3, i * 3 + 3)
      .map((v, axis) => (v - asset.position_um[axis]) * scale)
    const actualDelta = data.positions
      .slice(i * 3, i * 3 + 3)
      .map((v, axis) => v - data.positions[axis])
    assert.ok(Math.abs(Math.hypot(...rawDelta) - Math.hypot(...actualDelta)) < 1e-6)
    assert.ok(Math.abs(actualDelta[0] - rawDelta[0]) < 1e-6)
    assert.ok(Math.abs(actualDelta[1] + rawDelta[2]) < 1e-6)
    assert.ok(Math.abs(actualDelta[2] - rawDelta[1]) < 1e-6)
  }
})

test('group and cell-type filters return original backend indices, including an empty intersection', () => {
  const data = parseBrainData(asset, groups)
  assert.deepEqual(filterNeuronIndices(data, 'T4', ''), groups.T4)
  const type = asset.type[groups.T4[0]]
  assert.deepEqual(
    filterNeuronIndices(data, 'T4', String(type)),
    groups.T4.filter((i) => asset.type[i] === type),
  )
  const absentType = asset.type.find((code) => !groups.T4.some((i) => asset.type[i] === code))
  assert.deepEqual(filterNeuronIndices(data, 'T4', String(absentType)), [])
})

test('rejects truncated positions, invalid lookup codes, and unsupported versions', () => {
  assert.throws(
    () => parseBrainData({ ...asset, position_um: asset.position_um.slice(3) }, groups),
    /coordinates/,
  )
  assert.throws(
    () => parseBrainData({ ...asset, type: [-1, ...asset.type.slice(1)] }, groups),
    /type metadata/,
  )
  assert.throws(() => parseBrainData({ ...asset, schema_version: 2 }, groups), /schema/)
})

test('rejects unsafe body IDs, nonfinite coordinates, and out-of-range group indices', () => {
  assert.throws(
    () =>
      parseBrainData(
        { ...asset, body_id: [Number.MAX_SAFE_INTEGER + 1, ...asset.body_id.slice(1)] },
        groups,
      ),
    /IDs/,
  )
  assert.throws(
    () =>
      parseBrainData({ ...asset, position_um: [Infinity, ...asset.position_um.slice(1)] }, groups),
    /coordinates/,
  )
  assert.throws(() => parseBrainData(asset, { invalid: [asset.n_neurons] }), /group invalid/)
})
