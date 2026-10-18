import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'
import {
  filterEdgeIndices,
  filterNeuronIndices,
  neuronConnectionStats,
  parseBrainData,
  parseControls,
} from '../src/brain/data.ts'

const asset = JSON.parse(
  readFileSync(new URL('../public/data/brain/neurons.json', import.meta.url)),
)
const groups = JSON.parse(
  readFileSync(new URL('../public/data/brain/groups.json', import.meta.url)),
)
const edges = JSON.parse(readFileSync(new URL('../public/data/brain/edges.json', import.meta.url)))

test('the actual brain export keeps every index, ID, and position provenance', () => {
  const data = parseBrainData(asset, groups, edges)
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
  const data = parseBrainData(asset, groups, edges)
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
  const data = parseBrainData(asset, groups, edges)
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
    () => parseBrainData({ ...asset, position_um: asset.position_um.slice(3) }, groups, edges),
    /coordinates/,
  )
  assert.throws(
    () => parseBrainData({ ...asset, type: [-1, ...asset.type.slice(1)] }, groups, edges),
    /type metadata/,
  )
  assert.throws(() => parseBrainData({ ...asset, schema_version: 2 }, groups, edges), /schema/)
})

test('rejects unsafe body IDs, nonfinite coordinates, and out-of-range group indices', () => {
  assert.throws(
    () =>
      parseBrainData(
        { ...asset, body_id: [Number.MAX_SAFE_INTEGER + 1, ...asset.body_id.slice(1)] },
        groups,
        edges,
      ),
    /IDs/,
  )
  assert.throws(
    () =>
      parseBrainData(
        { ...asset, position_um: [Infinity, ...asset.position_um.slice(1)] },
        groups,
        edges,
      ),
    /coordinates/,
  )
  assert.throws(() => parseBrainData(asset, { invalid: [asset.n_neurons] }, edges), /group invalid/)
})

test('the complete trained-bundle topology matches the manifest and original neuron IDs', () => {
  const manifest = JSON.parse(
    readFileSync(new URL('../../bundles/real_fast/MANIFEST.json', import.meta.url)),
  )
  const data = parseBrainData(asset, groups, edges)
  assert.deepEqual(edges.body_id, asset.body_id)
  assert.equal(edges.n_edges, manifest.subgraph_fingerprint.n_edges)
  assert.equal(edges.n_neurons, manifest.subgraph_fingerprint.n_neurons)
  assert.equal(edges.n_synapses, manifest.subgraph_fingerprint.n_synapses)
  assert.deepEqual(
    filterEdgeIndices(data, filterNeuronIndices(data, '', '')),
    Array.from({ length: edges.n_edges }, (_, i) => i),
  )
})

test('edge filtering requires both endpoints and selection respects edge direction', () => {
  // Include a self-loop and a reciprocal pair to guard against double counting.
  const topology = {
    ...edges,
    n_edges: 5,
    n_synapses: 30,
    pre: [0, 1, 0, 2, 3],
    post: [1, 0, 0, 1, 4],
    weight: [4, 5, 6, 7, 8],
    sign: [0, 1, 0, 2, 3].map((i) => asset.sign[i]),
  }
  const data = parseBrainData(asset, groups, topology)
  assert.deepEqual(filterEdgeIndices(data, [0, 1, 2, 3, 4]), [0, 1, 2, 3, 4])
  assert.deepEqual(filterEdgeIndices(data, [0, 1]), [0, 1, 2])
  assert.deepEqual(filterEdgeIndices(data, [0, 1, 2, 3, 4], 1), [0, 1, 3])
  assert.deepEqual(filterEdgeIndices(data, [0, 1], 1), [0, 1])
  assert.deepEqual(filterEdgeIndices(data, [], 1), [])
  assert.deepEqual(filterEdgeIndices(data, [2, 3], 1), [])
  assert.deepEqual(neuronConnectionStats(data, 0), {
    incoming: 2,
    outgoing: 2,
    incomingSynapses: 11,
    outgoingSynapses: 10,
  })
  assert.deepEqual(neuronConnectionStats(data, 1), {
    incoming: 2,
    outgoing: 1,
    incomingSynapses: 11,
    outgoingSynapses: 5,
  })
})

test('rejects mismatched neuron order and malformed or incomplete edge exports', () => {
  const parse = (changed) => parseBrainData(asset, groups, changed)
  // Neurons render before edges.json arrives; an absent table is not an error.
  assert.equal(parseBrainData(asset, groups).edges, null)
  assert.throws(() => parse(null), /schema/)
  assert.throws(() => parse({ ...edges, schema_version: 2 }), /schema/)
  assert.throws(() => parse({ ...edges, body_id: [...edges.body_id].reverse() }), /ordering/)
  assert.throws(() => parse({ ...edges, n_neurons: asset.n_neurons - 1 }), /ordering/)
  assert.throws(() => parse({ ...edges, pre: edges.pre.slice(1) }), /arrays/)
  assert.throws(() => parse({ ...edges, pre: [-1, ...edges.pre.slice(1)] }), /endpoints/)
  assert.throws(
    () => parse({ ...edges, post: [asset.n_neurons, ...edges.post.slice(1)] }),
    /endpoints/,
  )
  assert.throws(() => parse({ ...edges, weight: [NaN, ...edges.weight.slice(1)] }), /arrays/)
  assert.throws(() => parse({ ...edges, weight: [0.5, ...edges.weight.slice(1)] }), /synapse/)
  assert.throws(() => parse({ ...edges, sign: [2, ...edges.sign.slice(1)] }), /sign/)
  assert.throws(() => parse({ ...edges, n_synapses: edges.n_synapses + 1 }), /total synapse/)
})

test('controls.json gives the lesionable sections and each mood with its range', () => {
  const raw = JSON.parse(readFileSync(new URL('../public/data/brain/controls.json', import.meta.url)))
  const controls = parseControls(raw)
  assert.ok(controls.sections.some((s) => s.group === 'looming_detectors'))
  const arousal = controls.moods.find((m) => m.name === 'arousal')
  assert.ok(arousal.min < 0 && arousal.max > 0)
  const inverted = { ...raw, moods: [{ ...raw.moods[0], min: 1, max: -1 }] }
  assert.throws(() => parseControls(inverted), /controls/)
})
