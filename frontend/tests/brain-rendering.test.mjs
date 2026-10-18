import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'
import { connectionIndex, MOVING_EDGE_LIMIT } from '../src/brain/rendering.ts'

const readAsset = (name) =>
  JSON.parse(readFileSync(new URL(`../public/data/brain/${name}.json`, import.meta.url)))
const data = { asset: readAsset('neurons'), edges: readAsset('edges') }

test('the compact full-graph buffer preserves every directed edge, including multiplicity', () => {
  const edges = Array.from({ length: data.edges.n_edges }, (_, i) => i)
  const { index } = connectionIndex(data, edges, null)
  assert.ok(index instanceof Uint16Array)
  assert.equal(index.length, edges.length * 2)
  const expected = new Map()
  for (const edge of edges) {
    const key = `${data.edges.pre[edge]},${data.edges.post[edge]}`
    expected.set(key, (expected.get(key) ?? 0) + 1)
  }
  for (let i = 0; i < index.length; i += 2) {
    const key = `${index[i]},${index[i + 1]}`
    assert.ok(expected.get(key) > 0, `unexpected edge ${key}`)
    expected.set(key, expected.get(key) - 1)
  }
  assert.ok([...expected.values()].every((count) => count === 0))
  const compactBytes = data.asset.n_neurons * 3 * 4 + index.byteLength
  assert.ok(compactBytes < (edges.length * 6 * 4) / 4)
})

test('the movement prefix samples across the entire filtered graph', () => {
  const edges = Array.from({ length: data.edges.n_edges }, (_, i) => i).filter((i) => i % 3 === 1)
  const { index } = connectionIndex(data, edges, null)
  for (let i = 0; i < MOVING_EDGE_LIMIT; i++) {
    const edge = edges[Math.floor((i * edges.length) / MOVING_EDGE_LIMIT)]
    assert.equal(index[i * 2], data.edges.pre[edge])
    assert.equal(index[i * 2 + 1], data.edges.post[edge])
  }
})

test('small and empty views retain every edge, with separate selected direction groups', () => {
  const small = { asset: { n_neurons: 5 }, edges: { pre: [0, 1, 0, 2, 3], post: [1, 0, 0, 1, 4] } }
  const { index, outgoing } = connectionIndex(small, [0, 1, 2], 0)
  // Outgoing 0->1 and self-loop first; reciprocal 1->0 stays in the incoming group.
  assert.deepEqual([...index], [0, 1, 0, 0, 1, 0])
  assert.equal(outgoing, 4)
  assert.deepEqual([...connectionIndex(small, [1, 3], null).index], [1, 0, 2, 1])
  assert.equal(connectionIndex(small, [], null).index.length, 0)
})

test('larger future subgraphs use 32-bit indices without truncating endpoints', () => {
  const large = { asset: { n_neurons: 70_000 }, edges: { pre: [65_536], post: [69_999] } }
  const { index } = connectionIndex(large, [0], null)
  assert.ok(index instanceof Uint32Array)
  assert.deepEqual([...index], [65_536, 69_999])
})
