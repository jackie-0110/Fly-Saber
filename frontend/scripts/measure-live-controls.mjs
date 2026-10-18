// Live effect of each brain and eye control on a served brain (results/controls/live_controls.md).
// One condition per session, the same song window each time, counted from `event` messages.
//
//   node --experimental-strip-types scripts/measure-live-controls.mjs [map] [start_s] [seconds] [first-words]
//   BACKEND=wss://host/ws/session ... (default: the public API)
// `first-words` limits the run to conditions starting with those words, comma separated.
import { interventionsMessage, noControls } from '../src/game/interventions.ts'

const BACKEND = process.env.BACKEND ?? 'wss://api.jackie-courtney.com/ws/session'
const BRAIN = process.env.BRAIN ?? 'real'

const MAP = process.argv[2] ?? '304be_expertplus'
const START = Number(process.argv[3] ?? 20)
const SECONDS = Number(process.argv[4] ?? 15)
const conditions = [
  ['untouched', {}],
  ['lesion motion detectors', { lesion: ['motion_detectors'] }],
  ['lesion looming detectors', { lesion: ['looming_detectors'] }],
  ['stimulate giant fiber 5', { stimulate: { giant_fiber: 5 } }],
  ...[2, 4, 6, 10, 20].map((v) => [`blur ${v} deg`, { blurDeg: v }]),
  ...[0.1, 0.2, 0.3, 0.5, 1].map((v) => [`facets lost ${v}`, { facetDropout: v }]),
  ...[0.02, 0.05, 0.1, 0.15, 0.3].map((v) => [`visual noise ${v}`, { eyeNoise: v }]),
  ...[0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5].map((v) => [
    `neural noise ${v}`,
    { noise: v },
  ]),
]
const only = process.argv[5]?.split(',')

async function run(name, change) {
  const ws = new WebSocket(BACKEND)
  ws.binaryType = 'arraybuffer'
  const counts = { hit: 0, miss: 0, bad_cut: 0, bomb: 0 }
  const errors = []
  let ready
  const readyP = new Promise((r) => (ready = r))
  ws.onmessage = (e) => {
    if (typeof e.data !== 'string') return
    const m = JSON.parse(e.data)
    if (m.type === 'ready') ready(m)
    else if (m.type === 'event') counts[m.kind] = (counts[m.kind] ?? 0) + 1
    else if (m.type === 'error') errors.push(m.message)
  }
  await new Promise((res, rej) => {
    ws.onopen = res
    ws.onerror = rej
  })
  ws.send(JSON.stringify({ type: 'start', game: 'saber', brains: [BRAIN], map_id: MAP, seed: 0 }))
  await Promise.race([
    readyP,
    new Promise((_, rej) => setTimeout(() => rej(new Error('no ready')), 90000)),
  ])
  ws.send(JSON.stringify(interventionsMessage({ ...noControls, ...change })))
  await new Promise((r) => setTimeout(r, 3000)) // let the server re-centre the readout
  ws.send(JSON.stringify({ type: 'play', song_t: START }))
  await new Promise((r) => setTimeout(r, SECONDS * 1000))
  ws.close()
  const judged = counts.hit + counts.miss + counts.bad_cut
  const rate = judged ? ((100 * counts.hit) / judged).toFixed(1) : 'n/a'
  console.log(
    `${name.padEnd(42)} hits ${String(counts.hit).padStart(3)} / ${String(judged).padStart(3)} judged = ${rate}%  (bad cuts ${counts.bad_cut}, bombs ${counts.bomb})${errors.length ? ' ERR ' + errors.join('; ') : ''}`,
  )
  await new Promise((r) => setTimeout(r, 1500))
}

console.log(`map ${MAP}, song ${START}s to ${START + SECONDS}s, brain ${BRAIN}`)
for (const [name, change] of conditions)
  if (!only || only.includes(name.split(' ')[0])) await run(name, change)
