import { expect, test } from '@playwright/test'
import type { Page, WebSocketRoute } from '@playwright/test'
import { wav } from '../tests/song-fixtures.ts'

const map = {
  schema_version: 1,
  map_id: 'abc12_hard',
  split: 'heldout',
  song: {
    title: 'Live Fixture',
    artist: 'Tester',
    mapper: 'Fixture mapper',
    bpm: 120,
    attribution: 'Original silent test audio',
    license: 'CC0',
  },
  difficulty: { name: 'Hard', note_speed_mps: 7, spawn_distance_m: 12 },
  notes: [{ id: 0, time_sec: 2, lane: 1, layer: 1, color: 0, cut_direction: 1, is_bomb: false }],
  walls: [],
}
const score = { score: 100, combo: 1, max_combo: 1, hits: 1, misses: 0, accuracy: 1 }
async function load(page: Page) {
  await page.goto('/')
  await page.getByRole('tab', { name: 'Songs', exact: true }).click()
  await page.getByRole('button', { name: 'Load live Live Fixture Hard', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Live Fixture', level: 1 })).toBeVisible()
}

test.beforeEach(async ({ page }) => {
  await page.route('https://fonts.googleapis.com/**', (route) =>
    route.fulfill({ contentType: 'text/css', body: '' }),
  )
  await page.route('**/api/games', (route) =>
    route.fulfill({ json: [{ id: 'saber', label: 'Fly Saber' }] }),
  )
  await page.route('**/api/brains', (route) =>
    route.fulfill({
      json: [
        { id: 'expert', label: 'Scripted expert', description: 'Reads the map.' },
        { id: 'real', label: 'Real wiring', description: 'A trained fly.' },
      ],
    }),
  )
  await page.route('**/api/maps', (route) => route.fulfill({ json: [{ ...map, note_count: 1 }] }))
  await page.route('**/api/maps/abc12_hard', (route) => route.fulfill({ json: map }))
  await page.route('**/api/maps/abc12_hard/audio', (route) =>
    route.fulfill({ body: Buffer.from(wav()), contentType: 'audio/wav' }),
  )
})

test('server catalog, brain choice, live motor/score events, clocks, pause, seek, restart and end', async ({
  page,
}, testInfo) => {
  const messages: Record<string, unknown>[] = []
  let socket: WebSocketRoute
  await page.routeWebSocket('**/ws/session', (ws) => {
    socket = ws
    ws.onMessage((raw) => {
      const m = JSON.parse(String(raw))
      messages.push(m)
      if (m.type === 'start')
        ws.send(
          JSON.stringify({
            type: 'ready',
            map_id: m.map_id,
            slots: [{ slot: 0, brain: m.brains[0] }],
            n_neurons: 0,
            cell_types: [],
            step_hz: 30,
          }),
        )
      if (m.type === 'play') {
        ws.send(
          JSON.stringify({
            type: 'frame',
            slot: 0,
            render_state: {
              t: m.song_t,
              step: 1,
              score,
              sabers: [
                { color: 0, cell: 5, cut: 1 },
                { color: 1, cell: null, cut: null },
              ],
            },
          }),
        )
        ws.send(JSON.stringify({ type: 'event', slot: 0, t: m.song_t, kind: 'hit', note_id: 0 }))
      }
    })
  })
  await load(page)
  await expect(page.getByLabel('Brain player', { exact: true })).toHaveValue('real')
  await page.getByLabel('Brain player', { exact: true }).selectOption('expert')
  await expect(page.getByText('Unseen map · Live play', { exact: true })).toBeVisible()
  await expect(page.getByText('Original silent test audio', { exact: true })).toBeVisible()
  await expect(page.getByRole('combobox', { name: 'Playback speed' })).toBeDisabled()
  await page.getByRole('button', { name: 'Play song', exact: true }).click()
  await expect(page.getByText('Live · playing', { exact: true })).toBeVisible()
  await expect(page.locator('.performance-values')).toContainText('100.0%')
  await expect(page.locator('.session-readouts')).toContainText('Left: cell 5')
  await expect.poll(() => messages.filter((m) => m.type === 'clock').length).toBeGreaterThan(0)
  expect(messages[0]).toEqual({
    type: 'start',
    game: 'saber',
    brains: ['expert'],
    map_id: 'abc12_hard',
    seed: 0,
  })
  await page.getByRole('button', { name: 'Pause song', exact: true }).click()
  await expect(page.getByText('Live · paused', { exact: true })).toBeVisible()
  const plays = messages.filter((m) => m.type === 'play').length
  await page.getByRole('button', { name: 'Play song', exact: true }).click()
  await expect(page.getByText('Live · playing', { exact: true })).toBeVisible()
  expect(messages.filter((m) => m.type === 'play').length).toBe(plays)
  await page.getByRole('button', { name: 'Restart song', exact: true }).click()
  await expect(page.getByRole('slider', { name: 'Song position', exact: true })).toHaveValue('0')
  await page.getByRole('button', { name: 'Play song', exact: true }).click()
  await expect.poll(() => messages.filter((m) => m.type === 'play').length).toBeGreaterThan(plays)
  socket!.send(JSON.stringify({ type: 'ended', slot: 0, score }))
  await expect(page.getByText('Live · ended', { exact: true })).toBeVisible()
  await page.getByRole('tab', { name: 'Songs', exact: true }).click()
  await page.screenshot({ path: testInfo.outputPath('live-session.png'), scale: 'css' })
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
})

test('socket failure stops audio, shows recovery, and play reconnects', async ({ page }) => {
  let connections = 0
  await page.routeWebSocket('**/ws/session', (ws) => {
    connections++
    ws.onMessage((raw) => {
      const m = JSON.parse(String(raw))
      if (m.type === 'start') {
        if (connections === 1)
          ws.send(JSON.stringify({ type: 'error', message: 'brain unavailable' }))
        else
          ws.send(
            JSON.stringify({
              type: 'ready',
              map_id: m.map_id,
              slots: [{ slot: 0, brain: m.brains[0] }],
            }),
          )
      }
    })
  })
  await load(page)
  await page.getByRole('button', { name: 'Play song', exact: true }).click()
  await expect(
    page.getByText('Live session: brain unavailable. Press play to retry.', { exact: true }),
  ).toBeVisible()
  await expect(page.getByRole('button', { name: 'Play song', exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'Play song', exact: true }).click()
  await expect(page.getByText('Live · playing', { exact: true })).toBeVisible()
  expect(connections).toBe(2)
})

test('empty and failed catalogs offer a refresh without community downloads', async ({ page }) => {
  let offline = true
  await page.route('**/api/maps', (route) =>
    offline ? route.fulfill({ status: 503 }) : route.fulfill({ json: [] }),
  )
  await page.goto('/')
  await page.getByRole('tab', { name: 'Songs', exact: true }).click()
  await expect(page.getByRole('alert')).toContainText('503')
  offline = false
  await page.getByRole('button', { name: 'Refresh server catalog', exact: true }).click()
  await expect(page.getByText('No server songs are available yet.', { exact: false })).toBeVisible()
})
