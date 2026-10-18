import { expect, test } from '@playwright/test'
import type { WebSocketRoute } from '@playwright/test'
import { wav } from '../tests/song-fixtures.ts'

const lattice = {
  left: [
    [-10, 0],
    [-5, 4.33],
    [0, 0],
  ],
  right: [
    [10, 0],
    [5, 4.33],
    [0, 0],
  ],
}
const song = {
  title: 'Eye Test',
  artist: 'Test Artist',
  mapper: 'Test Mapper',
  license: 'CC BY 4.0',
  bpm: 120,
}
const map = {
  schema_version: 1,
  map_id: 'abc12_hard',
  split: 'heldout',
  song,
  difficulty: { name: 'Hard', note_speed_mps: 10, spawn_distance_m: 12 },
  notes: [{ id: 0, time_sec: 10, lane: 1, layer: 1, color: 0, cut_direction: 1, is_bomb: false }],
  walls: [],
}
const score = { score: 100, combo: 1, max_combo: 1, hits: 1, misses: 0, accuracy: 1 }

function image(step: number, left = [0, 128, 255], right = [255, 128, 0]) {
  const data = Buffer.alloc(12 + left.length + right.length)
  data.writeUInt8(3, 0)
  data.writeUInt32LE(step, 4)
  data.writeUInt16LE(left.length, 8)
  data.writeUInt16LE(right.length, 10)
  data.set([...left, ...right], 12)
  return data
}

test.beforeEach(async ({ page }) => {
  await page.route('**/api/games', (route) =>
    route.fulfill({ json: [{ id: 'saber', label: 'Fly Saber' }] }),
  )
  await page.route('**/api/brains', (route) =>
    route.fulfill({
      json: [
        { id: 'real', label: 'Real fly brain', description: 'Fixture brain' },
        { id: 'expert', label: 'Scripted expert', description: 'Fixture player' },
      ],
    }),
  )
  await page.route('**/api/maps', (route) => route.fulfill({ json: [{ ...map, note_count: 1 }] }))
  await page.route('**/api/maps/abc12_hard', (route) => route.fulfill({ json: map }))
  await page.route('**/api/maps/abc12_hard/audio', (route) =>
    route.fulfill({ body: Buffer.from(wav()), contentType: 'audio/wav' }),
  )
  await page.route('**/api/eye_lattice?*', (route) => route.fulfill({ json: lattice }))
  await page.route('https://fonts.googleapis.com/**', (route) =>
    route.fulfill({ contentType: 'text/css', body: '' }),
  )
  await page.route('https://api.beatsaver.com/**', (route) => route.fulfill({ json: { docs: [] } }))
})

test('live unsigned eye pixels fill both mosaics and inset, survive pause and clear on restart and disconnect', async ({
  page,
}) => {
  let socket: WebSocketRoute
  let step = 2
  await page.routeWebSocket('**/ws/session', (ws) => {
    socket = ws
    ws.onMessage((raw) => {
      const message = JSON.parse(String(raw))
      if (message.type === 'start')
        ws.send(
          JSON.stringify({
            type: 'ready',
            map_id: map.map_id,
            slots: [{ slot: 0, brain: 'real' }],
            step_hz: 30,
            n_neurons: 3,
            cell_types: ['a'],
          }),
        )
      if (message.type === 'play') {
        ws.send(
          JSON.stringify({
            type: 'frame',
            slot: 0,
            render_state: { t: step / 30, step, score, sabers: [] },
          }),
        )
        ws.send(image(step))
      }
      if (message.type === 'set_interventions') {
        step += 2
        ws.send(image(step, message.eye.blind_left ? [128, 128, 128] : undefined))
      }
      if (message.type === 'pause' && !message.paused) {
        step += 2
        ws.send(image(step))
      }
    })
  })
  const errors: string[] = []
  page.on('pageerror', (error) => errors.push(error.message))
  await page.goto('/')
  await page.getByRole('tab', { name: 'Songs', exact: true }).click()
  await page.getByRole('button', { name: 'Load Eye Test by Test Artist', exact: true }).click()
  await page.getByRole('button', { name: 'Play song', exact: true }).click()
  const compact = page.getByRole('button', { name: "Enlarge fly's-eye view" })
  const left = page.locator('.eye-toggles').getByRole('img', { name: /^Left eye.*3 live facets$/ })
  await expect(left).toBeVisible()
  await expect(left.locator('polygon').nth(2)).toHaveAttribute('fill', 'rgb(255, 255, 255)')
  await page.getByRole('button', { name: 'Blindfold left eye', exact: true }).click()
  await expect(left.locator('polygon').first()).toHaveAttribute('fill', 'rgb(128, 128, 128)')
  await expect(
    page
      .locator('.eye-toggles')
      .getByRole('img', { name: 'Right eye: 3 live facets', exact: true })
      .locator('polygon')
      .first(),
  ).toHaveAttribute('fill', 'rgb(255, 255, 255)')
  await compact.click()
  const inset = page.getByRole('region', { name: "Fly's-eye view", exact: true })
  await expect(inset).toBeVisible()
  await expect(
    inset
      .getByRole('img', { name: 'Left eye: 3 live facets', exact: true })
      .locator('polygon')
      .first(),
  ).toHaveAttribute('fill', 'rgb(128, 128, 128)')
  await expect(page.getByRole('button', { name: "Close fly's-eye view" })).toBeFocused()
  await page.keyboard.press('Escape')
  await expect(inset).toHaveCount(0)
  await expect(compact).toBeFocused()
  await page.getByRole('button', { name: 'Pause song', exact: true }).click()
  await expect(page.getByText('Paused eye image', { exact: true })).toBeVisible()
  await expect(left).toBeVisible()
  await page.getByRole('button', { name: 'Restart song', exact: true }).click()
  await expect(left).toHaveCount(0)
  await page.getByRole('button', { name: 'Play song', exact: true }).click()
  await expect(left).toBeVisible()
  socket!.send(image(step + 2, [1, 2], [1, 2, 3]))
  await expect(page.getByText('Eye image does not match layout', { exact: true })).toBeVisible()
  await expect(left).toHaveCount(0)
  socket!.close()
  await expect(page.getByText("Awaiting fly's-eye stream", { exact: true })).toBeVisible()
  await expect(left).toHaveCount(0)
  expect(errors).toEqual([])
})

test('eye layout failures can be retried, and the inset fits a compact workspace', async ({
  page,
}) => {
  let fail = true
  await page.route('**/api/eye_lattice?*', (route) =>
    fail ? route.fulfill({ status: 503, body: '{}' }) : route.fulfill({ json: lattice }),
  )
  if (test.info().project.name === 'desktop-chromium')
    await page.setViewportSize({ width: 1280, height: 720 })
  await page.goto('/')
  await expect(page.getByText('Eye layout unavailable', { exact: true })).toBeVisible()
  fail = false
  await page.getByRole('button', { name: 'Retry eye layout', exact: true }).click()
  await expect(page.getByText('Eye layout unavailable', { exact: true })).toHaveCount(0)
  await page.getByRole('button', { name: "Enlarge fly's-eye view" }).click()
  const inset = page.getByRole('region', { name: "Fly's-eye view", exact: true })
  await expect(inset).toBeVisible()
  await expect(inset.locator('polygon')).toHaveCount(6)
  const box = (await inset.boundingBox())!
  expect(box.x).toBeGreaterThanOrEqual(0)
  expect(box.x + box.width).toBeLessThanOrEqual(page.viewportSize()!.width)
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(
    page.viewportSize()!.width,
  )
  if (page.viewportSize()!.width >= 900)
    expect(await page.evaluate(() => document.documentElement.scrollHeight)).toBeLessThanOrEqual(
      720,
    )
  await page.getByRole('button', { name: "Close fly's-eye view" }).click()
  await expect(inset).toHaveCount(0)
})
