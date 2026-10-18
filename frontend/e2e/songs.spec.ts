import { expect, test } from '@playwright/test'
import type { Page } from '@playwright/test'
import { mapZip, result } from '../tests/song-fixtures.ts'

const archive = Buffer.from(mapZip())
const cover = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aX1cAAAAASUVORK5CYII=',
  'base64',
)

test.beforeEach(async ({ page }) => {
  // These specs exercise local chart previews. A running developer backend must not
  // replace the three-note archive with the real BeatSaver map for the fixture key.
  await page.route('**/api/community/**', (route) =>
    route.fulfill({ status: 422, json: { detail: 'Fixture preview only' } }),
  )
  await page.route('**/api/eye_lattice?*', (route) =>
    route.fulfill({ json: { left: [[0, 0]], right: [[0, 0]] } }),
  )
  await page.route('**/api/maps', (route) => route.fulfill({ json: [] }))
  await page.route('**/api/brains', (route) =>
    route.fulfill({
      json: [{ id: 'expert', label: 'Scripted expert', description: 'Fixture player' }],
    }),
  )
  await page.route('**/api/games', (route) =>
    route.fulfill({ json: [{ id: 'saber', label: 'Fly Saber' }] }),
  )
  await page.route('https://fonts.googleapis.com/**', (route) =>
    route.fulfill({ contentType: 'text/css', body: '' }),
  )
  await page.route('https://api.beatsaver.com/**', (route) => {
    const url = new URL(route.request().url())
    return route.fulfill({
      json: url.pathname.startsWith('/maps/id/')
        ? result
        : { docs: url.searchParams.get('q') === 'no matching song' ? [] : [result] },
    })
  })
  await page.route('https://r2cdn.beatsaver.com/test.zip', (route) =>
    route.fulfill({ body: archive, contentType: 'application/zip' }),
  )
  await page.route('https://cdn.beatsaver.com/test.jpg', (route) =>
    route.fulfill({ body: cover, contentType: 'image/png' }),
  )
})

async function loadMap(page: Page) {
  await page.getByRole('tab', { name: 'Songs', exact: true }).click()
  await page.getByRole('button', { name: 'Load Orbit Test by Test Artist', exact: true }).click()
  await expect(
    page.getByRole('heading', { name: 'Orbit Test', exact: true, level: 1 }),
  ).toBeVisible()
  await expect(page.getByRole('button', { name: 'Play song', exact: true })).toBeEnabled()
}

async function seek(page: Page, seconds: number) {
  const slider = page.getByRole('slider', { name: 'Song position', exact: true })
  // Exercise the actual range control and its React handler at an exact timestamp.
  await slider.evaluate((element, value) => {
    const input = element as HTMLInputElement
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(
      input,
      String(value),
    )
    input.dispatchEvent(new Event('input', { bubbles: true }))
    input.dispatchEvent(new Event('change', { bubbles: true }))
  }, seconds)
  await expect(slider).toHaveValue(String(seconds))
}

test('song search and download load a community map without a fly session or allowlist', async ({
  page,
}) => {
  const errors: string[] = []
  page.on('pageerror', (error) => errors.push(error.message))
  await page.goto('/')
  await page.getByRole('tab', { name: 'Songs', exact: true }).click()
  await expect(page.getByRole('tab', { name: 'Songs', exact: true })).toHaveAttribute(
    'aria-selected',
    'true',
  )
  await page
    .getByRole('textbox', { name: 'Search songs', exact: true })
    .fill('https://beatsaver.com/maps/abcd')
  await page.getByRole('button', { name: 'Search', exact: true }).click()
  await page.getByRole('button', { name: 'Load Orbit Test by Test Artist', exact: true }).click()
  const header = page.getByRole('region', { name: 'Selected song information', exact: true })
  await expect(header.getByRole('heading', { name: 'Orbit Test', exact: true })).toBeVisible()
  await expect(header).toContainText('Test Artist')
  await expect(header).toContainText('Mapped by Test Mapper')
  await expect(header).toContainText('120 BPM')
  await expect(header).toContainText('3 notes')
  await expect(header.getByRole('link', { name: 'Map source' })).toHaveAttribute(
    'href',
    'https://beatsaver.com/maps/abcd',
  )
  await expect(page.getByText('CHART PREVIEW', { exact: true })).toHaveCount(0)
  await expect(page.getByLabel('Import map ZIP', { exact: true })).toHaveCount(0)
  await expect(
    page.getByRole('region', { name: 'Loaded song difficulties', exact: true }),
  ).toHaveCount(0)
  await expect(header.getByRole('button', { name: 'Change song', exact: true })).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Play song', exact: true })).toBeEnabled()
  await page.getByRole('tab', { name: 'Session', exact: true }).click()
  await expect(page.getByText('No live session', { exact: true })).toBeVisible()
  expect(errors).toEqual([])
})

test('loaded maps support playback, pause, backward seeks, rate changes, and restart', async ({
  page,
}) => {
  await page.goto('/')
  await loadMap(page)
  await page.getByRole('button', { name: 'Play song', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Pause song', exact: true })).toBeVisible()
  const slider = page.getByRole('slider', { name: 'Song position', exact: true })
  await expect.poll(async () => Number(await slider.inputValue())).toBeGreaterThan(0.2)
  await page.getByRole('button', { name: 'Pause song', exact: true }).click()
  const paused = await slider.inputValue()
  await page.waitForTimeout(200)
  await expect(slider).toHaveValue(paused)
  await seek(page, 10)
  await seek(page, 2)
  await page.getByRole('combobox', { name: 'Playback speed' }).selectOption('0.5')
  await expect(slider).toHaveValue('2')
  await page.getByRole('button', { name: 'Play song', exact: true }).click()
  await expect.poll(async () => Number(await slider.inputValue())).toBeGreaterThan(2.1)
  await page.getByRole('button', { name: 'Restart song', exact: true }).click()
  await expect(slider).toHaveValue('0')
  await expect(page.getByRole('button', { name: 'Play song', exact: true })).toBeVisible()
})

test('dragging the timeline previews the new position and resumes only if it was playing', async ({
  page,
}) => {
  await page.goto('/')
  await loadMap(page)
  const slider = page.getByRole('slider', { name: 'Song position', exact: true })
  await slider.scrollIntoViewIfNeeded()
  const box = (await slider.boundingBox())!
  await page.mouse.move(box.x + box.width * 0.2, box.y + box.height / 2)
  await page.mouse.down()
  await page.mouse.move(box.x + box.width * 0.65, box.y + box.height / 2, { steps: 5 })
  await page.mouse.up()
  await expect(page.getByRole('button', { name: 'Play song', exact: true })).toBeVisible()
  expect(Number(await slider.inputValue())).toBeGreaterThan(10)
  await page.getByRole('button', { name: 'Play song', exact: true }).click()
  await page.mouse.move(box.x + box.width * 0.3, box.y + box.height / 2)
  await page.mouse.down()
  await page.mouse.move(box.x + box.width * 0.4, box.y + box.height / 2, { steps: 5 })
  await expect(page.getByRole('button', { name: 'Play song', exact: true })).toBeVisible()
  await page.mouse.up()
  await expect(page.getByRole('button', { name: 'Pause song', exact: true })).toBeVisible()
})

test('difficulty changes keep the song position and the camera can orbit, zoom, and reset', async ({
  page,
}) => {
  await page.goto('/')
  await loadMap(page)
  await seek(page, 1.5)
  await page
    .locator('.song-card.is-loaded')
    .getByRole('button', { name: 'Normal', exact: true })
    .click()
  await expect(page.getByRole('slider', { name: 'Song position', exact: true })).toHaveValue('1.5')
  await expect(page.locator('.current-difficulty')).toHaveText('Normal')
  await page.getByRole('button', { name: 'Reset game camera', exact: true }).click()
  const scene = page.getByRole('region', { name: 'Interactive 3D game view', exact: true })
  await expect(page.getByRole('button', { name: 'Reset game camera', exact: true })).toBeEnabled()
  await page.waitForTimeout(200)
  const before = await scene.screenshot()
  await scene.focus()
  await scene.press('ArrowRight')
  await scene.press('+')
  await page.waitForTimeout(250)
  const after = await scene.screenshot()
  expect(after.equals(before)).toBe(false)
  await page.getByRole('button', { name: 'Track overview', exact: true }).click()
  await page.getByRole('button', { name: 'Player view', exact: true }).click()
  await page.getByRole('button', { name: 'Hide fly model', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Show fly model', exact: true })).toBeVisible()
  const overflow = await page.evaluate(() => ({
    width: document.documentElement.scrollWidth,
    height: document.documentElement.scrollHeight,
    viewportWidth: innerWidth,
    viewportHeight: innerHeight,
  }))
  expect(overflow.width).toBeLessThanOrEqual(overflow.viewportWidth)
  if (overflow.viewportWidth >= 900)
    expect(overflow.height).toBeLessThanOrEqual(overflow.viewportHeight)
})

test('failed downloads preserve the loaded song and a new valid download recovers', async ({
  page,
}) => {
  await page.goto('/')
  await loadMap(page)
  await seek(page, 5)
  const next = { ...result, id: 'abc1', metadata: { ...result.metadata, songName: 'Next Track' } }
  await page.route('https://api.beatsaver.com/**', (route) =>
    route.fulfill({ json: { docs: [next] } }),
  )
  await page.route('https://r2cdn.beatsaver.com/test.zip', (route) =>
    route.fulfill({ body: Buffer.from('broken zip'), contentType: 'application/zip' }),
  )
  await page.getByRole('button', { name: 'Search', exact: true }).click()
  await page.getByRole('button', { name: 'Load Next Track by Test Artist', exact: true }).click()
  await expect(page.getByText('Could not open that ZIP.', { exact: false })).toBeVisible()
  await expect(
    page.getByRole('heading', { name: 'Orbit Test', level: 1, exact: true }),
  ).toBeVisible()
  await expect(page.getByRole('slider', { name: 'Song position', exact: true })).toHaveValue('5')
  await expect(page.getByRole('button', { name: 'Play song', exact: true })).toBeEnabled()
  await page.route('https://r2cdn.beatsaver.com/test.zip', (route) =>
    route.fulfill({ body: archive, contentType: 'application/zip' }),
  )
  await page.getByRole('button', { name: 'Load Next Track by Test Artist', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Dismiss song error', exact: true })).toHaveCount(0)
  await expect(page.locator('.song-card.is-loaded')).toContainText('Next Track')
})

test('search failure, empty results, and retry have usable recovery states', async ({ page }) => {
  let fail = true
  await page.route('https://api.beatsaver.com/**', (route) =>
    fail ? route.fulfill({ status: 503, body: '{}' }) : route.fulfill({ json: { docs: [] } }),
  )
  await page.goto('/')
  await page.getByRole('tab', { name: 'Songs', exact: true }).click()
  await expect(page.getByText('The song service returned 503.', { exact: false })).toBeVisible()
  fail = false
  await page.getByRole('button', { name: 'Retry search', exact: true }).click()
  await expect(page.getByText('No maps found.', { exact: false })).toBeVisible()
  await page.getByRole('combobox', { name: 'Song sort order', exact: true }).selectOption('Latest')
  await page.getByRole('textbox', { name: 'Search songs', exact: true }).fill('Test Artist')
  const request = page.waitForRequest(
    (request) =>
      request.url().includes('/search/text/') &&
      new URL(request.url()).searchParams.get('q') === 'Test Artist',
  )
  await page.getByRole('button', { name: 'Search', exact: true }).click()
  expect(new URL((await request).url()).searchParams.get('sortOrder')).toBe('Latest')
})

test('cancelling a pending song download retains the previous track and position', async ({
  page,
}) => {
  await page.goto('/')
  await loadMap(page)
  await seek(page, 5)
  const stalled = {
    ...result,
    id: 'abc1',
    metadata: { ...result.metadata, songName: 'Stalled Track' },
    versions: [{ ...result.versions[0], downloadURL: 'https://r2cdn.beatsaver.com/stalled.zip' }],
  }
  await page.route('https://api.beatsaver.com/**', (route) =>
    route.fulfill({ json: { docs: [stalled] } }),
  )
  await page.route('https://r2cdn.beatsaver.com/stalled.zip', async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 800))
    await route.fulfill({ body: archive, contentType: 'application/zip' })
  })
  await page.getByRole('button', { name: 'Search', exact: true }).click()
  await page.getByRole('button', { name: 'Load Stalled Track by Test Artist', exact: true }).click()
  await page.getByRole('button', { name: 'Cancel song loading', exact: true }).click()
  await expect(
    page.getByRole('heading', { name: 'Orbit Test', level: 1, exact: true }),
  ).toBeVisible()
  await expect(page.getByRole('slider', { name: 'Song position', exact: true })).toHaveValue('5')
  await expect(page.getByRole('button', { name: 'Play song', exact: true })).toBeEnabled()
  await page.waitForTimeout(900)
  await expect(
    page.getByRole('heading', { name: 'Orbit Test', level: 1, exact: true }),
  ).toBeVisible()
})

test('WebGL context loss is recoverable without losing the selected song', async ({ page }) => {
  await page.goto('/')
  await loadMap(page)
  await expect(page.getByRole('button', { name: 'Reset game camera', exact: true })).toBeEnabled()
  await page.locator('.game-stage canvas').evaluate((canvas) => {
    const gl = (canvas as HTMLCanvasElement).getContext('webgl2')!
    gl.getExtension('WEBGL_lose_context')!.loseContext()
  })
  await expect(
    page.getByRole('heading', { name: 'The 3D track is unavailable.', exact: true }),
  ).toBeVisible()
  await page.getByRole('button', { name: 'Retry 3D view', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Reset game camera', exact: true })).toBeEnabled()
  await expect(
    page.getByRole('heading', { name: 'Orbit Test', level: 1, exact: true }),
  ).toBeVisible()
  await expect(page.getByRole('button', { name: 'Play song', exact: true })).toBeEnabled()
})

test('audio and song selection still work when WebGL is unavailable', async ({ page }) => {
  await page.addInitScript(() => {
    const getContext = HTMLCanvasElement.prototype.getContext
    HTMLCanvasElement.prototype.getContext = function (
      this: HTMLCanvasElement,
      ...args: Parameters<typeof getContext>
    ) {
      if (String(args[0]).startsWith('webgl')) return null
      return getContext.apply(this, args)
    } as typeof getContext
  })
  await page.goto('/')
  await expect(
    page.getByRole('heading', { name: 'The 3D track is unavailable.', exact: true }),
  ).toBeVisible()
  await loadMap(page)
  await page.getByRole('button', { name: 'Play song', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Pause song', exact: true })).toBeVisible()
  await expect(page.getByRole('slider', { name: 'Song position', exact: true })).toBeEnabled()
})
