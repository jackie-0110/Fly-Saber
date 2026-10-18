import { expect, test } from '@playwright/test'
import { readFileSync } from 'node:fs'
import { CAMERA_SETTLE_MS, MOVING_EDGE_LIMIT, SAMPLE_EDGE_LIMIT } from '../src/brain/rendering.ts'

declare global {
  interface Window {
    brainRendering: {
      lines: { count: number; indexed: boolean; matrix?: number[] }[]
      uploads: number
    }
  }
}

const neurons = JSON.parse(
  readFileSync(new URL('../public/data/brain/neurons.json', import.meta.url), 'utf8'),
) as { n_neurons: number; type: number[]; body_id: number[] }
const groups = JSON.parse(
  readFileSync(new URL('../public/data/brain/groups.json', import.meta.url), 'utf8'),
) as Record<string, number[]>
const edges = JSON.parse(
  readFileSync(new URL('../public/data/brain/edges.json', import.meta.url), 'utf8'),
) as {
  n_edges: number
  n_synapses: number
  pre: number[]
  post: number[]
  weight: number[]
  sign: number[]
}

// CI browsers rasterize WebGL in software, where drawing all 340k edges stalls the page.
// Tests that need "All" detail use a prefix of the real table: it still exceeds the
// movement preview, so preview, restore and upload behavior are exercised the same way.
const FIXTURE_EDGES = MOVING_EDGE_LIMIT + 5_007
const fixture = (() => {
  const slice = <T>(values: T[]) => values.slice(0, FIXTURE_EDGES)
  const weight = slice(edges.weight)
  return {
    ...edges,
    n_edges: FIXTURE_EDGES,
    n_synapses: weight.reduce((sum, value) => sum + value, 0),
    pre: slice(edges.pre),
    post: slice(edges.post),
    weight,
    sign: slice(edges.sign),
  }
})()
const format = (value: number) => value.toLocaleString('en-US')
const shown = (value: number, total = edges.n_edges) =>
  `${format(value)} of ${format(total)} connections shown`

test.beforeEach(async ({ page }) => {
  await page.route('**/api/maps', (route) => route.fulfill({ json: [] }))
  await page.route('**/api/brains', (route) => route.fulfill({ json: [] }))
  await page.route('**/api/games', (route) =>
    route.fulfill({ json: [{ id: 'saber', label: 'Fly Saber' }] }),
  )
  // Keep optional web fonts independent of network access and provider availability.
  await page.route('https://fonts.googleapis.com/**', (route) =>
    route.fulfill({ contentType: 'text/css', body: '' }),
  )
  await page.route('https://api.beatsaver.com/**', (route) => route.fulfill({ json: { docs: [] } }))
})

test('production workspace loads without runtime errors or failed assets', async ({
  page,
}, testInfo) => {
  const errors: string[] = []
  page.on('pageerror', (error) => errors.push(error.message))
  page.on('requestfailed', (request) => errors.push(request.url()))
  page.on('response', (response) => {
    if (response.status() >= 400) errors.push(`${response.status()} ${response.url()}`)
  })

  await page.goto('/')
  await expect(page.getByRole('link', { name: 'Fly Saber home' })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Fly brain neurons' })).toBeVisible()
  await expect(page.getByRole('main')).toBeVisible()
  await expect(page.getByRole('link', { name: 'Fly Saber home' })).toHaveAttribute('href', '/')
  await expect(page.getByRole('button', { name: 'Reset', exact: true })).toBeEnabled()
  await expect(page.locator('.neuron-count')).toContainText(
    `${neurons.n_neurons.toLocaleString('en-US')} of ${neurons.n_neurons.toLocaleString('en-US')}`,
  )
  await page.waitForLoadState('networkidle')
  expect(errors).toEqual([])
  await page.screenshot({ path: testInfo.outputPath('brain-connections.png'), fullPage: true })
})

test('playback controls stay disabled until a song is loaded', async ({ page }) => {
  await page.goto('/')
  for (const name of ['Play song', 'Restart song']) {
    await expect(page.getByRole('button', { name, exact: true })).toBeDisabled()
  }
  await expect(page.getByRole('slider', { name: 'Song position', exact: true })).toBeDisabled()
})

test('workspace fits the viewport and inspector tabs support the keyboard', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('tab', { name: 'Neurons', exact: true }).focus()
  await page.keyboard.press('ArrowRight')
  await expect(page.getByRole('tabpanel', { name: 'Songs', exact: true })).toBeVisible()
  await page.keyboard.press('ArrowRight')
  await expect(page.getByRole('tabpanel', { name: 'Session', exact: true })).toBeVisible()
  await page.keyboard.press('ArrowRight')
  await expect(page.getByRole('tab', { name: 'Details', exact: true })).toHaveAttribute(
    'aria-selected',
    'true',
  )
  await expect(page.getByRole('tabpanel', { name: 'Details', exact: true })).toBeVisible()
  await page.keyboard.press('ArrowRight')
  await expect(page.getByRole('tabpanel', { name: 'Next steps', exact: true })).toBeVisible()
  await page.keyboard.press('Home')
  await expect(page.getByRole('tabpanel', { name: 'Neurons', exact: true })).toBeVisible()

  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth > window.innerWidth,
  )
  expect(overflow).toBe(false)
  if (page.viewportSize()!.width >= 900) {
    expect(await page.evaluate(() => document.documentElement.scrollHeight)).toBeLessThanOrEqual(
      page.viewportSize()!.height,
    )
  }
})

test('group filters restrict cell types and preserve original neuron IDs', async ({ page }) => {
  await page.goto('/')
  const group = page.getByLabel('Neuron group', { exact: true })
  const type = page.getByLabel('Cell type', { exact: true })
  await expect(group).toBeEnabled()
  await group.selectOption('T4')
  const t4 = [...new Set(groups.T4.map((index) => neurons.type[index]))].sort((a, b) => a - b)
  expect(
    await type
      .locator('option')
      .evaluateAll((options) => options.map((option) => (option as HTMLOptionElement).value)),
  ).toEqual(['', ...t4.map(String)])
  await type.selectOption(String(t4[0]))
  await page.getByRole('button', { name: 'Select next neuron' }).click()
  await expect(page.locator('.neuron-metadata')).toContainText(
    String(neurons.body_id[groups.T4[0]]),
  )
  await group.selectOption('motion_detectors')
  await expect(type).toHaveValue(String(t4[0]))
  await group.selectOption('T5')
  await expect(type).toHaveValue('')
  const t5 = [...new Set(groups.T5.map((index) => neurons.type[index]))].sort((a, b) => a - b)
  expect(
    await type
      .locator('option')
      .evaluateAll((options) => options.map((option) => (option as HTMLOptionElement).value)),
  ).toEqual(['', ...t5.map(String)])
})

test('connections default to a sample, can be hidden, filtered, and traced for the selected neuron', async ({
  page,
}, testInfo) => {
  await page.goto('/')
  const summary = page.locator('.brain-connection-summary')
  const detail = page.getByLabel('Connection detail', { exact: true })
  await expect(summary).toContainText(shown(SAMPLE_EDGE_LIMIT))
  await expect(summary).toContainText('Evenly spaced sample')
  await expect(page.getByRole('button', { name: 'Reset', exact: true })).toBeEnabled()
  await detail.selectOption('off')
  await expect(summary).toContainText('Connections hidden')
  await detail.selectOption('sample')
  await page.getByRole('button', { name: 'Select next neuron' }).click()
  // Selection is never sampled.
  const incident = edges.pre.filter((pre, edge) => pre === 0 || edges.post[edge] === 0).length
  await expect(summary).toContainText(shown(incident))
  await expect(summary).toContainText('Violet: incoming')
  await expect(summary).toContainText('Gold: outgoing')
  const incoming = edges.post.filter((post) => post === 0).length
  await expect(page.locator('.neuron-metadata')).toContainText(`${format(incoming)} connections /`)
  await page.screenshot({ path: testInfo.outputPath('selected-connections.png'), fullPage: true })

  await page.getByLabel('Neuron group', { exact: true }).selectOption('T4')
  const visible = new Set(groups.T4)
  const filtered = edges.pre.filter(
    (pre, edge) => visible.has(pre) && visible.has(edges.post[edge]),
  ).length
  await expect(summary).toContainText(shown(Math.min(filtered, SAMPLE_EDGE_LIMIT)))
  await expect(summary).not.toContainText('Blue: incoming')
  await page.getByRole('button', { name: 'Clear filters', exact: true }).click()
  await expect(summary).toContainText(shown(SAMPLE_EDGE_LIMIT))
})

test('connections load after the neurons and a failed load is retried on its own', async ({
  page,
}) => {
  await page.route('**/data/brain/edges.json', (route) => route.fulfill({ status: 404, body: '' }))
  await page.goto('/')
  await expect(page.locator('.neuron-count')).toContainText(
    `${format(neurons.n_neurons)} of ${format(neurons.n_neurons)} neurons shown`,
  )
  await expect(page.getByRole('button', { name: 'Reset', exact: true })).toBeEnabled()
  await expect(page.getByText('Could not load edges.json.', { exact: true })).toBeVisible()
  await page.unroute('**/data/brain/edges.json')
  await page.getByRole('button', { name: 'Retry connections', exact: true }).click()
  await expect(page.locator('.brain-connection-summary')).toContainText(shown(SAMPLE_EDGE_LIMIT))
})

test('camera movement uses a resident preview buffer and restores every connection', async ({
  page,
}, testInfo) => {
  // Many phases (drag, settle, resize, selected drag) on a full-size software canvas.
  test.setTimeout(90_000)
  const errors: string[] = []
  page.on('pageerror', (error) => errors.push(error.message))
  // Inspect actual WebGL submissions rather than relying on a fast test machine.
  await page.addInitScript(() => {
    window.brainRendering = { lines: [], uploads: 0 }
    for (const Type of [WebGLRenderingContext, WebGL2RenderingContext]) {
      const draw = Type.prototype.drawElements
      Type.prototype.drawElements = function (...args) {
        if (
          this.canvas instanceof HTMLCanvasElement &&
          this.canvas.getAttribute('aria-label') === '3D neurons and directed connections' &&
          args[0] === this.LINES
        ) {
          const program = this.getParameter(this.CURRENT_PROGRAM) as WebGLProgram
          const location = this.getUniformLocation(program, 'modelViewMatrix')
          window.brainRendering.lines.push({
            count: args[1],
            indexed: true,
            matrix: location
              ? Array.from(this.getUniform(program, location) as Float32Array)
              : undefined,
          })
        }
        return draw.apply(this, args)
      }
      const upload = Type.prototype.bufferData
      Type.prototype.bufferData = function (...args: unknown[]) {
        if (
          this.canvas instanceof HTMLCanvasElement &&
          this.canvas.getAttribute('aria-label') === '3D neurons and directed connections'
        ) {
          window.brainRendering.uploads++
        }
        return Reflect.apply(upload, this, args)
      }
    }
  })
  await page.route('**/data/brain/edges.json', (route) => route.fulfill({ json: fixture }))
  await page.goto('/')
  const summary = page.locator('.brain-connection-summary')
  await page.getByLabel('Connection detail', { exact: true }).selectOption('all')
  const allShown = shown(fixture.n_edges, fixture.n_edges)
  await expect(summary).toContainText(allShown)
  await expect(page.getByRole('button', { name: 'Reset', exact: true })).toBeEnabled()
  const stage = page.locator('.brain-stage-3d')
  await stage.scrollIntoViewIfNeeded()
  const resolution = () =>
    stage.locator('canvas').evaluate((element) => {
      const canvas = element as HTMLCanvasElement
      return canvas.width / canvas.getBoundingClientRect().width
    })
  const restingResolution = await resolution()
  const box = (await stage.boundingBox())!
  await page.evaluate(() => {
    window.brainRendering = { lines: [], uploads: 0 }
  })
  await page.mouse.move(box.x + box.width * 0.4, box.y + box.height * 0.4)
  await page.mouse.down()
  await page.mouse.move(box.x + box.width * 0.6, box.y + box.height * 0.5, { steps: 8 })
  await expect(summary).toContainText('Movement preview')
  const movingBox = (await stage.boundingBox())!
  expect(movingBox.width).toBeCloseTo(box.width, 1)
  expect(movingBox.height).toBeCloseTo(box.height, 1)
  await expect(summary).toContainText(`${MOVING_EDGE_LIMIT.toLocaleString('en-US')} of`)
  await expect.poll(resolution).toBeLessThanOrEqual(1.01)
  await expect
    .poll(() => page.evaluate(() => window.brainRendering.lines.at(-1)?.count))
    .toBe(MOVING_EDGE_LIMIT * 2)
  await page.screenshot({ path: testInfo.outputPath('moving-connections.png'), fullPage: true })
  await page.mouse.up()
  await expect(summary).toContainText(allShown)
  await expect
    .poll(() => page.evaluate(() => window.brainRendering.lines.at(-1)?.count))
    .toBe(fixture.n_edges * 2)
  await expect.poll(resolution).toBeCloseTo(restingResolution, 1)
  const settledBox = (await stage.boundingBox())!
  expect(settledBox.width).toBeCloseTo(box.width, 1)
  expect(settledBox.height).toBeCloseTo(box.height, 1)
  const rendering = await page.evaluate(() => window.brainRendering)
  expect(rendering.lines.every((draw) => draw.indexed)).toBe(true)
  expect(rendering.uploads).toBe(0)
  // Watch beyond several settling intervals: late damping updates must not
  // flip the renderer back to the preview after full detail has returned.
  const stable = await page.evaluate(
    async ({ duration, fullCount }) => {
      const start = window.brainRendering.lines.length
      await new Promise((resolve) => setTimeout(resolve, duration))
      return {
        full: window.brainRendering.lines.slice(start).every((draw) => draw.count === fullCount),
        text: document.querySelector('.brain-connection-summary')?.textContent,
      }
    },
    { duration: CAMERA_SETTLE_MS * 4, fullCount: fixture.n_edges * 2 },
  )
  expect(stable.full).toBe(true)
  expect(stable.text).toContain(allShown)
  await page.screenshot({ path: testInfo.outputPath('settled-connections.png'), fullPage: true })

  // Layout changes must update the projection without refitting/resetting the
  // camera. Inspect the matrix actually submitted for the connection geometry.
  const cameraBeforeResize = await page.evaluate(() => window.brainRendering.lines.at(-1)?.matrix)
  expect(cameraBeforeResize).toHaveLength(16)
  const drawsBeforeResize = await page.evaluate(() => window.brainRendering.lines.length)
  const viewport = page.viewportSize()!
  await page.setViewportSize({ width: viewport.width - 32, height: viewport.height })
  await expect
    .poll(() => page.evaluate(() => window.brainRendering.lines.length))
    .toBeGreaterThan(drawsBeforeResize)
  await expect
    .poll(() =>
      page.evaluate((before) => {
        const after = window.brainRendering.lines.at(-1)?.matrix ?? []
        return Math.max(...before.map((value, i) => Math.abs(value - (after[i] ?? Infinity))))
      }, cameraBeforeResize!),
    )
    // A refit moves the camera by ~0.1 or more; leftover OrbitControls damping from the
    // drag may still apply ~1e-4 on the resize render.
    .toBeLessThan(0.001)

  // Selection retains all incident connections even while the camera is moving.
  await page.getByRole('button', { name: 'Select next neuron' }).click()
  const selectedCount = fixture.pre.filter(
    (pre, edge) => pre === 0 || fixture.post[edge] === 0,
  ).length
  await stage.scrollIntoViewIfNeeded()
  const selectedBox = (await stage.boundingBox())!
  await page.evaluate(() => {
    window.brainRendering.lines = []
  })
  await page.mouse.move(
    selectedBox.x + selectedBox.width * 0.4,
    selectedBox.y + selectedBox.height * 0.4,
  )
  await page.mouse.down()
  await page.mouse.move(
    selectedBox.x + selectedBox.width * 0.6,
    selectedBox.y + selectedBox.height * 0.5,
    { steps: 4 },
  )
  await expect(summary).toContainText(`${selectedCount.toLocaleString('en-US')} of`)
  await expect(summary).not.toContainText('Movement preview')
  await expect
    .poll(() =>
      page.evaluate(() =>
        window.brainRendering.lines.slice(-2).reduce((sum, draw) => sum + draw.count, 0),
      ),
    )
    .toBe(selectedCount * 2)
  await page.mouse.up()
  expect(errors).toEqual([])
})

test('backend badge detects a disconnect and recovery without implying a live session', async ({
  page,
}) => {
  await page.clock.install()
  let connected = true
  await page.route('**/api/games', (route) =>
    connected
      ? route.fulfill({ json: [{ id: 'saber', label: 'Fly Saber' }] })
      : route.fulfill({ status: 503, json: { detail: 'Unavailable' } }),
  )
  await page.goto('/')
  await expect(page.locator('.backend-status')).toHaveText('Backend connected')
  connected = false
  await page.clock.fastForward(5100)
  await expect(page.locator('.backend-status')).toHaveText('Backend disconnected')
  connected = true
  await page.clock.fastForward(5100)
  await expect(page.locator('.backend-status')).toHaveText('Backend connected')
  await page.getByRole('tab', { name: 'Session', exact: true }).click()
  await expect(page.getByText('No live session', { exact: true })).toBeVisible()
})

test('brain controls toggle before a song and reset together', async ({ page }) => {
  await page.goto('/')
  const panel = page.getByRole('region', { name: 'Brain controls' })
  const lesion = panel.getByRole('button', { name: 'Lesion looming detectors' })
  // Blindfolds live on the eyes in Visual input, outside the Brain stage.
  const leftEye = page.getByRole('button', { name: 'Blindfold left eye' })
  await expect(panel.getByRole('slider', { name: /Arousal/ })).toBeEnabled()
  await lesion.click()
  await leftEye.click()
  await expect(lesion).toHaveAttribute('aria-pressed', 'true')
  await expect(leftEye).toHaveAttribute('aria-pressed', 'true')
  await panel.getByRole('button', { name: 'Reset controls' }).click()
  await expect(lesion).toHaveAttribute('aria-pressed', 'false')
  await expect(leftEye).toHaveAttribute('aria-pressed', 'false')
})
