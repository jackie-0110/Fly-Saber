import { expect, test } from '@playwright/test'
import { readFileSync } from 'node:fs'

const neurons = JSON.parse(
  readFileSync(new URL('../public/data/brain/neurons.json', import.meta.url), 'utf8'),
) as { n_neurons: number; type: number[]; body_id: number[] }
const groups = JSON.parse(
  readFileSync(new URL('../public/data/brain/groups.json', import.meta.url), 'utf8'),
) as Record<string, number[]>

test.beforeEach(async ({ page }) => {
  await page.route('**/api/games', (route) =>
    route.fulfill({ json: [{ id: 'saber', label: 'Fly Saber' }] }),
  )
  // Keep optional web fonts independent of network access and provider availability.
  await page.route('https://fonts.googleapis.com/**', (route) =>
    route.fulfill({ contentType: 'text/css', body: '' }),
  )
})

test('production workspace loads without runtime errors or failed assets', async ({ page }) => {
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
})

test('playback controls stay disabled until a session is loaded', async ({ page }) => {
  await page.goto('/')
  for (const name of ['Play session', 'Pause session', 'Restart session']) {
    await expect(page.getByRole('button', { name, exact: true })).toBeDisabled()
  }
  await expect(page.getByText('No session loaded', { exact: true })).toBeVisible()
})

test('workspace fits the viewport and inspector tabs support the keyboard', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('tab', { name: 'Neurons', exact: true }).focus()
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
