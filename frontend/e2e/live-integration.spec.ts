import { expect, test } from '@playwright/test'

// Opt-in run against FastAPI serving frontend/dist and a staged catalog.
// FLY_SABER_INTEGRATION_URL=http://127.0.0.1:8000 npx playwright test live-integration
test('FastAPI serves the app and simulates the same catalog song the browser plays', async ({
  page,
}, testInfo) => {
  test.skip(
    !process.env.FLY_SABER_INTEGRATION_URL,
    'Needs a running FastAPI server and staged allowlisted audio',
  )
  const errors: string[] = []
  const sent: Record<string, unknown>[] = []
  const received: Record<string, unknown>[] = []
  page.on('pageerror', (error) => errors.push(error.message))
  page.on('websocket', (ws) => {
    ws.on('framesent', (event) => {
      try {
        sent.push(JSON.parse(String(event.payload)))
      } catch {
        /* Binary streams. */
      }
    })
    ws.on('framereceived', (event) => {
      try {
        received.push(JSON.parse(String(event.payload)))
      } catch {
        /* Binary streams. */
      }
    })
  })
  await page.route('https://fonts.googleapis.com/**', (route) =>
    route.fulfill({ contentType: 'text/css', body: '' }),
  )
  await page.goto(process.env.FLY_SABER_INTEGRATION_URL!)
  await page.getByRole('tab', { name: 'Songs', exact: true }).click()
  await expect(page.getByLabel('Brain player', { exact: true })).toBeEnabled()
  await page
    .getByLabel('Brain player', { exact: true })
    .selectOption(process.env.FLY_SABER_INTEGRATION_BRAIN ?? 'expert')
  await page
    .getByRole('button', { name: /^Load live / })
    .first()
    .click()
  await expect(page.getByRole('button', { name: 'Play song', exact: true })).toBeEnabled()
  await page.getByRole('button', { name: 'Play song', exact: true }).click()
  await expect(page.locator('.performance-values')).not.toContainText('--', { timeout: 15000 })
  await expect.poll(() => received.some((m) => m.type === 'event'), { timeout: 15000 }).toBe(true)
  await expect.poll(() => sent.some((m) => m.type === 'clock')).toBe(true)
  const start = sent.find((m) => m.type === 'start')!
  const ready = received.find((m) => m.type === 'ready')!
  expect(ready.map_id).toBe(start.map_id)
  expect(start.brains).toEqual([process.env.FLY_SABER_INTEGRATION_BRAIN ?? 'expert'])
  await page.getByRole('button', { name: 'Pause song', exact: true }).click()
  await expect(page.getByText('Live · paused', { exact: true })).toBeVisible()
  const position = page.getByRole('slider', { name: 'Song position', exact: true })
  const paused = await position.inputValue()
  await page.waitForTimeout(150)
  await expect(position).toHaveValue(paused)
  await page.screenshot({ path: testInfo.outputPath('fastapi-live-session.png'), scale: 'css' })
  expect(errors).toEqual([])
})
