import { expect, test } from '@playwright/test'

test.beforeEach(async ({ page }) => {
  await page.route('**/api/games', (route) =>
    route.fulfill({ json: [{ id: 'saber', label: 'Fly Saber' }] }),
  )
  await page.route('https://fonts.googleapis.com/**', (route) =>
    route.fulfill({ contentType: 'text/css', body: '' }),
  )
})

test('animation preview reaches different cells, freezes, and resumes without runtime errors', async ({
  page,
}, testInfo) => {
  // Software-rendered WebGL captures take several seconds on shared CI runners.
  test.setTimeout(60_000)
  const errors: string[] = []
  page.on('pageerror', (error) => errors.push(error.message))
  await page.goto('/?flyMotion=preview')
  await expect(page.getByRole('button', { name: 'Reset game camera', exact: true })).toBeEnabled()
  const preview = page.getByRole('region', { name: 'Fly animation preview', exact: true })
  await expect(preview).toContainText('Sample movements · no backend policy')
  await page.getByLabel('Preview movement', { exact: true }).selectOption('1')
  await page.getByLabel('Preview target cell', { exact: true }).selectOption('0')
  await page.waitForTimeout(250)
  await page.getByRole('button', { name: 'Pause preview', exact: true }).click()
  const canvas = page.locator('.game-stage canvas')
  // Compare the same CSS pixels on desktop and high-DPI mobile devices.
  const capture = () => canvas.screenshot({ scale: 'css' })
  await page.waitForTimeout(100)
  let low = await capture()
  await expect
    .poll(
      async () => {
        const next = await capture()
        const stable = next.equals(low)
        low = next
        return stable
      },
      { timeout: 20_000 },
    )
    .toBe(true)
  await page.getByRole('button', { name: 'Resume preview', exact: true }).click()
  await page.getByLabel('Preview target cell', { exact: true }).selectOption('11')
  await page.waitForTimeout(250)
  await page.getByRole('button', { name: 'Pause preview', exact: true }).click()
  await page.waitForTimeout(100)
  expect((await capture()).equals(low)).toBe(false)
  await page.screenshot({ path: testInfo.outputPath('fly-animation-preview.png'), scale: 'css' })
  await page.getByRole('button', { name: 'Resume preview', exact: true }).click()
  for (const move of ['2', '3', '4', '5', '6']) {
    await page.getByLabel('Preview movement', { exact: true }).selectOption(move)
    await page.waitForTimeout(170)
  }
  expect(errors).toEqual([])
  await expect(page.getByRole('button', { name: 'Play song', exact: true })).toBeDisabled()
})

test('normal workspace keeps sample animation controls out of the session', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('button', { name: 'Reset game camera', exact: true })).toBeEnabled()
  await expect(
    page.getByRole('region', { name: 'Fly animation preview', exact: true }),
  ).toHaveCount(0)
  await expect(page.getByText('Awaiting simulation', { exact: true })).toBeVisible()
})
