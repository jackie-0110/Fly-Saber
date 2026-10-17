import { expect, test } from '@playwright/test'

test.beforeEach(async ({ page }) => {
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
  await expect(page.getByRole('heading', { name: 'Fly Saber', exact: true })).toBeVisible()
  await expect(page.getByRole('main')).toBeVisible()
  await expect(page.getByRole('link', { name: 'FlyArcade home' })).toHaveAttribute('href', '/')
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
  await page.getByRole('tab', { name: 'Activity', exact: true }).focus()
  await page.keyboard.press('ArrowRight')
  await expect(page.getByRole('tab', { name: 'Details', exact: true })).toHaveAttribute(
    'aria-selected',
    'true',
  )
  await expect(page.getByRole('tabpanel', { name: 'Details', exact: true })).toBeVisible()
  await page.keyboard.press('ArrowRight')
  await expect(page.getByRole('tabpanel', { name: 'Next steps', exact: true })).toBeVisible()
  await page.keyboard.press('Home')
  await expect(page.getByRole('tabpanel', { name: 'Activity', exact: true })).toBeVisible()

  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth > window.innerWidth,
  )
  expect(overflow).toBe(false)
})
