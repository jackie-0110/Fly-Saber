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
  await expect(page.getByRole('heading', { name: 'FlySaber home', exact: true })).toBeVisible()
  await expect(page.getByRole('main')).toBeVisible()
  await expect(page.getByRole('link', { name: 'FlySaber home' })).toHaveAttribute('href', '/')
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

test('workspace fits the viewport and the connection status opens from the keyboard', async ({
  page,
}) => {
  await page.goto('/')
  const status = page.getByRole('button', { name: 'Offline preview' })
  await status.focus()
  await page.keyboard.press('Enter')
  await expect(page.getByText('Simulation server not connected', { exact: true })).toBeVisible()
  await page.keyboard.press('Escape')
  await expect(page.getByText('Simulation server not connected', { exact: true })).toBeHidden()
  await expect(status).toBeFocused()

  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth > window.innerWidth,
  )
  expect(overflow).toBe(false)
})
