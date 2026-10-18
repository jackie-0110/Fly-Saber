// One origin for REST, audio, and sessions. Empty means the app's own host.
const origin = (import.meta.env?.VITE_BACKEND_URL ?? '').replace(/\/$/, '')

export function backendUrl(path: string) {
  return `${origin}${path}`
}

export function sessionUrl() {
  const url = new URL(backendUrl('/ws/session'), window.location.href)
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:'
  return url.href
}

export async function backendJson(path: string, signal: AbortSignal): Promise<unknown> {
  const response = await fetch(backendUrl(path), {
    signal: AbortSignal.any([signal, AbortSignal.timeout(10000)]),
    cache: 'no-store',
  })
  if (!response.ok) throw new Error(`The server returned ${response.status}. Retry in a moment.`)
  return response.json()
}
