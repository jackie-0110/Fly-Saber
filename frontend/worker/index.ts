// Cloudflare Worker in front of the static build. Assets (with SPA fallback) are
// served by the platform; only /api/* and /ws/* reach this script, which forwards
// them to the FastAPI backend so the browser keeps a single origin (no CORS).

interface Env {
  ASSETS: { fetch(request: Request): Promise<Response> }
  // Origin of the FastAPI server, e.g. https://flysaber-api.example.com
  BACKEND_URL?: string
}

const BACKEND_PATHS = /^\/(api|ws)(\/|$)/

function backendOrigin(env: Env) {
  const raw = env.BACKEND_URL?.trim()
  if (!raw) return null
  try {
    const url = new URL(raw)
    return url.protocol === 'http:' || url.protocol === 'https:' ? url : null
  } catch {
    return null
  }
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url)
    if (!BACKEND_PATHS.test(url.pathname)) return env.ASSETS.fetch(request)

    const origin = backendOrigin(env)
    if (!origin) {
      return Response.json(
        { detail: 'Backend not configured: set BACKEND_URL on the flysaber Worker.' },
        { status: 503 },
      )
    }

    const target = new URL(url.pathname + url.search, origin)
    // Passing the original request keeps the method, body and Upgrade header, so
    // WebSocket handshakes for /ws/session are proxied end to end.
    const proxied = new Request(target, request)
    proxied.headers.set('X-Forwarded-Host', url.host)
    proxied.headers.set('X-Forwarded-Proto', url.protocol.slice(0, -1))
    try {
      return await fetch(proxied)
    } catch {
      return Response.json({ detail: 'Backend unreachable.' }, { status: 502 })
    }
  },
}
