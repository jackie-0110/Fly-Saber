import { useEffect, useState } from 'react'
import { Badge } from '@/components/ui/badge'

type BackendState = 'checking' | 'connected' | 'disconnected'

const labels: Record<BackendState, string> = {
  checking: 'Checking backend',
  connected: 'Backend connected',
  disconnected: 'Backend disconnected',
}

export function BackendStatus() {
  const [status, setStatus] = useState<BackendState>('checking')

  useEffect(() => {
    let disposed = false
    let request: AbortController | undefined
    let nextCheck: ReturnType<typeof setTimeout> | undefined
    let deadline: ReturnType<typeof setTimeout> | undefined

    async function check() {
      request = new AbortController()
      deadline = setTimeout(() => request?.abort(), 4000)
      try {
        const response = await fetch('/api/games', { signal: request.signal, cache: 'no-store' })
        if (!response.ok) throw new Error('Backend unavailable')
        const games: unknown = await response.json()
        // Vite's HTML fallback or an unrelated service must not count as a connection.
        const valid =
          Array.isArray(games) &&
          games.length > 0 &&
          games.every(
            (game) =>
              typeof game === 'object' &&
              game !== null &&
              typeof game.id === 'string' &&
              typeof game.label === 'string',
          )
        if (!disposed) setStatus(valid ? 'connected' : 'disconnected')
      } catch {
        if (!disposed) setStatus('disconnected')
      } finally {
        clearTimeout(deadline)
        if (!disposed) nextCheck = setTimeout(check, 5000)
      }
    }

    void check()
    return () => {
      disposed = true
      clearTimeout(nextCheck)
      clearTimeout(deadline)
      request?.abort()
    }
  }, [])

  return (
    <Badge
      variant="outline"
      className="backend-status"
      data-state={status}
      role="status"
      aria-live="polite"
    >
      <span className="status-dot" aria-hidden="true" />
      {labels[status]}
    </Badge>
  )
}
