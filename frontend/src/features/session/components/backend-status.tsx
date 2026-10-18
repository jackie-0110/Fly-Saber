import { useEffect, useState } from 'react'
import { ChevronDown } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { backendUrl } from '@/features/session/backend'

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
        const response = await fetch(backendUrl('/api/games'), {
          signal: request.signal,
          cache: 'no-store',
        })
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
    <Popover>
      <PopoverTrigger className="status-trigger" aria-label={`${labels[status]}: details`}>
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
        <ChevronDown aria-hidden="true" />
      </PopoverTrigger>
      <PopoverContent align="end" className="status-popover">
        <p className="status-popover-title">{details[status].title}</p>
        <p>{details[status].body}</p>
      </PopoverContent>
    </Popover>
  )
}

// What each state means for the visitor, in place of scattered "not connected" notes.
const details: Record<BackendState, { title: string; body: string }> = {
  checking: {
    title: 'Looking for the simulation server',
    body: 'Songs, audio and the neuron view work meanwhile. The fly moves once the server answers.',
  },
  connected: {
    title: 'Simulation server connected',
    body: 'Choose a brain and a Live play song in the Songs tab, then press play: the brain drives the fly.',
  },
  disconnected: {
    title: 'Simulation server unreachable',
    body: 'Songs still play as previews and the neuron view works, but no brain drives the fly. This retries every few seconds.',
  },
}
