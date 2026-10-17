import { create } from 'zustand'
import { emptySession } from './session-client.ts'
import type { SessionSnapshot } from './session-client.ts'

export const useSession = create<SessionSnapshot>(() => ({ ...emptySession }))
