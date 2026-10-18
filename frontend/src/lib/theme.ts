import { create } from 'zustand'

export type Theme = 'light' | 'dark'

const STORAGE_KEY = 'fly-saber-theme'

function initialTheme(): Theme {
  // index.html sets data-theme before first paint; read it back so React agrees.
  const set = document.documentElement.dataset.theme
  if (set === 'light' || set === 'dark') return set
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

function apply(theme: Theme) {
  const root = document.documentElement
  root.dataset.theme = theme
  root.classList.toggle('dark', theme === 'dark')
  document
    .querySelector('meta[name="theme-color"]')
    ?.setAttribute('content', theme === 'dark' ? '#111111' : '#f5f8f7')
}

export const useTheme = create<{ theme: Theme; toggle: () => void }>((set, get) => {
  const theme = initialTheme()
  apply(theme)
  return {
    theme,
    toggle: () => {
      const next = get().theme === 'dark' ? 'light' : 'dark'
      apply(next)
      try {
        localStorage.setItem(STORAGE_KEY, next)
      } catch {
        // Storage can be blocked; the choice then lasts for this page only.
      }
      set({ theme: next })
    },
  }
})

/*
 * WebGL colors can't read CSS variables, so each view takes its palette from here.
 * `field` matches --field in index.css for the active theme.
 */
const SCENE = {
  light: {
    field: '#e9eeec',
    restInk: '#96a1a1',
    excite: '#d54c25',
    inhibit: '#2a67bd',
    quietEdge: '#3f4a4b',
    selection: '#162021',
    floor: '#f4f8f6',
    floorLine: '#6a7474',
    rail: '#a5ada9',
    lane: '#d0d6d3',
    track: '#dfe4e2',
    hemiSky: '#ffffff',
    hemiGround: '#bec6c2',
    fill: '#e6eeeb',
    noteFace: '#f8fbf9',
  },
  dark: {
    field: '#121212',
    restInk: '#5c6360',
    excite: '#ff7a4d',
    inhibit: '#5b9bff',
    quietEdge: '#9aa39f',
    selection: '#3eeba3',
    floor: '#161716',
    floorLine: '#3eeba3',
    rail: '#2b2d2c',
    lane: '#1f2120',
    track: '#191a19',
    hemiSky: '#d8e6df',
    hemiGround: '#111111',
    fill: '#3eeba3',
    noteFace: '#ededed',
  },
} as const

export function useScenePalette() {
  return SCENE[useTheme((s) => s.theme)]
}
