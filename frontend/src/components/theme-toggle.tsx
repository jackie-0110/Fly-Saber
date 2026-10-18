import { Moon, Sun } from 'lucide-react'
import { useTheme } from '@/lib/theme'

export function ThemeToggle() {
  const theme = useTheme((s) => s.theme)
  const toggle = useTheme((s) => s.toggle)
  const dark = theme === 'dark'
  return (
    <button
      type="button"
      className="theme-toggle"
      aria-pressed={dark}
      aria-label="Dark mode"
      onClick={toggle}
    >
      {dark ? <Moon aria-hidden="true" /> : <Sun aria-hidden="true" />}
      {dark ? 'Dark' : 'Light'}
    </button>
  )
}
