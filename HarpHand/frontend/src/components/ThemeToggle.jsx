import { createContext, useContext, useEffect, useMemo, useState } from 'react'

const THEME_STORAGE_KEY = 'harphand_theme'
const ThemeContext = createContext(null)

function readStoredTheme() {
  try {
    const storedTheme = window.localStorage.getItem(THEME_STORAGE_KEY)
    return storedTheme === 'light' || storedTheme === 'dark' ? storedTheme : 'dark'
  } catch {
    return 'dark'
  }
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme
  document.documentElement.style.colorScheme = theme
}

const initialTheme = readStoredTheme()
applyTheme(initialTheme)

export function ThemeProvider({ children }) {
  const [theme, setTheme] = useState(initialTheme)

  useEffect(() => {
    applyTheme(theme)
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, theme)
    } catch {
      // The selected theme still applies for this session when storage is unavailable.
    }
  }, [theme])

  const value = useMemo(() => ({ theme, setTheme }), [theme])
  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
}

export default function ThemeToggle({ className = '' }) {
  const context = useContext(ThemeContext)
  if (!context) return null

  const { theme, setTheme } = context

  return (
    <div className={`theme-toggle ${className}`.trim()} role="group" aria-label="Color theme">
      <button
        type="button"
        className={theme === 'light' ? 'is-active' : ''}
        aria-pressed={theme === 'light'}
        onClick={() => setTheme('light')}
      >
        Light
      </button>
      <button
        type="button"
        className={theme === 'dark' ? 'is-active' : ''}
        aria-pressed={theme === 'dark'}
        onClick={() => setTheme('dark')}
      >
        Dark
      </button>
    </div>
  )
}
