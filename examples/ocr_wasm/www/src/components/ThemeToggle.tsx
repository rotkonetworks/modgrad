import { theme_, toggleTheme } from '../state/theme'

export function ThemeToggle() {
  return (
    <button
      onClick={toggleTheme}
      class="flr-btn gap-1.5 uppercase tracking-wider"
      title={`theme: ${theme_()}`}
    >
      <span class="text-amber">{theme_() === 'dark' ? '☾' : '☀'}</span>
      <span class="text-2xs">{theme_()}</span>
    </button>
  )
}
