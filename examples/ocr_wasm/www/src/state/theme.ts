// Two themes. Light is the flrval default; dark is the equivalent.
// User preference wins over system; system pref is the initial seed.

import { createSignal, createEffect } from 'solid-js'

export type Theme = 'light' | 'dark'

const STORE_KEY = 'modgrad-ocr.theme'

function initial(): Theme {
  try {
    const saved = localStorage.getItem(STORE_KEY)
    if (saved === 'light' || saved === 'dark') return saved
  } catch { /* private mode */ }
  if (typeof window !== 'undefined' && window.matchMedia?.('(prefers-color-scheme: dark)').matches) {
    return 'dark'
  }
  return 'light'
}

const [theme, setSig] = createSignal<Theme>(initial())
export const theme_ = theme

createEffect(() => {
  document.documentElement.dataset.theme = theme()
})

export function setTheme(t: Theme) {
  setSig(t)
  try { localStorage.setItem(STORE_KEY, t) } catch { /* ignore */ }
}

export function toggleTheme() {
  setTheme(theme() === 'dark' ? 'light' : 'dark')
}
