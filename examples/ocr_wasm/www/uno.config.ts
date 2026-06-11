import { defineConfig, presetUno } from 'unocss'

// Colors point at CSS custom properties defined in index.css so the
// `data-theme` attribute on <html> switches the whole palette at once.
export default defineConfig({
  presets: [presetUno()],
  theme: {
    colors: {
      bg: 'var(--bg)',
      panel: 'var(--panel)',
      'panel-alt': 'var(--panel-alt)',
      border: 'var(--border)',
      'border-strong': 'var(--border-strong)',
      ink: 'var(--ink)',
      muted: 'var(--muted)',
      key: 'var(--key)',
      accent: 'var(--accent)',
      'accent-bright': 'var(--accent-bright)',
      amber: 'var(--amber)',
      ok: 'var(--ok)',
      warn: 'var(--warn)',
      err: 'var(--err)',
      'on-accent': 'var(--on-accent)',
    },
    fontFamily: {
      sans: 'Verdana, Geneva, Tahoma, sans-serif',
      mono: 'ui-monospace, SFMono-Regular, "Cascadia Mono", Menlo, Consolas, monospace',
    },
    fontSize: { '2xs': ['10.5px', '14px'] },
  },
  shortcuts: {
    'flr-h1': 'bg-accent text-on-accent px-3.5 py-2.5 text-lg font-bold tracking-wide',
    'flr-section': 'bg-accent text-on-accent px-3 py-1.5 text-xs font-bold uppercase tracking-wider',
    'flr-sub': 'bg-panel-alt text-ink px-3.5 py-1.5 text-xs border-b border-border-strong',
    'flr-controls': 'bg-panel border border-border-strong',
    'flr-k': 'text-key',
    'flr-derived': 'font-bold text-ink',
    'flr-derived-y': 'font-bold text-amber',
    'flr-pr': 'font-bold text-ok',
    'flr-foot': 'text-2xs text-muted leading-relaxed',
    'flr-btn': 'inline-flex items-center h-7 px-2.5 border border-border-strong bg-panel text-ink text-xs hover:bg-panel-alt hover:border-key',
    'flr-btn-accent': 'inline-flex items-center h-7 px-2.5 border border-accent-bright bg-accent text-on-accent text-xs hover:bg-accent-bright',
    'flr-btn-danger': 'inline-flex items-center h-7 px-2.5 border border-border-strong bg-panel text-muted text-xs hover:border-err hover:text-err',
    'flr-badge': 'inline-flex items-center px-1.5 h-5 text-2xs font-bold uppercase tracking-wider',
    'flr-badge-pending': 'flr-badge bg-border text-muted',
    'flr-badge-processing': 'flr-badge bg-accent text-on-accent',
    'flr-badge-done': 'flr-badge text-ok border border-ok',
    'flr-badge-error': 'flr-badge text-err border border-err',
  },
})
