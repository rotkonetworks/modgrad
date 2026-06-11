import { Show } from 'solid-js'
import { ConfidentialBadge } from './ConfidentialBadge'
import { ModelPanel } from './ModelPanel'
import { ThemeToggle } from './ThemeToggle'
import { livePoolSize } from '../state/workers'

export function Header(props: {
  onPick: () => void
  onClear: () => void
  hasItems: boolean
  onDownloadAll: () => void
}) {
  return (
    <header class="sticky top-0 z-30">
      <h1 class="flr-h1 m-0 flex items-baseline gap-3">
        <span>OCR Online</span>
        <span class="text-xs font-normal opacity-70 hidden md:inline">
          browser-side · PDFs + images
        </span>
      </h1>
      <div class="flr-controls border-x-0 border-t-0 px-3.5 h-10 flex items-center gap-2.5">
        <button onClick={props.onPick} class="flr-btn-accent">+ files</button>
        <Show when={props.hasItems}>
          <button onClick={props.onDownloadAll} class="flr-btn">↓ .txt</button>
          <button onClick={props.onClear} class="flr-btn-danger">clear</button>
        </Show>
        <span class="text-border mx-1">|</span>
        <span class="flr-k text-2xs uppercase tracking-wider">workers</span>
        <span class="flr-derived text-xs">{livePoolSize()}</span>
        <span class="text-border mx-1 hidden md:inline">|</span>
        <a
          href="/embed-demo.html"
          class="flr-btn hidden md:inline-flex"
          title="Embed this OCR in your own page via a hidden iframe + SDK"
        >Use as API</a>
        <span class="flex-1" />
        <ModelPanel />
        <ConfidentialBadge />
        <ThemeToggle />
      </div>
    </header>
  )
}
