import { createSignal, Show } from 'solid-js'
import { NetworkMonitor } from './NetworkMonitor'

export function ConfidentialBadge() {
  const [open, setOpen] = createSignal(false)

  return (
    <div class="relative">
      <button onClick={() => setOpen(!open())} class="flr-btn gap-2">
        <span class="text-ok">●</span>
        <span>local</span>
        <NetworkMonitor />
      </button>

      <Show when={open()}>
        <div class="absolute right-0 top-full mt-1.5 w-72 border border-border-strong bg-panel shadow-xl z-50">
          <div class="flr-section">Runtime</div>
          <div class="px-3.5 py-2.5 text-2xs font-mono space-y-1">
            <Row k="ocr" v="this page (Web Worker)" />
            <Row k="weights" v="cached after first GET" />
            <Row k="documents" v="in-memory, not persisted" />
            <Row k="external requests" v="0 during OCR" />
            <Row k="csp" v="default-src 'self'" />
          </div>
        </div>
      </Show>
    </div>
  )
}

function Row(props: { k: string; v: string }) {
  return (
    <div class="flex justify-between gap-3">
      <span class="flr-k">{props.k}</span>
      <span class="flr-derived text-right">{props.v}</span>
    </div>
  )
}
