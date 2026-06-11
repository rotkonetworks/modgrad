import { createSignal, Show } from 'solid-js'

export function Dropzone(props: {
  onFiles: (files: File[]) => void
  onPick: () => void
  compact?: boolean
}) {
  const [over, setOver] = createSignal(false)
  function pickFiles(ev: DragEvent): File[] {
    const list = ev.dataTransfer?.files
    if (!list) return []
    return Array.from(list).filter(acceptable)
  }
  function onDrop(ev: DragEvent) {
    ev.preventDefault()
    setOver(false)
    const files = pickFiles(ev)
    if (files.length) props.onFiles(files)
  }

  return (
    <div
      class={`border-2 border-dashed text-center ${
        over() ? 'border-accent-bright bg-accent/5' : 'border-border-strong bg-panel'
      } ${props.compact ? 'p-3' : 'p-8'}`}
      onDragOver={(e) => { e.preventDefault(); setOver(true) }}
      onDragLeave={() => setOver(false)}
      onDrop={onDrop}
    >
      <Show
        when={!props.compact}
        fallback={
          <div class="flex items-center justify-between gap-3 text-2xs">
            <span class="flr-k uppercase tracking-wider">add more</span>
            <button class="flr-btn" onClick={props.onPick}>browse</button>
          </div>
        }
      >
        <h2 class="text-base font-bold mb-1">Drop PDFs or images</h2>
        <p class="flr-foot mb-4">
          PDF, PNG, JPG, WebP. Paste from clipboard with
          <kbd class="font-mono text-2xs px-1 mx-1 border border-border-strong">⌘/Ctrl + V</kbd>.
        </p>
        <button onClick={props.onPick} class="flr-btn-accent">browse</button>
      </Show>
    </div>
  )
}

const accepted = new Set([
  'application/pdf',
  'image/png',
  'image/jpeg',
  'image/webp',
])
function acceptable(f: File): boolean {
  if (accepted.has(f.type)) return true
  const n = f.name.toLowerCase()
  return n.endsWith('.pdf') || n.endsWith('.png') || n.endsWith('.jpg') ||
    n.endsWith('.jpeg') || n.endsWith('.webp')
}
