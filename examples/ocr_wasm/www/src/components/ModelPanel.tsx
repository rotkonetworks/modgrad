import { createSignal, Show } from 'solid-js'
import { modelBackend, DEFAULT_ORIGIN } from '../state/model'

export function ModelPanel() {
  const [open, setOpen] = createSignal(false)

  const label = () => {
    const b = modelBackend()
    if (b.kind === 'idle') return 'idle'
    if (b.kind === 'loading') return 'loading…'
    if (b.kind === 'ready') return 'ready'
    return 'error'
  }
  const color = () => {
    const b = modelBackend()
    if (b.kind === 'ready') return 'text-ok'
    if (b.kind === 'error') return 'text-err'
    if (b.kind === 'loading') return 'text-amber'
    return 'text-muted'
  }

  return (
    <div class="relative">
      <button onClick={() => setOpen(!open())} class="flr-btn">
        <span class="flr-k mr-1.5">model</span>
        <span class={color()}>{label()}</span>
      </button>

      <Show when={open()}>
        <div class="absolute right-0 top-full mt-1.5 w-80 border border-border-strong bg-panel shadow-xl z-50">
          <div class="flr-section">Inference backend</div>
          <div class="px-3.5 py-2.5 space-y-1 text-2xs font-mono">
            <Row k="arch" v="PP-OCRv5 mobile (det + rec)" />
            <Row k="runtime" v="onnxruntime-web · wasm · 1 thread" />
            <Row k="size" v="≈17 MB models + 26 MB runtime" amber />
            <Row k="cache" v="browser HTTP cache" />
            <Row k="origin" v={DEFAULT_ORIGIN} />
          </div>
          <Show when={modelBackend().kind === 'error'}>
            <div class="border-t border-border px-3.5 py-2.5 text-err text-2xs font-mono break-words">
              {(modelBackend() as { kind: 'error'; message: string }).message}
            </div>
          </Show>
        </div>
      </Show>
    </div>
  )
}

function Row(props: { k: string; v: string; amber?: boolean }) {
  return (
    <div class="flex justify-between gap-3">
      <span class="flr-k">{props.k}</span>
      <span class={`text-right ${props.amber ? 'flr-derived-y' : 'flr-derived'}`}>{props.v}</span>
    </div>
  )
}
