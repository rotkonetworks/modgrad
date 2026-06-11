// Embed entry point — the iframe document loads this. Pure
// postMessage protocol; no DOM, no UI. Wraps the existing WorkerPool.
//
// Wire protocol:
//   parent → embed: { type: 'ocr/recognize', id, blob|bitmap|imageData }
//   embed  → parent: { type: 'ocr/result', id, text, items, avgConfidence }
//                  | { type: 'ocr/error',  id, error }
//   embed  → parent (broadcast on load): { type: 'ocr/hello', version }
//
// The iframe makes its OWN network requests for models + wasm runtime
// (same trusted allowlist as the standalone app). It does NOT touch
// the parent page's network surface. The parent's documents stay in
// in-process structured-clone memory throughout.

import { pool } from './state/workers'
import { installNetworkMonitor } from './state/network'

installNetworkMonitor()

interface RecognizeMsg {
  type: 'ocr/recognize'
  id: number
  blob?: Blob
  bitmap?: ImageBitmap
  imageData?: ImageData
}

function isRecognizeMsg(x: unknown): x is RecognizeMsg {
  return !!x && typeof x === 'object' && (x as { type?: unknown }).type === 'ocr/recognize'
    && typeof (x as { id?: unknown }).id === 'number'
}

async function inputToBitmap(m: RecognizeMsg): Promise<ImageBitmap> {
  if (m.bitmap instanceof ImageBitmap) return m.bitmap
  if (m.blob instanceof Blob) return await createImageBitmap(m.blob)
  if (m.imageData instanceof ImageData) {
    const c = new OffscreenCanvas(m.imageData.width, m.imageData.height)
    const ctx = c.getContext('2d')
    if (!ctx) throw new Error('no 2d context')
    ctx.putImageData(m.imageData, 0, 0)
    return await createImageBitmap(c)
  }
  throw new Error('ocr/recognize requires `blob`, `bitmap`, or `imageData`')
}

window.addEventListener('message', async (ev: MessageEvent) => {
  if (!isRecognizeMsg(ev.data)) return
  const m = ev.data
  const target = ev.source as Window | null
  const origin = ev.origin
  if (!target) return
  try {
    const bitmap = await inputToBitmap(m)
    const result = await pool().recognize(bitmap)
    target.postMessage(
      {
        type: 'ocr/result',
        id: m.id,
        text: result.items.map((i) => i.text).join('\n'),
        items: result.items,
        avgConfidence: result.avgConfidence,
      },
      { targetOrigin: origin },
    )
  } catch (e) {
    target.postMessage(
      { type: 'ocr/error', id: m.id, error: e instanceof Error ? e.message : String(e) },
      { targetOrigin: origin },
    )
  }
})

// Wake the pool now so the model fetch starts immediately — that way
// the first `recognize()` from the parent doesn't include cold-cache
// download time.
void pool()

// Announce readiness once the pool's first worker reports `inited`.
// Until then we still accept messages — they queue inside the pool.
const announceReady = () => {
  parent.postMessage({ type: 'ocr/hello', version: 1, agent: 'modgrad-ocr-embed' }, '*')
}
announceReady() // immediate hello so the client can attach
// And re-broadcast once models are loaded, so callers can opt to
// wait for `ocr/ready` if they want.
import('./state/model').then(({ modelBackend }) => {
  const tick = () => {
    if (modelBackend().kind === 'ready') {
      parent.postMessage({ type: 'ocr/ready' }, '*')
    } else {
      setTimeout(tick, 200)
    }
  }
  tick()
})
