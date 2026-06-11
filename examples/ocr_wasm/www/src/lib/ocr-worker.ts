// Web Worker entry: PP-OCRv5 mobile via ppu-paddle-ocr + onnxruntime-web.
//
// Initializes once (loads det + rec ONNX models from CDN, ~17 MB total,
// HTTP-cached) then handles `recognize` messages with a transferred
// ImageBitmap per page. Returns flattened recognition results.

// ── Network monitor inside the worker (mirror of state/network.ts) ──
//
// The whole "0 external requests" claim in the UI is a lie if the
// worker can fetch without the main thread seeing it. Patch self.fetch,
// XHR, sendBeacon here too and post a message back to the main thread
// when a cross-origin call goes out. Same-origin loads (Vite asset
// chunks) are ignored.
{
  const workerOrigin = location.origin
  const isSameOrigin = (url: string): boolean => {
    try { return new URL(url, location.href).origin === workerOrigin }
    catch { return true }
  }
  const note = (url: string) => {
    if (isSameOrigin(url)) return
    ;(self as unknown as Worker).postMessage({ type: 'ext-call', url, t: Date.now() })
  }
  const origFetch = self.fetch.bind(self)
  self.fetch = ((input: RequestInfo | URL, init?: RequestInit) => {
    note(input instanceof Request ? input.url : String(input))
    return origFetch(input, init)
  }) as typeof self.fetch
  if (typeof XMLHttpRequest !== 'undefined') {
    const origOpen = XMLHttpRequest.prototype.open
    XMLHttpRequest.prototype.open = function (this: XMLHttpRequest, ...args: unknown[]) {
      note(String(args[1]))
      return (origOpen as (...a: unknown[]) => void).apply(this, args)
    } as typeof XMLHttpRequest.prototype.open
  }
  // Worker scopes don't have navigator.sendBeacon, but be defensive.
  const beacon = (self as unknown as { navigator?: { sendBeacon?: (u: string | URL, d?: BodyInit | null) => boolean } }).navigator?.sendBeacon
  if (beacon) {
    const wself = self as unknown as { navigator: { sendBeacon: (u: string | URL, d?: BodyInit | null) => boolean } }
    const orig = beacon.bind(wself.navigator)
    wself.navigator.sendBeacon = (url: string | URL, data?: BodyInit | null) => {
      note(String(url))
      return orig(url, data)
    }
  }
}

import * as ort from 'onnxruntime-web'

// onnxruntime-web's `package.json` exports field doesn't expose the
// raw `.wasm` paths, so we can't `?url` them. Point ort at jsdelivr —
// cross-origin but the asset is integrity-checkable and HTTP-cached
// after first fetch. We'll swap this for `cdn.rotko.net/ort/...` once
// we mirror onnxruntime-web there.
ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ort.env.versions.web}/dist/`
// SharedArrayBuffer needs COOP/COEP we don't have in dev; single thread.
ort.env.wasm.numThreads = 1

// ppu-paddle-ocr's web platform makes two DOM assumptions that workers
// don't satisfy:
//   1. `document.createElement('canvas')` for scratch canvases.
//   2. bare `image instanceof HTMLCanvasElement` (no typeof guard) in
//      its isCanvas check, which throws ReferenceError in workers.
// Polyfill the minimum surface needed for both to behave: a `document`
// shim returning OffscreenCanvas, and a never-matching HTMLCanvasElement
// stub so `instanceof` returns false instead of throwing.
{
  const g = globalThis as { document?: unknown; HTMLCanvasElement?: unknown }
  if (typeof g.document === 'undefined' && typeof OffscreenCanvas !== 'undefined') {
    g.document = {
      createElement(tag: string) {
        if (String(tag).toLowerCase() === 'canvas') return new OffscreenCanvas(1, 1)
        throw new Error(`document.createElement('${tag}') not supported in worker`)
      },
    }
  }
  if (typeof g.HTMLCanvasElement === 'undefined') {
    g.HTMLCanvasElement = class HTMLCanvasElementShim {}
  }
}

import { PaddleOcrService } from 'ppu-paddle-ocr/web'
import type { FlattenedPaddleOcrResult } from 'ppu-paddle-ocr/web'

let service: PaddleOcrService | null = null
let initPromise: Promise<void> | null = null

interface InitRequest { type: 'init' }
interface RecognizeRequest {
  type: 'recognize'
  id: number
  bitmap: ImageBitmap
}
type WorkerRequest = InitRequest | RecognizeRequest

interface ReadyResponse { type: 'ready' }
interface InitedResponse { type: 'inited' }
interface RecognitionItem {
  text: string
  confidence: number
  box: { x: number; y: number; width: number; height: number }
}
interface ResultResponse {
  type: 'result'
  id: number
  items: RecognitionItem[]
  avgConfidence: number
}
interface ErrorResponse { type: 'error'; id: number; error: string }
type WorkerResponse = ReadyResponse | InitedResponse | ResultResponse | ErrorResponse

function post(msg: WorkerResponse) {
  ;(self as unknown as Worker).postMessage(msg)
}

async function ensureInit() {
  if (service) return
  if (!initPromise) {
    initPromise = (async () => {
      // Default engine = opencv.js — better contour detection,
      // canvas-native produced zero boxes on real PDFs in testing.
      // ~9 MB extra one-time fetch, HTTP-cached after that.
      const s = new PaddleOcrService()
      await s.initialize()
      service = s
    })()
  }
  await initPromise
}

post({ type: 'ready' })

self.onmessage = async (ev: MessageEvent<WorkerRequest>) => {
  const req = ev.data
  if (req.type === 'init') {
    try { await ensureInit(); post({ type: 'inited' }) }
    catch (e) { post({ type: 'error', id: -1, error: e instanceof Error ? e.message : String(e) }) }
    return
  }
  if (req.type === 'recognize') {
    try {
      await ensureInit()
      const canvas = new OffscreenCanvas(req.bitmap.width, req.bitmap.height)
      const ctx = canvas.getContext('2d')
      if (!ctx) throw new Error('no 2d context in worker')
      ctx.drawImage(req.bitmap, 0, 0)
      req.bitmap.close()
      // Disable the package's globalImageCache: its key hashes only the
      // first 1024 bytes of the raw RGBA buffer (≈ 256 px from the top
      // row), so every white-margin document collides and we get the
      // first page's result for every subsequent page.
      const result = await service!.recognize(canvas as unknown as HTMLCanvasElement, {
        flatten: true,
        noCache: true,
      } as { flatten: true; noCache?: boolean }) as FlattenedPaddleOcrResult
      const items: RecognitionItem[] = result.results.map((r) => ({
        text: r.text,
        confidence: r.confidence,
        box: { x: r.box.x, y: r.box.y, width: r.box.width, height: r.box.height },
      }))
      post({ type: 'result', id: req.id, items, avgConfidence: result.confidence })
    } catch (e) {
      post({ type: 'error', id: req.id, error: e instanceof Error ? e.message : String(e) })
    }
  }
}
