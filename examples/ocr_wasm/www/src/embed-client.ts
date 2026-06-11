// Public consumer SDK for the OCR Online embed iframe.
//
// USAGE (from any third-party page):
//
//   import { OcrClient } from 'https://ocr.rotko.net/embed-client.js'
//   const ocr = await OcrClient.attach('https://ocr.rotko.net/embed.html')
//   const result = await ocr.recognize(file)
//   console.log(result.text)
//
// The image data is transferred from the parent page into the iframe
// via structured clone (in-memory, no network). The iframe performs
// OCR with its own WorkerPool and posts back the result. Documents
// never leave the user's browser.

export interface OcrItem {
  text: string
  confidence: number
  box: { x: number; y: number; width: number; height: number }
}

export interface OcrResult {
  text: string
  items: OcrItem[]
  avgConfidence: number
}

export interface AttachOptions {
  /** Maximum time (ms) to wait for the iframe to send `ocr/hello`. */
  helloTimeoutMs?: number
  /** If true, also wait for `ocr/ready` (models cached) before resolving. */
  awaitModelReady?: boolean
  /** Max time to wait for `ocr/ready`. */
  readyTimeoutMs?: number
}

export class OcrClient {
  private nextId = 0
  private pending = new Map<number, { resolve: (r: OcrResult) => void; reject: (e: Error) => void }>()
  private messageHandler: (ev: MessageEvent) => void

  private constructor(private iframe: HTMLIFrameElement, private targetOrigin: string) {
    this.messageHandler = (ev) => this.onMessage(ev)
    window.addEventListener('message', this.messageHandler)
  }

  /**
   * Mount or adopt an iframe pointing at the embed URL, then wait for
   * the embed to signal that it has loaded.
   */
  static async attach(
    srcOrIframe: string | HTMLIFrameElement,
    opts: AttachOptions = {},
  ): Promise<OcrClient> {
    const iframe = typeof srcOrIframe === 'string'
      ? mountHiddenIframe(srcOrIframe)
      : srcOrIframe

    const url = new URL(iframe.src || (typeof srcOrIframe === 'string' ? srcOrIframe : ''), location.href)
    const targetOrigin = url.origin

    await waitForMessage(iframe, 'ocr/hello', opts.helloTimeoutMs ?? 30000)
    if (opts.awaitModelReady) {
      await waitForMessage(iframe, 'ocr/ready', opts.readyTimeoutMs ?? 60000)
    }
    return new OcrClient(iframe, targetOrigin)
  }

  /** Run OCR on a file, blob, bitmap, or raw ImageData. */
  recognize(input: Blob | File | ImageBitmap | ImageData): Promise<OcrResult> {
    const id = this.nextId++
    return new Promise<OcrResult>((resolve, reject) => {
      this.pending.set(id, { resolve, reject })
      const cw = this.iframe.contentWindow
      if (!cw) {
        this.pending.delete(id)
        reject(new Error('OCR iframe has no contentWindow'))
        return
      }
      if (input instanceof ImageBitmap) {
        cw.postMessage({ type: 'ocr/recognize', id, bitmap: input }, this.targetOrigin, [input])
      } else if (input instanceof ImageData) {
        cw.postMessage({ type: 'ocr/recognize', id, imageData: input }, this.targetOrigin)
      } else {
        cw.postMessage({ type: 'ocr/recognize', id, blob: input }, this.targetOrigin)
      }
    })
  }

  /** Stop listening for messages. The iframe is left alone — caller owns it. */
  destroy() {
    window.removeEventListener('message', this.messageHandler)
    for (const { reject } of this.pending.values()) reject(new Error('OcrClient destroyed'))
    this.pending.clear()
  }

  private onMessage(ev: MessageEvent) {
    if (ev.source !== this.iframe.contentWindow) return
    const m = ev.data as { type?: string; id?: number; text?: string; items?: OcrItem[]; avgConfidence?: number; error?: string }
    if (!m || typeof m.id !== 'number') return
    const slot = this.pending.get(m.id)
    if (!slot) return
    this.pending.delete(m.id)
    if (m.type === 'ocr/result') {
      slot.resolve({ text: m.text ?? '', items: m.items ?? [], avgConfidence: m.avgConfidence ?? 0 })
    } else if (m.type === 'ocr/error') {
      slot.reject(new Error(m.error || 'OCR error'))
    }
  }
}

function mountHiddenIframe(src: string): HTMLIFrameElement {
  const iframe = document.createElement('iframe')
  iframe.src = src
  iframe.style.cssText = 'position:absolute;width:0;height:0;border:0;visibility:hidden;clip:rect(0 0 0 0)'
  iframe.setAttribute('aria-hidden', 'true')
  iframe.setAttribute('title', 'OCR Online embed')
  iframe.setAttribute('referrerpolicy', 'no-referrer')
  iframe.setAttribute('allow', '')
  document.body.appendChild(iframe)
  return iframe
}

function waitForMessage(iframe: HTMLIFrameElement, type: string, timeoutMs: number): Promise<void> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      cleanup()
      reject(new Error(`OCR iframe did not send '${type}' within ${timeoutMs}ms`))
    }, timeoutMs)
    const onMsg = (ev: MessageEvent) => {
      if (ev.source !== iframe.contentWindow) return
      const m = ev.data as { type?: string }
      if (m?.type === type) {
        cleanup()
        resolve()
      }
    }
    const cleanup = () => {
      clearTimeout(timer)
      window.removeEventListener('message', onMsg)
    }
    window.addEventListener('message', onMsg)
  })
}
