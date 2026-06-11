// Web Worker pool for OCR inference.
//
// Each worker holds a single PaddleOcrService instance (PP-OCRv5 mobile).
// Models load once on first init (~17 MB HTTP-cached). After that,
// each page is processed by sending an ImageBitmap to the next free
// worker. Pool defaults to navigator.hardwareConcurrency-1 capped at 4.

import { createSignal } from 'solid-js'
import OcrWorker from '../lib/ocr-worker.ts?worker'
import {
  markModelLoading, markModelReady, markModelError,
  modelOrigin, DEFAULT_ORIGIN,
} from './model'
import { recordExternal } from './network'

export interface OcrItem {
  text: string
  confidence: number
  box: { x: number; y: number; width: number; height: number }
}

export interface OcrPageResult {
  items: OcrItem[]
  avgConfidence: number
}

type Resolver = (r: OcrPageResult) => void
type Rejecter = (e: Error) => void

interface PendingJob { id: number; resolve: Resolver; reject: Rejecter }
interface WorkerSlot {
  worker: Worker
  ready: boolean
  initFailed: boolean
  currentJob: PendingJob | null
}
interface QueuedTask { job: PendingJob; bitmap: ImageBitmap }

// Reactive accessor so the header can show the live pool size as
// siblings come up after the first init.
const [poolSize, setPoolSize] = createSignal(0)
export const livePoolSize = poolSize

export class WorkerPool {
  private slots: WorkerSlot[]
  private queue: QueuedTask[] = []
  private nextId = 0
  private targetSize: number

  constructor(size: number) {
    this.targetSize = size
    // Spawn ONE worker first. Siblings come up only after that worker's
    // init completes, so the model files are in the browser HTTP cache
    // before the others race on the same remote fetches.
    this.slots = [this.spawn()]
    setPoolSize(this.slots.length)
    markModelLoading()
    this.slots[0]!.worker.postMessage({ type: 'init' })
  }

  get size() { return this.slots.length }

  private spawnSiblings() {
    while (this.slots.length < this.targetSize) this.slots.push(this.spawn())
    setPoolSize(this.slots.length)
  }

  private spawn(): WorkerSlot {
    const worker = new OcrWorker()
    const slot: WorkerSlot = { worker, ready: false, initFailed: false, currentJob: null }
    worker.onmessage = (ev) => this.onWorkerMessage(slot, ev)
    worker.onerror = (ev) => this.onWorkerError(slot, ev)
    return slot
  }

  private onWorkerMessage(slot: WorkerSlot, ev: MessageEvent) {
    const msg = ev.data as
      | { type: 'ready' }
      | { type: 'inited' }
      | { type: 'result'; id: number; items: OcrItem[]; avgConfidence: number }
      | { type: 'error'; id: number; error: string }
      | { type: 'ext-call'; url: string; t: number }

    if (msg.type === 'ext-call') {
      recordExternal(msg.url)
      return
    }
    if (msg.type === 'ready' || msg.type === 'inited') {
      slot.ready = true
      slot.initFailed = false
      if (msg.type === 'inited') {
        // Any previously-failed init recovers — re-assert ready.
        markModelReady(modelOrigin() || DEFAULT_ORIGIN)
        if (this.slots.length < this.targetSize) {
          this.spawnSiblings()
          for (const s of this.slots) {
            if (!s.ready && !s.initFailed) s.worker.postMessage({ type: 'init' })
          }
        }
      }
      this.drain()
      return
    }
    const job = slot.currentJob
    if (!job) {
      // Init error for the eager kickoff — surface in the model panel,
      // mark slot so we don't keep trying.
      if (msg.type === 'error') {
        slot.initFailed = true
        // Only mark global error if NO slot is healthy.
        if (!this.slots.some((s) => s.ready)) markModelError(msg.error)
      }
      return
    }
    slot.currentJob = null
    if (msg.type === 'result') {
      job.resolve({ items: msg.items, avgConfidence: msg.avgConfidence })
    } else if (msg.type === 'error') {
      job.reject(new Error(msg.error))
    }
    this.drain()
  }

  private onWorkerError(slot: WorkerSlot, ev: ErrorEvent) {
    const job = slot.currentJob
    slot.currentJob = null
    if (job) job.reject(new Error(ev.message || 'worker fault'))
    slot.worker.terminate()
    const fresh = this.spawn()
    const idx = this.slots.indexOf(slot)
    if (idx >= 0) this.slots[idx] = fresh
    // CRITICAL: queued bitmaps were waiting on this slot. Without
    // explicit drain here, the queue stalls until another slot
    // happens to finish a job.
    this.drain()
  }

  private drain() {
    for (const slot of this.slots) {
      if (!slot.ready || slot.currentJob) continue
      const next = this.queue.shift()
      if (!next) return
      slot.currentJob = next.job
      slot.worker.postMessage(
        { type: 'recognize', id: next.job.id, bitmap: next.bitmap },
        [next.bitmap],
      )
    }
  }

  recognize(bitmap: ImageBitmap): Promise<OcrPageResult> {
    return new Promise((resolve, reject) => {
      const job: PendingJob = { id: this.nextId++, resolve, reject }
      this.queue.push({ job, bitmap })
      this.drain()
    })
  }

  terminate() {
    for (const s of this.slots) s.worker.terminate()
    this.slots = []
    setPoolSize(0)
    for (const q of this.queue) {
      q.bitmap.close()
      q.job.reject(new Error('pool terminated'))
    }
    this.queue = []
  }
}

let _pool: WorkerPool | null = null
export function pool(): WorkerPool {
  if (!_pool) {
    const hc = (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 2
    // PP-OCRv5 sessions ~30-50 MB resident each.
    const size = Math.min(4, Math.max(1, hc - 1))
    _pool = new WorkerPool(size)
  }
  return _pool
}
