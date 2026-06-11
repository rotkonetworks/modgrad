// Reactive queue store.
//
// Items live entirely in memory. Refreshing the tab wipes them — that's
// the confidential default. Persistence (if added later) is opt-in
// behind a setting, never on by default.

import { createStore, produce } from 'solid-js/store'
import type { LineBox } from '../lib/segment'

export type ItemStatus = 'pending' | 'processing' | 'done' | 'error' | 'cancelled'
export type PageOcrStatus = 'pending' | 'processing' | 'done' | 'error'

export interface PageResult {
  pageNumber: number
  canvas: HTMLCanvasElement
  width: number
  height: number
  lines: LineBox[]
  texts: string[]
  confidences: number[]
  ocrStatus: PageOcrStatus
  ocrError?: string
  // Where the text came from: 'pdf-text' = lifted from the PDF's
  // embedded text layer (no OCR needed); 'ocr' = result of running
  // the recognition model on the page bitmap.
  source?: 'pdf-text' | 'ocr'
}

export interface QueueItem {
  id: string
  file: File
  fileSize: number
  status: ItemStatus
  // Progress: 0..1 across all pages × lines. Updated continuously.
  progress: number
  pages: PageResult[]
  // Total wall-clock processing time once done.
  elapsedMs: number
  error?: string
  addedAt: number
  startedAt?: number
  finishedAt?: number
  // User-corrected transcript. When present, takes precedence over the
  // auto-generated text in copy + download paths. Set by the full-text
  // editor in ItemDetail.
  editedText?: string
}

interface QueueState {
  items: QueueItem[]
}

const [state, setState] = createStore<QueueState>({ items: [] })
export const queue = state

let _id = 0
const nextId = () => `q${Date.now().toString(36)}-${(_id++).toString(36)}`

export const QueueActions = {
  enqueue(files: File[]): QueueItem[] {
    const fresh: QueueItem[] = files.map((f) => ({
      id: nextId(),
      file: f,
      fileSize: f.size,
      status: 'pending',
      progress: 0,
      pages: [],
      elapsedMs: 0,
      addedAt: Date.now(),
    }))
    setState('items', (cur) => [...cur, ...fresh])
    return fresh
  },

  update(id: string, patch: Partial<QueueItem>) {
    setState('items', (i) => i.id === id, patch)
  },

  setPages(id: string, pages: PageResult[]) {
    setState('items', (i) => i.id === id, 'pages', pages)
  },

  appendPage(id: string, page: PageResult) {
    setState(
      'items',
      (i) => i.id === id,
      produce((item) => {
        item.pages.push(page)
      }),
    )
  },

  updatePage(id: string, pageIdx: number, patch: Partial<PageResult>) {
    setState(
      'items',
      (i) => i.id === id,
      produce((item) => {
        const p = item.pages[pageIdx]
        if (!p) return
        Object.assign(p, patch)
      }),
    )
  },

  setLineText(id: string, pageIdx: number, lineIdx: number, text: string, conf: number) {
    setState(
      'items',
      (i) => i.id === id,
      produce((item) => {
        const page = item.pages[pageIdx]
        if (!page) return
        page.texts[lineIdx] = text
        page.confidences[lineIdx] = conf
      }),
    )
  },

  remove(id: string) {
    setState('items', (i) => i.filter((x) => x.id !== id))
  },

  clear() {
    setState('items', [])
  },

  clearDone() {
    setState('items', (i) => i.filter((x) => x.status !== 'done'))
  },
}

// ─── Derived ────────────────────────────────────────────────

export function itemStats(item: QueueItem) {
  let nLines = 0
  let nChars = 0
  for (const p of item.pages) {
    nLines += p.texts.length
    for (const t of p.texts) nChars += t.length
  }
  return { nPages: item.pages.length, nLines, nChars }
}

export function queueStats() {
  let pending = 0
  let processing = 0
  let done = 0
  let error = 0
  let pages = 0
  let lines = 0
  for (const it of state.items) {
    if (it.status === 'pending') pending++
    else if (it.status === 'processing') processing++
    else if (it.status === 'done') done++
    else if (it.status === 'error') error++
    pages += it.pages.length
    for (const p of it.pages) lines += p.texts.length
  }
  return { pending, processing, done, error, pages, lines, total: state.items.length }
}
