import { createSignal, createEffect, Show, onMount } from 'solid-js'
import { queue, QueueActions } from './state/queue'
import { pool } from './state/workers'
import { installNetworkMonitor } from './state/network'
import { openPdf, rasterizeImage } from './lib/pdf'
import type { RasterizedPage } from './lib/pdf'
import { recognizePage } from './lib/ocr'
import { downloadAllText, downloadItemText } from './lib/download'
import { Header } from './components/Header'
import { Dropzone } from './components/Dropzone'
import { QueueList } from './components/QueueList'
import { ItemDetail } from './components/ItemDetail'
import { LandingInfo } from './components/LandingInfo'

// Cap concurrent file-processing. Each open PDF holds its full
// ArrayBuffer in memory + per-page canvases; dropping 20 files at once
// would push >1 GB through the heap before any OCR ran. Two at a time
// is a safe default — the worker pool already saturates per-page
// parallelism inside each item.
const MAX_ITEM_CONCURRENCY = 2

export function App() {
  const [selectedId, setSelectedId] = createSignal<string | null>(null)
  const [dragOverlay, setDragOverlay] = createSignal(false)
  let fileInputRef: HTMLInputElement | undefined

  // Auto-select the first item once the queue has at least one entry,
  // unless the user has already picked something.
  createEffect(() => {
    if (!selectedId() && queue.items.length > 0) {
      setSelectedId(queue.items[0]!.id)
    }
  })

  onMount(() => {
    installNetworkMonitor()
    // Force the worker pool to come up at startup so the first worker
    // begins fetching models immediately. The model panel + WORKERS
    // counter need this signal to ever leave their idle state.
    void pool()
    // Whole-page drag handlers: stop the browser opening the file
    // directly if the user misses the drop zone, and show the overlay.
    window.addEventListener('dragenter', (e) => {
      if (e.dataTransfer?.types?.includes('Files')) setDragOverlay(true)
    })
    window.addEventListener('dragover', (e) => e.preventDefault())
    window.addEventListener('drop', (e) => {
      e.preventDefault()
      setDragOverlay(false)
      const files = Array.from(e.dataTransfer?.files ?? [])
      if (files.length) addFiles(files)
    })
    window.addEventListener('dragleave', (e) => {
      // dragleave fires for every child element; only hide overlay when
      // the drag exits the document entirely.
      if (e.relatedTarget === null) setDragOverlay(false)
    })

    // Ctrl+V paste — accept any image data from the system clipboard
    // (screenshots are the obvious use case). We don't intercept paste
    // inside form inputs.
    window.addEventListener('paste', (e: ClipboardEvent) => {
      const target = e.target as HTMLElement | null
      if (target && /^(INPUT|TEXTAREA|SELECT)$/i.test(target.tagName)) return
      const items = e.clipboardData?.items
      if (!items) return
      const files: File[] = []
      const stamp = new Date().toISOString().replace(/[:.]/g, '-')
      for (let i = 0; i < items.length; i++) {
        const it = items[i]
        if (!it || !it.type.startsWith('image/')) continue
        const file = it.getAsFile()
        if (!file) continue
        const ext = (it.type.split('/')[1] || 'png').replace('jpeg', 'jpg')
        files.push(
          new File([file], file.name && file.name !== 'image.png' ? file.name : `paste-${stamp}.${ext}`, {
            type: it.type,
            lastModified: Date.now(),
          }),
        )
      }
      if (files.length > 0) {
        e.preventDefault()
        void addFiles(files)
      }
    })
  })

  // ── Item-level concurrency semaphore ─────────────────────
  // Each enqueue goes into pending. A scheduler pulls up to
  // MAX_ITEM_CONCURRENCY items at a time and starts processing them.
  const activeIds = new Set<string>()

  function scheduleNext() {
    while (activeIds.size < MAX_ITEM_CONCURRENCY) {
      const next = queue.items.find(
        (it) => it.status === 'pending' && !activeIds.has(it.id),
      )
      if (!next) return
      activeIds.add(next.id)
      void processItem(next.id).finally(() => {
        activeIds.delete(next.id)
        scheduleNext()
      })
    }
  }

  // Anything bigger than this gets a confirm prompt — a 2 GB PDF
  // through `file.arrayBuffer()` OOMs the tab on most laptops. The
  // user can still proceed if they really want to.
  const HUGE_FILE_BYTES = 500 * 1024 * 1024
  function vetSize(files: File[]): File[] {
    const huge = files.filter((f) => f.size > HUGE_FILE_BYTES)
    if (huge.length === 0) return files
    const names = huge.map((f) => `${f.name} (${(f.size / 1024 / 1024).toFixed(0)} MB)`).join('\n')
    const ok = window.confirm(
      `The following file(s) are over ${HUGE_FILE_BYTES / 1024 / 1024} MB ` +
      `and may exhaust browser memory:\n\n${names}\n\nProceed anyway?`,
    )
    return ok ? files : files.filter((f) => f.size <= HUGE_FILE_BYTES)
  }

  async function addFiles(rawFiles: File[]) {
    const files = vetSize(rawFiles)
    if (files.length === 0) return
    const fresh = QueueActions.enqueue(files)
    if (!selectedId() && fresh.length > 0) setSelectedId(fresh[0]!.id)
    scheduleNext()
  }

  function retryItem(id: string) {
    QueueActions.update(id, {
      status: 'pending', progress: 0, error: undefined,
      pages: [], startedAt: undefined, finishedAt: undefined, elapsedMs: 0,
    })
    scheduleNext()
  }

  async function processItem(id: string) {
    const item = queue.items.find((x) => x.id === id)
    if (!item || item.status !== 'pending') return
    const startedAt = Date.now()
    QueueActions.update(id, { status: 'processing', startedAt, progress: 0 })

    // Progress weighting: rasterize counts for 30% of the bar, OCR for
    // 70%. With totalPages known up front from pdf.numPages, the bar
    // monotonically advances and never regresses.
    let totalPages = 0
    let rasterized = 0
    let ocrDone = 0
    const ocrPromises: Promise<void>[] = []
    const bumpProgress = () => {
      const ras = totalPages === 0 ? 0 : rasterized / totalPages
      const ocr = totalPages === 0 ? 0 : ocrDone / totalPages
      QueueActions.update(id, { progress: 0.3 * ras + 0.7 * ocr })
    }

    const handlePage = (page: RasterizedPage, pageIdx: number) => {
      // If pdf.js found a real text layer, use it straight — far
      // faster + perfect-quality vs running OCR on the rasterized
      // pixels. The user still sees the canvas + overlay boxes.
      if (page.textLayer) {
        const tl = page.textLayer.lines
        QueueActions.appendPage(id, {
          pageNumber: page.pageNumber,
          canvas: page.canvas,
          width: page.canvas.width,
          height: page.canvas.height,
          lines: tl.map((l) => l.box),
          texts: tl.map((l) => l.text),
          confidences: tl.map(() => 1),
          ocrStatus: 'done',
          source: 'pdf-text',
        })
        rasterized++
        ocrDone++
        bumpProgress()
        return
      }
      QueueActions.appendPage(id, {
        pageNumber: page.pageNumber,
        canvas: page.canvas,
        width: page.canvas.width,
        height: page.canvas.height,
        lines: [], texts: [], confidences: [],
        ocrStatus: 'processing',
        source: 'ocr',
      })
      rasterized++
      bumpProgress()
      ocrPromises.push(
        recognizePage(page.canvas)
          .then((results) => {
            QueueActions.updatePage(id, pageIdx, {
              lines: results.map((r) => r.line),
              texts: results.map((r) => r.text),
              confidences: results.map((r) => r.confidence),
              ocrStatus: 'done',
            })
          })
          .catch((e: unknown) => {
            const msg = e instanceof Error ? e.message : String(e)
            QueueActions.updatePage(id, pageIdx, { ocrStatus: 'error', ocrError: msg })
          })
          .finally(() => { ocrDone++; bumpProgress() }),
      )
    }

    try {
      const isPdf =
        item.file.type === 'application/pdf' ||
        item.file.name.toLowerCase().endsWith('.pdf')

      if (isPdf) {
        const { numPages, stream } = await openPdf(await item.file.arrayBuffer())
        totalPages = numPages
        let i = 0
        for await (const page of stream) handlePage(page, i++)
      } else {
        totalPages = 1
        handlePage(await rasterizeImage(item.file), 0)
      }

      await Promise.all(ocrPromises)

      const finishedAt = Date.now()
      QueueActions.update(id, {
        status: 'done', progress: 1,
        elapsedMs: finishedAt - startedAt, finishedAt,
      })
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e)
      QueueActions.update(id, { status: 'error', error: msg })
    }
  }

  function onPickClick() { fileInputRef?.click() }
  function onPickChange(ev: Event) {
    const inp = ev.currentTarget as HTMLInputElement
    const files = Array.from(inp.files ?? [])
    if (files.length) void addFiles(files)
    inp.value = '' // reset so picking the same file again re-fires
  }

  const selectedItem = () =>
    queue.items.find((x) => x.id === selectedId()) ?? null

  return (
    <div class="min-h-screen flex flex-col bg-bg text-ink">
      <Header
        onPick={onPickClick}
        onClear={() => { QueueActions.clear(); setSelectedId(null) }}
        hasItems={queue.items.length > 0}
        onDownloadAll={() => downloadAllText(queue.items)}
      />
      <input
        ref={fileInputRef}
        type="file"
        accept="application/pdf,image/png,image/jpeg,image/webp"
        multiple
        class="hidden"
        onChange={onPickChange}
      />

      <Show
        when={queue.items.length > 0}
        fallback={
          <main class="flex-1 overflow-y-auto p-6">
            <div class="w-full max-w-2xl mx-auto">
              <Dropzone onFiles={addFiles} onPick={onPickClick} />
              <LandingInfo />
            </div>
          </main>
        }
      >
        <main class="flex-1 grid grid-cols-[320px_1fr] min-h-0">
          <aside class="border-r border-border bg-panel flex flex-col min-h-0">
            <div class="p-3 border-b border-border">
              <Dropzone onFiles={addFiles} onPick={onPickClick} compact />
            </div>
            <QueueList
              items={queue.items}
              selectedId={selectedId()}
              onSelect={setSelectedId}
              onRemove={(id) => {
                QueueActions.remove(id)
                if (selectedId() === id) {
                  const next = queue.items.find((x) => x.id !== id)
                  setSelectedId(next?.id ?? null)
                }
              }}
              onDownload={downloadItemText}
              onClearDone={() => QueueActions.clearDone()}
              onRetry={retryItem}
            />
          </aside>

          <section class="min-h-0 min-w-0">
            <Show
              when={selectedItem()}
              fallback={
                <div class="h-full flex items-center justify-center text-muted text-sm">
                  Pick a document to view results
                </div>
              }
            >
              {(it) => <ItemDetail item={it()} />}
            </Show>
          </section>
        </main>
      </Show>

      <Show when={dragOverlay()}>
        <div class="fixed inset-0 bg-bg/90 backdrop-blur-sm flex items-center justify-center z-50 pointer-events-none">
          <div class="p-12 border-2 border-dashed border-accent rounded-2xl text-center">
            <div class="text-6xl mb-4 text-accent">⤓</div>
            <div class="text-xl">Drop to queue</div>
          </div>
        </div>
      </Show>
    </div>
  )
}
