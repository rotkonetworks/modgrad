// pdf.js wrapper — rasterize a PDF page to a canvas, return ImageData.

import * as pdfjs from 'pdfjs-dist'
// Vite-friendly worker import: the ?url suffix produces a hashed asset URL.
import workerUrl from 'pdfjs-dist/build/pdf.worker.min.mjs?url'

pdfjs.GlobalWorkerOptions.workerSrc = workerUrl

export interface RasterizedPage {
  pageNumber: number
  canvas: HTMLCanvasElement
  imageData: ImageData
  // If the PDF page had an embedded text layer with meaningful content,
  // this is the recovered lines + per-line geometry. When set, the
  // caller should skip OCR — the data is already perfect.
  textLayer?: {
    lines: { text: string; box: { x: number; y: number; w: number; h: number } }[]
  }
}

export interface RasterizeOptions {
  // Target page width in CSS px (height auto-scales by aspect).
  // Larger = better OCR fidelity, more memory. 1200 is a reasonable default
  // for ~A4 at ~150 DPI equivalent.
  pageWidthPx?: number
}

/**
 * Open the PDF and resolve to `numPages` + an async generator. The
 * caller knows the total page count *before* the first page lands, so
 * progress doesn't go backwards as pages stream in.
 */
export async function openPdf(
  source: ArrayBuffer | Uint8Array,
  opts: RasterizeOptions = {},
): Promise<{ numPages: number; stream: AsyncGenerator<RasterizedPage> }> {
  const pdf = await pdfjs.getDocument({ data: source }).promise
  const targetWidth = opts.pageWidthPx ?? 1200
  async function* gen() {
    for (let i = 1; i <= pdf.numPages; i++) {
      const page = await pdf.getPage(i)
      const baseViewport = page.getViewport({ scale: 1.0 })
      const scale = targetWidth / baseViewport.width
      const viewport = page.getViewport({ scale })
      const canvas = document.createElement('canvas')
      canvas.width = Math.ceil(viewport.width)
      canvas.height = Math.ceil(viewport.height)
      const ctx = canvas.getContext('2d', { willReadFrequently: true })
      if (!ctx) throw new Error('canvas 2D context unavailable')
      ctx.fillStyle = '#ffffff'
      ctx.fillRect(0, 0, canvas.width, canvas.height)
      await page.render({ canvasContext: ctx, viewport }).promise
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)

      // Try to extract the embedded text layer. Most modern PDFs have
      // real text; only scans need actual OCR. The viewport scale is
      // the same one we used for rendering, so the box coordinates
      // line up with the rasterized canvas.
      const textLayer = await extractTextLayer(page, viewport)

      yield { pageNumber: i, canvas, imageData, textLayer }
      page.cleanup()
    }
  }
  return { numPages: pdf.numPages, stream: gen() }
}

interface PdfTextItem {
  str: string
  // pdf.js gives a 6-element transform matrix [a, b, c, d, e, f].
  transform: number[]
  width: number
  height: number
}

/**
 * Extract the page's embedded text layer (if any). Returns `undefined`
 * when the page has no meaningful text — i.e. the document is scanned
 * imagery and we should rasterize + OCR it.
 *
 * The bounding boxes are converted into canvas-pixel coordinates so
 * the demo can overlay them just like OCR-detected boxes.
 */
async function extractTextLayer(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  page: any,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  viewport: any,
): Promise<RasterizedPage['textLayer']> {
  try {
    const content = await page.getTextContent({
      includeMarkedContent: false,
      disableNormalization: false,
    }) as { items: PdfTextItem[] }
    const items = content.items.filter((it) => it.str && it.str.trim().length > 0)
    if (items.length === 0) return undefined
    // Sanity: 20 non-whitespace chars total → call it a real text PDF.
    const total = items.reduce((s, it) => s + it.str.length, 0)
    if (total < 20) return undefined

    // Project pdf-coords → viewport-coords for each text item, then
    // greedily merge into reading-order lines by vertical proximity.
    const projected = items.map((it) => {
      // pdf.js places the text origin at the baseline; transform[5] is
      // y in pdf-space (y goes UP). The viewport transform applies
      // scale + flip to get screen coords.
      const tx = pdfjs.Util.transform(viewport.transform, it.transform)
      const x = tx[4]
      // The transformed y is the BASELINE in screen-space; subtract
      // the font height to get the top of the glyph box.
      const fontH = Math.hypot(tx[2], tx[3]) || 12
      const y = tx[5] - fontH
      const w = it.width * Math.hypot(viewport.transform[0], viewport.transform[1])
      const h = fontH
      return { str: it.str, x, y, w, h }
    })

    // Sort top-to-bottom, then left-to-right within a line.
    projected.sort((a, b) => a.y - b.y || a.x - b.x)
    const lines: { text: string; box: { x: number; y: number; w: number; h: number } }[] = []
    for (const it of projected) {
      const last = lines[lines.length - 1]
      const sameLine = last && Math.abs(it.y - last.box.y) < Math.max(it.h, last.box.h) * 0.5
      if (sameLine && last) {
        // Merge with a single space if there isn't already trailing space.
        const sep = last.text.endsWith(' ') || it.str.startsWith(' ') ? '' : ' '
        last.text = last.text + sep + it.str
        // Expand the bounding box rightward + downward.
        const right = Math.max(last.box.x + last.box.w, it.x + it.w)
        last.box.w = right - last.box.x
        last.box.h = Math.max(last.box.h, it.h)
      } else {
        lines.push({ text: it.str, box: { x: it.x, y: it.y, w: it.w, h: it.h } })
      }
    }
    // Drop lines that became empty after collapsing whitespace.
    const cleaned = lines.filter((l) => l.text.trim().length > 0)
    if (cleaned.length === 0) return undefined
    return { lines: cleaned }
  } catch {
    return undefined
  }
}

/**
 * Stream a PDF's pages one at a time. Kept for callers that don't
 * need the up-front page count.
 */
export async function* rasterizePdfStream(
  source: ArrayBuffer | Uint8Array,
  opts: RasterizeOptions = {},
): AsyncGenerator<RasterizedPage, { totalPages: number }> {
  const targetWidth = opts.pageWidthPx ?? 1200
  const pdf = await pdfjs.getDocument({ data: source }).promise
  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i)
    const baseViewport = page.getViewport({ scale: 1.0 })
    const scale = targetWidth / baseViewport.width
    const viewport = page.getViewport({ scale })
    const canvas = document.createElement('canvas')
    canvas.width = Math.ceil(viewport.width)
    canvas.height = Math.ceil(viewport.height)
    const ctx = canvas.getContext('2d', { willReadFrequently: true })
    if (!ctx) throw new Error('canvas 2D context unavailable')
    ctx.fillStyle = '#ffffff'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    await page.render({ canvasContext: ctx, viewport }).promise
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
    yield { pageNumber: i, canvas, imageData }
    // Help pdf.js release per-page memory between iterations.
    page.cleanup()
  }
  return { totalPages: pdf.numPages }
}

/**
 * Eager helper — collects the stream into an array. Kept for callers
 * that don't care about progressive display.
 */
export async function rasterizePdf(
  source: ArrayBuffer | Uint8Array,
  opts: RasterizeOptions = {},
): Promise<RasterizedPage[]> {
  const pages: RasterizedPage[] = []
  for await (const page of rasterizePdfStream(source, opts)) pages.push(page)
  return pages
}

/**
 * Render a single image file (PNG/JPG) into the same RasterizedPage shape
 * so the rest of the pipeline doesn't care whether the source was a PDF.
 */
export async function rasterizeImage(file: File): Promise<RasterizedPage> {
  const bitmap = await createImageBitmap(file)
  const canvas = document.createElement('canvas')
  canvas.width = bitmap.width
  canvas.height = bitmap.height
  const ctx = canvas.getContext('2d', { willReadFrequently: true })
  if (!ctx) throw new Error('canvas 2D context unavailable')
  ctx.drawImage(bitmap, 0, 0)
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
  bitmap.close()
  return { pageNumber: 1, canvas, imageData }
}
