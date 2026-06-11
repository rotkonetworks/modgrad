// Projection-profile line segmentation.
//
// For clean single-column printed pages: project a binarized image onto
// the y axis (count of dark pixels per row), threshold, and read off
// contiguous ranges of text rows as line bounding boxes. Cheap, robust
// for the demo target (PDF scans). Won't handle multi-column layouts,
// tables, or handwriting — those are explicit non-goals (see ocr-demo.md).

export interface LineBox {
  x: number
  y: number
  w: number
  h: number
}

export interface SegmentOptions {
  // Pixel value 0..1 below which a pixel counts as ink. 0.5 = midgray.
  inkThreshold?: number
  // Min dark-pixel fraction per row for the row to count as "text".
  // 0.01 is conservative enough to catch thin diacritics but ignores
  // 1-pixel speckle.
  rowDarkFraction?: number
  // Min line height in px. Drops cosmetic specks.
  minLineHeight?: number
  // Vertical padding around each detected line, px.
  pad?: number
}

const DEFAULT_OPTS: Required<SegmentOptions> = {
  inkThreshold: 0.55,
  rowDarkFraction: 0.012,
  minLineHeight: 6,
  pad: 2,
}

/**
 * Find text-line bounding boxes by projection profile.
 *
 * Algorithm:
 *   1. ImageData → per-row dark-pixel count (luma < threshold).
 *   2. Mark rows above `rowDarkFraction * width` as "text rows".
 *   3. Collapse consecutive text rows into ranges.
 *   4. Pad and emit as LineBox with x=0, w=width (full page width crop).
 */
export function segmentLines(img: ImageData, opts: SegmentOptions = {}): LineBox[] {
  const o = { ...DEFAULT_OPTS, ...opts }
  const { width: w, height: h, data } = img
  // Row dark count.
  const rowDark = new Uint32Array(h)
  for (let y = 0; y < h; y++) {
    const base = y * w * 4
    let darkInRow = 0
    for (let x = 0; x < w; x++) {
      const i = base + x * 4
      // Rec. 709 luma, normalize to 0..1.
      const luma = (0.2126 * data[i]! + 0.7152 * data[i + 1]! + 0.0722 * data[i + 2]!) / 255
      if (luma < o.inkThreshold) darkInRow++
    }
    rowDark[y] = darkInRow
  }
  const rowThreshold = Math.max(1, Math.floor(o.rowDarkFraction * w))
  const lines: LineBox[] = []
  let start = -1
  for (let y = 0; y < h; y++) {
    const isText = rowDark[y]! >= rowThreshold
    if (isText && start < 0) start = y
    if (!isText && start >= 0) {
      pushLine(lines, start, y, w, h, o)
      start = -1
    }
  }
  if (start >= 0) pushLine(lines, start, h, w, h, o)
  return lines
}

function pushLine(
  out: LineBox[],
  yStart: number,
  yEnd: number,
  pageW: number,
  pageH: number,
  o: Required<SegmentOptions>,
) {
  const height = yEnd - yStart
  if (height < o.minLineHeight) return
  const y = Math.max(0, yStart - o.pad)
  const yE = Math.min(pageH, yEnd + o.pad)
  out.push({ x: 0, y, w: pageW, h: yE - y })
}

/**
 * Crop a LineBox from a source canvas into a fresh canvas. Used to
 * extract per-line pixel buffers for the OCR call.
 */
export function cropLineCanvas(source: HTMLCanvasElement, box: LineBox): HTMLCanvasElement {
  const dst = document.createElement('canvas')
  dst.width = box.w
  dst.height = box.h
  const ctx = dst.getContext('2d')
  if (!ctx) throw new Error('canvas 2D context unavailable')
  ctx.drawImage(source, box.x, box.y, box.w, box.h, 0, 0, box.w, box.h)
  return dst
}
