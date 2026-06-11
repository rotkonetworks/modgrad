// Page-level OCR orchestrator.
//
// PP-OCRv5 mobile does detection + recognition end-to-end on a full
// page. We hand it the rasterized canvas; it returns line boxes +
// text + confidence. Projection-profile segmentation (segment.ts) is
// no longer in the inference path — kept around as a fallback / debug
// aid only.

import { pool, type OcrItem } from '../state/workers'

export interface OcrResult {
  line: { x: number; y: number; w: number; h: number }
  text: string
  confidence: number
}

export async function recognizePage(
  pageCanvas: HTMLCanvasElement,
): Promise<OcrResult[]> {
  const bitmap = await createImageBitmap(pageCanvas)
  const p = pool()
  const { items } = await p.recognize(bitmap)
  return items.map((it: OcrItem) => ({
    line: { x: it.box.x, y: it.box.y, w: it.box.width, h: it.box.height },
    text: it.text,
    confidence: it.confidence,
  }))
}
