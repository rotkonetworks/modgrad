// Bundle results into a download. .txt for raw text, .json for
// structure (line bboxes, confidences, per-page geometry).

import type { QueueItem } from '../state/queue'

function safeName(s: string): string {
  // Path separators become underscores. Filesystem-reserved characters
  // (Windows: <>:"|?*) and ascii control chars are stripped. Unicode
  // letters and digits round-trip into the download.
  let out = s.replace(/[\\/]/g, '_')
  out = out.replace(/[<>:"|?*]/g, '')
  // Strip ascii control chars (0x00-0x1f, 0x7f) without using a
  // literal control-char regex (lint).
  let stripped = ''
  for (const ch of out) {
    const code = ch.codePointAt(0) ?? 0
    if (code >= 0x20 && code !== 0x7f) stripped += ch
  }
  stripped = stripped.replace(/\s+/g, '_').trim().replace(/^[._]+|[._]+$/g, '')
  return (stripped || 'transcript').slice(0, 160)
}

function trigger(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

function itemToText(item: QueueItem): string {
  // User edits in the full-text editor take precedence over the raw
  // OCR output — that's why the editor exists.
  if (item.editedText && item.editedText.trim().length > 0) {
    return item.editedText.endsWith('\n') ? item.editedText : item.editedText + '\n'
  }
  const out: string[] = []
  out.push(item.file.name)
  out.push('')
  let anyContent = false
  for (const p of item.pages) {
    const lines = p.texts.filter((t) => t.trim().length > 0)
    if (lines.length === 0) continue // M4: skip pages with no recognised text
    anyContent = true
    if (item.pages.length > 1) {
      out.push(`— page ${p.pageNumber} —`)
    }
    for (const t of lines) out.push(t)
    out.push('')
  }
  if (!anyContent) out.push('(no recognised text)')
  return out.join('\n').trimEnd() + '\n'
}

function itemToJson(item: QueueItem) {
  return {
    filename: item.file.name,
    fileSize: item.fileSize,
    elapsedMs: item.elapsedMs,
    pages: item.pages.map((p) => ({
      pageNumber: p.pageNumber,
      width: p.width,
      height: p.height,
      lines: p.lines.map((b, i) => ({
        bbox: { x: b.x, y: b.y, w: b.w, h: b.h },
        text: p.texts[i] ?? '',
        confidence: p.confidences[i] ?? 0,
      })),
    })),
  }
}

export function downloadItemText(item: QueueItem) {
  trigger(new Blob([itemToText(item)], { type: 'text/plain' }), safeName(item.file.name) + '.txt')
}

export function downloadItemJson(item: QueueItem) {
  trigger(
    new Blob([JSON.stringify(itemToJson(item), null, 2)], { type: 'application/json' }),
    safeName(item.file.name) + '.json',
  )
}

export function downloadAllText(items: QueueItem[]) {
  const merged = items
    .filter((it) => it.status === 'done')
    .map(itemToText)
    .join('\n\n---\n\n')
  if (!merged.trim()) return
  trigger(new Blob([merged], { type: 'text/plain' }), 'ocr_bundle.txt')
}

export function downloadAllJson(items: QueueItem[]) {
  const bundle = {
    extractedAt: new Date().toISOString(),
    items: items.filter((it) => it.status === 'done').map(itemToJson),
  }
  trigger(
    new Blob([JSON.stringify(bundle, null, 2)], { type: 'application/json' }),
    'ocr_bundle.json',
  )
}
