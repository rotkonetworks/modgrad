import { createSignal, createEffect, createMemo, For, Show, onCleanup } from 'solid-js'
import type { QueueItem, PageResult } from '../state/queue'
import { downloadItemText, downloadItemJson } from '../lib/download'

export function ItemDetail(props: { item: QueueItem }) {
  const [activePage, setActivePage] = createSignal(0)
  const [showOverlay, setShowOverlay] = createSignal(true)
  const [showFullText, setShowFullText] = createSignal(false)

  // Clamp to the valid range so removals / re-processing don't read
  // past the end of the pages array (which the previous non-null
  // assertion would have rendered as a crash).
  const safePageIdx = createMemo(() => {
    const n = props.item.pages.length
    if (n === 0) return -1
    return Math.max(0, Math.min(activePage(), n - 1))
  })
  const activePageObj = createMemo(() => {
    const i = safePageIdx()
    return i < 0 ? null : props.item.pages[i] ?? null
  })

  // Live elapsed during processing, ticks every 200 ms. Stops once the
  // item finishes — at which point we display item.elapsedMs directly.
  const [now, setNow] = createSignal(Date.now())
  createEffect(() => {
    if (props.item.status !== 'processing' || !props.item.startedAt) return
    const t = setInterval(() => setNow(Date.now()), 200)
    onCleanup(() => clearInterval(t))
  })
  const elapsedDisplay = () => {
    if (props.item.status === 'processing' && props.item.startedAt) {
      return `${((now() - props.item.startedAt) / 1000).toFixed(1)} s`
    }
    if (props.item.elapsedMs > 0) return `${(props.item.elapsedMs / 1000).toFixed(2)} s`
    return null
  }

  // Materialise the full transcript on demand. Used for the copy
  // button and the in-page textarea view.
  const fullText = () => itemToPlainText(props.item)

  return (
    <div class="flex flex-col h-full">
      <div class="flr-section flex items-center gap-3">
        <span class="truncate flex-1" title={props.item.file.name}>{props.item.file.name}</span>
        <Show when={props.item.pages.length > 1}>
          <span class="text-2xs opacity-80">
            page {(safePageIdx() + 1)} / {props.item.pages.length}
          </span>
        </Show>
      </div>

      <div class="flr-sub flex items-center gap-2 flex-wrap">
        <button onClick={() => setShowOverlay(!showOverlay())} class="flr-btn">
          {showOverlay() ? 'Hide boxes' : 'Show boxes'}
        </button>
        <button onClick={() => setShowFullText(!showFullText())} class="flr-btn">
          {showFullText() ? 'Pages view' : 'Full text view'}
        </button>
        <Show when={props.item.pages.length > 0}>
          <button
            onClick={() => void copyToClipboard(fullText())}
            class="flr-btn"
            title="Copy the entire transcript to the clipboard"
          >📋 copy all</button>
        </Show>
        <Show when={props.item.status === 'done'}>
          <button onClick={() => downloadItemText(props.item)} class="flr-btn">↓ .txt</button>
          <button onClick={() => downloadItemJson(props.item)} class="flr-btn">↓ .json</button>
        </Show>
        <span class="flex-1" />
        <Show when={elapsedDisplay()}>
          <span class="flr-foot">
            <span class="flr-k">elapsed</span>{' '}
            <span class="flr-derived-y">{elapsedDisplay()}</span>
          </span>
        </Show>
      </div>

      <Show
        when={props.item.pages.length > 0}
        fallback={
          <div class="flex-1 flex items-center justify-center text-muted text-sm bg-panel">
            <Show
              when={props.item.status === 'pending'}
              fallback={<span>processing…</span>}
            >
              <span>queued — waiting to start</span>
            </Show>
          </div>
        }
      >
        <Show when={!showFullText()} fallback={<FullTextView item={props.item} text={fullText()} />}>
          <div class="flex-1 flex flex-col lg:flex-row min-h-0">
            <Show when={props.item.pages.length > 1}>
              <aside class="lg:w-12 flex lg:flex-col gap-1 p-1.5 border-b lg:border-b-0 lg:border-r border-border-strong bg-panel overflow-auto">
                <For each={props.item.pages}>
                  {(p, i) => (
                    <button
                      onClick={() => setActivePage(i())}
                      class={`relative shrink-0 h-7 lg:w-9 text-2xs border ${
                        i() === safePageIdx()
                          ? 'bg-accent text-on-accent border-accent-bright'
                          : 'bg-panel-alt border-border-strong text-muted hover:border-key hover:text-ink'
                      }`}
                      title={`page ${i() + 1} · ocr ${p.ocrStatus}`}
                    >
                      {i() + 1}
                      <span
                        class={`absolute top-0.5 right-0.5 w-1.5 h-1.5 ${
                          p.ocrStatus === 'done'
                            ? 'bg-ok'
                            : p.ocrStatus === 'processing'
                            ? 'bg-amber animate-pulse'
                            : p.ocrStatus === 'error'
                            ? 'bg-err'
                            : 'bg-muted'
                        }`}
                      />
                    </button>
                  )}
                </For>
              </aside>
            </Show>

            <Show when={activePageObj()}>
              {(pg) => <PagePreview page={pg()} showOverlay={showOverlay()} />}
            </Show>
            <Show when={activePageObj()}>
              {(pg) => <Transcript page={pg()} />}
            </Show>
          </div>
        </Show>
      </Show>
    </div>
  )
}

// ─── Full-text view (HackMD-style split editor) ──────────────
//
// Left pane: editable monospace textarea. Edits are saved into the
// queue item's `editedText` field, which the download path prefers
// over the auto-generated OCR concatenation.
//
// Right pane: live preview of the same text, lightly formatted
// (paragraph breaks, monospace-vs-prose toggle, line wrapping). No
// markdown engine — the OCR output is mostly plain prose and we'd
// rather not import a 50 KB library for one feature.

import { QueueActions } from '../state/queue'

function FullTextView(props: { item: QueueItem; text: string }) {
  const [mode, setMode] = createSignal<'split' | 'edit' | 'preview'>('split')
  const initial = () => props.item.editedText ?? props.text
  const [draft, setDraft] = createSignal(initial())
  const [dirty, setDirty] = createSignal(false)
  const [findOpen, setFindOpen] = createSignal(false)
  const [findQuery, setFindQuery] = createSignal('')
  const [replaceWith, setReplaceWith] = createSignal('')
  const [caseSensitive, setCaseSensitive] = createSignal(false)
  let ta: HTMLTextAreaElement | undefined
  let findInput: HTMLInputElement | undefined

  const matchCount = () => {
    const q = findQuery()
    if (!q) return 0
    const flags = caseSensitive() ? 'g' : 'gi'
    try {
      return (draft().match(new RegExp(escapeRe(q), flags)) || []).length
    } catch { return 0 }
  }

  function replaceAll() {
    const q = findQuery()
    if (!q) return
    const flags = caseSensitive() ? 'g' : 'gi'
    const next = draft().replace(new RegExp(escapeRe(q), flags), replaceWith())
    if (next === draft()) return
    setDraft(next)
    setDirty(true)
  }

  // Transcript cleanups. Each recipe is a pure string transform applied
  // to the current draft. They're conservative — only patterns that are
  // near-universally OCR artifacts (not text the author wrote on purpose).
  const recipes: { id: string; label: string; help: string; fn: (s: string) => string }[] = [
    {
      id: 'dehyphenate',
      label: 'join hyphenated breaks',
      help: 'word-\\nword → wordword',
      fn: (s) => s.replace(/(\p{L})-\n(\p{L})/gu, '$1$2'),
    },
    {
      id: 'squeeze-ws',
      label: 'squeeze whitespace',
      help: 'collapse runs of spaces/tabs and blank lines',
      fn: (s) =>
        s
          .replace(/[ \t]+/g, ' ')
          .replace(/ +\n/g, '\n')
          .replace(/\n{3,}/g, '\n\n'),
    },
    {
      id: 'strip-trailing',
      label: 'trim line ends',
      help: 'remove trailing whitespace on every line',
      fn: (s) => s.replace(/[ \t]+$/gm, ''),
    },
  ]
  const [recipesOpen, setRecipesOpen] = createSignal(false)
  function applyRecipe(fn: (s: string) => string) {
    const next = fn(draft())
    setRecipesOpen(false)
    if (next === draft()) return
    setDraft(next)
    setDirty(true)
  }

  // Keep the editor in sync when the source item changes (different
  // file selected, new OCR results landed). Don't clobber unsaved
  // edits the user has made.
  createEffect(() => {
    const next = initial()
    if (!dirty()) setDraft(next)
  })

  function onInput(ev: InputEvent & { currentTarget: HTMLTextAreaElement }) {
    setDraft(ev.currentTarget.value)
    setDirty(true)
  }
  function save() {
    QueueActions.update(props.item.id, { editedText: draft() })
    setDirty(false)
  }
  function revert() {
    QueueActions.update(props.item.id, { editedText: undefined })
    setDraft(props.text)
    setDirty(false)
  }
  // Save on blur so users don't lose work if they click away.
  function onBlur() { if (dirty()) save() }

  // Keyboard: Ctrl/Cmd+S save, Ctrl/Cmd+F open find, Tab indent.
  function onKeyDown(ev: KeyboardEvent & { currentTarget: HTMLTextAreaElement }) {
    const cmd = ev.ctrlKey || ev.metaKey
    if (cmd && ev.key.toLowerCase() === 's') {
      ev.preventDefault(); save(); return
    }
    if (cmd && ev.key.toLowerCase() === 'f') {
      ev.preventDefault()
      setFindOpen(true)
      setTimeout(() => findInput?.focus(), 0)
      return
    }
    if (ev.key === 'Escape' && findOpen()) {
      ev.preventDefault(); setFindOpen(false); return
    }
    if (ev.key === 'Tab') {
      ev.preventDefault()
      const t = ev.currentTarget
      const start = t.selectionStart, end = t.selectionEnd
      const v = t.value
      t.value = v.slice(0, start) + '  ' + v.slice(end)
      t.selectionStart = t.selectionEnd = start + 2
      setDraft(t.value); setDirty(true)
    }
  }

  return (
    <div class="flex-1 flex flex-col min-h-0 bg-panel">
      <div class="flr-sub flex items-center gap-2 flex-wrap">
        <span class="flr-k">transcript</span>
        <span class="text-2xs text-muted">
          {countLines(draft())} lines · {countWords(draft())} words · {draft().length} chars
        </span>
        <Show when={props.item.editedText !== undefined}>
          <span class="text-2xs px-1.5 py-0.5 bg-amber/15 text-amber" title="Saved user edits. Downloads use this version.">edited</span>
        </Show>
        <Show when={dirty()}>
          <span class="text-2xs text-amber animate-pulse">● unsaved</span>
        </Show>
        <span class="flex-1" />
        <div class="inline-flex border border-border-strong">
          <button
            onClick={() => setMode('edit')}
            class={`px-2 h-6 text-2xs uppercase tracking-wider ${mode() === 'edit' ? 'bg-accent text-on-accent' : 'bg-panel text-ink hover:bg-panel-alt'}`}
          >edit</button>
          <button
            onClick={() => setMode('split')}
            class={`px-2 h-6 text-2xs uppercase tracking-wider border-l border-border-strong ${mode() === 'split' ? 'bg-accent text-on-accent' : 'bg-panel text-ink hover:bg-panel-alt'}`}
          >split</button>
          <button
            onClick={() => setMode('preview')}
            class={`px-2 h-6 text-2xs uppercase tracking-wider border-l border-border-strong ${mode() === 'preview' ? 'bg-accent text-on-accent' : 'bg-panel text-ink hover:bg-panel-alt'}`}
          >preview</button>
        </div>
        <button
          onClick={() => {
            setFindOpen(!findOpen())
            if (!findOpen()) setTimeout(() => findInput?.focus(), 0)
          }}
          class={findOpen() ? 'flr-btn-accent' : 'flr-btn'}
          title="Find and replace (Ctrl/⌘+F)"
        >find</button>
        <div class="relative">
          <button
            onClick={() => setRecipesOpen(!recipesOpen())}
            class={recipesOpen() ? 'flr-btn-accent' : 'flr-btn'}
            title="Apply a one-shot cleanup to the transcript"
          >tidy ▾</button>
          <Show when={recipesOpen()}>
            <div
              class="absolute right-0 top-full mt-1 z-10 min-w-64 bg-panel border border-border-strong shadow-lg"
              onMouseLeave={() => setRecipesOpen(false)}
            >
              <For each={recipes}>
                {(r) => (
                  <button
                    onClick={() => applyRecipe(r.fn)}
                    class="w-full text-left px-3 py-2 hover:bg-panel-alt block border-b border-border last:border-b-0"
                  >
                    <div class="text-xs text-ink">{r.label}</div>
                    <div class="text-2xs text-muted font-mono">{r.help}</div>
                  </button>
                )}
              </For>
            </div>
          </Show>
        </div>
        <Show when={dirty()}>
          <button onClick={save} class="flr-btn-accent">save</button>
        </Show>
        <Show when={props.item.editedText !== undefined}>
          <button onClick={revert} class="flr-btn-danger" title="Discard edits, restore the OCR output">revert</button>
        </Show>
        <button
          onClick={() => { ta?.select(); void copyToClipboard(draft()) }}
          class="flr-btn"
        >copy</button>
      </div>

      <Show when={findOpen()}>
        <div class="flr-sub flex items-center gap-2 bg-panel-alt">
          <span class="flr-k text-2xs uppercase tracking-wider">find</span>
          <input
            ref={(el) => (findInput = el)}
            type="text"
            class="font-mono text-xs flex-1 min-w-0"
            placeholder="text to find"
            value={findQuery()}
            onInput={(e) => setFindQuery(e.currentTarget.value)}
            onKeyDown={(e) => { if (e.key === 'Escape') setFindOpen(false) }}
          />
          <span class="flr-k text-2xs uppercase tracking-wider">replace</span>
          <input
            type="text"
            class="font-mono text-xs flex-1 min-w-0"
            placeholder="replacement"
            value={replaceWith()}
            onInput={(e) => setReplaceWith(e.currentTarget.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') replaceAll() }}
          />
          <label class="text-2xs flex items-center gap-1 cursor-pointer" title="Match case">
            <input
              type="checkbox"
              checked={caseSensitive()}
              onChange={(e) => setCaseSensitive(e.currentTarget.checked)}
            />
            <span>Aa</span>
          </label>
          <span class="text-2xs text-muted font-mono">
            {findQuery() ? `${matchCount()} matches` : '—'}
          </span>
          <button onClick={replaceAll} disabled={!findQuery() || matchCount() === 0} class="flr-btn-accent">
            replace all
          </button>
          <button onClick={() => setFindOpen(false)} class="flr-btn" title="Close (Esc)">×</button>
        </div>
      </Show>

      <div class={`flex-1 grid min-h-0 ${mode() === 'split' ? 'grid-cols-2' : 'grid-cols-1'}`}>
        <Show when={mode() !== 'preview'}>
          <textarea
            ref={(el) => (ta = el)}
            class="font-mono text-xs p-3 m-0 resize-none border-0 border-r border-border-strong"
            spellcheck={true}
            value={draft()}
            placeholder="OCR transcript appears here. Edit freely; downloads will use your edits."
            onInput={onInput}
            onBlur={onBlur}
            onKeyDown={onKeyDown}
          />
        </Show>
        <Show when={mode() !== 'edit'}>
          <Preview text={draft()} item={props.item} />
        </Show>
      </div>
    </div>
  )
}

function Preview(props: { text: string; item?: QueueItem }) {
  // Build a lookup of {line text → min confidence} so we can colour
  // each line in the preview by how confident the OCR was. Multiple
  // OCR lines with identical text get the min confidence (worst case).
  const lineConf = () => {
    const map = new Map<string, number>()
    if (!props.item) return map
    for (const p of props.item.pages) {
      for (let i = 0; i < p.texts.length; i++) {
        const t = p.texts[i]?.trim()
        if (!t) continue
        const c = p.confidences[i] ?? 0
        const prev = map.get(t)
        if (prev === undefined || c < prev) map.set(t, c)
      }
    }
    return map
  }

  const blocks = () => {
    const paras: string[] = []
    let buf: string[] = []
    for (const raw of props.text.split('\n')) {
      const line = raw.replace(/\s+$/, '')
      if (line === '') {
        if (buf.length) { paras.push(buf.join('\n')); buf = [] }
      } else {
        buf.push(line)
      }
    }
    if (buf.length) paras.push(buf.join('\n'))
    return paras
  }

  const colourFor = (line: string): string | null => {
    const c = lineConf().get(line.trim())
    if (c === undefined) return null
    if (c >= 0.95) return null            // confident — no decoration
    if (c >= 0.80) return 'underline decoration-warn decoration-dotted decoration-from-font'
    return 'underline decoration-err decoration-wavy decoration-from-font'
  }

  // Counts of warn/err lines so we only show the legend when it's useful.
  const lowConfCounts = () => {
    let warn = 0, err = 0
    for (const c of lineConf().values()) {
      if (c >= 0.95) continue
      if (c >= 0.80) warn++
      else err++
    }
    return { warn, err }
  }

  return (
    <div class="overflow-y-auto p-4 bg-bg text-ink leading-relaxed">
      <Show when={blocks().length > 0} fallback={
        <p class="text-muted italic">(transcript is empty)</p>
      }>
        <For each={blocks()}>
          {(p) => {
            const isHeading = /^(—\s*page\s*\d+\s*—|##.*|#\s+.*)$/i.test(p)
            if (isHeading) {
              return <h4 class="mt-4 mb-1 font-mono text-2xs uppercase tracking-wider text-key">{p.replace(/^#+\s*/, '')}</h4>
            }
            const lines = p.split('\n')
            return (
              <p class="my-3 whitespace-pre-wrap break-words">
                <For each={lines}>
                  {(line, i) => {
                    const cls = colourFor(line)
                    return (
                      <>
                        <Show when={i() > 0}><br /></Show>
                        <Show when={cls} fallback={<span>{line}</span>}>
                          <span class={cls!} title="Low OCR confidence — verify">{line}</span>
                        </Show>
                      </>
                    )
                  }}
                </For>
              </p>
            )
          }}
        </For>
        <Show when={lowConfCounts().warn + lowConfCounts().err > 0}>
          <div class="mt-6 pt-3 border-t border-border text-2xs text-muted flex items-center gap-4">
            <span class="uppercase tracking-wider">confidence</span>
            <Show when={lowConfCounts().warn > 0}>
              <span class="underline decoration-warn decoration-dotted decoration-from-font">
                {lowConfCounts().warn} line{lowConfCounts().warn === 1 ? '' : 's'} ≥ 80%
              </span>
            </Show>
            <Show when={lowConfCounts().err > 0}>
              <span class="underline decoration-err decoration-wavy decoration-from-font">
                {lowConfCounts().err} line{lowConfCounts().err === 1 ? '' : 's'} &lt; 80%
              </span>
            </Show>
          </div>
        </Show>
      </Show>
    </div>
  )
}

function PagePreview(props: { page: PageResult; showOverlay: boolean }) {
  let host: HTMLDivElement | undefined
  createEffect(() => {
    const el = host
    if (!el) return
    const canvas = props.page.canvas
    el.innerHTML = ''
    el.appendChild(canvas)
    canvas.style.width = '100%'
    canvas.style.height = 'auto'
    canvas.style.display = 'block'
  })
  return (
    <section class="flex-1 min-w-0 overflow-auto bg-bg border-r border-border-strong">
      <div class="relative">
        <div ref={(el) => (host = el)} class="relative" />
        <Show when={props.showOverlay && props.page.lines.length > 0}>
          <svg
            class="absolute inset-0 w-full h-full pointer-events-none"
            viewBox={`0 0 ${props.page.canvas.width} ${props.page.canvas.height}`}
            preserveAspectRatio="none"
          >
            <For each={props.page.lines}>
              {(box, i) => (
                <g>
                  <rect
                    x={box.x} y={box.y} width={box.w} height={box.h}
                    fill="none" stroke="#d4a020" stroke-width="2" opacity="0.85"
                  />
                  <text
                    x={box.x + 4} y={box.y + 14}
                    fill="#d4a020" font-size="14" font-family="ui-monospace, monospace"
                  >{i() + 1}</text>
                </g>
              )}
            </For>
          </svg>
        </Show>
      </div>
    </section>
  )
}

function Transcript(props: { page: PageResult }) {
  const pageText = () => props.page.texts.join('\n')
  return (
    <aside class="lg:w-96 border-t lg:border-t-0 border-border-strong bg-panel flex flex-col">
      <div class="flr-section flex items-center gap-2">
        <span>Transcript</span>
        <Show when={props.page.ocrStatus !== 'processing'}>
          <span class="opacity-70">·</span>
          <span>{props.page.texts.length}</span>
          <span class="opacity-70">lines</span>
        </Show>
        <Show when={props.page.source === 'pdf-text'}>
          <span
            class="text-2xs px-1.5 py-0.5 bg-ok/20 text-ok normal-case tracking-normal"
            title="Text was lifted from the PDF's embedded text layer — no OCR was needed."
          >pdf text</span>
        </Show>
        <Show when={props.page.source === 'ocr'}>
          <span
            class="text-2xs px-1.5 py-0.5 bg-warn/15 text-warn normal-case tracking-normal"
            title="Page had no embedded text layer; OCR was used."
          >ocr</span>
        </Show>
        <span class="flex-1" />
        <Show when={props.page.texts.length > 0}>
          <button
            onClick={() => void copyToClipboard(pageText())}
            class="text-2xs uppercase tracking-wider opacity-80 hover:opacity-100 hover:underline"
            title="Copy this page's lines"
          >copy</button>
        </Show>
        <Show when={props.page.ocrStatus === 'processing'}>
          <span class="text-amber text-2xs normal-case tracking-normal">processing…</span>
        </Show>
        <Show when={props.page.ocrStatus === 'error'}>
          <span class="text-err text-2xs normal-case tracking-normal">error</span>
        </Show>
      </div>
      <Show when={props.page.ocrStatus === 'processing' && props.page.texts.length === 0}>
        <div class="px-3 py-4 text-2xs text-muted font-mono">
          Detecting + recognising… results stream in as they finish.
        </div>
      </Show>
      <Show when={props.page.ocrStatus === 'error' && props.page.ocrError}>
        <div class="px-3 py-3 text-2xs text-err font-mono break-words">{props.page.ocrError}</div>
      </Show>
      <div class="flex-1 overflow-y-auto">
        <For each={props.page.texts}>
          {(text, i) => {
            const conf = () => props.page.confidences[i()] ?? 0
            const alt = () => i() % 2 === 1
            return (
              <div class={`px-3 py-1.5 border-b border-border flex gap-3 items-start ${alt() ? 'bg-panel-alt' : 'bg-panel'}`}>
                <span class="flr-k text-2xs w-6 text-right pt-0.5 font-mono">
                  {String(i() + 1).padStart(2, '0')}
                </span>
                <div class="flex-1 font-mono text-xs break-words leading-relaxed">
                  {text || <span class="text-muted italic">(empty)</span>}
                </div>
                <Show when={conf() > 0}>
                  <span
                    class={`text-2xs px-1.5 py-0.5 font-bold ${
                      conf() > 0.8 ? 'bg-ok/15 text-ok'
                      : conf() > 0.5 ? 'bg-warn/15 text-warn'
                      : 'bg-err/15 text-err'
                    }`}
                  >{(conf() * 100).toFixed(0)}%</span>
                </Show>
              </div>
            )
          }}
        </For>
      </div>
    </aside>
  )
}

// ─── Helpers ────────────────────────────────────────────────

function itemToPlainText(item: QueueItem): string {
  const out: string[] = []
  out.push(item.file.name)
  out.push('')
  for (const p of item.pages) {
    const pageText = p.texts.filter((t) => t.trim().length > 0).join('\n')
    if (!pageText) continue
    if (item.pages.length > 1) out.push(`— page ${p.pageNumber} —`)
    out.push(pageText)
    out.push('')
  }
  return out.join('\n').trimEnd() + '\n'
}

function countLines(s: string): number {
  if (!s) return 0
  // Don't count a trailing newline as an extra empty line.
  const stripped = s.endsWith('\n') ? s.slice(0, -1) : s
  return stripped.split('\n').length
}

function countWords(s: string): number {
  return (s.match(/\S+/g) ?? []).length
}

function escapeRe(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

async function copyToClipboard(text: string): Promise<void> {
  // Modern path
  if (navigator.clipboard?.writeText) {
    try { await navigator.clipboard.writeText(text); return } catch { /* fall through */ }
  }
  // Fallback for older browsers / non-secure contexts
  const ta = document.createElement('textarea')
  ta.value = text
  ta.style.position = 'fixed'
  ta.style.left = '-9999px'
  document.body.appendChild(ta)
  ta.select()
  try { document.execCommand('copy') } finally { ta.remove() }
}
