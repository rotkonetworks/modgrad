import { For, Show, createSignal, createMemo } from 'solid-js'
import type { QueueItem } from '../state/queue'
import { queueStats, QueueActions } from '../state/queue'
import { downloadAllText } from '../lib/download'
import { QueueRow } from './QueueRow'

export function QueueList(props: {
  items: QueueItem[]
  selectedId: string | null
  onSelect: (id: string) => void
  onRemove: (id: string) => void
  onDownload: (item: QueueItem) => void
  onClearDone: () => void
  onRetry: (id: string) => void
}) {
  const s = () => queueStats()

  // Multi-select set for bulk actions. Clearing on every items-change
  // would lose the selection unnecessarily; instead drop stale ids
  // lazily on access.
  const [marks, setMarks] = createSignal<Set<string>>(new Set())
  const liveMarks = createMemo(() => {
    const live = new Set(props.items.map((i) => i.id))
    return new Set([...marks()].filter((id) => live.has(id)))
  })

  function toggleMark(id: string) {
    const m = new Set<string>(marks())
    if (m.has(id)) m.delete(id); else m.add(id)
    setMarks(m)
  }
  function clearMarks() { setMarks(new Set<string>()) }
  function markAll() { setMarks(new Set<string>(props.items.map((i) => i.id))) }

  const markedItems = () => props.items.filter((i) => liveMarks().has(i.id))

  function bulkRemove() {
    for (const id of liveMarks()) QueueActions.remove(id)
    clearMarks()
  }
  function bulkRetry() {
    for (const it of markedItems()) {
      if (it.status === 'error') props.onRetry(it.id)
    }
  }
  function bulkDownload() {
    downloadAllText(markedItems())
  }

  const errorCount = () => props.items.filter((i) => i.status === 'error').length

  return (
    <div class="flex flex-col h-full">
      {/* Section bar */}
      <div class="flr-section flex items-center gap-2">
        <span>Queue</span>
        <span class="opacity-70">·</span>
        <span class="opacity-90">{props.items.length}</span>
        <span class="flex-1" />
        <Show when={s().done > 0}>
          <button
            onClick={props.onClearDone}
            class="text-2xs uppercase tracking-wider opacity-80 hover:opacity-100 hover:underline"
            title="Remove finished documents from the queue"
          >clear done</button>
        </Show>
        <Show when={errorCount() > 0}>
          <button
            onClick={() => {
              for (const it of props.items) {
                if (it.status === 'error') props.onRetry(it.id)
              }
            }}
            class="text-2xs uppercase tracking-wider opacity-80 hover:opacity-100 hover:underline"
            title="Re-queue every errored item"
          >retry errors</button>
        </Show>
      </div>

      {/* Counts strip */}
      <div class="flr-sub flex items-center gap-3 flex-wrap">
        <Show when={s().processing > 0}>
          <span><span class="flr-k">working</span> <span class="flr-derived-y ml-1">{s().processing}</span></span>
        </Show>
        <Show when={s().pending > 0}>
          <span><span class="flr-k">pending</span> <span class="flr-derived ml-1">{s().pending}</span></span>
        </Show>
        <Show when={s().done > 0}>
          <span><span class="flr-k">done</span> <span class="flr-pr ml-1">{s().done}</span></span>
        </Show>
        <Show when={s().error > 0}>
          <span><span class="flr-k">error</span> <span class="text-err font-bold ml-1">{s().error}</span></span>
        </Show>
        <span class="flex-1" />
        <Show when={s().lines > 0}>
          <span class="flr-foot">{s().pages} pages · {s().lines} lines</span>
        </Show>
      </div>

      {/* Bulk toolbar */}
      <Show when={liveMarks().size > 0}>
        <div class="flr-sub bg-accent/10 border-b border-accent-bright flex items-center gap-2">
          <span class="text-2xs uppercase tracking-wider">
            <span class="flr-derived">{liveMarks().size}</span> selected
          </span>
          <button class="flr-btn" onClick={bulkDownload}>↓ .txt</button>
          <button class="flr-btn" onClick={bulkRetry}
            title="Retry errored items in the selection">↻ retry</button>
          <button class="flr-btn-danger" onClick={bulkRemove}>remove</button>
          <span class="flex-1" />
          <button
            class="text-2xs underline decoration-dotted text-muted hover:text-ink"
            onClick={clearMarks}
          >clear</button>
        </div>
      </Show>

      {/* Mark-all row when many items are present */}
      <Show when={props.items.length > 1 && liveMarks().size === 0}>
        <div class="px-3 py-1 border-b border-border flex items-center gap-2 text-2xs">
          <button
            class="underline decoration-dotted text-muted hover:text-ink"
            onClick={markAll}
          >select all</button>
        </div>
      </Show>

      <div class="flex-1 overflow-y-auto bg-panel">
        <For each={props.items}>
          {(item, i) => (
            <QueueRow
              item={item}
              index={i()}
              selected={item.id === props.selectedId}
              marked={liveMarks().has(item.id)}
              onSelect={() => props.onSelect(item.id)}
              onToggleMark={() => toggleMark(item.id)}
              onRemove={() => props.onRemove(item.id)}
              onDownload={() => props.onDownload(item)}
              onRetry={() => props.onRetry(item.id)}
            />
          )}
        </For>
      </div>
    </div>
  )
}
