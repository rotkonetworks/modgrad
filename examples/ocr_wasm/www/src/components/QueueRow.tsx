import { Show, createMemo } from 'solid-js'
import type { QueueItem } from '../state/queue'
import { itemStats } from '../state/queue'

function fmtSize(b: number): string {
  if (b < 1024) return `${b} B`
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`
  return `${(b / (1024 * 1024)).toFixed(1)} MB`
}
function fmtMs(ms: number): string {
  if (ms < 1000) return `${ms} ms`
  return `${(ms / 1000).toFixed(1)} s`
}

const BADGE: Record<QueueItem['status'], string> = {
  pending: 'flr-badge-pending',
  processing: 'flr-badge-processing',
  done: 'flr-badge-done',
  error: 'flr-badge-error',
  cancelled: 'flr-badge-pending',
}

export function QueueRow(props: {
  item: QueueItem
  selected: boolean
  marked: boolean
  index: number
  onSelect: () => void
  onToggleMark: () => void
  onRemove: () => void
  onDownload: () => void
  onRetry: () => void
}) {
  const stats = createMemo(() => itemStats(props.item))
  // Live elapsed while processing — refreshes by reading addedAt vs
  // current Date.now(). For a non-reactive ticking display we just
  // re-render on item changes (which happen via progress updates).
  const elapsedLabel = () => {
    if (props.item.status === 'processing' && props.item.startedAt) {
      return `${((Date.now() - props.item.startedAt) / 1000).toFixed(1)} s`
    }
    if (props.item.elapsedMs > 0) return fmtMs(props.item.elapsedMs)
    return null
  }
  const altRow = () => props.index % 2 === 1

  return (
    <div
      class={`group px-2.5 py-2 border-b border-border cursor-pointer transition-colors ${
        props.selected
          ? 'bg-accent/15 border-l-2 border-l-accent-bright'
          : altRow()
          ? 'bg-panel-alt hover:bg-panel'
          : 'bg-panel hover:bg-panel-alt'
      } ${props.marked ? 'ring-1 ring-accent-bright' : ''}`}
      onClick={props.onSelect}
    >
      <div class="flex items-center gap-2 mb-1">
        <input
          type="checkbox"
          checked={props.marked}
          onChange={props.onToggleMark}
          onClick={(e) => e.stopPropagation()}
          class="cursor-pointer"
          title="Select for bulk action"
        />
        <span class={BADGE[props.item.status]}>{props.item.status}</span>
        <span class="text-xs truncate flex-1" title={props.item.file.name}>
          {props.item.file.name}
        </span>
        <button
          onClick={(e) => { e.stopPropagation(); props.onRemove() }}
          class="opacity-0 group-hover:opacity-100 text-muted hover:text-err text-xs px-1"
          title="Remove"
        >✕</button>
      </div>

      <div class="flex items-center gap-2 text-2xs">
        <span class="flr-k">{fmtSize(props.item.fileSize)}</span>
        <Show when={stats().nPages > 0}>
          <span class="text-border">|</span>
          <span class="flr-k">{stats().nPages}p</span>
        </Show>
        <Show when={stats().nLines > 0}>
          <span class="text-border">|</span>
          <span class="flr-k">{stats().nLines} lines</span>
        </Show>
        <Show when={elapsedLabel()}>
          <span class="text-border">|</span>
          <span class="flr-derived-y">{elapsedLabel()}</span>
        </Show>
        <span class="flex-1" />
        <Show when={props.item.status === 'done'}>
          <button
            onClick={(e) => { e.stopPropagation(); props.onDownload() }}
            class="text-accent-bright hover:underline"
          >↓ .txt</button>
        </Show>
        <Show when={props.item.status === 'error'}>
          <button
            onClick={(e) => { e.stopPropagation(); props.onRetry() }}
            class="text-amber hover:underline"
          >↻ retry</button>
        </Show>
      </div>

      <Show when={props.item.status === 'processing'}>
        <div class="mt-2 h-1 bg-border">
          <div
            class="h-full bg-accent-bright transition-all"
            style={{ width: `${Math.max(2, props.item.progress * 100).toFixed(1)}%` }}
          />
        </div>
      </Show>

      <Show when={props.item.status === 'error' && props.item.error}>
        <div class="mt-1 text-2xs text-err break-words">{props.item.error}</div>
      </Show>
    </div>
  )
}
