import { Show, createMemo } from 'solid-js'
import { trustedCount, unknownCount } from '../state/network'

/**
 * Live counter of cross-origin network calls since page load. Trusted
 * (known model + runtime sources) shown amber, informational. Unknown
 * (anything outside the allowlist) shown red — that's the actual
 * exfil signal the user should watch.
 */
export function NetworkMonitor() {
  const t = createMemo(trustedCount)
  const u = createMemo(unknownCount)
  return (
    <span class="inline-flex items-center gap-1 font-mono text-2xs">
      <Show when={u() === 0} fallback={
        <span
          class="inline-flex items-center gap-1 px-1.5 h-5 bg-err/15 text-err border border-err/40"
          title={`${u()} request(s) to origins outside the allowlist — likely exfil`}
        >
          <span class="w-1.5 h-1.5 bg-err animate-pulse" />
          {u()} unknown
        </span>
      }>
        <span
          class="inline-flex items-center gap-1 px-1.5 h-5 bg-ok/10 text-ok border border-ok/40"
          title="No requests to unknown origins since page load"
        >
          <span class="w-1.5 h-1.5 bg-ok" />
          0 exfil
        </span>
      </Show>
      <Show when={t() > 0}>
        <span
          class="inline-flex items-center gap-1 px-1.5 h-5 bg-warn/10 text-warn border border-warn/40"
          title="Model + runtime loads from allowlisted origins (jsdelivr, githubusercontent, cdn.rotko.net)"
        >
          {t()} model
        </span>
      </Show>
    </span>
  )
}
