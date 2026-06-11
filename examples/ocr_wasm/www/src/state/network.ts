// Live network-call monitor. Installed at app boot so the user can see
// — in their own dev tools and on the page — that document data never
// leaves their browser.
//
// Cross-origin calls split into two buckets:
//   - "trusted"  — origins we knowingly fetch model + runtime from
//                  (jsdelivr.net, github.com mirrors, cdn.rotko.net).
//                  These happen at startup and stay flat after that.
//   - "unknown"  — anything else. If this ever ticks up, something
//                  is exfiltrating; the badge turns red.
// Same-origin loads are ignored.

import { createSignal } from 'solid-js'

export interface NetCall { url: string; t: number; trusted: boolean }

const [calls, setCalls] = createSignal<NetCall[]>([])

export const externalCalls = calls
export const trustedCount = () => calls().reduce((a, c) => a + (c.trusted ? 1 : 0), 0)
export const unknownCount = () => calls().reduce((a, c) => a + (c.trusted ? 0 : 1), 0)

// Allowlist of origins we knowingly call. Editing here is the only
// way to add more, so a future origin appearing in the counter is
// always an audit signal.
const TRUSTED_HOSTS = [
  'cdn.jsdelivr.net',
  'raw.githubusercontent.com',
  'media.githubusercontent.com',
  'cdn.rotko.net',
]

function isSameOrigin(url: string): boolean {
  try { return new URL(url, location.href).origin === location.origin }
  catch { return true }
}

function isTrusted(url: string): boolean {
  try { return TRUSTED_HOSTS.includes(new URL(url, location.href).hostname) }
  catch { return false }
}

function record(url: string) {
  if (isSameOrigin(url)) return
  const trusted = isTrusted(url)
  setCalls((prev) => [...prev, { url, t: Date.now(), trusted }])
}

// Public entry for the worker pool to forward worker-originated calls
// into the same counter. Without this, the worker can fetch anywhere
// and the badge silently lies.
export function recordExternal(url: string) { record(url) }

let installed = false
export function installNetworkMonitor() {
  if (installed) return
  installed = true

  const originalFetch = window.fetch.bind(window)
  window.fetch = (input: RequestInfo | URL, init?: RequestInit) => {
    const url = input instanceof Request ? input.url : String(input)
    record(url)
    return originalFetch(input, init)
  }

  // Patch prototype directly — simpler than subclassing, and dodges
  // the two-overload signature on XHR.open.
  const originalOpen = XMLHttpRequest.prototype.open
  XMLHttpRequest.prototype.open = function (this: XMLHttpRequest, ...args: unknown[]) {
    record(String(args[1]))
    return (originalOpen as (...a: unknown[]) => void).apply(this, args)
  } as typeof XMLHttpRequest.prototype.open

  // sendBeacon is the other common silent-exfil channel.
  const originalBeacon = navigator.sendBeacon?.bind(navigator)
  if (originalBeacon) {
    navigator.sendBeacon = (url: string | URL, data?: BodyInit | null) => {
      record(String(url))
      return originalBeacon(url, data)
    }
  }
}
