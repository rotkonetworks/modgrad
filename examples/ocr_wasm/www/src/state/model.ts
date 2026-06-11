// Model backend state.
//
// PP-OCRv5 mobile (detection + recognition) via ppu-paddle-ocr +
// onnxruntime-web. The package defaults are fine:
//   - onnxruntime wasm: cdn.jsdelivr.net (one-time, HTTP-cached)
//   - PP-OCRv5 det + rec + dict: media.githubusercontent.com mirror
//
// Both origins are on the trusted allowlist in `state/network.ts`. Any
// other origin appearing in the network monitor is the exfil signal.

import { createSignal } from 'solid-js'

export type Backend =
  | { kind: 'idle' }
  | { kind: 'ready'; source: string }
  | { kind: 'loading' }
  | { kind: 'error'; message: string }

export const DEFAULT_ORIGIN = 'github media + jsdelivr (built-in)'

const [backend, setBackend] = createSignal<Backend>({ kind: 'idle' })

export const modelBackend = backend
// Kept for backwards-compat with the model panel; just echoes the
// built-in default.
export const modelOrigin = () => DEFAULT_ORIGIN

export function markModelLoading() { setBackend({ kind: 'loading' }) }
export function markModelReady(source: string) { setBackend({ kind: 'ready', source }) }
export function markModelError(message: string) { setBackend({ kind: 'error', message }) }
