// OCR Online service worker.
//
// Goal: full offline operation after first visit. The privacy story —
// "your documents never leave the browser" — gets stronger when you
// can verify it by pulling the network cable.
//
// Strategy:
//   - App shell (HTML, CSS, app JS, worker chunks) — cache-first.
//   - Model + ORT wasm from public CDNs — cache-on-first-fetch,
//     network-first when cache is empty.
//   - All other navigations — network-first with shell fallback.

const VERSION = 'ocr-online-v1'
const SHELL_CACHE = VERSION + '-shell'
const RUNTIME_CACHE = VERSION + '-runtime'

const SHELL_URLS = [
  '/',
  '/index.html',
  '/embed.html',
  '/embed-demo.html',
  '/embed-client.js',
  '/manifest.webmanifest',
  '/robots.txt',
  '/sitemap.xml',
]

const TRUSTED_ORIGINS = [
  self.location.origin,
  'https://cdn.jsdelivr.net',
  'https://media.githubusercontent.com',
  'https://raw.githubusercontent.com',
]

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(SHELL_CACHE).then((cache) =>
      // Best-effort — if individual URLs 404 during dev, don't fail install.
      Promise.allSettled(SHELL_URLS.map((u) => cache.add(u))),
    ).then(() => self.skipWaiting()),
  )
})

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys
          .filter((k) => k.startsWith('ocr-online-') && k !== SHELL_CACHE && k !== RUNTIME_CACHE)
          .map((k) => caches.delete(k)),
      ),
    ).then(() => self.clients.claim()),
  )
})

self.addEventListener('fetch', (event) => {
  const req = event.request
  if (req.method !== 'GET') return

  const url = new URL(req.url)

  // Refuse to participate in requests to origins we don't know about.
  // The SW going silent here forces the network — if something is
  // exfiltrating, the request still goes out (we can't stop that),
  // but at least we don't accidentally cache it.
  if (!TRUSTED_ORIGINS.includes(url.origin)) return

  // App shell + hashed assets: cache-first, revalidate in background.
  if (url.origin === self.location.origin) {
    event.respondWith(cacheFirst(req))
    return
  }

  // Cross-origin (trusted): network-first, cache the response if OK.
  event.respondWith(networkFirstThenCache(req))
})

async function cacheFirst(req) {
  const cache = await caches.open(SHELL_CACHE)
  const cached = await cache.match(req, { ignoreSearch: false })
  if (cached) {
    // Refresh in the background so updates land on the next load.
    fetch(req).then((resp) => {
      if (resp.ok) cache.put(req, resp.clone())
    }).catch(() => { /* offline, fine */ })
    return cached
  }
  try {
    const resp = await fetch(req)
    if (resp.ok) cache.put(req, resp.clone())
    return resp
  } catch (e) {
    // Last resort: serve the SPA shell so the in-page router can deal.
    const shell = await cache.match('/index.html')
    if (shell && req.mode === 'navigate') return shell
    throw e
  }
}

async function networkFirstThenCache(req) {
  const cache = await caches.open(RUNTIME_CACHE)
  try {
    const resp = await fetch(req)
    // Opaque (no-cors) responses still cache, but we can't read status.
    if (resp.ok || resp.type === 'opaque') cache.put(req, resp.clone())
    return resp
  } catch (e) {
    const cached = await cache.match(req)
    if (cached) return cached
    throw e
  }
}
