import { render } from 'solid-js/web'
import 'virtual:uno.css'
import './index.css'
import { App } from './App'

const root = document.getElementById('root')
if (!root) throw new Error('#root missing')
render(() => <App />, root)

// Service worker. Skip on non-secure origins (other than localhost)
// where the API isn't available. Register either now (if `load` has
// already fired) or on the `load` event — the previous version only
// hooked `load`, which on a fast SPA could fire before this script
// even attached.
if ('serviceWorker' in navigator &&
    (location.protocol === 'https:' || location.hostname === 'localhost' || location.hostname === '127.0.0.1')) {
  const register = () => {
    void navigator.serviceWorker.register('/sw.js', { scope: '/' }).catch(() => {
      /* dev mode has stale SWs sometimes — non-fatal */
    })
  }
  if (document.readyState === 'complete') register()
  else window.addEventListener('load', register, { once: true })
}
