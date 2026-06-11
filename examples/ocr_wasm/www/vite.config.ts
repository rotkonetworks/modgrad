import { defineConfig } from 'vite'
import solid from 'vite-plugin-solid'
import unocss from 'unocss/vite'
import { resolve } from 'node:path'

export default defineConfig({
  plugins: [unocss(), solid()],
  optimizeDeps: {
    exclude: ['pdfjs-dist'],
  },
  server: { port: 5173 },
  build: {
    rollupOptions: {
      // The embed demo page imports '/embed-client.js' directly (the
      // public runtime URL). Tell rollup not to try to resolve it at
      // build time — it gets served at /embed-client.js from disk.
      external: ['/embed-client.js'],
      input: {
        main: resolve(__dirname, 'index.html'),
        embed: resolve(__dirname, 'embed.html'),
        'embed-demo': resolve(__dirname, 'embed-demo.html'),
        // Stand-alone ES module the consumer's page imports. `strict`
        // tells rollup not to tree-shake exports out — without this
        // the chunk comes out empty because no other entry imports it.
        'embed-client': resolve(__dirname, 'src/embed-client.ts'),
      },
      preserveEntrySignatures: 'strict',
      output: {
        // Keep `embed-client.js` at a predictable path so consumers
        // can pin a single URL.
        entryFileNames: (chunk) =>
          chunk.name === 'embed-client'
            ? 'embed-client.js'
            : 'assets/[name]-[hash].js',
      },
    },
  },
})
