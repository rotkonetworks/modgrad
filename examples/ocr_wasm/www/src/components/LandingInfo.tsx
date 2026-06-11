// Landing-page content shown only in the empty state.
//
// Source of truth is `/content.json`. Edit there, not here. Keeps the
// prose out of the React tree so it can be cached, swapped, or
// localised without recompiling.

import { createResource, For, Show } from 'solid-js'

interface ContentSection {
  id: string
  heading: string
  items: string[]
}
interface ContentFaq {
  q: string
  a: string
}
interface Content {
  title: string
  tagline: string
  summary: string
  sections: ContentSection[]
  faq: ContentFaq[]
  links: { source: string; embedDemo: string }
}

export function LandingInfo() {
  const [content] = createResource<Content>(() =>
    fetch('/content.json', { credentials: 'omit' }).then((r) => {
      if (!r.ok) throw new Error('content.json: ' + r.status)
      return r.json() as Promise<Content>
    }),
  )

  return (
    <Show
      when={content()}
      fallback={
        <div class="mt-8 text-2xs text-muted font-mono">loading…</div>
      }
    >
      {(c) => <Render content={c()} />}
    </Show>
  )
}

function Render(props: { content: Content }) {
  const c = props.content
  return (
    <section class="mt-8 space-y-6 text-sm" aria-labelledby="seo-summary">
      <p id="seo-summary" class="text-ink leading-relaxed">{c.summary}</p>

      <For each={c.sections}>
        {(s) => (
          <div class="flr-controls p-4">
            <h3 class="flr-section -m-4 mb-3">{s.heading}</h3>
            <ul class="space-y-1.5 text-ink">
              <For each={s.items}>{(item) => <li>{item}</li>}</For>
            </ul>
          </div>
        )}
      </For>

      <div class="flr-controls p-4">
        <h3 class="flr-section -m-4 mb-3">FAQ</h3>
        <For each={c.faq}>
          {(f) => (
            <details class="mb-2">
              <summary class="cursor-pointer flr-derived">{f.q}</summary>
              <p class="mt-1 text-muted">{f.a}</p>
            </details>
          )}
        </For>
      </div>

      <p class="text-2xs text-muted text-center">
        Source: <a href={c.links.source} target="_blank" rel="noopener">{c.links.source}</a>
        {' · '}
        <a href={c.links.embedDemo}>embed API</a>
      </p>
    </section>
  )
}
