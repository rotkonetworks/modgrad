# OCR demo — CTM + CTC line recognition, train native, ship wasm

**Design doc + task breakdown.**
Status: proposed (v2 — revised after learning the SDK and reading PaddleOCR
3.0 technical report).

Working document — edit in place. Git log is the change trail.

## What this is, what it isn't

**Is:** modgrad-CTM as the sequence model in a CRNN-style line recognizer.
Train on synthetic + Chars74K-style rendered/cropped lines. Ship the trained
weights as a wasm binary that a browser uses to OCR scanned PDFs.

**Isn't:** a PP-StructureV3 replacement. PaddleOCR's full pipeline (layout
detection, table/formula/chart/seal recognition, reading-order recovery,
markdown export) is a 10+ model orchestration with years of Baidu team-work
behind it. We pick one shape — single-column printed English line recognition
— and do it well.

## Why we picked this shape

After reading the SDK end-to-end (program.md, BRAIN_ARCHITECTURE, minictm,
lm_validate, pretrain_retina, modgrad-ctm/graph.rs) and the PP-OCRv5 tech
report:

1. **modgrad's CTM has a sequence axis we aren't using.** Each CTM region
   unrolls `ticks` inner steps. The public `regional_train_token` collapses
   that to a single classification per call. For recognition we want the
   per-tick logits exposed as a `[T × alphabet]` tensor.
2. **CTC is the standard answer to "no character-level boxes."** PP-OCRv5_rec
   ships SVTR-HGNet + CTC. Per-character classifiers need a segmenter;
   segmentation is the hard part. CTC sidesteps it.
3. **The CTM's tick axis is a natural CTC time axis.** No new sequence
   model needed — we already have one.
4. **CPU-only forward + greedy CTC decode is wasm-ready.** CTM default
   features pull only `matrixmultiply`, `rayon`, `bincode`, `half`, `wincode`
   — all wasm32 compatible. Inference path stays single-threaded; training
   stays parallel on native.

## The missing primitive

modgrad doesn't have CTC. Grepped the workspace — zero hits on `ctc` /
`connectionist`. This is the one thing we add before scaffolding the example.

**`modgrad-training::ctc`** — forward (negative-log-likelihood with blank,
log-space alpha pass) + backward (alpha-beta gradient w.r.t. logits) +
greedy decoder. Pure Rust, ~150 LOC. Reference: PyTorch `nn.CTCLoss`
internals or any CRNN reference implementation.

**`modgrad-ctm::graph::regional_forward_sequence`** — variant of
`regional_forward` that returns the per-tick output-region readouts instead
of collapsing to a single classification. The existing forward already
computes these internally; we just expose them.

These two together unlock OCR-style training. They're also useful primitives
beyond OCR (any seq2seq task with unknown alignment).

## Stack

| stage | crate | gradient path | notes |
|---|---|---|---|
| image → feature map | `modgrad-codec::retina` | frozen Hebbian (pretrained on STL-10) | use `pretrain_retina` checkpoint, no backward through retina in phase 0 |
| feature map → tick observations | trivial flatten/projection | trivial | one observation vector per CTM tick |
| sequence model | `modgrad-ctm` 8-region small + `regional_forward_sequence` (new) | yes | T = ticks ≈ 32 |
| loss | `modgrad-training::ctc` (new) | yes | alphabet size ~95 (printable ascii + blank) |
| inference decode | greedy CTC | n/a | beam search optional later |

Param budget: CTM 8-region small ≈ 187k. CTC loss adds zero params.
Final weight file: ≪ 1 MB after f16. Browser-friendly.

## Phases

### phase 0 — primitives + synthetic smoke

1. Implement `modgrad-training::ctc` with unit tests against a hand-computed
   reference on a 3-class, T=4 example.
2. Add `regional_forward_sequence` to `modgrad-ctm::graph`. Test that the
   collapse of the returned sequence reproduces `regional_forward`.
3. Write `examples/ocr_smoke/`:
   - `render.rs`: ascii line → 32×W grayscale via `fontdue`. Random length
     4–24, mild noise + ±2px translation. Zero data deps.
   - `main.rs`: retina (loaded checkpoint or fresh init) → flatten per
     horizontal strip → CTM with T=32 ticks → CTC loss vs string.
   - Mirror `lm_validate` for run shape: train curve, held-out greedy CER,
     PASS/WARN/FAIL gate.

Gate: held-out **character error rate < 5%** on the synthetic distribution.

### phase 1 — real data: Chars74K English printed lines (rendered)

Chars74K only ships single-character images, so use it as a font seed —
render lines using its character glyphs / fonts plus standard system fonts.
Then evaluate on real printed-line crops if a small clean test set is
available (IIIT5K word-level subset works as a sanity baseline).

Gate: case-insensitive word accuracy > 70% on the held-out split. (PaddleOCR
v5 is in the 95%+ range — we are not chasing SOTA, we are proving the loop
works on real images.)

### phase 2 — wasm32 + browser demo

- `cargo build --target wasm32-unknown-unknown` with default features only.
  Single-threaded inference path. If rayon's parallel iterators break the
  wasm build, replace with serial loops at inference (training stays
  parallel on native).
- `wasm-bindgen` wrapper exposing
  `classify_line(pixels: &[u8], w: u32, h: u32) -> String`.
- Browser harness:
  - `pdf.js` rasterizes page → `<canvas>` → `getImageData()`.
  - If the page already has a text layer (most modern PDFs do), use it
    directly — no OCR needed. The interesting demo is **scanned** PDFs.
  - Projection-profile line splitter (horizontal histogram of dark
    pixels → segments) for clean single-column printed scans.
  - For each line crop: wasm `classify_line` → string.
  - Overlay decoded text on the page.

### phase 3 (deferred) — anything that touches layout

Tables, formulas, multi-column, reading order, handwriting, multilingual.
Each of these is a project on its own. PaddleOCR ships a dedicated model
for each. Out of scope here so we don't paint into a corner thinking we'll
match PP-StructureV3 — we won't, and we shouldn't pretend to.

## Crate layout

```
crates/modgrad-training/
  src/
    ctc.rs           # NEW: forward, backward, greedy decode
    lib.rs           # re-export

crates/modgrad-ctm/
  src/
    graph.rs         # NEW: regional_forward_sequence (~50 LOC addition)

examples/ocr_smoke/
  Cargo.toml         # default-features only — wasm-ready
  src/
    main.rs          # train + eval loop
    render.rs        # synthetic line renderer (fontdue)
    data.rs          # phase 1: real dataset loader

examples/ocr_wasm/   # phase 2 — separate crate so wasm deps don't pollute
  Cargo.toml         # wasm-bindgen, no rayon, no rocm
  src/lib.rs         # classify_line wrapper
  www/               # static HTML harness with pdf.js
```

## Open questions

1. **Tick budget vs. line length.** CTC needs T ≥ |target| + collapse slack.
   Long lines need more ticks → more compute. Cap line width at render-time
   and pick T = ceil(width / stride). Concrete value: 32 ticks works for
   ≤16-char lines at the strides we'd use.
2. **Retina frozen or jointly trained?** Phase 0 uses frozen
   `pretrain_retina` STL-10 weights. Jointly training the retina with CTC
   gradient needs the V2/V4 backward path, which `pretrain_retina` doesn't
   exercise. Defer to phase 1 if frozen retina hits the gate.
3. **Alphabet.** Printable ascii is 95 chars + 1 blank = 96. Lowercase-only
   would be 27 (+ blank) and easier; the demo deserves the full ascii set.
4. **Where the tick observations come from.** Two options for going from a
   2D feature map to T per-tick observation vectors: (a) horizontal stripes
   averaged vertically (simple, 1D), (b) attention pooling per tick over
   the full feature map (more expressive, more compute). Start with (a).

## Hard non-goals

These are explicitly out of scope. Naming them here so the next person
doesn't try to creep them back in:

- Matching PP-OCRv5 / PP-StructureV3 accuracy or feature set.
- Handwriting, multilingual scripts, Chinese, formulas, tables.
- Layout detection / multi-column reading order.
- Beam search decoding, language-model rescoring.
- End-to-end training through the retina (phase 0 keeps it frozen).
