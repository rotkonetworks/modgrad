# BLT Backward Pass — Architectural Overview

Reference doc for `crates/modgrad-blt`. Audience: a developer who knows
transformer training but is new to this codebase, sitting down to fix a
backward-pass bug. The aim is orientation, not a tutorial.

The BLT model is a Byte Latent Transformer (Pagnoni et al. 2024,
`https://arxiv.org/abs/2412.09871`): three stacked transformers
(LocalEncoder, Latent, LocalDecoder) connected by per-layer
cross-attention. The byte-level pieces are local; the global
"workhorse" model is the latent. For training, `BltModel` exposes
`forward_for_backward` and `backward` that operate at sequence
granularity (not the per-position trait shape `LmTrainer<M>` uses for
plain GPT models). All paths require `--features rocm`.

## Pipeline at a glance

```text
   bytes [N]                                          d/d(bytes)
       │                                                  ↑
       ▼                                                  │ scatter
   ┌──────────────┐                                ┌────────────────┐
   │ LocalEncoder │── byte_reps (last layer) ────→ │  LocalDecoder  │
   │              │                                │  cross-attn    │ seed
   │ embed +      │                                │  (Q=bytes,     │
   │ n-gram       │                                │   K/V=patches) │
   │ + block × lE │                                │                │
   │ + cross-attn │                                │ block × lD     │
   └──────────────┘                                │ + final norm   │
       │                                           │ + LM head      │
       │ patch_reps_E                              └────────────────┘
       ▼                                                  ↑ d_patch_reps
   ┌──────────────┐                                       │
   │   Latent     │── patch_reps_L ─────────────→  decoder K/V
   │ block × lL   │                                       
   │ + final norm │                                
   └──────────────┘                                
       │                                           
       ▼  byte_logits [N × 256]                    
   forward →  ;  backward ←
```

Forward proceeds top-to-bottom-to-top (encoder, latent, decoder).
Backward reverses: decoder → latent → encoder. Two upstream gradients
land on the encoder: one through the latent (`d_patch_reps_pre_latent`)
and one through the decoder cross-attn seed
(`d_seed_byte_reps`); both are summed onto the encoder's last-layer
output before its block backward starts. See
`crates/modgrad-blt/src/model.rs:469-619` for the orchestration.

## Components

| Module                                       | Responsibility                                                                                  |
|----------------------------------------------|--------------------------------------------------------------------------------------------------|
| `encoder.rs`                                 | `LocalEncoder` — byte embed + n-gram hash augmentation + lE byte-transformer layers + cross-attn |
| `decoder.rs`                                 | `LocalDecoder` — lD byte-transformer layers (cross-attn before each) + final norm + LM head      |
| `cross_attn.rs`                              | `CrossAttention` (one direction per instance), used by both encoder and decoder                  |
| `model.rs`                                   | `BltModel` (assembly), `BltBackwardState` (per-step buffers), latent forward/backward inlined    |
| `trainer.rs`                                 | `BltModelTrainer` — per-`Linear` AdamW with key namespace + local/global LR routing              |

The Latent is held as a `GptModelResident`
(`modgrad_transformer::resident`) but its embed and lm_head are
unused — patch reps go straight into `latent.blocks`. The encoder
provides the byte embedding; the decoder provides the LM head. Holding
the latent as `GptModelResident` keeps the byteification recipe (load
Qwen2.5 weights into the latent) trivial — see `byteify.rs`.

## State and scratch shapes

`BltBackwardState` (model.rs:629-665) is the per-train-step container.
Allocated once per `BltModelTrainer`; `zero_resident()` clears
accumulators between steps.

| Field                                | Purpose                                                              | Shape                                |
|--------------------------------------|----------------------------------------------------------------------|--------------------------------------|
| `encoder_grads`                      | Encoder weight grad accumulators                                     | per-layer attn/mlp + cross-attn + `[256 × byte_dim]` byte embed |
| `encoder_cache`                      | Per-(layer, byte) saved attn/MLP scratch + per-layer cross-attn cache | `[lE][max_seq_len]` block scratches  |
| `encoder_bwd_scratch`                | Per-byte working buffers for the encoder backward                    | `dy_per_byte[byte_dim]`, etc.        |
| `latent_attn_grads[li]`              | Per-layer attention weight grads                                     | `lL` entries                         |
| `latent_mlp_grads[li]`               | Per-layer SwiGLU MLP weight grads                                    | `lL` entries                         |
| `latent_block_scratches[p][li]`      | Per-(patch, layer) saved block activations                           | `[max_patches][lL]`                  |
| `latent_pre_norm_per_patch_host`     | Host snapshot of pre-final-norm hidden state per patch               | `[max_patches × patch_dim]`          |
| `d_latent_final_norm_weight`         | Latent final-norm scale grad accumulator                             | `[patch_dim]`                        |
| `decoder_grads`                      | Decoder weight grad accumulators (incl. `dweight_lm_head`, `dweight_final_norm`) | per-layer attn/mlp + cross-attn |
| `decoder_cache`                      | Per-(layer, byte) saved scratch + per-layer cross-attn caches + pre/post-norm host slabs | `[lD][max_seq_len]` |
| `d_patch_reps_post_latent`           | Inter-stage: d/d(decoder K/V) — written by decoder, consumed by latent | `[max_patches × patch_dim]`        |
| `d_patch_reps_pre_latent`            | Inter-stage: d/d(latent input) — written by latent, consumed by encoder | `[max_patches × patch_dim]`       |
| `d_seed_byte_reps`                   | Inter-stage: d/d(decoder seed = encoder last-layer byte_reps)        | `[max_seq_len × byte_dim]`           |

`BltScratch` (model.rs:779-797) is the matched per-call forward scratch:
encoder/decoder/latent scratches plus one `[max_patches × patch_dim]`
post-latent staging buffer.

## Forward path

Three stages, orchestrated by `BltModel::forward_for_backward`
(model.rs:367-462). The plain `forward` (model.rs:276-358) is the
inference variant — same shape, no activation snapshots.

### Stage 1 — encoder (encoder.rs:701-772)

1. **Byte embed + n-gram augmentation** (host). For each byte `b` at
   position `t`, the augmented embedding is
   `(byte_embed[b] + Σ_n ngram_n[hash_n(t)]) / (n_tables + 1)`. The
   normalisation factor matters for backward — see bug-3 below. The
   augmented slab and the byte ids are snapshotted into
   `encoder_cache.augmented_input_host` / `encoder_cache.byte_ids`.
2. **Per-byte transformer block forward** (lE layers, but in production
   `lE = 1`). Each layer's per-byte forward is `block.forward_for_backward`,
   which writes the per-byte attn/MLP saved activations into
   `encoder_cache.block_scratches[li][t]`. After the block stack, each
   layer's full byte_reps slab is snapshotted into
   `encoder_cache.layer_outputs_host[li]`.
3. **Cross-attn (encoder direction)**. Per layer, queries are patch
   reps (initialised by max-pooling byte reps within each patch — see
   `cross_attn.rs:914-915`), keys/values are byte reps; output overwrites
   `patch_reps_out`. Only the *last* layer's cross-attn output survives,
   because each layer overwrites the same buffer (encoder.rs:826-835).

### Stage 2 — latent (model.rs:404-445)

Per-patch loop. For each patch `p`:

1. Copy `patch_reps_view[p]` into the shared dense `scratch.latent.hidden`.
2. For each layer `li`, run `block.forward_for_backward`, which writes
   K/V at cache position `p` and snapshots activations into
   `state.latent_block_scratches[p][li]`.
3. Snapshot the pre-final-norm hidden state into the host slab
   `latent_pre_norm_per_patch_host[p]`.
4. Apply RMSNorm with the latent's final-norm scale; write back into
   `scratch.patch_reps[p]`.

After the loop, `scratch.patch_reps` holds the post-latent reps; the
latent KV cache holds positions `0..n_patches`.

### Stage 3 — decoder (decoder.rs:399-560)

1. **Seed**. Copy the encoder's last-layer byte_reps into the
   decoder's rolling `byte_reps` slab.
2. **Per-layer (cross-attn → block stack)**. For each `li`:
   - Snapshot the cross-attn Q input (current `byte_reps`) into
     `decoder_cache.byte_reps_q_per_layer[li]` (decoder.rs:452-456).
   - Cross-attn forward (decoder direction, Q=bytes, K/V=patches),
     populating `decoder_cache.cross_caches[li]`. Output overwrites
     `byte_reps`.
   - Per-byte block forward, snapshotting per-(layer, byte)
     activations into `decoder_cache.block_scratches[li][t]`.
3. **Final norm + LM head**. Per byte, RMSNorm over the byte_reps row
   then matvec into `byte_logits[t × 256]`. The pre/post-norm host
   slabs are captured for backward (decoder.rs:500-540).
4. The decoder's `patch_reps_kv` cache field also gets an owned
   device clone of `patch_reps`, since the cross-attn backward
   needs the values it was forwarded with.

## Backward path

Three stages, in reverse: decoder, latent, encoder. Orchestrated by
`BltModel::backward` (model.rs:469-619).

### Stage 1 — decoder backward (decoder.rs:573-920)

Inputs: `d_byte_logits` (caller-supplied, from cross-entropy on
`bytes[1..]`).

1. **Per-byte LM head + final-norm backward** (decoder.rs:614-694).
   For each byte `t`:
   - LM head `backward(post_norm[t], d_logits[t]) → d_post_norm[t]`,
     and `dweight_lm_head += d_logits ⊗ post_norm`,
     `dbias_lm_head += d_logits` (both via resident Add — bug-2 lesson,
     see below).
   - Host-side RMSNorm backward into `d_pre_norm[t]`, accumulating
     `dweight_final_norm` into a host slab.
   - Stage `d_pre_norm[t]` into the slab `d_byte_reps[t·byte_dim..]`.
   - At the end, fold the host `dweight_final_norm` into the device
     accumulator (single H2D + Add).
2. **Per-layer reverse walk** (decoder.rs:801-917). For `li` from
   `lD-1` down to 0:
   - **Block backward, per byte** (decoder.rs:807-863). For each byte
     `t`, restore the per-(layer, byte) saved activations from
     `cache.block_scratches[li][t]` into the shared `attn_scratch` /
     `mlp_scratch`, run `byte_layers[li].backward` into a *temporary*
     `tmp_attn_grads` / `tmp_mlp_grads` bundle, then accumulate into
     `grads.attn_grads[li]` / `grads.mlp_grads[li]` via `op_tensor_resident
     Add`. The per-byte `dx` is written back into the slab.
   - **Cross-attn backward** (decoder.rs:876-890). Produces `d_q_in`
     (additive into a per-layer `d_byte_reps_q_layer` slab) and
     `d_patch_reps` (additive into `d_patch_reps_post_latent`).
   - Hand `d_byte_reps_q_layer` to layer `li-1` as its block-output
     gradient. For `li == 0`, add into `d_seed_byte_reps_out`
     (decoder.rs:897-911).

### Stage 2 — latent backward (model.rs:520-603)

Inputs: `d_patch_reps_post_latent` (filled by decoder stage above).

For each patch `p` from `n_patches-1` down to 0:

1. Copy `d_patch_reps_post_latent[p]` into a dense step buffer.
2. **Final RMSNorm backward** (host). Read pre-norm slab from
   `latent_pre_norm_per_patch_host[p]`, run host arithmetic, accumulate
   `dweight_final_norm_host`.
3. **Block stack backward, in reverse layer order**. For each `li` from
   `lL-1` down to 0:
   - Restore per-(patch, layer) saved scratch via
     `restore_block_scratch_to_shared` (model.rs:1127).
   - `block.backward` with `recompute=false`, accumulating into
     `state.latent_attn_grads[li]` / `state.latent_mlp_grads[li]`.
4. Write the layer-0 dx back into `d_patch_reps_pre_latent[p]`.

After the loop, fold the host-accumulated `dweight_final_norm_host`
into `state.d_latent_final_norm_weight` (model.rs:591-603).

### Stage 3 — encoder backward (encoder.rs:792-942)

Inputs: `d_patch_reps_pre_latent` (latent output) and the
optional `d_seed_byte_reps` (decoder seed contribution).

1. **Last-layer cross-attn backward** (encoder.rs:836-851). Only the
   deepest encoder layer's cross-attn produces a gradient, because
   earlier layers' cross-attn outputs were overwritten in the forward
   path. Produces `d_byte_reps` (additive). For shallower layers the
   cross-attn contribution is skipped.
2. **Add the seed gradient** (encoder.rs:860-870). For the deepest
   layer, sum `d_seed_byte_reps` into `d_layer_input` (host slab) —
   both contribute to the same byte_reps activation.
3. **Per-byte block backward in reverse byte order** (encoder.rs:872-907).
   For each byte `t` in `(0..n_bytes).rev()`:
   - Run `byte_layers[li].backward` with `recompute=false`, driven off
     the per-(layer, byte) cache slot. Accumulates directly into
     `grads.attn_grads[li]` / `grads.mlp_grads[li]`. (NB: the encoder
     does *not* use the per-byte temp-grad accumulator pattern from
     the decoder — see bug-2 below for context.)
4. **Layer-0 byte embed scatter** (encoder.rs:921-939). Scatter
   per-byte-id rows into `grads.d_byte_embed`, multiplied by
   `1 / (n_tables + 1)` to match the forward's normalisation factor —
   see bug-3 below.

## Why per-(layer, byte) activation cache, not replay/recompute

The block backward needs the matched forward's `q_proj`, `k_proj`,
`v_proj`, `q_normed`, `k_normed`, `scores_tight`, `head_out`, plus the
SwiGLU saved activations. There are two ways to provide them:

- **Recompute**: re-run the forward inside backward, regenerating
  scratch on demand.
- **Cache**: snapshot every layer-byte pair during
  `forward_for_backward`, then D2D-restore the saved scratch into the
  shared `attn_scratch`/`mlp_scratch` buffers before each
  `block.backward`.

The encoder *originally* used recompute. It was bug-1 (see below) —
the recompute path uses `block_scratch_replay`, but that buffer is
never populated with this byte's forward state, so the recompute
started from a zeroed `attn_input` and produced all-zero outer-product
weight grads for every encoder block.

Both encoder and decoder now use the cache pattern. Per-(layer, byte)
slots live in `encoder_cache.block_scratches[li][t]` and
`decoder_cache.block_scratches[li][t]`. Memory cost is `lE × N × ~8 KB`
per byte for tiny configs, scaling linearly with lE/N. For the
production BLT (lE = 1, lD = 9, N ≤ 4096) this fits comfortably.

The latent uses the same pattern, indexed by `[patch][layer]` rather
than `[layer][byte]` (model.rs:687-703).

## Gradient correctness — the gradcheck story

The 41 BLT lib tests (`grep -rn '#\[test\]' crates/modgrad-blt/src/`)
are *shape* tests: loss is finite, monotonic over a smoke loop, dims
match. They do not catch sign errors, magnitude errors, or missing
chain-rule terms.

The gradcheck test
(`crates/modgrad-blt/tests/gradcheck.rs:486-618`) compares analytic
weight gradients to finite-difference reference values for one weight
per parameter group. It is `#[ignore]`'d so `cargo test` stays green;
reproduce with:

```
cargo test -p modgrad-blt --features rocm --test gradcheck -- \
  --ignored --test-threads=1 --nocapture
```

Twelve groups span the full BLT key namespace. Each prints
`num`, `analytic`, `rel_err`, and the L2 norm of the entire grad
buffer. Tolerance is `1e-2`. The diagnostic output is the artifact —
copy it into bug reports verbatim.

### Catastrophic-zero bugs (FIXED — commit dd63d4d)

These were buffer-wide zero gradients hidden by shape tests. Each
landed with a `FIXME(blt-bwd-bug-N) fix` comment marker at the fix site.

- **bug-1 — encoder.block recompute path** (encoder.rs:876-885 fix
  marker). `LocalEncoder::backward` ran `recompute=true` against a
  never-populated `block_scratch_replay`. Fix: switch to
  `recompute=false` against `cache.block_scratches[li][t]`, mirroring
  the decoder's pattern.
- **bug-2 — decoder.block.wv was exactly zero** (decoder.rs:732-788
  detailed root-cause). Two-fault compound:
  1. Shared `attn_scratch`/`mlp_scratch` held the last-iteration
     forward's state by backward time. Fix: D2D-restore per-(layer,
     byte) saved activations into the shared scratch before each
     `block.backward` (decoder.rs:812-819).
  2. `LinearResident::backward` *overwrites* `dweight_*` (uses
     `matmul_resident_nn` with `C=dweight`, not matmul_add). Per-byte
     calls clobbered each other; with the gradcheck's single-position
     loss (only row 0 of `d_byte_logits` is nonzero), bytes 1..31
     wrote zero, clobbering byte 0's contribution. Fix: per-byte
     temporary `tmp_attn_grads`/`tmp_mlp_grads`, accumulated into the
     layer total via `op_tensor_resident Add` after each byte
     (decoder.rs:851-856).
- **bug-3 — encoder byte_embed scatter scale** (encoder.rs:924-933 fix
  marker). The forward `embed_bytes` divides the augmented
  embedding by `(n_tables + 1)`, so
  `∂augmented/∂byte_embed = 1/(n_tables + 1)`. The scatter previously
  ignored this factor, producing an `(n_tables + 1)×` overshoot vs
  finite differences. Fix: apply the scale in
  `scatter_byte_embed_grad` (encoder.rs:989).

### Open bugs

- **bug-A — cross-step KV gradient dropped**.
  `AttentionResident::backward` (in `crates/modgrad-transformer/src/resident.rs:560-589`)
  documents: *"Treats prior KV-cache entries as constants (causal
  training-style detached cache) — only the current step's Q, K, V
  projection gradients are produced."* For an autoregressive
  single-position backward (which `GptModelResident` uses correctly,
  via `LmTrainer<GptModelResident>`), this is correct: each training
  example backproppes through one position with prior K/V treated as
  constants of the past.

  For full-sequence training — exactly what the BLT encoder and
  decoder do — this is wrong. When backward runs over byte `t`, the
  gradient flowing back to `H[s]` (s < t) via `dK[s] · ∂softmax/∂K`
  is dropped entirely. The accumulated `dW_k` (and via residual fold,
  `dW_q`, MLP grads, byte embed) systematically undercounts. Today
  this surfaces in the gradcheck as `encoder.block.0.wq` (99% rel_err),
  `encoder.block.0.gate` (87%), and `encoder.byte_embed` (96%).

  Fix shape (out-of-scope for this slice): a new entry point
  `AttentionResident::backward_full_sequence` that runs the per-step
  backward but routes the dK/dV contributions through a
  `KvGradAccumulator` rather than discarding them. The per-step
  entry stays for the autoregressive path — they are different
  semantics, not interchangeable.

- **bug-B — ~~rms_norm_backward_per_head unconditional~~ FIXED in
  commit a76ef75**. Was: `AttentionResident::backward` called
  `rms_norm_backward_per_head` on Q/K even when the matched forward
  skipped QK norm (`use_qk_norm = false`, BLT's config). Fix: gate
  the call on `self.use_qk_norm`
  (`crates/modgrad-transformer/src/resident.rs:801-818`). The
  gradcheck doc-comment status table at the top of `gradcheck.rs`
  predates this fix and is stale; the actual code is correct.

After bug-A is resolved upstream, all eight currently-failing
gradcheck rows should resolve, the marginals likely tighten under
1e-2, and the test gets un-ignored.

## How to extend

### Adding a new component (e.g. an extra LayerNorm in the decoder)

1. **Forward**: write the activation snapshot into the matched cache
   slot (`decoder_cache.block_scratches[li][t]` or a new field if the
   shape doesn't fit the existing block scratch). Update
   `BltBackwardState::new` to size the new field correctly.
2. **Backward**: read the snapshot, run the gradient kernel, write
   into a fresh `*_grad` buffer in the relevant `*Grads` struct.
3. **Gradient buffer**: extend `LocalEncoderGrads` /
   `LocalDecoderGrads` / `BltBackwardState` with the new field; wire
   it through `zero_resident()`.
4. **AdamW**: extend the param-key namespace in `BltModelTrainer`
   (`trainer.rs:599-643` for the canonical key list), and extend
   `weight_dev_for_key` / `grad_dev_for_key` (trainer.rs:1010-1058)
   with the new key.
5. **Gradcheck**: add the key to the `keys` array in
   `tests/gradcheck.rs:535-548` and verify it lands inside 1e-2.

Avoid editing `Connection` / `RegionalConfig` core fields for
mechanism-specific knobs (see `feedback_modularity` in user memory).
New mechanisms plug in via traits.

### Adding a new backward smoke test

1. Inline the dimensional config (parallel to `tiny_config()` in
   gradcheck.rs:164-202 or `tiny_blt_config` in trainer.rs).
2. Pick a deterministic seed and a deterministic input
   (`(0..32u8).collect()` is fine for tiny shapes;
   `lcg_bytes(seed, n)` for varied content).
3. Print a binned loss curve (e.g. mean of first 5, mean of last 5).
4. Assert `mean(last %) < mean(first %) * threshold`. Threshold of
   0.5–0.7 catches "loss is monotonic" without flaking on stochastic
   noise.

### Adding gradcheck coverage

1. New `#[ignore]`'d test in `tests/gradcheck.rs`. Two variants
   already exist: `blt_backward_multi_idx` (3 indices per key) and
   `blt_backward_multi_input` (3 inputs per idx).
2. Use existing helpers (`read_f32_at`, `write_f32_at`,
   `grad_dev_for_key`, `weight_dev_for_key`, `forward_loss`).
3. Print a per-key (or per-input or per-idx) table — the verbose
   output is the proof artifact.
4. Strict `1e-2` tolerance. Don't loosen. If a key flags, bisect over
   `EPS` (currently `1e-3`) before assuming it's a backward bug — fp32
   round-off in resident kernels can land near `1e-2` on small
   inputs.
