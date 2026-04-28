//! `BltCerebellum` — adapter that wraps a device-resident
//! [`BltModel`](crate::model::BltModel) (Byte Latent Transformer)
//! as a [`FrozenCerebellum`].
//!
//! ## What this provides
//!
//! Per-layer hidden-state cache for the BLT's **latent transformer**,
//! produced by running a byte sequence through the full encoder + latent
//! pipeline. The cortex reads these via `cerebellum_at_position` +
//! `CerebProjection` exactly the same way it reads from
//! `modgrad_ctm::qwen_cerebellum::QwenCerebellum`.
//!
//! ## What this explicitly does NOT provide
//!
//! - **Encoder / decoder layers are NOT exposed.** Only the latent
//!   transformer's per-layer hidden states make it into the cache. The
//!   encoder is a byte-level adapter that turns bytes into patch reps;
//!   the decoder is a byte-level adapter that turns patch reps back
//!   into byte logits. They're treated as opaque pre-/post-processors.
//!   This matches the BLT paper's framing: the latent IS the world
//!   model — encoder/decoder exist only because we operate at byte
//!   granularity instead of token granularity.
//! - **No `forward(input: &[f32])`.** Same rationale as
//!   `QwenCerebellum`: the input would have to be a byte id or an
//!   embedding, not a generic activation vector. The trait method
//!   panics; use `encode_context_layers`.
//! - **No backward pass.** This struct is FROZEN. The wrapped
//!   `BltModel` is consumed by value in `from_model`, so callers can't
//!   mutate weights via aliasing. To fine-tune the BLT, drop this
//!   cerebellum, train via `modgrad-blt`, rebuild.
//!
//! ## CRITICAL — patch-vs-byte semantics
//!
//! **The cache row count is the patch count, NOT the byte count.**
//!
//! BLT is a hierarchical model:
//!
//! ```text
//!   bytes [N]
//!     ↓ encoder (byte-level transformer + cross-attn pool)
//!   patch reps [P × patch_dim]   (P < N — typically P = N/4 to N/8)
//!     ↓ latent transformer (the world model)
//!   patch reps [P × patch_dim]
//!     ↓ decoder (cross-attn + byte-level transformer)
//!   byte logits [N × 256]
//! ```
//!
//! The latent only sees `P` patches. Per-layer hidden states therefore
//! exist only at patch positions, not byte positions. A
//! [`CerebellumCache`] returned from this adapter has
//! `n_positions == P`, with `P = boundaries.len() - 1 < N`.
//!
//! **Cortex consumers must attend to PATCH positions, not byte
//! positions.** This is the structural difference from
//! `modgrad_ctm::qwen_cerebellum::QwenCerebellum`, which has one cache
//! row per token. A connection that streams "one cerebellum slot per
//! input timestep" assuming token-level alignment will silently
//! mis-align if it's wired to a `BltCerebellum`.
//!
//! ## Layout contract
//!
//! Identical to `QwenCerebellum`:
//! `[n_layers × n_positions × hidden_dim]` flat, layer-major,
//! position-major. `n_positions` here is `P` (patch count) and
//! `hidden_dim` is `patch_dim`.
//!
//! Only available with `--features rocm` (depends on the resident BLT
//! stack in `modgrad-blt`).

#![cfg(feature = "rocm")]

use crate::model::{BltBackwardState, BltModel, BltScratch};
use modgrad_compute::backend::{GpuVec, ResidencyError};
use modgrad_device::backend::HipBatch;

use modgrad_traits::cerebellum::{CerebellumCache, FrozenCerebellum};

/// Frozen cerebellum backed by a device-resident BLT.
///
/// Wraps [`BltModel`] and exposes the **latent transformer's** per-layer
/// hidden states for a context window. The encoder, latent, and decoder
/// all stay on device for the lifetime of this struct — no PCIe per
/// call.
///
/// **Frozen.** `from_model` takes the BLT by value; the only way to
/// mutate weights is to drop and rebuild.
///
/// **Single-pass per `encode_context_layers` call.** The KV caches
/// inside the BLT (encoder byte, latent patch, decoder byte) are reset
/// at the start of every call by `BltModel::forward_for_backward`.
/// Building a longer context is the caller's job (concatenate bytes,
/// call once).
///
/// **Cache rows = patches.** See module docs.
pub struct BltCerebellum {
    /// Device-resident BLT. Owned by value — no aliasing.
    model: BltModel,
    /// Caller-owned scratch for the BLT forward path. Allocated once;
    /// reused across calls.
    scratch: BltScratch,
    /// Backward state — used here only for its per-(patch, layer)
    /// activation snapshots, NOT for actually running backward. We pay
    /// the alloc cost (kilobytes for tiny configs, low MB for
    /// production sizes) so we can read per-layer hidden states without
    /// modifying `BltModel`'s public API. See module docs.
    state: BltBackwardState,
    /// Maximum supported byte-context width. The cache produced will
    /// have `n_positions = P ≤ max_patches`.
    #[allow(dead_code)]
    max_n_bytes: usize,
}

impl BltCerebellum {
    /// Wrap a [`BltModel`] as a frozen cerebellum.
    ///
    /// The BLT is consumed by value — frozen by construction. Both the
    /// forward scratch and a [`BltBackwardState`] are allocated up
    /// front; together they own the per-(patch, layer) activation
    /// snapshots we read in `encode_context_layers`.
    ///
    /// `max_n_bytes` is recorded for diagnostics; the operative cap is
    /// `model.config.latent.max_patches` (the largest `P` the latent's
    /// KV cache can hold).
    pub fn from_model(model: BltModel) -> Result<Self, ResidencyError> {
        let scratch = BltScratch::new(&model.config)?;
        let state = BltBackwardState::new(&model)?;
        let max_n_bytes = model.config.encoder.max_seq_len;
        Ok(Self { model, scratch, state, max_n_bytes })
    }

    /// Convert i64 token IDs to bytes (clamping into `0..=255`).
    ///
    /// `LanguageModel::forward_logits` for `BltModel` does the same
    /// clamp — keeping the conversion in one place (here) means
    /// downstream cortex code that already feeds `i64` token streams to
    /// other cerebellums (e.g. `QwenCerebellum`)
    /// can swap in a `BltCerebellum` without touching its caller.
    fn tokens_as_bytes(token_ids: &[i64]) -> Vec<u8> {
        token_ids.iter().map(|&t| t.clamp(0, 255) as u8).collect()
    }

    /// Default fixed-stride patcher matching the placeholder in
    /// `BltModel::forward_logits` (which mirrors `tiny_config`'s shape:
    /// 32 bytes → 8 patches of 4 bytes each).
    ///
    /// Stride is `ceil(n_bytes / 8)`, capped so the final boundary is
    /// always exactly `n_bytes`. The minimum stride is 1 — guards
    /// against `n_bytes == 0` producing an empty boundary list.
    ///
    /// This is a placeholder until leah's entropy patcher is integrated;
    /// real BLT inference will compute boundaries from byte-level
    /// next-byte entropy. For a frozen cerebellum the boundary scheme
    /// only affects how cache rows align with byte positions — the
    /// layer-blend semantics are unchanged either way.
    fn default_boundaries(n_bytes: usize) -> Vec<usize> {
        if n_bytes == 0 { return vec![0]; }
        let stride = ((n_bytes + 7) / 8).max(1);
        let mut boundaries = Vec::with_capacity(9);
        boundaries.push(0);
        let mut b = stride;
        while b < n_bytes {
            boundaries.push(b);
            b += stride;
        }
        boundaries.push(n_bytes);
        boundaries
    }

    /// Convenience accessor: maximum patches the underlying latent
    /// supports per call. The cache row count cannot exceed this.
    pub fn max_patches(&self) -> usize { self.model.config.latent.max_patches }
}

impl FrozenCerebellum for BltCerebellum {
    /// Latent patch dimension. **Not** byte_dim — the cache is
    /// patch-aligned (see module docs).
    fn hidden_dim(&self) -> usize { self.model.config.latent.patch_dim }

    /// Number of latent transformer layers. **Encoder / decoder layers
    /// are not counted here** — they don't appear in the cache.
    fn n_layers(&self) -> usize { self.model.latent.num_layers() }

    /// Run the BLT (encoder + latent) over `token_ids` interpreted as
    /// bytes, returning a `[n_layers × n_patches × patch_dim]` flat
    /// cache.
    ///
    /// ## Strategy
    ///
    /// We use [`BltModel::forward_for_backward`], **not**
    /// [`BltModel::forward`]. The "for_backward" variant is the only
    /// public BLT API that captures per-(patch, layer) activations:
    /// `state.latent_block_scratches[p][li]` snapshots layer `li`'s
    /// inputs at patch `p` (matches the resident transformer's
    /// `forward_for_backward` shape — see
    /// `crates/modgrad-transformer/src/resident.rs::TransformerBlock::forward_for_backward`).
    ///
    /// The `attn_input` field captures the input to layer `li` — which
    /// is the **output** of layer `li - 1` (or the encoder's patch reps
    /// for `li == 0`). So:
    ///
    /// - For layers `0..=L-2`: layer `li`'s output is read from
    ///   `block_scratches[p][li + 1].attn_input` (D2H copy).
    /// - For the last layer `L-1`: the output is the pre-final-norm
    ///   hidden state, which `forward_for_backward` already snapshots
    ///   to host in `state.latent_pre_norm_per_patch_host`.
    ///
    /// We never run the backward — we only read the activation
    /// snapshots that forward_for_backward populates as a side effect.
    /// This costs a `BltBackwardState` worth of memory but avoids
    /// modifying `BltModel`'s public API.
    ///
    /// ## Caveats
    ///
    /// - **Patch granularity.** `cache.n_positions` equals the number
    ///   of patches `P`, not the number of bytes. See module docs.
    /// - **Empty input.** Returns an empty cache (zero positions) for
    ///   `token_ids.is_empty()`. The byte-to-patch path can't run on
    ///   zero bytes.
    /// - **Length cap.** Sequences longer than
    ///   `min(model.config.encoder.max_seq_len, max_n_bytes)` will trip
    ///   debug asserts inside the BLT forward. Cap your context.
    fn encode_context_layers(&mut self, token_ids: &[i64]) -> CerebellumCache {
        let n_layers = self.model.latent.num_layers();
        let pd = self.model.config.latent.patch_dim;

        if token_ids.is_empty() {
            return CerebellumCache {
                hidden_states: Vec::new(),
                hidden_dim: pd,
                n_positions: 0,
                n_layers,
                modalities: None,
            };
        }

        let bytes = Self::tokens_as_bytes(token_ids);
        let n_bytes = bytes.len();
        debug_assert!(
            n_bytes <= self.model.config.encoder.max_seq_len,
            "BltCerebellum: byte sequence ({n_bytes}) exceeds encoder.max_seq_len ({})",
            self.model.config.encoder.max_seq_len,
        );

        let boundaries = Self::default_boundaries(n_bytes);
        let n_patches = boundaries.len().saturating_sub(1);
        debug_assert!(
            n_patches <= self.model.config.latent.max_patches,
            "BltCerebellum: patch count ({n_patches}) exceeds latent.max_patches ({})",
            self.model.config.latent.max_patches,
        );

        // Allocate the byte-logits sink — the BLT forward writes here
        // unconditionally; we never read it. Stays in scope until the
        // forward returns, then drops.
        let mut byte_logits = GpuVec::try_hip(n_bytes * 256)
            .expect("BltCerebellum: alloc byte_logits sink");

        let batch = HipBatch::new();
        self.model
            .forward_for_backward(
                &batch,
                &bytes,
                &boundaries,
                &mut self.scratch,
                &mut self.state,
                &mut byte_logits,
            )
            .expect("BltCerebellum: forward_for_backward");
        // One flush per encode — same cadence as QwenCerebellum.
        batch.flush().expect("BltCerebellum: flush");

        // Walk the per-(patch, layer) snapshots. Layout: layer-major,
        // then patch-major (matches `CerebellumCache::at(layer, pos)`).
        let mut hidden_states = vec![0.0f32; n_layers * n_patches * pd];
        let mut host_slab = vec![0.0f32; pd];

        for p in 0..n_patches {
            // Layers 0..L-2: output of layer `li` is the input to layer
            // `li + 1`, captured in `block_scratches[p][li+1].attn_input`.
            for li in 0..n_layers.saturating_sub(1) {
                let next_block = &self.state.latent_block_scratches[p][li + 1];
                next_block.attn_input.copy_to_host(&mut host_slab);
                let row_off = li * (n_patches * pd) + p * pd;
                hidden_states[row_off..row_off + pd]
                    .copy_from_slice(&host_slab);
            }
            // Last layer L-1: pre-final-norm hidden state — already
            // host-side in `latent_pre_norm_per_patch_host`.
            if n_layers > 0 {
                let li = n_layers - 1;
                let row_off = li * (n_patches * pd) + p * pd;
                let host_off = p * pd;
                hidden_states[row_off..row_off + pd].copy_from_slice(
                    &self.state.latent_pre_norm_per_patch_host
                        [host_off..host_off + pd],
                );
            }
        }

        CerebellumCache {
            hidden_states,
            hidden_dim: pd,
            n_positions: n_patches,
            n_layers,
            modalities: None,
        }
    }

    /// `forward` is meaningless for a BLT cerebellum — the input would
    /// have to be a byte id or a patch rep, not a generic activation
    /// vector. Use `encode_context_layers`.
    fn forward(&mut self, _input: &[f32]) -> Vec<f32> {
        panic!("BltCerebellum: tokens-as-floats meaningless; use encode_context_layers");
    }
}

// ─── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Construction + encode smoke test. Skipped if HIP runtime is
    /// unavailable or `MODGRAD_SKIP_HIP_TESTS` is set (CI without GPU).
    ///
    /// Mirrors `BltModel::tests::tiny_config()` shape (32 bytes → 8
    /// patches, byte_dim=32, patch_dim=64, lE=1, lL=2, lD=1). Same
    /// rationale as `QwenCerebellum`'s smoke test: rebuild the config
    /// inline because the BLT crate's `tiny_config()` is `pub(crate)`
    /// inside its tests module.
    #[test]
    fn blt_cerebellum_alloc_and_encode() {
        if std::env::var("MODGRAD_SKIP_HIP_TESTS").is_ok() {
            eprintln!("MODGRAD_SKIP_HIP_TESTS set, skipping");
            return;
        }
        if !modgrad_device::backend::rocm::ffi::runtime_available() {
            eprintln!("hip runtime unavailable, skipping");
            return;
        }

        use crate::decoder::LocalDecoderConfig;
        use crate::encoder::LocalEncoderConfig;
        use crate::model::{BltConfig, BltLatentConfig};

        let byte_dim = 32usize;
        let n_byte_heads = 4usize;
        let byte_head_dim = byte_dim / n_byte_heads;
        let patch_dim = 64usize;
        let n_patch_heads = 4usize;
        let patch_head_dim = patch_dim / n_patch_heads;
        let max_seq = 32usize;
        let max_patches = 16usize;

        let config = BltConfig {
            encoder: LocalEncoderConfig {
                n_layers: 1,
                byte_dim,
                patch_dim,
                n_heads: n_byte_heads,
                head_dim: byte_head_dim,
                mlp_dim: byte_dim * 2,
                norm_eps: 1e-5,
                rope_base: 10_000.0,
                max_seq_len: max_seq,
                ngram_min_n: 3,
                ngram_max_n: 5,
                ngram_vocab_per_n: 256,
            },
            latent: BltLatentConfig {
                n_layers: 2,
                patch_dim,
                n_heads: n_patch_heads,
                head_dim: patch_head_dim,
                mlp_dim: patch_dim * 2,
                norm_eps: 1e-5,
                rope_base: 10_000.0,
                max_patches,
            },
            decoder: LocalDecoderConfig {
                n_layers: 1,
                byte_dim,
                patch_dim,
                n_heads: n_byte_heads,
                head_dim: byte_head_dim,
                mlp_dim: byte_dim * 2,
                norm_eps: 1e-5,
                rope_base: 10_000.0,
                max_seq_len: max_seq,
            },
        };

        let model = BltModel::new(config).expect("BltModel::new");
        let mut cereb = BltCerebellum::from_model(model)
            .expect("BltCerebellum::from_model");

        // Trait surface checks.
        assert_eq!(cereb.hidden_dim(), 64, "patch_dim");
        assert_eq!(cereb.n_layers(), 2, "latent layers (encoder/decoder NOT counted)");
        assert_eq!(cereb.max_patches(), 16);

        // 32 bytes → 8 patches with the default fixed-stride patcher.
        let token_ids: Vec<i64> = (0..32).map(|i| i as i64).collect();
        let cache = cereb.encode_context_layers(&token_ids);

        assert_eq!(cache.n_layers, 2);
        assert_eq!(cache.n_positions, 8, "32 bytes / stride 4 = 8 patches");
        assert_eq!(cache.hidden_dim, 64);
        assert_eq!(cache.hidden_states.len(), 2 * 8 * 64);

        // Finite check — non-finite would mean a bad dispatch.
        for (i, &v) in cache.hidden_states.iter().enumerate() {
            assert!(v.is_finite(), "non-finite at offset {i}: {v}");
        }

        // Each layer must have at least one non-zero value at position
        // 0 — random init + a real forward should never collapse the
        // residual stream to all-zeros at any layer.
        for li in 0..cache.n_layers {
            let row = cache.at(li, 0);
            let any_nonzero = row.iter().any(|&v| v != 0.0);
            assert!(
                any_nonzero,
                "layer {li} pos 0 collapsed to all-zeros — bad forward"
            );
        }
    }
}
