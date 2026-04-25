//! `LanguageModel` trait — a residency-aware face for autoregressive
//! transformer-style language models that the trainer / inference
//! orchestrator drives.
//!
//! ## Why this trait, not `Layer`
//!
//! `runtime::layer::Layer` is shaped for fixed-shape residency-streamed
//! sub-blocks: it takes `(x_dev, out_dev)` of known `in_dim()` /
//! `out_dim()`. A whole transformer language model does not fit that
//! mold — its forward signature is variable-length token decode:
//!
//! ```ignore
//! fn forward(&mut self,
//!     batch: &HipBatch,
//!     token_ids: &[i64],
//!     positions: &[usize],
//!     kv_cache: &mut KvCacheResident,
//!     logits_out: &mut GpuVec,        // [n_tokens × vocab_size]
//! );
//! ```
//!
//! The shape is upstream of the residency window — it's the model's
//! semantics, not its memory layout. So `LanguageModel` is a sibling
//! trait in `runtime`, not a `Layer` impl. Concrete model wrappers
//! (`GptModelResident`, future `LlamaResident`, …) implement it.
//!
//! ## Why `&mut self`
//!
//! The forward pass mutates the KV cache (passed in as a borrow), but
//! it also mutates per-call scratch state when caching activations for
//! a future backward — see `eve`'s slice for the activation-cache
//! design. We take `&mut self` here so a model that *does* keep a
//! per-step activation tape is not blocked.
//!
//! ## Why `Send` only
//!
//! Same rationale as `Layer`: `HipBatch` is `!Send`; the model wraps
//! HIP runtime state that does not cross threads. We keep `Send` so
//! whole-model handoff between threads (host-prep on one, dispatch on
//! another) is not blocked, but `Sync` would be a lie.
//!
//! ## What this trait does NOT cover
//!
//! - **Training-only state.** The trainer owns the optimizer + master
//!   weights; the `LanguageModel` is just the forward face. AdamW
//!   state lives in `LmTrainer`.
//! - **Embedding / unembedding details.** A model that ties
//!   embeddings or runs sparse MoE projections still has the same
//!   forward face from outside; whether it shares parameters with the
//!   LM head is its own concern.
//! - **Tokenization.** Tokens come in as `&[i64]`; how they got there
//!   is the caller's problem.

#[cfg(feature = "rocm")]
use modgrad_compute::backend::{GpuVec, ResidencyError};
#[cfg(feature = "rocm")]
use modgrad_device::backend::HipBatch;
#[cfg(feature = "rocm")]
use modgrad_transformer::{GptModelResident, KvCacheResident};

// ─── Trait ────────────────────────────────────────────────────

/// Residency-aware forward face for an autoregressive language model.
/// Implementors typically wrap a device-resident model
/// (`GptModelResident` is the canonical example) and forward this
/// trait's calls 1-1.
#[cfg(feature = "rocm")]
pub trait LanguageModel: Send {
    /// Run a forward pass over `token_ids` at `positions`, writing
    /// `[n_tokens × vocab_size()]` into `logits_out`. The optional
    /// KV cache is mutated to accumulate K/V slabs for the consumed
    /// tokens.
    ///
    /// `logits_out.len()` must equal `token_ids.len() * vocab_size()`.
    /// `positions.len()` must equal `token_ids.len()`.
    fn forward_logits(
        &mut self,
        batch: &HipBatch,
        token_ids: &[i64],
        positions: &[usize],
        kv_cache: Option<&mut KvCacheResident>,
        logits_out: &mut GpuVec,
    ) -> Result<(), ResidencyError>;

    /// Number of transformer layers / blocks. Shape introspection.
    fn n_layers(&self) -> usize;

    /// Hidden / model dimension (the residual stream width).
    fn d_model(&self) -> usize;

    /// Vocabulary size — output rows in the LM head, input rows in
    /// the embedding table.
    fn vocab_size(&self) -> usize;

    /// Ensure all weights / KV scratch buffers are resident in VRAM.
    /// Idempotent. After a successful return, `forward_logits` may be
    /// called without further upload work.
    ///
    /// `batch` is taken so impls that need to record dequant kernel
    /// dispatches (Q4_K → fp32 unpack, future bf16 → fp32 promote)
    /// are not blocked by the trait shape.
    fn ensure_resident(&mut self, batch: &HipBatch) -> Result<(), ResidencyError>;
}

// ─── GptModelResident impl ────────────────────────────────────

/// `GptModelResident` is already device-resident at construction
/// (`from_model` uploads everything). The `LanguageModel` impl is
/// therefore a thin pass-through.
///
/// **Why we require a non-`None` `kv_cache` for now.** The current
/// resident `forward` signature in `modgrad-transformer` takes a
/// `&mut KvCacheResident` (not `Option`) — every block dispatches
/// through it. Training a flat batch of tokens conceptually doesn't
/// need a cache (causal mask + parallel attention), but that path
/// does not yet exist as resident kernels (no resident causal-mask
/// matmul prefill). The pragmatic answer: the trainer allocates a
/// cache the size of `seq_len`, resets it between micro-batches,
/// and treats the forward as a "single-shot prefill via decode
/// loop." That's the wrapping `LmTrainer` does.
#[cfg(feature = "rocm")]
impl LanguageModel for GptModelResident {
    fn forward_logits(
        &mut self,
        batch: &HipBatch,
        token_ids: &[i64],
        positions: &[usize],
        kv_cache: Option<&mut KvCacheResident>,
        logits_out: &mut GpuVec,
    ) -> Result<(), ResidencyError> {
        // The current resident forward takes `&mut KvCacheResident`,
        // not `Option`. If the caller didn't supply one, the trainer
        // has bugged out — surface it loudly.
        let cache = kv_cache.ok_or(ResidencyError::WrongVariant {
            expected: "Some(&mut KvCacheResident) — GptModelResident::forward needs the cache",
            got: "None",
        })?;
        self.forward(batch, token_ids, positions, cache, logits_out)
    }

    fn n_layers(&self) -> usize { self.num_layers() }
    fn d_model(&self) -> usize { self.model_dim() }
    fn vocab_size(&self) -> usize { GptModelResident::vocab_size(self) }

    fn ensure_resident(&mut self, _batch: &HipBatch) -> Result<(), ResidencyError> {
        // GptModelResident is fully resident from `from_model`. There
        // is no streaming variant yet (slice scope: harlan's c9f35d9
        // is the all-resident path). Layer streaming for foundation-
        // class models lands as a separate slice with
        // `LinearResidentStreaming`-flavored blocks.
        Ok(())
    }
}
