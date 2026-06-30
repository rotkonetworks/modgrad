//! Continuous Thought Machine — faithful implementation of Sakana AI's CTM.
//!
//! Two levels:
//!   - Single CTM: one neuron pool, U-Net synapse, MHA, sync readout, full BPTT.
//!   - CTM Graph: compose N CTMs into a directed graph. Embedding table, AdamW,
//!     token space, NeuralComputer. This is the main entry point for building brains.
//!
//! Reference: arxiv 2505.05522

// Threading prelude shim: rayon on native, serial fallback on wasm32.
pub mod rayon_shim;

pub mod config;
/// Fixed 2D sinusoidal positional encoding (no trainable params). Added
/// in place to a flattened `[n_tokens × raw_dim]` spatial-token buffer so
/// attention can recover token grid position.
pub mod pos;
/// Value-Iteration-Network readout (arXiv:1602.02867). Convolutional
/// value-propagation over the 2D grid of V4 spatial tokens, read
/// ego-centrically at the agent's own cell — never pooled. Forward-only
/// alternative readout that makes local walls linearly decodable. See
/// module docs and `examples/mazes --vin-readout`.
pub mod vin;
pub mod planning;
pub mod synapse;
pub mod weights;
pub mod forward;
pub mod latent;
pub mod train;
pub mod loss;
/// Per-dispatch-type timing for the brain forward path. Activated by
/// env var `MODGRAD_PROFILE_DISPATCH=1`. Off by default — zero
/// overhead on the production path.
pub mod dispatch_profile;
// FFN architecture moved to the `modgrad-ffn` crate — parallel to
// `modgrad-transformer`. Depend on it directly for SwiGLU language models.
pub mod graph;
/// Non-hand-designed brain search space. Encodes per-region shape,
/// connectivity mask and globals as a sampleable architecture point;
/// the `eight_region_*` presets are reproducible as encoded points.
pub mod search_space;
/// Zero-cost trainability proxies (NAS Survey §4.4): forward-entropy
/// scores at init, no backward, no training. Used by random search
/// to rank architectures without running a single optimiser step.
pub mod proxies;
pub mod schedule;
pub mod modulator;
/// Re-export of `modgrad_memory` for backward compatibility.
/// New code should import from `modgrad_memory` directly.
pub use modgrad_memory as memory;
pub mod bio;
pub mod curriculum;
pub mod plural;
pub mod organism;
pub mod cerebellum;
/// Sibling-service wrapper that mounts a `FrozenCerebellum` outside
/// the regional iteration. The cerebellum-region in `eight_region_v2`
/// stays a small placeholder; the heavy LLM compute happens in
/// `CerebellumService` once per context window, and the cortex reads
/// projections of the per-layer hidden cache per-tick. See
/// `docs/BRAIN_ARCHITECTURE.md` §7 (option **(b)**).
pub mod cerebellum_service;
/// Minimal frozen transformer loader for external LLMs (safetensors).
/// Stays here because it impls the `cerebellum::FrozenCerebellum` trait;
/// splitting requires a bridge module across crates — deferred until needed.
/// For trainable transformers, use the rest of `modgrad-transformer`.
pub mod frozen_transformer;
/// Device-resident `FrozenCerebellum` adapter wrapping
/// `GptModelResident` (Qwen2.5-class transformer). Only available with
/// `--features rocm` because it depends on the resident transformer
/// stack in `modgrad-transformer`. See module docs for the layout
/// contract and the explicit "no `forward()`" panic.
#[cfg(feature = "rocm")]
pub mod qwen_cerebellum;
/// Red-team validation: attack primitives with corresponding defenses.
/// Every attack function has a defense counterpart in bio/ or plural.
/// This is penetration testing tooling, not weaponization.
pub mod monarch;
/// Brain → LLM logit modulation seam. The brain produces an output
/// vector; this module projects it to vocab dim and folds it into the
/// LLM's logits before sampling. Closes the architectural one-way
/// gap where `CerebellumService` lets brain read from Qwen but never
/// write back. See module docs.
pub mod logit_modulator;

pub use config::CtmConfig;
pub use weights::{CtmWeights, CtmState};
pub use train::{Ctm, CtmGradients, CtmCache, RegionBackwardResult, train_step, backward_from_activated};
pub use forward::{ctm_forward, CtmInput};
pub use loss::{CtmLoss, LastTickCE, ThinkingLoss, ImaginationLoss};
pub use graph::{RegionalBrain, RegionalCache};

/// Compute L2 gradient norm over multiple slices, GPU-accelerated when available.
pub fn grad_norm(slices: &[&[f32]]) -> f32 {
    modgrad_compute::grad_norm(slices)
}
