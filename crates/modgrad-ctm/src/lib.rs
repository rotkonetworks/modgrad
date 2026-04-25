//! Continuous Thought Machine — faithful implementation of Sakana AI's CTM.
//!
//! Two levels:
//!   - Single CTM: one neuron pool, U-Net synapse, MHA, sync readout, full BPTT.
//!   - CTM Graph: compose N CTMs into a directed graph. Embedding table, AdamW,
//!     token space, NeuralComputer. This is the main entry point for building brains.
//!
//! Reference: arxiv 2505.05522

pub mod config;
pub mod synapse;
pub mod weights;
pub mod forward;
pub mod train;
pub mod loss;
// FFN architecture moved to the `modgrad-ffn` crate — parallel to
// `modgrad-transformer`. Depend on it directly for SwiGLU language models.
pub mod graph;
pub mod schedule;
#[cfg(feature = "rocm")]
pub mod resident;
/// Re-export of `modgrad_memory` for backward compatibility.
/// New code should import from `modgrad_memory` directly.
pub use modgrad_memory as memory;
pub mod bio;
pub mod curriculum;
pub mod plural;
pub mod organism;
pub mod cerebellum;
/// Minimal frozen transformer loader for external LLMs (safetensors).
/// Stays here because it impls the `cerebellum::FrozenCerebellum` trait;
/// splitting requires a bridge module across crates — deferred until needed.
/// For trainable transformers, use the rest of `modgrad-transformer`.
pub mod frozen_transformer;
/// Red-team validation: attack primitives with corresponding defenses.
/// Every attack function has a defense counterpart in bio/ or plural.
/// This is penetration testing tooling, not weaponization.
pub mod monarch;

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
