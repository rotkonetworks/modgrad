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
pub mod ffn;
pub mod graph;
pub mod memory;
pub mod bio;
pub mod curriculum;
pub mod plural;
pub mod organism;
pub mod cerebellum;
/// Minimal frozen transformer loader for external LLMs (safetensors).
/// For trainable transformers, use modgrad-transformer crate instead.
pub mod frozen_transformer;
/// Red-team validation: attack primitives with corresponding defenses.
/// Every attack function has a defense counterpart in bio/ or plural.
/// This is penetration testing tooling, not weaponization.
pub mod monarch;

pub use config::CtmConfig;
pub use weights::{CtmWeights, CtmState};
pub use train::{Ctm, CtmGradients, CtmCache, RegionBackwardResult, train_step, backward_from_activated};
pub use forward::ctm_forward;
pub use loss::{CtmLoss, LastTickCE, ThinkingLoss, ImaginationLoss};
pub use graph::{RegionalBrain, RegionalCache};

/// Compute L2 gradient norm over multiple slices, GPU-accelerated when available.
pub fn grad_norm(slices: &[&[f32]]) -> f32 {
    modgrad_compute::grad_norm(slices)
}
