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
pub mod graph;

pub use config::CtmConfig;
pub use weights::{CtmWeights, CtmState};
pub use train::{Ctm, CtmGradients, CtmCache, RegionBackwardResult, train_step, backward_from_activated};
pub use forward::ctm_forward;
