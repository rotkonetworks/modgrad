//! Continuous Thought Machine — faithful implementation of Sakana AI's CTM.
//!
//! Single neuron pool, U-Net synapse, multihead attention, sync readout.
//! Implements the Brain trait from modgrad-traits.
//!
//! Reference: arxiv 2505.05522

pub mod config;
pub mod synapse;
pub mod weights;
pub mod forward;
pub mod train;

pub use config::CtmConfig;
pub use weights::{CtmWeights, CtmState};
pub use train::{Ctm, CtmGradients, CtmCache, RegionBackwardResult, train_step, backward_from_activated};
pub use forward::ctm_forward;
