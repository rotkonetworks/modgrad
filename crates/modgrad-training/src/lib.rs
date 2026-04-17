//! Training and inference infrastructure.
//!
//! Training: optimizer, scheduler, checkpointing, metrics, dream/sleep.
//! Inference: generate, speculative decoding, sampling.

pub mod optim;
pub mod trainer;
pub mod dream;
pub mod inference;
pub mod checkpoint;
pub mod metrics;
pub mod grad_accum;
pub mod grad_ckpt;
