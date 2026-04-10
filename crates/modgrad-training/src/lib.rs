//! Training infrastructure: optimizer, scheduler, checkpointing, metrics.

pub mod optim;
pub mod trainer;
pub mod checkpoint;
pub mod metrics;
pub mod grad_accum;
pub mod grad_ckpt;
