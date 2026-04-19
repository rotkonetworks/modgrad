//! Training and inference infrastructure.
//!
//! Training: optimizer, scheduler, checkpointing, metrics, dream/sleep.
//! Inference: generate, speculative decoding, sampling.

pub mod optim;
pub mod trainer;
pub mod trainer_loop;
pub use trainer_loop::{TrainerLoop, TrainerConfig, StepReport, TrainerReport};
pub mod dream;
pub mod inference;
pub mod checkpoint;
pub mod checkpoint_bundle;
pub use checkpoint_bundle::{CheckpointBundle, BasicMeta, CheckpointError, CURRENT_SCHEMA, save_training_checkpoint};
pub mod autoresearch;
pub use autoresearch::AutoresearchSummary;
pub mod metrics;
pub mod grad_accum;
pub mod grad_ckpt;
