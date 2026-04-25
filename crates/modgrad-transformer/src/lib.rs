//! Native Rust transformer forward pass.
//!
//! Replaces InferenceService (ONNX wrapper) with a composable pipeline
//! of TransformerBlocks, each implementing Filter. Design follows:
//!   - composable pure-function pipelines

pub mod dims;
pub mod config;
pub mod error;
pub mod norm;
pub mod rope;
pub mod attention;
pub mod kv_cache;
pub mod value_embed;
pub mod smear;
pub mod residual;
pub mod mlp;
pub mod block;
pub mod model;
pub mod weights;
pub mod builder;
pub mod tensor;
pub mod ops;
pub mod position;
pub mod optim;
pub mod offload;
pub mod train;
pub mod loss;
pub mod loop_lm;

// Re-exports
pub use config::GptConfig;
pub use dims::*;
pub use error::TransformerError;
pub use builder::TransformerBuilder;
pub use ops::TransformerOps;

// ─── Resident path (--features rocm) ────────────────────────
//
// Device-resident wrappers over the host transformer types. Kept in
// separate modules so the host crate compiles on every host (no
// `rocm` link dependencies pulled in by default), and the resident
// path opts in by enabling the `rocm` feature.

#[cfg(feature = "rocm")]
pub mod kv_cache_resident;
#[cfg(feature = "rocm")]
pub mod resident;
#[cfg(feature = "rocm")]
pub use kv_cache_resident::KvCacheResident;
#[cfg(feature = "rocm")]
pub use resident::{
    AttentionResident, GptModelResident, SwigluResident, TransformerBlockResident,
};
