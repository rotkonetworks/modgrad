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

// Re-exports
pub use config::GptConfig;
pub use dims::*;
pub use error::TransformerError;
pub use builder::TransformerBuilder;
pub use ops::TransformerOps;
