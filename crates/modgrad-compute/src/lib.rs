//! Compute primitives for the modgrad ML SDK.
//!
//! Generic building blocks: Linear layers, activations, tensor ops.
//! No architecture-specific or runtime-specific code.

pub mod neuron;
pub mod ops;
pub mod tensor;
pub mod backend;
#[cfg(feature = "cuda")]
pub mod cuda_backend;
