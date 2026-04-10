//! IO, serialization, memory, and inference backends for the modgrad ML SDK.
//!
//! Pure data types, episodic memory, and model-agnostic inference abstractions.

pub mod types;
pub mod memory;
pub mod episode;
pub mod service;
pub mod config_file;
pub mod telemetry;
pub mod telemetry_transport;
pub mod dataloader;
pub mod control;
pub mod backend;
pub mod flatbuf;

#[cfg(feature = "onnx")]
pub mod inference;

#[cfg(feature = "gguf")]
pub mod gguf;
