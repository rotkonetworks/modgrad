//! modgrad — Modular Gradient SDK for building general intelligence.
//!
//! Two-layer architecture:
//!   - **sdk crates**: model-agnostic ML infrastructure (traits, training, compute, data, codecs, devices, io)
//!   - **runtime crate**: isis, our default model built on the SDK
//!
//! Anyone can build a different runtime on the same SDK.

// ─── SDK re-exports ────────────────────────────────────────

// Traits
pub use modgrad_traits as traits;

// Compute
pub use modgrad_compute as compute;
pub use modgrad_compute::neuron;
pub use modgrad_compute::ops;

// Architectures
pub use modgrad_ctm::config as ctm_config;
pub use modgrad_ctm::synapse as ctm_synapse;
pub use modgrad_ctm::weights as ctm_weights;
pub use modgrad_ctm::forward as ctm_forward;
pub use modgrad_ctm::train as ctm_train;
pub use modgrad_transformer as transformer;

// Training
pub use modgrad_training::optim;
pub use modgrad_training::checkpoint;
pub use modgrad_training::grad_accum;
pub use modgrad_training::trainer;

// Data
pub use modgrad_data::tokenize;
pub use modgrad_data::tokens;
pub use modgrad_data::stream;

// Codec
pub use modgrad_codec::retina;
pub use modgrad_codec::vqvae;
pub use modgrad_codec::ngram_hash;
pub use modgrad_codec::cifar;

// Device
pub use modgrad_device::device;
pub use modgrad_device::cuda;
pub use modgrad_device::gpu;
pub use modgrad_device::kfd;

// Persistence
pub use modgrad_persist::persist;
pub use modgrad_persist::quantize;
pub use modgrad_persist::vocab;
pub use modgrad_persist::bpe;

// IO
pub use modgrad_io::types;
pub use modgrad_io::memory;
pub use modgrad_io::episode;
pub use modgrad_io::service;
pub use modgrad_io::backend;
pub use modgrad_io::flatbuf;
pub use modgrad_io::dataloader;
pub use modgrad_io::telemetry;
pub use modgrad_io::telemetry_transport;
pub use modgrad_io::config_file;
pub use modgrad_io::control;
#[cfg(feature = "onnx")]
pub use modgrad_io::inference;
#[cfg(feature = "gguf")]
pub use modgrad_io::gguf;

// ─── Runtime re-exports ────────────────────────────────────

// Core CTM
pub use modgrad_runtime::ctm;
pub use modgrad_runtime::config as rt_config;

// Organism
pub use modgrad_runtime::organism as tabula_rasa;
pub use modgrad_runtime::homeostasis;
pub use modgrad_runtime::autonomic;
pub use modgrad_runtime::neuvola;
pub use modgrad_runtime::pedagogy;
pub use modgrad_runtime::curriculum;
pub use modgrad_runtime::brain;
pub use modgrad_runtime::accel as gpu_accel;

// Evaluation
pub use modgrad_runtime::eval::tasks;
pub use modgrad_runtime::eval::challenge;
pub use modgrad_runtime::eval::linalg;
pub use modgrad_runtime::eval::sdp;
pub use modgrad_runtime::eval::metrics;
pub use modgrad_runtime::eval::accuracy;
pub use modgrad_runtime::eval::probe;
pub use modgrad_runtime::eval::consolidation;
pub use modgrad_runtime::eval::loss;

// Tuning
pub use modgrad_runtime::tuning::tuning;
pub use modgrad_runtime::tuning::puberty;

// Architecture compositions
pub use modgrad_runtime::cortex;
pub use modgrad_runtime::bridge;

// IO glue
pub use modgrad_runtime::filter;
pub use modgrad_runtime::daemon;

// ─── Convenience re-exports ────────────────────────────────

pub use modgrad_io::backend::Backend;
pub use modgrad_io::memory::MemoryBank;
pub use modgrad_io::service::{InferenceService, Request, Response};
pub use modgrad_runtime::filter::BrainPipeline;
