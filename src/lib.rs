//! modgrad — Modular Gradient SDK for building general intelligence.
//!
//! Two layers:
//!   - **SDK crates**: generic ML infrastructure anyone can use
//!   - **Runtimes**: specific models built on the SDK (isis is ours)

// ─── SDK re-exports ────────────────────────────────────────

// Core
pub use modgrad_traits as traits;
pub use modgrad_compute as compute;
pub use modgrad_compute::neuron;

// CTM (the main architecture)
pub use modgrad_ctm as ctm;
pub use modgrad_ctm::graph;

// Training
pub use modgrad_training::optim;

// Data
pub use modgrad_data::tokenize;
pub use modgrad_data::tokens;
pub use modgrad_data::stream;
// Codec
pub use modgrad_codec::vqvae;
pub use modgrad_codec::audio_codec;
pub use modgrad_codec::retina;
pub use modgrad_codec::cifar;

// Device
pub use modgrad_device::device;

// Persistence
pub use modgrad_persist::persist;

// IO
pub use modgrad_io::backend;
pub use modgrad_io::telemetry;

// ─── isis runtime re-exports ──────────────────────────────

pub use modgrad_runtime::curriculum;
pub use modgrad_runtime::regional; // re-export of modgrad_ctm::graph
pub use modgrad_runtime::nc_io;
pub use modgrad_runtime::nc_socket;
