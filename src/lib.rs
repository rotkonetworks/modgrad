//! modgrad — Modular Gradient SDK for building general intelligence.
//!
//! # Quick start
//! ```no_run
//! use modgrad::prelude::*;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     modgrad::init()?;                           // picks AMD / CUDA / CPU
//!
//!     let mut model = FfnWeights::new(FfnConfig::large(256));
//!     let mut opt = FfnAdamW::new(&model).with_lr(1e-4);
//!     let mut grads = FfnGradients::zeros(&model);
//!
//!     let bytes = std::fs::read("train.txt")?;
//!     let tokens: Vec<usize> = bytes.iter().map(|&b| b as usize).collect();
//!
//!     for chunk in tokens.chunks(128) {
//!         ffn_train_step(&mut model, &mut opt, &mut grads, chunk);
//!     }
//!     model.save("cerebellum.bin")?;
//!     Ok(())
//! }
//! ```
//!
//! # Two layers
//!   - **SDK crates**: generic ML infrastructure (traits, tensors, ops,
//!     kernels) you can compose.
//!   - **Runtimes**: specific models built on the SDK (isis is ours).
//!
//! # Platform
//! `modgrad::init()` detects `/dev/kfd` (AMD KFD) and enables the GPU
//! automatically. CPU-only is the fallback — the same code runs. No
//! build-time feature flags for platform.

// ─── SDK re-exports ────────────────────────────────────────

// Core
pub use modgrad_traits as traits;
pub use modgrad_compute as compute;
pub use modgrad_compute::neuron;

// CTM (the main architecture)
pub use modgrad_ctm as ctm;
pub use modgrad_ctm::graph;

// Training + Inference
pub use modgrad_training::optim;
pub use modgrad_training::inference;
pub use modgrad_training::dream;
pub use modgrad_training::trainer;

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

// ─── Runtime detection + init ─────────────────────────────

/// Which compute backend `init()` turned on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    /// CPU only — rayon parallelism + matrixmultiply SIMD GEMM.
    Cpu,
    /// AMD GPU via direct KFD. Currently only gfx1102 kernels shipped.
    AmdKfd,
    /// NVIDIA GPU via cudarc. Not yet wired; returned here for future use.
    NvidiaCuda,
}

/// Runtime config tweaks — arena size, CPU/GPU preference.
/// `Default` is "auto-detect GPU, 1 GiB VRAM arena".
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Try to enable GPU if available. Fall back to CPU silently otherwise.
    pub prefer_gpu: bool,
    /// VRAM arena size in MiB. Activations allocated from this arena bypass
    /// PCIe during dispatch. Ignored on CPU-only platforms.
    pub vram_arena_mb: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self { prefer_gpu: true, vram_arena_mb: 1024 }
    }
}

/// One-call runtime init. Detects /dev/kfd (AMD) and enables GPU automatically,
/// else CPU. Returns which platform was selected so callers can log it.
///
/// Call once at program start. Safe to skip if you don't care about GPU.
/// Equivalent to `init_with(RuntimeConfig::default())`.
pub fn init() -> Result<Platform, std::io::Error> {
    init_with(RuntimeConfig::default())
}

/// `init` with an explicit config. Use when you want a non-default arena
/// size or to force CPU.
pub fn init_with(cfg: RuntimeConfig) -> Result<Platform, std::io::Error> {
    if cfg.prefer_gpu && std::path::Path::new("/dev/kfd").exists() {
        modgrad_compute::neuron::enable_gpu();
        modgrad_device::kfd::accel::init_arena(cfg.vram_arena_mb);
        return Ok(Platform::AmdKfd);
    }
    // Fallback: CPU. Nothing to init.
    Ok(Platform::Cpu)
}

// ─── Prelude ──────────────────────────────────────────────

/// Re-exports the 90% of types a runtime builder wants. Use with
/// `use modgrad::prelude::*;` — one line instead of a dozen.
///
/// Keep this curated: only types that a model-builder / trainer actually
/// touches. Internals (device singletons, kernel dispatch primitives, etc.)
/// are deliberately NOT here; reach into the sub-crates for those.
pub mod prelude {
    // ─── Runtime setup ───
    pub use crate::{init, init_with, Platform, RuntimeConfig};

    // ─── Architectures ───
    pub use modgrad_ffn::{FfnConfig, FfnWeights, FfnBlock, FfnAdamW,
                          FfnGradients, FfnCache, ffn_train_step,
                          ffn_forward, ffn_backward, ffn_loss};
    pub use modgrad_ctm::{CtmConfig, CtmWeights, Ctm,
                          CtmCache, CtmGradients, train_step, ctm_forward};
    pub use modgrad_ctm::{RegionalBrain, RegionalCache};

    // ─── Tensors and typed ops (device-dispatched) ───
    pub use modgrad_compute::tensor_device::{
        CpuTensor, VramTensor, DeviceElem,
        Matmul, LayerNorm, SwiGLU, AddInPlace, Numeric, matmul_checked,
    };

    // ─── Core traits ───
    pub use modgrad_traits::{ParamIter, Vjp};

    // ─── Optimizer state (device-agnostic, AMD/CUDA-portable) ───
    pub use modgrad_compute::optimizer_state::OptimizerState;
    pub use modgrad_compute::make_optimizer_state;

    // ─── Memory primitives ───
    pub use modgrad_memory::episodic::{EpisodicMemory, EpisodicConfig};
}
