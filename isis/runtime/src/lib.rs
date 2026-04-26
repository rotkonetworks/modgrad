//! isis runtime — our brain built on the modgrad SDK.
//!
//! isis is one composition of the SDK. It provides:
//!   - 8-region brain presets (four_region, eight_region)
//!   - Curriculum-based staged training
//!   - Real-time audio/camera I/O
//!   - TCP debug socket for live inspection
//!
//! The core graph composition, tokens, NeuralComputer, and AdamW
//! live in the SDK (modgrad-ctm::graph). isis re-exports them.

// Re-export the SDK graph module as `regional` for backwards compat
pub use modgrad_ctm::graph as regional;

// isis-specific
pub mod curriculum;
pub mod training_data;
pub mod challenges;

// Real-time I/O (generic but lives here for now)
pub mod nc_io;
pub mod nc_socket;

// Actor model (generic)
pub mod actors;

// Organism-aware NeuralComputer (pain, memory, dream, plural, monarch)
pub mod plural_nc;

// ONNX-backed frozen cerebellum
#[cfg(feature = "onnx")]
pub mod onnx_cerebellum;

// Layer trait + LayerScheduler — orchestrates which weights live in
// VRAM at any moment when training a foundation model that does not
// fit. See `layer.rs` for design notes.
pub mod layer;

// LanguageModel trait — residency-aware forward face for autoregressive
// transformer LMs. Implemented by `GptModelResident`. See
// `language_model.rs`.
pub mod language_model;

// LM trainer — wraps a `LanguageModel` with the orchestration loop
// (forward → CE loss → [TODO eve: backward → AdamW]). See
// `lm_trainer.rs` for slice-scope notes.
pub mod lm_trainer;

// Sampler — autoregressive decode driver, generic over `LanguageModel`.
// See `sampler.rs`.
pub mod sampler;

/// Generate tokens from a NeuralComputer using the SDK inference runtime.
pub fn generate_nc(
    nc: &mut modgrad_ctm::graph::NeuralComputer,
    input_tokens: &[usize],
    sampler: &mut dyn modgrad_traits::Sampler,
    stop: &modgrad_training::inference::Stop,
) -> modgrad_training::inference::GenerateResult {
    let initial_logits = nc.observe(input_tokens);
    modgrad_training::inference::generate(
        &mut |token| nc.step(token),
        initial_logits,
        sampler,
        stop,
    )
}
