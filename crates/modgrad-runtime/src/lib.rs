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

// Real-time I/O (generic but lives here for now)
pub mod nc_io;
pub mod nc_socket;

// Actor model (generic)
pub mod actors;

