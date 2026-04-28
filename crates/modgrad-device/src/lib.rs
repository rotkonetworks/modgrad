//! GPU/hardware backends for the modgrad ML SDK.
//!
//! Device management: GPU mesh, CUDA, Vulkan, KFD.
//!
//! New work (Phase 0 of the ROCm + `Op`/`Backend` refactor, see
//! `tasks/rocm-backend-refactor.md`) lives under `backend::`. Existing
//! modules stay untouched during Phase 0; they'll be ported to the
//! trait in Phase 2.

pub mod device;
pub mod cuda;
pub mod gpu;
pub mod kfd;
pub mod backend;

// Shared lock for HIP-using tests across the workspace. Gated behind
// `test-utils` so non-test builds are unaffected; downstream crates
// enable it under their `[dev-dependencies]`. See `test_lock.rs` for
// rationale and usage.
#[cfg(feature = "test-utils")]
pub mod test_lock;
