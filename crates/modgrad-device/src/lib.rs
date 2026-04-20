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
