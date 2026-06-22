//! GPU/hardware backends for the modgrad ML SDK.
//!
//! Device management: GPU mesh, CUDA, Vulkan, KFD.
//!
//! New work (Phase 0 of the ROCm + `Op`/`Backend` refactor, see
//! `tasks/rocm-backend-refactor.md`) lives under `backend::`. Existing
//! modules stay untouched during Phase 0; they'll be ported to the
//! trait in Phase 2.

// These modules pull in unix libc (mmap/ioctl), GPU FFI, and threads
// that don't exist on wasm32-unknown-unknown. Inference reaches none of
// them — it goes through `backend::cpu`. Gate them out on wasm; native
// is unchanged.
#[cfg(not(target_arch = "wasm32"))]
pub mod device;
#[cfg(not(target_arch = "wasm32"))]
pub mod cuda;
#[cfg(not(target_arch = "wasm32"))]
pub mod gpu;
#[cfg(not(target_arch = "wasm32"))]
pub mod kfd;
pub mod backend;

// Threading prelude shim: rayon on native, serial fallback on wasm32.
pub mod rayon_shim;
// NOTE: the Gemma-4 model (rocm_gemma) moved to the `modgrad-gemma` crate — it's
// an application, not part of this general device SDK. The reusable primitives it
// uses (HIP backend, K-quant matvec kernels, GGUF loader) stay here.

// Shared lock for HIP-using tests across the workspace. Gated behind
// `test-utils` so non-test builds are unaffected; downstream crates
// enable it under their `[dev-dependencies]`. See `test_lock.rs` for
// rationale and usage.
#[cfg(feature = "test-utils")]
pub mod test_lock;
