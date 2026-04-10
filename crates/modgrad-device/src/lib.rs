//! GPU/hardware backends for the modgrad ML SDK.
//!
//! Device management: GPU mesh, CUDA, Vulkan, KFD.

pub mod device;
pub mod cuda;
pub mod gpu;
pub mod kfd;
