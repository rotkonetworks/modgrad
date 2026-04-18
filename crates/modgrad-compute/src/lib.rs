//! Compute primitives for the modgrad ML SDK.
//!
//! Generic building blocks: Linear layers, activations, tensor ops.
//! No architecture-specific or runtime-specific code.

pub mod neuron;
pub mod ops;
pub mod tensor;
pub mod backend;
pub mod kv_buffer;
pub mod tensor_device;

/// Allocate a VRAM `GpuBuffer` of `bytes` bytes via the device singleton.
/// Returns `None` if GPU is unavailable. Exposed so `tensor_device::VramTensor`
/// can allocate without taking a direct modgrad-device visibility leak.
#[doc(hidden)]
pub fn alloc_device_vram(bytes: u64) -> Option<modgrad_device::kfd::memory::GpuBuffer> {
    modgrad_device::kfd::accel::alloc_vram(bytes)
}
/// Compute L2 gradient norm over multiple slices, GPU-accelerated when available.
pub fn grad_norm(slices: &[&[f32]]) -> f32 {
    let total_len: usize = slices.iter().map(|s| s.len()).sum();
    if total_len == 0 { return 0.0; }

    if total_len >= 1024 && neuron::gpu_enabled() {
        let mut buf = Vec::with_capacity(total_len);
        for s in slices { buf.extend_from_slice(s); }
        if let Some(norm) = modgrad_device::kfd::accel::try_l2_norm(&buf) {
            return norm;
        }
    }

    let mut total_sq = 0.0f32;
    for s in slices { for x in *s { total_sq += x * x; } }
    total_sq.sqrt()
}

#[cfg(feature = "cuda")]
pub mod cuda_backend;
