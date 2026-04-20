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
pub mod optimizer_state;

/// Factory: build a device-backed `OptimizerState` sized for the given
/// per-tensor element counts. Returns `None` if no device backend is
/// available (then the caller uses a CPU AdamW loop instead).
///
/// Today this resolves to the KFD `VramMirror` on AMD; a future CUDA
/// impl slots in here without caller-side changes.
pub fn make_optimizer_state(sizes: Vec<usize>) -> Option<Box<dyn optimizer_state::OptimizerState>> {
    let mirror = modgrad_device::kfd::accel::make_vram_mirror(sizes)?;
    Some(Box::new(mirror))
}

/// Compute L2 gradient norm over multiple slices. Dispatches via the
/// backend registry above `GPU_THRESHOLD` elements — whichever backend
/// claims `ReduceL2Sq` for the shape runs (KFD/ROCm/CUDA/Vulkan/CPU).
/// Below the threshold, dispatch overhead dominates; host-loop the sum.
pub fn grad_norm(slices: &[&[f32]]) -> f32 {
    const GPU_THRESHOLD: usize = 1024;
    let total_len: usize = slices.iter().map(|s| s.len()).sum();
    if total_len == 0 { return 0.0; }

    if total_len < GPU_THRESHOLD {
        let mut total_sq = 0.0f32;
        for s in slices { for x in *s { total_sq += x * x; } }
        return total_sq.sqrt();
    }

    let mut buf = Vec::with_capacity(total_len);
    for s in slices { buf.extend_from_slice(s); }
    let mut out = [0.0f32];
    modgrad_device::backend::ops::reduce_l2_sq(&buf, &mut out);
    out[0].sqrt()
}

#[cfg(feature = "cuda")]
pub mod cuda_backend;
