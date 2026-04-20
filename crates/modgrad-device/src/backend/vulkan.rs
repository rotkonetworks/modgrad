//! Vulkan compute backend — wraps `crate::gpu` (ash-based). Feature-
//! gated behind `--features gpu`. Cross-vendor fallback when neither
//! KFD (AMD gfx1102) nor CUDA (NVIDIA) is viable — runs on any GPU
//! with a Vulkan 1.3 compute-capable driver (Steam Deck, Intel Arc,
//! Apple Silicon via MoltenVK, etc.).
//!
//! Coverage matches existing shaders: `Matvec` only on first landing.
//! `try_superlinear` exists in `gpu.rs` too, but its argument shape
//! (no t_bias, no cache) doesn't cleanly align with `Op::SuperLinearFwd`;
//! deferred until the shader is extended or the Op trimmed.

use super::{Backend, BackendError, DeviceInfo, DeviceKind, Op};
#[cfg(feature = "gpu")]
use super::QuantKind;

/// Vulkan compute backend. Zero-size type when `gpu` feature is off.
pub struct VulkanBackend {
    #[cfg(feature = "gpu")]
    available: bool,
    #[cfg(not(feature = "gpu"))]
    _unused: (),
}

impl VulkanBackend {
    /// Probe the Vulkan runtime. Returns `None` when feature is disabled
    /// or no compatible GPU was found by `crate::gpu::init_global`.
    pub fn try_new() -> Option<Self> {
        #[cfg(feature = "gpu")]
        {
            // Ensure global init ran (idempotent).
            crate::gpu::init_global();
            if crate::gpu::available() {
                Some(Self { available: true })
            } else {
                None
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            None
        }
    }
}

impl Backend for VulkanBackend {
    fn name(&self) -> &'static str { "vulkan" }

    fn device_info(&self) -> DeviceInfo {
        DeviceInfo {
            kind: DeviceKind::Vulkan,
            name: "Vulkan compute".into(),
            total_mem_bytes: 0,  // filled in when gpu.rs exposes device props
            arch: None,
        }
    }

    fn supports(&self, op: &Op) -> bool {
        #[cfg(not(feature = "gpu"))]
        { let _ = op; return false; }
        #[cfg(feature = "gpu")]
        {
            if !self.available { return false; }
            matches!(op, Op::Matvec { quant: QuantKind::F32, .. })
        }
    }

    fn dispatch(&self, op: &mut Op) -> Result<(), BackendError> {
        #[cfg(not(feature = "gpu"))]
        {
            let _ = op;
            Err(BackendError::Unsupported { op: "n/a", backend: "vulkan" })
        }
        #[cfg(feature = "gpu")]
        {
            match op {
                Op::Matvec {
                    x, weight, bias, out,
                    out_dim, in_dim,
                    quant: QuantKind::F32,
                } => {
                    if crate::gpu::try_matvec(
                        x, weight, bias, out,
                        *out_dim as u32, *in_dim as u32,
                    ) {
                        Ok(())
                    } else {
                        Err(BackendError::Runtime(
                            "vulkan matvec dispatch failed".into(),
                        ))
                    }
                }
                _ => Err(BackendError::Unsupported {
                    op: op.name(),
                    backend: "vulkan",
                }),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_new_is_safe_without_vulkan() {
        // Constructor must not panic when feature is off or no GPU.
        let _ = VulkanBackend::try_new();
    }
}
