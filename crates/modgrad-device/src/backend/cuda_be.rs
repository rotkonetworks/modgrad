//! CUDA backend — thin wrapper around the existing cudarc-based
//! dispatch in `crate::cuda`. Feature-gated behind `--features cuda`.
//!
//! Coverage today matches the pre-refactor reality: just `Matvec`.
//! Everything else falls through to CPU (or KFD on gfx1102). Further
//! ops land as we port them — same pattern as `KfdBackend`.

use super::{Backend, BackendError, BufferBackend, ComputeCtx, DeviceInfo, DeviceKind, HostBuffer, Op};
#[cfg(feature = "cuda")]
use super::QuantKind;

/// CUDA backend. When the `cuda` feature is off this is a zero-size
/// type whose `supports()` always returns false — safe to include in
/// any registry without conditional compilation at the caller.
pub struct CudaBackend {
    #[cfg(feature = "cuda")]
    device: crate::cuda::CudaDevice,
    #[cfg(not(feature = "cuda"))]
    _unused: (),
}

impl CudaBackend {
    /// Probe the first CUDA device (ordinal 0). Returns `None` if CUDA
    /// is disabled at compile time or no device is present at runtime.
    ///
    /// cudarc can PANIC (not just Err) when libcuda.so is missing —
    /// typical on pure-AMD boxes. We `catch_unwind` so a default
    /// build with `cuda` enabled doesn't blow up on non-NVIDIA hosts.
    pub fn try_new() -> Option<Self> {
        #[cfg(feature = "cuda")]
        {
            std::panic::catch_unwind(|| crate::cuda::CudaDevice::new(0))
                .ok()
                .and_then(|r| r.ok())
                .map(|d| Self { device: d })
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    }
}

impl Backend for CudaBackend {
    fn name(&self) -> &'static str { "cuda" }

    fn device_info(&self) -> DeviceInfo {
        #[cfg(feature = "cuda")]
        {
            DeviceInfo {
                kind: DeviceKind::Cuda,
                name: format!("CUDA device {}", self.device.ordinal),
                total_mem_bytes: self.device.total_mem as u64,
                arch: None,  // filled in when we add driver query
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            DeviceInfo {
                kind: DeviceKind::Cuda,
                name: "cuda (feature disabled)".into(),
                total_mem_bytes: 0,
                arch: None,
            }
        }
    }

    fn supports(&self, op: &Op) -> bool {
        #[cfg(not(feature = "cuda"))]
        { let _ = op; return false; }
        #[cfg(feature = "cuda")]
        {
            match op {
                Op::Matvec { quant: QuantKind::F32, .. } => true,
                _ => false,
            }
        }
    }

    fn dispatch(&self, op: &mut Op) -> Result<(), BackendError> {
        #[cfg(not(feature = "cuda"))]
        {
            let _ = op;
            Err(BackendError::Unsupported { op: "n/a", backend: "cuda" })
        }
        #[cfg(feature = "cuda")]
        {
            match op {
                Op::Matvec {
                    x, weight, bias, out,
                    out_dim: _out_dim, in_dim: _in_dim,
                    quant: QuantKind::F32,
                } => {
                    self.device.matvec(weight, bias, x, out)
                        .map_err(BackendError::Runtime)
                }
                _ => Err(BackendError::Unsupported {
                    op: op.name(),
                    backend: "cuda",
                }),
            }
        }
    }
}

/// CUDA backend doesn't have VRAM allocation plumbed into this crate
/// yet — device-resident buffers will land when `crate::cuda` grows an
/// explicit alloc surface. Until then we default to `HostBuffer`, the
/// same as CPU. Dispatches still run on GPU (via `crate::cuda`); only
/// the `ComputeCtx::alloc_buffer` path is host-backed.
impl BufferBackend for CudaBackend {
    type Buffer = HostBuffer;

    fn alloc_buffer(&self, n: usize) -> Result<HostBuffer, BackendError> {
        Ok(HostBuffer::new(n))
    }
}

/// CUDA ComputeCtx hooks are no-ops until we wire a real device-arena
/// and an async stream in a follow-up. Having them present keeps
/// `ComputeCtx<CudaBackend>` a drop-in replacement for future lifecycle-
/// aware call sites.
impl ComputeCtx<CudaBackend> {
    /// No-op — no CUDA arena plumbed yet.
    pub fn arena_reset(&self) {}

    /// No-op — dispatch is still synchronous through `crate::cuda`.
    pub fn flush(&self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_new_none_without_device() {
        // Whether or not CUDA compiles in, a box without an NVIDIA GPU
        // returns None. Verify the constructor doesn't panic.
        let _ = CudaBackend::try_new();
    }
}
