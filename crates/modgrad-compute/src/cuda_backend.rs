//! CUDA compute backend using candle-kernels for PTX and cudarc for dispatch.
//!
//! Implements ComputeBackend by loading pre-compiled PTX from candle-kernels
//! (unary, reduce) and dispatching via cudarc. Uses cuBLAS for matmul.
//!
//! Feature-gated: `cargo build --features cuda`

#[cfg(feature = "cuda")]
mod inner {
    use super::super::backend::ComputeBackend;
    use cudarc::driver::CudaDevice;
    use cudarc::cublas::CudaBlas;
    use std::sync::Arc;

    /// CUDA compute backend.
    pub struct CudaBackend {
        dev: Arc<CudaDevice>,
        blas: CudaBlas,
    }

    impl CudaBackend {
        /// Create a CUDA backend on device 0.
        pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
            let dev = CudaDevice::new(0)?;

            // Load candle kernel modules (PTX)
            dev.load_ptx(
                candle_kernels::UNARY.ptx().into(),
                "unary",
                &["usilu_f32"],
            )?;
            dev.load_ptx(
                candle_kernels::REDUCE.ptx().into(),
                "reduce",
                &["layernorm_f32", "softmax_f32"],
            )?;

            let blas = CudaBlas::new(dev.clone())?;
            Ok(Self { dev, blas })
        }

        /// Create on a specific device ordinal.
        pub fn on_device(ordinal: usize) -> Result<Self, Box<dyn std::error::Error>> {
            let dev = CudaDevice::new(ordinal)?;
            dev.load_ptx(
                candle_kernels::UNARY.ptx().into(),
                "unary",
                &["usilu_f32"],
            )?;
            dev.load_ptx(
                candle_kernels::REDUCE.ptx().into(),
                "reduce",
                &["layernorm_f32", "softmax_f32"],
            )?;
            let blas = CudaBlas::new(dev.clone())?;
            Ok(Self { dev, blas })
        }

        /// Get the underlying cudarc device.
        pub fn device(&self) -> &Arc<CudaDevice> {
            &self.dev
        }
    }

    // Stage 6: trait collapsed to lifecycle-only. The CUDA backend keeps
    // the default (heap alloc, no-op arena_reset, no-op flush) — matmul /
    // glu / silu / layer_norm / superlinear / trace_shift / sync_update
    // now dispatch through `modgrad_device::backend::ops::*`, which will
    // pick up the CUDA device backend once it registers in the registry.
    // The old inherent-method implementations here were never called
    // through the registry plumbing and are deleted wholesale.
    impl ComputeBackend for CudaBackend {}
}

#[cfg(feature = "cuda")]
pub use inner::CudaBackend;
