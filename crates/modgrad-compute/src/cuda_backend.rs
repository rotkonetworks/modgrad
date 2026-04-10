//! CUDA compute backend using candle-kernels for PTX and cudarc for dispatch.
//!
//! Implements ComputeBackend by loading pre-compiled PTX from candle-kernels
//! (unary, reduce) and dispatching via cudarc. Uses cuBLAS for matmul.
//!
//! Feature-gated: `cargo build --features cuda`

#[cfg(feature = "cuda")]
mod inner {
    use super::super::backend::ComputeBackend;
    use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig};
    use cudarc::cublas::{CudaBlas, GemmOp};
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

    impl ComputeBackend for CudaBackend {
        fn matvec(&self, weight: &[f32], bias: &[f32], x: &[f32],
                  y: &mut [f32], out_dim: usize, in_dim: usize) {
            // Upload to device
            let d_w = self.dev.htod_sync_copy(weight).unwrap();
            let d_x = self.dev.htod_sync_copy(x).unwrap();
            let mut d_y = self.dev.htod_sync_copy(bias).unwrap(); // init with bias

            // y = W @ x + b via cuBLAS gemv
            // cuBLAS is column-major, our weights are row-major [out_dim × in_dim].
            // For row-major W: y = W * x is equivalent to column-major y = W^T * x
            // with m=out_dim, n=1, k=in_dim, lda=in_dim
            unsafe {
                self.blas.gemm(
                    GemmOp::T,    // transpose W (row-major → col-major)
                    GemmOp::N,    // x is a column vector
                    out_dim as i32,
                    1,            // single vector
                    in_dim as i32,
                    1.0f32,       // alpha
                    &d_w,         // A = W [in_dim × out_dim in col-major = out_dim × in_dim row-major]
                    in_dim as i32, // lda
                    &d_x,         // B = x [in_dim × 1]
                    in_dim as i32, // ldb
                    1.0f32,       // beta = 1 (add to bias already in d_y)
                    &mut d_y,     // C = y [out_dim × 1]
                    out_dim as i32, // ldc
                ).unwrap();
            }

            // Download result
            self.dev.dtoh_sync_copy_into(&d_y, y).unwrap();
        }

        fn superlinear(&self, weights: &[f32], biases: &[f32], trace: &[f32],
                       output: &mut [f32], n_neurons: usize, in_per: usize, out_per: usize) {
            // For now: batched matmul on CPU (GPU batched small-matrix not worth it at <64 neurons).
            // TODO: custom CUDA kernel for batched per-neuron MLP when n_neurons > 256.
            for n in 0..n_neurons {
                let w_offset = n * out_per * in_per;
                let b_offset = n * out_per;
                let t_offset = n * in_per;
                let o_offset = n * out_per;
                for o in 0..out_per {
                    let mut sum = biases[b_offset + o];
                    for i in 0..in_per {
                        sum += weights[w_offset + o * in_per + i] * trace[t_offset + i];
                    }
                    output[o_offset + o] = sum;
                }
            }
        }

        fn glu(&self, input: &[f32], output: &mut [f32]) {
            let half = output.len();
            for i in 0..half {
                let gate = 1.0 / (1.0 + (-input[half + i]).exp());
                output[i] = input[i] * gate;
            }
        }

        fn silu_inplace(&self, x: &mut [f32]) {
            // Use candle's CUDA SiLU kernel for large tensors
            let n = x.len();
            if n < 256 {
                // CPU fallback for tiny tensors
                for v in x.iter_mut() {
                    let s = 1.0 / (1.0 + (-*v).exp());
                    *v = *v * s;
                }
                return;
            }

            let d_x = self.dev.htod_sync_copy(x).unwrap();
            let mut d_out = self.dev.alloc_zeros::<f32>(n).unwrap();
            let func = self.dev.get_func("unary", "usilu_f32").unwrap();
            let cfg = LaunchConfig::for_num_elems(n as u32);
            unsafe {
                func.launch(cfg, (&d_x, &mut d_out, n)).unwrap();
            }
            self.dev.dtoh_sync_copy_into(&d_out, x).unwrap();
        }

        fn layer_norm_inplace(&self, x: &mut [f32]) {
            // CPU for now — candle's layernorm kernel expects (batch, dim) layout
            let n = x.len() as f32;
            let mean: f32 = x.iter().sum::<f32>() / n;
            let var: f32 = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
            let inv_std = 1.0 / (var + 1e-5).sqrt();
            for v in x.iter_mut() {
                *v = (*v - mean) * inv_std;
            }
        }

        fn trace_shift(&self, traces: &mut [f32], new_activations: &[f32],
                       n_neurons: usize, memory_length: usize) {
            // CPU — simple shift, not worth GPU dispatch
            for n in 0..n_neurons {
                let base = n * memory_length;
                traces.copy_within(base + 1..base + memory_length, base);
                traces[base + memory_length - 1] = new_activations[n];
            }
        }

        fn sync_update(&self, alpha: &mut [f32], beta: &mut [f32],
                       activations_left: &[f32], activations_right: &[f32],
                       phases_left: &[f32], phases_right: &[f32],
                       decay: &[f32], decay_shift: &[f32],
                       dopamine: f32, n_pairs: usize, initialized: bool,
                       sync_out: &mut [f32]) {
            // CPU — sync is tiny relative to matmul
            let has_phases = !phases_left.is_empty() && !phases_right.is_empty();
            for i in 0..n_pairs {
                let l = activations_left[i];
                let r = activations_right[i];

                let temporal = if has_phases {
                    let dp = phases_left[i] - phases_right[i];
                    (-dp * dp / (2.0 * 0.3 * 0.3)).exp()
                } else { 1.0 };

                let pw = l * r * dopamine * temporal;
                let d = decay[i] + decay_shift.get(i).copied().unwrap_or(0.0);
                let retention = (-d.clamp(0.0, 15.0)).exp();

                if initialized {
                    alpha[i] = retention * alpha[i] + pw;
                    beta[i] = retention * beta[i] + dopamine;
                } else {
                    alpha[i] = pw;
                    beta[i] = dopamine;
                }

                sync_out[i] = alpha[i] / beta[i].sqrt().max(1e-8);
            }
        }
    }
}

#[cfg(feature = "cuda")]
pub use inner::CudaBackend;
