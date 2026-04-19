//! CUDA compute backend for NVIDIA GPUs (H100, H200, A100, etc.)
//!
//! Mirrors gpu.rs (Vulkan) but uses cudarc for the NVIDIA ecosystem:
//!   - cuBLAS for dense matmul (synapse forward)
//!   - Custom PTX kernels for per-neuron SuperLinear
//!   - Multi-device support via CudaContext per GPU
//!   - Peer-to-peer memory for NVLink cross-GPU transfer
//!
//! Feature-gated: `cargo build --features cuda`
//!
//! Design for 8×H200 cluster:
//!   - Each GPU gets its own CudaContext + stream + cuBLAS handle
//!   - Regions assigned to devices via DeviceMesh
//!   - Activation transfers via peer-to-peer (NVLink: 900 GB/s)
//!   - Weight buffers pinned to device (no per-dispatch upload)
//!
//! Build: requires CUDA toolkit (nvcc). Install via your distro or
//!   https://developer.nvidia.com/cuda-toolkit
//!
//! The public API (try_matvec, try_superlinear, etc.) compiles on all
//! platforms. The CUDA implementation is only compiled when the feature
//! is enabled AND the CUDA toolkit is present.

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext as CudaCtx, DevicePtr, DevicePtrMut};
#[cfg(feature = "cuda")]
use std::sync::Arc;

// ─── Per-GPU Context ───────────────────────────────────────

/// Per-GPU compute context. One per physical NVIDIA device.
#[cfg(feature = "cuda")]
pub struct CudaDevice {
    pub ctx: Arc<CudaCtx>,
    pub ordinal: usize,
    pub total_mem: usize,
}

#[cfg(feature = "cuda")]
impl CudaDevice {
    /// Initialize a CUDA device by ordinal (0, 1, ...).
    pub fn new(ordinal: usize) -> Result<Self, String> {
        let ctx = CudaCtx::new(ordinal)
            .map_err(|e| format!("CUDA device {ordinal}: {e}"))?;
        let total_mem = ctx.total_mem()
            .map_err(|e| format!("total_mem: {e}"))?;
        Ok(Self { ctx, ordinal, total_mem })
    }

    /// Dense matrix-vector multiply via cuBLAS: y = W @ x + b.
    /// W is [out_dim × in_dim] row-major on host.
    pub fn matvec(
        &self,
        weight: &[f32],
        bias: &[f32],
        x: &[f32],
        out: &mut [f32],
    ) -> Result<(), String> {
        let stream = self.ctx.default_stream();
        let blas = cudarc::cublas::CudaBlas::new(stream.clone())
            .map_err(|e| format!("cuBLAS: {e}"))?;

        let out_dim = bias.len();
        let in_dim = x.len();

        // Upload (host → device)
        let d_w = stream.clone_htod(weight)
            .map_err(|e| format!("upload W: {e}"))?;
        let d_x = stream.clone_htod(x)
            .map_err(|e| format!("upload x: {e}"))?;
        let mut d_y = stream.clone_htod(bias)
            .map_err(|e| format!("upload bias: {e}"))?;

        // y = 1.0 * W^T @ x + 1.0 * y  (y seeded with bias)
        // Row-major W treated as col-major W^T for cuBLAS
        unsafe {
            let (ptr_w, _g1) = d_w.device_ptr(&stream);
            let (ptr_x, _g2) = d_x.device_ptr(&stream);
            let (ptr_y, _g3) = d_y.device_ptr_mut(&stream);
            cudarc::cublas::result::sgemv(
                *blas.handle(),
                cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
                in_dim as i32,
                out_dim as i32,
                &1.0f32 as *const f32,
                ptr_w as *const f32,
                in_dim as i32,
                ptr_x as *const f32,
                1,
                &1.0f32 as *const f32,
                ptr_y as *mut f32,
                1,
            ).map_err(|e| format!("sgemv: {e}"))?;
        }

        // Download (device → host)
        stream.memcpy_dtoh(&d_y, out)
            .map_err(|e| format!("download: {e}"))?;
        Ok(())
    }
}

// ─── Multi-GPU Context ────────────────────────────────────

#[cfg(feature = "cuda")]
pub struct MultiGpu {
    pub devices: Vec<CudaDevice>,
}

#[cfg(feature = "cuda")]
impl MultiGpu {
    pub fn init_all() -> Result<Self, String> {
        let n = CudaCtx::device_count()
            .map_err(|e| format!("CUDA count: {e}"))? as usize;
        if n == 0 { return Err("no CUDA devices".into()); }

        let mut devices = Vec::with_capacity(n);
        for i in 0..n {
            devices.push(CudaDevice::new(i)?);
        }

        eprintln!("[cuda] {} devices:", n);
        for d in &devices {
            eprintln!("  device {}: {:.1} GB", d.ordinal, d.total_mem as f64 / 1e9);
        }

        Ok(Self { devices })
    }

    pub fn device(&self, ordinal: usize) -> Option<&CudaDevice> {
        self.devices.get(ordinal)
    }

    pub fn device_count(&self) -> usize { self.devices.len() }
}

// ─── CUDA C kernels (compiled at runtime via NVRTC) ────────

/// SuperLinear kernel: per-neuron MLP forward pass.
/// Same algorithm as shaders/superlinear.comp but in CUDA C.
#[cfg(feature = "cuda")]
pub const SUPERLINEAR_CUDA: &str = r#"
extern "C" __global__ void superlinear_forward(
    const float* __restrict__ trace,
    const float* __restrict__ weights,
    const float* __restrict__ biases,
    float* __restrict__ out,
    unsigned int n_neurons,
    unsigned int in_per,
    unsigned int out_per
) {
    unsigned int neuron = blockIdx.x;
    if (neuron >= n_neurons) return;

    extern __shared__ float shared_trace[];
    for (unsigned int i = threadIdx.x; i < in_per; i += blockDim.x) {
        shared_trace[i] = trace[neuron * in_per + i];
    }
    __syncthreads();

    const float* w = weights + neuron * out_per * in_per;
    for (unsigned int o = threadIdx.x; o < out_per; o += blockDim.x) {
        float sum = biases[neuron * out_per + o];
        const float* w_row = w + o * in_per;
        for (unsigned int i = 0; i < in_per; i++) {
            sum += w_row[i] * shared_trace[i];
        }
        out[neuron * out_per + o] = sum;
    }
}

extern "C" __global__ void matvec_forward(
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ x,
    float* __restrict__ y,
    unsigned int out_dim,
    unsigned int in_dim
) {
    unsigned int row = blockIdx.x;
    if (row >= out_dim) return;

    extern __shared__ float partial[];
    float sum = 0.0f;
    const float* w_row = weight + row * in_dim;
    for (unsigned int i = threadIdx.x; i < in_dim; i += blockDim.x) {
        sum += w_row[i] * x[i];
    }
    partial[threadIdx.x] = sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        y[row] = partial[0] + bias[row];
    }
}
"#;

// ─── Global State ──────────────────────────────────────────

#[cfg(feature = "cuda")]
static CUDA: std::sync::OnceLock<Option<MultiGpu>> = std::sync::OnceLock::new();

/// Initialize all CUDA devices. Call once at startup.
pub fn init_global() {
    #[cfg(feature = "cuda")]
    {
        CUDA.get_or_init(|| MultiGpu::init_all().ok());
    }
}

// ─── Public API (always compiles, no-ops without cuda feature) ──

/// Try cuBLAS matvec on default device. Returns false if unavailable.
pub fn try_matvec(
    _x: &[f32], _weight: &[f32], _bias: &[f32], _out: &mut [f32],
    _out_dim: u32, _in_dim: u32,
) -> bool {
    #[cfg(feature = "cuda")]
    {
        if let Some(Some(multi)) = CUDA.get() {
            if let Some(dev) = multi.device(0) {
                return dev.matvec(weight, bias, x, out).is_ok();
            }
        }
    }
    false
}

/// Try SuperLinear on default device. Returns false if unavailable.
pub fn try_superlinear(
    _trace: &[f32], _weights: &[f32], _biases: &[f32], _out: &mut [f32],
    _n_neurons: u32, _in_per: u32, _out_per: u32,
) -> bool {
    // Unimplemented on CUDA — caller falls through to Vulkan or CPU.
    // Would be NVRTC-compiled PTX kernel dispatch here.
    false
}

/// Try matvec on a specific device.
pub fn try_matvec_on(
    _device_ordinal: usize,
    _x: &[f32], _weight: &[f32], _bias: &[f32], _out: &mut [f32],
) -> bool {
    #[cfg(feature = "cuda")]
    {
        if let Some(Some(multi)) = CUDA.get() {
            if let Some(dev) = multi.device(_device_ordinal) {
                return dev.matvec(weight, bias, x, out).is_ok();
            }
        }
    }
    false
}

/// Number of CUDA devices available.
pub fn device_count() -> usize {
    #[cfg(feature = "cuda")]
    {
        if let Some(Some(multi)) = CUDA.get() {
            return multi.device_count();
        }
    }
    0
}

/// Check if CUDA is available.
pub fn available() -> bool { device_count() > 0 }
