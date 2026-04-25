//! AMD ROCm backend — hand-rolled FFI to libamdhip64 + libhipblas.
//!
//! Feature-gated behind `--features rocm`. Linked dynamically at
//! runtime so no extra build-time dependencies beyond the system
//! ROCm install (headers not required — we declare the signatures
//! inline in the `ffi` sub-module).
//!
//! Coverage on first landing: `Matvec` only. This is the minimum that
//! validates the wiring end-to-end. Extending to Matmul is the next
//! commit; MIOpen-backed layer_norm/silu follows after.
//!
//! Complementary to KFD: on gfx1102 the hand-written KFD kernels
//! remain preferred (higher in `BackendRegistry::detect()` order).
//! ROCm covers every other AMD arch supported by the ROCm runtime,
//! plus falls through to KFD-uncovered ops on gfx1102 itself.

use super::{Backend, BackendError, BufferBackend, ComputeCtx, DeviceInfo, DeviceKind, Op};
#[cfg(feature = "rocm")]
use super::{ActivationMode, BinaryOpKind, DeviceBuffer, QuantKind};
#[cfg(not(feature = "rocm"))]
use super::HostBuffer;

#[cfg(feature = "rocm")]
pub mod ffi {
    #![allow(non_camel_case_types)]
    use std::os::raw::{c_char, c_int, c_uint, c_void};

    pub type hipError_t = c_int;
    pub type hipblasHandle_t = *mut c_void;
    pub type hipblasStatus_t = c_int;

    /// hipMemcpyKind enumeration.
    pub const HIP_MEMCPY_HOST_TO_DEVICE: c_int = 1;
    pub const HIP_MEMCPY_DEVICE_TO_HOST: c_int = 2;

    /// hipblasOperation_t.
    pub const HIPBLAS_OP_N: c_int = 111;
    pub const HIPBLAS_OP_T: c_int = 112;

    /// `hipDataType::HIP_R_16BF`. From `hip/library_types.h`:
    /// real bf16, value 14. The hipblasGemmEx API expects this enum
    /// (NOT the legacy `hipblasDatatype_t`) for its `*Type` arguments.
    pub const HIP_R_16BF: c_int = 14;

    /// `hipblasComputeType_t::HIPBLAS_COMPUTE_32F`. From
    /// `hipblas-common/hipblas-common.h` — the standard mixed-precision
    /// recipe: bf16 input/output, fp32 accumulate. Value 2.
    pub const HIPBLAS_COMPUTE_32F: c_int = 2;

    /// `hipblasGemmAlgo_t::HIPBLAS_GEMM_DEFAULT` — let hipblas pick the
    /// algorithm. Value 160 per `hipblas/hipblas.h`.
    pub const HIPBLAS_GEMM_DEFAULT: c_int = 160;

    #[link(name = "amdhip64")]
    unsafe extern "C" {
        pub fn hipGetDeviceCount(count: *mut c_int) -> hipError_t;
        pub fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> hipError_t;
        pub fn hipFree(ptr: *mut c_void) -> hipError_t;
        pub fn hipMemcpy(
            dst: *mut c_void,
            src: *const c_void,
            size: usize,
            kind: c_int,
        ) -> hipError_t;
        pub fn hipDeviceSynchronize() -> hipError_t;
        pub fn hipGetErrorString(error: hipError_t) -> *const c_char;
    }

    #[link(name = "hipblas")]
    unsafe extern "C" {
        pub fn hipblasCreate(handle: *mut hipblasHandle_t) -> hipblasStatus_t;
        pub fn hipblasDestroy(handle: hipblasHandle_t) -> hipblasStatus_t;
        pub fn hipblasSgemv(
            handle: hipblasHandle_t,
            trans: c_int,
            m: c_int,
            n: c_int,
            alpha: *const f32,
            a: *const f32,
            lda: c_int,
            x: *const f32,
            incx: c_int,
            beta: *const f32,
            y: *mut f32,
            incy: c_int,
        ) -> hipblasStatus_t;
        pub fn hipblasSgemm(
            handle: hipblasHandle_t,
            transa: c_int,
            transb: c_int,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: *const f32,
            a: *const f32,
            lda: c_int,
            b: *const f32,
            ldb: c_int,
            beta: *const f32,
            c: *mut f32,
            ldc: c_int,
        ) -> hipblasStatus_t;

        /// Modern hipblas mixed-dtype GEMM. Inputs/output may be bf16,
        /// fp16, or fp32; compute precision is selectable
        /// independently. We invoke it with `HIPBLAS_R_16BF` for A/B/C
        /// and `HIPBLAS_COMPUTE_32F` for the accumulator — the standard
        /// mixed-precision training recipe.
        ///
        /// `alpha`/`beta` are passed as `*const c_void` because the
        /// underlying type is determined by `compute_type`. For
        /// `HIPBLAS_COMPUTE_32F` they are fp32 scalars.
        pub fn hipblasGemmEx(
            handle: hipblasHandle_t,
            transa: c_int,
            transb: c_int,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: *const c_void,
            a: *const c_void,
            a_type: c_int,
            lda: c_int,
            b: *const c_void,
            b_type: c_int,
            ldb: c_int,
            beta: *const c_void,
            c: *mut c_void,
            c_type: c_int,
            ldc: c_int,
            compute_type: c_int,
            algo: c_int,
        ) -> hipblasStatus_t;
    }

    // ─── MIOpen ────────────────────────────────────────────────
    //
    // Bound directly per /opt/rocm/include/miopen/miopen.h. The handle
    // and descriptor types are all `MIOPEN_DECLARE_OBJECT`-style
    // pointers to opaque structs — i.e. `*mut c_void` from FFI's
    // perspective. Every function returns `miopenStatus_t` (an int);
    // `0` = `miopenStatusSuccess`. The signatures below mirror the
    // header literally — DO NOT reorder or retype arguments without
    // re-reading the header, MIOpen is unforgiving about ABI.
    //
    // Backward functions are declared but not invoked yet — they're
    // here so the FFI block ships complete and limit #1 (backward
    // through resident) can wire them without another FFI edit.

    pub type miopenHandle_t = *mut c_void;
    pub type miopenTensorDescriptor_t = *mut c_void;
    pub type miopenActivationDescriptor_t = *mut c_void;
    pub type miopenStatus_t = c_int;

    /// `miopenStatusSuccess` — the only success code from MIOpen.
    pub const MIOPEN_STATUS_SUCCESS: miopenStatus_t = 0;

    /// `miopenDataType_t::miopenFloat`.
    pub const MIOPEN_DATA_FLOAT: c_int = 1;

    /// `miopenSoftmaxAlgorithm_t::MIOPEN_SOFTMAX_ACCURATE`.
    /// We hard-pick ACCURATE so the kernel subtracts the row max before
    /// exp — matches the CPU reference numerically.
    pub const MIOPEN_SOFTMAX_ACCURATE: c_int = 1;
    /// `miopenSoftmaxAlgorithm_t::MIOPEN_SOFTMAX_LOG`.
    pub const MIOPEN_SOFTMAX_LOG: c_int = 2;

    /// `miopenSoftmaxMode_t::MIOPEN_SOFTMAX_MODE_INSTANCE`.
    /// Per-image normalisation (one row of the [n, c, 1, 1] tensor at a
    /// time) which is what we want for row-wise softmax.
    pub const MIOPEN_SOFTMAX_MODE_INSTANCE: c_int = 0;

    /// `miopenActivationMode_t` — only the modes we expose.
    pub const MIOPEN_ACTIVATION_LOGISTIC: c_int = 1;
    pub const MIOPEN_ACTIVATION_TANH: c_int = 2;
    pub const MIOPEN_ACTIVATION_RELU: c_int = 3;

    /// `miopenTensorOp_t` — element-wise tensor ops.
    pub const MIOPEN_OP_TENSOR_ADD: c_int = 0;
    pub const MIOPEN_OP_TENSOR_MUL: c_int = 1;
    pub const MIOPEN_OP_TENSOR_MIN: c_int = 2;
    pub const MIOPEN_OP_TENSOR_MAX: c_int = 3;

    /// `miopenNormMode_t::MIOPEN_WEIGHT_BIAS` — affine LayerNorm with
    /// learnable weight and bias. Matches PyTorch
    /// `nn.LayerNorm(elementwise_affine=True)`. The `MIOPEN_BETA_API`
    /// header guards this enum, but the symbol is always exported by
    /// libMIOpen.so on this hardware (verified by `nm -D`).
    pub const MIOPEN_NORM_WEIGHT_BIAS: c_int = 1;

    #[link(name = "MIOpen")]
    unsafe extern "C" {
        // Handle lifecycle. Probed working in examples/miopen_probe.rs
        // (commit 3e46684).
        pub fn miopenCreate(handle: *mut miopenHandle_t) -> miopenStatus_t;
        pub fn miopenDestroy(handle: miopenHandle_t) -> miopenStatus_t;

        // Tensor descriptor lifecycle.
        pub fn miopenCreateTensorDescriptor(
            tensorDesc: *mut miopenTensorDescriptor_t,
        ) -> miopenStatus_t;
        pub fn miopenDestroyTensorDescriptor(
            tensorDesc: miopenTensorDescriptor_t,
        ) -> miopenStatus_t;
        pub fn miopenSet4dTensorDescriptor(
            tensorDesc: miopenTensorDescriptor_t,
            dataType: c_int,
            n: c_int,
            c: c_int,
            h: c_int,
            w: c_int,
        ) -> miopenStatus_t;

        // Activation descriptor + forward/backward.
        pub fn miopenCreateActivationDescriptor(
            activDesc: *mut miopenActivationDescriptor_t,
        ) -> miopenStatus_t;
        pub fn miopenDestroyActivationDescriptor(
            activDesc: miopenActivationDescriptor_t,
        ) -> miopenStatus_t;
        pub fn miopenSetActivationDescriptor(
            activDesc: miopenActivationDescriptor_t,
            mode: c_int,
            activAlpha: f64,
            activBeta: f64,
            activGamma: f64,
        ) -> miopenStatus_t;
        pub fn miopenActivationForward(
            handle: miopenHandle_t,
            activDesc: miopenActivationDescriptor_t,
            alpha: *const c_void,
            xDesc: miopenTensorDescriptor_t,
            x: *const c_void,
            beta: *const c_void,
            yDesc: miopenTensorDescriptor_t,
            y: *mut c_void,
        ) -> miopenStatus_t;
        pub fn miopenActivationBackward(
            handle: miopenHandle_t,
            activDesc: miopenActivationDescriptor_t,
            alpha: *const c_void,
            yDesc: miopenTensorDescriptor_t,
            y: *const c_void,
            dyDesc: miopenTensorDescriptor_t,
            dy: *const c_void,
            xDesc: miopenTensorDescriptor_t,
            x: *const c_void,
            beta: *const c_void,
            dxDesc: miopenTensorDescriptor_t,
            dx: *mut c_void,
        ) -> miopenStatus_t;

        // Softmax V2.
        pub fn miopenSoftmaxForward_V2(
            handle: miopenHandle_t,
            alpha: *const c_void,
            xDesc: miopenTensorDescriptor_t,
            x: *const c_void,
            beta: *const c_void,
            yDesc: miopenTensorDescriptor_t,
            y: *mut c_void,
            algorithm: c_int,
            mode: c_int,
        ) -> miopenStatus_t;
        pub fn miopenSoftmaxBackward_V2(
            handle: miopenHandle_t,
            alpha: *const c_void,
            yDesc: miopenTensorDescriptor_t,
            y: *const c_void,
            dyDesc: miopenTensorDescriptor_t,
            dy: *const c_void,
            beta: *const c_void,
            dxDesc: miopenTensorDescriptor_t,
            dx: *mut c_void,
            algorithm: c_int,
            mode: c_int,
        ) -> miopenStatus_t;

        // LayerNorm (BETA_API in the header but symbols are exported).
        // The signature uses `float epsilon` (not f64) and
        // `int32_t normalized_dim` — bound here as f32 / c_int.
        pub fn miopenLayerNormForward(
            handle: miopenHandle_t,
            mode: c_int,
            xDesc: miopenTensorDescriptor_t,
            x: *const c_void,
            weightDesc: miopenTensorDescriptor_t,
            weight: *const c_void,
            biasDesc: miopenTensorDescriptor_t,
            bias: *const c_void,
            epsilon: f32,
            normalized_dim: c_int,
            yDesc: miopenTensorDescriptor_t,
            y: *mut c_void,
            meanDesc: miopenTensorDescriptor_t,
            mean: *mut c_void,
            rstdDesc: miopenTensorDescriptor_t,
            rstd: *mut c_void,
        ) -> miopenStatus_t;
        pub fn miopenLayerNormBackward(
            handle: miopenHandle_t,
            mode: c_int,
            workspace: *mut c_void,
            workspaceSizeInBytes: usize,
            dyDesc: miopenTensorDescriptor_t,
            dy: *const c_void,
            xDesc: miopenTensorDescriptor_t,
            x: *const c_void,
            weightDesc: miopenTensorDescriptor_t,
            weight: *const c_void,
            meanDesc: miopenTensorDescriptor_t,
            mean: *const c_void,
            rstdDesc: miopenTensorDescriptor_t,
            rstd: *const c_void,
            normalized_dim: c_int,
            dxDesc: miopenTensorDescriptor_t,
            dx: *mut c_void,
            dwDesc: miopenTensorDescriptor_t,
            dw: *mut c_void,
            dbDesc: miopenTensorDescriptor_t,
            db: *mut c_void,
        ) -> miopenStatus_t;

        // GLU (BETA_API in the header but symbols are exported).
        // `dim` is the split dimension — for the `n_rows × 2*half`
        // tensors we wire here, that's the last (W) axis = 3.
        pub fn miopenGLUForward(
            handle: miopenHandle_t,
            inputDesc: miopenTensorDescriptor_t,
            input: *const c_void,
            outputDesc: miopenTensorDescriptor_t,
            output: *mut c_void,
            dim: u32,
        ) -> miopenStatus_t;
        pub fn miopenGLUBackward(
            handle: miopenHandle_t,
            inputDesc: miopenTensorDescriptor_t,
            input: *const c_void,
            outputGradDesc: miopenTensorDescriptor_t,
            outputGrad: *const c_void,
            inputGradDesc: miopenTensorDescriptor_t,
            inputGrad: *mut c_void,
            dim: u32,
        ) -> miopenStatus_t;

        // OpTensor (binary elementwise).
        pub fn miopenOpTensor(
            handle: miopenHandle_t,
            tensorOp: c_int,
            alpha1: *const c_void,
            aDesc: miopenTensorDescriptor_t,
            a: *const c_void,
            alpha2: *const c_void,
            bDesc: miopenTensorDescriptor_t,
            b: *const c_void,
            beta: *const c_void,
            cDesc: miopenTensorDescriptor_t,
            c: *mut c_void,
        ) -> miopenStatus_t;
    }

    // ─── Custom hipcc kernels ──────────────────────────────────
    //
    // Compiled by `build.rs` from `kernels/rms_norm.hip` and linked in
    // when `MODGRAD_HIPCC_KERNELS` is set as a build-time cfg by the
    // build script. Hosts without hipcc don't define the cfg and the
    // FFI block is omitted entirely — no link-time dependency, no
    // runtime missing-symbol error, just a lower-tier `Op::supports()`
    // for `RmsNormResident`.
    //
    // Calling convention from `kernels/rms_norm.hip`: one block per
    // sample, `BLOCK_SIZE` threads per block (the kernel side picks
    // the value); the launcher computes the grid/block from `n` /
    // `hidden`. Returning a hipError_t lets the dispatch arm surface
    // launch failures the same way as hipblas / MIOpen calls.
    #[cfg(modgrad_hipcc_kernels)]
    unsafe extern "C" {
        pub fn launch_rms_norm(
            x: *const f32,
            weight: *const f32,
            y: *mut f32,
            n: c_int,
            hidden: c_int,
            eps: f32,
        ) -> hipError_t;

        // Q4_K_M dequantize. `q4k` is `n_blocks * 144` bytes of GGUF
        // Q4_K format; `fp32_out` receives `n_blocks * 256` fp32
        // values. See `kernels/dequant_q4k.hip` for the kernel.
        pub fn launch_dequant_q4k(
            q4k: *const u8,
            n_blocks: c_int,
            fp32_out: *mut f32,
        ) -> hipError_t;
    }

    /// Convert a hipError_t into a human-readable String.
    pub fn hip_err_str(code: hipError_t) -> String {
        unsafe {
            let p = hipGetErrorString(code);
            if p.is_null() {
                return format!("hip error {code}");
            }
            let cstr = std::ffi::CStr::from_ptr(p);
            cstr.to_string_lossy().into_owned()
        }
    }

    /// Runtime-level check: do we have at least one ROCm-capable device?
    pub fn runtime_available() -> bool {
        let mut count: c_int = 0;
        let err = unsafe { hipGetDeviceCount(&mut count) };
        if err != 0 {
            eprintln!("rocm probe: hipGetDeviceCount err={err} ({})", hip_err_str(err));
        } else if count == 0 {
            eprintln!("rocm probe: no HIP devices");
        }
        err == 0 && count > 0
    }

    /// Unused for now; keeps the no-mangling tag happy in lint runs.
    #[allow(dead_code)]
    pub fn _uint_marker(_: c_uint) {}
}

/// RAII wrapper around a single `hipMalloc` allocation. `Drop` calls
/// `hipFree`; ergonomic "?" error handling in the dispatcher stops
/// leaking on the first error path because the Drop runs automatically.
///
/// Public so `modgrad-compute::GpuVec::Hip` can wrap one for cross-call
/// device residency (weights uploaded once, kept device-side across
/// forwards). Pre-residency code used `HipBuffer` only privately
/// inside one dispatch.
#[cfg(feature = "rocm")]
pub struct HipBuffer {
    ptr: *mut std::os::raw::c_void,
    bytes: usize,
}

// Safe to send/sync a hipMalloc'd device pointer between threads —
// HIP's runtime is internally synchronized; only the holder of the
// pointer can read/write it. The pointer itself is plain data.
#[cfg(feature = "rocm")]
unsafe impl Send for HipBuffer {}
#[cfg(feature = "rocm")]
unsafe impl Sync for HipBuffer {}

#[cfg(feature = "rocm")]
impl HipBuffer {
    /// Public allocator — used by `GpuVec::Hip` and resident-weight
    /// caches in `modgrad-compute`.
    pub fn new(bytes: usize) -> Result<Self, BackendError> {
        Self::alloc(bytes)
    }

    /// Allocate `bytes` on the current HIP device.
    fn alloc(bytes: usize) -> Result<Self, BackendError> {
        let mut ptr: *mut std::os::raw::c_void = std::ptr::null_mut();
        let err = unsafe { ffi::hipMalloc(&mut ptr, bytes) };
        if err != 0 {
            return Err(BackendError::Runtime(format!(
                "hipMalloc({bytes}): {}", ffi::hip_err_str(err)
            )));
        }
        Ok(Self { ptr, bytes })
    }

    /// Raw device pointer. For passing to hipBLAS or kernel launches.
    pub fn device_ptr(&self) -> *mut std::os::raw::c_void { self.ptr }

    /// Allocation size in bytes.
    pub fn bytes(&self) -> usize { self.bytes }

    /// Capacity in f32 elements.
    pub fn len_f32(&self) -> usize { self.bytes / 4 }

    /// Public alias for `upload_f32` so callers in other crates can
    /// stage host data into this buffer.
    pub fn copy_from_host(&self, src: &[f32]) -> Result<(), BackendError> {
        self.upload_f32(src)
    }

    /// Public alias for `download_f32`.
    pub fn copy_to_host(&self, dst: &mut [f32]) -> Result<(), BackendError> {
        self.download_f32(dst)
    }

    /// Copy `src` (host) into this device buffer.
    /// Returns `Err(BackendError::Runtime)` if `src` is larger than
    /// the device allocation — release builds previously over-read
    /// host memory and silently truncated. The check is a hot-path
    /// branch but cheaper than the kernel launch it precedes.
    fn upload_f32(&self, src: &[f32]) -> Result<(), BackendError> {
        let nbytes = src.len() * 4;
        if nbytes > self.bytes {
            return Err(BackendError::Runtime(format!(
                "upload_f32: src is {} bytes, device buffer is {} bytes",
                nbytes, self.bytes,
            )));
        }
        let err = unsafe {
            ffi::hipMemcpy(
                self.ptr,
                src.as_ptr() as *const std::os::raw::c_void,
                nbytes,
                ffi::HIP_MEMCPY_HOST_TO_DEVICE,
            )
        };
        if err != 0 {
            return Err(BackendError::Runtime(format!(
                "hipMemcpy H2D: {}", ffi::hip_err_str(err)
            )));
        }
        Ok(())
    }

    /// Copy this device buffer into `dst` (host). Returns
    /// `Err(BackendError::Runtime)` if `dst` is larger than the
    /// device allocation — release builds previously short-wrote and
    /// left tail garbage. The check is a hot-path branch but cheaper
    /// than the kernel sync it precedes.
    fn download_f32(&self, dst: &mut [f32]) -> Result<(), BackendError> {
        let nbytes = dst.len() * 4;
        if nbytes > self.bytes {
            return Err(BackendError::Runtime(format!(
                "download_f32: dst is {} bytes, device buffer is {} bytes",
                nbytes, self.bytes,
            )));
        }
        let err = unsafe {
            ffi::hipMemcpy(
                dst.as_mut_ptr() as *mut std::os::raw::c_void,
                self.ptr,
                nbytes,
                ffi::HIP_MEMCPY_DEVICE_TO_HOST,
            )
        };
        if err != 0 {
            return Err(BackendError::Runtime(format!(
                "hipMemcpy D2H: {}", ffi::hip_err_str(err)
            )));
        }
        Ok(())
    }

    /// Raw device pointer as f32* (for BLAS calls).
    fn as_f32_ptr(&self) -> *mut f32 {
        self.ptr as *mut f32
    }
}

#[cfg(feature = "rocm")]
impl Drop for HipBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::hipFree(self.ptr) };
        }
    }
}

/// RAII guard around a sequence of asynchronous HIP dispatches.
///
/// **Invariant the caller buys by holding one of these:** any
/// resident dispatch (`hipblasSgemv`, `hipblasSgemm`, ...) executed
/// inside this scope is bounded — the queue of pending kernels
/// never exceeds `sync_every` between syncs, and a final
/// `hipDeviceSynchronize` runs unconditionally on `Drop`.
///
/// **Why this type exists.** `hipblasSgemv` returns the moment the
/// kernel is queued, not when it runs. The HIP runtime's command
/// queue has a finite depth. Submitting faster than the GPU
/// dispatches eventually overflows it; amdgpu's watchdog declares
/// the device hung and resets it. On a system where the same GPU
/// drives the display, the reset takes Xorg with it. This was hit
/// in `examples/bench_resident.rs` on 2026-04-25 — see
/// `memory/feedback_hip_queue_overflow.md`. Making
/// `LinearResident::forward` (and any future resident dispatch)
/// require a `&HipBatch` argument turns the failure mode from
/// "easy to forget" into "compile error if you do."
///
/// **Default cadence (256 dispatches between syncs):** chosen
/// empirically — well below the queue overflow threshold on
/// gfx1102 (which started backing up around 4 s ≈ ~600 k unsynced
/// hipblasSgemv at our shape), and well above the per-sync
/// overhead break-even point. Tighter cadences are safe but slow;
/// looser cadences risk the same crash.
///
/// **Not Send.** HIP runtime contexts are thread-local in the
/// general case (rocm 6.x docs). Holding a `HipBatch` and
/// dispatching from another thread would race against the
/// internal counter and the per-thread queue. The
/// `PhantomData<*const ()>` makes that a compile error.
#[cfg(feature = "rocm")]
pub struct HipBatch {
    pending: std::cell::Cell<usize>,
    sync_every: usize,
    _not_send: std::marker::PhantomData<*const ()>,
}

#[cfg(feature = "rocm")]
impl HipBatch {
    pub const DEFAULT_SYNC_EVERY: usize = 256;

    /// New batch with the default sync cadence.
    pub fn new() -> Self {
        Self::with_sync_every(Self::DEFAULT_SYNC_EVERY)
    }

    /// New batch with a caller-chosen cadence. `n` must be ≥ 1.
    /// Larger `n` ⇒ less per-call overhead but higher risk of
    /// queue overflow on weak silicon. The default
    /// (`DEFAULT_SYNC_EVERY`) is the empirically-validated safe
    /// point on gfx1102.
    pub fn with_sync_every(n: usize) -> Self {
        assert!(n >= 1, "HipBatch::with_sync_every: n must be ≥ 1");
        Self {
            pending: std::cell::Cell::new(0),
            sync_every: n,
            _not_send: std::marker::PhantomData,
        }
    }

    /// Called by resident dispatch ops AFTER they queue a kernel.
    /// When `pending` reaches `sync_every`, drains the queue
    /// synchronously. Failure mode: returns the hipError on a
    /// failed sync — caller (the dispatch op) propagates as
    /// `BackendError::Runtime`. The dispatcher is structured so
    /// "queue full → sync fails → dispatch fails" surfaces loudly
    /// as a Result, never silently.
    #[doc(hidden)]
    pub fn note_dispatch(&self) -> Result<(), BackendError> {
        let new = self.pending.get() + 1;
        if new >= self.sync_every {
            let err = unsafe { ffi::hipDeviceSynchronize() };
            if err != 0 {
                self.pending.set(0);  // reset even on failure to avoid feedback loop
                return Err(BackendError::Runtime(format!(
                    "HipBatch sync (every {}): {}",
                    self.sync_every, ffi::hip_err_str(err),
                )));
            }
            self.pending.set(0);
        } else {
            self.pending.set(new);
        }
        Ok(())
    }

    /// Force an immediate drain. Called by Drop and by callers who
    /// need a sync mid-scope (e.g. before reading device output
    /// back to host with `copy_to_host`).
    pub fn flush(&self) -> Result<(), BackendError> {
        let err = unsafe { ffi::hipDeviceSynchronize() };
        self.pending.set(0);
        if err != 0 {
            return Err(BackendError::Runtime(format!(
                "HipBatch::flush: {}", ffi::hip_err_str(err)
            )));
        }
        Ok(())
    }
}

#[cfg(feature = "rocm")]
impl Default for HipBatch {
    fn default() -> Self { Self::new() }
}

#[cfg(feature = "rocm")]
impl Drop for HipBatch {
    /// Final drain on scope exit. Cheap if `pending == 0`. We
    /// deliberately swallow errors here — the alternative is to
    /// abort, but a single failed sync at scope exit is recoverable
    /// (the next sync attempt will surface it). Aborting in Drop
    /// would convert any hip teardown wobble into a hard process
    /// kill, which is worse than letting the next operation report
    /// the error.
    fn drop(&mut self) {
        if self.pending.get() > 0 {
            unsafe { let _ = ffi::hipDeviceSynchronize(); }
        }
    }
}

/// MIOpen context — RAII handle plus lazily-cached descriptors.
///
/// Mirrors the hipBLAS-handle pattern on `RocmBackend`: one process-wide
/// instance, owned by `RocmBackend`, dropped via `Drop`. MIOpen handles
/// are **not** thread-safe (same as hipBLAS), so the inner pointer
/// lives behind `Mutex`. Tensor descriptors are cached per
/// `(n, c, h, w)` shape so we don't burn a `miopenCreateTensorDescriptor`
/// + `miopenSet4d` per dispatch — those calls are cheap individually
/// but allocate a small heap each time, which adds up across a
/// training loop.
///
/// The activation descriptor is per-mode (LOGISTIC/TANH/RELU); the
/// `Silu` activation reuses the LOGISTIC descriptor and then issues a
/// follow-up `miopenOpTensor` MUL — see [`ActivationMode::Silu`].
///
/// Dropping the context calls `miopenDestroy`; per the MIOpen docs,
/// destroying the handle implicitly invalidates the descriptors it
/// holds, but we explicitly destroy each cached descriptor first so
/// the order matches the create-then-destroy contract literally.
#[cfg(feature = "rocm")]
pub(crate) struct MiopenContext {
    /// MIOpen handle. Initialised once via `miopenCreate`. Wrapped
    /// in `Mutex` because the handle is not safe for concurrent
    /// dispatches per the MIOpen docs (and we serialise hipBLAS the
    /// same way for the same reason).
    handle: std::sync::Mutex<ffi::miopenHandle_t>,
    /// Cached 4D tensor descriptors keyed by `(n, c, h, w)`. Hit rate
    /// is high in steady-state training because most layers use the
    /// same shape every step. Lock-then-lookup-then-clone-pointer; the
    /// pointer remains valid because dropping the context destroys
    /// every entry before destroying the handle.
    tensor_descs:
        std::sync::Mutex<std::collections::HashMap<(i32, i32, i32, i32), ffi::miopenTensorDescriptor_t>>,
    /// Cached activation descriptors keyed by mode. Allocated on first
    /// use of each mode, never invalidated.
    activation_descs:
        std::sync::Mutex<std::collections::HashMap<i32, ffi::miopenActivationDescriptor_t>>,
}

#[cfg(feature = "rocm")]
impl MiopenContext {
    /// Create a fresh MIOpen handle. Returns `None` if `miopenCreate`
    /// fails — the caller (`RocmBackend::try_new`) treats that as "no
    /// MIOpen support" and degrades gracefully (the resident-MIOpen ops
    /// will fall through to whichever backend is next in the registry,
    /// or surface `Unsupported` if none is).
    fn try_new() -> Option<Self> {
        let mut handle: ffi::miopenHandle_t = std::ptr::null_mut();
        let status = unsafe { ffi::miopenCreate(&mut handle) };
        if status != ffi::MIOPEN_STATUS_SUCCESS || handle.is_null() {
            return None;
        }
        Some(Self {
            handle: std::sync::Mutex::new(handle),
            tensor_descs: std::sync::Mutex::new(std::collections::HashMap::new()),
            activation_descs: std::sync::Mutex::new(std::collections::HashMap::new()),
        })
    }

    /// Get-or-create a 4D tensor descriptor for the given NCHW shape.
    /// Returns the raw descriptor pointer, valid for the lifetime of
    /// the `MiopenContext`. Caller must NOT call
    /// `miopenDestroyTensorDescriptor` on it — the context owns it.
    fn tensor_4d(
        &self, n: i32, c: i32, h: i32, w: i32,
    ) -> Result<ffi::miopenTensorDescriptor_t, BackendError> {
        let key = (n, c, h, w);
        let mut map = self.tensor_descs.lock()
            .map_err(|_| BackendError::Runtime("miopen: tensor_descs mutex poisoned".into()))?;
        if let Some(desc) = map.get(&key) {
            return Ok(*desc);
        }
        let mut desc: ffi::miopenTensorDescriptor_t = std::ptr::null_mut();
        let st = unsafe { ffi::miopenCreateTensorDescriptor(&mut desc) };
        if st != ffi::MIOPEN_STATUS_SUCCESS {
            return Err(BackendError::Runtime(format!(
                "miopenCreateTensorDescriptor: status {st}"
            )));
        }
        let st = unsafe { ffi::miopenSet4dTensorDescriptor(desc, ffi::MIOPEN_DATA_FLOAT, n, c, h, w) };
        if st != ffi::MIOPEN_STATUS_SUCCESS {
            unsafe { ffi::miopenDestroyTensorDescriptor(desc); }
            return Err(BackendError::Runtime(format!(
                "miopenSet4dTensorDescriptor({n},{c},{h},{w}): status {st}"
            )));
        }
        map.insert(key, desc);
        Ok(desc)
    }

    /// Get-or-create an activation descriptor for the given mode.
    /// Alpha/beta/gamma are 0/0/0 for LOGISTIC, RELU; (1,1,1) for TANH
    /// (so the formula reduces to plain `tanh(x)`). Mode is the raw
    /// `miopenActivationMode_t` int (LOGISTIC=1, TANH=2, RELU=3, ...).
    fn activation_desc(
        &self, mode: i32,
    ) -> Result<ffi::miopenActivationDescriptor_t, BackendError> {
        let mut map = self.activation_descs.lock()
            .map_err(|_| BackendError::Runtime("miopen: activation_descs mutex poisoned".into()))?;
        if let Some(d) = map.get(&mode) { return Ok(*d); }
        let mut desc: ffi::miopenActivationDescriptor_t = std::ptr::null_mut();
        let st = unsafe { ffi::miopenCreateActivationDescriptor(&mut desc) };
        if st != ffi::MIOPEN_STATUS_SUCCESS {
            return Err(BackendError::Runtime(format!(
                "miopenCreateActivationDescriptor: status {st}"
            )));
        }
        // Alpha/beta/gamma defaults: LOGISTIC/RELU don't read them,
        // TANH wants alpha=beta=1 for plain tanh(x). Set 1/1/1 across
        // the board — extra params are ignored by the modes we expose.
        let st = unsafe { ffi::miopenSetActivationDescriptor(desc, mode, 1.0, 1.0, 1.0) };
        if st != ffi::MIOPEN_STATUS_SUCCESS {
            unsafe { ffi::miopenDestroyActivationDescriptor(desc); }
            return Err(BackendError::Runtime(format!(
                "miopenSetActivationDescriptor(mode={mode}): status {st}"
            )));
        }
        map.insert(mode, desc);
        Ok(desc)
    }
}

#[cfg(feature = "rocm")]
impl Drop for MiopenContext {
    fn drop(&mut self) {
        // Destroy descriptors first, then the handle. MIOpen would
        // tolerate the reverse order on most ROCm versions but the
        // documented contract is "destroy children before parent" —
        // matching it keeps us out of trouble on future ROCm bumps.
        if let Ok(mut m) = self.tensor_descs.lock() {
            for (_, d) in m.drain() {
                unsafe { ffi::miopenDestroyTensorDescriptor(d); }
            }
        }
        if let Ok(mut m) = self.activation_descs.lock() {
            for (_, d) in m.drain() {
                unsafe { ffi::miopenDestroyActivationDescriptor(d); }
            }
        }
        if let Ok(handle) = self.handle.lock() {
            if !handle.is_null() {
                unsafe { ffi::miopenDestroy(*handle); }
            }
        }
    }
}

// SAFETY: same argument as `RocmBackend` — MiopenContext holds only
// raw `*mut c_void` pointers, never mutated outside the per-field
// `Mutex`. MIOpen's handle is not thread-safe, but the Mutex
// serialises every access; the unsafe impls just tell Rust we've done
// the work.
#[cfg(feature = "rocm")]
unsafe impl Send for MiopenContext {}
#[cfg(feature = "rocm")]
unsafe impl Sync for MiopenContext {}

/// AMD ROCm backend. Holds a hipblas handle across ops; dropped cleanly
/// via `Drop`.
///
/// The handle is wrapped in a `Mutex` because hipBLAS handles are **not**
/// thread-safe per the ROCm 6.x docs — each host thread dispatching
/// concurrently needs its own handle. Serialising through one mutex
/// bounds multi-thread dispatch throughput but keeps correctness honest
/// until a handle pool lands. Today's caller (`BackendRegistry::dispatch`)
/// is sequential, so this is not a visible cost.
#[cfg(feature = "rocm")]
pub struct RocmBackend {
    handle: std::sync::Mutex<ffi::hipblasHandle_t>,
    /// MIOpen context (handle + cached descriptors). `None` when
    /// `miopenCreate` failed at probe time — the resident-MIOpen ops
    /// fall through to whichever backend is next in the registry, and
    /// every other dispatch path keeps working. In practice on this
    /// hardware MIOpen is always available alongside hipBLAS, but
    /// keeping it optional means no panic on machines that ship with
    /// rocm-libraries minus MIOpen.
    miopen: Option<MiopenContext>,
    /// Weight VRAM cache — mirrors the KFD `GpuQueue::weight_cache`
    /// pattern (see `kfd/dispatch_queue.rs`). Eliminates the per-dispatch
    /// `hipMalloc + hipMemcpy H2D + hipFree` triple that otherwise runs
    /// every forward pass for weights that never change within a step.
    ///
    /// Key: `(weight.as_ptr() as usize, weight.len())`. Value:
    /// `(content_fingerprint, vram_buffer)`.
    ///
    /// SAFETY / correctness:
    /// - Fingerprint is a 64-element sample hash of the weight buffer.
    ///   Cache hits are gated on fingerprint match, so in-place
    ///   optimizer updates that mutate the buffer automatically
    ///   invalidate the cached VRAM copy on the next dispatch. This
    ///   replaces the earlier pointer-only scheme that required
    ///   explicit `registry().invalidate_caches()` after every step —
    ///   modgrad-ctm training loops (mazes/minictm/dream_bench) did not
    ///   call it, producing the catastrophic divergence fixed in
    ///   commit 7f17f42 (ID first-step 37% → 9.5%).
    /// - Single entry per `(ptr, len)` slot: a fingerprint mismatch
    ///   replaces the prior entry (its `HipBuffer::drop` runs `hipFree`
    ///   so VRAM stays bounded to one buffer per live weight).
    /// - `invalidate_cache` still clears the whole map — callers that
    ///   *can* signal explicitly avoid the fingerprint check on first
    ///   post-step dispatch. Not required for correctness.
    /// - Wrapped in the same `Mutex` style as the hipBLAS handle; every
    ///   access locks first. The lock is held only during the
    ///   lookup/insert; dispatches do not hold it across the hipBLAS
    ///   call.
    cache: std::sync::Mutex<std::collections::HashMap<(usize, usize), (u64, HipBuffer)>>,
}

#[cfg(not(feature = "rocm"))]
pub struct RocmBackend {
    _unused: (),
}

impl RocmBackend {
    /// Probe the ROCm runtime. Returns `None` when either the `rocm`
    /// feature is disabled at compile time or no HIP device is present
    /// at runtime.
    pub fn try_new() -> Option<Self> {
        #[cfg(feature = "rocm")]
        {
            if !ffi::runtime_available() { return None; }
            let mut handle: ffi::hipblasHandle_t = std::ptr::null_mut();
            let status = unsafe { ffi::hipblasCreate(&mut handle) };
            if status != 0 { return None; }
            // MIOpen is best-effort — failing to create the handle
            // disables the resident-MIOpen ops without disabling
            // the whole backend. On this hardware (ROCm 6.x) the
            // probe should always succeed.
            let miopen = MiopenContext::try_new();
            Some(Self {
                handle: std::sync::Mutex::new(handle),
                miopen,
                cache: std::sync::Mutex::new(std::collections::HashMap::new()),
            })
        }
        #[cfg(not(feature = "rocm"))]
        {
            None
        }
    }
}

#[cfg(feature = "rocm")]
impl Drop for RocmBackend {
    fn drop(&mut self) {
        if let Ok(handle) = self.handle.lock() {
            if !handle.is_null() {
                unsafe { ffi::hipblasDestroy(*handle) };
            }
        }
    }
}

// SAFETY: `hipblasHandle_t` is a raw `*mut c_void`, so not Send/Sync by
// default. We wrap it in a `Mutex` above; every access locks first, and
// hipBLAS requires one thread at a time per handle anyway. The Mutex is
// the sync primitive; the unsafe impls just tell Rust we've done the work.
#[cfg(feature = "rocm")]
unsafe impl Send for RocmBackend {}
#[cfg(feature = "rocm")]
unsafe impl Sync for RocmBackend {}

/// True when `build.rs` successfully compiled and linked the custom
/// hipcc kernels (`kernels/rms_norm.hip`). Hosts without hipcc skip
/// the compile step in `build.rs`, the cfg stays unset, and this
/// returns `false` — `RocmBackend::supports(RmsNormResident)` then
/// returns false and the registry surfaces `Unsupported`.
#[cfg(feature = "rocm")]
#[inline]
fn rms_norm_kernel_present() -> bool {
    cfg!(modgrad_hipcc_kernels)
}

/// Bounds-check a `usize` dimension for the hipBLAS `int` FFI. Returns
/// a loud `Runtime` error rather than silently truncating — matters for
/// anyone who one day routes a 3B-param weight through here.
#[cfg(feature = "rocm")]
fn as_i32(dim: usize, name: &'static str) -> Result<i32, BackendError> {
    i32::try_from(dim).map_err(|_| {
        BackendError::Runtime(format!("rocm: {name}={dim} exceeds i32::MAX"))
    })
}

impl Backend for RocmBackend {
    fn name(&self) -> &'static str { "rocm" }

    fn device_info(&self) -> DeviceInfo {
        DeviceInfo {
            kind: DeviceKind::Rocm,
            name: "AMD ROCm (hipblas)".into(),
            total_mem_bytes: 0,
            arch: None, // filled in later via hipGetDeviceProperties
        }
    }

    fn supports(&self, op: &Op) -> bool {
        #[cfg(not(feature = "rocm"))]
        { let _ = op; return false; }
        #[cfg(feature = "rocm")]
        {
            // Gate on shape size: below ~64 elements per dim, hipBLAS
            // dispatch overhead (3× hipMalloc + 3× hipMemcpy + sgemv +
            // sync + 1× hipMemcpy back) dwarfs the actual compute, and
            // some tiny-shape configurations surface as silent zero-output
            // in practice (caught by isis-runtime's 8-region actor test
            // at obs_proj shapes like out_dim=16, in_dim=32). Conservative
            // size gate routes tiny matvecs to CPU; real training shapes
            // (out_dim >= 64) clear this and get the GPU path.
            match op {
                Op::Matvec { quant: QuantKind::F32, out_dim, in_dim, .. }
                    if *out_dim >= 64 && *in_dim >= 64 => true,
                // Resident dispatches skip ALL the per-call overhead
                // (no malloc / memcpy / cache lookup), so the size
                // gate is dropped — small ops are still profitable
                // when weights and activations are already on device.
                Op::MatvecResident { .. } => true,
                Op::MatmulResidentNN { .. }
                | Op::MatmulResidentNT { .. }
                | Op::MatmulResidentTN { .. } => true,
                // bf16 dispatches via hipblasGemmEx, which is part of
                // the same hipblas runtime — same supports() story as
                // the fp32 resident variants.
                Op::MatmulResidentBf16Nn { .. }
                | Op::MatmulResidentBf16Nt { .. }
                | Op::MatmulResidentBf16Tn { .. } => true,
                Op::MatvecResidentBf16 { .. } => true,
                // Custom hipcc kernel — only available when build.rs
                // successfully compiled `kernels/rms_norm.hip`. The
                // `rms_norm_kernel_present` constant is set from
                // build.rs via cfg, so a host without hipcc still
                // compiles (the symbol is gated to the same cfg as
                // the dispatch arm).
                Op::RmsNormResident { .. } => rms_norm_kernel_present(),
                // Q4_K_M dequant lives in the same hipcc-compiled
                // archive as RmsNormResident — same gate.
                Op::DequantQ4KResident { .. } => rms_norm_kernel_present(),
                Op::MatmulNN { m, k, n, .. }
                    if *m >= 64 && *k >= 64 && *n >= 64 => true,
                Op::MatmulNT { m, k, n, .. }
                    if *m >= 64 && *k >= 64 && *n >= 64 => true,
                Op::MatmulTN { m, k, n, .. }
                    if *m >= 64 && *k >= 64 && *n >= 64 => true,
                // MIOpen-backed resident ops — supported only when
                // the MIOpen handle was created successfully. Same
                // "no per-call malloc/memcpy" argument as
                // MatvecResident: small shapes are still profitable.
                Op::LayerNormResident { .. }
                | Op::SoftmaxResident { .. }
                | Op::ActivationResident { .. }
                | Op::GluResident { .. }
                | Op::OpTensorResident { .. }
                | Op::LayerNormBackwardResident { .. }
                | Op::SoftmaxBackwardResident { .. }
                | Op::ActivationBackwardResident { .. }
                | Op::GluBackwardResident { .. } => self.miopen.is_some(),
                _ => false,
            }
        }
    }

    fn invalidate_cache(&self) {
        // Drop every cached weight buffer. Each `HipBuffer::drop` calls
        // `hipFree`, so VRAM is released here rather than leaking to the
        // end of the process. Called by isis after every optimizer step
        // because AdamW mutates weights in place — see the cache field's
        // SAFETY note for the full invariant. When the rocm feature is
        // off the struct has no cache field and this is a no-op.
        #[cfg(feature = "rocm")]
        {
            if let Ok(mut c) = self.cache.lock() {
                c.clear();
            }
        }
    }

    fn dispatch(&self, op: &mut Op) -> Result<(), BackendError> {
        #[cfg(not(feature = "rocm"))]
        {
            let _ = op;
            Err(BackendError::Unsupported { op: "n/a", backend: "rocm" })
        }
        #[cfg(feature = "rocm")]
        {
            match op {
                Op::Matvec {
                    x, weight, bias, out,
                    out_dim, in_dim,
                    quant: QuantKind::F32,
                } => self.matvec_f32(x, weight, bias, out, *out_dim, *in_dim),
                Op::MatvecResident {
                    x_dev, weight_dev, bias_dev, out_dev,
                    out_dim, in_dim,
                } => self.matvec_resident_f32(
                    *x_dev as *const std::os::raw::c_void,
                    *weight_dev as *const std::os::raw::c_void,
                    *bias_dev as *const std::os::raw::c_void,
                    *out_dev as *mut std::os::raw::c_void,
                    *out_dim, *in_dim,
                ),
                Op::MatmulResidentNN {
                    a_dev, b_dev, out_dev, m, k, n,
                } => self.matmul_resident_nn_f32(
                    *a_dev, *b_dev, *out_dev, *m, *k, *n,
                ),
                Op::MatmulResidentNT {
                    a_dev, b_dev, out_dev, m, k, n,
                } => self.matmul_resident_nt_f32(
                    *a_dev, *b_dev, *out_dev, *m, *k, *n,
                ),
                Op::MatmulResidentTN {
                    a_dev, b_dev, out_dev, m, k, n,
                } => self.matmul_resident_tn_f32(
                    *a_dev, *b_dev, *out_dev, *m, *k, *n,
                ),
                Op::MatmulResidentBf16Nn {
                    a_dev, b_dev, c_dev, m, k, n, alpha, beta,
                } => self.matmul_resident_bf16_nn(
                    *a_dev, *b_dev, *c_dev, *m, *k, *n, *alpha, *beta,
                ),
                Op::MatmulResidentBf16Nt {
                    a_dev, b_dev, c_dev, m, k, n, alpha, beta,
                } => self.matmul_resident_bf16_nt(
                    *a_dev, *b_dev, *c_dev, *m, *k, *n, *alpha, *beta,
                ),
                Op::MatmulResidentBf16Tn {
                    a_dev, b_dev, c_dev, m, k, n, alpha, beta,
                } => self.matmul_resident_bf16_tn(
                    *a_dev, *b_dev, *c_dev, *m, *k, *n, *alpha, *beta,
                ),
                Op::MatvecResidentBf16 {
                    x_dev, weight_dev, bias_dev, out_dev,
                    out_dim, in_dim,
                } => self.matvec_resident_bf16(
                    *x_dev, *weight_dev, *bias_dev, *out_dev,
                    *out_dim, *in_dim,
                ),
                Op::RmsNormResident {
                    x_dev, weight_dev, y_dev, n, hidden, eps,
                } => self.rms_norm_resident_f32(
                    *x_dev, *weight_dev, *y_dev, *n, *hidden, *eps,
                ),
                Op::DequantQ4KResident {
                    q4k_dev, fp32_dev, n_blocks,
                } => self.dequant_q4k_resident_f32(
                    *q4k_dev, *fp32_dev, *n_blocks,
                ),
                Op::MatmulNN {
                    a, b, out, bias, m, k, n,
                } => {
                    self.matmul_nn_f32(a, b, out, *m, *k, *n)?;
                    // hipBLAS sgemm doesn't bias — apply on host.
                    // Cost: one pass over `out` (already resident on host
                    // after the download). Negligible relative to the
                    // GEMM itself.
                    if let Some(bias) = bias {
                        for r in 0..*m {
                            let row = &mut out[r * *n..(r + 1) * *n];
                            for c in 0..*n { row[c] += bias[c]; }
                        }
                    }
                    Ok(())
                },
                Op::MatmulNT {
                    a, b, out, bias, m, k, n,
                } => {
                    self.matmul_nt_f32(a, b, out, *m, *k, *n)?;
                    if let Some(bias) = bias {
                        for r in 0..*m {
                            let row = &mut out[r * *n..(r + 1) * *n];
                            for c in 0..*n { row[c] += bias[c]; }
                        }
                    }
                    Ok(())
                },
                Op::MatmulTN {
                    a, b, out, bias, m, k, n,
                } => {
                    self.matmul_tn_f32(a, b, out, *m, *k, *n)?;
                    if let Some(bias) = bias {
                        for r in 0..*m {
                            let row = &mut out[r * *n..(r + 1) * *n];
                            for c in 0..*n { row[c] += bias[c]; }
                        }
                    }
                    Ok(())
                },
                Op::LayerNormResident {
                    x_dev, weight_dev, bias_dev, y_dev,
                    n, normalized_size, epsilon,
                } => self.layer_norm_resident_f32(
                    *x_dev, *weight_dev, *bias_dev, *y_dev,
                    *n, *normalized_size, *epsilon,
                ),
                Op::SoftmaxResident {
                    x_dev, y_dev, n_rows, row_len, log,
                } => self.softmax_resident_f32(
                    *x_dev, *y_dev, *n_rows, *row_len, *log,
                ),
                Op::ActivationResident {
                    x_dev, y_dev, n, mode,
                } => self.activation_resident_f32(
                    *x_dev, *y_dev, *n, *mode,
                ),
                Op::GluResident {
                    x_dev, y_dev, n_rows, half_size,
                } => self.glu_resident_f32(
                    *x_dev, *y_dev, *n_rows, *half_size,
                ),
                Op::OpTensorResident {
                    a_dev, b_dev, c_dev, n,
                    alpha1, alpha2, beta, op: kind,
                } => self.op_tensor_resident_f32(
                    *a_dev, *b_dev, *c_dev, *n,
                    *alpha1, *alpha2, *beta, *kind,
                ),
                Op::LayerNormBackwardResident {
                    x_dev, dy_dev, weight_dev, mean_dev, rstd_dev,
                    dx_dev, dweight_dev, dbias_dev,
                    n, normalized_size,
                } => self.layer_norm_backward_resident_f32(
                    *x_dev, *dy_dev, *weight_dev, *mean_dev, *rstd_dev,
                    *dx_dev, *dweight_dev, *dbias_dev,
                    *n, *normalized_size,
                ),
                Op::SoftmaxBackwardResident {
                    y_dev, dy_dev, dx_dev, n_rows, row_len, log,
                } => self.softmax_backward_resident_f32(
                    *y_dev, *dy_dev, *dx_dev, *n_rows, *row_len, *log,
                ),
                Op::ActivationBackwardResident {
                    x_dev, y_dev, dy_dev, dx_dev, n, mode,
                } => self.activation_backward_resident_f32(
                    *x_dev, *y_dev, *dy_dev, *dx_dev, *n, *mode,
                ),
                Op::GluBackwardResident {
                    x_dev, dy_dev, dx_dev, n_rows, half_size,
                } => self.glu_backward_resident_f32(
                    *x_dev, *dy_dev, *dx_dev, *n_rows, *half_size,
                ),
                _ => Err(BackendError::Unsupported {
                    op: op.name(),
                    backend: "rocm",
                }),
            }
        }
    }
}

/// Sample-based content fingerprint of a weight buffer. Captures
/// enough of the buffer that a dense AdamW in-place update will
/// produce a different value with overwhelming probability, while
/// staying O(1) in the buffer size.
///
/// Takes 64 evenly-spaced samples (or every element for n <= 64).
/// Collision probability over a realistic training run is negligible:
/// every AdamW step touches every weight, so every sampled position
/// changes too.
#[cfg(feature = "rocm")]
fn weight_fingerprint(w: &[f32]) -> u64 {
    const SAMPLES: usize = 64;
    let n = w.len();
    let mut hash: u64 = n as u64;
    if n == 0 { return hash; }
    if n <= SAMPLES {
        for &v in w {
            hash = hash.rotate_left(13) ^ (v.to_bits() as u64);
        }
    } else {
        let stride = n / SAMPLES;
        for i in 0..SAMPLES {
            // SAFETY: i < SAMPLES ≤ n/stride, so i*stride < n.
            let v = unsafe { *w.get_unchecked(i * stride) };
            hash = hash.rotate_left(13) ^ (v.to_bits() as u64);
        }
    }
    hash
}

#[cfg(feature = "rocm")]
impl RocmBackend {
    /// Look up the weight VRAM mirror, or upload it if the fingerprint
    /// mismatches (or the entry is missing). Returns the cached device
    /// pointer.
    ///
    /// Fingerprint-based invalidation: in-place optimizer updates are
    /// detected on the next dispatch without needing an explicit
    /// `invalidate_caches()` call from the training loop. See the
    /// `cache` field's SAFETY note for why this is the correctness
    /// fence, and commit 7f17f42 for the divergence it prevents.
    ///
    /// The pointer remains valid after the lock is released: on a hit
    /// the entry isn't touched; on a miss the new entry owns the
    /// `HipBuffer`, and no concurrent dispatch can evict it because the
    /// hipBLAS handle is serialised through its own mutex.
    fn cached_weight_ptr(&self, weight: &[f32]) -> Result<*mut f32, BackendError> {
        let fingerprint = weight_fingerprint(weight);
        let key = (weight.as_ptr() as usize, weight.len());
        let mut cache = self.cache.lock()
            .map_err(|_| BackendError::Runtime("rocm: cache mutex poisoned".into()))?;
        if let Some((cached_fp, buf)) = cache.get(&key) {
            if *cached_fp == fingerprint {
                return Ok(buf.as_f32_ptr());
            }
        }
        // Miss or fingerprint mismatch: upload fresh. Dropping the old
        // HipBuffer on replace runs hipFree automatically so VRAM stays
        // bounded to one entry per live weight slot.
        let buf = HipBuffer::alloc(weight.len() * 4)?;
        buf.upload_f32(weight)?;
        let ptr = buf.as_f32_ptr();
        cache.insert(key, (fingerprint, buf));
        Ok(ptr)
    }

    /// y = W @ x + bias, f32, via hipblasSgemv.
    /// W is row-major [out_dim × in_dim]; hipBLAS is column-major so we
    /// compute W^T @ x with hipblas + seed y with bias.
    ///
    /// HipBuffer handles are dropped automatically on error via RAII,
    /// so the `?` operator is safe here — no manual cleanup needed.
    /// Weight buffers live in the VRAM cache and persist until
    /// `invalidate_cache` clears them.
    fn matvec_f32(
        &self,
        x: &[f32],
        weight: &[f32],
        bias: &[f32],
        out: &mut [f32],
        out_dim: usize,
        in_dim: usize,
    ) -> Result<(), BackendError> {
        debug_assert_eq!(bias.len(), out_dim, "rocm matvec: bias len must equal out_dim");
        debug_assert_eq!(weight.len(), out_dim * in_dim, "rocm matvec: weight len mismatch");
        debug_assert!(x.len() >= in_dim, "rocm matvec: x shorter than in_dim");
        debug_assert!(out.len() >= out_dim, "rocm matvec: out shorter than out_dim");

        let in_dim_i32 = as_i32(in_dim, "in_dim")?;
        let out_dim_i32 = as_i32(out_dim, "out_dim")?;

        // Weight reuses the VRAM cache — keyed on pointer identity so
        // hot training loops amortise the upload cost across steps.
        let d_w_ptr = self.cached_weight_ptr(weight)?;
        let d_x = HipBuffer::alloc(x.len() * 4)?;
        let d_y = HipBuffer::alloc(out_dim * 4)?;

        d_x.upload_f32(x)?;
        d_y.upload_f32(bias)?; // seed y with bias; SGEMV adds into it.

        // hipBLAS is column-major. Our W is row-major [out_dim × in_dim],
        // which reads as column-major [in_dim × out_dim]. TRANS=T gives
        // W_col^T @ x = our row-major W @ x.
        let alpha: f32 = 1.0;
        let beta: f32 = 1.0;
        let handle = self.handle.lock()
            .map_err(|_| BackendError::Runtime("rocm: hipblas handle poisoned".into()))?;
        let status = unsafe {
            ffi::hipblasSgemv(
                *handle,
                ffi::HIPBLAS_OP_T,
                in_dim_i32,
                out_dim_i32,
                &alpha,
                d_w_ptr,
                in_dim_i32,
                d_x.as_f32_ptr(),
                1,
                &beta,
                d_y.as_f32_ptr(),
                1,
            )
        };
        drop(handle);
        if status != 0 {
            return Err(BackendError::Runtime(format!("hipblasSgemv: status {status}")));
        }

        // The following `hipMemcpy` D2H is synchronous (blocking), and
        // it orders against prior default-stream work — so hipblasSgemv
        // is guaranteed to complete before the copy reads from d_y.
        // An explicit hipDeviceSynchronize() here would add another
        // host↔driver round-trip for no safety gain.
        d_y.download_f32(out)
    }

    /// Fully-resident matvec: every operand is a hip device pointer.
    /// Zero PCIe transfers in this call — the whole point. Steps:
    ///   1. D2D copy bias_dev → out_dev (hipMemcpy DEVICE_TO_DEVICE,
    ///      so out_dev = bias as the SGEMV initial y).
    ///   2. hipblasSgemv with weight, x, out — beta = 1.0 means the
    ///      result accumulates into out (= bias + W·x).
    /// Output stays on device; caller decides when (if ever) to
    /// download.
    fn matvec_resident_f32(
        &self,
        x_dev: *const std::os::raw::c_void,
        weight_dev: *const std::os::raw::c_void,
        bias_dev: *const std::os::raw::c_void,
        out_dev: *mut std::os::raw::c_void,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<(), BackendError> {
        let in_dim_i32 = as_i32(in_dim, "in_dim")?;
        let out_dim_i32 = as_i32(out_dim, "out_dim")?;

        // Step 1: D2D copy bias → out (initialises y for the
        // accumulating SGEMV). DEVICE_TO_DEVICE = 3 in the hip enum.
        const HIP_MEMCPY_DEVICE_TO_DEVICE: std::os::raw::c_int = 3;
        let bytes = out_dim * 4;
        let err = unsafe {
            ffi::hipMemcpy(out_dev, bias_dev, bytes, HIP_MEMCPY_DEVICE_TO_DEVICE)
        };
        if err != 0 {
            return Err(BackendError::Runtime(format!(
                "hipMemcpy D2D bias→out: {}", ffi::hip_err_str(err)
            )));
        }

        // Step 2: SGEMV. Same column-major-via-transpose trick as
        // `matvec_f32`. Weight is row-major [out_dim × in_dim], read
        // as column-major [in_dim × out_dim]; TRANS=T gives the
        // row-major matvec we want.
        let alpha: f32 = 1.0;
        let beta: f32 = 1.0;
        let handle = self.handle.lock()
            .map_err(|_| BackendError::Runtime("rocm: hipblas handle poisoned".into()))?;
        let status = unsafe {
            ffi::hipblasSgemv(
                *handle,
                ffi::HIPBLAS_OP_T,
                in_dim_i32,
                out_dim_i32,
                &alpha,
                weight_dev as *const f32,
                in_dim_i32,
                x_dev as *const f32,
                1,
                &beta,
                out_dev as *mut f32,
                1,
            )
        };
        drop(handle);
        if status != 0 {
            return Err(BackendError::Runtime(format!(
                "hipblasSgemv (resident): status {status}"
            )));
        }
        Ok(())
    }
}

#[cfg(feature = "rocm")]
impl RocmBackend {
    /// C[m×n] = A[m×k] @ B[k×n], all f32 row-major.
    ///
    /// Trick to use column-major hipBLAS on row-major data: compute
    /// C^T = B^T @ A^T by passing B as the first arg and A as the
    /// second, with no explicit transpose. Row-major C[m×n] is the
    /// same memory as col-major (C^T)[n×m].
    fn matmul_nn_f32(
        &self,
        a: &[f32],
        b: &[f32],
        out: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(), BackendError> {
        debug_assert_eq!(a.len(), m * k, "rocm matmul: a len mismatch");
        debug_assert_eq!(b.len(), k * n, "rocm matmul: b len mismatch");
        debug_assert!(out.len() >= m * n, "rocm matmul: out too small");

        let m_i32 = as_i32(m, "m")?;
        let k_i32 = as_i32(k, "k")?;
        let n_i32 = as_i32(n, "n")?;

        // `a` is the weight-shaped operand in training dispatch — reuse
        // the VRAM cache. `b` is typically the freshly-materialised input
        // activation, which wouldn't cache-hit, so we keep it on the
        // per-dispatch alloc/upload/free path.
        let d_a_ptr = self.cached_weight_ptr(a)?;
        let d_b = HipBuffer::alloc(k * n * 4)?;
        let d_c = HipBuffer::alloc(m * n * 4)?;

        d_b.upload_f32(b)?;

        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let handle = self.handle.lock()
            .map_err(|_| BackendError::Runtime("rocm: hipblas handle poisoned".into()))?;
        let status = unsafe {
            ffi::hipblasSgemm(
                *handle,
                ffi::HIPBLAS_OP_N, ffi::HIPBLAS_OP_N,
                n_i32, m_i32, k_i32,
                &alpha,
                d_b.as_f32_ptr(), n_i32,
                d_a_ptr, k_i32,
                &beta,
                d_c.as_f32_ptr(), n_i32,
            )
        };
        drop(handle);
        if status != 0 {
            return Err(BackendError::Runtime(format!("hipblasSgemm: status {status}")));
        }

        // `hipMemcpy` D2H is synchronous and orders against prior
        // default-stream work; hipblasSgemm is guaranteed to complete
        // before the copy runs.  Skipping the explicit
        // `hipDeviceSynchronize` trims one host↔driver round-trip per
        // dispatch with no correctness loss.
        d_c.download_f32(out)
    }

    /// `C[m×n] = A[m×k] @ B[n×k]^T`, row-major. Both operands are
    /// treated as per-dispatch ephemerals (neither is assumed to be a
    /// stable weight buffer, matching how NT is typically called from
    /// gradient computations — `dW = activation @ grad^T`). If a caller
    /// wants the cache for a weight-like arg here, adapt similarly to
    /// `matmul_nn_f32`'s `cached_weight_ptr` usage.
    ///
    /// Derivation: row-major C[m,n] has the same bytes as col-major
    /// C^T[n,m]. Computing C^T = B · A^T in column-major via hipBLAS
    /// reduces to SGEMM(T, N, n, m, k, B, lda=k, A, ldb=k, C, ldc=n) —
    /// the T on the B-first arg reinterprets B's row-major [n,k]
    /// storage (= col-major [k,n]) as its transpose [n,k] for the
    /// multiply.
    fn matmul_nt_f32(
        &self, a: &[f32], b: &[f32], out: &mut [f32],
        m: usize, k: usize, n: usize,
    ) -> Result<(), BackendError> {
        debug_assert_eq!(a.len(), m * k, "rocm matmul_nt: a len mismatch");
        debug_assert_eq!(b.len(), n * k, "rocm matmul_nt: b len mismatch");
        debug_assert!(out.len() >= m * n, "rocm matmul_nt: out too small");

        let m_i32 = as_i32(m, "m")?;
        let k_i32 = as_i32(k, "k")?;
        let n_i32 = as_i32(n, "n")?;

        let d_a = HipBuffer::alloc(m * k * 4)?;
        let d_b = HipBuffer::alloc(n * k * 4)?;
        let d_c = HipBuffer::alloc(m * n * 4)?;
        d_a.upload_f32(a)?;
        d_b.upload_f32(b)?;

        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let handle = self.handle.lock()
            .map_err(|_| BackendError::Runtime("rocm: hipblas handle poisoned".into()))?;
        let status = unsafe {
            ffi::hipblasSgemm(
                *handle,
                ffi::HIPBLAS_OP_T, ffi::HIPBLAS_OP_N,
                n_i32, m_i32, k_i32,
                &alpha,
                d_b.as_f32_ptr(), k_i32,
                d_a.as_f32_ptr(), k_i32,
                &beta,
                d_c.as_f32_ptr(), n_i32,
            )
        };
        drop(handle);
        if status != 0 {
            return Err(BackendError::Runtime(format!("hipblasSgemm (NT): status {status}")));
        }
        d_c.download_f32(out)
    }

    /// `C[m×n] = A[k×m]^T @ B[k×n]`, row-major. `a` is assumed to be
    /// the weight-shaped operand (same convention as `matmul_nn_f32`) —
    /// reuses the weight cache. `b` is uploaded fresh per dispatch.
    ///
    /// Derivation: same trick as NN/NT; here the transpose flag is on
    /// the second SGEMM arg so A's row-major [k,m] storage
    /// (= col-major [m,k]) is reinterpreted as [k,m] for the multiply.
    /// Result: SGEMM(N, T, n, m, k, B, lda=n, A, ldb=m, C, ldc=n).
    fn matmul_tn_f32(
        &self, a: &[f32], b: &[f32], out: &mut [f32],
        m: usize, k: usize, n: usize,
    ) -> Result<(), BackendError> {
        debug_assert_eq!(a.len(), k * m, "rocm matmul_tn: a len mismatch");
        debug_assert_eq!(b.len(), k * n, "rocm matmul_tn: b len mismatch");
        debug_assert!(out.len() >= m * n, "rocm matmul_tn: out too small");

        let m_i32 = as_i32(m, "m")?;
        let k_i32 = as_i32(k, "k")?;
        let n_i32 = as_i32(n, "n")?;

        let d_a_ptr = self.cached_weight_ptr(a)?;
        let d_b = HipBuffer::alloc(k * n * 4)?;
        let d_c = HipBuffer::alloc(m * n * 4)?;
        d_b.upload_f32(b)?;

        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let handle = self.handle.lock()
            .map_err(|_| BackendError::Runtime("rocm: hipblas handle poisoned".into()))?;
        let status = unsafe {
            ffi::hipblasSgemm(
                *handle,
                ffi::HIPBLAS_OP_N, ffi::HIPBLAS_OP_T,
                n_i32, m_i32, k_i32,
                &alpha,
                d_b.as_f32_ptr(), n_i32,
                d_a_ptr, m_i32,
                &beta,
                d_c.as_f32_ptr(), n_i32,
            )
        };
        drop(handle);
        if status != 0 {
            return Err(BackendError::Runtime(format!("hipblasSgemm (TN): status {status}")));
        }
        d_c.download_f32(out)
    }
}

#[cfg(feature = "rocm")]
impl RocmBackend {
    /// Fully-resident `C = A @ B`, all f32 row-major, every operand a
    /// hip device pointer. Mirrors `matmul_nn_f32`'s column-major-via-
    /// transpose trick: hipBLAS is column-major, so we compute
    /// `C^T = B^T @ A^T` by passing B as the first arg and A as the
    /// second. No transposes, no per-call malloc/upload/free.
    fn matmul_resident_nn_f32(
        &self,
        a_dev: *const f32,
        b_dev: *const f32,
        out_dev: *mut f32,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(), BackendError> {
        let m_i32 = as_i32(m, "m")?;
        let k_i32 = as_i32(k, "k")?;
        let n_i32 = as_i32(n, "n")?;

        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let handle = self.handle.lock()
            .map_err(|_| BackendError::Runtime("rocm: hipblas handle poisoned".into()))?;
        let status = unsafe {
            ffi::hipblasSgemm(
                *handle,
                ffi::HIPBLAS_OP_N, ffi::HIPBLAS_OP_N,
                n_i32, m_i32, k_i32,
                &alpha,
                b_dev, n_i32,
                a_dev, k_i32,
                &beta,
                out_dev, n_i32,
            )
        };
        drop(handle);
        if status != 0 {
            return Err(BackendError::Runtime(format!(
                "hipblasSgemm (resident NN): status {status}"
            )));
        }
        Ok(())
    }

    /// Fully-resident `C = A @ B^T`, row-major. See `matmul_nt_f32`
    /// for the trick — same transposition story, but operands are
    /// device pointers and we don't touch host memory.
    fn matmul_resident_nt_f32(
        &self,
        a_dev: *const f32,
        b_dev: *const f32,
        out_dev: *mut f32,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(), BackendError> {
        let m_i32 = as_i32(m, "m")?;
        let k_i32 = as_i32(k, "k")?;
        let n_i32 = as_i32(n, "n")?;

        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let handle = self.handle.lock()
            .map_err(|_| BackendError::Runtime("rocm: hipblas handle poisoned".into()))?;
        let status = unsafe {
            ffi::hipblasSgemm(
                *handle,
                ffi::HIPBLAS_OP_T, ffi::HIPBLAS_OP_N,
                n_i32, m_i32, k_i32,
                &alpha,
                b_dev, k_i32,
                a_dev, k_i32,
                &beta,
                out_dev, n_i32,
            )
        };
        drop(handle);
        if status != 0 {
            return Err(BackendError::Runtime(format!(
                "hipblasSgemm (resident NT): status {status}"
            )));
        }
        Ok(())
    }

    /// Fully-resident `C = A^T @ B`, row-major. See `matmul_tn_f32`
    /// for the derivation.
    fn matmul_resident_tn_f32(
        &self,
        a_dev: *const f32,
        b_dev: *const f32,
        out_dev: *mut f32,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(), BackendError> {
        let m_i32 = as_i32(m, "m")?;
        let k_i32 = as_i32(k, "k")?;
        let n_i32 = as_i32(n, "n")?;

        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let handle = self.handle.lock()
            .map_err(|_| BackendError::Runtime("rocm: hipblas handle poisoned".into()))?;
        let status = unsafe {
            ffi::hipblasSgemm(
                *handle,
                ffi::HIPBLAS_OP_N, ffi::HIPBLAS_OP_T,
                n_i32, m_i32, k_i32,
                &alpha,
                b_dev, n_i32,
                a_dev, m_i32,
                &beta,
                out_dev, n_i32,
            )
        };
        drop(handle);
        if status != 0 {
            return Err(BackendError::Runtime(format!(
                "hipblasSgemm (resident TN): status {status}"
            )));
        }
        Ok(())
    }

    /// Fully-resident bf16 matmul: `C = A @ B` with fp32 accumulate.
    /// All operands are bf16 stored as `u16`. Same column-major-via-
    /// transpose trick as `matmul_resident_nn_f32`: hipBLAS is
    /// column-major, so we compute `C^T = B^T @ A^T` by passing B as
    /// the first arg and A as the second.
    #[allow(clippy::too_many_arguments)]
    fn matmul_resident_bf16_nn(
        &self,
        a_dev: *const u16,
        b_dev: *const u16,
        c_dev: *mut u16,
        m: usize,
        k: usize,
        n: usize,
        alpha: f32,
        beta: f32,
    ) -> Result<(), BackendError> {
        let m_i32 = as_i32(m, "m")?;
        let k_i32 = as_i32(k, "k")?;
        let n_i32 = as_i32(n, "n")?;
        self.gemm_ex_bf16(
            ffi::HIPBLAS_OP_N, ffi::HIPBLAS_OP_N,
            n_i32, m_i32, k_i32,
            alpha, beta,
            b_dev, n_i32,
            a_dev, k_i32,
            c_dev, n_i32,
            "matmul_resident_bf16_nn",
        )
    }

    /// Fully-resident bf16 matmul: `C = A @ B^T`. See
    /// `matmul_resident_nt_f32` for the derivation.
    #[allow(clippy::too_many_arguments)]
    fn matmul_resident_bf16_nt(
        &self,
        a_dev: *const u16,
        b_dev: *const u16,
        c_dev: *mut u16,
        m: usize,
        k: usize,
        n: usize,
        alpha: f32,
        beta: f32,
    ) -> Result<(), BackendError> {
        let m_i32 = as_i32(m, "m")?;
        let k_i32 = as_i32(k, "k")?;
        let n_i32 = as_i32(n, "n")?;
        self.gemm_ex_bf16(
            ffi::HIPBLAS_OP_T, ffi::HIPBLAS_OP_N,
            n_i32, m_i32, k_i32,
            alpha, beta,
            b_dev, k_i32,
            a_dev, k_i32,
            c_dev, n_i32,
            "matmul_resident_bf16_nt",
        )
    }

    /// Fully-resident bf16 matmul: `C = A^T @ B`. See
    /// `matmul_resident_tn_f32` for the derivation.
    #[allow(clippy::too_many_arguments)]
    fn matmul_resident_bf16_tn(
        &self,
        a_dev: *const u16,
        b_dev: *const u16,
        c_dev: *mut u16,
        m: usize,
        k: usize,
        n: usize,
        alpha: f32,
        beta: f32,
    ) -> Result<(), BackendError> {
        let m_i32 = as_i32(m, "m")?;
        let k_i32 = as_i32(k, "k")?;
        let n_i32 = as_i32(n, "n")?;
        self.gemm_ex_bf16(
            ffi::HIPBLAS_OP_N, ffi::HIPBLAS_OP_T,
            n_i32, m_i32, k_i32,
            alpha, beta,
            b_dev, n_i32,
            a_dev, m_i32,
            c_dev, n_i32,
            "matmul_resident_bf16_tn",
        )
    }

    /// Internal: route a single hipblasGemmEx call with bf16 inputs and
    /// fp32 compute. Centralises the alpha/beta handle-lock dance so the
    /// three NN/NT/TN wrappers stay almost-trivial.
    ///
    /// `op_label` is included in the error string so failures point at
    /// the originating dispatch rather than this helper.
    #[allow(clippy::too_many_arguments)]
    fn gemm_ex_bf16(
        &self,
        transa: std::os::raw::c_int,
        transb: std::os::raw::c_int,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        a: *const u16,
        lda: i32,
        b: *const u16,
        ldb: i32,
        c: *mut u16,
        ldc: i32,
        op_label: &'static str,
    ) -> Result<(), BackendError> {
        let handle = self.handle.lock()
            .map_err(|_| BackendError::Runtime("rocm: hipblas handle poisoned".into()))?;
        let status = unsafe {
            ffi::hipblasGemmEx(
                *handle,
                transa, transb,
                m, n, k,
                &alpha as *const f32 as *const std::os::raw::c_void,
                a as *const std::os::raw::c_void, ffi::HIP_R_16BF, lda,
                b as *const std::os::raw::c_void, ffi::HIP_R_16BF, ldb,
                &beta as *const f32 as *const std::os::raw::c_void,
                c as *mut std::os::raw::c_void, ffi::HIP_R_16BF, ldc,
                ffi::HIPBLAS_COMPUTE_32F,
                ffi::HIPBLAS_GEMM_DEFAULT,
            )
        };
        drop(handle);
        if status != 0 {
            return Err(BackendError::Runtime(format!(
                "hipblasGemmEx ({op_label}): status {status}"
            )));
        }
        Ok(())
    }

    /// Fully-resident bf16 matvec: `out = weight @ x + bias`, all
    /// bf16 (`u16`), fp32 accumulate. Steps mirror `matvec_resident_f32`:
    ///   1. D2D copy `bias_dev` → `out_dev` so out_dev seeds with bias.
    ///   2. hipblasGemmEx with weight, x, out — `beta = 1.0` accumulates
    ///      `W·x` into the bias-seeded out.
    ///
    /// Implementation note: we issue this as a 1-column GEMM rather
    /// than a hypothetical hipblasGemvEx because the matvec dimension
    /// for our shapes is small enough that the GEMM overhead doesn't
    /// matter, and hipblas's matvec-Ex path is not in the hipblas
    /// runtime we link against.
    fn matvec_resident_bf16(
        &self,
        x_dev: *const u16,
        weight_dev: *const u16,
        bias_dev: *const u16,
        out_dev: *mut u16,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<(), BackendError> {
        let in_dim_i32 = as_i32(in_dim, "in_dim")?;
        let out_dim_i32 = as_i32(out_dim, "out_dim")?;

        // Step 1: D2D copy bias → out (bf16 = 2 bytes per elem).
        const HIP_MEMCPY_DEVICE_TO_DEVICE: std::os::raw::c_int = 3;
        let bytes = out_dim * 2;
        let err = unsafe {
            ffi::hipMemcpy(
                out_dev as *mut std::os::raw::c_void,
                bias_dev as *const std::os::raw::c_void,
                bytes, HIP_MEMCPY_DEVICE_TO_DEVICE,
            )
        };
        if err != 0 {
            return Err(BackendError::Runtime(format!(
                "hipMemcpy D2D bias→out (bf16): {}", ffi::hip_err_str(err)
            )));
        }

        // Step 2: 1-column GEMM. Same column-major-via-transpose trick
        // as `matvec_resident_f32`: weight is row-major
        // [out_dim × in_dim], read as col-major [in_dim × out_dim];
        // transposing the first operand recovers the row-major matvec
        // we want. We pose this as `C^T = W · x` of shape [out_dim × 1]:
        //   GemmEx(N, N, m=out_dim, n=1, k=in_dim,
        //          A = weight (transposed in op-order to get col-major
        //              [out_dim × in_dim] semantics), lda=in_dim,
        //          B = x, ldb=in_dim,
        //          C = out, ldc=out_dim).
        // Actually we follow the SGEMV pattern: hipblasSgemv(T, ...)
        // does the transpose; the GemmEx equivalent is to declare
        // A as [in_dim × out_dim] col-major and transpose it.
        let alpha: f32 = 1.0;
        let beta: f32 = 1.0;
        self.gemm_ex_bf16(
            ffi::HIPBLAS_OP_T, ffi::HIPBLAS_OP_N,
            out_dim_i32, 1, in_dim_i32,
            alpha, beta,
            weight_dev, in_dim_i32,
            x_dev, in_dim_i32,
            out_dev, out_dim_i32,
            "matvec_resident_bf16",
        )
    }

    /// Resident RMSNorm forward via the custom hipcc kernel built by
    /// `build.rs`. Modern LLM normalisation:
    ///
    ///   y[r, c] = x[r, c] / sqrt(mean(x[r, :]^2) + eps) * weight[c]
    ///
    /// MIOpen does not ship RMSNorm — only LayerNorm, which subtracts
    /// the mean and adds bias. Hence the custom kernel.
    ///
    /// Caller invariant: `supports()` already confirmed
    /// `rms_norm_kernel_present()`. We re-check here so that a
    /// supports/dispatch race surfaces as a loud `Unsupported` rather
    /// than a missing-symbol link error at runtime.
    #[cfg(modgrad_hipcc_kernels)]
    fn rms_norm_resident_f32(
        &self,
        x_dev: *const f32,
        weight_dev: *const f32,
        y_dev: *mut f32,
        n: usize,
        hidden: usize,
        eps: f32,
    ) -> Result<(), BackendError> {
        let n_i32 = as_i32(n, "n")?;
        let hidden_i32 = as_i32(hidden, "hidden")?;
        let err = unsafe {
            ffi::launch_rms_norm(x_dev, weight_dev, y_dev, n_i32, hidden_i32, eps)
        };
        if err != 0 {
            return Err(BackendError::Runtime(format!(
                "launch_rms_norm(n={n}, hidden={hidden}): {}",
                ffi::hip_err_str(err),
            )));
        }
        Ok(())
    }

    /// Fallback when the hipcc-built kernel isn't present (host without
    /// hipcc, build.rs skipped the compile). The `supports()` arm
    /// returns false in this configuration so the registry never
    /// dispatches here, but we keep the symbol so `dispatch()` still
    /// matches every `Op` variant.
    #[cfg(not(modgrad_hipcc_kernels))]
    #[allow(clippy::too_many_arguments)]
    fn rms_norm_resident_f32(
        &self,
        _x_dev: *const f32,
        _weight_dev: *const f32,
        _y_dev: *mut f32,
        _n: usize,
        _hidden: usize,
        _eps: f32,
    ) -> Result<(), BackendError> {
        Err(BackendError::Unsupported {
            op: "rms_norm_resident",
            backend: "rocm",
        })
    }

    /// Dispatch the Q4_K_M dequantize kernel. `q4k_dev` is a
    /// hip-device byte pointer covering `n_blocks * 144` bytes;
    /// `fp32_dev` receives `n_blocks * 256` fp32 values. The kernel
    /// is launched with one workgroup per block (256 threads), so
    /// each block reads the 16-byte header into shared memory once
    /// and the per-thread work is a single nibble extract + scale.
    ///
    /// Caller invariant: `supports()` already confirmed the hipcc
    /// archive is present. Re-checking here surfaces a
    /// supports/dispatch race as a loud `Unsupported` rather than a
    /// link-time missing-symbol crash.
    #[cfg(modgrad_hipcc_kernels)]
    fn dequant_q4k_resident_f32(
        &self,
        q4k_dev: *const u8,
        fp32_dev: *mut f32,
        n_blocks: usize,
    ) -> Result<(), BackendError> {
        let n_i32 = as_i32(n_blocks, "n_blocks")?;
        let err = unsafe {
            ffi::launch_dequant_q4k(q4k_dev, n_i32, fp32_dev)
        };
        if err != 0 {
            return Err(BackendError::Runtime(format!(
                "launch_dequant_q4k(n_blocks={n_blocks}): {}",
                ffi::hip_err_str(err),
            )));
        }
        Ok(())
    }

    #[cfg(not(modgrad_hipcc_kernels))]
    fn dequant_q4k_resident_f32(
        &self,
        _q4k_dev: *const u8,
        _fp32_dev: *mut f32,
        _n_blocks: usize,
    ) -> Result<(), BackendError> {
        Err(BackendError::Unsupported {
            op: "dequant_q4k_resident",
            backend: "rocm",
        })
    }
}

#[cfg(feature = "rocm")]
impl RocmBackend {
    /// Locked accessor for the MIOpen context. Resident-MIOpen
    /// dispatchers all start by calling this; if MIOpen wasn't
    /// available at probe time it surfaces here as a `Runtime` error
    /// (`supports()` returns false in that case so the registry
    /// shouldn't pick this backend, but defending against a
    /// supports/dispatch race is cheap).
    fn miopen(&self) -> Result<&MiopenContext, BackendError> {
        self.miopen.as_ref().ok_or_else(|| {
            BackendError::Runtime(
                "rocm: MIOpen handle not available — re-probe failed at startup".into(),
            )
        })
    }

    /// Resident LayerNorm: `y = ((x - mean) * rstd) * weight + bias`,
    /// row-wise over `n` rows of length `normalized_size`.
    ///
    /// MIOpen's `miopenLayerNormForward` always allocates per-row
    /// `mean` and `rstd` outputs. We can't elide them — the API
    /// requires both pointers to be non-null — so we burn two
    /// `hipMalloc`s per dispatch for them. Fixing that needs a
    /// resident scratch arena; today it's `hipMalloc + hipFree`,
    /// which the dispatcher already does for transient activations
    /// in `matmul_nn_f32`. Cost on this hardware is sub-microsecond
    /// per pair.
    ///
    /// The MIOpen layout we use is `[n, c=normalized_size, h=1, w=1]`
    /// with `normalized_dim = 1` (normalise over the C axis). That
    /// matches the row-wise affine LN every Phase-5b consumer
    /// expects.
    #[allow(clippy::too_many_arguments)]
    fn layer_norm_resident_f32(
        &self,
        x_dev: *const f32,
        weight_dev: *const f32,
        bias_dev: *const f32,
        y_dev: *mut f32,
        n: usize,
        normalized_size: usize,
        epsilon: f32,
    ) -> Result<(), BackendError> {
        let ctx = self.miopen()?;
        let n_i32 = as_i32(n, "n_rows")?;
        let c_i32 = as_i32(normalized_size, "normalized_size")?;

        // x and y have shape [n, c, 1, 1]; weight and bias have shape
        // [1, c, 1, 1] (broadcast across the batch axis); mean/rstd
        // have shape [n, 1, 1, 1] (one scalar per row).
        let x_desc = ctx.tensor_4d(n_i32, c_i32, 1, 1)?;
        let y_desc = x_desc;
        let wb_desc = ctx.tensor_4d(1, c_i32, 1, 1)?;
        let stat_desc = ctx.tensor_4d(n_i32, 1, 1, 1)?;

        // mean and rstd outputs — allocated and dropped per dispatch.
        // They're discarded for inference paths but the MIOpen API
        // requires non-null pointers.
        let mean_buf = HipBuffer::alloc(n * 4)?;
        let rstd_buf = HipBuffer::alloc(n * 4)?;

        let handle = ctx.handle.lock()
            .map_err(|_| BackendError::Runtime("miopen: handle mutex poisoned".into()))?;
        let st = unsafe {
            ffi::miopenLayerNormForward(
                *handle,
                ffi::MIOPEN_NORM_WEIGHT_BIAS,
                x_desc, x_dev as *const std::os::raw::c_void,
                wb_desc, weight_dev as *const std::os::raw::c_void,
                wb_desc, bias_dev as *const std::os::raw::c_void,
                epsilon,
                1, // normalize over C axis
                y_desc, y_dev as *mut std::os::raw::c_void,
                stat_desc, mean_buf.device_ptr(),
                stat_desc, rstd_buf.device_ptr(),
            )
        };
        drop(handle);
        if st != ffi::MIOPEN_STATUS_SUCCESS {
            return Err(BackendError::Runtime(format!(
                "miopenLayerNormForward(n={n}, c={normalized_size}): status {st}"
            )));
        }
        Ok(())
    }

    /// Resident row-wise softmax (or log-softmax) via
    /// `miopenSoftmaxForward_V2` with the ACCURATE/LOG algorithm.
    ///
    /// We model rows as the C (channel) axis of an [n_rows, row_len, 1, 1]
    /// tensor and pick `MIOPEN_SOFTMAX_MODE_INSTANCE`. With H=W=1,
    /// INSTANCE = "compute per N across C" — exactly row-wise softmax.
    /// `MIOPEN_SOFTMAX_MODE_CHANNEL` would compute per (N,H,W) across C
    /// which is the same when H=W=1 but the semantics are clearer this
    /// way.
    fn softmax_resident_f32(
        &self,
        x_dev: *const f32,
        y_dev: *mut f32,
        n_rows: usize,
        row_len: usize,
        log: bool,
    ) -> Result<(), BackendError> {
        let ctx = self.miopen()?;
        let n_i32 = as_i32(n_rows, "n_rows")?;
        let c_i32 = as_i32(row_len, "row_len")?;

        let desc = ctx.tensor_4d(n_i32, c_i32, 1, 1)?;
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let algo = if log { ffi::MIOPEN_SOFTMAX_LOG } else { ffi::MIOPEN_SOFTMAX_ACCURATE };

        let handle = ctx.handle.lock()
            .map_err(|_| BackendError::Runtime("miopen: handle mutex poisoned".into()))?;
        let st = unsafe {
            ffi::miopenSoftmaxForward_V2(
                *handle,
                &alpha as *const f32 as *const std::os::raw::c_void,
                desc, x_dev as *const std::os::raw::c_void,
                &beta as *const f32 as *const std::os::raw::c_void,
                desc, y_dev as *mut std::os::raw::c_void,
                algo,
                ffi::MIOPEN_SOFTMAX_MODE_INSTANCE,
            )
        };
        drop(handle);
        if st != ffi::MIOPEN_STATUS_SUCCESS {
            return Err(BackendError::Runtime(format!(
                "miopenSoftmaxForward_V2(n={n_rows}, c={row_len}, log={log}): status {st}"
            )));
        }
        Ok(())
    }

    /// Resident element-wise activation. `Logistic` / `Tanh` / `Relu`
    /// each map to a single `miopenActivationForward` call. `Silu` is
    /// `Logistic(x) -> y` followed by `OpTensor(MUL, x, y) -> y` —
    /// MIOpen has no native SiLU/Swish kernel.
    fn activation_resident_f32(
        &self,
        x_dev: *const f32,
        y_dev: *mut f32,
        n: usize,
        mode: ActivationMode,
    ) -> Result<(), BackendError> {
        let ctx = self.miopen()?;
        let n_i32 = as_i32(n, "n")?;
        // Flat 1D-into-NCHW: shape [n, 1, 1, 1] is a length-n vector.
        let desc = ctx.tensor_4d(n_i32, 1, 1, 1)?;

        // Map the public enum to the raw mode int.
        let raw_mode = match mode {
            ActivationMode::Logistic => ffi::MIOPEN_ACTIVATION_LOGISTIC,
            ActivationMode::Tanh => ffi::MIOPEN_ACTIVATION_TANH,
            ActivationMode::Relu => ffi::MIOPEN_ACTIVATION_RELU,
            ActivationMode::Silu => ffi::MIOPEN_ACTIVATION_LOGISTIC,
        };
        let activ_desc = ctx.activation_desc(raw_mode)?;
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;

        // Step 1: y = sigmoid(x) [or tanh/relu, pass-through for Silu].
        let handle = ctx.handle.lock()
            .map_err(|_| BackendError::Runtime("miopen: handle mutex poisoned".into()))?;
        let st = unsafe {
            ffi::miopenActivationForward(
                *handle, activ_desc,
                &alpha as *const f32 as *const std::os::raw::c_void,
                desc, x_dev as *const std::os::raw::c_void,
                &beta as *const f32 as *const std::os::raw::c_void,
                desc, y_dev as *mut std::os::raw::c_void,
            )
        };
        drop(handle);
        if st != ffi::MIOPEN_STATUS_SUCCESS {
            return Err(BackendError::Runtime(format!(
                "miopenActivationForward(n={n}, mode={raw_mode}): status {st}"
            )));
        }

        // Step 2 (Silu only): y = x * y. OpTensor(MUL) reads x and y,
        // writes back to y. alpha1 = alpha2 = 1.0, beta = 0.0 ⇒
        // c = (1.0 * a) * (1.0 * b) + 0.0 * c.
        if matches!(mode, ActivationMode::Silu) {
            self.op_tensor_resident_f32(
                x_dev, y_dev, y_dev, n,
                1.0, 1.0, 0.0, BinaryOpKind::Mul,
            )?;
        }
        Ok(())
    }

    /// Resident GLU forward.
    ///
    /// MIOpen's GLU solver only supports `dim == 0` splits — see
    /// `src/solver/glu/forward_glu.cpp` in the rocm-libraries source.
    /// We model the input as a 4D `[2, n_rows * half_size, 1, 1]`
    /// tensor and split on dim 0. That places the value half in the
    /// first `n_rows * half_size` floats and the gate half in the
    /// next `n_rows * half_size`, matching the MIOpen kernel's
    /// `inputFirstHalf = input; inputSecondHalf = input + N`
    /// arithmetic exactly. Output is `[1, n_rows * half_size, 1, 1]`.
    ///
    /// Callers whose input is laid out per-row (value then gate
    /// alternating per row) must scatter into the two planes before
    /// calling this op, or call once per row with `n_rows = 1`.
    fn glu_resident_f32(
        &self,
        x_dev: *const f32,
        y_dev: *mut f32,
        n_rows: usize,
        half_size: usize,
    ) -> Result<(), BackendError> {
        let ctx = self.miopen()?;
        let total = n_rows.checked_mul(half_size).ok_or_else(|| {
            BackendError::Runtime(format!(
                "glu_resident: n_rows*half_size overflow ({n_rows} * {half_size})"
            ))
        })?;
        let total_i32 = as_i32(total, "n_rows*half_size")?;

        // Input has split-dim 0 of size 2; output has size 1 on the
        // same axis. Other axes match.
        let in_desc = ctx.tensor_4d(2, total_i32, 1, 1)?;
        let out_desc = ctx.tensor_4d(1, total_i32, 1, 1)?;

        let handle = ctx.handle.lock()
            .map_err(|_| BackendError::Runtime("miopen: handle mutex poisoned".into()))?;
        let st = unsafe {
            ffi::miopenGLUForward(
                *handle,
                in_desc, x_dev as *const std::os::raw::c_void,
                out_desc, y_dev as *mut std::os::raw::c_void,
                0, // split on N axis (dim 0)
            )
        };
        drop(handle);
        if st != ffi::MIOPEN_STATUS_SUCCESS {
            return Err(BackendError::Runtime(format!(
                "miopenGLUForward(n={n_rows}, half={half_size}): status {st}"
            )));
        }
        Ok(())
    }

    /// Resident binary element-wise op:
    /// `c = op(alpha1 * a, alpha2 * b) + beta * c`.
    ///
    /// All three operands share the same shape — laid out as a flat
    /// `[n, 1, 1, 1]` tensor. MIOpen's OpTensor handles strided/broadcast
    /// shapes too, but we only expose the dense 1D variant from this
    /// op.
    #[allow(clippy::too_many_arguments)]
    fn op_tensor_resident_f32(
        &self,
        a_dev: *const f32,
        b_dev: *const f32,
        c_dev: *mut f32,
        n: usize,
        alpha1: f32,
        alpha2: f32,
        beta: f32,
        kind: BinaryOpKind,
    ) -> Result<(), BackendError> {
        let ctx = self.miopen()?;
        let n_i32 = as_i32(n, "n")?;
        let desc = ctx.tensor_4d(n_i32, 1, 1, 1)?;

        let raw_op = match kind {
            BinaryOpKind::Add => ffi::MIOPEN_OP_TENSOR_ADD,
            BinaryOpKind::Mul => ffi::MIOPEN_OP_TENSOR_MUL,
            BinaryOpKind::Min => ffi::MIOPEN_OP_TENSOR_MIN,
            BinaryOpKind::Max => ffi::MIOPEN_OP_TENSOR_MAX,
        };

        let handle = ctx.handle.lock()
            .map_err(|_| BackendError::Runtime("miopen: handle mutex poisoned".into()))?;
        let st = unsafe {
            ffi::miopenOpTensor(
                *handle, raw_op,
                &alpha1 as *const f32 as *const std::os::raw::c_void,
                desc, a_dev as *const std::os::raw::c_void,
                &alpha2 as *const f32 as *const std::os::raw::c_void,
                desc, b_dev as *const std::os::raw::c_void,
                &beta as *const f32 as *const std::os::raw::c_void,
                desc, c_dev as *mut std::os::raw::c_void,
            )
        };
        drop(handle);
        if st != ffi::MIOPEN_STATUS_SUCCESS {
            return Err(BackendError::Runtime(format!(
                "miopenOpTensor(n={n}, op={raw_op}): status {st}"
            )));
        }
        Ok(())
    }

    /// Resident LayerNorm backward.
    ///
    /// MIOpen's `miopenLayerNormBackward` consumes the per-row
    /// `mean`/`rstd` produced by the matching forward, plus `x`,
    /// `weight`, and `dy`, and writes `dx`, `dw`, `db`. The same
    /// `[n, c, 1, 1]` / `[1, c, 1, 1]` / `[n, 1, 1, 1]` descriptor
    /// shapes used by `miopenLayerNormForward` are reused here —
    /// MIOpen requires the descriptors to match the forward call
    /// for the saved stats to be interpreted correctly.
    ///
    /// `dweight_dev` and `dbias_dev` are accumulated into by the
    /// caller's higher-level wrapper; MIOpen itself overwrites these
    /// outputs, so we pre-stage host-zero scratch and add into the
    /// caller's slot afterwards. **NOTE.** MIOpen's
    /// `miopenLayerNormBackward` writes (not accumulates) to dW/dB —
    /// we route to the caller's pointer directly here, and the
    /// resident-brain caller is expected to use a fresh per-tick
    /// buffer (or zero-init before this dispatch). Same convention
    /// as host `linear_backward`'s d_input scratch.
    #[allow(clippy::too_many_arguments)]
    fn layer_norm_backward_resident_f32(
        &self,
        x_dev: *const f32,
        dy_dev: *const f32,
        weight_dev: *const f32,
        mean_dev: *const f32,
        rstd_dev: *const f32,
        dx_dev: *mut f32,
        dweight_dev: *mut f32,
        dbias_dev: *mut f32,
        n: usize,
        normalized_size: usize,
    ) -> Result<(), BackendError> {
        let ctx = self.miopen()?;
        let n_i32 = as_i32(n, "n_rows")?;
        let c_i32 = as_i32(normalized_size, "normalized_size")?;

        // Same descriptor shapes as the forward call so MIOpen
        // resolves the same solver path — see layer_norm_resident_f32.
        let x_desc = ctx.tensor_4d(n_i32, c_i32, 1, 1)?;
        let wb_desc = ctx.tensor_4d(1, c_i32, 1, 1)?;
        let stat_desc = ctx.tensor_4d(n_i32, 1, 1, 1)?;

        let handle = ctx.handle.lock()
            .map_err(|_| BackendError::Runtime("miopen: handle mutex poisoned".into()))?;
        let st = unsafe {
            ffi::miopenLayerNormBackward(
                *handle,
                ffi::MIOPEN_NORM_WEIGHT_BIAS,
                std::ptr::null_mut(), // workspace (unused — MIOpen allocates internally)
                0,                      // workspaceSizeInBytes
                x_desc, dy_dev as *const std::os::raw::c_void,
                x_desc, x_dev as *const std::os::raw::c_void,
                wb_desc, weight_dev as *const std::os::raw::c_void,
                stat_desc, mean_dev as *const std::os::raw::c_void,
                stat_desc, rstd_dev as *const std::os::raw::c_void,
                1, // normalize over C axis (matches forward)
                x_desc, dx_dev as *mut std::os::raw::c_void,
                wb_desc, dweight_dev as *mut std::os::raw::c_void,
                wb_desc, dbias_dev as *mut std::os::raw::c_void,
            )
        };
        drop(handle);
        if st != ffi::MIOPEN_STATUS_SUCCESS {
            return Err(BackendError::Runtime(format!(
                "miopenLayerNormBackward(n={n}, c={normalized_size}): status {st}"
            )));
        }
        Ok(())
    }

    /// Resident row-wise softmax backward via
    /// `miopenSoftmaxBackward_V2`. Mirrors the forward path's
    /// algorithm/mode pair — ACCURATE/INSTANCE for plain softmax,
    /// LOG/INSTANCE for log_softmax.
    fn softmax_backward_resident_f32(
        &self,
        y_dev: *const f32,
        dy_dev: *const f32,
        dx_dev: *mut f32,
        n_rows: usize,
        row_len: usize,
        log: bool,
    ) -> Result<(), BackendError> {
        let ctx = self.miopen()?;
        let n_i32 = as_i32(n_rows, "n_rows")?;
        let c_i32 = as_i32(row_len, "row_len")?;

        let desc = ctx.tensor_4d(n_i32, c_i32, 1, 1)?;
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let algo = if log { ffi::MIOPEN_SOFTMAX_LOG } else { ffi::MIOPEN_SOFTMAX_ACCURATE };

        let handle = ctx.handle.lock()
            .map_err(|_| BackendError::Runtime("miopen: handle mutex poisoned".into()))?;
        let st = unsafe {
            ffi::miopenSoftmaxBackward_V2(
                *handle,
                &alpha as *const f32 as *const std::os::raw::c_void,
                desc, y_dev as *const std::os::raw::c_void,
                desc, dy_dev as *const std::os::raw::c_void,
                &beta as *const f32 as *const std::os::raw::c_void,
                desc, dx_dev as *mut std::os::raw::c_void,
                algo,
                ffi::MIOPEN_SOFTMAX_MODE_INSTANCE,
            )
        };
        drop(handle);
        if st != ffi::MIOPEN_STATUS_SUCCESS {
            return Err(BackendError::Runtime(format!(
                "miopenSoftmaxBackward_V2(n={n_rows}, c={row_len}, log={log}): status {st}"
            )));
        }
        Ok(())
    }

    /// Resident element-wise activation backward.
    ///
    /// `Logistic`/`Tanh`/`Relu` each map to a single
    /// `miopenActivationBackward` call. `Silu` reverses the forward
    /// compose: forward is `y = x * sigmoid(x)`. The chain rule gives
    /// `dx = dy * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)))
    ///     = dy * (y/x + y * (1 - y/x))`. Numerically stable form
    /// uses the `sigmoid_x = sigmoid(x)` computed via
    /// `miopenActivationForward(LOGISTIC)` into a scratch buffer,
    /// then `dx = dy * sigmoid_x * (1 + x * (1 - sigmoid_x))` via
    /// elementwise compose. Implemented host-side via small
    /// hipMalloc'd scratch — the per-call cost is one extra pass over
    /// the activation tensor.
    #[allow(clippy::too_many_arguments)]
    fn activation_backward_resident_f32(
        &self,
        x_dev: *const f32,
        y_dev: *const f32,
        dy_dev: *const f32,
        dx_dev: *mut f32,
        n: usize,
        mode: ActivationMode,
    ) -> Result<(), BackendError> {
        let ctx = self.miopen()?;
        let n_i32 = as_i32(n, "n")?;
        let desc = ctx.tensor_4d(n_i32, 1, 1, 1)?;
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;

        match mode {
            ActivationMode::Logistic | ActivationMode::Tanh | ActivationMode::Relu => {
                let raw_mode = match mode {
                    ActivationMode::Logistic => ffi::MIOPEN_ACTIVATION_LOGISTIC,
                    ActivationMode::Tanh => ffi::MIOPEN_ACTIVATION_TANH,
                    ActivationMode::Relu => ffi::MIOPEN_ACTIVATION_RELU,
                    ActivationMode::Silu => unreachable!(),
                };
                let activ_desc = ctx.activation_desc(raw_mode)?;
                let handle = ctx.handle.lock()
                    .map_err(|_| BackendError::Runtime("miopen: handle mutex poisoned".into()))?;
                let st = unsafe {
                    ffi::miopenActivationBackward(
                        *handle, activ_desc,
                        &alpha as *const f32 as *const std::os::raw::c_void,
                        desc, y_dev as *const std::os::raw::c_void,
                        desc, dy_dev as *const std::os::raw::c_void,
                        desc, x_dev as *const std::os::raw::c_void,
                        &beta as *const f32 as *const std::os::raw::c_void,
                        desc, dx_dev as *mut std::os::raw::c_void,
                    )
                };
                drop(handle);
                if st != ffi::MIOPEN_STATUS_SUCCESS {
                    return Err(BackendError::Runtime(format!(
                        "miopenActivationBackward(n={n}, mode={raw_mode}): status {st}"
                    )));
                }
            }
            ActivationMode::Silu => {
                // SiLU: y = x * sigmoid(x). Let s = sigmoid(x). Then
                // dy/dx = s + x * s * (1 - s).  We need a scratch for
                // `s`.  Compute s = sigmoid(x) via miopenActivationForward.
                let scratch = HipBuffer::alloc(n * 4)?;
                let activ_desc = ctx.activation_desc(ffi::MIOPEN_ACTIVATION_LOGISTIC)?;
                let handle_g = ctx.handle.lock()
                    .map_err(|_| BackendError::Runtime("miopen: handle mutex poisoned".into()))?;
                let st = unsafe {
                    ffi::miopenActivationForward(
                        *handle_g, activ_desc,
                        &alpha as *const f32 as *const std::os::raw::c_void,
                        desc, x_dev as *const std::os::raw::c_void,
                        &beta as *const f32 as *const std::os::raw::c_void,
                        desc, scratch.device_ptr(),
                    )
                };
                drop(handle_g);
                if st != ffi::MIOPEN_STATUS_SUCCESS {
                    return Err(BackendError::Runtime(format!(
                        "silu_backward: miopenActivationForward(LOGISTIC) status {st}"
                    )));
                }

                // We now have s = sigmoid(x) in `scratch`. Compute
                // dx = dy * (s + x * s * (1 - s)).
                //
                // Implementation: stage helper bufs and use OpTensor
                // composition.
                //   tmp1 = 1 - s   (via OpTensor: MIN(1*ones, 1*s) won't work; we
                //                   instead use the identity 1 - s via OpTensor
                //                   ADD with alpha2=-1 against a pre-uploaded
                //                   ones vector). Simpler: derive the result
                //                   in two element-wise dispatches we already
                //                   have.
                //
                // Actually the cleanest path uses three OpTensor MUL passes:
                //   tmp_a = s * (1 - s)              <- requires 1-s
                //   tmp_b = x * tmp_a                <- "x * s * (1-s)"
                //   tmp_c = s + tmp_b                <- combined "ds/dx"
                //   dx    = dy * tmp_c
                //
                // To avoid a "ones" upload we exploit that for any tensor
                // shape, alpha1 * a + alpha2 * b with a = b = s and
                // (alpha1, alpha2) = (1, -1) ⇒ 0; with (alpha1, alpha2) =
                // (1, 0) ⇒ s. So we encode `1 - s` as
                //   OpTensor(ADD, alpha1=-1, a=s, alpha2=0, b=s, beta=0, c=tmp)
                // followed by adding 1 — but OpTensor lacks a scalar bias.
                //
                // Cleanest: ADD with a scratch ones-buffer. Allocate once
                // here per dispatch (small).
                let ones = HipBuffer::alloc(n * 4)?;
                {
                    let host_ones: Vec<f32> = vec![1.0; n];
                    ones.copy_from_host(&host_ones)?;
                }
                let one_minus_s = HipBuffer::alloc(n * 4)?;
                // one_minus_s = 1.0 * ones + (-1.0) * s
                self.op_tensor_resident_f32(
                    ones.device_ptr() as *const f32,
                    scratch.device_ptr() as *const f32,
                    one_minus_s.device_ptr() as *mut f32,
                    n, 1.0, -1.0, 0.0, BinaryOpKind::Add,
                )?;
                // tmp_a = s * (1 - s)
                let tmp_a = HipBuffer::alloc(n * 4)?;
                self.op_tensor_resident_f32(
                    scratch.device_ptr() as *const f32,
                    one_minus_s.device_ptr() as *const f32,
                    tmp_a.device_ptr() as *mut f32,
                    n, 1.0, 1.0, 0.0, BinaryOpKind::Mul,
                )?;
                // tmp_b = x * tmp_a
                let tmp_b = HipBuffer::alloc(n * 4)?;
                self.op_tensor_resident_f32(
                    x_dev,
                    tmp_a.device_ptr() as *const f32,
                    tmp_b.device_ptr() as *mut f32,
                    n, 1.0, 1.0, 0.0, BinaryOpKind::Mul,
                )?;
                // tmp_c = s + tmp_b   (alpha1=alpha2=1, ADD)
                // Reuse `scratch`'s data via a fresh allocation for `tmp_c`
                // because OpTensor's docs allow aliasing but doing so on
                // some MIOpen versions has read/write hazards on partial
                // updates; pay the alloc cost for safety.
                let tmp_c = HipBuffer::alloc(n * 4)?;
                self.op_tensor_resident_f32(
                    scratch.device_ptr() as *const f32,
                    tmp_b.device_ptr() as *const f32,
                    tmp_c.device_ptr() as *mut f32,
                    n, 1.0, 1.0, 0.0, BinaryOpKind::Add,
                )?;
                // dx = dy * tmp_c
                self.op_tensor_resident_f32(
                    dy_dev,
                    tmp_c.device_ptr() as *const f32,
                    dx_dev,
                    n, 1.0, 1.0, 0.0, BinaryOpKind::Mul,
                )?;
                // y_dev is read implicitly by the SiLU formula via x_dev/dy_dev;
                // we silence the unused warning since some MIOpen activation
                // backwards take y but Silu doesn't go through the kernel.
                let _ = y_dev;
            }
        }
        Ok(())
    }

    /// Resident GLU backward.
    ///
    /// MIOpen's `miopenGLUBackward` consumes the original input
    /// `x = [value | gate]` and the upstream gradient `dy`, and writes
    /// `dx` of the same shape as `x`. We use `dim = 0` to match the
    /// forward call.
    fn glu_backward_resident_f32(
        &self,
        x_dev: *const f32,
        dy_dev: *const f32,
        dx_dev: *mut f32,
        n_rows: usize,
        half_size: usize,
    ) -> Result<(), BackendError> {
        let ctx = self.miopen()?;
        let total = n_rows.checked_mul(half_size).ok_or_else(|| {
            BackendError::Runtime(format!(
                "glu_backward_resident: n_rows*half_size overflow ({n_rows} * {half_size})"
            ))
        })?;
        let total_i32 = as_i32(total, "n_rows*half_size")?;

        // Same descriptor shapes as forward — input/dx have split
        // dim 0 of size 2; dy has size 1 on the same axis.
        let in_desc = ctx.tensor_4d(2, total_i32, 1, 1)?;
        let out_desc = ctx.tensor_4d(1, total_i32, 1, 1)?;

        let handle = ctx.handle.lock()
            .map_err(|_| BackendError::Runtime("miopen: handle mutex poisoned".into()))?;
        let st = unsafe {
            ffi::miopenGLUBackward(
                *handle,
                in_desc, x_dev as *const std::os::raw::c_void,
                out_desc, dy_dev as *const std::os::raw::c_void,
                in_desc, dx_dev as *mut std::os::raw::c_void,
                0, // split on N axis (dim 0)
            )
        };
        drop(handle);
        if st != ffi::MIOPEN_STATUS_SUCCESS {
            return Err(BackendError::Runtime(format!(
                "miopenGLUBackward(n={n_rows}, half={half_size}): status {st}"
            )));
        }
        Ok(())
    }
}

/// ROCm device-resident buffer — a `DeviceBuffer`-shaped wrapper around
/// a `hipMalloc`'d allocation. Dropped via `HipBuffer`'s own `Drop`,
/// which runs `hipFree` automatically; `?`-propagated errors in the
/// dispatcher already rely on that, so re-using it here is no new
/// discipline.
#[cfg(feature = "rocm")]
pub struct RocmBuffer {
    inner: HipBuffer,
    len_f32: usize,
}

#[cfg(feature = "rocm")]
impl DeviceBuffer for RocmBuffer {
    fn backend_name(&self) -> &'static str { "rocm" }

    fn len(&self) -> usize { self.len_f32 }

    fn copy_from_host(&mut self, src: &[f32]) -> Result<(), BackendError> {
        if src.len() > self.len_f32 {
            return Err(BackendError::Runtime(format!(
                "RocmBuffer::copy_from_host: src.len()={} > buffer.len()={}",
                src.len(), self.len_f32,
            )));
        }
        self.inner.upload_f32(src)
    }

    fn copy_to_host(&self, dst: &mut [f32]) -> Result<(), BackendError> {
        if dst.len() > self.len_f32 {
            return Err(BackendError::Runtime(format!(
                "RocmBuffer::copy_to_host: dst.len()={} > buffer.len()={}",
                dst.len(), self.len_f32,
            )));
        }
        self.inner.download_f32(dst)
    }
}

// SAFETY: HipBuffer holds only an opaque device pointer + byte size.
// Neither is mutated across threads; the hipMemcpy / hipFree calls are
// thread-safe per the HIP runtime.
#[cfg(feature = "rocm")]
unsafe impl Send for RocmBuffer {}
#[cfg(feature = "rocm")]
unsafe impl Sync for RocmBuffer {}

#[cfg(feature = "rocm")]
impl BufferBackend for RocmBackend {
    type Buffer = RocmBuffer;

    fn alloc_buffer(&self, n_f32: usize) -> Result<RocmBuffer, BackendError> {
        let bytes = n_f32.checked_mul(4).ok_or_else(|| {
            BackendError::Runtime(format!("rocm alloc_buffer: size overflow (n={n_f32})"))
        })?;
        let inner = HipBuffer::alloc(bytes)?;
        Ok(RocmBuffer { inner, len_f32: n_f32 })
    }
}

/// When the `rocm` feature is off, `RocmBackend` is a zero-size stub
/// that never appears in the registry (`try_new` returns `None`). Still
/// needs a `BufferBackend` impl so `ComputeCtx<RocmBackend>` is
/// nameable as a type — default to `HostBuffer`, same pattern as CUDA
/// and Vulkan.
#[cfg(not(feature = "rocm"))]
impl BufferBackend for RocmBackend {
    type Buffer = HostBuffer;

    fn alloc_buffer(&self, n: usize) -> Result<HostBuffer, BackendError> {
        Ok(HostBuffer::new(n))
    }
}

/// ROCm lifecycle hooks — each dispatch ends with a synchronous
/// `hipMemcpy` D2H, so the caller's `out` slice is always up to date
/// by the time `dispatch` returns. `flush` is therefore a no-op.
/// Arena doesn't exist yet: we still allocate/free per dispatch today,
/// which is exactly the waste Stage 4 will address by letting callers
/// hold `RocmBuffer` across ops.
impl ComputeCtx<RocmBackend> {
    /// No-op — ROCm has no managed arena yet.
    pub fn arena_reset(&self) {}

    /// No-op — dispatcher already returns with `out` materialised on host.
    pub fn flush(&self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_new_is_safe_without_rocm() {
        // Must not panic whether or not the rocm feature is enabled.
        let _ = RocmBackend::try_new();
    }
}
