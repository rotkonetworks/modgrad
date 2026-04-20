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
use super::{DeviceBuffer, QuantKind};
#[cfg(not(feature = "rocm"))]
use super::HostBuffer;

#[cfg(feature = "rocm")]
mod ffi {
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
#[cfg(feature = "rocm")]
struct HipBuffer {
    ptr: *mut std::os::raw::c_void,
    bytes: usize,
}

#[cfg(feature = "rocm")]
impl HipBuffer {
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

    /// Copy `src` (host) into this device buffer.
    fn upload_f32(&self, src: &[f32]) -> Result<(), BackendError> {
        let nbytes = src.len() * 4;
        debug_assert!(nbytes <= self.bytes, "upload_f32 overflow");
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

    /// Copy this device buffer into `dst` (host).
    fn download_f32(&self, dst: &mut [f32]) -> Result<(), BackendError> {
        let nbytes = dst.len() * 4;
        debug_assert!(nbytes <= self.bytes, "download_f32 overflow");
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
    /// Weight VRAM cache — mirrors the KFD `GpuQueue::weight_cache`
    /// pattern (see `kfd/dispatch_queue.rs`). Eliminates the per-dispatch
    /// `hipMalloc + hipMemcpy H2D + hipFree` triple that otherwise runs
    /// every forward pass for weights that never change within a step.
    ///
    /// SAFETY invariants (pointer-identity caching):
    /// - Key is `(weight.as_ptr() as usize, weight.len())`. This is
    ///   stable across training steps because weights live inside their
    ///   owning struct (e.g. `CtmWeights`) and are never reallocated.
    /// - The cache ASSUMES the host buffer behind `ptr` is not mutated
    ///   in place while the cache entry is live. modgrad-device's AdamW
    ///   DOES mutate weights in place, so the isis training loop calls
    ///   `registry().invalidate_caches()` after every optimizer step
    ///   (see commit feea041). As long as that invariant holds, a cache
    ///   hit returns the same bytes the caller intended to upload.
    /// - Dropping a `HipBuffer` runs `hipFree`, so clearing the HashMap
    ///   automatically releases VRAM — no manual cleanup needed on
    ///   invalidation or backend drop.
    /// - Wrapped in the same `Mutex` style as the hipBLAS handle; every
    ///   access locks first. The lock is held only during the
    ///   lookup/insert; dispatches do not hold it across the hipBLAS
    ///   call.
    cache: std::sync::Mutex<std::collections::HashMap<(usize, usize), HipBuffer>>,
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
            Some(Self {
                handle: std::sync::Mutex::new(handle),
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
                Op::MatmulNN { m, k, n, .. }
                    if *m >= 64 && *k >= 64 && *n >= 64 => true,
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
                _ => Err(BackendError::Unsupported {
                    op: op.name(),
                    backend: "rocm",
                }),
            }
        }
    }
}

#[cfg(feature = "rocm")]
impl RocmBackend {
    /// Look up the weight VRAM mirror, or upload it if missing. Returns
    /// the cached device pointer.
    ///
    /// The pointer remains valid after the lock is released because the
    /// cache only ever drops entries wholesale via `invalidate_cache`,
    /// which callers schedule between dispatches (post-optimizer step),
    /// never concurrent with one. See the `cache` field's SAFETY note.
    fn cached_weight_ptr(&self, weight: &[f32]) -> Result<*mut f32, BackendError> {
        let key = (weight.as_ptr() as usize, weight.len());
        let mut cache = self.cache.lock()
            .map_err(|_| BackendError::Runtime("rocm: cache mutex poisoned".into()))?;
        if let Some(buf) = cache.get(&key) {
            return Ok(buf.as_f32_ptr());
        }
        let buf = HipBuffer::alloc(weight.len() * 4)?;
        buf.upload_f32(weight)?;
        let ptr = buf.as_f32_ptr();
        cache.insert(key, buf);
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

        let err = unsafe { ffi::hipDeviceSynchronize() };
        if err != 0 {
            return Err(BackendError::Runtime(format!(
                "deviceSync: {}", ffi::hip_err_str(err)
            )));
        }

        d_y.download_f32(out)
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

        let err = unsafe { ffi::hipDeviceSynchronize() };
        if err != 0 {
            return Err(BackendError::Runtime(format!(
                "matmul sync: {}", ffi::hip_err_str(err)
            )));
        }

        d_c.download_f32(out)
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

/// ROCm lifecycle hooks — the HIP runtime synchronises per-call in our
/// dispatcher (each matvec ends with `hipDeviceSynchronize`), so the
/// flush hook is a no-op at Stage 2. Arena doesn't exist yet: we do
/// allocate/free per dispatch today, which is exactly the waste Stage 4
/// will address by letting callers hold `RocmBuffer` across ops.
impl ComputeCtx<RocmBackend> {
    /// No-op — ROCm has no managed arena yet.
    pub fn arena_reset(&self) {}

    /// No-op — dispatcher already syncs on every op.
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
