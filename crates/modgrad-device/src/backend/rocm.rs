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
                // Resident dispatches skip ALL the per-call overhead
                // (no malloc / memcpy / cache lookup), so the size
                // gate is dropped — small ops are still profitable
                // when weights and activations are already on device.
                Op::MatvecResident { .. } => true,
                Op::MatmulNN { m, k, n, .. }
                    if *m >= 64 && *k >= 64 && *n >= 64 => true,
                Op::MatmulNT { m, k, n, .. }
                    if *m >= 64 && *k >= 64 && *n >= 64 => true,
                Op::MatmulTN { m, k, n, .. }
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
