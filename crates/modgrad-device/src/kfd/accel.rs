//! Transparent KFD acceleration: GPU-first, CPU-fallback, zero config.
//!
//! The brain calls `kfd::try_matvec(x, w, b, y, m, k)` — same signature
//! as cuda::try_matvec and gpu::try_matvec. Returns true if GPU handled it.
//!
//! Internally:
//!   1. Lazy-init HsaDevice on first call (once, thread-safe)
//!   2. Cache uploaded weight matrices in VRAM (keyed by pointer+size)
//!   3. Pad dimensions to tile boundaries (transparent to caller)
//!   4. Dispatch matmul kernel, read back result
//!   5. If anything fails → return false → caller does CPU
//!
//! The matmul is a composable service — GPU dispatch is a transparent
//! filter that composes with the CPU fallback path.

use super::{HsaDevice, memory::GpuBuffer};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Minimum flops to bother with GPU dispatch.
/// Below ~10M, CPU AVX is faster due to KFD ioctl overhead (~300μs per dispatch).
/// GPU wins for large matmuls where compute time >> dispatch overhead.
const MIN_FLOPS: usize = 10_000_000;

/// Cached weight matrix in VRAM.
struct CachedWeight {
    w_row: GpuBuffer,    // W [out_dim × in_dim] row-major
    w_t: GpuBuffer,      // W^T [in_dim × out_dim] row-major (for backward dx)
    bias: GpuBuffer,
    bias_zero: GpuBuffer, // zero bias for backward dispatch
    w_col: GpuBuffer,    // unused legacy
    m: u32,              // out_dim
    k: u32,              // in_dim
    m_pad: u32,
    k_pad: u32,
}

/// Global KFD accelerator: device + weight cache + reusable scratch buffers.
struct KfdAccel {
    dev: HsaDevice,
    cache: HashMap<(usize, usize), CachedWeight>,
    /// Reusable input buffer (resized as needed)
    x_buf: Option<GpuBuffer>,
    x_buf_cap: usize,
    /// Reusable output buffer (resized as needed)
    y_buf: Option<GpuBuffer>,
    y_buf_cap: usize,
    /// Reusable kernargs buffer
    args_buf: Option<GpuBuffer>,
}

fn align_up(x: u32, align: u32) -> u32 {
    (x + align - 1) / align * align
}

impl KfdAccel {
    /// Ensure weight is uploaded to VRAM. Returns the cache key.
    fn ensure_weight(&mut self, weight: &[f32], bias: &[f32], out_dim: u32, in_dim: u32)
        -> Option<(usize, usize)>
    {
        let key = (weight.as_ptr() as usize, weight.len());
        if self.cache.contains_key(&key) {
            return Some(key);
        }

        // Pad dimensions
        let m_pad = if out_dim >= 1536 { align_up(out_dim, 128) } else { align_up(out_dim, 32) };
        let k_pad = align_up(in_dim, 8);

        // Pad weight matrix (row-major)
        let mut padded_row = vec![0.0f32; m_pad as usize * k_pad as usize];
        for r in 0..(out_dim as usize) {
            let src_start = r * in_dim as usize;
            let dst_start = r * k_pad as usize;
            let len = in_dim as usize;
            if src_start + len <= weight.len() {
                padded_row[dst_start..dst_start + len]
                    .copy_from_slice(&weight[src_start..src_start + len]);
            }
        }

        // Column-major (transpose)
        let mut padded_col = vec![0.0f32; k_pad as usize * m_pad as usize];
        for r in 0..(out_dim as usize) {
            for c in 0..(in_dim as usize) {
                padded_col[c * m_pad as usize + r] = weight[r * in_dim as usize + c];
            }
        }

        // Pad bias
        let mut padded_bias = vec![0.0f32; m_pad as usize];
        for i in 0..(out_dim as usize).min(bias.len()) {
            padded_bias[i] = bias[i];
        }

        let w_row = self.dev.upload_f32(&padded_row).ok()?;
        let w_col = self.dev.upload_f32(&padded_col).ok()?;
        let bias_buf = self.dev.upload_f32(&padded_bias).ok()?;

        // Transpose + zero bias for backward
        let w_t = self.dev.upload_f32(&padded_col).unwrap_or_else(|_| {
            self.dev.upload_f32(&[0.0f32]).unwrap()
        });
        let bias_zero = self.dev.upload_f32(&vec![0.0f32; k_pad as usize]).unwrap_or_else(|_| {
            self.dev.upload_f32(&[0.0f32]).unwrap()
        });
        self.cache.insert(key, CachedWeight {
            w_row, w_t, bias: bias_buf, bias_zero, w_col,
            m: out_dim, k: in_dim, m_pad, k_pad,
        });
        Some(key)
    }
}

/// Global singleton — opened once, lives forever.
static KFD: OnceLock<Mutex<Option<KfdAccel>>> = OnceLock::new();

/// Once a dispatch times out, stop trying GPU for this session.
static KFD_DISABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn get_accel() -> &'static Mutex<Option<KfdAccel>> {
    KFD.get_or_init(|| {
        let accel = HsaDevice::open().ok().map(|dev| KfdAccel {
            dev,
            cache: HashMap::new(),
            x_buf: None, x_buf_cap: 0,
            y_buf: None, y_buf_cap: 0,
            args_buf: None,
        });
        Mutex::new(accel)
    })
}

/// Try to compute y = W @ x + b on KFD GPU.
/// Returns true if successful, false if GPU unavailable or dimensions unsupported.
/// Caller should fall back to CPU on false.
///
/// Uses the matvec kernel: each workitem computes one output row.
/// W is row-major [out_dim × in_dim].
pub fn try_matvec(
    x: &[f32], weight: &[f32], bias: &[f32], out: &mut [f32],
    out_dim: u32, in_dim: u32,
) -> bool {
    // Skip if too small or GPU previously failed
    if (out_dim as usize * in_dim as usize) < MIN_FLOPS {
        return false;
    }
    if KFD_DISABLED.load(std::sync::atomic::Ordering::Relaxed) {
        return false;
    }

    let mut guard = match get_accel().lock() {
        Ok(g) => g,
        Err(_) => return false,
    };
    let accel = match guard.as_mut() {
        Some(a) => a,
        None => return false,
    };

    // Upload weight + bias + transpose (cached by pointer identity)
    let key = (weight.as_ptr() as usize, weight.len());
    if !accel.cache.contains_key(&key) {
        let w_buf = match accel.dev.upload_f32(weight) {
            Ok(b) => b,
            Err(_) => return false,
        };
        let b_buf = match accel.dev.upload_f32(bias) {
            Ok(b) => b,
            Err(_) => return false,
        };
        // Transpose: W^T[j * out_dim + i] = W[i * in_dim + j]
        // Used by backward pass: dx = W^T @ d_out
        let m = out_dim as usize;
        let k = in_dim as usize;
        let mut w_t_data = vec![0.0f32; k * m];
        for i in 0..m {
            for j in 0..k {
                w_t_data[j * m + i] = weight[i * k + j];
            }
        }
        let w_t_buf = match accel.dev.upload_f32(&w_t_data) {
            Ok(b) => b,
            Err(_) => return false,
        };
        let b_zero = match accel.dev.upload_f32(&vec![0.0f32; k]) {
            Ok(b) => b,
            Err(_) => return false,
        };
        let w_col = match accel.dev.upload_f32(&[0.0f32]) {
            Ok(b) => b,
            Err(_) => return false,
        };
        accel.cache.insert(key, CachedWeight {
            w_row: w_buf, w_t: w_t_buf, bias: b_buf, bias_zero: b_zero, w_col,
            m: out_dim, k: in_dim, m_pad: out_dim, k_pad: in_dim,
        });
    }
    let cached = &accel.cache[&key];

    // Reuse or grow input buffer
    let x_bytes = x.len() * 4;
    if accel.x_buf_cap < x_bytes {
        let new_cap = ((x_bytes + 4095) & !4095).max(4096);
        accel.x_buf = accel.dev.upload_f32(x).ok();
        accel.x_buf_cap = new_cap;
    } else if let Some(ref xb) = accel.x_buf {
        xb.write_f32(0, x);
    }
    // Reuse or grow output buffer
    let y_bytes = out_dim as usize * 4;
    if accel.y_buf_cap < y_bytes {
        let new_cap = ((y_bytes + 4095) & !4095).max(4096);
        accel.y_buf = accel.dev.alloc_output(new_cap).ok();
        accel.y_buf_cap = new_cap;
    } else if let Some(ref yb) = accel.y_buf {
        // Zero just the output region
        unsafe { std::ptr::write_bytes(yb.cpu_ptr, 0, y_bytes); }
    }

    let (x_buf, y_buf) = match (&accel.x_buf, &accel.y_buf) {
        (Some(x), Some(y)) => (x, y),
        _ => return false,
    };

    // Build kernargs — write directly to pre-allocated buffer (no alloc per call)
    let nwg = (out_dim + 255) / 256;
    if accel.args_buf.is_none() {
        accel.args_buf = accel.dev.alloc.alloc_vram(4096).ok();
    }
    let args_buf = match &accel.args_buf {
        Some(b) => b,
        None => return false,
    };
    // Write kernargs: 4 pointers (32 bytes) + 2 u32s (8 bytes) = 40 bytes
    let mut kargs = [0u8; 48];
    kargs[0..8].copy_from_slice(&cached.w_row.va_addr.to_le_bytes());
    kargs[8..16].copy_from_slice(&cached.bias.va_addr.to_le_bytes());
    kargs[16..24].copy_from_slice(&x_buf.va_addr.to_le_bytes());
    kargs[24..32].copy_from_slice(&y_buf.va_addr.to_le_bytes());
    kargs[32..36].copy_from_slice(&out_dim.to_le_bytes());
    kargs[36..40].copy_from_slice(&in_dim.to_le_bytes());
    args_buf.write(0, &kargs[..40]);

    if !accel.dev.dispatch_enqueue("matvec", args_buf, [nwg, 1, 1], [256, 1, 1]) {
        return false;
    }
    if !accel.dev.submit_wait(2000) {
        eprintln!("kfd: matvec {}x{} timeout — disabling GPU dispatch", out_dim, in_dim);
        KFD_DISABLED.store(true, std::sync::atomic::Ordering::Relaxed);
        return false;
    }

    // Read back results
    let y_slice = unsafe {
        std::slice::from_raw_parts(y_buf.cpu_ptr as *const f32, out_dim as usize)
    };
    out[..out_dim as usize].copy_from_slice(y_slice);
    true
}

/// Compute d_input = W^T @ d_out on GPU (backward input gradient).
///
/// Same kernel as try_matvec, but uses the cached transposed weight.
/// W is [out_dim × in_dim], W^T is [in_dim × out_dim].
/// d_out is [out_dim], d_input is [in_dim].
pub fn try_matvec_t(
    d_out: &[f32], weight: &[f32], d_input: &mut [f32],
    out_dim: u32, in_dim: u32,
) -> bool {
    if (out_dim as usize * in_dim as usize) < MIN_FLOPS {
        return false;
    }
    if KFD_DISABLED.load(std::sync::atomic::Ordering::Relaxed) {
        return false;
    }

    let mut guard = match get_accel().lock() {
        Ok(g) => g,
        Err(_) => return false,
    };
    let accel = match guard.as_mut() {
        Some(a) => a,
        None => return false,
    };

    // Weight must already be cached (from a prior forward call)
    let key = (weight.as_ptr() as usize, weight.len());
    let cached = match accel.cache.get(&key) {
        Some(c) => c,
        None => return false, // not cached — forward wasn't called yet
    };

    // Reuse scratch buffers for d_out input and d_input output
    let d_out_bytes = d_out.len() * 4;
    if accel.x_buf_cap < d_out_bytes {
        accel.x_buf = accel.dev.upload_f32(d_out).ok();
        accel.x_buf_cap = ((d_out_bytes + 4095) & !4095).max(4096);
    } else if let Some(ref xb) = accel.x_buf {
        xb.write_f32(0, d_out);
    }

    let d_in_bytes = in_dim as usize * 4;
    if accel.y_buf_cap < d_in_bytes {
        accel.y_buf = accel.dev.alloc_output(((d_in_bytes + 4095) & !4095).max(4096)).ok();
        accel.y_buf_cap = ((d_in_bytes + 4095) & !4095).max(4096);
    } else if let Some(ref yb) = accel.y_buf {
        unsafe { std::ptr::write_bytes(yb.cpu_ptr, 0, d_in_bytes); }
    }

    let (x_buf, y_buf) = match (&accel.x_buf, &accel.y_buf) {
        (Some(x), Some(y)) => (x, y),
        _ => return false,
    };

    // Dispatch: matvec(W^T, d_out) → d_input
    // W^T is [in_dim × out_dim], so matvec output dim = in_dim
    let nwg = (in_dim + 255) / 256;
    if accel.args_buf.is_none() {
        accel.args_buf = accel.dev.alloc.alloc_vram(4096).ok();
    }
    let args_buf = match &accel.args_buf {
        Some(b) => b,
        None => return false,
    };
    let mut kargs = [0u8; 48];
    kargs[0..8].copy_from_slice(&cached.w_t.va_addr.to_le_bytes());
    kargs[8..16].copy_from_slice(&cached.bias_zero.va_addr.to_le_bytes());
    kargs[16..24].copy_from_slice(&x_buf.va_addr.to_le_bytes());
    kargs[24..32].copy_from_slice(&y_buf.va_addr.to_le_bytes());
    kargs[32..36].copy_from_slice(&in_dim.to_le_bytes());
    kargs[36..40].copy_from_slice(&out_dim.to_le_bytes());
    args_buf.write(0, &kargs[..40]);

    if !accel.dev.dispatch_enqueue("matvec", args_buf, [nwg, 1, 1], [256, 1, 1]) {
        return false;
    }
    if !accel.dev.submit_wait(2000) {
        return false;
    }

    let y_slice = unsafe {
        std::slice::from_raw_parts(y_buf.cpu_ptr as *const f32, in_dim as usize)
    };
    d_input[..in_dim as usize].copy_from_slice(y_slice);
    true
}

/// Get a reference to the KFD device for direct GPU buffer operations.
/// Used by DeviceWeightCache for upload/alloc without going through try_matvec.
pub fn get_device() -> Option<DeviceRef> {
    let guard = match get_accel().lock() {
        Ok(g) => g,
        Err(_) => return None,
    };
    if guard.is_some() { Some(DeviceRef) } else { None }
}

/// Handle for direct GPU operations. Locks the device on each call.
pub struct DeviceRef;

impl DeviceRef {
    pub fn upload_f32(&self, data: &[f32]) -> std::io::Result<super::memory::GpuBuffer> {
        let mut guard = get_accel().lock().map_err(|_| std::io::Error::new(
            std::io::ErrorKind::Other, "lock failed"))?;
        let accel = guard.as_mut().ok_or_else(|| std::io::Error::new(
            std::io::ErrorKind::NotFound, "no device"))?;
        accel.dev.upload_f32(data)
    }

    pub fn alloc_output(&self, size_bytes: usize) -> std::io::Result<super::memory::GpuBuffer> {
        let mut guard = get_accel().lock().map_err(|_| std::io::Error::new(
            std::io::ErrorKind::Other, "lock failed"))?;
        let accel = guard.as_mut().ok_or_else(|| std::io::Error::new(
            std::io::ErrorKind::NotFound, "no device"))?;
        accel.dev.alloc_output(size_bytes)
    }

    /// Dispatch matvec using pre-uploaded GPU buffers. No CPU data involved.
    pub fn dispatch_matvec(
        &self,
        w: &super::memory::GpuBuffer,
        bias: &super::memory::GpuBuffer,
        x: &super::memory::GpuBuffer,
        y: &super::memory::GpuBuffer,
        out_dim: u32, in_dim: u32,
    ) -> bool {
        let mut guard = match get_accel().lock() {
            Ok(g) => g,
            Err(_) => return false,
        };
        let accel = match guard.as_mut() {
            Some(a) => a,
            None => return false,
        };

        if accel.args_buf.is_none() {
            accel.args_buf = accel.dev.alloc.alloc_vram(4096).ok();
        }
        let args_buf = match &accel.args_buf {
            Some(b) => b,
            None => return false,
        };

        let nwg = (out_dim + 255) / 256;
        let mut kargs = [0u8; 48];
        kargs[0..8].copy_from_slice(&w.va_addr.to_le_bytes());
        kargs[8..16].copy_from_slice(&bias.va_addr.to_le_bytes());
        kargs[16..24].copy_from_slice(&x.va_addr.to_le_bytes());
        kargs[24..32].copy_from_slice(&y.va_addr.to_le_bytes());
        kargs[32..36].copy_from_slice(&out_dim.to_le_bytes());
        kargs[36..40].copy_from_slice(&in_dim.to_le_bytes());
        args_buf.write(0, &kargs[..40]);

        if !accel.dev.dispatch_enqueue("matvec", args_buf, [nwg, 1, 1], [256, 1, 1]) {
            return false;
        }
        accel.dev.submit_wait(2000)
    }
}

/// Check if KFD GPU is available.
pub fn available() -> bool {
    let guard = match get_accel().lock() {
        Ok(g) => g,
        Err(_) => return false,
    };
    guard.is_some()
}

/// Invalidate the weight cache (call after weights are modified, e.g. after sleep).
pub fn invalidate_cache() {
    if let Ok(mut guard) = get_accel().lock() {
        if let Some(ref mut accel) = *guard {
            accel.cache.clear();
        }
    }
}
