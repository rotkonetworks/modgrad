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
    w_row: GpuBuffer,
    w_col: GpuBuffer,
    bias: GpuBuffer,
    m: u32,
    k: u32,
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

        self.cache.insert(key, CachedWeight {
            w_row, w_col, bias: bias_buf,
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

    // Upload weight + bias (cached by pointer identity)
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
        // col-major not needed for matvec, store as dummy
        let w_col = match accel.dev.upload_f32(&[0.0f32]) {
            Ok(b) => b,
            Err(_) => return false,
        };
        accel.cache.insert(key, CachedWeight {
            w_row: w_buf, w_col, bias: b_buf,
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
