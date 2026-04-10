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
/// Below this, CPU is faster due to kernel launch overhead.
const MIN_FLOPS: usize = 50_000;

/// Cached weight matrix in VRAM: row-major, col-major, bias, all padded.
struct CachedWeight {
    w_row: GpuBuffer,
    w_col: GpuBuffer,
    bias: GpuBuffer,
    m: u32,      // original out_dim
    k: u32,      // original in_dim
    m_pad: u32,  // padded
    k_pad: u32,  // padded
}

/// Global KFD accelerator: device + weight cache.
struct KfdAccel {
    dev: HsaDevice,
    cache: HashMap<(usize, usize), CachedWeight>,  // (weight_ptr, weight_len) → VRAM
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

fn get_accel() -> &'static Mutex<Option<KfdAccel>> {
    KFD.get_or_init(|| {
        let accel = HsaDevice::open().ok().map(|dev| KfdAccel {
            dev,
            cache: HashMap::new(),
        });
        Mutex::new(accel)
    })
}

/// Try to compute y = W @ x + b on KFD GPU.
/// Returns true if successful, false if GPU unavailable or dimensions unsupported.
/// Caller should fall back to CPU on false.
///
/// This is the only function the brain needs to call.
pub fn try_matvec(
    x: &[f32], weight: &[f32], bias: &[f32], out: &mut [f32],
    out_dim: u32, in_dim: u32,
) -> bool {
    // Skip if too small for GPU
    if (out_dim as usize * in_dim as usize) < MIN_FLOPS {
        return false;
    }

    let mut guard = match get_accel().lock() {
        Ok(g) => g,
        Err(_) => return false,
    };
    let accel = match guard.as_mut() {
        Some(a) => a,
        None => return false,  // KFD not available
    };

    // Upload or retrieve cached weight
    let key = match accel.ensure_weight(weight, bias, out_dim, in_dim) {
        Some(k) => k,
        None => return false,
    };

    // Get cached dimensions (safe: key was just inserted/verified)
    let (m_pad, k_pad) = {
        let c = &accel.cache[&key];
        (c.m_pad, c.k_pad)
    };

    // Pack input as first column of a K×32 batch (kernel needs N%32==0)
    let n_pad = 32u32;
    let mut flat_x = vec![0.0f32; k_pad as usize * n_pad as usize];
    for i in 0..x.len().min(k_pad as usize) {
        flat_x[i] = x[i];
    }
    let x_batch = match accel.dev.upload_f32(&flat_x) {
        Ok(b) => b,
        Err(_) => return false,
    };
    let y_batch = match accel.dev.alloc_output(m_pad as usize * n_pad as usize * 4 + 64) {
        Ok(b) => b,
        Err(_) => return false,
    };

    // Dispatch (re-borrow cache after upload)
    let cached = &accel.cache[&key];
    let ok = accel.dev.dispatch_matmul_enqueue(
        &cached.w_row, &cached.w_col, &cached.bias,
        &x_batch, &y_batch,
        m_pad, k_pad, n_pad,
    );
    if !ok { return false; }
    if !accel.dev.submit_wait(5_000) { return false; }

    // Read back first column only
    let y_slice = unsafe {
        std::slice::from_raw_parts(y_batch.cpu_ptr as *const f32,
                                   m_pad as usize * n_pad as usize)
    };
    for i in 0..(out_dim as usize).min(out.len()) {
        out[i] = y_slice[i];
    }

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
