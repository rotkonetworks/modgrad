//! KFD GPU acceleration: three modes.
//!
//! CPU:    no GPU, pure Rust (default when --gpu/--vram not passed)
//! HYBRID: stream weights per-call (--gpu). Good for PCIe x16.
//! VRAM:   upload once, keep in VRAM (--vram). Good for PCIe x4.
//!
//! API (same for all modes):
//!   try_matvec(x, w, b, out, m, k)   → y = Wx + b
//!   try_matvec_t(d_out, w, dx, m, k) → dx = W^T @ d_out
//!   try_superlinear(...)              → batched per-neuron matvec
//!   invalidate_cache()                → call after optimizer step (VRAM mode)

use super::HsaDevice;
use std::sync::{Mutex, OnceLock};

struct Gpu {
    dev: HsaDevice,
    engine: super::stream::StreamEngine,
}

static GPU: OnceLock<Mutex<Option<Gpu>>> = OnceLock::new();
static DISABLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

fn gpu() -> &'static Mutex<Option<Gpu>> {
    GPU.get_or_init(|| {
        let g = HsaDevice::open().ok().map(|dev| Gpu {
            dev,
            engine: super::stream::StreamEngine::new(),
        });
        Mutex::new(g)
    })
}

/// Enable VRAM-resident mode. Call once at startup before any dispatch.
/// Weights are cached in VRAM on first use, never re-uploaded until
/// invalidate_cache() is called after optimizer step.
pub fn enable_vram_mode() {
    if let Ok(mut guard) = gpu().lock() {
        if let Some(ref mut g) = *guard {
            g.engine.vram_mode = true;
        }
    }
}

/// y = W @ x + b.
pub fn try_matvec(
    x: &[f32], weight: &[f32], bias: &[f32], out: &mut [f32],
    out_dim: u32, in_dim: u32,
) -> bool {
    if DISABLED.load(std::sync::atomic::Ordering::Relaxed) { return false; }
    let mut guard = match gpu().lock() { Ok(g) => g, Err(_) => return false };
    let g = match guard.as_mut() { Some(g) => g, None => return false };

    match g.engine.matvec(&mut g.dev, weight, bias, x, out_dim as usize, in_dim as usize) {
        Some(y) => { out[..out_dim as usize].copy_from_slice(&y); true }
        None => {
            DISABLED.store(true, std::sync::atomic::Ordering::Relaxed);
            false
        }
    }
}

/// dx = W^T @ d_out.
pub fn try_matvec_t(
    d_out: &[f32], weight: &[f32], d_input: &mut [f32],
    out_dim: u32, in_dim: u32,
) -> bool {
    if DISABLED.load(std::sync::atomic::Ordering::Relaxed) { return false; }
    let mut guard = match gpu().lock() { Ok(g) => g, Err(_) => return false };
    let g = match guard.as_mut() { Some(g) => g, None => return false };

    match g.engine.matvec_t(&mut g.dev, weight, d_out, out_dim as usize, in_dim as usize) {
        Some(dx) => { d_input[..in_dim as usize].copy_from_slice(&dx); true }
        None => false,
    }
}

/// SuperLinear: batched per-neuron matvec.
/// In VRAM mode: no flops threshold (weights are cached, no PCIe cost).
/// In stream mode: 8M flops minimum (PCIe transfer must be justified).
pub fn try_superlinear(
    trace: &[f32], weights: &[f32], biases: &[f32], out: &mut [f32],
    n_neurons: u32, in_per: u32, out_per: u32,
) -> bool {
    if DISABLED.load(std::sync::atomic::Ordering::Relaxed) { return false; }

    let flops = n_neurons as usize * in_per as usize * out_per as usize;

    // In stream mode, check threshold. In VRAM mode, always dispatch.
    {
        let guard = match gpu().lock() { Ok(g) => g, Err(_) => return false };
        let g = match guard.as_ref() { Some(g) => g, None => return false };
        if !g.engine.vram_mode && flops < 8_000_000 { return false; }
    }

    let mut guard = match gpu().lock() { Ok(g) => g, Err(_) => return false };
    let g = match guard.as_mut() { Some(g) => g, None => return false };

    match g.engine.superlinear(
        &mut g.dev, weights, biases, trace,
        n_neurons as usize, out_per as usize, in_per as usize,
    ) {
        Some(y) => {
            out[..(n_neurons * out_per) as usize].copy_from_slice(&y);
            true
        }
        None => {
            DISABLED.store(true, std::sync::atomic::Ordering::Relaxed);
            false
        }
    }
}

/// Check if KFD GPU is available.
pub fn available() -> bool {
    match gpu().lock() { Ok(g) => g.is_some(), Err(_) => false }
}

/// Invalidate VRAM weight cache. Call after optimizer step.
/// In stream mode, this is a no-op (nothing cached).
pub fn invalidate_cache() {
    if let Ok(mut guard) = gpu().lock() {
        if let Some(ref mut g) = *guard {
            g.engine.invalidate();
        }
    }
}
