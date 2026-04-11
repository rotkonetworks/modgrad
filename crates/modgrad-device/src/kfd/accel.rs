//! KFD GPU acceleration: stateless streaming dispatch.
//!
//! MegaTrain-style: GPU is a transient compute engine.
//! Weights stream in, result streams out, no persistent state.
//! One dispatch at a time — no concurrency, no cache coherence.
//!
//! API:
//!   try_matvec(x, w, b, out, m, k)   → y = Wx + b
//!   try_matvec_t(d_out, w, dx, m, k) → dx = W^T @ d_out
//!
//! Both return true if GPU handled it, false for CPU fallback.

use super::HsaDevice;
use std::sync::{Mutex, OnceLock};

/// GPU singleton: device + stream engine.
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

/// y = W @ x + b. Stateless: stream weights in, compute, stream result out.
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

/// dx = W^T @ d_out. Stateless backward.
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

/// SuperLinear forward: batched per-neuron matvec on GPU.
pub fn try_superlinear(
    trace: &[f32], weights: &[f32], biases: &[f32], out: &mut [f32],
    n_neurons: u32, in_per: u32, out_per: u32,
) -> bool {
    if DISABLED.load(std::sync::atomic::Ordering::Relaxed) { return false; }
    let mut guard = match gpu().lock() { Ok(g) => g, Err(_) => return false };
    let g = match guard.as_mut() { Some(g) => g, None => return false };

    match g.engine.superlinear(
        &mut g.dev, weights, biases, trace,
        n_neurons as usize, out_per as usize, in_per as usize,
    ) {
        Some(y) => {
            let n = (n_neurons * out_per) as usize;
            out[..n].copy_from_slice(&y);
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

/// No-op (no cache to invalidate in stateless design).
pub fn invalidate_cache() {}
