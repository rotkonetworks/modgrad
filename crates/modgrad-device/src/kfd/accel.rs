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
use super::arena::VramArena;
use std::sync::{Mutex, OnceLock};

struct Gpu {
    dev: HsaDevice,
    engine: super::stream::StreamEngine,
    /// VRAM arena for activation buffers. When slices point into this
    /// arena, dispatch skips upload/download and uses VRAM VAs directly.
    arena: Option<VramArena>,
}

static GPU: OnceLock<Mutex<Option<Gpu>>> = OnceLock::new();
static DISABLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

fn gpu() -> &'static Mutex<Option<Gpu>> {
    GPU.get_or_init(|| {
        let g = HsaDevice::open().ok().map(|dev| Gpu {
            dev,
            engine: super::stream::StreamEngine::new(),
            arena: None,
        });
        Mutex::new(g)
    })
}

/// Initialize the VRAM arena for zero-copy GPU dispatch.
/// Call once at startup. `size_mb` is the arena size in megabytes.
/// All activation buffers allocated from this arena will bypass PCIe
/// during GPU dispatch — data stays in VRAM between ops.
pub fn init_arena(size_mb: usize) {
    if let Ok(mut guard) = gpu().lock() {
        if let Some(ref mut g) = *guard {
            let bytes = size_mb * 1024 * 1024;
            g.arena = VramArena::new(&g.dev, bytes);
            if g.arena.is_some() {
                eprintln!("    VRAM arena: {} MB allocated", size_mb);
            }
        }
    }
}

/// Allocate `n` f32 elements from the VRAM arena.
/// Returns a raw pointer usable as `&mut [f32]` (BAR-mapped).
/// GPU dispatch will detect this pointer and skip PCIe transfers.
pub fn arena_alloc(n: usize) -> Option<*mut f32> {
    if let Ok(guard) = gpu().lock() {
        if let Some(ref g) = *guard {
            if let Some(ref arena) = g.arena {
                return arena.alloc(n).map(|s| s.ptr);
            }
        }
    }
    None
}

/// Reset the arena (free all allocations). Call between training steps.
pub fn arena_reset() {
    if let Ok(guard) = gpu().lock() {
        if let Some(ref g) = *guard {
            if let Some(ref arena) = g.arena {
                arena.reset();
            }
        }
    }
}

/// Check if a pointer is in the VRAM arena and return its GPU VA.
fn resolve_va(g: &Gpu, ptr: *const f32) -> Option<u64> {
    g.arena.as_ref()?.resolve_va(ptr)
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
/// If x and out are in the VRAM arena, dispatches zero-copy (no PCIe).
/// Otherwise falls back to upload/dispatch/download.
pub fn try_matvec(
    x: &[f32], weight: &[f32], bias: &[f32], out: &mut [f32],
    out_dim: u32, in_dim: u32,
) -> bool {
    if DISABLED.load(std::sync::atomic::Ordering::Relaxed) { return false; }
    let mut guard = match gpu().lock() { Ok(g) => g, Err(_) => return false };
    let g = match guard.as_mut() { Some(g) => g, None => return false };

    // Zero-copy path: both x and out are in VRAM arena
    if let (Some(x_va), Some(y_va)) = (resolve_va(g, x.as_ptr()), resolve_va(g, out.as_ptr())) {
        return g.engine.matvec_zerocopy(
            &mut g.dev, weight, bias,
            x_va, y_va, out_dim as usize, in_dim as usize,
        );
    }

    // Fallback: upload/dispatch/download (old path — proven to work)
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
/// Zero-copy when trace/out are in VRAM arena.
pub fn try_superlinear(
    trace: &[f32], weights: &[f32], biases: &[f32], out: &mut [f32],
    n_neurons: u32, in_per: u32, out_per: u32,
) -> bool {
    if DISABLED.load(std::sync::atomic::Ordering::Relaxed) { return false; }

    let mut guard = match gpu().lock() { Ok(g) => g, Err(_) => return false };
    let g = match guard.as_mut() { Some(g) => g, None => return false };

    // Zero-copy path
    if let (Some(t_va), Some(y_va)) = (resolve_va(g, trace.as_ptr()), resolve_va(g, out.as_ptr())) {
        return g.engine.superlinear_zerocopy(
            &mut g.dev, weights, biases,
            t_va, y_va, n_neurons as usize, out_per as usize, in_per as usize,
        );
    }

    // Direct dispatch with single-copy readback
    if g.engine.superlinear_into(
        &mut g.dev, weights, biases, trace, out,
        n_neurons as usize, out_per as usize, in_per as usize,
    ) {
        return true;
    }

    DISABLED.store(true, std::sync::atomic::Ordering::Relaxed);
    false
}

/// Fused synapse forward: matvec → GLU → SiLU → LayerNorm.
/// All 4 kernels run on GPU with data staying in VRAM between ops.
/// Single PCIe round-trip: upload x, download result.
/// `out_dim` is the FINAL output size (after GLU halving).
/// Weight is [2*out_dim x in_dim], bias is [2*out_dim].
pub fn try_synapse_forward(
    x: &[f32], weight: &[f32], bias: &[f32],
    output: &mut [f32], _scratch: &mut [f32],
    out_dim: u32, in_dim: u32,
) -> bool {
    if DISABLED.load(std::sync::atomic::Ordering::Relaxed) { return false; }
    // layer_norm kernel handles up to 1024 elements
    if out_dim > 1024 { return false; }
    let mut guard = match gpu().lock() { Ok(g) => g, Err(_) => return false };
    let g = match guard.as_mut() { Some(g) => g, None => return false };

    match g.engine.synapse_forward(
        &mut g.dev, weight, bias, x,
        out_dim as usize, in_dim as usize,
    ) {
        Some(y) => {
            output[..out_dim as usize].copy_from_slice(&y);
            true
        }
        None => false,
    }
}

/// GPU GLU activation.
pub fn try_glu(input: &[f32], output: &mut [f32], n: u32) -> bool {
    if DISABLED.load(std::sync::atomic::Ordering::Relaxed) { return false; }
    let mut guard = match gpu().lock() { Ok(g) => g, Err(_) => return false };
    let g = match guard.as_mut() { Some(g) => g, None => return false };

    match g.engine.glu(&mut g.dev, input, n as usize) {
        Some(y) => { output[..n as usize].copy_from_slice(&y); true }
        None => false,
    }
}

/// GPU SiLU in-place.
pub fn try_silu_inplace(x: &mut [f32]) -> bool {
    if DISABLED.load(std::sync::atomic::Ordering::Relaxed) { return false; }
    let mut guard = match gpu().lock() { Ok(g) => g, Err(_) => return false };
    let g = match guard.as_mut() { Some(g) => g, None => return false };
    g.engine.silu_inplace(&mut g.dev, x)
}

/// GPU layer norm in-place.
pub fn try_layer_norm_inplace(x: &mut [f32]) -> bool {
    if DISABLED.load(std::sync::atomic::Ordering::Relaxed) { return false; }
    if x.len() > 1024 { return false; }
    let mut guard = match gpu().lock() { Ok(g) => g, Err(_) => return false };
    let g = match guard.as_mut() { Some(g) => g, None => return false };
    g.engine.layer_norm_inplace(&mut g.dev, x)
}

/// GPU trace shift.
pub fn try_trace_shift(
    traces: &mut [f32], new_activations: &[f32],
    n_neurons: u32, memory_length: u32,
) -> bool {
    if DISABLED.load(std::sync::atomic::Ordering::Relaxed) { return false; }
    let mut guard = match gpu().lock() { Ok(g) => g, Err(_) => return false };
    let g = match guard.as_mut() { Some(g) => g, None => return false };
    g.engine.trace_shift(&mut g.dev, traces, new_activations,
        n_neurons as usize, memory_length as usize)
}

/// GPU sync update.
pub fn try_sync_update(
    alpha: &mut [f32], beta: &mut [f32],
    act_left: &[f32], act_right: &[f32],
    phases_left: &[f32], phases_right: &[f32],
    decay: &[f32], decay_shift: &[f32],
    dopamine: f32, n_pairs: u32, initialized: bool,
    sync_out: &mut [f32],
) -> bool {
    if DISABLED.load(std::sync::atomic::Ordering::Relaxed) { return false; }
    let mut guard = match gpu().lock() { Ok(g) => g, Err(_) => return false };
    let g = match guard.as_mut() { Some(g) => g, None => return false };
    g.engine.sync_update(&mut g.dev, alpha, beta,
        act_left, act_right, phases_left, phases_right,
        decay, decay_shift, dopamine, n_pairs as usize, initialized, sync_out)
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
