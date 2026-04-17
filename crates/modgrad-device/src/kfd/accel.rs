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

    // Direct dispatch: upload → dispatch → wait → copy to caller's slice (no Vec alloc)
    if g.engine.matvec_into(&mut g.dev, weight, bias, x, out, out_dim as usize, in_dim as usize) {
        return true;
    }
    DISABLED.store(true, std::sync::atomic::Ordering::Relaxed);
    false
}

/// dx = W^T @ d_out.
pub fn try_matvec_t(
    d_out: &[f32], weight: &[f32], d_input: &mut [f32],
    out_dim: u32, in_dim: u32,
) -> bool {
    if DISABLED.load(std::sync::atomic::Ordering::Relaxed) { return false; }
    let mut guard = match gpu().lock() { Ok(g) => g, Err(_) => return false };
    let g = match guard.as_mut() { Some(g) => g, None => return false };

    // Zero-copy path: both d_out and d_input are in VRAM arena
    if let (Some(dout_va), Some(dx_va)) = (resolve_va(g, d_out.as_ptr()), resolve_va(g, d_input.as_ptr())) {
        return g.engine.matvec_t_zerocopy(
            &mut g.dev, weight,
            dout_va, dx_va, out_dim as usize, in_dim as usize,
        );
    }

    // Direct dispatch: upload → dispatch → wait → copy to caller's slice (no Vec alloc)
    if g.engine.matvec_t_into(&mut g.dev, weight, d_out, d_input, out_dim as usize, in_dim as usize) {
        return true;
    }
    DISABLED.store(true, std::sync::atomic::Ordering::Relaxed);
    false
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

/// Fused synapse backward: SiLU_bwd → LN_bwd → matvec_t.
/// All 3 kernels run on GPU with data staying in VRAM between ops.
/// Single PCIe round-trip: upload cached activations + d_out,
/// download d_input + d_ln + d_gamma + d_beta.
///
/// Returns true on success, writing d_input and d_ln.
/// Accumulates into d_gamma, d_beta.
/// Caller uses d_ln for d_weight (outer product) and d_bias.
///
/// Constraint: out_dim <= 256 (ln_bwd kernel is single-WG).
pub fn try_synapse_backward(
    weight: &[f32],
    d_out: &[f32],
    pre_silu: &[f32],
    normalized: &[f32],
    gamma: &[f32],
    d_gamma: &mut [f32],
    d_beta: &mut [f32],
    d_input: &mut [f32],
    d_ln: &mut [f32],
    inv_std: f32,
    out_dim: u32, in_dim: u32,
) -> bool {
    if DISABLED.load(std::sync::atomic::Ordering::Relaxed) { return false; }
    if out_dim > 256 { return false; } // ln_bwd kernel limit
    let mut guard = match gpu().lock() { Ok(g) => g, Err(_) => return false };
    let g = match guard.as_mut() { Some(g) => g, None => return false };

    match g.engine.synapse_backward(
        &mut g.dev, weight, d_out, pre_silu, normalized, gamma,
        d_gamma, d_beta, inv_std,
        out_dim as usize, in_dim as usize,
    ) {
        Some((di, dln)) => {
            d_input[..in_dim as usize].copy_from_slice(&di);
            d_ln[..out_dim as usize].copy_from_slice(&dln);
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

/// GPU sync backward: scatter-add gradients from pairs back to neurons.
/// `left` and `right` are u32 neuron indices. Returns d_act[d_model].
pub fn try_sync_backward(
    d_sync: &[f32], activated: &[f32], beta: &[f32],
    left: &[u32], right: &[u32],
    n_pairs: u32, d_model: u32,
    d_act: &mut [f32],
) -> bool {
    if DISABLED.load(std::sync::atomic::Ordering::Relaxed) { return false; }
    let mut guard = match gpu().lock() { Ok(g) => g, Err(_) => return false };
    let g = match guard.as_mut() { Some(g) => g, None => return false };
    g.engine.sync_backward(&mut g.dev, d_sync, activated, beta,
        left, right, n_pairs as usize, d_model as usize, d_act)
}

/// GPU outer product accumulate: dW[i*k+j] += d_out[i] * input[j]
/// Used for gradient computation (d_weight = d_out ⊗ input).
pub fn try_outer_product(
    d_weight: &mut [f32], d_out: &[f32], input: &[f32],
    out_dim: u32, in_dim: u32,
) -> bool {
    if DISABLED.load(std::sync::atomic::Ordering::Relaxed) { return false; }
    let mut guard = match gpu().lock() { Ok(g) => g, Err(_) => return false };
    let g = match guard.as_mut() { Some(g) => g, None => return false };
    g.engine.outer_product(&mut g.dev, d_weight, d_out, input,
        out_dim as usize, in_dim as usize)
}

/// GPU SGD update: w[i] -= lr_scale * grad[i]; grad[i] = 0
pub fn try_sgd_update(
    weights: &mut [f32], grads: &mut [f32], lr_scale: f32,
) -> bool {
    if DISABLED.load(std::sync::atomic::Ordering::Relaxed) { return false; }
    let mut guard = match gpu().lock() { Ok(g) => g, Err(_) => return false };
    let g = match guard.as_mut() { Some(g) => g, None => return false };
    g.engine.sgd_update(&mut g.dev, weights, grads, lr_scale)
}

/// GPU AdamW optimizer: update weights, moments, zero grads.
/// Bias correction terms are pre-computed by caller: bc1_inv = 1/(1-beta1^t), bc2_inv = 1/(1-beta2^t).
pub fn try_adamw(
    weights: &mut [f32], grads: &mut [f32],
    m: &mut [f32], v: &mut [f32],
    lr: f32, beta1: f32, beta2: f32, eps: f32,
    weight_decay: f32, bc1_inv: f32, bc2_inv: f32,
) -> bool {
    if DISABLED.load(std::sync::atomic::Ordering::Relaxed) { return false; }
    let mut guard = match gpu().lock() { Ok(g) => g, Err(_) => return false };
    let g = match guard.as_mut() { Some(g) => g, None => return false };
    g.engine.adamw(&mut g.dev, weights, grads, m, v,
        lr, beta1, beta2, eps, weight_decay, bc1_inv, bc2_inv)
}

/// GPU SuperLinear backward: d_weight and d_input in two dispatches.
/// Returns d_input. Accumulates into d_weights.
pub fn try_superlinear_backward(
    weights: &[f32], d_out: &[f32], input: &[f32],
    d_weights: &mut [f32], d_input: &mut [f32],
    n_neurons: u32, out_per: u32, in_per: u32,
) -> bool {
    if DISABLED.load(std::sync::atomic::Ordering::Relaxed) { return false; }
    let mut guard = match gpu().lock() { Ok(g) => g, Err(_) => return false };
    let g = match guard.as_mut() { Some(g) => g, None => return false };
    g.engine.superlinear_backward(
        &mut g.dev, weights, d_out, input, d_weights, d_input,
        n_neurons as usize, out_per as usize, in_per as usize,
    )
}

/// GPU SiLU backward: d_input[i] = d_out[i] * (s + x*s*(1-s))
pub fn try_silu_backward(
    d_out: &[f32], pre_silu: &[f32], d_input: &mut [f32],
) -> bool {
    if DISABLED.load(std::sync::atomic::Ordering::Relaxed) { return false; }
    let mut guard = match gpu().lock() { Ok(g) => g, Err(_) => return false };
    let g = match guard.as_mut() { Some(g) => g, None => return false };
    g.engine.silu_backward(&mut g.dev, d_out, pre_silu, d_input)
}

/// GPU GLU backward: d_val and d_gate from d_out + cached input.
pub fn try_glu_backward(
    d_out: &[f32], cached_input: &[f32], d_input: &mut [f32], n: u32,
) -> bool {
    if DISABLED.load(std::sync::atomic::Ordering::Relaxed) { return false; }
    let mut guard = match gpu().lock() { Ok(g) => g, Err(_) => return false };
    let g = match guard.as_mut() { Some(g) => g, None => return false };
    g.engine.glu_backward(&mut g.dev, d_out, cached_input, d_input, n as usize)
}

/// GPU per-neuron GLU backward.
pub fn try_per_neuron_glu_backward(
    d_out: &[f32], cached_input: &[f32], d_input: &mut [f32],
    n_neurons: u32, out_per: u32,
) -> bool {
    if DISABLED.load(std::sync::atomic::Ordering::Relaxed) { return false; }
    let mut guard = match gpu().lock() { Ok(g) => g, Err(_) => return false };
    let g = match guard.as_mut() { Some(g) => g, None => return false };
    g.engine.per_neuron_glu_backward(&mut g.dev, d_out, cached_input, d_input,
        n_neurons as usize, out_per as usize)
}

/// GPU affine LayerNorm backward.
pub fn try_ln_backward(
    d_out: &[f32], normalized: &[f32], gamma: &[f32],
    d_gamma: &mut [f32], d_beta: &mut [f32], d_input: &mut [f32],
    inv_std: f32,
) -> bool {
    if DISABLED.load(std::sync::atomic::Ordering::Relaxed) { return false; }
    if d_out.len() > 256 { return false; } // single-WG kernel, max 256
    let mut guard = match gpu().lock() { Ok(g) => g, Err(_) => return false };
    let g = match guard.as_mut() { Some(g) => g, None => return false };
    g.engine.ln_backward(&mut g.dev, d_out, normalized, gamma,
        d_gamma, d_beta, d_input, inv_std)
}

/// GPU L2 norm: returns sqrt(sum(x[i]^2)).
/// Returns None if GPU unavailable or dispatch fails (caller should fall back to CPU).
pub fn try_l2_norm(data: &[f32]) -> Option<f32> {
    if DISABLED.load(std::sync::atomic::Ordering::Relaxed) { return None; }
    let mut guard = match gpu().lock() { Ok(g) => g, Err(_) => return None };
    let g = match guard.as_mut() { Some(g) => g, None => return None };
    g.engine.reduce_l2(&mut g.dev, data)
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
