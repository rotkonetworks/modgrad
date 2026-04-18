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

/// Y[n×m] = X[n×k] @ W^T[k×m] + B[m] using the GPU `matmul_blocked` kernel.
/// Defensive layer — validates every precondition before touching hardware.
///
/// # Preconditions (checked; returns `false` if violated)
/// * `m % 128 == 0`, `k % 8 == 0`, `n % 32 == 0` — required by matmul_blocked tiling
/// * `weight.len() >= m*k`, `bias.len() >= m`, `x.len() >= n*k`, `out.len() >= n*m`
///
/// # Quick-return behaviour
/// * If `n == 0` or `m == 0`: zero-sized output, returns `true` without dispatch.
/// * If `k == 0`: output is just broadcast bias; returns `true` after writing bias on CPU.
///
/// # Failure modes (returns `false`, `out` may be partially written)
/// * GPU globally disabled (prior hang / unavailable hardware)
/// * GPU lock contention
/// * VRAM alloc / upload / dispatch / timeout failure — sets session-wide DISABLED flag
///   on timeout so subsequent calls skip the hardware and fall to CPU.
///
/// # Safety (not unsafe, but documented)
/// This function never violates memory safety, but a mis-sized weight against the
/// declared `m, k` could cause the GPU kernel to read out-of-bounds of the uploaded
/// buffer → device hang + Xorg crash. The length checks above are the ONLY barrier
/// between caller bugs and a device reset. Do not weaken them.
pub fn try_matmul(
    x: &[f32], weight: &[f32], bias: &[f32], out: &mut [f32],
    n: u32, k: u32, m: u32,
) -> bool {
    // ─── Quick-return edge cases (success without dispatch) ───
    if n == 0 || m == 0 {
        return true;  // nothing to write
    }
    let n_us = n as usize;
    let k_us = k as usize;
    let m_us = m as usize;

    if k == 0 {
        // Y = bias broadcast; no matmul to do. Fill `out` row-wise on CPU — this
        // is small (n rows × m cols) and only hit for degenerate shapes.
        if bias.len() < m_us || out.len() < n_us * m_us { return false; }
        for row in 0..n_us {
            out[row * m_us..(row + 1) * m_us].copy_from_slice(&bias[..m_us]);
        }
        return true;
    }

    // ─── Kernel tiling preconditions (violation → GPU OOB → hang) ───
    if m % 128 != 0 || k % 8 != 0 || n % 32 != 0 {
        return false;
    }

    // ─── Buffer length validation — caller bug protection ───
    if weight.len() < m_us * k_us
        || bias.len() < m_us
        || x.len() < n_us * k_us
        || out.len() < n_us * m_us
    {
        return false;
    }

    // ─── GPU availability & locking ───
    if DISABLED.load(std::sync::atomic::Ordering::Relaxed) { return false; }
    let mut guard = match gpu().lock() { Ok(g) => g, Err(_) => return false };
    let g = match guard.as_mut() { Some(g) => g, None => return false };

    // ─── Dispatch. engine.matmul_into returns false on upload, alloc, or
    //     submit_wait failure. A submit_wait timeout is the most dangerous
    //     signal — we set DISABLED so a second bad call doesn't re-trigger.
    if g.engine.matmul_into(&mut g.dev, weight, bias, x, out, n_us, k_us, m_us) {
        return true;
    }

    // Couldn't complete — assume the queue is wedged. Disable for the session
    // and fall to CPU. Ops/ps this is safer than retrying into a hung GPU.
    DISABLED.store(true, std::sync::atomic::Ordering::Relaxed);
    false
}

/// Allocate a `VramMirror` holding permanent VRAM for a list of tensor sizes.
/// Returns `None` if the GPU is unavailable or any alloc fails.
///
/// Every tensor gets four VRAM buffers (weights, grads, m, v). `m` and `v`
/// start zero-initialised. Uploads to `weights` should follow via
/// `VramMirror::upload_weight`.
pub fn make_vram_mirror(sizes: Vec<usize>) -> Option<super::vram_mirror::VramMirror> {
    if DISABLED.load(std::sync::atomic::Ordering::Relaxed) { return None; }
    let guard = gpu().lock().ok()?;
    let g = guard.as_ref()?;
    super::vram_mirror::VramMirror::new(&g.dev, sizes)
}

/// Run AdamW in-place on a VRAM-resident mirror tensor (weights, grads, m, v
/// all live in the mirror). Zero PCIe / BAR traffic per call — kernel reads
/// and writes VRAM directly.
///
/// Returns `false` on dispatch failure. On timeout, sets the session-wide
/// DISABLED flag so subsequent calls short-circuit to the caller's fallback.
pub fn try_adamw_vram(
    mirror: &super::vram_mirror::VramMirror, idx: usize,
    lr: f32, beta1: f32, beta2: f32, eps: f32,
    weight_decay: f32, bc1_inv: f32, bc2_inv: f32,
) -> bool {
    if idx >= mirror.sizes.len() { return false; }
    if DISABLED.load(std::sync::atomic::Ordering::Relaxed) { return false; }
    let mut guard = match gpu().lock() { Ok(g) => g, Err(_) => return false };
    let g = match guard.as_mut() { Some(g) => g, None => return false };

    let n = mirror.sizes[idx];
    if g.engine.adamw_zerocopy(
        &mut g.dev,
        mirror.weight_va(idx), mirror.grad_va(idx),
        mirror.m_va(idx), mirror.v_va(idx),
        n, lr, beta1, beta2, eps, weight_decay, bc1_inv, bc2_inv,
    ) {
        return true;
    }
    DISABLED.store(true, std::sync::atomic::Ordering::Relaxed);
    false
}

/// Run AdamW on every tensor in `mirror` with one GPU submission.
/// Enqueues `mirror.sizes.len()` dispatches each using a distinct kernargs
/// slot, then a single cache writeback + submit_wait. For a 189 M param FFN
/// with ~100 tensors this turns 100 round trips into 1.
///
/// `wd_for_idx(idx)` supplies per-tensor weight decay (typically `wd` for
/// weights and 0.0 for biases / layer-norm params).
///
/// Returns `false` on any enqueue or the final submit_wait failing. On
/// timeout, sets the session-wide DISABLED flag.
pub fn try_adamw_vram_batch<F: Fn(usize) -> f32>(
    mirror: &super::vram_mirror::VramMirror,
    lr: f32, beta1: f32, beta2: f32, eps: f32,
    bc1_inv: f32, bc2_inv: f32,
    wd_for_idx: F,
) -> bool {
    let n_tensors = mirror.sizes.len();
    if n_tensors == 0 { return true; }
    if DISABLED.load(std::sync::atomic::Ordering::Relaxed) { return false; }
    let mut guard = match gpu().lock() { Ok(g) => g, Err(_) => return false };
    let g = match guard.as_mut() { Some(g) => g, None => return false };

    for idx in 0..n_tensors {
        let n = mirror.sizes[idx];
        let wd = wd_for_idx(idx);
        if !g.engine.adamw_enqueue(
            &mut g.dev, idx,
            mirror.weight_va(idx), mirror.grad_va(idx),
            mirror.m_va(idx), mirror.v_va(idx),
            n, lr, beta1, beta2, eps, wd, bc1_inv, bc2_inv,
        ) {
            // Enqueue failed mid-batch — still need to flush whatever already
            // went on the queue to avoid leaving the GPU with half-submitted work.
            let _ = g.engine.flush_queue(&mut g.dev, 10_000u32);
            DISABLED.store(true, std::sync::atomic::Ordering::Relaxed);
            return false;
        }
    }

    // Single submit_wait amortises ~100 dispatches into one round trip.
    if !g.engine.flush_queue(&mut g.dev, 30_000u32) {
        eprintln!("try_adamw_vram_batch: flush_queue timed out");
        DISABLED.store(true, std::sync::atomic::Ordering::Relaxed);
        return false;
    }
    true
}

/// Transposed matmul: dA[n×k] = dY[n×m] @ W[m×k] (ASSIGNS, does not accumulate).
/// Caller is responsible for zero-init / accumulation outside.
///
/// Implemented by physically transposing `weight` into a scratch `W^T` laid out
/// as [k×m] row-major, then calling `matmul_blocked` with `X=dY` and `W=W^T`.
///
/// # Preconditions (returns `false` on violation)
/// * `k % 128 == 0` (output M for kernel), `m % 8 == 0`, `n % 32 == 0`
/// * slice length checks as per `try_matmul`
pub fn try_matmul_t(
    dy: &[f32], weight: &[f32], da: &mut [f32],
    n: u32, k: u32, m: u32,
) -> bool {
    if n == 0 || k == 0 { return true; }
    let n_us = n as usize;
    let k_us = k as usize;
    let m_us = m as usize;

    // matmul_blocked requires M_kernel % 128 == 0. Here M_kernel = k.
    if k % 128 != 0 || m % 8 != 0 || n % 32 != 0 { return false; }

    if dy.len() < n_us * m_us
        || weight.len() < m_us * k_us
        || da.len() < n_us * k_us
    {
        return false;
    }

    if DISABLED.load(std::sync::atomic::Ordering::Relaxed) { return false; }

    // Transpose W[m,k] → W^T[k,m] once (CPU; rayon-parallel optional, but this
    // is small enough — e.g. 5120×1024 = 20MB, ~5ms scalar).
    let mut weight_t = vec![0.0f32; m_us * k_us];
    for row in 0..m_us {
        for col in 0..k_us {
            weight_t[col * m_us + row] = weight[row * k_us + col];
        }
    }
    // Zero bias slot for the kernel (we don't want an additive bias here).
    let zero_bias = vec![0.0f32; k_us];

    let mut guard = match gpu().lock() { Ok(g) => g, Err(_) => return false };
    let g = match guard.as_mut() { Some(g) => g, None => return false };

    // matmul_blocked: Y[n, k] = X[n, m] @ W^T_kernel + zero_bias
    //   where W^T_kernel = weight_t laid out as [k, m] row-major — exactly what
    //   the kernel expects as its "weight" argument.
    if g.engine.matmul_into(
        &mut g.dev, &weight_t, &zero_bias, dy, da, n_us, m_us, k_us,
    ) {
        return true;
    }
    DISABLED.store(true, std::sync::atomic::Ordering::Relaxed);
    false
}

/// Weight-gradient matmul: dW[m×k] = dY^T[m×n] @ A[n×k] (ASSIGNS, not accumulate).
///
/// Implemented via two one-off transposes (dY→dY^T, A→A^T) and a single
/// `matmul_blocked` dispatch with (N=m, K=n, M=k). Both transposes are small
/// compared to the matmul FLOPs (O(n·{m+k}) vs O(n·m·k)).
///
/// # Preconditions (returns `false` on violation)
/// * `k % 128 == 0`, `n % 8 == 0`, `m % 32 == 0`
/// * slice length checks
pub fn try_matmul_grad(
    dy: &[f32], a: &[f32], dw: &mut [f32],
    n: u32, k: u32, m: u32,
) -> bool {
    if m == 0 || k == 0 { return true; }
    if n == 0 {
        // dw = 0 (sum over empty axis). Assign zeros.
        for v in dw.iter_mut() { *v = 0.0; }
        return true;
    }
    let n_us = n as usize;
    let k_us = k as usize;
    let m_us = m as usize;

    // matmul_blocked requires kernel's M % 128 == 0. Kernel M = k here.
    if k % 128 != 0 || n % 8 != 0 || m % 32 != 0 { return false; }

    if dy.len() < n_us * m_us
        || a.len() < n_us * k_us
        || dw.len() < m_us * k_us
    {
        return false;
    }

    if DISABLED.load(std::sync::atomic::Ordering::Relaxed) { return false; }

    // dY^T: [m, n] row-major.
    let mut dy_t = vec![0.0f32; m_us * n_us];
    for row in 0..n_us {
        for col in 0..m_us {
            dy_t[col * n_us + row] = dy[row * m_us + col];
        }
    }
    // A^T: [k, n] row-major.
    let mut a_t = vec![0.0f32; k_us * n_us];
    for row in 0..n_us {
        for col in 0..k_us {
            a_t[col * n_us + row] = a[row * k_us + col];
        }
    }
    let zero_bias = vec![0.0f32; k_us];

    let mut guard = match gpu().lock() { Ok(g) => g, Err(_) => return false };
    let g = match guard.as_mut() { Some(g) => g, None => return false };

    // matmul_blocked: Y[m, k] = X[m, n] @ W^T_kernel + bias
    //   X = dy_t  ([m, n])
    //   kernel weight laid out as [k, n] row-major = a_t
    if g.engine.matmul_into(
        &mut g.dev, &a_t, &zero_bias, &dy_t, dw, m_us, n_us, k_us,
    ) {
        return true;
    }
    DISABLED.store(true, std::sync::atomic::Ordering::Relaxed);
    false
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
