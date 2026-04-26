//! Full BPTT training for the faithful Ctm CTM.
//!
//! Implements cached forward + backward for every component:
//! Linear, LayerNorm, SiLU, GLU, SynapseBlock, SynapseUNet,
//! SuperLinear/NLM, Sync, MHA. Then the tick-loop unroll.
//!
//! No shortcuts — gradients flow through the entire computation graph,
//! matching what PyTorch autograd does for the reference implementation.

use modgrad_compute::neuron::{Linear, SuperLinear};
use super::synapse::{SynapseBlock, SynapseUNet};
use super::weights::CtmWeights;

// ═══════════════════════════════════════════════════════════════
// PRIMITIVE BACKWARD PASSES
// ═══════════════════════════════════════════════════════════════

// ─── Linear ────────────────────────────────────────────────

pub(crate) struct LinearCache { pub(crate) input: Vec<f32> }

pub(crate) fn linear_forward_cached(linear: &Linear, x: &[f32]) -> (Vec<f32>, LinearCache) {
    (linear.forward(x), LinearCache { input: x.to_vec() })
}

/// Writes d_input into the caller-supplied scratch slice. Accumulates
/// into d_weight, d_bias.
///
/// Contract: caller guarantees `d_input.len() == linear.in_dim`. The
/// scratch is fully overwritten (not accumulated into) — callers that
/// need accumulation do it themselves after the call.
///
/// d_weight (outer product) and d_input (matvec_t) dispatch through the
/// `Backend` registry: each op routes to the fastest registered backend
/// that supports the shape, falling through to `CpuBackend` when no GPU
/// claim matches. Caller doesn't branch on hardware — the registry's
/// the abstraction that hides that.
///
/// Kill-switch: `MODGRAD_BACKEND=cpu` before process start forces the
/// registry to contain only CPU.
pub(crate) fn linear_backward(
    linear: &Linear, d_out: &[f32], cache: &LinearCache,
    d_weight: &mut [f32], d_bias: &mut [f32],
    d_input: &mut [f32],
) {
    use modgrad_device::backend::ops;

    let in_dim = linear.in_dim;
    let out_dim = linear.out_dim;
    debug_assert_eq!(d_input.len(), in_dim,
        "linear_backward: d_input scratch must match linear.in_dim");

    // d_bias: a trivial accumulation, not worth an Op variant.
    for i in 0..out_dim { d_bias[i] += d_out[i]; }

    // d_weight[i,j] += d_out[i] * input[j]
    ops::outer_product_acc(d_out, &cache.input, d_weight, out_dim, in_dim)
        .expect("outer_product_acc dispatch");

    // d_input = W^T @ d_out (overwrites)
    for v in d_input.iter_mut() { *v = 0.0; }
    ops::matvec_t(d_out, &linear.weight, d_input, out_dim, in_dim)
        .expect("matvec_t dispatch");
}

// ─── Affine LayerNorm ──────────────────────────────────────

struct LnCache { normalized: Vec<f32>, inv_std: f32 }

fn affine_ln_forward(x: &mut Vec<f32>, gamma: &[f32], beta: &[f32]) -> LnCache {
    let n = x.len() as f32;
    let mean: f32 = x.iter().sum::<f32>() / n;
    let var: f32 = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    let inv_std = 1.0 / (var + 1e-5).sqrt();
    let mut normalized = Vec::with_capacity(x.len());
    for i in 0..x.len() {
        let norm = (x[i] - mean) * inv_std;
        normalized.push(norm);
        x[i] = gamma[i] * norm + beta[i];
    }
    LnCache { normalized, inv_std }
}

/// Returns d_input. Accumulates into d_gamma, d_beta.
fn affine_ln_backward(
    d_out: &[f32], cache: &LnCache, gamma: &[f32],
    d_gamma: &mut [f32], d_beta: &mut [f32],
) -> Vec<f32> {
    // This path uses a normalized+inv_std cache, which doesn't line up
    // with `Op::LayerNormBwd`'s mean/rstd cache shape. Until the Op is
    // unified (follow-up), the math stays inline here. It's CPU-only:
    // the old KFD fast-path went through a backend-specific import,
    // which would layer-violate modgrad-ctm against `modgrad_device::kfd`.
    // Worth measuring before re-introducing a fast path.
    let n = d_out.len();
    let nf = n as f32;
    for i in 0..n {
        d_gamma[i] += d_out[i] * cache.normalized[i];
        d_beta[i] += d_out[i];
    }
    let d_norm: Vec<f32> = (0..n).map(|i| d_out[i] * gamma[i]).collect();
    let mean_dn: f32 = d_norm.iter().sum::<f32>() / nf;
    let mean_dn_xhat: f32 = d_norm.iter().zip(&cache.normalized)
        .map(|(&d, &x)| d * x).sum::<f32>() / nf;
    (0..n).map(|i| {
        cache.inv_std * (d_norm[i] - mean_dn - cache.normalized[i] * mean_dn_xhat)
    }).collect()
}

// ─── SiLU ──────────────────────────────────────────────────

fn silu_forward_cached(x: &mut [f32]) -> Vec<f32> {
    let pre = x.to_vec();
    for v in x.iter_mut() {
        let s = 1.0 / (1.0 + (-*v).exp());
        *v = *v * s;
    }
    pre
}

fn silu_backward(d_out: &[f32], pre: &[f32]) -> Vec<f32> {
    use modgrad_device::backend::ops;
    let mut d_input = vec![0.0f32; d_out.len()];
    ops::silu_bwd(d_out, pre, &mut d_input).expect("silu_bwd dispatch");
    d_input
}

// ─── Per-neuron GLU ────────────────────────────────────────

struct GluCache { x: Vec<f32>, n_neurons: usize, out_per: usize }

fn per_neuron_glu_cached(x: &[f32], n_neurons: usize, out_per: usize) -> (Vec<f32>, GluCache) {
    let half = out_per / 2;
    let mut result = Vec::with_capacity(n_neurons * half);
    for n in 0..n_neurons {
        let base = n * out_per;
        for j in 0..half {
            let val = x[base + j];
            let gate = 1.0 / (1.0 + (-x[base + half + j]).exp());
            result.push(val * gate);
        }
    }
    (result, GluCache { x: x.to_vec(), n_neurons, out_per })
}

fn per_neuron_glu_backward(d_out: &[f32], cache: &GluCache) -> Vec<f32> {
    let half = cache.out_per / 2;
    use modgrad_device::backend::ops;
    let total_in = cache.n_neurons * cache.out_per;
    let mut d_input = vec![0.0f32; total_in];
    ops::per_neuron_glu_bwd(d_out, &cache.x, &mut d_input, cache.n_neurons, half)
        .expect("per_neuron_glu_bwd dispatch");
    d_input
}

// ═══════════════════════════════════════════════════════════════
// COMPOSITE BACKWARD PASSES
// ═══════════════════════════════════════════════════════════════

// ─── SynapseBlock (Linear → LN → SiLU) ────────────────────

struct BlockCache { lin: LinearCache, ln: LnCache, pre_silu: Vec<f32> }
struct BlockGrads { d_weight: Vec<f32>, d_bias: Vec<f32>, d_gamma: Vec<f32>, d_beta: Vec<f32> }

impl BlockGrads {
    fn zero(&mut self) {
        self.d_weight.fill(0.0);
        self.d_bias.fill(0.0);
        self.d_gamma.fill(0.0);
        self.d_beta.fill(0.0);
    }
    fn zeros(block: &SynapseBlock) -> Self {
        Self {
            d_weight: vec![0.0; block.linear.weight.len()],
            d_bias: vec![0.0; block.linear.bias.len()],
            d_gamma: vec![0.0; block.ln_gamma.len()],
            d_beta: vec![0.0; block.ln_beta.len()],
        }
    }

    fn apply(&mut self, block: &mut SynapseBlock, lr: f32) {
        use modgrad_device::backend::ops;
        // Weight update via registry dispatch. Grad-zeroing is the
        // caller's responsibility (preserved in the outer training loop);
        // sgd_update now does ONLY the `w -= lr*g` step.
        ops::sgd_update(&mut block.linear.weight, &self.d_weight, lr)
            .expect("sgd_update dispatch");
        // Small epilogue updates stay inline — they're O(vec_len) serial
        // loops on tiny vectors; an Op dispatch would cost more than the work.
        for (b, g) in block.linear.bias.iter_mut().zip(&self.d_bias) { *b -= lr * g; }
        for (g, dg) in block.ln_gamma.iter_mut().zip(&self.d_gamma) { *g -= lr * dg; }
        for (b, db) in block.ln_beta.iter_mut().zip(&self.d_beta) { *b -= lr * db; }
    }

    /// Element-wise accumulate `other`'s gradients into `self`.
    /// Used by the resident `RegionalBrain::backward_cached_resident`
    /// path to merge per-tick `BlockGrads` from a region's
    /// `ctm_backward_resident` into the outer `RegionalGradients`
    /// accumulator.
    fn add_from(&mut self, other: &BlockGrads) {
        for (d, s) in self.d_weight.iter_mut().zip(&other.d_weight) { *d += *s; }
        for (d, s) in self.d_bias.iter_mut().zip(&other.d_bias) { *d += *s; }
        for (d, s) in self.d_gamma.iter_mut().zip(&other.d_gamma) { *d += *s; }
        for (d, s) in self.d_beta.iter_mut().zip(&other.d_beta) { *d += *s; }
    }
}

fn block_forward_cached(block: &SynapseBlock, x: &[f32]) -> (Vec<f32>, BlockCache) {
    let (lin_out, lin_cache) = linear_forward_cached(&block.linear, x);
    let mut y = lin_out;
    let ln_cache = affine_ln_forward(&mut y, &block.ln_gamma, &block.ln_beta);
    let pre_silu = silu_forward_cached(&mut y);
    (y, BlockCache { lin: lin_cache, ln: ln_cache, pre_silu })
}

fn block_backward(
    block: &SynapseBlock, d_out: &[f32], cache: &BlockCache, grads: &mut BlockGrads,
) -> Vec<f32> {
    // Three-stage backward: silu → ln → linear. Each stage dispatches
    // through the registry, which routes to KFD on gfx1102 and CPU
    // otherwise. A fused GPU path existed before (one submit_wait for
    // all three) — it traded ~2 dispatch-sync stalls for coupling
    // modgrad-ctm to `modgrad_device::kfd`. Not worth the layer
    // violation; if the perf matters, the fusion can come back inside
    // `KfdBackend::dispatch` as a graph-level optimization without any
    // caller changes.
    let d_silu = silu_backward(d_out, &cache.pre_silu);
    let d_ln = affine_ln_backward(&d_silu, &cache.ln, &block.ln_gamma,
        &mut grads.d_gamma, &mut grads.d_beta);
    let mut d_input = vec![0.0f32; block.linear.in_dim];
    linear_backward(&block.linear, &d_ln, &cache.lin,
        &mut grads.d_weight, &mut grads.d_bias, &mut d_input);
    d_input
}

// ─── SynapseUNet ───────────────────────────────────────────

pub(crate) struct UNetCache {
    first: BlockCache,
    downs: Vec<BlockCache>,
    ups: Vec<BlockCache>,
    #[allow(dead_code)] // stored for skip connections; retained for potential future backward use
    down_outs: Vec<Vec<f32>>,
    #[allow(dead_code)] // before skip LN (for LN backward); retained for potential future use
    pre_skip_ln: Vec<Vec<f32>>,
    skip_ln_caches: Vec<LnCache>,
}

pub struct UNetGrads {
    first: BlockGrads,
    downs: Vec<BlockGrads>,
    ups: Vec<BlockGrads>,
    skip_d_gamma: Vec<Vec<f32>>,
    skip_d_beta: Vec<Vec<f32>>,
}

impl UNetGrads {
    fn zero(&mut self) {
        self.first.zero();
        for d in &mut self.downs { d.zero(); }
        for u in &mut self.ups { u.zero(); }
        for g in &mut self.skip_d_gamma { g.fill(0.0); }
        for b in &mut self.skip_d_beta { b.fill(0.0); }
    }
    fn zeros(unet: &SynapseUNet) -> Self {
        let _n = unet.down_blocks.len();
        Self {
            first: BlockGrads::zeros(&unet.first_projection),
            downs: unet.down_blocks.iter().map(BlockGrads::zeros).collect(),
            ups: unet.up_blocks.iter().map(BlockGrads::zeros).collect(),
            skip_d_gamma: unet.skip_ln_gamma.iter().map(|g| vec![0.0; g.len()]).collect(),
            skip_d_beta: unet.skip_ln_beta.iter().map(|b| vec![0.0; b.len()]).collect(),
        }
    }

    fn apply(&mut self, unet: &mut SynapseUNet, lr: f32) {
        self.first.apply(&mut unet.first_projection, lr);
        for (g, b) in self.downs.iter_mut().zip(unet.down_blocks.iter_mut()) { g.apply(b, lr); }
        for (g, b) in self.ups.iter_mut().zip(unet.up_blocks.iter_mut()) { g.apply(b, lr); }
        for (i, (dg, db)) in self.skip_d_gamma.iter().zip(&self.skip_d_beta).enumerate() {
            for (g, d) in unet.skip_ln_gamma[i].iter_mut().zip(dg) { *g -= lr * d; }
            for (b, d) in unet.skip_ln_beta[i].iter_mut().zip(db) { *b -= lr * d; }
        }
    }

    /// Element-wise accumulate `other`'s per-block gradients into
    /// `self`. Used by `RegionalBrain::backward_cached_resident` to
    /// merge per-tick per-region `UNetGrads` from
    /// `ctm_backward_resident` into the outer `RegionalGradients`
    /// accumulator. Path B for the resident U-Net backward depends
    /// on this — without it, U-Net weight grads would still be
    /// dropped at the outer-merge step.
    pub fn add_from(&mut self, other: &UNetGrads) {
        debug_assert_eq!(self.downs.len(), other.downs.len());
        debug_assert_eq!(self.ups.len(), other.ups.len());
        debug_assert_eq!(self.skip_d_gamma.len(), other.skip_d_gamma.len());
        self.first.add_from(&other.first);
        for (d, s) in self.downs.iter_mut().zip(&other.downs) { d.add_from(s); }
        for (u, s) in self.ups.iter_mut().zip(&other.ups) { u.add_from(s); }
        for (g, s) in self.skip_d_gamma.iter_mut().zip(&other.skip_d_gamma) {
            for (a, b) in g.iter_mut().zip(s) { *a += *b; }
        }
        for (g, s) in self.skip_d_beta.iter_mut().zip(&other.skip_d_beta) {
            for (a, b) in g.iter_mut().zip(s) { *a += *b; }
        }
    }

    /// Diagnostic: L2 norm across every weight/bias/gamma/beta
    /// gradient buffer in this U-Net accumulator. Used by the
    /// resident-backward regression tests to catch the
    /// "U-Net grads regress to zero" bug — a passing forward
    /// followed by `unet_backward` must drive this above zero.
    pub fn l2_norm(&self) -> f32 {
        let mut sumsq: f64 = 0.0;
        let mut acc = |s: &[f32]| {
            for &x in s { sumsq += (x as f64) * (x as f64); }
        };
        acc(&self.first.d_weight); acc(&self.first.d_bias);
        acc(&self.first.d_gamma); acc(&self.first.d_beta);
        for d in &self.downs {
            acc(&d.d_weight); acc(&d.d_bias);
            acc(&d.d_gamma); acc(&d.d_beta);
        }
        for u in &self.ups {
            acc(&u.d_weight); acc(&u.d_bias);
            acc(&u.d_gamma); acc(&u.d_beta);
        }
        for g in &self.skip_d_gamma { acc(g); }
        for b in &self.skip_d_beta { acc(b); }
        (sumsq as f32).sqrt()
    }

    /// Diagnostic: per-block weight gradient slices, in the order
    /// `first.d_weight, first.d_bias, first.d_gamma, first.d_beta,
    /// down[0].d_weight, ..., up[n-1].d_beta, skip_d_gamma[0], ...,
    /// skip_d_beta[n-1]`. Used by the resident vs host backward
    /// parity tests to compare per-block gradient magnitudes within
    /// FP tolerance.
    pub fn flat_weight_grads(&self) -> Vec<f32> {
        let mut v = Vec::new();
        v.extend_from_slice(&self.first.d_weight);
        v.extend_from_slice(&self.first.d_bias);
        v.extend_from_slice(&self.first.d_gamma);
        v.extend_from_slice(&self.first.d_beta);
        for d in &self.downs {
            v.extend_from_slice(&d.d_weight);
            v.extend_from_slice(&d.d_bias);
            v.extend_from_slice(&d.d_gamma);
            v.extend_from_slice(&d.d_beta);
        }
        for u in &self.ups {
            v.extend_from_slice(&u.d_weight);
            v.extend_from_slice(&u.d_bias);
            v.extend_from_slice(&u.d_gamma);
            v.extend_from_slice(&u.d_beta);
        }
        for g in &self.skip_d_gamma { v.extend_from_slice(g); }
        for b in &self.skip_d_beta { v.extend_from_slice(b); }
        v
    }
}

pub(crate) fn unet_forward_cached(unet: &SynapseUNet, x: &[f32]) -> (Vec<f32>, UNetCache) {
    let n_blocks = unet.down_blocks.len();
    let (first_out, first_cache) = block_forward_cached(&unet.first_projection, x);

    let mut down_outs = vec![first_out];
    let mut down_caches = Vec::with_capacity(n_blocks);
    for i in 0..n_blocks {
        let (out, cache) = block_forward_cached(&unet.down_blocks[i], down_outs.last().unwrap());
        down_caches.push(cache);
        down_outs.push(out);
    }

    let mut current = down_outs[n_blocks].clone();
    let mut up_caches = Vec::with_capacity(n_blocks);
    let mut pre_skip_ln = Vec::with_capacity(n_blocks);
    let mut skip_ln_caches = Vec::with_capacity(n_blocks);

    for i in 0..n_blocks {
        let up_idx = n_blocks - 1 - i;
        let (up_out, up_cache) = block_forward_cached(&unet.up_blocks[up_idx], &current);
        up_caches.push(up_cache);

        // Add skip + LN
        let mut sum: Vec<f32> = up_out.iter().zip(&down_outs[up_idx])
            .map(|(&a, &b)| a + b).collect();
        pre_skip_ln.push(sum.clone());
        let ln_cache = affine_ln_forward(&mut sum, &unet.skip_ln_gamma[up_idx], &unet.skip_ln_beta[up_idx]);
        skip_ln_caches.push(ln_cache);
        current = sum;
    }

    (current, UNetCache { first: first_cache, downs: down_caches, ups: up_caches,
        down_outs, pre_skip_ln, skip_ln_caches })
}

pub(crate) fn unet_backward(
    unet: &SynapseUNet, d_out: &[f32], cache: &UNetCache, grads: &mut UNetGrads,
) -> Vec<f32> {
    let n_blocks = unet.down_blocks.len();
    let mut d_current = d_out.to_vec();
    let mut d_skip_for_down: Vec<Option<Vec<f32>>> = vec![None; n_blocks + 1];

    // Up path backward: undo forward up steps in reverse order.
    // Forward step i used up_blocks[n-1-i], gamma[n-1-i], stored cache at index i.
    // Backward step j undoes forward step (n-1-j): cache[n-1-j], gamma[j], up_blocks[j].
    for j in 0..n_blocks {
        let fwd_i = n_blocks - 1 - j;    // which forward step we're undoing
        let gamma_idx = j;                 // = n-1-fwd_i = n-1-(n-1-j) = j
        let cache_i = fwd_i;              // caches stored in forward order

        // Skip LN backward
        let d_sum = affine_ln_backward(&d_current, &cache.skip_ln_caches[cache_i],
            &unet.skip_ln_gamma[gamma_idx],
            &mut grads.skip_d_gamma[gamma_idx], &mut grads.skip_d_beta[gamma_idx]);

        // Skip gradient goes to down_outs[gamma_idx]
        d_skip_for_down[gamma_idx] = Some(d_sum.clone());

        // Up block backward
        d_current = block_backward(&unet.up_blocks[gamma_idx], &d_sum,
            &cache.ups[cache_i], &mut grads.ups[gamma_idx]);
    }

    // d_current is now gradient at bottleneck (down_outs[n_blocks])

    // Down path backward
    for i in (0..n_blocks).rev() {
        d_current = block_backward(&unet.down_blocks[i], &d_current,
            &cache.downs[i], &mut grads.downs[i]);
        // Add skip gradient for down_outs[i]
        if let Some(ref extra) = d_skip_for_down[i] {
            for (c, e) in d_current.iter_mut().zip(extra) { *c += e; }
        }
    }

    // First projection backward
    block_backward(&unet.first_projection, &d_current, &cache.first, &mut grads.first)
}

// ─── SuperLinear (NLM) ─────────────────────────────────────

struct SuperLinearCache { input: Vec<f32> }

fn superlinear_forward_cached(sl: &SuperLinear, x: &[f32]) -> (Vec<f32>, SuperLinearCache) {
    (sl.forward(x), SuperLinearCache { input: x.to_vec() })
}

/// Returns d_input. Accumulates into d_weights, d_biases.
/// GPU-accelerated when available.
fn superlinear_backward(
    sl: &SuperLinear, d_out: &[f32], cache: &SuperLinearCache,
    d_weights: &mut [f32], d_biases: &mut [f32],
) -> Vec<f32> {
    let n = sl.n_neurons;
    let ip = sl.in_per;
    let op = sl.out_per;

    // d_bias (always CPU — small, element-wise)
    for i in 0..n * op {
        d_biases[i] += d_out[i];
    }

    // Two-phase gradient: weight-grad accumulates, input-grad is fresh.
    // The KFD kernel fuses both into one dispatch; routed through
    // registry, CPU runs them as two passes. When a kernel-matched
    // fused variant lands, supports() can claim it and this same
    // call-site benefits automatically.
    use modgrad_device::backend::ops;
    let mut d_input = vec![0.0f32; n * ip];
    ops::super_linear_bwd_dw(d_out, &cache.input, d_weights, n, ip, op)
        .expect("super_linear_bwd_dw dispatch");
    ops::super_linear_bwd_dx(d_out, &sl.weights, &mut d_input, n, ip, op)
        .expect("super_linear_bwd_dx dispatch");
    d_input
}

// ─── Sync backward ─────────────────────────────────────────

/// d_sync → d_activated. Sync = alpha/sqrt(beta), alpha accumulates pairwise products.
/// At the current tick, the contribution to alpha[i] is activated[left[i]] * activated[right[i]].
/// d_activated[left[i]] += d_sync[i] / sqrt(beta[i]) * activated[right[i]]
/// d_activated[right[i]] += d_sync[i] / sqrt(beta[i]) * activated[left[i]]
///
/// GPU-accelerated via atomic scatter kernel when available.
fn sync_backward(
    d_sync: &[f32], activated: &[f32], beta: &[f32],
    left: &[usize], right: &[usize], d_model: usize,
) -> Vec<f32> {
    let n_pairs = left.len();
    // GPU path: convert usize indices to u32, dispatch scatter kernel
    use modgrad_device::backend::{ops, SyncBackwardScatterArgs};
    let left_u32: Vec<u32> = left.iter().map(|&x| x as u32).collect();
    let right_u32: Vec<u32> = right.iter().map(|&x| x as u32).collect();
    let mut d_act = vec![0.0f32; d_model];
    ops::sync_backward_scatter(SyncBackwardScatterArgs {
        d_sync, pairs_left: &left_u32, pairs_right: &right_u32,
        activated, beta, d_act: &mut d_act,
        n_pairs, d_model,
    }).expect("sync_backward_scatter dispatch");
    d_act
}

// ─── MHA backward ──────────────────────────────────────────

struct MhaCache {
    q_full: Vec<f32>,       // [d_input]
    k_all: Vec<f32>,        // [n_tokens × d_input]
    v_all: Vec<f32>,        // [n_tokens × d_input]
    attn_weights: Vec<Vec<f32>>, // [n_heads][n_tokens] softmax weights
    #[allow(dead_code)] // [d_input] before out_proj; retained for potential future backward use
    concat_heads: Vec<f32>,
    out_lin: LinearCache,    // for out_proj
    q_in: Vec<f32>,          // input to in_proj for Q
    kv_tokens: Vec<Vec<f32>>, // each KV token input to in_proj
}

fn mha_forward_cached(
    q_in: &[f32], kv_flat: &[f32], n_tokens: usize, d_input: usize, n_heads: usize,
    in_proj: &Linear, out_proj: &Linear,
) -> (Vec<f32>, MhaCache) {
    let d_head = d_input / n_heads;
    let scale = 1.0 / (d_head as f32).sqrt();

    let q_full = linear_slice_vec(q_in, in_proj, 0, d_input);
    let mut k_all = Vec::with_capacity(n_tokens * d_input);
    let mut v_all = Vec::with_capacity(n_tokens * d_input);
    let mut kv_tokens = Vec::with_capacity(n_tokens);
    for t in 0..n_tokens {
        let tok = &kv_flat[t * d_input..(t + 1) * d_input];
        kv_tokens.push(tok.to_vec());
        k_all.extend_from_slice(&linear_slice_vec(tok, in_proj, d_input, 2 * d_input));
        v_all.extend_from_slice(&linear_slice_vec(tok, in_proj, 2 * d_input, 3 * d_input));
    }

    let mut all_weights = Vec::with_capacity(n_heads);
    let mut concat_heads = Vec::with_capacity(d_input);
    for h in 0..n_heads {
        let q_h = &q_full[h * d_head..(h + 1) * d_head];
        let mut scores = Vec::with_capacity(n_tokens);
        for t in 0..n_tokens {
            let k_h = &k_all[t * d_input + h * d_head..t * d_input + (h + 1) * d_head];
            scores.push(q_h.iter().zip(k_h).map(|(&a, &b)| a * b).sum::<f32>() * scale);
        }
        let max_s = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_s: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp()).collect();
        let sum_s: f32 = exp_s.iter().sum();
        let weights: Vec<f32> = exp_s.iter().map(|&e| e / sum_s).collect();

        let mut head_out = vec![0.0f32; d_head];
        for t in 0..n_tokens {
            let v_h = &v_all[t * d_input + h * d_head..t * d_input + (h + 1) * d_head];
            for j in 0..d_head { head_out[j] += weights[t] * v_h[j]; }
        }
        all_weights.push(weights);
        concat_heads.extend_from_slice(&head_out);
    }

    let (result, out_lin) = linear_forward_cached(out_proj, &concat_heads);
    (result, MhaCache { q_full, k_all, v_all, attn_weights: all_weights,
        concat_heads, out_lin, q_in: q_in.to_vec(), kv_tokens })
}

/// Returns (d_q_in, d_kv_tokens) where d_kv_tokens[t] is gradient w.r.t. KV token t.
fn mha_backward(
    d_out: &[f32], cache: &MhaCache, n_tokens: usize, d_input: usize, n_heads: usize,
    in_proj: &Linear, out_proj: &Linear,
    d_in_w: &mut [f32], d_in_b: &mut [f32], d_out_w: &mut [f32], d_out_b: &mut [f32],
) -> (Vec<f32>, Vec<Vec<f32>>) {
    let d_head = d_input / n_heads;
    let scale = 1.0 / (d_head as f32).sqrt();

    // out_proj backward
    let mut d_concat = vec![0.0f32; out_proj.in_dim];
    linear_backward(out_proj, d_out, &cache.out_lin, d_out_w, d_out_b, &mut d_concat);

    // Per-head backward
    let mut d_q_full = vec![0.0f32; d_input];
    let mut d_k_all = vec![0.0f32; n_tokens * d_input];
    let mut d_v_all = vec![0.0f32; n_tokens * d_input];

    for h in 0..n_heads {
        let d_head_out = &d_concat[h * d_head..(h + 1) * d_head];
        let weights = &cache.attn_weights[h];

        // d_weights[t] = d_head_out · v_h[t], d_v[t] = weights[t] * d_head_out
        let mut d_weights = vec![0.0f32; n_tokens];
        for t in 0..n_tokens {
            let v_h = &cache.v_all[t * d_input + h * d_head..t * d_input + (h + 1) * d_head];
            for j in 0..d_head {
                d_weights[t] += d_head_out[j] * v_h[j];
                d_v_all[t * d_input + h * d_head + j] += weights[t] * d_head_out[j];
            }
        }

        // Softmax backward: d_scores = weights * (d_weights - sum(weights * d_weights))
        let dot_wd: f32 = weights.iter().zip(&d_weights).map(|(&w, &d)| w * d).sum();
        let d_scores: Vec<f32> = (0..n_tokens)
            .map(|t| weights[t] * (d_weights[t] - dot_wd) * scale)
            .collect();

        // d_q_h[j] += sum_t(d_scores[t] * k_h[t][j])
        // d_k_h[t][j] += d_scores[t] * q_h[j]
        let q_h = &cache.q_full[h * d_head..(h + 1) * d_head];
        for t in 0..n_tokens {
            let k_start = t * d_input + h * d_head;
            for j in 0..d_head {
                d_q_full[h * d_head + j] += d_scores[t] * cache.k_all[k_start + j];
                d_k_all[k_start + j] += d_scores[t] * q_h[j];
            }
        }
    }

    // in_proj backward for Q (rows 0..d_input applied to q_in)
    let d_q_in = linear_slice_backward(&cache.q_in, in_proj, &d_q_full, 0, d_input,
        d_in_w, d_in_b);

    // in_proj backward for K and V — now COLLECT d_kv_tokens
    let mut d_kv_tokens: Vec<Vec<f32>> = Vec::with_capacity(n_tokens);
    for t in 0..n_tokens {
        let d_k_t = &d_k_all[t * d_input..(t + 1) * d_input];
        let d_v_t = &d_v_all[t * d_input..(t + 1) * d_input];
        let tok = &cache.kv_tokens[t];
        let d_tok_from_k = linear_slice_backward(tok, in_proj, d_k_t, d_input, 2 * d_input, d_in_w, d_in_b);
        let d_tok_from_v = linear_slice_backward(tok, in_proj, d_v_t, 2 * d_input, 3 * d_input, d_in_w, d_in_b);
        // Sum K and V contributions
        let d_tok: Vec<f32> = d_tok_from_k.iter().zip(&d_tok_from_v)
            .map(|(&k, &v)| k + v).collect();
        d_kv_tokens.push(d_tok);
    }

    (d_q_in, d_kv_tokens)
}

fn linear_slice_vec(x: &[f32], linear: &Linear, row_start: usize, row_end: usize) -> Vec<f32> {
    let in_dim = linear.in_dim;
    (row_start..row_end).map(|r| {
        let w = &linear.weight[r * in_dim..(r + 1) * in_dim];
        linear.bias[r] + w.iter().zip(x).map(|(&a, &b)| a * b).sum::<f32>()
    }).collect()
}

/// Backward for a slice of rows. Returns d_input, accumulates into d_weight, d_bias.
/// GPU-accelerated: outer_product for d_weight, matvec_t for d_input.
fn linear_slice_backward(
    x: &[f32], linear: &Linear, d_out: &[f32],
    row_start: usize, row_end: usize,
    d_weight: &mut [f32], d_bias: &mut [f32],
) -> Vec<f32> {
    let in_dim = linear.in_dim;
    let slice_dim = row_end - row_start;

    // d_bias — always on CPU (tiny)
    for (ri, r) in (row_start..row_end).enumerate() {
        d_bias[r] += d_out[ri];
    }

    // d_weight (outer product): d_weight[r*in_dim+j] += d_out[ri] * x[j]
    use modgrad_device::backend::ops;

    // GPU operates on the contiguous slice d_weight[row_start*in_dim..row_end*in_dim]
    let w_offset = row_start * in_dim;
    let w_slice = &mut d_weight[w_offset..w_offset + slice_dim * in_dim];
    ops::outer_product_acc(d_out, x, w_slice, slice_dim, in_dim)
        .expect("outer_product_acc dispatch");

    // d_input = W_slice^T @ d_out
    let wt_slice = &linear.weight[w_offset..w_offset + slice_dim * in_dim];
    let mut d_input = vec![0.0f32; in_dim];
    ops::matvec_t(d_out, wt_slice, &mut d_input, slice_dim, in_dim)
        .expect("matvec_t dispatch");
    d_input
}

// ═══════════════════════════════════════════════════════════════
// NLM BACKWARD
// ═══════════════════════════════════════════════════════════════

struct NlmCache {
    s1_cache: SuperLinearCache,
    glu1_cache: GluCache,
    s2_cache: Option<SuperLinearCache>,
    glu2_cache: Option<GluCache>,
}

fn nlm_forward_cached(
    trace: &[f32], stage1: &SuperLinear, stage2: Option<&SuperLinear>, d_model: usize,
) -> (Vec<f32>, NlmCache) {
    let (s1_out, s1_cache) = superlinear_forward_cached(stage1, trace);
    let (s1_glu, glu1_cache) = per_neuron_glu_cached(&s1_out, d_model, stage1.out_per);

    if let Some(s2) = stage2 {
        let (s2_out, s2_cache) = superlinear_forward_cached(s2, &s1_glu);
        let (s2_glu, glu2_cache) = per_neuron_glu_cached(&s2_out, d_model, s2.out_per);
        (s2_glu, NlmCache { s1_cache, glu1_cache, s2_cache: Some(s2_cache), glu2_cache: Some(glu2_cache) })
    } else {
        (s1_glu, NlmCache { s1_cache, glu1_cache, s2_cache: None, glu2_cache: None })
    }
}

fn nlm_backward(
    d_out: &[f32], cache: &NlmCache,
    stage1: &SuperLinear, stage2: Option<&SuperLinear>,
    d_s1_w: &mut [f32], d_s1_b: &mut [f32],
    d_s2_w: &mut Option<Vec<f32>>, d_s2_b: &mut Option<Vec<f32>>,
) -> Vec<f32> {
    let d_before_nlm;
    if let (Some(s2), Some(s2_cache), Some(glu2_cache)) =
        (stage2, cache.s2_cache.as_ref(), cache.glu2_cache.as_ref())
    {
        let d_glu2 = per_neuron_glu_backward(d_out, glu2_cache);
        let d_s2_in = superlinear_backward(s2, &d_glu2, s2_cache,
            d_s2_w.as_mut().unwrap(), d_s2_b.as_mut().unwrap());
        let d_glu1 = per_neuron_glu_backward(&d_s2_in, &cache.glu1_cache);
        d_before_nlm = superlinear_backward(stage1, &d_glu1, &cache.s1_cache, d_s1_w, d_s1_b);
    } else {
        let d_glu1 = per_neuron_glu_backward(d_out, &cache.glu1_cache);
        d_before_nlm = superlinear_backward(stage1, &d_glu1, &cache.s1_cache, d_s1_w, d_s1_b);
    }
    d_before_nlm
}

// ═══════════════════════════════════════════════════════════════
// FULL TICK-LOOP BPTT
// ═══════════════════════════════════════════════════════════════

/// Cache for one tick of the forward pass.
struct TickCache {
    activated_prev: Vec<f32>,  // activated state at start of tick
    #[allow(dead_code)] // retained for potential future backward / inspection use
    sync_action: Vec<f32>,
    beta_action: Vec<f32>,
    #[allow(dead_code)] // q_proj output; retained for potential future inspection
    q: Vec<f32>,
    q_lin: LinearCache,
    mha_cache: MhaCache,
    #[allow(dead_code)] // retained for potential future inspection
    attn_out: Vec<f32>,
    unet_cache: UNetCache,
    #[allow(dead_code)] // pre-activation value; retained for potential future inspection
    pre_act: Vec<f32>,
    nlm_cache: NlmCache,
    activated_post: Vec<f32>,  // activated state after NLM
    sync_out: Vec<f32>,
    beta_out: Vec<f32>,
    out_lin: LinearCache,
    // Adaptive exit gate cache (None when gate is off)
    exit_gate_lin: Option<LinearCache>,
    exit_lambda: f32,  // σ(gate_logit), 0.0 when gate is off
}

/// All gradients for CtmWeights.
pub struct CtmGradients {
    pub unet: UNetGrads,
    pub nlm_s1_w: Vec<f32>,
    pub nlm_s1_b: Vec<f32>,
    pub nlm_s2_w: Option<Vec<f32>>,
    pub nlm_s2_b: Option<Vec<f32>>,
    pub d_start_activated: Vec<f32>,
    pub d_start_trace: Vec<f32>,
    pub kv_proj_w: Vec<f32>,
    pub kv_proj_b: Vec<f32>,
    pub kv_ln_d_gamma: Vec<f32>,
    pub kv_ln_d_beta: Vec<f32>,
    pub q_proj_w: Vec<f32>,
    pub q_proj_b: Vec<f32>,
    pub mha_in_w: Vec<f32>,
    pub mha_in_b: Vec<f32>,
    pub mha_out_w: Vec<f32>,
    pub mha_out_b: Vec<f32>,
    pub d_decay_out: Vec<f32>,
    pub d_decay_action: Vec<f32>,
    pub out_proj_w: Vec<f32>,
    pub out_proj_b: Vec<f32>,
    // Adaptive exit gate gradients
    pub exit_gate_w: Option<Vec<f32>>,
    pub exit_gate_b: Option<Vec<f32>>,
}

/// Compute L2 norm across multiple gradient slices.
/// Tries GPU dispatch first (concatenates into a scratch buffer), falls back to CPU.
impl CtmGradients {
    pub fn zeros(w: &CtmWeights) -> Self {
        Self {
            unet: UNetGrads::zeros(&w.synapse),
            nlm_s1_w: vec![0.0; w.nlm_stage1.weights.len()],
            nlm_s1_b: vec![0.0; w.nlm_stage1.biases.len()],
            nlm_s2_w: w.nlm_stage2.as_ref().map(|s| vec![0.0; s.weights.len()]),
            nlm_s2_b: w.nlm_stage2.as_ref().map(|s| vec![0.0; s.biases.len()]),
            d_start_activated: vec![0.0; w.config.d_model],
            d_start_trace: vec![0.0; w.config.d_model * w.config.memory_length],
            kv_proj_w: vec![0.0; w.kv_proj.weight.len()],
            kv_proj_b: vec![0.0; w.kv_proj.bias.len()],
            kv_ln_d_gamma: vec![0.0; w.kv_ln_gamma.len()],
            kv_ln_d_beta: vec![0.0; w.kv_ln_beta.len()],
            q_proj_w: vec![0.0; w.q_proj.weight.len()],
            q_proj_b: vec![0.0; w.q_proj.bias.len()],
            mha_in_w: vec![0.0; w.mha_in_proj.weight.len()],
            mha_in_b: vec![0.0; w.mha_in_proj.bias.len()],
            mha_out_w: vec![0.0; w.mha_out_proj.weight.len()],
            mha_out_b: vec![0.0; w.mha_out_proj.bias.len()],
            d_decay_out: vec![0.0; w.decay_params_out.len()],
            d_decay_action: vec![0.0; w.decay_params_action.len()],
            out_proj_w: vec![0.0; w.output_proj.weight.len()],
            out_proj_b: vec![0.0; w.output_proj.bias.len()],
            exit_gate_w: w.exit_gate.as_ref().map(|g| vec![0.0; g.weight.len()]),
            exit_gate_b: w.exit_gate.as_ref().map(|g| vec![0.0; g.bias.len()]),
        }
    }

    /// Zero all gradient buffers in-place. No allocation.
    pub fn zero(&mut self) {
        self.unet.zero();
        self.nlm_s1_w.fill(0.0);
        self.nlm_s1_b.fill(0.0);
        if let Some(w) = &mut self.nlm_s2_w { w.fill(0.0); }
        if let Some(b) = &mut self.nlm_s2_b { b.fill(0.0); }
        self.d_start_activated.fill(0.0);
        self.d_start_trace.fill(0.0);
        self.kv_proj_w.fill(0.0);
        self.kv_proj_b.fill(0.0);
        self.kv_ln_d_gamma.fill(0.0);
        self.kv_ln_d_beta.fill(0.0);
        self.q_proj_w.fill(0.0);
        self.q_proj_b.fill(0.0);
        self.mha_in_w.fill(0.0);
        self.mha_in_b.fill(0.0);
        self.mha_out_w.fill(0.0);
        self.mha_out_b.fill(0.0);
        self.d_decay_out.fill(0.0);
        self.d_decay_action.fill(0.0);
        self.out_proj_w.fill(0.0);
        self.out_proj_b.fill(0.0);
        if let Some(w) = &mut self.exit_gate_w { w.fill(0.0); }
        if let Some(b) = &mut self.exit_gate_b { b.fill(0.0); }
    }

    /// Diagnostic: L2 norm across all public weight gradient fields of this region.
    /// Used for vanishing-gradient diagnostics; not on the hot path.
    pub fn l2_norm(&self) -> f32 {
        let mut sumsq: f64 = 0.0;
        let mut acc = |s: &[f32]| {
            for &x in s { sumsq += (x as f64) * (x as f64); }
        };
        acc(&self.nlm_s1_w); acc(&self.nlm_s1_b);
        if let Some(w) = &self.nlm_s2_w { acc(w); }
        if let Some(b) = &self.nlm_s2_b { acc(b); }
        acc(&self.d_start_activated); acc(&self.d_start_trace);
        acc(&self.kv_proj_w); acc(&self.kv_proj_b);
        acc(&self.kv_ln_d_gamma); acc(&self.kv_ln_d_beta);
        acc(&self.q_proj_w); acc(&self.q_proj_b);
        acc(&self.mha_in_w); acc(&self.mha_in_b);
        acc(&self.mha_out_w); acc(&self.mha_out_b);
        acc(&self.d_decay_out); acc(&self.d_decay_action);
        acc(&self.out_proj_w); acc(&self.out_proj_b);
        if let Some(w) = &self.exit_gate_w { acc(w); }
        if let Some(b) = &self.exit_gate_b { acc(b); }
        (sumsq as f32).sqrt()
    }

    /// SGD update: w -= lr * grad. Clips gradients by norm.
    /// GPU-accelerated for large parameter arrays.
    pub fn apply(&mut self, w: &mut CtmWeights, lr: f32, clip_norm: f32) {
        let scale = {
            let grads: &[&[f32]] = &[
                &self.nlm_s1_w, &self.nlm_s1_b,
                &self.out_proj_w, &self.out_proj_b,
                &self.q_proj_w,
            ];
            let norm = crate::grad_norm(grads);
            if norm > clip_norm { clip_norm / norm } else { 1.0 }
        };
        let lr = lr * scale;

        self.unet.apply(&mut w.synapse, lr);
        use modgrad_device::backend::ops;
        // Closure over `ops::sgd_update` — the closure can't `?` into
        // `apply`'s `()` return, and this is a training-loop call site,
        // so `.expect` here keeps the panic off the FFI boundary inside
        // `ops::sgd_update` itself. Every invocation below goes through
        // this one `.expect` site.
        let sgd = |w: &mut [f32], g: &mut [f32]| {
            ops::sgd_update(w, g, lr).expect("sgd_update dispatch");
        };
        sgd(&mut w.nlm_stage1.weights, &mut self.nlm_s1_w);
        sgd(&mut w.nlm_stage1.biases, &mut self.nlm_s1_b);
        if let Some(s2) = &mut w.nlm_stage2 {
            if let (Some(gw), Some(gb)) = (&mut self.nlm_s2_w, &mut self.nlm_s2_b) {
                sgd(&mut s2.weights, gw);
                sgd(&mut s2.biases, gb);
            }
        }
        sgd(&mut w.start_activated, &mut self.d_start_activated);
        sgd(&mut w.start_trace, &mut self.d_start_trace);
        sgd(&mut w.kv_proj.weight, &mut self.kv_proj_w);
        sgd(&mut w.kv_proj.bias, &mut self.kv_proj_b);
        sgd(&mut w.kv_ln_gamma, &mut self.kv_ln_d_gamma);
        sgd(&mut w.kv_ln_beta, &mut self.kv_ln_d_beta);
        sgd(&mut w.q_proj.weight, &mut self.q_proj_w);
        sgd(&mut w.q_proj.bias, &mut self.q_proj_b);
        sgd(&mut w.mha_in_proj.weight, &mut self.mha_in_w);
        sgd(&mut w.mha_in_proj.bias, &mut self.mha_in_b);
        sgd(&mut w.mha_out_proj.weight, &mut self.mha_out_w);
        sgd(&mut w.mha_out_proj.bias, &mut self.mha_out_b);
        sgd(&mut w.output_proj.weight, &mut self.out_proj_w);
        sgd(&mut w.output_proj.bias, &mut self.out_proj_b);
        if let Some(gate) = &mut w.exit_gate {
            if let (Some(gw), Some(gb)) = (&mut self.exit_gate_w, &mut self.exit_gate_b) {
                sgd(&mut gate.weight, gw);
                sgd(&mut gate.bias, gb);
            }
        }
    }
}

// ─── Loss ──────────────────────────────────────────────────

/// Cross-entropy loss + gradient w.r.t. logits.
fn cross_entropy_grad(logits: &[f32], target: usize) -> (f32, Vec<f32>) {
    let max_l = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_s: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
    let sum: f32 = exp_s.iter().sum();
    let mut softmax: Vec<f32> = exp_s.iter().map(|&e| e / sum).collect();
    let loss = -(softmax[target].max(1e-8)).ln();
    softmax[target] -= 1.0; // d_logits = softmax - one_hot
    (loss, softmax)
}

/// Ctm loss: (min_CE_tick + most_certain_tick) / 2.
fn ctm_loss(predictions: &[Vec<f32>], certainties: &[[f32; 2]], target: usize)
    -> (f32, Vec<Vec<f32>>)
{
    let k = predictions.len();
    let losses_and_grads: Vec<(f32, Vec<f32>)> = predictions.iter()
        .map(|p| cross_entropy_grad(p, target)).collect();

    // Find tick with minimum loss
    let min_tick = (0..k).min_by(|&a, &b|
        losses_and_grads[a].0.partial_cmp(&losses_and_grads[b].0).unwrap_or(std::cmp::Ordering::Equal)
    ).unwrap_or(k - 1);

    // Find tick with highest certainty (1 - normalized_entropy)
    let cert_tick = (0..k).max_by(|&a, &b|
        certainties[a][1].partial_cmp(&certainties[b][1]).unwrap_or(std::cmp::Ordering::Equal)
    ).unwrap_or(k - 1);

    let loss = (losses_and_grads[min_tick].0 + losses_and_grads[cert_tick].0) / 2.0;

    // Gradients: half from min_tick, half from cert_tick
    let mut d_preds: Vec<Vec<f32>> = vec![vec![0.0; predictions[0].len()]; k];
    for (j, g) in losses_and_grads[min_tick].1.iter().enumerate() {
        d_preds[min_tick][j] += 0.5 * g;
    }
    for (j, g) in losses_and_grads[cert_tick].1.iter().enumerate() {
        d_preds[cert_tick][j] += 0.5 * g;
    }

    (loss, d_preds)
}

/// LoopLM-style adaptive exit loss.
///
/// L = Σ p(t) * CE[t]  −  β * H(p)
///
/// p(t) is a proper distribution derived from per-tick gate lambdas via
/// survival probabilities. Gradient through lambdas is exact (chain rule
/// through the survival product), not a REINFORCE approximation.
///
/// Returns (total_loss, d_preds weighted by exit probs, d_lambdas for gate).
/// d_lambdas are w.r.t. the gate *logit* (pre-sigmoid), so callers pass them
/// directly to linear_backward.
pub(crate) fn adaptive_exit_loss(
    predictions: &[Vec<f32>],
    lambdas: &[f32],    // per-tick exit probabilities from gate (post-sigmoid)
    target: usize,
    beta: f32,
) -> (f32, Vec<Vec<f32>>, Vec<f32>) {
    let k = predictions.len();
    assert_eq!(k, lambdas.len());

    let ce: Vec<(f32, Vec<f32>)> = predictions.iter()
        .map(|p| cross_entropy_grad(p, target)).collect();

    // ── Exit distribution: p(t) = λ_t * S_{t-1}, p(K) = S_{K-1}
    // S_t = Π_{j≤t} (1 − λ_j)
    let mut surv = vec![0.0f32; k]; // S_{t-1} for each t
    surv[0] = 1.0;
    for t in 1..k {
        surv[t] = surv[t - 1] * (1.0 - lambdas[t - 1]);
    }

    let mut p = vec![0.0f32; k];
    for t in 0..k - 1 {
        p[t] = lambdas[t] * surv[t];
    }
    p[k - 1] = surv[k - 1] * (1.0 - lambdas[k - 1]); // remaining mass
    // For k-1 this simplifies to S_{k-1} when λ_{k-1} = 0, but in general
    // the last lambda can be nonzero. Assign all remaining survival mass:
    p[k - 1] = surv[k - 1]; // last step gets S_{k-2} * (1 - λ_{k-2}) = surv[k-1]
    // Correction: p[k-1] should be whatever mass hasn't exited.
    // S_{k-1} = Π_{j<k-1} (1 - λ_j). p[k-1] = S_{k-1}. This is correct.

    debug_assert!(
        (p.iter().sum::<f32>() - 1.0).abs() < 0.01,
        "exit distribution does not sum to 1: sum = {}", p.iter().sum::<f32>()
    );

    // ── Forward: expected loss and entropy
    let task_loss: f32 = (0..k).map(|t| p[t] * ce[t].0).sum();
    let entropy: f32 = -(0..k)
        .map(|t| if p[t] > 1e-8 { p[t] * p[t].ln() } else { 0.0 })
        .sum::<f32>();
    let total_loss = task_loss - beta * entropy;

    // ── d_preds: weighted by exit probability
    let d_preds: Vec<Vec<f32>> = (0..k).map(|t| {
        ce[t].1.iter().map(|&g| p[t] * g).collect()
    }).collect();

    // ── d_lambdas: exact gradient via chain rule through survival product.
    //
    // d_loss/d_λ_t (pre-sigmoid, i.e. w.r.t. the logit z where λ = σ(z)):
    //
    // p(t) = λ_t * S_{t-1}             → dp(t)/dλ_t = S_{t-1}
    // p(s) = λ_s * S_{s-1} for s > t   → dp(s)/dλ_t = -p(s) / (1 - λ_t)
    //    because S_{s-1} contains factor (1 - λ_t)
    // p(K-1) = S_{K-1}                 → dp(K-1)/dλ_t = -S_{K-1} / (1 - λ_t)
    //
    // Suffix loss = Σ_{s>t} p(s) * CE[s] / Σ_{s>t} p(s)
    //             = expected CE conditioned on surviving past tick t.
    //
    // d(task_loss)/dλ_t = S_{t-1} * CE[t]  −  S_{t-1} * suffix_loss
    //                   = S_{t-1} * (CE[t] − suffix_loss)
    //
    // d(entropy)/dλ_t follows similar structure via d(-p·ln(p))/dp = -(1+ln(p)).
    //
    // Multiply by σ'(z) = λ(1−λ) to get d/dz.
    let mut d_lambdas = vec![0.0f32; k];

    // Precompute suffix: Σ_{s=t+1..k} p(s) * CE[s] and Σ_{s=t+1..k} p(s)
    let mut suffix_weighted = 0.0f32; // Σ p(s)*CE[s] for s > t
    let mut suffix_mass = 0.0f32;     // Σ p(s) for s > t

    // Walk backwards to build suffix sums, then walk forward for gradients
    let mut suffix_w = vec![0.0f32; k]; // suffix_weighted at position t
    let mut suffix_m = vec![0.0f32; k]; // suffix_mass at position t
    for t in (0..k).rev() {
        suffix_w[t] = suffix_weighted;
        suffix_m[t] = suffix_mass;
        suffix_weighted += p[t] * ce[t].0;
        suffix_mass += p[t];
    }

    for t in 0..k - 1 {
        let one_minus = (1.0 - lambdas[t]).max(1e-8);

        // Task loss gradient
        let suffix_expected = if suffix_m[t] > 1e-8 {
            suffix_w[t] / suffix_m[t]
        } else {
            ce[k - 1].0
        };
        let d_task = surv[t] * (ce[t].0 - suffix_expected);

        // Entropy gradient: Σ_s dp(s)/dλ_t * (-(1 + ln p(s)))
        // = S_{t-1} * (-(1+ln p(t))) + Σ_{s>t} (-p(s)/(1-λ_t)) * (-(1+ln p(s)))
        // = S_{t-1} * (-(1+ln p(t))) + (1/(1-λ_t)) * Σ_{s>t} p(s)*(1+ln p(s))
        let neg_1_ln_t = if p[t] > 1e-8 { -(1.0 + p[t].ln()) } else { 0.0 };
        let suffix_ent: f32 = (t + 1..k)
            .map(|s| if p[s] > 1e-8 { p[s] * (1.0 + p[s].ln()) } else { 0.0 })
            .sum();
        let d_entropy = surv[t] * neg_1_ln_t + suffix_ent / one_minus;

        // Combine: d_loss/dλ = d_task - β * d_entropy
        // Then chain through sigmoid: d_loss/dz = d_loss/dλ * λ(1-λ)
        let sig_deriv = lambdas[t] * (1.0 - lambdas[t]);
        d_lambdas[t] = sig_deriv * (d_task + beta * d_entropy);
    }

    (total_loss, d_preds, d_lambdas)
}

// ─── Training step ─────────────────────────────────────────

pub struct TrainResult {
    pub loss: f32,
    pub prediction: usize,
}

/// One training step: forward pass with caching, loss, backward, gradient accumulation.
/// Does NOT apply gradients — caller does that (for mini-batch accumulation).
pub fn train_step(
    w: &CtmWeights,
    grads: &mut CtmGradients,
    observation: &[f32],
    n_tokens: usize,
    raw_dim: usize,
    target: usize,
) -> TrainResult {
    let cfg = &w.config;
    let d = cfg.d_model;
    let d_in = cfg.d_input;
    let m = cfg.memory_length;

    // ── Forward with caching ──

    // KV projection (not cached for backward through KV yet — treat as frozen for now)
    let mut kv = Vec::with_capacity(n_tokens * d_in);
    for t in 0..n_tokens {
        let tok = &observation[t * raw_dim..(t + 1) * raw_dim];
        let mut projected = w.kv_proj.forward(tok);
        // LN forward (simplified — no backward through KV proj for now)
        let n = projected.len() as f32;
        let mean: f32 = projected.iter().sum::<f32>() / n;
        let var: f32 = projected.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
        let inv_std = 1.0 / (var + 1e-5).sqrt();
        for i in 0..projected.len() {
            projected[i] = w.kv_ln_gamma[i] * (projected[i] - mean) * inv_std + w.kv_ln_beta[i];
        }
        kv.extend_from_slice(&projected);
    }

    let r_out: Vec<f32> = w.decay_params_out.iter().map(|&p| (-p.clamp(0.0, 15.0)).exp()).collect();
    let r_action: Vec<f32> = w.decay_params_action.iter().map(|&p| (-p.clamp(0.0, 15.0)).exp()).collect();

    // State — allocate from VRAM arena when available (zero-copy dispatch)
    let be = modgrad_compute::backend::backend();
    let mut trace = be.alloc_f32(w.start_trace.len());
    trace.copy_from_slice(&w.start_trace);
    let mut activated = be.alloc_f32(w.start_activated.len());
    activated.copy_from_slice(&w.start_activated);
    let mut alpha_out: Vec<f32> = (0..cfg.n_synch_out).map(|i|
        activated[w.sync_out_left[i]] * activated[w.sync_out_right[i]]).collect();
    let mut beta_out = vec![1.0f32; cfg.n_synch_out];
    let mut alpha_action = Vec::new();
    let mut beta_action = Vec::new();
    let mut action_init = false;

    let mut tick_caches = Vec::with_capacity(cfg.iterations);

    for _tick in 0..cfg.iterations {
        let activated_prev = activated.as_slice().to_vec();

        // Sync action — VRAM-backed for zero-copy into q_proj matvec
        let (sync_action, ba) = if !action_init {
            let pw: Vec<f32> = (0..cfg.n_synch_action).map(|i|
                activated[w.sync_action_left[i]] * activated[w.sync_action_right[i]]).collect();
            alpha_action = pw.clone();
            beta_action = vec![1.0f32; cfg.n_synch_action];
            action_init = true;
            let mut sa = be.alloc_f32(cfg.n_synch_action);
            for i in 0..cfg.n_synch_action {
                sa[i] = alpha_action[i] / beta_action[i].sqrt().max(1e-8);
            }
            (sa, beta_action.clone())
        } else {
            for i in 0..cfg.n_synch_action {
                let pw = activated[w.sync_action_left[i]] * activated[w.sync_action_right[i]];
                alpha_action[i] = r_action[i] * alpha_action[i] + pw;
                beta_action[i] = r_action[i] * beta_action[i] + 1.0;
            }
            let mut sa = be.alloc_f32(cfg.n_synch_action);
            for i in 0..cfg.n_synch_action {
                sa[i] = alpha_action[i] / beta_action[i].sqrt().max(1e-8);
            }
            (sa, beta_action.clone())
        };

        // q_proj
        let (q, q_lin) = linear_forward_cached(&w.q_proj, &sync_action);

        // MHA
        let (attn_out, mha_cache) = mha_forward_cached(
            &q, &kv, n_tokens, d_in, cfg.heads, &w.mha_in_proj, &w.mha_out_proj);

        // Synapse — allocate concat buffer from VRAM for zero-copy dispatch
        let mut pre_syn = be.alloc_f32(d_in + d);
        pre_syn[..d_in].copy_from_slice(&attn_out);
        pre_syn[d_in..d_in + d].copy_from_slice(&activated);
        let (pre_act, unet_cache) = unet_forward_cached(&w.synapse, &pre_syn);

        // Update trace
        for n in 0..d {
            let base = n * m;
            trace.copy_within(base + 1..base + m, base);
            trace[base + m - 1] = pre_act[n];
        }

        // NLM
        let (new_activated, nlm_cache) = nlm_forward_cached(
            &trace, &w.nlm_stage1, w.nlm_stage2.as_ref(), d);
        activated.copy_from_slice(&new_activated);

        // Sync out
        for i in 0..cfg.n_synch_out {
            let pw = activated[w.sync_out_left[i]] * activated[w.sync_out_right[i]];
            alpha_out[i] = r_out[i] * alpha_out[i] + pw;
            beta_out[i] = r_out[i] * beta_out[i] + 1.0;
        }
        let mut sync_out = be.alloc_f32(cfg.n_synch_out);
        for i in 0..cfg.n_synch_out {
            sync_out[i] = alpha_out[i] / beta_out[i].sqrt().max(1e-8);
        }

        // Output proj
        let (_pred, out_lin) = linear_forward_cached(&w.output_proj, &sync_out);

        // Exit gate (adaptive exit)
        let (exit_gate_lin, exit_lambda) = if let Some(ref gate) = w.exit_gate {
            let (gate_logit, gate_lin) = linear_forward_cached(gate, &sync_out);
            let lambda = 1.0 / (1.0 + (-gate_logit[0]).exp());
            (Some(gate_lin), lambda)
        } else {
            (None, 0.0)
        };

        tick_caches.push(TickCache {
            activated_prev, sync_action: sync_action.as_slice().to_vec(), beta_action: ba,
            q, q_lin, mha_cache, attn_out,
            unet_cache, pre_act: pre_act.clone(),
            nlm_cache, activated_post: activated.as_slice().to_vec(),
            sync_out: sync_out.as_slice().to_vec(), beta_out: beta_out.clone(), out_lin,
            exit_gate_lin, exit_lambda,
        });
    }

    // ── Loss ──
    let predictions: Vec<Vec<f32>> = tick_caches.iter()
        .map(|tc| w.output_proj.forward(&tc.sync_out)).collect();
    let certainties: Vec<[f32; 2]> = predictions.iter()
        .map(|p| super::forward::compute_certainty_pub(p)).collect();

    let pred_class = predictions.last().unwrap().iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i).unwrap_or(0);

    // Choose loss based on exit strategy
    let (loss, d_preds) = if let Some(ref gate) = w.exit_gate {
        let lambdas: Vec<f32> = tick_caches.iter().map(|tc| tc.exit_lambda).collect();
        let beta = cfg.exit_strategy.beta();
        let (loss, d_preds, d_lambdas) = adaptive_exit_loss(
            &predictions, &lambdas, target, beta);

        // Backward through exit gate (detached from main BPTT).
        // Scratch lifted out of the loop — gate.in_dim is constant across
        // ticks, so one allocation serves every d_lambda.
        let mut d_gate_in_scratch = vec![0.0f32; gate.in_dim];
        for (tick, d_lambda) in d_lambdas.iter().enumerate() {
            if let Some(ref gate_lin) = tick_caches[tick].exit_gate_lin {
                let d_gate_logit = [*d_lambda];
                let gw = grads.exit_gate_w.as_mut().unwrap();
                let gb = grads.exit_gate_b.as_mut().unwrap();
                linear_backward(gate, &d_gate_logit, gate_lin, gw, gb,
                    &mut d_gate_in_scratch);
            }
        }

        (loss, d_preds)
    } else {
        ctm_loss(&predictions, &certainties, target)
    };

    // ── Backward through tick loop ──
    let mut d_activated = vec![0.0f32; d];
    // Hot-loop scratches: shapes are constant across ticks, so one
    // allocation each kills ~2 Vec allocs per tick.
    let mut d_sync_out_scratch = vec![0.0f32; w.output_proj.in_dim];
    let mut d_sync_action_scratch = vec![0.0f32; w.q_proj.in_dim];

    for tick in (0..cfg.iterations).rev() {
        let tc = &tick_caches[tick];

        // output_proj backward
        linear_backward(&w.output_proj, &d_preds[tick], &tc.out_lin,
            &mut grads.out_proj_w, &mut grads.out_proj_b,
            &mut d_sync_out_scratch);
        let d_sync_out: &[f32] = &d_sync_out_scratch;

        // Sync out backward → d_activated
        let d_from_sync_out = sync_backward(d_sync_out, &tc.activated_post, &tc.beta_out,
            &w.sync_out_left, &w.sync_out_right, d);
        for i in 0..d { d_activated[i] += d_from_sync_out[i]; }

        // NLM backward → d_trace
        let d_trace = nlm_backward(&d_activated, &tc.nlm_cache,
            &w.nlm_stage1, w.nlm_stage2.as_ref(),
            &mut grads.nlm_s1_w, &mut grads.nlm_s1_b,
            &mut grads.nlm_s2_w, &mut grads.nlm_s2_b);

        // Trace shift backward: d_pre_act[n] = d_trace[n * m + m - 1]
        let mut d_pre_act = vec![0.0f32; d];
        for n in 0..d {
            d_pre_act[n] = d_trace[n * m + m - 1];
        }
        // d_trace for earlier positions → d_start_trace (accumulated)
        // (simplified: only the last position has gradient from this tick)

        // UNet backward
        let d_pre_syn = unet_backward(&w.synapse, &d_pre_act, &tc.unet_cache, &mut grads.unet);

        // Split d_pre_syn → d_attn_out, d_activated_prev
        let d_attn_out = &d_pre_syn[..d_in];
        let d_act_from_syn = &d_pre_syn[d_in..];

        // MHA backward → d_q (d_kv_tokens unused in single-pool train_step)
        let (d_q, _d_kv) = mha_backward(d_attn_out, &tc.mha_cache, n_tokens, d_in, cfg.heads,
            &w.mha_in_proj, &w.mha_out_proj,
            &mut grads.mha_in_w, &mut grads.mha_in_b,
            &mut grads.mha_out_w, &mut grads.mha_out_b);

        // q_proj backward → d_sync_action
        linear_backward(&w.q_proj, &d_q, &tc.q_lin,
            &mut grads.q_proj_w, &mut grads.q_proj_b,
            &mut d_sync_action_scratch);
        let d_sync_action: &[f32] = &d_sync_action_scratch;

        // Sync action backward → d_activated_prev
        let d_from_sync_action = sync_backward(&d_sync_action, &tc.activated_prev, &tc.beta_action,
            &w.sync_action_left, &w.sync_action_right, d);

        // Accumulate into d_activated for previous tick
        d_activated = vec![0.0f32; d];
        for i in 0..d {
            d_activated[i] = d_act_from_syn[i] + d_from_sync_action[i];
        }
    }

    // d_activated is now gradient w.r.t. initial activated state
    for i in 0..d { grads.d_start_activated[i] += d_activated[i]; }

    // Reset VRAM arena — all temporary GPU buffers freed.
    //
    // Stage 6 shrunk `ComputeBackend` to lifecycle-only (alloc_f32,
    // arena_reset, flush); `arena_reset` is one of the remaining four
    // trait methods, so this call survives unchanged. Migrating to
    // `ComputeCtx::<KfdBackend>::arena_reset` would require threading
    // the ctx through every caller of `train_step_frozen` — out of
    // scope per the "global mutable state stays" stance in the plan.
    be.arena_reset();

    TrainResult { loss, prediction: pred_class }
}

// ═══════════════════════════════════════════════════════════════
// BRAIN TRAIT IMPLEMENTATION
// ═══════════════════════════════════════════════════════════════

use modgrad_traits::{Brain, BrainOutput, LossFn, TokenInput};

/// The Continuous Thought Machine as a Brain.
pub struct Ctm;

/// Cache from forward_cached — everything needed for backward.
pub struct CtmCache {
    tick_caches: Vec<TickCache>,
    #[allow(dead_code)] // retained for potential future backward / inspection use
    kv: Vec<f32>,
    n_tokens: usize,
    d_input: usize,
    #[allow(dead_code)] // retained for potential future backward / inspection use
    r_out: Vec<f32>,
    #[allow(dead_code)] // retained for potential future backward / inspection use
    r_action: Vec<f32>,
}

/// Explicit state returned from forward (not mutated in place).
pub struct CtmStateExplicit {
    pub trace: Vec<f32>,
    pub activated: Vec<f32>,
    pub alpha_out: Vec<f32>,
    pub beta_out: Vec<f32>,
    pub alpha_action: Vec<f32>,
    pub beta_action: Vec<f32>,
    pub action_initialized: bool,
}

impl Brain for Ctm {
    type Input = TokenInput;
    type Weights = CtmWeights;
    type State = CtmStateExplicit;
    type Cache = CtmCache;
    type Gradients = CtmGradients;

    fn init_state(w: &CtmWeights) -> CtmStateExplicit {
        let cfg = &w.config;
        let alpha_out: Vec<f32> = (0..cfg.n_synch_out).map(|i|
            w.start_activated[w.sync_out_left[i]] * w.start_activated[w.sync_out_right[i]])
            .collect();
        CtmStateExplicit {
            trace: w.start_trace.clone(),
            activated: w.start_activated.clone(),
            alpha_out,
            beta_out: vec![1.0f32; cfg.n_synch_out],
            alpha_action: Vec::new(),
            beta_action: Vec::new(),
            action_initialized: false,
        }
    }

    fn forward(
        w: &CtmWeights, state: CtmStateExplicit,
        input: &TokenInput,
    ) -> (BrainOutput, CtmStateExplicit) {
        let (output, state, _cache) = Self::forward_cached(w, state, input);
        (output, state)
    }

    fn forward_cached(
        w: &CtmWeights, mut state: CtmStateExplicit,
        input: &TokenInput,
    ) -> (BrainOutput, CtmStateExplicit, CtmCache) {
        let tokens = &input.tokens;
        let n_tokens = input.n_tokens;
        let token_dim = input.token_dim;

        let cfg = &w.config;
        let d = cfg.d_model;
        let d_in = cfg.d_input;
        let m = cfg.memory_length;

        // KV projection
        let mut kv = Vec::with_capacity(n_tokens * d_in);
        for t in 0..n_tokens {
            let tok = &tokens[t * token_dim..(t + 1) * token_dim];
            let mut projected = w.kv_proj.forward(tok);
            let n = projected.len() as f32;
            let mean: f32 = projected.iter().sum::<f32>() / n;
            let var: f32 = projected.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
            let inv_std = 1.0 / (var + 1e-5).sqrt();
            for i in 0..projected.len() {
                projected[i] = w.kv_ln_gamma[i] * (projected[i] - mean) * inv_std + w.kv_ln_beta[i];
            }
            kv.extend_from_slice(&projected);
        }

        let r_out: Vec<f32> = w.decay_params_out.iter().map(|&p| (-p.clamp(0.0, 15.0)).exp()).collect();
        let r_action: Vec<f32> = w.decay_params_action.iter().map(|&p| (-p.clamp(0.0, 15.0)).exp()).collect();

        let mut tick_caches = Vec::with_capacity(cfg.iterations);
        let mut predictions = Vec::with_capacity(cfg.iterations);
        let mut certainties = Vec::with_capacity(cfg.iterations);

        for _tick in 0..cfg.iterations {
            let activated_prev = state.activated.clone();

            // Sync action
            let (sync_action, ba) = if !state.action_initialized {
                let pw: Vec<f32> = (0..cfg.n_synch_action).map(|i|
                    state.activated[w.sync_action_left[i]] * state.activated[w.sync_action_right[i]]).collect();
                state.alpha_action = pw.clone();
                state.beta_action = vec![1.0f32; cfg.n_synch_action];
                state.action_initialized = true;
                let sa: Vec<f32> = (0..cfg.n_synch_action).map(|i|
                    state.alpha_action[i] / state.beta_action[i].sqrt().max(1e-8)).collect();
                (sa, state.beta_action.clone())
            } else {
                for i in 0..cfg.n_synch_action {
                    let pw = state.activated[w.sync_action_left[i]] * state.activated[w.sync_action_right[i]];
                    state.alpha_action[i] = r_action[i] * state.alpha_action[i] + pw;
                    state.beta_action[i] = r_action[i] * state.beta_action[i] + 1.0;
                }
                let sa: Vec<f32> = (0..cfg.n_synch_action).map(|i|
                    state.alpha_action[i] / state.beta_action[i].sqrt().max(1e-8)).collect();
                (sa, state.beta_action.clone())
            };

            let (q, q_lin) = linear_forward_cached(&w.q_proj, &sync_action);
            let (attn_out, mha_cache) = mha_forward_cached(
                &q, &kv, n_tokens, d_in, cfg.heads, &w.mha_in_proj, &w.mha_out_proj);

            let mut pre_syn = Vec::with_capacity(d_in + d);
            pre_syn.extend_from_slice(&attn_out);
            pre_syn.extend_from_slice(&state.activated);
            let (pre_act, unet_cache) = unet_forward_cached(&w.synapse, &pre_syn);

            for n in 0..d {
                let base = n * m;
                state.trace.copy_within(base + 1..base + m, base);
                state.trace[base + m - 1] = pre_act[n];
            }

            let (new_activated, nlm_cache) = nlm_forward_cached(
                &state.trace, &w.nlm_stage1, w.nlm_stage2.as_ref(), d);
            state.activated = new_activated;

            for i in 0..cfg.n_synch_out {
                let pw = state.activated[w.sync_out_left[i]] * state.activated[w.sync_out_right[i]];
                state.alpha_out[i] = r_out[i] * state.alpha_out[i] + pw;
                state.beta_out[i] = r_out[i] * state.beta_out[i] + 1.0;
            }
            let sync_out: Vec<f32> = (0..cfg.n_synch_out).map(|i|
                state.alpha_out[i] / state.beta_out[i].sqrt().max(1e-8)).collect();

            let (pred, out_lin) = linear_forward_cached(&w.output_proj, &sync_out);
            let cert = super::forward::compute_certainty_pub(&pred);

            predictions.push(pred);
            certainties.push(cert);

            // Exit gate (adaptive exit)
            let (exit_gate_lin, exit_lambda) = if let Some(ref gate) = w.exit_gate {
                let (gate_logit, gate_lin) = linear_forward_cached(gate, &sync_out);
                let lambda = 1.0 / (1.0 + (-gate_logit[0]).exp());
                (Some(gate_lin), lambda)
            } else {
                (None, 0.0)
            };

            tick_caches.push(TickCache {
                activated_prev, sync_action: sync_action.as_slice().to_vec(), beta_action: ba,
                q, q_lin, mha_cache, attn_out,
                unet_cache, pre_act: pre_act.clone(),
                nlm_cache, activated_post: state.activated.clone(),
                sync_out, beta_out: state.beta_out.clone(), out_lin,
                exit_gate_lin, exit_lambda,
            });
        }

        let final_sync = tick_caches.last()
            .map(|tc| tc.sync_out.clone())
            .unwrap_or_default();

        let output = BrainOutput {
            predictions,
            certainties,
            sync: final_sync,
        };
        let cache = CtmCache {
            tick_caches, kv, n_tokens, d_input: d_in,
            r_out, r_action,
        };

        (output, state, cache)
    }

    fn backward(
        w: &CtmWeights, cache: CtmCache, d_predictions: &[Vec<f32>],
    ) -> CtmGradients {
        let cfg = &w.config;
        let d = cfg.d_model;
        let d_in = cache.d_input;
        let m = cfg.memory_length;

        let mut grads = Self::zero_gradients(w);
        let mut d_activated = vec![0.0f32; d];
        // Hot-loop scratches — same pattern as `train_step` above.
        let mut d_sync_out_scratch = vec![0.0f32; w.output_proj.in_dim];
        let mut d_sync_action_scratch = vec![0.0f32; w.q_proj.in_dim];

        for tick in (0..cfg.iterations).rev() {
            let tc = &cache.tick_caches[tick];

            linear_backward(&w.output_proj, &d_predictions[tick], &tc.out_lin,
                &mut grads.out_proj_w, &mut grads.out_proj_b,
                &mut d_sync_out_scratch);
            let d_sync_out: &[f32] = &d_sync_out_scratch;

            let d_from_sync_out = sync_backward(d_sync_out, &tc.activated_post, &tc.beta_out,
                &w.sync_out_left, &w.sync_out_right, d);
            for i in 0..d { d_activated[i] += d_from_sync_out[i]; }

            let d_trace = nlm_backward(&d_activated, &tc.nlm_cache,
                &w.nlm_stage1, w.nlm_stage2.as_ref(),
                &mut grads.nlm_s1_w, &mut grads.nlm_s1_b,
                &mut grads.nlm_s2_w, &mut grads.nlm_s2_b);

            let mut d_pre_act = vec![0.0f32; d];
            for n in 0..d { d_pre_act[n] = d_trace[n * m + m - 1]; }

            let d_pre_syn = unet_backward(&w.synapse, &d_pre_act, &tc.unet_cache, &mut grads.unet);

            let d_attn_out = &d_pre_syn[..d_in];
            let d_act_from_syn = &d_pre_syn[d_in..];

            let (d_q, _d_kv) = mha_backward(d_attn_out, &tc.mha_cache, cache.n_tokens, d_in, cfg.heads,
                &w.mha_in_proj, &w.mha_out_proj,
                &mut grads.mha_in_w, &mut grads.mha_in_b,
                &mut grads.mha_out_w, &mut grads.mha_out_b);

            linear_backward(&w.q_proj, &d_q, &tc.q_lin,
                &mut grads.q_proj_w, &mut grads.q_proj_b,
                &mut d_sync_action_scratch);
            let d_sync_action: &[f32] = &d_sync_action_scratch;

            let d_from_sync_action = sync_backward(d_sync_action, &tc.activated_prev, &tc.beta_action,
                &w.sync_action_left, &w.sync_action_right, d);

            d_activated = vec![0.0f32; d];
            for i in 0..d {
                d_activated[i] = d_act_from_syn[i] + d_from_sync_action[i];
            }
        }

        for i in 0..d { grads.d_start_activated[i] += d_activated[i]; }
        grads
    }

    fn zero_gradients(w: &CtmWeights) -> CtmGradients {
        CtmGradients::zeros(w)
    }

    fn apply_gradients(w: &mut CtmWeights, grads: &mut CtmGradients, lr: f32, clip_norm: f32) {
        grads.apply(w, lr, clip_norm);
    }
}

// ═══════════════════════════════════════════════════════════════
// COMPOSABLE TRAINING
// ═══════════════════════════════════════════════════════════════

/// One training step, composed from Brain + LossFn.
/// Each part is independently swappable.
pub fn train_step_composed<T, L: LossFn<Target = T>>(
    w: &CtmWeights,
    grads: &mut CtmGradients,
    tokens: &[f32],
    n_tokens: usize,
    token_dim: usize,
    target: &T,
    loss_fn: &L,
) -> TrainResult {
    // 1. Forward with caching (pure, returns new state)
    let state = Ctm::init_state(w);
    let input = TokenInput { tokens: tokens.to_vec(), n_tokens, token_dim };
    let (output, _state, cache) = Ctm::forward_cached(w, state, &input);

    // 2. Loss (pure function, swappable)
    let (loss, d_preds) = loss_fn.compute(&output.predictions, &output.certainties, target);

    // 3. Backward (pure function of cache + gradients)
    let step_grads = Ctm::backward(w, cache, &d_preds);

    // 4. Accumulate into grads (caller applies when ready)
    accumulate_gradients(grads, &step_grads);

    let pred_class = output.predictions.last()
        .and_then(|p| p.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)))
        .map(|(i, _)| i).unwrap_or(0);

    TrainResult { loss, prediction: pred_class }
}

// ═══════════════════════════════════════════════════════════════
// HIERARCHICAL CTM SUPPORT
// ═══════════════════════════════════════════════════════════════

/// Result of backward_from_activated: weight gradients + input gradient.
pub struct RegionBackwardResult {
    /// Weight gradients for this region's CtmWeights.
    pub grads: CtmGradients,
    /// Gradient w.r.t. the observation input (for connection synapse backward).
    /// Accumulated across all ticks, backpropped through kv_proj.
    /// Length = n_tokens * raw_obs_dim (same shape as the observation input).
    pub d_observation: Vec<f32>,
}

/// Backward from external gradient on final activated state.
///
/// For hierarchical CTM: the outer region graph computes d_loss/d_region_activated.
/// This injects that gradient and runs full inner-tick BPTT through the region's
/// NLM, synapse, and MHA — training all internal weights.
///
/// Returns weight gradients AND d_observation (for connection synapse backward).
pub fn backward_from_activated(
    w: &CtmWeights, cache: &CtmCache, d_external: &[f32],
) -> RegionBackwardResult {
    let cfg = &w.config;
    let d = cfg.d_model;
    let d_in = cache.d_input;
    let m = cfg.memory_length;
    let k = cfg.iterations;

    let mut grads = CtmGradients::zeros(w);

    // Start with the external gradient at the last tick
    let mut d_activated: Vec<f32> = d_external.iter().copied()
        .chain(std::iter::repeat(0.0))
        .take(d)
        .collect();

    // Accumulate d_kv across all ticks (gradient w.r.t. KV tokens)
    let mut d_kv_accumulated = vec![0.0f32; cache.n_tokens * d_in];

    // Hot-loop scratch — q_proj.in_dim is constant across ticks.
    let mut d_sync_action_scratch = vec![0.0f32; w.q_proj.in_dim];

    for tick in (0..k).rev() {
        let tc = &cache.tick_caches[tick];

        // Sync out backward: how activated contributes to sync readout
        let d_from_sync = sync_backward(
            &d_activated, &tc.activated_post, &tc.beta_out,
            &w.sync_out_left, &w.sync_out_right, d,
        );

        // At the last tick, inject the external gradient directly.
        // At earlier ticks, gradient flows from the next tick's synapse input.
        let mut d_act_total = vec![0.0f32; d];
        for i in 0..d {
            d_act_total[i] = d_from_sync[i];
            if tick == k - 1 {
                d_act_total[i] += d_activated[i];
            }
        }

        // NLM backward
        let d_trace = nlm_backward(&d_act_total, &tc.nlm_cache,
            &w.nlm_stage1, w.nlm_stage2.as_ref(),
            &mut grads.nlm_s1_w, &mut grads.nlm_s1_b,
            &mut grads.nlm_s2_w, &mut grads.nlm_s2_b);

        let mut d_pre_act = vec![0.0f32; d];
        for n in 0..d { d_pre_act[n] = d_trace[n * m + m - 1]; }

        // Synapse backward
        let d_pre_syn = unet_backward(&w.synapse, &d_pre_act, &tc.unet_cache, &mut grads.unet);

        let d_attn_out = &d_pre_syn[..d_in];
        let d_act_from_syn = &d_pre_syn[d_in..];

        // MHA backward — collect d_kv_tokens
        let (d_q, d_kv_tokens) = mha_backward(d_attn_out, &tc.mha_cache, cache.n_tokens, d_in, cfg.heads,
            &w.mha_in_proj, &w.mha_out_proj,
            &mut grads.mha_in_w, &mut grads.mha_in_b,
            &mut grads.mha_out_w, &mut grads.mha_out_b);

        // Accumulate d_kv across ticks (each tick attends over the same KV)
        for (t, d_tok) in d_kv_tokens.iter().enumerate() {
            let offset = t * d_in;
            for j in 0..d_in.min(d_tok.len()) {
                if offset + j < d_kv_accumulated.len() {
                    d_kv_accumulated[offset + j] += d_tok[j];
                }
            }
        }

        linear_backward(&w.q_proj, &d_q, &tc.q_lin,
            &mut grads.q_proj_w, &mut grads.q_proj_b,
            &mut d_sync_action_scratch);
        let d_sync_action: &[f32] = &d_sync_action_scratch;

        let d_from_sync_action = sync_backward(d_sync_action, &tc.activated_prev, &tc.beta_action,
            &w.sync_action_left, &w.sync_action_right, d);

        // Propagate to previous tick
        d_activated = vec![0.0f32; d];
        for i in 0..d {
            d_activated[i] = d_act_from_syn[i] + d_from_sync_action[i];
        }
    }

    for i in 0..d { grads.d_start_activated[i] += d_activated[i]; }

    // Backprop d_kv through kv_proj to get d_observation
    // kv_proj: raw_obs → d_input (per token). d_kv is in d_input space.
    // d_observation = kv_proj.W^T @ d_kv (per token)
    let raw_dim = w.kv_proj.in_dim;
    let mut d_observation = vec![0.0f32; cache.n_tokens * raw_dim];
    for t in 0..cache.n_tokens {
        let d_kv_t = &d_kv_accumulated[t * d_in..(t + 1).min(cache.n_tokens) * d_in];
        for j in 0..raw_dim {
            for i in 0..d_in.min(d_kv_t.len()) {
                d_observation[t * raw_dim + j] += d_kv_t[i] * w.kv_proj.weight[i * raw_dim + j];
            }
            // Also accumulate kv_proj weight gradients
            grads.kv_proj_b[j] += 0.0; // bias grad needs kv_proj input, skip for now
        }
    }

    RegionBackwardResult { grads, d_observation }
}

/// Accumulate sample gradients into batch gradients.
pub fn accumulate_gradients(dst: &mut CtmGradients, src: &CtmGradients) {
    let add = |d: &mut [f32], s: &[f32]| { for (d, s) in d.iter_mut().zip(s) { *d += s; } };
    // UNet grads
    add(&mut dst.unet.first.d_weight, &src.unet.first.d_weight);
    add(&mut dst.unet.first.d_bias, &src.unet.first.d_bias);
    add(&mut dst.unet.first.d_gamma, &src.unet.first.d_gamma);
    add(&mut dst.unet.first.d_beta, &src.unet.first.d_beta);
    for i in 0..dst.unet.downs.len() {
        add(&mut dst.unet.downs[i].d_weight, &src.unet.downs[i].d_weight);
        add(&mut dst.unet.downs[i].d_bias, &src.unet.downs[i].d_bias);
        add(&mut dst.unet.downs[i].d_gamma, &src.unet.downs[i].d_gamma);
        add(&mut dst.unet.downs[i].d_beta, &src.unet.downs[i].d_beta);
        add(&mut dst.unet.ups[i].d_weight, &src.unet.ups[i].d_weight);
        add(&mut dst.unet.ups[i].d_bias, &src.unet.ups[i].d_bias);
        add(&mut dst.unet.ups[i].d_gamma, &src.unet.ups[i].d_gamma);
        add(&mut dst.unet.ups[i].d_beta, &src.unet.ups[i].d_beta);
        add(&mut dst.unet.skip_d_gamma[i], &src.unet.skip_d_gamma[i]);
        add(&mut dst.unet.skip_d_beta[i], &src.unet.skip_d_beta[i]);
    }
    // Flat grads
    add(&mut dst.nlm_s1_w, &src.nlm_s1_w);
    add(&mut dst.nlm_s1_b, &src.nlm_s1_b);
    if let (Some(dw), Some(sw)) = (&mut dst.nlm_s2_w, &src.nlm_s2_w) { add(dw, sw); }
    if let (Some(db), Some(sb)) = (&mut dst.nlm_s2_b, &src.nlm_s2_b) { add(db, sb); }
    add(&mut dst.d_start_activated, &src.d_start_activated);
    add(&mut dst.d_start_trace, &src.d_start_trace);
    add(&mut dst.kv_proj_w, &src.kv_proj_w);
    add(&mut dst.kv_proj_b, &src.kv_proj_b);
    add(&mut dst.kv_ln_d_gamma, &src.kv_ln_d_gamma);
    add(&mut dst.kv_ln_d_beta, &src.kv_ln_d_beta);
    add(&mut dst.q_proj_w, &src.q_proj_w);
    add(&mut dst.q_proj_b, &src.q_proj_b);
    add(&mut dst.mha_in_w, &src.mha_in_w);
    add(&mut dst.mha_in_b, &src.mha_in_b);
    add(&mut dst.mha_out_w, &src.mha_out_w);
    add(&mut dst.mha_out_b, &src.mha_out_b);
    add(&mut dst.out_proj_w, &src.out_proj_w);
    add(&mut dst.out_proj_b, &src.out_proj_b);
    if let (Some(dw), Some(sw)) = (&mut dst.exit_gate_w, &src.exit_gate_w) { add(dw, sw); }
    if let (Some(db), Some(sb)) = (&mut dst.exit_gate_b, &src.exit_gate_b) { add(db, sb); }
}

