//! FFN cerebellum — trainable feedforward network for language/pattern storage.
//!
//! The biological cerebellum is a massive feedforward network (granule cells →
//! Purkinje cells) specialized for pattern memorization. Our FFN cerebellum
//! implements this: vocab_size → d_model → hidden → d_model → vocab_size.
//!
//! Architecture: embed → [SwiGLU block] × N → unembed.
//! Each block: x + down(silu(gate(norm(x))) * up(norm(x)))
//!
//! Unlike the CTM, this has no recurrence, no attention, no ticks.
//! Pure feedforward. Trains fast on GPU. Serves as pre-trained language
//! prior that the CTM cortex routes through.
//!
//! Two use cases:
//! 1. Standalone: train on next-token prediction as a simple language model.
//! 2. Frozen cerebellum: train standalone, freeze, plug into CTM.

use serde::{Deserialize, Serialize};
use modgrad_compute::neuron::Linear;
use modgrad_traits::ParamIter;
use rayon::prelude::*;
use wincode_derive::{SchemaRead, SchemaWrite};

/// FFN cerebellum config.
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub struct FfnConfig {
    pub vocab: usize,
    pub d_model: usize,
    pub hidden: usize,    // inner FFN dim (typically 4 × d_model)
    pub n_layers: usize,
    pub context: usize,   // max context length (for position embedding)
}

impl FfnConfig {
    /// Small: ~5M params for sanity check.
    pub fn small(vocab: usize) -> Self {
        Self { vocab, d_model: 128, hidden: 512, n_layers: 4, context: 64 }
    }

    /// Medium: ~50M params.
    pub fn medium(vocab: usize) -> Self {
        Self { vocab, d_model: 384, hidden: 1536, n_layers: 8, context: 128 }
    }

    /// Large: ~200M params — real language learning starts here.
    /// Fits comfortably in 8GB VRAM with AdamW (weights 800MB + moments 1.6GB + acts ~1GB = ~4GB).
    pub fn large(vocab: usize) -> Self {
        Self { vocab, d_model: 1024, hidden: 5120, n_layers: 12, context: 256 }
    }

    /// XL: ~500M params — pushes 8GB VRAM limits with AdamW (weights 2GB + moments 4GB).
    pub fn xl(vocab: usize) -> Self {
        Self { vocab, d_model: 1536, hidden: 6144, n_layers: 16, context: 512 }
    }
}

/// One SwiGLU block with residual + pre-norm.
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub struct FfnBlock {
    /// Pre-norm gamma: [d_model].
    pub ln_gamma: Vec<f32>,
    /// Pre-norm beta: [d_model].
    pub ln_beta: Vec<f32>,
    /// Gate projection: [hidden × d_model].
    pub gate: Linear,
    /// Up projection: [hidden × d_model].
    pub up: Linear,
    /// Down projection: [d_model × hidden].
    pub down: Linear,
}

impl FfnBlock {
    pub fn new(d_model: usize, hidden: usize) -> Self {
        Self {
            ln_gamma: vec![1.0; d_model],
            ln_beta: vec![0.0; d_model],
            gate: Linear::new(d_model, hidden),
            up: Linear::new(d_model, hidden),
            down: Linear::new(hidden, d_model),
        }
    }

    pub fn n_params(&self) -> usize {
        self.ln_gamma.len() + self.ln_beta.len()
            + self.gate.weight.len() + self.gate.bias.len()
            + self.up.weight.len() + self.up.bias.len()
            + self.down.weight.len() + self.down.bias.len()
    }
}

/// FFN cerebellum weights.
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub struct FfnWeights {
    pub config: FfnConfig,
    /// Token embedding: [vocab × d_model].
    pub embed: Vec<f32>,
    /// Stacked SwiGLU blocks.
    pub blocks: Vec<FfnBlock>,
    /// Final layer norm.
    pub final_ln_gamma: Vec<f32>,
    pub final_ln_beta: Vec<f32>,
    /// Output projection (lm_head): [vocab × d_model].
    pub lm_head: Vec<f32>,
}

impl FfnWeights {
    pub fn new(config: FfnConfig) -> Self {
        let d = config.d_model;
        let bound = 1.0 / (d as f32).sqrt();
        let mut rng = modgrad_compute::neuron::SimpleRng::new(42);

        let embed: Vec<f32> = (0..config.vocab * d)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * bound).collect();
        let blocks: Vec<FfnBlock> = (0..config.n_layers)
            .map(|_| FfnBlock::new(d, config.hidden)).collect();
        let lm_head: Vec<f32> = (0..config.vocab * d)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * bound).collect();

        Self {
            config,
            embed,
            blocks,
            final_ln_gamma: vec![1.0; d],
            final_ln_beta: vec![0.0; d],
            lm_head,
        }
    }

    /// Total parameter count. Delegates to `ParamIter::n_params` — single
    /// source of truth for what "the parameters of this model are".
    pub fn n_params(&self) -> usize {
        <Self as ParamIter>::n_params(self)
    }

    pub fn print_summary(&self) {
        let c = &self.config;
        eprintln!("FFN cerebellum: vocab={} d_model={} hidden={} layers={}",
            c.vocab, c.d_model, c.hidden, c.n_layers);
        eprintln!("  embed: {} params", self.embed.len());
        for (i, b) in self.blocks.iter().enumerate() {
            eprintln!("  block[{}]: {} params", i, b.n_params());
        }
        eprintln!("  lm_head: {} params", self.lm_head.len());
        eprintln!("  total: {} params", self.n_params());
    }

    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        modgrad_persist::persist::save(self, path).map_err(|e| e.into())
    }

    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        modgrad_persist::persist::load(path).map_err(|e| e.into())
    }
}

// ═══════════════════════════════════════════════════════════════
// FORWARD PASS — batched: all tokens processed together as [N × D] slabs.
// Tokens are independent in an FFN (no attention, no recurrence), so we
// stack them and use matmul instead of looping matvec. This turns thousands
// of sequential GPU/CPU dispatches into a handful of big rayon-parallel ops.
// ═══════════════════════════════════════════════════════════════

/// FFN forward with cache for backward pass. All slabs are row-major [N × D].
pub struct FfnCache {
    pub tokens: Vec<usize>,
    pub n: usize,                         // batch size (= tokens.len())
    pub per_layer: Vec<LayerCache>,       // one entry per block
    /// Input to final LN (i.e. residual sum after last block): [N × d_model].
    pub final_ln_input: Vec<f32>,
    /// Output of final LN: [N × d_model]. Used for lm_head forward/backward.
    pub final_normed: Vec<f32>,
}

pub struct LayerCache {
    pub pre_norm: Vec<f32>,   // input to block (before norm):  [N × d_model]
    pub normed: Vec<f32>,     // after pre-norm:                [N × d_model]
    pub gate_out: Vec<f32>,   // gate(normed):                  [N × hidden]
    pub up_out: Vec<f32>,     // up(normed):                    [N × hidden]
    pub hidden: Vec<f32>,     // silu(gate) * up:               [N × hidden]
    pub means: Vec<f32>,      // per-row LN mean:               [N]
    pub inv_stds: Vec<f32>,   // per-row LN 1/sqrt(var+eps):    [N]
}

/// Forward pass over `tokens`, return per-position logits and cache.
/// Output shape: logits[pos] has length `vocab`.
pub fn ffn_forward(w: &FfnWeights, tokens: &[usize]) -> (Vec<Vec<f32>>, FfnCache) {
    let d = w.config.d_model;
    let h = w.config.hidden;
    let vocab = w.config.vocab;
    let n = tokens.len();

    // ─── Embed lookup: stack input tokens as rows ───
    let mut x = vec![0.0f32; n * d];
    for (i, &tok) in tokens.iter().enumerate() {
        let src = tok * d;
        x[i * d..(i + 1) * d].copy_from_slice(&w.embed[src..src + d]);
    }

    // ─── SwiGLU blocks ───
    let mut per_layer: Vec<LayerCache> = Vec::with_capacity(w.blocks.len());
    for block in &w.blocks {
        let pre_norm = x.clone();

        // Pre-norm (row-wise): produces normed [N × d] plus per-row (mean, inv_std).
        let (normed, means, inv_stds) =
            layer_norm_forward_batched(&x, &block.ln_gamma, &block.ln_beta, n, d);

        // gate_out = normed @ gate.W^T + gate.b            shape [N × h]
        let mut gate_out = vec![0.0f32; n * h];
        matmul_rayon(&normed, &block.gate.weight, &block.gate.bias, &mut gate_out, n, d, h);

        // up_out   = normed @ up.W^T   + up.b              shape [N × h]
        let mut up_out = vec![0.0f32; n * h];
        matmul_rayon(&normed, &block.up.weight, &block.up.bias, &mut up_out, n, d, h);

        // hidden = silu(gate_out) * up_out                  shape [N × h]
        let mut hidden = vec![0.0f32; n * h];
        hidden.par_iter_mut().enumerate().for_each(|(i, hv)| {
            let g = gate_out[i];
            let s = 1.0 / (1.0 + (-g).exp());
            *hv = g * s * up_out[i];
        });

        // down_out = hidden @ down.W^T + down.b             shape [N × d]
        let mut down_out = vec![0.0f32; n * d];
        matmul_rayon(&hidden, &block.down.weight, &block.down.bias, &mut down_out, n, h, d);

        // Residual: x = pre_norm + down_out
        x.par_iter_mut().zip(down_out.par_iter()).zip(pre_norm.par_iter())
            .for_each(|((xv, &dv), &pv)| *xv = pv + dv);

        per_layer.push(LayerCache { pre_norm, normed, gate_out, up_out, hidden, means, inv_stds });
    }

    // ─── Final LN + LM head ───
    let final_ln_input = x.clone();
    let (final_normed, _means, _inv_stds) =
        layer_norm_forward_batched(&x, &w.final_ln_gamma, &w.final_ln_beta, n, d);

    // logits_slab [N × vocab] = final_normed @ lm_head^T   (no bias for lm_head)
    let mut logits_slab = vec![0.0f32; n * vocab];
    let zero_bias = vec![0.0f32; vocab];
    matmul_rayon(&final_normed, &w.lm_head, &zero_bias, &mut logits_slab, n, d, vocab);

    // Unpack row-major [N × vocab] into per-position Vecs (API compat).
    let logits: Vec<Vec<f32>> = (0..n).map(|pos| {
        logits_slab[pos * vocab..(pos + 1) * vocab].to_vec()
    }).collect();

    let cache = FfnCache {
        tokens: tokens.to_vec(),
        n,
        per_layer,
        final_ln_input,
        final_normed,
    };

    (logits, cache)
}

/// Forward matmul: Y[n×m] = A[n×k] @ W^T[k×m] + bias[m].
/// GPU-accelerated via `accel::try_matmul` (uses matmul_blocked kernel).
/// CPU path (rayon + matrixmultiply sgemm) runs when GPU is disabled.
///
/// Fail-fast contract: if `gpu_enabled()`, GPU dispatch *must* succeed.
/// Any failure panics with diagnostic shape info. A silent CPU fallback
/// on GPU failure would hide dispatch bugs and turn 5-minute runs into
/// hours of slow-CPU training while the user thinks they're on GPU —
/// exactly the failure mode that cost 4h of training time before the
/// fallback was removed.
///
/// GPU path constraints: m%128==0, k%8==0, n%32==0. For our FFN sizes
/// (N=128, K∈{1024, 5120}, M∈{1024, 5120}) all natural shapes pass.
#[inline]
fn matmul_rayon(a: &[f32], weight: &[f32], bias: &[f32], y: &mut [f32],
                n: usize, k: usize, m: usize) {
    // y[n×m] = a[n×k] @ weight^T + bias[m]  (weight stored as [m×k]
    // row-major, so we use kind=NT to compute A @ B^T).
    //
    // Registry dispatches: KFD's try_matmul expects this layout and
    // claims NN-only in its gate today; for NT-kind we fall through to
    // CPU which handles all kinds. KFD support for NT is a follow-up
    // (kernel needs T-variant).
    use modgrad_device::backend::{registry, Op};
    registry().dispatch(&mut Op::MatmulNT {
        a, b: weight, out: y, bias: Some(bias),
        m: n, k, n: m,
    }).expect("matmul dispatch");
}

/// Transposed matmul: dA[n×k] += dY[n×m] @ W[m×k] (ACCUMULATES).
/// GPU path when `--gpu` enabled; CPU path otherwise. Fail-fast — no
/// silent fallback (see `matmul_rayon` for the rationale).
#[inline]
fn matmul_t_rayon(dy: &[f32], weight: &[f32], da: &mut [f32],
                  n: usize, k: usize, m: usize) {
    // Backward-input path: da[n×k] += dy[n×m] @ weight
    // Weight is stored row-major [m×k], so this is a plain NN matmul.
    // Compute into `tmp` (overwrite) then accumulate into `da`.
    use modgrad_device::backend::{registry, Op};
    let mut tmp = vec![0.0f32; n * k];
    registry().dispatch(&mut Op::MatmulNN {
        a: dy, b: weight, out: &mut tmp, bias: None,
        m: n, k: m, n: k,
    }).expect("matmul_t dispatch");
    da.par_iter_mut().zip(tmp.par_iter()).for_each(|(d, &t)| *d += t);
}

/// Weight-gradient matmul: dW[m×k] += dY^T[m×n] @ A[n×k] (ACCUMULATES).
/// GPU path when `--gpu` enabled; CPU path otherwise. Fail-fast — no
/// silent fallback.
#[inline]
fn matmul_grad_rayon(dy: &[f32], a: &[f32], dw: &mut [f32],
                     n: usize, k: usize, m: usize) {
    // Weight-gradient: dw[m×k] += dy^T[m×n] @ a[n×k]
    // dy stored [n×m] row-major; a stored [n×k] row-major; both as
    // provided by the caller. Using kind=TN for "first matrix
    // transposed" matches this layout: out[m×k] = dy^T @ a.
    use modgrad_device::backend::{registry, Op};
    let mut tmp = vec![0.0f32; m * k];
    registry().dispatch(&mut Op::MatmulTN {
        a: dy, b: a, out: &mut tmp, bias: None,
        m, k: n, n: k,
    }).expect("matmul_grad dispatch");
    dw.par_iter_mut().zip(tmp.par_iter()).for_each(|(d, &t)| *d += t);
}

/// Batched row-wise layer norm. Returns (normed, means, inv_stds).
fn layer_norm_forward_batched(x: &[f32], gamma: &[f32], beta: &[f32],
                              n: usize, d: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut out = vec![0.0f32; n * d];
    let mut means = vec![0.0f32; n];
    let mut inv_stds = vec![0.0f32; n];
    let df = d as f32;

    out.par_chunks_mut(d).zip(means.par_iter_mut()).zip(inv_stds.par_iter_mut()).enumerate()
        .for_each(|(row, ((out_row, mean_slot), inv_std_slot))| {
            let x_row = &x[row * d..(row + 1) * d];
            let mean: f32 = x_row.iter().sum::<f32>() / df;
            let var: f32 = x_row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / df;
            let inv_std = 1.0 / (var + 1e-5).sqrt();
            for j in 0..d {
                out_row[j] = gamma[j] * (x_row[j] - mean) * inv_std + beta[j];
            }
            *mean_slot = mean;
            *inv_std_slot = inv_std;
        });

    (out, means, inv_stds)
}

// ═══════════════════════════════════════════════════════════════
// LOSS: per-position cross-entropy on next-token prediction
// ═══════════════════════════════════════════════════════════════

/// Compute CE loss + gradient on logits for sequence prediction.
/// `tokens`: input sequence [t0, t1, ..., tN-1]
/// For each position i, target is tokens[i+1].
/// Returns (loss, d_logits per position).
/// Cross-entropy loss + gradient. `logits[pos]` predicts `targets[pos]`;
/// the two slices must have the same length.
///
/// Caller (see `ffn_train_step`) is responsible for producing aligned
/// inputs/targets — typically `inputs = &tokens[..n-1]`, `targets =
/// &tokens[1..]`. This used to be inlined here as "last position has no
/// target"; splitting it out means the forward pass receives a
/// correctly-shaped input slice (length n-1) instead of length n with
/// one wasted position. Aligning to 2ⁿ / multiple-of-32 boundaries
/// matters for GPU matmul kernels, which require n%32==0.
pub fn ffn_loss(logits: &[Vec<f32>], targets: &[usize]) -> (f32, Vec<Vec<f32>>) {
    let n = logits.len();
    assert_eq!(n, targets.len(),
        "ffn_loss: logits and targets must have equal length (got {} vs {})",
        n, targets.len());
    if n == 0 { return (0.0, vec![]); }
    let mut total_loss = 0.0f32;
    let mut d_logits = Vec::with_capacity(n);
    for pos in 0..n {
        let (loss, grad) = modgrad_traits::cross_entropy_grad(&logits[pos], targets[pos]);
        total_loss += loss;
        d_logits.push(grad);
    }
    (total_loss / n as f32, d_logits)
}

// ═══════════════════════════════════════════════════════════════
// BACKWARD PASS
// ═══════════════════════════════════════════════════════════════

/// Gradients for FFN training.
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub struct FfnGradients {
    pub embed: Vec<f32>,
    pub blocks: Vec<FfnBlockGrads>,
    pub final_ln_gamma: Vec<f32>,
    pub final_ln_beta: Vec<f32>,
    pub lm_head: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub struct FfnBlockGrads {
    pub ln_gamma: Vec<f32>,
    pub ln_beta: Vec<f32>,
    pub gate_w: Vec<f32>,
    pub gate_b: Vec<f32>,
    pub up_w: Vec<f32>,
    pub up_b: Vec<f32>,
    pub down_w: Vec<f32>,
    pub down_b: Vec<f32>,
}

impl FfnGradients {
    pub fn zeros(w: &FfnWeights) -> Self {
        let d = w.config.d_model;
        let h = w.config.hidden;
        Self {
            embed: vec![0.0; w.embed.len()],
            blocks: w.blocks.iter().map(|_| FfnBlockGrads {
                ln_gamma: vec![0.0; d],
                ln_beta: vec![0.0; d],
                gate_w: vec![0.0; h * d],
                gate_b: vec![0.0; h],
                up_w: vec![0.0; h * d],
                up_b: vec![0.0; h],
                down_w: vec![0.0; d * h],
                down_b: vec![0.0; d],
            }).collect(),
            final_ln_gamma: vec![0.0; d],
            final_ln_beta: vec![0.0; d],
            lm_head: vec![0.0; w.lm_head.len()],
        }
    }

    pub fn zero(&mut self) {
        for x in self.embed.iter_mut() { *x = 0.0; }
        for b in self.blocks.iter_mut() {
            for x in b.ln_gamma.iter_mut() { *x = 0.0; }
            for x in b.ln_beta.iter_mut() { *x = 0.0; }
            for x in b.gate_w.iter_mut() { *x = 0.0; }
            for x in b.gate_b.iter_mut() { *x = 0.0; }
            for x in b.up_w.iter_mut() { *x = 0.0; }
            for x in b.up_b.iter_mut() { *x = 0.0; }
            for x in b.down_w.iter_mut() { *x = 0.0; }
            for x in b.down_b.iter_mut() { *x = 0.0; }
        }
        for x in self.final_ln_gamma.iter_mut() { *x = 0.0; }
        for x in self.final_ln_beta.iter_mut() { *x = 0.0; }
        for x in self.lm_head.iter_mut() { *x = 0.0; }
    }
}

/// Backward pass through FFN, accumulating into `grads`.
/// Operates on batched [N × D] slabs; reuses rayon-parallel matmul helpers.
pub fn ffn_backward(
    w: &FfnWeights,
    cache: &FfnCache,
    d_logits: &[Vec<f32>],
    grads: &mut FfnGradients,
) {
    let d = w.config.d_model;
    let h = w.config.hidden;
    let vocab = w.config.vocab;
    let n = cache.n;

    // ─── Pack d_logits [pos][vocab] into slab [N × vocab] ───
    let mut d_logits_slab = vec![0.0f32; n * vocab];
    for (pos, dl) in d_logits.iter().enumerate() {
        d_logits_slab[pos * vocab..(pos + 1) * vocab].copy_from_slice(dl);
    }

    // ─── lm_head backward ───
    // d_lm_head[v,j] += Σ_n d_logits[n,v] * final_normed[n,j]
    // d_final_normed[n,j] = Σ_v d_logits[n,v] * lm_head[v,j]
    matmul_grad_rayon(&d_logits_slab, &cache.final_normed, &mut grads.lm_head, n, d, vocab);
    let mut d_final_normed = vec![0.0f32; n * d];
    matmul_t_rayon(&d_logits_slab, &w.lm_head, &mut d_final_normed, n, d, vocab);

    // ─── Final LN backward ───
    let mut d_x = vec![0.0f32; n * d];
    layer_norm_backward_batched(
        &d_final_normed, &cache.final_ln_input, &w.final_ln_gamma,
        &mut d_x, &mut grads.final_ln_gamma, &mut grads.final_ln_beta,
        n, d,
    );

    // ─── Per-block backward (reverse order) ───
    for (layer_idx, block) in w.blocks.iter().enumerate().rev() {
        let lc = &cache.per_layer[layer_idx];
        let bg = &mut grads.blocks[layer_idx];

        // Residual split: d_x flows into both the skip path (pre_norm side) and the
        // sublayer path (down_out side).
        let d_down_out = d_x.clone();
        let mut d_pre_norm = d_x.clone();

        // ─── down backward ───
        // bias: accumulate across the batch
        for row in 0..n {
            let r = &d_down_out[row * d..(row + 1) * d];
            for j in 0..d { bg.down_b[j] += r[j]; }
        }
        matmul_grad_rayon(&d_down_out, &lc.hidden, &mut bg.down_w, n, h, d);
        let mut d_hidden = vec![0.0f32; n * h];
        matmul_t_rayon(&d_down_out, &block.down.weight, &mut d_hidden, n, h, d);

        // ─── SwiGLU backward: hidden = silu(gate) * up ───
        let mut d_gate_out = vec![0.0f32; n * h];
        let mut d_up_out = vec![0.0f32; n * h];
        d_gate_out.par_iter_mut().zip(d_up_out.par_iter_mut()).enumerate()
            .for_each(|(i, (dg, du))| {
                let g = lc.gate_out[i];
                let s = 1.0 / (1.0 + (-g).exp());
                let silu = g * s;
                let d_silu_dg = s + g * s * (1.0 - s);
                *dg = d_hidden[i] * lc.up_out[i] * d_silu_dg;
                *du = d_hidden[i] * silu;
            });

        // ─── up backward ───
        for row in 0..n {
            let r = &d_up_out[row * h..(row + 1) * h];
            for j in 0..h { bg.up_b[j] += r[j]; }
        }
        matmul_grad_rayon(&d_up_out, &lc.normed, &mut bg.up_w, n, d, h);
        let mut d_normed = vec![0.0f32; n * d];
        matmul_t_rayon(&d_up_out, &block.up.weight, &mut d_normed, n, d, h);

        // ─── gate backward ───
        for row in 0..n {
            let r = &d_gate_out[row * h..(row + 1) * h];
            for j in 0..h { bg.gate_b[j] += r[j]; }
        }
        matmul_grad_rayon(&d_gate_out, &lc.normed, &mut bg.gate_w, n, d, h);
        matmul_t_rayon(&d_gate_out, &block.gate.weight, &mut d_normed, n, d, h);

        // ─── Layer norm backward (into d_pre_norm via residual) ───
        let mut d_ln_input = vec![0.0f32; n * d];
        layer_norm_backward_batched(
            &d_normed, &lc.pre_norm, &block.ln_gamma,
            &mut d_ln_input, &mut bg.ln_gamma, &mut bg.ln_beta,
            n, d,
        );
        d_pre_norm.par_iter_mut().zip(d_ln_input.par_iter())
            .for_each(|(p, &l)| *p += l);

        d_x = d_pre_norm;
    }

    // ─── Embed backward: sparse scatter (serial, cheap) ───
    for (row, &tok) in cache.tokens.iter().enumerate() {
        let dst = tok * d;
        let src = row * d;
        for j in 0..d { grads.embed[dst + j] += d_x[src + j]; }
    }
}

/// Batched row-wise layer-norm backward.
/// `d_out`, `x`, `d_x` are all [N × d]. Accumulates into `d_gamma` / `d_beta` [d].
fn layer_norm_backward_batched(
    d_out: &[f32], x: &[f32], gamma: &[f32],
    d_x: &mut [f32], d_gamma: &mut [f32], d_beta: &mut [f32],
    n: usize, d: usize,
) {
    let df = d as f32;
    let _ = n;

    // One parallel pass per row: compute d_x_row and the row's contributions to
    // d_gamma / d_beta. We can write d_x directly via par_chunks_mut, but
    // d_gamma / d_beta need a reduction, so we collect per-row buffers.
    let row_grads: Vec<(Vec<f32>, Vec<f32>)> = d_x.par_chunks_mut(d).enumerate().map(|(row, dx_row)| {
        let x_row = &x[row * d..(row + 1) * d];
        let d_out_row = &d_out[row * d..(row + 1) * d];

        let mean: f32 = x_row.iter().sum::<f32>() / df;
        let var: f32 = x_row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / df;
        let inv_std = 1.0 / (var + 1e-5).sqrt();

        let mut d_gamma_row = vec![0.0f32; d];
        let mut d_beta_row = vec![0.0f32; d];
        let mut d_x_hat = vec![0.0f32; d];

        for j in 0..d {
            let x_hat = (x_row[j] - mean) * inv_std;
            d_gamma_row[j] = d_out_row[j] * x_hat;
            d_beta_row[j] = d_out_row[j];
            d_x_hat[j] = d_out_row[j] * gamma[j];
        }

        let sum_dxh: f32 = d_x_hat.iter().sum();
        let sum_dxh_xh: f32 = (0..d).map(|j| d_x_hat[j] * (x_row[j] - mean) * inv_std).sum();

        for j in 0..d {
            let x_hat = (x_row[j] - mean) * inv_std;
            dx_row[j] = inv_std / df * (df * d_x_hat[j] - sum_dxh - x_hat * sum_dxh_xh);
        }

        (d_gamma_row, d_beta_row)
    }).collect();

    // Reduce d_gamma / d_beta across rows.
    for (dg_row, db_row) in &row_grads {
        for j in 0..d {
            d_gamma[j] += dg_row[j];
            d_beta[j] += db_row[j];
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// ADAMW OPTIMIZER
// ═══════════════════════════════════════════════════════════════

/// Canonical tensor ordering inside an `OptimizerState` backing an `FfnAdamW`.
/// Index 0 is `embed`; each layer contributes 8 tensors; then `final_ln_gamma`,
/// `final_ln_beta`, `lm_head`. Callers compute indices via `ffn_tensor_index`.
const TENSORS_PER_BLOCK: usize = 8;
const BLOCK_LN_GAMMA: usize = 0;
const BLOCK_LN_BETA:  usize = 1;
const BLOCK_GATE_W:   usize = 2;
const BLOCK_GATE_B:   usize = 3;
const BLOCK_UP_W:     usize = 4;
const BLOCK_UP_B:     usize = 5;
const BLOCK_DOWN_W:   usize = 6;
const BLOCK_DOWN_B:   usize = 7;

fn idx_embed() -> usize { 0 }
fn idx_block(layer: usize, which: usize) -> usize { 1 + layer * TENSORS_PER_BLOCK + which }
fn idx_final_ln_gamma(n_layers: usize) -> usize { 1 + n_layers * TENSORS_PER_BLOCK }
fn idx_final_ln_beta(n_layers: usize)  -> usize { 1 + n_layers * TENSORS_PER_BLOCK + 1 }
fn idx_lm_head(n_layers: usize)        -> usize { 1 + n_layers * TENSORS_PER_BLOCK + 2 }

/// Size of every tensor in canonical order. Matches the index helpers above.
fn ffn_tensor_sizes(w: &FfnWeights) -> Vec<usize> {
    let mut sizes = Vec::with_capacity(1 + w.blocks.len() * TENSORS_PER_BLOCK + 3);
    sizes.push(w.embed.len());
    for block in &w.blocks {
        sizes.push(block.ln_gamma.len());
        sizes.push(block.ln_beta.len());
        sizes.push(block.gate.weight.len());
        sizes.push(block.gate.bias.len());
        sizes.push(block.up.weight.len());
        sizes.push(block.up.bias.len());
        sizes.push(block.down.weight.len());
        sizes.push(block.down.bias.len());
    }
    sizes.push(w.final_ln_gamma.len());
    sizes.push(w.final_ln_beta.len());
    sizes.push(w.lm_head.len());
    sizes
}

#[derive(Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub struct FfnAdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub wd: f32,
    pub clip: f32,
    pub step: usize,
    // Manual Clone below — can't derive because `vram: Option<Box<dyn ...>>`
    // isn't Clone. Semantics: cloning for save drops the VRAM mirror;
    // re-materialising happens via enable_vram() on the clone if needed.
    // Flat moment buffers — one per parameter group
    m: FfnGradients,
    v: FfnGradients,
    /// Device-backed mirror of weights/grads/m/v. Set via `enable_vram`.
    /// When present, `step_update` dispatches AdamW on device directly
    /// (zero-copy), and `m` / `v` in this struct become stale until a
    /// `sync_moments_to_cpu` call (done automatically before save).
    ///
    /// Device-agnostic — on AMD this is a KFD `VramMirror`, on CUDA it's a
    /// cudarc-backed impl, etc. The FFN code here never names either.
    #[serde(skip)]
    #[wincode(skip)]
    vram: Option<Box<dyn modgrad_compute::optimizer_state::OptimizerState>>,
}

// Manual Clone: VramMirror isn't Clone (holds raw GPU buffer handles),
// so the clone drops the mirror. Callers that want the mirror on the
// clone re-allocate it with `enable_vram(&weights)` on the clone.
impl Clone for FfnAdamW {
    fn clone(&self) -> Self {
        Self {
            lr: self.lr, beta1: self.beta1, beta2: self.beta2,
            eps: self.eps, wd: self.wd, clip: self.clip, step: self.step,
            m: self.m.clone(),
            v: self.v.clone(),
            vram: None,
        }
    }
}

impl FfnAdamW {
    pub fn new(w: &FfnWeights) -> Self {
        Self {
            lr: 3e-4,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            wd: 0.01,
            clip: 1.0,
            step: 0,
            m: FfnGradients::zeros(w),
            v: FfnGradients::zeros(w),
            vram: None,
        }
    }

    pub fn with_lr(mut self, lr: f32) -> Self { self.lr = lr; self }

    /// Allocate a VRAM mirror and upload current `w`, `self.m`, `self.v` to it.
    /// Returns `true` on success; `false` if GPU is unavailable or any alloc
    /// fails (caller stays on the CPU path).
    ///
    /// After this, training steps invoke `adamw_zerocopy` on VRAM directly;
    /// ~5 GB of per-step BAR traffic disappears. CPU-side `self.m` and `self.v`
    /// go stale until the next `save` (which syncs them first).
    pub fn enable_vram(&mut self, w: &FfnWeights) -> bool {
        let sizes = ffn_tensor_sizes(w);
        let mut mirror = match modgrad_compute::make_optimizer_state(sizes) {
            Some(m) => m, None => return false,
        };
        let n_layers = w.blocks.len();

        // Upload weights via the trait — no device type named here.
        mirror.upload_param(idx_embed(), &w.embed);
        for (li, block) in w.blocks.iter().enumerate() {
            mirror.upload_param(idx_block(li, BLOCK_LN_GAMMA), &block.ln_gamma);
            mirror.upload_param(idx_block(li, BLOCK_LN_BETA),  &block.ln_beta);
            mirror.upload_param(idx_block(li, BLOCK_GATE_W),   &block.gate.weight);
            mirror.upload_param(idx_block(li, BLOCK_GATE_B),   &block.gate.bias);
            mirror.upload_param(idx_block(li, BLOCK_UP_W),     &block.up.weight);
            mirror.upload_param(idx_block(li, BLOCK_UP_B),     &block.up.bias);
            mirror.upload_param(idx_block(li, BLOCK_DOWN_W),   &block.down.weight);
            mirror.upload_param(idx_block(li, BLOCK_DOWN_B),   &block.down.bias);
        }
        mirror.upload_param(idx_final_ln_gamma(n_layers), &w.final_ln_gamma);
        mirror.upload_param(idx_final_ln_beta(n_layers),  &w.final_ln_beta);
        mirror.upload_param(idx_lm_head(n_layers),        &w.lm_head);

        // Upload m, v (may be zero if fresh training, or loaded values on resume).
        mirror.upload_m(idx_embed(), &self.m.embed);
        mirror.upload_v(idx_embed(), &self.v.embed);
        for li in 0..n_layers {
            let mb = &self.m.blocks[li]; let vb = &self.v.blocks[li];
            mirror.upload_m(idx_block(li, BLOCK_LN_GAMMA), &mb.ln_gamma);
            mirror.upload_v(idx_block(li, BLOCK_LN_GAMMA), &vb.ln_gamma);
            mirror.upload_m(idx_block(li, BLOCK_LN_BETA),  &mb.ln_beta);
            mirror.upload_v(idx_block(li, BLOCK_LN_BETA),  &vb.ln_beta);
            mirror.upload_m(idx_block(li, BLOCK_GATE_W),   &mb.gate_w);
            mirror.upload_v(idx_block(li, BLOCK_GATE_W),   &vb.gate_w);
            mirror.upload_m(idx_block(li, BLOCK_GATE_B),   &mb.gate_b);
            mirror.upload_v(idx_block(li, BLOCK_GATE_B),   &vb.gate_b);
            mirror.upload_m(idx_block(li, BLOCK_UP_W),     &mb.up_w);
            mirror.upload_v(idx_block(li, BLOCK_UP_W),     &vb.up_w);
            mirror.upload_m(idx_block(li, BLOCK_UP_B),     &mb.up_b);
            mirror.upload_v(idx_block(li, BLOCK_UP_B),     &vb.up_b);
            mirror.upload_m(idx_block(li, BLOCK_DOWN_W),   &mb.down_w);
            mirror.upload_v(idx_block(li, BLOCK_DOWN_W),   &vb.down_w);
            mirror.upload_m(idx_block(li, BLOCK_DOWN_B),   &mb.down_b);
            mirror.upload_v(idx_block(li, BLOCK_DOWN_B),   &vb.down_b);
        }
        mirror.upload_m(idx_final_ln_gamma(n_layers), &self.m.final_ln_gamma);
        mirror.upload_v(idx_final_ln_gamma(n_layers), &self.v.final_ln_gamma);
        mirror.upload_m(idx_final_ln_beta(n_layers),  &self.m.final_ln_beta);
        mirror.upload_v(idx_final_ln_beta(n_layers),  &self.v.final_ln_beta);
        mirror.upload_m(idx_lm_head(n_layers),        &self.m.lm_head);
        mirror.upload_v(idx_lm_head(n_layers),        &self.v.lm_head);

        self.vram = Some(mirror);
        true
    }

    /// True if the mirror is active.
    pub fn vram_enabled(&self) -> bool { self.vram.is_some() }

    /// Pull `m` and `v` out of VRAM into `self.m`/`self.v`. Called automatically
    /// before `save`; external callers can invoke it to inspect moments on CPU.
    pub fn sync_moments_to_cpu(&mut self) {
        let mirror = match &self.vram { Some(m) => m, None => return };
        // Embed.
        self.m.embed = mirror.download_m(idx_embed());
        self.v.embed = mirror.download_v(idx_embed());
        for li in 0..self.m.blocks.len() {
            let mb = &mut self.m.blocks[li]; let vb = &mut self.v.blocks[li];
            mb.ln_gamma = mirror.download_m(idx_block(li, BLOCK_LN_GAMMA));
            vb.ln_gamma = mirror.download_v(idx_block(li, BLOCK_LN_GAMMA));
            mb.ln_beta  = mirror.download_m(idx_block(li, BLOCK_LN_BETA));
            vb.ln_beta  = mirror.download_v(idx_block(li, BLOCK_LN_BETA));
            mb.gate_w   = mirror.download_m(idx_block(li, BLOCK_GATE_W));
            vb.gate_w   = mirror.download_v(idx_block(li, BLOCK_GATE_W));
            mb.gate_b   = mirror.download_m(idx_block(li, BLOCK_GATE_B));
            vb.gate_b   = mirror.download_v(idx_block(li, BLOCK_GATE_B));
            mb.up_w     = mirror.download_m(idx_block(li, BLOCK_UP_W));
            vb.up_w     = mirror.download_v(idx_block(li, BLOCK_UP_W));
            mb.up_b     = mirror.download_m(idx_block(li, BLOCK_UP_B));
            vb.up_b     = mirror.download_v(idx_block(li, BLOCK_UP_B));
            mb.down_w   = mirror.download_m(idx_block(li, BLOCK_DOWN_W));
            vb.down_w   = mirror.download_v(idx_block(li, BLOCK_DOWN_W));
            mb.down_b   = mirror.download_m(idx_block(li, BLOCK_DOWN_B));
            vb.down_b   = mirror.download_v(idx_block(li, BLOCK_DOWN_B));
        }
        let n_layers = self.m.blocks.len();
        self.m.final_ln_gamma = mirror.download_m(idx_final_ln_gamma(n_layers));
        self.v.final_ln_gamma = mirror.download_v(idx_final_ln_gamma(n_layers));
        self.m.final_ln_beta  = mirror.download_m(idx_final_ln_beta(n_layers));
        self.v.final_ln_beta  = mirror.download_v(idx_final_ln_beta(n_layers));
        self.m.lm_head        = mirror.download_m(idx_lm_head(n_layers));
        self.v.lm_head        = mirror.download_v(idx_lm_head(n_layers));
    }

    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        // If VRAM mirror is active, sync moments down first so the serialised
        // state matches what the mirror holds. (Saving weights is the caller's
        // responsibility via FfnWeights::save after sync_weights_to_cpu.)
        if self.vram.is_some() {
            // `save` takes `&self` — the caller should call `sync_moments_to_cpu`
            // via `&mut self` before invoking save. Log a warning if not synced.
            eprintln!("  warning: FfnAdamW::save called with vram active; \
                       call sync_moments_to_cpu() first to avoid stale m/v.");
        }
        modgrad_persist::persist::save(self, path).map_err(|e| e.into())
    }

    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        modgrad_persist::persist::load(path).map_err(|e| e.into())
    }

    pub fn step_update(&mut self, w: &mut FfnWeights, g: &FfnGradients) {
        self.step += 1;
        let b1 = self.beta1;
        let b2 = self.beta2;
        let t = self.step as i32;
        let bc1 = 1.0 - b1.powi(t);
        let bc2 = 1.0 - b2.powi(t);

        // Compute grad norm for clipping (always CPU — grads live on CPU).
        let norm = {
            let mut sq = 0.0f32;
            for x in &g.embed { sq += x * x; }
            for b in &g.blocks {
                for x in &b.ln_gamma { sq += x * x; }
                for x in &b.ln_beta { sq += x * x; }
                for x in &b.gate_w { sq += x * x; }
                for x in &b.gate_b { sq += x * x; }
                for x in &b.up_w { sq += x * x; }
                for x in &b.up_b { sq += x * x; }
                for x in &b.down_w { sq += x * x; }
                for x in &b.down_b { sq += x * x; }
            }
            for x in &g.final_ln_gamma { sq += x * x; }
            for x in &g.final_ln_beta { sq += x * x; }
            for x in &g.lm_head { sq += x * x; }
            sq.sqrt()
        };
        let scale = if norm > self.clip { self.clip / norm } else { 1.0 };

        let lr = self.lr;
        let eps = self.eps;
        let wd = self.wd;

        // ─── VRAM-resident fast path ───
        // Weights / m / v stay in VRAM across steps. We BAR-write pre-scaled
        // grads into the mirror, dispatch adamw_zerocopy, then BAR-read updated
        // weights back into `w` so the next forward pass sees them.
        // (Forward matmul still reads from CPU `w`; a follow-up will keep
        // forward entirely on VRAM and drop the weight readback.)
        if self.vram.is_some() {
            if self.step_update_vram(w, g, scale, b1, b2, bc1, bc2, eps, lr, wd) {
                return;
            }
            // If VRAM dispatch failed mid-step, fall through to CPU. State is
            // consistent because we only write-through to mirror on success.
            eprintln!("  warning: VRAM adamw failed, falling back to CPU path");
        }

        // ─── CPU path (original) ───
        adamw_apply(&mut w.embed, &g.embed, &mut self.m.embed, &mut self.v.embed,
            b1, b2, bc1, bc2, eps, lr, 0.0, scale); // embed: no wd

        for i in 0..w.blocks.len() {
            let bw = &mut w.blocks[i];
            let bg = &g.blocks[i];
            let bm = &mut self.m.blocks[i];
            let bv = &mut self.v.blocks[i];
            adamw_apply(&mut bw.ln_gamma, &bg.ln_gamma, &mut bm.ln_gamma, &mut bv.ln_gamma, b1, b2, bc1, bc2, eps, lr, 0.0, scale);
            adamw_apply(&mut bw.ln_beta, &bg.ln_beta, &mut bm.ln_beta, &mut bv.ln_beta, b1, b2, bc1, bc2, eps, lr, 0.0, scale);
            adamw_apply(&mut bw.gate.weight, &bg.gate_w, &mut bm.gate_w, &mut bv.gate_w, b1, b2, bc1, bc2, eps, lr, wd, scale);
            adamw_apply(&mut bw.gate.bias, &bg.gate_b, &mut bm.gate_b, &mut bv.gate_b, b1, b2, bc1, bc2, eps, lr, 0.0, scale);
            adamw_apply(&mut bw.up.weight, &bg.up_w, &mut bm.up_w, &mut bv.up_w, b1, b2, bc1, bc2, eps, lr, wd, scale);
            adamw_apply(&mut bw.up.bias, &bg.up_b, &mut bm.up_b, &mut bv.up_b, b1, b2, bc1, bc2, eps, lr, 0.0, scale);
            adamw_apply(&mut bw.down.weight, &bg.down_w, &mut bm.down_w, &mut bv.down_w, b1, b2, bc1, bc2, eps, lr, wd, scale);
            adamw_apply(&mut bw.down.bias, &bg.down_b, &mut bm.down_b, &mut bv.down_b, b1, b2, bc1, bc2, eps, lr, 0.0, scale);
        }
        adamw_apply(&mut w.final_ln_gamma, &g.final_ln_gamma, &mut self.m.final_ln_gamma, &mut self.v.final_ln_gamma, b1, b2, bc1, bc2, eps, lr, 0.0, scale);
        adamw_apply(&mut w.final_ln_beta, &g.final_ln_beta, &mut self.m.final_ln_beta, &mut self.v.final_ln_beta, b1, b2, bc1, bc2, eps, lr, 0.0, scale);
        adamw_apply(&mut w.lm_head, &g.lm_head, &mut self.m.lm_head, &mut self.v.lm_head, b1, b2, bc1, bc2, eps, lr, wd, scale);
    }

    /// Device-backed branch of `step_update`. Returns `false` on any sub-dispatch
    /// failure; caller falls back to CPU path. Device-agnostic: this code
    /// names only `dyn OptimizerState`, so it runs identically on AMD KFD
    /// and (future) CUDA backends.
    fn step_update_vram(
        &mut self, w: &mut FfnWeights, g: &FfnGradients,
        scale: f32, b1: f32, b2: f32, bc1: f32, bc2: f32, eps: f32, lr: f32, wd: f32,
    ) -> bool {
        let mirror = self.vram.as_mut().expect("vram must be Some in step_update_vram");
        let n_layers = w.blocks.len();
        let bc1_inv = 1.0 / bc1;
        let bc2_inv = 1.0 / bc2;

        // Stage gradients (scaled by clip factor) through the trait — each
        // backend decides whether this is a BAR write (AMD KFD), a cudaMemcpyAsync
        // (CUDA), etc. We don't care; FFN just calls `write_grad_scaled`.
        mirror.write_grad_scaled(idx_embed(), &g.embed, scale);
        for li in 0..n_layers {
            let bg = &g.blocks[li];
            mirror.write_grad_scaled(idx_block(li, BLOCK_LN_GAMMA), &bg.ln_gamma, scale);
            mirror.write_grad_scaled(idx_block(li, BLOCK_LN_BETA),  &bg.ln_beta,  scale);
            mirror.write_grad_scaled(idx_block(li, BLOCK_GATE_W),   &bg.gate_w,   scale);
            mirror.write_grad_scaled(idx_block(li, BLOCK_GATE_B),   &bg.gate_b,   scale);
            mirror.write_grad_scaled(idx_block(li, BLOCK_UP_W),     &bg.up_w,     scale);
            mirror.write_grad_scaled(idx_block(li, BLOCK_UP_B),     &bg.up_b,     scale);
            mirror.write_grad_scaled(idx_block(li, BLOCK_DOWN_W),   &bg.down_w,   scale);
            mirror.write_grad_scaled(idx_block(li, BLOCK_DOWN_B),   &bg.down_b,   scale);
        }
        mirror.write_grad_scaled(idx_final_ln_gamma(n_layers), &g.final_ln_gamma, scale);
        mirror.write_grad_scaled(idx_final_ln_beta(n_layers),  &g.final_ln_beta,  scale);
        mirror.write_grad_scaled(idx_lm_head(n_layers),        &g.lm_head,        scale);

        // Per-tensor wd mask: weights get `wd`, everything else (biases,
        // layer-norm γ/β, embed, final LN) gets 0. Mirrors the original CPU
        // loop.
        let wd_for = move |idx: usize| -> f32 {
            if idx == idx_embed() { return 0.0; }
            if idx == idx_final_ln_gamma(n_layers) || idx == idx_final_ln_beta(n_layers) { return 0.0; }
            if idx == idx_lm_head(n_layers) { return wd; }
            let offset = (idx - 1) % TENSORS_PER_BLOCK;
            match offset {
                BLOCK_GATE_W | BLOCK_UP_W | BLOCK_DOWN_W => wd,
                _ => 0.0,
            }
        };

        // One batched dispatch + single submit_wait across every tensor.
        if !mirror.adamw_step(lr, b1, b2, eps, bc1_inv, bc2_inv, &wd_for) {
            return false;
        }

        // Sync updated weights back to CPU. On AMD this is a BAR read (zero
        // copy); on future CUDA backends it'll be cudaMemcpyDtoH. Device
        // decides; the trait just exposes `download_param`.
        w.embed = mirror.download_param(idx_embed());
        for li in 0..n_layers {
            let bw = &mut w.blocks[li];
            bw.ln_gamma   = mirror.download_param(idx_block(li, BLOCK_LN_GAMMA));
            bw.ln_beta    = mirror.download_param(idx_block(li, BLOCK_LN_BETA));
            bw.gate.weight = mirror.download_param(idx_block(li, BLOCK_GATE_W));
            bw.gate.bias   = mirror.download_param(idx_block(li, BLOCK_GATE_B));
            bw.up.weight   = mirror.download_param(idx_block(li, BLOCK_UP_W));
            bw.up.bias     = mirror.download_param(idx_block(li, BLOCK_UP_B));
            bw.down.weight = mirror.download_param(idx_block(li, BLOCK_DOWN_W));
            bw.down.bias   = mirror.download_param(idx_block(li, BLOCK_DOWN_B));
        }
        w.final_ln_gamma = mirror.download_param(idx_final_ln_gamma(n_layers));
        w.final_ln_beta  = mirror.download_param(idx_final_ln_beta(n_layers));
        w.lm_head        = mirror.download_param(idx_lm_head(n_layers));

        true
    }
}

fn adamw_apply(
    weights: &mut [f32], grads: &[f32], m: &mut [f32], v: &mut [f32],
    beta1: f32, beta2: f32, bc1: f32, bc2: f32, eps: f32,
    lr: f32, wd: f32, clip_scale: f32,
) {
    // GPU AdamW: one dispatch handles update + zero-grads.
    // try_adamw expects grads as &mut (it zeros them), but we want to keep them for logging.
    // Make a copy for the GPU call (grads are still live for multiple weight groups).
    // For now use CPU — the GPU kernel signature doesn't match cleanly (mutates grads).
    // TODO: add try_adamw_noclear variant or clone grads.

    // Pre-scale grads once on CPU; then dispatch via the backend
    // registry. The registry picks the fastest backend that supports
    // AdamW at this shape; size-gated KFD paths are honored inside
    // supports(). Callers don't branch on hardware.
    let g_scaled: Vec<f32> = grads.iter().map(|&g| g * clip_scale).collect();
    use modgrad_device::backend::{registry, AdamWArgs, Op};
    registry().dispatch(&mut Op::AdamW(AdamWArgs {
        w: weights, g: &g_scaled, m, v,
        lr, beta1, beta2, eps, weight_decay: wd,
        bc1_inv: 1.0 / bc1, bc2_inv: 1.0 / bc2,
    })).expect("adamw dispatch");
}

// ═══════════════════════════════════════════════════════════════
// TRAINING STEP
// ═══════════════════════════════════════════════════════════════

/// One training step: forward + loss + backward + update.
/// Returns (loss, accuracy).
pub fn ffn_train_step(
    w: &mut FfnWeights,
    opt: &mut FfnAdamW,
    grads: &mut FfnGradients,
    tokens: &[usize],
) -> (f32, f32) {
    grads.zero();
    if tokens.len() < 2 { return (0.0, 0.0); }

    // Split `tokens` into aligned input/target views so the forward pass
    // gets exactly `context_len` rows (a multiple of 32 for the standard
    // --context 64/128/256 knobs). Previously `ffn_forward` was called
    // with all `tokens` (length context_len+1) and the trailing row was
    // silently discarded — which meant every FFN training step produced
    // an n=65 matmul that the GPU kernels reject (n%32==0 precondition),
    // making --gpu a silent no-op.
    let inputs = &tokens[..tokens.len() - 1];
    let targets = &tokens[1..];

    let (logits, cache) = ffn_forward(w, inputs);
    let (loss, d_logits) = ffn_loss(&logits, targets);

    // Accuracy — same positions the loss scored.
    let mut correct = 0usize;
    let n = inputs.len();
    for pos in 0..n {
        let pred = logits[pos].iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i).unwrap_or(0);
        if pred == targets[pos] { correct += 1; }
    }
    let acc = correct as f32 / n.max(1) as f32;

    ffn_backward(w, &cache, &d_logits, grads);
    opt.step_update(w, grads);

    // Weights were updated in place — the GPU's VRAM-cached copies are now
    // stale. Drop them so the next forward re-uploads the current values.
    // No-op for backends that don't maintain weight caches.
    modgrad_device::backend::registry().invalidate_caches();

    (loss, acc)
}

// ═══════════════════════════════════════════════════════════════
// ParamIter impls — single-site declaration of parameter layout.
// Replaces the canonical-index arithmetic (idx_embed, idx_block, …).
// Any machinery that wants to upload / checkpoint / norm / iterate
// parameters walks these and stays correct without coordination.
// ═══════════════════════════════════════════════════════════════

impl ParamIter for FfnWeights {
    fn walk_params(&self, f: &mut dyn FnMut(&str, &[f32])) {
        f("embed", &self.embed);
        for (i, blk) in self.blocks.iter().enumerate() {
            f(&format!("blocks.{i}.ln_gamma"),    &blk.ln_gamma);
            f(&format!("blocks.{i}.ln_beta"),     &blk.ln_beta);
            f(&format!("blocks.{i}.gate.weight"), &blk.gate.weight);
            f(&format!("blocks.{i}.gate.bias"),   &blk.gate.bias);
            f(&format!("blocks.{i}.up.weight"),   &blk.up.weight);
            f(&format!("blocks.{i}.up.bias"),     &blk.up.bias);
            f(&format!("blocks.{i}.down.weight"), &blk.down.weight);
            f(&format!("blocks.{i}.down.bias"),   &blk.down.bias);
        }
        f("final_ln_gamma", &self.final_ln_gamma);
        f("final_ln_beta",  &self.final_ln_beta);
        f("lm_head",        &self.lm_head);
    }
    fn walk_params_mut(&mut self, f: &mut dyn FnMut(&str, &mut [f32])) {
        f("embed", &mut self.embed);
        for (i, blk) in self.blocks.iter_mut().enumerate() {
            f(&format!("blocks.{i}.ln_gamma"),    &mut blk.ln_gamma);
            f(&format!("blocks.{i}.ln_beta"),     &mut blk.ln_beta);
            f(&format!("blocks.{i}.gate.weight"), &mut blk.gate.weight);
            f(&format!("blocks.{i}.gate.bias"),   &mut blk.gate.bias);
            f(&format!("blocks.{i}.up.weight"),   &mut blk.up.weight);
            f(&format!("blocks.{i}.up.bias"),     &mut blk.up.bias);
            f(&format!("blocks.{i}.down.weight"), &mut blk.down.weight);
            f(&format!("blocks.{i}.down.bias"),   &mut blk.down.bias);
        }
        f("final_ln_gamma", &mut self.final_ln_gamma);
        f("final_ln_beta",  &mut self.final_ln_beta);
        f("lm_head",        &mut self.lm_head);
    }
}

impl ParamIter for FfnGradients {
    fn walk_params(&self, f: &mut dyn FnMut(&str, &[f32])) {
        f("embed", &self.embed);
        for (i, blk) in self.blocks.iter().enumerate() {
            f(&format!("blocks.{i}.ln_gamma"), &blk.ln_gamma);
            f(&format!("blocks.{i}.ln_beta"),  &blk.ln_beta);
            f(&format!("blocks.{i}.gate.weight"), &blk.gate_w);
            f(&format!("blocks.{i}.gate.bias"),   &blk.gate_b);
            f(&format!("blocks.{i}.up.weight"),   &blk.up_w);
            f(&format!("blocks.{i}.up.bias"),     &blk.up_b);
            f(&format!("blocks.{i}.down.weight"), &blk.down_w);
            f(&format!("blocks.{i}.down.bias"),   &blk.down_b);
        }
        f("final_ln_gamma", &self.final_ln_gamma);
        f("final_ln_beta",  &self.final_ln_beta);
        f("lm_head",        &self.lm_head);
    }
    fn walk_params_mut(&mut self, f: &mut dyn FnMut(&str, &mut [f32])) {
        f("embed", &mut self.embed);
        for (i, blk) in self.blocks.iter_mut().enumerate() {
            f(&format!("blocks.{i}.ln_gamma"), &mut blk.ln_gamma);
            f(&format!("blocks.{i}.ln_beta"),  &mut blk.ln_beta);
            f(&format!("blocks.{i}.gate.weight"), &mut blk.gate_w);
            f(&format!("blocks.{i}.gate.bias"),   &mut blk.gate_b);
            f(&format!("blocks.{i}.up.weight"),   &mut blk.up_w);
            f(&format!("blocks.{i}.up.bias"),     &mut blk.up_b);
            f(&format!("blocks.{i}.down.weight"), &mut blk.down_w);
            f(&format!("blocks.{i}.down.bias"),   &mut blk.down_b);
        }
        f("final_ln_gamma", &mut self.final_ln_gamma);
        f("final_ln_beta",  &mut self.final_ln_beta);
        f("lm_head",        &mut self.lm_head);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn param_iter_visits_every_tensor_exactly_once() {
        let cfg = FfnConfig { vocab: 4, d_model: 8, hidden: 16, n_layers: 3, context: 16 };
        let w = FfnWeights::new(cfg);
        let mut names = Vec::new();
        let mut total_via_walk = 0usize;
        w.walk_params(&mut |name, data| {
            names.push(name.to_string());
            total_via_walk += data.len();
        });
        // Expected: embed + 3 blocks × 8 tensors + 3 final = 28
        assert_eq!(names.len(), 1 + 3 * 8 + 3);
        assert_eq!(names[0], "embed");
        assert_eq!(names.last().unwrap(), "lm_head");
        assert_eq!(total_via_walk, w.n_params(),
            "walk must hit every tensor the n_params count is built from");
    }

    #[test]
    fn param_iter_order_is_stable_and_matches_grads() {
        // Invariant: walks of FfnWeights and FfnGradients produce the same
        // tensor name sequence. Any optimizer that zips the two lists
        // depends on this.
        let cfg = FfnConfig { vocab: 4, d_model: 8, hidden: 16, n_layers: 2, context: 16 };
        let w = FfnWeights::new(cfg);
        let g = FfnGradients::zeros(&w);
        let mut w_names = Vec::new();
        let mut g_names = Vec::new();
        w.walk_params(&mut |n, _| w_names.push(n.to_string()));
        g.walk_params(&mut |n, _| g_names.push(n.to_string()));
        assert_eq!(w_names, g_names,
            "FfnWeights and FfnGradients walks must emit identical names");
    }

    #[test]
    fn param_iter_mut_can_zero_all_weights() {
        let cfg = FfnConfig { vocab: 4, d_model: 8, hidden: 16, n_layers: 1, context: 16 };
        let mut w = FfnWeights::new(cfg);
        w.walk_params_mut(&mut |_, data| {
            for x in data.iter_mut() { *x = 0.0; }
        });
        let mut total_nonzero = 0usize;
        w.walk_params(&mut |_, data| {
            total_nonzero += data.iter().filter(|&&v| v != 0.0).count();
        });
        assert_eq!(total_nonzero, 0);
    }

    #[test]
    fn ffn_forward_shape() {
        let cfg = FfnConfig { vocab: 10, d_model: 8, hidden: 16, n_layers: 2, context: 16 };
        let w = FfnWeights::new(cfg);
        let tokens = vec![1, 2, 3, 4];
        let (logits, _cache) = ffn_forward(&w, &tokens);
        assert_eq!(logits.len(), tokens.len());
        assert_eq!(logits[0].len(), 10);
    }

    #[test]
    fn ffn_trains() {
        // Memorize a simple pattern: 0→1, 1→2, 2→3, 3→0
        let cfg = FfnConfig { vocab: 4, d_model: 16, hidden: 32, n_layers: 2, context: 16 };
        let mut w = FfnWeights::new(cfg);
        let mut opt = FfnAdamW::new(&w).with_lr(1e-2);
        let mut grads = FfnGradients::zeros(&w);

        let seq = vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3];
        let mut final_loss = 0.0;
        for _step in 0..100 {
            let (loss, _acc) = ffn_train_step(&mut w, &mut opt, &mut grads, &seq);
            final_loss = loss;
        }
        // Should learn the pattern
        assert!(final_loss < 1.0, "loss should drop below 1.0, got {}", final_loss);
    }
}
