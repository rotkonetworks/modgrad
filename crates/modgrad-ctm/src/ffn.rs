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
use rayon::prelude::*;

/// FFN cerebellum config.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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

    pub fn n_params(&self) -> usize {
        self.embed.len()
            + self.blocks.iter().map(|b| b.n_params()).sum::<usize>()
            + self.final_ln_gamma.len() + self.final_ln_beta.len()
            + self.lm_head.len()
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

/// How many rows of the batch dimension each rayon thread processes.
/// At N=128 and 32 cores, a chunk size of 4 gives full core coverage.
const GEMM_N_CHUNK: usize = 4;

/// Batched matmul: Y[n×m] = A[n×k] @ W^T[k×m] + bias[m].
/// Splits the batch dimension across rayon threads; each chunk calls matrixmultiply
/// independently (SIMD-vectorized sgemm). Result: SIMD × multi-core parallelism.
#[inline]
fn matmul_rayon(a: &[f32], weight: &[f32], bias: &[f32], y: &mut [f32],
                _n: usize, k: usize, m: usize) {
    y.par_chunks_mut(GEMM_N_CHUNK * m)
        .zip(a.par_chunks(GEMM_N_CHUNK * k))
        .for_each(|(y_chunk, a_chunk)| {
            let rows = y_chunk.len() / m;
            // Stamp bias into this chunk (broadcast).
            for r in 0..rows {
                y_chunk[r * m..(r + 1) * m].copy_from_slice(bias);
            }
            // Y += A @ W^T   with A:[rows×k], W:[m×k] → W^T:[k×m], Y:[rows×m]
            unsafe {
                matrixmultiply::sgemm(
                    rows, k, m,
                    1.0, a_chunk.as_ptr(), k as isize, 1,
                    weight.as_ptr(), 1, k as isize,
                    1.0, y_chunk.as_mut_ptr(), m as isize, 1,
                );
            }
        });
}

/// Batched transposed matmul: dA[n×k] += dY[n×m] @ W[m×k].
/// Splits the batch dimension across rayon threads.
#[inline]
fn matmul_t_rayon(dy: &[f32], weight: &[f32], da: &mut [f32],
                  _n: usize, k: usize, m: usize) {
    da.par_chunks_mut(GEMM_N_CHUNK * k)
        .zip(dy.par_chunks(GEMM_N_CHUNK * m))
        .for_each(|(da_chunk, dy_chunk)| {
            let rows = da_chunk.len() / k;
            unsafe {
                matrixmultiply::sgemm(
                    rows, m, k,
                    1.0, dy_chunk.as_ptr(), m as isize, 1,
                    weight.as_ptr(), k as isize, 1,
                    1.0, da_chunk.as_mut_ptr(), k as isize, 1,
                );
            }
        });
}

/// Weight-gradient matmul: dW[m×k] += dY^T[m×n] @ A[n×k].
/// Split along output dim m (= rows of dW). Each thread owns a row range
/// and reads all n rows of dY and A — thread-safe since dW rows don't overlap.
#[inline]
fn matmul_grad_rayon(dy: &[f32], a: &[f32], dw: &mut [f32],
                     n: usize, k: usize, m: usize) {
    // Chunk size chosen for ~32 chunks on an m=5120 matmul (160 rows each).
    let m_chunk = (m / 32).max(32);
    dw.par_chunks_mut(m_chunk * k).enumerate().for_each(|(chunk_idx, dw_chunk)| {
        let rows = dw_chunk.len() / k;
        let m_start = chunk_idx * m_chunk;
        // dy_sub is a column-sub-slice of dY: still n rows, but only `rows` cols starting at m_start.
        // Its strides inside the full dY: row stride m, col stride 1.
        // For the transposed view used by sgemm, strides swap: row stride (of dY^T_sub) = 1, col stride = m.
        // Offset into dY: column m_start.
        let dy_sub = &dy[m_start..];
        unsafe {
            matrixmultiply::sgemm(
                rows, n, k,
                1.0, dy_sub.as_ptr(), 1, m as isize,
                a.as_ptr(), k as isize, 1,
                1.0, dw_chunk.as_mut_ptr(), k as isize, 1,
            );
        }
    });
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
pub fn ffn_loss(logits: &[Vec<f32>], tokens: &[usize]) -> (f32, Vec<Vec<f32>>) {
    let n = logits.len();
    if n < 2 { return (0.0, vec![vec![0.0; logits.get(0).map_or(0, |l| l.len())]; n]); }

    let mut total_loss = 0.0f32;
    let mut d_logits = Vec::with_capacity(n);

    for pos in 0..n - 1 {
        let target = tokens[pos + 1];
        let (loss, grad) = modgrad_traits::cross_entropy_grad(&logits[pos], target);
        total_loss += loss;
        d_logits.push(grad);
    }
    // Last position has no target
    d_logits.push(vec![0.0; logits[0].len()]);

    (total_loss / (n - 1) as f32, d_logits)
}

// ═══════════════════════════════════════════════════════════════
// BACKWARD PASS
// ═══════════════════════════════════════════════════════════════

/// Gradients for FFN training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FfnGradients {
    pub embed: Vec<f32>,
    pub blocks: Vec<FfnBlockGrads>,
    pub final_ln_gamma: Vec<f32>,
    pub final_ln_beta: Vec<f32>,
    pub lm_head: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FfnAdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub wd: f32,
    pub clip: f32,
    pub step: usize,
    // Flat moment buffers — one per parameter group
    m: FfnGradients,
    v: FfnGradients,
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
        }
    }

    pub fn with_lr(mut self, lr: f32) -> Self { self.lr = lr; self }

    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
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

        // Compute grad norm for clipping
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

        // Apply to each param group
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

    if modgrad_compute::neuron::gpu_enabled() && weights.len() >= 1024 {
        // Pre-scale grads on CPU (small op), then dispatch GPU AdamW.
        let mut g_scaled: Vec<f32> = grads.iter().map(|&g| g * clip_scale).collect();
        let bc1_inv = 1.0 / bc1;
        let bc2_inv = 1.0 / bc2;
        if modgrad_device::kfd::accel::try_adamw(
            weights, &mut g_scaled, m, v,
            lr, beta1, beta2, eps, wd, bc1_inv, bc2_inv,
        ) {
            return;
        }
    }

    // CPU fallback
    for i in 0..weights.len() {
        let g = grads[i] * clip_scale;
        m[i] = beta1 * m[i] + (1.0 - beta1) * g;
        v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;
        let m_hat = m[i] / bc1;
        let v_hat = v[i] / bc2;
        weights[i] -= lr * (m_hat / (v_hat.sqrt() + eps) + wd * weights[i]);
    }
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
    let (logits, cache) = ffn_forward(w, tokens);
    let (loss, d_logits) = ffn_loss(&logits, tokens);

    // Accuracy
    let mut correct = 0usize;
    let n = tokens.len() - 1;
    for pos in 0..n {
        let pred = logits[pos].iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i).unwrap_or(0);
        if pred == tokens[pos + 1] { correct += 1; }
    }
    let acc = correct as f32 / n.max(1) as f32;

    ffn_backward(w, &cache, &d_logits, grads);
    opt.step_update(w, grads);

    (loss, acc)
}

#[cfg(test)]
mod tests {
    use super::*;

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
