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
// FORWARD PASS
// ═══════════════════════════════════════════════════════════════

/// FFN forward with cache for backward pass.
pub struct FfnCache {
    /// Per-position, per-layer: pre-norm input, post-norm, gate, up, hidden.
    pub tokens: Vec<usize>,  // input token ids
    pub embeds: Vec<Vec<f32>>,                  // [pos][d_model]
    pub per_layer: Vec<Vec<LayerCache>>,        // [pos][layer]
    pub final_normed: Vec<Vec<f32>>,            // [pos][d_model]
}

pub struct LayerCache {
    pub pre_norm: Vec<f32>,      // input to block (before norm)
    pub normed: Vec<f32>,        // after pre-norm
    pub gate_out: Vec<f32>,      // gate(normed)
    pub up_out: Vec<f32>,        // up(normed)
    pub hidden: Vec<f32>,        // silu(gate) * up
    pub mean: f32,
    pub inv_std: f32,
}

/// Forward one position, return logits for next-token prediction.
/// Input: sequence of token ids, predicts next token for each position.
pub fn ffn_forward(w: &FfnWeights, tokens: &[usize]) -> (Vec<Vec<f32>>, FfnCache) {
    let d = w.config.d_model;
    let n = tokens.len();

    let mut embeds: Vec<Vec<f32>> = Vec::with_capacity(n);
    let mut per_layer: Vec<Vec<LayerCache>> = Vec::with_capacity(n);
    let mut final_normed: Vec<Vec<f32>> = Vec::with_capacity(n);
    let mut logits: Vec<Vec<f32>> = Vec::with_capacity(n);

    for &tok in tokens {
        // Embed lookup
        let start = tok * d;
        let mut x: Vec<f32> = w.embed[start..start + d].to_vec();
        embeds.push(x.clone());

        let mut layer_caches: Vec<LayerCache> = Vec::with_capacity(w.blocks.len());

        for block in &w.blocks {
            let pre_norm = x.clone();
            let (normed, mean, inv_std) = layer_norm_forward(&x, &block.ln_gamma, &block.ln_beta);

            let gate_out = block.gate.forward(&normed);
            let up_out = block.up.forward(&normed);

            let mut hidden = vec![0.0f32; w.config.hidden];
            for i in 0..w.config.hidden {
                let s = 1.0 / (1.0 + (-gate_out[i]).exp());
                hidden[i] = gate_out[i] * s * up_out[i];
            }

            let down_out = block.down.forward(&hidden);
            for i in 0..d {
                x[i] = pre_norm[i] + down_out[i]; // residual
            }

            layer_caches.push(LayerCache {
                pre_norm, normed, gate_out, up_out, hidden, mean, inv_std,
            });
        }
        per_layer.push(layer_caches);

        // Final layer norm
        let (normed, _, _) = layer_norm_forward(&x, &w.final_ln_gamma, &w.final_ln_beta);
        final_normed.push(normed.clone());

        // LM head
        let mut l = vec![0.0f32; w.config.vocab];
        for v in 0..w.config.vocab {
            let row_start = v * d;
            let mut s = 0.0f32;
            for j in 0..d {
                s += w.lm_head[row_start + j] * normed[j];
            }
            l[v] = s;
        }
        logits.push(l);
    }

    let cache = FfnCache {
        tokens: tokens.to_vec(),
        embeds,
        per_layer,
        final_normed,
    };

    (logits, cache)
}

fn layer_norm_forward(x: &[f32], gamma: &[f32], beta: &[f32]) -> (Vec<f32>, f32, f32) {
    let n = x.len() as f32;
    let mean: f32 = x.iter().sum::<f32>() / n;
    let var: f32 = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    let inv_std = 1.0 / (var + 1e-5).sqrt();
    let out: Vec<f32> = x.iter().zip(gamma.iter()).zip(beta.iter())
        .map(|((&xi, &g), &b)| g * (xi - mean) * inv_std + b).collect();
    (out, mean, inv_std)
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

/// GPU-accelerated outer product: `dw[i*in_dim + j] += d_out[i] * input[j]`.
/// Falls back to CPU if GPU not available.
#[inline]
fn accumulate_outer(dw: &mut [f32], d_out: &[f32], input: &[f32], out_dim: usize, in_dim: usize) {
    if modgrad_compute::neuron::gpu_enabled()
        && modgrad_device::kfd::accel::try_outer_product(
            dw, d_out, input, out_dim as u32, in_dim as u32)
    {
        return;
    }
    // CPU fallback
    for i in 0..out_dim {
        let gi = d_out[i];
        let row = i * in_dim;
        for j in 0..in_dim {
            dw[row + j] += gi * input[j];
        }
    }
}

/// GPU-accelerated transposed matvec: `d_in[j] += Σ W[i*in_dim + j] * d_out[i]`.
/// Falls back to CPU if GPU not available.
#[inline]
fn matvec_t_accum(d_in: &mut [f32], weight: &[f32], d_out: &[f32], out_dim: usize, in_dim: usize) {
    let mut d_in_tmp = vec![0.0f32; in_dim];
    if modgrad_compute::neuron::gpu_enabled()
        && modgrad_device::kfd::accel::try_matvec_t(
            d_out, weight, &mut d_in_tmp, out_dim as u32, in_dim as u32)
    {
        for j in 0..in_dim { d_in[j] += d_in_tmp[j]; }
        return;
    }
    // CPU fallback
    for i in 0..out_dim {
        let gi = d_out[i];
        let row = i * in_dim;
        for j in 0..in_dim {
            d_in[j] += gi * weight[row + j];
        }
    }
}

/// Backward pass through FFN, accumulating into `grads`.
/// Uses GPU kernels for the expensive matmul-shaped operations.
pub fn ffn_backward(
    w: &FfnWeights,
    cache: &FfnCache,
    d_logits: &[Vec<f32>],
    grads: &mut FfnGradients,
) {
    let d = w.config.d_model;
    let h = w.config.hidden;
    let vocab = w.config.vocab;
    let n = cache.tokens.len();

    for pos in 0..n {
        let dl = &d_logits[pos];
        let normed = &cache.final_normed[pos];

        // lm_head backward (largest matmul: vocab × d_model)
        // d_lm_head[v*d + j] += d_logits[v] * normed[j]
        // d_normed[j] += Σ_v d_logits[v] * lm_head[v*d + j]
        accumulate_outer(&mut grads.lm_head, dl, normed, vocab, d);
        let mut d_normed = vec![0.0f32; d];
        matvec_t_accum(&mut d_normed, &w.lm_head, dl, vocab, d);

        // Reconstruct final x (pre final-ln) from last block cache
        let final_ln_input: Vec<f32> = {
            let last_block = cache.per_layer[pos].last().unwrap();
            let down_out = w.blocks.last().unwrap().down.forward(&last_block.hidden);
            (0..d).map(|i| last_block.pre_norm[i] + down_out[i]).collect()
        };

        let (d_final_x, d_gamma, d_beta) = layer_norm_backward(
            &d_normed, &final_ln_input, &w.final_ln_gamma,
        );
        for i in 0..d {
            grads.final_ln_gamma[i] += d_gamma[i];
            grads.final_ln_beta[i] += d_beta[i];
        }

        let mut d_x = d_final_x;
        for (layer_idx, block) in w.blocks.iter().enumerate().rev() {
            let lc = &cache.per_layer[pos][layer_idx];
            let bg = &mut grads.blocks[layer_idx];

            // Residual: d_x flows to both pre_norm and down_out
            let d_down_out = d_x.clone();
            let mut d_pre_norm: Vec<f32> = d_x.clone();

            // down backward: down: hidden → d_model
            // d_down_w[i*h + j] += d_down_out[i] * hidden[j]
            // d_hidden[j] += Σ_i d_down_out[i] * down.w[i*h + j]
            for i in 0..d { bg.down_b[i] += d_down_out[i]; }
            accumulate_outer(&mut bg.down_w, &d_down_out, &lc.hidden, d, h);
            let mut d_hidden = vec![0.0f32; h];
            matvec_t_accum(&mut d_hidden, &block.down.weight, &d_down_out, d, h);

            // SwiGLU backward: hidden = silu(gate) * up
            let mut d_gate_out = vec![0.0f32; h];
            let mut d_up_out = vec![0.0f32; h];
            for j in 0..h {
                let g = lc.gate_out[j];
                let s = 1.0 / (1.0 + (-g).exp());
                let silu = g * s;
                let d_silu_dg = s + g * s * (1.0 - s);
                d_gate_out[j] = d_hidden[j] * lc.up_out[j] * d_silu_dg;
                d_up_out[j] = d_hidden[j] * silu;
            }

            // up backward: up: d_model → hidden
            for i in 0..h { bg.up_b[i] += d_up_out[i]; }
            accumulate_outer(&mut bg.up_w, &d_up_out, &lc.normed, h, d);
            let mut d_normed = vec![0.0f32; d];
            matvec_t_accum(&mut d_normed, &block.up.weight, &d_up_out, h, d);

            // gate backward
            for i in 0..h { bg.gate_b[i] += d_gate_out[i]; }
            accumulate_outer(&mut bg.gate_w, &d_gate_out, &lc.normed, h, d);
            matvec_t_accum(&mut d_normed, &block.gate.weight, &d_gate_out, h, d);

            // Layer norm backward
            let (d_ln_input, d_ln_g, d_ln_b) = layer_norm_backward(
                &d_normed, &lc.pre_norm, &block.ln_gamma,
            );
            for i in 0..d {
                bg.ln_gamma[i] += d_ln_g[i];
                bg.ln_beta[i] += d_ln_b[i];
                d_pre_norm[i] += d_ln_input[i];
            }

            d_x = d_pre_norm;
        }

        // Embed backward: sparse — only the input token's row
        let tok = cache.tokens[pos];
        let start = tok * d;
        for i in 0..d {
            grads.embed[start + i] += d_x[i];
        }
    }
}

fn layer_norm_backward(d_out: &[f32], x: &[f32], gamma: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = x.len();
    let nf = n as f32;
    let mean: f32 = x.iter().sum::<f32>() / nf;
    let var: f32 = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / nf;
    let inv_std = 1.0 / (var + 1e-5).sqrt();

    let mut d_gamma = vec![0.0f32; n];
    let mut d_beta = vec![0.0f32; n];
    let mut d_x_hat = vec![0.0f32; n];

    for i in 0..n {
        let x_hat = (x[i] - mean) * inv_std;
        d_gamma[i] = d_out[i] * x_hat;
        d_beta[i] = d_out[i];
        d_x_hat[i] = d_out[i] * gamma[i];
    }

    let sum_dxh: f32 = d_x_hat.iter().sum();
    let sum_dxh_xh: f32 = (0..n).map(|i| d_x_hat[i] * (x[i] - mean) * inv_std).sum();

    let mut d_x = vec![0.0f32; n];
    for i in 0..n {
        let x_hat = (x[i] - mean) * inv_std;
        d_x[i] = inv_std / nf * (nf * d_x_hat[i] - sum_dxh - x_hat * sum_dxh_xh);
    }

    (d_x, d_gamma, d_beta)
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
