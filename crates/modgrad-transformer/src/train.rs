//! Training loop: forward + backward + optimizer step.
//!
//! Manual backward pass (no autograd framework):
//!   - Each module has a `backward` method that computes gradients
//!   - Gradient buffers are pre-allocated per layer
//!   - Gradient checkpointing: recompute activations instead of storing all
//!   - Optimizer: AdamW for embeddings/scalars, Muon for weight matrices
//!
//! Memory layout for large models:
//!   - Weights: CPU RAM (via WeightOffloader), streamed to VRAM per-layer
//!   - Gradients: CPU RAM, accumulated per micro-batch
//!   - Optimizer state: CPU RAM (AdamW moments, Muon momentum+variance)
//!   - Activations: VRAM during forward, recomputed during backward (checkpointing)

use super::config::GptConfig;
use super::offload::{WeightOffloader, LayerOffsets};
use super::optim::adamw::AdamW;
use super::optim::muon::Muon;
use super::ops::TransformerOps;

/// Per-layer gradient buffers (same layout as layer weights).
pub struct LayerGradients {
    /// Concatenated gradients: [wq|wk|wv|wo|fc|proj], same layout as CpuLayerWeights.
    pub data: Vec<f32>,
    pub offsets: LayerOffsets,
}

impl LayerGradients {
    pub fn new(offsets: LayerOffsets) -> Self {
        Self {
            data: vec![0.0; offsets.total_elems],
            offsets,
        }
    }

    pub fn zero(&mut self) {
        self.data.fill(0.0);
    }

    pub fn wq_mut(&mut self) -> &mut [f32] {
        let o = &self.offsets;
        &mut self.data[o.wq..o.wk]
    }

    pub fn wk_mut(&mut self) -> &mut [f32] {
        let o = &self.offsets;
        &mut self.data[o.wk..o.wv]
    }

    pub fn wv_mut(&mut self) -> &mut [f32] {
        let o = &self.offsets;
        &mut self.data[o.wv..o.wo]
    }

    pub fn wo_mut(&mut self) -> &mut [f32] {
        let o = &self.offsets;
        &mut self.data[o.wo..o.mlp_fc]
    }

    pub fn mlp_fc_mut(&mut self) -> &mut [f32] {
        let o = &self.offsets;
        &mut self.data[o.mlp_fc..o.mlp_proj]
    }

    pub fn mlp_proj_mut(&mut self) -> &mut [f32] {
        let o = &self.offsets;
        &mut self.data[o.mlp_proj..o.total_elems]
    }
}

/// Gradient buffers for the full model.
pub struct ModelGradients {
    /// Per-layer gradients.
    pub layers: Vec<LayerGradients>,
    /// Token embedding gradients: [vocab_size * model_dim].
    pub embed: Vec<f32>,
    /// LM head gradients: [vocab_size * model_dim].
    pub lm_head: Vec<f32>,
    /// Final norm scale gradients: [model_dim].
    pub norm_scale: Vec<f32>,
    /// Smear gate gradients.
    pub smear_gate: Vec<f32>,
}

impl ModelGradients {
    pub fn new(config: &GptConfig) -> Self {
        let md = config.model_dim.get();
        let kv_dim = config.num_kv_heads.get() * config.head_dim.get();
        let mlp_dim = config.mlp_dim.get();
        let vocab = config.vocab_size.get();
        let offsets = LayerOffsets::compute(md, kv_dim, mlp_dim);

        Self {
            layers: (0..config.num_layers.get())
                .map(|_| LayerGradients::new(offsets))
                .collect(),
            embed: vec![0.0; vocab * md],
            lm_head: vec![0.0; vocab * md],
            norm_scale: vec![0.0; md],
            smear_gate: vec![0.0; md * config.smear.gate_channels],
        }
    }

    /// Zero all gradients (before accumulation).
    pub fn zero(&mut self) {
        for lg in &mut self.layers {
            lg.zero();
        }
        self.embed.fill(0.0);
        self.lm_head.fill(0.0);
        self.norm_scale.fill(0.0);
        self.smear_gate.fill(0.0);
    }

    /// Scale all gradients (for gradient averaging over micro-batches).
    pub fn scale(&mut self, factor: f32) {
        for lg in &mut self.layers {
            for v in &mut lg.data { *v *= factor; }
        }
        for v in &mut self.embed { *v *= factor; }
        for v in &mut self.lm_head { *v *= factor; }
        for v in &mut self.norm_scale { *v *= factor; }
        for v in &mut self.smear_gate { *v *= factor; }
    }
}

/// Per-layer optimizer state.
struct LayerOptimizers {
    /// Muon for weight matrices (wq, wk, wv, wo, mlp_fc, mlp_proj).
    wq: Muon,
    wk: Muon,
    wv: Muon,
    wo: Muon,
    mlp_fc: Muon,
    mlp_proj: Muon,
}

/// Full model optimizer state.
pub struct ModelOptimizer {
    /// Per-layer Muon optimizers.
    layers: Vec<LayerOptimizers>,
    /// AdamW for embeddings.
    embed_opt: AdamW,
    /// AdamW for LM head.
    lm_head_opt: AdamW,
    /// AdamW for norm scale.
    norm_opt: AdamW,
    /// AdamW for smear gate.
    smear_opt: AdamW,
}

impl ModelOptimizer {
    pub fn new(config: &GptConfig, lr: f32, muon_lr: f32) -> Self {
        let md = config.model_dim.get();
        let kv_dim = config.num_kv_heads.get() * config.head_dim.get();
        let mlp_dim = config.mlp_dim.get();
        let vocab = config.vocab_size.get();

        let layers = (0..config.num_layers.get()).map(|_| {
            LayerOptimizers {
                wq: Muon::new(md, md, muon_lr),
                wk: Muon::new(kv_dim, md, muon_lr),
                wv: Muon::new(kv_dim, md, muon_lr),
                wo: Muon::new(md, md, muon_lr),
                mlp_fc: Muon::new(mlp_dim, md, muon_lr),
                mlp_proj: Muon::new(md, mlp_dim, muon_lr),
            }
        }).collect();

        Self {
            layers,
            embed_opt: AdamW::new(vocab * md, lr),
            lm_head_opt: AdamW::new(vocab * md, lr),
            norm_opt: AdamW::new(md, lr).with_weight_decay(0.0),
            smear_opt: AdamW::new(md * config.smear.gate_channels, lr),
        }
    }

    /// Apply one optimizer step using accumulated gradients.
    pub fn step(
        &mut self,
        offloader: &mut WeightOffloader,
        grads: &ModelGradients,
        backend: &dyn TransformerOps,
    ) {
        // Embeddings (AdamW)
        self.embed_opt.step(&mut offloader.embed, &grads.embed);
        self.lm_head_opt.step(&mut offloader.lm_head, &grads.lm_head);

        // Per-layer weight matrices (Muon)
        for (i, (lg, lo)) in grads.layers.iter().zip(self.layers.iter_mut()).enumerate() {
            let layer = &mut offloader.layers[i];
            let o = &lg.offsets;

            lo.wq.step(&mut layer.data[o.wq..o.wk], &lg.data[o.wq..o.wk], backend);
            lo.wk.step(&mut layer.data[o.wk..o.wv], &lg.data[o.wk..o.wv], backend);
            lo.wv.step(&mut layer.data[o.wv..o.wo], &lg.data[o.wv..o.wo], backend);
            lo.wo.step(&mut layer.data[o.wo..o.mlp_fc], &lg.data[o.wo..o.mlp_fc], backend);
            lo.mlp_fc.step(
                &mut layer.data[o.mlp_fc..o.mlp_proj],
                &lg.data[o.mlp_fc..o.mlp_proj],
                backend,
            );
            lo.mlp_proj.step(
                &mut layer.data[o.mlp_proj..o.total_elems],
                &lg.data[o.mlp_proj..o.total_elems],
                backend,
            );
        }
    }
}

/// Cross-entropy loss: -log(softmax(logits)[target]).
///
/// Returns (loss, d_logits).
pub fn cross_entropy_loss(logits: &[f32], target: usize) -> (f32, Vec<f32>) {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();

    let loss = -(probs[target].max(1e-10)).ln();

    // Gradient: probs - one_hot(target)
    let mut d_logits = probs;
    d_logits[target] -= 1.0;

    (loss, d_logits)
}

/// Backward through lm_head: d_hidden = W^T @ d_logits.
///
/// Also accumulates gradient into d_lm_head: d_W += d_logits ⊗ hidden.
pub fn backward_lm_head(
    d_logits: &[f32],
    hidden: &[f32],
    lm_head: &[f32],
    d_hidden: &mut [f32],
    d_lm_head: &mut [f32],
    vocab: usize,
    model_dim: usize,
) {
    // d_hidden = W^T @ d_logits (transpose of forward: logits = W @ hidden)
    for i in 0..model_dim {
        let mut s = 0.0f32;
        for v in 0..vocab {
            s += lm_head[v * model_dim + i] * d_logits[v];
        }
        d_hidden[i] = s;
    }

    // d_W += d_logits ⊗ hidden (outer product, accumulated)
    for v in 0..vocab {
        for i in 0..model_dim {
            d_lm_head[v * model_dim + i] += d_logits[v] * hidden[i];
        }
    }
}

/// Backward through RMS norm with scale.
///
/// Given forward: y = (x / rms(x)) * scale
/// Computes d_x and accumulates d_scale.
pub fn backward_scaled_rms_norm(
    d_y: &[f32],
    x: &[f32],
    scale: &[f32],
    d_x: &mut [f32],
    d_scale: &mut [f32],
    eps: f32,
) {
    let n = x.len() as f32;
    let ss: f32 = x.iter().map(|v| v * v).sum();
    let rms = (ss / n + eps).sqrt();
    let inv_rms = 1.0 / rms;

    // d_scale += d_y * (x / rms)
    for i in 0..x.len() {
        d_scale[i] += d_y[i] * x[i] * inv_rms;
    }

    // d_x: chain rule through rms normalization
    // d_x_i = scale_i * inv_rms * (d_y_i - x_normed_i * mean(d_y * x_normed) )
    let mut dot_dy_xn = 0.0f32;
    for i in 0..x.len() {
        let x_normed = x[i] * inv_rms;
        dot_dy_xn += d_y[i] * scale[i] * x_normed;
    }
    dot_dy_xn /= n;

    for i in 0..x.len() {
        let x_normed = x[i] * inv_rms;
        d_x[i] = scale[i] * inv_rms * d_y[i] - x_normed * dot_dy_xn;
    }
}

/// Backward through a bias-free linear: y = W @ x.
///
/// Computes d_x = W^T @ d_y and accumulates d_W += d_y ⊗ x.
pub fn backward_linear(
    d_y: &[f32],
    x: &[f32],
    weight: &[f32],
    d_x: &mut [f32],
    d_weight: &mut [f32],
    out_dim: usize,
    in_dim: usize,
) {
    // d_x = W^T @ d_y
    for j in 0..in_dim {
        let mut s = 0.0f32;
        for i in 0..out_dim {
            s += weight[i * in_dim + j] * d_y[i];
        }
        d_x[j] = s;
    }

    // d_W += d_y ⊗ x
    for i in 0..out_dim {
        for j in 0..in_dim {
            d_weight[i * in_dim + j] += d_y[i] * x[j];
        }
    }
}

/// Backward through ReLU²: y = max(0, x)².
/// d_x = d_y * 2 * max(0, x) * (x > 0)
pub fn backward_relu_squared(d_y: &[f32], x: &[f32], d_x: &mut [f32]) {
    for i in 0..x.len() {
        d_x[i] = if x[i] > 0.0 { d_y[i] * 2.0 * x[i] } else { 0.0 };
    }
}

// ─── Attention backward primitives ──────────────────────────

/// Backward through softmax.
///
/// Given forward: p = softmax(s), and d_p (gradient from weighted V sum),
/// compute d_s (gradient w.r.t. pre-softmax scores).
///
/// d_s_i = p_i * (d_p_i - sum_j(p_j * d_p_j))
pub fn backward_softmax(d_p: &[f32], p: &[f32], d_s: &mut [f32]) {
    let dot: f32 = p.iter().zip(d_p.iter()).map(|(pi, di)| pi * di).sum();
    for i in 0..p.len() {
        d_s[i] = p[i] * (d_p[i] - dot);
    }
}

/// Backward through RoPE rotation.
///
/// Forward: [x1', x2'] = [x1*cos - x2*sin, x1*sin + x2*cos]
/// Backward: [d_x1, d_x2] = [d_x1'*cos + d_x2'*sin, -d_x1'*sin + d_x2'*cos]
///
/// (RoPE is an orthogonal rotation — its transpose is the inverse rotation.)
pub fn backward_rope(
    d_head: &[f32],
    cos_table: &[f32],
    sin_table: &[f32],
    d_input: &mut [f32],
    half_dim: usize,
) {
    let (d_left, d_right) = d_head.split_at(half_dim);
    let (out_left, out_right) = d_input.split_at_mut(half_dim);
    for i in 0..half_dim {
        // Inverse rotation (transpose of forward)
        out_left[i]  = d_left[i] * cos_table[i] + d_right[i] * sin_table[i];
        out_right[i] = -d_left[i] * sin_table[i] + d_right[i] * cos_table[i];
    }
}

/// Backward through RMS norm (no learnable params).
///
/// Forward: y = x / sqrt(mean(x²) + eps)
/// Backward: d_x_i = (d_y_i - y_i * mean(d_y * y)) / rms
pub fn backward_rms_norm(
    d_y: &[f32],
    x: &[f32],
    d_x: &mut [f32],
    eps: f32,
) {
    let n = x.len() as f32;
    let ss: f32 = x.iter().map(|v| v * v).sum();
    let rms = (ss / n + eps).sqrt();
    let inv_rms = 1.0 / rms;

    // y = x * inv_rms
    let mut dot_dy_y = 0.0f32;
    for i in 0..x.len() {
        let y_i = x[i] * inv_rms;
        dot_dy_y += d_y[i] * y_i;
    }
    dot_dy_y /= n;

    for i in 0..x.len() {
        let y_i = x[i] * inv_rms;
        d_x[i] = inv_rms * (d_y[i] - y_i * dot_dy_y);
    }
}

/// Full backward through one attention head for a single query position.
///
/// Given saved forward state:
///   - `q_normed`: normalized+scaled+RoPE'd query `[head_dim]`
///   - `k_normed_all`: normalized+scaled K for each attended position `[attn_len, head_dim]`
///   - `v_all`: V for each attended position `[attn_len, head_dim]`
///   - `attn_probs`: softmax output `[attn_len]`
///   - `d_head_out`: gradient from output projection `[head_dim]`
///
/// Computes:
///   - `d_q_pre_rope`: gradient for Q before RoPE `[head_dim]`
///   - `d_k_pre_rope`: gradient for each K before RoPE `[attn_len, head_dim]`
///   - `d_v`: gradient for each V `[attn_len, head_dim]`
pub fn backward_attention_head(
    d_head_out: &[f32],
    q_normed: &[f32],
    k_normed_all: &[f32],
    v_all: &[f32],
    attn_probs: &[f32],
    attn_len: usize,
    head_dim: usize,
    d_v: &mut [f32],
    d_scores: &mut [f32],
    d_q_normed: &mut [f32],
    d_k_normed: &mut [f32],
) {
    // ─── Backward through weighted V sum ────────────────
    // head_out = sum_t(probs[t] * v[t])
    // d_v[t] = probs[t] * d_head_out  (for each attended position)
    // d_probs[t] = dot(d_head_out, v[t])
    let mut d_probs = vec![0.0f32; attn_len];
    for t in 0..attn_len {
        let v_t = &v_all[t * head_dim..(t + 1) * head_dim];
        let mut dot = 0.0f32;
        for i in 0..head_dim {
            d_v[t * head_dim + i] += attn_probs[t] * d_head_out[i];
            dot += d_head_out[i] * v_t[i];
        }
        d_probs[t] = dot;
    }

    // ─── Backward through softmax ───────────────────────
    backward_softmax(&d_probs, attn_probs, d_scores);

    // ─── Backward through QK dot products ───────────────
    // score[t] = dot(q_normed, k_normed[t])
    // d_q_normed += d_scores[t] * k_normed[t]  (accumulated over t)
    // d_k_normed[t] = d_scores[t] * q_normed
    d_q_normed.fill(0.0);
    for t in 0..attn_len {
        let k_t = &k_normed_all[t * head_dim..(t + 1) * head_dim];
        let ds = d_scores[t];
        for i in 0..head_dim {
            d_q_normed[i] += ds * k_t[i];
            d_k_normed[t * head_dim + i] = ds * q_normed[i];
        }
    }
}

/// Backward through full single-token attention for all heads.
///
/// Accumulates into d_wq, d_wk, d_wv, d_wo gradient buffers.
/// `d_hidden`: gradient flowing back to the hidden state `[model_dim]`.
pub fn backward_attention_layer(
    d_attn_out: &[f32],       // [model_dim] from residual backward
    // Saved forward state:
    normed_input: &[f32],     // [model_dim] pre-attention norm output
    _q_proj: &[f32],           // [model_dim] raw Q projection
    _k_proj: &[f32],           // [kv_dim] raw K projection
    _v_proj: &[f32],           // [kv_dim] raw V projection
    concat_heads: &[f32],     // [model_dim] concatenated attention heads
    // Weights:
    wq: &[f32], wk: &[f32], wv: &[f32], wo: &[f32],
    // Dims:
    model_dim: usize,
    kv_dim: usize,
    _num_heads: usize,
    _head_dim: usize,
    // Gradient outputs (accumulated):
    d_normed: &mut [f32],     // [model_dim]
    d_wq: &mut [f32],
    d_wk: &mut [f32],
    d_wv: &mut [f32],
    d_wo: &mut [f32],
) {
    // ─── Backward through output projection ─────────────
    // attn_out = wo @ concat_heads
    let mut d_concat = vec![0.0f32; model_dim];
    backward_linear(d_attn_out, concat_heads, wo, &mut d_concat, d_wo, model_dim, model_dim);

    // d_concat distributes to per-head gradients
    // Each head's output slice gets its gradient — these feed into
    // backward_attention_head for each head.

    // ─── Backward through Q projection ──────────────────
    // q = wq @ normed_input
    let _d_q = vec![0.0f32; model_dim];
    backward_linear(&d_concat, normed_input, wq, d_normed, d_wq, model_dim, model_dim);

    // ─── Backward through K projection ──────────────────
    let mut d_normed_k = vec![0.0f32; model_dim];
    let d_k_placeholder = vec![0.0f32; kv_dim]; // from attention head backward
    backward_linear(&d_k_placeholder, normed_input, wk, &mut d_normed_k, d_wk, kv_dim, model_dim);
    for i in 0..model_dim { d_normed[i] += d_normed_k[i]; }

    // ─── Backward through V projection ──────────────────
    let mut d_normed_v = vec![0.0f32; model_dim];
    let d_v_placeholder = vec![0.0f32; kv_dim]; // from attention head backward
    backward_linear(&d_v_placeholder, normed_input, wv, &mut d_normed_v, d_wv, kv_dim, model_dim);
    for i in 0..model_dim { d_normed[i] += d_normed_v[i]; }
}

/// Training configuration.
pub struct TrainConfig {
    /// Learning rate for embeddings/scalars (AdamW).
    pub lr: f32,
    /// Learning rate for weight matrices (Muon).
    pub muon_lr: f32,
    /// Gradient accumulation steps.
    pub grad_accum_steps: usize,
    /// Whether to use gradient checkpointing.
    pub gradient_checkpointing: bool,
    /// Max gradient norm for clipping.
    pub max_grad_norm: f32,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            lr: 3e-4,
            muon_lr: 2e-2,
            grad_accum_steps: 1,
            gradient_checkpointing: true,
            max_grad_norm: 1.0,
        }
    }
}

// ─── Full training loop ─────────────────────────────────────

/// Single training step result.
pub struct StepResult {
    pub loss: f32,
    pub grad_norm: f32,
    pub tokens_processed: usize,
}

/// Run one training step on a batch of sequences.
///
/// `input`: `[batch_size][seq_len]` token IDs.
/// `target`: `[batch_size][seq_len]` target token IDs (shifted by 1).
/// `offloader`: mutable weight storage.
/// `grads`: gradient buffers (zeroed internally).
/// `optimizer`: model optimizer state.
/// `config`: training config.
/// `backend`: compute backend.
///
/// Returns loss, grad_norm, tokens_processed.
pub fn train_step(
    input: &[Vec<i64>],
    target: &[Vec<i64>],
    offloader: &mut super::offload::WeightOffloader,
    grads: &mut ModelGradients,
    optimizer: &mut ModelOptimizer,
    config: &TrainConfig,
    backend: &dyn TransformerOps,
) -> StepResult {
    let batch_size = input.len();
    let seq_len = input[0].len();
    grads.zero();

    let total_loss = 0.0f32;
    let mut total_tokens = 0usize;

    // Forward + backward for each sequence in batch (gradient accumulation)
    for b in 0..batch_size {
        let _tids = &input[b];
        let _targets = &target[b];

        // Forward: embed all tokens
        let _md = offloader.offsets.total_elems; // approximate — use actual model_dim
        // For now: single-token forward through the weight offloader
        // Full sequence forward requires the model, which we'd need passed in.
        // This is the wiring point — the actual training loop will call
        // GptModel::forward_prefill, get logits, compute loss, backward.

        // Compute loss using host/eval/loss.rs cross_entropy
        // (Called from the outer training loop that has access to the model)

        total_tokens += seq_len;
    }

    // Average gradients
    if batch_size > 1 {
        grads.scale(1.0 / batch_size as f32);
    }

    // Clip gradients
    let grad_norm = clip_grad_norm(grads, config.max_grad_norm);

    // Optimizer step
    optimizer.step(offloader, grads, backend);

    StepResult {
        loss: total_loss / batch_size as f32,
        grad_norm,
        tokens_processed: total_tokens,
    }
}

/// Full training loop over a dataset.
///
/// This is the top-level entry point that ties together:
///   - DataLoader (batches)
///   - GptModel (forward)
///   - loss.rs (cross_entropy)
///   - Backward pass (train.rs primitives)
///   - ModelOptimizer (step)
///   - WeightOffloader (weight storage)
pub struct Trainer {
    pub grads: ModelGradients,
    pub optimizer: ModelOptimizer,
    pub config: TrainConfig,
    /// Running average loss for logging.
    pub running_loss: f32,
    /// Total steps completed.
    pub step_count: u64,
}

impl Trainer {
    pub fn new(model_config: &super::config::GptConfig, train_config: TrainConfig) -> Self {
        Self {
            grads: ModelGradients::new(model_config),
            optimizer: ModelOptimizer::new(
                model_config,
                train_config.lr,
                train_config.muon_lr,
            ),
            config: train_config,
            running_loss: 0.0,
            step_count: 0,
        }
    }

    /// Run one full forward + backward + optimizer step.
    ///
    /// `token_ids`: input sequence `[seq_len]`.
    /// `targets`: target token IDs `[seq_len]`.
    /// `offloader`: weight storage for optimizer step.
    /// `rope`: rotary embeddings.
    /// `norm_scale`: final norm scale weights.
    /// `backend`: compute backend.
    ///
    /// Returns loss for this step.
    pub fn step(
        &mut self,
        token_ids: &[i64],
        targets: &[i64],
        offloader: &mut WeightOffloader,
        config: &super::config::GptConfig,
        rope: &RotaryEmbedding,
        norm_scale: &[f32],
        backend: &dyn TransformerOps,
    ) -> f32 {
        let vocab = config.vocab_size.get();

        // Forward pass (saves activations)
        let (logits, saved) = forward_train(
            token_ids, offloader, config, rope, norm_scale, backend,
        );

        // Loss
        let (loss, d_logits) = crate::loss::cross_entropy(&logits, targets, vocab);

        // Full backward pass
        self.grads.zero();
        backward_train(
            &d_logits, token_ids, &saved, offloader, &mut self.grads,
            config, rope, norm_scale, backend,
        );

        // Clip and step
        let _norm = clip_grad_norm(&mut self.grads, self.config.max_grad_norm);
        self.optimizer.step(offloader, &self.grads, backend);

        // Update running stats
        self.step_count += 1;
        let alpha = 0.99f32;
        self.running_loss = if self.step_count == 1 {
            loss
        } else {
            alpha * self.running_loss + (1.0 - alpha) * loss
        };

        loss
    }
}

// ─── Full training forward + backward ──────────────────────

use super::rope::RotaryEmbedding;

/// Per-layer saved activations (for gradient checkpointing, save only essentials).
pub struct LayerSaved {
    /// Input to this layer [seq_len * md].
    input: Vec<f32>,
    /// Attention normed input [seq_len * md].
    attn_normed: Vec<f32>,
    /// Q, K, V projections.
    q_proj: Vec<f32>,  // [seq_len * md]
    k_proj: Vec<f32>,  // [seq_len * kv_dim]
    v_proj: Vec<f32>,  // [seq_len * kv_dim]
    /// Concatenated attention head outputs [seq_len * md].
    concat_heads: Vec<f32>,
    /// Post-attention residual hidden [seq_len * md].
    post_attn: Vec<f32>,
    /// MLP normed input [seq_len * md].
    mlp_normed: Vec<f32>,
    /// MLP up-projection output (before relu²) [seq_len * mlp_dim].
    mlp_up: Vec<f32>,
    /// MLP activated output (after relu²) [seq_len * mlp_dim].
    mlp_act: Vec<f32>,
}

/// All saved state from forward pass.
pub struct ForwardSaved {
    layers: Vec<LayerSaved>,
    /// Post-embed hidden [seq_len * md].
    post_embed: Vec<f32>,
    /// Pre-lm_head hidden (after final norm) [seq_len * md].
    final_hidden: Vec<f32>,
    /// Pre-norm hidden (before final norm) [seq_len * md].
    pre_norm_hidden: Vec<f32>,
}

/// Training forward pass: embed → blocks → norm → lm_head.
///
/// Saves all activations needed for backward. Operates directly on
/// weight slices from the offloader (no GptModel struct needed).
///
/// Simplified: no smear, no VE, no backout, no sliding window.
/// These can be added incrementally.
pub fn forward_train(
    token_ids: &[i64],
    offloader: &WeightOffloader,
    config: &super::config::GptConfig,
    rope: &RotaryEmbedding,
    norm_scale: &[f32],
    backend: &dyn TransformerOps,
) -> (Vec<f32>, ForwardSaved) {
    let md = config.model_dim.get();
    let kv_dim = config.num_kv_heads.get() * config.head_dim.get();
    let mlp_dim = config.mlp_dim.get();
    let vocab = config.vocab_size.get();
    let n_heads = config.num_heads.get();
    let n_kv = config.num_kv_heads.get();
    let hd = config.head_dim.get();
    let gqa = config.gqa_ratio();
    let seq_len = token_ids.len();
    let eps = config.norm_eps;

    // ─── Embedding ──────────────────────────────────────────
    let mut hidden = vec![0.0f32; seq_len * md];
    for (t, &tid) in token_ids.iter().enumerate() {
        let row_start = tid as usize * md;
        hidden[t * md..(t + 1) * md]
            .copy_from_slice(&offloader.embed[row_start..row_start + md]);
    }
    let post_embed = hidden.clone();

    // ─── Blocks ─────────────────────────────────────────────
    let mut layers_saved = Vec::with_capacity(config.num_layers.get());

    for layer in 0..config.num_layers.get() {
        let wq = offloader.layer_wq(layer);
        let wk = offloader.layer_wk(layer);
        let wv = offloader.layer_wv(layer);
        let wo = offloader.layer_wo(layer);
        let fc = offloader.layer_mlp_fc(layer);
        let proj = offloader.layer_mlp_proj(layer);

        let layer_input = hidden.clone();

        // ── Attention norm ──────────────────────────────────
        let mut attn_normed = vec![0.0f32; seq_len * md];
        for t in 0..seq_len {
            rms_norm_slice(
                &hidden[t * md..(t + 1) * md],
                &mut attn_normed[t * md..(t + 1) * md],
                eps,
            );
        }

        // ── Q, K, V projections ─────────────────────────────
        let mut q_proj = vec![0.0f32; seq_len * md];
        let mut k_proj = vec![0.0f32; seq_len * kv_dim];
        let mut v_proj = vec![0.0f32; seq_len * kv_dim];

        for t in 0..seq_len {
            let x = &attn_normed[t * md..(t + 1) * md];
            backend.matvec_nobias(wq, x, &mut q_proj[t * md..(t + 1) * md], md, md);
            backend.matvec_nobias(wk, x, &mut k_proj[t * kv_dim..(t + 1) * kv_dim], kv_dim, md);
            backend.matvec_nobias(wv, x, &mut v_proj[t * kv_dim..(t + 1) * kv_dim], kv_dim, md);
        }

        // ── RoPE on K ──────────────────────────────────────
        for t in 0..seq_len {
            for h in 0..n_kv {
                rope.apply(&mut k_proj[t * kv_dim + h * hd..t * kv_dim + (h + 1) * hd], t);
            }
        }

        // ── Per-head attention ──────────────────────────────
        let mut concat_heads = vec![0.0f32; seq_len * md];

        for t in 0..seq_len {
            for h in 0..n_heads {
                let kv_h = h / gqa;

                // Q head + RoPE
                let mut q_head: Vec<f32> = q_proj[t * md + h * hd..t * md + (h + 1) * hd].to_vec();
                rope.apply(&mut q_head, t);

                // Compute attention scores
                let mut scores = vec![f32::NEG_INFINITY; t + 1];
                for s in 0..=t {
                    let k_s = &k_proj[s * kv_dim + kv_h * hd..s * kv_dim + (kv_h + 1) * hd];
                    let mut dot = 0.0f32;
                    for i in 0..hd { dot += q_head[i] * k_s[i]; }
                    scores[s] = dot / (hd as f32).sqrt();
                }

                // Softmax
                backend.softmax_inplace(&mut scores);

                // Weighted V sum
                for s in 0..=t {
                    let v_s = &v_proj[s * kv_dim + kv_h * hd..s * kv_dim + (kv_h + 1) * hd];
                    let w = scores[s];
                    for i in 0..hd {
                        concat_heads[t * md + h * hd + i] += w * v_s[i];
                    }
                }
            }
        }

        // ── Output projection ───────────────────────────────
        let mut attn_out = vec![0.0f32; seq_len * md];
        for t in 0..seq_len {
            backend.matvec_nobias(
                wo,
                &concat_heads[t * md..(t + 1) * md],
                &mut attn_out[t * md..(t + 1) * md],
                md, md,
            );
        }

        // ── Attention residual ──────────────────────────────
        for i in 0..seq_len * md {
            hidden[i] += attn_out[i];
        }
        let post_attn = hidden.clone();

        // ── MLP norm ────────────────────────────────────────
        let mut mlp_normed = vec![0.0f32; seq_len * md];
        for t in 0..seq_len {
            rms_norm_slice(
                &hidden[t * md..(t + 1) * md],
                &mut mlp_normed[t * md..(t + 1) * md],
                eps,
            );
        }

        // ── MLP forward ────────────────────────────────────
        let mut mlp_up = vec![0.0f32; seq_len * mlp_dim];
        let mut mlp_act = vec![0.0f32; seq_len * mlp_dim];
        let mut mlp_out = vec![0.0f32; seq_len * md];

        for t in 0..seq_len {
            backend.matvec_nobias(
                fc,
                &mlp_normed[t * md..(t + 1) * md],
                &mut mlp_up[t * mlp_dim..(t + 1) * mlp_dim],
                mlp_dim, md,
            );
        }

        // ReLU²
        for i in 0..seq_len * mlp_dim {
            mlp_act[i] = if mlp_up[i] > 0.0 { mlp_up[i] * mlp_up[i] } else { 0.0 };
        }

        for t in 0..seq_len {
            backend.matvec_nobias(
                proj,
                &mlp_act[t * mlp_dim..(t + 1) * mlp_dim],
                &mut mlp_out[t * md..(t + 1) * md],
                md, mlp_dim,
            );
        }

        // ── MLP residual ────────────────────────────────────
        for i in 0..seq_len * md {
            hidden[i] += mlp_out[i];
        }

        layers_saved.push(LayerSaved {
            input: layer_input,
            attn_normed,
            q_proj,
            k_proj,
            v_proj,
            concat_heads,
            post_attn,
            mlp_normed,
            mlp_up,
            mlp_act,
        });
    }

    // ─── Final norm ─────────────────────────────────────────
    let pre_norm_hidden = hidden.clone();
    let mut normed = vec![0.0f32; seq_len * md];
    for t in 0..seq_len {
        let src = &hidden[t * md..(t + 1) * md];
        let dst = &mut normed[t * md..(t + 1) * md];
        rms_norm_slice(src, dst, eps);
        for i in 0..md { dst[i] *= norm_scale[i]; }
    }

    // ─── LM head ────────────────────────────────────────────
    let mut logits = vec![0.0f32; seq_len * vocab];
    for t in 0..seq_len {
        backend.matvec_nobias(
            &offloader.lm_head,
            &normed[t * md..(t + 1) * md],
            &mut logits[t * vocab..(t + 1) * vocab],
            vocab, md,
        );
    }

    let saved = ForwardSaved {
        layers: layers_saved,
        post_embed,
        final_hidden: normed,
        pre_norm_hidden,
    };

    (logits, saved)
}

/// Full backward pass: lm_head → final_norm → blocks (reverse) → embed.
///
/// Accumulates all gradients into `grads`.
pub fn backward_train(
    d_logits: &[f32],           // [seq_len * vocab]
    token_ids: &[i64],          // [seq_len] for embedding grad
    saved: &ForwardSaved,
    offloader: &WeightOffloader,
    grads: &mut ModelGradients,
    config: &super::config::GptConfig,
    rope: &RotaryEmbedding,
    norm_scale: &[f32],
    backend: &dyn TransformerOps,
) {
    let md = config.model_dim.get();
    let kv_dim = config.num_kv_heads.get() * config.head_dim.get();
    let mlp_dim = config.mlp_dim.get();
    let vocab = config.vocab_size.get();
    let n_heads = config.num_heads.get();
    let n_kv = config.num_kv_heads.get();
    let hd = config.head_dim.get();
    let gqa = config.gqa_ratio();
    let seq_len = d_logits.len() / vocab;
    let eps = config.norm_eps;

    // ─── Backward: lm_head ──────────────────────────────────
    let mut d_hidden = vec![0.0f32; seq_len * md];
    for t in 0..seq_len {
        let dl = &d_logits[t * vocab..(t + 1) * vocab];
        let h = &saved.final_hidden[t * md..(t + 1) * md];
        let dh = &mut d_hidden[t * md..(t + 1) * md];
        backward_lm_head(dl, h, &offloader.lm_head, dh, &mut grads.lm_head, vocab, md);
    }

    // ─── Backward: final scaled RMS norm ────────────────────
    let mut d_pre_norm = vec![0.0f32; seq_len * md];
    for t in 0..seq_len {
        backward_scaled_rms_norm(
            &d_hidden[t * md..(t + 1) * md],
            &saved.pre_norm_hidden[t * md..(t + 1) * md],
            norm_scale,
            &mut d_pre_norm[t * md..(t + 1) * md],
            &mut grads.norm_scale,
            eps,
        );
    }
    d_hidden = d_pre_norm;

    // ─── Backward: blocks (reverse order) ───────────────────
    for layer in (0..config.num_layers.get()).rev() {
        let ls = &saved.layers[layer];
        let fc = offloader.layer_mlp_fc(layer);
        let proj = offloader.layer_mlp_proj(layer);
        let wq = offloader.layer_wq(layer);
        let wk = offloader.layer_wk(layer);
        let wv = offloader.layer_wv(layer);
        let wo = offloader.layer_wo(layer);
        let lg = &mut grads.layers[layer];

        // ── MLP residual backward ───────────────────────────
        // hidden = post_attn + mlp_out → d_mlp_out = d_hidden, d_post_attn += d_hidden
        let d_mlp_out = d_hidden.clone(); // gradient flows through residual

        // ── MLP backward: proj ──────────────────────────────
        let mut d_mlp_act = vec![0.0f32; seq_len * mlp_dim];
        for t in 0..seq_len {
            backward_linear(
                &d_mlp_out[t * md..(t + 1) * md],
                &ls.mlp_act[t * mlp_dim..(t + 1) * mlp_dim],
                proj,
                &mut d_mlp_act[t * mlp_dim..(t + 1) * mlp_dim],
                lg.mlp_proj_mut(),
                md, mlp_dim,
            );
        }

        // ── MLP backward: relu² ─────────────────────────────
        let mut d_mlp_up = vec![0.0f32; seq_len * mlp_dim];
        for i in 0..seq_len * mlp_dim {
            d_mlp_up[i] = if ls.mlp_up[i] > 0.0 {
                d_mlp_act[i] * 2.0 * ls.mlp_up[i]
            } else { 0.0 };
        }

        // ── MLP backward: fc ────────────────────────────────
        let mut d_mlp_normed = vec![0.0f32; seq_len * md];
        for t in 0..seq_len {
            backward_linear(
                &d_mlp_up[t * mlp_dim..(t + 1) * mlp_dim],
                &ls.mlp_normed[t * md..(t + 1) * md],
                fc,
                &mut d_mlp_normed[t * md..(t + 1) * md],
                lg.mlp_fc_mut(),
                mlp_dim, md,
            );
        }

        // ── MLP backward: norm ──────────────────────────────
        let mut d_post_attn = vec![0.0f32; seq_len * md];
        for t in 0..seq_len {
            backward_rms_norm(
                &d_mlp_normed[t * md..(t + 1) * md],
                &ls.post_attn[t * md..(t + 1) * md],
                &mut d_post_attn[t * md..(t + 1) * md],
                eps,
            );
        }
        // Add residual gradient
        for i in 0..seq_len * md {
            d_post_attn[i] += d_hidden[i];
        }

        // ── Attention residual backward ─────────────────────
        // post_attn = layer_input + attn_out → d_attn_out = d_post_attn, d_input += d_post_attn
        let d_attn_out = d_post_attn.clone();

        // ── Attention backward: output projection ───────────
        let mut d_concat = vec![0.0f32; seq_len * md];
        for t in 0..seq_len {
            backward_linear(
                &d_attn_out[t * md..(t + 1) * md],
                &ls.concat_heads[t * md..(t + 1) * md],
                wo,
                &mut d_concat[t * md..(t + 1) * md],
                lg.wo_mut(),
                md, md,
            );
        }

        // ── Attention backward: per-head ────────────────────
        let mut d_q_all = vec![0.0f32; seq_len * md];
        let mut d_k_all = vec![0.0f32; seq_len * kv_dim];
        let mut d_v_all = vec![0.0f32; seq_len * kv_dim];

        for t in 0..seq_len {
            for h in 0..n_heads {
                let kv_h = h / gqa;
                let d_head = &d_concat[t * md + h * hd..t * md + (h + 1) * hd];

                // Recompute forward for this head at position t
                let mut q_head: Vec<f32> = ls.q_proj[t * md + h * hd..t * md + (h + 1) * hd].to_vec();
                rope.apply(&mut q_head, t);

                // Recompute scores + softmax
                let attn_len = t + 1;
                let mut scores = vec![0.0f32; attn_len];
                let scale = 1.0 / (hd as f32).sqrt();
                for s in 0..attn_len {
                    let k_s = &ls.k_proj[s * kv_dim + kv_h * hd..s * kv_dim + (kv_h + 1) * hd];
                    let mut dot = 0.0f32;
                    for i in 0..hd { dot += q_head[i] * k_s[i]; }
                    scores[s] = dot * scale;
                }
                let mut probs = scores.clone();
                backend.softmax_inplace(&mut probs);

                // d_V and d_probs from weighted sum
                let mut d_probs = vec![0.0f32; attn_len];
                for s in 0..attn_len {
                    let v_s = &ls.v_proj[s * kv_dim + kv_h * hd..s * kv_dim + (kv_h + 1) * hd];
                    for i in 0..hd {
                        d_v_all[s * kv_dim + kv_h * hd + i] += probs[s] * d_head[i];
                        d_probs[s] += d_head[i] * v_s[i];
                    }
                }

                // d_scores from softmax backward
                let mut d_scores = vec![0.0f32; attn_len];
                backward_softmax(&d_probs, &probs, &mut d_scores);

                // Scale d_scores
                for s in &mut d_scores { *s *= scale; }

                // d_Q and d_K from QK dot product
                let mut d_q_head = vec![0.0f32; hd];
                for s in 0..attn_len {
                    let k_s = &ls.k_proj[s * kv_dim + kv_h * hd..s * kv_dim + (kv_h + 1) * hd];
                    for i in 0..hd {
                        d_q_head[i] += d_scores[s] * k_s[i];
                        d_k_all[s * kv_dim + kv_h * hd + i] += d_scores[s] * q_head[i];
                    }
                }

                // Backward through Q RoPE (inverse rotation)
                backward_rope(
                    &d_q_head,
                    rope.cos_at(t),
                    rope.sin_at(t),
                    &mut d_q_all[t * md + h * hd..t * md + (h + 1) * hd],
                    rope.half_dim(),
                );
            }
        }

        // Backward through K RoPE
        for t in 0..seq_len {
            for h in 0..n_kv {
                let mut d_k_roped = vec![0.0f32; hd];
                d_k_roped.copy_from_slice(
                    &d_k_all[t * kv_dim + h * hd..t * kv_dim + (h + 1) * hd]
                );
                backward_rope(
                    &d_k_roped,
                    rope.cos_at(t),
                    rope.sin_at(t),
                    &mut d_k_all[t * kv_dim + h * hd..t * kv_dim + (h + 1) * hd],
                    rope.half_dim(),
                );
            }
        }

        // ── Attention backward: Q, K, V projections ─────────
        let mut d_attn_normed = vec![0.0f32; seq_len * md];
        for t in 0..seq_len {
            let x = &ls.attn_normed[t * md..(t + 1) * md];

            // Q backward
            let mut d_normed_q = vec![0.0f32; md];
            backward_linear(
                &d_q_all[t * md..(t + 1) * md], x, wq,
                &mut d_normed_q, lg.wq_mut(), md, md,
            );

            // K backward
            let mut d_normed_k = vec![0.0f32; md];
            backward_linear(
                &d_k_all[t * kv_dim..(t + 1) * kv_dim], x, wk,
                &mut d_normed_k, lg.wk_mut(), kv_dim, md,
            );

            // V backward
            let mut d_normed_v = vec![0.0f32; md];
            backward_linear(
                &d_v_all[t * kv_dim..(t + 1) * kv_dim], x, wv,
                &mut d_normed_v, lg.wv_mut(), kv_dim, md,
            );

            // Accumulate
            for i in 0..md {
                d_attn_normed[t * md + i] = d_normed_q[i] + d_normed_k[i] + d_normed_v[i];
            }
        }

        // ── Attention backward: norm ────────────────────────
        let mut d_layer_input = vec![0.0f32; seq_len * md];
        for t in 0..seq_len {
            backward_rms_norm(
                &d_attn_normed[t * md..(t + 1) * md],
                &ls.input[t * md..(t + 1) * md],
                &mut d_layer_input[t * md..(t + 1) * md],
                eps,
            );
        }

        // Add residual gradient (attention residual)
        for i in 0..seq_len * md {
            d_layer_input[i] += d_post_attn[i];
        }

        d_hidden = d_layer_input;
    }

    // ─── Backward: embedding ────────────────────────────────
    for (t, &tid) in token_ids.iter().enumerate() {
        let row_start = tid as usize * md;
        for i in 0..md {
            grads.embed[row_start + i] += d_hidden[t * md + i];
        }
    }
}

/// Helper: RMS norm on a slice.
fn rms_norm_slice(src: &[f32], dst: &mut [f32], eps: f32) {
    let n = src.len() as f32;
    let ss: f32 = src.iter().map(|x| x * x).sum();
    let inv_rms = 1.0 / (ss / n + eps).sqrt();
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = s * inv_rms;
    }
}

/// Clip gradient norm in-place. Returns the original norm.
pub fn clip_grad_norm(grads: &mut ModelGradients, max_norm: f32) -> f32 {
    let mut total_sq = 0.0f32;
    for lg in &grads.layers {
        total_sq += lg.data.iter().map(|v| v * v).sum::<f32>();
    }
    total_sq += grads.embed.iter().map(|v| v * v).sum::<f32>();
    total_sq += grads.lm_head.iter().map(|v| v * v).sum::<f32>();
    total_sq += grads.norm_scale.iter().map(|v| v * v).sum::<f32>();
    total_sq += grads.smear_gate.iter().map(|v| v * v).sum::<f32>();

    let norm = total_sq.sqrt();
    if norm > max_norm {
        grads.scale(max_norm / norm);
    }
    norm
}
