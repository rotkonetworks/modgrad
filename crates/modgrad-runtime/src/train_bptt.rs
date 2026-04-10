//! BPTT training loop for the CTM on next-token prediction.
//!
//! Phase 0 of the roadmap: train the 8-region brain to predict the next byte
//! in a text sequence using backpropagation through time.
//!
//! Architecture per token:
//!   token → embed table → sensory MLP → CTM forward_split (K ticks) → sync → output_proj → logits
//!
//! The CTM processes each token position sequentially. Within each position,
//! K ticks of recurrent dynamics produce a sync vector. The sync vector is
//! projected to logits over the vocabulary.
//!
//! BPTT flows backward through: output_proj → sync → K ticks (reverse) → synapse + NLM gradients.
//! State (activations, traces) carries forward across token positions.

use super::weights::CtmWeights;
use super::session::CtmSession;
use modgrad_compute::neuron::Linear;
use super::forward::forward_split;
use super::session::BpttTickCache;

/// Training config for BPTT.
pub struct BpttTrainConfig {
    pub lr: f32,
    pub batch_size: usize,
    pub seq_len: usize,
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub log_every: usize,
}

impl Default for BpttTrainConfig {
    fn default() -> Self {
        Self {
            lr: 0.001,
            batch_size: 1,
            seq_len: 128,
            vocab_size: 256, // byte-level
            embed_dim: 128,
            log_every: 100,
        }
    }
}

/// Result of one training step.
pub struct StepResult {
    pub loss: f32,
    pub correct: usize,
    pub total: usize,
}

/// Cross-entropy loss + gradient w.r.t. logits.
/// Returns (loss, d_logits) where d_logits[i] = softmax[i] - one_hot[target][i].
fn cross_entropy_with_grad(logits: &[f32], target: usize) -> (f32, Vec<f32>) {
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|l| (l - max_l).exp()).sum();
    let softmax: Vec<f32> = logits.iter().map(|l| (l - max_l).exp() / exp_sum).collect();

    let p = softmax.get(target).copied().unwrap_or(1e-8).max(1e-8);
    let loss = -p.ln();

    let mut d_logits = softmax;
    if target < d_logits.len() {
        d_logits[target] -= 1.0;
    }

    (loss, d_logits)
}

/// One training step: process a sequence of tokens, compute loss, backprop.
///
/// Returns the average loss over the sequence.
/// Modifies weights in-place (SGD step).
pub fn train_step(
    weights: &mut CtmWeights,
    embed_table: &[f32],      // [vocab_size × embed_dim]
    sensory_w: &Linear,       // embed_dim → d_input
    output_proj: &mut Linear, // sync_dim → vocab_size
    token_ids: &[usize],
    config: &BpttTrainConfig,
) -> StepResult {
    let proprio = vec![0.0f32; weights.config.d_input];
    let k = weights.config.iterations;

    // Per-token: forward with BPTT caching, collect logits
    let mut session = CtmSession::new(&weights.config);
    let mut tick_state = weights.init_tick_state();

    let mut all_caches: Vec<Vec<BpttTickCache>> = Vec::new(); // [token_pos][tick]
    let mut all_logits: Vec<Vec<f32>> = Vec::new();
    let mut all_sync: Vec<Vec<f32>> = Vec::new();
    let mut total_loss = 0.0f32;
    let mut correct = 0usize;
    let mut total = 0usize;

    // Forward pass: token by token
    for pos in 0..token_ids.len().saturating_sub(1) {
        let tid = token_ids[pos];

        // Embed
        let emb_start = tid * config.embed_dim;
        let emb_end = emb_start + config.embed_dim;
        let emb = if emb_end <= embed_table.len() {
            &embed_table[emb_start..emb_end]
        } else {
            &embed_table[..config.embed_dim] // fallback for OOB
        };

        // Sensory projection: embed → d_input
        let obs = sensory_w.forward(emb);

        // Enable BPTT caching
        session.bptt_caches = Some(Vec::with_capacity(k));

        // Forward through CTM (K ticks)
        let (_preds, sync, _signals) = forward_split(
            weights, &mut session, &mut tick_state,
            &obs, &proprio, false,
        );

        // Extract caches
        let caches = session.bptt_caches.take().unwrap_or_default();

        // Project sync → logits
        let logits = output_proj.forward(&sync);

        // Loss against next token
        let target = token_ids[pos + 1];
        let (loss, _d_logits) = cross_entropy_with_grad(&logits, target);
        total_loss += loss;

        let pred = logits.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
        if pred == target { correct += 1; }
        total += 1;

        all_caches.push(caches);
        all_logits.push(logits);
        all_sync.push(sync);
    }

    // Backward pass: token by token in reverse
    // For now: only backprop through output_proj → sync.
    // Full BPTT through tick loop comes next.
    let sync_dim = if all_sync.is_empty() { 0 } else { all_sync[0].len() };
    let mut d_output_proj_w = vec![0.0f32; output_proj.weight.len()];
    let mut d_output_proj_b = vec![0.0f32; output_proj.bias.len()];

    for pos in 0..all_logits.len() {
        let target = token_ids[pos + 1];
        let (_loss, d_logits) = cross_entropy_with_grad(&all_logits[pos], target);

        // d_output_proj: dW = d_logits × sync^T, db = d_logits
        let sync = &all_sync[pos];
        for i in 0..output_proj.out_dim {
            d_output_proj_b[i] += d_logits[i];
            for j in 0..sync_dim.min(output_proj.in_dim) {
                d_output_proj_w[i * output_proj.in_dim + j] += d_logits[i] * sync[j];
            }
        }

        // d_sync = output_proj.W^T × d_logits
        // This would feed into BPTT through the tick caches
        // TODO: wire into backward through all_caches[pos]
    }

    // Apply output_proj gradients
    let scale = config.lr / total.max(1) as f32;
    for (w, &g) in output_proj.weight.iter_mut().zip(d_output_proj_w.iter()) {
        *w -= scale * g;
    }
    for (b, &g) in output_proj.bias.iter_mut().zip(d_output_proj_b.iter()) {
        *b -= scale * g;
    }

    StepResult {
        loss: total_loss / total.max(1) as f32,
        correct,
        total,
    }
}
