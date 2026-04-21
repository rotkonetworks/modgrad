//! Transformer–NMC bridge: trained features → neuromorphic dynamics.
//!
//! Architecture:
//!   Input text → Transformer (backprop-trained) → hidden states [model_dim]
//!                                                       ↓
//!                                                 NMC d_input = model_dim
//!                                                       ↓
//!                                             NMC tick loop (8 regions, N ticks)
//!                                                       ↓
//!                                             Sync/Activations readout
//!                                                       ↓
//!                                                 Task output
//!
//! The transformer provides learned features (the "what").
//! The NMC adds temporal deliberation and online adaptation (the "how to think").
//!
//! This solves the core problem: NMC with random projections can't learn because
//! the features are garbage. With transformer features, the NMC's region dynamics
//! and tick-based processing operate on meaningful representations.

use crate::config::CtmConfig;
use crate::weights::CtmWeights;
use crate::session::CtmSession;
use crate::forward::forward_split;
use modgrad_compute::neuron::SimpleRng;
use modgrad_transformer::model::GptModel;
use modgrad_transformer::ops::TransformerOps;
use modgrad_transformer::residual::ForwardCtx;
use modgrad_transformer::kv_cache::{KvCache, Decoding};
use crate::eval::linalg;

/// Bridge between a trained transformer and NMC neuromorphic compute.
///
/// The transformer runs first to produce rich features from input tokens.
/// Those features feed into the NMC as its d_input observation vector.
/// The NMC then runs its tick loop to produce task-specific output.
pub struct TransformerNmcBridge {
    /// Frozen transformer weights (backprop-trained, then frozen).
    pub transformer: GptModel,
    /// NMC brain weights (can be modified by perturbation learning).
    pub nmc_weights: CtmWeights,
    /// NMC config.
    pub nmc_config: CtmConfig,
    /// Which transformer layer to extract features from.
    /// -1 = final hidden state (after all layers), 0..N = intermediate layer.
    pub extract_layer: i32,
    /// Optional: project transformer features to NMC input dim.
    /// None = dims match directly (model_dim == d_input).
    pub projection: Option<Vec<f32>>,
}

/// Result of a bridge forward pass.
pub struct BridgeOutput {
    /// Transformer hidden state used as NMC input [model_dim].
    pub features: Vec<f32>,
    /// NMC sync output [n_sync_out] (for LS readout).
    pub sync: Vec<f32>,
    /// NMC raw activations [total_neurons] (for activations readout).
    pub activations: Vec<f32>,
    /// Transformer logits [vocab_size] (for language generation).
    pub logits: Vec<f32>,
}

impl TransformerNmcBridge {
    /// Create a bridge with matching dimensions (model_dim == d_input).
    pub fn new(transformer: GptModel, nmc_weights: CtmWeights) -> Self {
        let nmc_config = nmc_weights.config.clone();
        Self {
            transformer,
            nmc_weights,
            nmc_config,
            extract_layer: -1,
            projection: None,
        }
    }

    /// Forward pass: token → transformer features → NMC processing.
    ///
    /// `token_id`: current token to process.
    /// `cache`: transformer KV cache (for autoregressive generation).
    /// `ctx`: transformer forward context.
    /// `backend`: compute backend for transformer ops.
    pub fn forward(
        &self,
        token_id: usize,
        cache: &mut KvCache<Decoding>,
        ctx: &mut ForwardCtx,
        backend: &dyn TransformerOps,
    ) -> BridgeOutput {
        // Step 1: Transformer forward → hidden state + logits
        let logits = self.transformer.forward_one(token_id, cache, ctx, backend);

        // Get the last hidden state (before lm_head) as features.
        // We reconstruct it: the cache stores prev_embedding which is the
        // hidden state after all layers + backout (before final norm).
        let features = cache.prev_embedding.clone();

        // Step 2: Optionally project features to NMC input dim.
        // Pure matvec `out = proj @ features`, no bias — routes through
        // `ops::matvec` with an inline-constructed zero-bias so the
        // dispatch path stays consistent with other matvec call sites.
        let nmc_input = match &self.projection {
            Some(proj) => {
                let d_in = features.len();
                let d_out = self.nmc_config.d_input;
                let zero_bias = vec![0.0f32; d_out];
                let mut out = vec![0.0f32; d_out];
                modgrad_device::backend::ops::matvec(
                    &features, proj, &zero_bias, &mut out,
                    d_out, d_in,
                    modgrad_device::backend::QuantKind::F32,
                ).expect("bridge: matvec dispatch");
                out
            }
            None => {
                // Direct pass — dims must match
                debug_assert_eq!(features.len(), self.nmc_config.d_input,
                    "transformer model_dim ({}) != NMC d_input ({})",
                    features.len(), self.nmc_config.d_input);
                features.clone()
            }
        };

        // Step 3: NMC forward pass with transformer features as observation
        let proprio = vec![0.0f32; self.nmc_config.d_input];
        let mut session = CtmSession::new(&self.nmc_config);
        let mut tick = self.nmc_weights.init_tick_state();

        let (_, sync, _) = forward_split(
            &self.nmc_weights, &mut session, &mut tick,
            &nmc_input, &proprio, false,
        );

        let activations = tick.activations.clone();

        BridgeOutput { features, sync, activations, logits }
    }

    /// Batch forward: process multiple tokens through transformer, then NMC.
    ///
    /// Returns one BridgeOutput per token. Useful for evaluating a sequence
    /// where each token's transformer features feed into NMC independently.
    pub fn forward_batch(
        &self,
        token_ids: &[usize],
        cache: &mut KvCache<Decoding>,
        ctx: &mut ForwardCtx,
        backend: &dyn TransformerOps,
    ) -> Vec<BridgeOutput> {
        token_ids.iter().map(|&tid| {
            self.forward(tid, cache, ctx, backend)
        }).collect()
    }

    /// Evaluate classification accuracy using NMC sync features + LS readout.
    ///
    /// Takes (input_tokens, target_label) pairs.
    /// Each input sequence is processed through transformer → NMC,
    /// then LS regression classifies using NMC sync features.
    pub fn eval_sync_readout(
        &self,
        data: &[(Vec<usize>, usize)],
        num_classes: usize,
        cache: &mut KvCache<Decoding>,
        ctx: &mut ForwardCtx,
        backend: &dyn TransformerOps,
    ) -> f32 {
        let sd = self.nmc_config.n_sync_out;

        // Collect (sync_features, label) for all examples
        let mut features: Vec<(Vec<f32>, usize)> = Vec::with_capacity(data.len());
        for (tokens, label) in data {
            // Process last token through the bridge
            let last_token = *tokens.last().unwrap_or(&0);
            let output = self.forward(last_token, cache, ctx, backend);
            features.push((output.sync, *label));
        }

        // Least-squares readout
        let mut xtx = vec![0.0f32; sd * sd];
        let mut xty = vec![0.0f32; sd * num_classes];

        for (f, l) in &features {
            for r in 0..sd {
                for c in 0..sd {
                    xtx[r * sd + c] += f[r] * f[c];
                }
                xty[r * num_classes + l] += f[r];
            }
        }

        // Ridge regression with lambda sweep
        let mut best_acc = 0.0f32;
        for &lam in &[1e-6, 1e-4, 1e-2, 0.1, 1.0] {
            let mut xr = xtx.clone();
            for i in 0..sd { xr[i * sd + i] += lam; }

            if let Some(l) = linalg::cholesky(&xr, sd) {
                let mut weights = vec![0.0f32; sd * num_classes];
                for c in 0..num_classes {
                    let rhs: Vec<f32> = (0..sd).map(|r| xty[r * num_classes + c]).collect();
                    let z = linalg::forward_solve(&l, &rhs, sd);
                    let w = linalg::backward_solve(&l, &z, sd);
                    for r in 0..sd { weights[c * sd + r] = w[r]; }
                }

                // Evaluate accuracy
                let correct: usize = features.iter().map(|(f, lab)| {
                    let mut best_class = 0;
                    let mut best_score = f32::NEG_INFINITY;
                    for c in 0..num_classes {
                        let score: f32 = (0..sd).map(|r| weights[c * sd + r] * f[r]).sum();
                        if score > best_score {
                            best_score = score;
                            best_class = c;
                        }
                    }
                    if best_class == *lab { 1 } else { 0 }
                }).sum();

                let acc = correct as f32 / features.len() as f32;
                if acc > best_acc { best_acc = acc; }
            }
        }

        best_acc
    }

    /// Create a random projection matrix for dimension mismatch.
    ///
    /// Used when transformer model_dim != NMC d_input.
    pub fn random_projection(from_dim: usize, to_dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = SimpleRng::new(seed);
        let scale = (1.0 / from_dim as f32).sqrt();
        (0..to_dim * from_dim).map(|_| rng.next_normal() * scale).collect()
    }
}
