//! Sync accumulator and Hopfield attention readout.
//!
//! Two readout mechanisms:
//!   SyncAccumulator — pairwise products with exponential decay (original CTM)
//!   HopfieldReadout — modern Hopfield / attention (Ramsauer et al. 2020)
//!
//! The Hopfield readout is mathematically equivalent to transformer attention:
//!   z = softmax(β · query @ activations^T) @ activations
//! It has exponential storage capacity and guaranteed convergence.

use serde::{Deserialize, Serialize};

pub use modgrad_compute::neuron::*;

/// Content-dependent temporal repositioning for sync accumulator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionPredictor {
    pub n_activated: usize,  // input: number of activated neurons
    pub n_pairs: usize,      // output: one decay shift per sync pair
    pub mid_size: usize,     // bottleneck dimension

    // SwiGLU weights
    pub w_gate: Vec<f32>,    // [mid_size × n_activated]
    pub w_content: Vec<f32>, // [mid_size × n_activated]
    pub w_final: Vec<f32>,   // [n_pairs × mid_size]
}

impl PositionPredictor {
    pub fn new(n_activated: usize, n_pairs: usize) -> Self {
        let mid_size = (n_activated / 8).max(4);
        let mut rng = SimpleRng::new((n_activated * n_pairs + 42) as u64);
        let scale_g = (1.0 / n_activated as f32).sqrt();
        let scale_f = (1.0 / mid_size as f32).sqrt();

        Self {
            n_activated,
            n_pairs,
            mid_size,
            w_gate: (0..mid_size * n_activated).map(|_| rng.next_normal() * scale_g).collect(),
            w_content: (0..mid_size * n_activated).map(|_| rng.next_normal() * scale_g).collect(),
            w_final: (0..n_pairs * mid_size).map(|_| rng.next_normal() * scale_f * 0.01).collect(),
            // Small init for w_final → r_shift starts near 0 → backward compat
        }
    }

    /// Predict per-pair decay rate shifts from current activations.
    /// Returns `r_shift` in [-0.5, 0.5] for each sync pair.
    /// Add to the baseline decay to get the dynamic rate.
    /// Zero allocation: writes into provided buffer.
    pub fn predict_into(&self, activated: &[f32], out: &mut [f32]) {
        let n = self.n_activated.min(activated.len());
        let m = self.mid_size;

        // gate = sigmoid(W_gate @ activated)
        // content = W_content @ activated
        // (fused loop to avoid allocation)
        // gated = gate * content
        // r_shift = sigmoid(W_final @ gated) - 0.5

        for p in 0..self.n_pairs.min(out.len()) {
            // Compute gated vector and dot with w_final[p] in one pass
            let mut sum = 0.0f32;
            for j in 0..m {
                // gate[j] = sigmoid(W_gate[j] @ activated)
                let mut g = 0.0f32;
                let mut c = 0.0f32;
                let g_row = j * n;
                for i in 0..n {
                    g += self.w_gate[g_row + i] * activated[i];
                    c += self.w_content[g_row + i] * activated[i];
                }
                let gate = 1.0 / (1.0 + (-g).exp()); // sigmoid
                let gated = gate * c; // SwiGLU
                sum += self.w_final[p * m + j] * gated;
            }
            // sigmoid(sum) - 0.5 → shift in [-0.5, 0.5]
            out[p] = 1.0 / (1.0 + (-sum).exp()) - 0.5;
        }
    }

    pub fn param_count(&self) -> usize {
        self.w_gate.len() + self.w_content.len() + self.w_final.len()
    }
}

impl Default for PositionPredictor {
    fn default() -> Self { Self::new(0, 0) }
}

// ─── Sync accumulator ───────────────────────────────────────

/// Pairwise sync accumulator with exponential decay.
/// Optionally uses content-dependent decay rates (RePo-style positioning).
#[derive(Debug, Clone)]
pub struct SyncAccumulator {
    pub n_pairs: usize,
    pub indices_left: Vec<usize>,
    pub indices_right: Vec<usize>,
    pub decay: Vec<f32>,       // baseline decay (learned parameter)
    pub alpha: Vec<f32>,
    pub beta: Vec<f32>,
    pub initialized: bool,
    /// Content-dependent position predictor (RePo-style).
    /// When present, decay rates are modulated per-tick by activation content.
    pub position_predictor: Option<PositionPredictor>,
    /// Scratch buffer for dynamic decay shifts (avoids allocation per tick).
    pub r_shift_buf: Vec<f32>,
}

impl SyncAccumulator {
    pub fn new(n_pairs: usize, n_neurons: usize) -> Self {
        let mut rng = SimpleRng::new(n_pairs as u64 * 7919);
        let indices_left: Vec<usize> = (0..n_pairs).map(|_| rng.next_u64() as usize % n_neurons).collect();
        let indices_right: Vec<usize> = (0..n_pairs).map(|_| rng.next_u64() as usize % n_neurons).collect();
        // Initialize decay to ~0.7: exp(-0.7) ≈ 0.50 retention per tick.
        // This means each tick contributes ~50% of its history, so recent ticks
        // dominate the sync signal and different inputs produce distinct patterns.
        // With decay=0.0 (old default), exp(0)=1.0 = no forgetting, all ticks
        // equally weighted, sync converges to input-independent mean.
        let decay: Vec<f32> = (0..n_pairs).map(|i| {
            // Slight variation per pair so different timescales coexist
            0.5 + 0.4 * ((i as f32 / n_pairs as f32) * std::f32::consts::PI).sin()
        }).collect();
        // Enable position predictor if we have enough neurons to benefit
        let position_predictor = if n_neurons >= 16 && n_pairs >= 8 {
            Some(PositionPredictor::new(n_neurons, n_pairs))
        } else {
            None
        };

        Self {
            n_pairs,
            indices_left,
            indices_right,
            decay,
            alpha: vec![0.0; n_pairs],
            beta: vec![0.0; n_pairs],
            initialized: false,
            r_shift_buf: vec![0.0; n_pairs],
            position_predictor,
        }
    }

    /// Update sync from current activations, modulated by dopamine and phase.
    ///
    /// **Phase-aware temporal binding**: neurons that fire at similar times within
    /// the tick contribute MORE to the sync signal. This models gamma synchrony —
    /// the brain's mechanism for binding features into coherent representations.
    ///
    /// A neuron pair with phases (0.1, 0.1) = synchronous = full contribution.
    /// A neuron pair with phases (0.1, 0.9) = asynchronous = reduced contribution.
    ///
    /// Without phases: sync = left_mag × right_mag × dopamine
    /// With phases:    sync = left_mag × right_mag × dopamine × temporal_proximity
    pub fn update(&mut self, activated: &[f32], dopamine: f32) -> Vec<f32> {
        self.update_with_phase(activated, dopamine, &[])
    }

    /// Phase-aware update. If phases is empty, falls back to magnitude-only.
    pub fn update_with_phase(&mut self, activated: &[f32], dopamine: f32, phases: &[f32]) -> Vec<f32> {
        let has_phase = !phases.is_empty();
        let mut sync = vec![0.0f32; self.n_pairs];

        // Content-dependent positioning: predict decay shifts from activations
        if let Some(ref predictor) = self.position_predictor {
            predictor.predict_into(activated, &mut self.r_shift_buf);
        } else {
            self.r_shift_buf.fill(0.0); // no predictor → no shift → baseline decay
        }
        for i in 0..self.n_pairs {
            let li = self.indices_left[i];
            let ri = self.indices_right[i];
            let left = activated[li];
            let right = activated[ri];

            // Phase-aware temporal binding
            let temporal_proximity = if has_phase && li < phases.len() && ri < phases.len() {
                // Gaussian kernel: nearby phases → proximity≈1, distant → proximity→0
                // Width σ=0.3: phases within ~0.3 of each other are "synchronous"
                let phase_diff = phases[li] - phases[ri];
                (-phase_diff * phase_diff / (2.0 * 0.3 * 0.3)).exp()
            } else {
                1.0 // no phase info: all pairs contribute equally (backward compat)
            };

            let pairwise = left * right * dopamine * temporal_proximity;
            // Content-dependent decay: baseline + dynamic shift from position predictor
            let decay_val = self.decay[i] + self.r_shift_buf[i]; // r_shift_buf is 0 if no predictor
            let r = (-decay_val.clamp(0.0, 15.0)).exp();

            if !self.initialized {
                self.alpha[i] = pairwise;
                self.beta[i] = dopamine;
            } else {
                self.alpha[i] = r * self.alpha[i] + pairwise;
                self.beta[i] = r * self.beta[i] + dopamine;
            }
            sync[i] = self.alpha[i] / self.beta[i].sqrt().max(1e-8);
        }
        self.initialized = true;
        sync
    }

    pub fn reset(&mut self) {
        self.alpha.fill(0.0);
        self.beta.fill(0.0);
        self.initialized = false;
    }
}

// ─── Modern Hopfield / Attention readout ────────────────────

/// Hopfield readout: attention mechanism as the update rule of a modern
/// Hopfield network (Ramsauer et al. 2020, Widrich et al. 2020).
///
/// z = softmax(β · query @ keys^T) @ values
///
/// Where:
///   query: learned d_out-dimensional vectors (one per output dim)
///   keys = W_k @ activations (project to query space)
///   values = activations (or projected)
///   β = 1/√d_k (temperature)
///
/// Properties:
///   - Exponential storage capacity (can retrieve from huge activation space)
///   - One-step convergence (single softmax, no iteration needed)
///   - Differentiable (works with backprop AND SPSA)
///   - No random pair selection (uses ALL activations)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HopfieldReadout {
    /// Number of output dimensions
    pub d_out: usize,
    /// Key dimension (query/key comparison space)
    pub d_key: usize,
    /// Input dimension (number of neurons feeding into readout)
    pub d_in: usize,
    /// Query vectors: [d_out × d_key] — learned, one query per output dim
    pub queries: Vec<f32>,
    /// Key projection: [d_key × d_in] — projects activations to key space
    pub w_key: Vec<f32>,
    /// Temperature: 1/√d_key
    pub beta: f32,
    /// Running EMA of output for temporal smoothing across ticks
    pub ema: Vec<f32>,
    /// EMA decay factor (0 = no memory, 1 = infinite memory)
    pub ema_decay: f32,
    pub initialized: bool,
}

impl HopfieldReadout {
    pub fn new(d_in: usize, d_out: usize, d_key: usize) -> Self {
        let mut rng = SimpleRng::new((d_in * d_out + 12345) as u64);
        let scale_q = (1.0 / d_key as f32).sqrt();
        let scale_k = (1.0 / d_in as f32).sqrt();

        Self {
            d_out, d_key, d_in,
            queries: (0..d_out * d_key).map(|_| rng.next_normal() * scale_q).collect(),
            w_key: (0..d_key * d_in).map(|_| rng.next_normal() * scale_k).collect(),
            beta: 1.0 / (d_key as f32).sqrt(),
            ema: vec![0.0; d_out],
            ema_decay: 0.5,
            initialized: false,
        }
    }

    /// Compute readout from activations.
    /// activations: [d_in] — all neuron activations concatenated
    /// Returns: [d_out] — the readout vector (replaces sync)
    pub fn forward(&mut self, activations: &[f32], dopamine: f32) -> Vec<f32> {
        let d_in = self.d_in.min(activations.len());
        let d_key = self.d_key;
        let d_out = self.d_out;

        // Project activations to key space: key = W_key @ act  [d_key]
        let mut key = vec![0.0f32; d_key];
        for k in 0..d_key {
            let mut sum = 0.0f32;
            for i in 0..d_in {
                sum += self.w_key[k * d_in + i] * activations[i];
            }
            key[k] = sum;
        }

        // For each output dim: attention score = β * query_j · key
        // Then softmax over output dims? No — we want attention over NEURONS.
        //
        // Actually: treat each neuron activation as a "stored pattern" (value).
        // The query searches for the most relevant neurons.
        //
        // Attention: for each query q_j:
        //   scores[i] = β * q_j · k_i  where k_i = W_key @ e_i (projection of neuron i)
        //   But that requires projecting each neuron separately.
        //
        // Simpler and correct: use the full activation vector as one pattern.
        // score_j = β * query_j · key (scalar per output dim)
        // output_j = score_j * (sum of activations weighted by key similarity)
        //
        // Simplest correct form:
        //   z_j = query_j · key  (dot product in key space, one per output)
        //   This IS a linear readout in key space. The power comes from the
        //   key projection transforming the activation space.
        //
        // For proper Hopfield with multiple stored patterns, we'd need
        // activations from multiple ticks as separate patterns. Let's do that:
        // store the current activations as a new pattern each tick, query retrieves.
        //
        // For now: single-tick attention readout.
        // z_j = sum_i (softmax(β * query_j · w_key_col_i) * activations[i])

        // Compute attention weights: [d_out × d_in]
        // For each output j, attend over input neurons i
        // score_ji = β * sum_k(query[j,k] * w_key[k,i])
        let mut output = vec![0.0f32; d_out];
        for j in 0..d_out {
            // Compute attention scores for this query
            let mut scores = vec![0.0f32; d_in];
            for i in 0..d_in {
                let mut dot = 0.0f32;
                for k in 0..d_key {
                    dot += self.queries[j * d_key + k] * self.w_key[k * d_in + i];
                }
                scores[i] = self.beta * dot;
            }

            // Softmax
            let max_s: f32 = scores.iter().fold(f32::MIN, |a, &b| a.max(b));
            let mut sum_exp = 0.0f32;
            for s in &mut scores {
                *s = (*s - max_s).exp();
                sum_exp += *s;
            }
            if sum_exp > 0.0 {
                for s in &mut scores { *s /= sum_exp; }
            }

            // Weighted sum of activations
            let mut z = 0.0f32;
            for i in 0..d_in {
                z += scores[i] * activations[i];
            }
            output[j] = z * dopamine;
        }

        // Temporal EMA smoothing
        if self.initialized {
            for j in 0..d_out {
                self.ema[j] = self.ema_decay * self.ema[j] + (1.0 - self.ema_decay) * output[j];
                output[j] = self.ema[j];
            }
        } else {
            self.ema.copy_from_slice(&output);
            self.initialized = true;
        }

        output
    }

    pub fn reset(&mut self) {
        self.ema.fill(0.0);
        self.initialized = false;
    }

    pub fn param_count(&self) -> usize {
        self.queries.len() + self.w_key.len()
    }
}
