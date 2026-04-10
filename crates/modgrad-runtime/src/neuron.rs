//! isis-specific neuron layers: NeuronLayer, NeuronLayerWeights, plasticity, noise.
//!
//! These types compose SDK compute primitives (SuperLinear, SimpleRng) with
//! isis runtime concepts (LayerConfig, SleepPhase). Generic primitives live
//! in `modgrad_compute::neuron`.

use serde::{Deserialize, Serialize};

use modgrad_compute::neuron::{SuperLinear, SimpleRng};
use super::config::*;
use super::session::SleepPhase;

// ─── NeuronLayer split: Weights (immutable) + State (per-thread) ──

/// Immutable weight data for a brain region. Safe to share behind Arc.
/// Contains all NLM stages, inhibitory masks, and config.
/// Does NOT contain mutable per-neuron state (noise_scale, usefulness_ema).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronLayerWeights {
    pub config: LayerConfig,
    /// NLM: trace processor (per-neuron MLP chain)
    pub nlm_stage1: SuperLinear,
    pub nlm_stage2: Option<SuperLinear>,
    /// Start states (cloned into CtmState each forward)
    pub start_trace: Vec<f32>,     // [n_neurons × memory_length]
    pub start_activated: Vec<f32>, // [n_neurons]
    /// Inhibitory mask: true = inhibitory (GABA), false = excitatory.
    pub inhibitory: Vec<bool>,
}

// SAFETY: NeuronLayerWeights contains only Vec<f32>, Vec<bool>, SuperLinear
// and LayerConfig — all plain data with no interior mutability.
unsafe impl Sync for NeuronLayerWeights {}

impl NeuronLayerWeights {
    pub fn new(config: &LayerConfig) -> Self {
        let n = config.n_neurons;
        let m = config.memory_length;
        let h = config.memory_hidden_dims;

        // No GLU: output dims are direct (not 2× for gating)
        let (nlm_stage1, nlm_stage2) = if config.nlm_depth >= 2 {
            (SuperLinear::new(n, m, h), Some(SuperLinear::new(n, h, 1)))
        } else {
            (SuperLinear::new(n, m, 1), None)
        };

        let mut rng = SimpleRng::new((n * m) as u64);
        let scale_n = (1.0 / n as f32).sqrt();
        let scale_nm = (1.0 / (n + m) as f32).sqrt();
        let start_activated: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 * scale_n - scale_n).collect();
        let start_trace: Vec<f32> = (0..n * m).map(|_| rng.next_f32() * 2.0 * scale_nm - scale_nm).collect();

        let n_inhibitory = (n as f32 * config.inhibitory_fraction) as usize;
        let mut inhibitory = vec![false; n];
        for i in (n - n_inhibitory)..n {
            inhibitory[i] = true;
        }

        Self {
            config: config.clone(), nlm_stage1, nlm_stage2,
            start_trace, start_activated, inhibitory,
        }
    }

    /// One tick: update trace, apply NLM -> activated.
    /// Same logic as NeuronLayer::step but on the weights-only struct.
    pub fn step(&self, pre_activation: &[f32], trace: &mut Vec<f32>) -> Vec<f32> {
        let n = self.config.n_neurons;
        let m = self.config.memory_length;

        // Shift trace left, append new activation
        for neuron in 0..n {
            let off = neuron * m;
            for t in 0..m - 1 {
                trace[off + t] = trace[off + t + 1];
            }
            trace[off + m - 1] = pre_activation[neuron];
        }

        // NLM: SuperLinear + SiLU. No GLU (kills signal), no LayerNorm.
        let mut stage1_out = self.nlm_stage1.forward(trace);
        for v in &mut stage1_out {
            let sigmoid = 1.0 / (1.0 + (-*v).exp());
            *v *= sigmoid;
        }

        let mut activated = if let Some(ref stage2) = self.nlm_stage2 {
            let mut out = stage2.forward(&stage1_out);
            for v in &mut out {
                let sigmoid = 1.0 / (1.0 + (-*v).exp());
                *v *= sigmoid;
            }
            out
        } else {
            stage1_out
        };

        for (i, &is_inhib) in self.inhibitory.iter().enumerate() {
            if is_inhib && i < activated.len() {
                activated[i] = -activated[i].abs();
            }
        }

        let max_abs: f32 = activated.iter()
            .map(|x| x.abs())
            .fold(0.0f32, f32::max)
            .max(0.1);
        for v in &mut activated {
            *v /= max_abs;
        }

        activated
    }

    /// Compute firing phase for each neuron given activations.
    #[inline]
    pub fn compute_phase(activated: &[f32]) -> Vec<f32> {
        NeuronLayer::compute_phase(activated)
    }

    /// Cached NLM forward: stores intermediates for backward pass.
    pub fn step_cached(&self, pre_activation: &[f32], trace: &mut Vec<f32>) -> (Vec<f32>, NlmForwardCache) {
        let n = self.config.n_neurons;
        let m = self.config.memory_length;

        // Shift trace left, append new activation
        for neuron in 0..n {
            let off = neuron * m;
            for t in 0..m - 1 {
                trace[off + t] = trace[off + t + 1];
            }
            trace[off + m - 1] = pre_activation[neuron];
        }

        let trace_snapshot = trace.clone();

        // Stage 1: SuperLinear + SiLU
        let z1 = self.nlm_stage1.forward(&trace_snapshot);
        let mut h1 = z1.clone();
        for v in &mut h1 {
            let s = 1.0 / (1.0 + (-*v).exp());
            *v *= s;
        }

        let (z2, pre_norm) = if let Some(ref stage2) = self.nlm_stage2 {
            let z = stage2.forward(&h1);
            let mut out = z.clone();
            for v in &mut out {
                let s = 1.0 / (1.0 + (-*v).exp());
                *v *= s;
            }
            (Some(z), out)
        } else {
            (None, h1.clone())
        };

        // Inhibitory flip
        let mut post_inhib = pre_norm.clone();
        for (i, &is_inhib) in self.inhibitory.iter().enumerate() {
            if is_inhib && i < post_inhib.len() {
                post_inhib[i] = -post_inhib[i].abs();
            }
        }

        // Max-abs normalization
        let max_abs = post_inhib.iter().map(|x| x.abs()).fold(0.0f32, f32::max).max(0.1);
        let mut activated = post_inhib.clone();
        for v in &mut activated { *v /= max_abs; }

        let cache = NlmForwardCache {
            trace: trace_snapshot,
            z1, h1, z2,
            pre_norm, post_inhib, max_abs,
            n, m,
        };

        (activated, cache)
    }

    /// Backward through NLM step. Returns (d_pre_activation, NlmGradients).
    /// d_activated is dL/d(activated) — gradient w.r.t. the NLM output.
    pub fn backward_nlm(&self, d_activated: &[f32], cache: &NlmForwardCache) -> (Vec<f32>, NlmGradients) {
        let n = cache.n;
        let m = cache.m;

        // Backward through max-abs normalization
        // activated = post_inhib / max_abs
        // d_post_inhib[i] = d_activated[i] / max_abs  (simplified: ignore d_max_abs term)
        let d_post_inhib: Vec<f32> = d_activated.iter()
            .map(|&da| da / cache.max_abs)
            .collect();

        // Backward through inhibitory flip
        let mut d_pre_norm = d_post_inhib.clone();
        for (i, &is_inhib) in self.inhibitory.iter().enumerate() {
            if is_inhib && i < d_pre_norm.len() {
                // post_inhib[i] = -|pre_norm[i]|, d/dx(-|x|) = -sign(x)
                d_pre_norm[i] = -d_post_inhib[i] * cache.pre_norm[i].signum();
            }
        }

        // Backward through stage2 (if present) or use d_pre_norm as d_h1
        let (d_h1, dw_stage2) = if let Some(ref stage2) = self.nlm_stage2 {
            let z2 = cache.z2.as_ref().unwrap();
            // d_z2 = d_pre_norm * SiLU'(z2)
            let d_z2: Vec<f32> = d_pre_norm.iter().zip(z2.iter()).map(|(&dp, &z)| {
                let s = 1.0 / (1.0 + (-z).exp());
                dp * (s + z * s * (1.0 - s))
            }).collect();

            // dW_stage2: per-neuron gradients
            let mut dw2 = vec![0.0f32; stage2.weights.len()];
            let mut db2 = vec![0.0f32; stage2.biases.len()];
            let in2 = stage2.in_per;
            let out2 = stage2.out_per;
            for neuron in 0..n {
                for o in 0..out2 {
                    let grad = d_z2[neuron * out2 + o];
                    db2[neuron * out2 + o] = grad;
                    for i in 0..in2 {
                        dw2[neuron * out2 * in2 + o * in2 + i] = grad * cache.h1[neuron * in2 + i];
                    }
                }
            }

            // d_h1 = W_stage2^T @ d_z2 (per neuron)
            let mut d_h1 = vec![0.0f32; n * in2];
            for neuron in 0..n {
                for i in 0..in2 {
                    let mut sum = 0.0f32;
                    for o in 0..out2 {
                        sum += stage2.weights[neuron * out2 * in2 + o * in2 + i] * d_z2[neuron * out2 + o];
                    }
                    d_h1[neuron * in2 + i] = sum;
                }
            }

            (d_h1, Some((dw2, db2)))
        } else {
            (d_pre_norm, None)
        };

        // Backward through stage1 SiLU
        let d_z1: Vec<f32> = d_h1.iter().zip(cache.z1.iter()).map(|(&dh, &z)| {
            let s = 1.0 / (1.0 + (-z).exp());
            dh * (s + z * s * (1.0 - s))
        }).collect();

        // dW_stage1: per-neuron gradients
        let in1 = self.nlm_stage1.in_per;
        let out1 = self.nlm_stage1.out_per;
        let mut dw1 = vec![0.0f32; self.nlm_stage1.weights.len()];
        let mut db1 = vec![0.0f32; self.nlm_stage1.biases.len()];
        for neuron in 0..n {
            for o in 0..out1 {
                let grad = d_z1[neuron * out1 + o];
                db1[neuron * out1 + o] = grad;
                for i in 0..in1 {
                    dw1[neuron * out1 * in1 + o * in1 + i] = grad * cache.trace[neuron * m + i];
                }
            }
        }

        // d_trace = W_stage1^T @ d_z1 (per neuron) — we only need the last column
        // which is d_pre_activation (the signal that was appended to the trace)
        let mut d_pre_activation = vec![0.0f32; n];
        for neuron in 0..n {
            let mut sum = 0.0f32;
            for o in 0..out1 {
                // The last trace position (m-1) is where pre_activation was written
                sum += self.nlm_stage1.weights[neuron * out1 * in1 + o * in1 + (m - 1)] * d_z1[neuron * out1 + o];
            }
            d_pre_activation[neuron] = sum;
        }

        let grads = NlmGradients { dw_stage1: dw1, db_stage1: db1, dw_stage2 };
        (d_pre_activation, grads)
    }

    /// Apply NLM gradients with learning rate.
    pub fn apply_nlm_gradients(&mut self, grads: &NlmGradients, lr: f32) {
        for (w, &g) in self.nlm_stage1.weights.iter_mut().zip(grads.dw_stage1.iter()) {
            *w -= lr * g;
        }
        for (b, &g) in self.nlm_stage1.biases.iter_mut().zip(grads.db_stage1.iter()) {
            *b -= lr * g;
        }
        if let Some((ref dw2, ref db2)) = grads.dw_stage2 {
            if let Some(ref mut stage2) = self.nlm_stage2 {
                for (w, &g) in stage2.weights.iter_mut().zip(dw2.iter()) { *w -= lr * g; }
                for (b, &g) in stage2.biases.iter_mut().zip(db2.iter()) { *b -= lr * g; }
            }
        }
    }
}

/// Cached intermediates from NLM forward pass.
pub struct NlmForwardCache {
    pub trace: Vec<f32>,
    pub z1: Vec<f32>,
    pub h1: Vec<f32>,
    pub z2: Option<Vec<f32>>,
    pub pre_norm: Vec<f32>,
    pub post_inhib: Vec<f32>,
    pub max_abs: f32,
    pub n: usize,
    pub m: usize,
}

/// Gradients for NLM weights.
pub struct NlmGradients {
    pub dw_stage1: Vec<f32>,
    pub db_stage1: Vec<f32>,
    pub dw_stage2: Option<(Vec<f32>, Vec<f32>)>, // (weights, biases)
}

/// Per-thread mutable state for a brain region. NOT Sync.
/// Contains noise plasticity and usefulness tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronLayerState {
    /// Per-neuron learned noise scale (mechanism 4: noise plasticity).
    pub noise_scale: Vec<f32>,
    /// EMA of per-neuron usefulness (activation consistency across ticks).
    pub usefulness_ema: Vec<f32>,
}

impl NeuronLayerState {
    pub fn new(config: &LayerConfig) -> Self {
        let n = config.n_neurons;
        Self {
            noise_scale: vec![1.0; n],
            usefulness_ema: vec![0.5; n],
        }
    }
}

// ─── Local Hebbian plasticity ───────────────────────────────

/// Fast online Hebbian plasticity within a neuron layer.
/// Tracks baseline statistics and corrects drift.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalHebbian {
    pub n_neurons: usize,
    pub lr: f32,
    pub momentum: f32,
    pub baseline_mean: Vec<f32>,
    pub baseline_var: Vec<f32>,
    pub running_mean: Vec<f32>,
    pub calibrated: bool,
}

impl LocalHebbian {
    pub fn new(n_neurons: usize, lr: f32) -> Self {
        Self {
            n_neurons, lr, momentum: 0.95,
            baseline_mean: vec![0.0; n_neurons],
            baseline_var: vec![1.0; n_neurons],
            running_mean: vec![0.0; n_neurons],
            calibrated: false,
        }
    }

    /// Calibrate from a batch of activations.
    pub fn calibrate(&mut self, activations: &[f32], batch_size: usize) {
        let n = self.n_neurons;
        for i in 0..n {
            let mut sum = 0.0;
            let mut sum2 = 0.0;
            for b in 0..batch_size {
                let v = activations[b * n + i];
                sum += v;
                sum2 += v * v;
            }
            let mean = sum / batch_size as f32;
            let var = (sum2 / batch_size as f32 - mean * mean).max(1e-6);
            self.baseline_mean[i] = mean;
            self.baseline_var[i] = var;
            self.running_mean[i] = mean;
        }
        self.calibrated = true;
    }

    /// Apply Hebbian correction to activations (online, every tick).
    pub fn correct(&mut self, activated: &mut [f32]) {
        if !self.calibrated { return; }
        let n = self.n_neurons;
        for i in 0..n {
            self.running_mean[i] = self.momentum * self.running_mean[i]
                + (1.0 - self.momentum) * activated[i];
            let drift = self.running_mean[i] - self.baseline_mean[i];
            let correction = -self.lr * drift / self.baseline_var[i].sqrt();
            activated[i] += correction;
        }
    }

    /// Homeostatic rebalancing during sleep.
    /// Neurons that chronically over-fire get their baseline RAISED
    /// (harder to activate next time). Neurons that never fire get
    /// their baseline LOWERED (easier to activate).
    /// This is the slow, offline anti-collapse mechanism.
    pub fn rebalance(&mut self) {
        if !self.calibrated { return; }
        let n = self.n_neurons;
        let global_mean: f32 = self.running_mean.iter().sum::<f32>() / n as f32;

        for i in 0..n {
            // How far is this neuron from the population average?
            let deviation = self.running_mean[i] - global_mean;

            // Shift baseline toward the running mean — but gently.
            // Over-active neurons: baseline moves up → harder to activate.
            // Under-active neurons: baseline moves down → easier to activate.
            self.baseline_mean[i] += 0.1 * deviation;

            // Also adjust variance: over-active neurons get higher variance
            // (need stronger signal to stand out), under-active get lower.
            let target_var = self.baseline_var[i] * (1.0 + 0.05 * deviation.abs());
            self.baseline_var[i] = 0.9 * self.baseline_var[i] + 0.1 * target_var;
            self.baseline_var[i] = self.baseline_var[i].max(0.01); // floor
        }
    }
}

// ─── Noise injection ──────────────────────────────────────

/// Biologically-grounded noise with causal structure.
///
/// Five mechanisms, each with a specific causal role:
///
/// **1. Correlated column noise** (gap junctions)
///   Nearby neurons share noise — models cortical column coupling.
///   70% shared column noise + 30% independent per-neuron noise.
///
/// **2. Vesicle depletion** (activity-dependent dropout)
///   Hot neurons fail more often (short-term synaptic depression).
///
/// **3. Homeostatic scaling** (stochastic resonance)
///   Quiet neurons get MORE noise (upregulation), active get LESS.
///
/// **4. Spontaneous firing** (baseline activity)
///   Silenced neurons still fire at baseline rate.
///
/// **5. Per-neuron learned noise scale** (noise plasticity)
///   Each neuron's noise level adapts based on its usefulness.
///   Useful neurons → precise. Irrelevant → noisy (explore).
const COLUMN_SIZE: usize = 4;

#[inline]
pub(crate) fn inject_noise(
    activations: &mut [f32],
    rng: &mut SimpleRng,
    base_amplitude: f32,
    base_dropout: f32,
    noise_scale: &[f32],         // per-neuron learned scale
    column_buf: &mut Vec<f32>,   // scratch for correlated noise
    sleep_phase: SleepPhase,
) {
    // Sleep phase modulates overall noise level
    let phase_scale = match sleep_phase {
        SleepPhase::Awake => 1.0,
        SleepPhase::Nrem  => 0.3,  // clean consolidation signal
        SleepPhase::Rem   => 2.0,  // creative exploration
    };

    // 1. Generate correlated column noise
    let n_columns = (activations.len() + COLUMN_SIZE - 1) / COLUMN_SIZE;
    if column_buf.len() < n_columns { column_buf.resize(n_columns, 0.0); }
    for c in 0..n_columns {
        column_buf[c] = rng.next_normal();
    }

    for (i, v) in activations.iter_mut().enumerate() {
        let abs_v = v.abs();

        // Per-neuron learned noise scale (mechanism 5 / noise plasticity)
        let neuron_scale = if i < noise_scale.len() { noise_scale[i] } else { 1.0 };

        // 2. Activity-dependent vesicle depletion
        let dropout = (base_dropout * (0.5 + abs_v)).min(0.8);

        if rng.next_f32() < dropout {
            // 4. Spontaneous firing: small baseline kick even when silenced
            *v = base_amplitude * phase_scale * neuron_scale * 0.3 * rng.next_normal();
        } else {
            // 3. Homeostatic scaling (stochastic resonance)
            let homeo_scale = if abs_v < 0.1 {
                3.0  // quiet neuron: amplify (upregulation)
            } else if abs_v > 0.5 {
                0.3  // hyperactive: reduce (downregulation)
            } else {
                1.0
            };

            // 1. Correlated column noise: 70% shared + 30% independent
            let column_idx = i / COLUMN_SIZE;
            let shared = column_buf[column_idx];
            let independent = rng.next_normal();
            let noise = 0.7 * shared + 0.3 * independent;

            *v += base_amplitude * phase_scale * homeo_scale * neuron_scale * noise;
            *v /= 1.0 - base_dropout;
        }
    }
}

// ─── Neuron Layer (brain region) ────────────────────────────

fn default_noise_scale() -> Vec<f32> { Vec::new() }
fn default_usefulness() -> Vec<f32> { Vec::new() }

/// A group of neurons forming a functional brain region.
/// This is the mutable version that holds both weights and per-neuron state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronLayer {
    pub config: LayerConfig,
    /// NLM: trace processor (per-neuron MLP chain)
    pub nlm_stage1: SuperLinear,
    pub nlm_stage2: Option<SuperLinear>,
    /// Start states
    pub start_trace: Vec<f32>,     // [n_neurons × memory_length]
    pub start_activated: Vec<f32>, // [n_neurons]
    /// Inhibitory mask: true = inhibitory (GABA), false = excitatory.
    /// Inhibitory neurons negate their output.
    pub inhibitory: Vec<bool>,
    /// Per-neuron learned noise scale (mechanism 4: noise plasticity).
    /// Useful neurons -> lower noise (reliable). Irrelevant -> higher noise (explore).
    /// Updated during sleep based on activation consistency.
    #[serde(default = "default_noise_scale")]
    pub noise_scale: Vec<f32>,
    /// EMA of per-neuron usefulness (activation consistency across ticks).
    #[serde(default = "default_usefulness")]
    pub usefulness_ema: Vec<f32>,
}

impl NeuronLayer {
    pub fn new(config: &LayerConfig) -> Self {
        let n = config.n_neurons;
        let m = config.memory_length;
        let h = config.memory_hidden_dims;

        // No GLU: output dims are direct (not 2x for gating)
        let (nlm_stage1, nlm_stage2) = if config.nlm_depth >= 2 {
            (SuperLinear::new(n, m, h), Some(SuperLinear::new(n, h, 1)))
        } else {
            (SuperLinear::new(n, m, 1), None)
        };

        let mut rng = SimpleRng::new((n * m) as u64);
        let scale_n = (1.0 / n as f32).sqrt();
        let scale_nm = (1.0 / (n + m) as f32).sqrt();
        let start_activated: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 * scale_n - scale_n).collect();
        let start_trace: Vec<f32> = (0..n * m).map(|_| rng.next_f32() * 2.0 * scale_nm - scale_nm).collect();

        // Mark last N neurons as inhibitory (GABA)
        let n_inhibitory = (n as f32 * config.inhibitory_fraction) as usize;
        let mut inhibitory = vec![false; n];
        for i in (n - n_inhibitory)..n {
            inhibitory[i] = true;
        }

        Self {
            config: config.clone(), nlm_stage1, nlm_stage2,
            start_trace, start_activated, inhibitory,
            noise_scale: vec![1.0; n],
            usefulness_ema: vec![0.5; n],
        }
    }

    /// One tick: update trace, apply NLM -> activated.
    /// trace: [n_neurons x memory_length], pre_activation: [n_neurons]
    /// Returns: (activated: [n_neurons], updated trace)
    pub fn step(&self, pre_activation: &[f32], trace: &mut Vec<f32>) -> Vec<f32> {
        let n = self.config.n_neurons;
        let m = self.config.memory_length;

        // Shift trace left, append new activation
        for neuron in 0..n {
            let off = neuron * m;
            for t in 0..m - 1 {
                trace[off + t] = trace[off + t + 1];
            }
            trace[off + m - 1] = pre_activation[neuron];
        }

        // NLM: SuperLinear + SiLU. No GLU (kills signal), no LayerNorm.
        let mut stage1_out = self.nlm_stage1.forward(trace);
        for v in &mut stage1_out {
            let sigmoid = 1.0 / (1.0 + (-*v).exp());
            *v *= sigmoid;
        }

        let mut activated = if let Some(ref stage2) = self.nlm_stage2 {
            let mut out = stage2.forward(&stage1_out);
            for v in &mut out {
                let sigmoid = 1.0 / (1.0 + (-*v).exp());
                *v *= sigmoid;
            }
            out
        } else {
            stage1_out
        };

        // Apply inhibitory mask
        for (i, &is_inhib) in self.inhibitory.iter().enumerate() {
            if is_inhib && i < activated.len() {
                activated[i] = -activated[i].abs();
            }
        }

        // Competitive inhibition (k-WTA): only top-k% excitatory neurons survive.
        if self.config.sparsity_target > 0.0 && self.config.sparsity_target < 1.0 {
            let n_excitatory = activated.len() - self.inhibitory.iter().filter(|&&b| b).count();
            let k = ((1.0 - self.config.sparsity_target) * n_excitatory as f32) as usize;
            if k > 0 && k < n_excitatory {
                let mut excitatory_vals: Vec<f32> = activated.iter().enumerate()
                    .filter(|(i, _)| !self.inhibitory.get(*i).copied().unwrap_or(false))
                    .map(|(_, &v)| v.abs())
                    .collect();
                excitatory_vals.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
                let threshold = excitatory_vals.get(k).copied().unwrap_or(0.0);
                for (i, v) in activated.iter_mut().enumerate() {
                    if !self.inhibitory.get(i).copied().unwrap_or(false) && v.abs() < threshold {
                        *v *= 0.1; // soft suppression, not hard zero
                    }
                }
            }
        }

        // ANTI-COLLAPSE: soft normalization that preserves input structure.
        let max_abs: f32 = activated.iter()
            .map(|x| x.abs())
            .fold(0.0f32, f32::max)
            .max(0.1);
        for v in &mut activated {
            *v /= max_abs;
        }

        activated
    }

    /// Compute firing phase for each neuron given activations.
    /// Strong activation fires early (phase~0). Weak fires late (phase~1).
    /// Returns [n_neurons] phases in [0, 1].
    #[inline]
    pub fn compute_phase(activated: &[f32]) -> Vec<f32> {
        activated.iter().map(|&v| {
            let sigmoid = 1.0 / (1.0 + (v.abs() * 3.0).exp());
            sigmoid
        }).collect()
    }

    /// Extract the immutable weights portion.
    pub fn to_weights(&self) -> NeuronLayerWeights {
        NeuronLayerWeights {
            config: self.config.clone(),
            nlm_stage1: self.nlm_stage1.clone(),
            nlm_stage2: self.nlm_stage2.clone(),
            start_trace: self.start_trace.clone(),
            start_activated: self.start_activated.clone(),
            inhibitory: self.inhibitory.clone(),
        }
    }

    /// Extract the mutable state portion.
    pub fn to_state(&self) -> NeuronLayerState {
        NeuronLayerState {
            noise_scale: self.noise_scale.clone(),
            usefulness_ema: self.usefulness_ema.clone(),
        }
    }

    /// Reconstruct a NeuronLayer from weights + state.
    pub fn from_split(weights: &NeuronLayerWeights, state: &NeuronLayerState) -> Self {
        Self {
            config: weights.config.clone(),
            nlm_stage1: weights.nlm_stage1.clone(),
            nlm_stage2: weights.nlm_stage2.clone(),
            start_trace: weights.start_trace.clone(),
            start_activated: weights.start_activated.clone(),
            inhibitory: weights.inhibitory.clone(),
            noise_scale: state.noise_scale.clone(),
            usefulness_ema: state.usefulness_ema.clone(),
        }
    }
}
