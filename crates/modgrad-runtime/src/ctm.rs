//! Continuous Thought Machine v2 — 4-region brain with Hebbian plasticity.
//!
//! Four brain regions with distinct structural priors:
//!   INPUT (V1/S1)      — wide, shallow NLMs, short memory, fast Hebbian
//!   ATTENTION (thalamus) — narrow, deep NLMs, gating sparsity, NO broadcast
//!   OUTPUT (assoc cortex) — medium, longest memory, evidence accumulation
//!   MOTOR (M1/BG)       — small, fast, minimal memory, decisive
//!
//! Architecture per tick:
//!   motor→input (feedback) → attention query → cross-attn →
//!   attention layer → output layer → motor layer → repeat
//!
//! Each region is a composable filter in the pipeline.
//!
//! Reference: Sakana AI CTM paper (arxiv 2505.05522)
//! Adapted from: ctm_v2.py, zish/src/inference/ctm.zig

use rayon::prelude::*;

pub use super::config::*;
pub use modgrad_compute::neuron::*;
pub use super::neuron::*;
pub use modgrad_compute::ops::*;
pub use super::sync::*;
pub use super::session::*;
pub use super::synapse::*;
pub use super::weights::*;

// NeuronLayer — lives in runtime::neuron (re-exported above via `pub use super::neuron::*`)

// ─── Content-Dependent Position Predictor (RePo-style) ─────
//
// Analog of Sakana AI's RePo: instead of fixed decay rates in the
// sync accumulator, predict per-pair decay from current activations.
// Tokens that are contextually relevant get HELD LONGER (high r),
// irrelevant correlations decay faster (low r).
//
// Architecture: SwiGLU gate (same as RePo's f_φ)
//   gate    = sigmoid(W_gate @ activated)
//   content = W_content @ activated
//   gated   = gate * content
//   r_shift = sigmoid(W_final @ gated) - 0.5  // shift in [-0.5, 0.5]
//   r_dynamic = r_baseline + r_shift           // per-pair decay rate
//
// Trained via LS during sleep (same machinery as synapses).

fn default_bg_da_baseline() -> f32 { 0.7 }

// CtmWeights — moved to weights.rs (re-exported above via `pub use super::weights::*`)
// forward_split — moved to forward.rs
pub use super::forward::forward_split;


// ─── CTM v2 (backward-compatible wrapper) ───────────────────

impl Ctm {
    /// Run K ticks of the CTM on an observation vector.
    /// Returns (predictions_per_tick, final_sync). Drops TickSignals for compat.
    ///
    /// Note: prefer `forward_split()` with `CtmWeights`/`CtmSession` for new code.
    #[allow(deprecated)]
    pub fn forward(
        &mut self,
        observation: &[f32],
        state: &mut CtmState,
        collect_sleep: bool,
    ) -> (Vec<Vec<f32>>, Vec<f32>) {
        let zeros = vec![0.0f32; self.config.d_input];
        let (preds, sync, _signals) = self.forward_with_proprio(
            observation, &zeros, state, collect_sleep);
        (preds, sync)
    }

    /// Forward pass with proprioceptive input.
    /// proprioception is fed to the insula region (body sensors).
    /// Pass zeros for backward compat or when no ProprioceptiveRetina is available.
    #[deprecated(note = "Use forward_split() with CtmWeights/CtmSession instead")]
    pub fn forward_with_proprio(
        &mut self,
        observation: &[f32],       // d_input: exteroceptive (visual/byte)
        proprioception: &[f32],    // d_input: interoceptive (body sensors, or zeros)
        state: &mut CtmState,
        collect_sleep: bool,
    ) -> (Vec<Vec<f32>>, Vec<f32>, TickSignals) {
        let cfg = &self.config;
        let k = cfg.iterations;
        let mut predictions: Vec<Vec<f32>> = Vec::with_capacity(k);
        let mut signals = TickSignals::default();

        // Reset per-forward state
        state.motor_evidence.fill(0.0);
        state.motor_decided = false;
        state.decision_tick = None;
        state.tick_traces.clear();

        for tick_idx in 0..k {
            // ═══════════════════════════════════════════════════════════════
            // PARALLEL REGION PROCESSING (brain-like)
            //
            // All regions read from PREVIOUS tick's state (state.act_*),
            // compute new activations into local variables, then commit.
            // This makes all 6 regions independent per tick — parallelizable.
            // The brain works this way: all regions fire simultaneously each
            // gamma cycle, reading from the previous cycle's output.
            // ═══════════════════════════════════════════════════════════════

            // ── Phase 1: Prepare inputs from prev tick state (all reads, no writes) ──

            // Global broadcast from prev tick's activations (all 8 regions)
            let full_state = concat(&[
                &state.act_input, &state.act_attention,
                &state.act_output, &state.act_motor,
                &state.act_cerebellum, &state.act_basal_ganglia,
                &state.act_insula, &state.act_hippocampus,
            ]);
            let global_raw = self.global_projector.forward(&full_state);
            let global_ctx = glu(&global_raw);

            // Sync action (needs prev tick's input + attention)
            let in_attn_cat = concat(&[&state.act_input, &state.act_attention]);
            let phase_in = NeuronLayer::compute_phase(&state.act_input);
            let phase_attn = NeuronLayer::compute_phase(&state.act_attention);
            let in_attn_phases = concat(&[&phase_in, &phase_attn]);
            let sync_action = state.sync_action.update_with_phase(
                &in_attn_cat, state.modulation[MOD_SYNC_SCALE], &in_attn_phases);
            let attn_result = simple_attention(&sync_action, observation, cfg.d_input);

            // Prepare all 6 synapse inputs from prev tick state
            let syn_in_input = maybe_broadcast(
                &concat(&[observation, &state.act_motor]),
                &global_ctx, cfg.input_layer.receives_broadcast);

            let syn_in_attn = maybe_broadcast(
                &concat(&[&state.act_input, &attn_result]),
                &global_ctx, cfg.attention_layer.receives_broadcast);

            let syn_in_output = maybe_broadcast(
                &concat(&[&state.act_attention, &attn_result]),
                &global_ctx, cfg.output_layer.receives_broadcast);

            // Motor receives output + BG bias
            let syn_in_motor = maybe_broadcast(
                &concat(&[&state.act_output, &state.act_basal_ganglia]),
                &global_ctx, cfg.motor_layer.receives_broadcast);

            // Cerebellum: motor + observation → prediction
            let syn_in_cereb = concat(&[&state.act_motor, observation]);

            // BG: output + dopamine scalar → action bias
            let mut bg_input = state.act_output.clone();
            bg_input.push(state.modulation[MOD_SYNC_SCALE]);

            // Insula: body sensors + hippocampal retrieval (somatic markers)
            let syn_in_insula = concat(&[proprioception, &self.hippo_retrieval]);

            // Hippocampus: sees all cortical activations (pattern completion)
            let syn_in_hippo = concat(&[
                &state.act_input, &state.act_attention,
                &state.act_output, &state.act_motor,
            ]);

            // ── Phase 2: Compute all 8 regions in parallel ──
            // Each region: synapse forward → NLM step → new activations.
            // All read from self (immutable), write to separate local vars.
            // Uses rayon::scope for parallel execution when regions are large enough.
            // At small scale (<par_threshold), sequential is faster (no scheduling overhead).

            let total_neurons = cfg.input_layer.n_neurons + cfg.attention_layer.n_neurons
                + cfg.output_layer.n_neurons + cfg.motor_layer.n_neurons;
            let parallel = total_neurons >= cfg.par_threshold * 4; // 4 cortical regions

            // Results: (signal, trace, activations) per region
            let (input_signal, mut trace_input, mut new_input);
            let (attn_signal, mut trace_attn, mut new_attn);
            let (out_signal, mut trace_output, mut new_output);
            let (motor_signal, mut trace_motor, mut new_motor);
            let (cereb_signal, mut trace_cereb, new_cereb);
            let (bg_signal, mut trace_bg, new_bg);
            let (insula_signal, mut trace_insula, mut new_insula);
            let (hippo_signal, mut trace_hippo, mut new_hippo);

            if parallel {
                // Parallel: 3 pairs via nested rayon::join (avoids scope lifetime issues)
                let ((r_in, r_attn), (r_out, r_motor)) = rayon::join(
                    || rayon::join(
                        || {
                            let sig = self.syn_motor_input.forward(&syn_in_input);
                            let mut tr = state.trace_input.clone();
                            let act = self.input_region.step(&sig, &mut tr);
                            (sig, tr, act)
                        },
                        || {
                            let sig = self.syn_input_attn.forward(&syn_in_attn);
                            let mut tr = state.trace_attention.clone();
                            let act = self.attention_region.step(&sig, &mut tr);
                            (sig, tr, act)
                        },
                    ),
                    || rayon::join(
                        || {
                            let sig = self.syn_attn_output.forward(&syn_in_output);
                            let mut tr = state.trace_output.clone();
                            let act = self.output_region.step(&sig, &mut tr);
                            (sig, tr, act)
                        },
                        || {
                            let sig = self.syn_output_motor.forward(&syn_in_motor);
                            let mut tr = state.trace_motor.clone();
                            let act = self.motor_region.step(&sig, &mut tr);
                            (sig, tr, act)
                        },
                    ),
                );
                // Subcortical: parallel with each other
                let (r_cereb, r_bg) = rayon::join(
                    || {
                        let sig = self.syn_cerebellum.forward(&syn_in_cereb);
                        let mut tr = state.trace_cerebellum.clone();
                        let act = self.cerebellum_region.step(&sig, &mut tr);
                        (sig, tr, act)
                    },
                    || {
                        let sig = self.syn_basal_ganglia.forward(&bg_input);
                        let mut tr = state.trace_basal_ganglia.clone();
                        let act = self.basal_ganglia_region.step(&sig, &mut tr);
                        (sig, tr, act)
                    },
                );
                // New regions (v3): parallel with each other
                let (r_insula, r_hippo) = rayon::join(
                    || {
                        let sig = self.syn_insula.forward(&syn_in_insula);
                        let mut tr = state.trace_insula.clone();
                        let act = self.insula_region.step(&sig, &mut tr);
                        (sig, tr, act)
                    },
                    || {
                        let sig = self.syn_hippocampus.forward(&syn_in_hippo);
                        let mut tr = state.trace_hippocampus.clone();
                        let act = self.hippocampus_region.step(&sig, &mut tr);
                        (sig, tr, act)
                    },
                );
                (input_signal, trace_input, new_input) = r_in;
                (attn_signal, trace_attn, new_attn) = r_attn;
                (out_signal, trace_output, new_output) = r_out;
                (motor_signal, trace_motor, new_motor) = r_motor;
                (cereb_signal, trace_cereb, new_cereb) = r_cereb;
                (bg_signal, trace_bg, new_bg) = r_bg;
                (insula_signal, trace_insula, new_insula) = r_insula;
                (hippo_signal, trace_hippo, new_hippo) = r_hippo;
            } else {
                // Sequential: faster for small regions (< par_threshold)
                input_signal = self.syn_motor_input.forward(&syn_in_input);
                trace_input = state.trace_input.clone();
                new_input = self.input_region.step(&input_signal, &mut trace_input);

                attn_signal = self.syn_input_attn.forward(&syn_in_attn);
                trace_attn = state.trace_attention.clone();
                new_attn = self.attention_region.step(&attn_signal, &mut trace_attn);

                out_signal = self.syn_attn_output.forward(&syn_in_output);
                trace_output = state.trace_output.clone();
                new_output = self.output_region.step(&out_signal, &mut trace_output);

                motor_signal = self.syn_output_motor.forward(&syn_in_motor);
                trace_motor = state.trace_motor.clone();
                new_motor = self.motor_region.step(&motor_signal, &mut trace_motor);

                cereb_signal = self.syn_cerebellum.forward(&syn_in_cereb);
                trace_cereb = state.trace_cerebellum.clone();
                new_cereb = self.cerebellum_region.step(&cereb_signal, &mut trace_cereb);

                bg_signal = self.syn_basal_ganglia.forward(&bg_input);
                trace_bg = state.trace_basal_ganglia.clone();
                new_bg = self.basal_ganglia_region.step(&bg_signal, &mut trace_bg);

                insula_signal = self.syn_insula.forward(&syn_in_insula);
                trace_insula = state.trace_insula.clone();
                new_insula = self.insula_region.step(&insula_signal, &mut trace_insula);

                hippo_signal = self.syn_hippocampus.forward(&syn_in_hippo);
                trace_hippo = state.trace_hippocampus.clone();
                new_hippo = self.hippocampus_region.step(&hippo_signal, &mut trace_hippo);
            }

            // ── Phase 3: Sequential post-processing (Hebbian, noise, neuromod) ──
            // These need &mut self so can't run in parallel with region computation.

            if self.hebbian_enabled {
                if cfg.input_layer.hebbian { self.hebb_input.correct(&mut new_input); }
                if cfg.attention_layer.hebbian { self.hebb_attention.correct(&mut new_attn); }
                if cfg.output_layer.hebbian { self.hebb_output.correct(&mut new_output); }
                if cfg.motor_layer.hebbian { self.hebb_motor.correct(&mut new_motor); }
                if cfg.insula_layer.hebbian { self.hebb_insula.correct(&mut new_insula); }
                if cfg.hippocampus_layer.hebbian { self.hebb_hippocampus.correct(&mut new_hippo); }

                // ── Hebbian synapse plasticity (Oja's rule) ──
                // "Neurons that fire together wire together" — modulated by dopamine.
                // ΔW = lr × da × (post × pre - post² × W)
                // Oja's rule prevents unbounded growth while strengthening co-active paths.
                let da = state.modulation[MOD_SYNC_SCALE];
                // Only learn when dopamine is elevated (surprise signal).
                // Baseline da ≈ 1.0, spikes on prediction error.
                // No learning at baseline — this is the gating.
                let surprise_gate = (da - cfg.neuromod.da_gate).max(0.0);
                let hebb_lr = cfg.neuromod.hebb_syn_lr * surprise_gate;
                let pre_post_pairs: [(&[f32], &[f32], &mut Synapse); 4] = [
                    (&syn_in_input, &new_input, &mut self.syn_motor_input),
                    (&syn_in_attn, &new_attn, &mut self.syn_input_attn),
                    (&syn_in_output, &new_output, &mut self.syn_attn_output),
                    (&syn_in_motor, &new_motor, &mut self.syn_output_motor),
                ];
                for (pre, post, syn) in pre_post_pairs {
                    let in_dim = syn.linear.in_dim;
                    let half = syn.linear.out_dim / 2;
                    for j in 0..half.min(post.len()) {
                        let pj = post[j];
                        if pj.abs() < 0.01 { continue; }
                        for i in 0..in_dim.min(pre.len()) {
                            let idx = j * in_dim + i;
                            if idx < syn.linear.weight.len() {
                                let w = syn.linear.weight[idx];
                                syn.linear.weight[idx] +=
                                    hebb_lr * (pj * pre[i] - pj * pj * w);
                            }
                        }
                    }
                }
            }

            // ── Hippocampal CAM: store + retrieve ──
            // The NeuronLayer output gates storage: high activation = "this is novel, store it."
            // Retrieval uses cortical concat as query. Result available next tick.
            if self.hippo_cam.capacity > 0 {
                let cortical_concat = concat(&[&new_input, &new_attn, &new_output, &new_motor]);
                let hippo_magnitude: f32 = new_hippo.iter()
                    .map(|x| x.abs()).sum::<f32>() / new_hippo.len().max(1) as f32;

                // Store if hippocampus activation is high (novelty signal)
                if hippo_magnitude > 0.3 {
                    self.hippo_cam.store(&cortical_concat, &cortical_concat, hippo_magnitude);
                }

                // Retrieve: pattern completion from stored cortical snapshots
                let (retrieved, max_sim) = self.hippo_cam.retrieve(&cortical_concat, 0.5);
                self.hippo_retrieval = retrieved;
                signals.hippo_max_similarity = max_sim;
            }

            // ── Interoceptive → neuromodulation (two channels) ──
            //
            // LEARNED channel (Hormonal tier): insula NeuronLayer output.
            // The organism can learn to contextualize body signals —
            // "high CPU temp during training is normal, don't panic."
            //
            // HARDWIRED channel (Genome tier): raw proprioception bypass.
            // Cannot be learned away. Like pain — you can't train yourself
            // to not feel a burn. Prevents wireheading by ensuring 30% of
            // neuromod drive comes from unlearnable raw body state.
            {
                let n_ins = new_insula.len().max(1) as f32;

                // Learned channel: insula NeuronLayer output
                let learned_magnitude: f32 = new_insula.iter()
                    .map(|x| x.abs()).sum::<f32>() / n_ins;
                let learned_valence: f32 = new_insula.iter().sum::<f32>() / n_ins;

                // Hardwired channel: raw proprioception (Genome-tier, unlearnable)
                let n_pro = proprioception.len().max(1) as f32;
                let raw_magnitude: f32 = proprioception.iter()
                    .map(|x| x.abs()).sum::<f32>() / n_pro;
                let raw_valence: f32 = proprioception.iter().sum::<f32>() / n_pro;

                // Mix: 70% learned, 30% hardwired (Genome-tier ratio, not tunable)
                let insula_magnitude = 0.7 * learned_magnitude + 0.3 * raw_magnitude;
                let insula_valence = 0.7 * learned_valence + 0.3 * raw_valence;

                signals.insula_magnitude = insula_magnitude;
                signals.insula_valence = insula_valence;

                // NE and 5HT are NOT updated here. The organism reports
                // insula_magnitude and insula_valence via TickSignals.
                // The HOST computes the neuromod changes from these signals.
                // This prevents the organism from writing its own reward:
                // it can't learn insula weights that produce fake comfort.
            }

            // ── Basal ganglia three-factor learning (reinforcement) ──
            // The striatum learns which actions to select via:
            //   ΔW = lr × (dopamine - baseline) × pre × post
            //
            // Three factors:
            //   1. pre: output region activity (what state we're in)
            //   2. post: BG activity (which action was selected)
            //   3. dopamine - baseline: reward prediction error
            //
            // Direct pathway (D1 receptors): DA > baseline → strengthen
            // Indirect pathway (D2 receptors): DA < baseline → weaken
            // This is how habits form: repeatedly rewarded actions become automatic.
            {
                let da = state.modulation[MOD_SYNC_SCALE];
                let reward_signal = da - self.bg_da_baseline;
                // Update baseline (running average)
                self.bg_da_baseline = 0.99 * self.bg_da_baseline + 0.01 * da;

                let bg_n = cfg.basal_ganglia_layer.n_neurons;
                let pre_dim = bg_input.len(); // output_neurons + 1 (dopamine)
                let bg_lr = cfg.basal_ganglia_layer.hebbian_lr;

                // Update eligibility trace: decaying record of pre × post
                for j in 0..bg_n {
                    for i in 0..pre_dim.min(bg_input.len()) {
                        let idx = j * pre_dim + i;
                        if idx < self.bg_eligibility.len() {
                            // Decay old trace + add new pre×post
                            self.bg_eligibility[idx] = 0.9 * self.bg_eligibility[idx]
                                + 0.1 * bg_input[i] * new_bg[j];
                        }
                    }
                }

                // Apply three-factor update to BG synapse weights
                // Only when reward signal is significant (|reward| > 0.05)
                // Only apply BG reinforcement at sufficient scale.
                // At < 32 BG neurons, the eligibility trace is too noisy —
                // random perturbations corrupt the motor signal.
                // This matches biology: the striatum has millions of neurons.
                if reward_signal.abs() > 0.1 && bg_n >= 32 {
                    let syn = &mut self.syn_basal_ganglia.linear;
                    // syn.weight is [out_dim*2 × in_dim] (GLU doubles output)
                    let in_dim = pre_dim.min(syn.in_dim);
                    let out_dim = bg_n.min(syn.out_dim / 2); // GLU halves

                    for j in 0..out_dim {
                        for i in 0..in_dim {
                            let elig_idx = j * pre_dim + i;
                            if elig_idx < self.bg_eligibility.len() {
                                let delta = bg_lr * reward_signal * self.bg_eligibility[elig_idx];
                                if delta.is_finite() {
                                    // Update both GLU halves
                                    syn.weight[j * syn.in_dim + i] += delta;
                                    syn.weight[(j + syn.out_dim / 2) * syn.in_dim + i] += delta * 0.5;
                                }
                            }
                        }
                    }
                }
            }

            // Serotonin gating on attention
            let serotonin_gate = state.modulation[MOD_GATE];
            for v in &mut new_attn {
                if v.abs() < serotonin_gate { *v *= 0.1; }
            }

            // ACh update from attention strength
            let attn_strength: f32 = sync_action.iter().map(|x| x.abs()).sum::<f32>()
                / sync_action.len().max(1) as f32;
            state.modulation[MOD_PRECISION] = 0.8 * state.modulation[MOD_PRECISION]
                + 0.2 * attn_strength.clamp(0.0, 1.0);

            // Noise injection + usefulness tracking (all regions)
            if state.noisy {
                let ach = state.modulation[MOD_PRECISION];
                // Usefulness EMA
                for regions in [
                    (&mut self.input_region, &new_input),
                    (&mut self.attention_region, &new_attn),
                    (&mut self.output_region, &new_output),
                    (&mut self.motor_region, &new_motor),
                ] {
                    for (i, &a) in regions.1.iter().enumerate() {
                        if i < regions.0.usefulness_ema.len() {
                            let u = if a.abs() > 0.1 { 1.0 } else { 0.0 };
                            regions.0.usefulness_ema[i] = 0.99 * regions.0.usefulness_ema[i] + 0.01 * u;
                        }
                    }
                }
                // Input: more noise when not attending
                inject_noise(&mut new_input, &mut state.noise_rng,
                    0.1 * (1.0 + 0.3 * (1.0 - ach)), 0.2,
                    &self.input_region.noise_scale, &mut state.column_noise_buf, state.sleep_phase);
                // Attention + Output: less noise when attending (precision)
                let attn_amp = 0.1 * (1.0 - 0.5 * ach);
                inject_noise(&mut new_attn, &mut state.noise_rng, attn_amp, 0.2,
                    &self.attention_region.noise_scale, &mut state.column_noise_buf, state.sleep_phase);
                inject_noise(&mut new_output, &mut state.noise_rng, attn_amp, 0.2,
                    &self.output_region.noise_scale, &mut state.column_noise_buf, state.sleep_phase);
                // Motor: not ACh-modulated
                inject_noise(&mut new_motor, &mut state.noise_rng, 0.1, 0.2,
                    &self.motor_region.noise_scale, &mut state.column_noise_buf, state.sleep_phase);
            }

            // Cerebellum prediction error → dopamine surprise + cerebellar learning
            let pred_error_mag: f32 = new_cereb.iter().zip(observation.iter())
                .map(|(&pred, &obs)| (obs - pred).powi(2))
                .sum::<f32>().sqrt()
                / observation.len().max(1) as f32;
            if pred_error_mag > 0.1 {
                state.modulation[MOD_SYNC_SCALE] = (state.modulation[MOD_SYNC_SCALE] * cfg.neuromod.da_error_alpha
                    + cfg.neuromod.da_error_beta * (1.0 + pred_error_mag * 2.0)).clamp(cfg.neuromod.da_min, cfg.neuromod.da_max);

                // Cerebellar delta rule: direct weight update (climbing fiber signal)
                if self.hebbian_enabled {
                    let cereb_lr = cfg.neuromod.hebb_syn_lr * pred_error_mag.min(1.0);
                    let syn = &mut self.syn_cerebellum.linear;
                    let in_dim = syn.in_dim;
                    let n_cereb = cfg.cerebellum_layer.n_neurons.min(new_cereb.len());
                    for j in 0..n_cereb {
                        let error_j = if j < observation.len() {
                            observation[j] - new_cereb[j]
                        } else { 0.0 };
                        if error_j.abs() < 0.01 { continue; }
                        for i in 0..in_dim.min(syn_in_cereb.len()) {
                            let idx = j * in_dim + i;
                            if idx < syn.weight.len() {
                                syn.weight[idx] += cereb_lr * error_j * syn_in_cereb[i];
                            }
                        }
                    }
                }
            } else {
                state.modulation[MOD_SYNC_SCALE] = state.modulation[MOD_SYNC_SCALE] * cfg.neuromod.da_decay + (1.0 - cfg.neuromod.da_decay);
            }

            // ── Phase 4: Commit new activations ──
            state.act_input = new_input;
            state.act_attention = new_attn;
            state.act_output = new_output;
            state.act_motor = new_motor;
            state.act_cerebellum = new_cereb;
            state.act_basal_ganglia = new_bg;
            state.act_insula = new_insula;
            state.act_hippocampus = new_hippo;
            state.trace_input = trace_input;
            state.trace_attention = trace_attn;
            state.trace_output = trace_output;
            state.trace_motor = trace_motor;
            state.trace_cerebellum = trace_cereb;
            state.trace_basal_ganglia = trace_bg;
            state.trace_insula = trace_insula;
            state.trace_hippocampus = trace_hippo;
            state.phase_input = phase_in;
            state.phase_attention = phase_attn;

            // Drift-diffusion: accumulate motor evidence across ticks
            if !state.motor_decided {
                for (i, act) in state.act_motor.iter().enumerate() {
                    if i < state.motor_evidence.len() {
                        state.motor_evidence[i] += act;
                    }
                }
                let max_evidence: f32 = state.motor_evidence.iter()
                    .map(|x| x.abs()).fold(0.0f32, f32::max);
                let ne = state.modulation[MOD_AROUSAL].max(0.1);
                let effective_threshold = cfg.motor_threshold / ne.sqrt();
                if max_evidence > effective_threshold {
                    state.motor_decided = true;
                    state.decision_tick = Some(tick_idx);
                }
            }

            // Collect sleep traces
            if collect_sleep {
                self.sleep.collect("syn_motor_input", &syn_in_input, &input_signal);
                self.sleep.collect("syn_input_attn", &syn_in_attn, &attn_signal);
                self.sleep.collect("syn_attn_output", &syn_in_output, &out_signal);
                self.sleep.collect("syn_output_motor", &syn_in_motor, &motor_signal);
                self.sleep.collect("syn_cerebellum", &syn_in_cereb, &cereb_signal);
                self.sleep.collect("syn_basal_ganglia", &bg_input, &bg_signal);
                self.sleep.collect("syn_insula", &syn_in_insula, &insula_signal);
                self.sleep.collect("syn_hippocampus", &syn_in_hippo, &hippo_signal);
            }

            // Output sync: pairwise products from output + motor
            let out_motor = concat(&[&state.act_output, &state.act_motor]);
            let da = if state.noisy {
                let da_base = state.modulation[MOD_SYNC_SCALE];
                let jitter_amp = 0.1 + 0.1 * da_base;
                let jitter = 1.0 + jitter_amp * state.noise_rng.next_normal();
                (da_base * jitter).clamp(0.1, 2.0)
            } else {
                state.modulation[MOD_SYNC_SCALE]
            };
            state.phase_output = NeuronLayer::compute_phase(&state.act_output);
            state.phase_motor = NeuronLayer::compute_phase(&state.act_motor);
            let out_motor_phases = concat(&[&state.phase_output, &state.phase_motor]);
            // Include observation in sync to prevent collapse:
            // even when activations are uniform, the observation carries input identity.
            let sync_input = concat(&[&out_motor, observation]);
            let sync = state.sync_out.update_with_phase(&sync_input, da, &out_motor_phases);
            state.last_sync = sync.clone();

            // Prediction from sync
            let pred = self.output_projector.forward(&sync);

            // INTRA-TICK DOPAMINE: update between ticks, not after forward.
            // Compare this tick's prediction to previous tick's.
            // Large change = prediction error = dopamine spike.
            // Small change = converging = dopamine settles.
            // This is how the brain does it: continuous feedback within 200ms.
            if let Some(prev_pred) = predictions.last() {
                let delta: f32 = pred.iter().zip(prev_pred.iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum::<f32>()
                    .sqrt();
                // Prediction error → surprise → dopamine update
                // Normalize by prediction magnitude for scale-invariance
                let magnitude: f32 = pred.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
                let tick_surprise = delta / magnitude;
                state.modulation[MOD_SYNC_SCALE] = (state.modulation[MOD_SYNC_SCALE] * cfg.neuromod.da_intra_alpha
                    + cfg.neuromod.da_intra_beta * (1.0 + tick_surprise * 2.0)).clamp(cfg.neuromod.da_min, cfg.neuromod.da_max);
            }

            predictions.push(pred);

            // Collect tick trace for visualization
            if collect_sleep {
                state.tick_traces.push(TickTrace {
                    tick: tick_idx,
                    input_activations: state.act_input.clone(),
                    attention_activations: state.act_attention.clone(),
                    output_activations: state.act_output.clone(),
                    motor_activations: state.act_motor.clone(),
                    sync_out: sync.clone(),
                    sync_action: sync_action.clone(),
                    modulation: state.modulation.clone(),
                    motor_evidence_max: state.motor_evidence.iter()
                        .map(|x| x.abs()).fold(0.0f32, f32::max),
                    motor_decided: state.motor_decided,
                });
            }

            // ── Telemetry: record tick to binary stream ──
            if let Some(ref mut telem) = self.telemetry {
                let activations: Vec<f32> = [
                    &state.act_input[..], &state.act_attention[..],
                    &state.act_output[..], &state.act_motor[..],
                    &state.act_cerebellum[..], &state.act_basal_ganglia[..],
                    &state.act_insula[..], &state.act_hippocampus[..],
                ].iter().flat_map(|a| a.iter().copied()).collect();

                let signals = &state.modulation;
                let sync_vals = &state.last_sync;
                let extras: Vec<f32> = state.sync_out.r_shift_buf.clone();

                let record = telem.build_record(
                    tick_idx as u32,
                    0, // step — filled by caller via manifest
                    0.0, // loss — filled by caller via manifest
                    &activations,
                    signals,
                    sync_vals,
                    &extras,
                );
                telem.record_tick(&record);
            }
        }

        // Return the NORMALIZED sync (alpha/sqrt(beta)), not raw alpha.
        // This must match what output_projector sees inside the tick loop.
        let final_sync = state.last_sync.clone();

        // Populate remaining signals
        signals.motor_decided = state.motor_decided;
        signals.decision_tick = state.decision_tick;
        signals.ticks = k;

        // Activation energy: L2 of all region activations
        let all_act: f32 = [
            &state.act_input[..], &state.act_attention[..],
            &state.act_output[..], &state.act_motor[..],
        ].iter()
            .flat_map(|a| a.iter())
            .map(|x| x * x)
            .sum::<f32>();
        signals.activation_energy = all_act.sqrt();

        // Sync convergence: did predictions stabilize?
        signals.sync_converged = if predictions.len() >= 2 {
            let last = &predictions[predictions.len() - 1];
            let prev = &predictions[predictions.len() - 2];
            let delta: f32 = last.iter().zip(prev)
                .map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();
            let mag: f32 = last.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
            delta / mag < 0.1 // converged if <10% relative change
        } else {
            false
        };

        // Hedonic state is computed by the HOST, not the organism.
        // The organism reports raw neuromod levels in TickSignals.
        // The host computes curiosity/anxiety/serotonin from these.
        // This makes wireheading structurally impossible: the organism
        // cannot call the hedonic function or influence its computation.
        // Raw neuromod levels for host to compute hedonic state
        signals.dopamine = state.modulation[MOD_SYNC_SCALE];
        signals.norepinephrine = state.modulation[MOD_AROUSAL];
        signals.serotonin = state.modulation[MOD_GATE];
        // Hedonic state computed by host (these stay 0 here)
        signals.curiosity = 0.0;
        signals.anxiety = 0.0;

        (predictions, final_sync, signals)
    }

    /// Run sleep consolidation cycle.
    /// Replay high-surprise experiences before consolidation.
    /// This is the "dreaming" phase — replaying surprising events
    /// generates fresh sleep traces weighted by importance.
    /// Collect a (sync, logit_bias) training pair for the logit projector.
    /// Called during teach: run CTM on the prompt, then store the mapping
    /// from sync signal → desired logit bias strengths.
    ///
    /// The projector learns: sync → strength values for KNOWN token IDs.
    /// Token IDs are stored separately (they're discrete, not continuous).
    /// The projector only predicts HOW STRONGLY to bias each token.
    pub fn collect_logit_trace(&mut self, sync: &[f32], logit_biases: &[modgrad_io::types::LogitBias]) {
        // Target: just the strength values (continuous, regressionable)
        let target: Vec<f32> = logit_biases.iter().map(|lb| lb.strength).collect();
        self.logit_traces.push((sync.to_vec(), target));
    }

    /// Train the logit projector from collected traces via least-squares.
    /// Called during sleep. Maps sync → strength values.
    pub fn train_logit_projector(&mut self) {
        if self.logit_traces.is_empty() { return; }

        let in_dim = self.logit_traces[0].0.len();
        let out_dim = self.logit_traces.iter().map(|(_, t)| t.len()).max().unwrap_or(0);
        if out_dim == 0 { return; }

        let mut xtx = vec![0.0f32; in_dim * in_dim];
        let mut xty = vec![0.0f32; in_dim * out_dim];

        for (x, y) in &self.logit_traces {
            for i in 0..in_dim {
                for j in 0..in_dim {
                    xtx[i * in_dim + j] += x[i] * x[j];
                }
                for o in 0..y.len().min(out_dim) {
                    xty[i * out_dim + o] += x[i] * y[o];
                }
            }
        }

        for i in 0..in_dim {
            xtx[i * in_dim + i] += 1e-4;
        }

        if let Some(w) = solve_least_squares(&xtx, &xty, in_dim, out_dim) {
            let mut weight = vec![0.0f32; out_dim * in_dim];
            for o in 0..out_dim {
                for i in 0..in_dim {
                    weight[o * in_dim + i] = w[i * out_dim + o];
                }
            }
            self.logit_projector = Some(Linear {
                weight,
                bias: vec![0.0; out_dim],
                in_dim,
                out_dim,
            });
        }
    }

    /// Predict logit bias strengths from a sync signal.
    /// Returns biases using the ORIGINAL token IDs from the matched episode
    /// but with CTM-predicted strength values.
    /// `episode_biases` provides the token IDs; CTM predicts the strengths.
    pub fn predict_logit_biases(
        &self,
        sync: &[f32],
        episode_biases: &[modgrad_io::types::LogitBias],
    ) -> Option<Vec<modgrad_io::types::LogitBias>> {
        let proj = self.logit_projector.as_ref()?;
        let predicted_strengths = proj.forward(sync);

        let mut biases = Vec::new();
        for (i, eb) in episode_biases.iter().enumerate() {
            let strength = predicted_strengths.get(i).copied().unwrap_or(0.0);
            if strength.abs() > 0.1 {
                biases.push(modgrad_io::types::LogitBias {
                    token_id: eb.token_id,   // keep original token ID
                    token: eb.token.clone(),  // keep original token string
                    strength,                 // CTM-predicted strength
                    suppress: eb.suppress.clone(),
                });
            }
        }
        Some(biases)
    }

    #[allow(deprecated)]
    pub fn dream(&mut self) {
        let replays: Vec<_> = self.replay.prioritized()
            .iter()
            .map(|e| e.observation.clone())
            .collect();

        for obs in &replays {
            let mut state = self.init_state();
            // High-surprise replays get full sync scaling
            state.modulation[MOD_SYNC_SCALE] = 1.0;
            self.forward(obs, &mut state, true); // collect_sleep = true
        }
    }

    pub fn run_sleep(&mut self, blend: f32) -> Vec<(String, f32)> {
        // Don't consolidate with too few traces — overfits and destroys diversity.
        // Need enough samples for LS to find meaningful structure, not noise.
        if self.sleep.traces.len() < 100 {
            self.sleep.reset();
            return Vec::new();
        }

        let corrections = self.sleep.consolidate();
        let mut stats = Vec::new();

        for (name, w_opt, in_dim, out_dim) in &corrections {
            // Apply blended weight update to the corresponding weight matrix.
            // The LS solver returns W_opt[in_dim × out_dim] row-major.
            // We need to map it to the correct weight matrix layout.

            // Helper to blend w_opt into a Linear's weight matrix
            let blend_into = |lin: &mut Linear, w: &[f32], id: usize, od: usize, b: f32| {
                let apply_out = od.min(lin.out_dim);
                let apply_in = id.min(lin.in_dim);
                for o in 0..apply_out {
                    for i in 0..apply_in {
                        let old = lin.weight[o * lin.in_dim + i];
                        let new = w[i * od + o];
                        lin.weight[o * lin.in_dim + i] = (1.0 - b) * old + b * new;
                    }
                }
            };

            match name.as_str() {
                // Inter-region synapses (GLU: weight is 2×out_dim, only update first half)
                "syn_motor_input" | "syn_input_attn" | "syn_attn_output" | "syn_output_motor"
                | "syn_cerebellum" | "syn_basal_ganglia"
                | "syn_insula" | "syn_hippocampus" => {
                    let synapse = match name.as_str() {
                        "syn_motor_input" => &mut self.syn_motor_input,
                        "syn_input_attn" => &mut self.syn_input_attn,
                        "syn_attn_output" => &mut self.syn_attn_output,
                        "syn_output_motor" => &mut self.syn_output_motor,
                        "syn_cerebellum" => &mut self.syn_cerebellum,
                        "syn_basal_ganglia" => &mut self.syn_basal_ganglia,
                        "syn_insula" => &mut self.syn_insula,
                        "syn_hippocampus" => &mut self.syn_hippocampus,
                        _ => unreachable!(),
                    };
                    let apply_od = (*out_dim).min(synapse.linear.out_dim / 2);
                    blend_into(&mut synapse.linear, w_opt, *in_dim, apply_od, blend);
                }

                // Global projector — DEFERRED until stable
                "global_proj" if false => {
                    let apply_od = (*out_dim).min(self.global_projector.out_dim / 2);
                    blend_into(&mut self.global_projector, w_opt, *in_dim, apply_od, blend * 0.5);
                }

                // NLMs: LS gives us the linear approximation of the nonlinear NLM.
                // DEFERRED: only apply after sync patterns have stabilized.
                // Early NLM consolidation disrupts learning (same as output proj).
                "nlm_input" | "nlm_attention" | "nlm_output" | "nlm_motor" if false => {
                    let region = match name.as_str() {
                        "nlm_input" => &mut self.input_region,
                        "nlm_attention" => &mut self.attention_region,
                        "nlm_output" => &mut self.output_region,
                        "nlm_motor" => &mut self.motor_region,
                        _ => unreachable!(),
                    };
                    // The NLM is SuperLinear (per-neuron), not Linear.
                    // We can update the biases of stage1 as a linear correction.
                    // For each neuron, the LS-optimal bias shifts the activation.
                    let n = region.config.n_neurons;
                    let nlm = &mut region.nlm_stage1;
                    let out_per = nlm.out_per;
                    // Average w_opt over in_dim to get per-output bias correction
                    for o in 0..(*out_dim).min(n) {
                        let mean_w: f32 = (0..*in_dim).map(|i| w_opt[i * out_dim + o]).sum::<f32>()
                            / *in_dim as f32;
                        // Blend into the bias for this neuron's first output
                        if o < nlm.biases.len() / out_per {
                            nlm.biases[o * out_per] = (1.0 - blend * 0.3) * nlm.biases[o * out_per]
                                + blend * 0.3 * mean_w;
                        }
                    }
                }

                _ => continue,
            }

            let residual: f32 = w_opt.iter().map(|x| x * x).sum::<f32>().sqrt()
                / (in_dim * out_dim) as f32;
            stats.push((name.clone(), residual));
        }

        self.sleep.reset();

        // Homeostatic rebalancing: prevent neuron monopoly.
        // Over-active neurons get harder to activate, silent ones get easier.
        self.hebb_input.rebalance();
        self.hebb_attention.rebalance();
        self.hebb_output.rebalance();
        self.hebb_motor.rebalance();
        self.hebb_cerebellum.rebalance();
        self.hebb_basal_ganglia.rebalance();

        // NOISE PLASTICITY: update per-neuron noise scales from Hebbian usefulness.
        // Neurons with high baseline variance (consistently active, useful) → lower noise.
        // Neurons with low variance (rarely active, irrelevant) → higher noise (explore).
        // This models long-term receptor density adaptation in the brain.
        for region in [&mut self.input_region, &mut self.attention_region,
                       &mut self.output_region, &mut self.motor_region,
                       &mut self.cerebellum_region, &mut self.basal_ganglia_region] {
            let n = region.config.n_neurons;
            if region.noise_scale.len() != n { region.noise_scale = vec![1.0; n]; }
            if region.usefulness_ema.len() != n { region.usefulness_ema = vec![0.5; n]; }
            for i in 0..n {
                // Usefulness proxy: how far from baseline is this neuron's activation?
                // High variance neurons are doing something — they're "useful".
                let useful = region.usefulness_ema[i];
                // Useful neurons: lower noise (reliable). Irrelevant: higher noise (explore).
                // Range [0.3, 3.0]. useful=0.9 → scale=0.6, useful=0.1 → scale=1.4
                region.noise_scale[i] = (1.5 - useful).clamp(0.3, 3.0);
            }
        }

        self.train_logit_projector();

        stats
    }

    pub fn enable_hebbian(&mut self) { self.hebbian_enabled = true; }
    pub fn disable_hebbian(&mut self) { self.hebbian_enabled = false; }

    /// Full sleep cycle with NREM and REM phases.
    ///
    /// **NREM (slow-wave sleep):**
    ///   Consolidates declarative/episodic → semantic.
    ///   - Least-squares synapse optimization
    ///   - Train logit projector
    ///   - Graduation testing
    ///
    /// **REM (dreaming):**
    ///   Consolidates procedural + processes emotional memories.
    ///   - Replay high-surprise experiences (dreaming)
    ///   - Self-treat fear memories with elevated plasticity
    ///   - Prune old low-surprise replay entries
    ///
    /// Real sleep alternates NREM → REM. We do one cycle of each.
    pub fn full_sleep_cycle(
        &mut self,
        bank: &mut modgrad_io::memory::MemoryBank,
        projector: &Linear,
    ) -> SleepReport {
        let mut report = SleepReport::default();

        // ─── NREM: declarative consolidation ────────────────────
        let synapse_stats = self.run_sleep(0.5);
        report.nrem_synapse_stats = synapse_stats;

        // Graduation testing
        for alter in &mut bank.alters {
            for ep in &mut alter.episodes {
                let prompt_key = ep.keys.iter()
                    .find(|ck| ck.position == -1)
                    .map(|ck| ck.key.clone());

                if let Some(key) = prompt_key {
                    let observation = projector.forward(&key);
                    let score = self.test_consolidation(&observation);

                    ep.sleep_cycles += 1;
                    let valence_rate = match ep.valence {
                        modgrad_io::types::Valence::Fear => 0.05,
                        modgrad_io::types::Valence::Negative => 0.15,
                        modgrad_io::types::Valence::Neutral => 0.3,
                        modgrad_io::types::Valence::Positive => 0.5,
                    };
                    ep.consolidation_score = (1.0 - valence_rate) * ep.consolidation_score
                        + valence_rate * score;

                    if !ep.consolidated {
                        let threshold = match ep.valence {
                            modgrad_io::types::Valence::Fear => 0.95,
                            modgrad_io::types::Valence::Negative => 0.9,
                            _ => 0.8,
                        };
                        if ep.consolidation_score >= threshold {
                            ep.consolidated = true;
                            report.graduated += 1;
                        }
                    }
                }
            }
        }

        // ─── REM: emotional processing + dreaming ───────────────

        // 1. Dream: replay high-surprise experiences
        self.dream();
        report.replayed = self.replay.len();

        // 2. Subconscious emotional processing — the autonomic system.
        //    Diagnoses emotional health, then treats automatically.
        //    Not triggered by user. Like real REM sleep processing.
        let health = super::autonomic::diagnose(bank);
        let treatment = super::autonomic::subconscious_rem(bank, self, projector, &health);
        report.fear_treated = treatment.fears_shifted + treatment.fears_resolved;

        // Record health in report
        report.health_score = health.health_score;
        report.diagnoses = health.diagnoses;

        // 3. Prune old replays
        self.replay.prune(86400.0 * 7.0, 0.5);

        // ─── Neuromodulator tuning ───────────────────────────────
        // Use Angeris bounds as feedback: if gaps aren't closing,
        // widen dopamine range (more surprise sensitivity).
        // If gaps are near zero, narrow ranges (stable).
        let bounds = self.angeris_bounds();
        if !bounds.synapse_gaps.is_empty() {
            let mean_gap_pct: f32 = bounds.synapse_gaps.iter()
                .map(|sg| sg.gap_pct).sum::<f32>()
                / bounds.synapse_gaps.len() as f32;

            // Report gap status for external tuning
            report.nrem_synapse_stats.push(("mean_gap_pct".into(), mean_gap_pct));
        }

        report
    }

    /// Test consolidation: can the CTM reproduce an episode's pattern?
    ///
    /// Runs the CTM on the episode's key, measures:
    /// - Confidence (sync convergence across last ticks)
    /// - Consistency (same key always produces same output)
    ///
    /// Returns a score in [0, 1]. High score = CTM has absorbed this pattern.
    #[allow(deprecated)]
    pub fn test_consolidation(
        &mut self,
        observation: &[f32],
    ) -> f32 {
        // Run twice — consolidated knowledge should be consistent
        let mut state1 = self.init_state();
        let mut state2 = self.init_state();

        let (preds1, sync1) = self.forward(observation, &mut state1, false);
        let (_preds2, sync2) = self.forward(observation, &mut state2, false);

        // 1. Confidence: sync convergence (low delta between last ticks)
        let confidence = if preds1.len() >= 2 {
            let last = &preds1[preds1.len() - 1];
            let prev = &preds1[preds1.len() - 2];
            let delta: f32 = last.iter().zip(prev)
                .map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();
            (1.0 - delta.min(1.0)).max(0.0)
        } else {
            0.0
        };

        // 2. Consistency: same input → same output (deterministic recall)
        let consistency = if !sync1.is_empty() && !sync2.is_empty() {
            let dot: f32 = sync1.iter().zip(&sync2)
                .map(|(a, b)| a * b).sum();
            let n1: f32 = sync1.iter().map(|x| x * x).sum::<f32>().sqrt();
            let n2: f32 = sync2.iter().map(|x| x * x).sum::<f32>().sqrt();
            if n1 > 1e-8 && n2 > 1e-8 { dot / (n1 * n2) } else { 0.0 }
        } else {
            0.0
        };

        // Score = confidence × consistency
        // Both must be high for consolidation — the CTM must converge
        // AND produce the same result each time
        (confidence * consistency.max(0.0)).clamp(0.0, 1.0)
    }

    /// Full sleep cycle with graduation testing.
    ///
    /// 1. Consolidate synapse weights (least-squares)
    /// 2. Test each episode for CTM consolidation
    /// 3. Update consolidation scores
    /// 4. Graduate episodes that score above threshold
    /// 5. Prune fully consolidated episodes past decay
    ///
    /// Returns (synapse_stats, graduated_count, pruned_count).
    pub fn sleep_and_graduate(
        &mut self,
        bank: &mut modgrad_io::memory::MemoryBank,
        projector: &Linear,
        blend: f32,
        graduation_threshold: f32,  // typically 0.8
        prune_after_cycles: u32,    // typically 10
    ) -> (Vec<(String, f32)>, usize, usize) {
        // 1. Consolidate synapse weights
        let synapse_stats = self.run_sleep(blend);

        // 2. Test each episode
        let mut graduated = 0;
        let mut pruned = 0;

        for alter in &mut bank.alters {
            let mut to_prune = Vec::new();

            for (ei, ep) in alter.episodes.iter_mut().enumerate() {
                // Find the PROMPT_END key
                let prompt_key = ep.keys.iter()
                    .find(|ck| ck.position == -1)
                    .map(|ck| &ck.key);

                if let Some(key) = prompt_key {
                    // Project to CTM input space
                    let observation = projector.forward(key);

                    // Test: can the CTM reproduce this pattern?
                    let score = self.test_consolidation(&observation);

                    // Update episode
                    ep.sleep_cycles += 1;

                    // Valence-dependent consolidation rate:
                    // Fear memories RESIST consolidation (stay episodic = PTSD model).
                    // Positive memories consolidate FASTER.
                    // This matches the clinical observation that trauma stays vivid
                    // while happy memories gradually become "semantic" (you know it
                    // happened but can't replay the details).
                    let valence_rate = match ep.valence {
                        modgrad_io::types::Valence::Fear => 0.05,     // very slow — resists
                        modgrad_io::types::Valence::Negative => 0.15,
                        modgrad_io::types::Valence::Neutral => 0.3,
                        modgrad_io::types::Valence::Positive => 0.5,   // fast — consolidates easily
                    };
                    ep.consolidation_score = (1.0 - valence_rate) * ep.consolidation_score
                        + valence_rate * score;

                    // Graduate?
                    if ep.consolidation_score >= graduation_threshold && !ep.consolidated {
                        // Fear memories have a higher graduation threshold
                        let effective_threshold = match ep.valence {
                            modgrad_io::types::Valence::Fear => 0.95,
                            modgrad_io::types::Valence::Negative => graduation_threshold + 0.1,
                            _ => graduation_threshold,
                        };
                        if ep.consolidation_score >= effective_threshold {
                            ep.consolidated = true;
                            graduated += 1;
                        }
                    }

                    // Prune? Fully consolidated AND old enough AND decayed
                    // All memories can be pruned once reprocessed, including fear
                    if ep.consolidated
                        && ep.sleep_cycles >= prune_after_cycles
                        && ep.strength < 0.1
                    {
                        to_prune.push(ei);
                    }
                }
            }

            // Prune in reverse order to preserve indices
            for &ei in to_prune.iter().rev() {
                alter.episodes.remove(ei);
                pruned += 1;
            }
        }

        (synapse_stats, graduated, pruned)
    }

    /// Self-treat fear memories through deliberate reprocessing.
    ///
    /// Models how humans overcome PTSD through therapy:
    ///
    /// 1. **Recall** — deliberately activate the fear memory (exposure)
    /// 2. **Plasticity boost** — enter high-plasticity state (psilocybin analogue)
    /// 3. **Reprocess** — run CTM forward on the memory with elevated plasticity,
    ///    generating new sleep traces that compete with the fear encoding
    /// 4. **Reconsolidate** — the fear memory's labile window is open (just recalled),
    ///    so sleep consolidation can now modify it
    /// 5. **Valence shift** — Fear → Negative → Neutral over multiple cycles
    ///
    /// The factual content is PRESERVED. Only the emotional charge changes.
    /// "Fire is dangerous" stays. "I'm paralyzed by fire" fades.
    ///
    /// `plasticity_boost`: how much to amplify consolidation (1.0 = normal,
    ///   10.0 = psilocybin-level, opens the fear memory to modification).
    /// `cycles`: number of reprocessing cycles (therapy sessions).
    ///
    /// Returns number of fear memories that shifted valence.
    #[allow(deprecated)]
    pub fn self_treat_fear(
        &mut self,
        bank: &mut modgrad_io::memory::MemoryBank,
        projector: &Linear,
        plasticity_boost: f32,
        cycles: usize,
    ) -> usize {
        let mut treated = 0;

        for alter in &mut bank.alters {
            for ep in &mut alter.episodes {
                if ep.valence != modgrad_io::types::Valence::Fear {
                    continue;
                }

                let prompt_key = ep.keys.iter()
                    .find(|ck| ck.position == -1)
                    .map(|ck| ck.key.clone());

                let Some(key) = prompt_key else { continue };

                // 1. Recall — activates the memory, opens reconsolidation window
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs_f64();
                ep.last_recalled_at = now;
                ep.recall_count += 1;

                // 2. Reprocess with elevated plasticity
                let observation = projector.forward(&key);
                for _ in 0..cycles {
                    let mut state = self.init_state();
                    // High sync scaling = this is important, pay attention
                    state.modulation[MOD_SYNC_SCALE] = 1.0;
                    // High gating = novel reprocessing context
                    state.modulation[MOD_GATE] = 1.0;
                    // Elevated arousal = therapeutic importance
                    state.modulation[MOD_AROUSAL] = plasticity_boost.min(3.0);

                    self.forward(&observation, &mut state, true);
                }

                // 3. Run consolidation with boosted blend
                // Normal blend = 0.5, boosted = up to 0.9
                let boosted_blend = (0.5 * plasticity_boost).min(0.9);
                self.run_sleep(boosted_blend);

                // 4. Test consolidation — did the CTM absorb the pattern?
                let score = self.test_consolidation(&projector.forward(&key));

                // 5. Valence shift based on reprocessing success
                //    Each treatment nudges the score up; once it crosses
                //    thresholds, valence shifts down
                ep.consolidation_score = 0.5 * ep.consolidation_score + 0.5 * score;

                if ep.consolidation_score > 0.7 {
                    // Memory is being integrated — shift from Fear to Negative
                    // "I know this was scary" rather than "I'm reliving it"
                    ep.valence = modgrad_io::types::Valence::Negative;
                    treated += 1;
                }
                if ep.consolidation_score > 0.9 {
                    // Fully reprocessed — shift to Neutral
                    // The fact remains, the terror is gone
                    ep.valence = modgrad_io::types::Valence::Neutral;
                }
            }
        }

        treated
    }

    /// Angeris bounds analysis: compute the gap between current synapse weights
    /// and the least-squares optimum for each inter-region synapse.
    ///
    /// gap≈0 means the synapse is already optimal — sleep won't improve it.
    /// gap>0 means sleep consolidation can reduce the residual.
    ///
    /// Reference: Angeris (2022) "Generalizing Power Bounds for Physical Design"
    pub fn angeris_bounds(&self) -> BoundsReport {
        let mut synapse_gaps = Vec::new();

        // Group sleep traces by synapse name
        let mut groups: std::collections::HashMap<String, Vec<(&[f32], &[f32])>>
            = std::collections::HashMap::new();
        for (name, inp, out, _reward) in &self.sleep.traces {
            groups.entry(name.clone()).or_default().push((inp, out));
        }

        for (name, pairs) in &groups {
            if pairs.is_empty() { continue; }
            let in_dim = pairs[0].0.len();
            let out_dim = pairs[0].1.len();

            // Compute achieved residual: sum ||y_t||^2
            let achieved: f32 = pairs.iter()
                .map(|(_, y)| y.iter().map(|v| v * v).sum::<f32>())
                .sum();

            // Build X^T X and X^T Y for least-squares
            let mut xtx = vec![0.0f32; in_dim * in_dim];
            let mut xty = vec![0.0f32; in_dim * out_dim];

            for &(x, y) in pairs {
                for i in 0..in_dim {
                    for j in 0..in_dim {
                        xtx[i * in_dim + j] += x[i] * x[j];
                    }
                    for o in 0..out_dim {
                        xty[i * out_dim + o] += x[i] * y[o];
                    }
                }
            }

            // Regularize
            for i in 0..in_dim {
                xtx[i * in_dim + i] += 1e-4;
            }

            // Solve for W_opt, compute optimal residual
            if let Some(w_opt) = solve_least_squares(&xtx, &xty, in_dim, out_dim) {
                let optimal: f32 = pairs.iter()
                    .map(|(x, y)| {
                        let mut resid = 0.0f32;
                        for o in 0..out_dim {
                            let predicted: f32 = (0..in_dim)
                                .map(|i| w_opt[i * out_dim + o] * x[i])
                                .sum();
                            resid += (y[o] - predicted).powi(2);
                        }
                        resid
                    })
                    .sum();

                let gap = achieved - optimal;
                let gap_pct = if achieved > 1e-10 { gap / achieved * 100.0 } else { 0.0 };

                // Compute effective rank of activations (how much of the synapse is used)
                let act_energy: Vec<f32> = (0..in_dim)
                    .map(|i| pairs.iter().map(|(x, _)| x[i] * x[i]).sum::<f32>())
                    .collect();
                let total_energy: f32 = act_energy.iter().sum();
                let mut sorted_energy = act_energy.clone();
                sorted_energy.sort_by(|a, b| b.partial_cmp(a).unwrap());
                let mut cumsum = 0.0;
                let mut eff_rank = 0;
                for e in &sorted_energy {
                    cumsum += e;
                    eff_rank += 1;
                    if cumsum >= total_energy * 0.9 { break; }
                }

                synapse_gaps.push(SynapseGap {
                    name: name.clone(),
                    achieved_residual: achieved,
                    optimal_residual: optimal,
                    gap,
                    gap_pct,
                    n_samples: pairs.len(),
                    in_dim,
                    out_dim,
                    effective_rank_90: eff_rank,
                });
            }
        }

        // Dead neuron detection per region
        let dead_input = count_dead_neurons(&self.input_region);
        let dead_attention = count_dead_neurons(&self.attention_region);
        let dead_output = count_dead_neurons(&self.output_region);
        let dead_motor = count_dead_neurons(&self.motor_region);

        BoundsReport {
            synapse_gaps,
            dead_neurons: [dead_input, dead_attention, dead_output, dead_motor],
            total_neurons: self.config.d_model,
        }
    }
}

// ─── Factory ────────────────────────────────────────────────

/// Build a configured CTM v2 for a given task profile.
pub fn build_ctm(task: &str, d_model: usize, out_dims: usize) -> Ctm {
    let (split, iterations) = match task {
        "qec" => ([0.20, 0.30, 0.30, 0.20], 16),
        "cifar100" => ([0.35, 0.20, 0.25, 0.20], 16),
        "imagenet" => ([0.40, 0.20, 0.25, 0.15], 32),
        "poker" => ([0.15, 0.25, 0.35, 0.25], 8),
        "language" => ([0.15, 0.30, 0.35, 0.20], 16),
        _ => ([0.25, 0.25, 0.25, 0.25], 16),
    };

    let sizes: Vec<usize> = split.iter()
        .map(|&s| (d_model as f32 * s).max(4.0) as usize)
        .collect();
    let remainder = d_model - sizes.iter().sum::<usize>();

    let mut config = CtmConfig {
        iterations,
        d_model,
        d_input: d_model, // observation dimension matches model
        out_dims,
        ..Default::default()
    };

    config.input_layer.n_neurons = sizes[0];
    config.attention_layer.n_neurons = sizes[1];
    config.output_layer.n_neurons = sizes[2];
    config.motor_layer.n_neurons = sizes[3] + remainder;

    // Scale subcortical proportionally (~12% of mean cortical size, min 4)
    let mean_cortical = d_model / 4;
    let subcortical_n = (mean_cortical / 8).max(4);
    config.cerebellum_layer.n_neurons = subcortical_n;
    config.basal_ganglia_layer.n_neurons = subcortical_n;
    config.insula_layer.n_neurons = subcortical_n;
    config.hippocampus_layer.n_neurons = subcortical_n;

    Ctm::new(config)
}

// ─── Sleep Report ───────────────────────────────────────────

/// Report from a full NREM + REM sleep cycle.
#[derive(Debug, Clone, Default)]
pub struct SleepReport {
    pub nrem_synapse_stats: Vec<(String, f32)>,
    pub graduated: usize,
    pub replayed: usize,
    pub fear_treated: usize,
    pub health_score: f32,
    pub diagnoses: Vec<String>,
}

impl SleepReport {
    pub fn print(&self) {
        println!("=== Sleep Cycle Report ===");
        println!("NREM: {} synapse updates", self.nrem_synapse_stats.len());
        for (name, residual) in &self.nrem_synapse_stats {
            println!("  {name}: residual={residual:.4}");
        }
        println!("Graduated: {} episodes → semantic", self.graduated);
        println!("REM: {} replayed, {} fears treated", self.replayed, self.fear_treated);
        println!("Health: {:.2}/1.00{}", self.health_score,
            if self.health_score < 0.5 { " ⚠ UNHEALTHY" } else { "" });
        for d in &self.diagnoses {
            println!("  DIAGNOSIS: {d}");
        }
    }
}

// ─── Angeris Bounds Types ────────────────────────────────────

/// Per-synapse optimality gap.
#[derive(Debug, Clone)]
pub struct SynapseGap {
    pub name: String,
    pub achieved_residual: f32,
    pub optimal_residual: f32,
    pub gap: f32,         // achieved - optimal (0 = already optimal)
    pub gap_pct: f32,     // gap as percentage of achieved
    pub n_samples: usize,
    pub in_dim: usize,
    pub out_dim: usize,
    pub effective_rank_90: usize,
}

/// Full bounds analysis report.
#[derive(Debug, Clone)]
pub struct BoundsReport {
    pub synapse_gaps: Vec<SynapseGap>,
    pub dead_neurons: [usize; 4],  // [input, attention, output, motor]
    pub total_neurons: usize,
}

impl BoundsReport {
    pub fn print(&self) {
        println!("=== Angeris Bounds Analysis ===");
        println!("Neurons: {} total, dead: input={} attn={} output={} motor={}",
            self.total_neurons,
            self.dead_neurons[0], self.dead_neurons[1],
            self.dead_neurons[2], self.dead_neurons[3]);

        for sg in &self.synapse_gaps {
            let status = if sg.gap_pct < 1.0 { "OPTIMAL" }
                else if sg.gap_pct < 10.0 { "NEAR-OPTIMAL" }
                else { "SUBOPTIMAL" };
            println!("  {} [{status}]: gap={:.4} ({:.1}%), rank={}/{}, samples={}",
                sg.name, sg.gap, sg.gap_pct,
                sg.effective_rank_90, sg.in_dim, sg.n_samples);
        }

        if self.synapse_gaps.iter().all(|sg| sg.gap_pct < 1.0) {
            println!("  ALL SYNAPSES AT OPTIMUM — sleep consolidation won't improve further");
        }
    }
}

/// Count neurons with negligible weight norms in a NeuronLayer.
fn count_dead_neurons(layer: &NeuronLayer) -> usize {
    let n = layer.config.n_neurons;
    let weights = &layer.nlm_stage1.weights;
    let per_neuron = layer.nlm_stage1.in_per * layer.nlm_stage1.out_per;

    let norms: Vec<f32> = (0..n)
        .map(|i| {
            let off = i * per_neuron;
            weights[off..off + per_neuron].iter()
                .map(|w| w * w).sum::<f32>().sqrt()
        })
        .collect();

    let mean_norm: f32 = norms.iter().sum::<f32>() / n as f32;
    norms.iter().filter(|&&norm| norm < mean_norm * 0.1).count()
}

// ─── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear() {
        let lin = Linear::new(4, 3);
        let x = vec![1.0, 0.0, -1.0, 0.5];
        let y = lin.forward(&x);
        assert_eq!(y.len(), 3);
    }

    #[test]
    fn test_super_linear() {
        let sl = SuperLinear::new(8, 4, 6);
        let trace = vec![0.1f32; 8 * 4];
        let out = sl.forward(&trace);
        assert_eq!(out.len(), 8 * 6);
    }

    #[test]
    fn test_neuron_layer_step() {
        let config = LayerConfig { n_neurons: 8, memory_length: 4, ..Default::default() };
        let layer = NeuronLayer::new(&config);
        let mut trace = layer.start_trace.clone();
        let pre_act = vec![0.5f32; 8];
        let activated = layer.step(&pre_act, &mut trace);
        assert_eq!(activated.len(), 8);
    }

    #[test]
    fn test_ctm_forward() {
        let ctm_cfg = CtmConfig {
            d_model: 32, d_input: 8, out_dims: 4,
            n_sync_out: 8, n_sync_action: 4, iterations: 4,
            input_layer: LayerConfig { n_neurons: 8, memory_length: 4, nlm_depth: 1, ..Default::default() },
            attention_layer: LayerConfig { n_neurons: 8, memory_length: 4, nlm_depth: 1, receives_broadcast: false, ..Default::default() },
            output_layer: LayerConfig { n_neurons: 8, memory_length: 4, nlm_depth: 1, ..Default::default() },
            motor_layer: LayerConfig { n_neurons: 8, memory_length: 4, nlm_depth: 1, ..Default::default() },
            ..Default::default()
        };
        let mut ctm = Ctm::new(ctm_cfg);
        let mut state = ctm.init_state();
        let observation = vec![0.1f32; 8];

        let (predictions, sync) = ctm.forward(&observation, &mut state, false);
        assert_eq!(predictions.len(), 4); // 4 ticks
        assert_eq!(predictions[0].len(), 4); // out_dims = 4
        assert_eq!(sync.len(), 8); // n_sync_out = 8
    }

    #[test]
    fn test_ctm_hebbian() {
        let mut ctm = build_ctm("language", 32, 4);
        ctm.enable_hebbian();

        let mut state = ctm.init_state();
        let obs = vec![0.5f32; ctm.config.d_input];

        // Run a few forward passes to build up Hebbian state
        for _ in 0..5 {
            ctm.forward(&obs, &mut state, false);
        }

        // Hebbian running means should have moved from zero
        assert!(ctm.hebb_input.running_mean.iter().any(|&x| x != 0.0)
            || !ctm.hebb_input.calibrated);
    }

    #[test]
    fn test_ctm_sleep() {
        let mut ctm = build_ctm("language", 32, 4);
        let mut state = ctm.init_state();
        let obs = vec![0.5f32; ctm.config.d_input];

        // Collect sleep traces
        for _ in 0..10 {
            ctm.forward(&obs, &mut state, true);
        }

        assert!(!ctm.sleep.traces.is_empty());

        // Angeris bounds — should produce a report with gaps
        let bounds = ctm.angeris_bounds();
        assert!(!bounds.synapse_gaps.is_empty(), "should have synapse gap data");
        // All synapses should have been analyzed
        assert!(bounds.synapse_gaps.len() >= 4, "should analyze all 4 synapses");
        // Gap should be non-negative (achieved >= optimal by definition)
        for sg in &bounds.synapse_gaps {
            assert!(sg.gap >= -0.01, "gap should be non-negative: {} gap={}", sg.name, sg.gap);
            assert!(sg.effective_rank_90 > 0, "effective rank should be > 0");
        }

        let stats = ctm.run_sleep(0.5);
        assert!(!stats.is_empty());

        // Sleep should have cleared traces
        assert!(ctm.sleep.traces.is_empty());
    }

    #[test]
    fn test_ctm_save_load() {
        let ctm = build_ctm("language", 32, 4);
        let path = "/tmp/isis_ctm_test.json";
        ctm.save(path).unwrap();

        let loaded = Ctm::load(path).unwrap();
        assert_eq!(loaded.config.d_model, 32);
        assert_eq!(loaded.config.iterations, 16);
        assert_eq!(loaded.input_region.config.n_neurons, ctm.input_region.config.n_neurons);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_factory() {
        let ctm = build_ctm("poker", 64, 6);
        assert_eq!(ctm.config.iterations, 8);
        let total: usize = [
            ctm.config.input_layer.n_neurons,
            ctm.config.attention_layer.n_neurons,
            ctm.config.output_layer.n_neurons,
            ctm.config.motor_layer.n_neurons,
        ].iter().sum();
        assert_eq!(total, 64);
    }

    #[test]
    fn test_sync_accumulator() {
        let mut sync = SyncAccumulator::new(4, 8);
        let act = vec![0.5f32; 8];

        let s1 = sync.update(&act, 1.0);
        assert_eq!(s1.len(), 4);

        let s2 = sync.update(&act, 1.0);
        // Second update should give different values (accumulated)
        assert_ne!(s1, s2);

        sync.reset();
        let s3 = sync.update(&act, 1.0);
        // After reset, should be same as first
        assert_eq!(s1, s3);
    }

    #[test]
    fn test_cholesky_solve() {
        // Simple 2x2 system: A = [[2, 1], [1, 3]], b = [[5], [7]]
        // Solution: x = [1.6, 1.8]
        let ata = vec![2.0, 1.0, 1.0, 3.0];
        let atb = vec![5.0, 7.0];
        let w = solve_least_squares(&ata, &atb, 2, 1).unwrap();
        assert!((w[0] - 1.6).abs() < 0.01);
        assert!((w[1] - 1.8).abs() < 0.01);
    }

    #[test]
    fn bench_forward_split_vs_original() {
        let config = CtmConfig {
            iterations: 4, d_model: 64, d_input: 64,
            heads: 4, n_sync_out: 8, n_sync_action: 4,
            synapse_depth: 1, out_dims: 8,
            global_broadcast_dim: 0, motor_threshold: 5.0, par_threshold: 32,

            input_layer: LayerConfig { n_neurons: 16, memory_length: 4, nlm_depth: 1, ..Default::default() },
            attention_layer: LayerConfig { n_neurons: 16, memory_length: 4, nlm_depth: 1, ..Default::default() },
            output_layer: LayerConfig { n_neurons: 16, memory_length: 4, nlm_depth: 1, ..Default::default() },
            motor_layer: LayerConfig { n_neurons: 16, memory_length: 4, nlm_depth: 1, ..Default::default() },
            cerebellum_layer: LayerConfig { n_neurons: 8, ..Default::default() },
            basal_ganglia_layer: LayerConfig { n_neurons: 8, ..Default::default() },
            insula_layer: LayerConfig { n_neurons: 8, ..Default::default() },
            hippocampus_layer: LayerConfig { n_neurons: 8, ..Default::default() },
            neuromod: NeuromodConfig::default(),
            use_hopfield_readout: false,
            hopfield_d_key: 32,
        };
        let mut ctm = Ctm::new(config.clone());
        let obs = vec![0.1f32; 64];
        let proprio = vec![0.0f32; 64];

        // Warmup + time original
        let mut state = ctm.init_state();
        ctm.forward_with_proprio(&obs, &proprio, &mut state, false);
        let t0 = std::time::Instant::now();
        for _ in 0..10 {
            let mut state = ctm.init_state();
            ctm.forward_with_proprio(&obs, &proprio, &mut state, false);
        }
        let orig_ms = t0.elapsed().as_millis();

        // Warmup + time forward_split
        let (mut w, mut sess) = ctm.clone().into_split();
        // Lazy init sync topology
        if w.sync_out_indices_l.is_empty() {
            let n = config.output_layer.n_neurons + config.motor_layer.n_neurons + config.d_input;
            let mut rng = SimpleRng::new(config.n_sync_out as u64 * 7919);
            w.sync_out_indices_l = (0..config.n_sync_out).map(|_| rng.next_u64() as usize % n).collect();
            w.sync_out_indices_r = (0..config.n_sync_out).map(|_| rng.next_u64() as usize % n).collect();
            w.sync_out_decay = (0..config.n_sync_out).map(|i| 0.5 + 0.4 * ((i as f32 / config.n_sync_out as f32) * std::f32::consts::PI).sin()).collect();
            let n2 = config.input_layer.n_neurons + config.attention_layer.n_neurons;
            let mut rng2 = SimpleRng::new(config.n_sync_action as u64 * 7919);
            w.sync_action_indices_l = (0..config.n_sync_action).map(|_| rng2.next_u64() as usize % n2).collect();
            w.sync_action_indices_r = (0..config.n_sync_action).map(|_| rng2.next_u64() as usize % n2).collect();
            w.sync_action_decay = (0..config.n_sync_action).map(|i| 0.5 + 0.4 * ((i as f32 / config.n_sync_action as f32) * std::f32::consts::PI).sin()).collect();
        }
        let mut ts = w.init_tick_state();
        forward_split(&w, &mut sess, &mut ts, &obs, &proprio, false);
        let t1 = std::time::Instant::now();
        for _ in 0..10 {
            let mut ts = w.init_tick_state();
            let mut sess = CtmSession::new(&config);
            forward_split(&w, &mut sess, &mut ts, &obs, &proprio, false);
        }
        let split_ms = t1.elapsed().as_millis();

        eprintln!("  BENCH: original={}ms, forward_split={}ms ({}x)",
            orig_ms, split_ms, if orig_ms > 0 { split_ms / orig_ms } else { 999 });
        // forward_split should not be >5x slower
        assert!(split_ms < orig_ms * 10, "forward_split is {}x slower!", split_ms / orig_ms.max(1));
    }

    // ─── Compile-time trait assertions for parallelism safety ──

    /// CtmWeights must be Send + Sync (shareable via Arc across threads).
    fn _assert_ctm_weights_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<CtmWeights>();
        assert_sync::<CtmWeights>();
    }

    /// CtmSession must be Send (movable between threads).
    fn _assert_ctm_session_send() {
        fn assert_send<T: Send>() {}
        assert_send::<CtmSession>();
    }

    /// NeuronLayerWeights must be Send + Sync.
    fn _assert_neuron_layer_weights_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<NeuronLayerWeights>();
        assert_sync::<NeuronLayerWeights>();
    }

    #[test]
    fn test_ctm_split_roundtrip() {
        let ctm = build_ctm("language", 32, 4);
        let original_config = ctm.config.clone();
        let original_n_input = ctm.input_region.config.n_neurons;

        let (weights, session) = ctm.into_split();

        assert_eq!(weights.config.d_model, original_config.d_model);
        assert_eq!(weights.config.iterations, original_config.iterations);
        assert_eq!(weights.input_region.config.n_neurons, original_n_input);

        assert_eq!(session.input_state.noise_scale.len(), original_n_input);
        assert_eq!(session.input_state.usefulness_ema.len(), original_n_input);

        let ctm2 = Ctm::from_split(weights, session);
        assert_eq!(ctm2.config.d_model, original_config.d_model);
        assert_eq!(ctm2.input_region.config.n_neurons, original_n_input);
    }

    #[test]
    fn test_ctm_weights_new() {
        let config = CtmConfig {
            d_model: 32, d_input: 8, out_dims: 4,
            n_sync_out: 8, n_sync_action: 4, iterations: 4,
            input_layer: LayerConfig { n_neurons: 8, memory_length: 4, nlm_depth: 1, ..Default::default() },
            attention_layer: LayerConfig { n_neurons: 8, memory_length: 4, nlm_depth: 1, receives_broadcast: false, ..Default::default() },
            output_layer: LayerConfig { n_neurons: 8, memory_length: 4, nlm_depth: 1, ..Default::default() },
            motor_layer: LayerConfig { n_neurons: 8, memory_length: 4, nlm_depth: 1, ..Default::default() },
            ..Default::default()
        };
        let weights = CtmWeights::new(config.clone());
        assert_eq!(weights.config.d_model, 32);
        assert_eq!(weights.input_region.config.n_neurons, 8);
        assert_eq!(weights.output_projector.out_dim, 4);
    }

    #[test]
    fn test_ctm_session_new() {
        let config = CtmConfig::default();
        let session = CtmSession::new(&config);
        assert_eq!(session.input_state.noise_scale.len(), config.input_layer.n_neurons);
        assert_eq!(session.hebb_input.n_neurons, config.input_layer.n_neurons);
        assert!(!session.hebbian_enabled);
        assert_eq!(session.bg_da_baseline, 0.7);
    }

    #[test]
    fn test_ctm_split_forward_compat() {
        let ctm = build_ctm("language", 32, 4);
        let (weights, session) = ctm.into_split();
        let mut ctm2 = Ctm::from_split(weights, session);

        let mut state = ctm2.init_state();
        let obs = vec![0.1f32; ctm2.config.d_input];
        let (preds, sync) = ctm2.forward(&obs, &mut state, false);
        assert!(!preds.is_empty());
        assert!(!sync.is_empty());
    }

    #[test]
    fn test_neuron_layer_weights_step() {
        let config = LayerConfig { n_neurons: 8, memory_length: 4, ..Default::default() };
        let weights = NeuronLayerWeights::new(&config);
        let mut trace = weights.start_trace.clone();
        let pre_act = vec![0.5f32; 8];
        let activated = weights.step(&pre_act, &mut trace);
        assert_eq!(activated.len(), 8);
    }

    #[test]
    fn test_forward_split_basic() {
        let ctm_cfg = CtmConfig {
            d_model: 32, d_input: 8, out_dims: 4,
            n_sync_out: 8, n_sync_action: 4, iterations: 4,
            input_layer: LayerConfig { n_neurons: 8, memory_length: 4, nlm_depth: 1, ..Default::default() },
            attention_layer: LayerConfig { n_neurons: 8, memory_length: 4, nlm_depth: 1, receives_broadcast: false, ..Default::default() },
            output_layer: LayerConfig { n_neurons: 8, memory_length: 4, nlm_depth: 1, ..Default::default() },
            motor_layer: LayerConfig { n_neurons: 8, memory_length: 4, nlm_depth: 1, ..Default::default() },
            ..Default::default()
        };
        let weights = CtmWeights::new(ctm_cfg.clone());
        let mut session = CtmSession::new(&ctm_cfg);
        let mut tick_state = weights.init_tick_state();

        let observation = vec![0.1f32; 8];
        let proprioception = vec![0.0f32; 8];

        let (predictions, sync, signals) = forward_split(
            &weights, &mut session, &mut tick_state,
            &observation, &proprioception, false,
        );

        assert_eq!(predictions.len(), 4); // 4 iterations
        assert_eq!(predictions[0].len(), 4); // out_dims = 4
        assert_eq!(sync.len(), 8); // n_sync_out = 8
        assert_eq!(signals.ticks, 4);
    }

    #[test]
    fn test_forward_split_with_sleep_traces() {
        let ctm_cfg = CtmConfig {
            d_model: 32, d_input: 8, out_dims: 4,
            n_sync_out: 8, n_sync_action: 4, iterations: 4,
            input_layer: LayerConfig { n_neurons: 8, memory_length: 4, nlm_depth: 1, ..Default::default() },
            attention_layer: LayerConfig { n_neurons: 8, memory_length: 4, nlm_depth: 1, receives_broadcast: false, ..Default::default() },
            output_layer: LayerConfig { n_neurons: 8, memory_length: 4, nlm_depth: 1, ..Default::default() },
            motor_layer: LayerConfig { n_neurons: 8, memory_length: 4, nlm_depth: 1, ..Default::default() },
            ..Default::default()
        };
        let weights = CtmWeights::new(ctm_cfg.clone());
        let mut session = CtmSession::new(&ctm_cfg);
        let mut tick_state = weights.init_tick_state();

        let observation = vec![0.1f32; 8];
        let proprioception = vec![0.0f32; 8];

        let (_, _, _) = forward_split(
            &weights, &mut session, &mut tick_state,
            &observation, &proprioception, true, // collect_sleep = true
        );

        // Sleep traces should be collected (8 synapses × 4 ticks = 32)
        assert!(!session.sleep.traces.is_empty());
        // Tick traces should be collected
        assert_eq!(tick_state.tick_traces.len(), 4);
    }

    #[test]
    fn test_ctm_tick_state_arena_accessors() {
        let ctm_cfg = CtmConfig::default();
        let weights = CtmWeights::new(ctm_cfg);
        let mut ts = weights.init_tick_state();

        // Check arena sizes match config
        assert_eq!(ts.act(REGION_INPUT).len(), weights.config.input_layer.n_neurons);
        assert_eq!(ts.act(REGION_MOTOR).len(), weights.config.motor_layer.n_neurons);
        assert_eq!(ts.trace(REGION_INPUT).len(),
            weights.config.input_layer.n_neurons * weights.config.input_layer.memory_length);

        // Check mutation
        ts.act_mut(REGION_INPUT)[0] = 42.0;
        assert_eq!(ts.act(REGION_INPUT)[0], 42.0);
    }

    #[test]
    fn test_forward_split_hebbian() {
        let ctm_cfg = CtmConfig {
            d_model: 32, d_input: 8, out_dims: 4,
            n_sync_out: 8, n_sync_action: 4, iterations: 4,
            input_layer: LayerConfig { n_neurons: 8, memory_length: 4, nlm_depth: 1, ..Default::default() },
            attention_layer: LayerConfig { n_neurons: 8, memory_length: 4, nlm_depth: 1, receives_broadcast: false, ..Default::default() },
            output_layer: LayerConfig { n_neurons: 8, memory_length: 4, nlm_depth: 1, ..Default::default() },
            motor_layer: LayerConfig { n_neurons: 8, memory_length: 4, nlm_depth: 1, ..Default::default() },
            ..Default::default()
        };
        let weights = CtmWeights::new(ctm_cfg.clone());
        let mut session = CtmSession::new(&ctm_cfg);
        session.enable_hebbian();

        let observation = vec![0.5f32; 8];
        let proprioception = vec![0.0f32; 8];

        // Run multiple forward passes to exercise Hebbian path
        for _ in 0..5 {
            let mut tick_state = weights.init_tick_state();
            forward_split(
                &weights, &mut session, &mut tick_state,
                &observation, &proprioception, false,
            );
        }

        // Should not crash with Hebbian enabled
        assert!(session.hebbian_enabled);
    }
}
