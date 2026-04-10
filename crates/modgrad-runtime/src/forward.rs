//! forward_split: the main forward pass using split ownership.
//!
//! Extracted from ctm.rs — this is the hot path.

use rayon::prelude::*;

pub use super::config::*;
pub use modgrad_compute::neuron::*;
pub use super::neuron::*;
pub use modgrad_compute::ops::*;
pub use super::session::*;
pub use super::synapse::*;
pub use super::weights::*;

// ─── forward_split: new ownership model forward pass ─────────

/// Standalone forward pass using the split ownership model.
///
/// Reads weights from `&CtmWeights` (immutable, Arc-shareable),
/// writes learning state to `&mut CtmSession` (per-thread),
/// uses `&mut CtmTickState` for ephemeral per-tick state.
///
/// Produces identical numerical results to `Ctm::forward_with_proprio`.
///
/// Learning updates are accumulated in `session` during forward:
///   - `session.syn_deltas`: three-factor REINFORCE weight changes
///   - `session.bg_weight_delta`: basal ganglia reinforcement
///
/// To enable learning:
///   1. `session.init_syn_deltas(weights)` once before first forward
///   2. `session.hebbian_enabled = true`
///   3. After forward: `session.last_reward = <0.5 wrong | 1.5 right>`
///   4. Periodically: `session.apply_syn_deltas(&mut weights)`
///
/// The reward from step 3 is injected as dopamine into the NEXT forward call,
/// closing the loop: experience → reward → better synapses → better experience.
pub fn forward_split(
    weights: &CtmWeights,
    session: &mut CtmSession,
    tick_state: &mut CtmTickState,
    observation: &[f32],
    proprioception: &[f32],
    collect_sleep: bool,
) -> (Vec<Vec<f32>>, Vec<f32>, TickSignals) {
    let cfg = &weights.config;
    let k = cfg.iterations;
    let mut predictions: Vec<Vec<f32>> = Vec::with_capacity(k);
    let mut signals = TickSignals::default();

    // Reset per-forward state
    tick_state.motor_evidence.fill(0.0);
    tick_state.motor_decided = false;
    tick_state.decision_tick = None;
    tick_state.tick_traces.clear();

    // Inject external reward from previous forward pass as initial dopamine.
    // This closes the loop: caller sets session.last_reward after evaluating output,
    // next forward_split uses it for three-factor learning.
    if session.hebbian_enabled {
        tick_state.modulation[MOD_SYNC_SCALE] = session.last_reward;
    }

    // We need SyncAccumulator objects for the update logic.
    // Build them from the tick_state flat buffers + weights config.
    let n_out_motor = cfg.output_layer.n_neurons + cfg.motor_layer.n_neurons + cfg.d_input;
    let n_in_attn = cfg.input_layer.n_neurons + cfg.attention_layer.n_neurons;

    let mut sync_out = SyncAccumulator::new(cfg.n_sync_out, n_out_motor);
    // Copy position predictor from weights if available
    sync_out.position_predictor = weights.position_predictor.clone();
    // Restore state from tick_state
    sync_out.alpha.copy_from_slice(&tick_state.sync_alpha);
    sync_out.beta.copy_from_slice(&tick_state.sync_beta);
    sync_out.r_shift_buf.copy_from_slice(&tick_state.sync_r_shift);
    sync_out.initialized = tick_state.sync_initialized;

    // Hopfield readout (modern Hopfield network / attention)
    let mut hopfield_out = weights.hopfield_readout.clone();

    let mut sync_action = SyncAccumulator::new(cfg.n_sync_action, n_in_attn);
    sync_action.alpha.copy_from_slice(&tick_state.sync_action_alpha);
    sync_action.beta.copy_from_slice(&tick_state.sync_action_beta);
    sync_action.r_shift_buf.copy_from_slice(&tick_state.sync_action_r_shift);
    sync_action.initialized = tick_state.sync_action_initialized;

    for tick_idx in 0..k {
        // ═══════════════════════════════════════════════════════════════
        // PARALLEL REGION PROCESSING (brain-like)
        // ═══════════════════════════════════════════════════════════════

        // ── Phase 1: Prepare inputs from prev tick state (all reads, no writes) ──

        // Global broadcast from prev tick's activations (all 8 regions)
        let full_state = tick_state.activations.clone();
        let global_raw = weights.global_projector.forward(&full_state);
        let global_ctx = glu(&global_raw);

        // Sync action (needs prev tick's input + attention)
        let act_input = tick_state.act(REGION_INPUT).to_vec();
        let act_attention = tick_state.act(REGION_ATTENTION).to_vec();
        let act_output = tick_state.act(REGION_OUTPUT).to_vec();
        let act_motor = tick_state.act(REGION_MOTOR).to_vec();
        let _act_cerebellum = tick_state.act(REGION_CEREBELLUM).to_vec();
        let act_basal_ganglia = tick_state.act(REGION_BASAL_GANGLIA).to_vec();
        let _act_insula = tick_state.act(REGION_INSULA).to_vec();
        let _act_hippocampus = tick_state.act(REGION_HIPPOCAMPUS).to_vec();

        let in_attn_cat = concat(&[&act_input, &act_attention]);
        let phase_in = NeuronLayer::compute_phase(&act_input);
        let phase_attn = NeuronLayer::compute_phase(&act_attention);
        let in_attn_phases = concat(&[&phase_in, &phase_attn]);
        let sync_action_result = sync_action.update_with_phase(
            &in_attn_cat, tick_state.modulation[MOD_SYNC_SCALE], &in_attn_phases);
        let attn_result = simple_attention(&sync_action_result, observation, cfg.d_input);

        // Prepare all synapse inputs from prev tick state
        let syn_in_input = maybe_broadcast(
            &concat(&[observation, &act_motor]),
            &global_ctx, cfg.input_layer.receives_broadcast);

        let syn_in_attn = maybe_broadcast(
            &concat(&[&act_input, &attn_result]),
            &global_ctx, cfg.attention_layer.receives_broadcast);

        let syn_in_output = maybe_broadcast(
            &concat(&[&act_attention, &attn_result]),
            &global_ctx, cfg.output_layer.receives_broadcast);

        let syn_in_motor = maybe_broadcast(
            &concat(&[&act_output, &act_basal_ganglia]),
            &global_ctx, cfg.motor_layer.receives_broadcast);

        let syn_in_cereb = concat(&[&act_motor, observation]);

        let mut bg_input = act_output.clone();
        bg_input.push(tick_state.modulation[MOD_SYNC_SCALE]);

        let syn_in_insula = concat(&[proprioception, &session.hippo_retrieval]);

        let syn_in_hippo = concat(&[
            &act_input, &act_attention, &act_output, &act_motor,
        ]);

        // ── Phase 2: Compute all 8 regions in parallel ──
        let total_neurons = cfg.input_layer.n_neurons + cfg.attention_layer.n_neurons
            + cfg.output_layer.n_neurons + cfg.motor_layer.n_neurons;
        // Disable parallel when BPTT caching is active — cached path is sequential
        let parallel = total_neurons >= cfg.par_threshold * 4 && session.bptt_caches.is_none();

        let (input_signal, mut trace_input, mut new_input);
        let (attn_signal, mut trace_attn, mut new_attn);
        let (out_signal, mut trace_output, mut new_output);
        let (motor_signal, mut trace_motor, mut new_motor);
        let (mut cereb_signal, mut trace_cereb, mut new_cereb);
        let (mut bg_signal, mut trace_bg, mut new_bg);
        let (mut insula_signal, mut trace_insula, mut new_insula);
        let (mut hippo_signal, mut trace_hippo, mut new_hippo);

        if parallel {
            let ((r_in, r_attn), (r_out, r_motor)) = rayon::join(
                || rayon::join(
                    || {
                        let sig = weights.syn_motor_input.forward(&syn_in_input);
                        let mut tr = tick_state.trace(REGION_INPUT).to_vec();
                        let act = weights.input_region.step(&sig, &mut tr);
                        (sig, tr, act)
                    },
                    || {
                        let sig = weights.syn_input_attn.forward(&syn_in_attn);
                        let mut tr = tick_state.trace(REGION_ATTENTION).to_vec();
                        let act = weights.attention_region.step(&sig, &mut tr);
                        (sig, tr, act)
                    },
                ),
                || rayon::join(
                    || {
                        let sig = weights.syn_attn_output.forward(&syn_in_output);
                        let mut tr = tick_state.trace(REGION_OUTPUT).to_vec();
                        let act = weights.output_region.step(&sig, &mut tr);
                        (sig, tr, act)
                    },
                    || {
                        let sig = weights.syn_output_motor.forward(&syn_in_motor);
                        let mut tr = tick_state.trace(REGION_MOTOR).to_vec();
                        let act = weights.motor_region.step(&sig, &mut tr);
                        (sig, tr, act)
                    },
                ),
            );
            let (r_cereb, r_bg) = rayon::join(
                || {
                    let sig = weights.syn_cerebellum.forward(&syn_in_cereb);
                    let mut tr = tick_state.trace(REGION_CEREBELLUM).to_vec();
                    let act = weights.cerebellum_region.step(&sig, &mut tr);
                    (sig, tr, act)
                },
                || {
                    let sig = weights.syn_basal_ganglia.forward(&bg_input);
                    let mut tr = tick_state.trace(REGION_BASAL_GANGLIA).to_vec();
                    let act = weights.basal_ganglia_region.step(&sig, &mut tr);
                    (sig, tr, act)
                },
            );
            let (r_insula, r_hippo) = rayon::join(
                || {
                    let sig = weights.syn_insula.forward(&syn_in_insula);
                    let mut tr = tick_state.trace(REGION_INSULA).to_vec();
                    let act = weights.insula_region.step(&sig, &mut tr);
                    (sig, tr, act)
                },
                || {
                    let sig = weights.syn_hippocampus.forward(&syn_in_hippo);
                    let mut tr = tick_state.trace(REGION_HIPPOCAMPUS).to_vec();
                    let act = weights.hippocampus_region.step(&sig, &mut tr);
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
        } else if session.bptt_caches.is_some() {
            // ── BPTT mode: use cached forward for all synapses and NLMs ──
            let (sig, sc0) = weights.syn_motor_input.forward_cached(&syn_in_input);
            input_signal = sig;
            trace_input = tick_state.trace(REGION_INPUT).to_vec();
            let (act, nc0) = weights.input_region.step_cached(&input_signal, &mut trace_input);
            new_input = act;

            let (sig, sc1) = weights.syn_input_attn.forward_cached(&syn_in_attn);
            attn_signal = sig;
            trace_attn = tick_state.trace(REGION_ATTENTION).to_vec();
            let (act, nc1) = weights.attention_region.step_cached(&attn_signal, &mut trace_attn);
            new_attn = act;

            let (sig, sc2) = weights.syn_attn_output.forward_cached(&syn_in_output);
            out_signal = sig;
            trace_output = tick_state.trace(REGION_OUTPUT).to_vec();
            let (act, nc2) = weights.output_region.step_cached(&out_signal, &mut trace_output);
            new_output = act;

            let (sig, sc3) = weights.syn_output_motor.forward_cached(&syn_in_motor);
            motor_signal = sig;
            trace_motor = tick_state.trace(REGION_MOTOR).to_vec();
            let (act, nc3) = weights.motor_region.step_cached(&motor_signal, &mut trace_motor);
            new_motor = act;

            let (sig, sc4) = weights.syn_cerebellum.forward_cached(&syn_in_cereb);
            cereb_signal = sig;
            trace_cereb = tick_state.trace(REGION_CEREBELLUM).to_vec();
            let (act, nc4) = weights.cerebellum_region.step_cached(&cereb_signal, &mut trace_cereb);
            new_cereb = act;

            let (sig, sc5) = weights.syn_basal_ganglia.forward_cached(&bg_input);
            bg_signal = sig;
            trace_bg = tick_state.trace(REGION_BASAL_GANGLIA).to_vec();
            let (act, nc5) = weights.basal_ganglia_region.step_cached(&bg_signal, &mut trace_bg);
            new_bg = act;

            let (sig, sc6) = weights.syn_insula.forward_cached(&syn_in_insula);
            insula_signal = sig;
            trace_insula = tick_state.trace(REGION_INSULA).to_vec();
            let (act, nc6) = weights.insula_region.step_cached(&insula_signal, &mut trace_insula);
            new_insula = act;

            let (sig, sc7) = weights.syn_hippocampus.forward_cached(&syn_in_hippo);
            hippo_signal = sig;
            trace_hippo = tick_state.trace(REGION_HIPPOCAMPUS).to_vec();
            let (act, nc7) = weights.hippocampus_region.step_cached(&hippo_signal, &mut trace_hippo);
            new_hippo = act;

            // Store caches for backward pass
            if let Some(ref mut caches) = session.bptt_caches {
                caches.push(BpttTickCache {
                    syn_caches: [sc0, sc1, sc2, sc3, sc4, sc5, sc6, sc7],
                    nlm_caches: [nc0, nc1, nc2, nc3, nc4, nc5, nc6, nc7],
                    syn_inputs: [
                        syn_in_input.clone(), syn_in_attn.clone(),
                        syn_in_output.clone(), syn_in_motor.clone(),
                        syn_in_cereb.clone(), bg_input.clone(),
                        syn_in_insula.clone(), syn_in_hippo.clone(),
                    ],
                });
            }
        } else {
            input_signal = weights.syn_motor_input.forward(&syn_in_input);
            trace_input = tick_state.trace(REGION_INPUT).to_vec();
            new_input = weights.input_region.step(&input_signal, &mut trace_input);

            attn_signal = weights.syn_input_attn.forward(&syn_in_attn);
            trace_attn = tick_state.trace(REGION_ATTENTION).to_vec();
            new_attn = weights.attention_region.step(&attn_signal, &mut trace_attn);

            out_signal = weights.syn_attn_output.forward(&syn_in_output);
            trace_output = tick_state.trace(REGION_OUTPUT).to_vec();
            new_output = weights.output_region.step(&out_signal, &mut trace_output);

            motor_signal = weights.syn_output_motor.forward(&syn_in_motor);
            trace_motor = tick_state.trace(REGION_MOTOR).to_vec();
            new_motor = weights.motor_region.step(&motor_signal, &mut trace_motor);

            // ── Subcortical sub-ticks: faster clock rate ──
            // Each subcortical region ticks sub_ticks times per cortical tick.
            // Cortical activations are frozen during sub-ticks. Subcortical
            // regions iterate, refining their outputs at higher frequency.
            // Cerebellum ~200Hz vs cortex ~40Hz ≈ 5× sub-ticks.
            trace_cereb = tick_state.trace(REGION_CEREBELLUM).to_vec();
            trace_bg = tick_state.trace(REGION_BASAL_GANGLIA).to_vec();
            trace_insula = tick_state.trace(REGION_INSULA).to_vec();
            trace_hippo = tick_state.trace(REGION_HIPPOCAMPUS).to_vec();

            // First sub-tick (always runs — initializes the variables)
            cereb_signal = weights.syn_cerebellum.forward(&syn_in_cereb);
            new_cereb = weights.cerebellum_region.step(&cereb_signal, &mut trace_cereb);
            bg_signal = weights.syn_basal_ganglia.forward(&bg_input);
            new_bg = weights.basal_ganglia_region.step(&bg_signal, &mut trace_bg);
            insula_signal = weights.syn_insula.forward(&syn_in_insula);
            new_insula = weights.insula_region.step(&insula_signal, &mut trace_insula);
            hippo_signal = weights.syn_hippocampus.forward(&syn_in_hippo);
            new_hippo = weights.hippocampus_region.step(&hippo_signal, &mut trace_hippo);

            // Additional sub-ticks (sub=1..)
            let cereb_ticks = cfg.cerebellum_layer.sub_ticks.max(1);
            let bg_ticks = cfg.basal_ganglia_layer.sub_ticks.max(1);
            let insula_ticks = cfg.insula_layer.sub_ticks.max(1);
            let hippo_ticks = cfg.hippocampus_layer.sub_ticks.max(1);
            let max_sub = cereb_ticks.max(bg_ticks).max(insula_ticks).max(hippo_ticks);

            for sub in 1..max_sub {
                if sub < cereb_ticks {
                    cereb_signal = weights.syn_cerebellum.forward(&syn_in_cereb);
                    new_cereb = weights.cerebellum_region.step(&cereb_signal, &mut trace_cereb);
                }
                if sub < bg_ticks {
                    bg_signal = weights.syn_basal_ganglia.forward(&bg_input);
                    new_bg = weights.basal_ganglia_region.step(&bg_signal, &mut trace_bg);
                }
                if sub < insula_ticks {
                    insula_signal = weights.syn_insula.forward(&syn_in_insula);
                    new_insula = weights.insula_region.step(&insula_signal, &mut trace_insula);
                }
                if sub < hippo_ticks {
                    hippo_signal = weights.syn_hippocampus.forward(&syn_in_hippo);
                    new_hippo = weights.hippocampus_region.step(&hippo_signal, &mut trace_hippo);
                }
            }
        }

        // ── Phase 3: Sequential post-processing (Hebbian, noise, neuromod) ──

        if session.hebbian_enabled {
            if cfg.input_layer.hebbian { session.hebb_input.correct(&mut new_input); }
            if cfg.attention_layer.hebbian { session.hebb_attention.correct(&mut new_attn); }
            if cfg.output_layer.hebbian { session.hebb_output.correct(&mut new_output); }
            if cfg.motor_layer.hebbian { session.hebb_motor.correct(&mut new_motor); }
            if cfg.insula_layer.hebbian { session.hebb_insula.correct(&mut new_insula); }

            // ── Hebbian synapse plasticity: "neurons that fire together wire together" ──
            // ── NMDA-gated Hebbian plasticity (calcium dynamics) ──
            //
            // Real synaptic plasticity requires NMDA receptor activation:
            //   1. Presynaptic neuron fires (glutamate released)
            //   2. Postsynaptic neuron is already depolarized (Mg²⁺ block removed)
            //   3. NMDA channel opens → Ca²⁺ influx
            //   4. Calcium accumulates over multiple ticks (slow τ ≈ 10 ticks)
            //   5. Ca²⁺ > LTP threshold → strengthen (CaMKII pathway)
            //   6. Ca²⁺ > LTD threshold → weaken (calcineurin pathway, BCM rule)
            //
            // This means: Hebbian only fires on COINCIDENCE, not correlation.
            // Single-stream input rarely triggers it. Multi-sensory input does.
            if !session.syn_deltas.is_empty() && !session.syn_calcium.is_empty() {
                let da = tick_state.modulation[MOD_SYNC_SCALE];
                let surprise_gate = (da - cfg.neuromod.da_gate).max(0.0);
                let hebb_lr = cfg.neuromod.hebb_syn_lr * surprise_gate;

                let pairs: [(usize, &[f32], &[f32]); 4] = [
                    (0, &syn_in_input, &new_input),
                    (1, &syn_in_attn, &new_attn),
                    (2, &syn_in_output, &new_output),
                    (3, &syn_in_motor, &new_motor),
                ];

                for (s, pre, post) in pairs {
                    let syn = &weights.synapse_refs()[s];
                    let in_dim = syn.linear.in_dim;
                    let out_dim = syn.linear.out_dim;

                    // Step 1: Decay existing calcium
                    for ca in session.syn_calcium[s].iter_mut() {
                        *ca *= cfg.neuromod.calcium_decay;
                    }

                    // Step 2: NMDA coincidence detection → calcium influx
                    // Pre must be strong (glutamate released) AND post must be
                    // depolarized (Mg²⁺ block removed). Both conditions required.
                    let pre_mean: f32 = pre.iter().map(|x| x.abs()).sum::<f32>()
                        / pre.len().max(1) as f32;
                    let pre_active = pre_mean > cfg.neuromod.nmda_pre_threshold;

                    for j in 0..out_dim.min(post.len()) {
                        if !pre_active { break; }
                        let post_j = post[j].abs();
                        if post_j > cfg.neuromod.nmda_post_threshold {
                            // NMDA gate open: both pre and post active simultaneously
                            // Calcium influx proportional to post activation strength
                            if j < session.syn_calcium[s].len() {
                                session.syn_calcium[s][j] += post_j * pre_mean;
                            }
                        }
                    }

                    // Step 3: Calcium-dependent plasticity (BCM rule)
                    for j in 0..out_dim.min(post.len()) {
                        if j >= session.syn_calcium[s].len() { break; }
                        let ca = session.syn_calcium[s][j];

                        if ca < cfg.neuromod.calcium_ltp_threshold {
                            continue; // Below threshold: no plasticity
                        }

                        let post_j = post[j];
                        // BCM: moderate calcium → LTP, excessive → LTD
                        let plasticity_sign = if ca > cfg.neuromod.calcium_ltd_threshold {
                            -0.5 // LTD: weaken over-active synapses
                        } else {
                            1.0  // LTP: strengthen coincident synapses
                        };

                        for i in 0..in_dim.min(pre.len()) {
                            let idx = j * in_dim + i;
                            if idx < session.syn_deltas[s].len() {
                                // Oja's rule with NMDA gate and BCM sign:
                                // ΔW = lr × sign × (post × pre - post² × W)
                                let w = weights.synapse_refs()[s].linear.weight[idx];
                                session.syn_deltas[s][idx] += hebb_lr * plasticity_sign
                                    * (post_j * pre[i] - post_j * post_j * w);
                            }
                        }

                        // Calcium consumed by plasticity event
                        session.syn_calcium[s][j] *= 0.5;
                    }
                }
            }
            if cfg.hippocampus_layer.hebbian { session.hebb_hippocampus.correct(&mut new_hippo); }
        }

        // ── Hippocampal CAM: store + retrieve ──
        if session.hippo_cam.capacity > 0 {
            let cortical_concat = concat(&[&new_input, &new_attn, &new_output, &new_motor]);
            let hippo_magnitude: f32 = new_hippo.iter()
                .map(|x| x.abs()).sum::<f32>() / new_hippo.len().max(1) as f32;

            if hippo_magnitude > 0.3 {
                session.hippo_cam.store(&cortical_concat, &cortical_concat, hippo_magnitude);
            }

            let (retrieved, max_sim) = session.hippo_cam.retrieve(&cortical_concat, 0.5);
            session.hippo_retrieval = retrieved;
            signals.hippo_max_similarity = max_sim;

            // ── CRI motor memory: retrieve conditioned reflexes ──
            // Cosine-match current cortical state against stored episodes.
            // Retrieved motor bias injected into motor evidence for action selection.
            // This is Pavlovian conditioning: activation pattern → motor response.
            if session.motor_memory.count > 0 {
                let (motor_bias, motor_sim) = session.motor_memory.retrieve(&cortical_concat, 0.6);
                if motor_sim > 0.6 {
                    // Inject retrieved bias into motor evidence (drift-diffusion accumulator)
                    for (ev, &bias) in tick_state.motor_evidence.iter_mut().zip(motor_bias.iter()) {
                        *ev += bias * motor_sim; // scale by match confidence
                    }
                }
                session.motor_bias = motor_bias;
            }
        }

        // ── Interoceptive → neuromodulation ──
        {
            let n_ins = new_insula.len().max(1) as f32;
            let learned_magnitude: f32 = new_insula.iter()
                .map(|x| x.abs()).sum::<f32>() / n_ins;
            let learned_valence: f32 = new_insula.iter().sum::<f32>() / n_ins;

            let n_pro = proprioception.len().max(1) as f32;
            let raw_magnitude: f32 = proprioception.iter()
                .map(|x| x.abs()).sum::<f32>() / n_pro;
            let raw_valence: f32 = proprioception.iter().sum::<f32>() / n_pro;

            let insula_magnitude = 0.7 * learned_magnitude + 0.3 * raw_magnitude;
            let insula_valence = 0.7 * learned_valence + 0.3 * raw_valence;

            signals.insula_magnitude = insula_magnitude;
            signals.insula_valence = insula_valence;
        }

        // ── Basal ganglia three-factor learning (reinforcement) ──
        // Same logic but accumulates weight delta in session instead of mutating weights.
        {
            let da = tick_state.modulation[MOD_SYNC_SCALE];
            let reward_signal = da - session.bg_da_baseline;
            session.bg_da_baseline = 0.99 * session.bg_da_baseline + 0.01 * da;

            let bg_n = cfg.basal_ganglia_layer.n_neurons;
            let pre_dim = bg_input.len();
            let bg_lr = cfg.basal_ganglia_layer.hebbian_lr;

            // Update eligibility trace
            for j in 0..bg_n {
                for i in 0..pre_dim.min(bg_input.len()) {
                    let idx = j * pre_dim + i;
                    if idx < session.bg_eligibility.len() {
                        session.bg_eligibility[idx] = cfg.neuromod.bg_elig_decay * session.bg_eligibility[idx]
                            + 0.1 * bg_input[i] * new_bg[j];
                    }
                }
            }

            // Accumulate weight delta instead of modifying weights directly
            if reward_signal.abs() > 0.1 && bg_n >= 32 {
                let syn = &weights.syn_basal_ganglia.linear;
                let in_dim = pre_dim.min(syn.in_dim);
                let out_dim = bg_n.min(syn.out_dim);

                for j in 0..out_dim {
                    for i in 0..in_dim {
                        let elig_idx = j * pre_dim + i;
                        if elig_idx < session.bg_eligibility.len() {
                            let delta = bg_lr * reward_signal * session.bg_eligibility[elig_idx];
                            if delta.is_finite() {
                                let idx1 = j * syn.in_dim + i;
                                if idx1 < session.bg_weight_delta.len() {
                                    session.bg_weight_delta[idx1] += delta;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Serotonin gating on attention
        let serotonin_gate = tick_state.modulation[MOD_GATE];
        for v in &mut new_attn {
            if v.abs() < serotonin_gate { *v *= 0.1; }
        }

        // ── Titans-style three-factor REINFORCE ──
        //
        // Implements Behrouz et al. 2024 (Titans) learning dynamics adapted
        // for biologically plausible reward-driven plasticity:
        //
        //   Momentary surprise  = advantage = (reward - baseline)
        //   Past surprise       = eligibility trace (momentum of co-activation)
        //   Forgetting gate     = adaptive α_t via salience (applied in apply_syn_deltas)
        //   Data-dependent η_t  = eligibility decay modulated by salience
        //   Data-dependent θ_t  = learning rate scaled by salience
        //
        // Titans Eq.9-10, 13:
        //   S_t = η_t · S_{t-1}  -  θ_t · ∇ℓ(M; x_t)
        //   M_t = (1 - α_t) · M_{t-1}  +  S_t
        //
        // Mapped to neuroscience:
        //   S_t        = eligibility trace (synaptic tag)
        //   η_t        = trace decay, modulated by salience (salient → hold trace longer)
        //   θ_t · ∇ℓ  = lr · advantage · Hebbian term (three-factor rule)
        //   α_t        = weight decay toward baseline (synaptic homeostasis)
        //   salience   = RPE × motor conflict (insula + ACC)
        //
        // Only motor pathway synapses: 3=output→motor, 5=BG.
        // Other synapses are reservoir — learn from cerebellar delta rule only.
        if session.hebbian_enabled && !session.syn_eligibility.is_empty() {
            let da = tick_state.modulation[MOD_SYNC_SCALE];
            let elig_decay_base = cfg.neuromod.bg_elig_decay;

            // ── Salience: RPE × motor conflict ──
            let reward_rpe = (da - session.syn_reward_baseline.get(3).copied().unwrap_or(1.0)).abs();

            let mut motor_sorted: Vec<f32> = new_motor.iter().map(|x| x.abs()).collect();
            motor_sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let motor_conflict = if motor_sorted.len() >= 2 {
                1.0 / (1.0 + (motor_sorted[0] - motor_sorted[1]) * 5.0)
            } else {
                1.0
            };

            let salience = (reward_rpe * (0.3 + 0.7 * motor_conflict)).clamp(0.0, 2.0);
            let salience_gate = if salience > 0.05 { salience } else { 0.0 };

            // ── Data-dependent η_t: eligibility decay ──
            // High salience → η closer to 1.0 → hold trace longer (remember this)
            // Low salience  → η = base decay → normal forgetting
            let eta = elig_decay_base + (1.0 - elig_decay_base) * 0.5 * salience.min(1.0);

            // ── Data-dependent θ_t: learning rate ──
            let theta = cfg.neuromod.hebb_syn_lr * salience_gate;

            let motor_synapses: [(usize, &[f32], &[f32]); 2] = [
                (3, &syn_in_motor, &new_motor),
                (5, &bg_input, &new_bg),
            ];
            for &(s, pre, post) in &motor_synapses {
                if s >= session.syn_eligibility.len() { continue; }
                let in_dim = weights.synapse_refs()[s].linear.in_dim;
                let out_dim = weights.synapse_refs()[s].linear.out_dim;

                // Eligibility trace update: S_t = η_t · S_{t-1} + (1-η_t) · post · pre
                for j in 0..out_dim.min(post.len()) {
                    if post[j].abs() < 0.01 { continue; }
                    for i in 0..in_dim.min(pre.len()) {
                        let idx = j * in_dim + i;
                        if idx < session.syn_eligibility[s].len() {
                            session.syn_eligibility[s][idx] =
                                eta * session.syn_eligibility[s][idx]
                                + (1.0 - eta) * post[j] * pre[i];
                        }
                    }
                }

                // Weight delta: θ_t · advantage · eligibility
                let reward = da;
                let bl = session.syn_reward_baseline[s];
                session.syn_reward_baseline[s] = 0.99 * bl + 0.01 * reward;
                let advantage = reward - bl;

                if advantage.abs() > cfg.neuromod.bg_reward_threshold && theta > 0.0 {
                    for idx in 0..session.syn_deltas[s].len().min(session.syn_eligibility[s].len()) {
                        session.syn_deltas[s][idx] += theta * advantage * session.syn_eligibility[s][idx];
                    }
                }

                // Track per-synapse salience for adaptive forgetting gate
                if s < session.syn_salience.len() {
                    session.syn_salience[s] = 0.95 * session.syn_salience[s] + 0.05 * salience;
                }
            }
        }

        // ACh update from attention strength
        let attn_strength: f32 = sync_action_result.iter().map(|x| x.abs()).sum::<f32>()
            / sync_action_result.len().max(1) as f32;
        tick_state.modulation[MOD_PRECISION] = 0.8 * tick_state.modulation[MOD_PRECISION]
            + 0.2 * attn_strength.clamp(0.0, 1.0);

        // Noise injection + usefulness tracking
        if tick_state.noisy {
            let ach = tick_state.modulation[MOD_PRECISION];

            // Usefulness EMA — uses session's per-region state
            let region_activations: [&[f32]; 4] = [&new_input, &new_attn, &new_output, &new_motor];
            let region_states: [&mut NeuronLayerState; 4] = [
                &mut session.input_state, &mut session.attention_state,
                &mut session.output_state, &mut session.motor_state,
            ];
            for (state_ref, acts) in region_states.into_iter().zip(region_activations.iter()) {
                for (i, &a) in acts.iter().enumerate() {
                    if i < state_ref.usefulness_ema.len() {
                        let u = if a.abs() > 0.1 { 1.0 } else { 0.0 };
                        state_ref.usefulness_ema[i] = 0.99 * state_ref.usefulness_ema[i] + 0.01 * u;
                    }
                }
            }

            // Noise injection modulated by curiosity and anxiety.
            // Curiosity → more exploration noise (seek novel states)
            // Anxiety → less noise (precision mode, avoid errors)
            let curiosity_noise = tick_state.modulation[MOD_CURIOSITY];
            let anxiety_suppress = tick_state.modulation[MOD_ANXIETY];
            let will_factor = (1.0 + 0.5 * curiosity_noise) * (1.0 - 0.3 * anxiety_suppress).max(0.2);

            inject_noise(&mut new_input, &mut tick_state.noise_rng,
                0.1 * (1.0 + 0.3 * (1.0 - ach)) * will_factor, 0.2,
                &session.input_state.noise_scale, &mut tick_state.column_noise_buf, tick_state.sleep_phase);
            let attn_amp = 0.1 * (1.0 - 0.5 * ach) * will_factor;
            inject_noise(&mut new_attn, &mut tick_state.noise_rng, attn_amp, 0.2,
                &session.attention_state.noise_scale, &mut tick_state.column_noise_buf, tick_state.sleep_phase);
            inject_noise(&mut new_output, &mut tick_state.noise_rng, attn_amp, 0.2,
                &session.output_state.noise_scale, &mut tick_state.column_noise_buf, tick_state.sleep_phase);
            inject_noise(&mut new_motor, &mut tick_state.noise_rng, 0.1 * will_factor, 0.2,
                &session.motor_state.noise_scale, &mut tick_state.column_noise_buf, tick_state.sleep_phase);
        }

        // Cerebellum prediction error → dopamine surprise + cerebellar learning
        let pred_error_mag: f32 = new_cereb.iter().zip(observation.iter())
            .map(|(&pred, &obs)| (obs - pred).powi(2))
            .sum::<f32>().sqrt()
            / observation.len().max(1) as f32;
        if pred_error_mag > 0.1 {
            tick_state.modulation[MOD_SYNC_SCALE] = (tick_state.modulation[MOD_SYNC_SCALE] * cfg.neuromod.da_error_alpha
                + cfg.neuromod.da_error_beta * (1.0 + pred_error_mag * 2.0)).clamp(cfg.neuromod.da_min, cfg.neuromod.da_max);

            // Cerebellar delta rule: ΔW = lr × (obs - pred) × pre
            // The cerebellum learns to predict observation from motor + proprioception.
            // This is the climbing fiber signal — direct error correction.
            if session.hebbian_enabled && !session.syn_deltas.is_empty() {
                let cereb_lr = cfg.neuromod.hebb_syn_lr * pred_error_mag.min(1.0);
                let cereb_syn = &weights.syn_cerebellum.linear;
                let in_dim = cereb_syn.in_dim;
                let n_cereb = cfg.cerebellum_layer.n_neurons.min(new_cereb.len());
                let syn_idx = 4; // syn_cerebellum is index 4
                for j in 0..n_cereb {
                    let error_j = if j < observation.len() { observation[j] - new_cereb[j] } else { 0.0 };
                    if error_j.abs() < 0.01 { continue; }
                    for i in 0..in_dim.min(syn_in_cereb.len()) {
                        let idx = j * in_dim + i;
                        if idx < session.syn_deltas[syn_idx].len() {
                            session.syn_deltas[syn_idx][idx] += cereb_lr * error_j * syn_in_cereb[i];
                        }
                    }
                }
            }
        } else {
            tick_state.modulation[MOD_SYNC_SCALE] = tick_state.modulation[MOD_SYNC_SCALE] * cfg.neuromod.da_decay + (1.0 - cfg.neuromod.da_decay);
        }

        // ── Active Inference: curiosity / anxiety from prediction error × arousal ──
        // Friston's free energy decomposition: surprise splits into
        //   curiosity (surprise + calm → explore) vs anxiety (surprise + stress → flee)
        // This gives the organism *will*: it acts to reduce expected prediction error.
        {
            let da = tick_state.modulation[MOD_SYNC_SCALE];
            let ne = tick_state.modulation[MOD_AROUSAL];
            let calm = (1.0 - ne / 2.0).max(0.0).min(1.0);   // 1.0 at rest, 0.0 at max arousal
            let stress = (ne / 2.0).min(1.0);                  // inverse of calm

            // Curiosity: high prediction error + low arousal = "I want to understand this"
            let curiosity = pred_error_mag * da * calm;
            // Anxiety: high prediction error + high arousal = "I need to act NOW"
            let anxiety = pred_error_mag * da * stress;

            // Intrinsic motivation: track learning progress across ticks
            // (Schmidhuber's "fun" = rate of prediction improvement)
            let learning_progress = if tick_idx > 0 && predictions.len() >= 2 {
                let prev_err = if tick_idx >= 2 {
                    // Approximate prev prediction error from prediction stability
                    let p1 = &predictions[predictions.len() - 2];
                    let p0 = &predictions[predictions.len() - 1];
                    p1.iter().zip(p0.iter())
                        .map(|(&a, &b)| (a - b).powi(2))
                        .sum::<f32>().sqrt()
                } else {
                    pred_error_mag * 2.0 // assume started worse
                };
                (prev_err - pred_error_mag).max(0.0) // positive = getting better
            } else {
                0.0
            };

            // Curiosity boost from learning progress (it feels good to learn)
            let curiosity = curiosity + 0.5 * learning_progress;

            // EMA update: smooth over ticks to avoid jitter
            tick_state.modulation[MOD_CURIOSITY] =
                0.7 * tick_state.modulation[MOD_CURIOSITY] + 0.3 * curiosity;
            tick_state.modulation[MOD_ANXIETY] =
                0.7 * tick_state.modulation[MOD_ANXIETY] + 0.3 * anxiety;

            // Curiosity boosts arousal slightly (engagement)
            // Anxiety boosts arousal strongly (fight-or-flight)
            tick_state.modulation[MOD_AROUSAL] = (ne
                + 0.05 * curiosity   // mild engagement
                + 0.15 * anxiety     // strong stress response
            ).clamp(0.1, 2.0);

            // Serotonin: curiosity satisfied → serotonin up (contentment)
            //            anxiety → serotonin down (distress)
            tick_state.modulation[MOD_GATE] = (tick_state.modulation[MOD_GATE]
                + 0.02 * learning_progress  // learning feels good
                - 0.01 * anxiety            // anxiety feels bad
            ).clamp(0.1, 2.0);
        }

        // ── Phase 4: Commit new activations into tick_state arenas ──
        tick_state.act_mut(REGION_INPUT).copy_from_slice(&new_input);
        tick_state.act_mut(REGION_ATTENTION).copy_from_slice(&new_attn);
        tick_state.act_mut(REGION_OUTPUT).copy_from_slice(&new_output);
        tick_state.act_mut(REGION_MOTOR).copy_from_slice(&new_motor);
        tick_state.act_mut(REGION_CEREBELLUM).copy_from_slice(&new_cereb);
        tick_state.act_mut(REGION_BASAL_GANGLIA).copy_from_slice(&new_bg);
        tick_state.act_mut(REGION_INSULA).copy_from_slice(&new_insula);
        tick_state.act_mut(REGION_HIPPOCAMPUS).copy_from_slice(&new_hippo);

        tick_state.trace_mut(REGION_INPUT)[..trace_input.len()].copy_from_slice(&trace_input);
        tick_state.trace_mut(REGION_ATTENTION)[..trace_attn.len()].copy_from_slice(&trace_attn);
        tick_state.trace_mut(REGION_OUTPUT)[..trace_output.len()].copy_from_slice(&trace_output);
        tick_state.trace_mut(REGION_MOTOR)[..trace_motor.len()].copy_from_slice(&trace_motor);
        tick_state.trace_mut(REGION_CEREBELLUM)[..trace_cereb.len()].copy_from_slice(&trace_cereb);
        tick_state.trace_mut(REGION_BASAL_GANGLIA)[..trace_bg.len()].copy_from_slice(&trace_bg);
        tick_state.trace_mut(REGION_INSULA)[..trace_insula.len()].copy_from_slice(&trace_insula);
        tick_state.trace_mut(REGION_HIPPOCAMPUS)[..trace_hippo.len()].copy_from_slice(&trace_hippo);

        // Update phases
        tick_state.phase_mut(0).copy_from_slice(&phase_in);
        tick_state.phase_mut(1).copy_from_slice(&phase_attn);

        // Drift-diffusion with active inference: the organism DECIDES when to act.
        // High curiosity → raise threshold (keep thinking, I'm learning)
        // High anxiety → lower threshold (act now, threat detected)
        // This is the "will" — the organism controls its own decision timing.
        if !tick_state.motor_decided {
            let off = tick_state.act_offsets[REGION_MOTOR];
            let sz = tick_state.act_sizes[REGION_MOTOR];
            for i in 0..sz.min(tick_state.motor_evidence.len()) {
                tick_state.motor_evidence[i] += tick_state.activations[off + i];
            }
            let max_evidence: f32 = tick_state.motor_evidence.iter()
                .map(|x| x.abs()).fold(0.0f32, f32::max);

            let ne = tick_state.modulation[MOD_AROUSAL].max(0.1);
            let curiosity = tick_state.modulation[MOD_CURIOSITY];
            let anxiety = tick_state.modulation[MOD_ANXIETY];

            // Active inference threshold:
            //   Base: motor_threshold / sqrt(arousal)  (arousal speeds up decisions)
            //   Curiosity: +30% per unit (think longer when learning)
            //   Anxiety: -20% per unit (act faster under threat)
            let effective_threshold = cfg.motor_threshold / ne.sqrt()
                * (1.0 + 0.3 * curiosity)   // curiosity raises bar
                * (1.0 - 0.2 * anxiety).max(0.3); // anxiety lowers bar (floor at 30%)

            if max_evidence > effective_threshold {
                tick_state.motor_decided = true;
                tick_state.decision_tick = Some(tick_idx);
            }
        }

        // Collect sleep traces into session
        if collect_sleep {
            session.sleep.collect("syn_motor_input", &syn_in_input, &input_signal);
            session.sleep.collect("syn_input_attn", &syn_in_attn, &attn_signal);
            session.sleep.collect("syn_attn_output", &syn_in_output, &out_signal);
            session.sleep.collect("syn_output_motor", &syn_in_motor, &motor_signal);
            session.sleep.collect("syn_cerebellum", &syn_in_cereb, &cereb_signal);
            session.sleep.collect("syn_basal_ganglia", &bg_input, &bg_signal);
            session.sleep.collect("syn_insula", &syn_in_insula, &insula_signal);
            session.sleep.collect("syn_hippocampus", &syn_in_hippo, &hippo_signal);
        }

        // Output sync: pairwise products from output + motor
        let out_motor = concat(&[
            tick_state.act(REGION_OUTPUT),
            tick_state.act(REGION_MOTOR),
        ]);
        let da = if tick_state.noisy {
            let da_base = tick_state.modulation[MOD_SYNC_SCALE];
            let jitter_amp = 0.1 + 0.1 * da_base;
            let jitter = 1.0 + jitter_amp * tick_state.noise_rng.next_normal();
            (da_base * jitter).clamp(0.1, 2.0)
        } else {
            tick_state.modulation[MOD_SYNC_SCALE]
        };

        let phase_output = NeuronLayer::compute_phase(tick_state.act(REGION_OUTPUT));
        let phase_motor = NeuronLayer::compute_phase(tick_state.act(REGION_MOTOR));
        let out_motor_phases = concat(&[&phase_output, &phase_motor]);
        // Update phases in tick_state
        tick_state.phase_mut(2).copy_from_slice(&phase_output);
        tick_state.phase_mut(3).copy_from_slice(&phase_motor);

        let sync_input = concat(&[&out_motor, observation]);
        let sync = if let Some(ref mut hopfield) = hopfield_out {
            // Modern Hopfield / attention readout (hippocampal pattern completion)
            // Attends over ALL activations, not random pairs
            hopfield.forward(&sync_input, da)
        } else {
            // Legacy SyncAccumulator (pairwise products with decay)
            sync_out.update_with_phase(&sync_input, da, &out_motor_phases)
        };
        tick_state.last_sync = sync.clone();

        // Prediction from sync/hopfield output
        let pred = weights.output_projector.forward(&sync);

        // INTRA-TICK DOPAMINE
        if let Some(prev_pred) = predictions.last() {
            let delta: f32 = pred.iter().zip(prev_pred.iter())
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum::<f32>()
                .sqrt();
            let magnitude: f32 = pred.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
            let tick_surprise = delta / magnitude;
            // Intra-tick dopamine: proportional to prediction change
            tick_state.modulation[MOD_SYNC_SCALE] = (tick_state.modulation[MOD_SYNC_SCALE] * cfg.neuromod.da_intra_alpha
                + cfg.neuromod.da_intra_beta * (1.0 + tick_surprise * 2.0)).clamp(cfg.neuromod.da_min, cfg.neuromod.da_max);
        }

        predictions.push(pred);

        // Collect tick trace for visualization
        if collect_sleep {
            tick_state.tick_traces.push(TickTrace {
                tick: tick_idx,
                input_activations: tick_state.act(REGION_INPUT).to_vec(),
                attention_activations: tick_state.act(REGION_ATTENTION).to_vec(),
                output_activations: tick_state.act(REGION_OUTPUT).to_vec(),
                motor_activations: tick_state.act(REGION_MOTOR).to_vec(),
                sync_out: sync.clone(),
                sync_action: sync_action_result.clone(),
                modulation: tick_state.modulation.clone(),
                motor_evidence_max: tick_state.motor_evidence.iter()
                    .map(|x| x.abs()).fold(0.0f32, f32::max),
                motor_decided: tick_state.motor_decided,
            });
        }

        // ── Telemetry: record tick to binary stream ──
        if let Some(ref mut telem) = session.telemetry {
            let activations = tick_state.activations.clone();
            let signals_mod = &tick_state.modulation;
            let sync_vals = &tick_state.last_sync;
            let extras: Vec<f32> = sync_out.r_shift_buf.clone();

            let record = telem.build_record(
                tick_idx as u32,
                0,
                0.0,
                &activations,
                signals_mod,
                sync_vals,
                &extras,
            );
            telem.record_tick(&record);
        }
    }

    // Save sync state back to tick_state
    tick_state.sync_alpha.copy_from_slice(&sync_out.alpha);
    tick_state.sync_beta.copy_from_slice(&sync_out.beta);
    tick_state.sync_r_shift.copy_from_slice(&sync_out.r_shift_buf);
    tick_state.sync_initialized = sync_out.initialized;
    tick_state.sync_action_alpha.copy_from_slice(&sync_action.alpha);
    tick_state.sync_action_beta.copy_from_slice(&sync_action.beta);
    tick_state.sync_action_r_shift.copy_from_slice(&sync_action.r_shift_buf);
    tick_state.sync_action_initialized = sync_action.initialized;

    // Return the NORMALIZED sync
    let final_sync = tick_state.last_sync.clone();

    // Populate remaining signals
    signals.motor_decided = tick_state.motor_decided;
    signals.decision_tick = tick_state.decision_tick;
    signals.ticks = k;

    let all_act: f32 = [
        tick_state.act(REGION_INPUT),
        tick_state.act(REGION_ATTENTION),
        tick_state.act(REGION_OUTPUT),
        tick_state.act(REGION_MOTOR),
    ].iter()
        .flat_map(|a| a.iter())
        .map(|x| x * x)
        .sum::<f32>();
    signals.activation_energy = all_act.sqrt();

    signals.sync_converged = if predictions.len() >= 2 {
        let last = &predictions[predictions.len() - 1];
        let prev = &predictions[predictions.len() - 2];
        let delta: f32 = last.iter().zip(prev)
            .map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();
        let mag: f32 = last.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
        delta / mag < 0.1
    } else {
        false
    };

    signals.dopamine = tick_state.modulation[MOD_SYNC_SCALE];
    signals.norepinephrine = tick_state.modulation[MOD_AROUSAL];
    signals.serotonin = tick_state.modulation[MOD_GATE];
    signals.curiosity = tick_state.modulation[MOD_CURIOSITY];
    signals.anxiety = tick_state.modulation[MOD_ANXIETY];

    (predictions, final_sync, signals)
}
