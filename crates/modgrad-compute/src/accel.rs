//! GPU-accelerated batched synapse forward for CTM via KFD.
//!
//! Turns N independent brain forward passes into batched matmuls:
//!   N matvecs → 1 GEMM dispatch → 38× speedup at 50K neurons.
//!
//! Design:
//!   - GpuWeightCache: persistent VRAM mirror of synapse weights (upload once)
//!   - batched_synapse_forward: GPU synapse matmuls for B samples in lock-step
//!   - Zero-copy via KFD shared memory (CPU writes inputs, GPU reads directly)

use modgrad_device::kfd::HsaDevice;
use modgrad_device::kfd::memory::GpuBuffer;
use modgrad_device::kfd::dispatch::KernArgs;
use super::weights::CtmWeights;
use super::session::CtmSession;
use super::tick_state::{CtmTickState, TickSignals, REGION_INPUT, REGION_ATTENTION, REGION_OUTPUT, REGION_MOTOR, REGION_BASAL_GANGLIA};
use modgrad_compute::neuron::{glu, layer_norm, concat, maybe_broadcast, simple_attention};
use crate::neuron::{NeuronLayer, NeuronLayerWeights};
use super::sync::SyncAccumulator;
use super::config::{MOD_SYNC_SCALE, MOD_GATE, MOD_AROUSAL};

/// Persistent GPU mirror of all synapse weight matrices.
/// Created once from CtmWeights, reused across ticks and batches.
pub struct GpuWeightCache {
    pub synapses: Vec<GpuSynapseBuffers>,
}

pub struct GpuSynapseBuffers {
    pub w_row: GpuBuffer,
    pub w_col: GpuBuffer,
    pub bias: GpuBuffer,
    pub m: u32,      // real out_dim
    pub k: u32,      // real in_dim
    pub m_pad: u32,  // padded to tile boundary (multiple of 128)
    pub k_pad: u32,  // padded to tile boundary (multiple of 8)
}

/// Round up to next multiple of `align`.
fn align_up(x: u32, align: u32) -> u32 {
    (x + align - 1) / align * align
}

/// Pad a row-major [rows × cols] matrix to [rows_pad × cols_pad], zero-filling.
fn pad_matrix(data: &[f32], rows: usize, cols: usize, rows_pad: usize, cols_pad: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows_pad * cols_pad];
    for r in 0..rows {
        out[r * cols_pad..r * cols_pad + cols].copy_from_slice(&data[r * cols..r * cols + cols]);
    }
    out
}

/// Pad a bias vector [m] to [m_pad], zero-filling.
fn pad_vec(data: &[f32], len_pad: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; len_pad];
    out[..data.len()].copy_from_slice(data);
    out
}

impl GpuWeightCache {
    /// Upload all 8 synapse weight matrices to VRAM.
    /// Pads W and bias to tile-aligned dimensions so kernels never read OOB.
    pub fn new(dev: &HsaDevice, weights: &CtmWeights) -> std::io::Result<Self> {
        let synapses: Result<Vec<_>, std::io::Error> = weights.synapse_refs().iter().map(|syn| {
            let lin = &syn.linear;
            let m = lin.out_dim as u32;
            let k = lin.in_dim as u32;
            // Pad M to 128 (covers both TM=128 and TM=32), K to 8
            let m_pad = align_up(m, 128);
            let k_pad = align_up(k, 8);

            let w_padded = pad_matrix(&lin.weight, m as usize, k as usize,
                                       m_pad as usize, k_pad as usize);
            let b_padded = pad_vec(&lin.bias, m_pad as usize);

            let w_row = dev.upload_f32(&w_padded)?;
            let w_col = dev.upload_f32_col_major(&w_padded, m_pad as usize, k_pad as usize)?;
            let bias = dev.upload_f32(&b_padded)?;
            Ok(GpuSynapseBuffers {
                w_row, w_col, bias,
                m, k, m_pad, k_pad,
            })
        }).collect();
        Ok(Self { synapses: synapses? })
    }
}

/// Run GPU-batched synapse forward for B samples at one tick.
///
/// Returns per-sample, per-synapse output vectors (post-GLU, post-SiLU, post-LayerNorm).
/// Index: result[sample][synapse] = Vec<f32> of length n_neurons for that region.
///
/// The caller handles NLM step, Hebbian, neuromod, state commit (all CPU-side).
pub fn batched_synapse_forward(
    dev: &mut HsaDevice,
    gpu: &GpuWeightCache,
    weights: &CtmWeights,
    tick_states: &[CtmTickState],
    sync_actions: &mut [SyncAccumulator],
    observations: &[&[f32]],
    proprios: &[&[f32]],
) -> Vec<[Vec<f32>; 8]> {
    let batch = tick_states.len();
    let cfg = &weights.config;

    // Phase 1: CPU — prepare synapse inputs for all B samples
    let mut syn_inputs: Vec<Vec<Vec<f32>>> = (0..8).map(|_| Vec::with_capacity(batch)).collect();

    for b in 0..batch {
        let state = &tick_states[b];
        let obs = observations[b];
        let proprio = proprios[b];

        // Global broadcast from all 8 regions' activations (flat arena)
        let global_raw = weights.global_projector.forward(&state.activations);
        let global_ctx = glu(&global_raw);

        // Full sync_action: per-sample SyncAccumulator update + attention gating
        let act_input = state.act(REGION_INPUT);
        let act_attention = state.act(REGION_ATTENTION);
        let in_attn_cat = concat(&[act_input, act_attention]);
        let phase_in = NeuronLayer::compute_phase(act_input);
        let phase_attn = NeuronLayer::compute_phase(act_attention);
        let in_attn_phases = concat(&[&phase_in, &phase_attn]);
        let sync_action_result = sync_actions[b].update_with_phase(
            &in_attn_cat, state.modulation[MOD_SYNC_SCALE], &in_attn_phases);
        let attn_result = simple_attention(&sync_action_result, obs, cfg.d_input);

        // 0: syn_motor_input
        syn_inputs[0].push(maybe_broadcast(
            &concat(&[obs, state.act(REGION_MOTOR)]),
            &global_ctx, cfg.input_layer.receives_broadcast));

        // 1: syn_input_attn
        syn_inputs[1].push(maybe_broadcast(
            &concat(&[state.act(REGION_INPUT), &attn_result]),
            &global_ctx, cfg.attention_layer.receives_broadcast));

        // 2: syn_attn_output
        syn_inputs[2].push(maybe_broadcast(
            &concat(&[state.act(REGION_ATTENTION), &attn_result]),
            &global_ctx, cfg.output_layer.receives_broadcast));

        // 3: syn_output_motor
        syn_inputs[3].push(maybe_broadcast(
            &concat(&[state.act(REGION_OUTPUT), state.act(REGION_BASAL_GANGLIA)]),
            &global_ctx, cfg.motor_layer.receives_broadcast));

        // 4: syn_cerebellum
        syn_inputs[4].push(concat(&[state.act(REGION_MOTOR), obs]));

        // 5: syn_basal_ganglia
        let mut bg_input = state.act(REGION_OUTPUT).to_vec();
        bg_input.push(state.modulation[MOD_SYNC_SCALE]);
        syn_inputs[5].push(bg_input);

        // 6: syn_insula (hippo retrieval is zero for now — needs CtmSession)
        let hippo_retrieval = vec![0.0f32; cfg.hippocampus_layer.n_neurons];
        syn_inputs[6].push(concat(&[proprio, &hippo_retrieval]));

        // 7: syn_hippocampus
        syn_inputs[7].push(concat(&[
            state.act(REGION_INPUT), state.act(REGION_ATTENTION),
            state.act(REGION_OUTPUT), state.act(REGION_MOTOR),
        ]));
    }

    // Phase 2: dispatch all 8 synapses with 1 submit_wait.
    // All buffers are padded to tile boundaries — safe for any dimension.
    let mut results: Vec<[Vec<f32>; 8]> = (0..batch)
        .map(|_| Default::default())
        .collect();

    let n_pad = align_up(batch as u32, 32);

    // Prepare and enqueue all 8 synapse dispatches
    let mut y_bufs: Vec<GpuBuffer> = Vec::with_capacity(8);
    // Keep x_bufs and args_bufs alive until after submit_wait
    let mut _x_bufs: Vec<GpuBuffer> = Vec::with_capacity(8);
    let mut _args_bufs: Vec<GpuBuffer> = Vec::with_capacity(8);

    for s in 0..8 {
        let gs = &gpu.synapses[s];
        let m_pad = gs.m_pad;
        let k_pad = gs.k_pad;

        let mut flat_input = vec![0.0f32; n_pad as usize * k_pad as usize];
        for b in 0..batch {
            let row = &syn_inputs[s][b];
            let dst = &mut flat_input[b * k_pad as usize..b * k_pad as usize + row.len()];
            dst.copy_from_slice(row);
        }

        let (kernel_name, nwg, w_buf) = if m_pad >= 1536 {
            ("matmul_blocked", ((m_pad + 127) / 128) * ((n_pad + 31) / 32), &gs.w_row)
        } else {
            ("matmul_small", ((m_pad + 31) / 32) * ((n_pad + 31) / 32), &gs.w_col)
        };

        let x_buf = dev.upload_f32(&flat_input).unwrap();
        let y_buf = dev.alloc_output((n_pad as usize * m_pad as usize * 4 + 64) as usize).unwrap();

        let mut args = KernArgs::new();
        args.push_ptr(w_buf); args.push_ptr(&gs.bias);
        args.push_ptr(&x_buf); args.push_ptr(&y_buf);
        args.push_u32(m_pad); args.push_u32(k_pad); args.push_u32(n_pad);
        let ab = args.upload(&dev.alloc).unwrap();

        dev.dispatch_enqueue(kernel_name, &ab, [nwg, 1, 1], [256, 1, 1]);
        y_bufs.push(y_buf);
        _x_bufs.push(x_buf);
        _args_bufs.push(ab);
    }

    // Single submit for all 8 — padding guarantees no OOB
    assert!(dev.submit_wait(30_000), "GPU synapse batch timeout");

    // Read back + elementwise per sample
    for s in 0..8 {
        let m = gpu.synapses[s].m as usize;
        let m_pad = gpu.synapses[s].m_pad as usize;
        let y_slice = unsafe {
            std::slice::from_raw_parts(y_bufs[s].cpu_ptr as *const f32,
                                       n_pad as usize * m_pad)
        };

        let half = m / 2;
        for b in 0..batch {
            let row_start = b * m_pad;
            let raw = &y_slice[row_start..row_start + m];
            let mut out = glu(raw);
            for v in &mut out {
                let sigmoid = 1.0 / (1.0 + (-*v).exp());
                *v *= sigmoid;
            }
            layer_norm(&mut out);
            debug_assert_eq!(out.len(), half);
            results[b][s] = out;
        }
    }

    results
}

/// Full batched forward pass: K ticks × B samples, GPU synapse + CPU everything else.
///
/// Drop-in replacement for calling `forward_split` B times independently.
/// Returns (final_sync, per_tick_syncs, signals) per sample.
/// per_tick_syncs[t] is the sync vector at tick t — enables multi-tick readout.
pub fn forward_split_batched(
    dev: &mut HsaDevice,
    gpu: &GpuWeightCache,
    weights: &CtmWeights,
    sessions: &mut [CtmSession],
    tick_states: &mut [CtmTickState],
    observations: &[&[f32]],
    proprios: &[&[f32]],
    _collect_sleep: bool,
) -> Vec<(Vec<f32>, Vec<Vec<f32>>, TickSignals)> {
    forward_split_batched_inner(dev, gpu, None, weights, sessions, tick_states,
                                observations, proprios, _collect_sleep)
}

/// Full batched forward with optional GPU NLM acceleration.
pub fn forward_split_batched_gpu_nlm(
    dev: &mut HsaDevice,
    gpu: &GpuWeightCache,
    nlm: &GpuNlmCache,
    weights: &CtmWeights,
    sessions: &mut [CtmSession],
    tick_states: &mut [CtmTickState],
    observations: &[&[f32]],
    proprios: &[&[f32]],
    _collect_sleep: bool,
) -> Vec<(Vec<f32>, Vec<Vec<f32>>, TickSignals)> {
    forward_split_batched_inner(dev, gpu, Some(nlm), weights, sessions, tick_states,
                                observations, proprios, _collect_sleep)
}

fn forward_split_batched_inner(
    dev: &mut HsaDevice,
    gpu: &GpuWeightCache,
    nlm_cache: Option<&GpuNlmCache>,
    weights: &CtmWeights,
    sessions: &mut [CtmSession],
    tick_states: &mut [CtmTickState],
    observations: &[&[f32]],
    proprios: &[&[f32]],
    _collect_sleep: bool,
) -> Vec<(Vec<f32>, Vec<Vec<f32>>, TickSignals)> {
    let batch = tick_states.len();
    let cfg = &weights.config;
    let k = cfg.iterations;

    // Per-sample sync accumulators (built from tick_state flat buffers)
    let n_out_motor = cfg.output_layer.n_neurons + cfg.motor_layer.n_neurons + cfg.d_input;
    let n_in_attn = cfg.input_layer.n_neurons + cfg.attention_layer.n_neurons;

    let mut sync_outs: Vec<SyncAccumulator> = (0..batch).map(|b| {
        let mut sa = SyncAccumulator::new(cfg.n_sync_out, n_out_motor);
        sa.position_predictor = weights.position_predictor.clone();
        sa.alpha.copy_from_slice(&tick_states[b].sync_alpha);
        sa.beta.copy_from_slice(&tick_states[b].sync_beta);
        sa.r_shift_buf.copy_from_slice(&tick_states[b].sync_r_shift);
        sa.initialized = tick_states[b].sync_initialized;
        sa
    }).collect();

    let mut sync_actions: Vec<SyncAccumulator> = (0..batch).map(|b| {
        let mut sa = SyncAccumulator::new(cfg.n_sync_action, n_in_attn);
        sa.alpha.copy_from_slice(&tick_states[b].sync_action_alpha);
        sa.beta.copy_from_slice(&tick_states[b].sync_action_beta);
        sa.r_shift_buf.copy_from_slice(&tick_states[b].sync_action_r_shift);
        sa.initialized = tick_states[b].sync_action_initialized;
        sa
    }).collect();

    let mut all_predictions: Vec<Vec<Vec<f32>>> = (0..batch).map(|_| Vec::with_capacity(k)).collect();
    let mut per_tick_syncs: Vec<Vec<Vec<f32>>> = (0..batch).map(|_| Vec::with_capacity(k)).collect();
    let mut signals: Vec<TickSignals> = (0..batch).map(|_| TickSignals::default()).collect();

    // Reset per-forward state
    for b in 0..batch {
        tick_states[b].motor_evidence.fill(0.0);
        tick_states[b].motor_decided = false;
        tick_states[b].decision_tick = None;
    }

    let region_weights = [
        &weights.input_region, &weights.attention_region,
        &weights.output_region, &weights.motor_region,
        &weights.cerebellum_region, &weights.basal_ganglia_region,
        &weights.insula_region, &weights.hippocampus_region,
    ];

    for _tick_idx in 0..k {
        // Phase 1+2: GPU batched synapse forward (input prep + matmul)
        let syn_results = batched_synapse_forward(
            dev, gpu, weights, tick_states, &mut sync_actions, observations, proprios);

        // Phase 3: CPU per-sample NLM + post-processing
        for b in 0..batch {
            let state = &mut tick_states[b];
            let session = &mut sessions[b];

            // NLM step per region: signal -> trace update -> new activations
            let mut new_acts: [Vec<f32>; 8] = if let Some(nlm) = nlm_cache {
                // GPU-accelerated NLM: dispatch superlinear kernel for all 8 regions
                let sigs: [&[f32]; 8] = [
                    &syn_results[b][0], &syn_results[b][1], &syn_results[b][2], &syn_results[b][3],
                    &syn_results[b][4], &syn_results[b][5], &syn_results[b][6], &syn_results[b][7],
                ];
                let mut traces_arr: [Vec<f32>; 8] = [
                    state.trace(0).to_vec(), state.trace(1).to_vec(),
                    state.trace(2).to_vec(), state.trace(3).to_vec(),
                    state.trace(4).to_vec(), state.trace(5).to_vec(),
                    state.trace(6).to_vec(), state.trace(7).to_vec(),
                ];
                let acts = gpu_nlm_step(dev, nlm, &region_weights, &sigs, &mut traces_arr);
                for r in 0..8 {
                    state.trace_mut(r)[..traces_arr[r].len()].copy_from_slice(&traces_arr[r]);
                }
                acts
            } else {
                // CPU fallback
                let mut acts: [Vec<f32>; 8] = Default::default();
                for r in 0..8 {
                    let signal = &syn_results[b][r];
                    let mut trace = state.trace(r).to_vec();
                    acts[r] = region_weights[r].step(signal, &mut trace);
                    state.trace_mut(r)[..trace.len()].copy_from_slice(&trace);
                }
                acts
            };

            // Hebbian correction
            if session.hebbian_enabled {
                if cfg.input_layer.hebbian { session.hebb_input.correct(&mut new_acts[0]); }
                if cfg.attention_layer.hebbian { session.hebb_attention.correct(&mut new_acts[1]); }
                if cfg.output_layer.hebbian { session.hebb_output.correct(&mut new_acts[2]); }
                if cfg.motor_layer.hebbian { session.hebb_motor.correct(&mut new_acts[3]); }
                if cfg.insula_layer.hebbian { session.hebb_insula.correct(&mut new_acts[6]); }
                if cfg.hippocampus_layer.hebbian { session.hebb_hippocampus.correct(&mut new_acts[7]); }
            }

            // Serotonin gating on attention
            let serotonin_gate = state.modulation[MOD_GATE];
            for v in &mut new_acts[1] {
                if v.abs() < serotonin_gate { *v *= 0.1; }
            }

            // Cerebellum prediction error → dopamine
            let obs = observations[b];
            let pred_error_mag: f32 = new_acts[4].iter().zip(obs.iter())
                .map(|(&p, &o)| (o - p).powi(2))
                .sum::<f32>().sqrt()
                / obs.len().max(1) as f32;
            if pred_error_mag > 0.3 {
                state.modulation[MOD_SYNC_SCALE] = (state.modulation[MOD_SYNC_SCALE] * 0.9
                    + 0.1 * pred_error_mag.tanh()).clamp(0.5, 1.0);
            }

            // Commit activations
            for r in 0..8 {
                state.act_mut(r).copy_from_slice(&new_acts[r]);
            }

            // Drift-diffusion motor decision
            if !state.motor_decided {
                let off = state.act_offsets[REGION_MOTOR];
                let sz = state.act_sizes[REGION_MOTOR];
                for i in 0..sz.min(state.motor_evidence.len()) {
                    state.motor_evidence[i] += state.activations[off + i];
                }
                let max_ev: f32 = state.motor_evidence.iter()
                    .map(|x| x.abs()).fold(0.0f32, f32::max);
                let ne = state.modulation[MOD_AROUSAL].max(0.1);
                let effective_threshold = cfg.motor_threshold / ne.sqrt();
                if max_ev > effective_threshold {
                    state.motor_decided = true;
                    state.decision_tick = Some(_tick_idx);
                }
            }

            // Output sync
            let sync_input = concat(&[
                state.act(REGION_OUTPUT),
                state.act(REGION_MOTOR),
                obs,
            ]);
            let da = state.modulation[MOD_SYNC_SCALE];
            let phase_out = NeuronLayer::compute_phase(state.act(REGION_OUTPUT));
            let phase_mot = NeuronLayer::compute_phase(state.act(REGION_MOTOR));
            let out_motor_phases = concat(&[&phase_out, &phase_mot]);

            let sync = sync_outs[b].update_with_phase(&sync_input, da, &out_motor_phases);
            state.last_sync = sync.clone();

            // Augmented features: sync + per-region activation stats (mean, std, max)
            let mut augmented = sync.clone();
            for r in 0..8 {
                let act = state.act(r);
                let n = act.len().max(1) as f32;
                let mean = act.iter().sum::<f32>() / n;
                let var = act.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
                let max_abs = act.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                augmented.push(mean);
                augmented.push(var.sqrt());
                augmented.push(max_abs);
            }
            // Motor evidence magnitude
            let motor_ev: f32 = state.motor_evidence.iter()
                .map(|x| x.abs()).fold(0.0f32, f32::max);
            augmented.push(motor_ev);
            // Modulation state (dopamine, serotonin gate, arousal)
            augmented.push(state.modulation[MOD_SYNC_SCALE]);
            augmented.push(state.modulation[MOD_GATE]);
            augmented.push(state.modulation[MOD_AROUSAL]);
            per_tick_syncs[b].push(augmented);

            // Prediction + intra-tick dopamine
            let pred = weights.output_projector.forward(&sync);
            if let Some(prev_pred) = all_predictions[b].last() {
                let delta: f32 = pred.iter().zip(prev_pred.iter())
                    .map(|(&a, &b)| (a - b) * (a - b)).sum::<f32>().sqrt();
                let mag: f32 = pred.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
                let surprise = delta / mag;
                state.modulation[MOD_SYNC_SCALE] = (state.modulation[MOD_SYNC_SCALE] * 0.9
                    + 0.1 * surprise.tanh()).clamp(0.5, 1.0);
            }
            all_predictions[b].push(pred);
        }
    }

    // Save sync state back + build results
    for b in 0..batch {
        tick_states[b].sync_alpha.copy_from_slice(&sync_outs[b].alpha);
        tick_states[b].sync_beta.copy_from_slice(&sync_outs[b].beta);
        tick_states[b].sync_r_shift.copy_from_slice(&sync_outs[b].r_shift_buf);
        tick_states[b].sync_initialized = sync_outs[b].initialized;
        tick_states[b].sync_action_alpha.copy_from_slice(&sync_actions[b].alpha);
        tick_states[b].sync_action_beta.copy_from_slice(&sync_actions[b].beta);
        tick_states[b].sync_action_r_shift.copy_from_slice(&sync_actions[b].r_shift_buf);
        tick_states[b].sync_action_initialized = sync_actions[b].initialized;

        signals[b].motor_decided = tick_states[b].motor_decided;
        signals[b].decision_tick = tick_states[b].decision_tick;
        signals[b].ticks = k;
    }

    (0..batch).map(|b| {
        let sync = tick_states[b].last_sync.clone();
        let tick_syncs = std::mem::take(&mut per_tick_syncs[b]);
        let sig = std::mem::take(&mut signals[b]);
        (sync, tick_syncs, sig)
    }).collect()
}

// ─── GPU Backward (BPTT sleep) ────────────────────────────
//
// For each synapse, two GEMMs per backward:
//   dX[K×B] = W^T[K×M] @ dZ[M×B]   (input gradient, for upstream)
//   dW[M×K] = dZ[M×B] @ X^T[B×K]   (weight gradient, for update)
//
// W^T is available as w_col (column-major W = row-major W^T).
// Both dispatches use the same matmul kernels, just different dims.

/// Batched backward through all 8 synapses on GPU.
///
/// Inputs:
///   d_outputs[s][b]: gradient of loss w.r.t. synapse s's output for sample b
///   syn_inputs[s][b]: the synapse input that was used in the forward pass
///
/// Returns:
///   d_inputs[s][b]: gradient w.r.t. synapse s's input (for upstream backward)
///   dw[s]: accumulated weight gradient (M × K, same shape as synapse weight)
///   db[s]: accumulated bias gradient (M,)
///
/// Note: this does the LINEAR part only. SiLU derivative is applied on CPU
/// before calling this, and after receiving d_inputs.
pub fn batched_synapse_backward(
    dev: &mut HsaDevice,
    gpu: &GpuWeightCache,
    d_outputs: &[Vec<Vec<f32>>],  // [8][batch][out_dim] — d_loss/d(W@x+b)
    syn_inputs: &[Vec<Vec<f32>>], // [8][batch][in_dim] — cached inputs from forward
) -> (Vec<Vec<Vec<f32>>>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
    // d_outputs and syn_inputs: [synapse_idx][sample_idx][dim]
    let batch = d_outputs[0].len();
    let n_pad = align_up(batch as u32, 32);

    let mut all_d_inputs: Vec<Vec<Vec<f32>>> = vec![vec![]; 8];
    let mut all_dw: Vec<Vec<f32>> = vec![vec![]; 8];
    let mut all_db: Vec<Vec<f32>> = vec![vec![]; 8];

    // ── Pass 1: dX = W^T @ dZ (input gradients) ──
    // W^T is stored as w_col (column-major W = row-major W^T)
    // Dims: W^T is [K × M], dZ is [M × B], result dX is [K × B]
    let mut dx_bufs: Vec<GpuBuffer> = Vec::with_capacity(8);
    let mut _dz_bufs1: Vec<GpuBuffer> = Vec::with_capacity(8);
    let mut _args1: Vec<GpuBuffer> = Vec::with_capacity(8);

    for s in 0..8 {
        let gs = &gpu.synapses[s];
        let m_pad = gs.m_pad;
        let k_pad = gs.k_pad;
        let m = gs.m as usize;

        // Upload dZ as [B × M_pad] (each row is one sample's gradient)
        let mut flat_dz = vec![0.0f32; n_pad as usize * m_pad as usize];
        for b in 0..batch {
            let dz = &d_outputs[s][b];
            for i in 0..m.min(dz.len()) {
                flat_dz[b * m_pad as usize + i] = dz[i];
            }
        }

        // For W^T @ dZ: we need to compute (K × M) @ (M × B)
        // Using w_col which stores W column-major = W^T row-major
        // matmul_small computes: Y[M×N] = W[M×K] @ X[K×N]
        // We want: dX[K×B] = W^T[K×M] @ dZ[M×B]
        // So: M=K_pad, K=M_pad, N=n_pad, W=w_col
        let dz_buf = dev.upload_f32(&flat_dz).unwrap();
        let dx_buf = dev.alloc_output(n_pad as usize * k_pad as usize * 4 + 64).unwrap();

        // For the transpose multiply, swap M and K roles
        let (kernel, nwg) = if k_pad >= 1536 {
            ("matmul_blocked", ((k_pad + 127) / 128) * ((n_pad + 31) / 32))
        } else {
            ("matmul_small", ((k_pad + 31) / 32) * ((n_pad + 31) / 32))
        };

        // Create a zero bias for the transpose multiply
        let zero_bias = dev.upload_f32(&vec![0.0f32; k_pad as usize]).unwrap();

        let mut args = KernArgs::new();
        // W^T is w_col with dims [K_pad × M_pad]
        // But our kernel expects w_row for blocked, w_col for small
        // For the transpose: w_col of W = w_row of W^T
        // So for matmul_small (which uses w_col): we need w_row of W as the "w_col of W^T"
        if k_pad >= 1536 {
            args.push_ptr(&gs.w_col);  // w_row of W^T = w_col of W
        } else {
            args.push_ptr(&gs.w_row);  // w_col of W^T = w_row of W
        }
        args.push_ptr(&zero_bias);
        args.push_ptr(&dz_buf);
        args.push_ptr(&dx_buf);
        args.push_u32(k_pad); args.push_u32(m_pad); args.push_u32(n_pad);
        let ab = args.upload(&dev.alloc).unwrap();

        dev.dispatch_enqueue(kernel, &ab, [nwg, 1, 1], [256, 1, 1]);
        dx_bufs.push(dx_buf);
        _dz_bufs1.push(dz_buf);
        _args1.push(ab);
    }

    assert!(dev.submit_wait(30_000), "GPU backward dX timeout");

    // Read back dX results
    for s in 0..8 {
        let k = gpu.synapses[s].k as usize;
        let k_pad = gpu.synapses[s].k_pad as usize;
        let dx_slice = unsafe {
            std::slice::from_raw_parts(dx_bufs[s].cpu_ptr as *const f32, n_pad as usize * k_pad)
        };
        let mut d_inputs_s = Vec::with_capacity(batch);
        for b in 0..batch {
            d_inputs_s.push(dx_slice[b * k_pad..b * k_pad + k].to_vec());
        }
        all_d_inputs[s] = d_inputs_s;
    }

    // ── Pass 2: dW = dZ @ X^T (weight gradients, CPU for now) ──
    // This is a reduction across the batch: dW = sum_b (dZ_b * X_b^T)
    // Could be a GPU GEMM: [M × B] @ [B × K] but B is small (32).
    // CPU is fine for this — it's the reduction, not the bottleneck.
    for s in 0..8 {
        let m = gpu.synapses[s].m as usize;
        let k = gpu.synapses[s].k as usize;
        let mut dw = vec![0.0f32; m * k];
        let mut db = vec![0.0f32; m];
        for b in 0..batch {
            let dz = &d_outputs[s][b];
            let x = &syn_inputs[s][b];
            for i in 0..m.min(dz.len()) {
                db[i] += dz[i];
                for j in 0..k.min(x.len()) {
                    dw[i * k + j] += dz[i] * x[j];
                }
            }
        }
        all_dw[s] = dw;
        all_db[s] = db;
    }

    (all_d_inputs, all_dw, all_db)
}

// ─── GPU NLM (NeuronLayer) acceleration ────────────────────

/// Persistent VRAM mirror of all NLM (NeuronLayer) weight matrices.
/// 8 regions, each with stage1 and optional stage2 SuperLinear weights.
pub struct GpuNlmCache {
    pub regions: Vec<GpuNlmRegion>,
}

/// Per-region NLM weights in VRAM.
pub struct GpuNlmRegion {
    /// Stage1 SuperLinear: trace -> hidden (with GLU, so out_per = 2*hidden)
    pub stage1_w: GpuBuffer,
    pub stage1_b: GpuBuffer,
    pub stage1_n: u32,       // n_neurons
    pub stage1_o: u32,       // out_per (2*hidden for GLU)
    pub stage1_k: u32,       // in_per (memory_length)
    /// Stage2 SuperLinear (if nlm_depth >= 2): hidden -> output
    pub stage2: Option<GpuNlmStage>,
    /// Inhibitory mask
    pub inhibitory: Vec<bool>,
    /// Sparsity target
    pub sparsity_target: f32,
}

pub struct GpuNlmStage {
    pub w: GpuBuffer,
    pub b: GpuBuffer,
    pub n: u32,
    pub o: u32,
    pub k: u32,
}

impl GpuNlmCache {
    /// Upload all 8 region NLM weights to VRAM.
    pub fn new(dev: &HsaDevice, weights: &CtmWeights) -> std::io::Result<Self> {
        let region_weights = [
            &weights.input_region, &weights.attention_region,
            &weights.output_region, &weights.motor_region,
            &weights.cerebellum_region, &weights.basal_ganglia_region,
            &weights.insula_region, &weights.hippocampus_region,
        ];

        let mut regions = Vec::with_capacity(8);
        for rw in &region_weights {
            let s1 = &rw.nlm_stage1;
            let stage1_w = dev.upload_f32(&s1.weights)?;
            let stage1_b = dev.upload_f32(&s1.biases)?;

            let stage2 = if let Some(ref s2) = rw.nlm_stage2 {
                Some(GpuNlmStage {
                    w: dev.upload_f32(&s2.weights)?,
                    b: dev.upload_f32(&s2.biases)?,
                    n: s2.n_neurons as u32,
                    o: s2.out_per as u32,
                    k: s2.in_per as u32,
                })
            } else {
                None
            };

            regions.push(GpuNlmRegion {
                stage1_w, stage1_b,
                stage1_n: s1.n_neurons as u32,
                stage1_o: s1.out_per as u32,
                stage1_k: s1.in_per as u32,
                stage2,
                inhibitory: rw.inhibitory.clone(),
                sparsity_target: rw.config.sparsity_target,
            });
        }
        Ok(Self { regions })
    }
}

/// GPU-accelerated NLM step for all 8 regions, single sample.
///
/// Replaces the CPU NeuronLayerWeights::step() calls in the batched forward loop.
/// Dispatches stage1 for all 8 regions in one GPU submit, reads back, does
/// GLU+SiLU on CPU, then dispatches stage2 for regions that have it.
///
/// Returns new_acts[8] (same as calling region_weights[r].step() for each r).
pub fn gpu_nlm_step(
    dev: &mut HsaDevice,
    nlm: &GpuNlmCache,
    region_weights: &[&NeuronLayerWeights; 8],
    signals: &[&[f32]; 8],
    traces: &mut [Vec<f32>; 8],
) -> [Vec<f32>; 8] {
    // Phase 0 (CPU): trace shift + append pre_activation for all 8 regions.
    for r in 0..8 {
        let n = region_weights[r].config.n_neurons;
        let m = region_weights[r].config.memory_length;
        let trace = &mut traces[r];
        for neuron in 0..n {
            let off = neuron * m;
            for t in 0..m - 1 {
                trace[off + t] = trace[off + t + 1];
            }
            trace[off + m - 1] = signals[r][neuron];
        }
    }

    // Phase 1 (GPU): dispatch stage1 SuperLinear for all 8 regions.
    let mut stage1_x_bufs: Vec<GpuBuffer> = Vec::with_capacity(8);
    let mut stage1_y_bufs: Vec<GpuBuffer> = Vec::with_capacity(8);
    // Keep args alive until after submit_wait
    let mut _args_bufs: Vec<GpuBuffer> = Vec::with_capacity(16);

    for r in 0..8 {
        let reg = &nlm.regions[r];
        let n = reg.stage1_n;
        let o = reg.stage1_o;
        let k = reg.stage1_k;

        let x_buf = dev.upload_f32(&traces[r]).unwrap();
        let y_size = (n * o) as usize * 4;
        let y_buf = dev.alloc_output(y_size + 64).unwrap();

        let mut args = KernArgs::new();
        args.push_ptr(&reg.stage1_w); args.push_ptr(&reg.stage1_b);
        args.push_ptr(&x_buf); args.push_ptr(&y_buf);
        args.push_u32(n); args.push_u32(o); args.push_u32(k);
        let ab = args.upload(&dev.alloc).unwrap();

        let nwg = (n * o + 255) / 256;
        dev.dispatch_enqueue("superlinear_fwd", &ab, [nwg, 1, 1], [256, 1, 1]);

        stage1_x_bufs.push(x_buf);
        stage1_y_bufs.push(y_buf);
        _args_bufs.push(ab);
    }

    // Single submit for all 8 stage1 dispatches
    assert!(dev.submit_wait(30_000), "GPU NLM stage1 timeout");

    // Phase 2 (CPU): GLU + SiLU on stage1 outputs
    let mut after_glu1: [Vec<f32>; 8] = Default::default();
    for r in 0..8 {
        let reg = &nlm.regions[r];
        let n = reg.stage1_n as usize;
        let o = reg.stage1_o as usize;
        let y_slice = unsafe {
            std::slice::from_raw_parts(
                stage1_y_bufs[r].cpu_ptr as *const f32,
                n * o,
            )
        };
        let mut glu_out = glu(y_slice);
        for v in &mut glu_out {
            let sigmoid = 1.0 / (1.0 + (-*v).exp());
            *v *= sigmoid;
        }
        after_glu1[r] = glu_out;
    }

    // Phase 3 (GPU): dispatch stage2 for regions that have it
    let has_stage2: Vec<bool> = nlm.regions.iter().map(|r| r.stage2.is_some()).collect();
    let any_stage2 = has_stage2.iter().any(|&b| b);

    let mut stage2_y_bufs: Vec<Option<GpuBuffer>> = (0..8).map(|_| None).collect();

    if any_stage2 {
        for r in 0..8 {
            if let Some(ref s2) = nlm.regions[r].stage2 {
                let x_buf = dev.upload_f32(&after_glu1[r]).unwrap();
                let y_size = (s2.n * s2.o) as usize * 4;
                let y_buf = dev.alloc_output(y_size + 64).unwrap();

                let mut args = KernArgs::new();
                args.push_ptr(&s2.w); args.push_ptr(&s2.b);
                args.push_ptr(&x_buf); args.push_ptr(&y_buf);
                args.push_u32(s2.n); args.push_u32(s2.o); args.push_u32(s2.k);
                let ab = args.upload(&dev.alloc).unwrap();

                let nwg = (s2.n * s2.o + 255) / 256;
                dev.dispatch_enqueue("superlinear_fwd", &ab, [nwg, 1, 1], [256, 1, 1]);

                stage2_y_bufs[r] = Some(y_buf);
                _args_bufs.push(ab);
                // x_buf will be dropped after submit, that's fine since GPU has read it
                stage1_x_bufs.push(x_buf); // keep alive
            }
        }

        assert!(dev.submit_wait(30_000), "GPU NLM stage2 timeout");
    }

    // Phase 4 (CPU): final GLU + SiLU + inhibitory + normalize
    let mut new_acts: [Vec<f32>; 8] = Default::default();
    for r in 0..8 {
        let reg = &nlm.regions[r];

        let mut activated = if let Some(ref y_buf) = stage2_y_bufs[r] {
            let s2 = nlm.regions[r].stage2.as_ref().unwrap();
            let n = s2.n as usize;
            let o = s2.o as usize;
            let y_slice = unsafe {
                std::slice::from_raw_parts(y_buf.cpu_ptr as *const f32, n * o)
            };
            let mut out = glu(y_slice);
            for v in &mut out {
                let sigmoid = 1.0 / (1.0 + (-*v).exp());
                *v *= sigmoid;
            }
            out
        } else {
            std::mem::take(&mut after_glu1[r])
        };

        // Inhibitory mask
        for (i, &is_inhib) in reg.inhibitory.iter().enumerate() {
            if is_inhib && i < activated.len() {
                activated[i] = -activated[i].abs();
            }
        }

        // k-WTA sparsity
        if reg.sparsity_target > 0.0 && reg.sparsity_target < 1.0 {
            let n_excitatory = activated.len() - reg.inhibitory.iter().filter(|&&b| b).count();
            let k = ((1.0 - reg.sparsity_target) * n_excitatory as f32) as usize;
            if k > 0 && k < n_excitatory {
                let mut excitatory_vals: Vec<f32> = activated.iter().enumerate()
                    .filter(|(i, _)| !reg.inhibitory.get(*i).copied().unwrap_or(false))
                    .map(|(_, &v)| v.abs())
                    .collect();
                excitatory_vals.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
                let threshold = excitatory_vals.get(k).copied().unwrap_or(0.0);
                for (i, v) in activated.iter_mut().enumerate() {
                    if !reg.inhibitory.get(i).copied().unwrap_or(false) && v.abs() < threshold {
                        *v *= 0.1;
                    }
                }
            }
        }

        // Soft normalization
        let max_abs: f32 = activated.iter()
            .map(|x| x.abs())
            .fold(0.0f32, f32::max)
            .max(0.1);
        for v in &mut activated {
            *v /= max_abs;
        }

        new_acts[r] = activated;
    }

    new_acts
}
