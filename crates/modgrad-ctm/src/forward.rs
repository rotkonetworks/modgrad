//! Faithful Ctm CTM forward pass — single neuron pool.
//!
//! Matches `ContinuousThoughtMachine.forward()` from ctm.py exactly.
//! One synapse (U-Net), one NLM, two sync readouts, multihead attention.
//! No regions, no neuromod, no Hebbian. Just the core CTM algorithm.

use modgrad_compute::neuron::Linear;
use super::weights::{CtmWeights, CtmState};

/// Result of one forward pass.
pub struct CtmOutput {
    /// Predictions at each tick: predictions[t] has length out_dims.
    pub predictions: Vec<Vec<f32>>,
    /// Certainty at each tick: [normalized_entropy, 1 - normalized_entropy].
    pub certainties: Vec<[f32; 2]>,
    /// Final sync output.
    pub sync_out: Vec<f32>,
    /// Adaptive exit: per-tick instantaneous exit probability λ_t ∈ (0,1).
    /// Empty when exit gate is off.
    pub exit_lambdas: Vec<f32>,
    /// How many ticks actually ran (may be < iterations if early exit).
    pub ticks_used: usize,
    /// Per-tick activated states, flat `[ticks_used * d_model]`.
    /// Only populated when `config.collect_trajectories` is true — enables
    /// episodic-memory trajectory storage. Empty otherwise.
    pub trajectory: Vec<f32>,
}

/// Input to a CTM forward pass. Carries the projection state of the
/// observation; the episodic-memory behavior is owned by
/// `state.episodic.is_some()`, not by which constructor was called.
///
/// Per the SaaF symmetry-duals review (#6.4): three near-identical
/// `ctm_forward*` entry points with the same internal tick loop were
/// branching on input-shape encoded as a function name. This enum
/// makes the projection state explicit; callers that want to bypass
/// `kv_proj` use `Projected`.
pub enum CtmInput<'a> {
    /// Raw observation, will be projected through `kv_proj` + LayerNorm
    /// before MHA. Most callers want this.
    ///
    /// `obs`: `[n_tokens × raw_dim]` flat. `raw_dim` must equal
    /// `w.kv_proj.in_dim`.
    Raw { obs: &'a [f32], n_tokens: usize, raw_dim: usize },
    /// Pre-projected KV — already in `d_input` space. Used by callers
    /// that have their own projection chain (e.g. external embedding
    /// table, frozen language model bridge).
    Projected { kv: &'a [f32], n_tokens: usize },
}

/// Faithful CTM forward pass.
///
/// When `state.episodic` is `Some` and non-empty, episodic entries are
/// prepended to the MHA KV stream. After the tick loop completes, the
/// current observation's projected KV is appended to episodic memory.
/// `state.episodic = None` is the no-memory default.
pub fn ctm_forward(
    w: &CtmWeights,
    state: &mut CtmState,
    input: CtmInput<'_>,
) -> CtmOutput {
    let d_in = w.config.d_input;

    // Step 1: produce this call's KV (project Raw, pass through Projected).
    let (new_kv, n_new) = match input {
        CtmInput::Raw { obs, n_tokens, raw_dim } => {
            let mut kv = Vec::with_capacity(n_tokens * d_in);
            for t in 0..n_tokens {
                let tok = &obs[t * raw_dim..(t + 1) * raw_dim];
                let mut projected = {
                    let _g = crate::dispatch_profile::Guard::new(
                        crate::dispatch_profile::DispatchKind::KvProj);
                    w.kv_proj.forward(tok)
                };
                affine_ln(&mut projected, &w.kv_ln_gamma, &w.kv_ln_beta);
                kv.extend_from_slice(&projected);
            }
            (kv, n_tokens)
        }
        CtmInput::Projected { kv, n_tokens } => (kv.to_vec(), n_tokens),
    };

    // Step 2: prepend episodic entries if present. Episodic store owns
    // its own already-projected KV ring; concat is in d_input space.
    let has_episodic_entries =
        state.episodic.as_ref().map_or(false, |e| !e.is_empty());
    let (kv_used, n_total) = if has_episodic_entries {
        let ep = state.episodic.as_ref().unwrap();
        let ep_n = ep.n_tokens();
        let mut combined = Vec::with_capacity((ep_n + n_new) * d_in);
        combined.extend_from_slice(&ep.as_kv());
        combined.extend_from_slice(&new_kv);
        (combined, ep_n + n_new)
    } else {
        (new_kv.clone(), n_new)
    };

    // Step 3: tick loop on the combined KV.
    let output = ctm_forward_with_kv(w, state, &kv_used, n_total);

    // Step 4: persist this call's KV into episodic if the slot exists.
    // (Empty episodic still gets pushed to — first-call bootstrap.)
    if let Some(ref mut mem) = state.episodic {
        for t in 0..n_new {
            mem.push(&new_kv[t * d_in..(t + 1) * d_in]);
        }
    }

    output
}

/// Forward pass with pre-built KV buffer (skips kv_proj). Private
/// because the public surface routes everything through `CtmInput`.
///
/// **Cross-forward state contract** (relevant for AdaptiveGate /
/// Certainty early-exit and any caller reusing a `CtmState`):
///
/// - `state.alpha_out` / `state.beta_out`: **reset every call**. The
///   `sync_init` invocation below clears and re-initialises them
///   from `state.activated` regardless of how the previous forward
///   exited. There is no cross-forward bleed of sync accumulators
///   even when the prior forward terminated mid-trajectory.
/// - `state.alpha_action` / `state.beta_action` (Option): **vestigial
///   on `CtmState`**; the action sync uses local Vecs inside this
///   function and never writes back. The struct fields exist for
///   serialisation symmetry with `*_out`.
/// - `state.activated` / `state.trace`: **persist across forwards**.
///   `sync_init` *reads* the previous activated to seed `alpha_out`,
///   then the tick loop fully overwrites both. Continuous-thinking
///   semantics: a caller that reuses a `CtmState` is implicitly
///   running connected trajectories. A caller that wants fresh
///   thinking per sample must construct a new `CtmState`.
/// - `state.episodic`: persists by design (the memory store).
fn ctm_forward_with_kv(
    w: &CtmWeights,
    state: &mut CtmState,
    kv: &[f32],
    n_tokens: usize,
) -> CtmOutput {
    ctm_forward_with_kv_inner(w, state, kv, n_tokens, None)
}

/// Like `ctm_forward` but also returns the per-tick × per-head softmax
/// attention weights over input tokens, shape `[ticks_used][n_heads][n_tokens]`.
/// Used by visualisation tools (and any caller wanting to inspect what
/// the CTM is attending to). Numerics match `ctm_forward` exactly when
/// `state` is constructed identically.
pub fn ctm_forward_with_attn_trace(
    w: &CtmWeights,
    state: &mut CtmState,
    input: CtmInput<'_>,
) -> (CtmOutput, Vec<Vec<Vec<f32>>>) {
    let d_in = w.config.d_input;
    let (new_kv, n_new) = match input {
        CtmInput::Raw { obs, n_tokens, raw_dim } => {
            let mut kv = Vec::with_capacity(n_tokens * d_in);
            for t in 0..n_tokens {
                let tok = &obs[t * raw_dim..(t + 1) * raw_dim];
                let mut projected = w.kv_proj.forward(tok);
                affine_ln(&mut projected, &w.kv_ln_gamma, &w.kv_ln_beta);
                kv.extend_from_slice(&projected);
            }
            (kv, n_tokens)
        }
        CtmInput::Projected { kv, n_tokens } => (kv.to_vec(), n_tokens),
    };
    let has_episodic_entries =
        state.episodic.as_ref().map_or(false, |e| !e.is_empty());
    let (kv_used, n_total) = if has_episodic_entries {
        let ep = state.episodic.as_ref().unwrap();
        let ep_n = ep.n_tokens();
        let mut combined = Vec::with_capacity((ep_n + n_new) * d_in);
        combined.extend_from_slice(&ep.as_kv());
        combined.extend_from_slice(&new_kv);
        (combined, ep_n + n_new)
    } else {
        (new_kv.clone(), n_new)
    };
    let mut trace: Vec<Vec<Vec<f32>>> = Vec::with_capacity(w.config.iterations);
    let output = ctm_forward_with_kv_inner(w, state, &kv_used, n_total, Some(&mut trace));
    if let Some(ref mut mem) = state.episodic {
        for t in 0..n_new {
            mem.push(&new_kv[t * d_in..(t + 1) * d_in]);
        }
    }
    (output, trace)
}

fn ctm_forward_with_kv_inner(
    w: &CtmWeights,
    state: &mut CtmState,
    kv: &[f32],
    n_tokens: usize,
    mut attn_trace: Option<&mut Vec<Vec<Vec<f32>>>>,
) -> CtmOutput {
    let cfg = &w.config;
    let d = cfg.d_model;
    let d_in = cfg.d_input;
    let k = cfg.iterations;
    let m = cfg.memory_length;

    let r_out: Vec<f32> = w.decay_params_out.iter()
        .map(|&p| (-p.clamp(0.0, 15.0)).exp()).collect();
    let r_action: Vec<f32> = w.decay_params_action.iter()
        .map(|&p| (-p.clamp(0.0, 15.0)).exp()).collect();

    sync_init(&state.activated, &w.sync_out_left, &w.sync_out_right,
        &mut state.alpha_out, &mut state.beta_out);

    let mut alpha_action: Vec<f32> = Vec::new();
    let mut beta_action: Vec<f32> = Vec::new();
    let mut action_initialized = false;

    let mut predictions = Vec::with_capacity(k);
    let mut certainties = Vec::with_capacity(k);
    let mut exit_lambdas: Vec<f32> = Vec::new();
    let mut exit_cdf = 0.0f32;
    let mut survival = 1.0f32;
    let collect_traj = cfg.collect_trajectories;
    let mut trajectory = if collect_traj { Vec::with_capacity(k * d) } else { Vec::new() };

    for _tick in 0..k {
        let sync_action = if !action_initialized {
            sync_init(&state.activated, &w.sync_action_left, &w.sync_action_right,
                &mut alpha_action, &mut beta_action);
            action_initialized = true;
            sync_read(&alpha_action, &beta_action)
        } else {
            sync_update(&state.activated, &w.sync_action_left, &w.sync_action_right,
                &mut alpha_action, &mut beta_action, &r_action)
        };

        let q = {
            let _g = crate::dispatch_profile::Guard::new(
                crate::dispatch_profile::DispatchKind::QProj);
            w.q_proj.forward(&sync_action)
        };
        let attn_out = {
            let _g = crate::dispatch_profile::Guard::new(
                crate::dispatch_profile::DispatchKind::MhaIn);
            if attn_trace.is_some() {
                let mut tick_attn: Vec<Vec<f32>> = Vec::new();
                let out = multihead_attention_with_attn(
                    &q, kv, n_tokens, d_in, cfg.heads,
                    &w.mha_in_proj, &w.mha_out_proj,
                    Some(&mut tick_attn),
                );
                if let Some(ref mut t) = attn_trace { t.push(tick_attn); }
                out
            } else {
                multihead_attention(
                    &q, kv, n_tokens, d_in, cfg.heads,
                    &w.mha_in_proj, &w.mha_out_proj,
                )
            }
        };

        let mut pre_syn = Vec::with_capacity(d_in + d);
        pre_syn.extend_from_slice(&attn_out);
        pre_syn.extend_from_slice(&state.activated);
        let pre_act = {
            let _g = crate::dispatch_profile::Guard::new(
                crate::dispatch_profile::DispatchKind::Synapse);
            w.synapse.forward(&pre_syn)
        };

        for n in 0..d {
            let base = n * m;
            state.trace.copy_within(base + 1..base + m, base);
            state.trace[base + m - 1] = pre_act[n];
        }

        state.activated = {
            let _g = crate::dispatch_profile::Guard::new(
                crate::dispatch_profile::DispatchKind::NlmS1);
            nlm_forward(&state.trace, &w.nlm_stage1, w.nlm_stage2.as_ref(), d)
        };

        if collect_traj {
            trajectory.extend_from_slice(&state.activated);
        }

        let sync_out = sync_update(&state.activated, &w.sync_out_left, &w.sync_out_right,
            &mut state.alpha_out, &mut state.beta_out, &r_out);

        let pred = {
            let _g = crate::dispatch_profile::Guard::new(
                crate::dispatch_profile::DispatchKind::OutProjRegion);
            w.output_proj.forward(&sync_out)
        };
        let cert = compute_certainty(&pred);
        predictions.push(pred);
        certainties.push(cert);

        match &cfg.exit_strategy {
            crate::config::ExitStrategy::AdaptiveGate { threshold, .. } => {
                if let Some(ref gate) = w.exit_gate {
                    let _g = crate::dispatch_profile::Guard::new(
                        crate::dispatch_profile::DispatchKind::ExitGateRegion);
                    let gate_logit = gate.forward(&sync_out);
                    let lambda = 1.0 / (1.0 + (-gate_logit[0]).exp());
                    exit_lambdas.push(lambda);
                    let p_exit = lambda * survival;
                    exit_cdf += p_exit;
                    survival *= 1.0 - lambda;
                    if exit_cdf > *threshold { break; }
                }
            }
            crate::config::ExitStrategy::Certainty { threshold } => {
                if cert[1] > *threshold { break; }
            }
            crate::config::ExitStrategy::None => {}
        }
    }

    let ticks_used = predictions.len();
    let sync_out = sync_read(&state.alpha_out, &state.beta_out);
    CtmOutput { predictions, certainties, sync_out, exit_lambdas, ticks_used, trajectory }
}
// ─── Episodic memory bridge ────────────────────────────────

/// Store a CtmOutput as an episodic memory entry.
///
/// Requires the forward pass to have run with `config.collect_trajectories = true`;
/// panics on an empty trajectory to fail loudly on misconfigured callers rather
/// than silently storing zero-length episodes.
pub fn store_episodic_from_output(
    mem: modgrad_memory::episodic::EpisodicMemory,
    output: &CtmOutput,
    surprise: f32,
) -> (modgrad_memory::episodic::EpisodicMemory, bool) {
    assert!(
        !output.trajectory.is_empty(),
        "store_episodic_from_output: CtmOutput.trajectory is empty — \
         set config.collect_trajectories = true before the forward pass"
    );
    modgrad_memory::episodic::store(
        mem,
        &output.trajectory,
        &output.certainties,
        &output.exit_lambdas,
        output.ticks_used,
        surprise,
    )
}

/// Store a CtmOutput with an explicit valence receipt.
pub fn store_episodic_from_output_with_valence(
    mem: modgrad_memory::episodic::EpisodicMemory,
    output: &CtmOutput,
    surprise: f32,
    receipt: Option<modgrad_memory::episodic::ValenceReceipt>,
) -> (modgrad_memory::episodic::EpisodicMemory, bool) {
    assert!(
        !output.trajectory.is_empty(),
        "store_episodic_from_output_with_valence: CtmOutput.trajectory is empty — \
         set config.collect_trajectories = true before the forward pass"
    );
    modgrad_memory::episodic::store_with_valence(
        mem,
        &output.trajectory,
        &output.certainties,
        &output.exit_lambdas,
        output.ticks_used,
        surprise,
        receipt,
    )
}

// ─── Sync (random-pairing) ─────────────────────────────────

/// Initialize sync accumulators from current activations.
fn sync_init(
    activated: &[f32], left: &[usize], right: &[usize],
    alpha: &mut Vec<f32>, beta: &mut Vec<f32>,
) {
    let n = left.len();
    alpha.clear();
    alpha.reserve(n);
    beta.clear();
    beta.resize(n, 1.0);
    for i in 0..n {
        alpha.push(activated[left[i]] * activated[right[i]]);
    }
}

/// Update sync accumulators with exponential decay.
fn sync_update(
    activated: &[f32], left: &[usize], right: &[usize],
    alpha: &mut Vec<f32>, beta: &mut Vec<f32>, r: &[f32],
) -> Vec<f32> {
    let n = left.len();
    for i in 0..n {
        let pw = activated[left[i]] * activated[right[i]];
        alpha[i] = r[i] * alpha[i] + pw;
        beta[i] = r[i] * beta[i] + 1.0;
    }
    sync_read(alpha, beta)
}

/// Read normalized sync: alpha / sqrt(beta).
fn sync_read(alpha: &[f32], beta: &[f32]) -> Vec<f32> {
    alpha.iter().zip(beta).map(|(&a, &b)| a / b.sqrt().max(1e-8)).collect()
}

// ─── Batched sync (per-example offsets) ────────────────────────
// The sync recurrence is independent across examples, so the batched
// variants are the scalar ones with an outer batch loop: `activated` is
// `[batch × d_model]`, `alpha`/`beta`/`sync` are `[batch × n_pairs]`.
// (Tiny / host-side — no GEMM here; this is plumbing so the tick loop can
// carry B examples.)

/// Batched `sync_init`: `alpha[b,i] = activated[b,left]·activated[b,right]`,
/// `beta = 1`. Sizes `alpha`/`beta` to `batch × n_pairs`.
#[cfg_attr(not(test), allow(dead_code))]
fn sync_init_batched(
    activated: &[f32], left: &[usize], right: &[usize],
    alpha: &mut Vec<f32>, beta: &mut Vec<f32>, batch: usize, d_model: usize,
) {
    let n = left.len();
    alpha.clear();
    alpha.resize(batch * n, 0.0);
    beta.clear();
    beta.resize(batch * n, 1.0);
    for b in 0..batch {
        let ao = b * d_model;
        for i in 0..n {
            alpha[b * n + i] = activated[ao + left[i]] * activated[ao + right[i]];
        }
    }
}

/// Batched `sync_update` with exponential decay. Returns `[batch × n_pairs]`.
#[cfg_attr(not(test), allow(dead_code))]
fn sync_update_batched(
    activated: &[f32], left: &[usize], right: &[usize],
    alpha: &mut [f32], beta: &mut [f32], r: &[f32], batch: usize, d_model: usize,
) -> Vec<f32> {
    let n = left.len();
    let mut sync = vec![0.0f32; batch * n];
    for b in 0..batch {
        let ao = b * d_model;
        let so = b * n;
        for i in 0..n {
            let pw = activated[ao + left[i]] * activated[ao + right[i]];
            alpha[so + i] = r[i] * alpha[so + i] + pw;
            beta[so + i] = r[i] * beta[so + i] + 1.0;
            sync[so + i] = alpha[so + i] / beta[so + i].sqrt().max(1e-8);
        }
    }
    sync
}

/// Batched `sync_update_reverse_host` — the Bug-3 recurrence carry per
/// example. `d_sync`/`alpha`/`beta`/`d_alpha_carry` are `[batch × n_pairs]`,
/// `activated`/`d_activated` are `[batch × d_model]`.
#[cfg_attr(not(test), allow(dead_code))]
fn sync_update_reverse_host_batched(
    d_sync: &[f32], activated: &[f32], _alpha: &[f32], beta: &[f32], r: &[f32],
    left: &[usize], right: &[usize],
    d_activated: &mut [f32], d_alpha_carry: &mut [f32],
    batch: usize, d_model: usize,
) {
    let n = left.len();
    for b in 0..batch {
        let ao = b * d_model;
        let so = b * n;
        for i in 0..n {
            let inv_sqrt_beta = 1.0 / beta[so + i].max(1e-8).sqrt();
            let a_i = d_sync[so + i] * inv_sqrt_beta + r[i] * d_alpha_carry[so + i];
            d_activated[ao + left[i]] += a_i * activated[ao + right[i]];
            d_activated[ao + right[i]] += a_i * activated[ao + left[i]];
            d_alpha_carry[so + i] = a_i;
        }
    }
}

// ─── NLM (trace processor) ─────────────────────────────────

/// Per-neuron GLU: [n_neurons × 2k] → [n_neurons × k].
fn per_neuron_glu(x: &[f32], n_neurons: usize, out_per: usize) -> Vec<f32> {
    let half = out_per / 2;
    let mut result = Vec::with_capacity(n_neurons * half);
    for n in 0..n_neurons {
        let base = n * out_per;
        for j in 0..half {
            let val = x[base + j];
            let gate = 1.0 / (1.0 + (-x[base + half + j]).exp());
            result.push(val * gate);
        }
    }
    result
}

fn nlm_forward(
    trace: &[f32],
    stage1: &modgrad_compute::neuron::SuperLinear,
    stage2: Option<&modgrad_compute::neuron::SuperLinear>,
    d_model: usize,
) -> Vec<f32> {
    let s1 = stage1.forward(trace);
    let s1_glu = per_neuron_glu(&s1, d_model, stage1.out_per);

    if let Some(s2) = stage2 {
        let s2_out = s2.forward(&s1_glu);
        per_neuron_glu(&s2_out, d_model, s2.out_per)
    } else {
        s1_glu
    }
}

// ─── Multihead attention ───────────────────────────────────

fn multihead_attention(
    q_in: &[f32],
    kv_flat: &[f32],
    n_tokens: usize,
    d_input: usize,
    n_heads: usize,
    in_proj: &Linear,
    out_proj: &Linear,
) -> Vec<f32> {
    multihead_attention_with_attn(q_in, kv_flat, n_tokens, d_input, n_heads, in_proj, out_proj, None)
}

/// Like `multihead_attention` but optionally writes per-head softmax
/// weights into `attn_out` as `[head][token]`. Used by
/// `ctm_forward_with_attn_trace` for visualisation; behaviour and
/// numerics are identical to `multihead_attention` when `attn_out` is None.
fn multihead_attention_with_attn(
    q_in: &[f32],       // [d_input]
    kv_flat: &[f32],    // [n_tokens × d_input]
    n_tokens: usize,
    d_input: usize,
    n_heads: usize,
    in_proj: &Linear,    // d_input → 3×d_input
    out_proj: &Linear,   // d_input → d_input
    attn_out: Option<&mut Vec<Vec<f32>>>,
) -> Vec<f32> {
    let d_head = d_input / n_heads;
    let scale = 1.0 / (d_head as f32).sqrt();

    // Project query
    let q_full = linear_slice(q_in, in_proj, 0, d_input);

    // Project all KV tokens
    let mut k_all = Vec::with_capacity(n_tokens * d_input);
    let mut v_all = Vec::with_capacity(n_tokens * d_input);
    for t in 0..n_tokens {
        let tok = &kv_flat[t * d_input..(t + 1) * d_input];
        k_all.extend_from_slice(&linear_slice(tok, in_proj, d_input, 2 * d_input));
        v_all.extend_from_slice(&linear_slice(tok, in_proj, 2 * d_input, 3 * d_input));
    }

    // Per-head scaled dot-product attention
    let mut concat_heads = Vec::with_capacity(d_input);
    let collect_attn = attn_out.is_some();
    let mut per_head_weights: Vec<Vec<f32>> = if collect_attn {
        Vec::with_capacity(n_heads)
    } else { Vec::new() };
    for h in 0..n_heads {
        let q_h = &q_full[h * d_head..(h + 1) * d_head];

        // Scores
        let mut scores = Vec::with_capacity(n_tokens);
        for t in 0..n_tokens {
            let k_h = &k_all[t * d_input + h * d_head..t * d_input + (h + 1) * d_head];
            let dot: f32 = q_h.iter().zip(k_h).map(|(&a, &b)| a * b).sum();
            scores.push(dot * scale);
        }

        // Softmax
        let max_s = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_s: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp()).collect();
        let sum_s: f32 = exp_s.iter().sum();
        let inv_sum = 1.0 / sum_s;

        // Weighted sum of values
        let mut head_out = vec![0.0f32; d_head];
        let mut head_weights: Vec<f32> = if collect_attn {
            Vec::with_capacity(n_tokens)
        } else { Vec::new() };
        for t in 0..n_tokens {
            let w = exp_s[t] * inv_sum;
            if collect_attn { head_weights.push(w); }
            let v_h = &v_all[t * d_input + h * d_head..t * d_input + (h + 1) * d_head];
            for j in 0..d_head {
                head_out[j] += w * v_h[j];
            }
        }
        if collect_attn { per_head_weights.push(head_weights); }
        concat_heads.extend_from_slice(&head_out);
    }

    if let Some(out) = attn_out { *out = per_head_weights; }
    out_proj.forward(&concat_heads)
}

/// Partial matvec: compute rows `[row_start, row_end)` of
/// `out = linear.weight @ x + linear.bias`.
///
/// The slice `linear.weight[row_start * in_dim .. row_end * in_dim]`
/// is contiguous (weight is row-major `[out_dim × in_dim]`), so the
/// partial computation is itself a well-formed matvec on the weight
/// sub-block — dispatched through `ops::matvec`.
fn linear_slice(x: &[f32], linear: &Linear, row_start: usize, row_end: usize) -> Vec<f32> {
    let in_dim = linear.in_dim;
    let out_dim = row_end - row_start;
    let w_slice = &linear.weight[row_start * in_dim..row_end * in_dim];
    let b_slice = &linear.bias[row_start..row_end];
    let mut out = vec![0.0f32; out_dim];
    modgrad_device::backend::ops::matvec(
        x, w_slice, b_slice, &mut out,
        out_dim, in_dim,
        modgrad_device::backend::QuantKind::F32,
    ).expect("linear_slice: matvec dispatch");
    out
}

// ─── Certainty ─────────────────────────────────────────────

pub fn compute_certainty_pub(prediction: &[f32]) -> [f32; 2] { compute_certainty(prediction) }

fn compute_certainty(prediction: &[f32]) -> [f32; 2] {
    let n = prediction.len();
    if n <= 1 { return [0.0, 1.0]; }
    let max_p = prediction.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_s: Vec<f32> = prediction.iter().map(|&x| (x - max_p).exp()).collect();
    let sum: f32 = exp_s.iter().sum();
    let max_ent = (n as f32).ln();
    let ent: f32 = exp_s.iter()
        .map(|&e| { let p = e / sum; if p > 1e-10 { -p * p.ln() } else { 0.0 } })
        .sum();
    let ne = if max_ent > 0.0 { (ent / max_ent).clamp(0.0, 1.0) } else { 0.0 };
    [ne, 1.0 - ne]
}

// ─── Affine LayerNorm ──────────────────────────────────────

fn affine_ln(x: &mut [f32], gamma: &[f32], beta: &[f32]) {
    let n = x.len();
    if n == 0 { return; }
    let nf = n as f32;
    let mean: f32 = x.iter().sum::<f32>() / nf;
    let var: f32 = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / nf;
    let inv_std = 1.0 / (var + 1e-5).sqrt();
    for i in 0..n {
        x[i] = gamma[i] * (x[i] - mean) * inv_std + beta[i];
    }
}
// Device-generic CTM forward pass. v0 covers the core path:
//   - Raw single-token input
//   - Standard tick loop with kv_proj → MHA → synapse → trace shift
//     → NLM → output_proj
//   - ExitStrategy::None (no exit gate)
//   - No episodic memory
//   - No trajectory collection
//
// Big matvecs (kv_proj, q_proj, mha_in/out_proj, output_proj),
// synapse U-Net, and NLM SuperLinears all route through the typed
// cascade — for D = Rocm, weights stay on-device.
//
// Orchestration ops (sync_init/sync_update, per-head softmax+reduction
// in MHA, per-neuron GLU, trace shift) currently fall back to host
// computation per tick. They are small (O(d_model × n_heads × n_tokens))
// and the matmuls dwarf them; full Path-C kernelisation is a v1
// optimisation tracked as TODO. The download/upload bracket around
// these ops is the visible host-fallback PCIe cost.

use modgrad_device::backend::tensor as tensor_typed;
use modgrad_device::backend::tensor::{Device as TensorDevice, Tensor};
use modgrad_device::backend::BackendError;
use crate::weights::CtmWeightsTyped;

/// Forward pass through a typed CTM cell. Mirrors `ctm_forward` for
/// the supported feature subset (single-token Raw input, no episodic,
/// no exit gate, no trajectory). Returns identical output (within
/// float tolerance) to the untyped path on the same weights and input.
pub fn ctm_forward_typed<D: TensorDevice>(
    w: &CtmWeightsTyped<D>,
    activated: &mut Vec<f32>,   // [d_model], persistent across ticks
    trace: &mut Vec<f32>,       // [d_model × memory_length], persistent
    obs: &[f32],                 // [raw_input_dim] single token
) -> Result<CtmOutput, BackendError> {
    let cfg = &w.config;
    let d = cfg.d_model;
    let d_in = cfg.d_input;
    let k = cfg.iterations;
    let m = cfg.memory_length;

    // ── Step 1: project the single observation token → d_input ──
    let obs_t = Tensor::<D>::from_slice(obs)?;
    let proj_pre_ln = w.kv_proj.forward(&obs_t)?;
    let mut kv_t = Tensor::<D>::zeros(d_in)?;
    tensor_typed::layer_norm(
        &proj_pre_ln, &w.kv_ln_gamma, &w.kv_ln_beta,
        &mut kv_t, 1, d_in,
    )?;
    // kv on host for the multi-head softmax dance (host-fallback
    // orchestration; see module-level comment).
    let kv_host: Vec<f32> = kv_t.to_vec()?;
    let n_tokens = 1;

    // ── Step 2: tick-loop bookkeeping (mirrors ctm_forward_with_kv) ──
    let mut alpha_out: Vec<f32> = Vec::new();
    let mut beta_out: Vec<f32> = Vec::new();
    let decay_out_host: Vec<f32> = w.decay_params_out.to_vec()?
        .iter().map(|&p| (-p.clamp(0.0, 15.0)).exp()).collect();
    let decay_action_host: Vec<f32> = w.decay_params_action.to_vec()?
        .iter().map(|&p| (-p.clamp(0.0, 15.0)).exp()).collect();

    sync_init(activated, &w.sync_out_left, &w.sync_out_right,
        &mut alpha_out, &mut beta_out);

    let mut alpha_action: Vec<f32> = Vec::new();
    let mut beta_action: Vec<f32> = Vec::new();
    let mut action_initialized = false;

    let mut predictions = Vec::with_capacity(k);
    let mut certainties = Vec::with_capacity(k);

    // ── Step 3: tick loop. Same structure as ctm_forward_with_kv. ──
    for _tick in 0..k {
        // 3a. sync_action via host (small per-pair).
        let sync_action: Vec<f32> = if !action_initialized {
            sync_init(activated, &w.sync_action_left, &w.sync_action_right,
                &mut alpha_action, &mut beta_action);
            action_initialized = true;
            sync_read(&alpha_action, &beta_action)
        } else {
            sync_update(activated, &w.sync_action_left, &w.sync_action_right,
                &mut alpha_action, &mut beta_action, &decay_action_host)
        };

        // 3b. q_proj on device.
        let sa_t = Tensor::<D>::from_slice(&sync_action)?;
        let q_t = w.q_proj.forward(&sa_t)?;

        // 3c. MHA. v0: download q, run host MHA helper. The two big
        //     matvecs (mha_in_proj, mha_out_proj) go through typed
        //     Linear; the per-head softmax+sum stays host (it's
        //     n_heads × n_tokens × d_head — small for typical CTM).
        let q_host: Vec<f32> = q_t.to_vec()?;
        let attn_out_host = multihead_attention_typed_helper::<D>(
            &q_host, &kv_host, n_tokens, d_in, cfg.heads,
            &w.mha_in_proj, &w.mha_out_proj,
        )?;

        // 3d. concat(attn_out, activated) → synapse U-Net.
        let mut pre_syn_host = Vec::with_capacity(d_in + d);
        pre_syn_host.extend_from_slice(&attn_out_host);
        pre_syn_host.extend_from_slice(activated);
        let pre_syn_t = Tensor::<D>::from_slice(&pre_syn_host)?;
        let pre_act_t = w.synapse.forward(&pre_syn_t)?;
        let pre_act_host: Vec<f32> = pre_act_t.to_vec()?;

        // 3e. trace shift (host bookkeeping).
        for n in 0..d {
            let base = n * m;
            trace.copy_within(base + 1..base + m, base);
            trace[base + m - 1] = pre_act_host[n];
        }

        // 3f. NLM stage1 + GLU + optional stage2 + GLU.
        let trace_t = Tensor::<D>::from_slice(trace)?;
        let activated_new_host = nlm_forward_typed_helper::<D>(
            &trace_t, &w.nlm_stage1, w.nlm_stage2.as_ref(), d,
        )?;
        *activated = activated_new_host;

        // 3g. sync_out update.
        let sync_out = sync_update(activated, &w.sync_out_left, &w.sync_out_right,
            &mut alpha_out, &mut beta_out, &decay_out_host);

        // 3h. output_proj on device.
        let so_t = Tensor::<D>::from_slice(&sync_out)?;
        let pred_t = w.output_proj.forward(&so_t)?;
        let pred_host: Vec<f32> = pred_t.to_vec()?;
        let cert = compute_certainty(&pred_host);
        predictions.push(pred_host);
        certainties.push(cert);
    }

    let ticks_used = predictions.len();
    let sync_out_final = sync_read(&alpha_out, &beta_out);
    Ok(CtmOutput {
        predictions,
        certainties,
        sync_out: sync_out_final,
        exit_lambdas: Vec::new(),
        ticks_used,
        trajectory: Vec::new(),
    })
}

// ─── ctm_forward_typed_with_cache + ctm_backward_typed ───────
//
// Slice 4 orchestration: forward returns every intermediate the
// matched backward needs; backward walks reverse-tick consuming
// the cache. v0 covers the same feature subset as ctm_forward_typed
// (single-token, no episodic, no exit gate) — full features land
// after the parity loop is closed.

/// Per-tick cache populated by `ctm_forward_typed_with_cache`.
pub struct TickCacheTyped<D: TensorDevice> {
    pub sync_action: Vec<f32>,
    pub q: Tensor<D>,
    pub mha: MhaCacheTyped<D>,
    pub attn_out: Tensor<D>,
    pub activated_pre_tick: Vec<f32>,    // activated[] before this tick's NLM
    pub activated_post: Vec<f32>,         // activated[] after this tick's NLM (sync_out reads this)
    pub trace_pre_tick: Vec<f32>,         // trace before shift
    pub synapse_input: Tensor<D>,         // [d_input + d_model] the "concat(attn, activated)"
    pub synapse_cache: super::synapse::SynapseUNetCacheTyped<D>,
    pub pre_act: Tensor<D>,                // synapse output (= NLM input via shift)
    pub trace_post_shift: Vec<f32>,
    pub nlm: NlmCacheTyped<D>,
    pub sync_out_for_pred: Vec<f32>,       // post-update sync, fed into output_proj
    pub alpha_out_post: Vec<f32>,
    pub beta_out_post: Vec<f32>,
    pub alpha_action_post: Vec<f32>,
    pub beta_action_post: Vec<f32>,
}

/// Top-level cache: per-tick + KV path intermediates from before the loop.
pub struct CtmCacheTyped<D: TensorDevice> {
    pub obs: Tensor<D>,                    // raw obs token
    pub kv_pre_ln: Tensor<D>,              // kv_proj output before LN
    pub kv_ln_cache: Tensor<D>,            // mean+rstd from layer_norm_train
    pub kv: Tensor<D>,                     // post-LN, fed into MHA per tick
    pub ticks: Vec<TickCacheTyped<D>>,
    pub d_input: usize,
    pub d_model: usize,
    pub raw_input_dim: usize,
    pub memory_length: usize,
}

/// Forward pass with full cache for backward. Same algorithm as
/// `ctm_forward_typed` but captures every intermediate.
#[allow(clippy::too_many_arguments)]
pub fn ctm_forward_typed_with_cache<D: TensorDevice>(
    w: &CtmWeightsTyped<D>,
    activated: &mut Vec<f32>,
    trace: &mut Vec<f32>,
    obs: &[f32],
) -> Result<(CtmOutput, CtmCacheTyped<D>), BackendError> {
    let cfg = &w.config;
    let d = cfg.d_model;
    let d_in = cfg.d_input;
    let k = cfg.iterations;
    let m = cfg.memory_length;

    // KV projection + LN with cache.
    let obs_t = Tensor::<D>::from_slice(obs)?;
    let kv_pre_ln = w.kv_proj.forward(&obs_t)?;
    let mut kv_post = Tensor::<D>::zeros(d_in)?;
    let mut kv_ln_cache = Tensor::<D>::zeros(2)?;
    tensor_typed::layer_norm_train(
        &kv_pre_ln, &w.kv_ln_gamma, &w.kv_ln_beta,
        &mut kv_post, &mut kv_ln_cache, 1, d_in,
    )?;
    let kv_host: Vec<f32> = kv_post.to_vec()?;
    let n_tokens = 1;

    // Bookkeeping.
    let mut alpha_out: Vec<f32> = Vec::new();
    let mut beta_out: Vec<f32> = Vec::new();
    let decay_out_host: Vec<f32> = w.decay_params_out.to_vec()?
        .iter().map(|&p| (-p.clamp(0.0, 15.0)).exp()).collect();
    let decay_action_host: Vec<f32> = w.decay_params_action.to_vec()?
        .iter().map(|&p| (-p.clamp(0.0, 15.0)).exp()).collect();
    sync_init(activated, &w.sync_out_left, &w.sync_out_right,
        &mut alpha_out, &mut beta_out);

    let mut alpha_action: Vec<f32> = Vec::new();
    let mut beta_action: Vec<f32> = Vec::new();
    let mut action_initialized = false;

    let mut predictions = Vec::with_capacity(k);
    let mut certainties = Vec::with_capacity(k);
    let mut tick_caches: Vec<TickCacheTyped<D>> = Vec::with_capacity(k);

    for _tick in 0..k {
        let activated_pre_tick = activated.clone();
        let trace_pre_tick = trace.clone();

        // sync_action.
        let sync_action: Vec<f32> = if !action_initialized {
            sync_init(activated, &w.sync_action_left, &w.sync_action_right,
                &mut alpha_action, &mut beta_action);
            action_initialized = true;
            sync_read(&alpha_action, &beta_action)
        } else {
            sync_update(activated, &w.sync_action_left, &w.sync_action_right,
                &mut alpha_action, &mut beta_action, &decay_action_host)
        };
        let alpha_action_post = alpha_action.clone();
        let beta_action_post = beta_action.clone();

        // q_proj.
        let sa_t = Tensor::<D>::from_slice(&sync_action)?;
        let q_t = w.q_proj.forward(&sa_t)?;
        let q_host: Vec<f32> = q_t.to_vec()?;

        // MHA with cache.
        let (attn_out_h, mha_cache) = multihead_attention_with_cache_typed::<D>(
            &q_host, &kv_host, n_tokens, d_in, cfg.heads,
            &w.mha_in_proj, &w.mha_out_proj,
        )?;
        let attn_out_t = Tensor::<D>::from_slice(&attn_out_h)?;

        // synapse input = concat(attn_out, activated).
        let mut pre_syn_host = Vec::with_capacity(d_in + d);
        pre_syn_host.extend_from_slice(&attn_out_h);
        pre_syn_host.extend_from_slice(activated);
        let pre_syn_t = Tensor::<D>::from_slice(&pre_syn_host)?;
        let (pre_act_t, synapse_cache) = w.synapse.forward_with_cache(&pre_syn_t)?;
        let pre_act_host: Vec<f32> = pre_act_t.to_vec()?;

        // Trace shift.
        for n in 0..d {
            let base = n * m;
            trace.copy_within(base + 1..base + m, base);
            trace[base + m - 1] = pre_act_host[n];
        }
        let trace_post_shift = trace.clone();

        // NLM forward with cache.
        let trace_t = Tensor::<D>::from_slice(trace)?;
        let (activated_new_host, nlm_cache) = nlm_forward_with_cache::<D>(
            &trace_t, &w.nlm_stage1, w.nlm_stage2.as_ref(), d,
        )?;
        *activated = activated_new_host;
        let activated_post = activated.clone();

        // sync_out update.
        let sync_out = sync_update(activated, &w.sync_out_left, &w.sync_out_right,
            &mut alpha_out, &mut beta_out, &decay_out_host);
        let alpha_out_post = alpha_out.clone();
        let beta_out_post = beta_out.clone();

        // output_proj.
        let so_t = Tensor::<D>::from_slice(&sync_out)?;
        let pred_t = w.output_proj.forward(&so_t)?;
        let pred_host: Vec<f32> = pred_t.to_vec()?;
        let cert = compute_certainty(&pred_host);
        predictions.push(pred_host);
        certainties.push(cert);

        tick_caches.push(TickCacheTyped {
            sync_action,
            q: q_t,
            mha: mha_cache,
            attn_out: attn_out_t,
            activated_pre_tick,
            activated_post,
            trace_pre_tick,
            synapse_input: pre_syn_t,
            synapse_cache,
            pre_act: pre_act_t,
            trace_post_shift,
            nlm: nlm_cache,
            sync_out_for_pred: sync_out,
            alpha_out_post,
            beta_out_post,
            alpha_action_post,
            beta_action_post,
        });
    }

    let ticks_used = predictions.len();
    let sync_out_final = sync_read(&alpha_out, &beta_out);
    let output = CtmOutput {
        predictions, certainties,
        sync_out: sync_out_final,
        exit_lambdas: Vec::new(),
        ticks_used,
        trajectory: Vec::new(),
    };
    let cache = CtmCacheTyped {
        obs: obs_t,
        kv_pre_ln,
        kv_ln_cache,
        kv: kv_post,
        ticks: tick_caches,
        d_input: d_in,
        d_model: d,
        raw_input_dim: obs.len(),
        memory_length: m,
    };
    Ok((output, cache))
}

/// Batched per-neuron GLU over `[batch × n_neurons × out_per]`.
fn per_neuron_glu_batched(x: &[f32], n_neurons: usize, out_per: usize, batch: usize) -> Vec<f32> {
    let ex_in = n_neurons * out_per;
    let ex_out = n_neurons * (out_per / 2);
    let mut out = vec![0.0f32; batch * ex_out];
    for b in 0..batch {
        let g = per_neuron_glu(&x[b * ex_in..(b + 1) * ex_in], n_neurons, out_per);
        out[b * ex_out..(b + 1) * ex_out].copy_from_slice(&g);
    }
    out
}

/// Batched NLM forward — `trace` is `[batch × d_model × m]`, returns the
/// batched activated `[batch × d_model × out/2]` (flattened per example).
fn nlm_forward_batched<D: TensorDevice>(
    trace: &Tensor<D>,
    stage1: &tensor_typed::SuperLinear<D>,
    stage2: Option<&tensor_typed::SuperLinear<D>>,
    d_model: usize,
    batch: usize,
) -> Result<Vec<f32>, BackendError> {
    let s1 = stage1.forward_batched(trace, batch)?;
    let s1_glu = per_neuron_glu_batched(&s1.to_vec()?, d_model, stage1.out_per, batch);
    if let Some(s2) = stage2 {
        let s1_glu_t = Tensor::<D>::from_slice(&s1_glu)?;
        let s2_out = s2.forward_batched(&s1_glu_t, batch)?;
        Ok(per_neuron_glu_batched(&s2_out.to_vec()?, d_model, s2.out_per, batch))
    } else {
        Ok(s1_glu)
    }
}

/// Batched forward tick loop — the wiring capstone. Threads `batch`
/// through the whole per-tick op chain: GEMM-batched synapse / q_proj /
/// output_proj / kv_proj, batched NLM + sync, B-loop attention (small).
/// `activated` is `[batch × d_model]`, `trace` `[batch × d_model × m]`,
/// `obs` `[batch × raw_input_dim]`. Returns per-tick predictions, each
/// `[batch × out_dims]`. Forward only (no cache) — validates the batched
/// composition against B scalar `ctm_forward_typed_with_cache` calls.
/// (Backward + adaptive-exit halt-mask are the remaining wiring step.)
pub fn ctm_forward_typed_batched<D: TensorDevice>(
    w: &CtmWeightsTyped<D>,
    activated: &mut Vec<f32>,
    trace: &mut Vec<f32>,
    obs: &[f32],
    batch: usize,
) -> Result<(Vec<Vec<f32>>, Vec<usize>), BackendError> {
    let cfg = &w.config;
    let d = cfg.d_model;
    let d_in = cfg.d_input;
    let k = cfg.iterations;
    let m = cfg.memory_length;
    let raw = obs.len() / batch;
    let n_tokens = 1;

    // KV projection + LN (batched).
    let obs_t = Tensor::<D>::from_slice(obs)?;
    let mut kv_pre = Tensor::<D>::zeros(batch * d_in)?;
    w.kv_proj.forward_batched(&obs_t, &mut kv_pre, batch)?;
    let mut kv_post = Tensor::<D>::zeros(batch * d_in)?;
    let mut kv_ln_cache = Tensor::<D>::zeros(2 * batch)?;
    tensor_typed::layer_norm_train(&kv_pre, &w.kv_ln_gamma, &w.kv_ln_beta,
        &mut kv_post, &mut kv_ln_cache, batch, d_in)?;
    let kv_host: Vec<f32> = kv_post.to_vec()?;
    let _ = raw;

    let decay_out: Vec<f32> = w.decay_params_out.to_vec()?
        .iter().map(|&p| (-p.clamp(0.0, 15.0)).exp()).collect();
    let decay_action: Vec<f32> = w.decay_params_action.to_vec()?
        .iter().map(|&p| (-p.clamp(0.0, 15.0)).exp()).collect();

    let mut alpha_out = Vec::new();
    let mut beta_out = Vec::new();
    sync_init_batched(activated, &w.sync_out_left, &w.sync_out_right,
        &mut alpha_out, &mut beta_out, batch, d);
    let mut alpha_action = Vec::new();
    let mut beta_action = Vec::new();
    let mut action_init = false;

    let mut predictions: Vec<Vec<f32>> = Vec::with_capacity(k);
    for _tick in 0..k {
        // sync_action (batched).
        let sync_action: Vec<f32> = if !action_init {
            sync_init_batched(activated, &w.sync_action_left, &w.sync_action_right,
                &mut alpha_action, &mut beta_action, batch, d);
            action_init = true;
            sync_read(&alpha_action, &beta_action)
        } else {
            sync_update_batched(activated, &w.sync_action_left, &w.sync_action_right,
                &mut alpha_action, &mut beta_action, &decay_action, batch, d)
        };

        // q_proj (batched).
        let sa_t = Tensor::<D>::from_slice(&sync_action)?;
        let mut q_t = Tensor::<D>::zeros(batch * d_in)?;
        w.q_proj.forward_batched(&sa_t, &mut q_t, batch)?;
        let q_host: Vec<f32> = q_t.to_vec()?;

        // MHA (B-loop — small).
        let mut attn_all = vec![0.0f32; batch * d_in];
        for b in 0..batch {
            let (ao, _c) = multihead_attention_with_cache_typed::<D>(
                &q_host[b * d_in..(b + 1) * d_in],
                &kv_host[b * d_in..(b + 1) * d_in],
                n_tokens, d_in, cfg.heads, &w.mha_in_proj, &w.mha_out_proj,
            )?;
            attn_all[b * d_in..(b + 1) * d_in].copy_from_slice(&ao);
        }

        // synapse input = concat(attn_out, activated) per example.
        let mut pre_syn = vec![0.0f32; batch * (d_in + d)];
        for b in 0..batch {
            let o = b * (d_in + d);
            pre_syn[o..o + d_in].copy_from_slice(&attn_all[b * d_in..(b + 1) * d_in]);
            pre_syn[o + d_in..o + d_in + d].copy_from_slice(&activated[b * d..(b + 1) * d]);
        }
        let pre_syn_t = Tensor::<D>::from_slice(&pre_syn)?;
        let pre_act_t = w.synapse.forward_batched(&pre_syn_t, batch)?;
        let pre_act_host: Vec<f32> = pre_act_t.to_vec()?;

        // Trace shift (per example).
        for b in 0..batch {
            let to = b * d * m;
            let ao = b * d;
            for n in 0..d {
                let base = to + n * m;
                trace.copy_within(base + 1..base + m, base);
                trace[base + m - 1] = pre_act_host[ao + n];
            }
        }

        // NLM (batched).
        let trace_t = Tensor::<D>::from_slice(trace)?;
        *activated = nlm_forward_batched::<D>(&trace_t, &w.nlm_stage1, w.nlm_stage2.as_ref(), d, batch)?;

        // sync_out (batched) → output_proj (batched).
        let sync_out = sync_update_batched(activated, &w.sync_out_left, &w.sync_out_right,
            &mut alpha_out, &mut beta_out, &decay_out, batch, d);
        let so_t = Tensor::<D>::from_slice(&sync_out)?;
        let mut pred_t = Tensor::<D>::zeros(batch * cfg.out_dims)?;
        w.output_proj.forward_batched(&so_t, &mut pred_t, batch)?;
        predictions.push(pred_t.to_vec()?);
    }

    // ── Adaptive-exit halt mask (run max-T, mask the output per example). ──
    // All B examples ran all k ticks (uniform shape — GPU-friendly). Now,
    // per example, find the first tick whose exit condition fires and FREEZE
    // its output there: later ticks' predictions are overwritten with the
    // halt-tick prediction (equivalent to freezing that example's state,
    // since a frozen state re-emits the same output_proj result).
    let od = cfg.out_dims;
    let mut ticks_used = vec![k; batch];
    for b in 0..batch {
        for tick in 0..k {
            let pred_b = &predictions[tick][b * od..(b + 1) * od];
            let halt = match &cfg.exit_strategy {
                crate::config::ExitStrategy::Certainty { threshold } => {
                    compute_certainty(pred_b)[1] > *threshold
                }
                // None and AdaptiveGate (needs the per-tick gate forward —
                // a follow-up) run all ticks here.
                _ => false,
            };
            if halt {
                ticks_used[b] = tick + 1;
                let frozen: Vec<f32> = pred_b.to_vec();
                for later in (tick + 1)..k {
                    predictions[later][b * od..(b + 1) * od].copy_from_slice(&frozen);
                }
                break;
            }
        }
    }
    Ok((predictions, ticks_used))
}

/// CTM backward orchestration. Given upstream `d_predictions[t]` for
/// each tick (typically from a loss function), walk reverse-tick and
/// accumulate gradients into `grads`. Returns `d_obs` (gradient flowing
/// into the raw observation) for upstream connection-synapse backward.
///
/// **Output convention:** all weight grads ACCUMULATE; `d_obs` OVERWRITTEN.
#[allow(clippy::too_many_arguments)]
pub fn ctm_backward_typed<D: TensorDevice>(
    w: &CtmWeightsTyped<D>,
    cache: &CtmCacheTyped<D>,
    d_predictions: &[Vec<f32>],
    grads: &mut crate::weights::CtmGradientsTyped<D>,
    unet_grads: &mut crate::synapse::UNetGradsTyped<D>,
    d_obs: &mut Tensor<D>,
) -> Result<(), BackendError> {
    let cfg = &w.config;
    let d = cfg.d_model;
    let d_in = cfg.d_input;
    let m = cfg.memory_length;

    // Carries across ticks (reverse-tick walk):
    //   d_kv          — every tick reads the same kv from MHA, so its gradient sums.
    //   d_activated_carry — synapse U-Net backward at tick T emits
    //     d_synapse_input[d_in..] = d activated_pre_tick[T] = d activated_post_tick[T-1].
    //     We thread this back into the next iteration's d_activated (the earlier tick).
    let mut d_kv = Tensor::<D>::zeros(d_in)?;
    let mut d_activated_carry: Vec<f32> = vec![0.0f32; d];
    // Cross-tick trace gradient: trace is persistent state, so the
    // shift-adjoint's `d_trace_old` (gradient w.r.t. the pre-shift trace =
    // the previous tick's post-shift trace) must be threaded into the next
    // (earlier) reverse iteration. Dropping it severs NLM(tick T) → older
    // trace slots → pre_act(tick T-1).
    let mut d_trace_carry: Vec<f32> = vec![0.0f32; d * m];
    // Bug-3: sync accumulator recurrence trail (A_t = dL/d alpha_t) for the
    // out- and action-sync readouts. r[i] = exp(-decay[i]). Zero-init = A_{T+1}.
    let r_out: Vec<f32> = w.decay_params_out.to_vec()?
        .iter().map(|&p| (-p.clamp(0.0, 15.0)).exp()).collect();
    let r_action: Vec<f32> = w.decay_params_action.to_vec()?
        .iter().map(|&p| (-p.clamp(0.0, 15.0)).exp()).collect();
    let mut d_alpha_out_carry: Vec<f32> = vec![0.0f32; cfg.n_synch_out];
    let mut d_alpha_action_carry: Vec<f32> = vec![0.0f32; cfg.n_synch_action];

    // Walk ticks in reverse.
    let n_ticks = d_predictions.len().min(cache.ticks.len());
    for tick_idx in (0..n_ticks).rev() {
        let tick_cache = &cache.ticks[tick_idx];
        let d_pred_h = &d_predictions[tick_idx];
        let d_pred_t = Tensor::<D>::from_slice(d_pred_h)?;

        // Step 1: output_proj backward → d_sync_out.
        let so_t = Tensor::<D>::from_slice(&tick_cache.sync_out_for_pred)?;
        let mut d_sync_out = Tensor::<D>::zeros(cfg.n_synch_out)?;
        grads.backward_output_proj(w, &d_pred_t, &so_t, &mut d_sync_out)?;
        let d_sync_out_h: Vec<f32> = d_sync_out.to_vec()?;

        // Step 2: sync_update reverse → d_activated (host). Then add
        // the cross-tick carry from the synapse input of tick T+1
        // (which feeds this tick's post-tick activated state).
        let mut d_activated = vec![0.0f32; d];
        sync_update_reverse_host(
            &d_sync_out_h,
            &tick_cache.activated_post,      // sync_out readout is from POST-tick activated
            &tick_cache.alpha_out_post,
            &tick_cache.beta_out_post,
            &r_out,
            &w.sync_out_left, &w.sync_out_right,
            &mut d_activated,
            &mut d_alpha_out_carry,
        );
        for i in 0..d { d_activated[i] += d_activated_carry[i]; }

        // Step 3: NLM backward → d_trace.
        let d_activated_t = Tensor::<D>::from_slice(&d_activated)?;
        let mut d_trace = Tensor::<D>::zeros(d * m)?;
        let (s2_w, s2_b) = match (
            grads.nlm_s2_w.as_mut(),
            grads.nlm_s2_b.as_mut(),
        ) {
            (Some(w), Some(b)) => (Some(w), Some(b)),
            _ => (None, None),
        };
        nlm_backward_typed_helper::<D>(
            &d_activated_t, &tick_cache.nlm,
            &w.nlm_stage1, w.nlm_stage2.as_ref(),
            &mut grads.nlm_s1_w, &mut grads.nlm_s1_b,
            s2_w, s2_b,
            &mut d_trace,
        )?;

        // Step 4: trace shift adjoint → d_pre_act + d_trace_old.
        // Fold in the cross-tick trace carry, then refill it from this
        // tick's d_trace_old for the next (earlier) iteration.
        let mut d_trace_h: Vec<f32> = d_trace.to_vec()?;
        for i in 0..d * m { d_trace_h[i] += d_trace_carry[i]; }
        let mut d_pre_act = vec![0.0f32; d];
        let mut d_trace_old = vec![0.0f32; d * m];
        trace_shift_adjoint_host(&d_trace_h, &mut d_pre_act, &mut d_trace_old, d, m);
        d_trace_carry.copy_from_slice(&d_trace_old);

        // Step 5: synapse U-Net backward → d_synapse_input (= d concat(attn, activated)).
        let d_pre_act_t = Tensor::<D>::from_slice(&d_pre_act)?;
        let mut d_synapse_input = Tensor::<D>::zeros(d_in + d)?;
        w.synapse.backward(
            &d_pre_act_t, &tick_cache.synapse_cache, unet_grads, &mut d_synapse_input,
        )?;
        let d_synapse_input_h: Vec<f32> = d_synapse_input.to_vec()?;
        let d_attn_out = &d_synapse_input_h[..d_in];
        // The activated half of d_synapse_input carries gradient that
        // contributes to the *previous tick's* activated. Save it as
        // the carry for the next iteration of this reverse-tick loop.
        d_activated_carry.clear();
        d_activated_carry.extend_from_slice(&d_synapse_input_h[d_in..]);

        // Step 6: MHA backward → d_q_in + d_kv_tokens.
        let d_attn_out_t = Tensor::<D>::from_slice(d_attn_out)?;
        let (d_q_in, d_kv_tokens) = multihead_attention_backward_typed::<D>(
            &d_attn_out_t, &tick_cache.mha, 1, d_in, cfg.heads,
            &w.mha_in_proj, &w.mha_out_proj,
            &mut grads.mha_in_w, &mut grads.mha_in_b,
            &mut grads.mha_out_w, &mut grads.mha_out_b,
        )?;

        // Accumulate d_kv across ticks (each tick reads same kv).
        let d_kv_t0 = Tensor::<D>::from_slice(&d_kv_tokens[0])?;
        tensor_typed::add_assign(&mut d_kv, &d_kv_t0, d_in)?;

        // Step 7: q_proj backward → d_sync_action. sync_action[T] is a
        // function of activated_pre_tick[T] = activated_post_tick[T-1],
        // so d_sync_action's gradient feeds the cross-tick carry.
        let d_q_t = Tensor::<D>::from_slice(&d_q_in)?;
        let mut d_sync_action = Tensor::<D>::zeros(cfg.n_synch_action)?;
        grads.backward_q_proj(w, &d_q_t, &Tensor::<D>::from_slice(&tick_cache.sync_action)?,
            &mut d_sync_action)?;
        let d_sync_action_h: Vec<f32> = d_sync_action.to_vec()?;
        sync_update_reverse_host(
            &d_sync_action_h,
            &tick_cache.activated_pre_tick,
            &tick_cache.alpha_action_post,
            &tick_cache.beta_action_post,
            &r_action,
            &w.sync_action_left, &w.sync_action_right,
            &mut d_activated_carry,
            &mut d_alpha_action_carry,
        );

        let _ = tick_idx;
    }

    // After tick loop: kv_ln backward → d_kv_pre_ln, then kv_proj backward → d_obs.
    let mut d_kv_pre_ln = Tensor::<D>::zeros(d_in)?;
    grads.backward_kv_ln(
        w, &d_kv, &cache.kv_pre_ln, &cache.kv_ln_cache,
        &mut d_kv_pre_ln, d_in,
    )?;
    grads.backward_kv_proj(w, &d_kv_pre_ln, &cache.obs, d_obs)?;

    Ok(())
}

/// Alternate CTM backward entry point: gradient enters via the
/// post-last-tick `activated[]` state, NOT via per-tick predictions.
///
/// **Why:** in the regional cascade, the brain-level `output_proj`
/// produces predictions; per-region CTMs feed only their final
/// `activated[]` upward into the global sync. So the per-region
/// backward must accept `d_activated_post_last_tick` as the seed
/// instead of synthesising fake `d_predictions`.
///
/// **Semantics relative to `ctm_backward_typed`:**
/// - Skips Step 1 (output_proj backward) — no per-region prediction loss.
/// - Skips Step 2 (sync_update reverse) — no per-region sync produces
///   loss at the regional level.
/// - At the last tick: `d_activated` is seeded from
///   `d_activated_post_last_tick`.
/// - At earlier ticks: `d_activated` is zero (cross-tick carry through
///   activated_pre_tick is currently dropped — same v0 limitation as
///   `ctm_backward_typed`).
/// - Walks Step 3 (NLM) → 4 (trace shift) → 5 (synapse) → 6 (MHA)
///   → 7 (q_proj) for every tick.
/// - After tick loop: kv_ln backward → kv_proj backward → `d_obs`.
///
/// **Output convention:** all weight grads ACCUMULATE; `d_obs`
/// OVERWRITTEN.
#[allow(clippy::too_many_arguments)]
pub fn ctm_backward_from_d_activated<D: TensorDevice>(
    w: &CtmWeightsTyped<D>,
    cache: &CtmCacheTyped<D>,
    d_activated_post_last_tick: &Tensor<D>,
    grads: &mut crate::weights::CtmGradientsTyped<D>,
    unet_grads: &mut crate::synapse::UNetGradsTyped<D>,
    d_obs: &mut Tensor<D>,
) -> Result<(), BackendError> {
    let cfg = &w.config;
    let d = cfg.d_model;
    let d_in = cfg.d_input;
    let m = cfg.memory_length;

    // Carries across ticks:
    //   d_kv             — per-MHA-tick gradient sums (every tick reads same kv).
    //   d_activated_carry — synapse U-Net backward at tick T emits
    //     d_synapse_input[d_in..] = d activated_pre_tick[T] = d activated_post_tick[T-1].
    //     Threaded back into the next iteration's d_activated. At the last tick
    //     the carry is seeded from `d_activated_post_last_tick` (the regional
    //     gradient flowing in via global sync).
    let mut d_kv = Tensor::<D>::zeros(d_in)?;

    let n_ticks = cache.ticks.len();
    let last_tick = n_ticks.saturating_sub(1);

    let d_act_seed_host: Vec<f32> = d_activated_post_last_tick.to_vec()?;
    if d_act_seed_host.len() != d {
        return Err(BackendError::Runtime(format!(
            "ctm_backward_from_d_activated: d_activated len {} != d_model {}",
            d_act_seed_host.len(), d,
        )));
    }
    let mut d_activated_carry: Vec<f32> = vec![0.0f32; d];
    // Bug-3: sync-action recurrence trail (this entry-point reverses only
    // the action sync; gradient enters via activated, not predictions).
    let r_action: Vec<f32> = w.decay_params_action.to_vec()?
        .iter().map(|&p| (-p.clamp(0.0, 15.0)).exp()).collect();
    let mut d_alpha_action_carry: Vec<f32> = vec![0.0f32; cfg.n_synch_action];

    for tick_idx in (0..n_ticks).rev() {
        let tick_cache = &cache.ticks[tick_idx];

        // d_activated for this tick: at the LAST tick, seed from the
        // external `d_activated_post_last_tick`. At earlier ticks,
        // seed from the synapse-input carry of the next-later tick.
        let mut d_activated: Vec<f32> = if tick_idx == last_tick {
            d_act_seed_host.clone()
        } else {
            vec![0.0f32; d]
        };
        for i in 0..d { d_activated[i] += d_activated_carry[i]; }

        // Step 3: NLM backward → d_trace.
        let d_activated_t = Tensor::<D>::from_slice(&d_activated)?;
        let mut d_trace = Tensor::<D>::zeros(d * m)?;
        let (s2_w, s2_b) = match (
            grads.nlm_s2_w.as_mut(),
            grads.nlm_s2_b.as_mut(),
        ) {
            (Some(w), Some(b)) => (Some(w), Some(b)),
            _ => (None, None),
        };
        nlm_backward_typed_helper::<D>(
            &d_activated_t, &tick_cache.nlm,
            &w.nlm_stage1, w.nlm_stage2.as_ref(),
            &mut grads.nlm_s1_w, &mut grads.nlm_s1_b,
            s2_w, s2_b,
            &mut d_trace,
        )?;

        // Step 4: trace shift adjoint → d_pre_act + d_trace_old.
        let d_trace_h: Vec<f32> = d_trace.to_vec()?;
        let mut d_pre_act = vec![0.0f32; d];
        let mut d_trace_old = vec![0.0f32; d * m];
        trace_shift_adjoint_host(&d_trace_h, &mut d_pre_act, &mut d_trace_old, d, m);

        // Step 5: synapse U-Net backward → d_synapse_input
        // (= d concat(attn_out, activated)).
        let d_pre_act_t = Tensor::<D>::from_slice(&d_pre_act)?;
        let mut d_synapse_input = Tensor::<D>::zeros(d_in + d)?;
        w.synapse.backward(
            &d_pre_act_t, &tick_cache.synapse_cache, unet_grads, &mut d_synapse_input,
        )?;
        let d_synapse_input_h: Vec<f32> = d_synapse_input.to_vec()?;
        let d_attn_out = &d_synapse_input_h[..d_in];
        // Save the activated half as carry for the next iteration
        // (= the earlier tick) of this reverse-tick walk.
        d_activated_carry.clear();
        d_activated_carry.extend_from_slice(&d_synapse_input_h[d_in..]);

        // Step 6: MHA backward → d_q_in + d_kv_tokens.
        let d_attn_out_t = Tensor::<D>::from_slice(d_attn_out)?;
        let (d_q_in, d_kv_tokens) = multihead_attention_backward_typed::<D>(
            &d_attn_out_t, &tick_cache.mha, 1, d_in, cfg.heads,
            &w.mha_in_proj, &w.mha_out_proj,
            &mut grads.mha_in_w, &mut grads.mha_in_b,
            &mut grads.mha_out_w, &mut grads.mha_out_b,
        )?;

        let d_kv_t0 = Tensor::<D>::from_slice(&d_kv_tokens[0])?;
        tensor_typed::add_assign(&mut d_kv, &d_kv_t0, d_in)?;

        // Step 7: q_proj backward → d_sync_action. sync_action[T] uses
        // activated_pre_tick[T] = activated_post_tick[T-1]; reverse the
        // sync_action update to add a contribution to the cross-tick carry.
        let d_q_t = Tensor::<D>::from_slice(&d_q_in)?;
        let mut d_sync_action = Tensor::<D>::zeros(cfg.n_synch_action)?;
        grads.backward_q_proj(
            w, &d_q_t,
            &Tensor::<D>::from_slice(&tick_cache.sync_action)?,
            &mut d_sync_action,
        )?;
        let d_sync_action_h: Vec<f32> = d_sync_action.to_vec()?;
        sync_update_reverse_host(
            &d_sync_action_h,
            &tick_cache.activated_pre_tick,
            &tick_cache.alpha_action_post,
            &tick_cache.beta_action_post,
            &r_action,
            &w.sync_action_left, &w.sync_action_right,
            &mut d_activated_carry,
            &mut d_alpha_action_carry,
        );
    }

    // After tick loop: kv_ln backward → d_kv_pre_ln, then kv_proj backward → d_obs.
    let mut d_kv_pre_ln = Tensor::<D>::zeros(d_in)?;
    grads.backward_kv_ln(
        w, &d_kv, &cache.kv_pre_ln, &cache.kv_ln_cache,
        &mut d_kv_pre_ln, d_in,
    )?;
    grads.backward_kv_proj(w, &d_kv_pre_ln, &cache.obs, d_obs)?;

    Ok(())
}

/// Per-neuron NLM forward via typed SuperLinear<D>. v0 does the GLU
/// step on host between stages — typed `glu` exists but the layout
/// for per-neuron paired (value, gate) needs a small reshape.
fn nlm_forward_typed_helper<D: TensorDevice>(
    trace: &Tensor<D>,
    stage1: &tensor_typed::SuperLinear<D>,
    stage2: Option<&tensor_typed::SuperLinear<D>>,
    d_model: usize,
) -> Result<Vec<f32>, BackendError> {
    let s1_t = stage1.forward(trace)?;
    let s1_host: Vec<f32> = s1_t.to_vec()?;
    let s1_glu = per_neuron_glu(&s1_host, d_model, stage1.out_per);

    if let Some(s2) = stage2 {
        let s1_glu_t = Tensor::<D>::from_slice(&s1_glu)?;
        let s2_t = s2.forward(&s1_glu_t)?;
        let s2_host: Vec<f32> = s2_t.to_vec()?;
        Ok(per_neuron_glu(&s2_host, d_model, s2.out_per))
    } else {
        Ok(s1_glu)
    }
}

/// NLM forward + cache. Same as `nlm_forward_typed_helper` but
/// returns the intermediates needed by `nlm_backward_typed_helper`:
/// `s1` (pre-GLU stage1 output), `s1_glu` (post-GLU, = stage2 input),
/// `s2` (pre-GLU stage2 output, only when deep). Caller persists
/// these as part of `CtmCacheTyped<D>` for the backward pass.
pub struct NlmCacheTyped<D: TensorDevice> {
    pub trace: Tensor<D>,           // [d_model × in_per_1]
    pub s1: Tensor<D>,              // [d_model × out_per_1]
    pub s1_glu: Tensor<D>,          // [d_model × out_per_1/2]
    pub s2: Option<Tensor<D>>,      // [d_model × out_per_2] when stage2 present
}

#[allow(dead_code)] // wired by ctm_backward_typed in the next slice commit.
fn nlm_forward_with_cache<D: TensorDevice>(
    trace: &Tensor<D>,
    stage1: &tensor_typed::SuperLinear<D>,
    stage2: Option<&tensor_typed::SuperLinear<D>>,
    d_model: usize,
) -> Result<(Vec<f32>, NlmCacheTyped<D>), BackendError> {
    let s1_t = stage1.forward(trace)?;
    let s1_host: Vec<f32> = s1_t.to_vec()?;
    let s1_glu_host = per_neuron_glu(&s1_host, d_model, stage1.out_per);
    let s1_glu_t = Tensor::<D>::from_slice(&s1_glu_host)?;

    let (final_host, s2_cache) = if let Some(s2) = stage2 {
        let s2_t = s2.forward(&s1_glu_t)?;
        let s2_host: Vec<f32> = s2_t.to_vec()?;
        let final_glu = per_neuron_glu(&s2_host, d_model, s2.out_per);
        (final_glu, Some(s2_t))
    } else {
        (s1_glu_host.clone(), None)
    };

    let cache = NlmCacheTyped {
        trace: Tensor::<D>::from_slice(&trace.to_vec()?)?,
        s1: s1_t,
        s1_glu: s1_glu_t,
        s2: s2_cache,
    };
    Ok((final_host, cache))
}

/// BATCHED NLM forward saving the cache for the batched backward (M1 of the
/// GPU-training engine). `trace` is `[batch × d_model × in_per]`; mirrors
/// `nlm_forward_batched` but keeps the per-stage intermediates (each `[batch ×
/// d_model × out_per]`) so `nlm_backward_batched` can walk reverse. Forward
/// math is identical to `nlm_forward_batched` — verified bit-for-bit in tests.
#[allow(dead_code)] // consumed by nlm_backward_batched / the batched cached forward
fn nlm_forward_with_cache_batched<D: TensorDevice>(
    trace: &Tensor<D>,
    stage1: &tensor_typed::SuperLinear<D>,
    stage2: Option<&tensor_typed::SuperLinear<D>>,
    d_model: usize,
    batch: usize,
) -> Result<(Vec<f32>, NlmCacheTyped<D>), BackendError> {
    let s1_t = stage1.forward_batched(trace, batch)?;
    let s1_host: Vec<f32> = s1_t.to_vec()?;
    let s1_glu_host = per_neuron_glu_batched(&s1_host, d_model, stage1.out_per, batch);
    let s1_glu_t = Tensor::<D>::from_slice(&s1_glu_host)?;

    let (final_host, s2_cache) = if let Some(s2) = stage2 {
        let s2_t = s2.forward_batched(&s1_glu_t, batch)?;
        let s2_host: Vec<f32> = s2_t.to_vec()?;
        let final_glu = per_neuron_glu_batched(&s2_host, d_model, s2.out_per, batch);
        (final_glu, Some(s2_t))
    } else {
        (s1_glu_host.clone(), None)
    };

    let cache = NlmCacheTyped {
        trace: Tensor::<D>::from_slice(&trace.to_vec()?)?,
        s1: s1_t,
        s1_glu: s1_glu_t,
        s2: s2_cache,
    };
    Ok((final_host, cache))
}

/// Reverse the sync_update / sync_init at one tick.
///
/// Forward: `alpha[i] = r[i]·alpha_prev[i] + activated[left[i]]·activated[right[i]]`
///          `beta[i]  = r[i]·beta_prev[i]  + 1`
///          `sync[i]  = alpha[i] / sqrt(beta[i])`
///
/// Backward (Bug-3 fix): the sync accumulator is a linear recurrence
/// `alpha_t = r·alpha_{t-1} + product_t`, so the readout at tick t
/// receives gradient from every later tick through the `r` chain. Define
/// `A_t = dL/d(alpha_t)`. The reverse recurrence is
///   `A_t[i] = d_sync_t[i]/sqrt(beta_t[i]) + r[i]·A_{t+1}[i]`
/// and `A_t` scatters into `activated_t` via the product term. The
/// `r·A_{t+1}` carry (~37%/tick at r=e⁻¹) was previously DROPPED — only
/// the current tick's product term was backprop'd, losing all cross-tick
/// sync gradient. `d_alpha_carry` is the running `A` trail: it holds
/// `A_{t+1}` on entry and is updated to `A_t` on exit (the caller
/// zero-inits it before the reverse-tick loop and reuses it each tick).
/// (`d_beta` stays omitted — `beta_t = r·beta_{t-1} + 1` has no
/// dependence on `activated`, so its path is identically zero.)
///
/// **Output convention:** d_activated ACCUMULATES (multiple sync
/// pairs may touch the same neuron index).
pub fn sync_update_reverse_host(
    d_sync: &[f32],
    activated: &[f32],
    alpha: &[f32],
    beta: &[f32],
    r: &[f32],                    // exponentiated decay per pair, r[i] = exp(-decay[i])
    left: &[usize],
    right: &[usize],
    d_activated: &mut [f32],
    d_alpha_carry: &mut [f32],    // in: A_{t+1}; out: A_t
) {
    let n_pairs = left.len();
    debug_assert_eq!(d_sync.len(), n_pairs);
    debug_assert_eq!(d_alpha_carry.len(), n_pairs);
    for i in 0..n_pairs {
        let beta_i = beta[i].max(1e-8);
        let inv_sqrt_beta = 1.0 / beta_i.sqrt();
        // A_t[i] = d_sync_t/sqrt(beta_t) + r[i]·A_{t+1}[i].
        let a_i = d_sync[i] * inv_sqrt_beta + r[i] * d_alpha_carry[i];
        let l = left[i];
        let ri = right[i];
        // d_product / d_activated[left]  = activated[right]
        // d_product / d_activated[right] = activated[left]
        d_activated[l] += a_i * activated[ri];
        d_activated[ri] += a_i * activated[l];
        d_alpha_carry[i] = a_i; // carry A_t to the previous (earlier) tick
        let _ = alpha; // alpha read is reserved for the higher-order
                       // beta correction — kept in the signature so
                       // callers can plug it in when full correction
                       // is added. Today we use the dominant term.
    }
}

/// Reverse the per-neuron trace shift that `ctm_forward_typed` does:
///   forward:  trace_new[base+i] = trace_old[base+i+1]   for i in 0..m-1
///             trace_new[base+m-1] = pre_act[n]
///
/// backward: given `d_trace_new` (per neuron flat, length d_model*m),
/// produce `d_pre_act` (length d_model) and overwrite `d_trace_old`
/// with the shifted gradient.
///
/// **Output conventions:** d_pre_act OVERWRITTEN; d_trace_old OVERWRITTEN.
pub fn trace_shift_adjoint_host(
    d_trace_new: &[f32],
    d_pre_act: &mut [f32],
    d_trace_old: &mut [f32],
    d_model: usize,
    memory_length: usize,
) {
    debug_assert_eq!(d_trace_new.len(), d_model * memory_length);
    debug_assert_eq!(d_pre_act.len(), d_model);
    debug_assert_eq!(d_trace_old.len(), d_model * memory_length);
    for n in 0..d_model {
        let base = n * memory_length;
        // d_pre_act[n] picks up the gradient on the newly-inserted slot.
        d_pre_act[n] = d_trace_new[base + memory_length - 1];
        // Other slots: gradient on trace_new[base+i] flows to trace_old[base+i+1].
        // Equivalently: d_trace_old[base + i + 1] = d_trace_new[base + i] for i in 0..m-1.
        // Slot trace_old[base + 0] receives no gradient through the shift
        // (it was discarded by the forward; conceptually the oldest entry
        // falls out of the window).
        d_trace_old[base] = 0.0;
        for i in 0..memory_length - 1 {
            d_trace_old[base + i + 1] = d_trace_new[base + i];
        }
    }
}

/// NLM backward: given upstream `d_activated` (gradient w.r.t. NLM
/// output, length = `d_model × final_per_half`), compose backwards
/// through optional stage2-GLU and stage1-GLU, accumulating into
/// `d_W`/`d_b` for each stage and overwriting `d_trace`.
///
/// **Output convention:** d_W and d_b ACCUMULATE; d_trace is OVERWRITTEN.
pub fn nlm_backward_typed_helper<D: TensorDevice>(
    d_activated: &Tensor<D>,    // [d_model × out_per_2/2] (deep) or [d_model × out_per_1/2]
    cache: &NlmCacheTyped<D>,
    stage1: &tensor_typed::SuperLinear<D>,
    stage2: Option<&tensor_typed::SuperLinear<D>>,
    d_stage1_w: &mut Tensor<D>,
    d_stage1_b: &mut Tensor<D>,
    d_stage2_w: Option<&mut Tensor<D>>,
    d_stage2_b: Option<&mut Tensor<D>>,
    d_trace: &mut Tensor<D>,
) -> Result<(), BackendError> {
    let n_neurons = stage1.n_neurons;

    // d_s1_glu — gradient flowing into the stage1 GLU output.
    // For deep NLM: comes from stage2 backward.
    // For shallow: equals d_activated directly.
    let d_s1_glu = if let Some(s2) = stage2 {
        let s2_cache = cache.s2.as_ref()
            .expect("deep NLM cache requires s2");
        // Step 1: d_s2 = glu_bwd(d_activated, s2) — overwrite into a fresh buffer.
        let mut d_s2 = Tensor::<D>::zeros(n_neurons * s2.out_per)?;
        tensor_typed::glu_bwd(d_activated, s2_cache, &mut d_s2,
            n_neurons, s2.out_per / 2)?;
        // Step 2: super_linear backward — d_W2 += d_s2 ⊗ s1_glu, d_s1_glu = W2^T·d_s2.
        let mut d_s1_glu_buf = Tensor::<D>::zeros(n_neurons * s2.in_per)?;
        if let (Some(dw), Some(db)) = (d_stage2_w, d_stage2_b) {
            tensor_typed::super_linear_bwd_dw(&d_s2, &cache.s1_glu, dw,
                n_neurons, s2.in_per, s2.out_per)?;
            // d_b += d_s2 (per-neuron passthrough — same shape as biases).
            tensor_typed::add_assign(db, &d_s2, n_neurons * s2.out_per)?;
        }
        tensor_typed::super_linear_bwd_dx(&d_s2, &s2.weights, &mut d_s1_glu_buf,
            n_neurons, s2.in_per, s2.out_per)?;
        d_s1_glu_buf
    } else {
        // Shallow: d_activated IS the gradient w.r.t. s1_glu. Clone it
        // (we need a Tensor<D> to feed into the next glu_bwd call).
        let host: Vec<f32> = d_activated.to_vec()?;
        Tensor::<D>::from_slice(&host)?
    };

    // Step 3: d_s1 = glu_bwd(d_s1_glu, s1)
    let mut d_s1 = Tensor::<D>::zeros(n_neurons * stage1.out_per)?;
    tensor_typed::glu_bwd(&d_s1_glu, &cache.s1, &mut d_s1,
        n_neurons, stage1.out_per / 2)?;

    // Step 4: super_linear stage1 backward — d_W1 += d_s1 ⊗ trace, d_trace = W1^T·d_s1.
    tensor_typed::super_linear_bwd_dw(&d_s1, &cache.trace, d_stage1_w,
        n_neurons, stage1.in_per, stage1.out_per)?;
    tensor_typed::add_assign(d_stage1_b, &d_s1, n_neurons * stage1.out_per)?;
    tensor_typed::super_linear_bwd_dx(&d_s1, &stage1.weights, d_trace,
        n_neurons, stage1.in_per, stage1.out_per)?;

    Ok(())
}

/// MHA forward + cache. Same algorithm as `multihead_attention_typed_helper`
/// but returns the intermediates the matched backward needs:
/// `q_full` (= in_proj(q_in)[..d_input]), `k_all`, `v_all` (per-token K/V),
/// per-head softmax `weights`, the concat_heads input to out_proj, and
/// the q_in / kv_tokens used for in_proj backward.
pub struct MhaCacheTyped<D: TensorDevice> {
    pub q_full: Vec<f32>,
    pub k_all: Vec<f32>,
    pub v_all: Vec<f32>,
    pub attn_weights: Vec<Vec<f32>>,    // [n_heads][n_tokens]
    pub concat_heads: Tensor<D>,         // out_proj input — needed for Linear::backward
    pub q_in: Vec<f32>,
    pub kv_tokens: Vec<Vec<f32>>,
}

#[allow(dead_code)] // wired by ctm_backward_typed in next slice commit.
fn multihead_attention_with_cache_typed<D: TensorDevice>(
    q_in: &[f32],
    kv_flat: &[f32],
    n_tokens: usize,
    d_input: usize,
    n_heads: usize,
    in_proj: &tensor_typed::Linear<D>,
    out_proj: &tensor_typed::Linear<D>,
) -> Result<(Vec<f32>, MhaCacheTyped<D>), BackendError> {
    let d_head = d_input / n_heads;
    let scale = 1.0 / (d_head as f32).sqrt();

    // Q projection.
    let q_in_t = Tensor::<D>::from_slice(q_in)?;
    let q_full_t = in_proj.forward(&q_in_t)?;
    let q_full_h: Vec<f32> = q_full_t.to_vec()?;
    let q_full = q_full_h[..d_input].to_vec();

    // KV projection — store per-token K + V slices.
    let mut k_all = Vec::with_capacity(n_tokens * d_input);
    let mut v_all = Vec::with_capacity(n_tokens * d_input);
    let mut kv_tokens = Vec::with_capacity(n_tokens);
    for t in 0..n_tokens {
        let tok = &kv_flat[t * d_input..(t + 1) * d_input];
        kv_tokens.push(tok.to_vec());
        let tok_t = Tensor::<D>::from_slice(tok)?;
        let proj_t = in_proj.forward(&tok_t)?;
        let proj_h: Vec<f32> = proj_t.to_vec()?;
        k_all.extend_from_slice(&proj_h[d_input..2 * d_input]);
        v_all.extend_from_slice(&proj_h[2 * d_input..3 * d_input]);
    }

    // Per-head softmax + weighted sum (host).
    let mut all_weights = Vec::with_capacity(n_heads);
    let mut concat_heads_h = Vec::with_capacity(d_input);
    for h in 0..n_heads {
        let q_h = &q_full[h * d_head..(h + 1) * d_head];
        let mut scores = Vec::with_capacity(n_tokens);
        for t in 0..n_tokens {
            let k_h = &k_all[t * d_input + h * d_head..t * d_input + (h + 1) * d_head];
            scores.push(q_h.iter().zip(k_h).map(|(&a, &b)| a * b).sum::<f32>() * scale);
        }
        let max_s = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_s: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp()).collect();
        let sum_s: f32 = exp_s.iter().sum();
        let weights: Vec<f32> = exp_s.iter().map(|&e| e / sum_s).collect();
        let mut head_out = vec![0.0f32; d_head];
        for t in 0..n_tokens {
            let v_h = &v_all[t * d_input + h * d_head..t * d_input + (h + 1) * d_head];
            for j in 0..d_head { head_out[j] += weights[t] * v_h[j]; }
        }
        all_weights.push(weights);
        concat_heads_h.extend_from_slice(&head_out);
    }

    // out_proj.
    let concat_t = Tensor::<D>::from_slice(&concat_heads_h)?;
    let out_t = out_proj.forward(&concat_t)?;
    let out_h: Vec<f32> = out_t.to_vec()?;

    let cache = MhaCacheTyped {
        q_full, k_all, v_all,
        attn_weights: all_weights,
        concat_heads: Tensor::<D>::from_slice(&concat_heads_h)?,
        q_in: q_in.to_vec(),
        kv_tokens,
    };
    Ok((out_h, cache))
}

/// MHA backward — composes Linear<D>::backward (out_proj, in_proj × per-token) with
/// per-head host softmax_bwd + Q·K^T adjoint + value-weighted-sum adjoint. Returns
/// (d_q_in, d_kv_tokens).
///
/// in_proj backward: we don't have a row-sliced backward, so we build masked
/// upstream gradients (length 3*d_input) and call Linear::backward once per
/// input (q_in, then per-token kv). All calls accumulate into the same d_W/d_b.
#[allow(dead_code)] // wired by ctm_backward_typed in next slice commit.
fn multihead_attention_backward_typed<D: TensorDevice>(
    d_out: &Tensor<D>,
    cache: &MhaCacheTyped<D>,
    n_tokens: usize,
    d_input: usize,
    n_heads: usize,
    in_proj: &tensor_typed::Linear<D>,
    out_proj: &tensor_typed::Linear<D>,
    d_in_w: &mut Tensor<D>, d_in_b: &mut Tensor<D>,
    d_out_w: &mut Tensor<D>, d_out_b: &mut Tensor<D>,
) -> Result<(Vec<f32>, Vec<Vec<f32>>), BackendError> {
    let d_head = d_input / n_heads;
    let scale = 1.0 / (d_head as f32).sqrt();

    // 1. out_proj backward → d_concat (length d_input).
    let mut d_concat = Tensor::<D>::zeros(d_input)?;
    out_proj.backward(d_out, &cache.concat_heads, d_out_w, d_out_b, &mut d_concat)?;
    let d_concat_h: Vec<f32> = d_concat.to_vec()?;

    // 2. Per-head softmax+sum reverse (host).
    let mut d_q_full = vec![0.0f32; d_input];
    let mut d_k_all = vec![0.0f32; n_tokens * d_input];
    let mut d_v_all = vec![0.0f32; n_tokens * d_input];
    for h in 0..n_heads {
        let d_head_out = &d_concat_h[h * d_head..(h + 1) * d_head];
        let weights = &cache.attn_weights[h];
        let mut d_weights = vec![0.0f32; n_tokens];
        for t in 0..n_tokens {
            let v_h = &cache.v_all[t * d_input + h * d_head..t * d_input + (h + 1) * d_head];
            for j in 0..d_head {
                d_weights[t] += d_head_out[j] * v_h[j];
                d_v_all[t * d_input + h * d_head + j] += weights[t] * d_head_out[j];
            }
        }
        // softmax_bwd in closed form (Jacobian-vector product).
        let dot_wd: f32 = weights.iter().zip(&d_weights).map(|(&w, &d)| w * d).sum();
        let d_scores: Vec<f32> = (0..n_tokens)
            .map(|t| weights[t] * (d_weights[t] - dot_wd) * scale)
            .collect();
        // Reverse Q·K^T.
        let q_h = &cache.q_full[h * d_head..(h + 1) * d_head];
        for t in 0..n_tokens {
            let k_start = t * d_input + h * d_head;
            for j in 0..d_head {
                d_q_full[h * d_head + j] += d_scores[t] * cache.k_all[k_start + j];
                d_k_all[k_start + j] += d_scores[t] * q_h[j];
            }
        }
    }

    // 3. in_proj backward for Q from q_in.
    //    Mask: upstream = [d_q_full, 0, 0] (length 3*d_input).
    let mut q_upstream = vec![0.0f32; 3 * d_input];
    q_upstream[..d_input].copy_from_slice(&d_q_full);
    let q_upstream_t = Tensor::<D>::from_slice(&q_upstream)?;
    let q_in_t = Tensor::<D>::from_slice(&cache.q_in)?;
    let mut d_q_in_t = Tensor::<D>::zeros(d_input)?;
    in_proj.backward(&q_upstream_t, &q_in_t, d_in_w, d_in_b, &mut d_q_in_t)?;
    let d_q_in: Vec<f32> = d_q_in_t.to_vec()?;

    // 4. in_proj backward for K + V from each kv_token.
    //    Mask: upstream = [0, d_k_t, d_v_t].
    let mut d_kv_tokens: Vec<Vec<f32>> = Vec::with_capacity(n_tokens);
    for t in 0..n_tokens {
        let mut kv_upstream = vec![0.0f32; 3 * d_input];
        kv_upstream[d_input..2 * d_input].copy_from_slice(
            &d_k_all[t * d_input..(t + 1) * d_input],
        );
        kv_upstream[2 * d_input..3 * d_input].copy_from_slice(
            &d_v_all[t * d_input..(t + 1) * d_input],
        );
        let upstream_t = Tensor::<D>::from_slice(&kv_upstream)?;
        let tok_t = Tensor::<D>::from_slice(&cache.kv_tokens[t])?;
        let mut d_tok = Tensor::<D>::zeros(d_input)?;
        in_proj.backward(&upstream_t, &tok_t, d_in_w, d_in_b, &mut d_tok)?;
        d_kv_tokens.push(d_tok.to_vec()?);
    }

    Ok((d_q_in, d_kv_tokens))
}

/// Multi-head attention via typed `Linear<D>` for the two big matmuls;
/// per-head dot-products + softmax + value-weighted-sum are host
/// (small, intricate to typify). Mirrors `multihead_attention` exactly.
fn multihead_attention_typed_helper<D: TensorDevice>(
    q_in: &[f32],            // [d_input]
    kv_flat: &[f32],         // [n_tokens × d_input]
    n_tokens: usize,
    d_input: usize,
    n_heads: usize,
    in_proj: &tensor_typed::Linear<D>,    // d_input → 3×d_input
    out_proj: &tensor_typed::Linear<D>,   // d_input → d_input
) -> Result<Vec<f32>, BackendError> {
    let d_head = d_input / n_heads;
    let scale = 1.0 / (d_head as f32).sqrt();

    // Project query through full in_proj (we'll slice the [Q | K | V]
    // result on host since Q only uses the first d_input of 3*d_input).
    let q_in_t = Tensor::<D>::from_slice(q_in)?;
    let q_full_t = in_proj.forward(&q_in_t)?;   // [3*d_input]
    let q_full_host: Vec<f32> = q_full_t.to_vec()?;
    let q_full = &q_full_host[..d_input];

    // Project all KV tokens — for each token, we need K (slice
    // d_input..2*d_input) and V (slice 2*d_input..3*d_input).
    let mut k_all = Vec::with_capacity(n_tokens * d_input);
    let mut v_all = Vec::with_capacity(n_tokens * d_input);
    for t in 0..n_tokens {
        let tok = &kv_flat[t * d_input..(t + 1) * d_input];
        let tok_t = Tensor::<D>::from_slice(tok)?;
        let proj_t = in_proj.forward(&tok_t)?;     // [3*d_input]
        let proj_host: Vec<f32> = proj_t.to_vec()?;
        k_all.extend_from_slice(&proj_host[d_input..2 * d_input]);
        v_all.extend_from_slice(&proj_host[2 * d_input..3 * d_input]);
    }

    // Per-head scaled dot-product attention (host arithmetic).
    let mut concat_heads = Vec::with_capacity(d_input);
    for h in 0..n_heads {
        let q_h = &q_full[h * d_head..(h + 1) * d_head];

        let mut scores = Vec::with_capacity(n_tokens);
        for t in 0..n_tokens {
            let k_h = &k_all[t * d_input + h * d_head..t * d_input + (h + 1) * d_head];
            let dot: f32 = q_h.iter().zip(k_h).map(|(&a, &b)| a * b).sum();
            scores.push(dot * scale);
        }

        let max_s = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_s: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp()).collect();
        let sum_s: f32 = exp_s.iter().sum();

        let mut head_out = vec![0.0f32; d_head];
        for t in 0..n_tokens {
            let w = exp_s[t] / sum_s;
            let v_h = &v_all[t * d_input + h * d_head..t * d_input + (h + 1) * d_head];
            for j in 0..d_head {
                head_out[j] += w * v_h[j];
            }
        }
        concat_heads.extend_from_slice(&head_out);
    }

    let concat_t = Tensor::<D>::from_slice(&concat_heads)?;
    let out_t = out_proj.forward(&concat_t)?;
    out_t.to_vec()
}

#[cfg(test)]
mod typed_forward_tests {
    use super::*;
    use modgrad_device::backend::tensor::Cpu;
    #[cfg(feature = "rocm")]
    use modgrad_device::backend::tensor::Rocm;
    use crate::config::CtmConfig;
    use crate::weights::{CtmWeights, CtmState};

    fn small_cfg() -> CtmConfig {
        CtmConfig {
            iterations: 2,
            d_model: 4,
            d_input: 8,
            heads: 2,
            n_synch_out: 4,
            n_synch_action: 4,
            synapse_depth: 2,
            memory_length: 4,
            deep_nlms: false,
            memory_hidden_dims: 0,
            out_dims: 3,
            n_random_pairing_self: 0,
            min_width: 2,
            exit_strategy: crate::config::ExitStrategy::None,
            collect_trajectories: false,
            spatial: None,
        }
    }

    /// Run untyped ctm_forward on Raw single-token input and the
    /// typed Cpu version on the same weights — assert predictions
    /// agree to 1e-4. The parity proof: typed cascade reproduces
    /// the brain forward semantics exactly.
    #[test]
    fn cpu_typed_ctm_forward_matches_untyped_single_token() {
        let cfg = small_cfg();
        let raw_input_dim = 6;
        let w = CtmWeights::new(cfg.clone(), raw_input_dim);
        let typed = CtmWeightsTyped::<Cpu>::from_untyped(&w)
            .expect("from_untyped");

        let obs: Vec<f32> = (0..raw_input_dim).map(|i| (i as f32 - 2.5) * 0.3).collect();

        // Untyped reference.
        let mut state = CtmState::new(&w);
        let ref_out = ctm_forward(&w, &mut state, CtmInput::Raw {
            obs: &obs, n_tokens: 1, raw_dim: raw_input_dim,
        });

        // Typed.
        let mut activated = w.start_activated.clone();
        let mut trace = w.start_trace.clone();
        let typed_out = ctm_forward_typed::<Cpu>(
            &typed, &mut activated, &mut trace, &obs,
        ).expect("typed forward");

        assert_eq!(ref_out.predictions.len(), typed_out.predictions.len());
        assert_eq!(ref_out.ticks_used, typed_out.ticks_used);
        for (tick, (a, b)) in ref_out.predictions.iter()
            .zip(&typed_out.predictions).enumerate()
        {
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                assert!((x - y).abs() < 1e-3,
                    "tick {} pred[{}] disagree: untyped={} typed={}",
                    tick, i, x, y);
            }
        }
    }

    /// **The closing capability test.** Run ctm_forward_typed_with_cache,
    /// then ctm_backward_typed with hand-cooked d_predictions, verify
    /// every CtmGradientsTyped buffer has non-zero entries (gradient
    /// flowed through every weight). Confirms the full backward chain
    /// composes correctly.
    /// The wiring capstone: the batched forward tick loop must produce the
    /// same per-tick predictions as `batch` independent scalar forwards.
    /// Validates the whole batched composition end-to-end.
    #[test]
    fn cpu_ctm_forward_typed_batched_matches_scalar() {
        let cfg = small_cfg();
        let raw = 6usize;
        let batch = 3usize;
        let w = CtmWeights::new(cfg.clone(), raw);
        let typed = CtmWeightsTyped::<Cpu>::from_untyped(&w).unwrap();
        let obs: Vec<f32> = (0..batch * raw).map(|i| ((i * 3 % 7) as f32 - 3.0) * 0.2).collect();

        let mut act_b: Vec<f32> = (0..batch).flat_map(|_| w.start_activated.clone()).collect();
        let mut tr_b: Vec<f32> = (0..batch).flat_map(|_| w.start_trace.clone()).collect();
        // exit_strategy is None in small_cfg ⇒ all ticks run, no masking.
        let (preds_b, ticks_used) = ctm_forward_typed_batched::<Cpu>(&typed, &mut act_b, &mut tr_b, &obs, batch).unwrap();
        assert!(ticks_used.iter().all(|&t| t == cfg.iterations));

        for b in 0..batch {
            let mut a = w.start_activated.clone();
            let mut t = w.start_trace.clone();
            let (out, _) = ctm_forward_typed_with_cache::<Cpu>(
                &typed, &mut a, &mut t, &obs[b * raw..(b + 1) * raw],
            ).unwrap();
            for tick in 0..cfg.iterations {
                let pb = &preds_b[tick][b * cfg.out_dims..(b + 1) * cfg.out_dims];
                for j in 0..cfg.out_dims {
                    let exp = out.predictions[tick][j];
                    assert!((pb[j] - exp).abs() < 1e-4,
                        "pred[tick{tick}][b{b}][{j}]: {} vs {}", pb[j], exp);
                }
            }
        }
    }

    /// M1: the batched NLM forward-with-cache matches B per-sample calls
    /// bit-for-bit (the cache/backward groundwork for GPU training).
    #[test]
    fn cpu_nlm_forward_with_cache_batched_matches_scalar() {
        let cfg = small_cfg(); // shallow NLM (deep_nlms=false → stage2=None)
        let raw = 6usize;
        let batch = 4usize;
        let w = CtmWeights::new(cfg.clone(), raw);
        let typed = CtmWeightsTyped::<Cpu>::from_untyped(&w).unwrap();
        let (d, m) = (cfg.d_model, cfg.memory_length);

        let trace_b: Vec<f32> = (0..batch * d * m)
            .map(|i| (((i * 5 % 11) as f32) - 5.0) * 0.1)
            .collect();
        let trace_b_t = Tensor::<Cpu>::from_slice(&trace_b).unwrap();
        let (out_b, _cache) = nlm_forward_with_cache_batched::<Cpu>(
            &trace_b_t, &typed.nlm_stage1, typed.nlm_stage2.as_ref(), d, batch,
        ).unwrap();

        for b in 0..batch {
            let tr = &trace_b[b * d * m..(b + 1) * d * m];
            let tr_t = Tensor::<Cpu>::from_slice(tr).unwrap();
            let (out_s, _) = nlm_forward_with_cache::<Cpu>(
                &tr_t, &typed.nlm_stage1, typed.nlm_stage2.as_ref(), d,
            ).unwrap();
            let per = out_s.len();
            for j in 0..per {
                assert!((out_b[b * per + j] - out_s[j]).abs() < 1e-5,
                    "nlm[b{b}][{j}]: batched {} vs scalar {}", out_b[b * per + j], out_s[j]);
            }
        }
    }

    /// Adaptive-exit halt mask: with a Certainty exit, each example's output
    /// freezes at its halt tick (later ticks == halt-tick prediction) and
    /// ticks_used records where it stopped.
    #[test]
    fn cpu_ctm_forward_batched_halt_mask() {
        let mut cfg = small_cfg();
        cfg.iterations = 4;
        // Low threshold so at least some examples halt before the last tick.
        cfg.exit_strategy = crate::config::ExitStrategy::Certainty { threshold: 0.0 };
        let raw = 6usize;
        let batch = 3usize;
        let w = CtmWeights::new(cfg.clone(), raw);
        let typed = CtmWeightsTyped::<Cpu>::from_untyped(&w).unwrap();
        let obs: Vec<f32> = (0..batch * raw).map(|i| ((i * 5 % 7) as f32 - 3.0) * 0.2).collect();
        let mut act_b: Vec<f32> = (0..batch).flat_map(|_| w.start_activated.clone()).collect();
        let mut tr_b: Vec<f32> = (0..batch).flat_map(|_| w.start_trace.clone()).collect();
        let (preds, ticks_used) = ctm_forward_typed_batched::<Cpu>(&typed, &mut act_b, &mut tr_b, &obs, batch).unwrap();
        let od = cfg.out_dims;
        for b in 0..batch {
            let tu = ticks_used[b];
            assert!(tu >= 1 && tu <= cfg.iterations);
            // Post-halt ticks must equal the halt-tick prediction (frozen).
            if tu < cfg.iterations {
                let halt_pred = preds[tu - 1][b * od..b * od + od].to_vec();
                for tick in tu..cfg.iterations {
                    let later = &preds[tick][b * od..b * od + od];
                    for j in 0..od {
                        assert!((later[j] - halt_pred[j]).abs() < 1e-6,
                            "frozen pred mismatch b{b} tick{tick}");
                    }
                }
            }
        }
    }

    #[test]
    fn cpu_ctm_backward_typed_full_chain_flows_gradients() {
        use crate::weights::{CtmWeights, CtmGradientsTyped};
        use crate::synapse::UNetGradsTyped;

        let cfg = small_cfg();
        let raw_input_dim = 6;
        let w = CtmWeights::new(cfg.clone(), raw_input_dim);
        let typed = CtmWeightsTyped::<Cpu>::from_untyped(&w).unwrap();

        let obs: Vec<f32> = (0..raw_input_dim).map(|i| (i as f32 - 2.5) * 0.3).collect();
        let mut activated = w.start_activated.clone();
        let mut trace = w.start_trace.clone();

        let (out, cache) = ctm_forward_typed_with_cache::<Cpu>(
            &typed, &mut activated, &mut trace, &obs,
        ).expect("forward");

        // Hand-cooked d_predictions: ones at every tick.
        let d_preds: Vec<Vec<f32>> = out.predictions.iter()
            .map(|p| vec![1.0; p.len()])
            .collect();

        let mut grads = CtmGradientsTyped::<Cpu>::zeros(&typed).unwrap();
        let mut unet_grads = UNetGradsTyped::<Cpu>::zeros(&typed.synapse).unwrap();
        let mut d_obs = Tensor::<Cpu>::zeros(raw_input_dim).unwrap();
        ctm_backward_typed::<Cpu>(
            &typed, &cache, &d_preds,
            &mut grads, &mut unet_grads, &mut d_obs,
        ).expect("backward");

        // Every primary weight gradient must be non-zero (the chain
        // succeeded in propagating gradient to that weight).
        // Note: q_proj_w is intentionally NOT probed — with n_tokens=1
        // (single-token KV) the softmax output is trivially [1.0]
        // and ∂attn/∂Q = 0, so q_proj sees zero gradient. Multi-token
        // tests will exercise the Q path.
        let probes = [
            ("out_proj_w", &grads.out_proj_w),
            ("kv_proj_w", &grads.kv_proj_w),
            ("mha_in_w", &grads.mha_in_w),
            ("mha_out_w", &grads.mha_out_w),
            ("nlm_s1_w", &grads.nlm_s1_w),
        ];
        for (name, t) in probes {
            let v = t.to_vec().unwrap();
            assert!(v.iter().any(|&x| x.abs() > 1e-6),
                "{} all zero — backward did not flow into this weight", name);
        }
        // Synapse first projection grad should also be non-zero.
        let dw_first = unet_grads.first.d_w.to_vec().unwrap();
        assert!(dw_first.iter().any(|&x| x.abs() > 1e-6),
            "unet_grads.first.d_w all zero");
        // d_obs should be non-zero (gradient flowed all the way back).
        let do_h = d_obs.to_vec().unwrap();
        assert!(do_h.iter().any(|&x| x.abs() > 1e-6),
            "d_obs all zero — kv_proj backward did not flow");
    }

    /// Finite-difference gradcheck for the typed CTM backward.
    ///
    /// The existing `cpu_ctm_backward_typed_full_chain_flows_gradients`
    /// only checks gradients are *non-zero* (a smoke test). This is the
    /// numerical check the crate was missing: with `d_preds = ones` at
    /// every tick the loss is the smooth scalar `L = Σ_{tick,i}
    /// predictions[tick][i]`, so `∂L/∂w` from `ctm_backward_typed` must
    /// match central finite differences of the forward.
    ///
    /// Limitation (same shape as the BLT lesson): the typed forward is
    /// single-token, so the softmax is trivially `[1.0]` and the
    /// attention-score / Q path carries ~zero gradient — this gradcheck
    /// validates synapse, NLM, sync, output_proj, and the K/V + kv_proj
    /// projections, but NOT the multi-token attention score path. A
    /// multi-token typed forward is needed to close that gap.
    ///
    /// STATUS 2026-06-10: PASSES. Found and fixed four bugs in the typed
    /// backward: (1) sync_out reverse used the pre-tick activated instead
    /// of post-tick; (2) CPU `glu_bwd` dispatched the flat GLU instead of
    /// the per-neuron variant, scrambling NLM grads across neurons; (3)
    /// the cross-tick trace gradient `d_trace_old` was discarded; (4) the
    /// sync accumulator recurrence (`r·alpha_prev`) was dropped in
    /// `sync_update_reverse_host`. With all four fixed, every probed weight
    /// matches finite differences within tolerance (small attention grads
    /// sit at the FD floor — see ABS_FLOOR).
    #[test]
    fn cpu_ctm_backward_typed_matches_finite_difference() {
        use crate::weights::{CtmWeights, CtmGradientsTyped};
        use crate::synapse::UNetGradsTyped;

        let cfg = small_cfg();
        let raw_input_dim = 6;
        let w = CtmWeights::new(cfg.clone(), raw_input_dim);
        let obs: Vec<f32> = (0..raw_input_dim).map(|i| (i as f32 - 2.5) * 0.3).collect();

        // L = sum of all per-tick predictions  ⇒  d_preds = ones.
        let loss_of = |w: &CtmWeights| -> f32 {
            let typed = CtmWeightsTyped::<Cpu>::from_untyped(w).unwrap();
            let mut activated = w.start_activated.clone();
            let mut trace = w.start_trace.clone();
            let (out, _) = ctm_forward_typed_with_cache::<Cpu>(
                &typed, &mut activated, &mut trace, &obs,
            ).expect("forward");
            out.predictions.iter().flat_map(|p| p.iter()).sum::<f32>()
        };

        // Analytic grads at the base point.
        let typed = CtmWeightsTyped::<Cpu>::from_untyped(&w).unwrap();
        let mut activated = w.start_activated.clone();
        let mut trace = w.start_trace.clone();
        let (out, cache) = ctm_forward_typed_with_cache::<Cpu>(
            &typed, &mut activated, &mut trace, &obs,
        ).expect("forward");
        let d_preds: Vec<Vec<f32>> =
            out.predictions.iter().map(|p| vec![1.0; p.len()]).collect();
        let mut grads = CtmGradientsTyped::<Cpu>::zeros(&typed).unwrap();
        let mut unet_grads = UNetGradsTyped::<Cpu>::zeros(&typed.synapse).unwrap();
        let mut d_obs = Tensor::<Cpu>::zeros(raw_input_dim).unwrap();
        ctm_backward_typed::<Cpu>(
            &typed, &cache, &d_preds, &mut grads, &mut unet_grads, &mut d_obs,
        ).expect("backward");

        const EPS: f32 = 1e-3;
        const TOL: f32 = 2e-2;
        // Central-difference floor: at EPS=1e-3 with loss ~O(1) and f32 ULP
        // ~1.2e-7 the absolute FD noise is ~|L|·ulp/(2·EPS) ≈ 1e-4. Small-
        // magnitude grads (the ~1e-3 attention weights) hit that floor and
        // show 3-5% relative error that is pure noise, not a backward bug.
        // A key passes if EITHER the relative error is within TOL OR the
        // absolute difference is below the FD floor. (Real backward bugs
        // were 30-190% on grads ≥1e-2 — abs diffs far above this floor.)
        const ABS_FLOOR: f32 = 3e-4;

        // (name, analytic grad host vec, mutable access to the weight Vec)
        // We FD a few interior indices per weight group. q_proj is skipped
        // (single-token ⇒ zero grad, documented above).
        let probes: &[(&str, Vec<f32>, fn(&mut CtmWeights) -> &mut Vec<f32>)] = &[
            ("mha_in_w",   grads.mha_in_w.to_vec().unwrap(),   |w| &mut w.mha_in_proj.weight),
            ("mha_out_w",  grads.mha_out_w.to_vec().unwrap(),  |w| &mut w.mha_out_proj.weight),
            ("nlm_s1_w",   grads.nlm_s1_w.to_vec().unwrap(),   |w| &mut w.nlm_stage1.weights),
            ("out_proj_w", grads.out_proj_w.to_vec().unwrap(), |w| &mut w.output_proj.weight),
            ("kv_proj_w",  grads.kv_proj_w.to_vec().unwrap(),  |w| &mut w.kv_proj.weight),
        ];

        let mut failures: Vec<String> = Vec::new();
        for (name, analytic, accessor) in probes {
            let n = analytic.len();
            // Sample up to 4 indices spread across the buffer.
            let idxs: Vec<usize> = (0..4).map(|j| (j * n / 4).min(n - 1)).collect();
            for &idx in &idxs {
                let mut wp = w.clone();
                let orig = accessor(&mut wp)[idx];
                accessor(&mut wp)[idx] = orig + EPS;
                let lp = loss_of(&wp);
                accessor(&mut wp)[idx] = orig - EPS;
                let lm = loss_of(&wp);
                let fd = (lp - lm) / (2.0 * EPS);
                let a = analytic[idx];
                let abs_diff = (fd - a).abs();
                let denom = fd.abs().max(a.abs()).max(1e-4);
                let rel = abs_diff / denom;
                eprintln!(
                    "ctm-gradcheck: {name}[{idx}] analytic={a:+.6e} fd={fd:+.6e} rel={rel:.3e} abs={abs_diff:.3e}"
                );
                if rel > TOL && abs_diff > ABS_FLOOR {
                    failures.push(format!(
                        "{name}[{idx}]: analytic={a:+e} fd={fd:+e} rel={rel:.3e} abs={abs_diff:.3e}"
                    ));
                }
            }
        }
        assert!(
            failures.is_empty(),
            "CTM typed backward disagrees with finite differences:\n  {}",
            failures.join("\n  "),
        );
    }

    #[test]
    fn cpu_mha_backward_smoke() {
        // Build mha_in_proj (d_input → 3*d_input) and mha_out_proj (d_input → d_input).
        // d_input=4, n_heads=2, d_head=2. n_tokens=2.
        let d_input = 4;
        let n_heads = 2;
        let n_tokens = 2;

        // Random-ish weights using deterministic init.
        let in_w: Vec<f32> = (0..3 * d_input * d_input).map(|i| (i as f32 - 24.0) * 0.05).collect();
        let in_b = vec![0.0f32; 3 * d_input];
        let in_proj = tensor_typed::Linear::<Cpu>::from_host(
            &in_w, &in_b, d_input, 3 * d_input,
        ).unwrap();
        let out_w: Vec<f32> = (0..d_input * d_input).map(|i| (i as f32 - 8.0) * 0.1).collect();
        let out_b = vec![0.0f32; d_input];
        let out_proj = tensor_typed::Linear::<Cpu>::from_host(
            &out_w, &out_b, d_input, d_input,
        ).unwrap();

        let q_in: Vec<f32> = (0..d_input).map(|i| (i as f32 + 1.0) * 0.2).collect();
        let kv: Vec<f32> = (0..n_tokens * d_input).map(|i| (i as f32 - 4.0) * 0.15).collect();

        let (out, cache) = multihead_attention_with_cache_typed::<Cpu>(
            &q_in, &kv, n_tokens, d_input, n_heads, &in_proj, &out_proj,
        ).unwrap();
        assert_eq!(out.len(), d_input);

        // Run backward with a hand-cooked d_out.
        let d_out_h = vec![1.0, 0.5, -0.5, -1.0];
        let d_out_t = Tensor::<Cpu>::from_slice(&d_out_h).unwrap();
        let mut d_in_w = Tensor::<Cpu>::zeros(in_w.len()).unwrap();
        let mut d_in_b = Tensor::<Cpu>::zeros(in_b.len()).unwrap();
        let mut d_out_w_t = Tensor::<Cpu>::zeros(out_w.len()).unwrap();
        let mut d_out_b_t = Tensor::<Cpu>::zeros(out_b.len()).unwrap();
        let (d_q_in, d_kv_tokens) = multihead_attention_backward_typed::<Cpu>(
            &d_out_t, &cache, n_tokens, d_input, n_heads,
            &in_proj, &out_proj,
            &mut d_in_w, &mut d_in_b, &mut d_out_w_t, &mut d_out_b_t,
        ).unwrap();

        // Sanity: d_q_in should be non-zero (gradient flowed through Q path).
        assert!(d_q_in.iter().any(|&v| v.abs() > 1e-6),
            "d_q_in all zero — Q backward did not flow");
        // d_kv_tokens should have an entry per token, each non-zero overall.
        assert_eq!(d_kv_tokens.len(), n_tokens);
        for (t, dt) in d_kv_tokens.iter().enumerate() {
            assert!(dt.iter().any(|&v| v.abs() > 1e-6),
                "d_kv_tokens[{}] all zero — K/V backward did not flow", t);
        }
        // in_proj.d_W must have non-zero entries (accumulated from Q + K + V calls).
        let dw = d_in_w.to_vec().unwrap();
        assert!(dw.iter().any(|&v| v.abs() > 1e-6),
            "d_in_w all zero — in_proj weight grad did not accumulate");
    }

    #[test]
    fn cpu_trace_shift_adjoint_redistributes_gradient() {
        // 2 neurons, memory_length=3. d_trace_new = [10,20,30, 100,200,300].
        // For neuron 0:
        //   d_pre_act[0] = d_trace_new[2] = 30
        //   d_trace_old[0] = 0 (oldest entry discarded)
        //   d_trace_old[1] = d_trace_new[0] = 10
        //   d_trace_old[2] = d_trace_new[1] = 20
        // Symmetric for neuron 1.
        let d_trace_new = vec![10.0, 20.0, 30.0, 100.0, 200.0, 300.0];
        let mut d_pre_act = vec![0.0f32; 2];
        let mut d_trace_old = vec![0.0f32; 6];
        trace_shift_adjoint_host(&d_trace_new, &mut d_pre_act, &mut d_trace_old, 2, 3);
        assert_eq!(d_pre_act, vec![30.0, 300.0]);
        assert_eq!(d_trace_old, vec![0.0, 10.0, 20.0, 0.0, 100.0, 200.0]);
    }

    #[test]
    fn cpu_sync_batched_matches_scalar() {
        let (batch, d_model) = (3usize, 4usize);
        let left = vec![0usize, 1];
        let right = vec![2usize, 3];
        let n = left.len();
        let r = vec![0.5f32, 0.7];
        let activated: Vec<f32> = (0..batch * d_model)
            .map(|i| ((i * 3 % 5) as f32 - 2.0) * 0.4 + 0.1).collect();

        // Batched: init + one update.
        let mut alpha_b = Vec::new();
        let mut beta_b = Vec::new();
        sync_init_batched(&activated, &left, &right, &mut alpha_b, &mut beta_b, batch, d_model);
        let sync_b = sync_update_batched(&activated, &left, &right, &mut alpha_b, &mut beta_b, &r, batch, d_model);

        // Scalar per example.
        for b in 0..batch {
            let act_b = &activated[b * d_model..(b + 1) * d_model];
            let mut a = Vec::new();
            let mut be = Vec::new();
            sync_init(act_b, &left, &right, &mut a, &mut be);
            let s = sync_update(act_b, &left, &right, &mut a, &mut be, &r);
            for i in 0..n {
                assert!((sync_b[b * n + i] - s[i]).abs() < 1e-5, "sync[{b}][{i}]");
                assert!((alpha_b[b * n + i] - a[i]).abs() < 1e-5, "alpha[{b}][{i}]");
            }
        }

        // Batched reverse vs scalar reverse.
        let d_sync: Vec<f32> = (0..batch * n).map(|i| (i as f32 - 2.0) * 0.3).collect();
        let mut d_act_b = vec![0.0f32; batch * d_model];
        let mut carry_b = vec![0.0f32; batch * n];
        sync_update_reverse_host_batched(&d_sync, &activated, &alpha_b, &beta_b, &r,
            &left, &right, &mut d_act_b, &mut carry_b, batch, d_model);
        for b in 0..batch {
            let act_b = &activated[b * d_model..(b + 1) * d_model];
            let mut da = vec![0.0f32; d_model];
            let mut carry = vec![0.0f32; n];
            sync_update_reverse_host(&d_sync[b * n..(b + 1) * n], act_b,
                &alpha_b[b * n..(b + 1) * n], &beta_b[b * n..(b + 1) * n], &r,
                &left, &right, &mut da, &mut carry);
            for j in 0..d_model {
                assert!((d_act_b[b * d_model + j] - da[j]).abs() < 1e-5, "d_act[{b}][{j}]");
            }
        }
    }

    #[test]
    fn cpu_sync_update_reverse_accumulates_into_activated() {
        // 4 neurons, 2 sync pairs.
        // alpha = [1.0, 2.0], beta = [4.0, 4.0]  → sqrt(beta) = 2.
        // left = [0, 1], right = [2, 3].
        // d_sync = [4.0, 8.0] →
        //   d_alpha[0] = 4.0/2 = 2.0
        //   d_alpha[1] = 8.0/2 = 4.0
        // For activated = [1, 1, 5, 5]:
        //   d_activated[0] += 2.0 * 5 = 10
        //   d_activated[2] += 2.0 * 1 = 2
        //   d_activated[1] += 4.0 * 5 = 20
        //   d_activated[3] += 4.0 * 1 = 4
        let d_sync = vec![4.0, 8.0];
        let activated = vec![1.0, 1.0, 5.0, 5.0];
        let alpha = vec![1.0, 2.0];
        let beta = vec![4.0, 4.0];
        let left = vec![0, 1];
        let right = vec![2, 3];
        let mut d_activated = vec![0.0f32; 4];
        // Single-tick: zero d_alpha carry ⇒ no recurrence term, output is
        // purely this tick's product backward (r is unused when carry=0).
        let r = vec![0.5f32, 0.5];
        let mut d_alpha_carry = vec![0.0f32; 2];
        sync_update_reverse_host(&d_sync, &activated, &alpha, &beta,
            &r, &left, &right, &mut d_activated, &mut d_alpha_carry);
        assert!((d_activated[0] - 10.0).abs() < 1e-5);
        assert!((d_activated[1] - 20.0).abs() < 1e-5);
        assert!((d_activated[2] - 2.0).abs() < 1e-5);
        assert!((d_activated[3] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn cpu_nlm_backward_smoke_shallow() {
        // Shallow NLM: 4 neurons, in_per=4, out_per=2 (so post-GLU = 1 per neuron).
        // Build a SuperLinear, run forward-with-cache, backward.
        // Verifies wiring: d_W and d_b accumulate, d_trace overwrites,
        // shapes line up.
        let n = 4; let in_per = 4; let out_per = 2;
        let weights: Vec<f32> = (0..n * out_per * in_per).map(|i| (i as f32) * 0.1).collect();
        let biases = vec![0.0f32; n * out_per];
        let stage1 = tensor_typed::SuperLinear::<Cpu>::from_host(
            &weights, &biases, n, in_per, out_per,
        ).unwrap();

        let trace: Vec<f32> = (0..n * in_per).map(|i| (i as f32 + 1.0) * 0.05).collect();
        let trace_t = Tensor::<Cpu>::from_slice(&trace).unwrap();

        let (output, cache) = nlm_forward_with_cache::<Cpu>(
            &trace_t, &stage1, None, n,
        ).unwrap();
        assert_eq!(output.len(), n * (out_per / 2));

        let d_activated = Tensor::<Cpu>::from_slice(&vec![1.0; n * (out_per / 2)]).unwrap();
        let mut d_w1 = Tensor::<Cpu>::zeros(n * out_per * in_per).unwrap();
        let mut d_b1 = Tensor::<Cpu>::zeros(n * out_per).unwrap();
        let mut d_trace = Tensor::<Cpu>::zeros(n * in_per).unwrap();
        nlm_backward_typed_helper::<Cpu>(
            &d_activated, &cache, &stage1, None,
            &mut d_w1, &mut d_b1, None, None,
            &mut d_trace,
        ).expect("nlm backward");

        // Sanity: d_trace must have non-zero entries (gradient flowed).
        let dt = d_trace.to_vec().unwrap();
        assert!(dt.iter().any(|&v| v.abs() > 1e-6),
            "d_trace all zero — backward did not flow");
        // d_W1 must have non-zero entries (outer product accumulation).
        let dw = d_w1.to_vec().unwrap();
        assert!(dw.iter().any(|&v| v.abs() > 1e-6),
            "d_W1 all zero — outer product accumulation failed");
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn cpu_and_rocm_typed_ctm_forward_match() {
        let cfg = small_cfg();
        let raw_input_dim = 6;
        let w = CtmWeights::new(cfg.clone(), raw_input_dim);

        let obs: Vec<f32> = (0..raw_input_dim).map(|i| (i as f32 - 2.5) * 0.3).collect();

        // Cpu typed.
        let cpu_typed = CtmWeightsTyped::<Cpu>::from_untyped(&w).unwrap();
        let mut a_cpu = w.start_activated.clone();
        let mut t_cpu = w.start_trace.clone();
        let cpu_out = ctm_forward_typed::<Cpu>(
            &cpu_typed, &mut a_cpu, &mut t_cpu, &obs,
        ).unwrap();

        // Rocm typed.
        let rocm_typed = match CtmWeightsTyped::<Rocm>::from_untyped(&w) {
            Ok(t) => t,
            Err(BackendError::Runtime(msg)) if msg.contains("hipMalloc") => return,
            Err(e) => panic!("rocm CtmWeightsTyped build failed: {:?}", e),
        };
        let mut a_rocm = w.start_activated.clone();
        let mut t_rocm = w.start_trace.clone();
        let rocm_out = match ctm_forward_typed::<Rocm>(
            &rocm_typed, &mut a_rocm, &mut t_rocm, &obs,
        ) {
            Ok(o) => o,
            Err(BackendError::Runtime(msg)) if msg.contains("hipMalloc") => return,
            Err(BackendError::Unsupported { .. }) => return,
            Err(e) => panic!("rocm typed forward failed: {:?}", e),
        };

        for (tick, (a, b)) in cpu_out.predictions.iter()
            .zip(&rocm_out.predictions).enumerate()
        {
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                assert!((x - y).abs() < 1e-2,
                    "tick {} pred[{}] CPU vs Rocm disagree: cpu={} rocm={}",
                    tick, i, x, y);
            }
        }
    }
}
