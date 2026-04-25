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
                let mut projected = w.kv_proj.forward(tok);
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

        let q = w.q_proj.forward(&sync_action);
        let attn_out = multihead_attention(
            &q, kv, n_tokens, d_in, cfg.heads,
            &w.mha_in_proj, &w.mha_out_proj,
        );

        let mut pre_syn = Vec::with_capacity(d_in + d);
        pre_syn.extend_from_slice(&attn_out);
        pre_syn.extend_from_slice(&state.activated);
        let pre_act = w.synapse.forward(&pre_syn);

        for n in 0..d {
            let base = n * m;
            state.trace.copy_within(base + 1..base + m, base);
            state.trace[base + m - 1] = pre_act[n];
        }

        state.activated = nlm_forward(&state.trace, &w.nlm_stage1, w.nlm_stage2.as_ref(), d);

        if collect_traj {
            trajectory.extend_from_slice(&state.activated);
        }

        let sync_out = sync_update(&state.activated, &w.sync_out_left, &w.sync_out_right,
            &mut state.alpha_out, &mut state.beta_out, &r_out);

        let pred = w.output_proj.forward(&sync_out);
        let cert = compute_certainty(&pred);
        predictions.push(pred);
        certainties.push(cert);

        match &cfg.exit_strategy {
            crate::config::ExitStrategy::AdaptiveGate { threshold, .. } => {
                if let Some(ref gate) = w.exit_gate {
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

// ─── Device-resident tick loop ─────────────────────────────
//
// `ctm_forward_resident` mirrors `ctm_forward_with_kv` but routes the
// big fixed-cost matvecs (q_proj, MHA Q/K/V, MHA out, synapse U-Net,
// NLM stages, output_proj, exit_gate) through `CtmResidentCache` —
// weights stay device-resident across ticks, zero PCIe per dispatch.
//
// **What stays host.** The CTM tick state (activated, trace,
// alpha/beta accumulators, predictions, certainties, exit lambdas,
// trajectory) is small Vec<f32> bookkeeping driven by scalar ops
// (sync_init/sync_update, per_neuron_glu, softmax, certainty,
// trace shift, exit-gate sigmoid). Forcing it onto device buys
// nothing and complicates code. We upload to GpuVec::Hip on the
// boundaries where a resident matvec needs it (sync_action →
// q_proj, q_full → MHA, attn||activated → synapse, trace → NLM,
// sync_out → output_proj/exit_gate) and download the matvec result
// back to host before the next host op.
//
// **MHA softmax now resident.** Q (1 vector) and per-token K/V are
// produced by resident matvecs against the row-sliced mha_in_q /
// mha_in_k / mha_in_v sub-Linears. The per-head scores (host
// dot-product against pre-projected host K) are packed into a
// `[n_heads × n_tokens]` device buffer and run through a SINGLE
// `softmax_resident` dispatch (ACCURATE algo, INSTANCE mode → per-N
// softmax across the C axis), then downloaded as softmax weights.
// One MIOpen kernel launch per tick instead of n_heads. The
// per-head V·weights weighted sum stays host because V is in
// [n_tokens × d_head] layout — turning that into a resident matvec
// needs a transpose (or rocBLAS sgemv with transpose flag); keeping
// it host costs only the small per-head reduction. TODO
// (a) pre-transpose V on device so the weighted sum becomes a
// resident matvec, or (b) batch the QK dot-product as a resident
// matvec too so scores never round-trip; either subsumes the
// current upload/download bracket around the softmax dispatch.
//
// **NLM per-neuron GLU now resident.** Stage1 produces a device
// `[d * 2*h]` block; we issue `d` `glu_resident` dispatches (one
// per neuron) writing into a device `[d * h]` GLU output buffer,
// then optionally feed that directly into stage2 + a second
// per-neuron GLU loop. The whole NLM pipeline stays on device;
// only the final activated state downloads back to host where
// sync_update consumes it.

#[cfg(feature = "rocm")]
use modgrad_compute::backend::{GpuVec, ResidencyError};
#[cfg(feature = "rocm")]
use modgrad_device::backend::HipBatch;

/// Device-resident equivalent of `ctm_forward_with_kv`. Drives the
/// inner CTM tick loop with the heavy matvecs dispatched through
/// `CtmResidentCache`. Host state (predictions, certainties,
/// sync accumulators, exit bookkeeping) is identical to the host
/// path; only the matmul transport changes.
///
/// **Device-resident KV input.** Takes pre-projected KV in d_input
/// space as a `&GpuVec` (the `kv_proj` projection happens in the
/// outer caller — typically `RegionalBrain::forward_cached_resident`
/// — so this function focuses purely on the inner tick loop).
/// Accepting `&GpuVec` rather than `&[f32]` lets the caller keep
/// the connection-synapse output device-resident: the per-connection
/// D2H bounce in `forward_cached_resident` disappears.
///
/// **Strategy (a) — single D2H per call.** Today this function
/// downloads `kv` to a host scratch once at the top, then keeps the
/// existing per-token MHA K/V dispatch logic (which uploads each
/// token's d_in slice via `mha_kv_in_dev.copy_from`). Net D2H per
/// inner forward: 1 (down from `n_tokens` worth of upload pressure
/// previously fed by an outer-caller D2H). The per-token H2D
/// uploads remain. TODO: ship strategy (b) as a follow-up — keep
/// `kv` device-resident across the whole forward and replace the
/// per-token H2D upload with a D2D copy from a sub-range of the
/// input GpuVec into `mha_kv_in_dev`. Doable without API changes
/// to `LinearResident::forward` once a `hipMemcpy` D2D wrapper is
/// in place; deferred for honest scope.
///
/// **Cross-forward state contract.** Same as `ctm_forward_with_kv`:
/// `alpha_out`/`beta_out` reset every call via `sync_init`, action
/// accumulators are local, `activated`/`trace` persist across
/// forwards (continuous-thinking semantics).
///
/// Returns `CtmOutput` matching the host path bit-for-bit within
/// 1e-3 FP tolerance (rocBLAS reduces in a different order than
/// AVX-512 dot, hence the loose bound).
#[cfg(feature = "rocm")]
pub fn ctm_forward_resident(
    w: &CtmWeights,
    cache: &crate::ctm_resident::CtmResidentCache,
    state: &mut CtmState,
    batch: &HipBatch,
    kv: &GpuVec,
    n_tokens: usize,
) -> Result<CtmOutput, ResidencyError> {
    let cfg = &w.config;
    let d = cfg.d_model;
    let d_in = cfg.d_input;
    let k = cfg.iterations;
    let m = cfg.memory_length;
    let n_heads = cfg.heads;
    let d_head = d_in / n_heads;
    let scale = 1.0 / (d_head as f32).sqrt();

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

    // Pre-allocated device scratch buffers reused across ticks.
    // q_proj output and MHA Q-projection share d_in shape so they
    // could share a buffer, but cleanest to keep them distinct;
    // hipMalloc cost amortises across the tick loop anyway.
    let n_action = w.q_proj.in_dim;
    let mut sync_action_dev = GpuVec::try_hip(n_action)?;
    let mut q_proj_out_dev = GpuVec::try_hip(d_in)?;
    let mut mha_q_dev = GpuVec::try_hip(d_in)?;
    let mut mha_kv_in_dev = GpuVec::try_hip(d_in)?;
    let mut mha_k_dev = GpuVec::try_hip(d_in)?;
    let mut mha_v_dev = GpuVec::try_hip(d_in)?;
    let mut mha_concat_dev = GpuVec::try_hip(d_in)?;
    let mut attn_out_dev = GpuVec::try_hip(d_in)?;
    let mut pre_syn_dev = GpuVec::try_hip(d_in + d)?;
    let mut pre_act_dev = GpuVec::try_hip(d)?;
    let mut nlm_in_dev = GpuVec::try_hip(d * m)?;
    let mut nlm_s1_out_dev = GpuVec::try_hip(d * w.nlm_stage1.out_per)?;
    // Stage1 GLU output: half_size = out_per / 2, n_neurons = d.
    let s1_half = w.nlm_stage1.out_per / 2;
    let mut nlm_s1_glu_dev = GpuVec::try_hip(d * s1_half)?;
    // Stage2 buffers (raw output and GLU output) — only allocated
    // when stage2 is present.
    let nlm_s2_out_dev: Option<GpuVec> = match &w.nlm_stage2 {
        Some(s2) => Some(GpuVec::try_hip(d * s2.out_per)?),
        None => None,
    };
    let mut nlm_s2_out_dev = nlm_s2_out_dev;
    let nlm_s2_glu_dev: Option<GpuVec> = match &w.nlm_stage2 {
        Some(s2) => Some(GpuVec::try_hip(d * (s2.out_per / 2))?),
        None => None,
    };
    let mut nlm_s2_glu_dev = nlm_s2_glu_dev;
    // MHA softmax scratch — `[n_heads × n_tokens]` packed across
    // all heads. One input buffer (scores per head) + one output
    // buffer (softmax weights per head). Both reused across ticks.
    // We dispatch n_heads softmax_resident calls per tick at offsets
    // into these buffers, then a single flush + D2H drains them all
    // — one device sync per tick instead of per head.
    let mut mha_scores_dev = GpuVec::try_hip(n_heads * n_tokens)?;
    let mut mha_softmax_dev = GpuVec::try_hip(n_heads * n_tokens)?;
    let mut sync_out_dev = GpuVec::try_hip(cfg.n_synch_out)?;
    let mut pred_dev = GpuVec::try_hip(cfg.out_dims)?;
    let mut gate_dev = if w.exit_gate.is_some() {
        Some(GpuVec::try_hip(1)?)
    } else {
        None
    };

    // Strategy (a): download `kv` (device-resident) to a host scratch
    // ONCE up front, then keep the per-token MHA K/V dispatch logic
    // intact (it slices into `kv_host` for `mha_kv_in_dev.copy_from`).
    // This eliminates the caller-side per-connection D2H — graph.rs
    // used to download every connection-synapse output to feed
    // `&[f32]` here. Net D2H per inner forward: 1.
    //
    // TODO: lift to strategy (b) — replace the per-token H2D
    // `mha_kv_in_dev.copy_from` upload with a D2D copy from a
    // sub-range of `kv` directly. That keeps the input fully
    // device-resident and removes the host scratch entirely.
    let kv_host: Vec<f32> = match kv {
        GpuVec::Hip(buf) => {
            let mut host = vec![0.0f32; n_tokens * d_in];
            debug_assert_eq!(buf.len_f32(), n_tokens * d_in,
                "kv GpuVec length mismatch (n_tokens × d_in)");
            buf.copy_to_host(&mut host)?;
            host
        }
        // Non-Hip variants are valid host inputs (Heap/Vram are host-
        // visible per `GpuVec::is_host_visible`); just clone the
        // backing slice. Keeps the function callable from CPU-only
        // tests too.
        _ => kv.as_slice().to_vec(),
    };

    // Project all KV tokens once per call (kv doesn't change across
    // ticks). Each token: K = mha_in_k @ kv_token + bias_k,
    //                     V = mha_in_v @ kv_token + bias_v.
    // Stored host-side because softmax + weighted sum is host.
    // n_tokens × d_in each.
    let mut k_all_host = vec![0.0f32; n_tokens * d_in];
    let mut v_all_host = vec![0.0f32; n_tokens * d_in];
    for t in 0..n_tokens {
        mha_kv_in_dev.copy_from(&kv_host[t * d_in..(t + 1) * d_in]);
        cache.mha_in_k.forward(batch, &mha_kv_in_dev, &mut mha_k_dev)?;
        cache.mha_in_v.forward(batch, &mha_kv_in_dev, &mut mha_v_dev)?;
        // D2H to host scratch. flush guarantees the matvecs above
        // are finished before we read.
        batch.flush()?;
        let mut k_tok = vec![0.0f32; d_in];
        let mut v_tok = vec![0.0f32; d_in];
        match &mha_k_dev {
            GpuVec::Hip(buf) => buf.copy_to_host(&mut k_tok)?,
            _ => unreachable!("hip alloc"),
        }
        match &mha_v_dev {
            GpuVec::Hip(buf) => buf.copy_to_host(&mut v_tok)?,
            _ => unreachable!("hip alloc"),
        }
        k_all_host[t * d_in..(t + 1) * d_in].copy_from_slice(&k_tok);
        v_all_host[t * d_in..(t + 1) * d_in].copy_from_slice(&v_tok);
    }

    for _tick in 0..k {
        // ── Sync (action). Identical scalar op to the host path. ──
        let sync_action = if !action_initialized {
            sync_init(&state.activated, &w.sync_action_left, &w.sync_action_right,
                &mut alpha_action, &mut beta_action);
            action_initialized = true;
            sync_read(&alpha_action, &beta_action)
        } else {
            sync_update(&state.activated, &w.sync_action_left, &w.sync_action_right,
                &mut alpha_action, &mut beta_action, &r_action)
        };

        // ── Q projection: sync_action → d_input via q_proj. ──
        sync_action_dev.copy_from(&sync_action);
        cache.q_proj.forward(batch, &sync_action_dev, &mut q_proj_out_dev)?;

        // ── MHA Q row of in_proj. ──
        // Host code: linear_slice(q_in, in_proj, 0, d_input).
        // Resident: mha_in_q @ q_proj_out → mha_q_dev.
        cache.mha_in_q.forward(batch, &q_proj_out_dev, &mut mha_q_dev)?;
        batch.flush()?;
        let mut q_full = vec![0.0f32; d_in];
        match &mha_q_dev {
            GpuVec::Hip(buf) => buf.copy_to_host(&mut q_full)?,
            _ => unreachable!("hip alloc"),
        }

        // ── Per-head scaled dot-product attention. ──
        // K/V tokens were pre-projected once above. Per-head
        // QK dot-product runs host (K/V live host). Softmax runs
        // resident via `softmax_resident` (ACCURATE algo, INSTANCE
        // mode): all per-head score rows are uploaded packed into
        // `[n_heads × n_tokens]`, a single softmax_resident
        // dispatch handles every head row in one MIOpen call,
        // then we flush + D2H once. The V·weights weighted sum
        // stays host. TODO: lift QK (resident matvec against
        // packed K) and the V·weights weighted sum (V transpose +
        // resident matvec, or sgemv with trans flag) — that
        // subsumes both the upload-scores and download-weights
        // bracket around the softmax kernel.
        let mut concat_heads = vec![0.0f32; d_in];
        let mut scores_host = vec![0.0f32; n_heads * n_tokens];
        for h in 0..n_heads {
            let q_h = &q_full[h * d_head..(h + 1) * d_head];
            let row = &mut scores_host[h * n_tokens..(h + 1) * n_tokens];
            for t in 0..n_tokens {
                let k_h = &k_all_host[
                    t * d_in + h * d_head..t * d_in + (h + 1) * d_head];
                let dot: f32 = q_h.iter().zip(k_h).map(|(&a, &b)| a * b).sum();
                row[t] = dot * scale;
            }
        }

        // Single resident softmax over all head rows at once.
        // [n_heads, n_tokens, 1, 1] tensor with INSTANCE mode → per-N
        // softmax across the row_len axis. ACCURATE algo (max-
        // subtracting) matches the host numeric path within fp
        // tolerance.
        mha_scores_dev.copy_from(&scores_host);
        let scores_ptr = match &mha_scores_dev {
            GpuVec::Hip(b) => b.device_ptr() as *const f32,
            _ => unreachable!("hip alloc"),
        };
        let softmax_ptr = match &mut mha_softmax_dev {
            GpuVec::Hip(b) => b.device_ptr() as *mut f32,
            _ => unreachable!("hip alloc"),
        };
        unsafe {
            modgrad_device::backend::ops::softmax_resident(
                scores_ptr, softmax_ptr,
                n_heads, n_tokens, false,
            )?;
        }
        batch.note_dispatch()?;
        batch.flush()?;
        let mut softmax_host = vec![0.0f32; n_heads * n_tokens];
        match &mha_softmax_dev {
            GpuVec::Hip(buf) => buf.copy_to_host(&mut softmax_host)?,
            _ => unreachable!("hip alloc"),
        }

        // Per-head V·weights weighted sum (host). V is shaped
        // [n_tokens × d_head] per head, weights are [n_tokens]. TODO
        // above.
        for h in 0..n_heads {
            let weights = &softmax_host[h * n_tokens..(h + 1) * n_tokens];
            let head_out = &mut concat_heads[h * d_head..(h + 1) * d_head];
            for j in 0..d_head { head_out[j] = 0.0; }
            for t in 0..n_tokens {
                let w_attn = weights[t];
                let v_h = &v_all_host[
                    t * d_in + h * d_head..t * d_in + (h + 1) * d_head];
                for j in 0..d_head {
                    head_out[j] += w_attn * v_h[j];
                }
            }
        }

        // ── MHA out projection: concat → d_input via mha_out_proj. ──
        mha_concat_dev.copy_from(&concat_heads);
        cache.mha_out_proj.forward(batch, &mha_concat_dev, &mut attn_out_dev)?;
        batch.flush()?;
        let mut attn_out = vec![0.0f32; d_in];
        match &attn_out_dev {
            GpuVec::Hip(buf) => buf.copy_to_host(&mut attn_out)?,
            _ => unreachable!("hip alloc"),
        }

        // ── Synapse U-Net: concat(attn_out, activated) → d_model. ──
        // Build the concat host-side then upload as one block.
        let mut pre_syn = Vec::with_capacity(d_in + d);
        pre_syn.extend_from_slice(&attn_out);
        pre_syn.extend_from_slice(&state.activated);
        pre_syn_dev.copy_from(&pre_syn);
        cache.synapse.forward(batch, &pre_syn_dev, &mut pre_act_dev)?;
        batch.flush()?;
        let mut pre_act = vec![0.0f32; d];
        match &pre_act_dev {
            GpuVec::Hip(buf) => buf.copy_to_host(&mut pre_act)?,
            _ => unreachable!("hip alloc"),
        }

        // ── Trace shift (host, scalar). ──
        for n in 0..d {
            let base = n * m;
            state.trace.copy_within(base + 1..base + m, base);
            state.trace[base + m - 1] = pre_act[n];
        }

        // ── NLM forward: trace → activated, fully resident. ──
        // Stage 1: nlm_stage1 (resident SuperLinear) → s1_out_dev.
        // Per-neuron GLU dispatch loop writes into nlm_s1_glu_dev.
        // If stage2 is Some: stage2 (resident SuperLinear) reads
        // nlm_s1_glu_dev → nlm_s2_out_dev, then a second per-neuron
        // GLU dispatch loop writes nlm_s2_glu_dev. Final activated
        // is the only host download.
        nlm_in_dev.copy_from(&state.trace);
        cache.nlm_stage1.forward(batch, &nlm_in_dev, &mut nlm_s1_out_dev)?;

        // Per-neuron GLU on stage1 output. Each neuron's slice in
        // nlm_s1_out_dev is `[2*s1_half]` — laid out as the values
        // half followed by the gates half WITHIN that neuron — which
        // matches MIOpen's dim=0 GLU layout when viewed as a single
        // [2, s1_half, 1, 1] tensor (`n_rows = 1, half_size = s1_half`).
        // n_neurons (= d) dispatches per stage; same cadence as the
        // SuperLinearResident matvec loop above.
        {
            let s1_out_base = match &nlm_s1_out_dev {
                GpuVec::Hip(b) => b.device_ptr() as *const f32,
                _ => unreachable!("hip alloc"),
            };
            let s1_glu_base = match &mut nlm_s1_glu_dev {
                GpuVec::Hip(b) => b.device_ptr() as *mut f32,
                _ => unreachable!("hip alloc"),
            };
            let s1_in_stride = w.nlm_stage1.out_per; // 2 * s1_half
            for n in 0..d {
                unsafe {
                    modgrad_device::backend::ops::glu_resident(
                        s1_out_base.add(n * s1_in_stride),
                        s1_glu_base.add(n * s1_half),
                        1, s1_half,
                    )?;
                }
                batch.note_dispatch()?;
            }
        }

        let new_activated = if let Some(s2_resident) = &cache.nlm_stage2 {
            // stage2 reads stage1's GLU output (already device-resident
            // in nlm_s1_glu_dev). Output goes into nlm_s2_out_dev.
            let s2_weights = w.nlm_stage2.as_ref()
                .expect("nlm_stage2 host weights present when cache.nlm_stage2 is Some");
            let s2_out_dev = nlm_s2_out_dev.as_mut()
                .expect("s2 out buffer allocated when stage2 present");
            s2_resident.forward(batch, &nlm_s1_glu_dev, s2_out_dev)?;

            // Per-neuron GLU on stage2 output, same pattern as stage1.
            let s2_half = s2_weights.out_per / 2;
            let s2_glu_dev = nlm_s2_glu_dev.as_mut()
                .expect("s2 glu buffer allocated when stage2 present");
            {
                let s2_out_base = match &*s2_out_dev {
                    GpuVec::Hip(b) => b.device_ptr() as *const f32,
                    _ => unreachable!("hip alloc"),
                };
                let s2_glu_base = match &mut *s2_glu_dev {
                    GpuVec::Hip(b) => b.device_ptr() as *mut f32,
                    _ => unreachable!("hip alloc"),
                };
                let s2_in_stride = s2_weights.out_per;
                for n in 0..d {
                    unsafe {
                        modgrad_device::backend::ops::glu_resident(
                            s2_out_base.add(n * s2_in_stride),
                            s2_glu_base.add(n * s2_half),
                            1, s2_half,
                        )?;
                    }
                    batch.note_dispatch()?;
                }
            }
            batch.flush()?;
            let mut activated_host = vec![0.0f32; d * s2_half];
            match &*s2_glu_dev {
                GpuVec::Hip(buf) => buf.copy_to_host(&mut activated_host)?,
                _ => unreachable!("hip alloc"),
            }
            activated_host
        } else {
            batch.flush()?;
            let mut activated_host = vec![0.0f32; d * s1_half];
            match &nlm_s1_glu_dev {
                GpuVec::Hip(buf) => buf.copy_to_host(&mut activated_host)?,
                _ => unreachable!("hip alloc"),
            }
            activated_host
        };
        state.activated = new_activated;

        if collect_traj {
            trajectory.extend_from_slice(&state.activated);
        }

        // ── Sync (out): identical scalar op to host. ──
        let sync_out = sync_update(&state.activated, &w.sync_out_left, &w.sync_out_right,
            &mut state.alpha_out, &mut state.beta_out, &r_out);

        // ── Output projection: sync_out → out_dims via output_proj. ──
        sync_out_dev.copy_from(&sync_out);
        cache.output_proj.forward(batch, &sync_out_dev, &mut pred_dev)?;
        batch.flush()?;
        let mut pred = vec![0.0f32; cfg.out_dims];
        match &pred_dev {
            GpuVec::Hip(buf) => buf.copy_to_host(&mut pred)?,
            _ => unreachable!("hip alloc"),
        }
        let cert = compute_certainty(&pred);
        predictions.push(pred);
        certainties.push(cert);

        // ── Exit strategy: identical to host. ──
        match &cfg.exit_strategy {
            crate::config::ExitStrategy::AdaptiveGate { threshold, .. } => {
                if let (Some(gate_resident), Some(gate_buf)) =
                    (cache.exit_gate.as_ref(), gate_dev.as_mut())
                {
                    gate_resident.forward(batch, &sync_out_dev, gate_buf)?;
                    batch.flush()?;
                    let mut gate_out = [0.0f32; 1];
                    match &*gate_buf {
                        GpuVec::Hip(buf) => buf.copy_to_host(&mut gate_out)?,
                        _ => unreachable!("hip alloc"),
                    }
                    let lambda = 1.0 / (1.0 + (-gate_out[0]).exp());
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
    Ok(CtmOutput {
        predictions, certainties, sync_out, exit_lambdas, ticks_used, trajectory,
    })
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
    q_in: &[f32],       // [d_input]
    kv_flat: &[f32],    // [n_tokens × d_input]
    n_tokens: usize,
    d_input: usize,
    n_heads: usize,
    in_proj: &Linear,    // d_input → 3×d_input
    out_proj: &Linear,   // d_input → d_input
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

        // Weighted sum of values
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CtmConfig;
    use crate::weights::{CtmWeights, CtmState};

    #[test]
    fn smoke_forward() {
        let cfg = CtmConfig {
            iterations: 4,
            d_model: 64,
            d_input: 32,
            heads: 4,
            n_synch_out: 32,
            n_synch_action: 32,
            synapse_depth: 3,
            memory_length: 8,
            deep_nlms: true,
            memory_hidden_dims: 4,
            out_dims: 10,
            n_random_pairing_self: 0,
            min_width: 16,
            ..Default::default()
        };
        let raw_dim = 16;
        let w = CtmWeights::new(cfg.clone(), raw_dim);
        let mut state = CtmState::new(&w);

        eprintln!("  params: {}", w.n_params());
        eprintln!("  synapse widths: {:?}", w.synapse.widths);

        // Single token observation
        let obs = vec![0.5f32; raw_dim];
        let out = ctm_forward(&w, &mut state, CtmInput::Raw {
            obs: &obs, n_tokens: 1, raw_dim,
        });

        assert_eq!(out.predictions.len(), cfg.iterations);
        assert_eq!(out.predictions[0].len(), cfg.out_dims);
        assert_eq!(out.certainties.len(), cfg.iterations);

        eprintln!("  tick 0 pred: {:?}", &out.predictions[0][..5]);
        eprintln!("  tick 0 cert: {:?}", out.certainties[0]);
        eprintln!("  tick {} pred: {:?}", cfg.iterations - 1,
            &out.predictions[cfg.iterations - 1][..5]);
        eprintln!("  tick {} cert: {:?}", cfg.iterations - 1,
            out.certainties[cfg.iterations - 1]);

        // Predictions should differ across ticks (thinking changes output)
        let diff: f32 = out.predictions[0].iter()
            .zip(&out.predictions[cfg.iterations - 1])
            .map(|(a, b)| (a - b).abs())
            .sum();
        eprintln!("  pred diff (tick 0 vs last): {:.4}", diff);
        assert!(diff > 0.0, "predictions should change across ticks");
    }

    /// Host `ctm_forward_with_kv` and resident `ctm_forward_resident`
    /// must produce matching predictions (first tick, last tick) and
    /// matching final sync_out within 1e-3 FP tolerance. rocBLAS
    /// reduces in a different order than AVX-512 dot, so the bound
    /// is loose. AdaptiveGate is enabled to exercise the resident
    /// exit_gate dispatch path.
    ///
    /// **Why two states.** Both forwards mutate `CtmState` (trace,
    /// activated, alpha/beta_out). To keep the comparison clean each
    /// path gets its own state, both seeded identically from
    /// `CtmState::new(&w)`.
    #[cfg(feature = "rocm")]
    #[test]
    fn ctm_forward_resident_matches_host() {
        use crate::config::ExitStrategy;
        use crate::ctm_resident::CtmResidentCache;
        use modgrad_device::backend::HipBatch;
        use modgrad_device::backend::rocm::ffi::runtime_available;
        use modgrad_compute::neuron::SimpleRng;

        if !runtime_available() {
            eprintln!("hip runtime unavailable, skipping");
            return;
        }

        let cfg = CtmConfig {
            iterations: 4,
            d_model: 64,
            d_input: 32,
            heads: 4,
            n_synch_out: 32,
            n_synch_action: 32,
            synapse_depth: 3,
            memory_length: 8,
            deep_nlms: true,
            memory_hidden_dims: 4,
            out_dims: 10,
            n_random_pairing_self: 0,
            min_width: 16,
            exit_strategy: ExitStrategy::AdaptiveGate { beta: 0.1, threshold: 0.99 },
            collect_trajectories: false,
        };
        let raw_input_dim = 16;
        let w = CtmWeights::new(cfg.clone(), raw_input_dim);

        // Pre-projected kv in d_input space; n_tokens = 3.
        let n_tokens = 3;
        let mut rng = SimpleRng::new(0xCD_FE_E0);
        let kv: Vec<f32> = (0..n_tokens * cfg.d_input)
            .map(|_| rng.next_normal()).collect();

        // Host run.
        let mut host_state = CtmState::new(&w);
        let host_out = ctm_forward_with_kv(&w, &mut host_state, &kv, n_tokens);

        // Resident run with a fresh state. Upload `kv` to a
        // device-resident `GpuVec::Hip` to match the new
        // `ctm_forward_resident` signature (`&GpuVec`).
        let mut resident_state = CtmState::new(&w);
        let cache = CtmResidentCache::from_weights(&w)
            .expect("CtmResidentCache::from_weights");
        let batch = HipBatch::new();
        let mut kv_dev = modgrad_compute::backend::GpuVec::try_hip(kv.len())
            .expect("GpuVec::try_hip(kv)");
        kv_dev.copy_from(&kv);
        let resident_out = ctm_forward_resident(
            &w, &cache, &mut resident_state, &batch, &kv_dev, n_tokens,
        ).expect("ctm_forward_resident");
        batch.flush().expect("flush");

        assert_eq!(host_out.ticks_used, resident_out.ticks_used,
            "ticks_used mismatch (exit gate decided differently)");
        let last = host_out.ticks_used.saturating_sub(1);
        assert!(host_out.ticks_used >= 1);

        let max_diff = |a: &[f32], b: &[f32]| {
            a.iter().zip(b).map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        };

        let p0 = max_diff(&host_out.predictions[0], &resident_out.predictions[0]);
        assert!(p0 < 1e-3, "predictions[0] mismatch: max |Δ| = {p0}");

        let pl = max_diff(&host_out.predictions[last], &resident_out.predictions[last]);
        assert!(pl < 1e-3, "predictions[last] mismatch: max |Δ| = {pl}");

        let sd = max_diff(&host_out.sync_out, &resident_out.sync_out);
        assert!(sd < 1e-3, "sync_out mismatch: max |Δ| = {sd}");

        eprintln!("  ticks_used = {}", resident_out.ticks_used);
        eprintln!("  max |Δ predictions[0]|    = {p0:.6}");
        eprintln!("  max |Δ predictions[last]| = {pl:.6}");
        eprintln!("  max |Δ sync_out|          = {sd:.6}");
    }
}
