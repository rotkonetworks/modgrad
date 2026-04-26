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
// **NLM per-neuron GLU now resident and batched.** Stage1 produces
// a device `[d * 2*h]` block; a single
// `per_neuron_glu_batched_resident` dispatch (custom hipcc kernel)
// writes into a device `[d * h]` GLU output buffer, then optionally
// feeds that directly into stage2 + a second batched GLU dispatch.
// The whole NLM pipeline stays on device; only the final activated
// state downloads back to host where sync_update consumes it.
// Replacing the previous N-call MIOpen GLU loop drops `2*d` MIOpen
// dispatches per CTM tick to 2 hipcc launches.

#[cfg(feature = "rocm")]
use modgrad_compute::backend::{GpuVec, ResidencyError};
#[cfg(feature = "rocm")]
use modgrad_device::backend::HipBatch;

// ─── Resident-forward cache (PART A of backward-resident) ───────
//
// `CtmCacheResident` is the device-resident equivalent of host
// `crate::train::CtmCache`. It exists separately rather than reusing
// the host type because:
//   1. The host `CtmCache` and its inner `TickCache` are private to
//      `train.rs` (no public constructor); we cannot construct one
//      from this file.
//   2. The cache shape that resident BACKWARD will consume is shaped
//      around MIOpen's *Backward* dispatch contracts (LN backward,
//      activation backward, GLU backward, softmax backward), which
//      take (input, grad_output) and recompute internal stats — so
//      the resident cache only needs to save inputs to each forward
//      stage, not the normalized/inv_std/pre-silu trio that the host
//      CPU backward consumes. Same logical content; flatter shape.
//
// Storage strategy. Everything is host-side `Vec<f32>` for THIS
// slice. Resident-backward (next slice) re-uploads whatever it
// needs. Rationale: keeping (iterations × U-Net depth × per-block
// buffers) as `GpuVec` would explode VRAM (every outer-tick × region
// × inner-tick × block keeps a buffer live across the whole backward
// pass) — and the cache lifecycle straddles the entire outer-tick
// loop in `RegionalBrain::forward_cached_resident`. Host-side keeps
// the device-residency window tight to just-the-forward.
//
// What the resident path materialises vs. doesn't.
//   - Already host-resident in `ctm_forward_resident`: q_full,
//     k_all, v_all, attn_weights, attn_out, pre_syn (synapse input),
//     pre_act (synapse output), state.activated (post-NLM),
//     sync_action, sync_out, beta_action, beta_out, predictions,
//     exit_lambda, exit_gate input (= sync_out). These flow into
//     the cache directly — no extra D2H.
//   - Need an extra D2H to fill the cache: q_proj output (= input
//     to mha_in_q), nlm stage1 raw output (pre per-neuron-GLU), nlm
//     stage1 GLU output (= stage2 input when present), nlm stage2
//     raw output. Four small D2Hs per tick total (the last one
//     gated on `cfg.deep_nlms`).
//   - U-Net interior intermediates (per-block linear input, per-
//     block ln norm/invstd/pre-silu, skip-LN cache, down_outs,
//     pre_skip_ln). NOT captured here. The resident U-Net dispatch
//     (`SynapseUNetResident::forward`) is monolithic — pre_syn in,
//     pre_act out, no per-block hooks. Resident backward will need
//     either (a) a host-side `synapse.forward_cached(pre_syn)` to
//     reconstruct intermediates, or (b) a future
//     `SynapseUNetResident::forward_cached` that exposes per-block
//     buffers. Both are out-of-scope for this slice and tracked on
//     the next-slice plan. The cache stores `pre_syn` and `pre_act`
//     so either approach is unblocked.

/// Per-tick host-side cache from a resident inner CTM forward.
///
/// Mirrors the host `crate::train::TickCache` content needed for
/// backward, but flat (one struct, no nested `LinearCache`/`MhaCache`/
/// `UNetCache`/`NlmCache`) because resident backward will dispatch
/// directly through MIOpen *Backward* symbols which take only
/// (input, grad_output) — no need to thread normalized/inv_std
/// per-LN.
///
/// All fields are host-resident `Vec<f32>` for this slice. Resident
/// backward re-uploads as needed; see the module-level comment on
/// `CtmCacheResident` for the storage rationale.
#[cfg(feature = "rocm")]
pub struct CtmTickCacheResident {
    /// Activated state at start of tick (pre-NLM, pre-trace-shift).
    /// Used by sync_action backward (gradient w.r.t. activated_prev)
    /// and as the U-Net residual half (`pre_syn[d_in..]`).
    pub activated_prev: Vec<f32>,
    /// `sync_action` vector at this tick (= input to `q_proj`).
    /// Length `cfg.n_synch_action`.
    pub sync_action: Vec<f32>,
    /// `beta_action` accumulator at this tick (read by sync_action
    /// backward). Length `cfg.n_synch_action`.
    pub beta_action: Vec<f32>,
    /// `q_proj` output (= input to `mha_in_proj`). Length `d_input`.
    /// Captured via an extra D2H of `q_proj_out_dev` per tick (the
    /// existing path only downloads `mha_q_dev`).
    pub q_proj_out: Vec<f32>,
    /// MHA Q row of in_proj (= `mha_in_q @ q_proj_out`).
    /// Length `d_input`. Used by MHA backward (per-head Q gradient).
    pub q_full: Vec<f32>,
    /// MHA K all-tokens (`[n_tokens × d_input]`, packed by token then
    /// head). Used by MHA backward (per-head K gradient and the
    /// QK-attention gradient).
    pub k_all: Vec<f32>,
    /// MHA V all-tokens (`[n_tokens × d_input]`). Used by MHA
    /// backward (V gradient and softmax-input gradient).
    pub v_all: Vec<f32>,
    /// Per-token KV input to `mha_in_k`/`mha_in_v` (= the kv slice
    /// passed in, sliced per token). `kv_tokens[t]` length =
    /// `d_input`. Used by MHA backward to compute the gradient
    /// w.r.t. the KV input — itself the synapse input for upstream
    /// regions.
    pub kv_tokens: Vec<Vec<f32>>,
    /// Softmax weights per head, packed `[n_heads × n_tokens]`.
    /// Used by MHA softmax backward.
    pub attn_weights: Vec<f32>,
    /// Concatenated heads (= input to `mha_out_proj`). Length
    /// `d_input`. Used by `mha_out_proj` backward.
    pub concat_heads: Vec<f32>,
    /// Attention output (= `mha_out_proj` output, which is the first
    /// half of the synapse input). Length `d_input`.
    pub attn_out: Vec<f32>,
    /// Synapse U-Net input — the concat `[attn_out, activated_prev]`
    /// fed to `synapse.forward()`. Length `d_input + d_model`.
    /// **Backward gap.** The resident U-Net dispatch is monolithic;
    /// per-block intermediates (linear input, ln norm/invstd,
    /// pre-silu, skip caches) are NOT captured here. Resident
    /// backward must either (a) re-run the host-side
    /// `unet_forward_cached(pre_syn)` to recover intermediates, or
    /// (b) wait for `SynapseUNetResident::forward_cached` to expose
    /// per-block buffers. See module-level comment on
    /// `CtmCacheResident`.
    pub pre_syn: Vec<f32>,
    /// Synapse output (= U-Net output, pre-trace-shift). Length
    /// `d_model`. Saved as the U-Net upstream gradient target so
    /// resident backward can verify the dx shape.
    pub pre_act: Vec<f32>,
    /// Trace state AFTER trace-shift, BEFORE NLM forward (=
    /// `nlm_stage1` input). Length `d_model × memory_length`.
    /// **No D2H needed** — trace lives host-side already
    /// (`state.trace`); cache holds a clone.
    pub trace: Vec<f32>,
    /// `nlm_stage1` raw output (BEFORE per-neuron GLU). Length
    /// `d_model × stage1.out_per`. **D2H added by this slice.**
    /// Required by GLU backward (the per-neuron GLU input is the raw
    /// stage1 output).
    pub nlm_s1_out: Vec<f32>,
    /// `nlm_stage1` GLU output (= `nlm_stage2` input when stage2 is
    /// present, else = `activated_post`). Length
    /// `d_model × (stage1.out_per / 2)`. **D2H added by this slice.**
    pub nlm_s1_glu: Vec<f32>,
    /// `nlm_stage2` raw output (BEFORE per-neuron GLU). `Some` iff
    /// `cfg.deep_nlms`. Length `d_model × stage2.out_per`. **D2H
    /// added by this slice when stage2 is present.**
    pub nlm_s2_out: Option<Vec<f32>>,
    /// Activated state AFTER NLM (= region output). Length
    /// `d_model`.
    pub activated_post: Vec<f32>,
    /// `sync_out` at this tick (= input to `output_proj` and to
    /// `exit_gate`). Length `cfg.n_synch_out`.
    pub sync_out: Vec<f32>,
    /// `beta_out` accumulator at this tick. Length `cfg.n_synch_out`.
    pub beta_out: Vec<f32>,
    /// Output projection logits (`output_proj.forward(sync_out)`).
    /// Length `cfg.out_dims`. Used to short-circuit certainty/loss
    /// recomputation in backward.
    pub pred: Vec<f32>,
    /// Exit gate logit (saved as length-1 vec for shape consistency
    /// with the other LinearCache slots). `Some` iff
    /// `cfg.exit_strategy == AdaptiveGate { .. }` AND
    /// `w.exit_gate.is_some()`. Length 1.
    pub exit_gate_logit: Option<Vec<f32>>,
    /// `λ_t = σ(gate_logit)` at this tick. `0.0` when gate is off.
    pub exit_lambda: f32,
}

/// Cache from `ctm_forward_resident` — everything resident backward
/// will need to drive a BPTT pass through `RegionalBrain` regions.
///
/// Mirrors the layout of host `crate::train::CtmCache` (top-level KV
/// + per-tick caches + decay rates); see `CtmTickCacheResident` for
/// the per-tick content. Built and consumed by:
/// - `ctm_forward_resident` (this file) — populates.
/// - `RegionalBrain::backward_cached_resident` (next slice) —
///   consumes through `OuterTickCache::region_caches_resident`.
#[cfg(feature = "rocm")]
pub struct CtmCacheResident {
    /// Per-tick caches. Length = `ticks_used` (may be less than
    /// `cfg.iterations` when adaptive exit fires).
    pub tick_caches: Vec<CtmTickCacheResident>,
    /// KV stream for the inner forward (host-side copy of the
    /// downloaded device KV — same as `kv_host` in the inner forward
    /// fn). `[n_tokens × d_input]`.
    pub kv: Vec<f32>,
    /// Number of KV tokens (matches `n_tokens` arg to inner forward).
    pub n_tokens: usize,
    /// `cfg.d_input`. Cached for backward shape arithmetic without
    /// re-reading config.
    pub d_input: usize,
    /// Sync-out decay rates `r_out[i] = exp(-decay_params_out[i])`.
    /// Used by sync-out backward.
    pub r_out: Vec<f32>,
    /// Sync-action decay rates `r_action[i] = exp(-decay_params_action[i])`.
    /// Used by sync-action backward.
    pub r_action: Vec<f32>,
}

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
/// Returns `(CtmOutput, CtmCacheResident)` matching the host path
/// bit-for-bit within 1e-3 FP tolerance (rocBLAS reduces in a
/// different order than AVX-512 dot, hence the loose bound). The
/// cache contains everything resident backward needs — see the
/// module-level comment on `CtmCacheResident` for storage rationale
/// and the known U-Net interior gap.
///
/// **PART A of backward-resident.** Returning a populated
/// `CtmCacheResident` (rather than just `CtmOutput`) is the
/// prerequisite for `RegionalBrain::backward_cached_resident` (next
/// slice). The forward numerics are unchanged; the only behavioral
/// difference is four extra small D2H downloads per tick to
/// capture q_proj output, nlm stage1 raw output, stage1 GLU output,
/// and (when `deep_nlms`) stage2 raw output. The U-Net pre-syn /
/// pre-act pair is already host-resident in this function (no
/// extra D2H).
#[cfg(feature = "rocm")]
pub fn ctm_forward_resident(
    w: &CtmWeights,
    cache: &crate::ctm_resident::CtmResidentCache,
    state: &mut CtmState,
    batch: &HipBatch,
    kv: &GpuVec,
    n_tokens: usize,
) -> Result<(CtmOutput, CtmCacheResident), ResidencyError> {
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

    // Resident-backward cache accumulator. Pushed exactly once per
    // tick that runs to completion (mirrors `predictions.push`
    // cadence); when adaptive exit fires mid-tick we still push for
    // the just-completed tick before breaking, so
    // `tick_caches.len() == predictions.len() == ticks_used` holds.
    let mut tick_caches: Vec<CtmTickCacheResident> = Vec::with_capacity(k);

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
        // ── Cache: activated_prev (snapshot before NLM mutates it). ──
        // Backward needs this for the sync_action gradient and for
        // the U-Net residual half (`pre_syn[d_in..]` = activated_prev).
        let activated_prev_cache: Vec<f32> = state.activated.clone();

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
        // beta_action snapshot for sync_action backward (the action
        // sync accumulators decay across the next tick — clone now).
        let beta_action_cache: Vec<f32> = beta_action.clone();

        // ── Q projection: sync_action → d_input via q_proj. ──
        sync_action_dev.copy_from(&sync_action);
        cache.q_proj.forward(batch, &sync_action_dev, &mut q_proj_out_dev)?;

        // ── Cache: q_proj_out (= input to mha_in_q). ──
        // Existing path only downloads `mha_q_dev` (= mha_in_q @
        // q_proj_out). For backward through `mha_in_proj` we need the
        // q_proj output itself (which is the `q_in` of `MhaCache` in
        // the host). Extra D2H of d_in floats per tick.
        batch.flush()?;
        let mut q_proj_out_cache = vec![0.0f32; d_in];
        match &q_proj_out_dev {
            GpuVec::Hip(buf) => buf.copy_to_host(&mut q_proj_out_cache)?,
            _ => unreachable!("hip alloc"),
        }

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
        // Cache the U-Net input/output. Per-block intermediates
        // remain a known gap; see CtmCacheResident doc-comment.
        let pre_syn_cache: Vec<f32> = pre_syn.clone();
        let pre_act_cache: Vec<f32> = pre_act.clone();

        // ── Trace shift (host, scalar). ──
        for n in 0..d {
            let base = n * m;
            state.trace.copy_within(base + 1..base + m, base);
            state.trace[base + m - 1] = pre_act[n];
        }
        // Cache trace AFTER shift (= input to NLM stage1). The host
        // CtmCache analogue is `s1_cache.input` inside `NlmCache`.
        // Cloned host-side (no D2H needed: trace lives host-side).
        let trace_cache: Vec<f32> = state.trace.clone();

        // ── NLM forward: trace → activated, fully resident. ──
        // Stage 1: nlm_stage1 (resident SuperLinear) → s1_out_dev.
        // A single per-neuron-batched GLU dispatch (custom hipcc
        // kernel) writes into nlm_s1_glu_dev. If stage2 is Some:
        // stage2 (resident SuperLinear) reads nlm_s1_glu_dev →
        // nlm_s2_out_dev, then a second batched GLU dispatch writes
        // nlm_s2_glu_dev. Final activated is the only host download.
        nlm_in_dev.copy_from(&state.trace);
        cache.nlm_stage1.forward(batch, &nlm_in_dev, &mut nlm_s1_out_dev)?;

        // Per-neuron GLU on stage1 output, batched into a single
        // dispatch. Each neuron's slice in nlm_s1_out_dev is
        // `[2*s1_half]` — values half followed by gates half
        // WITHIN that neuron. MIOpen's GLU solver only splits along
        // the leading axis (`src/solver/glu/forward_glu.cpp`), so the
        // per-neuron interleaved layout cannot be batched in one
        // MIOpen call; the custom hipcc kernel
        // (`crates/modgrad-device/kernels/per_neuron_glu_batched.hip`)
        // handles the interleaved layout in one launch and replaces
        // the previous `d`-call MIOpen loop. Total threads launched =
        // `d * s1_half`. The debug_assert pins `out_per == 2 * s1_half`
        // — the kernel derives the per-neuron stride from `2 * half_size`.
        debug_assert_eq!(w.nlm_stage1.out_per, 2 * s1_half);
        {
            let s1_out_base = match &nlm_s1_out_dev {
                GpuVec::Hip(b) => b.device_ptr() as *const f32,
                _ => unreachable!("hip alloc"),
            };
            let s1_glu_base = match &mut nlm_s1_glu_dev {
                GpuVec::Hip(b) => b.device_ptr() as *mut f32,
                _ => unreachable!("hip alloc"),
            };
            unsafe {
                modgrad_device::backend::ops::per_neuron_glu_batched_resident(
                    s1_out_base, s1_glu_base, d, s1_half,
                )?;
            }
            batch.note_dispatch()?;
        }

        // ── Cache: D2H stage1 raw output (BEFORE per-neuron GLU). ──
        // Required by GLU backward (the per-neuron GLU input is the
        // raw stage1 output). The kernel is async, so flush before
        // download. d_model × stage1.out_per floats per tick.
        batch.flush()?;
        let nlm_s1_out_cache: Vec<f32> = {
            let mut h = vec![0.0f32; d * w.nlm_stage1.out_per];
            match &nlm_s1_out_dev {
                GpuVec::Hip(buf) => buf.copy_to_host(&mut h)?,
                _ => unreachable!("hip alloc"),
            }
            h
        };
        // ── Cache: D2H stage1 GLU output (= stage2 input or =
        // activated when no stage2). Required by stage2 backward
        // (input gradient w.r.t. stage1 GLU). d_model × s1_half
        // floats per tick.
        let nlm_s1_glu_cache: Vec<f32> = {
            let mut h = vec![0.0f32; d * s1_half];
            match &nlm_s1_glu_dev {
                GpuVec::Hip(buf) => buf.copy_to_host(&mut h)?,
                _ => unreachable!("hip alloc"),
            }
            h
        };

        let (new_activated, nlm_s2_out_cache_opt): (Vec<f32>, Option<Vec<f32>>) =
        if let Some(s2_resident) = &cache.nlm_stage2 {
            // stage2 reads stage1's GLU output (already device-resident
            // in nlm_s1_glu_dev). Output goes into nlm_s2_out_dev.
            let s2_weights = w.nlm_stage2.as_ref()
                .expect("nlm_stage2 host weights present when cache.nlm_stage2 is Some");
            let s2_out_dev = nlm_s2_out_dev.as_mut()
                .expect("s2 out buffer allocated when stage2 present");
            s2_resident.forward(batch, &nlm_s1_glu_dev, s2_out_dev)?;

            // ── Cache: D2H stage2 raw output (BEFORE per-neuron GLU). ──
            // Required by GLU backward (the per-neuron GLU input is
            // the raw stage2 output). Flush before download to wait
            // on the stage2 matvec dispatches. d_model × stage2.out_per
            // floats per tick when deep_nlms.
            batch.flush()?;
            let s2_out_cache: Vec<f32> = {
                let mut h = vec![0.0f32; d * s2_weights.out_per];
                match &*s2_out_dev {
                    GpuVec::Hip(buf) => buf.copy_to_host(&mut h)?,
                    _ => unreachable!("hip alloc"),
                }
                h
            };

            // Per-neuron GLU on stage2 output, same pattern as
            // stage1: one batched dispatch into the custom hipcc
            // kernel rather than `d` MIOpen calls.
            let s2_half = s2_weights.out_per / 2;
            debug_assert_eq!(s2_weights.out_per, 2 * s2_half);
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
                unsafe {
                    modgrad_device::backend::ops::per_neuron_glu_batched_resident(
                        s2_out_base, s2_glu_base, d, s2_half,
                    )?;
                }
                batch.note_dispatch()?;
            }
            batch.flush()?;
            let mut activated_host = vec![0.0f32; d * s2_half];
            match &*s2_glu_dev {
                GpuVec::Hip(buf) => buf.copy_to_host(&mut activated_host)?,
                _ => unreachable!("hip alloc"),
            }
            (activated_host, Some(s2_out_cache))
        } else {
            // No stage2: activated = nlm_s1_glu (already downloaded
            // above into `nlm_s1_glu_cache`). Reuse that download to
            // avoid a second D2H of the same buffer. The flush above
            // already ensured the GLU dispatches completed.
            (nlm_s1_glu_cache.clone(), None)
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
        predictions.push(pred.clone());
        certainties.push(cert);

        // ── Exit strategy: identical to host. ──
        // Run the gate dispatch first (when AdaptiveGate is on) so
        // its logit can be captured into the per-tick cache before
        // we decide whether to break out of the loop. We push the
        // cache UNCONDITIONALLY (mirroring `predictions.push` above)
        // so `tick_caches.len() == predictions.len()` holds across
        // both early-exit and full runs.
        let mut should_break = false;
        let mut exit_gate_logit_cache: Option<Vec<f32>> = None;
        let mut exit_lambda_cache: f32 = 0.0;
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
                    exit_gate_logit_cache = Some(vec![gate_out[0]]);
                    exit_lambda_cache = lambda;
                    if exit_cdf > *threshold { should_break = true; }
                }
            }
            crate::config::ExitStrategy::Certainty { threshold } => {
                if cert[1] > *threshold { should_break = true; }
            }
            crate::config::ExitStrategy::None => {}
        }

        // ── Cache push (mirrors host `tick_caches.push`). ──
        // Same cadence as `predictions.push` above so length
        // invariants hold for backward. Per-tick locals (sync_action,
        // q_full, softmax_host, concat_heads, attn_out, sync_out)
        // are MOVED in; loop-invariant scratches (k_all_host,
        // v_all_host, kv_host slices) are CLONED because backward
        // wants per-tick MhaCache content (host TickCache layout).
        tick_caches.push(CtmTickCacheResident {
            activated_prev: activated_prev_cache,
            sync_action,
            beta_action: beta_action_cache,
            q_proj_out: q_proj_out_cache,
            q_full,
            k_all: k_all_host.clone(),
            v_all: v_all_host.clone(),
            kv_tokens: (0..n_tokens)
                .map(|t| kv_host[t * d_in..(t + 1) * d_in].to_vec())
                .collect(),
            attn_weights: softmax_host,
            concat_heads,
            attn_out,
            pre_syn: pre_syn_cache,
            pre_act: pre_act_cache,
            trace: trace_cache,
            nlm_s1_out: nlm_s1_out_cache,
            nlm_s1_glu: nlm_s1_glu_cache,
            nlm_s2_out: nlm_s2_out_cache_opt,
            activated_post: state.activated.clone(),
            sync_out,
            beta_out: state.beta_out.clone(),
            pred,
            exit_gate_logit: exit_gate_logit_cache,
            exit_lambda: exit_lambda_cache,
        });

        if should_break { break; }
    }

    let ticks_used = predictions.len();
    let sync_out = sync_read(&state.alpha_out, &state.beta_out);
    let output = CtmOutput {
        predictions, certainties, sync_out, exit_lambdas, ticks_used, trajectory,
    };
    let ctm_cache_resident = CtmCacheResident {
        tick_caches,
        kv: kv_host,
        n_tokens,
        d_input: d_in,
        r_out,
        r_action,
    };
    Ok((output, ctm_cache_resident))
}

// ─── Resident inner-CTM backward (PART B) ─────────────────────
//
// `ctm_backward_resident` mirrors `train::backward_from_activated`
// structurally — same tick-loop reverse traversal, same gradient
// accumulation pattern — but consumes a `CtmCacheResident` instead
// of a host `CtmCache`.
//
// The math itself stays host-side for THIS slice. Resident BACKWARD
// dispatch (MIOpen *Backward symbols routed through the new
// `Op::*BackwardResident` variants) is plumbed end-to-end at the
// device layer (Part 1) but the inner-CTM call sites here continue
// to use the host-slice helpers (`crate::train::linear_backward`,
// `sync_backward`, etc.) that are already battle-tested. Lifting
// individual sub-graphs (LN backward, softmax backward) to resident
// dispatch is mechanical from here — every `*BackwardResident` op
// is callable, the descriptors match the forward-side dispatch.
//
// Synapse U-Net BACKWARD GAP — DOCUMENTED REGRESSION.
//
// `SynapseUNetResident::forward` is monolithic (no per-block
// intermediate cache exposed). Of the two repair paths:
//   (a) Extend `SynapseUNetResident::forward_cached` to expose
//       per-block buffers — significant resident-dispatch refactor;
//       deferred.
//   (b) Promote `train::unet_forward_cached` / `train::unet_backward`
//       (and `UNetCache`) to `pub(crate)`, then call them from here
//       on the saved `tc.pre_syn`, accumulating into
//       `grads.unet` — this is the path taken.
//
// Path B trades one extra host-side U-Net forward per tick for
// minimal-surface code and the existing battle-tested host backward
// math. The cross-tick BPTT chain (`d_pre_syn` propagation) was
// already correct; this fix adds full weight/bias/gamma/beta
// gradient accumulation into `CtmGradients.unet`.
//
// Path A remains a future option if the per-tick host U-Net forward
// becomes a measurable bottleneck. The contract this function
// exposes (`grads.unet` populated) does not depend on which path
// computes the cache.

/// Result of `ctm_backward_resident` — gradients plus the
/// cross-tick observable propagation needed by the outer brain
/// backward.
#[cfg(feature = "rocm")]
pub struct RegionBackwardResultResident {
    /// Per-region weight gradients (matches host `RegionBackwardResult.grads`).
    /// All fields including `grads.unet` are populated — the U-Net
    /// regression documented in earlier slices is FIXED via Path B
    /// (see module-level doc).
    pub grads: crate::train::CtmGradients,
    /// Gradient w.r.t. observation tokens fed in (per-token, flat).
    /// Length `cache.n_tokens * raw_dim`.
    pub d_observation: Vec<f32>,
}

/// Inner-CTM backward over a `CtmCacheResident`.
///
/// Mirrors `crate::train::backward_from_activated` structurally;
/// see module-level doc above for the U-Net regression.
///
/// Returns `(grads, d_observation)` matching
/// `crate::train::backward_from_activated`'s return shape minus the
/// regressed U-Net subgraph.
#[cfg(feature = "rocm")]
pub fn ctm_backward_resident(
    w: &CtmWeights,
    cache: &CtmCacheResident,
    d_external: &[f32],
) -> RegionBackwardResultResident {
    let cfg = &w.config;
    let d = cfg.d_model;
    let d_in = cache.d_input;
    let m = cfg.memory_length;
    let n_tokens = cache.n_tokens;

    let mut grads = crate::train::CtmGradients::zeros(w);

    // d_activated propagates across ticks (the cross-tick BPTT
    // backbone). Initialised with the external gradient at the
    // last tick — same convention as host.
    let mut d_activated: Vec<f32> = d_external.iter().copied()
        .chain(std::iter::repeat(0.0))
        .take(d)
        .collect();

    // Accumulator for d_kv across ticks (each tick attends over the
    // same KV stream; gradient adds across ticks).
    let mut d_kv_accumulated = vec![0.0f32; n_tokens * d_in];

    // Hot-loop scratch — q_proj.in_dim is constant.
    let mut d_sync_action_scratch = vec![0.0f32; w.q_proj.in_dim];

    let ticks_used = cache.tick_caches.len();
    for tick in (0..ticks_used).rev() {
        let tc = &cache.tick_caches[tick];

        // ── Sync-out backward (host scalar, identical to host path). ──
        let d_from_sync = sync_backward_host(
            &d_activated, &tc.activated_post, &tc.beta_out,
            &w.sync_out_left, &w.sync_out_right, d,
        );

        // External gradient is injected at the LAST tick only (matches
        // host `backward_from_activated`). Earlier ticks see only the
        // gradient routed back through the next tick's synapse input.
        let mut d_act_total = vec![0.0f32; d];
        for i in 0..d {
            d_act_total[i] = d_from_sync[i];
            if tick == ticks_used - 1 {
                d_act_total[i] += d_activated[i];
            }
        }

        // ── NLM backward (host helpers — `train::nlm_backward` is
        // private; we replicate the call signature using public
        // SuperLinear API). ──
        let d_trace = nlm_backward_resident(
            &d_act_total,
            tc,
            &w.nlm_stage1, w.nlm_stage2.as_ref(),
            &mut grads.nlm_s1_w, &mut grads.nlm_s1_b,
            &mut grads.nlm_s2_w, &mut grads.nlm_s2_b,
        );

        // d_pre_act is the last column of the trace (the freshest
        // entry that came from the synapse).
        let mut d_pre_act = vec![0.0f32; d];
        for n in 0..d { d_pre_act[n] = d_trace[n * m + m - 1]; }

        // ── Synapse U-Net backward (Path B — host re-cache). ──
        // Reconstruct per-block intermediates host-side via the now
        // pub(crate) `train::unet_forward_cached`, then run
        // `train::unet_backward` directly into `grads.unet`. Weight,
        // bias, gamma, beta gradients all accumulate correctly; the
        // returned `d_pre_syn` keeps the cross-tick BPTT chain intact.
        // (See module-level GAP doc for why Path B vs Path A.)
        let (_unet_out, unet_cache) = crate::train::unet_forward_cached(
            &w.synapse, &tc.pre_syn,
        );
        let d_pre_syn = crate::train::unet_backward(
            &w.synapse, &d_pre_act, &unet_cache, &mut grads.unet,
        );

        let d_attn_out = &d_pre_syn[..d_in];
        let d_act_from_syn = &d_pre_syn[d_in..];

        // ── MHA backward (host helpers — `train::mha_backward` is
        // private; we replicate inline). ──
        let (d_q, d_kv_tokens) = mha_backward_resident(
            d_attn_out, tc, n_tokens, d_in, cfg.heads,
            &w.mha_in_proj, &w.mha_out_proj,
            &mut grads.mha_in_w, &mut grads.mha_in_b,
            &mut grads.mha_out_w, &mut grads.mha_out_b,
        );

        // Accumulate d_kv across ticks.
        for (t, d_tok) in d_kv_tokens.iter().enumerate() {
            let offset = t * d_in;
            for j in 0..d_in.min(d_tok.len()) {
                if offset + j < d_kv_accumulated.len() {
                    d_kv_accumulated[offset + j] += d_tok[j];
                }
            }
        }

        // ── q_proj backward (`Linear` backward via train.rs helper).
        // Reconstruct the LinearCache from the resident-cache field
        // (we saved q_proj_out's INPUT = sync_action). ──
        let q_lin_cache = crate::train::LinearCache { input: tc.sync_action.clone() };
        crate::train::linear_backward(
            &w.q_proj, &d_q, &q_lin_cache,
            &mut grads.q_proj_w, &mut grads.q_proj_b,
            &mut d_sync_action_scratch,
        );
        let d_sync_action: &[f32] = &d_sync_action_scratch;

        // ── Sync-action backward — feeds back into d_activated for
        // the previous tick. ──
        let d_from_sync_action = sync_backward_host(
            d_sync_action, &tc.activated_prev, &tc.beta_action,
            &w.sync_action_left, &w.sync_action_right, d,
        );

        // Combined gradient for the previous tick's d_activated.
        d_activated = vec![0.0f32; d];
        for i in 0..d {
            d_activated[i] = d_act_from_syn[i] + d_from_sync_action[i];
        }
    }

    for i in 0..d { grads.d_start_activated[i] += d_activated[i]; }

    // ── kv_proj backward (host scalar, identical to
    // `backward_from_activated`). ──
    let raw_dim = w.kv_proj.in_dim;
    let mut d_observation = vec![0.0f32; n_tokens * raw_dim];
    for t in 0..n_tokens {
        let d_kv_t = &d_kv_accumulated[t * d_in..(t + 1).min(n_tokens) * d_in];
        for j in 0..raw_dim {
            for i in 0..d_in.min(d_kv_t.len()) {
                d_observation[t * raw_dim + j] +=
                    d_kv_t[i] * w.kv_proj.weight[i * raw_dim + j];
            }
        }
    }

    RegionBackwardResultResident { grads, d_observation }
}

// ─── Local backward helpers used by ctm_backward_resident ───────
//
// These mirror the private `train::*_backward` functions. We
// duplicate (rather than reach in) because the slice constraint
// is "do not modify train.rs" — without `pub(crate)` exposure
// these helpers are not reachable. Functions are minimal copies;
// any divergence from train.rs is a bug.

/// Mirrors `train::sync_backward`.
#[cfg(feature = "rocm")]
fn sync_backward_host(
    d_sync: &[f32], activated: &[f32], beta: &[f32],
    left: &[usize], right: &[usize], d_model: usize,
) -> Vec<f32> {
    use modgrad_device::backend::{ops, SyncBackwardScatterArgs};
    let n_pairs = left.len();
    let left_u32: Vec<u32> = left.iter().map(|&x| x as u32).collect();
    let right_u32: Vec<u32> = right.iter().map(|&x| x as u32).collect();
    let mut d_act = vec![0.0f32; d_model];
    ops::sync_backward_scatter(SyncBackwardScatterArgs {
        d_sync, pairs_left: &left_u32, pairs_right: &right_u32,
        activated, beta, d_act: &mut d_act,
        n_pairs, d_model,
    }).expect("sync_backward_scatter dispatch");
    d_act
}

/// Mirrors `train::nlm_backward` — reconstructs the NLM cache from
/// the resident tick cache's saved activations and dispatches the
/// per-stage backwards through `super_linear_bwd_*` ops.
///
/// The host path's `NlmCache` saves `s1_cache.input` (= trace),
/// `glu1_cache.x` (= s1_out raw), `s2_cache.input` (= s1_glu),
/// `glu2_cache.x` (= s2_out raw). Resident cache stores all four
/// under `tc.trace`, `tc.nlm_s1_out`, `tc.nlm_s1_glu`, `tc.nlm_s2_out`.
#[cfg(feature = "rocm")]
fn nlm_backward_resident(
    d_out: &[f32],
    tc: &CtmTickCacheResident,
    stage1: &modgrad_compute::neuron::SuperLinear,
    stage2: Option<&modgrad_compute::neuron::SuperLinear>,
    d_s1_w: &mut [f32], d_s1_b: &mut [f32],
    d_s2_w: &mut Option<Vec<f32>>, d_s2_b: &mut Option<Vec<f32>>,
) -> Vec<f32> {
    use modgrad_device::backend::ops;
    let d_model = stage1.n_neurons;

    // Reverse the (optional) stage2 first.
    let d_glu1 = if let (Some(s2), Some(s2_out_raw)) = (stage2, tc.nlm_s2_out.as_ref()) {
        // d_out has shape [d_model * (s2.out_per/2)].
        // GLU2 backward: invert per-neuron GLU on s2_out_raw.
        let s2_half = s2.out_per / 2;
        let mut d_s2_in = vec![0.0f32; d_model * s2.out_per];
        ops::per_neuron_glu_bwd(d_out, s2_out_raw, &mut d_s2_in, d_model, s2_half)
            .expect("per_neuron_glu_bwd dispatch (s2)");
        // SuperLinear stage2 backward — input was tc.nlm_s1_glu.
        let dw_s2 = d_s2_w.as_mut().expect("d_s2_w present when stage2 present");
        let db_s2 = d_s2_b.as_mut().expect("d_s2_b present when stage2 present");
        let n = s2.n_neurons;
        let ip = s2.in_per;
        let op = s2.out_per;
        // d_bias accumulate.
        for i in 0..n * op { db_s2[i] += d_s2_in[i]; }
        let mut d_s1_glu = vec![0.0f32; n * ip];
        ops::super_linear_bwd_dw(&d_s2_in, &tc.nlm_s1_glu, dw_s2, n, ip, op)
            .expect("super_linear_bwd_dw (s2)");
        ops::super_linear_bwd_dx(&d_s2_in, &s2.weights, &mut d_s1_glu, n, ip, op)
            .expect("super_linear_bwd_dx (s2)");
        // GLU1 backward: invert per-neuron GLU on tc.nlm_s1_out.
        d_s1_glu
    } else {
        // No stage2 — d_out IS d_glu1 (post-stage1 GLU output is
        // activated_post directly).
        d_out.to_vec()
    };

    // Stage 1 GLU backward + SuperLinear backward.
    let s1_half = stage1.out_per / 2;
    let mut d_s1_in = vec![0.0f32; d_model * stage1.out_per];
    ops::per_neuron_glu_bwd(&d_glu1, &tc.nlm_s1_out, &mut d_s1_in, d_model, s1_half)
        .expect("per_neuron_glu_bwd (s1)");

    let n = stage1.n_neurons;
    let ip = stage1.in_per;
    let op = stage1.out_per;
    for i in 0..n * op { d_s1_b[i] += d_s1_in[i]; }
    let mut d_trace = vec![0.0f32; n * ip];
    ops::super_linear_bwd_dw(&d_s1_in, &tc.trace, d_s1_w, n, ip, op)
        .expect("super_linear_bwd_dw (s1)");
    ops::super_linear_bwd_dx(&d_s1_in, &stage1.weights, &mut d_trace, n, ip, op)
        .expect("super_linear_bwd_dx (s1)");
    d_trace
}

/// Mirrors `train::mha_backward`.
///
/// MHA forward in `ctm_forward_resident` projects Q via `mha_in_q`
/// (a sliced row sub-view of the full `mha_in_proj` weight) and
/// projects K/V via `mha_in_k`/`mha_in_v` (other sliced row
/// sub-views). The backward writes weight gradients into
/// `d_in_w`/`d_in_b` at the row offsets matching each sub-view —
/// `0..d_input` for Q, `d_input..2*d_input` for K, `2*d_input..3*d_input`
/// for V.
#[cfg(feature = "rocm")]
fn mha_backward_resident(
    d_out: &[f32], tc: &CtmTickCacheResident,
    n_tokens: usize, d_input: usize, n_heads: usize,
    in_proj: &Linear, out_proj: &Linear,
    d_in_w: &mut [f32], d_in_b: &mut [f32],
    d_out_w: &mut [f32], d_out_b: &mut [f32],
) -> (Vec<f32>, Vec<Vec<f32>>) {
    use modgrad_device::backend::ops;
    let d_head = d_input / n_heads;
    let scale = 1.0 / (d_head as f32).sqrt();

    // out_proj backward — input was tc.concat_heads.
    let out_lin_cache = crate::train::LinearCache { input: tc.concat_heads.clone() };
    let mut d_concat = vec![0.0f32; out_proj.in_dim];
    crate::train::linear_backward(out_proj, d_out, &out_lin_cache,
        d_out_w, d_out_b, &mut d_concat);

    // Per-head backward through softmax.
    let mut d_q_full = vec![0.0f32; d_input];
    let mut d_k_all = vec![0.0f32; n_tokens * d_input];
    let mut d_v_all = vec![0.0f32; n_tokens * d_input];

    for h in 0..n_heads {
        let d_head_out = &d_concat[h * d_head..(h + 1) * d_head];
        let weights = &tc.attn_weights[h * n_tokens..(h + 1) * n_tokens];

        let mut d_weights = vec![0.0f32; n_tokens];
        for t in 0..n_tokens {
            let v_h = &tc.v_all[
                t * d_input + h * d_head..t * d_input + (h + 1) * d_head];
            for j in 0..d_head {
                d_weights[t] += d_head_out[j] * v_h[j];
                d_v_all[t * d_input + h * d_head + j] += weights[t] * d_head_out[j];
            }
        }

        // Softmax backward: d_scores[t] = w[t] * (d_w[t] - sum(w * d_w)) * scale
        let dot_wd: f32 = weights.iter().zip(&d_weights).map(|(&w, &d)| w * d).sum();
        let d_scores: Vec<f32> = (0..n_tokens)
            .map(|t| weights[t] * (d_weights[t] - dot_wd) * scale)
            .collect();

        // d_q_h[j] += sum_t(d_scores[t] * k_h[t][j])
        // d_k_h[t][j] += d_scores[t] * q_h[j]
        let q_h = &tc.q_full[h * d_head..(h + 1) * d_head];
        for t in 0..n_tokens {
            let k_start = t * d_input + h * d_head;
            let k_h = &tc.k_all[k_start..k_start + d_head];
            for j in 0..d_head {
                d_q_full[h * d_head + j] += d_scores[t] * k_h[j];
                d_k_all[k_start + j] += d_scores[t] * q_h[j];
            }
        }
    }

    // in_proj backward — three slice-row sub-paths feeding the same
    // big weight tensor `mha_in_proj`. Q sees rows 0..d_input over
    // `tc.q_proj_out` (= q_in). K sees rows d_input..2*d_input over
    // `tc.kv_tokens[t]`. V sees rows 2*d_input..3*d_input over
    // `tc.kv_tokens[t]`.
    //
    // d_bias for the rows (ri ∈ [start, end)): d_in_b[ri] += d_out[ri-start]
    // d_weight at row ri: d_in_w[ri * in_dim..(ri+1) * in_dim] += d_out[ri-start] * x
    // d_input (= d for the corresponding x) accumulates from W^T·d_out
    // across the matching slice.

    let in_dim = in_proj.in_dim;

    // Q rows [0, d_input). x = tc.q_proj_out.
    let q_x = &tc.q_proj_out;
    {
        // d_bias slice
        for r in 0..d_input { d_in_b[r] += d_q_full[r]; }
        // d_weight slice (rows 0..d_input)
        let w_slice = &mut d_in_w[..d_input * in_dim];
        ops::outer_product_acc(&d_q_full, q_x, w_slice, d_input, in_dim)
            .expect("outer_product_acc (Q)");
    }
    let mut d_q_in = vec![0.0f32; in_dim];
    {
        let wt_slice = &in_proj.weight[..d_input * in_dim];
        ops::matvec_t(&d_q_full, wt_slice, &mut d_q_in, d_input, in_dim)
            .expect("matvec_t (Q)");
    }

    // K rows [d_input, 2*d_input).  V rows [2*d_input, 3*d_input).
    // The host path treats each (token, K) as one matvec into the
    // K slice, accumulating d_in_w over the K sub-view across all
    // tokens. Same for V.  We collect d_kv_tokens[t] = d_k_in_t +
    // d_v_in_t (downstream gradient w.r.t. KV input).
    let mut d_kv_tokens: Vec<Vec<f32>> = Vec::with_capacity(n_tokens);
    let k_offset = d_input;
    let v_offset = 2 * d_input;
    for t in 0..n_tokens {
        let kv_x = &tc.kv_tokens[t];
        let d_k_t = &d_k_all[t * d_input..(t + 1) * d_input];
        let d_v_t = &d_v_all[t * d_input..(t + 1) * d_input];

        // K slice
        for r in 0..d_input { d_in_b[k_offset + r] += d_k_t[r]; }
        let w_slice = &mut d_in_w[k_offset * in_dim..(k_offset + d_input) * in_dim];
        ops::outer_product_acc(d_k_t, kv_x, w_slice, d_input, in_dim)
            .expect("outer_product_acc (K)");

        // V slice
        for r in 0..d_input { d_in_b[v_offset + r] += d_v_t[r]; }
        let w_slice = &mut d_in_w[v_offset * in_dim..(v_offset + d_input) * in_dim];
        ops::outer_product_acc(d_v_t, kv_x, w_slice, d_input, in_dim)
            .expect("outer_product_acc (V)");

        // d_kv_t = (W_K^T @ d_k_t) + (W_V^T @ d_v_t)
        let mut d_kv_t = vec![0.0f32; in_dim];
        let mut d_from_k = vec![0.0f32; in_dim];
        let mut d_from_v = vec![0.0f32; in_dim];
        ops::matvec_t(d_k_t, &in_proj.weight[k_offset * in_dim..(k_offset + d_input) * in_dim],
            &mut d_from_k, d_input, in_dim)
            .expect("matvec_t (K)");
        ops::matvec_t(d_v_t, &in_proj.weight[v_offset * in_dim..(v_offset + d_input) * in_dim],
            &mut d_from_v, d_input, in_dim)
            .expect("matvec_t (V)");
        for j in 0..in_dim { d_kv_t[j] = d_from_k[j] + d_from_v[j]; }
        d_kv_tokens.push(d_kv_t);
    }

    (d_q_in, d_kv_tokens)
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
        let (resident_out, resident_cache) = ctm_forward_resident(
            &w, &cache, &mut resident_state, &batch, &kv_dev, n_tokens,
        ).expect("ctm_forward_resident");
        batch.flush().expect("flush");

        // Forward correctness still the contract; the cache is a
        // bonus payload. PART A check: tick_caches.len() must equal
        // ticks_used (so backward sees one cache per emitted prediction).
        assert_eq!(resident_cache.tick_caches.len(), resident_out.ticks_used,
            "CtmCacheResident.tick_caches.len() = {} but ticks_used = {} — \
             cache push cadence drifted from predictions push cadence",
            resident_cache.tick_caches.len(), resident_out.ticks_used);

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
