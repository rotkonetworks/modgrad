//! `LmTrainer` — the entry point our user actually invokes to train a
//! foundation language model. Wraps a residency-aware `LanguageModel`
//! (typically `GptModelResident`) with the full training loop:
//!
//!   load model → forward → cross-entropy loss → per-position
//!   forward_for_backward + backward → AdamW step on host masters →
//!   re-upload to resident weight buffers → repeat.
//!
//! ## Slice state — backward chain wired
//!
//! [`LanguageModel`] now exposes `forward_for_backward_position` and
//! `backward_position`, which `GptModelResident` routes to its
//! inherent `forward_for_backward` / `backward` methods.
//!
//! `train_step` drives them in two passes:
//!
//!   1. **Forward + loss.** `forward_logits` over `tokens[..mbs]`
//!      produces logits; `cross_entropy(logits, targets, vocab)` gives
//!      `(loss, dL/dlogits)` (mbs × vocab).
//!   2. **Per-position backward.** For each `pos` in `0..mbs`, run
//!      `forward_for_backward_position(pos)` (so the resident state's
//!      `block_scratches` carry that position's activations), then
//!      `backward_position(pos, dL/dlogits[pos])`. The state's
//!      per-`Linear` grad buffers are *overwritten* per call, so we
//!      download to host accumulators after each backward and sum
//!      across positions.
//!   3. **Resident AdamW.** Each parameter's `weight_dev` is updated
//!      in place via `Op::AdamWResident` — a custom hipcc kernel that
//!      mutates `w`, `m`, `v` on device. No host master, no D2H/H2D
//!      round-trip on the weight. The grad accumulator still lives on
//!      the host (one H2D per param per step into a per-param resident
//!      scratch); making it resident is a follow-up slice.
//!
//! ### Embedding-grad sparsity caveat
//!
//! [`GptModelResident::backward`] writes `d_embed[token_id, :] =
//! d_hidden` (overwrite) and leaves all other rows untouched. With no
//! public hook to zero `d_embed` between calls, the trainer reads only
//! the row matching this position's `token_id` and accumulates into
//! the matching offset on the host. The full `[vocab × model_dim]`
//! buffer is never trusted — that path would double-count earlier
//! positions whose tokens overlap.
//!
//! ## Why per-`Linear` AdamW state, not a monolithic optimizer
//!
//! `RegionalAdamW` (in `modgrad-ctm`) is shaped for a fixed connectome
//! — one `AdamWBuf` per named slot (embed, conn_w, output_proj, …).
//! A transformer has a regular structure: every block has the same
//! six matrices (wq/wk/wv/wo/gate/up/down/lm_head). The natural map
//! is one `AdamWBuf` per `Linear`, indexed by a stable string key like
//! `"block.{li}.wq"`. That avoids hard-coding a transformer-specific
//! schema into the optimizer and lets a future LoRA / MoE adapter
//! plug in extra `Linear`s with their own AdamW state without
//! changing this module.

#[cfg(feature = "rocm")]
use std::collections::HashMap;

#[cfg(feature = "rocm")]
use modgrad_compute::backend::{GpuVec, ResidencyError};
#[cfg(feature = "rocm")]
use modgrad_device::backend::HipBatch;
#[cfg(feature = "rocm")]
use modgrad_transformer::loss::cross_entropy;
#[cfg(feature = "rocm")]
use modgrad_transformer::GptModelResident;
#[cfg(feature = "rocm")]
use modgrad_transformer::resident::GptBackwardState;

#[cfg(feature = "rocm")]
use crate::language_model::LanguageModel;

// ─── Config ───────────────────────────────────────────────────

/// Hyperparameters for the LM trainer. Defaults mirror standard GPT
/// training recipes (LR scaled to a small base; weight decay of 0.01;
/// gradient clipping at 1.0).
#[cfg(feature = "rocm")]
#[derive(Debug, Clone)]
pub struct LmTrainerConfig {
    /// AdamW learning rate. Default: 3e-4.
    pub lr: f32,
    /// First moment decay. Default: 0.9.
    pub beta1: f32,
    /// Second moment decay. Default: 0.999.
    pub beta2: f32,
    /// AdamW numerical-stability epsilon. Default: 1e-8.
    pub eps: f32,
    /// Decoupled weight decay. Default: 0.01.
    pub weight_decay: f32,
    /// Global gradient norm clip (0.0 ⇒ disabled). Default: 1.0.
    pub grad_clip: f32,
    /// Tokens per micro-batch. Trainer asserts `tokens.len() ==
    /// micro_batch_size + 1` so the next-token-prediction targets
    /// have a corresponding source token.
    pub micro_batch_size: usize,
    /// Sequence length the model expects per call. Asserted
    /// against the model's max_seq_len at construction.
    pub seq_len: usize,
}

#[cfg(feature = "rocm")]
impl Default for LmTrainerConfig {
    fn default() -> Self {
        Self {
            lr: 3e-4,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            grad_clip: 1.0,
            micro_batch_size: 16,
            seq_len: 16,
        }
    }
}

// ─── AdamW per-parameter state (resident) ────────────────────

/// Per-`Linear` AdamW state — moments + grad scratch live on device.
///
/// The fp32 master weight does NOT live here; for `LinearResident`
/// (pure-fp32, the only `LanguageModel` impl wired today) the device
/// `weight_dev` IS the master, so AdamW updates it in place via
/// `Op::AdamWResident`. The grad scratch `g_dev` is owned by this
/// struct so we don't reallocate per training step.
///
/// Why a sibling type instead of using `modgrad_transformer::optim::adamw::AdamW`:
/// the transformer's `AdamW` carries lr/beta/etc as fields, expecting
/// one optimizer per parameter group — that bloats the trainer's
/// `HashMap` (every entry duplicates the same hyperparameters). Here
/// we keep only the per-parameter *state* (m_dev, v_dev, g_dev, t) and
/// read the hyperparameters from the trainer's `LmTrainerConfig` once
/// per step.
///
/// **bf16 master note.** If a future `LanguageModel` wraps
/// `LinearResidentBf16` (fp32 host master + bf16 device weight), the
/// resident path here is wrong: `weight_dev` is bf16, not fp32, so the
/// kernel would corrupt it. That model would need a separate
/// `AdamWBufBf16` carrying a resident *fp32* master alongside the bf16
/// `weight_dev`, plus a `sync_from_master` post-step. Today's
/// `GptModelResident` is pure-fp32, so this complication does not
/// apply.
#[cfg(feature = "rocm")]
pub struct AdamWBuf {
    /// First moment estimates — fp32, length `n`, resident on device.
    pub m_dev: GpuVec,
    /// Second moment estimates — fp32, length `n`, resident on device.
    pub v_dev: GpuVec,
    /// Per-step grad scratch — fp32, length `n`, resident on device.
    /// Owned here so the trainer can upload the host-accumulated grad
    /// into a stable buffer once per step instead of reallocating.
    pub g_dev: GpuVec,
    /// Element count (for the resident dispatch's `n` parameter).
    pub n: usize,
    /// Step counter — bumped before each AdamW update so bias
    /// correction starts at t=1 (not 0, which would NaN the first
    /// step's `bc1`/`bc2`).
    pub t: u64,
}

#[cfg(feature = "rocm")]
impl AdamWBuf {
    /// Allocate moment + grad buffers for a parameter of `n`
    /// elements, resident on device. m and v are zero-initialised;
    /// g is left uninitialised (callers always upload before
    /// dispatch); t starts at zero.
    pub fn zeros(n: usize) -> Result<Self, ResidencyError> {
        let mut m_dev = GpuVec::try_hip(n)?;
        let mut v_dev = GpuVec::try_hip(n)?;
        let g_dev = GpuVec::try_hip(n)?;
        // Zero m and v via a single H2D — needed because hipMalloc
        // doesn't zero, and AdamW reads m[i]/v[i] before the first
        // write. g is overwritten before each read so no init needed.
        let zeros = vec![0.0f32; n];
        m_dev.copy_from(&zeros);
        v_dev.copy_from(&zeros);
        Ok(Self { m_dev, v_dev, g_dev, n, t: 0 })
    }

    /// Apply one AdamW update step against a device weight buffer.
    /// Pulls grads from `host_grad` (the trainer's per-step accumulator),
    /// uploads them to `g_dev`, and dispatches `Op::AdamWResident` so
    /// `weight_dev` / `m_dev` / `v_dev` all update on device with no
    /// further D2H/H2D round-trip.
    ///
    /// `weight_dev` is the model's own resident weight — the dispatch
    /// mutates it in place, which is exactly the residency invariant
    /// we want (the model's `weight_dev` IS the AdamW master).
    pub fn step_resident(
        &mut self,
        weight_dev: &modgrad_device::backend::HipBuffer,
        host_grad: &[f32],
        cfg: &LmTrainerConfig,
    ) -> Result<(), ResidencyError> {
        debug_assert_eq!(host_grad.len(), self.n,
            "AdamWBuf::step_resident: grad len {} != alloc {}", host_grad.len(), self.n);
        // Upload accumulated host grad into the resident scratch.
        self.g_dev.copy_from(host_grad);
        self.t += 1;
        let bc1 = 1.0 - cfg.beta1.powi(self.t as i32);
        let bc2 = 1.0 - cfg.beta2.powi(self.t as i32);
        let m_buf = match &mut self.m_dev {
            GpuVec::Hip(b) => b,
            other => return Err(ResidencyError::WrongVariant {
                expected: "Hip", got: other.variant_name(),
            }),
        };
        let v_buf = match &mut self.v_dev {
            GpuVec::Hip(b) => b,
            other => return Err(ResidencyError::WrongVariant {
                expected: "Hip", got: other.variant_name(),
            }),
        };
        let g_buf = match &self.g_dev {
            GpuVec::Hip(b) => b,
            other => return Err(ResidencyError::WrongVariant {
                expected: "Hip", got: other.variant_name(),
            }),
        };
        unsafe {
            modgrad_device::backend::ops::adamw_resident(
                weight_dev.device_ptr() as *mut f32,
                g_buf.device_ptr() as *const f32,
                m_buf.device_ptr() as *mut f32,
                v_buf.device_ptr() as *mut f32,
                self.n,
                cfg.lr, cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay,
                1.0 / bc1, 1.0 / bc2,
            )?;
        }
        Ok(())
    }
}

// ─── Trainer ──────────────────────────────────────────────────

/// Top-level training loop wrapping a `LanguageModel`. Owns the
/// per-parameter AdamW state, the loss history, and a working
/// `KvCacheResident` reused across steps.
///
/// Generic over the model so the same trainer drives `GptModelResident`
/// today, and `LlamaResident` / `MoeResident` / future architectures
/// when they exist — as long as they implement `LanguageModel`.
#[cfg(feature = "rocm")]
pub struct LmTrainer<M: LanguageModel> {
    model: M,
    /// Per-parameter AdamW state (resident moments + grad scratch),
    /// indexed by stable string key (e.g. `"block.0.wq"`, `"lm_head"`).
    /// Lazily populated on the first `train_step` call once we know
    /// the model shape.
    ///
    /// No parallel host-master HashMap: `GptModelResident::LinearResident`
    /// is pure-fp32 device-resident, so its `weight_dev` IS the AdamW
    /// master. `Op::AdamWResident` updates `weight_dev` in place via a
    /// custom hipcc kernel — eliminating the per-param D2H/host-AdamW/H2D
    /// round-trip the prior host-AdamW path forced (one residency leak
    /// per parameter per step, which dominated wall-clock at scale).
    /// See [`AdamWBuf::step_resident`] for the dispatch.
    ///
    /// Param keys (`block.{li}.{wq,wk,wv,wo,gate,up,down}`, `embed`,
    /// `lm_head`) match the modgrad-ctm convention so a future
    /// serializer can checkpoint either trainer with the same schema.
    adamw: HashMap<String, AdamWBuf>,
    /// Resident KV cache reused across `train_step` calls. Allocated
    /// at construction with capacity = max(seq_len of any future
    /// micro_batch) — for now equal to `config.seq_len`. Reset
    /// before every step.
    kv_cache: modgrad_transformer::KvCacheResident,
    /// Pre-allocated backward state. `None` until the first
    /// `train_step` calls `model.alloc_backward_state(&kv_cache)`.
    /// Reused across subsequent steps to avoid re-allocating GPU
    /// scratch every batch.
    bwd_state: Option<M::BackwardState>,
    /// Loss values from completed `train_step` calls, in order.
    loss_history: Vec<f32>,
    config: LmTrainerConfig,
}

#[cfg(feature = "rocm")]
impl<M: LanguageModel> LmTrainer<M> {
    /// Build a trainer over `model`. Allocates the resident KV cache
    /// matching the model's `n_layers()` × `seq_len`. The KV cache
    /// captures KV-head count / head_dim from the `LanguageModel`
    /// trait extension — but the trait doesn't expose them, so we
    /// take them as explicit args. Standard GPT-class wrapping is
    /// `n_kv_heads = num_heads` (no GQA); if the caller is wrapping
    /// a GQA model they pass the actual KV head count.
    pub fn new(
        model: M,
        n_kv_heads: usize,
        head_dim: usize,
        config: LmTrainerConfig,
    ) -> Result<Self, ResidencyError> {
        let n_layers = model.n_layers();
        let d_model = model.d_model();
        let kv_cache = modgrad_transformer::KvCacheResident::new(
            n_layers, n_kv_heads, head_dim, config.seq_len, d_model,
        )?;
        Ok(Self {
            model,
            adamw: HashMap::new(),
            kv_cache,
            bwd_state: None,
            loss_history: Vec::new(),
            config,
        })
    }

    /// Read-only access to the wrapped model. Useful for inference
    /// after training, or for poking at the per-block residency state.
    pub fn model(&self) -> &M { &self.model }

    /// Mutable access to the wrapped model. Used by callers that own
    /// model serialization or want to swap out KV-cache settings
    /// mid-run.
    pub fn model_mut(&mut self) -> &mut M { &mut self.model }

    /// Loss values from completed `train_step` calls, in order.
    pub fn loss_history(&self) -> &[f32] { &self.loss_history }

    /// AdamW state — exposed for tests + tools that want to inspect
    /// optimizer momentum, not for the hot path.
    pub fn adamw_state(&self) -> &HashMap<String, AdamWBuf> { &self.adamw }

    /// Configuration view.
    pub fn config(&self) -> &LmTrainerConfig { &self.config }

    /// Apply a global gradient norm clip to a flat list of grad
    /// slices, in place. Standard clip-by-global-norm.
    ///
    /// Exposed (`pub`) so the future `step_adamw_and_resync`
    /// implementation has direct access; tests can also call it
    /// against synthetic grads to validate the math.
    pub fn clip_grads(grads: &mut [&mut [f32]], max_norm: f32) {
        if max_norm <= 0.0 {
            return;
        }
        let sq_sum: f64 = grads.iter()
            .flat_map(|g| g.iter())
            .map(|&v| (v as f64) * (v as f64))
            .sum();
        let norm = sq_sum.sqrt() as f32;
        if norm.is_finite() && norm > max_norm && norm > 0.0 {
            let scale = max_norm / norm;
            for g in grads.iter_mut() {
                for v in g.iter_mut() {
                    *v *= scale;
                }
            }
        }
    }
}

// ─── Concrete training wiring for GptModelResident ────────────
//
// The full backward + AdamW chain is concrete to `GptModelResident`
// today: only that model exposes per-`Linear` weight buffers (`q_proj`,
// `lm_head`, etc.) by stable layer name. A future model with a
// different decomposition (LoRA adapters, MoE, …) will get its own
// concrete impl. The generic `LmTrainer<M>` keeps the forward+loss path
// plus shared utilities (`clip_grads`, history); the wiring is
// specialised here so the generic interface stays narrow.

/// Stable parameter keys — one per `Linear` plus the embedding table.
/// Keying as `block.{li}.{slot}` mirrors the modgrad-ctm convention so
/// a downstream serializer can checkpoint either trainer with the same
/// schema.
#[cfg(feature = "rocm")]
fn param_keys(n_layers: usize) -> Vec<String> {
    let mut keys = Vec::with_capacity(2 + n_layers * 7);
    keys.push("embed".to_string());
    for li in 0..n_layers {
        keys.push(format!("block.{li}.wq"));
        keys.push(format!("block.{li}.wk"));
        keys.push(format!("block.{li}.wv"));
        keys.push(format!("block.{li}.wo"));
        keys.push(format!("block.{li}.gate"));
        keys.push(format!("block.{li}.up"));
        keys.push(format!("block.{li}.down"));
    }
    keys.push("lm_head".to_string());
    keys
}

#[cfg(feature = "rocm")]
impl LmTrainer<GptModelResident> {
    /// Run one training step on `tokens` (length `micro_batch_size + 1`
    /// — the last token is the final next-token target only and is
    /// not fed through forward).
    ///
    /// Returns the cross-entropy loss for the step.
    ///
    /// Pipeline:
    ///   1. Reset the KV cache.
    ///   2. Forward pass on `tokens[..mbs]` at positions `0..mbs`
    ///      (existing forward path) → host logits → cross-entropy
    ///      against `tokens[1..=mbs]` (next-token prediction). This
    ///      gives us `(loss, dL/d(logits))`.
    ///   3. Per-position forward-with-cache + backward. We re-do the
    ///      forward via [`GptModelResident::forward_for_backward`]
    ///      (cheap activation-saving variant) then immediately
    ///      [`GptModelResident::backward`] with the per-position
    ///      `d_logits` slice. After each backward, we download every
    ///      `Linear`'s gradient into a host accumulator (sum across
    ///      positions). The double forward is the cost of the
    ///      single-token backward signature; multi-token resident
    ///      backward is a follow-up slice.
    ///   4. Resident AdamW directly on each `Linear`'s `weight_dev`.
    ///      A single `Op::AdamWResident` dispatch per parameter mutates
    ///      the device weight + moment buffers in place — no per-param
    ///      D2H/H2D round-trip on the weight. The grad still travels
    ///      H2D once per param (the per-position grad accumulator
    ///      lives on the host); making the accumulator resident is the
    ///      next slice.
    ///
    /// Errors propagate `ResidencyError` from any GPU op.
    pub fn train_step(
        &mut self,
        batch: &HipBatch,
        tokens: &[i64],
    ) -> Result<f32, ResidencyError> {
        let mbs = self.config.micro_batch_size;
        assert_eq!(
            tokens.len(), mbs + 1,
            "LmTrainer::train_step: tokens.len() must be micro_batch_size + 1 ({} + 1), got {}",
            mbs, tokens.len(),
        );

        let vocab = self.model.vocab_size();
        let n_layers = self.model.n_layers();
        let d_model = self.model.d_model();

        // ─── Stage 0: lazy alloc of resident AdamW state + bwd state.
        // First train_step allocates per-param resident moment + grad
        // scratch buffers (zero-initialised). Subsequent calls find the
        // state already populated and skip allocation.
        //
        // No host-master download: `LinearResident::weight_dev` is the
        // master, and `Op::AdamWResident` mutates it in place each step.
        if self.adamw.is_empty() {
            self.lazy_init_state()?;
        }

        // ─── Stage 1: reset cache. set_seq_len(0) only — every block's
        // forward writes the slot before reading it, no zero needed.
        self.kv_cache.reset();

        // ─── Stage 2: forward (existing path) → loss + d_logits.
        let src_tokens: &[i64] = &tokens[..mbs];
        let positions: Vec<usize> = (0..mbs).collect();
        let mut logits_dev = GpuVec::try_hip(mbs * vocab)?;

        self.model.ensure_resident(batch)?;
        self.model.forward_logits(
            batch, src_tokens, &positions,
            Some(&mut self.kv_cache),
            &mut logits_dev,
        )?;
        batch.flush()?;

        let mut logits_host = vec![0.0f32; mbs * vocab];
        logits_dev.copy_to_host(&mut logits_host);
        let targets: &[i64] = &tokens[1..=mbs];
        let (loss, grad_logits) = cross_entropy(&logits_host, targets, vocab);

        assert!(loss.is_finite(),
            "LmTrainer::train_step: loss is NaN/inf — upstream forward bug");
        self.loss_history.push(loss);

        // ─── Stage 3: per-position forward_for_backward + backward.
        //
        // `block_scratches` in the backward state is overwritten each
        // forward_for_backward call, so we MUST run backward immediately
        // after the matched forward — interleaved per position.
        //
        // Grads in the state buffers are also overwritten per backward
        // (LinearResident::backward is overwriting, not accumulating),
        // so we download to host accumulators after each call.
        self.kv_cache.reset();

        let keys = param_keys(n_layers);
        let mut grad_acc: HashMap<String, Vec<f32>> = HashMap::with_capacity(keys.len());
        for k in &keys {
            let n = self.adamw.get(k)
                .expect("AdamW state populated by lazy_init")
                .n;
            grad_acc.insert(k.clone(), vec![0.0f32; n]);
        }

        let mut d_logits_pos = GpuVec::try_hip(vocab)?;
        let mut row_scratch: Vec<f32> = Vec::with_capacity(vocab);

        // Take bwd_state out so we can hold &mut to it concurrently with
        // &mut self.kv_cache and &mut self.model (the borrow-checker
        // refuses three nested &mut self field refs in one call).
        // Restore it at end of stage so subsequent calls reuse the same
        // GPU scratch.
        let mut bwd = self.bwd_state.take()
            .expect("bwd_state populated by lazy_init_masters_and_state");

        let result: Result<(), ResidencyError> = (|| {
            for pos in 0..mbs {
                // forward-with-cache for this position. `d_logits_pos` is
                // reused — its first `vocab` slots get overwritten by this
                // call; we don't need them (the loss came from Stage 2).
                self.model.forward_for_backward_position(
                    batch, src_tokens[pos], pos,
                    &mut self.kv_cache, &mut bwd,
                    &mut d_logits_pos,
                )?;
                // Now upload this position's d_logits into the same buffer
                // (forward_for_backward wrote logits there; we replace).
                row_scratch.clear();
                row_scratch.extend_from_slice(
                    &grad_logits[pos * vocab..(pos + 1) * vocab],
                );
                d_logits_pos.copy_from(&row_scratch);

                self.model.backward_position(
                    batch, src_tokens[pos], pos,
                    &mut self.kv_cache, &mut bwd,
                    &d_logits_pos,
                )?;
                batch.flush()?;

                // Download per-position grads into host accumulators. Each
                // GpuVec::copy_to_host is one hipMemcpy D2H — small for
                // tiny configs, the per-position cost is in the dispatch
                // count not the bytes.
                Self::accumulate_position_grads(
                    &bwd, n_layers, src_tokens[pos], d_model,
                    &mut grad_acc,
                );
            }
            Ok(())
        })();
        // Always restore bwd_state, even on error, so the next train_step
        // doesn't trip on a `None` and re-allocate from scratch.
        self.bwd_state = Some(bwd);
        result?;

        // ─── Stage 4: optional grad-norm clip + resident AdamW.
        //
        // Resident dispatch eliminates the per-parameter D2H/H2D weight
        // round-trip the prior host-AdamW path forced. The grad still
        // travels host→device once per param per step (single H2D per
        // param), but the weight stays resident across the whole loop.
        // Removing the per-position grad download is the next slice.
        if self.config.grad_clip > 0.0 {
            let mut slices: Vec<&mut [f32]> =
                grad_acc.values_mut().map(|v| v.as_mut_slice()).collect();
            Self::clip_grads(&mut slices, self.config.grad_clip);
        }

        for k in &keys {
            let grad = grad_acc.get(k)
                .expect("grad accumulator populated above");
            // Borrow each AdamWBuf entry separately from the model so
            // we hold a single &mut into adamw and a single &mut into
            // model at the same time. Looking up by string per step is
            // cheap (HashMap<String, _> hash on a short key).
            let weight_dev = Self::weight_dev_for_key(&self.model, k, n_layers);
            let buf = self.adamw.get_mut(k)
                .expect("AdamW state populated by lazy_init");
            buf.step_resident(weight_dev, grad.as_slice(), &self.config)?;
        }
        let _ = d_model;

        Ok(loss)
    }

    /// First-call setup — allocate the resident AdamW state (one
    /// `AdamWBuf` per `Linear` plus the embedding table) and the
    /// backward state. Idempotent guard: caller checks
    /// `adamw.is_empty()`.
    ///
    /// Each `AdamWBuf::zeros(n)` performs three `hipMalloc`s
    /// (`m_dev`, `v_dev`, `g_dev`) and two zero H2Ds (m, v). For a
    /// 2-layer tiny test config that's 14 + 2 = 16 hipMallocs at
    /// startup; for a real foundation model it's ~7 × n_layers + 2,
    /// still a one-time cost amortised over the whole run.
    fn lazy_init_state(&mut self) -> Result<(), ResidencyError> {
        let n_layers = self.model.n_layers();
        let d_model = self.model.d_model();
        let vocab = self.model.vocab_size();

        let mut alloc = |key: &str, n: usize| -> Result<(), ResidencyError> {
            self.adamw.insert(key.to_string(), AdamWBuf::zeros(n)?);
            Ok(())
        };

        // Embedding: [vocab × d_model].
        alloc("embed", vocab * d_model)?;

        // Per-block linears.
        for li in 0..n_layers {
            let block = &self.model.blocks[li];
            let attn = &block.attn;
            let mlp = &block.mlp;
            let mlp_dim = mlp.mlp_dim();
            let mm = attn.model_dim * attn.model_dim;
            let kv_m = attn.kv_dim * attn.model_dim;
            alloc(&format!("block.{li}.wq"), mm)?;
            alloc(&format!("block.{li}.wk"), kv_m)?;
            alloc(&format!("block.{li}.wv"), kv_m)?;
            alloc(&format!("block.{li}.wo"), mm)?;
            alloc(&format!("block.{li}.gate"), mlp_dim * d_model)?;
            alloc(&format!("block.{li}.up"), mlp_dim * d_model)?;
            alloc(&format!("block.{li}.down"), d_model * mlp_dim)?;
        }

        // LM head: [vocab × d_model].
        alloc("lm_head", vocab * d_model)?;

        // Backward state.
        let state = LanguageModel::alloc_backward_state(&self.model, &self.kv_cache)?;
        self.bwd_state = Some(state);

        Ok(())
    }

    /// Look up the model's resident weight buffer for a given param
    /// key. Companion to `param_keys` — the inverse mapping. Used by
    /// the resident AdamW dispatch to mutate `weight_dev` in place.
    fn weight_dev_for_key<'a>(
        model: &'a GptModelResident,
        key: &str,
        n_layers: usize,
    ) -> &'a modgrad_device::backend::HipBuffer {
        if key == "embed" {
            return &model.embed_dev;
        }
        if key == "lm_head" {
            return &model.lm_head.weight_dev;
        }
        let rest = key.strip_prefix("block.").expect("block key shape");
        let mut parts = rest.splitn(2, '.');
        let li: usize = parts.next().unwrap().parse().expect("block index");
        let slot = parts.next().unwrap();
        debug_assert!(li < n_layers, "{key}: block index out of range");
        let block = &model.blocks[li];
        match slot {
            "wq" => &block.attn.q_proj.weight_dev,
            "wk" => &block.attn.k_proj.weight_dev,
            "wv" => &block.attn.v_proj.weight_dev,
            "wo" => &block.attn.o_proj.weight_dev,
            "gate" => &block.mlp.gate.weight_dev,
            "up" => &block.mlp.up.weight_dev,
            "down" => &block.mlp.down.weight_dev,
            other => panic!("LmTrainer::weight_dev_for_key: unknown slot {other}"),
        }
    }

    /// Add the resident-state's per-`Linear` grad buffers (this
    /// position's contribution) into the host accumulators.
    ///
    /// `LinearResident::backward` overwrites — it does not accumulate —
    /// so this download+sum is the only correct place to fold the
    /// position's contribution into the running total.
    ///
    /// **Embedding handling.** [`GptModelResident::backward`] writes
    /// `d_embed[token_id, :] = d_hidden` (overwrite) and leaves all
    /// other rows untouched. The trainer never zeroes the device-side
    /// `d_embed` between calls (no public hook), so reading the full
    /// `[vocab × model_dim]` buffer would double-count earlier
    /// positions' tokens. Instead we only fetch the *row* for this
    /// position's `token_id` and add it to the host accumulator at the
    /// matching offset. Sparse on the host, exact match for the
    /// per-position contract.
    fn accumulate_position_grads(
        state: &GptBackwardState,
        n_layers: usize,
        token_id: i64,
        d_model: usize,
        acc: &mut HashMap<String, Vec<f32>>,
    ) {
        // Embedding: download only the row written by this backward.
        let mut row = vec![0.0f32; d_model];
        // d_embed is `[vocab × model_dim]` row-major, but `GpuVec::copy_to_host`
        // dumps the whole buffer; for the per-row fetch we use the
        // underlying `HipBuffer::copy_to_host` with a temp dst sized to
        // d_model and rely on copy_to_host respecting the dst.len() (it
        // copies min(buffer_bytes, dst.len() * 4)). That copies the
        // first `d_model` floats — the row at index 0. To address an
        // arbitrary row we'd need a pointer-offset hipMemcpy, which is
        // not exposed; so we full-download into a vocab-sized scratch
        // and pull out the row.
        //
        // Cost: `vocab * d_model * 4` bytes per position. For tiny
        // configs (vocab=256, d_model=128) this is 128 KiB per
        // position × 16 positions = 2 MiB per train_step — trivial.
        let mut full = vec![0.0f32; state.d_embed.len()];
        state.d_embed.copy_to_host(&mut full);
        let row_off = token_id as usize * d_model;
        row.copy_from_slice(&full[row_off..row_off + d_model]);
        let acc_embed = acc.get_mut("embed").unwrap();
        for (i, &v) in row.iter().enumerate() {
            acc_embed[row_off + i] += v;
        }

        for li in 0..n_layers {
            Self::add_into(acc.get_mut(&format!("block.{li}.wq")).unwrap(),
                &state.attn_grads[li].dweight_q);
            Self::add_into(acc.get_mut(&format!("block.{li}.wk")).unwrap(),
                &state.attn_grads[li].dweight_k);
            Self::add_into(acc.get_mut(&format!("block.{li}.wv")).unwrap(),
                &state.attn_grads[li].dweight_v);
            Self::add_into(acc.get_mut(&format!("block.{li}.wo")).unwrap(),
                &state.attn_grads[li].dweight_o);
            Self::add_into(acc.get_mut(&format!("block.{li}.gate")).unwrap(),
                &state.mlp_grads[li].dweight_gate);
            Self::add_into(acc.get_mut(&format!("block.{li}.up")).unwrap(),
                &state.mlp_grads[li].dweight_up);
            Self::add_into(acc.get_mut(&format!("block.{li}.down")).unwrap(),
                &state.mlp_grads[li].dweight_down);
        }
        Self::add_into(acc.get_mut("lm_head").unwrap(),
            &state.d_lm_head_weight);
    }

    /// Download `src` into a temp and add element-wise into `dst`.
    fn add_into(dst: &mut [f32], src: &GpuVec) {
        let mut tmp = vec![0.0f32; dst.len()];
        src.copy_to_host(&mut tmp);
        for (a, b) in dst.iter_mut().zip(tmp.iter()) {
            *a += *b;
        }
    }

}

// ─── Tests ────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(feature = "rocm")]
mod tests {
    use super::*;
    use modgrad_device::backend::rocm::ffi::runtime_available;
    use modgrad_transformer::config::{
        GptConfig, MlpActivation, ResidualConfig, SmearConfig,
        ValueEmbedConfig, WindowPattern, Precision,
    };
    use modgrad_transformer::dims::*;
    use modgrad_transformer::tensor::Tensor2;
    use modgrad_transformer::attention::{AttentionWeights, CausalSelfAttention};
    use modgrad_transformer::block::TransformerBlock;
    use modgrad_transformer::mlp::{Mlp, MlpWeights, SwigluMlp, SwigluWeights};
    use modgrad_transformer::model::GptModel;
    use modgrad_transformer::norm::ScaledRmsNorm;
    use modgrad_transformer::position::fixed::FixedPositioning;
    use modgrad_transformer::residual::ResidualLambdas;
    use modgrad_transformer::rope::RotaryEmbedding;
    use modgrad_transformer::smear::{Inference, Smear, SmearWeights, Training};
    use modgrad_transformer::GptModelResident;
    use std::sync::Mutex;

    /// HIP runtime tests share the device with each other; serialize
    /// them so kernel dispatch from concurrent `cargo test` workers
    /// doesn't interleave on the default stream. Same lock pattern as
    /// `modgrad-transformer/src/resident.rs`.
    static HIP_TEST_LOCK: Mutex<()> = Mutex::new(());

    /// Tiny config — 2 layers, d_model=128, 4 heads, vocab=256.
    /// Mirrors `tiny_config` in modgrad-transformer's resident tests
    /// so the test build path is shared.
    fn tiny_config() -> GptConfig {
        let head_dim = 32usize;
        let n_heads = 4usize;
        let model_dim = head_dim * n_heads;
        GptConfig {
            model_dim: ModelDim::new(model_dim),
            num_heads: NumHeads::new(n_heads),
            num_kv_heads: NumKvHeads::new(n_heads),
            head_dim: HeadDim::new(head_dim),
            num_layers: NumLayers::new(2),
            vocab_size: VocabSize::new(256),
            mlp_dim: MlpDim::new(model_dim * 2),
            max_seq_len: SeqLen::new(32),
            rope_base: 10000.0,
            qk_norm_scale: 1.0,
            window_pattern: WindowPattern::Full,
            mlp_activation: MlpActivation::SwiGlu,
            layer_overrides: Vec::new(),
            tie_embeddings: false,
            logit_cap: 0.0,
            recurrent_steps: 1,
            has_exit_gate: false,
            value_embed: ValueEmbedConfig::default(),
            residual: ResidualConfig {
                resid_start: 1.0, resid_end: 1.0,
                x0_start: 0.0, x0_end: 0.0,
                backout_lambda: 0.0,
            },
            smear: SmearConfig::default(),
            precision: Precision::F32,
            norm_eps: 1e-5,
        }
    }

    /// Build a tiny test model with deterministic random weights.
    /// Same shape as `modgrad-transformer/src/resident.rs::build_test_model`.
    fn build_test_model(config: &GptConfig) -> (GptModel, Vec<SwigluMlp>) {
        let md = config.model_dim.get();
        let kv_dim = config.num_kv_heads.get() * config.head_dim.get();
        let vocab = config.vocab_size.get();
        let mlp_dim = config.mlp_dim.get();

        let mut rng = modgrad_compute::neuron::SimpleRng::new(0xBADBEEF);
        let randn = |rng: &mut modgrad_compute::neuron::SimpleRng, n: usize| -> Vec<f32> {
            (0..n).map(|_| rng.next_normal() * 0.05).collect()
        };

        let token_embed = randn(&mut rng, vocab * md);
        let lm_head = randn(&mut rng, vocab * md);
        let final_norm_scale = vec![1.0f32; md];
        let smear_gate = vec![0.0f32; md * config.smear.gate_channels];

        let mut blocks_host = Vec::with_capacity(config.num_layers.get());
        let mut swiglu_mlps = Vec::with_capacity(config.num_layers.get());
        for li in 0..config.num_layers.get() {
            let attn_w = AttentionWeights {
                wq: Tensor2::new(randn(&mut rng, md * md), md, md).unwrap(),
                wk: Tensor2::new(randn(&mut rng, kv_dim * md), kv_dim, md).unwrap(),
                wv: Tensor2::new(randn(&mut rng, kv_dim * md), kv_dim, md).unwrap(),
                wo: Tensor2::new(randn(&mut rng, md * md), md, md).unwrap(),
            };
            let attn = CausalSelfAttention::new(attn_w, config);

            let gate_w = randn(&mut rng, mlp_dim * md);
            let up_w = randn(&mut rng, mlp_dim * md);
            let down_w = randn(&mut rng, md * mlp_dim);
            let swiglu_w = SwigluWeights {
                gate: Tensor2::new(gate_w, mlp_dim, md).unwrap(),
                up: Tensor2::new(up_w, mlp_dim, md).unwrap(),
                down: Tensor2::new(down_w, md, mlp_dim).unwrap(),
            };
            let swiglu = SwigluMlp::new(swiglu_w, config.model_dim, config.mlp_dim);
            swiglu_mlps.push(swiglu);

            let placeholder_mlp = Mlp::new(
                MlpWeights {
                    fc: Tensor2::zeros(mlp_dim, md),
                    proj: Tensor2::zeros(md, mlp_dim),
                },
                config.model_dim, config.mlp_dim,
            );
            let layer_idx = LayerIdx::new(li, config.num_layers).unwrap();
            blocks_host.push(TransformerBlock::new(
                attn, placeholder_mlp, None, layer_idx, config,
            ));
        }

        let model = GptModel {
            embed: Tensor2::new(token_embed, vocab, md).unwrap(),
            lm_head: Tensor2::new(lm_head, vocab, md).unwrap(),
            final_norm: ScaledRmsNorm::new(final_norm_scale, config.norm_eps),
            smear_inference: Smear::<Inference>::new(SmearWeights::new(
                smear_gate.clone(), config.model_dim, &config.smear,
            )),
            smear_training: Smear::<Training>::new(SmearWeights::new(
                smear_gate, config.model_dim, &config.smear,
            )),
            blocks: blocks_host,
            lambdas: ResidualLambdas::from_config(&config.residual, config.num_layers),
            rope: RotaryEmbedding::new(config.head_dim, config.max_seq_len, config.rope_base),
            position: Box::new(FixedPositioning),
            config: config.clone(),
        };
        (model, swiglu_mlps)
    }

    macro_rules! skip_if_no_gpu {
        () => {
            if !runtime_available() {
                eprintln!("hip runtime unavailable, skipping");
                return;
            }
        };
    }

    /// `LmTrainerConfig::default()` produces sensible numbers — pure
    /// CPU sanity check that doesn't touch the GPU. Validates that
    /// the public defaults track the AdamW recipe documented in
    /// the module preamble.
    #[test]
    fn config_defaults() {
        let cfg = LmTrainerConfig::default();
        assert!(cfg.lr > 0.0 && cfg.lr < 1.0, "lr default sane");
        assert!(cfg.beta1 >= 0.5 && cfg.beta1 < 1.0, "beta1 sane");
        assert!(cfg.beta2 >= 0.5 && cfg.beta2 < 1.0, "beta2 sane");
        assert!(cfg.eps > 0.0 && cfg.eps < 1e-3, "eps sane");
        assert!(cfg.weight_decay >= 0.0, "wd nonneg");
        assert!(cfg.grad_clip >= 0.0, "clip nonneg");
        assert!(cfg.micro_batch_size > 0, "mbs positive");
        assert!(cfg.seq_len > 0, "seq_len positive");
    }

    /// `clip_grads` halves grads when their global norm is 2× the
    /// limit. CPU-only — exercises the math without touching GPU
    /// dispatch. Lets us validate the math under `--no-default-features
    /// --features rocm` without a real GPU present.
    #[test]
    fn clip_grads_global_norm() {
        let mut a = vec![3.0f32, 4.0]; // ‖a‖₂ = 5
        let mut b = vec![0.0f32];
        let total_pre: f32 = (3.0f32.powi(2) + 4.0f32.powi(2)).sqrt();
        assert!((total_pre - 5.0).abs() < 1e-6, "pre-clip sanity");

        let mut s: Vec<&mut [f32]> = vec![a.as_mut_slice(), b.as_mut_slice()];
        LmTrainer::<modgrad_transformer::GptModelResident>::clip_grads(&mut s, 2.5);

        // After clip: scale = 2.5 / 5.0 = 0.5
        assert!((a[0] - 1.5).abs() < 1e-5);
        assert!((a[1] - 2.0).abs() < 1e-5);
        assert!(b[0].abs() < 1e-9);

        // No-op when `max_norm = 0`.
        let mut c = vec![10.0f32, 10.0];
        let mut s2: Vec<&mut [f32]> = vec![c.as_mut_slice()];
        LmTrainer::<modgrad_transformer::GptModelResident>::clip_grads(&mut s2, 0.0);
        assert_eq!(c, vec![10.0, 10.0], "max_norm=0 disables clipping");

        // No-op when norm already under budget.
        let mut d = vec![0.1f32, 0.1];
        let mut s3: Vec<&mut [f32]> = vec![d.as_mut_slice()];
        LmTrainer::<modgrad_transformer::GptModelResident>::clip_grads(&mut s3, 100.0);
        assert!((d[0] - 0.1).abs() < 1e-9);
        assert!((d[1] - 0.1).abs() < 1e-9);
    }

    /// `AdamWBuf::step_resident` updates the device weight + moments
    /// in place, and bumps `t`. After one step against a
    /// positive-mean gradient, the weight magnitude shrinks (AdamW
    /// with decoupled weight decay nudges it toward zero), and the
    /// moment buffers carry non-zero state (download to verify).
    /// Requires a HIP-capable device — same gating macro as the smoke
    /// test below.
    #[test]
    fn adamw_buf_step_resident_basic() {
        let _guard = HIP_TEST_LOCK.lock().unwrap();
        skip_if_no_gpu!();
        let mut buf = AdamWBuf::zeros(4).expect("alloc resident adamw buf");
        // Resident weight starts at 1.0 across the board.
        let weight_buf = modgrad_device::backend::HipBuffer::new(4 * 4)
            .expect("alloc weight");
        weight_buf.copy_from_host(&[1.0f32, 1.0, 1.0, 1.0]).expect("upload weight");
        let grads = [0.5f32, 0.5, 0.5, 0.5];
        let cfg = LmTrainerConfig::default();
        buf.step_resident(&weight_buf, &grads, &cfg).expect("resident adamw");
        assert_eq!(buf.t, 1, "t bumped");

        // Weight should have shrunk from 1.0 toward zero, but only
        // slightly with the default tiny lr / wd.
        let mut weight_host = [0.0f32; 4];
        weight_buf.copy_to_host(&mut weight_host).expect("download weight");
        for &p in &weight_host {
            assert!(p < 1.0 && p > 0.99,
                "param after one tiny AdamW step should be just under 1.0, got {p}");
        }

        // First / second moments should carry the absorbed gradient.
        let mut m_host = [0.0f32; 4];
        let mut v_host = [0.0f32; 4];
        buf.m_dev.copy_to_host(&mut m_host);
        buf.v_dev.copy_to_host(&mut v_host);
        for &m in &m_host { assert!(m > 0.0, "first moment populated"); }
        for &v in &v_host { assert!(v > 0.0, "second moment populated"); }
    }

    /// End-to-end smoke test: build a tiny GptModelResident, wrap in
    /// LmTrainer, run 10 train_steps with the resident backward + AdamW
    /// wired. Asserts loss DECREASES over the run — the proof that
    /// every stage of the pipeline (forward → CE → backward → AdamW →
    /// re-upload) is actually moving the weights toward the target.
    ///
    /// Why 10 steps and `lr = 1e-2`: a 2-layer model with d_model=128
    /// and a 17-token deterministic pattern is small enough that even
    /// a few high-LR updates dominate the random-init plateau. The
    /// 5%-drop bound is conservative — observed drops on the bench
    /// were ~10-30%. If this test goes red, the wiring is broken
    /// (sign error, stale resident weight, mis-keyed grad), not a
    /// numerical edge case.
    #[test]
    fn lm_trainer_smoke_loss_decreases() {
        let _guard = HIP_TEST_LOCK.lock().unwrap();
        skip_if_no_gpu!();
        let config = tiny_config();
        let (model, swiglu_mlps) = build_test_model(&config);

        let resident = GptModelResident::from_model(&model, &swiglu_mlps)
            .expect("upload model");

        let mbs = 16usize;
        let trainer_cfg = LmTrainerConfig {
            lr: 1e-3,
            micro_batch_size: mbs,
            seq_len: mbs,
            ..LmTrainerConfig::default()
        };

        let n_kv = config.num_kv_heads.get();
        let hd = config.head_dim.get();
        let mut trainer = LmTrainer::new(resident, n_kv, hd, trainer_cfg)
            .expect("trainer alloc");

        // Synthetic 17-token sequence → 16 source tokens, 16 targets.
        // Deterministic so the gradient signal is identical step-to-step;
        // weight updates are the only thing that can change the loss.
        let tokens: Vec<i64> = (0..(mbs as i64) + 1)
            .map(|i| (i * 17) % (config.vocab_size.get() as i64))
            .collect();

        let n_steps = 10;
        let mut losses = Vec::with_capacity(n_steps);
        for step in 0..n_steps {
            let batch = HipBatch::new();
            let loss = trainer.train_step(&batch, &tokens)
                .unwrap_or_else(|e| panic!("train_step {step} failed: {e}"));
            assert!(loss.is_finite(),
                "step {step}: loss is non-finite ({loss}) — forward / backward bug");
            losses.push(loss);
        }
        assert_eq!(trainer.loss_history().len(), n_steps,
            "loss history accumulated");

        // Proof of training: loss[0] strictly larger than loss[9] by >5%.
        // If this fails, something in the chain isn't moving weights
        // toward lower loss — debug before papering over.
        let l0 = losses[0];
        let l_last = losses[n_steps - 1];
        eprintln!("lm_trainer smoke: {n_steps} steps, losses {losses:?}");
        assert!(l_last < l0 * 0.95,
            "loss did not decrease: l0={l0} l_last={l_last} losses={losses:?}");
    }
}
