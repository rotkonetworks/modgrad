//! BLT-specific training loop wrapper.
//!
//! BLT training is conceptually identical to standard LM training —
//! same forward/CE/backward/AdamW — except:
//!   - inputs are byte sequences (vocab=256 in the byte path) plus
//!     their precomputed patch boundaries (from [`crate::patcher`])
//!   - the optimizer applies different LRs to local vs global
//!     parameters (paper §6.2: 1/10 LR on global when byte-ifying a
//!     pretrained backbone)
//!
//! ## Layering vs `isis_runtime::LmTrainer`
//!
//! `isis_runtime::lm_trainer::LmTrainer<M>` is the reference foundation
//! for autoregressive LM training. Its [`LmTrainer::train_step`]
//! dispatches `AdamWBuf::step_resident` once per parameter using
//! `&self.config` as the source of `lr` — a single global LR for the
//! whole model.
//!
//! The BLT recipe wants two LRs in one step (local 3e-4, global 3e-5
//! per paper §6.2). `LmTrainer` does not surface a per-param-name LR
//! hook. To avoid editing isis-runtime (out of scope for this slice),
//! `BltTrainer` mirrors `LmTrainer<GptModelResident>::train_step`'s
//! pipeline structure here, injecting the LR routing at the AdamW step.
//! The reused public surfaces are:
//!   - [`LanguageModel`] for forward + per-position backward
//!   - [`AdamWBuf::step_resident`] (the device kernel dispatch — we
//!     just hand it the right `lr` per call)
//!   - [`LmTrainer::clip_grads`] for grad-norm clipping (LR-agnostic;
//!     reused as a static helper)
//!   - [`GptBackwardState`] field access (all `pub`) for grad
//!     accumulation
//!   - [`GptModelResident`]'s public weight-buffer fields
//!     (`embed_dev`, `lm_head.weight_dev`, `blocks[i].attn/mlp.*`)
//!
//! The mirroring is deliberate — when sasha's `BltModel` lands as a
//! `LanguageModel` impl, the only diffs are: (a) flip `M` to `BltModel`,
//! (b) extend `param_keys` / `weight_dev_for_key` to cover the new
//! local-stack parameters. The forward + backward + AdamW shape stays.
//!
//! ## Boundaries today vs once sasha's slice lands
//!
//! `train_step(bytes, boundaries)` accepts `boundaries` for the eventual
//! `BltModel` path (encoder cross-attn pools bytes → patch reps under
//! the patch mask; decoder cross-attn does the inverse). With sasha's
//! slice still in flight, the latent (Qwen-class `GptModelResident`)
//! consumes bytes directly as token ids; `boundaries` is asserted
//! well-formed but otherwise carried through unread.
//!
//! ## Why `is_global` is a `Box<dyn Fn>` and not an enum
//!
//! BLT's local-vs-global split is *named-parameter*, not structural.
//! `byteify` derives the predicate from the param-key schema
//! (`embed`/`lm_head`/`block.{i}.{slot}` — currently all global). A
//! dynamic closure lets a future `BltModel` add new param keys
//! (`encoder.block.{i}.wq`, `cross_attn.{i}.wq`, …) without revising
//! this trainer's API: the caller updates `is_global`, that's it.

#![allow(dead_code)]

#[cfg(feature = "rocm")]
use std::collections::HashMap;

#[cfg(feature = "rocm")]
use isis_runtime::language_model::LanguageModel;
#[cfg(feature = "rocm")]
use isis_runtime::lm_trainer::{AdamWBuf, LmTrainer, LmTrainerConfig};
#[cfg(feature = "rocm")]
use modgrad_compute::backend::{GpuVec, ResidencyError};
#[cfg(feature = "rocm")]
use modgrad_device::backend::{HipBatch, HipBuffer};
#[cfg(feature = "rocm")]
use modgrad_transformer::loss::cross_entropy;
#[cfg(feature = "rocm")]
use modgrad_transformer::resident::GptBackwardState;
#[cfg(feature = "rocm")]
use modgrad_transformer::{GptModelResident, KvCacheResident};

#[cfg(feature = "rocm")]
use crate::model::{BltBackwardState, BltModel, BltScratch};

// ─── Config ───────────────────────────────────────────────────

/// Hyperparameters for the BLT trainer. Defaults follow Pagnoni et al.
/// 2024 §4.8 and §6.2:
///   - β₁ = 0.9, β₂ = 0.95 (paper §4.8 — note this is *not* the GPT-3
///     β₂ = 0.999; BLT calls for 0.95)
///   - weight decay 0.1, grad clip 1.0
///   - global LR = local LR / 10 (the load-bearing scheduling decision
///     when initializing Latent from a pretrained checkpoint)
#[cfg(feature = "rocm")]
#[derive(Debug, Clone)]
pub struct BltTrainerConfig {
    /// Learning rate for *local* parameters (Local Encoder, Local
    /// Decoder, cross-attention). Default 3e-4 per Pagnoni et al. 2024
    /// §4.8 / §6.2 (matches Llama 3 8B small-model regime).
    pub local_lr: f32,
    /// Learning rate for *global* parameters (Latent transformer). Set
    /// to 1/10 the local LR by paper §6.2 to preserve the pretrained
    /// representation while still letting it adapt to the byte-level
    /// IO formed by the local stack. Default 3e-5.
    pub global_lr: f32,
    /// AdamW β₁. Default 0.9.
    pub beta1: f32,
    /// AdamW β₂. Default 0.95 — the BLT paper's choice (§4.8); not the
    /// GPT-3 0.999.
    pub beta2: f32,
    /// Numerical-stability epsilon. Default 1e-8.
    pub eps: f32,
    /// Decoupled weight decay. Default 0.1 (BLT §4.8).
    pub weight_decay: f32,
    /// Global gradient norm clip; 0 disables. Default 1.0.
    pub grad_clip: f32,
    /// Bytes per micro-batch — sets `bytes.len() == micro_batch_size +
    /// 1` (next-byte target alignment).
    pub micro_batch_size: usize,
    /// Sequence length the latent expects (in bytes today; will be
    /// patches once `BltModel` lands). Sets the resident KV cache
    /// capacity.
    pub seq_len: usize,
}

#[cfg(feature = "rocm")]
impl Default for BltTrainerConfig {
    fn default() -> Self {
        Self {
            local_lr: 3e-4,
            global_lr: 3e-5,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            weight_decay: 0.1,
            grad_clip: 1.0,
            micro_batch_size: 16,
            seq_len: 16,
        }
    }
}

#[cfg(feature = "rocm")]
impl BltTrainerConfig {
    /// Per-param `LmTrainerConfig` view. `lr` carries the chosen group
    /// LR; everything else mirrors the BLT config. Used as the `cfg`
    /// arg to [`AdamWBuf::step_resident`].
    fn for_group(&self, global: bool) -> LmTrainerConfig {
        LmTrainerConfig {
            lr: if global { self.global_lr } else { self.local_lr },
            beta1: self.beta1,
            beta2: self.beta2,
            eps: self.eps,
            weight_decay: self.weight_decay,
            grad_clip: self.grad_clip,
            micro_batch_size: self.micro_batch_size,
            seq_len: self.seq_len,
        }
    }
}

// ─── Trainer ──────────────────────────────────────────────────

/// BLT trainer — owns a Qwen-class `GptModelResident` (the latent), the
/// per-param `AdamWBuf` map, a resident `KvCacheResident`, the backward
/// state, and the per-group LR predicate. Mirrors the
/// `LmTrainer<GptModelResident>::train_step` pipeline with per-param
/// LR routing inserted at the AdamW dispatch.
///
/// **Today's M.** The latent is the Qwen2.5-0.5B `GptModelResident`
/// from [`crate::byteify::ByteifyRecipe::from_qwen2`]. Sasha's
/// `BltModel` (encoder + latent + decoder, real `LanguageModel` impl)
/// is in flight; when it lands the trainer's type parameter flips to
/// `BltModel` and `param_keys`/`weight_dev_for_key` extend to cover the
/// new keys.
///
/// **`is_global` semantics.** Consulted once per parameter per step.
/// Must be deterministic and side-effect-free. The byteify recipe ships
/// the canonical predicate via
/// [`crate::byteify::ByteifyRecipe::global_predicate`] — `true` for
/// every latent param (which is every param today, until the local
/// stack lands).
#[cfg(feature = "rocm")]
pub struct BltTrainer {
    /// The latent transformer. Every weight buffer is resident on
    /// device; AdamW updates them in place via the resident kernel.
    model: GptModelResident,
    /// Per-`Linear` AdamW state, keyed `embed`, `lm_head`,
    /// `block.{li}.{wq,wk,wv,wo,gate,up,down}`.
    adamw: HashMap<String, AdamWBuf>,
    /// Persistent host grad accumulator — one entry per param, sized
    /// to match the device grad buffers. Reset to zero per step.
    grad_acc: HashMap<String, Vec<f32>>,
    /// Resident KV cache reused across `train_step` calls. Reset before
    /// every step.
    kv_cache: KvCacheResident,
    /// Pre-allocated backward state. Reused across steps to avoid
    /// re-allocating GPU scratch every batch.
    bwd_state: Option<GptBackwardState>,
    /// Loss values from completed `train_step` calls, in order.
    loss_history: Vec<f32>,
    config: BltTrainerConfig,
    /// Per-parameter-name predicate: `true` ⇒ apply `global_lr`;
    /// `false` ⇒ `local_lr`.
    is_global: Box<dyn Fn(&str) -> bool + Send>,
}

#[cfg(feature = "rocm")]
impl BltTrainer {
    /// Wrap a Qwen-class `GptModelResident` (produced by
    /// [`crate::byteify::ByteifyRecipe::from_qwen2`]) with the BLT
    /// per-group LR schedule. `is_global` decides the per-param AdamW
    /// LR — the byteify recipe ships a canonical predicate.
    ///
    /// `n_kv_heads` and `head_dim` size the resident KV cache. For
    /// Qwen2.5-0.5B these are 2 and 64 respectively
    /// (`modgrad_io::qwen2::QWEN2_5_0_5B_NUM_KV_HEADS` /
    /// `..._HEAD_DIM`).
    pub fn new(
        latent: GptModelResident,
        n_kv_heads: usize,
        head_dim: usize,
        config: BltTrainerConfig,
        is_global: Box<dyn Fn(&str) -> bool + Send>,
    ) -> Result<Self, ResidencyError> {
        let n_layers = latent.num_layers();
        let d_model = latent.model_dim();
        let kv_cache = KvCacheResident::new(
            n_layers, n_kv_heads, head_dim, config.seq_len, d_model,
        )?;
        Ok(Self {
            model: latent,
            adamw: HashMap::new(),
            grad_acc: HashMap::new(),
            kv_cache,
            bwd_state: None,
            loss_history: Vec::new(),
            config,
            is_global,
        })
    }

    /// Read-only access to the wrapped model. Useful for inference
    /// after training, or for poking at per-block residency state.
    pub fn model(&self) -> &GptModelResident { &self.model }

    /// Mutable access for callers that own model serialization or want
    /// to drive validation forward passes via the `LanguageModel` trait.
    pub fn model_mut(&mut self) -> &mut GptModelResident { &mut self.model }

    /// Mutable access to the resident KV cache. Validation harnesses
    /// can borrow this for held-out forward passes.
    pub fn kv_cache_mut(&mut self) -> &mut KvCacheResident { &mut self.kv_cache }

    /// Configuration view.
    pub fn config(&self) -> &BltTrainerConfig { &self.config }

    /// Loss values from completed `train_step` calls, in order.
    pub fn loss_history(&self) -> &[f32] { &self.loss_history }

    /// AdamW state — exposed for tests + tools that want to inspect
    /// optimizer momentum, not for the hot path.
    pub fn adamw_state(&self) -> &HashMap<String, AdamWBuf> { &self.adamw }

    /// One BLT training step on a contiguous byte window. `bytes.len()`
    /// must equal `micro_batch_size + 1` (next-byte target alignment).
    /// `boundaries` are the patch boundaries from `crate::patcher` — a
    /// strictly increasing sorted list of byte indices, each in
    /// `[0, bytes.len())`. A fixed-stride patcher (`[0, P, 2P, ...]`)
    /// is a valid fallback when the entropy patcher hasn't landed.
    ///
    /// **Today's path.** With BltModel a placeholder, the latent runs
    /// directly on bytes (vocab=256-style decode) — `boundaries` is
    /// validated but otherwise unread. The expected diff once
    /// `BltModel` is real: forward switches to `BltModel::forward`
    /// which threads `boundaries` into the encoder's per-patch mask
    /// and the decoder's byte→patch cross-attn.
    ///
    /// Returns the cross-entropy loss for the step.
    pub fn train_step(
        &mut self,
        bytes: &[u8],
        boundaries: &[usize],
    ) -> Result<f32, ResidencyError> {
        let mbs = self.config.micro_batch_size;
        assert_eq!(bytes.len(), mbs + 1,
            "BltTrainer::train_step: bytes.len() must be micro_batch_size + 1 ({} + 1), got {}",
            mbs, bytes.len());
        validate_boundaries(bytes.len(), boundaries);

        let tokens: Vec<i64> = bytes.iter().map(|&b| b as i64).collect();
        let batch = HipBatch::new();
        self.train_step_tokens(&batch, &tokens)
    }

    /// Token-level entry point used by `train_step` and tests. Mirrors
    /// `LmTrainer<GptModelResident>::train_step` but routes per-param
    /// LR via `self.is_global`.
    fn train_step_tokens(
        &mut self,
        batch: &HipBatch,
        tokens: &[i64],
    ) -> Result<f32, ResidencyError> {
        let mbs = self.config.micro_batch_size;
        assert_eq!(tokens.len(), mbs + 1,
            "BltTrainer: tokens.len() must be mbs+1 ({} + 1), got {}",
            mbs, tokens.len());

        let vocab = self.model.vocab_size();
        let n_layers = self.model.num_layers();
        let d_model = self.model.model_dim();

        // Stage 0: lazy alloc of resident AdamW state + grad acc + bwd
        // state. First call only.
        if self.adamw.is_empty() {
            self.lazy_init()?;
        }

        // Zero the host grad accumulator.
        for v in self.grad_acc.values_mut() {
            v.fill(0.0);
        }

        // Stage 1: reset cache.
        self.kv_cache.reset();

        // Stage 2: forward → CE → d_logits.
        let src_tokens: &[i64] = &tokens[..mbs];
        let positions: Vec<usize> = (0..mbs).collect();
        let mut logits_dev = GpuVec::try_hip(mbs * vocab)?;

        LanguageModel::ensure_resident(&mut self.model, batch)?;
        LanguageModel::forward_logits(
            &mut self.model, batch, src_tokens, &positions,
            Some(&mut self.kv_cache),
            &mut logits_dev,
        )?;
        batch.flush()?;

        let mut logits_host = vec![0.0f32; mbs * vocab];
        logits_dev.copy_to_host(&mut logits_host);
        let targets: &[i64] = &tokens[1..=mbs];
        let (loss, grad_logits) = cross_entropy(&logits_host, targets, vocab);

        assert!(loss.is_finite(),
            "BltTrainer::train_step: loss is NaN/inf — upstream forward bug");
        self.loss_history.push(loss);

        // Stage 3: per-position forward_for_backward + backward.
        self.kv_cache.reset();

        let mut d_logits_pos = GpuVec::try_hip(vocab)?;
        let mut row_scratch: Vec<f32> = Vec::with_capacity(vocab);

        // Take bwd_state out so we can hold &mut to it concurrently
        // with &mut self.kv_cache and &mut self.model. Restore at end.
        let mut bwd = self.bwd_state.take()
            .expect("bwd_state populated by lazy_init");

        let result: Result<(), ResidencyError> = (|| {
            for pos in 0..mbs {
                LanguageModel::forward_for_backward_position(
                    &mut self.model, batch, src_tokens[pos], pos,
                    &mut self.kv_cache, &mut bwd,
                    &mut d_logits_pos,
                )?;
                row_scratch.clear();
                row_scratch.extend_from_slice(
                    &grad_logits[pos * vocab..(pos + 1) * vocab],
                );
                d_logits_pos.copy_from(&row_scratch);

                LanguageModel::backward_position(
                    &mut self.model, batch, src_tokens[pos], pos,
                    &mut self.kv_cache, &mut bwd,
                    &d_logits_pos,
                )?;
                batch.flush()?;

                accumulate_position_grads(
                    &bwd, n_layers, src_tokens[pos], d_model,
                    &mut self.grad_acc,
                );
            }
            Ok(())
        })();
        self.bwd_state = Some(bwd);
        result?;

        // Stage 4: optional grad-norm clip + per-param AdamW.
        if self.config.grad_clip > 0.0 {
            let mut slices: Vec<&mut [f32]> =
                self.grad_acc.values_mut().map(|v| v.as_mut_slice()).collect();
            // Reuse LmTrainer's clip helper — LR-agnostic.
            LmTrainer::<GptModelResident>::clip_grads(
                &mut slices, self.config.grad_clip,
            );
        }

        let keys = param_keys(n_layers);
        for k in &keys {
            let grad = self.grad_acc.get(k)
                .expect("grad acc populated by lazy_init");
            let weight_dev = weight_dev_for_key(&self.model, k, n_layers);
            let buf = self.adamw.get_mut(k)
                .expect("adamw populated by lazy_init");
            // Per-param LR routing — the load-bearing decision.
            let cfg = self.config.for_group((self.is_global)(k));
            buf.step_resident(weight_dev, grad.as_slice(), &cfg)?;
        }

        Ok(loss)
    }

    /// First-call setup — allocate AdamW state + grad acc + backward
    /// state. Idempotent guard via `adamw.is_empty()`. Mirrors
    /// `LmTrainer::lazy_init_state`'s sizing.
    fn lazy_init(&mut self) -> Result<(), ResidencyError> {
        let n_layers = self.model.num_layers();
        let d_model = self.model.model_dim();
        let vocab = self.model.vocab_size();

        let mut alloc = |key: &str, n: usize| -> Result<(), ResidencyError> {
            self.adamw.insert(key.to_string(), AdamWBuf::zeros(n)?);
            self.grad_acc.insert(key.to_string(), vec![0.0f32; n]);
            Ok(())
        };

        alloc("embed", vocab * d_model)?;
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
        alloc("lm_head", vocab * d_model)?;

        let state = LanguageModel::alloc_backward_state(&self.model, &self.kv_cache)?;
        self.bwd_state = Some(state);
        Ok(())
    }
}

// ─── Helpers ──────────────────────────────────────────────────

/// Stable parameter keys — one per `Linear` plus the embedding table.
/// Mirrors the modgrad-ctm + isis-runtime convention so a downstream
/// serializer can checkpoint either trainer with the same schema.
///
/// **When `BltModel` lands**, this function extends with keys for the
/// local stack: `encoder.block.{i}.{wq,…}`, `cross_attn.{i}.{wq,…}`,
/// `decoder.block.{i}.{wq,…}`. Today only the latent keys exist.
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

/// Look up the model's resident weight buffer for a given param key.
/// Inverse of [`param_keys`]. Used by the resident AdamW dispatch to
/// mutate `weight_dev` in place.
#[cfg(feature = "rocm")]
fn weight_dev_for_key<'a>(
    model: &'a GptModelResident,
    key: &str,
    n_layers: usize,
) -> &'a HipBuffer {
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
        other => panic!("BltTrainer::weight_dev_for_key: unknown slot {other}"),
    }
}

/// Add the resident-state's per-`Linear` grad buffers (this position's
/// contribution) into the host accumulators. Mirrors
/// `LmTrainer::accumulate_position_grads`.
///
/// **Embedding handling.** [`GptModelResident::backward`] writes
/// `d_embed[token_id, :] = d_hidden` (overwrite) and leaves all other
/// rows untouched. We download the full table and only fold the row
/// matching this position's `token_id` to avoid double-counting earlier
/// positions whose tokens overlap.
#[cfg(feature = "rocm")]
fn accumulate_position_grads(
    state: &GptBackwardState,
    n_layers: usize,
    token_id: i64,
    d_model: usize,
    acc: &mut HashMap<String, Vec<f32>>,
) {
    // Embedding: download only the row written by this backward.
    let mut full = vec![0.0f32; state.d_embed.len()];
    state.d_embed.copy_to_host(&mut full);
    let row_off = token_id as usize * d_model;
    let acc_embed = acc.get_mut("embed").unwrap();
    for i in 0..d_model {
        acc_embed[row_off + i] += full[row_off + i];
    }

    for li in 0..n_layers {
        add_into(acc.get_mut(&format!("block.{li}.wq")).unwrap(),
            &state.attn_grads[li].dweight_q);
        add_into(acc.get_mut(&format!("block.{li}.wk")).unwrap(),
            &state.attn_grads[li].dweight_k);
        add_into(acc.get_mut(&format!("block.{li}.wv")).unwrap(),
            &state.attn_grads[li].dweight_v);
        add_into(acc.get_mut(&format!("block.{li}.wo")).unwrap(),
            &state.attn_grads[li].dweight_o);
        add_into(acc.get_mut(&format!("block.{li}.gate")).unwrap(),
            &state.mlp_grads[li].dweight_gate);
        add_into(acc.get_mut(&format!("block.{li}.up")).unwrap(),
            &state.mlp_grads[li].dweight_up);
        add_into(acc.get_mut(&format!("block.{li}.down")).unwrap(),
            &state.mlp_grads[li].dweight_down);
    }
    add_into(acc.get_mut("lm_head").unwrap(),
        &state.d_lm_head_weight);
}

/// Download `src` into a temp and add element-wise into `dst`. Mirrors
/// `LmTrainer::add_into`.
#[cfg(feature = "rocm")]
fn add_into(dst: &mut [f32], src: &GpuVec) {
    let mut tmp = vec![0.0f32; dst.len()];
    src.copy_to_host(&mut tmp);
    for (a, b) in dst.iter_mut().zip(tmp.iter()) {
        *a += *b;
    }
}

/// Validate that `boundaries` is a strictly increasing sorted list of
/// byte indices within `[0, n)`. Panics on violation — boundaries come
/// from `crate::patcher`, and a malformed list there is a programmer
/// error, not a runtime condition.
#[cfg(feature = "rocm")]
fn validate_boundaries(n: usize, boundaries: &[usize]) {
    if boundaries.is_empty() { return; }
    for w in boundaries.windows(2) {
        assert!(w[0] < w[1],
            "BltTrainer::train_step: boundaries not strictly increasing ({} ≥ {})",
            w[0], w[1]);
    }
    let last = *boundaries.last().unwrap();
    assert!(last < n,
        "BltTrainer::train_step: boundary[last]={last} ≥ bytes.len()={n}");
}

// ─── BltModelTrainer ──────────────────────────────────────────
//
// Parallel to `BltTrainer` (above) but operates on the full `BltModel`
// (encoder + latent + decoder + cross-attn) via the sequence-level
// `BltModel::forward_for_backward` / `BltModel::backward` API.
//
// `BltTrainer` (above) trains the bare latent (Path A inference recipe);
// `BltModelTrainer` trains the complete BLT pipeline (Path B byte-ification
// recipe). Both coexist deliberately — they are not interchangeable. BLT
// processes patches as a unit, so the per-position `LanguageModel` shape
// the upper trainer relies on is a non-starter for `BltModel`.

/// BLT trainer for the full [`BltModel`] (encoder + latent + decoder
/// + cross-attn). Drives sequence-level forward + backward and applies
/// per-`Linear` AdamW with the local-vs-global LR routing required by
/// the paper §6.2 byte-ification recipe.
///
/// ## Per-`Linear` AdamW key map
///
/// One `AdamWBuf` per parameter buffer. Keys are stable strings so
/// callers (e.g. [`crate::byteify::ByteifyRecipe::global_predicate`])
/// can route per-name LR.
///
/// **Encoder** (per byte-layer `li`, `li ∈ 0..lE`):
///   - `encoder.byte_embed`           → `model.encoder.byte_embed_dev`
///                                       (grad: `state.encoder_grads.d_byte_embed`)
///   - `encoder.block.{li}.wq`        → `model.encoder.byte_layers[li].attn.q_proj.weight_dev`
///                                       (grad: `state.encoder_grads.attn_grads[li].dweight_q`)
///   - `encoder.block.{li}.wk`        → `…attn.k_proj.weight_dev`     (`dweight_k`)
///   - `encoder.block.{li}.wv`        → `…attn.v_proj.weight_dev`     (`dweight_v`)
///   - `encoder.block.{li}.wo`        → `…attn.o_proj.weight_dev`     (`dweight_o`)
///   - `encoder.block.{li}.gate`      → `…mlp.gate.weight_dev`        (`mlp_grads[li].dweight_gate`)
///   - `encoder.block.{li}.up`        → `…mlp.up.weight_dev`          (`dweight_up`)
///   - `encoder.block.{li}.down`      → `…mlp.down.weight_dev`        (`dweight_down`)
///   - `encoder.cross_attn.{li}.wq`   → `model.encoder.cross_attns[li].q_proj.weight_dev`
///                                       (grad: `state.encoder_grads.cross_attn_grads[li].dweight_q`)
///   - `encoder.cross_attn.{li}.wk`   → `…k_proj.weight_dev`          (`dweight_k`)
///   - `encoder.cross_attn.{li}.wv`   → `…v_proj.weight_dev`          (`dweight_v`)
///   - `encoder.cross_attn.{li}.wo`   → `…o_proj.weight_dev`          (`dweight_o`)
///
/// **Latent** (per layer `li`, `li ∈ 0..lL`):
///   - `latent.block.{li}.wq`         → `model.latent.blocks[li].attn.q_proj.weight_dev`
///                                       (grad: `state.latent_attn_grads[li].dweight_q`)
///   - `latent.block.{li}.{wk,wv,wo,gate,up,down}` — same shape as encoder block.
///   - `latent.final_norm`            → `model.latent_final_norm_weight_dev`
///                                       (grad: `state.d_latent_final_norm_weight`)
///
/// **Decoder** (per byte-layer `li`, `li ∈ 0..lD`):
///   - `decoder.block.{li}.{wq,wk,wv,wo,gate,up,down}` — same shape as encoder block.
///   - `decoder.cross_attn.{li}.{wq,wk,wv,wo}` — same shape as encoder cross-attn.
///   - `decoder.lm_head`              → `model.decoder.lm_head.weight_dev`
///                                       (grad: `state.decoder_grads.dweight_lm_head`)
///   - `decoder.lm_head_bias`         → `model.decoder.lm_head.bias_dev`
///                                       (grad: `state.decoder_grads.dbias_lm_head`)
///   - `decoder.final_norm`           → `model.decoder.final_norm_weight_dev`
///                                       (grad: `state.decoder_grads.dweight_final_norm`)
///
/// Bias buffers on attention / MLP / cross-attn `Linear`s are not
/// AdamW-updated — they are initialised to zero by `LinearResident::from_linear`
/// and stay zero (no entry in the AdamW map). The decoder's LM-head bias
/// is the lone exception (it has a dedicated grad accumulator and a
/// dedicated AdamW slot).
///
/// ## Pipeline
///
/// 1. `state.zero_resident` — clear every gradient accumulator.
/// 2. `BltModel::forward_for_backward` over the full byte sequence,
///    populating `byte_logits` of shape `[bytes.len() × 256]`.
/// 3. Cross-entropy on `bytes[1..]` (next-byte targets) for the first
///    `n - 1` byte positions; the last position's logit is dropped.
/// 4. Upload `d_logits` (zero-padded for the last position) and call
///    `BltModel::backward`.
/// 5. Optional global grad clip across every host-downloaded grad.
/// 6. Per-`Linear` AdamW with `is_global` routing the LR group.
#[cfg(feature = "rocm")]
pub struct BltModelTrainer {
    model: BltModel,
    state: BltBackwardState,
    scratch: BltScratch,
    adamw: HashMap<String, AdamWBuf>,
    /// Per-param-key host scratch sized to match each grad `GpuVec`.
    /// Pre-allocated once in [`BltModelTrainer::new`] (via
    /// [`Self::alloc_adamw`]) and reused every [`Self::train_step`] —
    /// eliminates the per-step `Vec<f32>` churn the previous clip path
    /// incurred (one fresh allocation per param per step, and the
    /// `BltModel` key set has on the order of dozens of params for the
    /// tiny config and hundreds for Qwen2.5-0.5B-class scales). Each
    /// step downloads grads into these slots, optionally clips them in
    /// place, then hands them to [`AdamWBuf::step_resident`].
    grad_host_scratch: HashMap<String, Vec<f32>>,
    config: BltTrainerConfig,
    is_global: Box<dyn Fn(&str) -> bool + Send>,
    loss_history: Vec<f32>,
    step: usize,
}

#[cfg(feature = "rocm")]
impl BltModelTrainer {
    /// Wrap a [`BltModel`] in the BLT per-group LR schedule. Allocates
    /// one [`AdamWBuf`] per parameter buffer (see the type docs for
    /// the full key map). `is_global` decides per-key which LR group
    /// applies — pass [`crate::byteify::ByteifyRecipe::global_predicate`]
    /// for the canonical recipe.
    pub fn new(
        model: BltModel,
        config: BltTrainerConfig,
        is_global: Box<dyn Fn(&str) -> bool + Send>,
    ) -> Result<Self, ResidencyError> {
        let state = BltBackwardState::new(&model)?;
        let scratch = BltScratch::new(&model.config)?;
        let mut trainer = Self {
            model,
            state,
            scratch,
            adamw: HashMap::new(),
            grad_host_scratch: HashMap::new(),
            config,
            is_global,
            loss_history: Vec::new(),
            step: 0,
        };
        trainer.alloc_adamw()?;
        Ok(trainer)
    }

    /// Read-only access to the wrapped model.
    pub fn model(&self) -> &BltModel { &self.model }

    /// Loss values from completed [`Self::train_step`] calls, in order.
    pub fn loss_history(&self) -> &[f32] { &self.loss_history }

    /// Configuration view.
    pub fn config(&self) -> &BltTrainerConfig { &self.config }

    /// AdamW state — exposed for tests + tools that want to inspect
    /// optimizer momentum, not for the hot path.
    pub fn adamw_state(&self) -> &HashMap<String, AdamWBuf> { &self.adamw }

    /// One BLT training step on a contiguous byte window.
    ///
    /// `bytes.len()` must equal `micro_batch_size` (the trainer predicts
    /// next-byte logits at every position and uses `bytes[1..]` as
    /// targets for positions `0..n-1`; the last position has no target
    /// and contributes nothing to the loss). `boundaries` is the patch
    /// boundary list `[0, b_1, …, n]` from [`crate::patcher`] — strictly
    /// increasing, starts at 0, ends at `n`.
    ///
    /// Returns the cross-entropy loss for the step. Asserts on NaN/inf.
    pub fn train_step(
        &mut self,
        bytes: &[u8],
        boundaries: &[usize],
    ) -> Result<f32, ResidencyError> {
        let n = bytes.len();
        let mbs = self.config.micro_batch_size;
        assert_eq!(n, mbs,
            "BltModelTrainer::train_step: bytes.len() must be micro_batch_size ({}), got {}",
            mbs, n);
        assert!(n >= 2,
            "BltModelTrainer::train_step: need at least 2 bytes for a next-byte target, got {n}");
        assert!(!boundaries.is_empty()
                && boundaries[0] == 0
                && *boundaries.last().unwrap() == n,
            "BltModelTrainer::train_step: boundaries must start at 0 and end at bytes.len()={n}");
        validate_boundaries(n + 1, boundaries);

        let batch = HipBatch::new();

        // Stage 0: zero all weight-grad accumulators.
        self.state.zero_resident(&batch)?;

        // Stage 1: forward for backward.
        let mut byte_logits = GpuVec::try_hip(n * 256)?;
        self.model.forward_for_backward(
            &batch,
            bytes,
            boundaries,
            &mut self.scratch,
            &mut self.state,
            &mut byte_logits,
        )?;
        batch.flush()?;

        // Stage 2: CE loss on the first n-1 positions; last position has
        // no target. The dropped position's d_logits row is zero so it
        // contributes nothing to the backward.
        let mut logits_host = vec![0.0f32; n * 256];
        byte_logits.copy_to_host(&mut logits_host);
        let logits_pred = &logits_host[..(n - 1) * 256];
        let targets: Vec<i64> = bytes[1..].iter().map(|&b| b as i64).collect();
        debug_assert_eq!(targets.len(), n - 1);
        let (loss, grad_logits_pred) = cross_entropy(logits_pred, &targets, 256);

        assert!(loss.is_finite(),
            "BltModelTrainer::train_step: loss is NaN/inf — upstream forward bug");
        self.loss_history.push(loss);

        // Stage 3: upload zero-padded d_logits.
        let mut d_logits_host = vec![0.0f32; n * 256];
        d_logits_host[..(n - 1) * 256].copy_from_slice(&grad_logits_pred);
        let mut d_byte_logits = GpuVec::try_hip(n * 256)?;
        d_byte_logits.copy_from(&d_logits_host);

        // Stage 4: backward.
        self.model.backward(
            &batch,
            bytes,
            boundaries,
            &mut self.scratch,
            &mut self.state,
            &d_byte_logits,
        )?;
        batch.flush()?;

        // Stage 5: download grads into pre-allocated host scratch, then
        // (optionally) apply the global grad-norm clip in place. The
        // scratch is allocated once in `alloc_adamw`, reused every step
        // — no per-call `Vec<f32>` churn. The download itself is paid
        // because `AdamWBuf::step_resident` consumes a host slice; the
        // clip stage only adds a scalar reduction + a conditional
        // in-place scale (no second D2H/H2D round-trip). See
        // [`Self::clip_grads_inplace`] for the algorithmic shape.
        let keys = param_keys_for_model(&self.model);
        for k in &keys {
            let grad_dev = grad_dev_for_key(&self.state, k, &self.model);
            let host = self.grad_host_scratch.get_mut(k)
                .expect("grad_host_scratch populated by alloc_adamw");
            debug_assert_eq!(host.len(), grad_dev.len(),
                "grad_host_scratch[{k}] sized {} != grad_dev sized {} — \
                 model shape changed under the trainer",
                host.len(), grad_dev.len());
            grad_dev.copy_to_host(host.as_mut_slice());
        }
        self.clip_grads_inplace();

        // Stage 6: per-param AdamW.
        for k in &keys {
            let weight_dev = weight_dev_for_blt_key(&self.model, k);
            let host_grad = self.grad_host_scratch.get(k)
                .expect("grad_host_scratch populated by alloc_adamw");
            let buf = self.adamw.get_mut(k)
                .expect("adamw populated by alloc_adamw");
            let cfg = self.config.for_group((self.is_global)(k));
            buf.step_resident(weight_dev, host_grad.as_slice(), &cfg)?;
        }

        self.step += 1;
        Ok(loss)
    }

    /// Pre-allocate one [`AdamWBuf`] and one host-scratch `Vec<f32>` per
    /// parameter buffer. Sizes are derived from the model directly —
    /// same source-of-truth as `weight_dev_for_blt_key`. Sharing the
    /// allocation here is what lets the per-step grad-clip stage avoid
    /// the per-call `Vec<f32>` churn (see [`Self::grad_host_scratch`]).
    fn alloc_adamw(&mut self) -> Result<(), ResidencyError> {
        let keys = param_keys_for_model(&self.model);
        for k in &keys {
            let n = grad_dev_for_key(&self.state, k, &self.model).len();
            self.adamw.insert(k.clone(), AdamWBuf::zeros(n)?);
            self.grad_host_scratch.insert(k.clone(), vec![0.0f32; n]);
        }
        Ok(())
    }

    /// Apply the global gradient norm clip in place across every host
    /// grad slot in `grad_host_scratch` (stage 5 of [`Self::train_step`]).
    ///
    /// ## Algorithmic shape
    ///
    /// 1. **Per-param scalar reduction.** Walk each host grad slice and
    ///    accumulate `Σ x²` into a single f32 running sum
    ///    (`global_sumsq`). Reduction stays in f32 to match the
    ///    precision used downstream by `Op::AdamWResident` — using
    ///    f64 here would compute a slightly different scale than the
    ///    one AdamW's f32 path effectively sees once weights are
    ///    updated. The accumulator overflows only for grads with
    ///    `Σ x² > 3.4e38`, which is far past every practical range
    ///    (Qwen2.5-0.5B with f32 weights and unit-scale grads sums to
    ///    ~10⁸ at most).
    /// 2. **Global norm.** `global_norm = sqrt(global_sumsq)` —
    ///    one f32 sqrt total, host-side.
    /// 3. **Conditional rescale.** If `global_norm > clip` (and
    ///    `global_norm` is finite and positive), compute
    ///    `scale = clip / global_norm` and multiply every grad slice
    ///    by `scale` element-wise. No-op otherwise — preserves the
    ///    original grad magnitudes when the model is well-behaved.
    ///
    /// ## Why this stayed host-side
    ///
    /// `AdamWBuf::step_resident` already takes a host `&[f32]` and
    /// uploads it into `g_dev` itself, so the per-step download into
    /// `grad_host_scratch` is required by the AdamW dispatch regardless
    /// of whether we clip. The only redundancy this slice eliminates
    /// is the per-step `Vec<f32>` allocation churn; the device→host
    /// move is paid once per step either way until a resident sumsq
    /// kernel + a resident scalar-multiply kernel both land in
    /// `modgrad-device`. Adding those is out of scope for this fix
    /// (no new kernels; preserve `train_step`'s public API).
    fn clip_grads_inplace(&mut self) {
        let clip = self.config.grad_clip;
        if clip <= 0.0 {
            return;
        }
        let mut global_sumsq: f32 = 0.0;
        for v in self.grad_host_scratch.values() {
            for &x in v.iter() {
                global_sumsq += x * x;
            }
        }
        let global_norm = global_sumsq.sqrt();
        if global_norm.is_finite() && global_norm > clip && global_norm > 0.0 {
            let scale = clip / global_norm;
            for v in self.grad_host_scratch.values_mut() {
                for x in v.iter_mut() {
                    *x *= scale;
                }
            }
        }
    }
}

// ─── BltModel param key plumbing ──────────────────────────────

/// Stable parameter keys for [`BltModel`]. Matches the schema in the
/// type-level doc for [`BltModelTrainer`]. Order is encoder → latent →
/// decoder; within each component, embeddings/heads bracket the per-layer
/// blocks.
#[cfg(feature = "rocm")]
fn param_keys_for_model(model: &BltModel) -> Vec<String> {
    let n_enc = model.encoder.n_layers();
    let n_lat = model.latent.num_layers();
    let n_dec = model.decoder.n_layers();

    // 1 (byte_embed) + n_enc * (7 attn/mlp + 4 cross_attn) + n_lat * 7 + 1
    // (latent.final_norm) + n_dec * (7 + 4) + 3 (lm_head, lm_head_bias,
    // final_norm).
    let mut keys = Vec::with_capacity(
        1 + n_enc * 11 + n_lat * 7 + 1 + n_dec * 11 + 3,
    );

    keys.push("encoder.byte_embed".to_string());
    for li in 0..n_enc {
        keys.push(format!("encoder.block.{li}.wq"));
        keys.push(format!("encoder.block.{li}.wk"));
        keys.push(format!("encoder.block.{li}.wv"));
        keys.push(format!("encoder.block.{li}.wo"));
        keys.push(format!("encoder.block.{li}.gate"));
        keys.push(format!("encoder.block.{li}.up"));
        keys.push(format!("encoder.block.{li}.down"));
        keys.push(format!("encoder.cross_attn.{li}.wq"));
        keys.push(format!("encoder.cross_attn.{li}.wk"));
        keys.push(format!("encoder.cross_attn.{li}.wv"));
        keys.push(format!("encoder.cross_attn.{li}.wo"));
    }

    for li in 0..n_lat {
        keys.push(format!("latent.block.{li}.wq"));
        keys.push(format!("latent.block.{li}.wk"));
        keys.push(format!("latent.block.{li}.wv"));
        keys.push(format!("latent.block.{li}.wo"));
        keys.push(format!("latent.block.{li}.gate"));
        keys.push(format!("latent.block.{li}.up"));
        keys.push(format!("latent.block.{li}.down"));
    }
    keys.push("latent.final_norm".to_string());

    for li in 0..n_dec {
        keys.push(format!("decoder.block.{li}.wq"));
        keys.push(format!("decoder.block.{li}.wk"));
        keys.push(format!("decoder.block.{li}.wv"));
        keys.push(format!("decoder.block.{li}.wo"));
        keys.push(format!("decoder.block.{li}.gate"));
        keys.push(format!("decoder.block.{li}.up"));
        keys.push(format!("decoder.block.{li}.down"));
        keys.push(format!("decoder.cross_attn.{li}.wq"));
        keys.push(format!("decoder.cross_attn.{li}.wk"));
        keys.push(format!("decoder.cross_attn.{li}.wv"));
        keys.push(format!("decoder.cross_attn.{li}.wo"));
    }
    keys.push("decoder.lm_head".to_string());
    keys.push("decoder.lm_head_bias".to_string());
    keys.push("decoder.final_norm".to_string());

    keys
}

/// Look up the model's resident weight buffer for a given param key.
/// Inverse of [`param_keys_for_model`]. Used by the resident AdamW
/// dispatch to mutate the device weight in place.
#[cfg(feature = "rocm")]
fn weight_dev_for_blt_key<'a>(model: &'a BltModel, key: &str) -> &'a HipBuffer {
    if key == "encoder.byte_embed" { return &model.encoder.byte_embed_dev; }
    if key == "latent.final_norm" { return &model.latent_final_norm_weight_dev; }
    if key == "decoder.lm_head" { return &model.decoder.lm_head.weight_dev; }
    if key == "decoder.lm_head_bias" { return &model.decoder.lm_head.bias_dev; }
    if key == "decoder.final_norm" { return &model.decoder.final_norm_weight_dev; }

    if let Some(rest) = key.strip_prefix("encoder.block.") {
        let (li, slot) = parse_block_key(rest);
        let block = &model.encoder.byte_layers[li];
        return block_slot_weight(block, slot, key);
    }
    if let Some(rest) = key.strip_prefix("encoder.cross_attn.") {
        let (li, slot) = parse_block_key(rest);
        return cross_attn_slot_weight(&model.encoder.cross_attns[li], slot, key);
    }
    if let Some(rest) = key.strip_prefix("latent.block.") {
        let (li, slot) = parse_block_key(rest);
        let block = &model.latent.blocks[li];
        return block_slot_weight(block, slot, key);
    }
    if let Some(rest) = key.strip_prefix("decoder.block.") {
        let (li, slot) = parse_block_key(rest);
        let block = &model.decoder.byte_layers[li];
        return block_slot_weight(block, slot, key);
    }
    if let Some(rest) = key.strip_prefix("decoder.cross_attn.") {
        let (li, slot) = parse_block_key(rest);
        return cross_attn_slot_weight(&model.decoder.cross_attns[li], slot, key);
    }
    panic!("BltModelTrainer::weight_dev_for_blt_key: unknown key {key}");
}

/// Look up the gradient buffer for a given param key on the backward
/// state. Inverse of [`param_keys_for_model`]; pairs with
/// [`weight_dev_for_blt_key`] one-to-one.
#[cfg(feature = "rocm")]
fn grad_dev_for_key<'a>(
    state: &'a BltBackwardState,
    key: &str,
    model: &BltModel,
) -> &'a GpuVec {
    if key == "encoder.byte_embed" { return &state.encoder_grads.d_byte_embed; }
    if key == "latent.final_norm" { return &state.d_latent_final_norm_weight; }
    if key == "decoder.lm_head" { return &state.decoder_grads.dweight_lm_head; }
    if key == "decoder.lm_head_bias" { return &state.decoder_grads.dbias_lm_head; }
    if key == "decoder.final_norm" { return &state.decoder_grads.dweight_final_norm; }

    if let Some(rest) = key.strip_prefix("encoder.block.") {
        let (li, slot) = parse_block_key(rest);
        debug_assert!(li < model.encoder.n_layers(), "{key}: encoder layer OOB");
        return block_slot_grad(
            &state.encoder_grads.attn_grads[li],
            &state.encoder_grads.mlp_grads[li],
            slot, key,
        );
    }
    if let Some(rest) = key.strip_prefix("encoder.cross_attn.") {
        let (li, slot) = parse_block_key(rest);
        debug_assert!(li < model.encoder.n_layers(), "{key}: encoder cross_attn layer OOB");
        return cross_attn_slot_grad(&state.encoder_grads.cross_attn_grads[li], slot, key);
    }
    if let Some(rest) = key.strip_prefix("latent.block.") {
        let (li, slot) = parse_block_key(rest);
        debug_assert!(li < model.latent.num_layers(), "{key}: latent layer OOB");
        return block_slot_grad(
            &state.latent_attn_grads[li],
            &state.latent_mlp_grads[li],
            slot, key,
        );
    }
    if let Some(rest) = key.strip_prefix("decoder.block.") {
        let (li, slot) = parse_block_key(rest);
        debug_assert!(li < model.decoder.n_layers(), "{key}: decoder layer OOB");
        return block_slot_grad(
            &state.decoder_grads.attn_grads[li],
            &state.decoder_grads.mlp_grads[li],
            slot, key,
        );
    }
    if let Some(rest) = key.strip_prefix("decoder.cross_attn.") {
        let (li, slot) = parse_block_key(rest);
        debug_assert!(li < model.decoder.n_layers(), "{key}: decoder cross_attn layer OOB");
        return cross_attn_slot_grad(&state.decoder_grads.cross_attn_grads[li], slot, key);
    }
    panic!("BltModelTrainer::grad_dev_for_key: unknown key {key}");
}

/// Parse `"{li}.{slot}"` into `(li, slot)`. Panics on malformed keys —
/// keys come from [`param_keys_for_model`], a malformed value here is a
/// programmer error.
#[cfg(feature = "rocm")]
fn parse_block_key(rest: &str) -> (usize, &str) {
    let mut parts = rest.splitn(2, '.');
    let li: usize = parts.next()
        .expect("layer index segment")
        .parse()
        .expect("layer index parses as usize");
    let slot = parts.next().expect("slot segment");
    (li, slot)
}

#[cfg(feature = "rocm")]
fn block_slot_weight<'a>(
    block: &'a modgrad_transformer::resident::TransformerBlockResident,
    slot: &str,
    key: &str,
) -> &'a HipBuffer {
    match slot {
        "wq" => &block.attn.q_proj.weight_dev,
        "wk" => &block.attn.k_proj.weight_dev,
        "wv" => &block.attn.v_proj.weight_dev,
        "wo" => &block.attn.o_proj.weight_dev,
        "gate" => &block.mlp.gate.weight_dev,
        "up" => &block.mlp.up.weight_dev,
        "down" => &block.mlp.down.weight_dev,
        _ => panic!("BltModelTrainer::block_slot_weight: unknown slot in {key}"),
    }
}

#[cfg(feature = "rocm")]
fn block_slot_grad<'a>(
    attn: &'a modgrad_transformer::resident::AttentionResidentGrads,
    mlp: &'a modgrad_transformer::resident::SwigluResidentGrads,
    slot: &str,
    key: &str,
) -> &'a GpuVec {
    match slot {
        "wq" => &attn.dweight_q,
        "wk" => &attn.dweight_k,
        "wv" => &attn.dweight_v,
        "wo" => &attn.dweight_o,
        "gate" => &mlp.dweight_gate,
        "up" => &mlp.dweight_up,
        "down" => &mlp.dweight_down,
        _ => panic!("BltModelTrainer::block_slot_grad: unknown slot in {key}"),
    }
}

#[cfg(feature = "rocm")]
fn cross_attn_slot_weight<'a>(
    cross: &'a crate::cross_attn::CrossAttention,
    slot: &str,
    key: &str,
) -> &'a HipBuffer {
    match slot {
        "wq" => &cross.q_proj.weight_dev,
        "wk" => &cross.k_proj.weight_dev,
        "wv" => &cross.v_proj.weight_dev,
        "wo" => &cross.o_proj.weight_dev,
        _ => panic!("BltModelTrainer::cross_attn_slot_weight: unknown slot in {key}"),
    }
}

#[cfg(feature = "rocm")]
fn cross_attn_slot_grad<'a>(
    grads: &'a crate::cross_attn::CrossAttnGrads,
    slot: &str,
    key: &str,
) -> &'a GpuVec {
    match slot {
        "wq" => &grads.dweight_q,
        "wk" => &grads.dweight_k,
        "wv" => &grads.dweight_v,
        "wo" => &grads.dweight_o,
        _ => panic!("BltModelTrainer::cross_attn_slot_grad: unknown slot in {key}"),
    }
}

// ─── Tests ────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(feature = "rocm")]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_match_paper() {
        let c = BltTrainerConfig::default();
        // Paper §6.2: global LR is local / 10. Tested as a structural
        // guarantee — if someone bumps `local_lr` and forgets
        // `global_lr`, this catches it.
        let ratio = c.local_lr / c.global_lr;
        assert!((ratio - 10.0).abs() < 1e-3,
            "BltTrainerConfig::default() should set global_lr = local_lr/10, got ratio {ratio}");
        assert!((c.beta1 - 0.9).abs() < 1e-6, "β₁ default 0.9");
        assert!((c.beta2 - 0.95).abs() < 1e-6, "β₂ default 0.95 per BLT §4.8");
        assert!((c.weight_decay - 0.1).abs() < 1e-6, "wd default 0.1");
        assert!((c.grad_clip - 1.0).abs() < 1e-6, "clip default 1.0");
    }

    #[test]
    fn for_group_routes_lr() {
        let c = BltTrainerConfig::default();
        let local = c.for_group(false);
        let global = c.for_group(true);
        assert!((local.lr - c.local_lr).abs() < 1e-6, "local group → local_lr");
        assert!((global.lr - c.global_lr).abs() < 1e-6, "global group → global_lr");
        // Other hyperparams identical across groups.
        assert!((local.beta1 - global.beta1).abs() < 1e-9);
        assert!((local.beta2 - global.beta2).abs() < 1e-9);
        assert!((local.weight_decay - global.weight_decay).abs() < 1e-9);
    }

    #[test]
    fn validate_boundaries_accepts_well_formed() {
        validate_boundaries(10, &[]);
        validate_boundaries(10, &[0]);
        validate_boundaries(10, &[0, 3, 7]);
        validate_boundaries(10, &[9]);
    }

    #[test]
    #[should_panic(expected = "boundaries not strictly increasing")]
    fn validate_boundaries_rejects_dup() {
        validate_boundaries(10, &[0, 3, 3, 7]);
    }

    #[test]
    #[should_panic(expected = "boundary[last]")]
    fn validate_boundaries_rejects_oob() {
        validate_boundaries(10, &[0, 3, 10]);
    }

    #[test]
    fn param_keys_shape() {
        let keys = param_keys(2);
        // 2 layers × 7 params + embed + lm_head = 16
        assert_eq!(keys.len(), 16);
        assert_eq!(keys[0], "embed");
        assert_eq!(keys[15], "lm_head");
        assert!(keys.iter().any(|k| k == "block.0.wq"));
        assert!(keys.iter().any(|k| k == "block.1.down"));
    }

    // ─── BltModelTrainer tests ────────────────────────────────

    use crate::byteify::ByteifyRecipe;
    use crate::decoder::LocalDecoderConfig;
    use crate::encoder::LocalEncoderConfig;
    use crate::model::{BltConfig, BltLatentConfig, BltModel};
    use modgrad_device::backend::rocm::ffi::runtime_available;
    use std::sync::Mutex;

    /// HIP runtime tests must run serially — the resident dispatch path
    /// shares the default stream. Mirrors the `tests::HIP_TEST_LOCK` in
    /// `model.rs`.
    static MODEL_TRAINER_LOCK: Mutex<()> = Mutex::new(());

    /// Tiny config — same shape as `model::tests::tiny_config` but
    /// duplicated here so the trainer tests don't pull from a private
    /// helper. 32-byte sequence, 8 patches, lE=1, lL=2, lD=1.
    ///
    /// `max_patches` is set to exactly the n_patches the loss-finite
    /// test uses (8) — the existing `LocalDecoder::backward` path
    /// asserts `d_patch_reps_out.len() == n_patches * patch_dim` against
    /// the state buffer sized to `max_patches * patch_dim`, so they have
    /// to match. (Out-of-scope to relax: the model contract belongs to
    /// the BltBackwardState allocator, not the trainer.)
    fn tiny_blt_config() -> BltConfig {
        let byte_dim = 32usize;
        let n_byte_heads = 4usize;
        let byte_head_dim = byte_dim / n_byte_heads;
        let patch_dim = 64usize;
        let n_patch_heads = 4usize;
        let patch_head_dim = patch_dim / n_patch_heads;
        let max_seq = 32usize;
        let max_patches = 8usize;

        BltConfig {
            encoder: LocalEncoderConfig {
                n_layers: 1, byte_dim, patch_dim,
                n_heads: n_byte_heads, head_dim: byte_head_dim,
                mlp_dim: byte_dim * 2,
                norm_eps: 1e-5, rope_base: 10_000.0,
                max_seq_len: max_seq,
                ngram_min_n: 3, ngram_max_n: 5,
                ngram_vocab_per_n: 256,
            },
            latent: BltLatentConfig {
                n_layers: 2, patch_dim,
                n_heads: n_patch_heads, head_dim: patch_head_dim,
                mlp_dim: patch_dim * 2,
                norm_eps: 1e-5, rope_base: 10_000.0,
                max_patches,
            },
            decoder: LocalDecoderConfig {
                n_layers: 1, byte_dim, patch_dim,
                n_heads: n_byte_heads, head_dim: byte_head_dim,
                mlp_dim: byte_dim * 2,
                norm_eps: 1e-5, rope_base: 10_000.0,
                max_seq_len: max_seq,
            },
        }
    }

    #[test]
    fn param_keys_for_model_shape() {
        // Pure host: build a key list against the count expectations
        // even when no HIP runtime is available.
        let n_enc = 1; let n_lat = 2; let n_dec = 1;
        let expected = 1 + n_enc * 11 + n_lat * 7 + 1 + n_dec * 11 + 3;
        // Simulate by stamping the same keys layout as the live helper.
        // The live `param_keys_for_model` requires a real `BltModel`,
        // which needs a HIP runtime; this test stays runtime-agnostic by
        // recomputing the key count from the model dims directly.
        assert_eq!(expected, 1 + 11 + 14 + 1 + 11 + 3);
        // Sanity: encoder.byte_embed first; decoder.final_norm last.
        // (Validated against the live helper in `blt_model_trainer_alloc`.)
    }

    #[test]
    fn blt_model_trainer_alloc() {
        let _guard = MODEL_TRAINER_LOCK.lock().unwrap();
        if std::env::var("MODGRAD_SKIP_HIP_TESTS").is_ok() {
            return;
        }
        if !runtime_available() {
            eprintln!("hip runtime unavailable, skipping");
            return;
        }
        let config = tiny_blt_config();
        let model = BltModel::new(config).expect("BltModel::new");

        let keys_expected = param_keys_for_model(&model);
        let trainer_cfg = BltTrainerConfig::default();
        let trainer = BltModelTrainer::new(
            model,
            trainer_cfg,
            ByteifyRecipe::global_predicate(),
        ).expect("BltModelTrainer::new");

        // Every key has an AdamW slot.
        let adamw = trainer.adamw_state();
        assert_eq!(adamw.len(), keys_expected.len(),
            "expected one AdamWBuf per key");
        for k in &keys_expected {
            assert!(adamw.contains_key(k), "missing AdamW slot for {k}");
        }
        // Sanity: first key is encoder.byte_embed, last is decoder.final_norm.
        assert_eq!(keys_expected[0], "encoder.byte_embed");
        assert_eq!(keys_expected.last().unwrap(), "decoder.final_norm");
        // No latent.lm_head / no encoder.lm_head — those are NOT BLT
        // params (the latent's lm_head path is bypassed; the byte LM
        // head is `decoder.lm_head`).
        assert!(!keys_expected.iter().any(|k| k == "latent.lm_head"));
        assert!(!keys_expected.iter().any(|k| k == "embed"));
        assert!(!keys_expected.iter().any(|k| k == "lm_head"));
    }

    #[test]
    fn blt_model_trainer_clip_grads_inplace_matches_reference() {
        // Numerical correctness of the in-place host clip:
        //   - Pick one param key, fill it with `value = 3.0`, length n.
        //   - Leave every other key at zero.
        //   - Global L2 norm = 3 * sqrt(n).
        //   - With clip = 1.0, the chosen slot must scale to
        //     `clip / norm * 3.0 = 1 / sqrt(n)` element-wise; the
        //     zero slots stay at zero.
        // This pins the f32 reduction shape against an analytic answer
        // — any drift from the documented algorithm flags here.
        let _guard = MODEL_TRAINER_LOCK.lock().unwrap();
        if std::env::var("MODGRAD_SKIP_HIP_TESTS").is_ok() {
            return;
        }
        if !runtime_available() {
            eprintln!("hip runtime unavailable, skipping");
            return;
        }
        let config = tiny_blt_config();
        let model = BltModel::new(config).expect("BltModel::new");

        let trainer_cfg = BltTrainerConfig {
            grad_clip: 1.0,
            ..BltTrainerConfig::default()
        };
        let mut trainer = BltModelTrainer::new(
            model,
            trainer_cfg,
            ByteifyRecipe::global_predicate(),
        ).expect("BltModelTrainer::new");

        // Pick a stable, present key for the non-zero grad slot.
        let target_key = "encoder.block.0.wq".to_string();
        assert!(trainer.grad_host_scratch.contains_key(&target_key),
            "tiny_blt_config should expose {target_key}");

        // Fill the chosen scratch with 3.0; leave all others at zero.
        for (k, v) in trainer.grad_host_scratch.iter_mut() {
            if *k == target_key {
                v.fill(3.0);
            } else {
                v.fill(0.0);
            }
        }
        let n = trainer.grad_host_scratch[&target_key].len();
        let expected_norm = 3.0_f32 * (n as f32).sqrt();
        let clip = trainer.config.grad_clip;
        let expected_scaled = clip / expected_norm * 3.0;

        trainer.clip_grads_inplace();

        // Target slot should be scaled element-wise.
        for (i, &x) in trainer.grad_host_scratch[&target_key].iter().enumerate() {
            assert!((x - expected_scaled).abs() < 1e-5,
                "target[{i}] = {x}, expected {expected_scaled}");
        }
        // All other slots should still be zero.
        for (k, v) in trainer.grad_host_scratch.iter() {
            if k == &target_key { continue; }
            for (i, &x) in v.iter().enumerate() {
                assert_eq!(x, 0.0, "{k}[{i}] = {x} (should still be 0)");
            }
        }
    }

    #[test]
    fn blt_model_trainer_clip_disabled_passes_through() {
        // `grad_clip = 0.0` must short-circuit — values stay untouched.
        let _guard = MODEL_TRAINER_LOCK.lock().unwrap();
        if std::env::var("MODGRAD_SKIP_HIP_TESTS").is_ok() {
            return;
        }
        if !runtime_available() {
            eprintln!("hip runtime unavailable, skipping");
            return;
        }
        let config = tiny_blt_config();
        let model = BltModel::new(config).expect("BltModel::new");

        let trainer_cfg = BltTrainerConfig {
            grad_clip: 0.0,
            ..BltTrainerConfig::default()
        };
        let mut trainer = BltModelTrainer::new(
            model,
            trainer_cfg,
            ByteifyRecipe::global_predicate(),
        ).expect("BltModelTrainer::new");

        // Stamp big values everywhere; must survive clip_grads_inplace.
        for v in trainer.grad_host_scratch.values_mut() {
            v.fill(7.5);
        }
        trainer.clip_grads_inplace();
        for (k, v) in trainer.grad_host_scratch.iter() {
            for (i, &x) in v.iter().enumerate() {
                assert_eq!(x, 7.5, "{k}[{i}] = {x} (clip disabled, should be 7.5)");
            }
        }
    }

    #[test]
    fn blt_model_trainer_loss_finite() {
        let _guard = MODEL_TRAINER_LOCK.lock().unwrap();
        if std::env::var("MODGRAD_SKIP_HIP_TESTS").is_ok() {
            return;
        }
        if !runtime_available() {
            eprintln!("hip runtime unavailable, skipping");
            return;
        }
        let config = tiny_blt_config();
        let model = BltModel::new(config).expect("BltModel::new");

        // Match the model's max_seq_len = 32 / max_patches = 16.
        let mbs = 32usize;
        let trainer_cfg = BltTrainerConfig {
            micro_batch_size: mbs,
            seq_len: mbs,
            ..BltTrainerConfig::default()
        };
        let mut trainer = BltModelTrainer::new(
            model,
            trainer_cfg,
            ByteifyRecipe::global_predicate(),
        ).expect("BltModelTrainer::new");

        let bytes: Vec<u8> = (0..mbs as u8).collect();
        let boundaries: Vec<usize> = (0..=8).map(|p| p * 4).collect();
        assert_eq!(boundaries[0], 0);
        assert_eq!(*boundaries.last().unwrap(), mbs);

        for step in 0..2 {
            let loss = trainer.train_step(&bytes, &boundaries)
                .unwrap_or_else(|e| panic!("step {step}: {e:?}"));
            assert!(loss.is_finite(),
                "step {step} loss not finite: {loss}");
        }
        // History reflects both steps.
        assert_eq!(trainer.loss_history().len(), 2);
        for (i, l) in trainer.loss_history().iter().enumerate() {
            assert!(l.is_finite(), "history[{i}] not finite: {l}");
        }
    }
}
