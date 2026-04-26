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
}
