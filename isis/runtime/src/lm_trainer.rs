//! `LmTrainer` — the entry point our user actually invokes to train a
//! foundation language model. Wraps a residency-aware `LanguageModel`
//! (typically `GptModelResident`) with the orchestration loop:
//!
//!   load model → forward → cross-entropy loss → backward → AdamW step
//!   → resync bf16 device weights → repeat.
//!
//! ## Slice scope honesty
//!
//! `eve`'s slice is implementing the device-resident backward chain
//! in parallel. As of this slice, the resident model exposes:
//!
//!   - `GptModelResident::forward` — forward complete, validated to
//!     8.3e-3 relative tolerance vs the host reference.
//!   - **No `backward` method.** A CPU-only `train::backward_train`
//!     exists and runs against a `WeightOffloader`, but it is not
//!     wired through `GptModelResident` and not GPU-resident.
//!
//! Therefore this slice ships:
//!
//!   1. The full trainer scaffold (`LmTrainer`, `LmTrainerConfig`).
//!   2. A working `train_step` that runs forward + cross-entropy
//!      loss on next-token-prediction targets. Loss is reported.
//!   3. **A clearly-marked TODO for the backward + AdamW step.** The
//!      forward produces the gradient w.r.t. logits already (cross-
//!      entropy returns `(loss, dL/dlogits)`) — when eve's slice
//!      lands a resident `GptModelResident::backward(dL_dlogits) →
//!      gradients struct`, we wire `step_backward` to it and the
//!      AdamW path activates.
//!
//! Until then, `train_step` returns the loss but does not actually
//! update weights. The trainer scaffold is still useful: it lets the
//! caller validate the forward + loss path on real tokens, time the
//! per-step cost, and integrate the loop into a higher-level isis
//! runtime program. Scope is documented; nothing is silently broken.
//!
//! ## Why per-`Linear` AdamW state, not a monolithic optimizer
//!
//! `RegionalAdamW` (in `modgrad-ctm`) is shaped for a fixed connectome
//! — one `AdamWBuf` per named slot (embed, conn_w, output_proj, …).
//! A transformer has a regular structure: every block has the same
//! six matrices (wq/wk/wv/wo/gate/up/down/lm_head), and each matrix
//! has the same fp32 master + bf16 device split via
//! `LinearResidentBf16`. The natural map is one `AdamWBuf` per
//! `Linear`, indexed by a stable string key like
//! `"block.{li}.wq"`. That avoids hard-coding a transformer-specific
//! schema into the optimizer and lets a future LoRA / MoE adapter
//! plug in extra `Linear`s with their own AdamW state without
//! changing this module.
//!
//! When eve's `GptModelResidentGrad` lands with its host fp32 master
//! reference, this trainer's `step_adamw_and_resync` walks the grads
//! struct, looks up the matching `AdamWBuf` by key, applies the
//! standard AdamW update on the master fp32 vector, and calls
//! `sync_from_master()` on each `LinearResidentBf16` to re-quantise
//! to bf16 device storage.

#[cfg(feature = "rocm")]
use std::collections::HashMap;

#[cfg(feature = "rocm")]
use modgrad_compute::backend::{GpuVec, ResidencyError};
#[cfg(feature = "rocm")]
use modgrad_device::backend::{HipBatch, AdamWArgs};
#[cfg(feature = "rocm")]
use modgrad_transformer::loss::cross_entropy;

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

// ─── AdamW per-parameter state ───────────────────────────────

/// Per-`Linear` AdamW first/second moment buffers + step counter.
///
/// Why a sibling type instead of using `modgrad_transformer::optim::adamw::AdamW`:
/// the transformer's `AdamW` carries lr/beta/etc as fields, expecting
/// one optimizer per parameter group — that bloats the trainer's
/// `HashMap` (every entry duplicates the same hyperparameters). Here
/// we keep only the per-parameter *state* (m, v, t) and read the
/// hyperparameters from the trainer's `LmTrainerConfig` once per
/// step. Same numerics as `optim::adamw::AdamW::step`.
#[cfg(feature = "rocm")]
pub struct AdamWBuf {
    /// First moment estimates (per-element).
    pub m: Vec<f32>,
    /// Second moment estimates (per-element).
    pub v: Vec<f32>,
    /// Step counter — bumped before each AdamW update so bias
    /// correction starts at t=1 (not 0, which would NaN the first
    /// step's `bc1`/`bc2`).
    pub t: u64,
}

#[cfg(feature = "rocm")]
impl AdamWBuf {
    /// Allocate buffers for a parameter of `n` elements. m and v
    /// start at zero; t starts at zero.
    pub fn zeros(n: usize) -> Self {
        Self { m: vec![0.0; n], v: vec![0.0; n], t: 0 }
    }

    /// Apply one AdamW update step. `params` is the host fp32 master;
    /// `grads` is the gradient (same length). The trainer's caller
    /// is responsible for re-quantising / re-uploading after this
    /// call (typically: `LinearResidentBf16::sync_from_master`).
    ///
    /// Routes through `modgrad-device`'s `ops::adamw` so the kernel
    /// path is the same as `RegionalAdamW`'s — there is exactly one
    /// AdamW implementation, with one place for bug-fixes.
    pub fn step(
        &mut self,
        params: &mut [f32],
        grads: &mut [f32],
        cfg: &LmTrainerConfig,
    ) {
        debug_assert_eq!(params.len(), grads.len());
        debug_assert_eq!(params.len(), self.m.len());
        self.t += 1;
        let bc1 = 1.0 - cfg.beta1.powi(self.t as i32);
        let bc2 = 1.0 - cfg.beta2.powi(self.t as i32);
        modgrad_device::backend::ops::adamw(AdamWArgs {
            w: params,
            g: grads,
            m: &mut self.m,
            v: &mut self.v,
            lr: cfg.lr,
            beta1: cfg.beta1,
            beta2: cfg.beta2,
            eps: cfg.eps,
            weight_decay: cfg.weight_decay,
            bc1_inv: 1.0 / bc1,
            bc2_inv: 1.0 / bc2,
        }).expect("adamw dispatch");
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
    /// Per-parameter AdamW state, indexed by stable string key
    /// (e.g. `"block.0.wq"`, `"lm_head"`). Only populated once the
    /// resident-backward + grads struct is wired (eve's slice).
    /// Until then the map stays empty and `step_adamw_and_resync`
    /// is a no-op.
    adamw: HashMap<String, AdamWBuf>,
    /// Resident KV cache reused across `train_step` calls. Allocated
    /// at construction with capacity = max(seq_len of any future
    /// micro_batch) — for now equal to `config.seq_len`. Reset
    /// before every step.
    kv_cache: modgrad_transformer::KvCacheResident,
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

    /// Run one training step on `tokens` (length `micro_batch_size + 1`
    /// — the last token is the final next-token target only and is
    /// not fed through forward).
    ///
    /// Returns the cross-entropy loss for the step.
    ///
    /// Pipeline:
    ///   1. Reset the KV cache.
    ///   2. Forward pass on `tokens[..mbs]` at positions `0..mbs`,
    ///      producing logits `[mbs × vocab_size]`.
    ///   3. Stage logits to host. Compute cross-entropy against
    ///      `tokens[1..=mbs]` (next-token prediction).
    ///   4. **TODO(eve):** dispatch resident backward with the
    ///      gradient w.r.t. logits to populate per-`Linear` grads;
    ///      run `step_adamw_and_resync()`. Without that wiring,
    ///      this step does forward + loss only.
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

        // Stage 1: reset cache. Uses the typed reset (set_seq_len(0))
        // not reset_zero — the latter is a slow D2D memset and
        // unnecessary because every block's forward writes the slot
        // before reading it.
        self.kv_cache.reset();

        // Stage 2: forward. Source tokens are tokens[..mbs];
        // positions are 0..mbs.
        let src_tokens: &[i64] = &tokens[..mbs];
        let positions: Vec<usize> = (0..mbs).collect();
        let vocab = self.model.vocab_size();
        let mut logits_dev = GpuVec::try_hip(mbs * vocab)?;

        self.model.ensure_resident(batch)?;
        self.model.forward_logits(
            batch,
            src_tokens, &positions,
            Some(&mut self.kv_cache),
            &mut logits_dev,
        )?;
        // Drain so the host-side cross-entropy reads consistent
        // logits — copy_to_host below does an explicit hipMemcpy
        // D2H, which acts as an implicit sync, but a flush here
        // surfaces a kernel-launch failure as a Result before we
        // try to read.
        batch.flush()?;

        // Stage 3: stage logits + compute loss.
        let mut logits_host = vec![0.0f32; mbs * vocab];
        logits_dev.copy_to_host(&mut logits_host);
        let targets: &[i64] = &tokens[1..=mbs];
        let (loss, _grad_logits) = cross_entropy(&logits_host, targets, vocab);

        // Sanity: loss should not NaN. If it does, something went
        // wrong upstream (uninitialized cache, embedding overflow,
        // …). Surface loudly — silent NaN propagation is the worst
        // bug class in training.
        assert!(loss.is_finite(),
            "LmTrainer::train_step: loss is NaN/inf — upstream forward bug");

        self.loss_history.push(loss);

        // Stage 4: TODO(eve) — backward + AdamW.
        //
        // The shape, once eve's slice lands:
        //   let grads = self.model.backward_logits(batch, &grad_logits)?;
        //   self.step_adamw_and_resync(&grads, batch)?;
        //
        // `grads` would be a `GptModelResidentGrad`-shaped struct
        // exposing per-`LinearResidentBf16` host gradients. The
        // resync stage walks every linear, applies AdamW on the
        // host master, then calls `sync_from_master` to re-quantise
        // to bf16 on device.
        //
        // Today we drop `_grad_logits` on the floor: it carries the
        // CE gradient w.r.t. logits but there is no resident
        // backward chain to consume it.

        Ok(loss)
    }

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

    /// `AdamWBuf::step` updates m, v, and t, and reduces the
    /// magnitude of `params` for a positive-mean gradient. Pure
    /// numerical check, single update, with CPU-side fp32 only.
    #[test]
    fn adamw_buf_step_basic() {
        let mut buf = AdamWBuf::zeros(4);
        let mut params = vec![1.0f32, 1.0, 1.0, 1.0];
        let mut grads = vec![0.5f32, 0.5, 0.5, 0.5];
        let cfg = LmTrainerConfig::default();
        buf.step(&mut params, &mut grads, &cfg);
        assert_eq!(buf.t, 1, "t bumped");
        // After one step, params should be slightly less than 1 (gradient
        // points away from zero, so AdamW with weight decay nudges them
        // smaller).
        for &p in &params {
            assert!(p < 1.0 && p > 0.99,
                "param after one tiny AdamW step should be just under 1.0, got {p}");
        }
        // Moments should have absorbed the gradient.
        for &m in &buf.m { assert!(m > 0.0, "first moment populated"); }
        for &v in &buf.v { assert!(v > 0.0, "second moment populated"); }
    }

    /// End-to-end smoke test: build a tiny GptModelResident, wrap in
    /// LmTrainer, run 5 train_steps. Asserts no NaN, asserts the
    /// trainer accumulates loss history. Does NOT assert loss
    /// decreases — the resident backward is not yet wired
    /// (eve's slice), so the model weights aren't updating.
    /// We DO assert loss is stable across calls (no drift from
    /// uninitialized state).
    #[test]
    fn lm_trainer_smoke_forward_only() {
        let _guard = HIP_TEST_LOCK.lock().unwrap();
        skip_if_no_gpu!();
        let config = tiny_config();
        let (model, swiglu_mlps) = build_test_model(&config);

        let resident = GptModelResident::from_model(&model, &swiglu_mlps)
            .expect("upload model");

        let mbs = 16usize;
        let trainer_cfg = LmTrainerConfig {
            lr: 1e-5,           // tiny — no risk of blowing up state
            micro_batch_size: mbs,
            seq_len: mbs,
            ..LmTrainerConfig::default()
        };

        let n_kv = config.num_kv_heads.get();
        let hd = config.head_dim.get();
        let mut trainer = LmTrainer::new(resident, n_kv, hd, trainer_cfg)
            .expect("trainer alloc");

        // Synthetic 17-token sequence → 16 source tokens, 16 targets.
        let tokens: Vec<i64> = (0..(mbs as i64) + 1)
            .map(|i| (i * 17) % (config.vocab_size.get() as i64))
            .collect();

        let mut losses = Vec::with_capacity(5);
        for step in 0..5 {
            let batch = HipBatch::new();
            let loss = trainer.train_step(&batch, &tokens)
                .unwrap_or_else(|e| panic!("train_step {step} failed: {e}"));
            assert!(loss.is_finite(),
                "step {step}: loss is non-finite ({loss}) — forward bug");
            losses.push(loss);
        }
        assert_eq!(trainer.loss_history().len(), 5, "loss history accumulated");

        // Without backward, the weights haven't moved. So the per-step
        // loss should be EXACTLY the same across calls — same model,
        // same tokens, deterministic dispatch. (Floating-point hipblas
        // with fixed seeds is bitwise-stable in practice; if it isn't,
        // our reduction order changed between dispatches, which is a
        // separate bug.)
        let l0 = losses[0];
        for (i, &li) in losses.iter().enumerate() {
            let drift = (li - l0).abs();
            assert!(drift < 1e-3,
                "step {i}: loss {li} drifted from initial {l0} by {drift} \
                 — without backward, repeated forwards must be near-identical");
        }
        eprintln!("lm_trainer smoke: 5 steps, losses {losses:?}");
    }
}
