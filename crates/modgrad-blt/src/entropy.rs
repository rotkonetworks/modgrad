//! Small byte-level autoregressive language model used to estimate
//! next-byte entropy `H(x_i)` for entropy-based patch boundaries.
//!
//! Per BLT paper §2.3 / §4.2: a transformer with ~100M params,
//! 14 layers, hidden 512, sliding-window attention 512 bytes. (We can
//! train smaller variants — paper §7 shows diminishing returns past
//! 50M params.)
//!
//! Architecture-wise this is just a regular small transformer over
//! byte vocab (256). We reuse [`modgrad_transformer::GptModelResident`]
//! configured with `vocab_size = 256`, `tie_embeddings = true`,
//! `use_qk_norm = false`, `mlp_activation = SwiGlu`, RoPE on. The
//! forward returns `[seq_len × 256]` logits; we softmax + entropy on
//! host since entropy patching is offline.
//!
//! ## Entropy units
//!
//! `entropies()` returns entropy in **bits** (`log₂` base). Random-
//! init or near-uniform predictions therefore land near
//! `log₂(256) = 8.0`. Paper-style thresholds (e.g. `θ_g ≈ 1.34`,
//! `θ_r ≈ 0.03`) in nats convert by × `1/ln(2) ≈ 1.4427`; if you are
//! porting numerical thresholds from a nats-based reference, scale
//! them up by `ln(2)`.
//!
//! ## Newline reset
//!
//! Per paper §4.4, the entropy model's KV context should be reset on
//! newline bytes to avoid baseline drift in long contexts. This
//! method handles **one contiguous chunk only** — the caller chunks
//! input on `b'\n'` and calls `entropies()` once per chunk. The KV
//! cache is reset internally at the start of each call.
//!
//! Owner: agent leah (tightly coupled with [`crate::patcher`]).

use serde::{Deserialize, Serialize};

#[cfg(feature = "rocm")]
use modgrad_compute::backend::{GpuVec, ResidencyError};
#[cfg(feature = "rocm")]
use modgrad_device::backend::HipBatch;
#[cfg(feature = "rocm")]
use modgrad_transformer::config::{
    GptConfig, MlpActivation, Precision, ResidualConfig, SmearConfig,
    ValueEmbedConfig, WindowPattern,
};
#[cfg(feature = "rocm")]
use modgrad_transformer::dims::{
    HeadDim, LayerIdx, MlpDim, ModelDim, NumHeads, NumKvHeads, NumLayers,
    SeqLen, VocabSize,
};
#[cfg(feature = "rocm")]
use modgrad_transformer::{GptModelResident, KvCacheResident};

// ─── Public config ────────────────────────────────────────────────

/// Hyperparameters for the byte-level entropy model. Defaults match
/// the small end of the paper's exploration (§4.2): 4 layers,
/// d_model 256, 4 heads × 64 = 256, max_seq_len 512.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntropyConfig {
    /// Hidden / residual stream width. Paper sweeps 128–512.
    pub d_model: usize,
    /// Number of transformer layers.
    pub n_layers: usize,
    /// Number of attention heads. Must satisfy
    /// `n_heads × head_dim = d_model`.
    pub n_heads: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// Sliding-window attention context. Paper uses 512.
    pub max_seq_len: usize,
    /// RMSNorm epsilon.
    pub norm_eps: f32,
    /// RoPE base frequency. 10_000 is the paper-typical choice for a
    /// short-context byte model.
    pub rope_base: f32,
}

impl Default for EntropyConfig {
    fn default() -> Self {
        // Small / cheap default. 4 layers × d_model 256 ≈ 4 M params
        // at vocab 256 with tied embeddings — tiny enough to train as
        // a dataset-side preprocessor.
        Self {
            d_model: 256,
            n_layers: 4,
            n_heads: 4,
            head_dim: 64,
            max_seq_len: 512,
            norm_eps: 1e-5,
            rope_base: 10_000.0,
        }
    }
}

impl EntropyConfig {
    /// Validate dim consistency. `n_heads × head_dim` must equal
    /// `d_model`.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.n_heads * self.head_dim != self.d_model {
            return Err("EntropyConfig: n_heads × head_dim != d_model");
        }
        if self.n_layers == 0 {
            return Err("EntropyConfig: n_layers must be ≥ 1");
        }
        if self.max_seq_len == 0 {
            return Err("EntropyConfig: max_seq_len must be ≥ 1");
        }
        Ok(())
    }
}

// ─── Resident path (--features rocm) ───────────────────────────

/// Byte-level autoregressive transformer that emits per-byte entropy
/// estimates for the BLT patcher.
///
/// Wraps a [`GptModelResident`] keyed for byte vocab (256), tied
/// embeddings, no QK norm, RoPE on. Owns its own
/// [`KvCacheResident`] sized to `config.max_seq_len`. The KV cache
/// is reset at the start of every [`Self::entropies`] call — the
/// caller is expected to chunk input on newlines (or whatever
/// reset boundary is desired) per paper §4.4.
///
/// The host-side logits scratch is grown on demand inside
/// [`Self::entropies`]. The device-side logits buffer is allocated
/// fresh each call because [`GptModelResident::forward`] asserts
/// `logits_out.len() == n × vocab` exactly — there is no view /
/// slice API that would let us share a fixed-size slab. Entropy
/// patching is offline (per paper §2.3) so the per-call
/// `hipMalloc` overhead is irrelevant.
#[cfg(feature = "rocm")]
pub struct EntropyModel {
    inner: GptModelResident,
    kv: KvCacheResident,
    /// Reusable host scratch for logits D2H. Grown on demand.
    logits_host: Vec<f32>,
    config: EntropyConfig,
}

#[cfg(feature = "rocm")]
impl EntropyModel {
    /// Build a fresh (random-init) entropy model. Used when training
    /// the entropy model from scratch on the same data as the BLT
    /// (paper §4.2 trains the byte-LM jointly on the patching corpus).
    ///
    /// Random weights use a deterministic seeded RNG so back-to-back
    /// constructions yield the same model; pass through a re-seeding
    /// step (e.g. via the trainer) if you need different initial
    /// state.
    pub fn new(config: EntropyConfig) -> Result<Self, ResidencyError> {
        config.validate().map_err(|msg| ResidencyError::WrongVariant {
            expected: "valid EntropyConfig", got: msg,
        })?;

        let gpt_cfg = build_byte_level_config(&config);
        let (host_model, swiglu_mlps) = build_random_byte_model(&gpt_cfg);
        let inner = GptModelResident::from_model(&host_model, &swiglu_mlps)?;

        let kv = KvCacheResident::new(
            gpt_cfg.num_layers.get(),
            gpt_cfg.num_kv_heads.get(),
            gpt_cfg.head_dim.get(),
            gpt_cfg.max_seq_len.get(),
            gpt_cfg.model_dim.get(),
        )?;

        Ok(Self {
            inner, kv,
            logits_host: Vec::new(),
            config,
        })
    }

    /// Read-only access to the configured hyperparameters.
    pub fn config(&self) -> &EntropyConfig { &self.config }

    /// Sliding-window context size — the maximum number of bytes a
    /// single [`Self::entropies`] call accepts.
    pub fn max_seq_len(&self) -> usize { self.config.max_seq_len }

    /// Compute next-byte entropy at every position in `bytes`.
    /// `entropies_out[i]` = H(byte_i | bytes[0..i]) in **bits**.
    ///
    /// Specifically: at each position `i`, the model is fed the
    /// causal prefix `bytes[0..i+1]` and the entropy of the
    /// resulting next-byte distribution is written to
    /// `entropies_out[i]`. This matches the paper's convention that
    /// "entropy at position t" is the uncertainty over the byte
    /// observed at that position given everything before it. (For
    /// `i = 0` the prefix is just `bytes[0]` itself; the cache is
    /// fresh and the prediction reflects the base-rate distribution
    /// the model learned for the first byte after a reset.)
    ///
    /// `entropies_out.len()` must equal `bytes.len()`. The caller
    /// owns the chunk boundary policy: call once per
    /// newline-bounded chunk per paper §4.4. Each call resets the
    /// internal KV cache.
    ///
    /// Returns a [`ResidencyError`] if `bytes.len() > max_seq_len`,
    /// or on any underlying HIP failure.
    pub fn entropies(
        &mut self,
        batch: &HipBatch,
        bytes: &[u8],
        entropies_out: &mut [f32],
    ) -> Result<(), ResidencyError> {
        if bytes.len() != entropies_out.len() {
            return Err(ResidencyError::WrongVariant {
                expected: "bytes.len() == entropies_out.len()",
                got: "mismatched lengths",
            });
        }
        if bytes.is_empty() {
            return Ok(());
        }
        if bytes.len() > self.config.max_seq_len {
            return Err(ResidencyError::WrongVariant {
                expected: "bytes.len() <= EntropyConfig::max_seq_len",
                got: "input chunk exceeds sliding-window context",
            });
        }

        let n = bytes.len();
        let vocab = 256usize;
        let token_ids: Vec<i64> = bytes.iter().map(|&b| b as i64).collect();
        let positions: Vec<usize> = (0..n).collect();
        let needed = n * vocab;

        // Fresh KV — caller's chunk is one contiguous reset window.
        self.kv.reset();

        // Grow host scratch if needed. Device scratch is allocated
        // per call to satisfy the forward's exact-length assert.
        if self.logits_host.len() < needed {
            self.logits_host.resize(needed, 0.0);
        }

        let mut logits_dev = GpuVec::try_hip(needed)?;
        self.inner.forward(batch, &token_ids, &positions, &mut self.kv, &mut logits_dev)?;
        // Sync before D2H — `copy_to_host` reads device-resident
        // memory and the documented contract is to flush pending
        // dispatches first (see comment on `HipBatch::flush`).
        batch.flush()?;
        logits_dev.copy_to_host(&mut self.logits_host[..needed]);

        // Per-position softmax + entropy in bits. Numerical recipe:
        //   m  = max_i logits[i]
        //   s  = Σ_i exp(logits[i] - m)
        //   p_i = exp(logits[i] - m) / s
        //   H  = -Σ_i p_i · log₂(p_i)
        //      = log₂(s) + m / ln(2) - Σ_i p_i · logits[i] / ln(2)
        // We use the explicit Σ p log₂ p formula — slightly more
        // arithmetic but no fewer subtractive cancellations.
        let inv_ln2 = std::f32::consts::LOG2_E; // 1 / ln(2) = log₂(e)
        for t in 0..n {
            let row = &self.logits_host[t * vocab..(t + 1) * vocab];
            let mut max = f32::NEG_INFINITY;
            for &v in row {
                if v > max { max = v; }
            }
            let mut sum_exp = 0.0f32;
            for &v in row {
                sum_exp += (v - max).exp();
            }
            let log_sum_exp = max + sum_exp.ln();
            // H_nats = log_sum_exp - Σ p_i · logits[i]
            //       = log_sum_exp - (1/sum_exp) Σ exp(l-m) · logits[i]
            //       = log_sum_exp - (1/sum_exp) Σ exp(l-m) · l
            // numerically: just compute over (l - m) and add the
            // missing m back via log_sum_exp.
            let inv_s = 1.0 / sum_exp;
            let mut weighted = 0.0f32;
            for &v in row {
                let p = (v - max).exp() * inv_s;
                weighted += p * v;
            }
            let h_nats = log_sum_exp - weighted;
            // Clamp at zero — float roundoff can push this slightly
            // negative when one logit dominates. Entropy is
            // mathematically ≥ 0; the patcher's threshold tests
            // assume that.
            let h_bits = (h_nats * inv_ln2).max(0.0);
            entropies_out[t] = h_bits;
        }

        Ok(())
    }
}

// ─── Internal helpers ──────────────────────────────────────────

/// Build the underlying [`GptConfig`] for a byte-level entropy LM.
/// Vocab fixed at 256, embeddings tied, no QK norm, SwiGLU MLP,
/// `mlp_dim = 2 × d_model` (matches the test fixture
/// `build_tiny_model_for_byte_level`).
#[cfg(feature = "rocm")]
fn build_byte_level_config(cfg: &EntropyConfig) -> GptConfig {
    let model_dim = cfg.d_model;
    GptConfig {
        model_dim: ModelDim::new(model_dim),
        num_heads: NumHeads::new(cfg.n_heads),
        num_kv_heads: NumKvHeads::new(cfg.n_heads),
        head_dim: HeadDim::new(cfg.head_dim),
        num_layers: NumLayers::new(cfg.n_layers),
        vocab_size: VocabSize::new(256),
        mlp_dim: MlpDim::new(model_dim * 2),
        max_seq_len: SeqLen::new(cfg.max_seq_len),
        rope_base: cfg.rope_base,
        // No QK norm — paper-class byte LM is plain Llama/Qwen-style
        // attention. `qk_norm_scale` is unused when `use_qk_norm`
        // is false but kept at 1.0 for self-documentation.
        qk_norm_scale: 1.0,
        use_qk_norm: false,
        window_pattern: WindowPattern::Full,
        mlp_activation: MlpActivation::SwiGlu,
        layer_overrides: Vec::new(),
        tie_embeddings: true,
        logit_cap: 0.0,
        recurrent_steps: 1,
        has_exit_gate: false,
        value_embed: ValueEmbedConfig::default(),
        // Flat residual lambdas — the residency path's resid_start /
        // resid_end / x0_* dance is for the foundation-model recipe.
        // For the entropy LM we want a plain transformer.
        residual: ResidualConfig {
            resid_start: 1.0, resid_end: 1.0,
            x0_start: 0.0, x0_end: 0.0,
            backout_lambda: 0.0,
        },
        smear: SmearConfig::default(),
        precision: Precision::F32,
        norm_eps: cfg.norm_eps,
    }
}

/// Construct a host-side [`GptModel`] + parallel `Vec<SwigluMlp>`
/// with deterministically-seeded random init. Stride-for-stride the
/// same recipe as `lm_validate::build_tiny_model_for_byte_level`,
/// generalised to arbitrary [`GptConfig`] dims.
#[cfg(feature = "rocm")]
fn build_random_byte_model(
    config: &GptConfig,
) -> (
    modgrad_transformer::model::GptModel,
    Vec<modgrad_transformer::mlp::SwigluMlp>,
) {
    use modgrad_compute::neuron::SimpleRng;
    use modgrad_transformer::attention::{AttentionWeights, CausalSelfAttention};
    use modgrad_transformer::block::TransformerBlock;
    use modgrad_transformer::mlp::{Mlp, MlpWeights, SwigluMlp, SwigluWeights};
    use modgrad_transformer::model::GptModel;
    use modgrad_transformer::norm::ScaledRmsNorm;
    use modgrad_transformer::position::fixed::FixedPositioning;
    use modgrad_transformer::residual::ResidualLambdas;
    use modgrad_transformer::rope::RotaryEmbedding;
    use modgrad_transformer::smear::{Inference, Smear, SmearWeights, Training};
    use modgrad_transformer::tensor::Tensor2;

    let md = config.model_dim.get();
    let kv_dim = config.num_kv_heads.get() * config.head_dim.get();
    let vocab = config.vocab_size.get();
    let mlp_dim = config.mlp_dim.get();

    // Seed: stable across runs so two `EntropyModel::new` calls with
    // the same config produce byte-identical models. Trainer can
    // re-seed via fresh-fill if true randomness is desired.
    let mut rng = SimpleRng::new(0xB7E_BA17_E000_0000);
    let randn = |rng: &mut SimpleRng, n: usize| -> Vec<f32> {
        // 0.05 stddev — small init keeps the pre-trained softmax
        // well away from saturation and matches the lm_validate
        // recipe.
        (0..n).map(|_| rng.next_normal() * 0.05).collect()
    };

    let token_embed = randn(&mut rng, vocab * md);
    // Tied embeddings: lm_head shares the embed table. The resident
    // upload still copies each weight independently (no aliasing on
    // device) — that's fine; the math is right at random init and
    // the trainer is responsible for tying gradients if it wants
    // tied training. For inference / random-init entropy estimation
    // we just need a valid fp32 matrix here.
    let lm_head = if config.tie_embeddings {
        token_embed.clone()
    } else {
        randn(&mut rng, vocab * md)
    };
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

        // Placeholder MLP — host model expects a `Mlp` field even
        // when the resident path uses the parallel `SwigluMlp` slab.
        // Same shape contract as `lm_validate::build_tiny_model_for_byte_level`.
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

// ─── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_default_is_consistent() {
        let cfg = EntropyConfig::default();
        cfg.validate().expect("default config validates");
        assert_eq!(cfg.n_heads * cfg.head_dim, cfg.d_model);
    }

    #[test]
    fn config_validate_catches_dim_mismatch() {
        let cfg = EntropyConfig {
            d_model: 256, n_heads: 4, head_dim: 32, // 4*32=128 ≠ 256
            n_layers: 2, max_seq_len: 64,
            norm_eps: 1e-5, rope_base: 10_000.0,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_validate_catches_zero_layers() {
        let cfg = EntropyConfig {
            d_model: 64, n_heads: 2, head_dim: 32,
            n_layers: 0, max_seq_len: 64,
            norm_eps: 1e-5, rope_base: 10_000.0,
        };
        assert!(cfg.validate().is_err());
    }

    /// Tiny entropy-model smoke test against a real GPU. Skipped
    /// cleanly without `--features rocm` or when the HIP runtime
    /// is missing. Verifies entropies are (a) finite, (b) ≥ 0, and
    /// (c) close to `log₂(256) = 8` for a random-init model. The
    /// upper bound is a hard 8.0 (uniform); the lower bound is a
    /// soft 5.0 — random init can be modestly peaked but should
    /// not be highly confident.
    #[cfg(feature = "rocm")]
    #[test]
    fn entropies_random_init_close_to_uniform() {
        if !modgrad_device::backend::rocm::ffi::runtime_available() {
            eprintln!("entropy: HIP runtime unavailable — skipping");
            return;
        }

        // Tiny config so the test runs in seconds even on weak silicon.
        let cfg = EntropyConfig {
            d_model: 64, n_heads: 2, head_dim: 32,
            n_layers: 2, max_seq_len: 32,
            norm_eps: 1e-5, rope_base: 10_000.0,
        };
        let mut model = EntropyModel::new(cfg).expect("build EntropyModel");

        let bytes = b"the quick brown fox";
        let mut entropies = vec![0.0f32; bytes.len()];
        let batch = HipBatch::new();
        model.entropies(&batch, bytes, &mut entropies)
            .expect("entropy forward");

        for (i, &h) in entropies.iter().enumerate() {
            assert!(h.is_finite(), "entropy at {i} is non-finite: {h}");
            assert!(h >= 0.0,        "entropy at {i} is negative: {h}");
            assert!(h <= 8.001,      "entropy at {i} > log₂(256): {h}");
            // Random init with stddev 0.05 produces near-uniform
            // softmax — entropy should be very close to 8 bits.
            // Loose lower bound to absorb both numerical and per-
            // position attention variation.
            assert!(h >= 5.0,
                "entropy at {i} unexpectedly low for random init: {h} \
                 (expected near 8.0)");
        }
    }

    /// Entropy must error out (not panic) when the input chunk
    /// exceeds the configured sliding-window context.
    #[cfg(feature = "rocm")]
    #[test]
    fn entropies_rejects_oversized_chunk() {
        if !modgrad_device::backend::rocm::ffi::runtime_available() {
            eprintln!("entropy: HIP runtime unavailable — skipping");
            return;
        }
        let cfg = EntropyConfig {
            d_model: 64, n_heads: 2, head_dim: 32,
            n_layers: 1, max_seq_len: 8,
            norm_eps: 1e-5, rope_base: 10_000.0,
        };
        let mut model = EntropyModel::new(cfg).expect("build");
        let bytes = vec![0u8; 16];
        let mut out = vec![0.0f32; 16];
        let batch = HipBatch::new();
        let err = model.entropies(&batch, &bytes, &mut out);
        assert!(err.is_err(), "oversized chunk should error");
    }

    /// Empty input is a no-op and produces no error.
    #[cfg(feature = "rocm")]
    #[test]
    fn entropies_empty_input_noop() {
        if !modgrad_device::backend::rocm::ffi::runtime_available() {
            eprintln!("entropy: HIP runtime unavailable — skipping");
            return;
        }
        let cfg = EntropyConfig {
            d_model: 64, n_heads: 2, head_dim: 32,
            n_layers: 1, max_seq_len: 8,
            norm_eps: 1e-5, rope_base: 10_000.0,
        };
        let mut model = EntropyModel::new(cfg).expect("build");
        let batch = HipBatch::new();
        model.entropies(&batch, &[], &mut []).expect("empty input ok");
    }
}
