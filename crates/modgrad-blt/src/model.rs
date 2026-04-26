//! `BltModel` — top-level assembly of LocalEncoder + LatentTransformer +
//! LocalDecoder, with the `LanguageModel` impl (`isis_runtime`'s trait)
//! so existing `LmTrainer<M>` and `Sampler` compose unchanged.
//!
//! The Latent Transformer is [`GptModelResident`] (canonically — for
//! byte-ification, initialised from a pretrained Qwen2.5 backbone via
//! [`crate::byteify`]).
//!
//! ## Forward path (paper Figure 5)
//!
//! ```text
//!   bytes [N]
//!     ↓                       LocalEncoder (lE byte-level layers)
//!   patch reps [P × patch_dim]
//!     ↓                       Latent transformer (block stack only —
//!                              we bypass the embed lookup since
//!                              patches are continuous reps already)
//!   patch reps [P × patch_dim]
//!     ↓                       LocalDecoder (lD layers, cross-attn first)
//!   byte logits [N × 256]
//! ```
//!
//! Owner: agent sasha (with [`crate::encoder`] and [`crate::decoder`]).
//!
//! Only available with `--features rocm`.

#![cfg(feature = "rocm")]
#![allow(clippy::too_many_arguments)]

use modgrad_compute::backend::{GpuVec, ResidencyError};
use modgrad_device::backend::{HipBatch, HipBuffer};
use modgrad_device::backend::ops::rms_norm_resident;
use modgrad_transformer::config::{
    GptConfig, MlpActivation, Precision, ResidualConfig, SmearConfig,
    ValueEmbedConfig, WindowPattern,
};
use modgrad_transformer::dims::*;
use modgrad_transformer::kv_cache_resident::KvCacheResident;
use modgrad_transformer::resident::{
    AttentionScratch, GptModelResident, SwigluScratch, TransformerBlockScratch,
};

use crate::decoder::{LocalDecoder, LocalDecoderConfig, LocalDecoderScratch};
use crate::encoder::{LocalEncoder, LocalEncoderConfig, LocalEncoderScratch};

/// Hyperparameters for the latent transformer (`GptConfig` slice with
/// only the relevant fields exposed). The latent operates over patch
/// representations, not byte tokens — so its `model_dim` must equal
/// the encoder's `patch_dim` and the decoder's `patch_dim`.
#[derive(Debug, Clone)]
pub struct BltLatentConfig {
    pub n_layers: usize,
    pub patch_dim: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub mlp_dim: usize,
    pub norm_eps: f32,
    pub rope_base: f32,
    /// Maximum number of patches the latent may see in one call.
    pub max_patches: usize,
}

impl BltLatentConfig {
    /// Validate dim consistency.
    pub fn validate(&self) -> Result<(), ResidencyError> {
        if self.n_heads * self.head_dim != self.patch_dim {
            return Err(ResidencyError::WrongVariant {
                expected: "n_heads * head_dim == patch_dim",
                got: "latent dim mismatch",
            });
        }
        Ok(())
    }

    /// Build a [`GptConfig`] for the latent. The vocab is set to a
    /// nominal value; we never use the embed lookup.
    pub(crate) fn to_gpt_config(&self) -> GptConfig {
        GptConfig {
            model_dim: ModelDim::new(self.patch_dim),
            num_heads: NumHeads::new(self.n_heads),
            num_kv_heads: NumKvHeads::new(self.n_heads),
            head_dim: HeadDim::new(self.head_dim),
            num_layers: NumLayers::new(self.n_layers.max(1)),
            vocab_size: VocabSize::new(256),
            mlp_dim: MlpDim::new(self.mlp_dim),
            max_seq_len: SeqLen::new(self.max_patches),
            rope_base: self.rope_base,
            qk_norm_scale: 1.0,
            use_qk_norm: false,
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
            norm_eps: self.norm_eps,
        }
    }
}

/// Top-level assembly of the BLT model.
///
/// **The latent transformer is held as a [`GptModelResident`] but the
/// embed lookup is bypassed.** We feed patch reps into the block stack
/// directly via the model's `blocks` field; the embed and lm_head of
/// the latent are unused (the byte LM head lives in the decoder, the
/// byte embedding lives in the encoder). Holding `GptModelResident`
/// rather than `Vec<TransformerBlockResident>` keeps the byteification
/// recipe (replacing this field with a Qwen2.5-loaded resident model)
/// trivially compatible.
pub struct BltModel {
    pub encoder: LocalEncoder,
    pub latent: GptModelResident,
    pub latent_kv_cache: KvCacheResident,
    /// Scaled-RMSNorm weight buffer for the latent's final norm.
    /// `[patch_dim]` ones. Kept here rather than reusing
    /// `latent.final_norm_weight_dev` because the latent's lm_head
    /// path is disabled — keeping our own norm buffer makes the
    /// patch-level forward self-contained.
    pub latent_final_norm_weight_dev: HipBuffer,
    pub decoder: LocalDecoder,
    pub config: BltConfig,
}

/// Combined configuration carried by [`BltModel`].
#[derive(Debug, Clone)]
pub struct BltConfig {
    pub encoder: LocalEncoderConfig,
    pub latent: BltLatentConfig,
    pub decoder: LocalDecoderConfig,
}

impl BltConfig {
    /// Validate the full model config — checks per-component validity
    /// and the patch-dim equality required across encoder, latent, and
    /// decoder.
    pub fn validate(&self) -> Result<(), ResidencyError> {
        self.encoder.validate()?;
        self.latent.validate()?;
        self.decoder.validate()?;
        if self.encoder.patch_dim != self.latent.patch_dim {
            return Err(ResidencyError::WrongVariant {
                expected: "encoder.patch_dim == latent.patch_dim",
                got: "patch dim mismatch (encoder vs latent)",
            });
        }
        if self.latent.patch_dim != self.decoder.patch_dim {
            return Err(ResidencyError::WrongVariant {
                expected: "latent.patch_dim == decoder.patch_dim",
                got: "patch dim mismatch (latent vs decoder)",
            });
        }
        if self.encoder.byte_dim != self.decoder.byte_dim {
            return Err(ResidencyError::WrongVariant {
                expected: "encoder.byte_dim == decoder.byte_dim",
                got: "byte dim mismatch (encoder vs decoder)",
            });
        }
        Ok(())
    }
}

impl BltModel {
    /// Allocate device buffers and build all three components from
    /// deterministic random weights. Latent transformer is initialised
    /// from a temporary host [`modgrad_transformer::model::GptModel`]
    /// — same pattern as `GptModelResident::from_model` uses elsewhere
    /// in the SDK. Byteification (initialising the latent from a Qwen2.5
    /// safetensors checkpoint) is the [`crate::byteify`] slice.
    pub fn new(config: BltConfig) -> Result<Self, ResidencyError> {
        config.validate()?;

        let encoder = LocalEncoder::new(config.encoder.clone())?;
        let decoder = LocalDecoder::new(config.decoder.clone())?;

        let (latent, latent_final_norm_weight_dev) = build_latent(&config.latent)?;
        let latent_gpt = config.latent.to_gpt_config();
        let latent_kv_cache = KvCacheResident::new(
            latent_gpt.num_layers.get(),
            latent_gpt.num_kv_heads.get(),
            latent_gpt.head_dim.get(),
            latent_gpt.max_seq_len.get(),
            latent_gpt.model_dim.get(),
        )?;

        Ok(Self {
            encoder,
            latent,
            latent_kv_cache,
            latent_final_norm_weight_dev,
            decoder,
            config,
        })
    }

    /// Total number of layers across encoder + latent + decoder. Useful
    /// for diagnostic logging; not load-bearing.
    pub fn n_layers(&self) -> usize {
        self.encoder.n_layers() + self.latent.num_layers() + self.decoder.n_layers()
    }

    /// Vocabulary size — always 256 (byte-level).
    #[inline]
    pub fn vocab_size(&self) -> usize { 256 }

    /// Forward bytes through the full BLT pipeline.
    ///
    /// `bytes`: input byte sequence `[N]`.
    /// `boundaries`: patch boundaries `[0, b_1, …, N]` (provided by
    /// leah's patcher; see [`crate::patcher`]).
    /// `byte_logits_out`: `[N × 256]` row-major output.
    ///
    /// All three KV caches (encoder byte, latent patch, decoder byte)
    /// are reset to length 0 at the start of every call. Each forward
    /// is treated as an independent prompt — the BLT training loop
    /// never wants stale K/V across micro-batches. Without the reset,
    /// the second call's positions would index beyond the cache slots
    /// the previous call wrote.
    pub fn forward(
        &mut self,
        batch: &HipBatch,
        bytes: &[u8],
        boundaries: &[usize],
        scratch: &mut BltScratch,
        byte_logits_out: &mut GpuVec,
    ) -> Result<(), ResidencyError> {
        let n_bytes = bytes.len();
        let n_patches = boundaries.len().saturating_sub(1);
        let patch_dim = self.config.latent.patch_dim;

        debug_assert_eq!(byte_logits_out.len(), n_bytes * 256);
        debug_assert!(n_patches > 0);
        debug_assert_eq!(boundaries[0], 0);
        debug_assert_eq!(boundaries[boundaries.len() - 1], n_bytes);

        // Reset KV caches between calls. `reset()` is cheap (just sets
        // `seq_len = 0`); the underlying buffers are not zeroed because
        // attention only reads slots `0..attn_len = position + 1`, and
        // the current call overwrites those before they're read.
        self.encoder.byte_kv_cache.reset();
        self.latent_kv_cache.reset();
        self.decoder.byte_kv_cache.reset();

        // ── Stage 1: encoder bytes → patch reps ──
        let patch_reps_buf = &mut scratch.patch_reps;
        debug_assert_eq!(patch_reps_buf.len(), self.config.latent.max_patches * patch_dim);

        // We reuse `scratch.patch_reps_view` as the n_patches × patch_dim
        // output. The encoder owns its scratch and writes into the
        // caller-provided buffer; size to the actual patch count.
        let mut patch_reps_view = GpuVec::try_hip(n_patches * patch_dim)?;
        self.encoder.forward(
            batch, bytes, boundaries, &mut scratch.encoder, &mut patch_reps_view,
        )?;

        // ── Stage 2: latent transformer over patches ──
        //
        // The latent processes patches as if they were tokens, but the
        // input is patch reps not token IDs. We bypass the embedding
        // lookup and feed patch reps straight into the block stack.
        // Per-patch forward through every block, with the latent's KV
        // cache accumulating positions 0..n_patches.
        run_latent_forward(
            batch,
            &self.latent,
            &self.latent_final_norm_weight_dev,
            &mut self.latent_kv_cache,
            &patch_reps_view,
            n_patches,
            patch_dim,
            self.config.latent.norm_eps,
            &mut scratch.latent,
            patch_reps_buf,
        )?;

        // After the latent, `patch_reps_buf` holds the final patch reps.
        // Trim to the actual count for the decoder's contract.
        let mut patch_reps_for_decoder = GpuVec::try_hip(n_patches * patch_dim)?;
        copy_d2d(patch_reps_buf, &mut patch_reps_for_decoder, n_patches * patch_dim)?;

        // ── Stage 3: decoder patches → byte logits ──
        //
        // Seed byte_reps = encoder's last-layer byte_reps (paper Fig 5).
        // The encoder leaves its byte_reps in `scratch.encoder.byte_reps`
        // — sized to `max_seq_len × byte_dim`, with valid data at
        // positions 0..n_bytes.
        self.decoder.forward(
            batch,
            &patch_reps_for_decoder,
            boundaries,
            Some(&scratch.encoder.byte_reps),
            &mut scratch.decoder,
            byte_logits_out,
        )?;

        // Reset KV caches between calls. The smoke test allocates a
        // fresh BltModel per call so this is a courtesy — production
        // training will need a `reset_caches` method.

        Ok(())
    }
}

// ─── Scratch ─────────────────────────────────────────────────

/// Caller-owned scratch for [`BltModel::forward`]. Sized once at
/// construction; reused across calls.
pub struct BltScratch {
    pub encoder: LocalEncoderScratch,
    pub latent: BltLatentScratch,
    pub decoder: LocalDecoderScratch,
    /// `[max_patches × patch_dim]` post-latent patch reps.
    pub patch_reps: GpuVec,
}

impl BltScratch {
    /// Allocate scratch for the given model config.
    pub fn new(config: &BltConfig) -> Result<Self, ResidencyError> {
        let encoder = LocalEncoderScratch::new(&config.encoder)?;
        let latent = BltLatentScratch::new(&config.latent)?;
        let decoder = LocalDecoderScratch::new(&config.decoder)?;
        let patch_reps = GpuVec::try_hip(
            config.latent.max_patches * config.latent.patch_dim,
        )?;
        Ok(Self { encoder, latent, decoder, patch_reps })
    }
}

/// Scratch for the latent transformer's per-patch forward. Mirrors
/// `TransformerBlockScratch` etc. but sized to the latent's dims.
pub struct BltLatentScratch {
    pub block_scratch: TransformerBlockScratch,
    pub attn_scratch: AttentionScratch,
    pub mlp_scratch: SwigluScratch,
    /// `[patch_dim]` per-patch hidden state.
    pub hidden: GpuVec,
    /// `[patch_dim]` x0 for the residual stream (unused under our
    /// `x_lambda = 0` profile but the resident block API requires it).
    pub x0: GpuVec,
    /// `[patch_dim]` post-final-norm output.
    pub normed: GpuVec,
}

impl BltLatentScratch {
    pub fn new(config: &BltLatentConfig) -> Result<Self, ResidencyError> {
        let pd = config.patch_dim;
        let kv_dim = config.n_heads * config.head_dim;
        let max_p = config.max_patches;

        Ok(Self {
            block_scratch: TransformerBlockScratch::with_dims(
                pd, kv_dim, config.mlp_dim, config.n_heads * max_p,
            )?,
            attn_scratch: AttentionScratch::new(
                config.n_heads, config.head_dim, kv_dim, max_p,
            )?,
            mlp_scratch: SwigluScratch::new(pd, config.mlp_dim)?,
            hidden: GpuVec::try_hip(pd)?,
            x0: GpuVec::try_hip(pd)?,
            normed: GpuVec::try_hip(pd)?,
        })
    }
}

// ─── Internals ───────────────────────────────────────────────

/// Build the latent transformer + a fresh `[patch_dim]` ones final-norm
/// buffer. Wraps the existing `GptModelResident::from_model` recipe
/// from the test harness in `modgrad-transformer`.
fn build_latent(
    config: &BltLatentConfig,
) -> Result<(GptModelResident, HipBuffer), ResidencyError> {
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

    let gpt_config = config.to_gpt_config();
    let pd = config.patch_dim;
    let kv_dim = gpt_config.num_kv_heads.get() * gpt_config.head_dim.get();
    let vocab = gpt_config.vocab_size.get();
    let mlp_dim = gpt_config.mlp_dim.get();

    let mut rng = SimpleRng::new(
        ((pd * config.n_layers).max(1) * config.mlp_dim.max(1)) as u64 ^ 0x1A7E_17F0,
    );
    let scale = 0.05_f32;
    let randn = |rng: &mut SimpleRng, n: usize| -> Vec<f32> {
        (0..n).map(|_| rng.next_normal() * scale).collect()
    };

    let token_embed = randn(&mut rng, vocab * pd);
    let lm_head = randn(&mut rng, vocab * pd);
    let final_norm_scale = vec![1.0f32; pd];
    let smear_gate = vec![0.0f32; pd * gpt_config.smear.gate_channels];

    let mut blocks_host = Vec::with_capacity(config.n_layers.max(1));
    let mut swiglu_mlps = Vec::with_capacity(config.n_layers.max(1));
    for li in 0..config.n_layers.max(1) {
        let attn_w = AttentionWeights {
            wq: Tensor2::new(randn(&mut rng, pd * pd), pd, pd).unwrap(),
            wk: Tensor2::new(randn(&mut rng, kv_dim * pd), kv_dim, pd).unwrap(),
            wv: Tensor2::new(randn(&mut rng, kv_dim * pd), kv_dim, pd).unwrap(),
            wo: Tensor2::new(randn(&mut rng, pd * pd), pd, pd).unwrap(),
        };
        let attn = CausalSelfAttention::new(attn_w, &gpt_config);

        let swiglu_w = SwigluWeights {
            gate: Tensor2::new(randn(&mut rng, mlp_dim * pd), mlp_dim, pd).unwrap(),
            up: Tensor2::new(randn(&mut rng, mlp_dim * pd), mlp_dim, pd).unwrap(),
            down: Tensor2::new(randn(&mut rng, pd * mlp_dim), pd, mlp_dim).unwrap(),
        };
        let swiglu = SwigluMlp::new(swiglu_w, gpt_config.model_dim, gpt_config.mlp_dim);
        swiglu_mlps.push(swiglu);

        let placeholder_mlp = Mlp::new(
            MlpWeights {
                fc: Tensor2::zeros(mlp_dim, pd),
                proj: Tensor2::zeros(pd, mlp_dim),
            },
            gpt_config.model_dim, gpt_config.mlp_dim,
        );
        let layer_idx = LayerIdx::new(li, gpt_config.num_layers).unwrap();
        blocks_host.push(TransformerBlock::new(attn, placeholder_mlp, None, layer_idx, &gpt_config));
    }

    let model = GptModel {
        embed: Tensor2::new(token_embed, vocab, pd).unwrap(),
        lm_head: Tensor2::new(lm_head, vocab, pd).unwrap(),
        final_norm: ScaledRmsNorm::new(final_norm_scale.clone(), gpt_config.norm_eps),
        smear_inference: Smear::<Inference>::new(SmearWeights::new(
            smear_gate.clone(), gpt_config.model_dim, &gpt_config.smear,
        )),
        smear_training: Smear::<Training>::new(SmearWeights::new(
            smear_gate, gpt_config.model_dim, &gpt_config.smear,
        )),
        blocks: blocks_host,
        lambdas: ResidualLambdas::from_config(&gpt_config.residual, gpt_config.num_layers),
        rope: RotaryEmbedding::new(
            gpt_config.head_dim, gpt_config.max_seq_len, gpt_config.rope_base,
        ),
        position: Box::new(FixedPositioning),
        config: gpt_config.clone(),
    };
    let resident = GptModelResident::from_model(&model, &swiglu_mlps)?;

    let final_norm_buf = HipBuffer::new(pd * 4)?;
    final_norm_buf.copy_from_host(&final_norm_scale)?;

    Ok((resident, final_norm_buf))
}

/// Per-patch forward through the latent's block stack, bypassing the
/// embed lookup. Reads `n_patches` patches from `patch_reps_in` and
/// writes the final post-norm reps into `patch_reps_out`.
fn run_latent_forward(
    batch: &HipBatch,
    latent: &GptModelResident,
    final_norm_weight_dev: &HipBuffer,
    kv_cache: &mut KvCacheResident,
    patch_reps_in: &GpuVec,
    n_patches: usize,
    patch_dim: usize,
    norm_eps: f32,
    scratch: &mut BltLatentScratch,
    patch_reps_out: &mut GpuVec,
) -> Result<(), ResidencyError> {
    debug_assert_eq!(patch_reps_in.len(), n_patches * patch_dim);
    debug_assert!(patch_reps_out.len() >= n_patches * patch_dim);

    for p in 0..n_patches {
        // Load patch[p] into scratch.hidden.
        copy_slab_to_dense(patch_reps_in, p, patch_dim, &mut scratch.hidden)?;
        batch.note_dispatch()?;
        // x0 mirrors hidden — same convention as `GptModelResident::forward`.
        copy_slab_to_dense(patch_reps_in, p, patch_dim, &mut scratch.x0)?;
        batch.note_dispatch()?;

        for block in &latent.blocks {
            block.forward(
                batch,
                &mut scratch.hidden,
                &scratch.x0,
                kv_cache,
                p,
                &latent.rope,
                &mut scratch.attn_scratch,
                &mut scratch.mlp_scratch,
                &mut scratch.block_scratch,
            )?;
        }

        // Final norm into normed, then write into patch_reps_out[p].
        unsafe {
            rms_norm_resident(
                hip_buf(&scratch.hidden)?.device_ptr() as *const f32,
                final_norm_weight_dev.device_ptr() as *const f32,
                hip_buf(&scratch.normed)?.device_ptr() as *mut f32,
                1, patch_dim, norm_eps,
            )?;
        }
        batch.note_dispatch()?;

        copy_dense_to_slab(&scratch.normed, p, patch_dim, patch_reps_out)?;
        batch.note_dispatch()?;
    }
    Ok(())
}

#[inline]
fn hip_buf<'a>(g: &'a GpuVec) -> Result<&'a modgrad_device::backend::HipBuffer, ResidencyError> {
    match g {
        GpuVec::Hip(b) => Ok(b),
        other => Err(ResidencyError::WrongVariant {
            expected: "Hip", got: other.variant_name(),
        }),
    }
}

#[inline]
fn hip_buf_mut<'a>(g: &'a mut GpuVec)
    -> Result<&'a mut modgrad_device::backend::HipBuffer, ResidencyError>
{
    match g {
        GpuVec::Hip(b) => Ok(b),
        other => Err(ResidencyError::WrongVariant {
            expected: "Hip", got: other.variant_name(),
        }),
    }
}

fn copy_d2d(src: &GpuVec, dst: &mut GpuVec, n: usize) -> Result<(), ResidencyError> {
    use modgrad_device::backend::rocm::ffi;
    let s = hip_buf(src)?;
    let d = hip_buf_mut(dst)?;
    const HIP_D2D: std::os::raw::c_int = 3;
    let err = unsafe {
        ffi::hipMemcpy(
            d.device_ptr(),
            s.device_ptr() as *const std::os::raw::c_void,
            n * 4, HIP_D2D,
        )
    };
    if err != 0 {
        return Err(ResidencyError::Backend(
            modgrad_device::backend::BackendError::Runtime(format!(
                "BltModel D2D ({} f32): {}", n, ffi::hip_err_str(err),
            )),
        ));
    }
    Ok(())
}

fn copy_slab_to_dense(
    slab: &GpuVec,
    t: usize,
    dim: usize,
    dense: &mut GpuVec,
) -> Result<(), ResidencyError> {
    use modgrad_device::backend::rocm::ffi;
    let src = hip_buf(slab)?;
    let dst = hip_buf_mut(dense)?;
    let off_bytes = t * dim * 4;
    const HIP_D2D: std::os::raw::c_int = 3;
    let err = unsafe {
        ffi::hipMemcpy(
            dst.device_ptr(),
            (src.device_ptr() as *const u8).add(off_bytes) as *const std::os::raw::c_void,
            dim * 4, HIP_D2D,
        )
    };
    if err != 0 {
        return Err(ResidencyError::Backend(
            modgrad_device::backend::BackendError::Runtime(format!(
                "slab → dense D2D: {}", ffi::hip_err_str(err),
            )),
        ));
    }
    Ok(())
}

fn copy_dense_to_slab(
    dense: &GpuVec,
    t: usize,
    dim: usize,
    slab: &mut GpuVec,
) -> Result<(), ResidencyError> {
    use modgrad_device::backend::rocm::ffi;
    let src = hip_buf(dense)?;
    let dst = hip_buf_mut(slab)?;
    let off_bytes = t * dim * 4;
    const HIP_D2D: std::os::raw::c_int = 3;
    let err = unsafe {
        ffi::hipMemcpy(
            (dst.device_ptr() as *mut u8).add(off_bytes) as *mut std::os::raw::c_void,
            src.device_ptr() as *const std::os::raw::c_void,
            dim * 4, HIP_D2D,
        )
    };
    if err != 0 {
        return Err(ResidencyError::Backend(
            modgrad_device::backend::BackendError::Runtime(format!(
                "dense → slab D2D: {}", ffi::hip_err_str(err),
            )),
        ));
    }
    Ok(())
}

// ─── LanguageModel impl ──────────────────────────────────────
//
// Option (A) from the brief: we expose `BltModel` as a `LanguageModel`
// where `token_ids` are bytes (i64) and patch boundaries are computed
// inline (TODO: hook into leah's patcher once it lands; for now we use
// a simple fixed-size-stride patcher as a placeholder so tests can
// exercise the trait without a real entropy patcher). `positions` is
// unused — BLT bytes don't carry an absolute-position notion outside
// the byte stream itself.
//
// Implementation is gated on noah's cross-attn landing (the trait uses
// the full forward path, so a stubbed cross-attn produces zero logits).
// We still wire the impl so `LmTrainer<BltModel>` type-checks today.

#[cfg(feature = "rocm")]
impl isis_runtime::language_model::LanguageModel for BltModel {
    fn forward_logits(
        &mut self,
        batch: &HipBatch,
        token_ids: &[i64],
        _positions: &[usize],
        _kv_cache: Option<&mut KvCacheResident>,
        logits_out: &mut GpuVec,
    ) -> Result<(), ResidencyError> {
        // BLT consumes byte sequences with patch boundaries. Treat
        // `token_ids` as i64-encoded bytes (the trainer feeds bytes via
        // the same channel; values must lie in 0..=255).
        let bytes: Vec<u8> = token_ids.iter()
            .map(|&t| t.clamp(0, 255) as u8)
            .collect();

        // Placeholder patcher: fixed stride. Real BLT integration will
        // call leah's entropy patcher inline. Stride chosen so a 32-byte
        // sequence yields 8 patches of 4 bytes each.
        let n = bytes.len();
        let stride = ((n + 7) / 8).max(1);
        let mut boundaries = Vec::new();
        boundaries.push(0);
        let mut b = stride;
        while b < n {
            boundaries.push(b);
            b += stride;
        }
        boundaries.push(n);

        // Need a scratch — allocate fresh. Caller-owned scratch via
        // `LanguageModel` would require a trait extension; we keep the
        // allocation here for now and revisit when the foundation-model
        // trainer integrates BLT.
        let mut scratch = BltScratch::new(&self.config)?;
        self.forward(batch, &bytes, &boundaries, &mut scratch, logits_out)
    }

    fn n_layers(&self) -> usize { BltModel::n_layers(self) }
    fn d_model(&self) -> usize { self.config.latent.patch_dim }
    fn vocab_size(&self) -> usize { 256 }

    fn ensure_resident(&mut self, _batch: &HipBatch) -> Result<(), ResidencyError> {
        // BLT components are fully resident from `BltModel::new`.
        Ok(())
    }

    type BackwardState = ();

    fn alloc_backward_state(
        &self,
        _kv_cache: &KvCacheResident,
    ) -> Result<Self::BackwardState, ResidencyError> {
        // Backward chain is a follow-up slice — see module docs.
        Ok(())
    }

    fn forward_for_backward_position(
        &mut self,
        _batch: &HipBatch,
        _token_id: i64,
        _position: usize,
        _kv_cache: &mut KvCacheResident,
        _state: &mut Self::BackwardState,
        _logits_out: &mut GpuVec,
    ) -> Result<(), ResidencyError> {
        Err(ResidencyError::WrongVariant {
            expected: "BltModel backward (follow-up slice)",
            got: "called forward_for_backward_position on BltModel",
        })
    }

    fn backward_position(
        &mut self,
        _batch: &HipBatch,
        _token_id: i64,
        _position: usize,
        _kv_cache: &mut KvCacheResident,
        _state: &mut Self::BackwardState,
        _d_logits: &GpuVec,
    ) -> Result<(), ResidencyError> {
        Err(ResidencyError::WrongVariant {
            expected: "BltModel backward (follow-up slice)",
            got: "called backward_position on BltModel",
        })
    }
}

// ─── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use modgrad_device::backend::rocm::ffi::runtime_available;
    use std::sync::Mutex;

    /// HIP runtime tests must run serially — same rationale as the
    /// `modgrad-transformer::resident` test guard. Multiple concurrent
    /// resident dispatches share the default stream.
    static HIP_TEST_LOCK: Mutex<()> = Mutex::new(());

    /// Tiny config the smoke test uses: 32 bytes → 8 patches, byte_dim=32,
    /// patch_dim=64, lE=1, lL=2, lD=1. Total params are kilobytes — the
    /// test runs in a few seconds even on a contented GPU.
    fn tiny_config() -> BltConfig {
        let byte_dim = 32usize;
        let n_byte_heads = 4usize;
        let byte_head_dim = byte_dim / n_byte_heads;
        let patch_dim = 64usize;
        let n_patch_heads = 4usize;
        let patch_head_dim = patch_dim / n_patch_heads;
        let max_seq = 32usize;
        let max_patches = 16usize;

        BltConfig {
            encoder: LocalEncoderConfig {
                n_layers: 1,
                byte_dim,
                patch_dim,
                n_heads: n_byte_heads,
                head_dim: byte_head_dim,
                mlp_dim: byte_dim * 2,
                norm_eps: 1e-5,
                rope_base: 10_000.0,
                max_seq_len: max_seq,
                ngram_min_n: 3,
                ngram_max_n: 5,
                ngram_vocab_per_n: 256,
            },
            latent: BltLatentConfig {
                n_layers: 2,
                patch_dim,
                n_heads: n_patch_heads,
                head_dim: patch_head_dim,
                mlp_dim: patch_dim * 2,
                norm_eps: 1e-5,
                rope_base: 10_000.0,
                max_patches,
            },
            decoder: LocalDecoderConfig {
                n_layers: 1,
                byte_dim,
                patch_dim,
                n_heads: n_byte_heads,
                head_dim: byte_head_dim,
                mlp_dim: byte_dim * 2,
                norm_eps: 1e-5,
                rope_base: 10_000.0,
                max_seq_len: max_seq,
            },
        }
    }

    #[test]
    fn blt_model_forward_smoke() {
        let _guard = HIP_TEST_LOCK.lock().unwrap();
        if !runtime_available() {
            eprintln!("hip runtime unavailable, skipping");
            return;
        }

        let config = tiny_config();
        let mut model = BltModel::new(config.clone()).expect("BltModel::new");
        let mut scratch = BltScratch::new(&config).expect("BltScratch::new");

        // 32 synthetic bytes → 8 patches of 4 bytes each. Patcher
        // boundaries are computed inline (the smoke test doesn't depend
        // on leah's entropy patcher).
        let bytes: Vec<u8> = (0..32u8).collect();
        let boundaries: Vec<usize> = (0..=8).map(|p| p * 4).collect();
        assert_eq!(boundaries[0], 0);
        assert_eq!(boundaries[8], 32);

        let n_bytes = bytes.len();
        let mut logits = GpuVec::try_hip(n_bytes * 256).expect("alloc logits");
        let batch = HipBatch::new();
        model.forward(&batch, &bytes, &boundaries, &mut scratch, &mut logits)
            .expect("BltModel::forward");
        batch.flush().expect("flush");

        // Shape: vocab × bytes.
        assert_eq!(logits.len(), n_bytes * 256);

        // Read back and sanity-check: no NaN/inf, plausible spread.
        let mut host = vec![0.0f32; n_bytes * 256];
        logits.copy_to_host(&mut host);
        let n_finite = host.iter().filter(|v| v.is_finite()).count();
        assert_eq!(n_finite, host.len(), "logits contain non-finite values");

        // Spread check: with random weights and zero-stub cross-attn
        // the logits won't be diverse (cross-attn is the source of
        // patch-token interaction), but the per-byte LM head still
        // produces a non-trivial spread because each byte's hidden
        // state is the residual stream after the byte-side blocks.
        let max = host.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min = host.iter().cloned().fold(f32::INFINITY, f32::min);
        eprintln!("blt logits range: [{min}, {max}]");
        assert!(max > min, "logits collapsed to a single value");
        // Loose magnitude bound — random init should keep |logits| < 100.
        assert!(max.abs() < 100.0 && min.abs() < 100.0,
                "logits magnitude implausibly large: {min}, {max}");
    }

    #[test]
    fn blt_config_validates() {
        // Pure-host validation: no HIP runtime needed.
        let mut bad = tiny_config();
        bad.latent.patch_dim = 128;  // mismatch with encoder.patch_dim = 64
        assert!(bad.validate().is_err(), "patch_dim mismatch must reject");

        let mut bad2 = tiny_config();
        bad2.encoder.byte_dim = 16;  // mismatch with decoder.byte_dim = 32
        bad2.encoder.n_heads = 1;
        bad2.encoder.head_dim = 16;
        assert!(bad2.validate().is_err(), "byte_dim mismatch must reject");
    }
}

