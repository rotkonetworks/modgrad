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
    AttentionBackwardScratch, AttentionResidentGrads, AttentionScratch,
    GptModelResident, SwigluBackwardScratch, SwigluResidentGrads, SwigluScratch,
    TransformerBlockScratch,
};

use crate::decoder::{
    LocalDecoder, LocalDecoderBwdCache, LocalDecoderConfig, LocalDecoderGrads,
    LocalDecoderScratch,
};
use crate::encoder::{
    LocalEncoder, LocalEncoderBackwardScratch, LocalEncoderBwdCache,
    LocalEncoderConfig, LocalEncoderGrads, LocalEncoderScratch,
};

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
        Self::build(config, None)
    }

    /// Like [`Self::new`], but explicitly initialises the latent's final
    /// RMSNorm scale from the supplied weights (length must equal
    /// `config.latent.patch_dim`). Used by the byteification recipe
    /// (paper §6.2) to preserve the pretrained model's final-norm scale
    /// rather than resetting it to all-ones — `BltModel::new` keeps the
    /// canonical RMSNorm-at-init convention (scale = 1) for greenfield
    /// training; `with_pretrained_final_norm` is the path for
    /// initialising from a Qwen2.5 (or similar) safetensors checkpoint
    /// where the trained final-norm scale is non-trivial and must flow
    /// through into both the host model and the resident device buffer.
    ///
    /// Returns [`ResidencyError::WrongVariant`] if
    /// `pretrained_final_norm.len() != config.latent.patch_dim`.
    pub fn with_pretrained_final_norm(
        config: BltConfig,
        pretrained_final_norm: &[f32],
    ) -> Result<Self, ResidencyError> {
        if pretrained_final_norm.len() != config.latent.patch_dim {
            return Err(ResidencyError::WrongVariant {
                expected: "pretrained_final_norm.len() == config.latent.patch_dim",
                got: "pretrained final-norm scale length mismatch",
            });
        }
        Self::build(config, Some(pretrained_final_norm))
    }

    /// Shared constructor body — both [`Self::new`] and
    /// [`Self::with_pretrained_final_norm`] route through here. `None`
    /// preserves the all-ones RMSNorm-at-init convention; `Some(scale)`
    /// flows the supplied scale into both the host `GptModel` (via its
    /// `final_norm.scale`) and the parallel resident buffer
    /// `latent_final_norm_weight_dev`, keeping the two views in sync.
    fn build(
        config: BltConfig,
        pretrained_final_norm: Option<&[f32]>,
    ) -> Result<Self, ResidencyError> {
        config.validate()?;

        let encoder = LocalEncoder::new(config.encoder.clone())?;
        let decoder = LocalDecoder::new(config.decoder.clone())?;

        let (latent, latent_final_norm_weight_dev) =
            build_latent(&config.latent, pretrained_final_norm)?;
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

    /// Forward over a full byte sequence, populating `state` with the
    /// per-patch / per-layer activations the matching [`Self::backward`]
    /// needs. Sequence-level — does not fit the per-position
    /// `LanguageModel` trait shape (which is why the trait impl below
    /// keeps `forward_for_backward_position` returning a
    /// `WrongVariant` error). A future `BltTrainer<BltModel>` will call
    /// this method directly.
    pub fn forward_for_backward(
        &mut self,
        batch: &HipBatch,
        bytes: &[u8],
        boundaries: &[usize],
        scratch: &mut BltScratch,
        state: &mut BltBackwardState,
        byte_logits_out: &mut GpuVec,
    ) -> Result<(), ResidencyError> {
        let n_bytes = bytes.len();
        let n_patches = boundaries.len().saturating_sub(1);
        let patch_dim = self.config.latent.patch_dim;

        debug_assert_eq!(byte_logits_out.len(), n_bytes * 256);
        debug_assert!(n_patches > 0);
        debug_assert!(n_patches <= self.config.latent.max_patches);
        debug_assert_eq!(boundaries[0], 0);
        debug_assert_eq!(boundaries[boundaries.len() - 1], n_bytes);

        self.encoder.byte_kv_cache.reset();
        self.latent_kv_cache.reset();
        self.decoder.byte_kv_cache.reset();

        // Stage 1: encoder forward-for-backward.
        let mut patch_reps_view = GpuVec::try_hip(n_patches * patch_dim)?;
        self.encoder.forward_for_backward(
            batch,
            bytes,
            boundaries,
            &mut scratch.encoder,
            &mut state.encoder_cache,
            &mut patch_reps_view,
        )?;

        // Stage 2: latent forward — per-patch through the block stack
        // with per-(patch, layer) activation snapshots into
        // `state.latent_block_scratches[p][li]`.
        for p in 0..n_patches {
            copy_slab_to_dense(&patch_reps_view, p, patch_dim, &mut scratch.latent.hidden)?;
            batch.note_dispatch()?;
            copy_slab_to_dense(&patch_reps_view, p, patch_dim, &mut scratch.latent.x0)?;
            batch.note_dispatch()?;

            for (li, block) in self.latent.blocks.iter().enumerate() {
                block.forward_for_backward(
                    batch,
                    &mut scratch.latent.hidden,
                    &scratch.latent.x0,
                    &mut self.latent_kv_cache,
                    p,
                    &self.latent.rope,
                    &mut state.latent_attn_scratch,
                    &mut state.latent_mlp_scratch,
                    &mut state.latent_block_scratches[p][li],
                )?;
            }

            // Capture pre-final-norm hidden state for patch p (host
            // slab) so the host-side final-norm backward can read it.
            scratch.latent.hidden.copy_to_host(
                &mut state.latent_pre_norm_per_patch_host
                    [p * patch_dim..(p + 1) * patch_dim],
            );

            unsafe {
                rms_norm_resident(
                    hip_buf(&scratch.latent.hidden)?.device_ptr() as *const f32,
                    self.latent_final_norm_weight_dev.device_ptr() as *const f32,
                    hip_buf(&scratch.latent.normed)?.device_ptr() as *mut f32,
                    1, patch_dim, self.config.latent.norm_eps,
                )?;
            }
            batch.note_dispatch()?;

            copy_dense_to_slab(
                &scratch.latent.normed, p, patch_dim, &mut scratch.patch_reps,
            )?;
            batch.note_dispatch()?;
        }

        let mut patch_reps_for_decoder = GpuVec::try_hip(n_patches * patch_dim)?;
        copy_d2d(&scratch.patch_reps, &mut patch_reps_for_decoder, n_patches * patch_dim)?;

        // Stage 3: decoder forward-for-backward.
        self.decoder.forward_for_backward(
            batch,
            &patch_reps_for_decoder,
            boundaries,
            Some(&scratch.encoder.byte_reps),
            &mut scratch.decoder,
            &mut state.decoder_cache,
            byte_logits_out,
        )?;

        Ok(())
    }

    /// Backward pass for a sequence-level forward. Mirrors the chain
    /// in [`Self::forward_for_backward`] in reverse: decoder, latent,
    /// encoder. Weight gradients accumulate into the matching `*_grads`
    /// fields of `state`; per-stage activation buffers are read from
    /// `state.*_cache` / `state.latent_block_scratches`.
    pub fn backward(
        &mut self,
        batch: &HipBatch,
        bytes: &[u8],
        boundaries: &[usize],
        scratch: &mut BltScratch,
        state: &mut BltBackwardState,
        d_byte_logits: &GpuVec,
    ) -> Result<(), ResidencyError> {
        let n_bytes = bytes.len();
        let n_patches = boundaries.len().saturating_sub(1);
        let patch_dim = self.config.latent.patch_dim;
        let n_layers = self.latent.num_layers();

        debug_assert_eq!(d_byte_logits.len(), n_bytes * 256);
        debug_assert!(n_patches > 0);
        debug_assert!(n_patches <= self.config.latent.max_patches);

        // Zero inter-stage gradient buffers.
        zero_gpuvec_full(&mut state.d_patch_reps_post_latent)?;
        zero_gpuvec_full(&mut state.d_patch_reps_pre_latent)?;
        zero_gpuvec_full(&mut state.d_seed_byte_reps)?;

        // Stage 1: decoder backward.
        self.decoder.backward(
            batch,
            boundaries,
            &mut state.decoder_cache,
            d_byte_logits,
            &mut state.decoder_grads,
            &mut state.d_patch_reps_post_latent,
            Some(&mut state.d_seed_byte_reps),
            &mut scratch.decoder,
        )?;

        // Stage 2: latent backward.
        //
        // Forward at patch p: hidden_p = patch_reps[p]; for each layer
        // li, block.forward(hidden_p) writes K/V[layer=li, pos=p] and
        // updates hidden_p; final-norm produces post-latent
        // patch_reps[p]. After all patches the latent KV cache holds
        // positions 0..n_patches.
        //
        // Backward at patch p (reverse): start from
        // d_patch_reps_post_latent[p], walk final-norm backward (host),
        // then walk layers in reverse, restoring saved per-(patch,
        // layer) attn/mlp activations into the shared scratch before
        // each block.backward (matching GptModelResident::backward).
        // The KV cache stays at post-forward seq_len = n_patches; each
        // block.backward at position p reads K/V[0..=p] independently
        // of subsequent positions, so no truncation is needed.
        let mut dy_dev = GpuVec::try_hip(patch_dim)?;
        let mut d_pre_norm_dev = GpuVec::try_hip(patch_dim)?;

        let mut d_post_norm_host = vec![0.0f32; patch_dim];
        let mut pre_norm_host = vec![0.0f32; patch_dim];
        let mut d_pre_norm_host = vec![0.0f32; patch_dim];
        let mut dweight_final_norm_host = vec![0.0f32; patch_dim];

        for p in (0..n_patches).rev() {
            // Final-norm backward (host) for patch p.
            copy_slab_to_dense(
                &state.d_patch_reps_post_latent, p, patch_dim, &mut dy_dev,
            )?;
            batch.note_dispatch()?;
            dy_dev.copy_to_host(&mut d_post_norm_host);
            pre_norm_host.copy_from_slice(
                &state.latent_pre_norm_per_patch_host
                    [p * patch_dim..(p + 1) * patch_dim],
            );
            host_rms_norm_backward(
                &pre_norm_host, &d_post_norm_host, &mut d_pre_norm_host,
                patch_dim, 1.0, self.config.latent.norm_eps,
            );
            host_accumulate_rmsnorm_dweight(
                &pre_norm_host, &d_post_norm_host, &mut dweight_final_norm_host,
                patch_dim, self.config.latent.norm_eps,
            );
            d_pre_norm_dev.copy_from(&d_pre_norm_host);

            // Block stack backward (reverse). For each li, restore
            // per-(p, li) saved activations into shared scratch then
            // block.backward.
            for li in (0..n_layers).rev() {
                let block = &self.latent.blocks[li];
                restore_block_scratch_to_shared(
                    batch,
                    &state.latent_block_scratches[p][li],
                    &mut state.latent_attn_scratch,
                    &mut state.latent_mlp_scratch,
                    self.latent_kv_cache.max_seq_len(),
                    block,
                )?;

                block.backward(
                    batch,
                    &mut d_pre_norm_dev,
                    None,
                    &mut self.latent_kv_cache,
                    p,
                    &self.latent.rope,
                    &mut state.latent_attn_scratch,
                    &mut state.latent_attn_bwd,
                    &mut state.latent_mlp_scratch,
                    &mut state.latent_mlp_bwd,
                    &mut state.latent_block_scratches[p][li],
                    &mut state.latent_attn_grads[li],
                    &mut state.latent_mlp_grads[li],
                    /* recompute = */ false,
                )?;
            }

            // d_pre_norm_dev now holds d/d(latent input at patch p) —
            // the gradient flowing back through layer 0.
            copy_dense_to_slab(
                &d_pre_norm_dev, p, patch_dim, &mut state.d_patch_reps_pre_latent,
            )?;
            batch.note_dispatch()?;
        }

        // Fold the host-accumulated final-norm scale grad onto the
        // device accumulator (single H2D + add).
        let mut tmp = GpuVec::try_hip(patch_dim)?;
        tmp.copy_from(&dweight_final_norm_host);
        unsafe {
            use modgrad_device::backend::op::BinaryOpKind;
            use modgrad_device::backend::ops::op_tensor_resident;
            op_tensor_resident(
                hip_buf(&state.d_latent_final_norm_weight)?.device_ptr() as *const f32,
                hip_buf(&tmp)?.device_ptr() as *const f32,
                hip_buf_mut(&mut state.d_latent_final_norm_weight)?.device_ptr() as *mut f32,
                patch_dim, 1.0, 1.0, 0.0, BinaryOpKind::Add,
            )?;
        }
        batch.note_dispatch()?;

        // Stage 3: encoder backward.
        self.encoder.backward(
            batch,
            bytes,
            boundaries,
            &mut scratch.encoder,
            &mut state.encoder_bwd_scratch,
            &state.encoder_cache,
            &state.d_patch_reps_pre_latent,
            Some(&state.d_seed_byte_reps),
            &mut state.encoder_grads,
        )?;

        Ok(())
    }
}

// ─── Backward state ──────────────────────────────────────────

/// Caller-owned per-step backward state for [`BltModel`].
///
/// One allocation per training run; `forward_for_backward` repopulates
/// per call. `zero_resident` resets weight-grad accumulators across
/// micro-batches.
pub struct BltBackwardState {
    pub encoder_grads: LocalEncoderGrads,
    pub encoder_cache: LocalEncoderBwdCache,
    pub encoder_bwd_scratch: LocalEncoderBackwardScratch,

    // Latent — per-patch activation snapshots + per-layer grads.
    pub latent_attn_grads: Vec<AttentionResidentGrads>,
    pub latent_mlp_grads: Vec<SwigluResidentGrads>,
    pub latent_block_scratches: Vec<Vec<TransformerBlockScratch>>,
    pub latent_attn_scratch: AttentionScratch,
    pub latent_attn_bwd: AttentionBackwardScratch,
    pub latent_mlp_scratch: SwigluScratch,
    pub latent_mlp_bwd: SwigluBackwardScratch,
    /// Host snapshot of pre-final-norm latent hidden state per patch:
    /// `[max_patches × patch_dim]`. Captured at end of latent forward.
    pub latent_pre_norm_per_patch_host: Vec<f32>,
    /// `[patch_dim]` — accumulator for the latent's final RMSNorm
    /// scale grad. Final-norm scale is host-managed (matches the
    /// resident-path convention of all-ones scale during training).
    pub d_latent_final_norm_weight: GpuVec,

    pub decoder_grads: LocalDecoderGrads,
    pub decoder_cache: LocalDecoderBwdCache,

    /// `[max_patches × patch_dim]` — d/d(decoder K/V input = latent
    /// output). Filled by `decoder.backward`, consumed by latent
    /// backward.
    pub d_patch_reps_post_latent: GpuVec,
    /// `[max_patches × patch_dim]` — d/d(latent input = encoder
    /// output). Filled by latent backward, consumed by encoder
    /// backward.
    pub d_patch_reps_pre_latent: GpuVec,
    /// `[max_seq_len × byte_dim]` — d/d(decoder seed = encoder last-layer
    /// byte_reps). Filled by `decoder.backward`, consumed by
    /// `encoder.backward` as the path-#2 upstream.
    pub d_seed_byte_reps: GpuVec,
}

impl BltBackwardState {
    /// Allocate every gradient + activation buffer for one BLT training
    /// loop. Sized to the model's max patches / max bytes so the same
    /// state is reusable across calls of varying length.
    pub fn new(model: &BltModel) -> Result<Self, ResidencyError> {
        let cfg = &model.config;
        let pd = cfg.latent.patch_dim;
        let n_layers = model.latent.num_layers();
        let max_patches = cfg.latent.max_patches;
        let n_heads = cfg.latent.n_heads.max(1);
        let head_dim = cfg.latent.head_dim.max(1);
        let kv_dim = n_heads * head_dim;
        let mlp_dim = cfg.latent.mlp_dim;

        let encoder_grads = LocalEncoderGrads::zeros_for(&model.encoder)?;
        let encoder_cache = LocalEncoderBwdCache::new(&model.encoder, max_patches)?;
        let encoder_bwd_scratch = LocalEncoderBackwardScratch::new(&cfg.encoder)?;

        let mut latent_attn_grads = Vec::with_capacity(n_layers);
        let mut latent_mlp_grads = Vec::with_capacity(n_layers);
        let mut latent_block_scratches: Vec<Vec<TransformerBlockScratch>> =
            Vec::with_capacity(max_patches);
        for _ in 0..n_layers {
            latent_attn_grads.push(AttentionResidentGrads::new(pd, kv_dim)?);
            latent_mlp_grads.push(SwigluResidentGrads::new(pd, mlp_dim)?);
        }
        // [n_patches][n_layers] — outer indexed by patch, inner by layer
        // (matches the reverse-loop pattern in `backward`).
        for _ in 0..max_patches {
            let mut per_patch = Vec::with_capacity(n_layers);
            for _ in 0..n_layers {
                per_patch.push(TransformerBlockScratch::with_dims(
                    pd, kv_dim, mlp_dim, n_heads * max_patches,
                )?);
            }
            latent_block_scratches.push(per_patch);
        }
        let latent_attn_scratch =
            AttentionScratch::new(n_heads, head_dim, kv_dim, max_patches)?;
        let latent_attn_bwd =
            AttentionBackwardScratch::new(n_heads, head_dim, kv_dim, max_patches)?;
        let latent_mlp_scratch = SwigluScratch::new(pd, mlp_dim)?;
        let latent_mlp_bwd = SwigluBackwardScratch::new(pd, mlp_dim)?;

        let latent_pre_norm_per_patch_host = vec![0.0f32; max_patches * pd];
        let d_latent_final_norm_weight = GpuVec::try_hip(pd)?;

        let decoder_grads = LocalDecoderGrads::zeros(&model.decoder)?;
        let decoder_cache = LocalDecoderBwdCache::new(&model.decoder)?;

        let d_patch_reps_post_latent = GpuVec::try_hip(max_patches * pd)?;
        let d_patch_reps_pre_latent = GpuVec::try_hip(max_patches * pd)?;
        let d_seed_byte_reps = GpuVec::try_hip(
            cfg.encoder.max_seq_len * cfg.encoder.byte_dim,
        )?;

        Ok(Self {
            encoder_grads,
            encoder_cache,
            encoder_bwd_scratch,
            latent_attn_grads,
            latent_mlp_grads,
            latent_block_scratches,
            latent_attn_scratch,
            latent_attn_bwd,
            latent_mlp_scratch,
            latent_mlp_bwd,
            latent_pre_norm_per_patch_host,
            d_latent_final_norm_weight,
            decoder_grads,
            decoder_cache,
            d_patch_reps_post_latent,
            d_patch_reps_pre_latent,
            d_seed_byte_reps,
        })
    }

    /// Reset every weight-grad accumulator + inter-stage gradient
    /// buffer to zero. Caller drives once per train step.
    pub fn zero_resident(&mut self, batch: &HipBatch) -> Result<(), ResidencyError> {
        self.encoder_grads.zero_resident(batch)?;
        self.decoder_grads.zero_resident(batch)?;
        for g in self.latent_attn_grads.iter_mut() {
            zero_gpuvec_full(&mut g.dweight_q)?;
            zero_gpuvec_full(&mut g.dbias_q)?;
            zero_gpuvec_full(&mut g.dweight_k)?;
            zero_gpuvec_full(&mut g.dbias_k)?;
            zero_gpuvec_full(&mut g.dweight_v)?;
            zero_gpuvec_full(&mut g.dbias_v)?;
            zero_gpuvec_full(&mut g.dweight_o)?;
            zero_gpuvec_full(&mut g.dbias_o)?;
        }
        for g in self.latent_mlp_grads.iter_mut() {
            zero_gpuvec_full(&mut g.dweight_gate)?;
            zero_gpuvec_full(&mut g.dbias_gate)?;
            zero_gpuvec_full(&mut g.dweight_up)?;
            zero_gpuvec_full(&mut g.dbias_up)?;
            zero_gpuvec_full(&mut g.dweight_down)?;
            zero_gpuvec_full(&mut g.dbias_down)?;
        }
        zero_gpuvec_full(&mut self.d_latent_final_norm_weight)?;
        zero_gpuvec_full(&mut self.d_patch_reps_post_latent)?;
        zero_gpuvec_full(&mut self.d_patch_reps_pre_latent)?;
        zero_gpuvec_full(&mut self.d_seed_byte_reps)?;
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

/// Build the latent transformer + a `[patch_dim]` final-norm buffer.
/// Wraps the existing `GptModelResident::from_model` recipe from the
/// test harness in `modgrad-transformer`.
///
/// `pretrained_final_norm`:
/// - `None` — initialise the host model's `final_norm.scale` and the
///   parallel device buffer to all-ones (canonical RMSNorm-at-init
///   convention; correct for greenfield training).
/// - `Some(scale)` — copy `scale` (length checked by the caller) into
///   both the host `final_norm.scale` and the resident device buffer.
///   This is the byteification path: the BLT loads a pretrained
///   Qwen-class checkpoint whose final-norm scale has been trained away
///   from 1.0, so the latent must be initialised to the actual values
///   from safetensors rather than silently reset to ones.
fn build_latent(
    config: &BltLatentConfig,
    pretrained_final_norm: Option<&[f32]>,
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
    // Final-norm scale: pretrained override if supplied, else canonical
    // RMSNorm-at-init = ones. Length is validated by the public caller
    // (`with_pretrained_final_norm`); guard with `debug_assert` here so
    // a bad direct call to `build_latent` in this module fails loudly.
    let final_norm_scale: Vec<f32> = match pretrained_final_norm {
        Some(scale) => {
            debug_assert_eq!(scale.len(), pd);
            scale.to_vec()
        }
        None => vec![1.0f32; pd],
    };
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

/// Zero a `GpuVec::Hip` over its full capacity. H2D upload because the
/// device backend does not yet expose a resident memset — the buffers
/// involved are small (≤ KBs in practice for the latent grad bundle).
fn zero_gpuvec_full(g: &mut GpuVec) -> Result<(), ResidencyError> {
    let n = g.len();
    let zeros = vec![0.0f32; n];
    g.copy_from(&zeros);
    Ok(())
}

/// Restore the per-(patch, layer) snapshotted attn/mlp activations
/// from a [`TransformerBlockScratch`] into the shared
/// `attn_scratch` / `mlp_scratch` that `block.backward` reads.
///
/// Mirrors the pattern in `GptModelResident::backward` (which copies
/// from `state.block_scratches[li].saved_*` into `state.attn_scratch.*`
/// before each block.backward call).
fn restore_block_scratch_to_shared(
    batch: &HipBatch,
    block_scratch: &TransformerBlockScratch,
    attn_scratch: &mut AttentionScratch,
    mlp_scratch: &mut SwigluScratch,
    max_seq: usize,
    block: &modgrad_transformer::resident::TransformerBlockResident,
) -> Result<(), ResidencyError> {
    let model_dim = block.attn.model_dim;
    let kv_dim = block.attn.kv_dim;
    let num_heads = block.attn.num_heads;
    let mlp_dim = block.mlp.mlp_dim();

    unsafe fn d2d(
        dst: *mut std::os::raw::c_void,
        src: *const std::os::raw::c_void,
        n_bytes: usize,
    ) -> Result<(), ResidencyError> {
        use modgrad_device::backend::rocm::ffi;
        // `3` = hipMemcpyDeviceToDevice — the HIP enum value that
        // matches `HIP_D2D` constant used elsewhere in this file.
        let err = unsafe { ffi::hipMemcpy(dst, src, n_bytes, 3) };
        if err != 0 {
            return Err(ResidencyError::Backend(
                modgrad_device::backend::BackendError::Runtime(format!(
                    "restore_block_scratch D2D: {}", ffi::hip_err_str(err),
                )),
            ));
        }
        Ok(())
    }

    unsafe {
        d2d(
            hip_buf_mut(&mut attn_scratch.q_proj)?.device_ptr(),
            hip_buf(&block_scratch.saved_q_proj)?.device_ptr() as *const _,
            model_dim * 4,
        )?;
        d2d(
            hip_buf_mut(&mut attn_scratch.k_proj)?.device_ptr(),
            hip_buf(&block_scratch.saved_k_proj)?.device_ptr() as *const _,
            kv_dim * 4,
        )?;
        d2d(
            hip_buf_mut(&mut attn_scratch.v_proj)?.device_ptr(),
            hip_buf(&block_scratch.saved_v_proj)?.device_ptr() as *const _,
            kv_dim * 4,
        )?;
        d2d(
            hip_buf_mut(&mut attn_scratch.q_normed)?.device_ptr(),
            hip_buf(&block_scratch.saved_q_normed)?.device_ptr() as *const _,
            model_dim * 4,
        )?;
        d2d(
            hip_buf_mut(&mut attn_scratch.scores_tight)?.device_ptr(),
            hip_buf(&block_scratch.saved_scores_tight)?.device_ptr() as *const _,
            num_heads * max_seq * 4,
        )?;
        d2d(
            hip_buf_mut(&mut attn_scratch.head_out)?.device_ptr(),
            hip_buf(&block_scratch.saved_head_out)?.device_ptr() as *const _,
            model_dim * 4,
        )?;
        d2d(
            hip_buf_mut(&mut mlp_scratch.gate_out)?.device_ptr(),
            hip_buf(&block_scratch.saved_gate_out)?.device_ptr() as *const _,
            mlp_dim * 4,
        )?;
        d2d(
            hip_buf_mut(&mut mlp_scratch.up_out)?.device_ptr(),
            hip_buf(&block_scratch.saved_up_out)?.device_ptr() as *const _,
            mlp_dim * 4,
        )?;
        d2d(
            hip_buf_mut(&mut mlp_scratch.silu)?.device_ptr(),
            hip_buf(&block_scratch.saved_silu)?.device_ptr() as *const _,
            mlp_dim * 4,
        )?;
        d2d(
            hip_buf_mut(&mut mlp_scratch.hidden)?.device_ptr(),
            hip_buf(&block_scratch.saved_hidden)?.device_ptr() as *const _,
            mlp_dim * 4,
        )?;
    }
    for _ in 0..10 { batch.note_dispatch()?; }
    Ok(())
}

/// Host-side RMSNorm backward, single row, scalar `scale`. Mirrors the
/// formula in `decoder::rms_norm_backward_host` / the resident path's
/// `rms_norm_backward_per_head`.
fn host_rms_norm_backward(
    x: &[f32], dy: &[f32], dx: &mut [f32],
    head_dim: usize, scale: f32, eps: f32,
) {
    let n = head_dim as f32;
    let mean_sq: f32 = x.iter().take(head_dim).map(|&v| v * v).sum::<f32>() / n;
    let rms = (mean_sq + eps).sqrt();
    let inv_rms = 1.0 / rms;
    let inv_rms_sq = inv_rms * inv_rms;
    let acc: f32 = x.iter().take(head_dim).zip(dy.iter().take(head_dim))
        .map(|(&a, &b)| a * b).sum();
    for i in 0..head_dim {
        dx[i] = scale * inv_rms * (dy[i] - x[i] * acc * inv_rms_sq / n);
    }
}

/// Accumulate `dweight += dy * x / rms(x)` per component, host-side.
fn host_accumulate_rmsnorm_dweight(
    x: &[f32], dy: &[f32],
    dweight: &mut [f32], head_dim: usize, eps: f32,
) {
    let n = head_dim as f32;
    let mean_sq: f32 = x.iter().take(head_dim).map(|&v| v * v).sum::<f32>() / n;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    for i in 0..head_dim {
        dweight[i] += dy[i] * x[i] * inv_rms;
    }
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
// **Per-position backward intentionally unsupported.** The
// `LanguageModel` trait's `forward_for_backward_position` /
// `backward_position` are autoregressive per-token entry points;
// BLT processes patches as a unit (a forward needs the full byte
// sequence + patch boundaries), so those methods return
// `ResidencyError::WrongVariant`. The sequence-level backward lives
// on [`BltModel::forward_for_backward`] / [`BltModel::backward`]
// directly — a future `BltTrainer<BltModel>` will call those instead
// of going through the trait. `BackwardState` and
// `alloc_backward_state` are still wired so a downstream caller can
// allocate the per-step state via the trait.

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

    type BackwardState = BltBackwardState;

    fn alloc_backward_state(
        &self,
        _kv_cache: &KvCacheResident,
    ) -> Result<Self::BackwardState, ResidencyError> {
        BltBackwardState::new(self)
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
            expected: "BltModel sequence-level backward (call BltModel::forward_for_backward)",
            got: "per-position not supported",
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
            expected: "BltModel sequence-level backward (call BltModel::backward)",
            got: "per-position not supported",
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

    #[test]
    fn blt_backward_state_alloc_and_zero() {
        let _guard = HIP_TEST_LOCK.lock().unwrap();
        if !runtime_available() {
            eprintln!("hip runtime unavailable, skipping");
            return;
        }
        let config = tiny_config();
        let model = BltModel::new(config).expect("BltModel::new");
        let mut state = BltBackwardState::new(&model).expect("BltBackwardState::new");

        // Per-layer counts match the model.
        assert_eq!(state.latent_attn_grads.len(), model.latent.num_layers());
        assert_eq!(state.latent_mlp_grads.len(), model.latent.num_layers());
        // Per-patch outer × per-layer inner block-scratch grid.
        let max_patches = model.config.latent.max_patches;
        assert_eq!(state.latent_block_scratches.len(), max_patches);
        for per_patch in state.latent_block_scratches.iter() {
            assert_eq!(per_patch.len(), model.latent.num_layers());
        }
        // Inter-stage gradient buffers.
        let pd = model.config.latent.patch_dim;
        assert_eq!(state.d_patch_reps_post_latent.len(), max_patches * pd);
        assert_eq!(state.d_patch_reps_pre_latent.len(), max_patches * pd);
        // Final-norm scale grad shape.
        assert_eq!(state.d_latent_final_norm_weight.len(), pd);
        // Pre-norm host slab.
        assert_eq!(state.latent_pre_norm_per_patch_host.len(), max_patches * pd);

        // zero_resident must succeed without panicking.
        let batch = HipBatch::new();
        state.zero_resident(&batch).expect("BltBackwardState::zero_resident");
        let _ = batch.flush();
    }

    /// `with_pretrained_final_norm` must propagate the supplied scale
    /// into the resident `latent_final_norm_weight_dev` buffer (rather
    /// than silently overriding it with the all-ones init that
    /// `BltModel::new` uses). This is the byteification correctness
    /// check: load Qwen-class final-norm scale → BLT must serve it
    /// back, not the canonical RMSNorm-at-init = 1.0 vector.
    #[test]
    fn pretrained_final_norm_propagates() {
        let _guard = HIP_TEST_LOCK.lock().unwrap();
        if !runtime_available() {
            eprintln!("hip runtime unavailable, skipping");
            return;
        }
        if std::env::var("MODGRAD_SKIP_HIP_TESTS").is_ok() {
            return;
        }
        let cfg = tiny_config();
        let pd = cfg.latent.patch_dim;
        // Distinct, non-trivial scale — index-encoded so any silent
        // reset to ones / shuffle would be obvious in the assert.
        let scale: Vec<f32> = (0..pd).map(|i| 1.0 + (i as f32) * 0.01).collect();
        let model = BltModel::with_pretrained_final_norm(cfg.clone(), &scale)
            .expect("with_pretrained_final_norm");

        // Read back the resident device buffer and assert exact match.
        let mut readback = vec![0.0_f32; pd];
        model
            .latent_final_norm_weight_dev
            .copy_to_host(&mut readback)
            .expect("copy_to_host");
        for i in 0..pd {
            assert!(
                (readback[i] - scale[i]).abs() < 1e-6,
                "final_norm_weight_dev[{i}] should be {} not {}",
                scale[i],
                readback[i],
            );
        }

        // Length-mismatch path returns WrongVariant rather than
        // panicking — guard the byteify caller against bad config.
        let bad = vec![1.0_f32; pd + 1];
        let err = BltModel::with_pretrained_final_norm(cfg, &bad).err()
            .expect("length-mismatch must error");
        match err {
            ResidencyError::WrongVariant { .. } => (),
            other => panic!("expected WrongVariant, got {other:?}"),
        }
    }
}

