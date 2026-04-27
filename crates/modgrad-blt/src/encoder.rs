//! Local encoder: bytes → patch representations.
//!
//! Per BLT paper §3.2 (Pagnoni et al. 2024): a lightweight transformer
//! with `lE` byte-level layers (typically 1, paper §4.8). Each layer is
//! a transformer block followed by an encoder cross-attention block;
//! the cross-attention pools byte representations within each patch
//! into a patch query (paper §3.2.2).
//!
//! Initial patch query pooling + linear projection is part of the
//! cross-attention bridge owned by `crate::cross_attn` — by the time
//! the bridge produces `patch_reps_out` the byte→patch projection is
//! already applied (BLT §4.8 max-pool then linear).
//!
//! Owner: agent sasha (with [`crate::decoder`] and [`crate::model`]).
//!
//! Only available with `--features rocm`.

#![cfg(feature = "rocm")]
#![allow(clippy::too_many_arguments)]

use modgrad_codec::ngram_hash::NgramHashEmbeddings;
use modgrad_compute::backend::{GpuVec, ResidencyError};
use modgrad_compute::neuron::SimpleRng;
use modgrad_device::backend::{HipBatch, HipBuffer};
use modgrad_transformer::attention::{AttentionWeights, CausalSelfAttention};
use modgrad_transformer::block::TransformerBlock;
use modgrad_transformer::config::{
    GptConfig, MlpActivation, Precision, ResidualConfig, SmearConfig,
    ValueEmbedConfig, WindowPattern,
};
use modgrad_transformer::dims::*;
use modgrad_transformer::kv_cache_resident::KvCacheResident;
use modgrad_transformer::mlp::{Mlp, MlpWeights, SwigluMlp, SwigluWeights};
use modgrad_transformer::residual::ResidualLambdas;
use modgrad_transformer::resident::{
    AttentionBackwardScratch, AttentionResidentGrads, AttentionScratch,
    SwigluBackwardScratch, SwigluResidentGrads, SwigluScratch,
    TransformerBlockResident, TransformerBlockScratch,
};
use modgrad_transformer::rope::RotaryEmbedding;
use modgrad_transformer::tensor::Tensor2;

use crate::cross_attn::{
    cross_attn_encoder_backward, CrossAttention, CrossAttnBwdCache,
    CrossAttnConfig, CrossAttnDirection, CrossAttnGrads, CrossAttnScratch,
};

// ─── Config ──────────────────────────────────────────────────

/// Hyperparameters for [`LocalEncoder`].
///
/// Field meanings track BLT paper §3.2 / §4.8 verbatim where possible.
#[derive(Debug, Clone)]
pub struct LocalEncoderConfig {
    /// Number of byte-level transformer layers (`lE`). Paper §4.8 uses 1.
    pub n_layers: usize,
    /// Residual stream width on the byte side.
    pub byte_dim: usize,
    /// Residual stream width on the patch side. Cross-attention bridges
    /// `byte_dim → patch_dim`.
    pub patch_dim: usize,
    /// Number of attention heads in the byte transformer.
    pub n_heads: usize,
    /// Per-head dimension. `n_heads * head_dim == byte_dim`.
    pub head_dim: usize,
    /// FFN inner dimension (SwiGLU intermediate width).
    pub mlp_dim: usize,
    /// RMSNorm epsilon for both pre-norm sites in the byte block.
    pub norm_eps: f32,
    /// RoPE base frequency.
    pub rope_base: f32,
    /// Maximum byte sequence length (sizes the byte KV cache).
    pub max_seq_len: usize,
    /// Smallest n-gram size for the hash embedding augmentation.
    /// Paper §3.2.1 uses 3.
    pub ngram_min_n: usize,
    /// Largest n-gram size. Paper §3.2.1 uses 8.
    pub ngram_max_n: usize,
    /// Per-n hash table size. Paper §3.2.1 uses ~500K.
    pub ngram_vocab_per_n: usize,
}

impl LocalEncoderConfig {
    /// Validate dimensional consistency. Returns `Err` on a misconfigured
    /// shape so construction fails loudly rather than silently producing
    /// shape-mismatched buffers.
    pub fn validate(&self) -> Result<(), ResidencyError> {
        if self.n_heads == 0 || self.head_dim == 0 {
            return Err(ResidencyError::WrongVariant {
                expected: "n_heads > 0 and head_dim > 0",
                got: "zero head config",
            });
        }
        if self.n_heads * self.head_dim != self.byte_dim {
            return Err(ResidencyError::WrongVariant {
                expected: "n_heads * head_dim == byte_dim",
                got: "encoder dim mismatch",
            });
        }
        if self.ngram_max_n < self.ngram_min_n {
            return Err(ResidencyError::WrongVariant {
                expected: "ngram_max_n >= ngram_min_n",
                got: "invalid ngram range",
            });
        }
        Ok(())
    }

    /// Project the byte-level config into a [`GptConfig`] suitable for
    /// constructing host [`TransformerBlock`]s. We reuse the existing
    /// transformer machinery rather than reimplementing the byte
    /// transformer.
    pub(crate) fn to_gpt_config(&self, num_layers: usize) -> GptConfig {
        GptConfig {
            model_dim: ModelDim::new(self.byte_dim),
            num_heads: NumHeads::new(self.n_heads),
            num_kv_heads: NumKvHeads::new(self.n_heads),
            head_dim: HeadDim::new(self.head_dim),
            num_layers: NumLayers::new(num_layers.max(1)),
            // The host-side block knows nothing about a byte vocab — vocab
            // is consumed by the embedding table the encoder owns directly.
            // 256 keeps the validator happy.
            vocab_size: VocabSize::new(256),
            mlp_dim: MlpDim::new(self.mlp_dim),
            max_seq_len: SeqLen::new(self.max_seq_len),
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
            // x_lambda = 0 throughout collapses the residual stream to
            // the standard pre-norm transformer the resident path is
            // tuned for; matches `tiny_config` in `modgrad-transformer`.
            residual: ResidualConfig {
                resid_start: 1.0,
                resid_end: 1.0,
                x0_start: 0.0,
                x0_end: 0.0,
                backout_lambda: 0.0,
            },
            smear: SmearConfig::default(),
            precision: Precision::F32,
            norm_eps: self.norm_eps,
        }
    }
}

// ─── Encoder ─────────────────────────────────────────────────

/// Local byte-side encoder. See module docs.
pub struct LocalEncoder {
    /// Byte embedding table `[256 × byte_dim]` row-major on device.
    pub byte_embed_dev: HipBuffer,
    /// Hash n-gram embedding tables (paper §3.2.1). Host-side because
    /// augmentation is a sparse table lookup with `O(N · n_gram_sizes)`
    /// host cost — small relative to the resident matvec budget.
    pub ngram: NgramHashEmbeddings,
    /// `lE` byte-level transformer blocks.
    pub byte_layers: Vec<TransformerBlockResident>,
    /// One cross-attention bridge per byte layer.
    pub cross_attns: Vec<CrossAttention>,
    /// Byte-level KV cache. One slab per byte layer.
    pub byte_kv_cache: KvCacheResident,
    /// Byte-level RoPE (`head_dim` × `max_seq_len`).
    pub rope: RotaryEmbedding,
    /// Configuration carried alongside for shape introspection.
    pub config: LocalEncoderConfig,
}

impl LocalEncoder {
    /// Allocate device buffers, upload deterministic random weights,
    /// and build the byte transformer + cross-attention bridges.
    ///
    /// Random weights are seeded from the config so two `new` calls with
    /// the same config produce the same model — useful for the smoke
    /// test's deterministic assertions.
    pub fn new(config: LocalEncoderConfig) -> Result<Self, ResidencyError> {
        config.validate()?;
        let byte_dim = config.byte_dim;
        let n_layers = config.n_layers.max(1);

        // ── Byte embedding table ──
        let mut rng = SimpleRng::new(
            ((byte_dim * config.patch_dim).max(1) * n_layers.max(1)) as u64 ^ 0xB17E_B17E,
        );
        // Scale chosen to give ~unit variance on the augmented embedding
        // (BLT divides by `n_gram_sizes + 1` after the table sum).
        let embed_scale = 0.05_f32;
        let byte_embed: Vec<f32> = (0..256 * byte_dim)
            .map(|_| rng.next_normal() * embed_scale)
            .collect();
        let byte_embed_dev = HipBuffer::new(byte_embed.len() * 4)?;
        byte_embed_dev.copy_from_host(&byte_embed)?;

        // ── N-gram hash augmentation ──
        let ngram = NgramHashEmbeddings::new(
            byte_dim,
            config.ngram_vocab_per_n,
            config.ngram_min_n,
            config.ngram_max_n,
        );

        // ── Byte transformer blocks ──
        let gpt_config = config.to_gpt_config(n_layers);
        let lambdas = ResidualLambdas::from_config(&gpt_config.residual, gpt_config.num_layers);
        let kv_dim = gpt_config.num_kv_heads.get() * gpt_config.head_dim.get();

        let mut byte_layers = Vec::with_capacity(n_layers);
        for li in 0..n_layers {
            let attn_w = AttentionWeights {
                wq: Tensor2::new(randn(&mut rng, byte_dim * byte_dim, embed_scale), byte_dim, byte_dim).unwrap(),
                wk: Tensor2::new(randn(&mut rng, kv_dim * byte_dim, embed_scale), kv_dim, byte_dim).unwrap(),
                wv: Tensor2::new(randn(&mut rng, kv_dim * byte_dim, embed_scale), kv_dim, byte_dim).unwrap(),
                wo: Tensor2::new(randn(&mut rng, byte_dim * byte_dim, embed_scale), byte_dim, byte_dim).unwrap(),
            };
            let attn = CausalSelfAttention::new(attn_w, &gpt_config);

            let swiglu_w = SwigluWeights {
                gate: Tensor2::new(randn(&mut rng, config.mlp_dim * byte_dim, embed_scale), config.mlp_dim, byte_dim).unwrap(),
                up: Tensor2::new(randn(&mut rng, config.mlp_dim * byte_dim, embed_scale), config.mlp_dim, byte_dim).unwrap(),
                down: Tensor2::new(randn(&mut rng, byte_dim * config.mlp_dim, embed_scale), byte_dim, config.mlp_dim).unwrap(),
            };
            let swiglu = SwigluMlp::new(swiglu_w, gpt_config.model_dim, gpt_config.mlp_dim);

            // Build a host TransformerBlock with a placeholder ReLU² Mlp
            // that the resident path never invokes (it requires SwiGLU).
            let placeholder_mlp = Mlp::new(
                MlpWeights {
                    fc: Tensor2::zeros(config.mlp_dim, byte_dim),
                    proj: Tensor2::zeros(byte_dim, config.mlp_dim),
                },
                gpt_config.model_dim,
                gpt_config.mlp_dim,
            );
            let layer_idx = LayerIdx::new(li, gpt_config.num_layers).unwrap();
            let block = TransformerBlock::new(attn, placeholder_mlp, None, layer_idx, &gpt_config);

            let resident = TransformerBlockResident::from_block(&block, &swiglu, &lambdas, &gpt_config)?;
            byte_layers.push(resident);
        }

        // ── Cross-attention bridges ──
        let cross_cfg = CrossAttnConfig {
            byte_dim,
            patch_dim: config.patch_dim,
            num_heads: config.n_heads.max(1),
            head_dim: (config.patch_dim / config.n_heads.max(1)).max(1),
            norm_eps: config.norm_eps,
            direction: CrossAttnDirection::Encoder,
        };
        let mut cross_attns = Vec::with_capacity(n_layers);
        for li in 0..n_layers {
            // Per-layer deterministic seed so the cross-attn weights are
            // reproducible across reconstructions of the encoder.
            let seed = 0xC0FF_EE00_u64 ^ ((li as u64) << 16) ^ (byte_dim as u64);
            cross_attns.push(CrossAttention::new(&cross_cfg, seed)?);
        }

        // ── Byte KV cache ──
        let byte_kv_cache = KvCacheResident::new(
            n_layers,
            gpt_config.num_kv_heads.get(),
            gpt_config.head_dim.get(),
            gpt_config.max_seq_len.get(),
            byte_dim,
        )?;

        let rope = RotaryEmbedding::new(
            gpt_config.head_dim,
            gpt_config.max_seq_len,
            gpt_config.rope_base,
        );

        Ok(Self {
            byte_embed_dev,
            ngram,
            byte_layers,
            cross_attns,
            byte_kv_cache,
            rope,
            config,
        })
    }

    /// Number of byte-level layers (`lE`).
    #[inline]
    pub fn n_layers(&self) -> usize { self.byte_layers.len() }

    /// Byte-side residual width.
    #[inline]
    pub fn byte_dim(&self) -> usize { self.config.byte_dim }

    /// Patch-side residual width.
    #[inline]
    pub fn patch_dim(&self) -> usize { self.config.patch_dim }

    /// Forward bytes → patch reps.
    ///
    /// `bytes`: input byte sequence `[N]`.
    /// `boundaries`: patch boundaries `[0, b_1, b_2, …, N]`. The number
    /// of patches is `boundaries.len() - 1`. Bytes in `[boundaries[p],
    /// boundaries[p+1])` are pooled into patch `p`.
    /// `scratch`: caller-owned scratch reused across calls.
    /// `patch_reps_out`: `[n_patches × patch_dim]` row-major device
    /// buffer; written in-place.
    ///
    /// Caller must reset `byte_kv_cache` between calls if reusing the
    /// encoder for a fresh sequence (the cache `seq_len` accumulates
    /// across calls). For the smoke test we allocate per-call.
    pub fn forward(
        &mut self,
        batch: &HipBatch,
        bytes: &[u8],
        boundaries: &[usize],
        scratch: &mut LocalEncoderScratch,
        patch_reps_out: &mut GpuVec,
    ) -> Result<(), ResidencyError> {
        let n_bytes = bytes.len();
        let n_patches = boundaries.len().saturating_sub(1);
        let byte_dim = self.config.byte_dim;
        let patch_dim = self.config.patch_dim;

        debug_assert_eq!(scratch.byte_reps.len(), self.config.max_seq_len * byte_dim);
        debug_assert_eq!(patch_reps_out.len(), n_patches * patch_dim);
        debug_assert!(n_bytes <= self.config.max_seq_len);
        debug_assert!(n_patches > 0);
        debug_assert_eq!(boundaries[0], 0);
        debug_assert_eq!(boundaries[boundaries.len() - 1], n_bytes);

        // ── Stage 1: byte embedding + n-gram hash augmentation (host) ──
        //
        // Embedding lookup + n-gram augmentation is sparse; we do it on
        // the host then upload `[n_bytes × byte_dim]` once. The cost is
        // O(N · byte_dim · n_gram_sizes) host work, which is dominated
        // by the device-side block forward for any non-trivial N.
        let mut byte_embed_table = vec![0.0f32; 256 * byte_dim];
        self.byte_embed_dev.copy_to_host(&mut byte_embed_table)?;
        let augmented = self.ngram.embed_bytes(bytes, &byte_embed_table, byte_dim);
        debug_assert_eq!(augmented.len(), n_bytes * byte_dim);

        // ── Stage 2: byte transformer ──
        //
        // Loop bytes through each block as a per-token decode. The byte
        // KV cache must be reset by the caller between sequences;
        // positions run 0..n_bytes for this call.
        let mut hidden_dev = GpuVec::try_hip(byte_dim)?;
        let x0_dev = GpuVec::try_hip(byte_dim)?;

        // Initial byte_reps = augmented embeddings.
        upload_byte_reps(&augmented, n_bytes, byte_dim, &mut scratch.byte_reps)?;

        for (block, cross) in self.byte_layers.iter().zip(self.cross_attns.iter()) {
            // Per-byte forward through this block. We reload `hidden_dev`
            // from the current byte_reps slab so each token sees the
            // post-previous-layer state.
            for t in 0..n_bytes {
                copy_byte_rep_to_hidden(
                    &scratch.byte_reps, t, byte_dim, &mut hidden_dev,
                )?;
                batch.note_dispatch()?;

                block.forward(
                    batch,
                    &mut hidden_dev,
                    &x0_dev,
                    &mut self.byte_kv_cache,
                    t,
                    &self.rope,
                    &mut scratch.attn_scratch,
                    &mut scratch.mlp_scratch,
                    &mut scratch.block_scratch,
                )?;

                copy_hidden_to_byte_rep(
                    &hidden_dev, t, byte_dim, &mut scratch.byte_reps,
                )?;
                batch.note_dispatch()?;
            }

            // Encoder cross-attn: pool byte reps within each patch and
            // project to `patch_dim`. The bridge writes its full output
            // into `patch_reps_out`.
            cross.forward_encoder(
                batch,
                &scratch.byte_reps,
                boundaries,
                &mut scratch.cross_scratch,
                patch_reps_out,
            )?;
        }

        let _ = patch_dim;
        Ok(())
    }
}

// ─── Scratch ─────────────────────────────────────────────────

/// Caller-owned scratch buffers for [`LocalEncoder::forward`].
///
/// One allocation up front, reused across forward passes. Sized to
/// fit `max_seq_len` bytes through the byte transformer.
pub struct LocalEncoderScratch {
    /// `[max_seq_len × byte_dim]` row-major; rolling per-byte
    /// representations after the most-recent byte layer.
    pub byte_reps: GpuVec,
    /// Per-block forward scratch reused across the byte stack.
    pub block_scratch: TransformerBlockScratch,
    /// Per-block attention scratch.
    pub attn_scratch: AttentionScratch,
    /// Per-block SwiGLU scratch.
    pub mlp_scratch: SwigluScratch,
    /// Per-call cross-attn scratch.
    pub cross_scratch: CrossAttnScratch,
}

impl LocalEncoderScratch {
    /// Allocate scratch sized for the given encoder config.
    pub fn new(config: &LocalEncoderConfig) -> Result<Self, ResidencyError> {
        let byte_dim = config.byte_dim;
        let n_heads = config.n_heads.max(1);
        let head_dim = config.head_dim.max(1);
        let kv_dim = n_heads * head_dim;
        let mlp_dim = config.mlp_dim;
        let max_seq = config.max_seq_len;

        Ok(Self {
            byte_reps: GpuVec::try_hip(max_seq * byte_dim)?,
            block_scratch: TransformerBlockScratch::with_dims(
                byte_dim, kv_dim, mlp_dim, n_heads * max_seq,
            )?,
            attn_scratch: AttentionScratch::new(n_heads, head_dim, kv_dim, max_seq)?,
            mlp_scratch: SwigluScratch::new(byte_dim, mlp_dim)?,
            cross_scratch: CrossAttnScratch::new(
                &CrossAttnConfig {
                    byte_dim: config.byte_dim,
                    patch_dim: config.patch_dim,
                    num_heads: config.n_heads.max(1),
                    head_dim: (config.patch_dim / config.n_heads.max(1)).max(1),
                    norm_eps: config.norm_eps,
                    direction: CrossAttnDirection::Encoder,
                },
                /* max_kv_len = */ max_seq,
                /* max_n_bytes = */ max_seq,
                /* max_n_patches = */ max_seq,
            )?,
        })
    }
}

// ─── Helpers ─────────────────────────────────────────────────

/// Sample `n` floats from a `N(0, scale²)` distribution.
fn randn(rng: &mut SimpleRng, n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|_| rng.next_normal() * scale).collect()
}

/// Upload `augmented[0..n_bytes * byte_dim]` into the leading rows of
/// `byte_reps`. Tail rows (`n_bytes..max_seq`) are left untouched —
/// the per-byte forward only reads positions `0..n_bytes`.
fn upload_byte_reps(
    augmented: &[f32],
    n_bytes: usize,
    byte_dim: usize,
    byte_reps: &mut GpuVec,
) -> Result<(), ResidencyError> {
    let buf = match byte_reps {
        GpuVec::Hip(b) => b,
        other => return Err(ResidencyError::WrongVariant {
            expected: "Hip", got: other.variant_name(),
        }),
    };
    // copy_from_host length-checks against the buffer's allocated size,
    // not the f32 count we want — use the slice length directly.
    let n = n_bytes * byte_dim;
    buf.copy_from_host(&augmented[..n])?;
    Ok(())
}

/// D2D copy one byte rep `byte_reps[t * byte_dim..(t+1) * byte_dim]` →
/// `hidden_dev[..byte_dim]`.
fn copy_byte_rep_to_hidden(
    byte_reps: &GpuVec,
    t: usize,
    byte_dim: usize,
    hidden_dev: &mut GpuVec,
) -> Result<(), ResidencyError> {
    use modgrad_device::backend::rocm::ffi;
    let src = match byte_reps {
        GpuVec::Hip(b) => b,
        other => return Err(ResidencyError::WrongVariant {
            expected: "Hip", got: other.variant_name(),
        }),
    };
    let dst = match hidden_dev {
        GpuVec::Hip(b) => b,
        other => return Err(ResidencyError::WrongVariant {
            expected: "Hip", got: other.variant_name(),
        }),
    };
    let off_bytes = t * byte_dim * 4;
    let n_bytes = byte_dim * 4;
    const HIP_D2D: std::os::raw::c_int = 3;
    let err = unsafe {
        ffi::hipMemcpy(
            dst.device_ptr(),
            (src.device_ptr() as *const u8).add(off_bytes) as *const std::os::raw::c_void,
            n_bytes,
            HIP_D2D,
        )
    };
    if err != 0 {
        return Err(ResidencyError::Backend(
            modgrad_device::backend::BackendError::Runtime(format!(
                "byte_rep → hidden D2D: {}", ffi::hip_err_str(err),
            )),
        ));
    }
    Ok(())
}

// ─── Backward — grads, cache, forward_for_backward, backward ────

/// Per-layer device-resident weight gradient bundle for [`LocalEncoder`].
///
/// Mirrors the model: each byte transformer layer has the standard
/// attention + SwiGLU resident grad pair (re-used from
/// `modgrad-transformer`), plus a [`CrossAttnGrads`] for the per-layer
/// cross-attention bridge. The byte embedding gradient is sparse —
/// only the rows corresponding to bytes seen in the call get non-zero
/// values — but we hold the full `[256 × byte_dim]` slab for parity
/// with `GptBackwardState::d_embed`.
pub struct LocalEncoderGrads {
    /// One per byte transformer layer.
    pub attn_grads: Vec<AttentionResidentGrads>,
    /// One per byte transformer layer (SwiGLU MLP).
    pub mlp_grads: Vec<SwigluResidentGrads>,
    /// One per byte layer (each layer has its own cross-attn bridge).
    pub cross_attn_grads: Vec<CrossAttnGrads>,
    /// `[256 × byte_dim]` row-major. Sparse: row `b` accumulates the
    /// gradient for byte id `b`.
    pub d_byte_embed: GpuVec,
    byte_dim: usize,
}

impl LocalEncoderGrads {
    /// Allocate fresh zero grads for the given encoder.
    pub fn zeros_for(encoder: &LocalEncoder) -> Result<Self, ResidencyError> {
        let n_layers = encoder.n_layers();
        let byte_dim = encoder.byte_dim();
        let n_heads = encoder.config.n_heads.max(1);
        let head_dim = encoder.config.head_dim.max(1);
        let kv_dim = n_heads * head_dim;
        let mlp_dim = encoder.config.mlp_dim;

        let mut attn_grads = Vec::with_capacity(n_layers);
        let mut mlp_grads = Vec::with_capacity(n_layers);
        let mut cross_attn_grads = Vec::with_capacity(n_layers);
        let cross_cfg = CrossAttnConfig {
            byte_dim,
            patch_dim: encoder.patch_dim(),
            num_heads: n_heads,
            head_dim: (encoder.patch_dim() / n_heads).max(1),
            norm_eps: encoder.config.norm_eps,
            direction: CrossAttnDirection::Encoder,
        };
        for _ in 0..n_layers {
            attn_grads.push(AttentionResidentGrads::new(byte_dim, kv_dim)?);
            mlp_grads.push(SwigluResidentGrads::new(byte_dim, mlp_dim)?);
            cross_attn_grads.push(CrossAttnGrads::zeros_for_config(&cross_cfg)?);
        }
        let d_byte_embed = GpuVec::try_hip(256 * byte_dim)?;
        Ok(Self { attn_grads, mlp_grads, cross_attn_grads, d_byte_embed, byte_dim })
    }

    /// Reset every accumulator to zero.
    pub fn zero_resident(&mut self, batch: &HipBatch) -> Result<(), ResidencyError> {
        // Attn / MLP grads: re-allocate-style zero via H2D upload.
        // TODO(blt-bwd): replace with `memset_resident` once exposed.
        for g in &mut self.attn_grads {
            zero_gpuvec(&mut g.dweight_q)?;
            zero_gpuvec(&mut g.dbias_q)?;
            zero_gpuvec(&mut g.dweight_k)?;
            zero_gpuvec(&mut g.dbias_k)?;
            zero_gpuvec(&mut g.dweight_v)?;
            zero_gpuvec(&mut g.dbias_v)?;
            zero_gpuvec(&mut g.dweight_o)?;
            zero_gpuvec(&mut g.dbias_o)?;
        }
        for g in &mut self.mlp_grads {
            zero_gpuvec(&mut g.dweight_gate)?;
            zero_gpuvec(&mut g.dbias_gate)?;
            zero_gpuvec(&mut g.dweight_up)?;
            zero_gpuvec(&mut g.dbias_up)?;
            zero_gpuvec(&mut g.dweight_down)?;
            zero_gpuvec(&mut g.dbias_down)?;
        }
        for g in &mut self.cross_attn_grads {
            g.zero_resident(batch)?;
        }
        let zeros = vec![0.0f32; 256 * self.byte_dim];
        self.d_byte_embed.copy_from(&zeros);
        Ok(())
    }
}

/// Per-call backward cache for [`LocalEncoder::forward_for_backward`].
///
/// Holds one [`TransformerBlockScratch`] per (layer, byte position) so
/// each token's per-layer activations are recoverable in backward.
/// (The reference `GptBackwardState` reuses one scratch per layer
/// because it processes one token; the encoder iterates over many
/// bytes per layer so we need per-byte scratch.) It also retains a
/// [`CrossAttnBwdCache`] per layer for the cross-attention bridge.
pub struct LocalEncoderBwdCache {
    /// `[n_layers][n_bytes]` block scratch (per-byte forward
    /// activations: attn_input, attn_normed, attn_out, mlp_input,
    /// mlp_normed, mlp_out, plus the saved q/k/v/scores/head/silu
    /// snapshots).
    pub block_scratches: Vec<Vec<TransformerBlockScratch>>,
    /// One cross-attn cache per layer.
    pub cross_caches: Vec<CrossAttnBwdCache>,
    /// `[n_bytes × byte_dim]` host snapshot of the augmented embedding
    /// after stage 1 (byte embed + n-gram). Used by backward to know
    /// what the input to layer 0's attention was so the byte-embed
    /// gradient can be scattered correctly.
    pub augmented_input_host: Vec<f32>,
    /// `[n_bytes]` byte ids for the current call (for scattering the
    /// embedding gradient).
    pub byte_ids: Vec<u8>,
    /// `[n_layers][n_bytes × byte_dim]` host snapshot of the byte_reps
    /// at each layer's *output*. Layer 0 reads `augmented_input_host`,
    /// layer 1 reads `layer_outputs[0]`, etc.
    pub layer_outputs_host: Vec<Vec<f32>>,
    cap_n_bytes: usize,
}

impl LocalEncoderBwdCache {
    pub fn new(
        encoder: &LocalEncoder,
        max_n_patches: usize,
    ) -> Result<Self, ResidencyError> {
        let n_layers = encoder.n_layers();
        let byte_dim = encoder.byte_dim();
        let n_heads = encoder.config.n_heads.max(1);
        let head_dim = encoder.config.head_dim.max(1);
        let kv_dim = n_heads * head_dim;
        let mlp_dim = encoder.config.mlp_dim;
        let max_seq = encoder.config.max_seq_len;
        let cross_cfg = CrossAttnConfig {
            byte_dim,
            patch_dim: encoder.patch_dim(),
            num_heads: n_heads,
            head_dim: (encoder.patch_dim() / n_heads).max(1),
            norm_eps: encoder.config.norm_eps,
            direction: CrossAttnDirection::Encoder,
        };

        let mut block_scratches = Vec::with_capacity(n_layers);
        let mut cross_caches = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            let mut layer_scratches = Vec::with_capacity(max_seq);
            for _ in 0..max_seq {
                layer_scratches.push(TransformerBlockScratch::with_dims(
                    byte_dim, kv_dim, mlp_dim, n_heads * max_seq,
                )?);
            }
            block_scratches.push(layer_scratches);
            cross_caches.push(CrossAttnBwdCache::new(
                &cross_cfg,
                /* max_n_queries = */ max_n_patches,
                /* max_kv_len    = */ max_seq,
            )?);
        }
        Ok(Self {
            block_scratches,
            cross_caches,
            augmented_input_host: vec![0.0; max_seq * byte_dim],
            byte_ids: vec![0u8; max_seq],
            layer_outputs_host: vec![vec![0.0; max_seq * byte_dim]; n_layers],
            cap_n_bytes: max_seq,
        })
    }
}

impl LocalEncoder {
    /// Forward pass that additionally populates `cache` for backward.
    ///
    /// Same semantics as [`LocalEncoder::forward`]; the only differences
    /// are (a) per-byte block forward uses `forward_for_backward` so
    /// activations are saved into `cache.block_scratches[li][t]`, and
    /// (b) the cross-attention bridge uses
    /// [`CrossAttention::forward_encoder_for_backward`] so the cross-
    /// attn cache is populated for the per-layer backward call.
    pub fn forward_for_backward(
        &mut self,
        batch: &HipBatch,
        bytes: &[u8],
        boundaries: &[usize],
        scratch: &mut LocalEncoderScratch,
        cache: &mut LocalEncoderBwdCache,
        patch_reps_out: &mut GpuVec,
    ) -> Result<(), ResidencyError> {
        let n_bytes = bytes.len();
        let n_patches = boundaries.len().saturating_sub(1);
        let byte_dim = self.config.byte_dim;
        let patch_dim = self.config.patch_dim;
        debug_assert!(n_bytes <= cache.cap_n_bytes);
        debug_assert_eq!(patch_reps_out.len(), n_patches * patch_dim);

        // Stage 1: byte embed + n-gram augmentation (host).
        let mut byte_embed_table = vec![0.0f32; 256 * byte_dim];
        self.byte_embed_dev.copy_to_host(&mut byte_embed_table)?;
        let augmented = self.ngram.embed_bytes(bytes, &byte_embed_table, byte_dim);
        cache.augmented_input_host[..n_bytes * byte_dim]
            .copy_from_slice(&augmented[..n_bytes * byte_dim]);
        cache.byte_ids[..n_bytes].copy_from_slice(bytes);

        // Stage 2: byte transformer with per-byte forward_for_backward.
        let mut hidden_dev = GpuVec::try_hip(byte_dim)?;
        let x0_dev = GpuVec::try_hip(byte_dim)?;
        upload_byte_reps(&augmented, n_bytes, byte_dim, &mut scratch.byte_reps)?;

        for (li, (block, cross)) in self.byte_layers.iter()
            .zip(self.cross_attns.iter()).enumerate() {
            for t in 0..n_bytes {
                copy_byte_rep_to_hidden(
                    &scratch.byte_reps, t, byte_dim, &mut hidden_dev,
                )?;
                batch.note_dispatch()?;

                block.forward_for_backward(
                    batch,
                    &mut hidden_dev,
                    &x0_dev,
                    &mut self.byte_kv_cache,
                    t,
                    &self.rope,
                    &mut scratch.attn_scratch,
                    &mut scratch.mlp_scratch,
                    &mut cache.block_scratches[li][t],
                )?;

                copy_hidden_to_byte_rep(
                    &hidden_dev, t, byte_dim, &mut scratch.byte_reps,
                )?;
                batch.note_dispatch()?;
            }

            // Snapshot this layer's byte_reps to the host cache so
            // backward can stage the upstream gradient against the
            // matching forward state.
            let layer_out = &mut cache.layer_outputs_host[li][..n_bytes * byte_dim];
            scratch.byte_reps.copy_to_host(layer_out);

            cross.forward_encoder_for_backward(
                batch,
                &scratch.byte_reps,
                boundaries,
                &mut scratch.cross_scratch,
                &mut cache.cross_caches[li],
                patch_reps_out,
            )?;
        }
        Ok(())
    }

    /// Backward pass.
    ///
    /// `d_patch_reps`: upstream gradient on the encoder's output
    /// `[n_patches × patch_dim]`.
    ///
    /// Walks every layer's cross-attention backward (accumulating into
    /// `grads.cross_attn_grads[li]`), folds the result with the
    /// upstream byte-side gradient, then walks each transformer block
    /// backward in reverse byte order, accumulating per-layer attn /
    /// mlp grads. The byte-embed gradient (sparse, per-byte-id) is
    /// scattered into `grads.d_byte_embed` row-by-row.
    ///
    /// Note on shape contracts: per-layer cross-attn backward writes
    /// `+=` into the byte-side gradient buffer. Across layers we
    /// process the cross-attn back-prop first (using each layer's
    /// snapshotted `byte_reps`), accumulate into a single per-layer
    /// `d_byte_in` buffer, then the block backward extends that with
    /// the gradient flowing through the transformer block.
    pub fn backward(
        &mut self,
        batch: &HipBatch,
        bytes: &[u8],
        boundaries: &[usize],
        scratch: &mut LocalEncoderScratch,
        bwd_scratch: &mut LocalEncoderBackwardScratch,
        cache: &LocalEncoderBwdCache,
        d_patch_reps: &GpuVec,
        grads: &mut LocalEncoderGrads,
    ) -> Result<(), ResidencyError> {
        let n_bytes = bytes.len();
        let n_patches = boundaries.len().saturating_sub(1);
        let byte_dim = self.config.byte_dim;
        let patch_dim = self.config.patch_dim;

        debug_assert_eq!(d_patch_reps.len(), n_patches * patch_dim);

        // Per-layer running gradient w.r.t. byte_reps at this layer's
        // input. Initialised to zero each time.
        let zero_byte_reps = vec![0.0f32; n_bytes * byte_dim];

        // Walk layers in reverse so the gradient flowing into each
        // layer's input becomes the upstream for the next-shallower
        // layer.
        let n_layers = self.byte_layers.len();
        let mut d_layer_input = GpuVec::try_hip(n_bytes * byte_dim)?;
        // Initial upstream into the deepest layer's cross-attn output:
        // d_patch_reps. The cross-attn at every layer writes into the
        // *same* `patch_reps_out` (no residual chain across layers in
        // the current encoder); only the deepest layer's gradient flows
        // back through its cross-attn. Earlier layers' cross-attn
        // outputs were overwritten by later layers, so their gradient
        // is zero at the cross-attn level. This matches the
        // `LocalEncoder::forward` pattern where every layer overwrites
        // patch_reps_out and the final write is the only one observed.
        // TODO(blt-bwd): if a future revision adds per-layer skip
        // connections from cross-attn to model output, this needs to
        // accumulate across all `li`.
        for li in (0..n_layers).rev() {
            d_layer_input.copy_from(&zero_byte_reps);
            let last_layer = li == n_layers - 1;
            if last_layer {
                cross_attn_encoder_backward(
                    batch,
                    &self.cross_attns[li],
                    /* byte_reps */ &scratch.byte_reps, // unused: cache holds the values
                    boundaries,
                    &cache.cross_caches[li],
                    d_patch_reps,
                    &mut grads.cross_attn_grads[li],
                    &mut d_layer_input,
                    &mut scratch.cross_scratch,
                )?;
            }

            // Block backward — reverse byte order so each token's KV-
            // cache view is consistent with the forward-time state at
            // that position. The host-side `d_layer_input` slab feeds
            // the per-byte backward via `bwd_scratch.dy_per_byte`.
            let mut d_layer_input_host = vec![0.0f32; n_bytes * byte_dim];
            d_layer_input.copy_to_host(&mut d_layer_input_host);

            for t in (0..n_bytes).rev() {
                bwd_scratch.dy_per_byte.copy_from(
                    &d_layer_input_host[t * byte_dim..(t + 1) * byte_dim],
                );
                self.byte_layers[li].backward(
                    batch,
                    &mut bwd_scratch.dy_per_byte,
                    None,
                    &mut self.byte_kv_cache,
                    t,
                    &self.rope,
                    &mut scratch.attn_scratch,
                    &mut bwd_scratch.attn_bwd,
                    &mut scratch.mlp_scratch,
                    &mut bwd_scratch.mlp_bwd,
                    &mut bwd_scratch.block_scratch_replay,
                    &mut grads.attn_grads[li],
                    &mut grads.mlp_grads[li],
                    /* recompute = */ true,
                )?;
                // After backward, dy_per_byte holds dx for this byte.
                let mut dx_host = vec![0.0f32; byte_dim];
                bwd_scratch.dy_per_byte.copy_to_host(&mut dx_host);
                d_layer_input_host[t * byte_dim..(t + 1) * byte_dim]
                    .copy_from_slice(&dx_host);
            }
            // d_layer_input_host now holds d/d(layer_input). For the
            // shallower layer this becomes the upstream into its
            // cross-attn-output buffer; for layer 0 this is the byte-
            // embed gradient.
            if li > 0 {
                d_layer_input.copy_from(&d_layer_input_host);
                // The cross-attn at layer (li-1) has already been
                // skipped (cross_attn output is overwritten by later
                // layers). The `d_layer_input` here is the gradient
                // flowing through layer li's block back to layer
                // (li-1)'s block output. We hand it off via
                // `bwd_scratch.d_byte_carry`.
                bwd_scratch.d_byte_carry.copy_from(&d_layer_input_host);
            } else {
                // Layer 0: scatter into d_byte_embed by byte id.
                scatter_byte_embed_grad(
                    &d_layer_input_host, &cache.byte_ids[..n_bytes],
                    n_bytes, byte_dim,
                    &mut grads.d_byte_embed,
                )?;
            }
        }
        Ok(())
    }
}

/// Per-call backward scratch for [`LocalEncoder::backward`]. Holds the
/// transient per-byte gradient buffers, one block-scratch replay slot
/// (used in the activation-checkpointing path), and the shared
/// attention / MLP backward scratches.
pub struct LocalEncoderBackwardScratch {
    /// `[byte_dim]` — per-byte rolling gradient (in-and-out of block.backward).
    pub dy_per_byte: GpuVec,
    /// `[n_bytes × byte_dim]` — staging for the layer-to-layer carry.
    pub d_byte_carry: GpuVec,
    /// Single replay slot for the per-byte block backward. The encoder
    /// uses recompute mode (the per-byte cache holds only the per-byte
    /// `attn_input` snapshot; the rest is regenerated on demand).
    pub block_scratch_replay: TransformerBlockScratch,
    pub attn_bwd: AttentionBackwardScratch,
    pub mlp_bwd: SwigluBackwardScratch,
}

impl LocalEncoderBackwardScratch {
    pub fn new(config: &LocalEncoderConfig) -> Result<Self, ResidencyError> {
        let byte_dim = config.byte_dim;
        let n_heads = config.n_heads.max(1);
        let head_dim = config.head_dim.max(1);
        let kv_dim = n_heads * head_dim;
        let mlp_dim = config.mlp_dim;
        let max_seq = config.max_seq_len;
        Ok(Self {
            dy_per_byte: GpuVec::try_hip(byte_dim)?,
            d_byte_carry: GpuVec::try_hip(max_seq * byte_dim)?,
            block_scratch_replay: TransformerBlockScratch::with_dims(
                byte_dim, kv_dim, mlp_dim, n_heads * max_seq,
            )?,
            attn_bwd: AttentionBackwardScratch::new(n_heads, head_dim, kv_dim, max_seq)?,
            mlp_bwd: SwigluBackwardScratch::new(byte_dim, mlp_dim)?,
        })
    }
}

/// Scatter `d_input[t, :]` into `d_byte_embed[byte_ids[t], :]` for each
/// byte position `t`. Multiple positions sharing the same byte id sum
/// into the same row.
fn scatter_byte_embed_grad(
    d_input_host: &[f32],
    byte_ids: &[u8],
    n_bytes: usize,
    byte_dim: usize,
    d_byte_embed: &mut GpuVec,
) -> Result<(), ResidencyError> {
    let mut d_embed_host = vec![0.0f32; 256 * byte_dim];
    d_byte_embed.copy_to_host(&mut d_embed_host);
    for t in 0..n_bytes {
        let id = byte_ids[t] as usize;
        let src = &d_input_host[t * byte_dim..(t + 1) * byte_dim];
        let dst = &mut d_embed_host[id * byte_dim..(id + 1) * byte_dim];
        for i in 0..byte_dim {
            dst[i] += src[i];
        }
    }
    d_byte_embed.copy_from(&d_embed_host);
    Ok(())
}

/// Zero a `GpuVec::Hip` slab. Uses a host upload until a resident
/// memset op is exposed.
fn zero_gpuvec(g: &mut GpuVec) -> Result<(), ResidencyError> {
    let n = g.len();
    let zeros = vec![0.0f32; n];
    g.copy_from(&zeros);
    Ok(())
}

/// D2D copy `hidden_dev[..byte_dim]` → `byte_reps[t * byte_dim..]`.
fn copy_hidden_to_byte_rep(
    hidden_dev: &GpuVec,
    t: usize,
    byte_dim: usize,
    byte_reps: &mut GpuVec,
) -> Result<(), ResidencyError> {
    use modgrad_device::backend::rocm::ffi;
    let src = match hidden_dev {
        GpuVec::Hip(b) => b,
        other => return Err(ResidencyError::WrongVariant {
            expected: "Hip", got: other.variant_name(),
        }),
    };
    let dst = match byte_reps {
        GpuVec::Hip(b) => b,
        other => return Err(ResidencyError::WrongVariant {
            expected: "Hip", got: other.variant_name(),
        }),
    };
    let off_bytes = t * byte_dim * 4;
    let n_bytes = byte_dim * 4;
    const HIP_D2D: std::os::raw::c_int = 3;
    let err = unsafe {
        ffi::hipMemcpy(
            (dst.device_ptr() as *mut u8).add(off_bytes) as *mut std::os::raw::c_void,
            src.device_ptr() as *const std::os::raw::c_void,
            n_bytes,
            HIP_D2D,
        )
    };
    if err != 0 {
        return Err(ResidencyError::Backend(
            modgrad_device::backend::BackendError::Runtime(format!(
                "hidden → byte_rep D2D: {}", ffi::hip_err_str(err),
            )),
        ));
    }
    Ok(())
}

// ─── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test for the local encoder backward path. Builds a tiny
    /// encoder, drives `forward_for_backward` then `backward`, and
    /// asserts the gradient buffers are finite and non-zero.
    ///
    /// Skipped without a real HIP device — the `try_hip` allocator
    /// returns Err which the test treats as "no device, skip".
    #[test]
    fn local_encoder_backward_smoke() {
        if std::env::var("MODGRAD_SKIP_HIP_TESTS").is_ok() { return; }

        const BYTE_DIM: usize = 8;
        const PATCH_DIM: usize = 8;
        const N_HEADS: usize = 2;
        const HEAD_DIM: usize = 4;
        const MLP_DIM: usize = 16;
        const N_BYTES: usize = 8;
        const N_PATCHES: usize = 2;
        const MAX_SEQ: usize = 16;

        let cfg = LocalEncoderConfig {
            n_layers: 1,
            byte_dim: BYTE_DIM,
            patch_dim: PATCH_DIM,
            n_heads: N_HEADS,
            head_dim: HEAD_DIM,
            mlp_dim: MLP_DIM,
            norm_eps: 1e-5,
            rope_base: 10000.0,
            max_seq_len: MAX_SEQ,
            ngram_min_n: 3,
            ngram_max_n: 4,
            ngram_vocab_per_n: 16,
        };

        let mut encoder = match LocalEncoder::new(cfg.clone()) {
            Ok(e) => e, Err(_) => return,
        };
        let mut scratch = match LocalEncoderScratch::new(&cfg) {
            Ok(s) => s, Err(_) => return,
        };
        let mut bwd_scratch = match LocalEncoderBackwardScratch::new(&cfg) {
            Ok(s) => s, Err(_) => return,
        };
        let mut cache = match LocalEncoderBwdCache::new(&encoder, N_PATCHES) {
            Ok(c) => c, Err(_) => return,
        };
        let mut grads = match LocalEncoderGrads::zeros_for(&encoder) {
            Ok(g) => g, Err(_) => return,
        };

        let batch = HipBatch::new();
        let _ = grads.zero_resident(&batch);

        let bytes: Vec<u8> = (0..N_BYTES as u8).collect();
        let boundaries = vec![0, 4, N_BYTES];
        let mut patch_reps_out = match GpuVec::try_hip(N_PATCHES * PATCH_DIM) {
            Ok(v) => v, Err(_) => return,
        };

        encoder.byte_kv_cache.reset();
        encoder.forward_for_backward(
            &batch, &bytes, &boundaries, &mut scratch, &mut cache, &mut patch_reps_out,
        ).expect("forward_for_backward");

        // Synthetic upstream gradient.
        let d_patch_host: Vec<f32> = (0..N_PATCHES * PATCH_DIM)
            .map(|i| ((i as f32) * 0.01) - 0.05).collect();
        let mut d_patch_reps = match GpuVec::try_hip(N_PATCHES * PATCH_DIM) {
            Ok(v) => v, Err(_) => return,
        };
        d_patch_reps.copy_from(&d_patch_host);

        encoder.backward(
            &batch, &bytes, &boundaries, &mut scratch, &mut bwd_scratch,
            &cache, &d_patch_reps, &mut grads,
        ).expect("encoder backward");
        batch.flush().expect("flush");

        // Spot-check finiteness on the cross-attn weight grads + byte
        // embed grad. Non-zero on at least one of them is enough — the
        // exact magnitude is the integration-test territory.
        let inner = N_HEADS * HEAD_DIM;
        let mut dwq = vec![0.0f32; inner * BYTE_DIM];
        grads.cross_attn_grads[0].dweight_q.copy_to_host(&mut dwq);
        let mut sumsq = 0.0f32;
        for &v in &dwq {
            assert!(v.is_finite(), "cross-attn dweight_q non-finite");
            sumsq += v * v;
        }
        assert!(sumsq > 0.0, "cross-attn dweight_q should be non-zero");

        let mut d_embed = vec![0.0f32; 256 * BYTE_DIM];
        grads.d_byte_embed.copy_to_host(&mut d_embed);
        for &v in &d_embed { assert!(v.is_finite(), "d_byte_embed non-finite"); }
    }
}
