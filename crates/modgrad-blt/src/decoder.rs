//! Local decoder: patches → bytes.
//!
//! Per BLT paper §3.3 (Pagnoni et al. 2024): a lightweight transformer
//! with `lD` layers (typically 9 for the 8B-class config). Each layer
//! is a decoder cross-attention block followed by a transformer block:
//! cross-attn reads byte queries and patch K/V (roles inverted from the
//! encoder), then the transformer block runs over byte representations.
//!
//! The final byte transformer layer feeds a byte LM head (`vocab = 256`).
//!
//! Owner: agent sasha (with [`crate::encoder`] and [`crate::model`]).
//!
//! Only available with `--features rocm`.

#![cfg(feature = "rocm")]
#![allow(clippy::too_many_arguments)]

use modgrad_compute::backend::{GpuVec, ResidencyError};
use modgrad_compute::neuron::{Linear, LinearResident, SimpleRng};
use modgrad_device::backend::{HipBatch, HipBuffer};
use modgrad_device::backend::ops::rms_norm_resident;
use modgrad_transformer::attention::{AttentionWeights, CausalSelfAttention};
use modgrad_transformer::block::TransformerBlock;
use modgrad_transformer::config::GptConfig;
use modgrad_transformer::dims::*;
use modgrad_transformer::kv_cache_resident::KvCacheResident;
use modgrad_transformer::mlp::{Mlp, MlpWeights, SwigluMlp, SwigluWeights};
use modgrad_transformer::residual::ResidualLambdas;
use modgrad_transformer::resident::{
    AttentionScratch, SwigluScratch, TransformerBlockResident, TransformerBlockScratch,
};
use modgrad_transformer::rope::RotaryEmbedding;
use modgrad_transformer::tensor::Tensor2;

use crate::cross_attn::{CrossAttention, CrossAttnConfig, CrossAttnDirection, CrossAttnScratch};
use crate::encoder::LocalEncoderConfig;

// ─── Config ──────────────────────────────────────────────────

/// Hyperparameters for [`LocalDecoder`].
///
/// `byte_dim` / `patch_dim` etc. mirror [`LocalEncoderConfig`] so a BLT
/// model's encoder and decoder can share dims without indirection.
#[derive(Debug, Clone)]
pub struct LocalDecoderConfig {
    /// Number of byte-level decoder layers (`lD`). Paper §4.8 uses 9
    /// for the 8B config.
    pub n_layers: usize,
    pub byte_dim: usize,
    pub patch_dim: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub mlp_dim: usize,
    pub norm_eps: f32,
    pub rope_base: f32,
    pub max_seq_len: usize,
}

impl LocalDecoderConfig {
    /// Validate dimensional consistency.
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
                got: "decoder dim mismatch",
            });
        }
        Ok(())
    }

    /// Mirror of [`LocalEncoderConfig::to_gpt_config`] for the byte-side
    /// transformer block stack.
    pub(crate) fn to_gpt_config(&self, num_layers: usize) -> GptConfig {
        // Re-use the encoder's projection logic — same shape contract.
        let proxy = LocalEncoderConfig {
            n_layers: num_layers,
            byte_dim: self.byte_dim,
            patch_dim: self.patch_dim,
            n_heads: self.n_heads,
            head_dim: self.head_dim,
            mlp_dim: self.mlp_dim,
            norm_eps: self.norm_eps,
            rope_base: self.rope_base,
            max_seq_len: self.max_seq_len,
            // Decoder doesn't use n-grams; placeholder values keep the
            // helper general.
            ngram_min_n: 3,
            ngram_max_n: 8,
            ngram_vocab_per_n: 1,
        };
        proxy.to_gpt_config(num_layers)
    }
}

// ─── Decoder ─────────────────────────────────────────────────

/// Local byte-side decoder. See module docs.
pub struct LocalDecoder {
    /// `lD` byte-level transformer blocks.
    pub byte_layers: Vec<TransformerBlockResident>,
    /// One cross-attention bridge per byte layer (runs *before* the
    /// transformer block per paper §3.3.1).
    pub cross_attns: Vec<CrossAttention>,
    /// Byte-level KV cache, one slab per byte layer.
    pub byte_kv_cache: KvCacheResident,
    /// Byte-level RoPE.
    pub rope: RotaryEmbedding,
    /// Final RMSNorm weight applied before the LM head; `[byte_dim]`,
    /// initialised to ones (host RmsNorm has no learnable scale outside
    /// QK norm — matches the existing resident path's convention).
    pub final_norm_weight_dev: HipBuffer,
    /// Byte LM head: `[256 × byte_dim]` row-major. Vocab = 256.
    pub lm_head: LinearResident,
    /// Configuration carried alongside.
    pub config: LocalDecoderConfig,
}

impl LocalDecoder {
    /// Allocate device buffers, upload deterministic random weights,
    /// and build the byte transformer stack + cross-attention bridges.
    pub fn new(config: LocalDecoderConfig) -> Result<Self, ResidencyError> {
        config.validate()?;
        let byte_dim = config.byte_dim;
        let n_layers = config.n_layers.max(1);
        let mut rng = SimpleRng::new(
            ((byte_dim * config.patch_dim).max(1) * n_layers.max(1)) as u64 ^ 0xDEC0_DE42,
        );
        let scale = 0.05_f32;

        let gpt_config = config.to_gpt_config(n_layers);
        let lambdas = ResidualLambdas::from_config(&gpt_config.residual, gpt_config.num_layers);
        let kv_dim = gpt_config.num_kv_heads.get() * gpt_config.head_dim.get();

        let mut byte_layers = Vec::with_capacity(n_layers);
        for li in 0..n_layers {
            let attn_w = AttentionWeights {
                wq: Tensor2::new(randn(&mut rng, byte_dim * byte_dim, scale), byte_dim, byte_dim).unwrap(),
                wk: Tensor2::new(randn(&mut rng, kv_dim * byte_dim, scale), kv_dim, byte_dim).unwrap(),
                wv: Tensor2::new(randn(&mut rng, kv_dim * byte_dim, scale), kv_dim, byte_dim).unwrap(),
                wo: Tensor2::new(randn(&mut rng, byte_dim * byte_dim, scale), byte_dim, byte_dim).unwrap(),
            };
            let attn = CausalSelfAttention::new(attn_w, &gpt_config);

            let swiglu_w = SwigluWeights {
                gate: Tensor2::new(randn(&mut rng, config.mlp_dim * byte_dim, scale), config.mlp_dim, byte_dim).unwrap(),
                up: Tensor2::new(randn(&mut rng, config.mlp_dim * byte_dim, scale), config.mlp_dim, byte_dim).unwrap(),
                down: Tensor2::new(randn(&mut rng, byte_dim * config.mlp_dim, scale), byte_dim, config.mlp_dim).unwrap(),
            };
            let swiglu = SwigluMlp::new(swiglu_w, gpt_config.model_dim, gpt_config.mlp_dim);

            let placeholder_mlp = Mlp::new(
                MlpWeights {
                    fc: Tensor2::zeros(config.mlp_dim, byte_dim),
                    proj: Tensor2::zeros(byte_dim, config.mlp_dim),
                },
                gpt_config.model_dim, gpt_config.mlp_dim,
            );
            let layer_idx = LayerIdx::new(li, gpt_config.num_layers).unwrap();
            let block = TransformerBlock::new(attn, placeholder_mlp, None, layer_idx, &gpt_config);

            let resident = TransformerBlockResident::from_block(&block, &swiglu, &lambdas, &gpt_config)?;
            byte_layers.push(resident);
        }

        let cross_cfg = CrossAttnConfig {
            byte_dim,
            patch_dim: config.patch_dim,
            num_heads: config.n_heads.max(1),
            head_dim: (config.patch_dim / config.n_heads.max(1)).max(1),
            norm_eps: config.norm_eps,
            direction: CrossAttnDirection::Decoder,
        };
        let mut cross_attns = Vec::with_capacity(n_layers);
        for li in 0..n_layers {
            let seed = 0xDEC0_DE42_u64 ^ ((li as u64) << 16) ^ (byte_dim as u64);
            cross_attns.push(CrossAttention::new(&cross_cfg, seed)?);
        }

        let byte_kv_cache = KvCacheResident::new(
            n_layers,
            gpt_config.num_kv_heads.get(),
            gpt_config.head_dim.get(),
            gpt_config.max_seq_len.get(),
            byte_dim,
        )?;
        let rope = RotaryEmbedding::new(
            gpt_config.head_dim, gpt_config.max_seq_len, gpt_config.rope_base,
        );

        // Final norm weight: ones to match the resident path's
        // unscaled-RMSNorm convention.
        let final_norm = vec![1.0f32; byte_dim];
        let final_norm_weight_dev = HipBuffer::new(final_norm.len() * 4)?;
        final_norm_weight_dev.copy_from_host(&final_norm)?;

        // Byte LM head: vocab = 256, in_dim = byte_dim. Bias is zero
        // (transformer LM heads are bias-free in our convention).
        let lm_head_w = randn(&mut rng, 256 * byte_dim, scale);
        let lm_head_lin = Linear {
            weight: lm_head_w,
            bias: vec![0.0f32; 256],
            in_dim: byte_dim,
            out_dim: 256,
        };
        let lm_head = LinearResident::from_linear(&lm_head_lin)?;

        Ok(Self {
            byte_layers,
            cross_attns,
            byte_kv_cache,
            rope,
            final_norm_weight_dev,
            lm_head,
            config,
        })
    }

    /// Number of byte-level layers (`lD`).
    #[inline]
    pub fn n_layers(&self) -> usize { self.byte_layers.len() }

    /// Forward `(patch_reps, boundaries) → byte_logits`.
    ///
    /// `patch_reps`: `[n_patches × patch_dim]`. Provided by the latent
    /// transformer (handled at the [`crate::model::BltModel`] level).
    /// `boundaries`: `[0, b_1, …, N]` patch boundaries.
    /// `seed_byte_reps`: optional `[N × byte_dim]` initial byte
    /// representations (typically the encoder's last-layer byte_reps);
    /// if `None`, decoder starts from the cross-attn bridge alone.
    /// `byte_logits_out`: `[N × 256]` row-major.
    pub fn forward(
        &mut self,
        batch: &HipBatch,
        patch_reps: &GpuVec,
        boundaries: &[usize],
        seed_byte_reps: Option<&GpuVec>,
        scratch: &mut LocalDecoderScratch,
        byte_logits_out: &mut GpuVec,
    ) -> Result<(), ResidencyError> {
        let n_bytes = boundaries[boundaries.len() - 1];
        let n_patches = boundaries.len().saturating_sub(1);
        let byte_dim = self.config.byte_dim;
        let patch_dim = self.config.patch_dim;

        debug_assert_eq!(scratch.byte_reps.len(), self.config.max_seq_len * byte_dim);
        debug_assert_eq!(byte_logits_out.len(), n_bytes * 256);
        debug_assert_eq!(patch_reps.len(), n_patches * patch_dim);
        debug_assert!(n_bytes <= self.config.max_seq_len);
        debug_assert!(n_patches > 0);
        debug_assert_eq!(boundaries[0], 0);

        // ── Stage 1: seed byte_reps ──
        //
        // The standard BLT decoder seeds byte queries from the encoder's
        // last-layer byte_reps (paper Figure 5 wires them through). If
        // the caller didn't supply a seed, we zero the workspace and let
        // the per-layer cross-attn fill it (real cross-attn output adds
        // a residual onto the seed; with the zero seed the residual is
        // the cross-attn output alone).
        match seed_byte_reps {
            Some(seed) => {
                debug_assert!(seed.len() >= n_bytes * byte_dim);
                copy_byte_reps_d2d(seed, &mut scratch.byte_reps, n_bytes * byte_dim)?;
            }
            None => {
                let host = vec![0.0f32; n_bytes * byte_dim];
                let dst = match &mut scratch.byte_reps {
                    GpuVec::Hip(b) => b,
                    other => return Err(ResidencyError::WrongVariant {
                        expected: "Hip", got: other.variant_name(),
                    }),
                };
                dst.copy_from_host(&host)?;
            }
        }

        let mut hidden_dev = GpuVec::try_hip(byte_dim)?;
        let x0_dev = GpuVec::try_hip(byte_dim)?;

        // ── Stage 2: layered (cross-attn → block) per byte_layer ──
        //
        // Per paper §3.3.1: each decoder layer applies the decoder
        // cross-attn first, then the transformer block. Both consume +
        // produce byte_reps; the cross-attn injects patch context via
        // residual update.
        for (block, cross) in self.byte_layers.iter().zip(self.cross_attns.iter()) {
            // Cross-attn first (paper §3.3.1).
            cross.forward_decoder(
                batch,
                &scratch.byte_reps,
                patch_reps,
                boundaries,
                &mut scratch.cross_scratch,
                &mut scratch.cross_out,
            )?;
            // The cross-attn output is the residual update; for now we
            // overwrite byte_reps with it (real impl from noah will add
            // residually inside `forward_decoder`).
            copy_byte_reps_d2d(&scratch.cross_out, &mut scratch.byte_reps, n_bytes * byte_dim)?;

            // Per-byte transformer block forward.
            for t in 0..n_bytes {
                copy_slab_to_hidden(
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

                copy_hidden_to_slab(
                    &hidden_dev, t, byte_dim, &mut scratch.byte_reps,
                )?;
                batch.note_dispatch()?;
            }
        }

        // ── Stage 3: per-byte final norm + LM head ──
        //
        // For each byte position, run the final RMSNorm (resident) and
        // dispatch a `[256 × byte_dim]` matvec into the right output
        // slab. We could fuse this into a `[256 × N]` matmul once we
        // have a multi-token resident path; the per-token matvec is
        // strictly correct and matches `GptModelResident::forward`'s
        // pattern.
        for t in 0..n_bytes {
            copy_slab_to_hidden(
                &scratch.byte_reps, t, byte_dim, &mut hidden_dev,
            )?;
            batch.note_dispatch()?;

            // Resident RMSNorm into the dedicated normed buffer.
            unsafe {
                rms_norm_resident(
                    hip_buf(&hidden_dev)?.device_ptr() as *const f32,
                    self.final_norm_weight_dev.device_ptr() as *const f32,
                    hip_buf(&scratch.normed)?.device_ptr() as *mut f32,
                    1, byte_dim, self.config.norm_eps,
                )?;
            }
            batch.note_dispatch()?;

            // LM head: logits[t] = lm_head.W · normed.
            let logits_buf = hip_buf_mut(byte_logits_out)?;
            let logits_base = logits_buf.device_ptr() as *mut f32;
            unsafe {
                modgrad_device::backend::ops::matvec_resident(
                    hip_buf(&scratch.normed)?.device_ptr() as *const f32,
                    self.lm_head.weight_dev.device_ptr() as *const f32,
                    self.lm_head.bias_dev.device_ptr() as *const f32,
                    logits_base.add(t * 256),
                    256,
                    byte_dim,
                )?;
            }
            batch.note_dispatch()?;
        }

        Ok(())
    }
}

// ─── Scratch ─────────────────────────────────────────────────

/// Caller-owned scratch buffers for [`LocalDecoder::forward`].
pub struct LocalDecoderScratch {
    /// `[max_seq_len × byte_dim]` rolling byte representations.
    pub byte_reps: GpuVec,
    /// Cross-attn output staging buffer; same shape as `byte_reps`.
    pub cross_out: GpuVec,
    /// `[byte_dim]` RMSNorm output workspace.
    pub normed: GpuVec,
    pub block_scratch: TransformerBlockScratch,
    pub attn_scratch: AttentionScratch,
    pub mlp_scratch: SwigluScratch,
    pub cross_scratch: CrossAttnScratch,
}

impl LocalDecoderScratch {
    pub fn new(config: &LocalDecoderConfig) -> Result<Self, ResidencyError> {
        let byte_dim = config.byte_dim;
        let n_heads = config.n_heads.max(1);
        let head_dim = config.head_dim.max(1);
        let kv_dim = n_heads * head_dim;
        let mlp_dim = config.mlp_dim;
        let max_seq = config.max_seq_len;

        Ok(Self {
            byte_reps: GpuVec::try_hip(max_seq * byte_dim)?,
            cross_out: GpuVec::try_hip(max_seq * byte_dim)?,
            normed: GpuVec::try_hip(byte_dim)?,
            block_scratch: TransformerBlockScratch::with_dims(
                byte_dim, kv_dim, mlp_dim, n_heads * max_seq,
            )?,
            attn_scratch: AttentionScratch::new(n_heads, head_dim, kv_dim, max_seq)?,
            mlp_scratch: SwigluScratch::new(byte_dim, mlp_dim)?,
            cross_scratch: CrossAttnScratch::new(
                &CrossAttnConfig {
                    byte_dim,
                    patch_dim: config.patch_dim,
                    num_heads: config.n_heads.max(1),
                    head_dim: (config.patch_dim / config.n_heads.max(1)).max(1),
                    norm_eps: config.norm_eps,
                    direction: CrossAttnDirection::Decoder,
                },
                /* max_kv_len = */ max_seq,
                /* max_n_bytes = */ max_seq,
                /* max_n_patches = */ max_seq,
            )?,
        })
    }
}

// ─── Helpers ─────────────────────────────────────────────────

fn randn(rng: &mut SimpleRng, n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|_| rng.next_normal() * scale).collect()
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
fn hip_buf_mut<'a>(g: &'a mut GpuVec) -> Result<&'a mut modgrad_device::backend::HipBuffer, ResidencyError> {
    match g {
        GpuVec::Hip(b) => Ok(b),
        other => Err(ResidencyError::WrongVariant {
            expected: "Hip", got: other.variant_name(),
        }),
    }
}

/// D2D copy `n` floats from `src` into `dst`. Both must be `Hip`.
fn copy_byte_reps_d2d(src: &GpuVec, dst: &mut GpuVec, n: usize)
    -> Result<(), ResidencyError>
{
    use modgrad_device::backend::rocm::ffi;
    let s = match src {
        GpuVec::Hip(b) => b,
        other => return Err(ResidencyError::WrongVariant {
            expected: "Hip", got: other.variant_name(),
        }),
    };
    let d = match dst {
        GpuVec::Hip(b) => b,
        other => return Err(ResidencyError::WrongVariant {
            expected: "Hip", got: other.variant_name(),
        }),
    };
    const HIP_D2D: std::os::raw::c_int = 3;
    let err = unsafe {
        ffi::hipMemcpy(
            d.device_ptr(),
            s.device_ptr() as *const std::os::raw::c_void,
            n * 4,
            HIP_D2D,
        )
    };
    if err != 0 {
        return Err(ResidencyError::Backend(
            modgrad_device::backend::BackendError::Runtime(format!(
                "byte_reps D2D ({} f32): {}", n, ffi::hip_err_str(err),
            )),
        ));
    }
    Ok(())
}

/// D2D copy `slab[t * byte_dim..(t+1) * byte_dim]` → `hidden[..byte_dim]`.
fn copy_slab_to_hidden(
    slab: &GpuVec,
    t: usize,
    byte_dim: usize,
    hidden: &mut GpuVec,
) -> Result<(), ResidencyError> {
    use modgrad_device::backend::rocm::ffi;
    let src = hip_buf(slab)?;
    let dst = hip_buf_mut(hidden)?;
    let off_bytes = t * byte_dim * 4;
    const HIP_D2D: std::os::raw::c_int = 3;
    let err = unsafe {
        ffi::hipMemcpy(
            dst.device_ptr(),
            (src.device_ptr() as *const u8).add(off_bytes) as *const std::os::raw::c_void,
            byte_dim * 4,
            HIP_D2D,
        )
    };
    if err != 0 {
        return Err(ResidencyError::Backend(
            modgrad_device::backend::BackendError::Runtime(format!(
                "slab → hidden D2D: {}", ffi::hip_err_str(err),
            )),
        ));
    }
    Ok(())
}

/// D2D copy `hidden[..byte_dim]` → `slab[t * byte_dim..]`.
fn copy_hidden_to_slab(
    hidden: &GpuVec,
    t: usize,
    byte_dim: usize,
    slab: &mut GpuVec,
) -> Result<(), ResidencyError> {
    use modgrad_device::backend::rocm::ffi;
    let src = hip_buf(hidden)?;
    let dst = hip_buf_mut(slab)?;
    let off_bytes = t * byte_dim * 4;
    const HIP_D2D: std::os::raw::c_int = 3;
    let err = unsafe {
        ffi::hipMemcpy(
            (dst.device_ptr() as *mut u8).add(off_bytes) as *mut std::os::raw::c_void,
            src.device_ptr() as *const std::os::raw::c_void,
            byte_dim * 4,
            HIP_D2D,
        )
    };
    if err != 0 {
        return Err(ResidencyError::Backend(
            modgrad_device::backend::BackendError::Runtime(format!(
                "hidden → slab D2D: {}", ffi::hip_err_str(err),
            )),
        ));
    }
    Ok(())
}
