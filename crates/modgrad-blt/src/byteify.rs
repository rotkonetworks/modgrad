//! Byte-ification recipe — initialise a `BltModel`'s Latent Transformer
//! from a pretrained `GptModelResident` (Qwen2.5 / Llama).
//!
//! Per BLT paper §6.2: copy the global transformer weights from the
//! pretrained checkpoint, train the local encoder + decoder + cross-
//! attention from scratch, with 1/10 the LR applied to the global
//! parameters.
//!
//! For us, the pretrained checkpoint comes from
//! `modgrad_io::qwen2::load_qwen2_5_0_5b` — already loaded into a
//! `GptModelResident`. The byte-ification routine returns a model
//! handle whose Latent is *that exact resident*, plus stores the local
//! configs that sasha's slice will consume to assemble the encoder /
//! decoder / cross-attention.
//!
//! ## Today's seam vs once sasha's slice lands
//!
//! `sasha`'s `BltModel` (`crate::model::BltModel`) is a placeholder
//! today; the encoder / decoder / cross-attn are likewise stubs. So
//! [`ByteifyRecipe::from_qwen2`] can't produce a real `BltModel` — it
//! returns a [`ByteifiedLatent`] that bundles the latent with the
//! configs needed to build the local stack later. Once sasha lands
//! `BltModel { encoder, latent, decoder, cross_attn }`, the recipe
//! flips its return type to `BltModel` and the rest of the API surface
//! is unchanged.
//!
//! The `BltTrainer` already accepts a `GptModelResident` directly (the
//! latent), so the smoke test can run today against
//! `ByteifiedLatent::latent` without waiting for sasha.

#![allow(dead_code)]

#[cfg(feature = "rocm")]
use modgrad_compute::backend::ResidencyError;
#[cfg(feature = "rocm")]
use modgrad_transformer::GptModelResident;

// ─── Local component configs ────────────────────────────────
//
// These are placeholders that sasha's slice replaces with real
// `LocalEncoderConfig` / `LocalDecoderConfig` / `CrossAttnConfig`. We
// define them here so the byteify API has stable types to consume —
// when sasha lands, the field shapes + names migrate verbatim and the
// `ByteifyConfig` shape stays.

/// Local encoder config (placeholder — sasha owns the real one).
///
/// Per BLT §3.2 / §4.8 (Llama 3 8B byte-ification regime):
/// `n_layers = 1`, `model_dim = 1024` (Llama 3 8B small variant).
/// For Qwen2.5-0.5B (model_dim 896) the natural pairing is
/// `byte_dim = 256` (compressed) or `byte_dim = 896` (matched to the
/// latent's residual stream — simpler, no projection needed).
#[derive(Debug, Clone)]
pub struct LocalEncoderConfig {
    /// Number of encoder transformer layers (`l_E` in the paper).
    /// Default `1` per §4.8 Table 13 (8B-class).
    pub n_layers: usize,
    /// Byte-stream embedding dim. Default 256.
    pub byte_dim: usize,
    /// Number of attention heads. Default 8.
    pub n_heads: usize,
    /// Head dimension. Default 32.
    pub head_dim: usize,
    /// MLP hidden dim. Default `4 * byte_dim`.
    pub mlp_dim: usize,
    /// Hash n-gram embedding params (paper §3.2.1). The number of
    /// n-gram banks; each bank embeds a different n.
    pub n_hash_banks: usize,
    /// Per-bank hash table size. Default 50_000.
    pub hash_table_size: usize,
}

impl Default for LocalEncoderConfig {
    fn default() -> Self {
        Self {
            n_layers: 1,
            byte_dim: 256,
            n_heads: 8,
            head_dim: 32,
            mlp_dim: 1024,
            n_hash_banks: 4,
            hash_table_size: 50_000,
        }
    }
}

/// Local decoder config (placeholder — sasha owns the real one).
///
/// Per BLT §3.3 / §4.8: deeper than the encoder
/// (`l_D = 9` in §4.8 Table 13), each layer is cross-attn → transformer.
/// Final LM head is `[byte_dim → 256]`.
#[derive(Debug, Clone)]
pub struct LocalDecoderConfig {
    /// Number of decoder transformer layers (`l_D`). Default `9`
    /// (paper §4.8 Table 13, 8B-class).
    pub n_layers: usize,
    /// Byte-stream embedding dim. Should match
    /// `LocalEncoderConfig::byte_dim` so the cross-attn key/value space
    /// is consistent.
    pub byte_dim: usize,
    /// Number of attention heads. Default 8.
    pub n_heads: usize,
    /// Head dimension. Default 32.
    pub head_dim: usize,
    /// MLP hidden dim. Default `4 * byte_dim`.
    pub mlp_dim: usize,
}

impl Default for LocalDecoderConfig {
    fn default() -> Self {
        Self {
            n_layers: 9,
            byte_dim: 256,
            n_heads: 8,
            head_dim: 32,
            mlp_dim: 1024,
        }
    }
}

/// Cross-attention config (placeholder — noah owns the real one).
///
/// Per BLT §3.2.2 / §3.3.1: pre-layer-norm on Q/K/V, no positional
/// embeddings, multi-headed.
#[derive(Debug, Clone)]
pub struct CrossAttnConfig {
    /// Number of cross-attn heads. Default 8.
    pub n_heads: usize,
    /// Per-head dim. Default 32.
    pub head_dim: usize,
}

impl Default for CrossAttnConfig {
    fn default() -> Self {
        Self {
            n_heads: 8,
            head_dim: 32,
        }
    }
}

// ─── Byteify config + recipe ────────────────────────────────

/// Top-level byteify config — bundles the local-component shapes the
/// recipe initialises from scratch. The latent (Qwen2.5 / Llama) is
/// passed by value to [`ByteifyRecipe::from_qwen2`] — its config is
/// already implicit in the residency layout.
#[derive(Debug, Clone, Default)]
pub struct ByteifyConfig {
    pub local_encoder_config: LocalEncoderConfig,
    pub local_decoder_config: LocalDecoderConfig,
    pub cross_attn_config: CrossAttnConfig,
}

/// Output of [`ByteifyRecipe::from_qwen2`] — bundles the consumed
/// latent with the local-component configs that sasha's slice will use
/// to assemble the encoder / decoder / cross-attn.
///
/// **Today** (sasha placeholder): `BltTrainer::new(byteified.latent,
/// …)` is the path; the local configs ride along for documentation
/// and future use.
///
/// **Once sasha lands** `BltModel`: this struct's contents migrate into
/// `BltModel { encoder, latent, decoder, cross_attn }` and the trainer's
/// `M` flips to `BltModel`.
#[cfg(feature = "rocm")]
pub struct ByteifiedLatent {
    /// The pretrained latent transformer (`GptModelResident`,
    /// canonically Qwen2.5-0.5B). Taken by move from the caller; the
    /// recipe does not clone — there is one resident copy and the
    /// trainer owns it from here on.
    pub latent: GptModelResident,
    /// Local encoder config — used by `BltModel::from_byteified` once
    /// sasha's slice lands.
    pub local_encoder_config: LocalEncoderConfig,
    /// Local decoder config — same.
    pub local_decoder_config: LocalDecoderConfig,
    /// Cross-attention config — same.
    pub cross_attn_config: CrossAttnConfig,
}

/// Byteify recipe per BLT paper §6.2.
///
/// The recipe itself is stateless — it's a namespace for the two
/// associated functions: `from_qwen2` (the construction) and
/// `global_predicate` (the LR-routing closure used by `BltTrainer`).
pub struct ByteifyRecipe;

#[cfg(feature = "rocm")]
impl ByteifyRecipe {
    /// Byte-ify a pretrained Qwen2.5-0.5B (or any `GptModelResident`).
    ///
    /// Per paper §6.2:
    ///   - Latent ← the supplied pretrained resident (consumed by move)
    ///   - LocalEncoder, LocalDecoder, CrossAttention ← *to be built
    ///     fresh* by sasha's slice from `config.local_*` /
    ///     `config.cross_attn_config`.
    ///
    /// Today this returns a [`ByteifiedLatent`] that the caller plugs
    /// into [`crate::trainer::BltTrainer::new`] alongside the LR-routing
    /// closure from [`Self::global_predicate`]. Once sasha lands
    /// `BltModel`, the return type flips and `BltTrainer::new` accepts
    /// the `BltModel` directly — the caller-side code stays the same
    /// modulo that one type swap.
    ///
    /// Returns `Err(ResidencyError)` if the latent is in an unsupported
    /// residency state — currently always succeeds because
    /// `GptModelResident::from_model` is the only constructor and it
    /// guarantees full residency. Kept fallible for forward-compat.
    pub fn from_qwen2(
        latent: GptModelResident,
        config: ByteifyConfig,
    ) -> Result<ByteifiedLatent, ResidencyError> {
        // Sanity: encoder and decoder byte_dim should match — cross-attn
        // expects a single byte-stream width on both sides.
        let enc_byte_dim = config.local_encoder_config.byte_dim;
        let dec_byte_dim = config.local_decoder_config.byte_dim;
        if enc_byte_dim != dec_byte_dim {
            eprintln!(
                "byteify: WARN — local_encoder.byte_dim ({enc_byte_dim}) != \
                 local_decoder.byte_dim ({dec_byte_dim}); cross-attn shape \
                 may not align."
            );
        }
        eprintln!(
            "byteify: latent d_model={}, n_layers={}, vocab={}",
            latent.model_dim(), latent.num_layers(), latent.vocab_size(),
        );

        Ok(ByteifiedLatent {
            latent,
            local_encoder_config: config.local_encoder_config,
            local_decoder_config: config.local_decoder_config,
            cross_attn_config: config.cross_attn_config,
        })
    }

    /// Returns the canonical "is this parameter a global (latent)
    /// parameter" predicate, suitable for [`crate::trainer::BltTrainer::new`].
    ///
    /// **Today's schema.** The trainer's `param_keys` produces
    /// `embed` / `lm_head` / `block.{i}.{slot}` for every latent
    /// parameter. With the local stack still on placeholders, *every*
    /// param the trainer touches is a latent param ⇒ the predicate
    /// returns `true` for every key. The 1/10×-LR rule applies
    /// uniformly.
    ///
    /// **Once sasha lands** `BltModel`, the local stack contributes new
    /// keys (e.g. `encoder.block.{i}.wq`, `decoder.block.{i}.wq`,
    /// `cross_attn.{i}.wq`). This predicate then returns `false` for
    /// any key starting with `encoder.` / `decoder.` / `cross_attn.`,
    /// and `true` for the latent's `embed` / `lm_head` /
    /// `block.{i}.{slot}` (no prefix). The trainer's per-group AdamW
    /// dispatch routes accordingly.
    ///
    /// The closure is `Send` so it can ride a `BltTrainer` across
    /// thread handoff (e.g. host-prep on one thread, GPU dispatch on
    /// another).
    pub fn global_predicate() -> Box<dyn Fn(&str) -> bool + Send> {
        Box::new(|name: &str| -> bool {
            // Local-stack prefixes — `false` ⇒ apply local_lr.
            if name.starts_with("encoder.")
                || name.starts_with("decoder.")
                || name.starts_with("cross_attn.")
            {
                return false;
            }
            // Everything else is the latent (Qwen / Llama). True ⇒
            // apply global_lr (= local_lr / 10 by paper §6.2).
            true
        })
    }
}

// ─── Tests ────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(feature = "rocm")]
mod tests {
    use super::*;

    #[test]
    fn global_predicate_today_all_latent() {
        let f = ByteifyRecipe::global_predicate();
        // Today's keys — all latent (return true ⇒ global_lr).
        assert!(f("embed"));
        assert!(f("lm_head"));
        assert!(f("block.0.wq"));
        assert!(f("block.23.down"));
    }

    #[test]
    fn global_predicate_local_stack_returns_false() {
        let f = ByteifyRecipe::global_predicate();
        // Once sasha's slice lands, these keys appear and must route
        // to `local_lr`.
        assert!(!f("encoder.block.0.wq"));
        assert!(!f("encoder.embed"));
        assert!(!f("decoder.block.5.wo"));
        assert!(!f("decoder.lm_head"));
        assert!(!f("cross_attn.0.wq"));
        assert!(!f("cross_attn.8.wo"));
    }

    #[test]
    fn config_defaults_paper_8b() {
        let c = ByteifyConfig::default();
        assert_eq!(c.local_encoder_config.n_layers, 1,
            "paper §4.8 Table 13: l_E = 1");
        assert_eq!(c.local_decoder_config.n_layers, 9,
            "paper §4.8 Table 13: l_D = 9 (8B)");
        // Encoder and decoder byte_dim should match — cross-attn
        // shape relies on it.
        assert_eq!(c.local_encoder_config.byte_dim, c.local_decoder_config.byte_dim);
    }
}
