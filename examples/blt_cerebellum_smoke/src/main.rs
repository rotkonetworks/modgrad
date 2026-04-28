//! Proof-of-life: a tiny `BltModel` wrapped as `BltCerebellum` produces
//! a per-layer hidden-state cache for the latent transformer, and
//! `CerebProjection::with_layers` reads that cache through
//! `cerebellum_at_position` to produce finite cortex activations.
//!
//! ## Patch-vs-byte semantic — the load-bearing difference from `qwen_cerebellum_smoke`
//!
//! `QwenCerebellum` returns one cache row per **token** (`n_positions == N_TOKENS`).
//! `BltCerebellum` returns one cache row per **patch** (`n_positions == N_PATCHES`,
//! and `N_PATCHES < N_BYTES`). The latent transformer is the world model in
//! BLT — it only sees patches — so a cortex consumer wired to a BLT cerebellum
//! must attend at patch granularity, not byte granularity. A connection that
//! assumes "one cerebellum slot per input timestep" will silently mis-align.
//!
//! HIP is required at runtime — the binary exits cleanly with status 0 on
//! hosts without a HIP device, same as `qwen_cerebellum_smoke` and
//! `multimodal_smoke`.
//!
//! Run:
//!   cargo run --release -p blt_cerebellum_smoke

use modgrad_blt::cerebellum::BltCerebellum;
use modgrad_blt::decoder::LocalDecoderConfig;
use modgrad_blt::encoder::LocalEncoderConfig;
use modgrad_blt::model::{BltConfig, BltLatentConfig, BltModel};
use modgrad_ctm::cerebellum::{cerebellum_at_position, CerebProjection};
use modgrad_device::backend::rocm::ffi::runtime_available;
use modgrad_traits::cerebellum::FrozenCerebellum;

// ── Encoder dims (byte-level) ───────────────────────────────────
const BYTE_DIM: usize = 32;
const N_BYTE_HEADS: usize = 4;
const BYTE_HEAD_DIM: usize = BYTE_DIM / N_BYTE_HEADS; // 8
const ENCODER_MLP_DIM: usize = BYTE_DIM * 2;          // 64
const N_LAYERS_ENCODER: usize = 1;

// ── Latent dims (patch-level — what the cerebellum exposes) ─────
const PATCH_DIM: usize = 64;
const N_PATCH_HEADS: usize = 4;
const PATCH_HEAD_DIM: usize = PATCH_DIM / N_PATCH_HEADS; // 16
const LATENT_MLP_DIM: usize = PATCH_DIM * 2;             // 128
const N_LAYERS_LATENT: usize = 2;
const MAX_PATCHES: usize = 8;

// ── Decoder dims (byte-level) ───────────────────────────────────
const N_LAYERS_DECODER: usize = 1;
const DECODER_MLP_DIM: usize = BYTE_DIM * 2; // 64

// ── Sequence dims ───────────────────────────────────────────────
const MAX_SEQ_LEN: usize = 32;
const N_BYTES: usize = 32;
const N_PATCHES_EXPECTED: usize = 8; // 32 bytes / stride 4 with default patcher

// ── N-gram embedding dims (encoder) ─────────────────────────────
const NGRAM_MIN_N: usize = 3;
const NGRAM_MAX_N: usize = 5;
const NGRAM_VOCAB_PER_N: usize = 256;

// ── RoPE / norm ─────────────────────────────────────────────────
const ROPE_BASE: f32 = 10_000.0;
const NORM_EPS: f32 = 1e-5;

// ── Cortex projection dims ──────────────────────────────────────
const CORTEX_DIM: usize = 32;

fn main() {
    if !runtime_available() {
        println!("blt_cerebellum_smoke: HIP runtime unavailable, skipping");
        std::process::exit(0);
    }

    // ── Build a tiny BltModel (mirrors BltModel::tests::tiny_config) ──
    let config = BltConfig {
        encoder: LocalEncoderConfig {
            n_layers: N_LAYERS_ENCODER,
            byte_dim: BYTE_DIM,
            patch_dim: PATCH_DIM,
            n_heads: N_BYTE_HEADS,
            head_dim: BYTE_HEAD_DIM,
            mlp_dim: ENCODER_MLP_DIM,
            norm_eps: NORM_EPS,
            rope_base: ROPE_BASE,
            max_seq_len: MAX_SEQ_LEN,
            ngram_min_n: NGRAM_MIN_N,
            ngram_max_n: NGRAM_MAX_N,
            ngram_vocab_per_n: NGRAM_VOCAB_PER_N,
        },
        latent: BltLatentConfig {
            n_layers: N_LAYERS_LATENT,
            patch_dim: PATCH_DIM,
            n_heads: N_PATCH_HEADS,
            head_dim: PATCH_HEAD_DIM,
            mlp_dim: LATENT_MLP_DIM,
            norm_eps: NORM_EPS,
            rope_base: ROPE_BASE,
            max_patches: MAX_PATCHES,
        },
        decoder: LocalDecoderConfig {
            n_layers: N_LAYERS_DECODER,
            byte_dim: BYTE_DIM,
            patch_dim: PATCH_DIM,
            n_heads: N_BYTE_HEADS,
            head_dim: BYTE_HEAD_DIM,
            mlp_dim: DECODER_MLP_DIM,
            norm_eps: NORM_EPS,
            rope_base: ROPE_BASE,
            max_seq_len: MAX_SEQ_LEN,
        },
    };

    let model = BltModel::new(config).expect("BltModel::new");
    let mut cereb = BltCerebellum::from_model(model)
        .expect("BltCerebellum::from_model");

    // ── Trait surface — patch-aligned, encoder/decoder NOT counted ──
    assert_eq!(
        cereb.hidden_dim(), PATCH_DIM,
        "BltCerebellum.hidden_dim: expected {} (patch_dim), got {}",
        PATCH_DIM, cereb.hidden_dim(),
    );
    assert_eq!(
        cereb.n_layers(), N_LAYERS_LATENT,
        "BltCerebellum.n_layers: expected {} (latent only — NOT encoder+latent+decoder), got {}",
        N_LAYERS_LATENT, cereb.n_layers(),
    );

    // ── Encode a 32-byte sequence ───────────────────────────────
    let tokens: Vec<i64> = (0..N_BYTES as i64).map(|i| i % 256).collect();
    assert_eq!(
        tokens.len(), N_BYTES,
        "tokens.len(): expected {}, got {}", N_BYTES, tokens.len(),
    );

    let cache = cereb.encode_context_layers(&tokens);

    // ── Cache shape — patch-aligned (load-bearing) ──────────────
    assert_eq!(
        cache.n_layers, N_LAYERS_LATENT,
        "cache.n_layers: expected {} (latent layers only), got {}",
        N_LAYERS_LATENT, cache.n_layers,
    );
    assert_eq!(
        cache.n_positions, N_PATCHES_EXPECTED,
        "cache.n_positions: expected {} (= n_patches, NOT n_bytes={}); BLT cerebellum is patch-aligned",
        N_PATCHES_EXPECTED, N_BYTES,
    );
    assert_eq!(
        cache.hidden_dim, PATCH_DIM,
        "cache.hidden_dim: expected {} (patch_dim), got {}",
        PATCH_DIM, cache.hidden_dim,
    );
    let expected_len = N_LAYERS_LATENT * N_PATCHES_EXPECTED * PATCH_DIM;
    assert_eq!(
        cache.hidden_states.len(), expected_len,
        "cache.hidden_states.len(): expected {} (= {} * {} * {}), got {}",
        expected_len, N_LAYERS_LATENT, N_PATCHES_EXPECTED, PATCH_DIM,
        cache.hidden_states.len(),
    );

    // ── Modality tracking — BltCerebellum doesn't track modalities ──
    assert!(
        cache.modalities.is_none(),
        "cache.modalities: expected None (BltCerebellum doesn't tag modalities — that's the caller's job), got Some(_)",
    );

    // ── Finite check ────────────────────────────────────────────
    for (i, &v) in cache.hidden_states.iter().enumerate() {
        assert!(
            v.is_finite(),
            "non-finite cache entry at flat offset {}: expected finite, got {}",
            i, v,
        );
    }

    // ── Per-layer non-trivial activation at position 0 ──────────
    for li in 0..N_LAYERS_LATENT {
        let h = cache.at(li, 0);
        assert_eq!(
            h.len(), PATCH_DIM,
            "cache.at({}, 0).len(): expected {}, got {}",
            li, PATCH_DIM, h.len(),
        );
        let sq: f32 = h.iter().map(|x| x * x).sum();
        assert!(
            sq > 0.0,
            "layer {} pos 0 collapsed to all-zeros (||h||² expected > 0, got {}); encoder/latent dispatch suspect",
            li, sq,
        );
    }

    // ── Cache → cortex projection ───────────────────────────────
    let proj = CerebProjection::with_layers(
        CORTEX_DIM,
        /* frozen_input_dim  */ PATCH_DIM,
        /* frozen_output_dim */ PATCH_DIM,
        N_LAYERS_LATENT,
    );
    assert_eq!(
        proj.layer_weight_logits.len(), N_LAYERS_LATENT,
        "CerebProjection.layer_weight_logits.len(): expected {}, got {}",
        N_LAYERS_LATENT, proj.layer_weight_logits.len(),
    );

    let mut out = vec![0.0f32; CORTEX_DIM];
    cerebellum_at_position(&cache, &proj, 0, &mut out);
    assert_eq!(
        out.len(), CORTEX_DIM,
        "cortex projection length: expected {}, got {}",
        CORTEX_DIM, out.len(),
    );
    for (i, &v) in out.iter().enumerate() {
        assert!(
            v.is_finite(),
            "non-finite cortex activation at idx {}: expected finite, got {}",
            i, v,
        );
    }
    let abs_max = out.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    assert!(
        abs_max > 0.0,
        "cortex projection collapsed to all-zeros (max |out| expected > 0, got 0); proj_out matvec suspect",
    );

    println!(
        "PASS: BltCerebellum encodes patch-aligned cache, projection produces finite cortex activations"
    );
}
