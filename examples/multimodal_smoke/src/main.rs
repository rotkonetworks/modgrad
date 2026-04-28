//! Proof-of-life: text + image VQ codes flow through `UnifiedTokenizer`
//! → `QwenCerebellum` (per-layer hidden cache) → modality-aware cortex
//! projection via `cerebellum_modality_pool`. Two pooled cortex vectors
//! (one per modality, Byte vs ImageVq) come out measurably distinct.
//!
//! For this binary to fail, one of: the unified-token offset table,
//! the per-position modality tagging, the resident transformer forward,
//! the per-layer cache layout, or the modality-aware mean-pool +
//! `project_out` matvec dispatch would have to be broken. HIP is
//! required at runtime — the binary exits cleanly with status 0 on
//! hosts without a HIP device.
//!
//! The deterministic seed `0xC0DE_FACE` drives both the synthetic 8×8
//! image grid and the transformer weight init; the L2-distance
//! assertion is reproducible against that seed.
//!
//! Run:
//!   cargo run --release -p multimodal_smoke

use modgrad_compute::neuron::SimpleRng;
use modgrad_ctm::cerebellum::{
    cerebellum_modality_pool, CerebProjection, FrozenCerebellum,
    Modality as CerebModality,
};
use modgrad_ctm::qwen_cerebellum::QwenCerebellum;
use modgrad_data::{Delimiter, Modality as DataModality, UnifiedTokenizer};
use modgrad_device::backend::rocm::ffi::runtime_available;
use modgrad_transformer::attention::{AttentionWeights, CausalSelfAttention};
use modgrad_transformer::block::TransformerBlock;
use modgrad_transformer::config::{
    GptConfig, MlpActivation, Precision, ResidualConfig, SmearConfig,
    ValueEmbedConfig, WindowPattern,
};
use modgrad_transformer::dims::*;
use modgrad_transformer::mlp::{Mlp, MlpWeights, SwigluMlp, SwigluWeights};
use modgrad_transformer::model::GptModel;
use modgrad_transformer::norm::ScaledRmsNorm;
use modgrad_transformer::position::fixed::FixedPositioning;
use modgrad_transformer::residual::ResidualLambdas;
use modgrad_transformer::resident::GptModelResident;
use modgrad_transformer::rope::RotaryEmbedding;
use modgrad_transformer::smear::{Inference, Smear, SmearWeights, Training};
use modgrad_transformer::tensor::Tensor2;

// ── Deterministic seed for reproducibility ─────────────────────
const SEED: u64 = 0xC0DE_FACE;

// ── Tiny LLM dims ───────────────────────────────────────────────
// vocab=152000 (NOT 256!) because the unified token range parks
// image-VQ codes at offset 140008..144104 — the embed table must
// cover those high-offset slots. 152000 × 32 × 4B ≈ 19 MB, fine.
const VOCAB_SIZE: usize = 152_000;
const MODEL_DIM: usize = 32;
const NUM_HEADS: usize = 4;
const NUM_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 8;
const NUM_LAYERS: usize = 3;
const MLP_DIM: usize = 64;
const MAX_SEQ_LEN: usize = 32;
const ROPE_BASE: f32 = 10_000.0;
const NORM_EPS: f32 = 1e-5;

// ── Cortex projection dims ──────────────────────────────────────
const CORTEX_DIM: usize = 64;

// ── Sequence layout ─────────────────────────────────────────────
//   "describe: " (10 bytes) + <img> + 4 image VQ codes + </img>
const PROMPT: &[u8] = b"describe: ";
const N_IMAGE_CODES: usize = 4;
const SEQ_LEN: usize = PROMPT.len() + 1 + N_IMAGE_CODES + 1; // 16

// ── L2-distance threshold for the modality-discrimination
// assertion. Below this would mean the two pooled cortex vectors
// are essentially identical — a degenerate seam.
const L2_MIN: f32 = 1e-2;

/// Map `modgrad-data`'s `Modality` into `modgrad-ctm`'s `Modality`.
/// They are kept as parallel enums (per the cerebellum-side comment)
/// to avoid a cross-crate dependency; this trivial map is the seam.
fn map_modality(m: DataModality) -> CerebModality {
    match m {
        DataModality::Byte => CerebModality::Byte,
        DataModality::Delimiter => CerebModality::Delimiter,
        DataModality::ImageVq => CerebModality::ImageVq,
        DataModality::AudioVq => CerebModality::AudioVq,
        DataModality::Timestamp => CerebModality::Timestamp,
        DataModality::Action => CerebModality::Action,
        DataModality::Other => CerebModality::Other,
    }
}

/// Format a modality tag in 4 chars for compact diagnostic printing.
fn tag(m: CerebModality) -> &'static str {
    match m {
        CerebModality::Byte => "BYTE",
        CerebModality::Delimiter => "DELM",
        CerebModality::ImageVq => "IMVQ",
        CerebModality::AudioVq => "AUVQ",
        CerebModality::Timestamp => "TIME",
        CerebModality::Action => "ACTN",
        CerebModality::Other => "OTHR",
    }
}

/// Stand-in for a real VQ-VAE: split an 8×8 grid into 4 quadrants,
/// sum each quadrant, mod 4096 → 4 image VQ codes (u16).
/// The goal here is just to produce 4 distinct image-VQ token IDs;
/// a real VQ-VAE round-trip is a future slice.
fn fake_vq_encode(grid: &[f32; 64]) -> [u16; N_IMAGE_CODES] {
    let mut sums = [0.0f32; N_IMAGE_CODES];
    // 8×8 layout: (row, col) → grid[row*8 + col]
    // Quadrants: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
    for row in 0..8 {
        for col in 0..8 {
            let q = (if row < 4 { 0 } else { 2 }) + if col < 4 { 0 } else { 1 };
            sums[q] += grid[row * 8 + col];
        }
    }
    let mut codes = [0u16; N_IMAGE_CODES];
    for q in 0..N_IMAGE_CODES {
        // Scale to a wide integer so distinct quadrant sums likely
        // fall on distinct mod-4096 buckets. The seed makes this
        // reproducible.
        let scaled = (sums[q].abs() * 1_000_000.0) as u64;
        codes[q] = (scaled % 4096) as u16;
    }
    codes
}

/// Build the synthetic 8×8 grid from the deterministic seed. Values
/// are `f32` in roughly [-1, 1].
fn build_grid(rng: &mut SimpleRng) -> [f32; 64] {
    let mut grid = [0.0f32; 64];
    for v in grid.iter_mut() {
        *v = rng.next_normal();
    }
    grid
}

/// L2 distance between two equal-length vectors.
fn l2(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "l2 length mismatch: {} vs {}", a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s.sqrt()
}

/// Per-layer variance (across hidden_dim, then averaged) for the
/// positions in `positions`. Returns one scalar per layer.
fn variance_signature(
    cache: &modgrad_ctm::cerebellum::CerebellumCache,
    positions: &[usize],
) -> Vec<f32> {
    let mut out = Vec::with_capacity(cache.n_layers);
    for li in 0..cache.n_layers {
        // For each hidden-dim coordinate, compute the variance across
        // the requested positions; average those variances over the
        // hidden-dim axis to get one scalar per layer.
        let n = positions.len() as f32;
        if positions.is_empty() {
            out.push(0.0);
            continue;
        }
        let mut mean = vec![0.0f32; cache.hidden_dim];
        for &p in positions {
            let h = cache.at(li, p);
            for i in 0..cache.hidden_dim {
                mean[i] += h[i];
            }
        }
        for v in mean.iter_mut() {
            *v /= n;
        }
        let mut var_sum = 0.0f32;
        for &p in positions {
            let h = cache.at(li, p);
            for i in 0..cache.hidden_dim {
                let d = h[i] - mean[i];
                var_sum += d * d;
            }
        }
        out.push(var_sum / (n * cache.hidden_dim as f32));
    }
    out
}

fn main() {
    if !runtime_available() {
        println!("multimodal_smoke: HIP runtime unavailable, skipping");
        std::process::exit(0);
    }

    // ── 1. Deterministic synthetic 8×8 image ────────────────────
    let mut rng = SimpleRng::new(SEED);
    let grid = build_grid(&mut rng);

    // ── 2. "VQ-encode" via deterministic quadrant hashing ───────
    let image_codes = fake_vq_encode(&grid);
    println!(
        "multimodal_smoke: synthetic image VQ codes (stand-in for real VQ-VAE) = {:?}",
        image_codes,
    );
    // Sanity: codes are in valid VQ range.
    for &c in &image_codes {
        assert!((c as i64) < 4096, "VQ code {} >= codebook size 4096", c);
    }

    // ── 3. Build unified token sequence ────────────────────────
    let tokenizer = UnifiedTokenizer::for_qwen2_5();
    let mut tokens: Vec<i64> = Vec::with_capacity(SEQ_LEN);
    for &b in PROMPT {
        tokens.push(tokenizer.encode_byte(b));
    }
    tokens.push(tokenizer.encode_delimiter(Delimiter::ImgOpen));
    for &c in &image_codes {
        tokens.push(tokenizer.encode_image_vq(c));
    }
    tokens.push(tokenizer.encode_delimiter(Delimiter::ImgClose));
    assert_eq!(
        tokens.len(),
        SEQ_LEN,
        "unified token sequence length: expected {}, got {}",
        SEQ_LEN,
        tokens.len(),
    );

    // ── 4. Parallel modality vector via decode_modality ────────
    let modalities: Vec<CerebModality> = tokens
        .iter()
        .map(|&t| map_modality(tokenizer.decode_modality(t)))
        .collect();
    assert_eq!(modalities.len(), tokens.len());

    // ── Per-position diagnostic table ──────────────────────────
    println!("multimodal_smoke: token sequence (pos, token_id, modality):");
    for (i, (&t, &m)) in tokens.iter().zip(modalities.iter()).enumerate() {
        println!("  [{:2}] {:>7}  {}", i, t, tag(m));
    }

    // Layout sanity-check: 10 Byte, 1 Delim, 4 ImageVq, 1 Delim.
    let n_byte = modalities.iter().filter(|&&m| m == CerebModality::Byte).count();
    let n_delim = modalities.iter().filter(|&&m| m == CerebModality::Delimiter).count();
    let n_image = modalities.iter().filter(|&&m| m == CerebModality::ImageVq).count();
    println!(
        "multimodal_smoke: per-modality counts → Byte={}, Delim={}, ImageVq={}",
        n_byte, n_delim, n_image,
    );
    assert_eq!(n_byte, PROMPT.len(), "expected {} Byte tokens, got {}", PROMPT.len(), n_byte);
    assert_eq!(n_delim, 2, "expected 2 Delimiter tokens, got {}", n_delim);
    assert_eq!(n_image, N_IMAGE_CODES, "expected {} ImageVq tokens, got {}", N_IMAGE_CODES, n_image);

    // ── 5. Build a tiny GptModelResident ────────────────────────
    let config = GptConfig {
        model_dim: ModelDim::new(MODEL_DIM),
        num_heads: NumHeads::new(NUM_HEADS),
        num_kv_heads: NumKvHeads::new(NUM_KV_HEADS),
        head_dim: HeadDim::new(HEAD_DIM),
        num_layers: NumLayers::new(NUM_LAYERS),
        vocab_size: VocabSize::new(VOCAB_SIZE),
        mlp_dim: MlpDim::new(MLP_DIM),
        max_seq_len: SeqLen::new(MAX_SEQ_LEN),
        rope_base: ROPE_BASE,
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
        norm_eps: NORM_EPS,
        use_qk_norm: false,
    };

    let md = config.model_dim.get();
    let kv_dim = config.num_kv_heads.get() * config.head_dim.get();
    let vocab = config.vocab_size.get();
    let mlp_dim = config.mlp_dim.get();

    let randn = |rng: &mut SimpleRng, n: usize| -> Vec<f32> {
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
        let attn = CausalSelfAttention::new(attn_w, &config);

        let swiglu_w = SwigluWeights {
            gate: Tensor2::new(randn(&mut rng, mlp_dim * md), mlp_dim, md).unwrap(),
            up:   Tensor2::new(randn(&mut rng, mlp_dim * md), mlp_dim, md).unwrap(),
            down: Tensor2::new(randn(&mut rng, md * mlp_dim), md, mlp_dim).unwrap(),
        };
        let swiglu = SwigluMlp::new(swiglu_w, config.model_dim, config.mlp_dim);
        swiglu_mlps.push(swiglu);

        let placeholder_mlp = Mlp::new(
            MlpWeights {
                fc:   Tensor2::zeros(mlp_dim, md),
                proj: Tensor2::zeros(md, mlp_dim),
            },
            config.model_dim, config.mlp_dim,
        );
        let layer_idx = LayerIdx::new(li, config.num_layers).unwrap();
        blocks_host.push(TransformerBlock::new(
            attn, placeholder_mlp, None, layer_idx, &config,
        ));
    }

    let model = GptModel {
        embed:     Tensor2::new(token_embed, vocab, md).unwrap(),
        lm_head:   Tensor2::new(lm_head, vocab, md).unwrap(),
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

    let resident = GptModelResident::from_model(&model, &swiglu_mlps)
        .expect("upload GptModelResident");
    let mut cereb = QwenCerebellum::from_resident(resident, MAX_SEQ_LEN)
        .expect("construct QwenCerebellum");

    // ── 6. Encode the unified token sequence ────────────────────
    let mut cache = cereb.encode_context_layers(&tokens);
    cache.modalities = Some(modalities.clone());

    assert_eq!(
        cache.n_layers, NUM_LAYERS,
        "cache.n_layers: expected {}, got {}", NUM_LAYERS, cache.n_layers,
    );
    assert_eq!(
        cache.n_positions, SEQ_LEN,
        "cache.n_positions: expected {}, got {}", SEQ_LEN, cache.n_positions,
    );
    assert_eq!(
        cache.hidden_dim, MODEL_DIM,
        "cache.hidden_dim: expected {}, got {}", MODEL_DIM, cache.hidden_dim,
    );
    for (i, &v) in cache.hidden_states.iter().enumerate() {
        assert!(v.is_finite(), "non-finite cache entry at flat offset {}: {}", i, v);
    }

    // ── 7. Build CerebProjection and pool per modality ──────────
    let proj = CerebProjection::with_layers(
        CORTEX_DIM,
        /* frozen_input_dim  */ MODEL_DIM,
        /* frozen_output_dim */ MODEL_DIM,
        NUM_LAYERS,
    );

    let mut text_out = vec![0.0f32; CORTEX_DIM];
    let matched_text =
        cerebellum_modality_pool(&cache, &proj, CerebModality::Byte, &mut text_out);
    assert!(matched_text, "Byte-modality pool reported no matches; cache.modalities tag suspect");

    let mut image_out = vec![0.0f32; CORTEX_DIM];
    let matched_image =
        cerebellum_modality_pool(&cache, &proj, CerebModality::ImageVq, &mut image_out);
    assert!(matched_image, "ImageVq-modality pool reported no matches; cache.modalities tag suspect");

    for (i, &v) in text_out.iter().enumerate() {
        assert!(v.is_finite(), "non-finite text cortex output at idx {}: {}", i, v);
    }
    for (i, &v) in image_out.iter().enumerate() {
        assert!(v.is_finite(), "non-finite image cortex output at idx {}: {}", i, v);
    }

    // ── 8. Per-layer variance signatures ────────────────────────
    let byte_positions: Vec<usize> = modalities
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| if m == CerebModality::Byte { Some(i) } else { None })
        .collect();
    let image_positions: Vec<usize> = modalities
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| if m == CerebModality::ImageVq { Some(i) } else { None })
        .collect();

    let byte_var = variance_signature(&cache, &byte_positions);
    let image_var = variance_signature(&cache, &image_positions);
    println!(
        "multimodal_smoke: per-layer variance (Byte positions {:?}): {:?}",
        byte_positions, byte_var,
    );
    println!(
        "multimodal_smoke: per-layer variance (ImageVq positions {:?}): {:?}",
        image_positions, image_var,
    );

    // ── 9. The hdevalence assertion ─────────────────────────────
    // A degenerate run where text and image tokens hit the model
    // identically would produce nearly-equal cortex vectors. With
    // distinct embeddings + per-layer hidden states diverging across
    // the two contiguous token regions, the L2 distance must clear
    // the threshold.
    let dist = l2(&text_out, &image_out);
    println!(
        "multimodal_smoke: L2(text_cortex, image_cortex) = {:.6} (threshold > {})",
        dist, L2_MIN,
    );
    assert!(
        dist > L2_MIN,
        "cortex projections collapsed: L2 = {:.6} <= {} — text and image regions are indistinguishable",
        dist, L2_MIN,
    );

    // Also assert each pooled output is non-trivially non-zero.
    let text_max = text_out.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let image_max = image_out.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    assert!(text_max > 0.0, "text cortex projection collapsed to all-zeros");
    assert!(image_max > 0.0, "image cortex projection collapsed to all-zeros");

    println!(
        "PASS: text/image cortex projections distinguishable (L2 distance = {:.4})",
        dist,
    );
}
