//! Proof-of-life: a synthetic cortex region actually CONSUMES the
//! cerebellum cache during its forward pass. The orchestrator encodes
//! a 16-token unified sequence (text + image VQ) once, attaches the
//! parallel modality vector, and then — position-by-position — pools
//! the cerebellum cache for that position's modality and feeds the
//! pooled vector into a `Linear` standing in for what an actual cortex
//! region's input projection would be. Per-modality cortex L2 norms
//! must come out measurably distinct, otherwise the seam is degenerate.
//!
//! For this binary to fail, one of: the resident transformer forward,
//! the per-position modality tagging, the modality-aware mean-pool +
//! `project_out` matvec dispatch, or the `Linear::forward_into` matvec
//! would have to be broken — or the cerebellum would be returning the
//! same pooled signal for every modality (e.g. zero-fill on miss).
//! HIP is required at runtime — the binary exits cleanly with status 0
//! on hosts without a HIP device.
//!
//! The synthetic "cortex region" here is just a `Linear` from
//! `modgrad-compute::neuron`; it stands in for what an actual cortex
//! region's input-side projection would be. Wiring this through the
//! full `RegionalConfig` forward path (so a real region in
//! `eight_region_v2` reads the cerebellum on every tick) is a separate
//! slice — that requires architectural decisions about WHEN/HOW a
//! region reads, which are deliberately out of scope here.
//!
//! Run:
//!   cargo run --release -p cortex_cerebellum_reader

use modgrad_compute::neuron::{Linear, SimpleRng};
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
const SEED: u64 = 0xC0FFEE_BEEF;

// ── Tiny LLM dims (assumptions readable in 30 s) ────────────────
// vocab=152000 to match `multimodal_smoke` so we can consume the
// unified token vocab (audio-VQ codes park at offsets up to 148200,
// even though we don't use them here — keeping the embed table sized
// for the full unified range avoids ad-hoc surprises).
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

// ── Sequence layout (16 tokens) ─────────────────────────────────
//   "describe: " (10 bytes) + <img> + 4 image VQ codes + </img>
const PROMPT: &[u8] = b"describe: ";
const N_IMAGE_VQ: usize = 4;
//  10 byte +  1 open + 4 imvq + 1 close = 16
const SEQ_LEN: usize = PROMPT.len() + 1 + N_IMAGE_VQ + 1;

// ── Discrimination threshold ────────────────────────────────────
// The mean cortex-output L2 for byte positions versus image-VQ
// positions must differ by at least this ratio (max/min). A run that
// regressed to "always pool the same vector" would push this ratio
// to 1.0 and trip the assertion below.
const MIN_RATIO: f32 = 1.05;

/// Map `modgrad-data`'s `Modality` into `modgrad-ctm`'s `Modality`.
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
/// sum each quadrant, mod 4096 → 4 image VQ codes (u16). Same shape
/// as `multimodal_smoke::fake_image_vq_encode`.
fn fake_image_vq_encode(grid: &[f32; 64]) -> [u16; N_IMAGE_VQ] {
    let mut sums = [0.0f32; N_IMAGE_VQ];
    for row in 0..8 {
        for col in 0..8 {
            let q = (if row < 4 { 0 } else { 2 }) + if col < 4 { 0 } else { 1 };
            sums[q] += grid[row * 8 + col];
        }
    }
    let mut codes = [0u16; N_IMAGE_VQ];
    for q in 0..N_IMAGE_VQ {
        let scaled = (sums[q].abs() * 1_000_000.0) as u64;
        codes[q] = (scaled % 4096) as u16;
    }
    codes
}

/// L2 norm of a vector.
fn l2_norm(x: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for &v in x {
        s += v * v;
    }
    s.sqrt()
}

/// Mean of a slice (returns 0.0 on empty input).
fn mean(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f32>() / xs.len() as f32
}

fn main() {
    if !runtime_available() {
        println!("cortex_cerebellum_reader: HIP runtime unavailable, skipping");
        std::process::exit(0);
    }

    // ── 1. Deterministic synthetic 8×8 image → 4 image VQ codes ──
    let mut rng = SimpleRng::new(SEED);
    let mut grid = [0.0f32; 64];
    for v in grid.iter_mut() {
        *v = rng.next_normal();
    }
    let image_codes = fake_image_vq_encode(&grid);
    println!(
        "cortex_cerebellum_reader: synthetic image VQ codes = {:?}",
        image_codes,
    );

    // ── 2. Build unified 16-token sequence ──────────────────────
    //   "describe: " + <img> + imvq×4 + </img>
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

    // ── 3. Parallel modality vector via decode_modality ─────────
    let modalities: Vec<CerebModality> = tokens
        .iter()
        .map(|&t| map_modality(tokenizer.decode_modality(t)))
        .collect();
    assert_eq!(modalities.len(), tokens.len());

    println!(
        "cortex_cerebellum_reader: token sequence (pos, token_id, modality):"
    );
    for (i, (&t, &m)) in tokens.iter().zip(modalities.iter()).enumerate() {
        println!("  [{:2}] {:>7}  {}", i, t, tag(m));
    }

    let n_byte = modalities.iter().filter(|&&m| m == CerebModality::Byte).count();
    let n_delim = modalities.iter().filter(|&&m| m == CerebModality::Delimiter).count();
    let n_image = modalities.iter().filter(|&&m| m == CerebModality::ImageVq).count();
    println!(
        "cortex_cerebellum_reader: per-modality counts → Byte={}, Delim={}, ImageVq={}",
        n_byte, n_delim, n_image,
    );
    assert_eq!(n_byte, PROMPT.len());
    assert_eq!(n_delim, 2);
    assert_eq!(n_image, N_IMAGE_VQ);

    // ── 4. Build a tiny GptModelResident (same shape as smokes) ─
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

    // ── 5. Encode the unified sequence + attach modality tags ───
    let mut cache = cereb.encode_context_layers(&tokens);
    cache.modalities = Some(modalities.clone());

    assert_eq!(cache.n_layers, NUM_LAYERS);
    assert_eq!(cache.n_positions, SEQ_LEN);
    assert_eq!(cache.hidden_dim, MODEL_DIM);
    for (i, &v) in cache.hidden_states.iter().enumerate() {
        assert!(v.is_finite(), "non-finite cache entry at {}: {}", i, v);
    }

    // ── 6. Build CerebProjection (cache → cortex) ───────────────
    let proj = CerebProjection::with_layers(
        CORTEX_DIM,
        /* frozen_input_dim  */ MODEL_DIM,
        /* frozen_output_dim */ MODEL_DIM,
        NUM_LAYERS,
    );

    // ── 7. Build the synthetic "cortex region" — a Linear ───────
    // This stands in for what an actual cortex region's input-side
    // projection would be. The full RegionalConfig forward path is a
    // separate slice (see top-of-file note).
    let cortex_linear = Linear::new(CORTEX_DIM, CORTEX_DIM);

    // ── 8. THE ACTUAL DEMONSTRATION ─────────────────────────────
    // Per-position synthetic-cortex forward, reading the cerebellum
    // cache through `cerebellum_modality_pool` for that position's
    // tagged modality.
    let mut cereb_input = vec![0.0f32; CORTEX_DIM];
    let mut cortex_output = vec![0.0f32; CORTEX_DIM];

    // (byte_pos, modality, ||cortex_output||₂)
    let mut all_outputs: Vec<(usize, CerebModality, f32)> = Vec::with_capacity(SEQ_LEN);

    for byte_pos in 0..SEQ_LEN {
        let modality = modalities[byte_pos];

        // Pool cerebellum cache for that modality (zero-fills + returns
        // false when the modality isn't present, but every modality in
        // our sequence has at least one match, so all positions get a
        // real pooled vector).
        cereb_input.fill(0.0);
        let matched = cerebellum_modality_pool(&cache, &proj, modality, &mut cereb_input);
        assert!(
            matched,
            "pool reported no match at pos {} for modality {:?}; cache.modalities tag suspect",
            byte_pos, modality,
        );

        // Synthetic cortex forward: cortex_output = Linear(cereb_input).
        cortex_output.fill(0.0);
        cortex_linear.forward_into(&cereb_input, &mut cortex_output);
        for (i, &v) in cortex_output.iter().enumerate() {
            assert!(
                v.is_finite(),
                "non-finite cortex output at pos {} idx {}: {}",
                byte_pos, i, v,
            );
        }

        all_outputs.push((byte_pos, modality, l2_norm(&cortex_output)));
    }

    // ── 9. Per-position diagnostic table ────────────────────────
    println!("cortex_cerebellum_reader: per-position cortex output L2 norms:");
    println!("  pos  modality   ||y||₂");
    for &(p, m, n) in &all_outputs {
        println!("  [{:2}]  {}    {:.6}", p, tag(m), n);
    }

    // ── 10. Per-modality mean L2 + assertion ────────────────────
    let byte_norms: Vec<f32> = all_outputs.iter()
        .filter(|(_, m, _)| *m == CerebModality::Byte)
        .map(|(_, _, n)| *n)
        .collect();
    let image_norms: Vec<f32> = all_outputs.iter()
        .filter(|(_, m, _)| *m == CerebModality::ImageVq)
        .map(|(_, _, n)| *n)
        .collect();
    let delim_norms: Vec<f32> = all_outputs.iter()
        .filter(|(_, m, _)| *m == CerebModality::Delimiter)
        .map(|(_, _, n)| *n)
        .collect();

    let mean_byte = mean(&byte_norms);
    let mean_image = mean(&image_norms);
    let mean_delim = mean(&delim_norms);

    println!("cortex_cerebellum_reader: per-modality mean cortex L2:");
    println!("  Byte    (n={}): {:.6}", byte_norms.len(), mean_byte);
    println!("  ImageVq (n={}): {:.6}", image_norms.len(), mean_image);
    println!("  Delim   (n={}): {:.6}", delim_norms.len(), mean_delim);

    // **The hdevalence assertion**: byte vs image mean L2 must differ by
    // at least MIN_RATIO. A degenerate `cerebellum_modality_pool` that
    // returned the same vector regardless of modality (e.g. always-zero,
    // or always all-positions average) would push this ratio toward 1.0
    // and trip the assertion. We also confirm both means are nonzero.
    assert!(mean_byte > 0.0, "byte cortex outputs collapsed to zero — cerebellum/projection seam dead");
    assert!(mean_image > 0.0, "image cortex outputs collapsed to zero — cerebellum/projection seam dead");

    let ratio = mean_byte.max(mean_image) / mean_byte.min(mean_image);
    assert!(
        ratio >= MIN_RATIO,
        "byte vs image mean cortex L2 too close (ratio={:.4} < {:.4}); cerebellum_modality_pool may be returning a position-agnostic signal",
        ratio, MIN_RATIO,
    );

    // Also confirm byte and image positions don't all produce identical
    // outputs WITHIN a modality — every position in a modality goes
    // through the same pool but the Linear is shared, so all per-modality
    // norms within a modality should be identical (mean-pool collapses
    // per-position info). That's expected for THIS slice; we just record
    // the within-modality variance for the human reader.
    let var = |xs: &[f32], m: f32| -> f32 {
        if xs.is_empty() {
            return 0.0;
        }
        xs.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / xs.len() as f32
    };
    println!(
        "cortex_cerebellum_reader: within-modality variance (expected ≈0, mean-pool collapses positions):"
    );
    println!("  Byte    var: {:.2e}", var(&byte_norms, mean_byte));
    println!("  ImageVq var: {:.2e}", var(&image_norms, mean_image));
    println!("  Delim   var: {:.2e}", var(&delim_norms, mean_delim));

    println!(
        "PASS: cortex reads modality-distinguishing signal from cerebellum \
         (mean L2 byte={:.4}, image={:.4}, ratio={:.4})",
        mean_byte, mean_image, ratio,
    );
}
