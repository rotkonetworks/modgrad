//! Proof-of-life: a tiny `QwenCerebellum` exposes per-layer hidden
//! states via `encode_context_layers`; `CerebProjection::with_layers`
//! reads them through `cerebellum_at_position` and produces finite
//! cortex activations; `RegionalConfig::eight_region_v2` builds the
//! cerebellum-dominant 8-region brain config without panicking.
//!
//! For this binary to fail, one of: the resident transformer upload
//! path, the per-layer hidden-state cache layout, the layer-blend +
//! `project_out` matvec dispatch, or the `eight_region_v2` topology
//! would have to be broken. HIP is required at runtime — the binary
//! exits cleanly with status 0 on hosts without a HIP device.
//!
//! Run:
//!   cargo run --release -p qwen_cerebellum_smoke

use modgrad_compute::neuron::SimpleRng;
use modgrad_ctm::cerebellum::{
    cerebellum_at_position, CerebProjection, FrozenCerebellum,
};
use modgrad_ctm::graph::RegionalConfig;
use modgrad_ctm::qwen_cerebellum::QwenCerebellum;
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

// ── Tiny LLM dims (assumptions readable in 30 s) ────────────────
const VOCAB_SIZE: usize = 256;
const MODEL_DIM: usize = 32;
const NUM_HEADS: usize = 4;
const NUM_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 8;
const NUM_LAYERS: usize = 3;
const MLP_DIM: usize = 64;
const MAX_SEQ_LEN: usize = 32;
const ROPE_BASE: f32 = 10_000.0;
const NORM_EPS: f32 = 1e-5;

// ── Brain dims (eight_region_v2) ────────────────────────────────
const OBS_DIM: usize = 128;
const OUT_DIMS: usize = 64;
const TICKS: usize = 16;

// ── Cortex projection dims ──────────────────────────────────────
const CORTEX_DIM: usize = 64;

fn main() {
    if !runtime_available() {
        println!("qwen_cerebellum_smoke: HIP runtime unavailable, skipping");
        std::process::exit(0);
    }

    // ── Build a tiny GptModelResident ────────────────────────────
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

    let mut rng = SimpleRng::new(0xC0FFEE_BEEF);
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

        // Placeholder MLP — `GptModelResident` consumes the SwigluMlp
        // slice for the actual MLP; the host-side `TransformerBlock`
        // only needs *some* MLP to satisfy its constructor.
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

    // ── Encode 32-token sequence ────────────────────────────────
    let tokens: Vec<i64> = (0..MAX_SEQ_LEN as i64)
        .map(|i| i % VOCAB_SIZE as i64)
        .collect();
    assert_eq!(tokens.len(), MAX_SEQ_LEN);

    let cache = cereb.encode_context_layers(&tokens);

    // ── Cache shape ─────────────────────────────────────────────
    assert_eq!(
        cache.n_layers, NUM_LAYERS,
        "cerebellum should expose {} layers, got {}", NUM_LAYERS, cache.n_layers,
    );
    assert_eq!(
        cache.n_positions, MAX_SEQ_LEN,
        "cache n_positions: expected {}, got {}", MAX_SEQ_LEN, cache.n_positions,
    );
    assert_eq!(
        cache.hidden_dim, MODEL_DIM,
        "cache hidden_dim: expected {}, got {}", MODEL_DIM, cache.hidden_dim,
    );
    let expected_len = NUM_LAYERS * MAX_SEQ_LEN * MODEL_DIM;
    assert_eq!(
        cache.hidden_states.len(), expected_len,
        "cache flat length: expected {} (= {} * {} * {}), got {}",
        expected_len, NUM_LAYERS, MAX_SEQ_LEN, MODEL_DIM, cache.hidden_states.len(),
    );

    // ── Finite check ────────────────────────────────────────────
    for (i, &v) in cache.hidden_states.iter().enumerate() {
        assert!(
            v.is_finite(),
            "non-finite cache entry at flat offset {}: {}", i, v,
        );
    }

    // ── Per-layer non-trivial activation at position 0 ──────────
    for li in 0..NUM_LAYERS {
        let h = cache.at(li, 0);
        let sq: f32 = h.iter().map(|x| x * x).sum();
        assert!(
            sq > 0.0,
            "layer {} position 0 is all-zeros (||h||² = {}); embed/block dispatch suspect",
            li, sq,
        );
    }

    // ── Sanity: uniform-weight blend at position 0 == mean(per-layer h[0]) ──
    let uniform = vec![1.0f32 / NUM_LAYERS as f32; NUM_LAYERS];
    let blended = cache.blend_layers(0, &uniform);
    assert_eq!(
        blended.len(), MODEL_DIM,
        "blend_layers output: expected {}, got {}", MODEL_DIM, blended.len(),
    );
    let mut manual = vec![0.0f32; MODEL_DIM];
    for li in 0..NUM_LAYERS {
        let h = cache.at(li, 0);
        for i in 0..MODEL_DIM {
            manual[i] += h[i] / NUM_LAYERS as f32;
        }
    }
    for i in 0..MODEL_DIM {
        let diff = (blended[i] - manual[i]).abs();
        assert!(
            diff < 1e-5,
            "blend disagrees with manual mean at idx {}: blended={}, manual={}, diff={}",
            i, blended[i], manual[i], diff,
        );
    }

    // ── eight_region_v2 topology ────────────────────────────────
    let cfg = RegionalConfig::eight_region_v2(OBS_DIM, OUT_DIMS, TICKS);
    assert_eq!(
        cfg.regions.len(), 8,
        "eight_region_v2 should have 8 regions, got {}", cfg.regions.len(),
    );
    let expected_names = [
        "input", "attention", "output", "motor",
        "cerebellum", "basal_ganglia", "insula", "hippocampus",
    ];
    for name in expected_names {
        let found = cfg.region_names.iter().any(|n| n == name);
        assert!(
            found,
            "eight_region_v2 missing region '{}'; got {:?}", name, cfg.region_names,
        );
    }
    // Arch doc §2: 8 edges, with the all-cortex→hippocampus row as ONE
    // Connection record carrying `from = [INPUT, ATTENTION, OUTPUT, MOTOR]`.
    assert_eq!(
        cfg.connections.len(), 8,
        "eight_region_v2 should have 8 connection records (arch doc §2), got {}",
        cfg.connections.len(),
    );
    assert_eq!(
        cfg.outer_ticks, TICKS,
        "outer_ticks: expected {}, got {}", TICKS, cfg.outer_ticks,
    );
    assert_eq!(
        cfg.out_dims, OUT_DIMS,
        "out_dims: expected {}, got {}", OUT_DIMS, cfg.out_dims,
    );
    assert_eq!(
        cfg.raw_obs_dim, OBS_DIM,
        "raw_obs_dim: expected {}, got {}", OBS_DIM, cfg.raw_obs_dim,
    );

    // ── Cache → cortex projection ───────────────────────────────
    let proj = CerebProjection::with_layers(
        CORTEX_DIM,
        /* frozen_input_dim  */ MODEL_DIM,
        /* frozen_output_dim */ MODEL_DIM,
        NUM_LAYERS,
    );
    assert_eq!(
        proj.layer_weight_logits.len(), NUM_LAYERS,
        "CerebProjection layer_weight_logits: expected {}, got {}",
        NUM_LAYERS, proj.layer_weight_logits.len(),
    );

    let mut out = vec![0.0f32; CORTEX_DIM];
    cerebellum_at_position(&cache, &proj, 0, &mut out);
    assert_eq!(
        out.len(), CORTEX_DIM,
        "cortex projection length: expected {}, got {}", CORTEX_DIM, out.len(),
    );
    for (i, &v) in out.iter().enumerate() {
        assert!(
            v.is_finite(),
            "non-finite cortex activation at idx {}: {}", i, v,
        );
    }
    // The projection isn't required to be non-zero for every output
    // index, but at least one component should differ from the bias —
    // a stuck-at-bias projection means the matvec didn't dispatch.
    let abs_max = out.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    assert!(
        abs_max > 0.0,
        "cortex projection collapsed to all-zeros (max |out| = 0); proj_out matvec suspect",
    );

    println!(
        "PASS: cerebellum cache shape OK, projection produces finite cortex activations"
    );
}
