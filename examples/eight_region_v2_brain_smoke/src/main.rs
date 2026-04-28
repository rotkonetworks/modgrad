//! End-to-end smoke: the `eight_region_v2` regional brain actually
//! consults a [`QwenCerebellum`]-backed sibling service during its
//! forward pass.
//!
//! ## What this proves
//!
//! Per `docs/BRAIN_ARCHITECTURE.md` §5–§7, the cerebellum is a
//! sibling-service LLM that the cortex regions read from. Until this
//! slice the seam was wired in isolation
//! (`examples/cortex_cerebellum_reader`) but the regional forward
//! itself never consulted the cerebellum. Today's wiring closes the
//! loop:
//!
//! 1. Build a tiny `GptModelResident` (3 layers, model_dim=32, vocab
//!    256) standing in for Qwen2.5-0.5B.
//! 2. Wrap as `QwenCerebellum` (frozen, exposes per-layer hidden
//!    cache).
//! 3. Build `RegionalConfig::eight_region_v2` (8 regions + the
//!    cerebellum→attention edge added in this slice) +
//!    `RegionalWeights::new`.
//! 4. Construct a `CerebellumService` from the cerebellum +
//!    `CerebProjection` sized cortex_dim = cerebellum-region d_model.
//! 5. `service.set_context(&token_ids)` for a small sequence.
//! 6. Run `regional_forward_with_service` for one and two ticks (the
//!    cortex consults the service per-tick at the cerebellum→attention
//!    edge).
//! 7. Run a baseline `regional_forward` (no service — placeholder
//!    cerebellum-region CTM is the only signal source) for the same
//!    inputs.
//! 8. Assert: the attention region's activation differs measurably
//!    between the two forwards. `delta_l2 > 0` proves the service is
//!    actually consulted; a zero delta would mean the wiring is dead.
//!
//! ## What this does NOT prove
//!
//! - **No backward pass.** Frozen cerebellum, per arch doc §7 (LoRA /
//!   low-LR backward is a separate slice).
//! - **No multimodal pooling.** Position-based read only (v0); the
//!   modality-aware path is exercised by `cortex_cerebellum_reader`.
//! - **No isis-runtime training.** That wires `CerebellumService` into
//!   the per-token training step; separate slice.
//!
//! Run:
//!
//! ```text
//! cargo run --release -p eight_region_v2_brain_smoke
//! ```

use modgrad_compute::neuron::SimpleRng;
use modgrad_ctm::cerebellum::CerebProjection;
use modgrad_ctm::cerebellum_service::CerebellumService;
use modgrad_ctm::graph::{
    regional_forward, regional_forward_with_service, RegionalConfig, RegionalState,
    RegionalWeights,
};
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

// ─── Tiny LLM dims (matching qwen_cerebellum_smoke) ────────────
const VOCAB_SIZE: usize = 256;
const MODEL_DIM: usize = 32;       // also = eight_region_v2's CEREB_D_MODEL placeholder
const NUM_HEADS: usize = 4;
const NUM_KV_HEADS: usize = 4;
const HEAD_DIM: usize = 8;
const NUM_LAYERS: usize = 3;
const MLP_DIM: usize = 64;
const MAX_SEQ_LEN: usize = 16;
const ROPE_BASE: f32 = 10_000.0;
const NORM_EPS: f32 = 1e-5;

// ─── Regional brain dims ────────────────────────────────────────
// `obs_dim = 128` matches eight_region_v2's intended cortex sizing
// (CORTEX_D_MODEL = 128 elsewhere). out_dims = 64 is the vocab/head
// size — small but non-trivial. ticks = 2 keeps the smoke fast; the
// service-vs-baseline delta surfaces on tick 0, but running 2 ticks
// confirms the per-tick read pattern doesn't accidentally degrade
// over time.
const OBS_DIM: usize = 128;
const OUT_DIMS: usize = 64;
const N_TICKS: usize = 2;

// ─── Position used for service.read_at ──────────────────────────
// We encode a 4-token sequence; position 0 is the read site. Any
// in-range position works; 0 is the canonical "current token".
const READ_POSITION: usize = 0;

// ─── Discrimination threshold ───────────────────────────────────
// L2 of (attention_with_service - attention_without_service) must
// exceed this. A degenerate path (service signal collapses to zero,
// or never reaches the attention region) would push this to ~0 and
// trip the assert.
const MIN_DELTA_L2: f32 = 1e-4;

const SEED: u64 = 0xCAFE_BEEF;

fn l2(x: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for &v in x { s += v * v; }
    s.sqrt()
}

fn l2_diff(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut s = 0.0f32;
    for i in 0..n {
        let d = a[i] - b[i];
        s += d * d;
    }
    s.sqrt()
}

fn build_tiny_resident_qwen() -> GptModelResident {
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

    let mut rng = SimpleRng::new(SEED);
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

    GptModelResident::from_model(&model, &swiglu_mlps).expect("upload resident")
}

fn main() {
    if !runtime_available() {
        println!("eight_region_v2_brain_smoke: HIP runtime unavailable, skipping");
        std::process::exit(0);
    }

    println!("eight_region_v2_brain_smoke: building tiny resident Qwen-class transformer");
    let resident = build_tiny_resident_qwen();
    let cereb = QwenCerebellum::from_resident(resident, MAX_SEQ_LEN)
        .expect("construct QwenCerebellum");
    println!(
        "eight_region_v2_brain_smoke: cerebellum hidden_dim={}, n_layers={}",
        modgrad_ctm::cerebellum::FrozenCerebellum::hidden_dim(&cereb),
        modgrad_ctm::cerebellum::FrozenCerebellum::n_layers(&cereb),
    );

    println!("eight_region_v2_brain_smoke: building eight_region_v2 brain ({} obs_dim, {} out_dims, {} ticks)",
        OBS_DIM, OUT_DIMS, N_TICKS);
    let cfg = RegionalConfig::eight_region_v2(OBS_DIM, OUT_DIMS, N_TICKS);
    let n_connections = cfg.connections.len();
    let cereb_idx = cfg.region_names.iter().position(|n| n == "cerebellum").expect("cerebellum region");
    let attention_idx = cfg.region_names.iter().position(|n| n == "attention").expect("attention region");
    let cereb_d_model = cfg.regions[cereb_idx].d_model;
    println!(
        "eight_region_v2_brain_smoke: regions={}, connections={}, cerebellum d_model={}, attention idx={}",
        cfg.regions.len(), n_connections, cereb_d_model, attention_idx,
    );
    assert_eq!(n_connections, 9, "eight_region_v2 should have 9 edges (8 base + cerebellum→attention)");
    let weights = RegionalWeights::new(cfg);

    // ── Mount the service ─────────────────────────────────────
    // Per arch doc §7 option (b): the cerebellum is a sibling service.
    // Projection sizing: cortex_dim = cerebellum-region d_model. The
    // service signal substitutes for `prev_outputs[CEREBELLUM]` in the
    // synapse source vector for any edge with `from = [CEREBELLUM]`,
    // so the wire dim must match.
    let proj = CerebProjection::with_layers(
        cereb_d_model,            // cortex_dim — what the synapse expects
        MODEL_DIM,                // frozen_input_dim (unused on this path)
        MODEL_DIM,                // frozen_output_dim — must equal cereb.hidden_dim()
        NUM_LAYERS,
    );
    let mut service = CerebellumService::new(Box::new(cereb), proj);

    // ── Encode a tiny sequence ────────────────────────────────
    let tokens: Vec<i64> = vec![7, 31, 91, 137];
    println!("eight_region_v2_brain_smoke: service.set_context(token_ids = {:?})", tokens);
    service.set_context(&tokens);
    assert_eq!(service.n_positions(), tokens.len());

    // ── Run forward WITHOUT service (baseline) ────────────────
    let observation: Vec<f32> = (0..OBS_DIM).map(|i| (i as f32 * 0.013).sin()).collect();
    let mut state_baseline = RegionalState::new(&weights);
    let _baseline = regional_forward(&weights, &mut state_baseline, &observation);
    let attn_baseline = state_baseline.region_outputs[attention_idx].clone();
    println!(
        "eight_region_v2_brain_smoke: baseline attention activation: dim={}, ||y||₂={:.6}",
        attn_baseline.len(), l2(&attn_baseline),
    );

    // ── Run forward WITH service ──────────────────────────────
    let mut state_with = RegionalState::new(&weights);
    let _with = regional_forward_with_service(
        &weights, &mut state_with, &observation, &service, READ_POSITION,
    );
    let attn_with = state_with.region_outputs[attention_idx].clone();
    println!(
        "eight_region_v2_brain_smoke: service-aware attention activation: dim={}, ||y||₂={:.6}",
        attn_with.len(), l2(&attn_with),
    );

    // ── The architectural assertion ───────────────────────────
    // Per arch doc §5: the cerebellum's per-layer hidden states ARE
    // the world-model representation the cortex reads. If the wiring
    // is dead (service never consulted, or its signal collapses to
    // zero before reaching attention) the two attention activations
    // collapse to identical, delta_l2 → 0, and we trip this assert.
    let delta = l2_diff(&attn_baseline, &attn_with);
    assert!(
        delta > MIN_DELTA_L2,
        "attention activation identical between with-service and without-service forwards \
         (delta L2 = {:.2e} ≤ {:.2e}); cerebellum→attention edge is wired but unread",
        delta, MIN_DELTA_L2,
    );

    // Per-region smoke: every region produced finite, dim-correct output.
    for (r, out) in state_with.region_outputs.iter().enumerate() {
        let expected = weights.regions[r].config.d_model;
        assert_eq!(out.len(), expected,
            "region {} output dim mismatch: {} != {}", r, out.len(), expected);
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "non-finite at region {} idx {}: {}", r, i, v);
        }
    }

    println!("PASS: brain consults cerebellum during forward (delta L2 = {:.6})", delta);
}
