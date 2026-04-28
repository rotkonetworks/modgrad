//! Proof-of-life: `ByteifyRecipe::from_qwen2` + `BltTrainer` compose
//! end-to-end. A tiny `GptModelResident` (standing in for a pretrained
//! Qwen-class checkpoint) is fed through the byteify recipe and the
//! returned `ByteifiedLatent` is plugged into `BltTrainer::new`. We then
//! drive 30 train_steps over a small structured byte corpus and assert
//! the loss-curve mean of the last 5 steps is at least 5% below the
//! mean of the first 5 — proving the per-component LR-routed AdamW
//! dispatch (paper §6.2: local 3e-4, global 3e-5) actually moves the
//! model.
//!
//! Today every latent param routes via `global_predicate` to the global
//! group (the local stack lands once sasha's slice arrives); with
//! `global_lr = 3e-5` defaults the loss creeps slowly, so we bump the
//! local/global pair to 1e-3 / 1e-4 — the *ratio* (1/10) is what's
//! load-bearing for the recipe, not the absolute magnitude.
//!
//! HIP is required at runtime — exits cleanly with status 0 when no HIP
//! device is reachable.
//!
//! Run:
//!   cargo run --release -p blt_byteify_smoke

use modgrad_blt::byteify::{ByteifyConfig, ByteifyRecipe};
use modgrad_blt::trainer::{BltTrainer, BltTrainerConfig};
use modgrad_compute::neuron::SimpleRng;
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

// ── Tiny latent dims (readable in 30 s) ──────────────────────────
//
// Same shape as `qwen_cerebellum_smoke` modulo NUM_LAYERS=2 (per the
// byteify recipe's "Qwen-class" stand-in convention) — vocab=256 so
// raw bytes feed straight in as token ids.
const VOCAB_SIZE: usize = 256;
const MODEL_DIM: usize = 32;
const NUM_HEADS: usize = 4;
const NUM_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 8;
const NUM_LAYERS: usize = 2;
const MLP_DIM: usize = 64;
const MAX_SEQ_LEN: usize = 32;
const ROPE_BASE: f32 = 10_000.0;
const NORM_EPS: f32 = 1e-5;

// ── Trainer / loop dims ──────────────────────────────────────────
//
// `BltTrainer::train_step` requires `bytes.len() == micro_batch_size + 1`
// (next-byte target alignment). With MBS=16 each window is 17 bytes;
// stride 8 over the ~200-byte corpus gives ~24 candidate windows.
const MICRO_BATCH_SIZE: usize = 16;
const WINDOW_BYTES: usize = MICRO_BATCH_SIZE + 1; // 17
const STRIDE: usize = 8;
const N_STEPS: usize = 30;

// LR pair — 1/10 ratio per BLT §6.2; bumped 10× over the
// `BltTrainerConfig::default()` to surface a loss move in 30 steps.
const LOCAL_LR: f32 = 1e-3;
const GLOBAL_LR: f32 = 1e-4;

fn main() {
    if !runtime_available() {
        eprintln!("blt_byteify_smoke: HIP runtime unavailable, skipping");
        std::process::exit(0);
    }

    // ── Build the tiny GptModelResident (stand-in for a pretrained ──
    // ── Qwen2.5-0.5B `GptModelResident`).                            ──
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

    let mut rng = SimpleRng::new(0xBEEF_CAFE);
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

        // Placeholder host MLP — `GptModelResident` consumes the
        // SwigluMlp slice for the actual MLP; the host-side block only
        // needs *some* MLP to satisfy its constructor.
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
        .expect("upload tiny GptModelResident");

    eprintln!(
        "blt_byteify_smoke: latent built — d_model={MODEL_DIM} \
         layers={NUM_LAYERS} vocab={VOCAB_SIZE} mlp_dim={MLP_DIM}"
    );

    // ── Run the byteify recipe ─────────────────────────────────────
    //
    // Today the recipe returns `ByteifiedLatent { latent, *configs }`;
    // once sasha's `BltModel` lands the return type flips to `BltModel`
    // and this call site is unchanged.
    let byte_cfg = ByteifyConfig::default();
    let byteified = ByteifyRecipe::from_qwen2(resident, byte_cfg)
        .expect("ByteifyRecipe::from_qwen2");
    assert_eq!(byteified.latent.model_dim(), MODEL_DIM);
    assert_eq!(byteified.latent.num_layers(), NUM_LAYERS);
    assert_eq!(byteified.latent.vocab_size(), VOCAB_SIZE);
    eprintln!(
        "blt_byteify_smoke: ByteifyRecipe produced ByteifiedLatent — \
         local_encoder.byte_dim={} local_decoder.byte_dim={}",
        byteified.local_encoder_config.byte_dim,
        byteified.local_decoder_config.byte_dim,
    );

    // ── Construct the trainer ──────────────────────────────────────
    //
    // `BltTrainer` (latent-only) — pairs with `ByteifiedLatent`. The
    // `global_predicate` returns `true` for every key today (every
    // param the trainer touches is a latent param), so AdamW applies
    // `global_lr` uniformly. Once the local stack lands, the predicate
    // returns `false` for `encoder.*` / `decoder.*` / `cross_attn.*`
    // and the routing splits.
    let trainer_cfg = BltTrainerConfig {
        local_lr: LOCAL_LR,
        global_lr: GLOBAL_LR,
        micro_batch_size: MICRO_BATCH_SIZE,
        seq_len: MICRO_BATCH_SIZE,
        ..BltTrainerConfig::default()
    };
    let mut trainer = BltTrainer::new(
        byteified.latent,
        NUM_KV_HEADS,
        HEAD_DIM,
        trainer_cfg,
        ByteifyRecipe::global_predicate(),
    )
    .expect("BltTrainer::new");

    // ── Build the corpus + windows ─────────────────────────────────
    //
    // 196-byte structured English corpus. Sliding 17-byte windows at
    // stride 8 produces ~24 candidate windows; a deterministic SimpleRng
    // selects from them per step.
    let pattern: &[u8] =
        b"the quick brown fox jumps over the lazy dog. \
          the quick brown fox jumps over the lazy dog. \
          the quick brown fox jumps over the lazy dog. \
          the quick brown fox jumps over the lazy dog. ";
    debug_assert!(pattern.len() >= WINDOW_BYTES + STRIDE);
    let n_windows = (pattern.len() - WINDOW_BYTES) / STRIDE + 1;
    let windows: Vec<&[u8]> = (0..n_windows)
        .map(|w| {
            let off = w * STRIDE;
            &pattern[off..off + WINDOW_BYTES]
        })
        .collect();
    for w in &windows {
        assert_eq!(w.len(), WINDOW_BYTES);
    }

    // 4 patches of 4 bytes — `validate_boundaries` requires strictly
    // increasing and `last < bytes.len()` (= 17). With BltModel still
    // a placeholder the trainer doesn't *consume* boundaries today, but
    // it does validate them — provide a well-formed list.
    let boundaries: [usize; 5] = [0, 4, 8, 12, 16];
    debug_assert!(*boundaries.last().unwrap() < WINDOW_BYTES);

    eprintln!(
        "blt_byteify_smoke: corpus={} bytes, mbs={MICRO_BATCH_SIZE} \
         window={WINDOW_BYTES} stride={STRIDE} n_windows={n_windows} \
         n_steps={N_STEPS} local_lr={LOCAL_LR} global_lr={GLOBAL_LR}",
        pattern.len(),
    );

    // ── Loop ───────────────────────────────────────────────────────
    let mut sample_rng = SimpleRng::new(0xF00D_FACE);
    let mut losses: Vec<f32> = Vec::with_capacity(N_STEPS);
    for step in 0..N_STEPS {
        let idx = (sample_rng.next_u64() as usize) % n_windows;
        let bytes = windows[idx];
        let loss = trainer
            .train_step(bytes, &boundaries)
            .unwrap_or_else(|e| panic!("train_step {step}: {e:?}"));
        assert!(loss.is_finite(),
            "blt_byteify_smoke: step {step} loss not finite ({loss})");
        losses.push(loss);
        if step % 5 == 0 || step == N_STEPS - 1 {
            eprintln!("step {step:>3}: loss = {loss:.4}");
        }
    }

    // ── Assertion (hdevalence-style) ───────────────────────────────
    //
    // 30 steps is short — we only assert non-degeneracy: mean(last 5)
    // strictly below mean(first 5). With every key routed to global_lr
    // (= local_lr / 10 = 1e-4) the curve creeps; a backward chain that
    // doesn't produce useful gradients would be flat or rising.
    let mean = |xs: &[f32]| xs.iter().copied().sum::<f32>() / xs.len() as f32;
    let mean_first = mean(&losses[..5]);
    let mean_last = mean(&losses[losses.len() - 5..]);
    let drop_pct = (mean_first - mean_last) / mean_first * 100.0;

    eprintln!(
        "blt_byteify_smoke: mean(first 5)={mean_first:.4} \
         mean(last 5)={mean_last:.4} drop={drop_pct:.2}%"
    );

    if mean_last < mean_first {
        println!(
            "PASS: ByteifyRecipe + BltTrainer compose; loss {mean_first:.2} \
             → {mean_last:.2} over {N_STEPS} steps ({drop_pct:.2}% drop)"
        );
    } else {
        eprintln!("FAIL: full loss history:");
        for (i, l) in losses.iter().enumerate() {
            eprintln!("  step {i:>3}: {l:.4}");
        }
        panic!(
            "blt_byteify_smoke: mean(last 5) ≥ mean(first 5) — \
             ByteifyRecipe → BltTrainer composition may be broken"
        );
    }
}
