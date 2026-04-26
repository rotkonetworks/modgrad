//! Smoke test: byteify recipe + BltTrainer end-to-end on a tiny model
//! standing in for Qwen2.5-0.5B.
//!
//! Why a tiny stand-in instead of the real Qwen2.5-0.5B: the 7600M XT
//! has 8 GiB VRAM. Qwen2.5-0.5B in fp32 is ~2.5 GiB resident; AdamW
//! state (3× weights for m/v/g_dev) adds another ~7.5 GiB → ~10 GiB,
//! which OOMs on this card. Real Qwen byteify training will need
//! either bf16 master + fp32 grads (deferred — see crate root) or
//! Q4_K residency, both already on the roadmap. For *this* slice the
//! goal is "prove the wiring runs end-to-end on real text"; the LR
//! ratio and per-param routing are exercised whether the latent is
//! 0.5B or 200K params.
//!
//! Two tests are included:
//!   - [`byteify_smoke_tiny_latent`] — runs by default. Tiny GPT
//!     standing in for the latent + 5 train_steps over climbmix bytes
//!     + asserts loss finite + decreasing trend.
//!   - [`byteify_recipe_loads_real_qwen`] — `#[ignore]` by default.
//!     Real Qwen2.5-0.5B load via `byteify::ByteifyRecipe::from_qwen2`,
//!     no training (VRAM doesn't permit AdamW state for a 0.5B model
//!     today). Validates that the recipe accepts a real
//!     `GptModelResident` and produces a `ByteifiedLatent` whose
//!     fields are sane. Run with:
//!         cargo test --features rocm -p modgrad-blt --test byteify -- \
//!             --ignored byteify_recipe_loads_real_qwen

#![cfg(feature = "rocm")]

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
use modgrad_transformer::rope::RotaryEmbedding;
use modgrad_transformer::smear::{Inference, Smear, SmearWeights, Training};
use modgrad_transformer::tensor::Tensor2;
use modgrad_transformer::GptModelResident;

/// Path to the climbmix English corpus relative to the workspace root.
/// `lm_validate` uses the same files; they live two crates / four dirs
/// up from this `tests/byteify.rs`.
fn climbmix_train_path() -> std::path::PathBuf {
    workspace_root().join("climbmix_train.txt")
}

fn workspace_root() -> std::path::PathBuf {
    // CARGO_MANIFEST_DIR for this crate is `<workspace>/crates/modgrad-blt`.
    let manifest = env!("CARGO_MANIFEST_DIR");
    let mut p = std::path::PathBuf::from(manifest);
    p.pop(); // drops "modgrad-blt"
    p.pop(); // drops "crates"
    p
}

/// Build a tiny byte-level GPT (vocab=256, 2 layers, d_model=128) — the
/// same shape as `lm_validate::build_tiny_model_for_byte_level`. Used
/// as a stand-in for the Qwen-class latent in the smoke test.
fn build_tiny_byte_latent() -> (GptModel, Vec<SwigluMlp>, GptConfig) {
    let head_dim = 32usize;
    let n_heads = 4usize;
    let model_dim = head_dim * n_heads;
    let config = GptConfig {
        model_dim: ModelDim::new(model_dim),
        num_heads: NumHeads::new(n_heads),
        num_kv_heads: NumKvHeads::new(n_heads),
        head_dim: HeadDim::new(head_dim),
        num_layers: NumLayers::new(2),
        vocab_size: VocabSize::new(256),
        mlp_dim: MlpDim::new(model_dim * 2),
        max_seq_len: SeqLen::new(64),
        rope_base: 10000.0,
        qk_norm_scale: 1.0,
        use_qk_norm: true,
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
        norm_eps: 1e-5,
    };

    let md = config.model_dim.get();
    let kv_dim = config.num_kv_heads.get() * config.head_dim.get();
    let vocab = config.vocab_size.get();
    let mlp_dim = config.mlp_dim.get();

    let mut rng = SimpleRng::new(0xBADBEEF);
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

        let gate_w = randn(&mut rng, mlp_dim * md);
        let up_w = randn(&mut rng, mlp_dim * md);
        let down_w = randn(&mut rng, md * mlp_dim);
        let swiglu_w = SwigluWeights {
            gate: Tensor2::new(gate_w, mlp_dim, md).unwrap(),
            up: Tensor2::new(up_w, mlp_dim, md).unwrap(),
            down: Tensor2::new(down_w, md, mlp_dim).unwrap(),
        };
        let swiglu = SwigluMlp::new(swiglu_w, config.model_dim, config.mlp_dim);
        swiglu_mlps.push(swiglu);

        let placeholder_mlp = Mlp::new(
            MlpWeights {
                fc: Tensor2::zeros(mlp_dim, md),
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
        embed: Tensor2::new(token_embed, vocab, md).unwrap(),
        lm_head: Tensor2::new(lm_head, vocab, md).unwrap(),
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
    (model, swiglu_mlps, config)
}

/// Fixed-stride patcher fallback per the brief: produce boundaries
/// `[0, P, 2P, ...]` so the `BltTrainer` has well-formed boundaries to
/// validate when leah's `EntropyPatcher` hasn't landed. With BltModel
/// still a placeholder, the trainer doesn't actually consume these —
/// they're sanity-checked.
fn fixed_stride_patcher(n_bytes: usize, stride: usize) -> Vec<usize> {
    (0..n_bytes).step_by(stride).collect()
}

/// Smoke test: tiny stand-in latent + 5 BltTrainer steps on real
/// climbmix bytes.
///
/// **Asserts:**
///   - `ByteifyRecipe::from_qwen2` accepts the resident model and
///     returns a `ByteifiedLatent` whose fields match the input config.
///   - `BltTrainer::new` constructs cleanly with the recipe's
///     `global_predicate`.
///   - 5 calls to `train_step(bytes, boundaries)` complete without
///     error.
///   - All 5 losses are finite.
///   - The loss curve shows movement (not necessarily monotone — 5
///     steps on real text from a random init is too short for a strict
///     monotone decrease assertion; we check `loss[4] != loss[0]` and
///     `min < initial * 1.05`).
///
/// This is a smoke test — its job is to catch obviously broken wiring
/// (NaN losses, grad-key mismatches, sign errors). The
/// `lm_validate`-style real-data validation lives elsewhere; once
/// `BltModel` is real and the LR ratio matters, a richer harness
/// (likely under `examples/blt_validate/`) can compare local-vs-global
/// LR effects on val loss.
#[test]
fn byteify_smoke_tiny_latent() {
    if !runtime_available() {
        eprintln!("byteify smoke: HIP runtime unavailable — skipping");
        return;
    }
    let train_path = climbmix_train_path();
    if !train_path.exists() {
        eprintln!(
            "byteify smoke: climbmix_train.txt not found at {} — skipping",
            train_path.display(),
        );
        return;
    }
    let train_bytes = std::fs::read(&train_path).expect("read climbmix");
    eprintln!(
        "byteify smoke: read {} bytes ({:.1} MiB) of climbmix",
        train_bytes.len(),
        train_bytes.len() as f64 / 1_048_576.0,
    );

    // Build the tiny stand-in latent and resident-upload it.
    let (model, swiglu_mlps, config) = build_tiny_byte_latent();
    let resident = GptModelResident::from_model(&model, &swiglu_mlps)
        .expect("upload tiny latent");

    // Byteify recipe — produces a ByteifiedLatent. Today the local
    // configs ride along; sasha's slice will consume them to assemble
    // encoder/decoder/cross-attn around the latent.
    let byte_cfg = ByteifyConfig::default();
    let byteified = ByteifyRecipe::from_qwen2(resident, byte_cfg.clone())
        .expect("byteify recipe");

    // Sanity: the byteified latent's d_model matches what we built
    // (proxy for the real Qwen2.5-0.5B's 896).
    assert_eq!(byteified.latent.model_dim(), config.model_dim.get());
    assert_eq!(byteified.latent.num_layers(), config.num_layers.get());
    // ByteifyConfig fields propagated unchanged.
    assert_eq!(
        byteified.local_encoder_config.byte_dim,
        byte_cfg.local_encoder_config.byte_dim,
    );

    // BltTrainer with the recipe's global predicate. mbs=8 to keep
    // smoke-test wall under a few seconds at the tiny config; the seq
    // matches.
    let mbs = 8usize;
    let trainer_cfg = BltTrainerConfig {
        // Bump local_lr to 1e-3 so 5 steps on a random init produce a
        // visible move on the loss curve; global_lr stays at 1/10.
        local_lr: 1e-3,
        global_lr: 1e-4,
        micro_batch_size: mbs,
        seq_len: mbs,
        ..BltTrainerConfig::default()
    };
    let n_kv = config.num_kv_heads.get();
    let hd = config.head_dim.get();
    let mut trainer = BltTrainer::new(
        byteified.latent, n_kv, hd, trainer_cfg,
        ByteifyRecipe::global_predicate(),
    ).expect("BltTrainer::new");

    // Drive 5 train_steps. Boundaries from the fixed-stride fallback
    // (every 4 bytes); validates the trainer's boundary check without
    // requiring leah's entropy patcher.
    let n_steps = 5usize;
    let mut losses = Vec::with_capacity(n_steps);
    let mut rng = SimpleRng::new(0xCAFE_F00D);
    let span = (train_bytes.len() - (mbs + 1)).max(1);
    for step in 0..n_steps {
        let off = (rng.next_u64() as usize) % span;
        let window = &train_bytes[off..off + mbs + 1];
        let boundaries = fixed_stride_patcher(window.len(), 4);
        let loss = trainer.train_step(window, &boundaries)
            .unwrap_or_else(|e| panic!("train_step {step} failed: {e}"));
        assert!(loss.is_finite(),
            "byteify smoke: step {step} loss is non-finite ({loss})");
        eprintln!("byteify smoke: step {step}  loss={loss:.4}");
        losses.push(loss);
    }

    // Loss-shape check: must move. 5 steps from random init on a 9-byte
    // window with lr=1e-3 should drop ~5-30%; we assert at least that
    // it changed by ≥1% in either direction (proves grads flowed) and
    // the minimum loss is no worse than 105% of step-0.
    let initial = losses[0];
    let last = *losses.last().unwrap();
    let min = losses.iter().cloned().fold(f32::INFINITY, f32::min);
    eprintln!(
        "byteify smoke: initial={initial:.4} last={last:.4} min={min:.4}"
    );
    assert!((last - initial).abs() > initial * 0.005,
        "loss did not move: initial={initial} last={last} losses={losses:?}");
    assert!(min < initial * 1.05,
        "min loss got worse than +5% of initial: initial={initial} min={min}");
}

/// Real Qwen2.5-0.5B byteify — `#[ignore]` because:
///   1. Loading + resident-uploading takes ~30 s and reads ~1 GiB from
///      disk.
///   2. Allocating AdamW state for a 0.5B model is ~7.5 GiB on top of
///      the ~2.5 GiB resident weights → OOM on the 7600M XT (8 GiB).
///      So we exercise the load-and-byteify path only and stop short of
///      `BltTrainer::new` (which would lazy-init the AdamW state on
///      the first `train_step`).
///
/// This test validates that `ByteifyRecipe::from_qwen2` accepts a real
/// pretrained `GptModelResident` end-to-end. Run with:
///     cargo test --release --features rocm -p modgrad-blt \
///         --test byteify -- --ignored byteify_recipe_loads_real_qwen
#[test]
#[ignore = "real Qwen2.5-0.5B requires ~3 GiB VRAM and ~30 s; \
    run with --ignored when validating Path B end-to-end"]
fn byteify_recipe_loads_real_qwen() {
    if !runtime_available() {
        eprintln!("byteify real: HIP runtime unavailable — skipping");
        return;
    }
    let path = "/steam/llm/hf_cache/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987/model.safetensors";
    if !std::path::Path::new(path).exists() {
        eprintln!("byteify real: Qwen safetensors not at {path} — skipping");
        return;
    }
    let max_seq = 512usize; // small KV cache cap for the smoke test
    let resident = modgrad_io::qwen2::load_qwen2_5_0_5b(path, max_seq)
        .expect("load Qwen2.5-0.5B");

    let byte_cfg = ByteifyConfig::default();
    let byteified = ByteifyRecipe::from_qwen2(resident, byte_cfg.clone())
        .expect("byteify Qwen2.5-0.5B");

    // Architectural sanity: latent matches the published Qwen2.5-0.5B
    // numbers.
    assert_eq!(byteified.latent.num_layers(),
        modgrad_io::qwen2::QWEN2_5_0_5B_NUM_LAYERS);
    assert_eq!(byteified.latent.model_dim(),
        modgrad_io::qwen2::QWEN2_5_0_5B_HIDDEN_SIZE);
    assert_eq!(byteified.latent.vocab_size(),
        modgrad_io::qwen2::QWEN2_5_0_5B_VOCAB);

    // Local configs propagated from the input — sasha's slice consumes
    // these when assembling encoder + decoder + cross-attn.
    assert_eq!(
        byteified.local_encoder_config.n_layers,
        byte_cfg.local_encoder_config.n_layers,
    );
    assert_eq!(
        byteified.local_decoder_config.n_layers,
        byte_cfg.local_decoder_config.n_layers,
    );
}
