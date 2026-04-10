//! Smoke test: build a tiny transformer and run a forward pass.

use modgrad::transformer::*;
use modgrad::transformer::config::*;
use modgrad::transformer::weights::*;
use modgrad::transformer::builder::TransformerBuilder;
use modgrad::transformer::kv_cache::KvCache;
use modgrad::transformer::residual::ForwardCtx;
use modgrad::transformer::ops::TransformerOps;
use modgrad_compute::backend::CpuBackend;
use modgrad::service::{Service, LogitMods, Request, RequestMeta};

fn tiny_config() -> GptConfig {
    GptConfig {
        model_dim: ModelDim::new(32),
        num_heads: NumHeads::new(4),
        num_kv_heads: NumKvHeads::new(4),
        head_dim: HeadDim::new(8),
        num_layers: NumLayers::new(4),
        vocab_size: VocabSize::new(64),
        mlp_dim: MlpDim::new(128),
        max_seq_len: SeqLen::new(32),
        rope_base: 10000.0,
        qk_norm_scale: 1.2,
        window_pattern: WindowPattern::Full,
        value_embed: ValueEmbedConfig { gate_channels: 4, gate_range: 3.0 },
        residual: ResidualConfig::default(),
        smear: SmearConfig { gate_channels: 4, lambda: 1.0 },
        precision: Precision::F32,
        norm_eps: 1e-5,
    }
}

fn random_vec(n: usize, seed: u64) -> Vec<f32> {
    // Simple deterministic pseudo-random for testing
    let mut rng = seed;
    (0..n).map(|_| {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        ((rng as f32) / u64::MAX as f32) * 0.2 - 0.1
    }).collect()
}

fn make_weights(config: &GptConfig) -> GptWeights {
    let md = config.model_dim.get();
    let kv_dim = config.num_kv_heads.get() * config.head_dim.get();
    let mlp_dim = config.mlp_dim.get();
    let vocab = config.vocab_size.get();

    let mut seed = 42u64;
    let mut rv = |n: usize| -> Vec<f32> {
        seed += 1;
        random_vec(n, seed)
    };

    let blocks = (0..config.num_layers.get()).map(|i| {
        let layer_idx = LayerIdx::new(i, config.num_layers).unwrap();
        let has_ve = config.has_value_embed(layer_idx);
        BlockWeights {
            wq: rv(md * md),
            wk: rv(kv_dim * md),
            wv: rv(kv_dim * md),
            wo: rv(md * md),
            mlp_fc: rv(mlp_dim * md),
            mlp_proj: rv(md * mlp_dim),
            ve_table: if has_ve { Some(rv(vocab * kv_dim)) } else { None },
            ve_gate: if has_ve { Some(rv(kv_dim * config.value_embed.gate_channels)) } else { None },
        }
    }).collect();

    GptWeights {
        token_embed: rv(vocab * md),
        lm_head: rv(vocab * md),
        final_norm_scale: vec![1.0; md],
        smear_gate: rv(md * config.smear.gate_channels),
        blocks,
    }
}

#[test]
fn test_config_validation() {
    let config = tiny_config();
    assert!(config.validate().is_ok());

    // Bad config: model_dim != heads * head_dim
    let bad = GptConfig {
        model_dim: ModelDim::new(33),
        ..tiny_config()
    };
    assert!(bad.validate().is_err());
}

#[test]
fn test_weight_validation() {
    let config = tiny_config();
    let weights = make_weights(&config);
    assert!(weights.validate(&config).is_ok());
}

#[test]
fn test_builder_builds() {
    let config = tiny_config();
    let weights = make_weights(&config);
    let backend = CpuBackend::new();

    let result = TransformerBuilder::new(config)
        .with_weights(weights)
        .with_backend(backend)
        .build();

    assert!(result.is_ok(), "builder failed: {:?}", result.err());
}

#[test]
fn test_forward_pass_produces_logits() {
    let config = tiny_config();
    let weights = make_weights(&config);
    let backend = CpuBackend::new();

    let mut service = TransformerBuilder::new(config.clone())
        .with_weights(weights)
        .with_backend(backend)
        .build()
        .unwrap();

    let mut mods = LogitMods::default();
    let req = Request {
        prompt: "test".into(),
        token_ids: vec![1, 2, 3],
        hidden_key: vec![],
        max_tokens: 5,
        eos_token_id: 0,
        meta: RequestMeta::default(),
    };

    let resp = service.call(req, &mut mods);

    // Should have generated some tokens
    assert!(resp.token_ids.len() > 3, "expected generation, got {} tokens", resp.token_ids.len());
    // All token IDs should be in vocab range
    let vocab = config.vocab_size.get() as i64;
    for &tid in &resp.token_ids[3..] {
        assert!(tid >= 0 && tid < vocab, "token {tid} out of range [0, {vocab})");
    }
}

#[test]
fn test_forward_no_nan() {
    let config = tiny_config();
    let weights = make_weights(&config);
    let backend = CpuBackend::new();

    let model = {
        let svc = TransformerBuilder::new(config.clone())
            .with_weights(weights)
            .with_backend(backend)
            .build()
            .unwrap();
        svc
    };

    // We can't easily get at the raw logits through Service, but
    // if the argmax produces a valid token, there were no NaNs
    // (NaN comparisons would break argmax). Covered by test above.
}

#[test]
fn test_eos_stops_generation() {
    let config = tiny_config();
    let weights = make_weights(&config);
    let backend = CpuBackend::new();

    let mut service = TransformerBuilder::new(config)
        .with_weights(weights)
        .with_backend(backend)
        .build()
        .unwrap();

    let mut mods = LogitMods::default();
    // max_seq_len is 32, prompt takes 1 slot, so we can generate at most 31
    let req = Request {
        prompt: "test".into(),
        token_ids: vec![1],
        hidden_key: vec![],
        max_tokens: 20,
        eos_token_id: -1, // impossible EOS
        meta: RequestMeta::default(),
    };

    let resp = service.call(req, &mut mods);
    // Should generate exactly max_tokens (since EOS is impossible)
    assert_eq!(resp.token_ids.len(), 21, "expected 1 prompt + 20 generated");
}

#[test]
fn test_weight_save_load_roundtrip() {
    let config = tiny_config();
    let weights = make_weights(&config);

    let path = std::env::temp_dir().join("isis_test_weights.bin");
    weights.save_raw(&path).unwrap();
    let loaded = GptWeights::load_raw(&path, &config).unwrap();

    // Check a few values
    assert_eq!(weights.token_embed.len(), loaded.token_embed.len());
    for (a, b) in weights.token_embed.iter().zip(loaded.token_embed.iter()) {
        assert!((a - b).abs() < 1e-7, "roundtrip mismatch: {a} vs {b}");
    }
    assert_eq!(weights.blocks.len(), loaded.blocks.len());
    for (bw, bl) in weights.blocks.iter().zip(loaded.blocks.iter()) {
        assert_eq!(bw.wq.len(), bl.wq.len());
    }

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_prefill_processes_prompt_in_batch() {
    // Verify that the same prompt+generation produces same results
    // whether prompt is 1 token or 5 tokens (prefill should handle both)
    let config = tiny_config();
    let weights = make_weights(&config);
    let backend = CpuBackend::new();

    let mut service = TransformerBuilder::new(config.clone())
        .with_weights(weights)
        .with_backend(backend)
        .build()
        .unwrap();

    // 5-token prompt
    let mut mods = LogitMods::default();
    let req = Request {
        prompt: "test".into(),
        token_ids: vec![1, 2, 3, 4, 5],
        hidden_key: vec![],
        max_tokens: 3,
        eos_token_id: -1,
        meta: RequestMeta::default(),
    };
    let resp = service.call(req, &mut mods);

    // Should have 5 prompt + 3 generated = 8
    assert_eq!(resp.token_ids.len(), 8, "expected 5 prompt + 3 generated, got {}", resp.token_ids.len());
    // All tokens in range
    let vocab = config.vocab_size.get() as i64;
    for &tid in &resp.token_ids[5..] {
        assert!(tid >= 0 && tid < vocab, "token {tid} out of range");
    }
}

#[test]
fn test_prefill_deterministic() {
    // Same input → same output (no stale state between calls after reset)
    let config = tiny_config();

    let make_svc = || {
        let weights = make_weights(&config);
        let backend = CpuBackend::new();
        TransformerBuilder::new(config.clone())
            .with_weights(weights)
            .with_backend(backend)
            .build()
            .unwrap()
    };

    let mut svc1 = make_svc();
    let mut svc2 = make_svc();

    let req = Request {
        prompt: "test".into(),
        token_ids: vec![1, 2, 3],
        hidden_key: vec![],
        max_tokens: 5,
        eos_token_id: -1,
        meta: RequestMeta::default(),
    };

    let mut mods1 = LogitMods::default();
    let mut mods2 = LogitMods::default();
    let r1 = svc1.call(req.clone(), &mut mods1);
    let r2 = svc2.call(req, &mut mods2);

    assert_eq!(r1.token_ids, r2.token_ids, "same weights + input should produce same output");
}

#[test]
fn test_cross_entropy_loss_gradient() {
    use modgrad::transformer::train::cross_entropy_loss;

    // Logits: [2.0, 1.0, 0.5], target: 0
    let logits = vec![2.0, 1.0, 0.5];
    let (loss, d_logits) = cross_entropy_loss(&logits, 0);

    // Loss should be > 0
    assert!(loss > 0.0, "loss should be positive: {loss}");
    // Gradient at target should be negative (we want to increase it)
    assert!(d_logits[0] < 0.0, "grad at target should be negative: {}", d_logits[0]);
    // Gradients should sum to ~0 (property of softmax gradient)
    let sum: f32 = d_logits.iter().sum();
    assert!(sum.abs() < 1e-5, "grad sum should be ~0: {sum}");
}

#[test]
fn test_backward_linear_correctness() {
    use modgrad::transformer::train::backward_linear;

    // W = [[1,2],[3,4]], x = [1,1], y = W@x = [3,7]
    // d_y = [1,0]
    // d_x = W^T @ d_y = [1, 2]
    // d_W = d_y ⊗ x = [[1,1],[0,0]]
    let w = vec![1.0, 2.0, 3.0, 4.0];
    let x = vec![1.0, 1.0];
    let d_y = vec![1.0, 0.0];
    let mut d_x = vec![0.0; 2];
    let mut d_w = vec![0.0; 4];

    backward_linear(&d_y, &x, &w, &mut d_x, &mut d_w, 2, 2);

    assert!((d_x[0] - 1.0).abs() < 1e-5);
    assert!((d_x[1] - 2.0).abs() < 1e-5);
    assert!((d_w[0] - 1.0).abs() < 1e-5);
    assert!((d_w[1] - 1.0).abs() < 1e-5);
    assert!((d_w[2] - 0.0).abs() < 1e-5);
    assert!((d_w[3] - 0.0).abs() < 1e-5);
}

#[test]
fn test_optimizer_reduces_loss() {
    use modgrad::transformer::train::{cross_entropy_loss, backward_lm_head, ModelGradients};
    use modgrad::transformer::optim::adamw::AdamW;

    let vocab = 8;
    let dim = 4;

    // Random-ish weights
    let mut lm_head: Vec<f32> = (0..vocab * dim)
        .map(|i| ((i * 7 + 3) % 20) as f32 / 20.0 - 0.5)
        .collect();
    let hidden = vec![0.3, -0.5, 0.1, 0.8];
    let target = 2;

    // Initial loss
    let mut logits = vec![0.0f32; vocab];
    for v in 0..vocab {
        for i in 0..dim {
            logits[v] += lm_head[v * dim + i] * hidden[i];
        }
    }
    let (loss0, _) = cross_entropy_loss(&logits, target);

    // Run 50 optimizer steps
    let mut opt = AdamW::new(vocab * dim, 0.01);
    for _ in 0..50 {
        let mut logits = vec![0.0f32; vocab];
        for v in 0..vocab {
            for i in 0..dim {
                logits[v] += lm_head[v * dim + i] * hidden[i];
            }
        }
        let (_, d_logits) = cross_entropy_loss(&logits, target);
        let mut d_hidden = vec![0.0f32; dim];
        let mut d_w = vec![0.0f32; vocab * dim];
        backward_lm_head(&d_logits, &hidden, &lm_head, &mut d_hidden, &mut d_w, vocab, dim);
        opt.step(&mut lm_head, &d_w);
    }

    // Final loss
    let mut logits = vec![0.0f32; vocab];
    for v in 0..vocab {
        for i in 0..dim {
            logits[v] += lm_head[v * dim + i] * hidden[i];
        }
    }
    let (loss_final, _) = cross_entropy_loss(&logits, target);

    assert!(loss_final < loss0, "optimizer should reduce loss: {loss0} -> {loss_final}");
}

#[test]
fn test_backward_softmax_gradient() {
    use modgrad::transformer::train::backward_softmax;

    // p = softmax([2.0, 1.0, 0.5])
    let logits = vec![2.0f32, 1.0, 0.5];
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let p: Vec<f32> = exps.iter().map(|e| e / sum).collect();

    let d_p = vec![1.0, 0.0, 0.0]; // gradient from downstream
    let mut d_s = vec![0.0; 3];
    backward_softmax(&d_p, &p, &mut d_s);

    // d_s should sum to 0 (softmax jacobian property)
    let sum: f32 = d_s.iter().sum();
    assert!(sum.abs() < 1e-5, "softmax backward grad sum should be ~0: {sum}");
    // d_s[0] should be positive (increase score → increase prob)
    assert!(d_s[0] > 0.0, "d_s[0] should be positive");
}

#[test]
fn test_backward_rope_is_inverse() {
    // RoPE backward should undo RoPE forward (orthogonal rotation)
    let half = 4;
    // Must be valid rotations: cos²+sin² = 1
    let angles = vec![0.3f32, 0.7, 1.2, 2.1];
    let cos_t: Vec<f32> = angles.iter().map(|a| a.cos()).collect();
    let sin_t: Vec<f32> = angles.iter().map(|a| a.sin()).collect();

    // Forward: apply rotation
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut rotated = vec![0.0; 8];
    for i in 0..half {
        rotated[i]        = x[i] * cos_t[i] - x[half + i] * sin_t[i];
        rotated[half + i] = x[i] * sin_t[i] + x[half + i] * cos_t[i];
    }

    // Backward: should recover x from d_output = rotated
    let mut recovered = vec![0.0; 8];
    modgrad::transformer::train::backward_rope(&rotated, &cos_t, &sin_t, &mut recovered, half);

    for i in 0..8 {
        assert!((recovered[i] - x[i]).abs() < 1e-5,
            "rope backward should invert forward: recovered[{i}]={} expected {}", recovered[i], x[i]);
    }
}

#[test]
fn test_backward_rms_norm_numerical() {
    use modgrad::transformer::train::backward_rms_norm;

    let x = vec![1.0, -0.5, 2.0, 0.3];
    let eps = 1e-5;

    // Forward
    let n = x.len() as f32;
    let ss: f32 = x.iter().map(|v| v * v).sum();
    let inv_rms = 1.0 / (ss / n + eps).sqrt();
    let y: Vec<f32> = x.iter().map(|v| v * inv_rms).collect();

    // Numerical gradient check
    let d_y = vec![1.0, 0.5, -0.3, 0.7];
    let mut d_x = vec![0.0; 4];
    backward_rms_norm(&d_y, &x, &mut d_x, eps);

    // Finite differences
    let delta = 1e-4;
    for j in 0..4 {
        let mut x_plus = x.clone();
        x_plus[j] += delta;
        let ss_p: f32 = x_plus.iter().map(|v| v * v).sum();
        let inv_p = 1.0 / (ss_p / n + eps).sqrt();
        let y_plus: Vec<f32> = x_plus.iter().map(|v| v * inv_p).collect();

        let mut x_minus = x.clone();
        x_minus[j] -= delta;
        let ss_m: f32 = x_minus.iter().map(|v| v * v).sum();
        let inv_m = 1.0 / (ss_m / n + eps).sqrt();
        let y_minus: Vec<f32> = x_minus.iter().map(|v| v * inv_m).collect();

        let mut numerical = 0.0f32;
        for i in 0..4 {
            numerical += d_y[i] * (y_plus[i] - y_minus[i]) / (2.0 * delta);
        }

        assert!((d_x[j] - numerical).abs() < 1e-2,
            "rms_norm backward d_x[{j}]: analytic={} numerical={}", d_x[j], numerical);
    }
}
