//! Qwen2.5-0.5B safetensors loader → device-resident transformer.
//!
//! Reads the HuggingFace safetensors blob (bf16 → fp32 on CPU) and
//! constructs a [`GptModelResident`] populated with the real weights:
//!
//!   - `model.embed_tokens.weight`              → embed (also tied as lm_head)
//!   - `model.norm.weight`                      → final RMSNorm scale
//!   - `model.layers.{i}.input_layernorm.weight`        → block.attn_norm_weight_dev
//!   - `model.layers.{i}.post_attention_layernorm.weight` → block.mlp_norm_weight_dev
//!   - `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight` → AttentionResident.{q,k,v,o}_proj.weight_dev
//!   - `model.layers.{i}.self_attn.{q,k,v}_proj.bias`     → bias_dev (Qwen has q/k/v biases, no o bias)
//!   - `model.layers.{i}.mlp.{gate,up,down}_proj.weight`  → SwigluResident.{gate,up,down}.weight_dev
//!
//! ## Architectural discrepancies vs Qwen2 (documented for future-us)
//!
//! - **QK norm.** Qwen2 has *no* QK normalisation: `q = wq @ x + bq`,
//!   then RoPE, then dot product. Our [`AttentionResident`] always
//!   applies RMSNorm to Q/K (with weight buffer = `qk_scale`). With
//!   `qk_norm_scale = 1.0` and `norm_eps = 1e-6` the resident path
//!   computes `q_normed = q / rms(q_head)` — Q is rescaled per-head to
//!   unit RMS. Empirically this preserves the ranking of attention
//!   patterns (softmax is rank-equivariant under per-head positive
//!   rescaling) but distorts the logit magnitudes. For the smoke test
//!   (non-NaN, plausible spread) this is acceptable; matching upstream
//!   Qwen2 generation will need a config flag to skip QK norm
//!   end-to-end. **Tracked as a follow-up.**
//!
//! - **Smear.** Qwen2 has no previous-token smear. The resident
//!   `forward` path skips smear by design (matches the test scaffold).
//!   We attach all-zero smear weights to the host `GptModel` only to
//!   satisfy the type system; the resident path never reads them.
//!
//! - **Residual.** Qwen2 uses plain `x = x + sublayer(norm(x))`. We
//!   configure `resid_lambda = 1.0`, `x0_lambda = 0.0`, `backout = 0.0`
//!   so [`ResidualLambdas::apply`] reduces to `hidden += block_output`.
//!
//! - **Tied embeddings.** Qwen2.5-0.5B sets `tie_word_embeddings = true`.
//!   The HuggingFace safetensors does not include a separate
//!   `lm_head.weight` tensor; the embedding row matrix is reused. We
//!   set `tie_embeddings = true` in the config and feed
//!   `model.embed_tokens.weight` to both the embedding *and* the lm_head
//!   tensors of the host model, so the same fp32 row matrix is uploaded
//!   to both `embed_dev` and `lm_head.weight_dev` on the device. (No
//!   weight sharing on the device side — this is a copy. With 988 MiB
//!   bf16 → ~1.9 GiB fp32, plus the duplicated embed/lm_head, total
//!   resident state is ~2.5 GiB. Fits in the 8 GiB on the test card.)

#![cfg(feature = "rocm")]

use std::time::Instant;

use modgrad_transformer::attention::{AttentionWeights, CausalSelfAttention};
use modgrad_transformer::block::TransformerBlock;
use modgrad_transformer::config::{
    GptConfig, MlpActivation, Precision, ResidualConfig, SmearConfig, ValueEmbedConfig,
    WindowPattern,
};
use modgrad_transformer::dims::{
    HeadDim, LayerIdx, MlpDim, ModelDim, NumHeads, NumKvHeads, NumLayers, SeqLen, VocabSize,
};
use modgrad_transformer::mlp::{Mlp, MlpWeights, SwigluMlp, SwigluWeights};
use modgrad_transformer::model::GptModel;
use modgrad_transformer::norm::ScaledRmsNorm;
use modgrad_transformer::position::fixed::FixedPositioning;
use modgrad_transformer::resident::GptModelResident;
use modgrad_transformer::residual::ResidualLambdas;
use modgrad_transformer::rope::RotaryEmbedding;
use modgrad_transformer::smear::{Inference, Smear, SmearWeights, Training};
use modgrad_transformer::tensor::Tensor2;

use crate::backend::BoxErr;
use crate::safetensors::SafetensorsFile;

/// Qwen2.5-0.5B architectural constants. Confirmed from
/// `config.json` in the HF snapshot.
pub const QWEN2_5_0_5B_NUM_LAYERS: usize = 24;
pub const QWEN2_5_0_5B_HIDDEN_SIZE: usize = 896;
pub const QWEN2_5_0_5B_NUM_HEADS: usize = 14;
pub const QWEN2_5_0_5B_NUM_KV_HEADS: usize = 2;
pub const QWEN2_5_0_5B_HEAD_DIM: usize = 64; // hidden / num_heads
pub const QWEN2_5_0_5B_MLP_DIM: usize = 4864;
pub const QWEN2_5_0_5B_VOCAB: usize = 151_936;
pub const QWEN2_5_0_5B_RMS_EPS: f32 = 1e-6;
pub const QWEN2_5_0_5B_ROPE_BASE: f32 = 1_000_000.0;

/// Construct a [`GptConfig`] tuned for Qwen2.5-0.5B. `max_seq` is the
/// runtime KV cache cap — Qwen2's training context is 32_768, but the
/// resident KV cache scales with `max_seq * n_kv_heads * head_dim` so
/// for a one-shot smoke test 2_048 is plenty.
pub fn qwen2_5_0_5b_config(max_seq: usize) -> GptConfig {
    GptConfig {
        model_dim: ModelDim::new(QWEN2_5_0_5B_HIDDEN_SIZE),
        num_heads: NumHeads::new(QWEN2_5_0_5B_NUM_HEADS),
        num_kv_heads: NumKvHeads::new(QWEN2_5_0_5B_NUM_KV_HEADS),
        head_dim: HeadDim::new(QWEN2_5_0_5B_HEAD_DIM),
        num_layers: NumLayers::new(QWEN2_5_0_5B_NUM_LAYERS),
        vocab_size: VocabSize::new(QWEN2_5_0_5B_VOCAB),
        mlp_dim: MlpDim::new(QWEN2_5_0_5B_MLP_DIM),
        max_seq_len: SeqLen::new(max_seq),
        rope_base: QWEN2_5_0_5B_ROPE_BASE,
        // Qwen2 has no QK norm — disable it via `use_qk_norm: false`.
        // The `qk_norm_scale` is then unused (kept at default 1.0).
        qk_norm_scale: 1.0,
        use_qk_norm: false,
        window_pattern: WindowPattern::Full,
        mlp_activation: MlpActivation::SwiGlu,
        layer_overrides: Vec::new(),
        tie_embeddings: true,
        logit_cap: 0.0,
        recurrent_steps: 1,
        has_exit_gate: false,
        // Bypass smear/value-embed/x0/backout — Qwen2 has none.
        value_embed: ValueEmbedConfig::default(),
        residual: ResidualConfig {
            resid_start: 1.0,
            resid_end: 1.0,
            x0_start: 0.0,
            x0_end: 0.0,
            backout_lambda: 0.0,
        },
        smear: SmearConfig::default(),
        precision: Precision::F32,
        norm_eps: QWEN2_5_0_5B_RMS_EPS,
    }
}

/// Load Qwen2.5-0.5B safetensors → [`GptModelResident`].
///
/// `safetensors_path` points at `model.safetensors` in the HF snapshot.
/// `max_seq` sets the KV cache capacity (the resident KV cache itself
/// is allocated by the caller — see `examples/qwen_load_smoke`).
///
/// On success the host weights are dropped before returning; only the
/// device buffers persist (~2.5 GiB for the 0.5B model in fp32).
pub fn load_qwen2_5_0_5b(
    safetensors_path: &str,
    max_seq: usize,
) -> Result<GptModelResident, BoxErr> {
    let started = Instant::now();
    eprintln!("qwen2_5_0_5b: opening {safetensors_path}");
    let mut sf = SafetensorsFile::load(safetensors_path)?;
    eprintln!(
        "qwen2_5_0_5b: header parsed ({} tensors, {} ms)",
        sf.len(),
        started.elapsed().as_millis()
    );

    let config = qwen2_5_0_5b_config(max_seq);
    config
        .validate()
        .map_err(|e| format!("qwen2 config validate: {e:?}"))?;
    let n_layers = config.num_layers.get();
    let model_dim = config.model_dim.get();
    let kv_dim = config.num_kv_heads.get() * config.head_dim.get();
    let mlp_dim = config.mlp_dim.get();
    let vocab = config.vocab_size.get();

    // ── Embedding (also serves as lm_head when tied) ──────────
    eprintln!("qwen2_5_0_5b: reading embed_tokens.weight ...");
    let embed_data = sf.read_tensor("model.embed_tokens.weight")?;
    sanity_check("model.embed_tokens.weight", &embed_data, vocab * model_dim)?;

    // ── Final RMSNorm scale ───────────────────────────────────
    let final_norm_scale = sf.read_tensor("model.norm.weight")?;
    sanity_check("model.norm.weight", &final_norm_scale, model_dim)?;

    // ── Per-layer attention/MLP weights + biases + norm scales ─
    //
    // We construct the host `GptModel` directly (bypassing
    // `TransformerBuilder` and its `GptWeights::validate`) because
    // the validator's `has_value_embed` heuristic insists on
    // `[vocab, kv_dim]` VE tables for half the layers — Qwen has no
    // value embeddings, and allocating ~78 MiB per VE layer × 12
    // layers (~940 MiB) just to satisfy a check we don't need is
    // wasteful.  The resident path skips smear / VE / x0 / backout
    // by design, so the `Smear`, `Mlp` (placeholder ReLU²), and
    // `value_embed` fields below carry the smallest-possible legal
    // payloads.
    let mut blocks_host: Vec<TransformerBlock> = Vec::with_capacity(n_layers);
    let mut swiglu_mlps: Vec<SwigluMlp> = Vec::with_capacity(n_layers);
    let mut q_biases: Vec<Vec<f32>> = Vec::with_capacity(n_layers);
    let mut k_biases: Vec<Vec<f32>> = Vec::with_capacity(n_layers);
    let mut v_biases: Vec<Vec<f32>> = Vec::with_capacity(n_layers);
    let mut attn_norm_scales: Vec<Vec<f32>> = Vec::with_capacity(n_layers);
    let mut mlp_norm_scales: Vec<Vec<f32>> = Vec::with_capacity(n_layers);

    for li in 0..n_layers {
        if li == 0 || li == n_layers - 1 || li == n_layers / 2 {
            eprintln!("qwen2_5_0_5b: reading layer {li}/{}", n_layers - 1);
        }
        let prefix = format!("model.layers.{li}");

        let wq = sf.read_tensor(&format!("{prefix}.self_attn.q_proj.weight"))?;
        sanity_check("q_proj.weight", &wq, model_dim * model_dim)?;
        let wk = sf.read_tensor(&format!("{prefix}.self_attn.k_proj.weight"))?;
        sanity_check("k_proj.weight", &wk, kv_dim * model_dim)?;
        let wv = sf.read_tensor(&format!("{prefix}.self_attn.v_proj.weight"))?;
        sanity_check("v_proj.weight", &wv, kv_dim * model_dim)?;
        let wo = sf.read_tensor(&format!("{prefix}.self_attn.o_proj.weight"))?;
        sanity_check("o_proj.weight", &wo, model_dim * model_dim)?;

        let bq = sf.read_tensor(&format!("{prefix}.self_attn.q_proj.bias"))?;
        sanity_check("q_proj.bias", &bq, model_dim)?;
        let bk = sf.read_tensor(&format!("{prefix}.self_attn.k_proj.bias"))?;
        sanity_check("k_proj.bias", &bk, kv_dim)?;
        let bv = sf.read_tensor(&format!("{prefix}.self_attn.v_proj.bias"))?;
        sanity_check("v_proj.bias", &bv, kv_dim)?;

        let mlp_gate = sf.read_tensor(&format!("{prefix}.mlp.gate_proj.weight"))?;
        sanity_check("gate_proj.weight", &mlp_gate, mlp_dim * model_dim)?;
        let mlp_up = sf.read_tensor(&format!("{prefix}.mlp.up_proj.weight"))?;
        sanity_check("up_proj.weight", &mlp_up, mlp_dim * model_dim)?;
        let mlp_down = sf.read_tensor(&format!("{prefix}.mlp.down_proj.weight"))?;
        sanity_check("down_proj.weight", &mlp_down, model_dim * mlp_dim)?;

        let attn_norm_w = sf.read_tensor(&format!("{prefix}.input_layernorm.weight"))?;
        sanity_check("input_layernorm.weight", &attn_norm_w, model_dim)?;
        let mlp_norm_w =
            sf.read_tensor(&format!("{prefix}.post_attention_layernorm.weight"))?;
        sanity_check("post_attention_layernorm.weight", &mlp_norm_w, model_dim)?;

        let attn_weights = AttentionWeights {
            wq: Tensor2::new(wq, model_dim, model_dim)
                .ok_or_else(|| format!("wq shape layer {li}"))?,
            wk: Tensor2::new(wk, kv_dim, model_dim)
                .ok_or_else(|| format!("wk shape layer {li}"))?,
            wv: Tensor2::new(wv, kv_dim, model_dim)
                .ok_or_else(|| format!("wv shape layer {li}"))?,
            wo: Tensor2::new(wo, model_dim, model_dim)
                .ok_or_else(|| format!("wo shape layer {li}"))?,
        };
        let attn = CausalSelfAttention::new(attn_weights, &config);

        // Placeholder ReLU² MLP (the resident path uses the parallel
        // SwiGLU below).  Zeros are fine — host blocks never run.
        let placeholder_mlp = Mlp::new(
            MlpWeights {
                fc: Tensor2::zeros(mlp_dim, model_dim),
                proj: Tensor2::zeros(model_dim, mlp_dim),
            },
            config.model_dim,
            config.mlp_dim,
        );

        let layer_idx = LayerIdx::new(li, config.num_layers)
            .ok_or_else(|| format!("layer idx {li}/{n_layers} out of range"))?;
        blocks_host.push(TransformerBlock::new(
            attn,
            placeholder_mlp,
            None, // no value embedding
            layer_idx,
            &config,
        ));

        let swiglu_w = SwigluWeights {
            gate: Tensor2::new(mlp_gate, mlp_dim, model_dim)
                .ok_or_else(|| format!("swiglu gate shape layer {li}"))?,
            up: Tensor2::new(mlp_up, mlp_dim, model_dim)
                .ok_or_else(|| format!("swiglu up shape layer {li}"))?,
            down: Tensor2::new(mlp_down, model_dim, mlp_dim)
                .ok_or_else(|| format!("swiglu down shape layer {li}"))?,
        };
        swiglu_mlps.push(SwigluMlp::new(
            swiglu_w,
            config.model_dim,
            config.mlp_dim,
        ));

        q_biases.push(bq);
        k_biases.push(bk);
        v_biases.push(bv);
        attn_norm_scales.push(attn_norm_w);
        mlp_norm_scales.push(mlp_norm_w);
    }

    eprintln!(
        "qwen2_5_0_5b: weights loaded ({} ms total)",
        started.elapsed().as_millis()
    );

    // ── Build the host GptModel directly (no validator) ───────
    // Tied embeddings: lm_head reuses the embed buffer.  Same byte
    // pattern, separate Vec allocations on host (the device only sees
    // one upload of each — `embed_dev` and `lm_head.weight_dev`, which
    // are separate device buffers but with identical contents).
    let lm_head = embed_data.clone();
    let smear_gate = vec![0.0f32; model_dim * config.smear.gate_channels];
    let smear_inference = Smear::<Inference>::new(SmearWeights::new(
        smear_gate.clone(),
        config.model_dim,
        &config.smear,
    ));
    let smear_training = Smear::<Training>::new(SmearWeights::new(
        smear_gate,
        config.model_dim,
        &config.smear,
    ));
    let lambdas = ResidualLambdas::from_config(&config.residual, config.num_layers);
    let rope = RotaryEmbedding::new(config.head_dim, config.max_seq_len, config.rope_base);

    let model = GptModel {
        embed: Tensor2::new(embed_data, vocab, model_dim)
            .ok_or_else(|| "embed tensor shape".to_string())?,
        lm_head: Tensor2::new(lm_head, vocab, model_dim)
            .ok_or_else(|| "lm_head tensor shape".to_string())?,
        final_norm: ScaledRmsNorm::new(final_norm_scale, config.norm_eps),
        smear_inference,
        smear_training,
        blocks: blocks_host,
        lambdas,
        rope,
        position: Box::new(FixedPositioning),
        config: config.clone(),
    };

    // ── Upload to device ──────────────────────────────────────
    eprintln!("qwen2_5_0_5b: uploading to GPU ...");
    let upload_start = Instant::now();
    let mut resident = GptModelResident::from_model(&model, &swiglu_mlps)
        .map_err(|e| format!("resident upload: {e:?}"))?;
    eprintln!(
        "qwen2_5_0_5b: GPU upload done ({} ms)",
        upload_start.elapsed().as_millis()
    );

    // ── Patch in Qwen-specific bits the default resident builder zeroed ──
    //
    // 1. q/k/v_proj biases — the default `linear_from_weight` makes
    //    bias = 0; we re-upload the real tensors directly into the
    //    HipBuffer that backs `bias_dev`. This is a non-mutating side-
    //    channel update; matvec_resident reads bias_dev on every call.
    // 2. attn_norm_weight_dev / mlp_norm_weight_dev — likewise, the
    //    default builder uploads `[1.0; model_dim]` (the host RmsNorm
    //    has no learnable scale). Qwen has per-block input/post norms,
    //    so we re-upload here.
    for (li, block) in resident.blocks.iter_mut().enumerate() {
        block.attn.q_proj.bias_dev.copy_from_host(&q_biases[li])
            .map_err(|e| format!("upload q_proj.bias layer {li}: {e:?}"))?;
        block.attn.k_proj.bias_dev.copy_from_host(&k_biases[li])
            .map_err(|e| format!("upload k_proj.bias layer {li}: {e:?}"))?;
        block.attn.v_proj.bias_dev.copy_from_host(&v_biases[li])
            .map_err(|e| format!("upload v_proj.bias layer {li}: {e:?}"))?;
        block.attn_norm_weight_dev.copy_from_host(&attn_norm_scales[li])
            .map_err(|e| format!("upload attn_norm layer {li}: {e:?}"))?;
        block.mlp_norm_weight_dev.copy_from_host(&mlp_norm_scales[li])
            .map_err(|e| format!("upload mlp_norm layer {li}: {e:?}"))?;
    }

    // Drop the host model now — the device has its own copy and the
    // host f32 weights are ~1.9 GiB. Keeping them through generation
    // would double the working set unnecessarily.
    drop(model);
    drop(swiglu_mlps);

    eprintln!(
        "qwen2_5_0_5b: total load+upload+patch took {} ms",
        started.elapsed().as_millis()
    );

    Ok(resident)
}

fn sanity_check(name: &str, t: &[f32], expected: usize) -> Result<(), BoxErr> {
    if t.len() != expected {
        return Err(format!(
            "tensor `{name}`: numel {} ≠ expected {expected}",
            t.len()
        )
        .into());
    }
    if t.iter().any(|v| !v.is_finite()) {
        return Err(format!("tensor `{name}`: contains non-finite values").into());
    }
    Ok(())
}

