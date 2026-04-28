//! Minimal frozen transformer forward pass for cerebellum use.
//!
//! Loads weights from safetensors and runs a Qwen2/Llama-style transformer
//! using our own compute primitives. No external inference runtime needed.
//!
//! Architecture: embed → [RMSNorm → GQA → RMSNorm → SwiGLU] × N → RMSNorm
//!
//! All operations use the SDK's Linear/matvec — same compute backend
//! that handles CTM training. GPU dispatch works automatically.

use serde::{Deserialize, Serialize};

// ─── Config ────────────────────────────────────────────────

/// Configuration for a frozen transformer (Qwen2/Llama family).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerConfig {
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub intermediate_dim: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
}

impl TransformerConfig {
    /// Qwen2.5-0.5B defaults.
    pub fn qwen2_0_5b() -> Self {
        Self {
            hidden_dim: 896,
            n_layers: 24,
            n_heads: 14,
            n_kv_heads: 2,
            intermediate_dim: 4864,
            vocab_size: 151936,
            max_seq_len: 32768,
            rms_norm_eps: 1e-6,
            rope_theta: 1000000.0,
        }
    }
}

// ─── Weights ───────────────────────────────────────────────

/// Weights for one transformer layer.
pub struct TransformerLayer {
    // Attention
    pub q_proj_w: Vec<f32>,    // [n_heads * head_dim, hidden_dim]
    pub q_proj_b: Vec<f32>,    // [n_heads * head_dim]
    pub k_proj_w: Vec<f32>,    // [n_kv_heads * head_dim, hidden_dim]
    pub k_proj_b: Vec<f32>,    // [n_kv_heads * head_dim]
    pub v_proj_w: Vec<f32>,    // [n_kv_heads * head_dim, hidden_dim]
    pub v_proj_b: Vec<f32>,    // [n_kv_heads * head_dim]
    pub o_proj_w: Vec<f32>,    // [hidden_dim, hidden_dim]
    // MLP (SwiGLU)
    pub gate_proj_w: Vec<f32>, // [intermediate, hidden_dim]
    pub up_proj_w: Vec<f32>,   // [intermediate, hidden_dim]
    pub down_proj_w: Vec<f32>, // [hidden_dim, intermediate]
    // Norms
    pub input_norm_w: Vec<f32>,  // [hidden_dim]
    pub post_attn_norm_w: Vec<f32>, // [hidden_dim]
}

/// Full frozen transformer weights.
pub struct FrozenTransformer {
    pub config: TransformerConfig,
    pub embed_w: Vec<f32>,      // [vocab_size, hidden_dim]
    pub layers: Vec<TransformerLayer>,
    pub final_norm_w: Vec<f32>, // [hidden_dim]
    // Pre-computed RoPE frequencies
    rope_cos: Vec<f32>,         // [max_seq, head_dim/2]
    rope_sin: Vec<f32>,         // [max_seq, head_dim/2]
}

impl FrozenTransformer {
    /// Load from safetensors file.
    pub fn load(path: &str, config: TransformerConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read(path)?;
        let tensors = parse_safetensors(&data)?;

        let hd = config.hidden_dim;
        let head_dim = hd / config.n_heads;
        let kv_dim = config.n_kv_heads * head_dim;
        let inter = config.intermediate_dim;

        // Embedding
        let embed_w = get_tensor_f32(&tensors, &data, "model.embed_tokens.weight",
            config.vocab_size * hd)?;

        // Layers
        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let prefix = format!("model.layers.{i}");
            let layer = TransformerLayer {
                q_proj_w: get_tensor_f32(&tensors, &data, &format!("{prefix}.self_attn.q_proj.weight"), hd * hd)?,
                q_proj_b: get_tensor_f32(&tensors, &data, &format!("{prefix}.self_attn.q_proj.bias"), hd)?,
                k_proj_w: get_tensor_f32(&tensors, &data, &format!("{prefix}.self_attn.k_proj.weight"), kv_dim * hd)?,
                k_proj_b: get_tensor_f32(&tensors, &data, &format!("{prefix}.self_attn.k_proj.bias"), kv_dim)?,
                v_proj_w: get_tensor_f32(&tensors, &data, &format!("{prefix}.self_attn.v_proj.weight"), kv_dim * hd)?,
                v_proj_b: get_tensor_f32(&tensors, &data, &format!("{prefix}.self_attn.v_proj.bias"), kv_dim)?,
                o_proj_w: get_tensor_f32(&tensors, &data, &format!("{prefix}.self_attn.o_proj.weight"), hd * hd)?,
                gate_proj_w: get_tensor_f32(&tensors, &data, &format!("{prefix}.mlp.gate_proj.weight"), inter * hd)?,
                up_proj_w: get_tensor_f32(&tensors, &data, &format!("{prefix}.mlp.up_proj.weight"), inter * hd)?,
                down_proj_w: get_tensor_f32(&tensors, &data, &format!("{prefix}.mlp.down_proj.weight"), hd * inter)?,
                input_norm_w: get_tensor_f32(&tensors, &data, &format!("{prefix}.input_layernorm.weight"), hd)?,
                post_attn_norm_w: get_tensor_f32(&tensors, &data, &format!("{prefix}.post_attention_layernorm.weight"), hd)?,
            };
            layers.push(layer);
        }

        let final_norm_w = get_tensor_f32(&tensors, &data, "model.norm.weight", hd)?;

        // Pre-compute RoPE
        let max_seq = config.max_seq_len.min(4096); // cap for memory
        let half_head = head_dim / 2;
        let mut rope_cos = vec![0.0f32; max_seq * half_head];
        let mut rope_sin = vec![0.0f32; max_seq * half_head];
        for pos in 0..max_seq {
            for i in 0..half_head {
                let freq = 1.0 / config.rope_theta.powf(2.0 * i as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                rope_cos[pos * half_head + i] = angle.cos();
                rope_sin[pos * half_head + i] = angle.sin();
            }
        }

        eprintln!("Loaded transformer: {} layers, hidden_dim={}, params={}",
            config.n_layers, hd,
            embed_w.len() + layers.iter().map(|l| layer_params(l)).sum::<usize>() + final_norm_w.len());

        Ok(Self { config, embed_w, layers, final_norm_w, rope_cos, rope_sin })
    }

    /// Forward pass: token_ids → hidden states per layer.
    /// Returns one Vec<f32> per layer, each [seq_len × hidden_dim] contiguous.
    pub fn forward_tokens(&self, token_ids: &[usize]) -> Vec<Vec<f32>> {
        let cfg = &self.config;
        let hd = cfg.hidden_dim;
        let head_dim = hd / cfg.n_heads;
        let kv_dim = cfg.n_kv_heads * head_dim;
        let seq_len = token_ids.len();

        // Embed
        let mut hidden = vec![0.0f32; seq_len * hd];
        for (t, &tid) in token_ids.iter().enumerate() {
            let src = tid * hd;
            if src + hd <= self.embed_w.len() {
                hidden[t * hd..(t + 1) * hd].copy_from_slice(&self.embed_w[src..src + hd]);
            }
        }

        // Collect per-layer outputs for multi-layer blending
        let mut layer_outputs: Vec<Vec<f32>> = Vec::with_capacity(cfg.n_layers);

        for layer in &self.layers {
            // RMSNorm
            let normed = rms_norm(&hidden, &layer.input_norm_w, cfg.rms_norm_eps, seq_len, hd);

            // Q, K, V projections
            let q = matvec_bias_batch(&layer.q_proj_w, &layer.q_proj_b, &normed, hd, hd, seq_len);
            let k = matvec_bias_batch(&layer.k_proj_w, &layer.k_proj_b, &normed, kv_dim, hd, seq_len);
            let v = matvec_bias_batch(&layer.v_proj_w, &layer.v_proj_b, &normed, kv_dim, hd, seq_len);

            // Apply RoPE to Q and K
            let q = apply_rope(&q, &self.rope_cos, &self.rope_sin, seq_len, cfg.n_heads, head_dim);
            let k = apply_rope(&k, &self.rope_cos, &self.rope_sin, seq_len, cfg.n_kv_heads, head_dim);

            // GQA attention
            let attn_out = gqa_attention(&q, &k, &v, seq_len, cfg.n_heads, cfg.n_kv_heads, head_dim);

            // O projection
            let projected = matvec_batch(&layer.o_proj_w, &attn_out, hd, hd, seq_len);

            // Residual
            for i in 0..hidden.len() { hidden[i] += projected[i]; }

            // Post-attention norm
            let normed2 = rms_norm(&hidden, &layer.post_attn_norm_w, cfg.rms_norm_eps, seq_len, hd);

            // SwiGLU MLP: gate = SiLU(gate_proj(x)), up = up_proj(x),
            //             out = down_proj(gate * up).
            // SiLU dispatches through `ops::silu_fwd_inplace`; the
            // elementwise `silu * up` stays as an inline scalar loop
            // because the registry has no fused "multiply two tensors"
            // op and adding one for a single call site would be
            // premature (no other caller would share it).
            let mut gate = matvec_batch(&layer.gate_proj_w, &normed2, cfg.intermediate_dim, hd, seq_len);
            let up = matvec_batch(&layer.up_proj_w, &normed2, cfg.intermediate_dim, hd, seq_len);
            modgrad_device::backend::ops::silu_fwd_inplace(&mut gate)
                .expect("SwiGLU SiLU dispatch");
            for i in 0..gate.len() { gate[i] *= up[i]; }
            let gated = gate;
            let mlp_out = matvec_batch(&layer.down_proj_w, &gated, hd, cfg.intermediate_dim, seq_len);

            // Residual
            for i in 0..hidden.len() { hidden[i] += mlp_out[i]; }

            // Store this layer's output
            layer_outputs.push(hidden.clone());
        }

        // Final RMSNorm on last hidden state (stored as the last layer output)
        if let Some(last) = layer_outputs.last_mut() {
            *last = rms_norm(last, &self.final_norm_w, cfg.rms_norm_eps, seq_len, hd);
        }

        layer_outputs
    }
}

// ─── FrozenCerebellum impl ─────────────────────────────────

impl crate::cerebellum::FrozenCerebellum for FrozenTransformer {
    fn hidden_dim(&self) -> usize { self.config.hidden_dim }
    fn n_layers(&self) -> usize { self.config.n_layers }

    fn encode_context_layers(&mut self, token_ids: &[i64]) -> crate::cerebellum::CerebellumCache {
        let ids: Vec<usize> = token_ids.iter().map(|&t| t as usize).collect();
        let seq_len = ids.len();
        let d = self.config.hidden_dim;
        let layer_outputs = self.forward_tokens(&ids);
        let n_layers = layer_outputs.len();

        // Pack into contiguous [n_layers × n_positions × hidden_dim]
        let mut hidden_states = Vec::with_capacity(n_layers * seq_len * d);
        for layer_out in &layer_outputs {
            hidden_states.extend_from_slice(layer_out);
        }

        crate::cerebellum::CerebellumCache {
            hidden_states,
            hidden_dim: d,
            n_positions: seq_len,
            n_layers,
            modalities: None,
        }
    }

    fn forward(&mut self, _input: &[f32]) -> Vec<f32> {
        vec![0.0; self.config.hidden_dim]
    }
}

// ─── Math primitives ───────────────────────────────────────

fn rms_norm(x: &[f32], w: &[f32], eps: f32, seq_len: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; seq_len * dim];
    for t in 0..seq_len {
        let offset = t * dim;
        let slice = &x[offset..offset + dim];
        let rms = (slice.iter().map(|&v| v * v).sum::<f32>() / dim as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;
        for i in 0..dim {
            out[offset + i] = slice[i] * inv_rms * w[i];
        }
    }
    out
}

/// Batched linear: `out[t, :] = W @ x[t, :]` for each token t.
///
/// Equivalent to one GEMM: `out = X @ W^T` where X is [seq_len, in_dim],
/// W is [out_dim, in_dim] (row-major), and out is [seq_len, out_dim].
/// Dispatched through `ops::matmul_nt` so every Q/K/V/O/gate/up/down
/// projection in the transformer stack hits the Backend registry as
/// one op per layer per projection, not seq_len separate matvecs.
fn matvec_batch(w: &[f32], x: &[f32], out_dim: usize, in_dim: usize, seq_len: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; seq_len * out_dim];
    modgrad_device::backend::ops::matmul_nt(
        x, w, &mut out, None,
        seq_len, in_dim, out_dim,
    ).expect("matvec_batch: matmul_nt dispatch");
    out
}

/// Same as `matvec_batch` but with an added bias per output unit,
/// broadcast across tokens. `ops::matmul_nt` accepts an optional
/// bias so we get a single dispatched op, no extra pass.
fn matvec_bias_batch(w: &[f32], b: &[f32], x: &[f32], out_dim: usize, in_dim: usize, seq_len: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; seq_len * out_dim];
    modgrad_device::backend::ops::matmul_nt(
        x, w, &mut out, Some(b),
        seq_len, in_dim, out_dim,
    ).expect("matvec_bias_batch: matmul_nt dispatch");
    out
}

fn apply_rope(x: &[f32], cos: &[f32], sin: &[f32], seq_len: usize, n_heads: usize, head_dim: usize) -> Vec<f32> {
    let half = head_dim / 2;
    let total_dim = n_heads * head_dim;
    let mut out = x.to_vec();
    for t in 0..seq_len {
        for h in 0..n_heads {
            let base = t * total_dim + h * head_dim;
            for i in 0..half {
                let cos_val = cos[t * half + i];
                let sin_val = sin[t * half + i];
                let x0 = x[base + i];
                let x1 = x[base + half + i];
                out[base + i] = x0 * cos_val - x1 * sin_val;
                out[base + half + i] = x0 * sin_val + x1 * cos_val;
            }
        }
    }
    out
}

fn gqa_attention(q: &[f32], k: &[f32], v: &[f32], seq_len: usize,
                 n_heads: usize, n_kv_heads: usize, head_dim: usize) -> Vec<f32> {
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let heads_per_kv = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut out = vec![0.0f32; seq_len * q_dim];

    for h in 0..n_heads {
        let kv_h = h / heads_per_kv;
        for t in 0..seq_len {
            // Compute attention scores for this head at position t
            let q_off = t * q_dim + h * head_dim;
            let mut scores = vec![0.0f32; t + 1]; // causal: only attend to 0..=t

            for s in 0..=t {
                let k_off = s * kv_dim + kv_h * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[q_off + d] * k[k_off + d];
                }
                scores[s] = dot * scale;
            }

            // Softmax
            let max_s = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp()).collect();
            let sum: f32 = exp_scores.iter().sum::<f32>().max(1e-8);
            for s in &mut exp_scores { *s /= sum; }

            // Weighted sum of values
            let o_off = t * q_dim + h * head_dim;
            for s in 0..=t {
                let v_off = s * kv_dim + kv_h * head_dim;
                for d in 0..head_dim {
                    out[o_off + d] += exp_scores[s] * v[v_off + d];
                }
            }
        }
    }
    out
}

fn layer_params(l: &TransformerLayer) -> usize {
    l.q_proj_w.len() + l.q_proj_b.len() + l.k_proj_w.len() + l.k_proj_b.len()
        + l.v_proj_w.len() + l.v_proj_b.len() + l.o_proj_w.len()
        + l.gate_proj_w.len() + l.up_proj_w.len() + l.down_proj_w.len()
        + l.input_norm_w.len() + l.post_attn_norm_w.len()
}

// ─── Safetensors parser ────────────────────────────────────

use std::collections::HashMap;

#[derive(Debug)]
struct TensorInfo {
    dtype: String,
    #[allow(dead_code)] // populated from safetensors header; retained for debug output and potential validation
    shape: Vec<usize>,
    data_offsets: (usize, usize),
}

fn parse_safetensors(data: &[u8]) -> Result<HashMap<String, TensorInfo>, Box<dyn std::error::Error>> {
    if data.len() < 8 {
        return Err("file too small".into());
    }
    let header_len = u64::from_le_bytes(data[0..8].try_into()?) as usize;
    let header_end = 8 + header_len;
    if header_end > data.len() {
        return Err("header length exceeds file size".into());
    }

    let header_str = std::str::from_utf8(&data[8..header_end])?;
    let header: serde_json::Value = serde_json::from_str(header_str)?;

    let mut tensors = HashMap::new();
    if let serde_json::Value::Object(map) = header {
        for (name, info) in map {
            if name == "__metadata__" { continue; }
            if let serde_json::Value::Object(ref tinfo) = info {
                let dtype = tinfo.get("dtype")
                    .and_then(|v| v.as_str())
                    .unwrap_or("F32")
                    .to_string();
                let shape: Vec<usize> = tinfo.get("shape")
                    .and_then(|v| v.as_array())
                    .map(|a| a.iter().filter_map(|v| v.as_u64().map(|u| u as usize)).collect())
                    .unwrap_or_default();
                let offsets = tinfo.get("data_offsets")
                    .and_then(|v| v.as_array())
                    .and_then(|a| {
                        if a.len() == 2 {
                            Some((a[0].as_u64()? as usize, a[1].as_u64()? as usize))
                        } else { None }
                    })
                    .unwrap_or((0, 0));

                tensors.insert(name, TensorInfo {
                    dtype, shape, data_offsets: offsets,
                });
            }
        }
    }
    Ok(tensors)
}

fn get_tensor_f32(
    tensors: &HashMap<String, TensorInfo>,
    data: &[u8],
    name: &str,
    _expected_len: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let info = tensors.get(name)
        .ok_or_else(|| format!("tensor '{}' not found", name))?;

    let header_len = u64::from_le_bytes(data[0..8].try_into()?) as usize;
    let data_start = 8 + header_len;
    let (off_start, off_end) = info.data_offsets;
    let tensor_data = &data[data_start + off_start..data_start + off_end];

    match info.dtype.as_str() {
        "F32" => {
            let floats: Vec<f32> = tensor_data.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            Ok(floats)
        }
        "F16" => {
            let floats: Vec<f32> = tensor_data.chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes(c.try_into().unwrap());
                    half::f16::from_bits(bits).to_f32()
                })
                .collect();
            Ok(floats)
        }
        "BF16" => {
            let floats: Vec<f32> = tensor_data.chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes(c.try_into().unwrap());
                    half::bf16::from_bits(bits).to_f32()
                })
                .collect();
            Ok(floats)
        }
        other => Err(format!("unsupported dtype '{}' for tensor '{}'", other, name).into()),
    }
}

// ─── Tests ─────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_unit() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![1.0; 4];
        let out = rms_norm(&x, &w, 1e-6, 1, 4);
        // RMS = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.738
        let rms = (30.0f32 / 4.0).sqrt();
        assert!((out[0] - 1.0 / rms).abs() < 1e-5);
    }

    #[test]
    fn rope_preserves_norm() {
        let x = vec![1.0; 128]; // 2 heads × 64 head_dim
        let cos = vec![1.0; 32]; // pos=0, all cos=1
        let sin = vec![0.0; 32]; // pos=0, all sin=0
        let out = apply_rope(&x, &cos, &sin, 1, 2, 64);
        // cos=1, sin=0 → identity
        assert!((out[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn config_qwen() {
        let cfg = TransformerConfig::qwen2_0_5b();
        assert_eq!(cfg.hidden_dim, 896);
        assert_eq!(cfg.n_layers, 24);
        assert_eq!(cfg.n_heads, 14);
        assert_eq!(cfg.n_kv_heads, 2);
    }
}
