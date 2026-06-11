//! Fully GPU-resident Gemma-4 inference on the ROCm/HIP runtime.
//!
//! Every weight lives in VRAM (Q4_K/Q5_K/Q6_K + f32 norms), the activation
//! never leaves VRAM during a token, and every op is a HIP kernel chained on
//! the default stream — one `hip_sync` per token, zero PCIe bounce. This is
//! the stable replacement for the KFD engine (which crashed Xorg).
//!
//! The forward math mirrors the validated CPU `kfd::inference::Gemma4Model`
//! (NEOX RoPE, 1/sqrt(head_dim) attention scale, V=K on global layers,
//! per-layer rope base, sandwich norms, logit softcap).

#![cfg(all(feature = "rocm", modgrad_hipcc_kernels))]

use std::collections::HashMap;
use crate::backend::rocm::{HipBuffer, kern};
use crate::backend::BackendError;
use crate::kfd::gguf::{GgufFile, GgmlType, dequantize_row_q6_k};

/// Apply a repetition penalty to the logits of tokens seen in the last `last_n`
/// of `hist` (deduped — once per distinct token), matching llama.cpp/ollama.
fn repeat_penalty(logits: &mut [f32], hist: &[u32], last_n: usize, penalty: f32) {
    if penalty == 1.0 { return; }
    let start = hist.len().saturating_sub(last_n);
    let mut seen = std::collections::HashSet::new();
    for &t in &hist[start..] {
        if !seen.insert(t) { continue; }
        if let Some(l) = logits.get_mut(t as usize) {
            *l = if *l > 0.0 { *l / penalty } else { *l * penalty };
        }
    }
}

/// argmax over logits, NaN-safe (NaN never wins).
fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv { bv = x; best = i; }
    }
    best as u32
}

/// Per-layer derived metadata (from tensor shapes, like the CPU engine).
struct Layer {
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    q_dim: usize,
    kv_dim: usize,
    is_local: bool,    // has attn_v → sliding-window local layer
    rope_base: f32,
}

pub struct RocmGemma {
    n_layers: usize,
    d_model: usize,
    d_ff: usize,
    vocab: usize,
    eps: f32,
    sliding_window: usize,
    logit_cap: f32,
    layers: Vec<Layer>,
    out_scale: Vec<f32>,

    // ALL quantized layer weights packed into ONE allocation (less VRAM
    // fragmentation than hundreds of hipMallocs); qoff = byte offset per tensor.
    qbase: HipBuffer,
    qoff: HashMap<String, usize>,
    qdt: HashMap<String, GgmlType>,   // actual dtype per quant weight (varies per layer!)
    nw: HashMap<String, HipBuffer>,
    embd_bytes: Vec<u8>,          // host Q6_K token_embd (embedding row; CPU lm_head fallback)
    token_embd: Option<HipBuffer>, // resident lm_head (GPU) if VRAM allows
    logits: Option<HipBuffer>,     // resident logits buffer (paired with token_embd)
    output_norm: HipBuffer,
    // Gemma4 "proportional" RoPE freq factors (rope_freqs.weight) — applied on
    // FULL-attention (global) layers only; zeroes rotation on the upper dims.
    rope_freqs: Option<HipBuffer>,

    // resident activations
    h: HipBuffer, normed: HipBuffer, q: HipBuffer, k: HipBuffer, v: HipBuffer,
    attn: HipBuffer, proj: HipBuffer, gate: HipBuffer, up: HipBuffer,
    glu: HipBuffer, ones: HipBuffer,
    // dp4a scratch: int8-quantized activation + per-32-block scale/sum
    xq: HipBuffer, xd: HipBuffer, xsum: HipBuffer,

    // KV cache (resident, per layer)
    kv_k: Vec<HipBuffer>, kv_v: Vec<HipBuffer>,
    kv_len: usize,
    max_seq: usize,
    use_dp4a: bool,
}

impl RocmGemma {
    pub fn load(gguf: &GgufFile, file: &[u8], max_seq: usize) -> Result<Self, String> {
        let arch = gguf.architecture().unwrap_or("gemma4").to_string();
        let mu = |k: &str| gguf.meta_u32(&format!("{arch}.{k}"));
        let mf = |k: &str| gguf.meta_f32(&format!("{arch}.{k}"));
        let n_layers = mu("block_count").unwrap_or(48) as usize;
        let d_model = mu("embedding_length").unwrap_or(3840) as usize;
        let d_ff = mu("feed_forward_length").unwrap_or(15360) as usize;
        let n_heads_meta = mu("attention.head_count").unwrap_or(16) as usize;
        let eps = mf("attention.layer_norm_rms_epsilon").unwrap_or(1e-6);
        let rope_base = mf("rope.freq_base").unwrap_or(1_000_000.0);
        let rope_base_swa = mf("rope.freq_base_swa").unwrap_or(10_000.0);
        let sliding_window = mu("attention.sliding_window").unwrap_or(1024) as usize;
        let logit_cap = mf("final_logit_softcapping").unwrap_or(0.0);
        let vocab = gguf.tensors.get("token_embd.weight")
            .and_then(|t| t.dims.get(1).copied()).unwrap_or(262144);

        // ── VRAM guard: never over-commit the display GPU (that resets it and
        // crashes Xorg). Estimate the resident footprint and bail cleanly if it
        // won't fit with headroom. token_embd stays host-only (CPU lm_head).
        let mut resident: usize = 0;
        for (name, info) in &gguf.tensors {
            if name == "token_embd.weight" { continue; }
            let n: usize = info.dims.iter().product();
            let (bb, be) = info.dtype.block_size();
            resident += if be == 0 { n * bb } else { n / be * bb };
        }
        for i in 0..n_layers {
            let kvd = gguf.tensors.get(&format!("blk.{i}.attn_k.weight"))
                .and_then(|t| t.dims.get(1).copied()).unwrap_or(2048);
            resident += max_seq * kvd * 4 * 2;
        }
        resident += d_model * 4 + d_ff * 4 * 3 + 8192 * 4 * 2 + 2048 * 4 * 3;
        // 1.5 GiB headroom: the amdgpu driver needs room for command-submission
        // buffers + the display. Too small a margin → "Not enough memory for
        // command submission" → GPU reset → Xorg crash (learned the hard way).
        let margin: usize = 1536 * 1024 * 1024;
        let (free, total) = crate::backend::rocm::kern::vram_free_total().map_err(|e| e.to_string())?;
        eprintln!("VRAM guard: need ~{:.2} GiB resident; free {:.2} / {:.2} GiB",
            resident as f64 / 1.073741824e9, free as f64 / 1.073741824e9, total as f64 / 1.073741824e9);
        if resident + margin > free {
            return Err(format!(
                "would over-commit VRAM: need {:.2} GiB + {:.2} GiB margin > {:.2} GiB free. \
                 Close GPU apps (chromium/steam/slack) or lower max_seq, then retry.",
                resident as f64 / 1.073741824e9, margin as f64 / 1.073741824e9, free as f64 / 1.073741824e9));
        }

        let dev_bytes = |name: &str| -> Result<HipBuffer, String> {
            let info = gguf.tensors.get(name).ok_or_else(|| format!("missing {name}"))?;
            let n: usize = info.dims.iter().product();
            let (bb, be) = info.dtype.block_size();
            let data_bytes = if be == 0 { n * bb } else { n / be * bb };
            let start = gguf.data_offset + info.offset;
            let buf = HipBuffer::new(data_bytes).map_err(|e| e.to_string())?;
            buf.copy_from_host_bytes(&file[start..start + data_bytes]).map_err(|e| e.to_string())?;
            Ok(buf)
        };
        let dev_f32 = |name: &str| -> Result<HipBuffer, String> {
            let info = gguf.tensors.get(name).ok_or_else(|| format!("missing {name}"))?;
            let n: usize = info.dims.iter().product();
            let start = gguf.data_offset + info.offset;
            let v: Vec<f32> = file[start..start + n * 4].chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            let buf = HipBuffer::new(n * 4).map_err(|e| e.to_string())?;
            buf.copy_from_host(&v).map_err(|e| e.to_string())?;
            Ok(buf)
        };
        let host_f32 = |name: &str| -> Vec<f32> {
            let info = &gguf.tensors[name];
            let n: usize = info.dims.iter().product();
            let start = gguf.data_offset + info.offset;
            file[start..start + n * 4].chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
        };

        let mut qnames: Vec<String> = Vec::new();
        let mut qdt: HashMap<String, GgmlType> = HashMap::new();
        let mut nw = HashMap::new();
        let mut layers = Vec::with_capacity(n_layers);
        let mut out_scale = Vec::with_capacity(n_layers);
        let mut max_q = d_model; let mut max_kv = 0usize;

        for i in 0..n_layers {
            let p = |s: &str| format!("blk.{i}.{s}.weight");
            let has = |s: &str| gguf.tensors.contains_key(&p(s));
            let head_dim = gguf.tensors[&p("attn_q_norm")].dims[0];
            let q_dim = gguf.tensors[&p("attn_q")].dims[1];
            let kv_dim = gguf.tensors[&p("attn_k")].dims[1];
            let is_local = has("attn_v");
            layers.push(Layer {
                head_dim,
                n_heads: q_dim / head_dim,
                n_kv_heads: kv_dim / head_dim,
                q_dim, kv_dim, is_local,
                rope_base: if is_local { rope_base_swa } else { rope_base },
            });
            max_q = max_q.max(q_dim); max_kv = max_kv.max(kv_dim);
            for t in ["attn_q", "attn_k", "attn_output", "ffn_gate", "ffn_up", "ffn_down"] {
                qnames.push(p(t));
                qdt.insert(p(t), gguf.tensors[&p(t)].dtype);
            }
            if is_local {
                qnames.push(p("attn_v"));
                qdt.insert(p("attn_v"), gguf.tensors[&p("attn_v")].dtype);
            }
            for t in ["attn_norm", "attn_q_norm", "attn_k_norm",
                      "post_attention_norm", "ffn_norm", "post_ffw_norm"] {
                nw.insert(p(t), dev_f32(&p(t))?);
            }
            out_scale.push(if has("layer_output_scale") { host_f32(&p("layer_output_scale"))[0] } else { 1.0 });
            let _ = n_heads_meta;
        }

        // Pack every quantized layer weight into ONE big VRAM allocation.
        let qbytes = |name: &str| -> usize {
            let info = &gguf.tensors[name];
            let n: usize = info.dims.iter().product();
            let (bb, be) = info.dtype.block_size();
            if be == 0 { n * bb } else { n / be * bb }
        };
        let mut qoff: HashMap<String, usize> = HashMap::new();
        let mut qtotal = 0usize;
        for name in &qnames { qoff.insert(name.clone(), qtotal); qtotal += qbytes(name); }
        let qbase = HipBuffer::new(qtotal).map_err(|e| e.to_string())?;
        for name in &qnames {
            let info = &gguf.tensors[name];
            let start = gguf.data_offset + info.offset;
            qbase.copy_from_host_bytes_at(qoff[name], &file[start..start + qbytes(name)])
                .map_err(|e| e.to_string())?;
        }
        eprintln!("    packed {} quant tensors into 1 alloc ({:.2} GiB)",
            qnames.len(), qtotal as f64 / 1.073741824e9);

        let embd_info = &gguf.tensors["token_embd.weight"];
        let embd_start = gguf.data_offset + embd_info.offset;
        let embd_n: usize = embd_info.dims.iter().product();
        let (eb, ee) = embd_info.dtype.block_size();
        let embd_data_bytes = embd_n / ee * eb;
        let embd_bytes = file[embd_start..embd_start + embd_data_bytes].to_vec();
        let output_norm = dev_f32("output_norm.weight")?;
        let alloc = |n: usize| HipBuffer::new(n * 4).map_err(|e| e.to_string());

        // Keep token_embd resident (GPU lm_head) if it still fits with margin —
        // otherwise CPU lm_head fallback. Re-query free VRAM after the layer uploads.
        let (free_now, _) = crate::backend::rocm::kern::vram_free_total().map_err(|e| e.to_string())?;
        // Keep token_embd resident (GPU lm_head) if ≥0.9 GiB headroom remains
        // after. GPU lm_head at AUTO clocks was proven stable at ~1 GiB headroom
        // (the crash only happened when ALSO forcing clocks — do NOT force clocks
        // with the lm_head resident; this box is too tight for that combination).
        let lm_margin: usize = if std::env::var("ROCM_CPU_LMHEAD").is_ok() { 99 << 30 } else { 900 * 1024 * 1024 };
        let want = embd_data_bytes + vocab * 4 + lm_margin;
        let (token_embd, logits) = if want < free_now {
            eprintln!("    lm_head: resident on GPU (token_embd {:.2} GiB)", embd_data_bytes as f64 / 1.073741824e9);
            (Some(dev_bytes("token_embd.weight")?), Some(alloc(vocab)?))
        } else {
            eprintln!("    lm_head: CPU (not enough VRAM for resident token_embd)");
            (None, None)
        };

        let rope_freqs = if gguf.tensors.contains_key("rope_freqs.weight") {
            Some(dev_f32("rope_freqs.weight")?)
        } else { None };

        let mut kv_k = Vec::with_capacity(n_layers);
        let mut kv_v = Vec::with_capacity(n_layers);
        for l in &layers {
            kv_k.push(alloc(max_seq * l.kv_dim)?);
            kv_v.push(alloc(max_seq * l.kv_dim)?);
        }
        let ones_host = vec![1.0f32; 512.max(64)];
        let ones = alloc(ones_host.len())?;
        ones.copy_from_host(&ones_host).map_err(|e| e.to_string())?;

        Ok(RocmGemma {
            n_layers, d_model, d_ff, vocab, eps, sliding_window, logit_cap, layers, out_scale,
            qbase, qoff, qdt, nw, embd_bytes, token_embd, logits, output_norm, rope_freqs,
            h: alloc(d_model)?, normed: alloc(d_model)?, q: alloc(max_q)?,
            k: alloc(max_kv)?, v: alloc(max_kv)?, attn: alloc(max_q)?,
            proj: alloc(d_model)?, gate: alloc(d_ff)?, up: alloc(d_ff)?,
            glu: alloc(d_ff)?, ones,
            // dp4a scratch must fit the LARGEST dp4a in_dim — that's ffn_down's
            // d_ff (15360), NOT d_model. ffn_down is Q4_K (dp4a arm) in 42/48
            // layers; sizing by only max_q/d_model overran it by 7168 bytes and
            // silently corrupted the FFN of every layer ≥6 (the dp4a "garbage" bug).
            xq: HipBuffer::new(max_q.max(d_model).max(d_ff)).map_err(|e| e.to_string())?,
            xd: alloc(max_q.max(d_model).max(d_ff) / 32)?,
            xsum: alloc(max_q.max(d_model).max(d_ff) / 32)?,
            kv_k, kv_v, kv_len: 0, max_seq,
            // dp4a is correct (argmax 532) and ~1.8× faster — default ON, opt out
            // with ROCM_NO_DP4A=1 to fall back to the f32 matvec.
            use_dp4a: std::env::var("ROCM_NO_DP4A").is_err(),
        })
    }

    pub fn reset(&mut self) { self.kv_len = 0; }

    /// Quantized matvec dispatched by the weight's ACTUAL dtype (Q4_K/Q5_K
    /// vary per layer — esp. attn_v). `bpr = in_dim/256`.
    unsafe fn qmatvec(&self, wname: &str, x: *const f32, y: *mut f32,
                      out_dim: usize, in_dim: usize) -> Result<(), BackendError> {
        let w = self.qbase.u8_at(self.qoff[wname]);
        let bpr = in_dim / 256;
        match self.qdt[wname] {
            GgmlType::Q5_K => kern::matvec_q5k(w, x, y, out_dim, bpr),
            GgmlType::Q6_K => kern::matvec_q6k(w, x, y, out_dim, bpr),
            _ if self.use_dp4a => {
                // Q4_K dp4a path (ROCM_DP4A=1): ~2.3× faster — quantize activation
                // to int8 Q8_1, then wide-int32-load integer matvec. The math is
                // exact (per-element quant error is zero-mean: bias/std ≈ -0.01).
                kern::quantize_q8_1(x, self.xq.device_ptr() as *mut i8,
                    self.xd.f32_at(0), self.xsum.f32_at(0), in_dim)?;
                kern::matvec_q4k_dp4a(w, self.xq.device_ptr() as *const i8,
                    self.xd.f32_ptr(), self.xsum.f32_ptr(), y, out_dim, bpr)
            }
            _ => kern::matvec_q4k(w, x, y, out_dim, bpr),
        }
    }

    /// One token. `token_id` at absolute position = current kv_len. Returns logits.
    pub fn forward_token(&mut self, token_id: u32) -> Result<Vec<f32>, String> {
        let d = self.d_model;
        let pos = self.kv_len;
        // ── embedding (CPU dequant of one Q6_K row → ×sqrt(d) → upload) ──
        let bpr_e = d / 256;
        let row_bytes = bpr_e * 210;
        let off = token_id as usize * row_bytes;
        let mut emb = vec![0.0f32; d];
        dequantize_row_q6_k(&self.embd_bytes[off..off + row_bytes], &mut emb, bpr_e);
        let scale = (d as f32).sqrt();
        for e in emb.iter_mut() { *e *= scale; }
        self.h.copy_from_host(&emb).map_err(|e| e.to_string())?;

        let timing = std::env::var("ROCM_TIMING").is_ok();
        let t0 = std::time::Instant::now();
        unsafe {
            for li in 0..self.n_layers {
                self.layer(li, pos).map_err(|e| e.to_string())?;
            }
            // final norm on GPU
            kern::rms_norm(self.h.f32_ptr(), self.output_norm.f32_ptr(), self.normed.f32_at(0), 1, d, self.eps).map_err(|e| e.to_string())?;
            if timing { kern::hip_sync().map_err(|e| e.to_string())?;
                eprintln!("  layers+norm: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0); }
            let t1 = std::time::Instant::now();
            if let Some(te) = &self.token_embd {
                // GPU lm_head (token_embd resident)
                kern::matvec_q6k(te.u8_ptr(), self.normed.f32_ptr(), self.logits.as_ref().unwrap().f32_at(0), self.vocab, d / 256).map_err(|e| e.to_string())?;
                if self.logit_cap > 0.0 {
                    kern::softcap(self.logits.as_ref().unwrap().f32_at(0), self.logit_cap, self.vocab).map_err(|e| e.to_string())?;
                }
                kern::hip_sync().map_err(|e| e.to_string())?;
            } else {
                kern::hip_sync().map_err(|e| e.to_string())?;
            }
            if timing && self.token_embd.is_some() {
                eprintln!("  lm_head(GPU): {:.1}ms", t1.elapsed().as_secs_f64() * 1000.0); }
        }
        self.kv_len += 1;
        if let Some(lg) = &self.logits {
            let mut out = vec![0.0f32; self.vocab];
            lg.copy_to_host(&mut out).map_err(|e| e.to_string())?;
            Ok(out)
        } else {
            // CPU lm_head fallback (token_embd host-only)
            let tlm = std::time::Instant::now();
            let mut hn = vec![0.0f32; d];
            self.normed.copy_to_host(&mut hn).map_err(|e| e.to_string())?;
            let r = self.cpu_lm_head(&hn);
            if timing { eprintln!("  lm_head(CPU): {:.1}ms", tlm.elapsed().as_secs_f64() * 1000.0); }
            Ok(r)
        }
    }

    /// Current sequence length (number of tokens in the KV cache).
    pub fn kv_len(&self) -> usize { self.kv_len }
    pub fn max_seq(&self) -> usize { self.max_seq }

    /// Greedy decode: prefill `prompt`, then emit up to `max_new` tokens, stopping
    /// at any id in `stop` (e.g. eos). Returns ONLY the newly generated ids.
    /// Caller should `reset()` first for an independent sequence.
    pub fn generate(&mut self, prompt: &[u32], max_new: usize, stop: &[u32]) -> Result<Vec<u32>, String> {
        if prompt.is_empty() { return Err("empty prompt".into()); }
        let mut logits = Vec::new();
        for &t in prompt {
            if self.kv_len >= self.max_seq { break; }
            logits = self.forward_token(t)?;
        }
        let mut out = Vec::with_capacity(max_new);
        let mut hist: Vec<u32> = prompt.to_vec();
        for _ in 0..max_new {
            // repeat penalty (ollama default 1.1 over last 64) — without it, pure
            // greedy on this reasoning model loops on the "thought" channel.
            repeat_penalty(&mut logits, &hist, 64, 1.1);
            let next = argmax(&logits);
            if stop.contains(&next) || self.kv_len >= self.max_seq { break; }
            out.push(next); hist.push(next);
            logits = self.forward_token(next)?;
        }
        Ok(out)
    }

    /// Streaming greedy decode: same as `generate` but calls `on_token(id)` for each
    /// new token as it is produced (return `false` to stop early).
    pub fn generate_stream(&mut self, prompt: &[u32], max_new: usize, stop: &[u32],
                           mut on_token: impl FnMut(u32) -> bool) -> Result<(), String> {
        if prompt.is_empty() { return Err("empty prompt".into()); }
        let mut logits = Vec::new();
        for &t in prompt {
            if self.kv_len >= self.max_seq { break; }
            logits = self.forward_token(t)?;
        }
        for _ in 0..max_new {
            let next = argmax(&logits);
            if stop.contains(&next) || self.kv_len >= self.max_seq { break; }
            if !on_token(next) { break; }
            logits = self.forward_token(next)?;
        }
        Ok(())
    }

    /// Tied lm_head on CPU: logits = Q6_K(token_embd) @ h, with softcap.
    fn cpu_lm_head(&self, h: &[f32]) -> Vec<f32> {
        use rayon::prelude::*;
        let d = self.d_model;
        let bpr = d / 256;
        let rb = bpr * 210;
        let cap = self.logit_cap;
        (0..self.vocab).into_par_iter().map(|r| {
            let mut w = vec![0.0f32; d];
            dequantize_row_q6_k(&self.embd_bytes[r * rb..(r + 1) * rb], &mut w, bpr);
            let mut a = 0.0f64;
            for k in 0..d { a += w[k] as f64 * h[k] as f64; }
            let l = a as f32;
            if cap > 0.0 { cap * (l / cap).tanh() } else { l }
        }).collect()
    }

    unsafe fn layer(&mut self, li: usize, pos: usize) -> Result<(), BackendError> {
        let d = self.d_model;
        let L = &self.layers[li];
        let (hd, nh, nkv, q_dim, kv_dim, is_local, rope_base) =
            (L.head_dim, L.n_heads, L.n_kv_heads, L.q_dim, L.kv_dim, L.is_local, L.rope_base);
        let p = |s: &str| format!("blk.{li}.{s}.weight");
        let n = |s: &str| self.nw[&p(s)].f32_ptr();
        let seq = pos + 1;

        // ── attention ──
        kern::rms_norm(self.h.f32_ptr(), n("attn_norm"), self.normed.f32_at(0), 1, d, self.eps)?;
        self.qmatvec(&p("attn_q"), self.normed.f32_ptr(), self.q.f32_at(0), q_dim, d)?;
        self.qmatvec(&p("attn_k"), self.normed.f32_ptr(), self.k.f32_at(0), kv_dim, d)?;
        if is_local {
            self.qmatvec(&p("attn_v"), self.normed.f32_ptr(), self.v.f32_at(0), kv_dim, d)?;
        } else {
            kern::copy(self.v.f32_at(0), self.k.f32_ptr(), kv_dim)?; // V = K
        }
        // qk-norm (per head, scaled), v-norm (per head, ones), rope on q,k
        kern::rms_norm(self.q.f32_ptr(), n("attn_q_norm"), self.q.f32_at(0), nh, hd, self.eps)?;
        kern::rms_norm(self.k.f32_ptr(), n("attn_k_norm"), self.k.f32_at(0), nkv, hd, self.eps)?;
        kern::rms_norm(self.v.f32_ptr(), self.ones.f32_ptr(), self.v.f32_at(0), nkv, hd, self.eps)?;
        // Proportional RoPE (rope_freqs) on FULL-attention (global) layers only;
        // local/sliding layers use full rotation (null factors).
        let ff = if !is_local {
            self.rope_freqs.as_ref().map(|b| b.f32_ptr()).unwrap_or(std::ptr::null())
        } else { std::ptr::null() };
        kern::rope_neox(self.q.f32_at(0), pos, nh, hd, rope_base, ff)?;
        kern::rope_neox(self.k.f32_at(0), pos, nkv, hd, rope_base, ff)?;
        // append K,V to cache at pos
        kern::copy(self.kv_k[li].f32_at(pos * kv_dim), self.k.f32_ptr(), kv_dim)?;
        kern::copy(self.kv_v[li].f32_at(pos * kv_dim), self.v.f32_ptr(), kv_dim)?;
        // attention
        let win_start = if is_local && seq > self.sliding_window { seq - self.sliding_window } else { 0 };
        // Gemma4 attention scale = 1.0 (NO 1/sqrt(head_dim)). The qk-norm already
        // normalizes Q/K, so llama.cpp sets hparams.f_attention_scale = 1.0 for
        // gemma4. Using 1/sqrt(head_dim) over-scaled scores 16-22× → flat softmax
        // → garbage. (ref: llama.cpp llama-model.cpp GEMMA4 case, line ~1273.)
        let ascale = 1.0f32;
        kern::sdpa_decode(self.q.f32_ptr(), self.kv_k[li].f32_ptr(), self.kv_v[li].f32_ptr(),
            self.attn.f32_at(0), nh, nkv, hd, seq, kv_dim, ascale, win_start)?;
        // output proj + post-norm + residual
        self.qmatvec(&p("attn_output"), self.attn.f32_ptr(), self.proj.f32_at(0), d, q_dim)?;
        kern::rms_norm(self.proj.f32_ptr(), n("post_attention_norm"), self.proj.f32_at(0), 1, d, self.eps)?;
        kern::residual_add(self.h.f32_at(0), self.proj.f32_ptr(), d)?;


        // ── FFN (GeGLU) ──
        kern::rms_norm(self.h.f32_ptr(), n("ffn_norm"), self.normed.f32_at(0), 1, d, self.eps)?;
        self.qmatvec(&p("ffn_gate"), self.normed.f32_ptr(), self.gate.f32_at(0), self.d_ff, d)?;
        self.qmatvec(&p("ffn_up"), self.normed.f32_ptr(), self.up.f32_at(0), self.d_ff, d)?;
        kern::geglu(self.gate.f32_ptr(), self.up.f32_ptr(), self.glu.f32_at(0), self.d_ff)?;
        self.qmatvec(&p("ffn_down"), self.glu.f32_ptr(), self.proj.f32_at(0), d, self.d_ff)?;
        kern::rms_norm(self.proj.f32_ptr(), n("post_ffw_norm"), self.proj.f32_at(0), 1, d, self.eps)?;
        kern::residual_add(self.h.f32_at(0), self.proj.f32_ptr(), d)?;

        // layer_output_scale
        let s = self.out_scale[li];
        if s != 1.0 { kern::scale(self.h.f32_at(0), s, d)?; }
        Ok(())
    }
}
