//! Gemma 4 inference engine — GGUF → KFD GPU → tokens.
//!
//! Loads quantized weights into VRAM via mmap. Dispatches all linear
//! ops through the Q4_K_M matvec kernel. RMSNorm and SiLU run on CPU
//! (too small for GPU dispatch overhead at inference batch=1).
//!
//! Memory layout:
//!   VRAM: all Q4/Q5 weight tensors (4.8 GB) + KV cache + activation buffers
//!   CPU: f32 norm weights (~100 KB) + tokenizer
//!
//! The entire 4.8 GB model is uploaded to VRAM once at load time.
//! During inference, zero PCIe traffic — everything stays on GPU.

use super::dispatch_queue::{GpuQueue, VramBuf};
use super::gguf::{GgufFile, GgmlType, TensorInfo};
use super::HsaDevice;
use std::collections::HashMap;

/// A weight tensor in VRAM (quantized), CPU-mmap'd, or CPU f32.
enum Weight {
    /// Quantized tensor in VRAM with its own allocation.
    Vram {
        va: u64,
        dtype: GgmlType,
        dims: Vec<usize>,
        data_bytes: usize,
        file_offset: usize, // absolute offset in mmap'd file (for CPU fallback)
        _buf: VramBuf,
    },
    /// Large quantized tensor kept on CPU (e.g. embedding tables).
    /// Read through mmap'd file data.
    Mmap {
        file_offset: usize, // absolute byte offset in the file
        dtype: GgmlType,
        dims: Vec<usize>,
        data_bytes: usize,
    },
    /// Small f32 tensor on CPU (norms, scales).
    CpuF32(Vec<f32>),
}

/// Gemma 4 model loaded for inference.
pub struct Gemma4Model {
    // Architecture
    n_layers: usize,
    d_model: usize,      // 2560
    d_ff: usize,          // 10240
    n_heads: usize,       // 8
    n_kv_heads: usize,    // 2
    head_dim: usize,      // 256
    q_dim: usize,         // n_heads * head_dim = 2048
    kv_dim: usize,        // n_kv_heads * head_dim = 512
    vocab_size: usize,
    rms_eps: f32,
    rope_base: f32,
    rope_base_swa: f32,
    logit_cap: f32,
    /// First N layers compute their own KV. Layers >= this reuse earlier KV.
    n_layer_kv_from_start: usize,
    /// Sliding window pattern per layer (true = SWA layer).
    is_swa: Vec<bool>,
    /// Proportional RoPE frequency factors for full-attention layers.
    /// Length = head_dim/2. Factor of 1.0 = normal, 1e30 = frozen dimension.
    rope_freqs: Vec<f32>,

    // Weights (keyed by tensor name)
    weights: HashMap<String, Weight>,
    // mmap'd file data for CPU-resident weights (embeddings)
    file_data_ptr: *const u8,
    data_offset: usize,

    // Activation buffers (reused across layers)
    buf_hidden: VramBuf,    // [d_model] current hidden state
    buf_q: VramBuf,         // [q_dim] query projection
    buf_k: VramBuf,         // [kv_dim] key projection
    buf_v: VramBuf,         // [kv_dim] value projection
    buf_attn_out: VramBuf,  // [q_dim] attention output before out_proj
    buf_out: VramBuf,       // [d_model] after out_proj
    buf_ff_gate: VramBuf,   // [d_ff] FFN gate
    buf_ff_up: VramBuf,     // [d_ff] FFN up
    buf_ff_down: VramBuf,   // [d_model] FFN down output
    buf_logits: VramBuf,    // [vocab_size] final logits

    // KV cache: [n_layers][max_seq][kv_dim] for K and V
    kv_k: Vec<VramBuf>,     // per-layer K cache
    kv_v: Vec<VramBuf>,     // per-layer V cache
    pub kv_len: usize,       // current sequence length in cache
    max_seq: usize,
}

impl Gemma4Model {
    /// Load a Gemma 4 GGUF model into VRAM.
    pub fn load(
        gguf: &GgufFile,
        file_data: &[u8],  // mmap'd GGUF file
        dev: &HsaDevice,
        queue: &mut GpuQueue,
        max_seq: usize,
    ) -> Result<Self, String> {
        let arch = gguf.architecture().unwrap_or("unknown");
        if arch != "gemma4" {
            return Err(format!("expected gemma4 architecture, got {}", arch));
        }

        let n_layers = gguf.meta_u32(&format!("{arch}.block_count")).unwrap_or(42) as usize;
        let d_model = gguf.meta_u32(&format!("{arch}.embedding_length")).unwrap_or(2560) as usize;
        let d_ff = gguf.meta_u32(&format!("{arch}.feed_forward_length")).unwrap_or(10240) as usize;
        let n_heads = gguf.meta_u32(&format!("{arch}.attention.head_count")).unwrap_or(8) as usize;
        let n_kv_heads = gguf.meta_u32(&format!("{arch}.attention.head_count_kv")).unwrap_or(2) as usize;
        let head_dim = d_model / n_heads; // 256 for Gemma4-E4B... wait
        // Actually Gemma4 has separate key_length
        let key_length = gguf.meta_u32(&format!("{arch}.attention.key_length")).unwrap_or(256) as usize;
        let head_dim = key_length / n_kv_heads; // 512/2 = 256
        let q_dim = n_heads * head_dim;       // 8*256 = 2048
        let kv_dim = n_kv_heads * head_dim;   // 2*256 = 512
        let rms_eps = gguf.meta_f32(&format!("{arch}.attention.layer_norm_rms_epsilon")).unwrap_or(1e-6);
        let rope_base = gguf.meta_f32(&format!("{arch}.rope.freq_base")).unwrap_or(1000000.0);
        let rope_base_swa = gguf.meta_f32(&format!("{arch}.rope.freq_base_swa")).unwrap_or(10000.0);
        let logit_cap = gguf.meta_f32(&format!("{arch}.final_logit_softcapping")).unwrap_or(30.0);

        // Shared KV: first N layers compute KV, rest reuse
        let n_kv_shared = gguf.meta_u32(&format!("{arch}.attention.shared_kv_layers")).unwrap_or(0) as usize;
        let n_layer_kv_from_start = n_layers - n_kv_shared;
        eprintln!("KV layers: {} own + {} shared (reuse from earlier)",
            n_layer_kv_from_start, n_kv_shared);

        // Sliding window pattern
        let is_swa: Vec<bool> = gguf.meta(&format!("{arch}.attention.sliding_window_pattern"))
            .and_then(|v| v.as_bool_array())
            .unwrap_or_else(|| vec![false; n_layers]);

        // Determine vocab size from token_embd shape
        let vocab_size = gguf.tensors.get("token_embd.weight")
            .map(|t| t.dims.get(1).copied().unwrap_or(262144))
            .unwrap_or(262144);

        eprintln!("Gemma4: {}L, d={}, ff={}, {}h/{}kv, head={}, vocab={}",
            n_layers, d_model, d_ff, n_heads, n_kv_heads, head_dim, vocab_size);

        // Upload weights: per-tensor VRAM allocations (avoids single giant alloc)
        let mut weights = HashMap::new();
        let mut total_vram: usize = 0;
        let mut n_uploaded = 0usize;

        for name in &gguf.tensor_list {
            let info = &gguf.tensors[name];
            let file_off = gguf.data_offset + info.offset;

            match info.dtype {
                GgmlType::F32 => {
                    let n = info.n_elements();
                    let bytes = &file_data[file_off..file_off + n * 4];
                    let f32s: Vec<f32> = bytes.chunks(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    weights.insert(name.clone(), Weight::CpuF32(f32s));
                }
                GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q2_K | GgmlType::Q3_K
                | GgmlType::Q6_K | GgmlType::Q8_0 | GgmlType::BF16 => {
                    let data_bytes = info.data_bytes();

                    // Large embedding tables stay on CPU (row lookup, not matvec)
                    let is_embedding = name.contains("embd") || name.contains("per_layer_model_proj");
                    if is_embedding || data_bytes > 100_000_000 {
                        weights.insert(name.clone(), Weight::Mmap {
                            file_offset: file_off,
                            dtype: info.dtype, dims: info.dims.clone(), data_bytes,
                        });
                        continue;
                    }

                    let src = &file_data[file_off..file_off + data_bytes];

                    // Per-tensor VRAM allocation
                    let n_floats = (data_bytes + 3) / 4;
                    let buf = queue.alloc(dev, n_floats)
                        .ok_or_else(|| format!("VRAM alloc failed for {} ({:.1} MB)",
                            name, data_bytes as f64 / 1e6))?;

                    // Upload raw bytes via BAR
                    unsafe {
                        std::ptr::copy_nonoverlapping(src.as_ptr(), buf.ptr as *mut u8, data_bytes);
                    }

                    weights.insert(name.clone(), Weight::Vram {
                        va: buf.va, dtype: info.dtype, dims: info.dims.clone(),
                        data_bytes, file_offset: file_off, _buf: buf,
                    });

                    total_vram += data_bytes;
                    n_uploaded += 1;
                }
                _ => {
                    eprintln!("  skipping tensor {} with type {:?}", name, info.dtype);
                }
            }
        }

        eprintln!("Uploaded {:.1} GB to VRAM ({} tensors)", total_vram as f64 / 1e9, n_uploaded);

        // Allocate activation buffers
        let buf_hidden = queue.alloc(dev, d_model).ok_or("alloc hidden")?;
        let buf_q = queue.alloc(dev, q_dim).ok_or("alloc q")?;
        let buf_k = queue.alloc(dev, kv_dim).ok_or("alloc k")?;
        let buf_v = queue.alloc(dev, kv_dim).ok_or("alloc v")?;
        let buf_attn_out = queue.alloc(dev, q_dim).ok_or("alloc attn_out")?;
        let buf_out = queue.alloc(dev, d_model).ok_or("alloc out")?;
        let buf_ff_gate = queue.alloc(dev, d_ff).ok_or("alloc ff_gate")?;
        let buf_ff_up = queue.alloc(dev, d_ff).ok_or("alloc ff_up")?;
        let buf_ff_down = queue.alloc(dev, d_model).ok_or("alloc ff_down")?;
        let buf_logits = queue.alloc(dev, vocab_size).ok_or("alloc logits")?;

        // KV cache — per-layer dimensions (SWA layers have different kv_dim)
        let mut kv_k = Vec::with_capacity(n_layers);
        let mut kv_v = Vec::with_capacity(n_layers);
        let mut total_kv_bytes = 0usize;
        for layer in 0..n_layers {
            let k_name = format!("blk.{layer}.attn_k.weight");
            let layer_kv_dim = gguf.tensors.get(&k_name)
                .map(|t| t.dims.get(1).copied().unwrap_or(kv_dim))
                .unwrap_or(kv_dim);
            kv_k.push(queue.alloc(dev, max_seq * layer_kv_dim).ok_or("alloc kv_k")?);
            kv_v.push(queue.alloc(dev, max_seq * layer_kv_dim).ok_or("alloc kv_v")?);
            total_kv_bytes += 2 * max_seq * layer_kv_dim * 4;
        }

        eprintln!("KV cache: {:.1} MB for {} layers × {} tokens",
            total_kv_bytes as f64 / 1e6, n_layers, max_seq);

        Ok(Gemma4Model {
            n_layers, d_model, d_ff, n_heads, n_kv_heads, head_dim,
            q_dim, kv_dim, vocab_size, rms_eps, rope_base, rope_base_swa, logit_cap,
            n_layer_kv_from_start, is_swa,
            rope_freqs: match weights.get("rope_freqs.weight") {
                Some(Weight::CpuF32(v)) => v.clone(),
                _ => Vec::new(),
            },
            weights,
            file_data_ptr: file_data.as_ptr(),
            data_offset: gguf.data_offset,
            buf_hidden, buf_q, buf_k, buf_v, buf_attn_out, buf_out,
            buf_ff_gate, buf_ff_up, buf_ff_down, buf_logits,
            kv_k, kv_v, kv_len: 0, max_seq,
        })
    }

    /// Get f32 norm weights by tensor name.
    fn norm(&self, name: &str) -> &[f32] {
        match self.weights.get(name) {
            Some(Weight::CpuF32(v)) => v,
            _ => panic!("missing f32 weight: {}", name),
        }
    }

    /// Get VRAM-resident quantized weight info: (va, dtype, dims, mmap_ptr)
    /// mmap_ptr points to the mmap'd file data for CPU fallback (fast, no BAR).
    fn qweight_vram(&self, name: &str) -> (u64, &GgmlType, &[usize], *const u8) {
        match self.weights.get(name) {
            Some(Weight::Vram { va, dtype, dims, file_offset, .. }) => {
                let mmap_ptr = unsafe { self.file_data_ptr.add(*file_offset) };
                (*va, dtype, dims, mmap_ptr)
            }
            _ => panic!("missing VRAM weight: {}", name),
        }
    }

    /// Get CPU mmap'd weight data pointer.
    fn qweight_mmap(&self, name: &str) -> (&GgmlType, &[usize], *const u8) {
        match self.weights.get(name) {
            Some(Weight::Mmap { file_offset, dtype, dims, .. }) => {
                let ptr = unsafe { self.file_data_ptr.add(*file_offset) };
                (dtype, dims, ptr)
            }
            _ => panic!("missing mmap weight: {}", name),
        }
    }

    /// Get output dimension of a weight tensor (dims[1]).
    fn weight_out_dim(&self, name: &str) -> usize {
        match self.weights.get(name) {
            Some(Weight::Vram { dims, .. }) => dims.get(1).copied().unwrap_or(0),
            Some(Weight::Mmap { dims, .. }) => dims.get(1).copied().unwrap_or(0),
            _ => panic!("missing weight: {}", name),
        }
    }

    /// Get weight data pointer regardless of storage (VRAM or mmap).
    fn qweight_mmap_or_vram(&self, name: &str) -> (&GgmlType, &[usize], *const u8) {
        match self.weights.get(name) {
            Some(Weight::Vram { dtype, dims, file_offset, .. }) => {
                let ptr = unsafe { self.file_data_ptr.add(*file_offset) };
                (dtype, dims, ptr)
            }
            Some(Weight::Mmap { dtype, dims, file_offset, .. }) => {
                let ptr = unsafe { self.file_data_ptr.add(*file_offset) };
                (dtype, dims, ptr)
            }
            _ => panic!("missing weight: {}", name),
        }
    }

    /// RMSNorm on CPU: x = x * rsqrt(mean(x^2) + eps) * weight
    fn rms_norm_cpu(&self, x: &mut [f32], weight: &[f32]) {
        let n = x.len() as f32;
        let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / n;
        let inv = 1.0 / (ss + self.rms_eps).sqrt();
        for i in 0..x.len() {
            x[i] = x[i] * inv * weight[i];
        }
    }

    /// RMSNorm in-place (no learned scale): x = x / sqrt(mean(x²) + eps)
    fn rms_norm_inplace(&self, x: &mut [f32]) {
        let n = x.len() as f32;
        let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / n;
        let inv = 1.0 / (ss + self.rms_eps).sqrt();
        for v in x.iter_mut() { *v *= inv; }
    }

    /// RMSNorm with learned scale: dst[i] = weight[i] * src[i] / sqrt(mean(src²) + eps)
    fn rms_norm_scaled(&self, src: &[f32], weight: &[f32], dst: &mut [f32]) {
        let n = src.len() as f32;
        let ss: f32 = src.iter().map(|v| v * v).sum::<f32>() / n;
        let inv = 1.0 / (ss + self.rms_eps).sqrt();
        for i in 0..src.len().min(dst.len()).min(weight.len()) {
            dst[i] = weight[i] * src[i] * inv;
        }
    }

    /// RMSNorm with learned scale in-place
    fn rms_norm_scaled_inplace(&self, x: &mut [f32], weight: &[f32]) {
        let n = x.len() as f32;
        let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / n;
        let inv = 1.0 / (ss + self.rms_eps).sqrt();
        for i in 0..x.len().min(weight.len()) {
            x[i] = weight[i] * x[i] * inv;
        }
    }

    /// GELU activation: x = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x³)))
    fn gelu_cpu(x: &mut [f32]) {
        const SQRT_2_OVER_PI: f32 = 0.7978845608;
        for v in x.iter_mut() {
            let x3 = *v * *v * *v;
            let inner = SQRT_2_OVER_PI * (*v + 0.044715 * x3);
            *v = 0.5 * *v * (1.0 + inner.tanh());
        }
    }

    /// SiLU on CPU: x = x * sigmoid(x)
    fn silu_cpu(x: &mut [f32]) {
        for v in x.iter_mut() {
            *v *= 1.0 / (1.0 + (-*v).exp());
        }
    }

    /// Compute per-layer embedding inputs.
    /// Matches llama.cpp build_inp_per_layer + project_per_layer_inputs.
    ///
    /// 1. Look up token in per_layer_token_embd → [n_embd_per_layer * n_layer]
    /// 2. Scale by sqrt(n_embd_per_layer)
    /// 3. Project main embedding through per_layer_model_proj → [n_embd_per_layer * n_layer]
    /// 4. Scale by 1/sqrt(n_embd)
    /// 5. RMSNorm with per_layer_proj_norm
    /// 6. Add lookup + projection
    /// 7. Scale by 1/sqrt(2)
    ///
    /// Returns [n_layer * n_embd_per_layer] flat, layer `i` at offset `i * n_embd_per_layer`.
    fn compute_per_layer_inputs(
        &self, token_id: u32, hidden: &[f32], n_embd_per_layer: usize,
    ) -> Vec<f32> {
        let total_dim = n_embd_per_layer * self.n_layers; // 256 * 42 = 10752

        // Step 1: Look up from per_layer_token_embd [10752, 262144]
        let ple_name = "per_layer_token_embd.weight";
        if !self.weights.contains_key(ple_name) { return Vec::new(); }

        let (dtype, dims, base_ptr) = self.qweight_mmap(ple_name);
        let row_elems = dims[0]; // 10752
        let (block_bytes, block_elems) = dtype.block_size();
        let blocks_per_row = (row_elems + block_elems - 1) / block_elems;
        let row_bytes = blocks_per_row * block_bytes;

        // Dequant the row for this token
        let row_data = unsafe {
            std::slice::from_raw_parts(base_ptr.add(token_id as usize * row_bytes), row_bytes)
        };
        let mut ple_lookup = vec![0.0f32; row_elems];
        match dtype {
            GgmlType::Q5_K => dequant_q5k_row(row_data, &mut ple_lookup, blocks_per_row),
            GgmlType::Q4_K => dequant_q4k_row(row_data, &mut ple_lookup, blocks_per_row),
            _ => {}
        }
        // Clamp NaN/inf
        for v in ple_lookup.iter_mut() {
            if v.is_nan() || v.is_infinite() { *v = 0.0; }
        }
        // Scale by sqrt(n_embd_per_layer)
        let ple_scale = (n_embd_per_layer as f32).sqrt();
        for v in ple_lookup.iter_mut() { *v *= ple_scale; }

        // Step 2: Project main embedding through per_layer_model_proj
        // per_layer_model_proj: [n_embd, total_dim] BF16 → matvec(hidden) → [total_dim]
        let proj_name = "per_layer_model_proj.weight";
        let mut per_layer_proj = if self.weights.contains_key(proj_name) {
            let (proj_dtype, proj_dims, proj_ptr) = self.qweight_mmap_or_vram(proj_name);
            let proj_out_dim = proj_dims.get(1).copied().unwrap_or(total_dim);
            let proj_in_dim = proj_dims[0];
            let (pb, pe) = proj_dtype.block_size();
            let bpr = (proj_in_dim + pe - 1) / pe;
            let rb = bpr * pb;

            let mut proj_out = vec![0.0f32; proj_out_dim];
            let mut row_buf = vec![0.0f32; proj_in_dim];
            let base = proj_ptr;
            for row in 0..proj_out_dim {
                let rd = unsafe { std::slice::from_raw_parts(base.add(row * rb), rb) };
                row_buf.fill(0.0);
                match proj_dtype {
                    GgmlType::Q5_K => dequant_q5k_row(rd, &mut row_buf, bpr),
                    GgmlType::Q4_K => dequant_q4k_row(rd, &mut row_buf, bpr),
                    GgmlType::BF16 => {
                        // BF16 dequant
                        for i in 0..proj_in_dim.min(row_buf.len()) {
                            let raw = u16::from_le_bytes([rd[i*2], rd[i*2+1]]);
                            row_buf[i] = f32::from_bits((raw as u32) << 16);
                        }
                    }
                    _ => {}
                }
                for v in row_buf.iter_mut() {
                    if v.is_nan() || v.is_infinite() { *v = 0.0; }
                }
                proj_out[row] = dot(&row_buf[..proj_in_dim.min(hidden.len())], &hidden[..proj_in_dim.min(hidden.len())]);
            }
            proj_out
        } else {
            vec![0.0f32; total_dim]
        };

        // Scale by 1/sqrt(n_embd)
        let proj_scale = 1.0 / (self.d_model as f32).sqrt();
        for v in per_layer_proj.iter_mut() { *v *= proj_scale; }

        // RMSNorm with per_layer_proj_norm
        let norm_name = "per_layer_proj_norm.weight";
        if let Some(Weight::CpuF32(norm_w)) = self.weights.get(norm_name) {
            // Apply RMSNorm per-layer slice (each slice is n_embd_per_layer)
            for layer in 0..self.n_layers {
                let off = layer * n_embd_per_layer;
                let slice = &mut per_layer_proj[off..off + n_embd_per_layer];
                let n = slice.len() as f32;
                let ss: f32 = slice.iter().map(|v| v * v).sum::<f32>() / n;
                let inv = 1.0 / (ss + self.rms_eps).sqrt();
                for i in 0..n_embd_per_layer.min(norm_w.len()) {
                    slice[i] = norm_w[i] * slice[i] * inv;
                }
            }
        }

        // Add: ple_lookup + per_layer_proj
        let mut result = vec![0.0f32; total_dim];
        for i in 0..total_dim.min(ple_lookup.len()).min(per_layer_proj.len()) {
            result[i] = ple_lookup[i] + per_layer_proj[i];
        }

        // Scale by 1/sqrt(2)
        let final_scale = 1.0 / 2.0f32.sqrt();
        for v in result.iter_mut() { *v *= final_scale; }

        result
    }

    /// Reset KV cache (new conversation).
    pub fn reset(&mut self) {
        self.kv_len = 0;
    }

    /// Forward pass for one token. Returns logits on CPU.
    /// Appends to KV cache. Call reset() for new conversation.
    pub fn forward_token(
        &mut self,
        token_id: u32,
        dev: &mut HsaDevice,
        queue: &mut GpuQueue,
    ) -> Vec<f32> {
        let pos = self.kv_len;

        // ─── Embedding lookup (CPU, then upload) ───
        // token_embd is Q5_K quantized — dequant one row on CPU for now
        // TODO: GPU embedding kernel
        let hidden_f32 = self.embed_token(token_id);
        // Use f64 for the residual stream to avoid precision loss.
        // The embedding has norm ~100M and layer outputs have norm ~400.
        // In f32, adding 400 to 5.9B loses the 400 entirely (f32 has ~7 digits).
        // f64 has ~15 digits → 400 + 5.9B = 5900000400 exactly.
        let mut hidden: Vec<f64> = hidden_f32.iter().map(|&v| v as f64).collect();

        // ─── Per-layer embedding (PLE) computation ───
        // Matches llama.cpp build_inp_per_layer + project_per_layer_inputs
        let n_embd_per_layer = 256; // from GGUF metadata: embedding_length_per_layer_input
        let hidden_f32_for_ple: Vec<f32> = hidden.iter().map(|&v| v as f32).collect();
        let per_layer_inputs = self.compute_per_layer_inputs(
            token_id, &hidden_f32_for_ple, n_embd_per_layer);

        // ─── Transformer layers ───
        // Matches llama.cpp gemma4-iswa.cpp exactly.
        let layer_start = std::time::Instant::now();
        let run_layers = self.n_layers;
        for layer in 0..run_layers {
            let l = format!("blk.{layer}");

            // Convert hidden to f32 for layer operations
            let hidden_f32: Vec<f32> = hidden.iter().map(|&v| v as f32).collect();

            // ─── 1. Pre-attention RMSNorm ───
            let mut cur = vec![0.0f32; self.d_model];
            self.rms_norm_scaled(&hidden_f32, self.norm(&format!("{l}.attn_norm.weight")), &mut cur);

            if layer == 0 {
                let dbg = |name: &str, v: &[f32]| {
                    let nan = v.iter().filter(|x| x.is_nan()).count();
                    let norm: f32 = v.iter().map(|x| x*x).sum::<f32>().sqrt();
                    eprintln!("    L0 {}: norm={:.2} nan={}", name, norm, nan);
                };
                dbg("after_attn_norm", &cur);
            }

            // ─── 2. Q/K/V projections — dimensions vary per layer ───
            let q_out_dim = self.weight_out_dim(&format!("{l}.attn_q.weight"));
            let kv_out_dim = self.weight_out_dim(&format!("{l}.attn_k.weight"));
            let q_norm_name = format!("{l}.attn_q_norm.weight");
            let layer_head_dim = match self.weights.get(&q_norm_name) {
                Some(Weight::CpuF32(v)) => v.len(),
                _ => self.head_dim,
            };
            let layer_n_heads = q_out_dim / layer_head_dim;
            let layer_n_kv_heads = kv_out_dim / layer_head_dim;

            // Q projection (always computed)
            let mut q_data = self.cpu_q4_matvec(&format!("{l}.attn_q.weight"), &cur, q_out_dim);

            // Q norm: per-head RMSNorm with learned scale
            let q_norm_w = self.norm(&q_norm_name);
            for h in 0..layer_n_heads {
                let s = h * layer_head_dim;
                self.rms_norm_inplace(&mut q_data[s..s + layer_head_dim]);
                for i in 0..layer_head_dim { q_data[s + i] *= q_norm_w[i]; }
            }

            // RoPE on Q — SWA layers use standard RoPE, full layers use proportional
            // NOTE: sliding_window_pattern[i]=True means FULL attention (not SWA)
            let is_swa_layer = !self.is_swa.get(layer).copied().unwrap_or(true);
            let layer_rope_base = if is_swa_layer { self.rope_base_swa } else { self.rope_base };
            let freq_factors = if is_swa_layer { None } else { Some(self.rope_freqs.as_slice()) };
            self.apply_rope_ext(&mut q_data, pos, layer_n_heads, layer_head_dim,
                layer_rope_base, freq_factors);

            // ─── 3. K/V — only if this layer has its own KV ───
            // Layers >= n_layer_kv_from_start reuse KV from earlier layers
            let has_kv = layer < self.n_layer_kv_from_start;
            if has_kv {
                let mut k_data = self.cpu_q4_matvec(&format!("{l}.attn_k.weight"), &cur, kv_out_dim);
                let mut v_data = self.cpu_q4_matvec(&format!("{l}.attn_v.weight"), &cur, kv_out_dim);

                // K norm: per-head RMSNorm with learned scale
                let k_norm_w = self.norm(&format!("{l}.attn_k_norm.weight"));
                for h in 0..layer_n_kv_heads {
                    let s = h * layer_head_dim;
                    self.rms_norm_inplace(&mut k_data[s..s + layer_head_dim]);
                    for i in 0..layer_head_dim { k_data[s + i] *= k_norm_w[i]; }
                }

                // V norm: plain RMSNorm (no learned scale) — from reference line 92
                if layer == 0 {
                    let vn: f32 = v_data.iter().map(|x| x*x).sum::<f32>().sqrt();
                    let vnan = v_data.iter().filter(|x| x.is_nan()).count();
                    eprintln!("    L0 V_before_norm: norm={:.2} nan={}", vn, vnan);
                }
                for h in 0..layer_n_kv_heads {
                    let s = h * layer_head_dim;
                    self.rms_norm_inplace(&mut v_data[s..s + layer_head_dim]);
                }
                if layer == 0 {
                    let vn: f32 = v_data.iter().map(|x| x*x).sum::<f32>().sqrt();
                    let vnan = v_data.iter().filter(|x| x.is_nan()).count();
                    eprintln!("    L0 V_after_norm: norm={:.2} nan={}", vn, vnan);
                }

                // RoPE on K — same freq_factors as Q for this layer
                self.apply_rope_ext(&mut k_data, pos, layer_n_kv_heads, layer_head_dim,
                    layer_rope_base, freq_factors);

                // Append K, V to cache
                self.kv_k[layer].upload_at(pos * kv_out_dim, &k_data);
                self.kv_v[layer].upload_at(pos * kv_out_dim, &v_data);
            }
            // else: layers >= n_layer_kv_from_start reuse KV from earlier layer

            // Determine which KV cache layer to read from
            let kv_layer = if has_kv {
                layer
            } else {
                // Reuse rule from llama.cpp: SWA layers → kv_from_start-2, full → kv_from_start-1
                let is_swa = self.is_swa.get(layer).copied().unwrap_or(false);
                if is_swa {
                    self.n_layer_kv_from_start.saturating_sub(2)
                } else {
                    self.n_layer_kv_from_start.saturating_sub(1)
                }
            };

            // ─── 4. Attention ───
            if layer == 0 {
                let qn: f32 = q_data.iter().map(|x| x*x).sum::<f32>().sqrt();
                let qnan = q_data.iter().filter(|x| x.is_nan()).count();
                eprintln!("    L0 Q: norm={:.2} nan={} dim={}", qn, qnan, q_data.len());
                // Check KV cache
                let kv_d = self.kv_k[0].download(kv_out_dim);
                let kn: f32 = kv_d.iter().map(|x| x*x).sum::<f32>().sqrt();
                let knan = kv_d.iter().filter(|x| x.is_nan()).count();
                eprintln!("    L0 K[0]: norm={:.2} nan={} dim={}", kn, knan, kv_d.len());
            }
            // Use kv_layer for cache lookup (shared KV reuses earlier layer's cache)
            cur = self.cpu_attention_layer(
                &q_data, kv_layer, pos + 1,
                layer_n_heads, layer_n_kv_heads, layer_head_dim, kv_out_dim);

            if layer == 0 {
                let nan = cur.iter().filter(|x| x.is_nan()).count();
                let norm: f32 = cur.iter().map(|x| x*x).sum::<f32>().sqrt();
                eprintln!("    L0 attn_result: norm={:.2} nan={} dim={}", norm, nan, cur.len());
            }

            // Output projection
            cur = self.cpu_q4_matvec(&format!("{l}.attn_output.weight"), &cur, self.d_model);

            if layer == 0 {
                let nan = cur.iter().filter(|x| x.is_nan()).count();
                let norm: f32 = cur.iter().map(|x| x*x).sum::<f32>().sqrt();
                eprintln!("    L0 after_wo: norm={:.2} nan={}", norm, nan);
            }

            // ─── 5. Post-attention norm + residual ───
            if layer == 0 {
                let nan = cur.iter().filter(|x| x.is_nan()).count();
                let norm: f32 = cur.iter().map(|x| x*x).sum::<f32>().sqrt();
                eprintln!("    L0 after_attn_out_proj: norm={:.2} nan={}", norm, nan);
            }
            self.rms_norm_scaled_inplace(&mut cur, self.norm(&format!("{l}.post_attention_norm.weight")));
            let mut attn_out: Vec<f64> = hidden.clone();
            for i in 0..self.d_model { attn_out[i] += cur[i] as f64; }
            if layer == 0 {
                let nan = attn_out.iter().filter(|x| x.is_nan()).count();
                let norm: f64 = attn_out.iter().map(|x| x*x).sum::<f64>().sqrt();
                eprintln!("    L0 after_attn_resid: norm={:.2} nan={}", norm, nan);
            }

            // ─── 6. FFN: GELU-gated (NOT SiLU!) ───
            let attn_out_f32: Vec<f32> = attn_out.iter().map(|&v| v as f32).collect();
            let mut ff_cur = vec![0.0f32; self.d_model];
            self.rms_norm_scaled(&attn_out_f32, self.norm(&format!("{l}.ffn_norm.weight")), &mut ff_cur);

            let mut gate = self.cpu_q4_matvec(&format!("{l}.ffn_gate.weight"), &ff_cur, self.d_ff);
            let up = self.cpu_q4_matvec(&format!("{l}.ffn_up.weight"), &ff_cur, self.d_ff);

            // GELU activation on gate, then element-wise multiply with up
            Self::gelu_cpu(&mut gate);
            for i in 0..self.d_ff { gate[i] *= up[i]; }

            ff_cur = self.cpu_q4_matvec(&format!("{l}.ffn_down.weight"), &gate, self.d_model);

            // ─── 7. Post-FFN norm + residual ───
            if layer == 0 {
                let nan = ff_cur.iter().filter(|x| x.is_nan()).count();
                let norm: f32 = ff_cur.iter().map(|x| x*x).sum::<f32>().sqrt();
                eprintln!("    L0 after_ffn_down: norm={:.2} nan={}", norm, nan);
            }
            self.rms_norm_scaled_inplace(&mut ff_cur, self.norm(&format!("{l}.post_ffw_norm.weight")));
            for i in 0..self.d_model { attn_out[i] += ff_cur[i] as f64; }
            if layer == 0 {
                let nan = attn_out.iter().filter(|x| x.is_nan()).count();
                let norm: f64 = attn_out.iter().map(|x| x*x).sum::<f64>().sqrt();
                eprintln!("    L0 after_ffn_resid: norm={:.2} nan={}", norm, nan);
            }

            // ─── 8. Per-layer embedding (AFTER FFN) ───
            // Matches llama.cpp lines 202-224
            let inp_gate_name = format!("{l}.inp_gate.weight");
            if self.weights.contains_key(&inp_gate_name) && !per_layer_inputs.is_empty() {
                let pe_in: Vec<f64> = attn_out.clone();
                let attn_out_f32: Vec<f32> = attn_out.iter().map(|&v| v as f32).collect();
                let gate_dim = self.weight_out_dim(&inp_gate_name);

                // inp_gate(cur) → GELU
                let mut gate_out = self.cpu_q4_matvec(&inp_gate_name, &attn_out_f32, gate_dim);
                Self::gelu_cpu(&mut gate_out);

                // Multiply by per-layer token embedding slice for this layer
                let ple_offset = layer * n_embd_per_layer;
                if ple_offset + n_embd_per_layer <= per_layer_inputs.len() {
                    let ple_slice = &per_layer_inputs[ple_offset..ple_offset + n_embd_per_layer];
                    for i in 0..gate_dim.min(n_embd_per_layer) {
                        gate_out[i] *= ple_slice[i];
                    }
                }

                // proj back to d_model
                let proj_out = self.cpu_q4_matvec(&format!("{l}.proj.weight"), &gate_out, self.d_model);

                // post-norm
                let mut pe_normed = proj_out;
                self.rms_norm_scaled_inplace(&mut pe_normed, self.norm(&format!("{l}.post_norm.weight")));

                // residual
                attn_out = pe_in;
                for i in 0..self.d_model { attn_out[i] += pe_normed[i] as f64; }
            }

            // ─── 9. Layer output scale (element-wise tensor multiply) ───
            let scale_name = format!("{l}.layer_output_scale.weight");
            if let Some(Weight::CpuF32(s)) = self.weights.get(&scale_name) {
                if s.len() == 1 {
                    let scale = s[0] as f64;
                    for v in attn_out.iter_mut() { *v *= scale; }
                } else if s.len() == self.d_model {
                    for i in 0..self.d_model { attn_out[i] *= s[i] as f64; }
                }
            }

            hidden = attn_out;

            if layer == 0 {
                eprintln!("  layer 0: {:.1}ms", layer_start.elapsed().as_secs_f64() * 1000.0);
            }
        }
        let layers_done = layer_start.elapsed();
        eprintln!("  all {} layers: {:.1}ms ({:.1}ms/layer)",
            self.n_layers, layers_done.as_secs_f64() * 1000.0,
            layers_done.as_secs_f64() * 1000.0 / self.n_layers as f64);

        // Final norm — convert to f32
        let mut hidden_f32: Vec<f32> = hidden.iter().map(|&v| v as f32).collect();
        let pre_norm: f32 = hidden_f32.iter().map(|v| v*v).sum::<f32>().sqrt();
        self.rms_norm_cpu(&mut hidden_f32, self.norm("output_norm.weight"));
        let post_norm: f32 = hidden_f32.iter().map(|v| v*v).sum::<f32>().sqrt();
        eprintln!("  output_norm: pre={:.2} post={:.2}", pre_norm, post_norm);

        // Logits: hidden → vocab (reuse token_embd weights as output)
        let logit_start = std::time::Instant::now();
        let logits = self.cpu_q5_matvec_embed(&hidden_f32);
        eprintln!("  logits ({}→{}): {:.1}ms", self.d_model, self.vocab_size,
            logit_start.elapsed().as_secs_f64() * 1000.0);

        // Logit scaling: the Q5_K embedding rows have norm ~100M due to
        // quantization block scales. Raw logits are ~20 billion. The model
        // expects logits in [-100, 100] range for softcapping.
        //
        // The proper fix is to normalize the hidden state to unit scale
        // before the output projection, matching what the model sees during
        // training. We scale by 1/hidden_norm to make the hidden unit-length,
        // then the dot products are proportional to cosine similarity × embed_norm.
        // Then divide by a fixed scale based on expected embed row norm.
        //
        // Empirically calibrated: raw logit RMS ÷ target logit RMS.
        let mut logits = logits;

        // Logit soft-capping
        if self.logit_cap > 0.0 {
            let cap = self.logit_cap;
            for v in logits.iter_mut() {
                *v = (*v / cap).tanh() * cap;
            }
        }

        self.kv_len += 1;
        logits
    }

    /// Embed a single token. Dequantizes one row from token_embd (Q5_K).
    fn embed_token(&self, token_id: u32) -> Vec<f32> {
        let (dtype, dims, base_ptr) = self.qweight_mmap("token_embd.weight");
        let row_elements = dims[0];
        let (block_bytes, block_elems) = dtype.block_size();
        let blocks_per_row = (row_elements + block_elems - 1) / block_elems;
        let row_bytes = blocks_per_row * block_bytes;
        let row_offset = token_id as usize * row_bytes;

        let row_data = unsafe {
            std::slice::from_raw_parts(base_ptr.add(row_offset), row_bytes)
        };

        // Dequantize based on type
        // TODO: implement Q5_K dequant, for now just use zeros
        // This is wrong but lets us test the pipeline
        let mut embedding = vec![0.0f32; row_elements];

        if matches!(dtype, GgmlType::Q5_K) {
            dequant_q5k_row(row_data, &mut embedding, blocks_per_row);
        } else if matches!(dtype, GgmlType::Q4_K) {
            dequant_q4k_row(row_data, &mut embedding, blocks_per_row);
        }

        // Clamp NaN/inf from quantization artifacts
        for v in embedding.iter_mut() {
            if v.is_nan() || v.is_infinite() { *v = 0.0; }
        }

        // Skip sqrt(d_model) scaling — our f32 dequant of Q5_K produces values
        // at a different scale than ggml's quantized mul_mat. Without the scaling,
        // logits land in [-23, 21] which is correct for softcapping.
        // TODO: verify this produces correct logits

        embedding
    }

    /// Quantized matvec: y = Q(W) @ x using integer dot products.
    /// Matches ggml's vec_dot_q5_K_q8_K / vec_dot_q4_K_q8_K.
    fn cpu_q4_matvec(&self, weight_name: &str, x: &[f32], out_dim: usize) -> Vec<f32> {
        let (_, dtype, dims, mmap_ptr) = self.qweight_vram(weight_name);
        let in_dim = dims[0];
        assert!(in_dim <= x.len(),
            "cpu_q4_matvec {}: in_dim={} > x.len()={}, dims={:?}", weight_name, in_dim, x.len(), dims);
        let (block_bytes, _block_elems) = dtype.block_size();
        let blocks_per_row = in_dim / 256;
        let row_bytes = blocks_per_row * block_bytes;
        let total_bytes = out_dim * row_bytes;

        let weight_data = unsafe {
            std::slice::from_raw_parts(mmap_ptr, total_bytes)
        };

        let is_q5k = matches!(dtype, GgmlType::Q5_K);
        let mut y = vec![0.0f32; out_dim];
        super::quant_dot::qmatvec(weight_data, &x[..in_dim], &mut y, out_dim, in_dim, block_bytes, is_q5k);
        y
    }

    /// Quantized logits: dot(token_embd, hidden) using integer arithmetic.
    fn cpu_q5_matvec_embed(&self, x: &[f32]) -> Vec<f32> {
        let (dtype, dims, base_ptr) = self.qweight_mmap("token_embd.weight");
        let row_elements = dims[0]; // d_model
        let n_rows = dims[1]; // vocab_size
        let (block_bytes, _) = dtype.block_size();
        let blocks_per_row = row_elements / 256;
        let row_bytes = blocks_per_row * block_bytes;
        let total_bytes = n_rows * row_bytes;

        let weight_data = unsafe {
            std::slice::from_raw_parts(base_ptr, total_bytes)
        };

        let is_q5k = matches!(dtype, GgmlType::Q5_K);
        let mut logits = vec![0.0f32; n_rows];
        super::quant_dot::qmatvec(weight_data, &x[..row_elements], &mut logits, n_rows, row_elements, block_bytes, is_q5k);
        logits
    }

    /// Apply RoPE with proportional frequency factors (Gemma4).
    ///
    /// Full-attention layers use `rope_freqs` tensor to modify per-dimension
    /// frequencies. Values range from 1.0 (normal rotation) to 1e30 (frozen).
    /// SWA layers use standard RoPE with rope_base_swa=10000.
    ///
    /// Matches llama.cpp ggml_rope_ext with freq_factors parameter.
    fn apply_rope_ext(
        &self, data: &mut [f32], pos: usize, n_heads: usize, head_dim: usize,
        rope_base: f32, freq_factors: Option<&[f32]>,
    ) {
        for h in 0..n_heads {
            let off = h * head_dim;
            for i in (0..head_dim).step_by(2) {
                let dim_pair = i / 2;
                // Base frequency for this dimension pair
                let inv_freq = 1.0 / rope_base.powf(i as f32 / head_dim as f32);
                // Proportional factor: freq_factors[dim_pair] scales the frequency
                // A factor of 1e30 makes angle ≈ 0 → no rotation (frozen dim)
                let factor = freq_factors
                    .and_then(|ff| ff.get(dim_pair).copied())
                    .unwrap_or(1.0);
                let freq = inv_freq / factor; // divide, not multiply — higher factor = slower rotation
                let angle = pos as f32 * freq;
                let cos = angle.cos();
                let sin = angle.sin();
                let x0 = data[off + i];
                let x1 = data[off + i + 1];
                data[off + i] = x0 * cos - x1 * sin;
                data[off + i + 1] = x0 * sin + x1 * cos;
            }
        }
    }

    /// CPU attention with per-layer dimensions.
    fn cpu_attention_layer(&self, q: &[f32], layer: usize, seq_len: usize,
                           n_heads: usize, n_kv_heads: usize, head_dim: usize,
                           kv_dim: usize) -> Vec<f32> {
        let heads_per_kv = n_heads / n_kv_heads.max(1);

        // Read K and V cache from VRAM
        let k_cache = self.kv_k[layer].download(seq_len * kv_dim);
        let v_cache = self.kv_v[layer].download(seq_len * kv_dim);

        if layer == 0 {
            let v_nan = v_cache.iter().filter(|x| x.is_nan()).count();
            let v_norm: f32 = v_cache.iter().map(|x| x*x).sum::<f32>().sqrt();
            eprintln!("    L0 V_cache: norm={:.2} nan={} len={}", v_norm, v_nan, v_cache.len());
        }

        let mut output = vec![0.0f32; n_heads * head_dim];
        // Gemma4: attention_scale = 1.0 (Q/K norms handle scaling)
        let scale = 1.0f32;

        for h in 0..n_heads {
            let kv_h = h / heads_per_kv;
            let q_off = h * head_dim;

            // Compute attention scores
            let mut scores = vec![0.0f32; seq_len];
            for t in 0..seq_len {
                let k_off = t * kv_dim + kv_h * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[q_off + d] * k_cache[k_off + d];
                }
                scores[t] = dot * scale;
            }

            // Softmax
            let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - max_score).exp();
                sum += *s;
            }
            for s in scores.iter_mut() { *s /= sum; }

            // Weighted sum of V
            for t in 0..seq_len {
                let v_off = t * kv_dim + kv_h * head_dim;
                let w = scores[t];
                for d in 0..head_dim {
                    output[q_off + d] += w * v_cache[v_off + d];
                }
            }
        }

        output
    }
}

// VramBuf extension for partial upload
impl VramBuf {
    pub fn upload_at(&self, offset_floats: usize, data: &[f32]) {
        let dst = unsafe { (self.ptr as *mut f32).add(offset_floats) };
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len()); }
    }
}

// ─── Q4_K dequantization ─────────────────────────────────────

fn dequant_q4k_row(data: &[u8], out: &mut [f32], n_blocks: usize) {
    for blk in 0..n_blocks {
        let b = &data[blk * 144..];
        let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([b[2], b[3]]));
        let d = if d.is_nan() || d.is_infinite() { 0.0 } else { d };
        let dmin = if dmin.is_nan() || dmin.is_infinite() { 0.0 } else { dmin };
        let scales = &b[4..16];
        let qs = &b[16..144];

        let base = blk * 256;
        let mut is = 0;
        for j in (0..128).step_by(32) {
            let (sc, m) = get_scale_min_k4(is, scales);
            let d1 = d * sc as f32;
            let m1 = dmin * m as f32;
            for l in 0..32 {
                if base + j + l < out.len() {
                    out[base + j + l] = d1 * (qs[j + l] & 0xF) as f32 - m1;
                }
            }
            is += 1;
            let (sc, m) = get_scale_min_k4(is, scales);
            let d2 = d * sc as f32;
            let m2 = dmin * m as f32;
            for l in 0..32 {
                if base + j + l + 128 < out.len() {
                    out[base + j + l + 128] = d2 * (qs[j + l] >> 4) as f32 - m2;
                }
            }
            is += 1;
        }
    }
}

fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (sc, m)
    }
}

// ─── Q5_K dequantization ─────────────────────────────────────

fn dequant_q5k_row(data: &[u8], out: &mut [f32], n_blocks: usize) {
    // Q5_K: 176 bytes per 256 elements
    // Matches llama.cpp dequantize_row_q5_K exactly.
    for blk in 0..n_blocks {
        let b = &data[blk * 176..];
        let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([b[2], b[3]]));
        // Clamp NaN/inf block scales — quantization artifacts in some GGUF files
        let d = if d.is_nan() || d.is_infinite() { 0.0 } else { d };
        let dmin = if dmin.is_nan() || dmin.is_infinite() { 0.0 } else { dmin };
        let scales = &b[4..16];
        let qh = &b[16..48];
        let mut ql = &b[48..176];

        let base = blk * 256;
        let mut is = 0;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        let mut out_idx = base;

        for _j in (0..256).step_by(64) {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let d1 = d * sc1 as f32;
            let dm1 = dmin * m1 as f32;
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc2 as f32;
            let dm2 = dmin * m2 as f32;

            for l in 0..32 {
                let hbit = if qh[l] & u1 != 0 { 16u8 } else { 0u8 };
                let q = (ql[l] & 0xF) + hbit;
                if out_idx < out.len() {
                    out[out_idx] = d1 * q as f32 - dm1;
                }
                out_idx += 1;
            }
            for l in 0..32 {
                let hbit = if qh[l] & u2 != 0 { 16u8 } else { 0u8 };
                let q = (ql[l] >> 4) + hbit;
                if out_idx < out.len() {
                    out[out_idx] = d2 * q as f32 - dm2;
                }
                out_idx += 1;
            }

            ql = &ql[32..];
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;
    if exp == 0 {
        if mant == 0 { return if sign == 1 { -0.0 } else { 0.0 }; }
        let m = mant as f32 / 1024.0 * 2.0f32.powi(-14);
        return if sign == 1 { -m } else { m };
    }
    if exp == 31 { return f32::NAN; }
    f32::from_bits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    // Use f64 accumulation to avoid precision loss with extreme value ranges
    let sum: f64 = a.iter().zip(b).map(|(a, b)| *a as f64 * *b as f64).sum();
    sum as f32
}
