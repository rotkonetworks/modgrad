//! LLM inference as a pure function.
//!
//! `Gemma4Model` was a monolith — load + state + forward all tangled
//! together, with Gemma-specific quirks baked into every routine. Adding
//! Llama meant copy-paste-edit. This module factors the model into:
//!
//!   - `LlmConfig`  — architecture knobs (norm style, softcap, etc.).
//!     Same struct works for Gemma, Llama, Qwen, Mistral, …
//!   - `LlmWeights` — the bag of `Weight` entries, keyed by tensor name.
//!   - `LlmCache`   — per-layer K/V buffers + current sequence length.
//!   - `forward_token(cfg, weights, cache, dev, q, token) -> Vec<f32>`
//!     stateless function: same input → same logits, no hidden state.
//!
//! Adding a new architecture = filling out a `LlmConfig` + a tensor-name
//! resolver. No new forward path.
//!
//! This is currently a sketch. Once the API shape is agreed, we wire
//! the body using the same VRAM buffers + matvec dispatch the existing
//! Gemma4Model uses, then port Gemma4Model on top of it as a
//! compatibility layer.

use super::HsaDevice;
use super::dispatch_queue::{GpuQueue, VramBuf};
use super::gguf::{self, GgmlType, GgufFile};
#[allow(unused_imports)]
use GgufFile as _GgufFile;  // used in LlmWeights::load_from_gguf signature
use std::collections::HashMap;
use rayon::prelude::*;

// ─── Architecture knobs ─────────────────────────────────────────────────────

/// Which side of the residual gets the norm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormKind {
    /// `x / rms(x) * weight` — Llama-3, Qwen, Mistral.
    Plain,
    /// `x / rms(x) * (1 + weight)` — Gemma-2/3/4.
    GemmaPlusOne,
}

/// Which FFN flavour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlpKind {
    /// `down(silu(gate(x)) * up(x))` — Llama, Gemma, Qwen, Mistral.
    SwiGlu,
    /// `down(gelu(up(x)))` — older Llama-1, GPT.
    GeGlu,
}

/// Optional per-architecture quirks.
#[derive(Debug, Clone)]
pub struct LlmConfig {
    pub n_layers: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,

    pub rms_eps: f32,
    pub rope_base: f32,

    pub norm_kind: NormKind,
    pub mlp_kind: MlpKind,

    /// Soft-cap on attention scores: `score = cap · tanh(score/cap)`.
    /// `None` disables (Llama). `Some(50.0)` for Gemma-2/3/4.
    pub attn_softcap: Option<f32>,

    /// Soft-cap on final logits. `None` disables (Llama).
    pub logit_softcap: Option<f32>,

    /// Per-head QK-norm (Gemma-only). When true, expect
    /// `blk.N.attn_q_norm.weight` + `blk.N.attn_k_norm.weight` tensors.
    pub qk_norm: bool,

    /// Per-layer model projection (Gemma's PLE pathway). When true,
    /// expects `per_layer_token_embd` + `per_layer_model_proj`.
    pub per_layer_inputs: bool,

    /// Sliding-window mask pattern. Empty = full attention everywhere.
    /// Length must equal `n_layers` if non-empty.
    pub is_swa: Vec<bool>,
    pub rope_base_swa: f32,

    /// First N layers compute K/V; remaining reuse the last computed
    /// (Gemma's `attention.shared_kv_layers`). `n_layer_kv_from_start ==
    /// n_layers` means "every layer owns its KV" (Llama).
    pub n_layer_kv_from_start: usize,

    /// Llama-3.1+ rope frequency scaling (None = original Llama-3 RoPE).
    /// See `apply_rope` for the formula. Orpheus 3B has:
    ///   factor=32, low_freq_factor=1, high_freq_factor=4,
    ///   original_max_position_embeddings=8192
    pub rope_scaling: Option<RopeScaling>,
}

#[derive(Debug, Clone, Copy)]
pub struct RopeScaling {
    pub factor: f32,
    pub low_freq_factor: f32,
    pub high_freq_factor: f32,
    pub original_max_position_embeddings: u32,
}

impl LlmConfig {
    /// Llama-3 family preset. Fill `n_layers/d_model/...` from GGUF.
    pub fn llama3() -> Self {
        Self {
            n_layers: 0, d_model: 0, d_ff: 0, n_heads: 0, n_kv_heads: 0,
            head_dim: 0, vocab_size: 0,
            rms_eps: 1e-5,
            rope_base: 500_000.0,
            rope_scaling: None,
            norm_kind: NormKind::Plain,
            mlp_kind: MlpKind::SwiGlu,
            attn_softcap: None,
            logit_softcap: None,
            qk_norm: false,
            per_layer_inputs: false,
            is_swa: Vec::new(),
            rope_base_swa: 0.0,
            n_layer_kv_from_start: 0,  // = n_layers after fill
        }
    }

    /// Gemma-4 family preset.
    pub fn gemma4() -> Self {
        Self {
            n_layers: 0, d_model: 0, d_ff: 0, n_heads: 0, n_kv_heads: 0,
            head_dim: 0, vocab_size: 0,
            rms_eps: 1e-6,
            rope_base: 1_000_000.0,
            norm_kind: NormKind::GemmaPlusOne,
            mlp_kind: MlpKind::SwiGlu,
            attn_softcap: Some(50.0),
            logit_softcap: Some(30.0),
            qk_norm: true,
            per_layer_inputs: true,
            is_swa: Vec::new(),  // populated from sliding_window_pattern
            rope_base_swa: 10_000.0,
            n_layer_kv_from_start: 0,
            rope_scaling: None,
        }
    }

    /// Read arch params from GGUF metadata. Caller picks the family
    /// preset; this fills in the dimensional fields.
    pub fn populate_from_gguf(&mut self, gguf: &super::gguf::GgufFile) -> Result<(), String> {
        let arch = gguf.architecture().ok_or("no general.architecture")?;
        self.n_layers = gguf.meta_u32(&format!("{arch}.block_count"))
            .ok_or_else(|| format!("missing {arch}.block_count"))? as usize;
        self.d_model = gguf.meta_u32(&format!("{arch}.embedding_length"))
            .ok_or_else(|| format!("missing {arch}.embedding_length"))? as usize;
        self.d_ff = gguf.meta_u32(&format!("{arch}.feed_forward_length"))
            .ok_or_else(|| format!("missing {arch}.feed_forward_length"))? as usize;
        self.n_heads = gguf.meta_u32(&format!("{arch}.attention.head_count"))
            .ok_or_else(|| format!("missing {arch}.attention.head_count"))? as usize;
        self.n_kv_heads = gguf.meta_u32(&format!("{arch}.attention.head_count_kv"))
            .unwrap_or(self.n_heads as u32) as usize;
        // head_dim: prefer explicit key_length, else divide d_model by n_heads.
        self.head_dim = gguf.meta_u32(&format!("{arch}.attention.key_length"))
            .map(|k| k as usize)
            .unwrap_or(self.d_model / self.n_heads.max(1));
        self.rms_eps = gguf.meta_f32(&format!("{arch}.attention.layer_norm_rms_epsilon"))
            .unwrap_or(self.rms_eps);
        self.rope_base = gguf.meta_f32(&format!("{arch}.rope.freq_base"))
            .unwrap_or(self.rope_base);
        // vocab_size: prefer token_embd shape (ground truth), else metadata.
        self.vocab_size = gguf.tensors.get("token_embd.weight")
            .and_then(|t| t.dims.get(1).copied())
            .or_else(|| gguf.meta_u32(&format!("{arch}.vocab_size")).map(|v| v as usize))
            .ok_or("can't determine vocab_size")?;
        if self.n_layer_kv_from_start == 0 {
            self.n_layer_kv_from_start = self.n_layers;
        }
        Ok(())
    }
}

// ─── Weights + cache ────────────────────────────────────────────────────────

/// A weight tensor in VRAM (quantized), CPU-mmap'd, or CPU f32.
///
/// Lifted from `inference.rs` so both engines share the same storage.
/// Made `pub(super)` here so the existing Gemma4Model can also use this
/// type once we factor it out.
#[allow(dead_code)]
pub enum Weight {
    Vram {
        va: u64,
        dtype: GgmlType,
        dims: Vec<usize>,
        data_bytes: usize,
        file_offset: usize,
        buf: VramBuf,
    },
    Mmap {
        file_offset: usize,
        dtype: GgmlType,
        dims: Vec<usize>,
        data_bytes: usize,
    },
    /// 1-D CPU vector (used for RMSNorm scales).
    CpuF32(Vec<f32>),
    /// 2-D CPU matrix stored row-major as F32. Used when the model is loaded
    /// from FP16 safetensors instead of a quantized GGUF — bypasses the
    /// dequant path and gives bit-exact arithmetic at higher RAM cost.
    /// `dims[0]` is `in_dim`, `dims[1]` is `out_dim` (matvec semantics).
    CpuF32Mat {
        data: Vec<f32>,
        dims: Vec<usize>,
    },
}

/// Bag of weights keyed by GGUF tensor name. Same storage rules as
/// Gemma4Model (large embeddings stay mmap'd, everything else uploads
/// to VRAM).
pub struct LlmWeights {
    pub by_name: HashMap<String, Weight>,
    /// mmap base pointer for CPU fallback paths.
    pub file_data_ptr: *const u8,
}

impl LlmWeights {
    /// Walk every tensor in the GGUF and decide storage:
    ///   - F32 → CPU (norm scales)
    ///   - K-quant / Q8_0 / BF16 → VRAM (upload),
    ///     except embedding tables / huge tensors which stay mmap'd
    ///
    /// Matches `Gemma4Model::load` rules but architecture-agnostic.
    pub fn load_from_gguf(
        gguf: &GgufFile, file_data: &[u8],
        dev: &HsaDevice, queue: &mut GpuQueue,
    ) -> Result<Self, String> {
        let mut by_name = HashMap::new();
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
                    by_name.insert(name.clone(), Weight::CpuF32(f32s));
                }
                GgmlType::Q2_K | GgmlType::Q3_K | GgmlType::Q4_K | GgmlType::Q5_K
                | GgmlType::Q6_K | GgmlType::Q8_0 | GgmlType::BF16 => {
                    let data_bytes = info.data_bytes();

                    // Embedding tables: row-lookup not matvec — stay mmap'd.
                    // Anything > 100 MB also stays mmap'd to keep VRAM use sane.
                    let is_embedding = name.contains("embd") || name.contains("per_layer_model_proj");
                    if is_embedding || data_bytes > 100_000_000 {
                        by_name.insert(name.clone(), Weight::Mmap {
                            file_offset: file_off,
                            dtype: info.dtype, dims: info.dims.clone(), data_bytes,
                        });
                        continue;
                    }

                    let src = &file_data[file_off..file_off + data_bytes];
                    let n_floats = (data_bytes + 3) / 4;
                    let buf = queue.alloc(dev, n_floats)
                        .ok_or_else(|| format!("VRAM alloc failed for {} ({:.1} MB)",
                            name, data_bytes as f64 / 1e6))?;
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            src.as_ptr(), buf.ptr as *mut u8, data_bytes);
                    }
                    by_name.insert(name.clone(), Weight::Vram {
                        va: buf.va, dtype: info.dtype, dims: info.dims.clone(),
                        data_bytes, file_offset: file_off, buf,
                    });
                    total_vram += data_bytes;
                    n_uploaded += 1;
                }
                other => {
                    eprintln!("  skipping {name} (dtype {other:?})");
                }
            }
        }

        eprintln!("LlmWeights: uploaded {:.2} GB to VRAM ({} tensors)",
            total_vram as f64 / 1e9, n_uploaded);

        Ok(Self { by_name, file_data_ptr: file_data.as_ptr() })
    }
}

/// KV cache per layer + current sequence length.
///
/// **Plain CPU memory** — earlier version stored kv_k/kv_v in VRAM (VramBuf)
/// and downloaded the full cache through BAR-mapped reads every layer every
/// token. BAR reads on mobile GPUs are ~200 ns per float, which made
/// attention dominate (~45 MB of BAR traffic per token at seq 400). Plain
/// `Vec<f32>` reads stay in CPU cache, ~50× faster.
pub struct LlmCache {
    pub kv_k: Vec<Vec<f32>>,
    pub kv_v: Vec<Vec<f32>>,
    pub kv_len: usize,
    pub max_seq: usize,
}

/// Activation scratch buffers reused across layers. Allocated once at
/// `new()` time.
///
/// `matvec_x`, `matvec_y`, `matvec_bias` are GPU matvec scratch — sized
/// to the LARGEST input/output any layer needs (d_ff for input, vocab_size
/// for LM-head output). Reused for every matvec call so we don't pay KFD
/// alloc ioctl costs per call (196× per token, was hundreds of ms of pure
/// host overhead).
pub struct LlmScratch {
    pub hidden:   VramBuf,
    pub q:        VramBuf,
    pub k:        VramBuf,
    pub v:        VramBuf,
    pub attn_out: VramBuf,
    pub out:      VramBuf,
    pub ff_gate:  VramBuf,
    pub ff_up:    VramBuf,
    pub ff_down:  VramBuf,
    pub logits:   VramBuf,
    // Stage-1 resident matvec scratch ─────────────────────────────────
    pub matvec_x:    VramBuf,  // sized for max in_dim across all matvecs
    pub matvec_y:    VramBuf,  // sized for max out_dim (vocab_size for LM head)
    pub matvec_bias: VramBuf,  // pre-zeroed, sized like matvec_y
    // Per-matvec output scratches for BATCHED enqueue (Q+K+V flushed once,
    // gate+up flushed once). Each matvec writes to its own buffer so they
    // can run in parallel before the single sync barrier.
    pub mv_q:     VramBuf,
    pub mv_k:     VramBuf,
    pub mv_v:     VramBuf,
    pub mv_gate:  VramBuf,
    pub mv_up:    VramBuf,
}

impl LlmScratch {
    pub fn allocate(cfg: &LlmConfig, dev: &HsaDevice, q: &mut GpuQueue) -> Result<Self, String> {
        let q_dim = cfg.n_heads * cfg.head_dim;
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;

        // The largest matvec input across the model: ffn_down reads d_ff.
        // The largest output: the LM head writes vocab_size.
        let max_in  = cfg.d_ff.max(cfg.d_model).max(q_dim);
        let max_out = cfg.vocab_size.max(cfg.d_ff).max(cfg.d_model);

        let matvec_x    = q.alloc(dev, max_in).ok_or("alloc matvec_x")?;
        let matvec_y    = q.alloc(dev, max_out).ok_or("alloc matvec_y")?;
        let matvec_bias = q.alloc(dev, max_out).ok_or("alloc matvec_bias")?;
        matvec_bias.zero();
        // Per-matvec output scratches for batching. Size each to the max
        // possible output it'll see across all layers (so reused across
        // every block). Q/K/V are bounded by q_dim; gate/up by d_ff.
        let mv_q    = q.alloc(dev, q_dim).ok_or("alloc mv_q")?;
        let mv_k    = q.alloc(dev, kv_dim).ok_or("alloc mv_k")?;
        let mv_v    = q.alloc(dev, kv_dim).ok_or("alloc mv_v")?;
        let mv_gate = q.alloc(dev, cfg.d_ff).ok_or("alloc mv_gate")?;
        let mv_up   = q.alloc(dev, cfg.d_ff).ok_or("alloc mv_up")?;

        Ok(Self {
            hidden:   q.alloc(dev, cfg.d_model).ok_or("alloc hidden")?,
            q:        q.alloc(dev, q_dim).ok_or("alloc q")?,
            k:        q.alloc(dev, kv_dim).ok_or("alloc k")?,
            v:        q.alloc(dev, kv_dim).ok_or("alloc v")?,
            attn_out: q.alloc(dev, q_dim).ok_or("alloc attn_out")?,
            out:      q.alloc(dev, cfg.d_model).ok_or("alloc out")?,
            ff_gate:  q.alloc(dev, cfg.d_ff).ok_or("alloc ff_gate")?,
            ff_up:    q.alloc(dev, cfg.d_ff).ok_or("alloc ff_up")?,
            ff_down:  q.alloc(dev, cfg.d_model).ok_or("alloc ff_down")?,
            logits:   q.alloc(dev, cfg.vocab_size).ok_or("alloc logits")?,
            matvec_x, matvec_y, matvec_bias,
            mv_q, mv_k, mv_v, mv_gate, mv_up,
        })
    }
}

impl LlmCache {
    pub fn allocate(cfg: &LlmConfig, _dev: &HsaDevice, _q: &mut GpuQueue, max_seq: usize)
        -> Result<Self, String>
    {
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        let mut kv_k = Vec::with_capacity(cfg.n_layers);
        let mut kv_v = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            // Pre-allocate the full max_seq buffer so position writes are
            // direct slice copies (no growth, no realloc, contiguous).
            kv_k.push(vec![0.0f32; max_seq * kv_dim]);
            kv_v.push(vec![0.0f32; max_seq * kv_dim]);
        }
        Ok(Self { kv_k, kv_v, kv_len: 0, max_seq })
    }
}

// ─── The function ───────────────────────────────────────────────────────────

// ─── Helpers (CPU paths first; GPU dispatch slots in later) ─────────────────

fn dequant_row_into(dtype: GgmlType, row: &[u8], out: &mut [f32], n_blocks: usize) {
    match dtype {
        GgmlType::Q4_K => gguf::dequantize_row_q4_k(row, out, n_blocks),
        GgmlType::Q5_K => gguf::dequantize_row_q5_k(row, out, n_blocks),
        GgmlType::Q6_K => gguf::dequantize_row_q6_k(row, out, n_blocks),
        other => panic!("dequant_row_into: unsupported dtype {other:?}"),
    }
}

impl LlmWeights {
    /// Look up an f32 weight (norm scales etc.). Panics if missing/wrong kind.
    pub fn f32(&self, name: &str) -> &[f32] {
        match self.by_name.get(name) {
            Some(Weight::CpuF32(v)) => v,
            _ => panic!("missing or non-f32 weight: {name}"),
        }
    }

    pub fn has(&self, name: &str) -> bool {
        self.by_name.contains_key(name)
    }

    fn qweight(&self, name: &str) -> (&GgmlType, &[usize], *const u8, usize) {
        match self.by_name.get(name) {
            Some(Weight::Vram { dtype, dims, file_offset, data_bytes, .. }) => {
                let ptr = unsafe { self.file_data_ptr.add(*file_offset) };
                (dtype, dims, ptr, *data_bytes)
            }
            Some(Weight::Mmap { dtype, dims, file_offset, data_bytes }) => {
                let ptr = unsafe { self.file_data_ptr.add(*file_offset) };
                (dtype, dims, ptr, *data_bytes)
            }
            _ => panic!("missing quantized weight: {name}"),
        }
    }

    /// Get VRAM-resident weight info INCLUDING the VRAM buffer reference,
    /// for direct GPU kernel dispatch. Returns None if the weight is
    /// mmap'd or not present, or if dtype is not GPU-kernel-supported.
    fn vram_q4k(&self, name: &str) -> Option<(&VramBuf, &[usize])> {
        match self.by_name.get(name) {
            Some(Weight::Vram { buf, dtype, dims, .. }) if matches!(dtype, GgmlType::Q4_K) => {
                Some((buf, dims))
            }
            _ => None,
        }
    }
}

/// Quantized matrix-vector: y = W · x, where W is [in=dims[0], out=dims[1]]
/// stored row-by-row.
///
/// **CPU dequant + rayon dot products, always.** GPU `matvec_q4k` dispatch
/// adds 5-50ms per-call sync overhead × 196 matvecs/token = pure slowdown.
/// Until full GPU residency, CPU wins on this hardware at batch=1.
///
/// Q4_K hot path is FUSED: per row we walk blocks reading 144 weight bytes
/// at a time and accumulate the dot directly, with NO heap allocation and
/// NO separate dequant pass. The original code allocated a `Vec<f32>` per
/// row (3072 floats × 3072 rows × 196 matvecs = ~1.9B allocs amortised)
/// and read its data back from memory after writing — the cache traffic was
/// dominating the matvec cost on this CPU.
fn matvec(
    weights: &LlmWeights, name: &str, x: &[f32],
    _dev: &mut HsaDevice, _queue: &mut GpuQueue,
    _scratch: &LlmScratch,
) -> Vec<f32> {
    // F32 matrix fast-path (used when model is loaded from FP16 safetensors).
    // No dequant, just plain dot products with AVX-512 SIMD via auto-vectorization.
    if let Some(Weight::CpuF32Mat { data, dims }) = weights.by_name.get(name) {
        let in_dim = dims[0];
        let out_dim = dims[1];
        assert_eq!(data.len(), in_dim * out_dim, "F32Mat {name} size mismatch");
        let xin = &x[..in_dim];
        let mut y = vec![0.0f32; out_dim];
        y.par_iter_mut().enumerate().for_each(|(r, yr)| {
            let row = &data[r * in_dim..(r + 1) * in_dim];
            // f64 accumulator to match the quant path's precision.
            let mut acc = 0.0f64;
            for k in 0..in_dim { acc += row[k] as f64 * xin[k] as f64; }
            *yr = acc as f32;
        });
        return y;
    }

    let (dtype, dims, base_ptr, _) = weights.qweight(name);
    let in_dim = dims[0];
    let out_dim = dims[1];
    let (block_bytes, block_elems) = dtype.block_size();
    let blocks_per_row = (in_dim + block_elems - 1) / block_elems;
    let row_bytes = blocks_per_row * block_bytes;
    let dt = *dtype;

    let total = out_dim * row_bytes;
    let raw = unsafe { std::slice::from_raw_parts(base_ptr, total) };
    let xin = &x[..in_dim];

    let mut y = vec![0.0f32; out_dim];
    match dt {
        // Q4_K fast path: fused dequant+dot, AVX-512 SIMD when available.
        GgmlType::Q4_K => {
            y.par_iter_mut().enumerate().for_each(|(r, yr)| {
                let row = &raw[r * row_bytes..(r + 1) * row_bytes];
                *yr = dot_q4k_row(row, xin, blocks_per_row);
            });
        }
        // Q6_K fast path: scalar dequant to a stack buffer, then SIMD dot.
        // (Q6_K's bit-packing is too irregular to inline SIMD-dequant cleanly,
        // but the dequant is small enough that bandwidth/dot is what hurts.)
        GgmlType::Q6_K => {
            y.par_iter_mut().enumerate().for_each(|(r, yr)| {
                let row = &raw[r * row_bytes..(r + 1) * row_bytes];
                *yr = dot_q6k_row(row, xin, blocks_per_row);
            });
        }
        // Slow path for any other quant type: legacy dequant-then-dot.
        _ => {
            y.par_iter_mut().enumerate().for_each(|(r, yr)| {
                let row = &raw[r * row_bytes..(r + 1) * row_bytes];
                let mut w = vec![0.0f32; in_dim];
                dequant_row_into(dt, row, &mut w, blocks_per_row);
                let mut acc = 0.0f64;
                for k in 0..in_dim { acc += w[k] as f64 * xin[k] as f64; }
                *yr = acc as f32;
            });
        }
    }
    y
}

/// Q6_K dot product. 210 bytes / 256 weights per block. We dequant the block
/// scalar into a stack buffer (fast — 256 muls + ~256 adds; no heap alloc)
/// then run an AVX-512 dot over the 256 floats × xin slice.
#[inline]
fn dot_q6k_row(row: &[u8], xin: &[f32], n_blocks: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma") {
            return unsafe { dot_q6k_row_avx512(row, xin, n_blocks) };
        }
    }
    // Fallback: full dequant + scalar dot.
    let mut w = vec![0.0f32; n_blocks * 256];
    super::gguf::dequantize_row_q6_k(row, &mut w, n_blocks);
    let mut acc = 0.0f64;
    for k in 0..n_blocks * 256 { acc += w[k] as f64 * xin[k] as f64; }
    acc as f32
}

/// AVX-512 fused Q6_K dequant+dot. Replaces the scalar block-dequant in the
/// earlier version (which was the remaining hotspot on Q6_K tensors: LM head,
/// V projection, FFN down).
///
/// Q6_K block layout (210 bytes / 256 weights):
///   ql[0..128]   — low 4 bits of each weight
///   qh[128..192] — high 2 bits (4 weights packed per byte)
///   sc[192..208] — 16 signed i8 scales
///   d[208..210]  — fp16 master scale
///
/// Reference dequant pattern (per group g ∈ {0,1}, l ∈ 0..32):
///   q1 = ((ql[g*64+l]    & 0x0F)   | ((qh[g*32+l]      & 3) << 4)) - 32
///   q2 = ((ql[g*64+l+32] & 0x0F)   | ((qh[g*32+l] >> 2 & 3) << 4)) - 32
///   q3 = ((ql[g*64+l]      >> 4)   | ((qh[g*32+l] >> 4 & 3) << 4)) - 32
///   q4 = ((ql[g*64+l+32]   >> 4)   | ((qh[g*32+l] >> 6 & 3) << 4)) - 32
///   out[g*128 + l    ] = d · sc[g*8 + l/16    ] · q1
///   out[g*128 + l + 32] = d · sc[g*8 + l/16 + 2] · q2
///   out[g*128 + l + 64] = d · sc[g*8 + l/16 + 4] · q3
///   out[g*128 + l + 96] = d · sc[g*8 + l/16 + 6] · q4
///
/// We vectorize across l in chunks of 16. Each chunk: 16 nibbles + 16 hbit
/// pairs → 16 signed weights → 16-float fmadd against x. 8 chunks per group ×
/// 2 groups = 16 chunks per block (256 weights).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,fma")]
unsafe fn dot_q6k_row_avx512(row: &[u8], xin: &[f32], n_blocks: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut acc = _mm512_setzero_ps();
    let mask4 = _mm512_set1_epi32(0x0F);
    let mask2 = _mm512_set1_epi32(0x03);
    let bias  = _mm512_set1_epi32(32);

    for blk in 0..n_blocks {
        let b_ptr = row.as_ptr().add(blk * 210);
        let d_raw = u16::from_le_bytes([*b_ptr.add(208), *b_ptr.add(209)]);
        let d = super::gguf::f16_to_f32(d_raw);
        let d = if d.is_finite() { d } else { 0.0 };

        let sc_ptr = b_ptr.add(192) as *const i8;
        // Two halves of the 256-weight block — each with its own qh stride.
        for grp in 0..2 {
            let ql_grp = b_ptr.add(grp * 64);
            let qh_grp = b_ptr.add(128 + grp * 32);
            let sc_grp = sc_ptr.add(grp * 8);
            let x_grp  = xin.as_ptr().add(blk * 256 + grp * 128);

            // For each `l_chunk` we cover 16 of the 32 l-values per q_id.
            for l_half in 0..2 {
                let l_base = l_half * 16;
                let scale_idx = l_half;
                // Load 16 bytes of ql[l_base..l_base+16]    → low-half source
                //  + 16 bytes of ql[l_base+32..l_base+48]   → high-half source
                let ql_low_b  = _mm_loadu_si128(ql_grp.add(l_base) as *const __m128i);
                let ql_high_b = _mm_loadu_si128(ql_grp.add(l_base + 32) as *const __m128i);
                let qh_b      = _mm_loadu_si128(qh_grp.add(l_base) as *const __m128i);

                // Expand each 16-byte vector into 16 i32 lanes for arithmetic.
                let ql_low  = _mm512_cvtepu8_epi32(ql_low_b);
                let ql_high = _mm512_cvtepu8_epi32(ql_high_b);
                let qh      = _mm512_cvtepu8_epi32(qh_b);

                // q1 = (ql_low & 0xF) | ((qh & 3) << 4); then - 32
                let q1 = _mm512_sub_epi32(
                    _mm512_or_epi32(
                        _mm512_and_epi32(ql_low, mask4),
                        _mm512_slli_epi32::<4>(_mm512_and_epi32(qh, mask2)),
                    ),
                    bias,
                );
                // q2 = (ql_high & 0xF) | ((qh >> 2 & 3) << 4); - 32
                let q2 = _mm512_sub_epi32(
                    _mm512_or_epi32(
                        _mm512_and_epi32(ql_high, mask4),
                        _mm512_slli_epi32::<4>(_mm512_and_epi32(_mm512_srli_epi32::<2>(qh), mask2)),
                    ),
                    bias,
                );
                // q3 = (ql_low >> 4) | ((qh >> 4 & 3) << 4); - 32
                let q3 = _mm512_sub_epi32(
                    _mm512_or_epi32(
                        _mm512_srli_epi32::<4>(ql_low),
                        _mm512_slli_epi32::<4>(_mm512_and_epi32(_mm512_srli_epi32::<4>(qh), mask2)),
                    ),
                    bias,
                );
                // q4 = (ql_high >> 4) | ((qh >> 6 & 3) << 4); - 32
                let q4 = _mm512_sub_epi32(
                    _mm512_or_epi32(
                        _mm512_srli_epi32::<4>(ql_high),
                        _mm512_slli_epi32::<4>(_mm512_and_epi32(_mm512_srli_epi32::<6>(qh), mask2)),
                    ),
                    bias,
                );

                let scl = |off: isize| _mm512_set1_ps(d * (*sc_grp.offset(scale_idx as isize + off)) as f32);
                let s1 = scl(0);
                let s2 = scl(2);
                let s3 = scl(4);
                let s4 = scl(6);

                let w1 = _mm512_mul_ps(_mm512_cvtepi32_ps(q1), s1);
                let w2 = _mm512_mul_ps(_mm512_cvtepi32_ps(q2), s2);
                let w3 = _mm512_mul_ps(_mm512_cvtepi32_ps(q3), s3);
                let w4 = _mm512_mul_ps(_mm512_cvtepi32_ps(q4), s4);

                let x1 = _mm512_loadu_ps(x_grp.offset(l_base as isize));
                let x2 = _mm512_loadu_ps(x_grp.offset(l_base as isize + 32));
                let x3 = _mm512_loadu_ps(x_grp.offset(l_base as isize + 64));
                let x4 = _mm512_loadu_ps(x_grp.offset(l_base as isize + 96));

                acc = _mm512_fmadd_ps(w1, x1, acc);
                acc = _mm512_fmadd_ps(w2, x2, acc);
                acc = _mm512_fmadd_ps(w3, x3, acc);
                acc = _mm512_fmadd_ps(w4, x4, acc);
            }
        }
    }
    _mm512_reduce_add_ps(acc)
}

/// Fused Q4_K dot product: `sum_i (W[i] * x[i])` where W is dequantized
/// inline from `row` (bytes) one block at a time. No heap allocation.
///
/// Block layout (144 bytes): 2 fp16 (d, dmin) + 12 packed scales + 128 q-nibbles.
/// 256 weights per block — 8 sub-blocks of 32, each with its own scale.
///
/// Dispatches to AVX-512 when available (10-20× the scalar throughput on this
/// CPU); falls through to the scalar reference otherwise.
#[inline]
fn dot_q4k_row(row: &[u8], xin: &[f32], n_blocks: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma") {
            return unsafe { dot_q4k_row_avx512(row, xin, n_blocks) };
        }
    }
    dot_q4k_row_scalar(row, xin, n_blocks)
}

#[inline]
fn dot_q4k_row_scalar(row: &[u8], xin: &[f32], n_blocks: usize) -> f32 {
    use super::gguf::{Q4K_BLOCK_BYTES, Q4K_BLOCK_ELEMS, q4k_get_scale_min_k4};
    let mut acc = 0.0f32;
    for blk in 0..n_blocks {
        let b = &row[blk * Q4K_BLOCK_BYTES..(blk + 1) * Q4K_BLOCK_BYTES];
        let d_raw = u16::from_le_bytes([b[0], b[1]]);
        let dmin_raw = u16::from_le_bytes([b[2], b[3]]);
        let d = super::gguf::f16_to_f32(d_raw);
        let dmin = super::gguf::f16_to_f32(dmin_raw);
        let d = if d.is_finite() { d } else { 0.0 };
        let dmin = if dmin.is_finite() { dmin } else { 0.0 };
        let scales = &b[4..16];
        let qs = &b[16..144];
        let xblk = &xin[blk * Q4K_BLOCK_ELEMS..(blk + 1) * Q4K_BLOCK_ELEMS];
        let mut is = 0;
        // 4 iterations × 64 weights each = 256 weights per block.
        for j in (0..128).step_by(32) {
            // Low nibbles, sub-block `is`
            let (sc, m) = q4k_get_scale_min_k4(is, scales);
            let d1 = d * sc as f32;
            let m1 = dmin * m as f32;
            let xs_lo = &xblk[(j/32) * 64..(j/32) * 64 + 32];
            for l in 0..32 {
                let w = d1 * (qs[j + l] & 0xF) as f32 - m1;
                acc += w * xs_lo[l];
            }
            is += 1;
            // High nibbles, sub-block `is`
            let (sc, m) = q4k_get_scale_min_k4(is, scales);
            let d2 = d * sc as f32;
            let m2 = dmin * m as f32;
            let xs_hi = &xblk[(j/32) * 64 + 32..(j/32) * 64 + 64];
            for l in 0..32 {
                let w = d2 * (qs[j + l] >> 4) as f32 - m2;
                acc += w * xs_hi[l];
            }
            is += 1;
        }
    }
    acc
}

/// AVX-512 Q4_K dot product. Per sub-block of 32 weights:
///   - Pull 32 nibbles (low or high half of 32 bytes from `qs`).
///   - Convert to two zmm registers of 16 fp32 each (`vcvtdq2ps`).
///   - Apply `scale * nibble - offset` with FMSUB.
///   - Multiply pointwise by 32 floats of `x`, accumulate in a 16-wide zmm acc.
///
/// 96 sub-blocks per row × 2 FMA-zmm each = 192 FMAs per row vs ~3072 scalar
/// fmas + 3072 scalar muls in the reference. ~16× ALU reduction; cache traffic
/// also drops since we never materialise the dequantised row.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,fma")]
unsafe fn dot_q4k_row_avx512(row: &[u8], xin: &[f32], n_blocks: usize) -> f32 {
    use super::gguf::{Q4K_BLOCK_BYTES, Q4K_BLOCK_ELEMS, q4k_get_scale_min_k4};
    use std::arch::x86_64::*;
    let mut sum_lo = _mm512_setzero_ps();
    let mut sum_hi = _mm512_setzero_ps();
    let mask_byte = _mm256_set1_epi8(0x0F);
    for blk in 0..n_blocks {
        let b_ptr = row.as_ptr().add(blk * Q4K_BLOCK_BYTES);
        let d_raw = u16::from_le_bytes([*b_ptr, *b_ptr.add(1)]);
        let dmin_raw = u16::from_le_bytes([*b_ptr.add(2), *b_ptr.add(3)]);
        let d = super::gguf::f16_to_f32(d_raw);
        let dmin = super::gguf::f16_to_f32(dmin_raw);
        let d = if d.is_finite() { d } else { 0.0 };
        let dmin = if dmin.is_finite() { dmin } else { 0.0 };
        let scales = std::slice::from_raw_parts(b_ptr.add(4), 12);
        let qs_ptr = b_ptr.add(16);
        let xptr = xin.as_ptr().add(blk * Q4K_BLOCK_ELEMS);

        let mut is: usize = 0;
        // 4 outer iterations: each consumes 32 bytes of qs and emits 64 weights
        // (32 low-nibble, 32 high-nibble — two sub-blocks).
        for j_blk in 0..4 {
            let j = j_blk * 32;
            let qs_chunk = _mm256_loadu_si256(qs_ptr.add(j) as *const __m256i);
            let nibbles_lo = _mm256_and_si256(qs_chunk, mask_byte);
            let nibbles_hi = _mm256_and_si256(_mm256_srli_epi16::<4>(qs_chunk), mask_byte);

            // Low-nibble sub-block: 32 weights at offset is*32 in the dequantised row.
            let (sc, m) = q4k_get_scale_min_k4(is, scales);
            let scale_v  = _mm512_set1_ps(d * sc as f32);
            let offset_v = _mm512_set1_ps(dmin * m as f32);
            let lo_p0 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm256_castsi256_si128(nibbles_lo)));
            let lo_p1 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm256_extracti128_si256::<1>(nibbles_lo)));
            let w0 = _mm512_fmsub_ps(lo_p0, scale_v, offset_v);
            let w1 = _mm512_fmsub_ps(lo_p1, scale_v, offset_v);
            let x_off = (j_blk * 64) as isize;
            let x0 = _mm512_loadu_ps(xptr.offset(x_off));
            let x1 = _mm512_loadu_ps(xptr.offset(x_off + 16));
            sum_lo = _mm512_fmadd_ps(w0, x0, sum_lo);
            sum_hi = _mm512_fmadd_ps(w1, x1, sum_hi);
            is += 1;

            // High-nibble sub-block: next 32 weights.
            let (sc, m) = q4k_get_scale_min_k4(is, scales);
            let scale_v  = _mm512_set1_ps(d * sc as f32);
            let offset_v = _mm512_set1_ps(dmin * m as f32);
            let hi_p0 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm256_castsi256_si128(nibbles_hi)));
            let hi_p1 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm256_extracti128_si256::<1>(nibbles_hi)));
            let w0 = _mm512_fmsub_ps(hi_p0, scale_v, offset_v);
            let w1 = _mm512_fmsub_ps(hi_p1, scale_v, offset_v);
            let x0 = _mm512_loadu_ps(xptr.offset(x_off + 32));
            let x1 = _mm512_loadu_ps(xptr.offset(x_off + 48));
            sum_lo = _mm512_fmadd_ps(w0, x0, sum_lo);
            sum_hi = _mm512_fmadd_ps(w1, x1, sum_hi);
            is += 1;
        }
    }
    _mm512_reduce_add_ps(_mm512_add_ps(sum_lo, sum_hi))
}

/// Batch three Q4_K matvecs (Q, K, V) that share the same input. One upload,
/// three enqueues, ONE flush (= one CPU↔GPU sync), three downloads.
///
/// Saves 2/3 of the per-matvec sync overhead. Returns None if all three
/// weights aren't Q4_K-in-VRAM (caller falls back to serial CPU/GPU).
fn batch_matvec_qkv(
    weights: &LlmWeights,
    q_name: &str, k_name: &str, v_name: &str, x: &[f32],
    dev: &mut HsaDevice, queue: &mut GpuQueue, scratch: &LlmScratch,
) -> Option<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let (wq, qdims) = weights.vram_q4k(q_name)?;
    let (wk, kdims) = weights.vram_q4k(k_name)?;
    let (wv, vdims) = weights.vram_q4k(v_name)?;

    let in_dim = qdims[0];
    let q_out  = qdims[1];
    let k_out  = kdims[1];
    let v_out  = vdims[1];
    let bpr = in_dim / 256;

    // Single upload of the shared input.
    scratch.matvec_x.upload(&x[..in_dim]);

    if !queue.enqueue_matvec_q4k(dev, wq, &scratch.matvec_x, &scratch.matvec_bias,
                                  &scratch.mv_q, q_out, bpr) { return None; }
    if !queue.enqueue_matvec_q4k(dev, wk, &scratch.matvec_x, &scratch.matvec_bias,
                                  &scratch.mv_k, k_out, bpr) { return None; }
    if !queue.enqueue_matvec_q4k(dev, wv, &scratch.matvec_x, &scratch.matvec_bias,
                                  &scratch.mv_v, v_out, bpr) { return None; }

    // ONE flush for all three.
    if !queue.flush(dev) { return None; }

    Some((scratch.mv_q.download(q_out),
          scratch.mv_k.download(k_out),
          scratch.mv_v.download(v_out)))
}

/// Batch two Q4_K matvecs (FFN gate, FFN up) sharing the same input.
fn batch_matvec_gate_up(
    weights: &LlmWeights,
    gate_name: &str, up_name: &str, x: &[f32],
    dev: &mut HsaDevice, queue: &mut GpuQueue, scratch: &LlmScratch,
) -> Option<(Vec<f32>, Vec<f32>)> {
    let (wg, gdims) = weights.vram_q4k(gate_name)?;
    let (wu, udims) = weights.vram_q4k(up_name)?;

    let in_dim = gdims[0];
    let g_out  = gdims[1];
    let u_out  = udims[1];
    let bpr = in_dim / 256;

    scratch.matvec_x.upload(&x[..in_dim]);

    if !queue.enqueue_matvec_q4k(dev, wg, &scratch.matvec_x, &scratch.matvec_bias,
                                  &scratch.mv_gate, g_out, bpr) { return None; }
    if !queue.enqueue_matvec_q4k(dev, wu, &scratch.matvec_x, &scratch.matvec_bias,
                                  &scratch.mv_up, u_out, bpr) { return None; }

    if !queue.flush(dev) { return None; }

    Some((scratch.mv_gate.download(g_out), scratch.mv_up.download(u_out)))
}

/// Q4_K matvec on the GPU using RESIDENT scratch buffers.
///
/// Per call: 1 upload (x), 1 dispatch, 1 submit, 1 wait, 1 download (y).
/// **Zero** VRAM allocations — `matvec_x`, `matvec_y`, `matvec_bias` were
/// allocated once at LlmScratch::allocate time and reused across all matvecs
/// in the entire forward pass.
fn gpu_matvec_q4k(
    dev: &mut HsaDevice, queue: &mut GpuQueue,
    w_buf: &VramBuf, x: &[f32],
    in_dim: usize, out_dim: usize, blocks_per_row: usize,
    scratch: &LlmScratch,
) -> Option<Vec<f32>> {
    scratch.matvec_x.upload(&x[..in_dim]);

    if !queue.enqueue_matvec_q4k(dev, w_buf, &scratch.matvec_x,
                                  &scratch.matvec_bias, &scratch.matvec_y,
                                  out_dim, blocks_per_row) {
        return None;
    }
    // flush = submit + wait + cache_wb so CPU sees the writes.
    if !queue.flush(dev) { return None; }
    Some(scratch.matvec_y.download(out_dim))
}

/// Embedding lookup: one row of `token_embd.weight` for the given token ID.
/// Returns `[d_model]` f32. Handles both quantized GGUF and F32 safetensors.
fn embed_token(weights: &LlmWeights, token: u32) -> Vec<f32> {
    // F32 safetensors path.
    if let Some(Weight::CpuF32Mat { data, dims }) = weights.by_name.get("token_embd.weight") {
        let d_model = dims[0];
        let off = token as usize * d_model;
        return data[off..off + d_model].to_vec();
    }
    // Quantized GGUF path.
    let (dtype, dims, base_ptr, _) = weights.qweight("token_embd.weight");
    let d_model = dims[0];
    let (block_bytes, block_elems) = dtype.block_size();
    let blocks_per_row = (d_model + block_elems - 1) / block_elems;
    let row_bytes = blocks_per_row * block_bytes;
    let row = unsafe {
        std::slice::from_raw_parts(base_ptr.add(token as usize * row_bytes), row_bytes)
    };
    let mut out = vec![0.0f32; d_model];
    dequant_row_into(*dtype, row, &mut out, blocks_per_row);
    out
}

/// RMSNorm: `dst = src / sqrt(mean(src²) + eps) · scale(weight)`.
/// `cfg.norm_kind` decides whether the scale is `w` (Llama) or `1+w` (Gemma).
fn rms_norm(src: &[f32], weight: &[f32], dst: &mut [f32], eps: f32, kind: NormKind) {
    let n = src.len() as f32;
    let ss: f32 = src.iter().map(|v| v * v).sum::<f32>() / n;
    let inv = 1.0 / (ss + eps).sqrt();
    match kind {
        NormKind::Plain => {
            for i in 0..src.len() { dst[i] = src[i] * inv * weight[i]; }
        }
        NormKind::GemmaPlusOne => {
            for i in 0..src.len() { dst[i] = src[i] * inv * (1.0 + weight[i]); }
        }
    }
}

/// NEOX-style RoPE (pair dim `i` with `i+half`, not adjacent `(i, i+1)`).
/// Used by Llama-3, Gemma, Qwen — basically all current decoder-only LLMs.
///
/// `scaling` applies the Llama-3.1+ frequency-aware adjustment:
///   low frequencies (long wavelength) are divided by `factor`,
///   high frequencies (short wavelength) are kept unscaled,
///   middle band is smoothly interpolated.
/// See Meta's "Llama-3.1 RoPE scaling" reference; this matches the formula
/// in HuggingFace transformers `modeling_rope_utils.py::_compute_llama3_parameters`.
fn apply_rope(
    data: &mut [f32], pos: usize,
    n_heads: usize, head_dim: usize, base: f32,
    scaling: Option<RopeScaling>,
) {
    let half = head_dim / 2;
    for h in 0..n_heads {
        let off = h * head_dim;
        for i in 0..half {
            let mut inv_freq = base.powf(-2.0 * i as f32 / head_dim as f32);

            if let Some(s) = scaling {
                // wavelength λ = 2π / inv_freq
                let wavelen = (2.0 * std::f32::consts::PI) / inv_freq;
                let low_wavelen  = s.original_max_position_embeddings as f32 / s.low_freq_factor;
                let high_wavelen = s.original_max_position_embeddings as f32 / s.high_freq_factor;
                if wavelen < high_wavelen {
                    // high freq, no scaling
                } else if wavelen > low_wavelen {
                    // low freq, divide by factor
                    inv_freq /= s.factor;
                } else {
                    // smooth band: smoothfactor goes 0 (high freq) → 1 (low freq)
                    let smooth = (s.original_max_position_embeddings as f32 / wavelen
                                  - s.low_freq_factor)
                                 / (s.high_freq_factor - s.low_freq_factor);
                    inv_freq = (1.0 - smooth) * (inv_freq / s.factor) + smooth * inv_freq;
                }
            }

            let angle = pos as f32 * inv_freq;
            let (sin, cos) = angle.sin_cos();
            let x0 = data[off + i];
            let x1 = data[off + i + half];
            data[off + i]        = x0 * cos - x1 * sin;
            data[off + i + half] = x0 * sin + x1 * cos;
        }
    }
}

/// Causal attention against the layer's accumulated KV cache.
/// `kv_buf` holds [seq_len × kv_dim] flat (k for K, v for V).
fn cpu_attention(
    q: &[f32], k_buf: &[f32], v_buf: &[f32],
    seq_len: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize,
    softcap: Option<f32>,
) -> Vec<f32> {
    let kv_dim = n_kv_heads * head_dim;
    let q_dim = n_heads * head_dim;
    let heads_per_kv = n_heads / n_kv_heads.max(1);
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; q_dim];

    for h in 0..n_heads {
        let kv_h = h / heads_per_kv;
        let q_off = h * head_dim;
        let kv_off = kv_h * head_dim;

        // Scores against all past positions (including current).
        let mut scores = vec![0.0f32; seq_len];
        for t in 0..seq_len {
            let k_t = &k_buf[t * kv_dim + kv_off .. t * kv_dim + kv_off + head_dim];
            let mut s = 0.0f32;
            for i in 0..head_dim { s += q[q_off + i] * k_t[i]; }
            let mut s = s * scale;
            if let Some(cap) = softcap {
                s = cap * (s / cap).tanh();
            }
            scores[t] = s;
        }

        // softmax
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in scores.iter_mut() { *s = (*s - max).exp(); sum += *s; }
        for s in scores.iter_mut() { *s /= sum; }

        // Weighted V sum
        let dst = &mut out[q_off .. q_off + head_dim];
        for t in 0..seq_len {
            let v_t = &v_buf[t * kv_dim + kv_off .. t * kv_dim + kv_off + head_dim];
            let p = scores[t];
            for i in 0..head_dim { dst[i] += p * v_t[i]; }
        }
    }
    out
}

/// SwiGLU MLP: `down(silu(gate(x)) * up(x))`. Standard for Llama, Gemma, Qwen.
///
/// gate+up are batched (1 sync instead of 2). down is separate (depends on
/// the element-wise product of gate*up which runs on CPU).
fn swiglu(
    weights: &LlmWeights, layer: usize, x: &[f32],
    dev: &mut HsaDevice, queue: &mut GpuQueue, scratch: &LlmScratch,
) -> Vec<f32> {
    let l = format!("blk.{layer}");
    let mut gate = matvec(weights, &format!("{l}.ffn_gate.weight"), x, dev, queue, scratch);
    let up       = matvec(weights, &format!("{l}.ffn_up.weight"),   x, dev, queue, scratch);
    // SwiGLU: gate[i] = silu(gate[i]) * up[i] = (gate[i] * sigmoid(gate[i])) * up[i]
    // SIMD over the 8192-element vector. Per token this saves ~280k scalar exp calls.
    swiglu_inplace(&mut gate, &up);
    matvec(weights, &format!("{l}.ffn_down.weight"), &gate, dev, queue, scratch)
}

/// SwiGLU activation in-place: `gate[i] = silu(gate[i]) * up[i]`.
/// silu(x) = x / (1 + exp(-x)). AVX-512 fast-path with rational sigmoid
/// approximation that's accurate to ~5e-4 over [-10, 10] — way above the
/// precision the LM needs since silu saturates outside [-10, 10] anyway.
#[inline]
fn swiglu_inplace(gate: &mut [f32], up: &[f32]) {
    // Tested: Padé approximation produced ~identical output (same top1 tokens
    // across 10 greedy steps). The accumulated drift comes from elsewhere.
    // Keep SIMD path for speed.
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma") {
            unsafe { swiglu_inplace_avx512(gate, up) };
            return;
        }
    }
    for i in 0..gate.len() {
        let g = gate[i];
        let sig = 1.0 / (1.0 + (-g).exp());
        gate[i] = (g * sig) * up[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn swiglu_inplace_avx512(gate: &mut [f32], up: &[f32]) {
    use std::arch::x86_64::*;
    let n = gate.len();
    let chunks = n / 16;
    let one = _mm512_set1_ps(1.0);
    // Coefficients for sigmoid via a degree-7 odd Padé-style approximation
    // applied to x/2:
    //   sigmoid(x) = 0.5 + 0.5 * tanh(x/2)
    //   tanh(x) ≈ x * (a + x² * (b + x² * (c + x² * d)))
    //            / (e + x² * (f + x² * (g + x² * h)))
    // Coeffs from glibc's vector tanh, accurate ~5e-4 over [-10, 10]; SiLU
    // saturates to its asymptotes outside that range so we clamp instead.
    let tanh_xmax = _mm512_set1_ps(8.0);
    for c in 0..chunks {
        let g = _mm512_loadu_ps(gate.as_ptr().add(c * 16));
        let u = _mm512_loadu_ps(up.as_ptr().add(c * 16));
        // sigmoid(g) using tanh: sigmoid(g) = 0.5 + 0.5*tanh(g/2)
        let half_g = _mm512_mul_ps(g, _mm512_set1_ps(0.5));
        // Clamp |half_g| ≤ 8 (saturate beyond).
        let clamped = _mm512_max_ps(_mm512_set1_ps(-8.0),
                                     _mm512_min_ps(tanh_xmax, half_g));
        // Pade approximation: tanh(x) ≈ x*(135135 + x²*(17325 + x²*(378 + x²)))
        //                              / (135135 + x²*(62370 + x²*(3150 + x²*28)))
        let x2 = _mm512_mul_ps(clamped, clamped);
        let num = _mm512_fmadd_ps(x2, _mm512_set1_ps(1.0),    _mm512_set1_ps(378.0));
        let num = _mm512_fmadd_ps(x2, num,                     _mm512_set1_ps(17325.0));
        let num = _mm512_fmadd_ps(x2, num,                     _mm512_set1_ps(135135.0));
        let num = _mm512_mul_ps(num, clamped);
        let den = _mm512_fmadd_ps(x2, _mm512_set1_ps(28.0),   _mm512_set1_ps(3150.0));
        let den = _mm512_fmadd_ps(x2, den,                     _mm512_set1_ps(62370.0));
        let den = _mm512_fmadd_ps(x2, den,                     _mm512_set1_ps(135135.0));
        let tanh = _mm512_div_ps(num, den);
        let sigmoid = _mm512_fmadd_ps(_mm512_set1_ps(0.5), tanh, _mm512_set1_ps(0.5));
        // silu(g) * up = (g * sigmoid) * up
        let silu = _mm512_mul_ps(g, sigmoid);
        let out  = _mm512_mul_ps(silu, u);
        _mm512_storeu_ps(gate.as_mut_ptr().add(c * 16), out);
    }
    // Scalar tail
    for i in (chunks * 16)..n {
        let g = gate[i];
        let sig = 1.0 / (1.0 + (-g).exp());
        gate[i] = (g * sig) * up[i];
    }
}

// ─── The function ───────────────────────────────────────────────────────────

/// Single-token forward. Pure function: same `(cfg, weights, cache, token)`
/// produces the same logits. KV cache mutates (sequence length grows by 1).
///
/// Standard llama.cpp tensor names: `blk.N.attn_q.weight`, `attn_k`, `attn_v`,
/// `attn_output`, `attn_norm`, `ffn_norm`, `ffn_gate`, `ffn_up`, `ffn_down`,
/// `token_embd.weight`, `output_norm.weight`, `output.weight` (or tied to
/// `token_embd.weight` if `output.weight` is absent).
///
/// Returns logits `[vocab_size]`.
/// Set `MODGRAD_PROFILE=1` to log per-section wall-time accumulators
/// from `forward_token`. Off by default — wall-time reads aren't free.
fn profile_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("MODGRAD_PROFILE").is_ok())
}

static PROFILE_NORM:    std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PROFILE_QKV:     std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PROFILE_ROPE:    std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PROFILE_ATTN:    std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PROFILE_O:       std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PROFILE_MLP:     std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PROFILE_LMHEAD:  std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PROFILE_TOKENS:  std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Print accumulated forward_token timings (one line) and zero counters.
pub fn print_and_reset_profile() {
    use std::sync::atomic::Ordering::Relaxed;
    let n = PROFILE_TOKENS.swap(0, Relaxed).max(1);
    let mk = |a: &std::sync::atomic::AtomicU64| {
        let total_us = a.swap(0, Relaxed);
        (total_us / n, total_us as f32 / 1000.0)
    };
    let (qkv_us, qkv_ms)       = mk(&PROFILE_QKV);
    let (o_us, o_ms)           = mk(&PROFILE_O);
    let (mlp_us, mlp_ms)       = mk(&PROFILE_MLP);
    let (norm_us, norm_ms)     = mk(&PROFILE_NORM);
    let (rope_us, rope_ms)     = mk(&PROFILE_ROPE);
    let (attn_us, attn_ms)     = mk(&PROFILE_ATTN);
    let (lm_us, lm_ms)         = mk(&PROFILE_LMHEAD);
    eprintln!("    PROFILE n={n} (per-token µs / total ms):");
    eprintln!("      qkv  {qkv_us:>6}µs / {qkv_ms:>7.1}ms total");
    eprintln!("      attn {attn_us:>6}µs / {attn_ms:>7.1}ms total");
    eprintln!("      o    {o_us:>6}µs / {o_ms:>7.1}ms total");
    eprintln!("      mlp  {mlp_us:>6}µs / {mlp_ms:>7.1}ms total");
    eprintln!("      norm {norm_us:>6}µs / {norm_ms:>7.1}ms total");
    eprintln!("      rope {rope_us:>6}µs / {rope_ms:>7.1}ms total");
    eprintln!("      lmhd {lm_us:>6}µs / {lm_ms:>7.1}ms total");
}

#[inline]
fn add_us(counter: &std::sync::atomic::AtomicU64, t: std::time::Instant) {
    counter.fetch_add(t.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
}

pub fn forward_token(
    cfg: &LlmConfig,
    weights: &LlmWeights,
    cache: &mut LlmCache,
    scratch: &mut LlmScratch,
    dev: &mut HsaDevice,
    queue: &mut GpuQueue,
    token: u32,
) -> Vec<f32> {
    assert!(cfg.is_swa.is_empty() || cfg.is_swa.len() == cfg.n_layers);
    assert!(!cfg.qk_norm, "qk_norm not yet supported in unified forward_token");
    assert!(!cfg.per_layer_inputs, "per-layer inputs (Gemma PLE) not yet supported");

    let profile = profile_enabled();
    let pos = cache.kv_len;
    let kv_dim = cfg.n_kv_heads * cfg.head_dim;

    // ── Embedding ──
    // f32 residual stream. Gemma4Model used f64 "for precision" but for short
    // generations the precision diff is invisible and the f64↔f32 round-trip
    // per layer (× 2 per layer × 28 layers) was costing ~3ms/token.
    let mut hidden: Vec<f32> = embed_token(weights, token);

    // KV cache lives in plain Vec<f32> — no per-layer download needed.
    // The existing `cache.kv_k[layer]` is directly accessible as a slice.

    for layer in 0..cfg.n_layers {
        let l = format!("blk.{layer}");

        // ── 1. Pre-attention RMSNorm ──
        let t = std::time::Instant::now();
        let mut cur = vec![0.0f32; cfg.d_model];
        rms_norm(&hidden, weights.f32(&format!("{l}.attn_norm.weight")),
                 &mut cur, cfg.rms_eps, cfg.norm_kind);
        if profile { add_us(&PROFILE_NORM, t); }

        // ── 2. Q/K/V projections ──
        let t = std::time::Instant::now();
        let mut q_data = matvec(weights, &format!("{l}.attn_q.weight"), &cur, dev, queue, scratch);
        let mut k_data = matvec(weights, &format!("{l}.attn_k.weight"), &cur, dev, queue, scratch);
        let     v_data = matvec(weights, &format!("{l}.attn_v.weight"), &cur, dev, queue, scratch);
        if profile { add_us(&PROFILE_QKV, t); }

        // ── 3. RoPE on Q and K ──
        let t = std::time::Instant::now();
        let rope_base = if !cfg.is_swa.is_empty() && cfg.is_swa[layer] {
            cfg.rope_base_swa
        } else {
            cfg.rope_base
        };
        apply_rope(&mut q_data, pos, cfg.n_heads,    cfg.head_dim, rope_base, cfg.rope_scaling);
        apply_rope(&mut k_data, pos, cfg.n_kv_heads, cfg.head_dim, rope_base, cfg.rope_scaling);
        if profile { add_us(&PROFILE_ROPE, t); }

        // ── 4. Append to KV cache (direct slice write, no PCIe) ──
        let base = pos * kv_dim;
        cache.kv_k[layer][base..base + kv_dim].copy_from_slice(&k_data);
        cache.kv_v[layer][base..base + kv_dim].copy_from_slice(&v_data);

        // ── 5. Attention (reads K/V directly from CPU memory) ──
        let t = std::time::Instant::now();
        let seq_end = (pos + 1) * kv_dim;
        cur = cpu_attention(
            &q_data, &cache.kv_k[layer][..seq_end], &cache.kv_v[layer][..seq_end],
            pos + 1, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.attn_softcap,
        );
        if profile { add_us(&PROFILE_ATTN, t); }

        // ── 6. Output projection + residual ──
        let t = std::time::Instant::now();
        cur = matvec(weights, &format!("{l}.attn_output.weight"), &cur, dev, queue, scratch);
        for i in 0..cfg.d_model { hidden[i] += cur[i]; }
        if profile { add_us(&PROFILE_O, t); }

        // ── 7. Pre-FFN RMSNorm ──
        let t = std::time::Instant::now();
        let mut ff = vec![0.0f32; cfg.d_model];
        rms_norm(&hidden, weights.f32(&format!("{l}.ffn_norm.weight")),
                 &mut ff, cfg.rms_eps, cfg.norm_kind);
        if profile { add_us(&PROFILE_NORM, t); }

        // ── 8. SwiGLU (assumed for now; GeGLU branch lands when needed) ──
        let t = std::time::Instant::now();
        let ff_out = match cfg.mlp_kind {
            MlpKind::SwiGlu => swiglu(weights, layer, &ff, dev, queue, scratch),
            MlpKind::GeGlu  => unimplemented!("GeGLU FFN not yet wired"),
        };

        // ── 9. Residual ──
        for i in 0..cfg.d_model { hidden[i] += ff_out[i]; }
        if profile { add_us(&PROFILE_MLP, t); }
    }

    // KV cache was written in-place during the layer loop — just bump the length.
    cache.kv_len += 1;

    // ── Final norm + LM head ──
    let t = std::time::Instant::now();
    let mut out_norm = vec![0.0f32; cfg.d_model];
    rms_norm(&hidden, weights.f32("output_norm.weight"),
             &mut out_norm, cfg.rms_eps, cfg.norm_kind);
    if profile { add_us(&PROFILE_NORM, t); }

    // Tied embedding: many Llama-3 checkpoints omit `output.weight` and
    // reuse `token_embd.weight` for the LM head.
    let lm_head = if weights.has("output.weight") {
        "output.weight"
    } else {
        "token_embd.weight"
    };
    let t = std::time::Instant::now();
    let mut logits = matvec(weights, lm_head, &out_norm, dev, queue, scratch);
    if profile { add_us(&PROFILE_LMHEAD, t); }
    if profile { PROFILE_TOKENS.fetch_add(1, std::sync::atomic::Ordering::Relaxed); }

    if let Some(cap) = cfg.logit_softcap {
        for v in logits.iter_mut() {
            *v = cap * (*v / cap).tanh();
        }
    }
    logits
}
