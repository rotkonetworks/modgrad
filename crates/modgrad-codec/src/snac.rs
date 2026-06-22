//! SNAC — Multi-Scale Neural Audio Codec (Siuzdak 2024).
//!
//! Decoder-only port for the 24 kHz variant published at
//! [hubertsiuzdak/snac_24khz](https://huggingface.co/hubertsiuzdak/snac_24khz).
//! Used by Orpheus TTS to turn the LM's emitted audio-code tokens into a
//! 24 kHz waveform.
//!
//! ## Pipeline
//!
//! Three codebooks at strides `[4, 2, 1]` (coarsest to finest, codes/frame
//! at the decoder's native rate). Each codebook is independently looked up
//! to retrieve `code_dim=8` embeddings, upsampled to the finest stride,
//! and summed. The resulting `[1, 8, T]` tensor enters the decoder.
//!
//! ## Decoder architecture (24 kHz config: encoder_dim=48, decoder_dim=1024,
//! rates `[8,8,4,2]`, vq_strides `[4,2,1]`, codebook 4096×8, depthwise=true,
//! noise=true)
//!
//! ```text
//!   input  [1, 8, T]
//!   ── Conv1d(k=7, pad=3, 8→1024) ─────────────────  initial projection
//!   ── DecoderBlock(rate=8, 1024→ 512) ─────────────  8x upsample
//!   ── DecoderBlock(rate=8,  512→ 256) ─────────────  8x upsample
//!   ── DecoderBlock(rate=4,  256→ 128) ─────────────  4x upsample
//!   ── DecoderBlock(rate=2,  128→  64) ─────────────  2x upsample
//!   ── Snake1d(64) ─────────────────────────────────  final activation
//!   ── Conv1d(k=7, pad=3, 64→1) ────────────────────  to mono waveform
//!   ── Tanh ────────────────────────────────────────  amplitude clip
//!   output [1, 1, T·512]                              waveform @ 24kHz
//! ```
//!
//! Each `DecoderBlock(rate=r, in=Cin, out=Cout=Cin/2)`:
//!
//! ```text
//!   Snake1d(Cin)
//!   ConvTranspose1d(k=2r, stride=r, in=Cin, out=Cout)
//!   NoiseBlock(Cout)                              # learnable noise gate
//!   ResidualUnit(Cout, dilation=1)
//!   ResidualUnit(Cout, dilation=3)
//!   ResidualUnit(Cout, dilation=9)
//! ```
//!
//! Each `ResidualUnit(C, dilation=d)`:
//!
//! ```text
//!   y = x
//!   x → Snake1d(C) → Conv1d(C→C, k=7, pad=3·d, dilation=d) → Snake1d(C) → Conv1d(C→C, k=1)
//!   return x + y                                  # skip connection
//! ```
//!
//! ## Weight-norm handling
//!
//! HuggingFace checkpoint uses `torch.nn.utils.weight_norm` (which stores
//! `weight_g` + `weight_v`). The effective weight is
//! `weight_g * weight_v / ||weight_v||`. The weight converter (Python
//! offline tool) collapses these into plain conv weights so this runtime
//! never sees the parametrisation — it just runs standard Conv1d.
//!
//! ## Status
//!
//! Currently scaffolding (types + signatures). Forward pass implementations
//! land alongside the weight converter so we can verify per-layer outputs
//! against the PyTorch reference (or its ONNX export) at each step.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use wincode_derive::{SchemaRead, SchemaWrite};

// ─── Primitives ─────────────────────────────────────────────────────────────

/// 1D convolution with `padding` (symmetric, same on both sides) and
/// `dilation`. Unlike the simpler `Conv1d` in `audio_codec.rs`, this one
/// supports the dilated kernels SNAC's residual blocks need.
///
/// Weight layout: `[out_channels, in_channels / groups, kernel_size]`
/// flattened row-major.
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub struct SnacConv1d {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub dilation: usize,
    pub padding: usize,
    /// `groups = in_channels` for depthwise convolutions; `1` for full conv.
    pub groups: usize,
}

impl SnacConv1d {
    /// Output length given input length, accounting for padding + dilation.
    pub fn out_len(&self, in_len: usize) -> usize {
        let eff_k = self.dilation * (self.kernel_size - 1) + 1;
        (in_len + 2 * self.padding).saturating_sub(eff_k) / self.stride + 1
    }

    /// Forward. Input is `[in_channels, in_len]` row-major, output is
    /// `[out_channels, out_len]`.
    ///
    /// Supports symmetric padding (zero-pad both ends), dilation, and
    /// groups (`groups = in_channels` for depthwise; `groups = 1` for full).
    ///
    /// Parallelized over output channels — disjoint output rows, same per-row
    /// accumulation order as the scalar version (bit-identical results).
    pub fn forward(&self, input: &[f32], in_len: usize) -> (Vec<f32>, usize) {
        debug_assert_eq!(input.len(), self.in_channels * in_len);
        debug_assert_eq!(self.in_channels % self.groups, 0);
        debug_assert_eq!(self.out_channels % self.groups, 0);

        let k = self.kernel_size;
        let d = self.dilation;
        let s = self.stride;
        let p = self.padding as isize;
        let in_per_group = self.in_channels / self.groups;
        let out_per_group = self.out_channels / self.groups;
        let out_len = self.out_len(in_len);
        let mut output = vec![0.0f32; self.out_channels * out_len];

        output
            .par_chunks_mut(out_len)
            .enumerate()
            .for_each(|(oc, out_row)| {
                let g = oc / out_per_group;
                let oc_local = oc % out_per_group;
                let ic_base = g * in_per_group;
                let w_row_off = oc * (in_per_group * k);
                let bias = self.bias[oc];
                for t in 0..out_len {
                    let mut acc = bias;
                    let padded_t0 = (t * s) as isize - p;
                    for ic_local in 0..in_per_group {
                        let ic = ic_base + ic_local;
                        let in_row_off = ic * in_len;
                        let w_off = w_row_off + ic_local * k;
                        for ki in 0..k {
                            let pt = padded_t0 + (ki * d) as isize;
                            if pt >= 0 && (pt as usize) < in_len {
                                acc += input[in_row_off + pt as usize]
                                     * self.weight[w_off + ki];
                            }
                        }
                    }
                    out_row[t] = acc;
                }
                let _ = oc_local;
            });
        (output, out_len)
    }
}

/// 1D transposed convolution for SNAC's per-block upsampling.
///
/// PyTorch ConvTranspose1d output length:
///   out_len = (in_len - 1) * stride - 2*padding + kernel_size + output_padding
///
/// SNAC uses `padding = math.ceil(stride/2)` on every ConvTranspose1d so
/// the leading and trailing `padding` samples of the raw transposed-conv
/// output get cropped, leaving a clean `in_len * stride`-ish sample count
/// per upsample stage. Without this padding the kernel boundaries leak
/// and the audio becomes garbage.
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub struct SnacConvTranspose1d {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub output_padding: usize,
}

impl SnacConvTranspose1d {
    pub fn out_len(&self, in_len: usize) -> usize {
        (in_len - 1) * self.stride + self.kernel_size + self.output_padding
            - 2 * self.padding
    }

    /// Transposed-conv forward (a.k.a. fractionally-strided conv). Input
    /// is `[in_channels, in_len]`, output is `[out_channels, out_len]`.
    ///
    /// PyTorch weight layout: `[in_channels, out_channels, kernel_size]`
    /// (in_channels is the leading axis, NOT out like normal Conv1d).
    ///
    /// We compute the "full" output (without cropping) conceptually but
    /// only write to in-range output positions — i.e. skip the leading
    /// `padding` and trailing `padding - output_padding` samples per channel.
    ///
    /// Reordered (oc outer, ic inner) so each parallel task owns one disjoint
    /// output row. Accumulation order changes vs the original scatter version,
    /// so results may differ by a few ULPs — perceptually identical, and the
    /// inline tests use integer weights so they still pass exactly.
    pub fn forward(&self, input: &[f32], in_len: usize) -> (Vec<f32>, usize) {
        debug_assert_eq!(input.len(), self.in_channels * in_len);
        let k = self.kernel_size;
        let s = self.stride;
        let p = self.padding as isize;
        let out_len = self.out_len(in_len);
        let in_ch = self.in_channels;
        let out_ch = self.out_channels;
        let mut output = vec![0.0f32; out_ch * out_len];

        output
            .par_chunks_mut(out_len)
            .enumerate()
            .for_each(|(oc, out_row)| {
                out_row.fill(self.bias[oc]);
                for ic in 0..in_ch {
                    let in_row_off = ic * in_len;
                    let w_off = ic * (out_ch * k) + oc * k;
                    for ki in 0..k {
                        let w = self.weight[w_off + ki];
                        if w == 0.0 { continue; }
                        // ot = t*s - p + ki. Valid range: ot ∈ [0, out_len).
                        // Solve for t (s > 0):
                        //   t >= ceil((p - ki) / s)        for ot >= 0
                        //   t <  ceil((out_len + p - ki) / s) for ot < out_len
                        let off = p - ki as isize;            // = p - ki
                        let lo  = if off <= 0 { 0 }
                                  else { ((off as usize) + s - 1) / s };
                        let hi_v: isize = out_len as isize + off;
                        let hi  = if hi_v <= 0 { 0 }
                                  else { (((hi_v - 1) as usize) / s + 1).min(in_len) };
                        let t_min = lo.min(in_len);
                        let t_max = hi;
                        // ot at t_min, increments by `s` per t step.
                        let mut ot = (t_min * s) as isize + (ki as isize) - p;
                        for t in t_min..t_max {
                            // Guaranteed 0 <= ot < out_len by construction.
                            out_row[ot as usize] += input[in_row_off + t] * w;
                            ot += s as isize;
                        }
                    }
                }
            });
        (output, out_len)
    }
}

/// Per-channel learnable Snake activation: `y = x + (1/α)·sin(αx)²`.
/// Length-of-α equals the number of channels.
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub struct Snake1d {
    pub alpha: Vec<f32>,
    pub channels: usize,
}

impl Snake1d {
    /// Apply in place to a `[channels, length]` buffer.
    ///
    /// SIMD path: process 8 lanes at a time via `wide::f32x8::sin` (polynomial
    /// approximation, ~5–10× faster than scalar `f32::sin`). Tail handled
    /// scalar. Parallelized over channels.
    pub fn forward_inplace(&self, x: &mut [f32], length: usize) {
        debug_assert_eq!(x.len(), self.channels * length);
        let alpha = &self.alpha;
        x.par_chunks_mut(length).enumerate().for_each(|(c, row)| {
            let a = alpha[c].max(1e-6);
            let inv_a = 1.0 / a;
            let a_v = wide::f32x8::splat(a);
            let inv_a_v = wide::f32x8::splat(inv_a);
            let (chunks, tail) = row.as_chunks_mut::<8>();
            for chunk in chunks {
                let v = wide::f32x8::from(*chunk);
                let s = (a_v * v).sin();
                let out = v + s * s * inv_a_v;
                *chunk = out.to_array();
            }
            for v in tail.iter_mut() {
                let s = (a * *v).sin();
                *v += s * s * inv_a;
            }
        });
    }
}

/// Learnable noise gate (SNAC's `NoiseBlock`). PyTorch impl is:
///
/// ```python
/// noise = torch.randn_like(x)         # [B, C, T]
/// h = self.linear(x)                  # 1x1 conv: scale per-channel
/// x = x + noise * h                   # add gated noise
/// ```
///
/// For deterministic offline decode we seed RNG per generation so outputs
/// are reproducible. The 1x1 conv is just `[in_channels=C, out_channels=C]`
/// without bias.
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub struct NoiseBlock {
    /// 1x1 conv weight `[channels, channels, 1]`, flattened `[C·C]`.
    pub gate_weight: Vec<f32>,
    pub channels: usize,
}

impl NoiseBlock {
    /// `x += randn(C, T) * (W₁ₓ₁ @ x)`.
    /// `gate_weight` is `[C × C]` flattened (the 1×1 conv kernel).
    ///
    /// Parallelism strategy: two stages, both parallel over output channels:
    ///   (1) gate[oc, t] = Σ_ic W[oc, ic] · x[ic, t]   — rayon over oc, t inner
    ///   (2) x[oc, t]   += randn_oc(t) · gate[oc, t]   — rayon over oc, each
    ///                                                    channel owns its
    ///                                                    own splitmix64 RNG
    ///
    /// Per-channel seed = splitmix64(master_seed XOR channel_id). This means
    /// the noise pattern differs from the old single-threaded sequential
    /// version (you'll get a different waveform sample-by-sample), but it's
    /// noise by design — the model was trained against arbitrary draws from
    /// the same distribution, and perceptual audio is identical.
    pub fn forward_inplace(&self, x: &mut [f32], length: usize, seed: u64) {
        let c = self.channels;
        debug_assert_eq!(x.len(), c * length);

        // ── Stage 1: gate = W @ x  (shape [c, length], row-major) ──────────
        let mut gate = vec![0.0f32; c * length];
        gate.par_chunks_mut(length).enumerate().for_each(|(oc, gate_row)| {
            let w_row = &self.gate_weight[oc * c .. (oc + 1) * c];
            for t in 0..length {
                let mut acc = 0.0f32;
                for ic in 0..c {
                    acc += w_row[ic] * x[ic * length + t];
                }
                gate_row[t] = acc;
            }
        });

        // ── Stage 2: per-channel parallel noise modulation ─────────────────
        x.par_chunks_mut(length).enumerate().for_each(|(oc, x_row)| {
            let mut state = mix_seed(seed, oc as u64);
            let g = &gate[oc * length .. (oc + 1) * length];
            let mut t = 0;
            while t + 1 < length {
                let (n0, n1) = next_norm_pair(&mut state);
                x_row[t]     += n0 * g[t];
                x_row[t + 1] += n1 * g[t + 1];
                t += 2;
            }
            if t < length {
                let (n0, _) = next_norm_pair(&mut state);
                x_row[t] += n0 * g[t];
            }
        });
    }
}

#[inline]
fn mix_seed(master: u64, lane: u64) -> u64 {
    // splitmix64 of master XOR (lane scrambled by golden ratio) — gives
    // statistically-independent RNG state per parallel lane.
    let mut z = master ^ lane.wrapping_mul(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    (z ^ (z >> 31)).max(1)
}

#[inline]
fn splitmix64_step(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

#[inline]
fn next_norm_pair(state: &mut u64) -> (f32, f32) {
    // box-muller: two unit gaussians from two uniforms.
    let u1 = ((splitmix64_step(state) >> 40) as f32 / (1u64 << 24) as f32).max(1e-10);
    let u2 =  (splitmix64_step(state) >> 40) as f32 / (1u64 << 24) as f32;
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f32::consts::PI * u2;
    (r * theta.cos(), r * theta.sin())
}

// ─── Composite blocks ───────────────────────────────────────────────────────

/// `ResidualUnit(C, dilation=d)`: skip-connection with two-conv body.
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub struct ResidualUnit {
    pub snake_a:  Snake1d,
    pub conv_a:   SnacConv1d,     // dilated
    pub snake_b:  Snake1d,
    pub conv_b:   SnacConv1d,     // kernel=1
}

impl ResidualUnit {
    /// `y = x + body(x)` where body = Snake → DilatedConv → Snake → Conv(k=1).
    ///
    /// Input and output have the same `(channels, length)` shape — the
    /// dilated conv uses `padding = (k-1)·d / 2` (computed once at load
    /// time and stored on `conv_a.padding`) so length is preserved.
    pub fn forward(&self, x: &[f32], length: usize) -> (Vec<f32>, usize) {
        let channels = self.snake_a.channels;
        debug_assert_eq!(x.len(), channels * length);

        // body(x):
        let mut h = x.to_vec();
        self.snake_a.forward_inplace(&mut h, length);
        let (h, len) = self.conv_a.forward(&h, length);
        debug_assert_eq!(len, length, "ResidualUnit: dilated conv changed length");
        let mut h = h;
        self.snake_b.forward_inplace(&mut h, length);
        let (mut h, _) = self.conv_b.forward(&h, length);

        // skip connection
        for i in 0..h.len() { h[i] += x[i]; }
        (h, length)
    }
}

/// `DecoderBlock(rate, in→out)`: Snake → upsample → noise → 3× residual.
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub struct DecoderBlock {
    pub snake_pre:  Snake1d,
    pub upsample:   SnacConvTranspose1d,
    pub noise:      NoiseBlock,
    pub res1:       ResidualUnit,  // dilation 1
    pub res3:       ResidualUnit,  // dilation 3
    pub res9:       ResidualUnit,  // dilation 9
}

impl DecoderBlock {
    /// Snake → ConvT (upsample) → NoiseBlock → 3× ResidualUnit (dil 1, 3, 9).
    /// Output channels = `Cin / 2`, output length = `length * stride + output_padding`.
    pub fn forward(&self, x: &[f32], length: usize, seed: u64) -> (Vec<f32>, usize) {
        let mut h = x.to_vec();
        self.snake_pre.forward_inplace(&mut h, length);
        let (h, up_len) = self.upsample.forward(&h, length);

        let mut h = h;
        self.noise.forward_inplace(&mut h, up_len, seed);

        let (h, l) = self.res1.forward(&h, up_len);
        let (h, l) = self.res3.forward(&h, l);
        let (h, l) = self.res9.forward(&h, l);
        (h, l)
    }
}

// ─── Residual vector quantizer (lookup-only for decode) ─────────────────────

/// One codebook: `n_codes × code_dim` learned embeddings.
/// SNAC stores these as `embedding.weight` after L2-normalising the input
/// projections; for decode we just lookup + optionally apply the inverse
/// projection.
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub struct Codebook {
    /// `[n_codes, code_dim]`, row-major.
    pub embedding: Vec<f32>,
    pub n_codes: usize,
    pub code_dim: usize,
    /// Output projection back to the decoder's input channel count.
    /// SNAC keeps this `code_dim → decoder_in` (8 → 8 for 24 kHz, which is
    /// effectively identity but learned).
    pub out_proj_weight: Vec<f32>,
    pub out_proj_bias: Vec<f32>,
    /// Stride at which this codebook samples the decoder time axis.
    /// E.g. stride=4 means one code covers 4 finest-resolution frames; we
    /// nearest-neighbour upsample to the finest stride before summing.
    pub stride: usize,
}

impl Codebook {
    /// Look up `codes` (length `T_at_stride`), project, nearest-neighbour
    /// upsample by `self.stride` so output matches the finest time axis.
    ///
    /// Returns `(buf, t_finest)` with shape `[out_channels, t_finest]`
    /// where `out_channels = out_proj_weight.len() / code_dim`.
    pub fn lookup_and_upsample(&self, codes: &[u32]) -> (Vec<f32>, usize) {
        let t_in = codes.len();
        let t_out = t_in * self.stride;
        let out_ch = self.out_proj_weight.len() / self.code_dim;
        debug_assert_eq!(self.out_proj_weight.len(), out_ch * self.code_dim);
        debug_assert_eq!(self.out_proj_bias.len(), out_ch);

        // Step 1: embed lookup → [t_in, code_dim] (row-major)
        let mut emb = vec![0.0f32; t_in * self.code_dim];
        for (t, &c) in codes.iter().enumerate() {
            let c = (c as usize).min(self.n_codes - 1);
            let src = &self.embedding[c * self.code_dim .. (c + 1) * self.code_dim];
            emb[t * self.code_dim .. (t + 1) * self.code_dim].copy_from_slice(src);
        }

        // Step 2: out_proj — embedding (t_in × code_dim) @ Wᵀ + b → (t_in × out_ch)
        // SNAC keeps weight as [out_ch, code_dim] row-major (PyTorch Conv1d k=1 layout).
        let mut projected = vec![0.0f32; t_in * out_ch];
        for t in 0..t_in {
            let in_row = &emb[t * self.code_dim .. (t + 1) * self.code_dim];
            for oc in 0..out_ch {
                let w_row = &self.out_proj_weight[oc * self.code_dim .. (oc + 1) * self.code_dim];
                let mut acc = self.out_proj_bias[oc];
                for k in 0..self.code_dim { acc += w_row[k] * in_row[k]; }
                projected[t * out_ch + oc] = acc;
            }
        }

        // Step 3: lay out as [out_ch, t_in] (channel-first) and nearest-neighbour
        // upsample along the time axis by `self.stride`. Output `[out_ch, t_out]`.
        let mut out = vec![0.0f32; out_ch * t_out];
        for oc in 0..out_ch {
            for t in 0..t_in {
                let v = projected[t * out_ch + oc];
                let base = oc * t_out + t * self.stride;
                for s in 0..self.stride {
                    out[base + s] = v;
                }
            }
        }
        (out, t_out)
    }
}

/// `ResidualVectorQuantize` for decode. The 3 SNAC codebooks at strides
/// `[4, 2, 1]` get summed after each one is looked-up + upsampled to the
/// finest (stride=1) time axis.
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub struct ResidualVQ {
    pub codebooks: Vec<Codebook>,  // 3 entries for 24 kHz config
}

impl ResidualVQ {
    /// `codes_per_book[i]` has length `T_finest / strides[i]` (each entry 0..4095).
    /// Each codebook independently emits its `[out_ch, T_finest]` activation
    /// (after lookup + projection + NN-upsample by its own stride), then the
    /// codebooks are summed element-wise. Returns `[out_ch, T_finest]`.
    pub fn decode(&self, codes_per_book: &[Vec<u32>]) -> (Vec<f32>, usize) {
        assert!(!self.codebooks.is_empty(), "ResidualVQ::decode: no codebooks");
        assert_eq!(codes_per_book.len(), self.codebooks.len(),
            "ResidualVQ::decode: codebook count mismatch");

        let (mut acc, t_finest) = self.codebooks[0].lookup_and_upsample(&codes_per_book[0]);
        for (i, cb) in self.codebooks.iter().enumerate().skip(1) {
            let (buf, t) = cb.lookup_and_upsample(&codes_per_book[i]);
            assert_eq!(t, t_finest,
                "ResidualVQ::decode: codebook {i} produced T={t} but finest is {t_finest}");
            assert_eq!(buf.len(), acc.len());
            for j in 0..acc.len() { acc[j] += buf[j]; }
        }
        (acc, t_finest)
    }
}

// ─── Full decoder ───────────────────────────────────────────────────────────

/// SNAC 24 kHz: codes → 24 kHz mono waveform.
///
/// State-dict layout (matches `hubertsiuzdak/snac_24khz` exactly):
///
/// ```text
///   decoder.model.0  → conv_in_dw    Conv1d(768, k=7, depthwise=true)
///   decoder.model.1  → conv_in_proj  Conv1d(768 → 1024, k=1)
///   decoder.model.2  → blocks[0]     DecoderBlock(rate=8, 1024→ 512)
///   decoder.model.3  → blocks[1]     DecoderBlock(rate=8,  512→ 256)
///   decoder.model.4  → blocks[2]     DecoderBlock(rate=4,  256→ 128)
///   decoder.model.5  → blocks[3]     DecoderBlock(rate=2,  128→  64)
///   decoder.model.6  → snake_final   Snake1d(64)
///   decoder.model.7  → conv_out      Conv1d(64 → 1, k=7)
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub struct SnacDecoder24k {
    pub vq:            ResidualVQ,
    /// Initial depthwise conv on the 768-channel VQ output (decoder.model.0).
    pub conv_in_dw:    SnacConv1d,
    /// 1×1 projection 768 → 1024 (decoder.model.1).
    pub conv_in_proj:  SnacConv1d,
    /// 4 upsampling blocks (decoder.model.2..5).
    pub blocks:        Vec<DecoderBlock>,
    /// Snake on the 64-channel pre-output (decoder.model.6).
    pub snake_final:   Snake1d,
    /// Final 64 → 1 conv with k=7 to produce mono waveform (decoder.model.7).
    pub conv_out:      SnacConv1d,
}

impl SnacDecoder24k {
    /// Decode 3 lists of codes into a 24 kHz mono waveform (samples in
    /// `[-1.0, 1.0]`).
    ///
    /// `seed` controls the NoiseBlock RNG so per-generation audio is
    /// reproducible. Use any non-zero value; the same seed always gives
    /// the same waveform.
    pub fn decode_seeded(&self, codes_per_book: &[Vec<u32>], seed: u64) -> Vec<f32> {
        // ── VQ: codes → [768, T_finest] (sum of 3 projected codebook lookups) ──
        let (vq_out, t_in) = self.vq.decode(codes_per_book);

        // ── Initial depthwise conv (k=7, channels stay at 768) ──
        let (h, l) = self.conv_in_dw.forward(&vq_out, t_in);
        // ── 1×1 projection up to 1024 channels ──
        let (h, l) = self.conv_in_proj.forward(&h, l);

        // ── 4 DecoderBlocks ──
        let mut h = h;
        let mut l = l;
        // Vary seed per block so noise is decorrelated but reproducible.
        let mut bseed = seed;
        for blk in &self.blocks {
            let (h2, l2) = blk.forward(&h, l, bseed);
            h = h2; l = l2;
            bseed = bseed.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(1);
        }

        // ── Final Snake → Conv → Tanh ──
        self.snake_final.forward_inplace(&mut h, l);
        let (mut h, l) = self.conv_out.forward(&h, l);
        debug_assert_eq!(h.len(), l, "expected mono output [1, T]");
        for v in h.iter_mut() { *v = v.tanh(); }
        h
    }

    /// Convenience: deterministic decode with seed=1.
    pub fn decode(&self, codes_per_book: &[Vec<u32>]) -> Vec<f32> {
        self.decode_seeded(codes_per_book, 1)
    }
}

#[cfg(test)]
mod conv_tests {
    use super::*;

    /// Identity-style 1×1 conv: weight=1 (single in→out, k=1), bias=0,
    /// no padding/dilation. Output must equal input.
    #[test]
    fn snac_conv1d_identity_kernel1() {
        let c = SnacConv1d {
            weight: vec![1.0],
            bias: vec![0.0],
            in_channels: 1, out_channels: 1,
            kernel_size: 1, stride: 1, dilation: 1, padding: 0, groups: 1,
        };
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let (y, len) = c.forward(&x, 4);
        assert_eq!(len, 4);
        assert_eq!(y, x);
    }

    /// k=3, stride=1, pad=0, dilation=1, weight=[1,1,1] → moving sum.
    ///   x = [1,2,3,4,5], y[t] = x[t]+x[t+1]+x[t+2]
    ///   → [6, 9, 12]
    #[test]
    fn snac_conv1d_moving_sum() {
        let c = SnacConv1d {
            weight: vec![1.0, 1.0, 1.0],
            bias: vec![0.0],
            in_channels: 1, out_channels: 1,
            kernel_size: 3, stride: 1, dilation: 1, padding: 0, groups: 1,
        };
        let (y, len) = c.forward(&[1.0, 2.0, 3.0, 4.0, 5.0], 5);
        assert_eq!(len, 3);
        assert_eq!(y, vec![6.0, 9.0, 12.0]);
    }

    /// Same-padding: k=3, pad=1 keeps length the same (with zero padding).
    ///   y[0] = 0 + x[0] + x[1] = 0+1+2 = 3
    ///   y[1] = x[0] + x[1] + x[2] = 6
    ///   y[2] = x[1] + x[2] + 0 = 5
    #[test]
    fn snac_conv1d_padding_preserves_length() {
        let c = SnacConv1d {
            weight: vec![1.0, 1.0, 1.0],
            bias: vec![0.0],
            in_channels: 1, out_channels: 1,
            kernel_size: 3, stride: 1, dilation: 1, padding: 1, groups: 1,
        };
        let (y, len) = c.forward(&[1.0, 2.0, 3.0], 3);
        assert_eq!(len, 3);
        assert_eq!(y, vec![3.0, 6.0, 5.0]);
    }

    /// Dilation=2, k=3 (effective kernel span 5), pad=0, stride=1.
    ///   y[t] = x[t]+x[t+2]+x[t+4]
    ///   x=[1,2,3,4,5,6,7]: y=[1+3+5, 2+4+6, 3+5+7] = [9, 12, 15]
    #[test]
    fn snac_conv1d_dilation_skip() {
        let c = SnacConv1d {
            weight: vec![1.0, 1.0, 1.0],
            bias: vec![0.0],
            in_channels: 1, out_channels: 1,
            kernel_size: 3, stride: 1, dilation: 2, padding: 0, groups: 1,
        };
        let (y, len) = c.forward(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 7);
        assert_eq!(len, 3);
        assert_eq!(y, vec![9.0, 12.0, 15.0]);
    }

    /// Depthwise (groups = in_ch = out_ch): each channel uses its own
    /// weight, no cross-channel mixing.
    ///   c0 weight=[2], c1 weight=[3], k=1
    ///   x = [[1,2,3], [10,20,30]]
    ///   y = [[2,4,6], [30,60,90]]
    #[test]
    fn snac_conv1d_depthwise() {
        let c = SnacConv1d {
            weight: vec![2.0, 3.0],
            bias: vec![0.0, 0.0],
            in_channels: 2, out_channels: 2,
            kernel_size: 1, stride: 1, dilation: 1, padding: 0, groups: 2,
        };
        let (y, len) = c.forward(&[1.0, 2.0, 3.0,  10.0, 20.0, 30.0], 3);
        assert_eq!(len, 3);
        assert_eq!(y, vec![2.0, 4.0, 6.0,  30.0, 60.0, 90.0]);
    }

    /// Bias add: output is offset by the per-channel bias.
    #[test]
    fn snac_conv1d_bias() {
        let c = SnacConv1d {
            weight: vec![1.0],
            bias: vec![0.5],
            in_channels: 1, out_channels: 1,
            kernel_size: 1, stride: 1, dilation: 1, padding: 0, groups: 1,
        };
        let (y, _) = c.forward(&[1.0, 2.0], 2);
        assert_eq!(y, vec![1.5, 2.5]);
    }

    /// ConvTranspose1d: stride=2, k=2, weight=[1, 1], bias=0
    ///   Each input value contributes to 2 consecutive output positions.
    ///   x=[a,b]: out=[a, a+b, b]  (overlap-add with stride 2, kernel 2)
    /// Actually: stride 2, kernel 2 → x[0] writes to out[0..2], x[1] writes
    /// to out[2..4]. No overlap. out_len = (2-1)*2+2 = 4.
    ///   x=[3,5]: out=[3, 3, 5, 5]
    #[test]
    fn snac_convt1d_basic_upsample() {
        let c = SnacConvTranspose1d {
            // PyTorch ConvTranspose1d weight: [in_channels, out_channels, k]
            weight: vec![1.0, 1.0],
            bias: vec![0.0],
            in_channels: 1, out_channels: 1,
            kernel_size: 2, stride: 2, padding: 0, output_padding: 0,
        };
        let (y, len) = c.forward(&[3.0, 5.0], 2);
        assert_eq!(len, 4);
        assert_eq!(y, vec![3.0, 3.0, 5.0, 5.0]);
    }

    /// ConvTranspose1d with overlap: stride=1, k=3, weight=[1,2,1]
    ///   x=[a,b]: out_len = (2-1)*1+3 = 4
    ///     x[0]=a → out[0..3] += [a, 2a, a]
    ///     x[1]=b → out[1..4] += [b, 2b, b]
    ///   out = [a, 2a+b, a+2b, b]
    ///   x=[1,3]: out=[1, 5, 7, 3]
    #[test]
    fn snac_convt1d_overlap_add() {
        let c = SnacConvTranspose1d {
            weight: vec![1.0, 2.0, 1.0],
            bias: vec![0.0],
            in_channels: 1, out_channels: 1,
            kernel_size: 3, stride: 1, padding: 0, output_padding: 0,
        };
        let (y, len) = c.forward(&[1.0, 3.0], 2);
        assert_eq!(len, 4);
        assert_eq!(y, vec![1.0, 5.0, 7.0, 3.0]);
    }
}
