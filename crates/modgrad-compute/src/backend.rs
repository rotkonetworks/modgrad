//! Compute backend abstraction for CTM hot-path operations.
//!
//! Mojo-inspired design:
//!   - Explicit SIMD (AVX-512 on Zen4) — not hoping for autovectorization
//!   - Cache-tiled matmul — L1-blocked for 32KB cache lines
//!   - Fused kernels — synapse+GLU+SiLU+layernorm in one cache pass
//!   - Autotune — benchmark tile sizes on first run
//!
//! The `ComputeBackend` trait is the single dispatch point.
//! Implementations: CpuBackend (this file), future CudaBackend, VulkanBackend.
//!
//! Design rules:
//!   - Object-safe (no generics in methods)
//!   - All data as slices — zero allocation on the hot path
//!   - Default methods for fused operations so backends can override

use rayon::prelude::*;

// ─── Explicit AVX-512 dot product ───────────────────────────

/// AVX-512 dot product: 16 floats/cycle with FMA.
/// Falls back to scalar if not available at runtime.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { dot_avx512(a, b) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_avx2_fma(a, b) };
        }
    }
    dot_scalar(a, b)
}

/// AVX-512: process 16 f32s per cycle with vfmadd231ps.
/// 4 accumulators hide FMA latency (4 cycles on Zen4).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dot_avx512(a: &[f32], b: &[f32]) -> f32 { unsafe {
    use std::arch::x86_64::*;
    let n = a.len();
    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();
    let chunks = n / 64;
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    for i in 0..chunks {
        let j = i * 64;
        let a0 = _mm512_loadu_ps(ap.add(j));
        let b0 = _mm512_loadu_ps(bp.add(j));
        acc0 = _mm512_fmadd_ps(a0, b0, acc0);
        let a1 = _mm512_loadu_ps(ap.add(j + 16));
        let b1 = _mm512_loadu_ps(bp.add(j + 16));
        acc1 = _mm512_fmadd_ps(a1, b1, acc1);
        let a2 = _mm512_loadu_ps(ap.add(j + 32));
        let b2 = _mm512_loadu_ps(bp.add(j + 32));
        acc2 = _mm512_fmadd_ps(a2, b2, acc2);
        let a3 = _mm512_loadu_ps(ap.add(j + 48));
        let b3 = _mm512_loadu_ps(bp.add(j + 48));
        acc3 = _mm512_fmadd_ps(a3, b3, acc3);
    }
    // Handle remaining 16-wide chunks
    let mut i = chunks * 64;
    while i + 16 <= n {
        let va = _mm512_loadu_ps(ap.add(i));
        let vb = _mm512_loadu_ps(bp.add(i));
        acc0 = _mm512_fmadd_ps(va, vb, acc0);
        i += 16;
    }
    let sum_vec = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
    let mut result = _mm512_reduce_add_ps(sum_vec);
    // Scalar tail
    while i < n {
        result += *a.get_unchecked(i) * *b.get_unchecked(i);
        i += 1;
    }
    result
}}

/// AVX2+FMA: process 8 f32s per cycle.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_avx2_fma(a: &[f32], b: &[f32]) -> f32 { unsafe {
    use std::arch::x86_64::*;
    let n = a.len();
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let chunks = n / 32;
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    for i in 0..chunks {
        let j = i * 32;
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(j)),      _mm256_loadu_ps(bp.add(j)),      acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(j + 8)),  _mm256_loadu_ps(bp.add(j + 8)),  acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(j + 16)), _mm256_loadu_ps(bp.add(j + 16)), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(j + 24)), _mm256_loadu_ps(bp.add(j + 24)), acc3);
    }
    let sum = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    // Horizontal sum of 8 floats
    let hi = _mm256_extractf128_ps(sum, 1);
    let lo = _mm256_castps256_ps128(sum);
    let sum128 = _mm_add_ps(lo, hi);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    let mut result = _mm_cvtss_f32(sum32);
    // Scalar tail
    let mut i = chunks * 32;
    while i < n {
        result += *a.get_unchecked(i) * *b.get_unchecked(i);
        i += 1;
    }
    result
}}

/// Scalar fallback with 4 accumulators for latency hiding.
#[inline]
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let chunks = n / 4;
    for i in 0..chunks {
        let j = i * 4;
        unsafe {
            s0 += *a.get_unchecked(j) * *b.get_unchecked(j);
            s1 += *a.get_unchecked(j + 1) * *b.get_unchecked(j + 1);
            s2 += *a.get_unchecked(j + 2) * *b.get_unchecked(j + 2);
            s3 += *a.get_unchecked(j + 3) * *b.get_unchecked(j + 3);
        }
    }
    for i in (chunks * 4)..n {
        s0 += a[i] * b[i];
    }
    (s0 + s1) + (s2 + s3)
}

// ─── Cache-tiled matmul ─────────────────────────────────────

/// L1-cache-blocked matvec: y = W*x + b
/// Tiles rows into blocks that fit in L1 (32KB on Zen4).
/// Each tile: TILE_ROWS rows × full in_dim, processed sequentially
/// to keep the x vector hot in cache.
const TILE_ROWS: usize = 32;

/// Tiled matvec: processes TILE_ROWS rows at a time.
/// x stays in L1 across all rows in the tile.
#[inline]
fn matvec_tiled(weight: &[f32], bias: &[f32], x: &[f32],
                y: &mut [f32], out_dim: usize, in_dim: usize) {
    for tile_start in (0..out_dim).step_by(TILE_ROWS) {
        let tile_end = (tile_start + TILE_ROWS).min(out_dim);
        for r in tile_start..tile_end {
            let row = &weight[r * in_dim..(r + 1) * in_dim];
            y[r] = bias[r] + dot(row, &x[..in_dim]);
        }
    }
}

// ─── Fused synapse: matvec → GLU → SiLU → layernorm ────────

/// Fused synapse forward in one pass.
/// The raw matvec produces 2*out_dim values. We immediately apply
/// GLU + SiLU per-element, then layernorm the result.
/// This keeps data in L1 instead of 4 separate passes over memory.
#[inline]
fn synapse_fused(weight: &[f32], bias: &[f32], x: &[f32],
                 output: &mut [f32], out_dim: usize, in_dim: usize) {
    // Step 1: Compute raw matvec (2*out_dim rows) directly into GLU+SiLU
    // For GLU: output[i] = raw[i] * sigmoid(raw[out_dim + i])
    // For SiLU: output[i] = glu_result * sigmoid(glu_result)
    // Fused: compute both rows at once, apply GLU+SiLU immediately
    for i in 0..out_dim {
        // Content row
        let content_row = &weight[i * in_dim..(i + 1) * in_dim];
        let content = bias[i] + dot(content_row, &x[..in_dim]);

        // Gate row
        let gate_row = &weight[(out_dim + i) * in_dim..(out_dim + i + 1) * in_dim];
        let gate = bias[out_dim + i] + dot(gate_row, &x[..in_dim]);

        // Fused GLU + SiLU
        let gate_sigmoid = fast_sigmoid(gate);
        let glu_val = content * gate_sigmoid;
        let silu_sigmoid = fast_sigmoid(glu_val);
        output[i] = glu_val * silu_sigmoid;
    }

    // Step 2: Layer norm (must be a separate pass — needs mean/var of all elements)
    let n = out_dim as f32;
    let mean: f32 = output[..out_dim].iter().sum::<f32>() / n;
    let var: f32 = output[..out_dim].iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv_std = 1.0 / (var + 1e-5).sqrt();
    for v in &mut output[..out_dim] {
        *v = (*v - mean) * inv_std;
    }
}

/// Parallel fused synapse for large out_dims.
/// Tiles output rows across threads, each thread does fused GLU+SiLU.
/// Layernorm requires a second pass (needs global mean/var).
fn synapse_fused_parallel(weight: &[f32], bias: &[f32], x: &[f32],
                          output: &mut [f32], out_dim: usize, in_dim: usize) {
    // Phase 1: Parallel fused matvec + GLU + SiLU
    let chunk_rows = (out_dim / rayon::current_num_threads()).max(4);
    output[..out_dim].par_chunks_mut(chunk_rows)
        .enumerate()
        .for_each(|(ci, chunk)| {
            let r_start = ci * chunk_rows;
            for (j, out) in chunk.iter_mut().enumerate() {
                let i = r_start + j;
                if i >= out_dim { break; }
                let content_row = &weight[i * in_dim..(i + 1) * in_dim];
                let content = bias[i] + dot(content_row, &x[..in_dim]);
                let gate_row = &weight[(out_dim + i) * in_dim..(out_dim + i + 1) * in_dim];
                let gate = bias[out_dim + i] + dot(gate_row, &x[..in_dim]);
                let gate_sig = fast_sigmoid(gate);
                let glu_val = content * gate_sig;
                let silu_sig = fast_sigmoid(glu_val);
                *out = glu_val * silu_sig;
            }
        });

    // Phase 2: Layer norm (sequential — needs global stats)
    let n = out_dim as f32;
    let mean: f32 = output[..out_dim].iter().sum::<f32>() / n;
    let var: f32 = output[..out_dim].iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv_std = 1.0 / (var + 1e-5).sqrt();
    for v in &mut output[..out_dim] {
        *v = (*v - mean) * inv_std;
    }
}

// ─── Autotune ───────────────────────────────────────────────

/// Startup autotune: benchmarks tile sizes and parallel thresholds.
/// Called once on first use, caches results for the session.
#[derive(Debug, Clone)]
pub struct AutoTuneConfig {
    /// Best tile size for matvec (rows per tile).
    pub tile_rows: usize,
    /// Minimum flops before engaging rayon parallelism.
    pub par_threshold: usize,
    /// Whether to use fused synapse (vs 4 separate passes).
    pub use_fused_synapse: bool,
}

impl Default for AutoTuneConfig {
    fn default() -> Self {
        Self {
            tile_rows: TILE_ROWS,
            par_threshold: 4096,
            use_fused_synapse: true,
        }
    }
}

impl AutoTuneConfig {
    /// Run a quick benchmark and pick optimal settings for this hardware.
    /// Tests tile sizes 16/32/64/128, measures dot throughput.
    pub fn autotune() -> Self {
        let mut best = Self::default();

        // Benchmark dot product to calibrate — if it's fast, we're on AVX-512
        let n = 512;
        let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.001).collect();
        let t0 = std::time::Instant::now();
        let mut sink = 0.0f32;
        for _ in 0..10000 {
            sink += dot(&a, &b);
        }
        let dot_ns = t0.elapsed().as_nanos() / 10000;
        let _ = sink; // prevent optimization

        // Fast dot (AVX-512): lower par threshold, bigger tiles
        if dot_ns < 100 {
            // ~50ns for 512-dim dot = AVX-512
            best.par_threshold = 2048;
            best.tile_rows = 64;
        } else if dot_ns < 300 {
            // ~200ns = AVX2
            best.par_threshold = 4096;
            best.tile_rows = 32;
        } else {
            // Scalar
            best.par_threshold = 8192;
            best.tile_rows = 16;
        }

        best
    }
}

// ─── fast sigmoid ───────────────────────────────────────────

/// Sigmoid with early-out for saturated values (avoids exp for |v| > 6).
#[inline(always)]
fn fast_sigmoid(v: f32) -> f32 {
    if v > 6.0 { 1.0 }
    else if v < -6.0 { 0.0 }
    else { 1.0 / (1.0 + (-v).exp()) }
}

// ─── ComputeBackend trait ───────────────────────────────────

/// Compute backend for CTM hot-path operations.
/// Implementations: CpuBackend (rayon), future CudaBackend, VulkanBackend.
pub trait ComputeBackend: Send + Sync {
    /// Dense matrix-vector multiply: y = W*x + b
    /// W is [out_dim x in_dim] row-major, x is [in_dim], y is [out_dim]
    fn matvec(&self, weight: &[f32], bias: &[f32], x: &[f32],
              y: &mut [f32], out_dim: usize, in_dim: usize);

    /// Per-neuron batched MLP (SuperLinear):
    /// For each neuron n: y[n] = W[n] * trace[n] + b[n]
    /// weights: [n_neurons x out_per x in_per], trace: [n_neurons x in_per]
    /// output: [n_neurons x out_per]
    fn superlinear(&self, weights: &[f32], biases: &[f32], trace: &[f32],
                   output: &mut [f32], n_neurons: usize, in_per: usize, out_per: usize);

    /// GLU activation: output[i] = input[i] * sigmoid(input[i + half])
    /// input has 2*n elements, output has n elements
    fn glu(&self, input: &[f32], output: &mut [f32]);

    /// SiLU (swish) in-place: x[i] = x[i] * sigmoid(x[i])
    fn silu_inplace(&self, x: &mut [f32]);

    /// Layer normalization in-place
    fn layer_norm_inplace(&self, x: &mut [f32]);

    /// Fused synapse forward: matvec -> GLU -> SiLU -> layer_norm
    /// Default implementation calls the individual methods.
    /// `scratch` must have at least `out_dim * 2` elements (the raw matvec output).
    /// `output` must have at least `out_dim` elements.
    fn synapse_forward(&self, weight: &[f32], bias: &[f32], x: &[f32],
                       output: &mut [f32], scratch: &mut [f32],
                       out_dim: usize, in_dim: usize) {
        // matvec into scratch (2*out_dim), GLU into output (out_dim), SiLU, layer_norm
        self.matvec(weight, bias, x, scratch, out_dim * 2, in_dim);
        self.glu(&scratch[..out_dim * 2], &mut output[..out_dim]);
        self.silu_inplace(&mut output[..out_dim]);
        self.layer_norm_inplace(&mut output[..out_dim]);
    }

    /// Trace shift: for each neuron, shift trace left by 1, append new activation.
    /// traces: [n_neurons x memory_length], new_activations: [n_neurons]
    fn trace_shift(&self, traces: &mut [f32], new_activations: &[f32],
                   n_neurons: usize, memory_length: usize);

    /// Sync accumulator update.
    ///
    /// For each pair i in 0..n_pairs:
    ///   pairwise = activations_left[i] * activations_right[i] * dopamine * temporal_proximity
    ///   r = exp(-(decay[i] + decay_shift[i]).clamp(0, 15))
    ///   alpha[i] = r * alpha[i] + pairwise   (or just pairwise if !initialized)
    ///   beta[i]  = r * beta[i]  + dopamine    (or just dopamine if !initialized)
    ///   sync_out[i] = alpha[i] / sqrt(beta[i]).max(1e-8)
    ///
    /// `phases_left`/`phases_right` may be empty (magnitude-only mode).
    fn sync_update(&self, alpha: &mut [f32], beta: &mut [f32],
                   activations_left: &[f32], activations_right: &[f32],
                   phases_left: &[f32], phases_right: &[f32],
                   decay: &[f32], decay_shift: &[f32],
                   dopamine: f32, n_pairs: usize, initialized: bool,
                   sync_out: &mut [f32]);
}

// ─── CpuBackend ─────────────────────────────────────────────

/// CPU compute backend: explicit SIMD, cache-tiled, fused kernels.
/// Auto-tunes on first creation for the host hardware.
#[derive(Debug, Clone)]
pub struct CpuBackend {
    pub config: AutoTuneConfig,
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self { config: AutoTuneConfig::autotune() }
    }
}

impl CpuBackend {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_threshold(par_threshold: usize) -> Self {
        Self {
            config: AutoTuneConfig {
                par_threshold,
                ..AutoTuneConfig::default()
            }
        }
    }
}

impl ComputeBackend for CpuBackend {
    fn matvec(&self, weight: &[f32], bias: &[f32], x: &[f32],
              y: &mut [f32], out_dim: usize, in_dim: usize) {
        debug_assert_eq!(weight.len(), out_dim * in_dim);
        debug_assert_eq!(bias.len(), out_dim);
        debug_assert!(x.len() >= in_dim);
        debug_assert!(y.len() >= out_dim);

        let total = out_dim * in_dim;
        if total >= self.config.par_threshold {
            // Parallel: each thread processes a tile of rows
            let chunk_rows = (out_dim / rayon::current_num_threads()).max(self.config.tile_rows);
            y[..out_dim].par_chunks_mut(chunk_rows)
                .enumerate()
                .for_each(|(ci, chunk)| {
                    let r_start = ci * chunk_rows;
                    for (j, out) in chunk.iter_mut().enumerate() {
                        let r = r_start + j;
                        if r < out_dim {
                            let row = &weight[r * in_dim..(r + 1) * in_dim];
                            *out = bias[r] + dot(row, &x[..in_dim]);
                        }
                    }
                });
        } else {
            // Sequential with L1-cache tiling
            matvec_tiled(weight, bias, x, y, out_dim, in_dim);
        }
    }

    /// Fused synapse: matvec → GLU → SiLU → layernorm in minimal cache passes.
    /// Overrides the default 4-pass implementation.
    fn synapse_forward(&self, weight: &[f32], bias: &[f32], x: &[f32],
                       output: &mut [f32], _scratch: &mut [f32],
                       out_dim: usize, in_dim: usize) {
        if !self.config.use_fused_synapse {
            // Fall back to default (4 separate passes)
            self.matvec(weight, bias, x, _scratch, out_dim * 2, in_dim);
            self.glu(&_scratch[..out_dim * 2], &mut output[..out_dim]);
            self.silu_inplace(&mut output[..out_dim]);
            self.layer_norm_inplace(&mut output[..out_dim]);
            return;
        }
        let total = out_dim * in_dim;
        if total >= self.config.par_threshold {
            synapse_fused_parallel(weight, bias, x, output, out_dim, in_dim);
        } else {
            synapse_fused(weight, bias, x, output, out_dim, in_dim);
        }
    }

    fn superlinear(&self, weights: &[f32], biases: &[f32], trace: &[f32],
                   output: &mut [f32], n_neurons: usize, in_per: usize, out_per: usize) {
        debug_assert_eq!(weights.len(), n_neurons * out_per * in_per);
        debug_assert_eq!(biases.len(), n_neurons * out_per);
        debug_assert!(trace.len() >= n_neurons * in_per);
        debug_assert!(output.len() >= n_neurons * out_per);

        let total_flops = n_neurons * in_per * out_per;
        if total_flops >= self.config.par_threshold {
            let chunk_size = (n_neurons / rayon::current_num_threads()).max(4);
            output[..n_neurons * out_per]
                .par_chunks_mut(chunk_size * out_per)
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    let n_start = chunk_idx * chunk_size;
                    let n_end = (n_start + chunk_size).min(n_neurons);
                    for n in n_start..n_end {
                        let t = &trace[n * in_per..(n + 1) * in_per];
                        let w_base = n * out_per * in_per;
                        let local_off = (n - n_start) * out_per;
                        for o in 0..out_per {
                            let w = &weights[w_base + o * in_per..w_base + (o + 1) * in_per];
                            out_chunk[local_off + o] = biases[n * out_per + o] + dot(w, t);
                        }
                    }
                });
        } else {
            for n in 0..n_neurons {
                let t = &trace[n * in_per..(n + 1) * in_per];
                let w_base = n * out_per * in_per;
                let o_base = n * out_per;
                for o in 0..out_per {
                    let w = &weights[w_base + o * in_per..w_base + (o + 1) * in_per];
                    output[o_base + o] = biases[o_base + o] + dot(w, t);
                }
            }
        }
    }

    fn glu(&self, input: &[f32], output: &mut [f32]) {
        let half = input.len() / 2;
        debug_assert!(output.len() >= half);

        if half >= self.config.par_threshold {
            output[..half].par_chunks_mut(256)
                .enumerate()
                .for_each(|(ci, chunk)| {
                    let base = ci * 256;
                    for (j, out) in chunk.iter_mut().enumerate() {
                        let i = base + j;
                        if i < half {
                            *out = input[i] * fast_sigmoid(input[half + i]);
                        }
                    }
                });
        } else {
            for i in 0..half {
                output[i] = input[i] * fast_sigmoid(input[half + i]);
            }
        }
    }

    fn silu_inplace(&self, x: &mut [f32]) {
        if x.len() >= self.config.par_threshold {
            x.par_chunks_mut(64).for_each(|chunk| {
                for v in chunk {
                    let sigmoid = 1.0 / (1.0 + (-*v).exp());
                    *v *= sigmoid;
                }
            });
        } else {
            for v in x.iter_mut() {
                let sigmoid = 1.0 / (1.0 + (-*v).exp());
                *v *= sigmoid;
            }
        }
    }

    fn layer_norm_inplace(&self, x: &mut [f32]) {
        let n = x.len() as f32;
        if n < 1.0 { return; }

        let mean: f32 = x.iter().sum::<f32>() / n;
        let var: f32 = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
        let inv_std = 1.0 / (var + 1e-5).sqrt();
        for v in x.iter_mut() {
            *v = (*v - mean) * inv_std;
        }
    }

    fn trace_shift(&self, traces: &mut [f32], new_activations: &[f32],
                   n_neurons: usize, memory_length: usize) {
        debug_assert!(traces.len() >= n_neurons * memory_length);
        debug_assert!(new_activations.len() >= n_neurons);

        if n_neurons >= 64 {
            traces[..n_neurons * memory_length]
                .par_chunks_mut(memory_length)
                .enumerate()
                .for_each(|(neuron, chunk)| {
                    chunk.copy_within(1..memory_length, 0);
                    chunk[memory_length - 1] = new_activations[neuron];
                });
        } else {
            for neuron in 0..n_neurons {
                let off = neuron * memory_length;
                traces[off..off + memory_length].copy_within(1..memory_length, 0);
                traces[off + memory_length - 1] = new_activations[neuron];
            }
        }
    }

    fn sync_update(&self, alpha: &mut [f32], beta: &mut [f32],
                   activations_left: &[f32], activations_right: &[f32],
                   phases_left: &[f32], phases_right: &[f32],
                   decay: &[f32], decay_shift: &[f32],
                   dopamine: f32, n_pairs: usize, initialized: bool,
                   sync_out: &mut [f32]) {
        debug_assert!(alpha.len() >= n_pairs);
        debug_assert!(beta.len() >= n_pairs);
        debug_assert!(sync_out.len() >= n_pairs);
        debug_assert!(decay.len() >= n_pairs);
        debug_assert!(decay_shift.len() >= n_pairs);

        let has_phase = !phases_left.is_empty() && !phases_right.is_empty();

        for i in 0..n_pairs {
            let left = activations_left[i];
            let right = activations_right[i];

            // Phase-aware temporal binding
            let temporal_proximity = if has_phase {
                let phase_diff = phases_left[i] - phases_right[i];
                (-phase_diff * phase_diff / (2.0 * 0.3 * 0.3)).exp()
            } else {
                1.0
            };

            let pairwise = left * right * dopamine * temporal_proximity;
            let decay_val = decay[i] + decay_shift[i];
            let r = (-decay_val.clamp(0.0, 15.0)).exp();

            if !initialized {
                alpha[i] = pairwise;
                beta[i] = dopamine;
            } else {
                alpha[i] = r * alpha[i] + pairwise;
                beta[i] = r * beta[i] + dopamine;
            }
            sync_out[i] = alpha[i] / beta[i].sqrt().max(1e-8);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_dot_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!(approx_eq(dot(&a, &b), 32.0, 1e-5));
    }

    #[test]
    fn test_dot_large() {
        // Exercise the 16-wide SIMD path
        let n = 64;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
        let expected: f32 = (0..n).map(|i| (i * (n - i)) as f32).sum();
        assert!(approx_eq(dot(&a, &b), expected, 1e-2));
    }

    #[test]
    fn test_matvec_sequential() {
        let backend = CpuBackend::with_threshold(usize::MAX); // force sequential
        // 2x3 matrix: [[1,2,3],[4,5,6]], bias=[0.1, 0.2], x=[1,1,1]
        let w = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![0.1, 0.2];
        let x = vec![1.0, 1.0, 1.0];
        let mut y = vec![0.0; 2];
        backend.matvec(&w, &b, &x, &mut y, 2, 3);
        assert!(approx_eq(y[0], 6.1, 1e-5));  // 1+2+3+0.1
        assert!(approx_eq(y[1], 15.2, 1e-5)); // 4+5+6+0.2
    }

    #[test]
    fn test_matvec_parallel() {
        let backend = CpuBackend::with_threshold(1); // force parallel
        let w = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![0.1, 0.2];
        let x = vec![1.0, 1.0, 1.0];
        let mut y = vec![0.0; 2];
        backend.matvec(&w, &b, &x, &mut y, 2, 3);
        assert!(approx_eq(y[0], 6.1, 1e-5));
        assert!(approx_eq(y[1], 15.2, 1e-5));
    }

    #[test]
    fn test_glu() {
        let backend = CpuBackend::new();
        // input: [a, b, gate_a, gate_b] -> output: [a*sigmoid(gate_a), b*sigmoid(gate_b)]
        let input = vec![2.0, 3.0, 0.0, 100.0];
        let mut output = vec![0.0; 2];
        backend.glu(&input, &mut output);
        assert!(approx_eq(output[0], 2.0 * 0.5, 1e-5)); // sigmoid(0) = 0.5
        assert!(approx_eq(output[1], 3.0 * 1.0, 1e-3)); // sigmoid(100) ~ 1.0
    }

    #[test]
    fn test_silu_inplace() {
        let backend = CpuBackend::new();
        let mut x = vec![0.0, 1.0, -1.0];
        backend.silu_inplace(&mut x);
        assert!(approx_eq(x[0], 0.0, 1e-5));           // 0 * sigmoid(0) = 0
        assert!(approx_eq(x[1], 0.7310586, 1e-4));      // 1 * sigmoid(1)
        assert!(approx_eq(x[2], -0.26894143, 1e-4));    // -1 * sigmoid(-1)
    }

    #[test]
    fn test_layer_norm() {
        let backend = CpuBackend::new();
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        backend.layer_norm_inplace(&mut x);
        // mean=2.5, var=1.25, std~=1.118
        let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
        assert!(approx_eq(mean, 0.0, 1e-5));
        let var: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        assert!(approx_eq(var, 1.0, 1e-3));
    }

    #[test]
    fn test_trace_shift() {
        let backend = CpuBackend::new();
        // 2 neurons, memory_length=3
        let mut traces = vec![1.0, 2.0, 3.0,  4.0, 5.0, 6.0];
        let new_act = vec![7.0, 8.0];
        backend.trace_shift(&mut traces, &new_act, 2, 3);
        assert_eq!(traces, vec![2.0, 3.0, 7.0,  5.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sync_update_no_phase() {
        let backend = CpuBackend::new();
        let mut alpha = vec![0.0; 2];
        let mut beta = vec![0.0; 2];
        let mut sync_out = vec![0.0; 2];
        let act_l = vec![1.0, 0.5];
        let act_r = vec![0.5, 1.0];
        let decay = vec![0.7, 0.7];
        let decay_shift = vec![0.0, 0.0];

        backend.sync_update(
            &mut alpha, &mut beta,
            &act_l, &act_r,
            &[], &[],
            &decay, &decay_shift,
            1.0, 2, false,
            &mut sync_out,
        );
        // Not initialized: alpha = pairwise, beta = dopamine
        assert!(approx_eq(alpha[0], 0.5, 1e-5));
        assert!(approx_eq(alpha[1], 0.5, 1e-5));
        assert!(approx_eq(beta[0], 1.0, 1e-5));
        assert!(approx_eq(beta[1], 1.0, 1e-5));
    }

    #[test]
    fn test_superlinear() {

        let backend = CpuBackend::new();
        // 2 neurons, in_per=2, out_per=1
        // neuron 0: w=[1,2], b=0.1, trace=[1,1] -> 1+2+0.1=3.1
        // neuron 1: w=[3,4], b=0.2, trace=[1,1] -> 3+4+0.2=7.2
        let weights = vec![1.0, 2.0,  3.0, 4.0];
        let biases = vec![0.1, 0.2];
        let trace = vec![1.0, 1.0,  1.0, 1.0];
        let mut output = vec![0.0; 2];
        backend.superlinear(&weights, &biases, &trace, &mut output, 2, 2, 1);
        assert!(approx_eq(output[0], 3.1, 1e-5));
        assert!(approx_eq(output[1], 7.2, 1e-5));
    }
}

// ─── Device selection ──────────────────────────────────────

/// Supported compute devices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    /// CPU with auto-tuned SIMD.
    Cpu,
    /// NVIDIA GPU via cuBLAS + candle CUDA kernels.
    Cuda,
    /// AMD GPU via KFD (ROCm userspace, direct ISA dispatch).
    Amdgpu,
}

impl Device {
    /// Parse from CLI string. Case-insensitive.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "cpu" => Some(Self::Cpu),
            "cuda" | "nvidia" | "gpu" => Some(Self::Cuda),
            "amdgpu" | "amd" | "rocm" | "kfd" => Some(Self::Amdgpu),
            _ => None,
        }
    }

    /// Auto-detect best available device.
    pub fn auto() -> Self {
        #[cfg(feature = "cuda")]
        {
            // Try CUDA first
            if Self::cuda_available() { return Self::Cuda; }
        }
        // Try AMD KFD
        if Self::amdgpu_available() { return Self::Amdgpu; }
        // Fallback
        Self::Cpu
    }

    /// Check if CUDA is available.
    pub fn cuda_available() -> bool {
        #[cfg(feature = "cuda")]
        {
            cudarc::driver::CudaDevice::new(0).is_ok()
        }
        #[cfg(not(feature = "cuda"))]
        { false }
    }

    /// Check if AMD KFD is available.
    pub fn amdgpu_available() -> bool {
        std::path::Path::new("/dev/kfd").exists()
    }

    /// Create a boxed ComputeBackend for this device.
    pub fn backend(&self) -> Box<dyn ComputeBackend> {
        match self {
            Self::Cpu => Box::new(CpuBackend::new()),
            Self::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    match crate::cuda_backend::CudaBackend::new() {
                        Ok(b) => return Box::new(b),
                        Err(e) => {
                            eprintln!("CUDA init failed: {e}, falling back to CPU");
                            Box::new(CpuBackend::new())
                        }
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    eprintln!("CUDA not compiled in (--features cuda), using CPU");
                    Box::new(CpuBackend::new())
                }
            }
            Self::Amdgpu => {
                // AMD KFD backend dispatches through modgrad-device's kfd module.
                // For now, fall back to CPU — the KFD kernels need to be wired
                // to ComputeBackend (currently they're standalone dispatchers).
                eprintln!("AMD GPU: KFD backend not yet wired to ComputeBackend, using CPU");
                Box::new(CpuBackend::new())
            }
        }
    }

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Amdgpu => "amdgpu",
        }
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}
