//! Generic compute primitives: Linear, activations, RNG.
//!
//! Pure building blocks with no runtime dependency.
//! isis-specific neuron layers (NeuronLayer, NeuronLayerWeights, etc.)
//! live in `crate::runtime::neuron`.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use super::ops::dot;

/// Global GPU enable flag. Off by default. Caller sets via `enable_gpu()`.
static GPU_ENABLED: AtomicBool = AtomicBool::new(false);

// ─── Weight identity for GPU cache coherence ──────────────

/// Monotonic counter for unique weight IDs.
static WEIGHT_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Stable identity for a Linear layer's GPU cache entry.
/// Survives optimizer steps (unlike pointer identity which breaks on realloc).
/// Fresh ID on construction and clone — no aliasing.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WeightId(pub u64);

impl WeightId {
    pub fn next() -> Self { Self(WEIGHT_ID_COUNTER.fetch_add(1, Ordering::Relaxed)) }
}

impl Default for WeightId {
    fn default() -> Self { Self::next() }
}

/// Enable GPU dispatch for Linear::forward(). Call once at startup.
pub fn enable_gpu() { GPU_ENABLED.store(true, Ordering::Relaxed); }

/// Disable GPU dispatch.
pub fn disable_gpu() { GPU_ENABLED.store(false, Ordering::Relaxed); }

/// Check if GPU dispatch is enabled.
pub fn gpu_enabled() -> bool { GPU_ENABLED.load(Ordering::Relaxed) }

// ─── Activation functions ──────────────────────────────────

/// GLU activation: x[..half] * sigmoid(x[half..])
#[inline(always)]
pub fn glu(x: &[f32]) -> Vec<f32> {
    let half = x.len() / 2;
    let mut out = Vec::with_capacity(half);
    for i in 0..half {
        // Fast sigmoid: avoid exp() for small values
        let v = x[half + i];
        let gate = if v > 6.0 { 1.0 }
            else if v < -6.0 { 0.0 }
            else { 1.0 / (1.0 + (-v).exp()) };
        out.push(x[i] * gate);
    }
    out
}

/// GLU in-place: write result into `out` slice, avoiding allocation.
#[inline(always)]
pub fn glu_into(x: &[f32], out: &mut [f32]) {
    let half = x.len() / 2;
    for i in 0..half.min(out.len()) {
        let v = x[half + i];
        let gate = if v > 6.0 { 1.0 }
            else if v < -6.0 { 0.0 }
            else { 1.0 / (1.0 + (-v).exp()) };
        out[i] = x[i] * gate;
    }
}

pub fn layer_norm(x: &mut [f32]) {
    let n = x.len() as f32;
    let mean: f32 = x.iter().sum::<f32>() / n;
    let var: f32 = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    let std = (var + 1e-5).sqrt();
    for v in x.iter_mut() {
        *v = (*v - mean) / std;
    }
}

// ─── Weight matrices ────────────────────────────────────────

/// Dense linear layer: y = Wx + b
///
/// GPU coherence: `weight_id` is a stable handle for the GPU weight cache.
/// `cpu_gen` is bumped on any CPU-side weight mutation. The GPU cache
/// compares generations to decide when to re-upload.
#[derive(Debug, Serialize, Deserialize)]
pub struct Linear {
    pub weight: Vec<f32>,  // [out_dim × in_dim] row-major
    pub bias: Vec<f32>,    // [out_dim]
    pub in_dim: usize,
    pub out_dim: usize,
    /// Stable ID for GPU weight cache. Fresh on construction/clone/deserialize.
    #[serde(skip, default = "WeightId::next")]
    pub weight_id: WeightId,
    /// Bumped on CPU-side weight mutation. GPU re-uploads when stale.
    #[serde(skip, default)]
    pub cpu_gen: u64,
}

impl Clone for Linear {
    fn clone(&self) -> Self {
        Self {
            weight: self.weight.clone(),
            bias: self.bias.clone(),
            in_dim: self.in_dim,
            out_dim: self.out_dim,
            weight_id: WeightId::next(), // fresh ID — own GPU cache entry
            cpu_gen: 0,                  // force re-upload
        }
    }
}

impl Linear {
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        let scale = (2.0 / in_dim as f32).sqrt();
        let mut rng = SimpleRng::new(in_dim as u64 ^ out_dim as u64);
        let weight: Vec<f32> = (0..out_dim * in_dim)
            .map(|_| rng.next_normal() * scale)
            .collect();
        let bias = vec![0.0; out_dim];
        Self { weight, bias, in_dim, out_dim, weight_id: WeightId::next(), cpu_gen: 0 }
    }

    /// Mark weights as modified on CPU. GPU cache will re-upload on next use.
    pub fn mark_dirty(&mut self) { self.cpu_gen += 1; }

    /// GPU-resident forward: weights stay in VRAM, only x uploaded per call.
    fn try_gpu_forward(&self, x: &[f32]) -> Option<Vec<f32>> {
        use modgrad_device::weight_cache::weight_cache;

        let mut cache = weight_cache().lock().ok()?;
        let id = self.weight_id.0;

        // Ensure GPU has current weights (no-op if already uploaded and gen matches)
        if !cache.ensure_uploaded(id, self.cpu_gen, &self.weight, &self.bias,
                                   self.out_dim, self.in_dim) {
            return None;
        }

        let dev = modgrad_device::kfd::accel::get_device()?;

        // Upload x to scratch (reuse buffer)
        let x_bytes = x.len() * 4;
        if cache.scratch_x_cap < x_bytes {
            cache.scratch_x = dev.upload_f32(x).ok();
            cache.scratch_x_cap = ((x_bytes + 4095) & !4095).max(4096);
        } else if let Some(ref xb) = cache.scratch_x {
            xb.write_f32(0, x);
        }

        let y_bytes = self.out_dim * 4;
        if cache.scratch_y_cap < y_bytes {
            cache.scratch_y = dev.alloc_output(((y_bytes + 4095) & !4095).max(4096)).ok();
            cache.scratch_y_cap = ((y_bytes + 4095) & !4095).max(4096);
        } else if let Some(ref yb) = cache.scratch_y {
            unsafe { std::ptr::write_bytes(yb.cpu_ptr, 0, y_bytes); }
        }

        let (x_buf, y_buf) = match (&cache.scratch_x, &cache.scratch_y) {
            (Some(x), Some(y)) => (x, y),
            _ => return None,
        };

        let entry = cache.get(id)?;
        if dev.dispatch_matvec(&entry.weight, &entry.bias, x_buf, y_buf,
                               self.out_dim as u32, self.in_dim as u32) {
            let y_slice = unsafe {
                std::slice::from_raw_parts(y_buf.cpu_ptr as *const f32, self.out_dim)
            };
            let mut y = vec![0.0f32; self.out_dim];
            y.copy_from_slice(y_slice);
            Some(y)
        } else {
            None
        }
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        // GPU: MegaTrain-style streaming (layer-wise, one dispatch at a time)
        if GPU_ENABLED.load(std::sync::atomic::Ordering::Relaxed)
            && self.in_dim * self.out_dim >= 8_000_000
        {
            let mut y = vec![0.0f32; self.out_dim];
            if modgrad_device::kfd::accel::try_stream_matvec(
                x, &self.weight, &self.bias, &mut y,
                self.out_dim as u32, self.in_dim as u32,
            ) {
                return y;
            }
        }

        // CPU fallback
        let mut y = Vec::with_capacity(self.out_dim);
        for r in 0..self.out_dim {
            let row = &self.weight[r * self.in_dim..(r + 1) * self.in_dim];
            y.push(self.bias[r] + dot(row, x));
        }
        y
    }
}

/// Minimal PRNG for weight init.
#[derive(Debug, Clone)]
pub struct SimpleRng(u64);

impl SimpleRng {
    pub fn new(seed: u64) -> Self { Self(seed.wrapping_add(1)) }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }

    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    pub fn next_normal(&mut self) -> f32 {
        // Box-Muller
        let u1 = self.next_f32().max(1e-10);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

// ─── SuperLinear (per-neuron MLP) ───────────────────────────

/// Per-neuron parallel MLP: each neuron has its own weight matrix.
/// Input: [n_neurons, memory_length] → Output: [n_neurons, out_per_neuron]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuperLinear {
    /// Weights: [n_neurons × out_per_neuron × in_per_neuron]
    pub weights: Vec<f32>,
    pub biases: Vec<f32>,  // [n_neurons × out_per_neuron]
    pub n_neurons: usize,
    pub in_per: usize,
    pub out_per: usize,
}

impl SuperLinear {
    pub fn new(n_neurons: usize, in_per: usize, out_per: usize) -> Self {
        let scale = (2.0 / in_per as f32).sqrt();
        let mut rng = SimpleRng::new((n_neurons * in_per * out_per) as u64);
        let weights: Vec<f32> = (0..n_neurons * out_per * in_per)
            .map(|_| rng.next_normal() * scale)
            .collect();
        let biases = vec![0.0; n_neurons * out_per];
        Self { weights, biases, n_neurons, in_per, out_per }
    }

    /// Forward: trace[n_neurons][in_per] → activated[n_neurons][out_per]
    ///
    /// Dispatch priority: CUDA → Vulkan → rayon (CPU parallel) → sequential.
    pub fn forward(&self, trace: &[f32]) -> Vec<f32> {
        let n_neurons = self.n_neurons;
        let in_per = self.in_per;
        let out_per = self.out_per;
        let total_flops = n_neurons * in_per * out_per;

        // Try GPU first (only worth it for large dispatches)
        if total_flops >= 100_000 {
            let mut out = vec![0.0f32; n_neurons * out_per];
            // CUDA (NVIDIA) first
            if modgrad_device::cuda::try_superlinear(
                trace, &self.weights, &self.biases, &mut out,
                n_neurons as u32, in_per as u32, out_per as u32,
            ) {
                return out;
            }
            // Vulkan (AMD/Intel/portable) fallback
            if modgrad_device::gpu::try_superlinear(
                trace, &self.weights, &self.biases, &mut out,
                n_neurons as u32, in_per as u32, out_per as u32,
            ) {
                return out;
            }
        }

        // CPU parallel (rayon) for medium workloads
        if total_flops >= 100_000 {
            // Parallel: chunks of neurons distributed across threads.
            // Each thread processes a batch to amortize rayon scheduling overhead.
            let chunk_size = (n_neurons / rayon::current_num_threads()).max(4);
            let mut out = vec![0.0f32; n_neurons * out_per];
            out.par_chunks_mut(chunk_size * out_per)
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    let n_start = chunk_idx * chunk_size;
                    let n_end = (n_start + chunk_size).min(n_neurons);
                    for n in n_start..n_end {
                        let t = &trace[n * in_per..(n + 1) * in_per];
                        let w_base = n * out_per * in_per;
                        let local_off = (n - n_start) * out_per;
                        for o in 0..out_per {
                            let w = &self.weights[w_base + o * in_per..w_base + (o + 1) * in_per];
                            out_chunk[local_off + o] = self.biases[n * out_per + o] + dot(w, t);
                        }
                    }
                });
            out
        } else {
            // Sequential for small neuron counts (rayon overhead > compute)
            let mut out = vec![0.0f32; n_neurons * out_per];
            for n in 0..n_neurons {
                let t = &trace[n * in_per..(n + 1) * in_per];
                let w_base = n * out_per * in_per;
                let o_base = n * out_per;
                for o in 0..out_per {
                    let w = &self.weights[w_base + o * in_per..w_base + (o + 1) * in_per];
                    out[o_base + o] = self.biases[o_base + o] + dot(w, t);
                }
            }
            out
        }
    }
}

// ─── Helpers ────────────────────────────────────────────────

pub fn concat(slices: &[&[f32]]) -> Vec<f32> {
    let total: usize = slices.iter().map(|s| s.len()).sum();
    let mut out = Vec::with_capacity(total);
    for s in slices {
        out.extend_from_slice(s);
    }
    out
}

pub fn maybe_broadcast(local: &[f32], global: &[f32], receives: bool) -> Vec<f32> {
    if receives {
        concat(&[local, global])
    } else {
        local.to_vec()
    }
}

/// Simple scaled dot-product attention: query × observation.
/// query: [n_sync], observation: [d_input]
/// Returns: [d_input] weighted observation.
pub fn simple_attention(query: &[f32], observation: &[f32], d_input: usize) -> Vec<f32> {
    // For single KV pair, attention is just a scaled gate
    let q_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    let scale = 1.0 / (d_input as f32).sqrt();
    // Use mean of query as attention weight
    let weight = (query.iter().sum::<f32>() / q_norm * scale).tanh();
    observation.iter().map(|&v| v * weight).collect()
}
