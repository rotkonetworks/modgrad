//! Generic compute primitives: Linear, activations, RNG.
//!
//! Pure building blocks with no runtime dependency.
//! isis-specific neuron layers (NeuronLayer, NeuronLayerWeights, etc.)
//! live in `crate::runtime::neuron`.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use wincode_derive::{SchemaRead, SchemaWrite};

use super::ops::dot;

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
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub struct Linear {
    pub weight: Vec<f32>,  // [out_dim × in_dim] row-major
    pub bias: Vec<f32>,    // [out_dim]
    pub in_dim: usize,
    pub out_dim: usize,
}

impl Linear {
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        let scale = (2.0 / in_dim as f32).sqrt();
        let mut rng = SimpleRng::new(in_dim as u64 ^ out_dim as u64);
        let weight: Vec<f32> = (0..out_dim * in_dim)
            .map(|_| rng.next_normal() * scale)
            .collect();
        let bias = vec![0.0; out_dim];
        Self { weight, bias, in_dim, out_dim }
    }

    /// Forward into pre-allocated output buffer. Zero allocation.
    /// Dispatches through `modgrad_device::backend::ops::matvec`, which
    /// routes through the `BackendRegistry` (KFD > ROCm > CUDA > CPU,
    /// shape-permitting).
    pub fn forward_into(&self, x: &[f32], y: &mut [f32]) {
        modgrad_device::backend::ops::matvec(
            x, &self.weight, &self.bias, y,
            self.out_dim, self.in_dim,
            modgrad_device::backend::QuantKind::F32,
        ).expect("matvec dispatch");
    }

    /// Allocating forward (backward compat). Prefer forward_into.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut y = vec![0.0f32; self.out_dim];
        self.forward_into(x, &mut y);
        y
    }

    /// Forward with VRAM-aware allocation. Output may be GPU-resident.
    /// Allocation goes through the lifecycle `ComputeBackend::alloc_f32`
    /// (heap on CPU, arena-backed VRAM on `VramGpuBackend`); the dispatch
    /// itself still goes through `ops::matvec` via `forward_into`.
    pub fn forward_gpu(&self, x: &[f32]) -> super::backend::GpuVec {
        let mut y = super::backend::backend().alloc_f32(self.out_dim);
        self.forward_into(x, &mut y);
        y
    }
}

/// Device-resident wrapper over a `Linear`. Uploads weight + bias to
/// hipMalloc'd buffers once, then `forward` runs hipblasSgemv with
/// device pointers — zero PCIe transfers per call.
///
/// Lifecycle:
///   - Construct via `LinearResident::from_linear(&lin)?`.
///   - Call `forward(x_dev, &mut out_dev)` per inference step.
///   - After an optimizer step (AdamW etc.) that mutated `lin.weight`,
///     call `sync_weights_from(&lin)` to re-upload.
///
/// Only available with `--features rocm`. Without that feature the
/// type doesn't exist — callers fall back to host-slice `Linear::forward`.
#[cfg(feature = "rocm")]
pub struct LinearResident {
    pub weight_dev: modgrad_device::backend::HipBuffer,
    pub bias_dev: modgrad_device::backend::HipBuffer,
    pub in_dim: usize,
    pub out_dim: usize,
}

#[cfg(feature = "rocm")]
impl LinearResident {
    /// Allocate device buffers and upload weight + bias.
    pub fn from_linear(lin: &Linear) -> Result<Self, String> {
        let weight_dev = modgrad_device::backend::HipBuffer::new(lin.weight.len() * 4)
            .map_err(|e| format!("hipMalloc(weight): {e:?}"))?;
        weight_dev.copy_from_host(&lin.weight)
            .map_err(|e| format!("upload weight: {e:?}"))?;
        let bias_dev = modgrad_device::backend::HipBuffer::new(lin.bias.len() * 4)
            .map_err(|e| format!("hipMalloc(bias): {e:?}"))?;
        bias_dev.copy_from_host(&lin.bias)
            .map_err(|e| format!("upload bias: {e:?}"))?;
        Ok(Self {
            weight_dev, bias_dev,
            in_dim: lin.in_dim, out_dim: lin.out_dim,
        })
    }

    /// Re-upload weights after an in-place optimizer step. Bias too,
    /// in case it was updated.
    pub fn sync_weights_from(&mut self, lin: &Linear) -> Result<(), String> {
        debug_assert_eq!(lin.in_dim, self.in_dim);
        debug_assert_eq!(lin.out_dim, self.out_dim);
        self.weight_dev.copy_from_host(&lin.weight)
            .map_err(|e| format!("re-upload weight: {e:?}"))?;
        self.bias_dev.copy_from_host(&lin.bias)
            .map_err(|e| format!("re-upload bias: {e:?}"))?;
        Ok(())
    }

    /// Resident forward: x and out are `GpuVec::Hip`. Zero PCIe
    /// transfers; everything stays on device.
    ///
    /// **Requires a `&HipBatch`.** The batch is the only construct
    /// that can guarantee the HIP command queue gets drained
    /// periodically; submitting kernels without one risks queue
    /// overflow → GPU hang → Xorg crash on single-GPU systems
    /// (see `memory/feedback_hip_queue_overflow.md`). By taking
    /// `&HipBatch` as a parameter we turn the sync requirement
    /// from a runtime contract into a compile-time obligation —
    /// callers cannot forget. The batch's `Drop` runs the final
    /// sync; intermediate syncs happen automatically every
    /// `HipBatch::DEFAULT_SYNC_EVERY` (256) dispatches.
    ///
    /// `out_dev` must already be allocated to `out_dim` f32s.
    pub fn forward(
        &self,
        batch: &modgrad_device::backend::HipBatch,
        x_dev: &super::backend::GpuVec,
        out_dev: &mut super::backend::GpuVec,
    ) -> Result<(), String> {
        use super::backend::GpuVec;
        debug_assert_eq!(x_dev.len(), self.in_dim);
        debug_assert_eq!(out_dev.len(), self.out_dim);
        let x_buf = match x_dev {
            GpuVec::Hip(b) => b,
            _ => return Err("LinearResident::forward requires GpuVec::Hip x".into()),
        };
        let out_buf = match out_dev {
            GpuVec::Hip(b) => b,
            _ => return Err("LinearResident::forward requires GpuVec::Hip out".into()),
        };
        unsafe {
            modgrad_device::backend::ops::matvec_resident(
                x_buf.device_ptr() as *const f32,
                self.weight_dev.device_ptr() as *const f32,
                self.bias_dev.device_ptr() as *const f32,
                out_buf.device_ptr() as *mut f32,
                self.out_dim,
                self.in_dim,
            ).map_err(|e| format!("matvec_resident: {e:?}"))?;
        }
        // Bookkeeping for the auto-sync cadence. If the batch's
        // pending count just hit sync_every, this drains the queue
        // before returning — keeping us strictly bounded against
        // the watchdog deadline.
        batch.note_dispatch().map_err(|e| format!("HipBatch::note_dispatch: {e:?}"))?;
        Ok(())
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
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
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

    /// Forward into pre-allocated buffer. Zero allocation.
    /// Dispatches through `modgrad_device::backend::ops::super_linear_fwd`,
    /// which routes via the `BackendRegistry`. Forward-only fused variant
    /// (`cache=None`).
    pub fn forward_into(&self, trace: &[f32], out: &mut [f32]) {
        modgrad_device::backend::ops::super_linear_fwd(
            trace, &self.weights, &self.biases, out, None,
            self.n_neurons, self.in_per, self.out_per,
        ).expect("super_linear_fwd dispatch");
    }

    /// CPU-only forward (used by backends internally).
    pub fn forward_cpu(&self, trace: &[f32], out: &mut [f32]) {
        let n_neurons = self.n_neurons;
        let in_per = self.in_per;
        let out_per = self.out_per;

        if n_neurons * in_per * out_per >= 100_000 {
            let chunk_size = (n_neurons / rayon::current_num_threads()).max(4);
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
        } else {
            // Sequential for small neuron counts
            for n in 0..n_neurons {
                let t = &trace[n * in_per..(n + 1) * in_per];
                let w_base = n * out_per * in_per;
                let o_base = n * out_per;
                for o in 0..out_per {
                    let w = &self.weights[w_base + o * in_per..w_base + (o + 1) * in_per];
                    out[o_base + o] = self.biases[o_base + o] + dot(w, t);
                }
            }
        }
    }

    /// Allocating forward (backward compat). Prefer forward_into.
    pub fn forward(&self, trace: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.n_neurons * self.out_per];
        self.forward_into(trace, &mut out);
        out
    }

    /// Forward with VRAM-aware allocation. Output may be GPU-resident.
    pub fn forward_gpu(&self, trace: &[f32]) -> super::backend::GpuVec {
        let mut out = super::backend::backend().alloc_f32(self.n_neurons * self.out_per);
        self.forward_into(trace, &mut out);
        out
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

#[cfg(test)]
#[cfg(feature = "rocm")]
mod resident_tests {
    use super::*;
    use modgrad_device::backend::rocm::ffi::runtime_available;

    /// LinearResident matches Linear (CPU) bit-by-bit-ish on a
    /// medium shape. Proves the device-resident dispatch produces
    /// the same arithmetic as the host path. Tolerance is loose
    /// (1e-3) because rocBLAS uses different accumulation order.
    #[test]
    fn linear_resident_matches_host() {
        if !runtime_available() {
            eprintln!("hip runtime unavailable, skipping");
            return;
        }
        let lin = Linear::new(64, 128);  // out_dim=128, in_dim=64
        let mut rng = SimpleRng::new(0x1234);
        let host_x: Vec<f32> = (0..64).map(|_| rng.next_normal()).collect();

        // Host reference path
        let mut host_y = vec![0.0f32; 128];
        lin.forward_into(&host_x, &mut host_y);

        // Device-resident path
        let resident = LinearResident::from_linear(&lin)
            .expect("LinearResident::from_linear");
        let mut x_dev = super::super::backend::GpuVec::try_hip(64).expect("alloc x");
        x_dev.copy_from(&host_x);
        let mut out_dev = super::super::backend::GpuVec::try_hip(128).expect("alloc out");

        let batch = modgrad_device::backend::HipBatch::new();
        resident.forward(&batch, &x_dev, &mut out_dev).expect("resident forward");
        batch.flush().expect("flush");

        let mut device_y = vec![0.0f32; 128];
        out_dev.copy_to_host(&mut device_y);

        let max_diff = host_y.iter().zip(&device_y)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-3,
            "host vs resident mismatch: max |Δ| = {max_diff}");
    }

    /// Calling forward many times in a row is the actual perf win
    /// path — weights uploaded ONCE, dispatched N times. Verify the
    /// loop runs without errors and produces consistent output.
    #[test]
    fn linear_resident_loop_no_drift() {
        if !runtime_available() {
            eprintln!("hip runtime unavailable, skipping");
            return;
        }
        let lin = Linear::new(64, 64);
        let mut rng = SimpleRng::new(0x5678);
        let host_x: Vec<f32> = (0..64).map(|_| rng.next_normal()).collect();

        let resident = LinearResident::from_linear(&lin).expect("resident");
        let mut x_dev = super::super::backend::GpuVec::try_hip(64).expect("x");
        x_dev.copy_from(&host_x);

        // Run 32 forwards in a row; output of every call should be
        // bit-identical (deterministic GPU execution, no PRNG).
        // The HipBatch ensures the queue stays bounded and a final
        // sync runs on Drop.
        let batch = modgrad_device::backend::HipBatch::new();
        let mut first_y: Option<Vec<f32>> = None;
        for _ in 0..32 {
            let mut out_dev = super::super::backend::GpuVec::try_hip(64).expect("out");
            resident.forward(&batch, &x_dev, &mut out_dev).expect("forward");
            // copy_to_host implicitly synchronises (hipMemcpy D2H is
            // synchronous against the default stream), so reads are
            // safe even mid-batch.
            let mut host_y = vec![0.0f32; 64];
            out_dev.copy_to_host(&mut host_y);
            match &first_y {
                None => first_y = Some(host_y),
                Some(y0) => assert_eq!(*y0, host_y, "drift across calls"),
            }
        }
    }
}

