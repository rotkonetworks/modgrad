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
    /// Allocate device buffers and upload weight + bias. Returns
    /// `ResidencyError::Backend(_)` on hipMalloc / hipMemcpy failure;
    /// match the inner `BackendError` variant for typed recovery.
    pub fn from_linear(lin: &Linear) -> Result<Self, super::backend::ResidencyError> {
        let weight_dev = modgrad_device::backend::HipBuffer::new(lin.weight.len() * 4)?;
        weight_dev.copy_from_host(&lin.weight)?;
        let bias_dev = modgrad_device::backend::HipBuffer::new(lin.bias.len() * 4)?;
        bias_dev.copy_from_host(&lin.bias)?;
        Ok(Self {
            weight_dev, bias_dev,
            in_dim: lin.in_dim, out_dim: lin.out_dim,
        })
    }

    /// Re-upload weights after an in-place optimizer step. Bias too,
    /// in case it was updated.
    pub fn sync_weights_from(&mut self, lin: &Linear) -> Result<(), super::backend::ResidencyError> {
        debug_assert_eq!(lin.in_dim, self.in_dim);
        debug_assert_eq!(lin.out_dim, self.out_dim);
        self.weight_dev.copy_from_host(&lin.weight)?;
        self.bias_dev.copy_from_host(&lin.bias)?;
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
    ) -> Result<(), super::backend::ResidencyError> {
        use super::backend::{GpuVec, ResidencyError};
        debug_assert_eq!(x_dev.len(), self.in_dim);
        debug_assert_eq!(out_dev.len(), self.out_dim);
        let x_buf = match x_dev {
            GpuVec::Hip(b) => b,
            other => return Err(ResidencyError::WrongVariant {
                expected: "Hip", got: other.variant_name(),
            }),
        };
        let out_buf = match out_dev {
            GpuVec::Hip(b) => b,
            other => return Err(ResidencyError::WrongVariant {
                expected: "Hip", got: other.variant_name(),
            }),
        };
        unsafe {
            modgrad_device::backend::ops::matvec_resident(
                x_buf.device_ptr() as *const f32,
                self.weight_dev.device_ptr() as *const f32,
                self.bias_dev.device_ptr() as *const f32,
                out_buf.device_ptr() as *mut f32,
                self.out_dim,
                self.in_dim,
            )?;
        }
        // Bookkeeping for the auto-sync cadence. If the batch's
        // pending count just hit sync_every, this drains the queue
        // before returning — keeping us strictly bounded against
        // the watchdog deadline.
        batch.note_dispatch()?;
        Ok(())
    }
}

/// **Mixed-precision** device-resident wrapper. The device-side
/// weight + bias are bf16 (half the size of fp32), and the forward
/// path runs `hipblasGemmEx` with bf16 input/output and fp32 compute
/// — the standard mixed-precision recipe.
///
/// Master fp32 copies live host-side: AdamW updates the fp32 master,
/// then `sync_from_master` re-quantises and re-uploads. fp32 master
/// weights preserve optimiser stability without inflating the
/// activation footprint (forward / backward pass tensors stay bf16).
///
/// Lifecycle:
///   - `LinearResidentBf16::from_linear(&lin)?` — quantise lin's fp32
///     weights to bf16 + upload. Master copy starts equal to `lin`.
///   - `forward(batch, x_dev, &mut out_dev)` — bf16 in, bf16 out.
///   - After AdamW (which runs on the host fp32 master),
///     `sync_from_master()` re-quantises and re-uploads.
///
/// Only available with `--features rocm`.
#[cfg(feature = "rocm")]
pub struct LinearResidentBf16 {
    /// bf16 weights on device (in-VRAM, half the size of fp32).
    pub weight_dev: modgrad_device::backend::HipBuffer,
    /// bf16 bias on device.
    pub bias_dev: modgrad_device::backend::HipBuffer,
    /// fp32 master copy of weights — for AdamW stability.
    pub host_master_weight: Vec<f32>,
    /// fp32 master copy of bias.
    pub host_master_bias: Vec<f32>,
    pub in_dim: usize,
    pub out_dim: usize,
}

#[cfg(feature = "rocm")]
impl LinearResidentBf16 {
    /// Build a bf16 resident wrapper from an fp32 `Linear`. Quantises
    /// weights + bias to bf16 once and uploads; the fp32 master is
    /// retained for the AdamW path.
    pub fn from_linear(lin: &Linear) -> Result<Self, super::backend::ResidencyError> {
        // bf16 storage is half the bytes of fp32. We can't pass a `&[u16]`
        // straight to `HipBuffer::copy_from_host` (it expects `&[f32]`),
        // so pre-quantise into a `Vec<u16>` and reinterpret as bytes via
        // a fresh fp32 view of the same allocation. The byte count is
        // identical either way.
        use modgrad_device::backend::op::f32_to_bf16;

        let wt_bf16: Vec<u16> = lin.weight.iter().map(|&v| f32_to_bf16(v)).collect();
        let bs_bf16: Vec<u16> = lin.bias.iter().map(|&v| f32_to_bf16(v)).collect();

        let weight_dev = modgrad_device::backend::HipBuffer::new(wt_bf16.len() * 2)?;
        let bias_dev = modgrad_device::backend::HipBuffer::new(bs_bf16.len() * 2)?;

        upload_u16(&weight_dev, &wt_bf16)?;
        upload_u16(&bias_dev, &bs_bf16)?;

        Ok(Self {
            weight_dev, bias_dev,
            host_master_weight: lin.weight.clone(),
            host_master_bias: lin.bias.clone(),
            in_dim: lin.in_dim,
            out_dim: lin.out_dim,
        })
    }

    /// Re-quantise the fp32 master to bf16 and re-upload. Call after
    /// every AdamW step that mutated `host_master_weight` /
    /// `host_master_bias`.
    pub fn sync_from_master(&mut self) -> Result<(), super::backend::ResidencyError> {
        use modgrad_device::backend::op::f32_to_bf16;
        let wt_bf16: Vec<u16> = self.host_master_weight.iter().map(|&v| f32_to_bf16(v)).collect();
        let bs_bf16: Vec<u16> = self.host_master_bias.iter().map(|&v| f32_to_bf16(v)).collect();
        upload_u16(&self.weight_dev, &wt_bf16)?;
        upload_u16(&self.bias_dev, &bs_bf16)?;
        Ok(())
    }

    /// bf16 forward — `x_dev` and `out_dev` must be bf16-flavored
    /// `GpuVec::Hip` buffers (from `GpuVec::try_hip_bf16`). The fp32
    /// element count is *not* a meaningful interpretation of these
    /// buffers; this method takes raw `&GpuVec` for plumbing
    /// uniformity but reads only the underlying `HipBuffer`.
    pub fn forward(
        &self,
        batch: &modgrad_device::backend::HipBatch,
        x_dev: &super::backend::GpuVec,
        out_dev: &mut super::backend::GpuVec,
    ) -> Result<(), super::backend::ResidencyError> {
        use super::backend::{GpuVec, ResidencyError};
        let x_buf = match x_dev {
            GpuVec::Hip(b) => b,
            other => return Err(ResidencyError::WrongVariant {
                expected: "Hip", got: other.variant_name(),
            }),
        };
        let out_buf = match out_dev {
            GpuVec::Hip(b) => b,
            other => return Err(ResidencyError::WrongVariant {
                expected: "Hip", got: other.variant_name(),
            }),
        };
        unsafe {
            modgrad_device::backend::ops::matvec_resident_bf16(
                x_buf.device_ptr() as *const u16,
                self.weight_dev.device_ptr() as *const u16,
                self.bias_dev.device_ptr() as *const u16,
                out_buf.device_ptr() as *mut u16,
                self.out_dim,
                self.in_dim,
            )?;
        }
        batch.note_dispatch()?;
        Ok(())
    }
}

/// Helper: upload a `&[u16]` (bf16 stored as `u16`) into a `HipBuffer`
/// via the same `hipMemcpy(H2D)` as the fp32 path. The HipBuffer
/// `copy_from_host` API is fp32-typed; we reinterpret the slice as
/// bytes-of-the-correct-length so the underlying memcpy is identical.
#[cfg(feature = "rocm")]
fn upload_u16(buf: &modgrad_device::backend::HipBuffer, src: &[u16])
    -> Result<(), super::backend::ResidencyError>
{
    // Reinterpret `[u16]` as `[f32]` of half the length. bf16 buffers
    // are aligned to 2 bytes; f32 needs 4-byte alignment, but since we
    // only call `copy_from_host` (which routes through hipMemcpy
    // ultimately taking a `*const c_void`) the alignment is not
    // dereferenced — only `as_ptr() / len * 4` are read by the FFI.
    // Pad the src to an even u16 count so the f32 view has integer
    // length; the HipBuffer was allocated to `src.len() * 2` bytes
    // (matching `n * 2`), so a one-element pad just costs 2 bytes per
    // odd dim.
    let n_u16 = src.len();
    if n_u16 % 2 == 0 {
        // Safety: we only reinterpret to satisfy the f32-typed copy
        // API; alignment of u16 (2) is < f32 (4) but the underlying
        // memcpy never derefs through the pointer as f32.
        let view = unsafe {
            std::slice::from_raw_parts(src.as_ptr() as *const f32, n_u16 / 2)
        };
        buf.copy_from_host(view)?;
    } else {
        // Odd length — pad to even and upload via a temporary owned
        // vector so the memcpy still sees an aligned source.
        let mut padded: Vec<u16> = Vec::with_capacity(n_u16 + 1);
        padded.extend_from_slice(src);
        padded.push(0);
        let view = unsafe {
            std::slice::from_raw_parts(padded.as_ptr() as *const f32, padded.len() / 2)
        };
        buf.copy_from_host(view)?;
    }
    Ok(())
}

/// Streaming variant of `LinearResident` — the host weights live in an
/// `Arc<Linear>` shared with the optimizer, and the device buffers are
/// allocated/uploaded on demand and (optionally) torn down right after
/// dispatch. Foundation for the foundation-model layer-streaming use
/// case where the full weight set doesn't fit in 8GB VRAM and layers
/// have to rotate through.
///
/// Lifecycle:
///   - `LinearResidentStreaming::from_linear_arc(host)` — no upload yet.
///   - `forward(batch, x_dev, &mut out_dev, auto_evict)` — uploads if not
///     already resident, dispatches, optionally evicts.
///   - `evict()` — drop device buffers (frees VRAM via `hipFree`).
///
/// **Caller-mutation contract.** `LinearResidentStreaming` does *not*
/// hash or generation-count the host weights. If the caller mutates
/// `Arc<Linear>` (typically via `Arc::get_mut` after an AdamW step),
/// the caller MUST call `evict()` before the next `forward` so the
/// next dispatch re-uploads the new weights. This mirrors the
/// `sync_weights_from` contract on the non-streaming `LinearResident`.
/// We keep the contract caller-side because adding a generation
/// counter to `Linear` would be a much wider surface change than this
/// slice warrants — see the design notes in the implementing PR.
///
/// Only available with `--features rocm`.
#[cfg(feature = "rocm")]
pub struct LinearResidentStreaming {
    /// Host-side weights — shared with the optimizer (so AdamW updates
    /// land in this same memory). Cloning the `Arc` is O(1); the
    /// streamer does not own the only reference.
    host: std::sync::Arc<Linear>,
    /// Device-side weight buffer, present only when uploaded. `None`
    /// means not currently resident in VRAM.
    weight_dev: Option<modgrad_device::backend::HipBuffer>,
    /// Device-side bias buffer, present only when uploaded.
    bias_dev: Option<modgrad_device::backend::HipBuffer>,
}

#[cfg(feature = "rocm")]
impl LinearResidentStreaming {
    /// Wrap a host `Arc<Linear>` for streaming. Does *not* upload — the
    /// first `forward` (or an explicit `ensure_resident()`) does that.
    pub fn from_linear_arc(host: std::sync::Arc<Linear>) -> Self {
        Self { host, weight_dev: None, bias_dev: None }
    }

    /// Read-only view of the host `Arc<Linear>`. Useful if the caller
    /// wants to clone the `Arc` for the optimizer.
    pub fn host(&self) -> &std::sync::Arc<Linear> { &self.host }

    /// Mutable access to the inner `Arc<Linear>` slot. Useful when the
    /// caller is the *only* holder of strong references and wants to
    /// `Arc::get_mut` for an in-place weight rewrite (test scaffolding,
    /// or rare cases where the streamer also owns the optimizer's
    /// reference). In production the optimizer typically clones its
    /// own `Arc` and mutates through that — both copies see the same
    /// underlying allocation, so the streamer's view stays consistent.
    pub fn host_arc_mut(&mut self) -> &mut std::sync::Arc<Linear> {
        &mut self.host
    }

    /// Are weight + bias currently uploaded to VRAM?
    pub fn is_resident(&self) -> bool {
        self.weight_dev.is_some() && self.bias_dev.is_some()
    }

    /// Input dimension (from the host `Linear`).
    pub fn in_dim(&self) -> usize { self.host.in_dim }

    /// Output dimension (from the host `Linear`).
    pub fn out_dim(&self) -> usize { self.host.out_dim }

    /// Ensure both device buffers are allocated and contain the current
    /// host weights. No-op if already resident. After an `evict()` (or
    /// for the very first dispatch) this allocates fresh `HipBuffer`s
    /// and uploads — the cost is one `hipMalloc` + one H2D `hipMemcpy`
    /// per buffer, dominated by the H2D for typical weight sizes.
    pub fn ensure_resident(&mut self) -> Result<(), super::backend::ResidencyError> {
        if self.weight_dev.is_none() {
            let buf = modgrad_device::backend::HipBuffer::new(self.host.weight.len() * 4)?;
            buf.copy_from_host(&self.host.weight)?;
            self.weight_dev = Some(buf);
        }
        if self.bias_dev.is_none() {
            let buf = modgrad_device::backend::HipBuffer::new(self.host.bias.len() * 4)?;
            buf.copy_from_host(&self.host.bias)?;
            self.bias_dev = Some(buf);
        }
        Ok(())
    }

    /// Drop the device buffers. `hipFree` runs on each `HipBuffer::Drop`,
    /// returning the VRAM to the allocator. The next `forward` (or
    /// `ensure_resident()`) will re-upload from the host.
    pub fn evict(&mut self) {
        // Order doesn't matter for hipFree; just drop both.
        self.weight_dev = None;
        self.bias_dev = None;
    }

    /// Streaming forward — uploads if needed, dispatches, optionally
    /// evicts. `auto_evict = true` is the tight-memory path: the device
    /// buffers are released immediately after the dispatch is recorded
    /// on the queue. `auto_evict = false` keeps the buffers resident
    /// for the next call (hot path, equivalent to the non-streaming
    /// `LinearResident` once the first upload has happened).
    ///
    /// **Requires a `&HipBatch`** for the same queue-bookkeeping reason
    /// as `LinearResident::forward`. See its doc-comment for the
    /// compile-time-obligation rationale.
    ///
    /// Note on `auto_evict` semantics: dropping the `HipBuffer` calls
    /// `hipFree`, which on the runtime we use is synchronous against
    /// in-flight work touching that allocation. The dispatch's
    /// `note_dispatch` may have triggered an in-batch sync already; if
    /// not, the eventual `HipBatch` flush (or the explicit `flush()` /
    /// `Drop`) will drain the queue. We deliberately do NOT add an
    /// eager sync here — the batch's invariant is the canonical
    /// place for that, and double-syncing would cost throughput in
    /// the non-evict case.
    pub fn forward(
        &mut self,
        batch: &modgrad_device::backend::HipBatch,
        x_dev: &super::backend::GpuVec,
        out_dev: &mut super::backend::GpuVec,
        auto_evict: bool,
    ) -> Result<(), super::backend::ResidencyError> {
        use super::backend::{GpuVec, ResidencyError};
        debug_assert_eq!(x_dev.len(), self.host.in_dim);
        debug_assert_eq!(out_dev.len(), self.host.out_dim);

        // Stage 1: make sure weights are on device.
        self.ensure_resident()?;

        // Stage 2: unwrap GpuVec variants and dispatch. The
        // `is_resident()` postcondition of `ensure_resident` makes
        // these `Option::unwrap`s infallible, but we still match
        // defensively to keep the error surface honest.
        let x_buf = match x_dev {
            GpuVec::Hip(b) => b,
            other => return Err(ResidencyError::WrongVariant {
                expected: "Hip", got: other.variant_name(),
            }),
        };
        let out_buf = match out_dev {
            GpuVec::Hip(b) => b,
            other => return Err(ResidencyError::WrongVariant {
                expected: "Hip", got: other.variant_name(),
            }),
        };
        let weight_dev = self.weight_dev.as_ref()
            .expect("ensure_resident postcondition: weight_dev is Some");
        let bias_dev = self.bias_dev.as_ref()
            .expect("ensure_resident postcondition: bias_dev is Some");

        unsafe {
            modgrad_device::backend::ops::matvec_resident(
                x_buf.device_ptr() as *const f32,
                weight_dev.device_ptr() as *const f32,
                bias_dev.device_ptr() as *const f32,
                out_buf.device_ptr() as *mut f32,
                self.host.out_dim,
                self.host.in_dim,
            )?;
        }
        batch.note_dispatch()?;

        // Stage 3: if requested, drop device buffers now. The next
        // `forward` will re-upload from `self.host`.
        if auto_evict {
            self.evict();
        }
        Ok(())
    }
}

/// Q4_K-quantised, on-demand-dequantised wrapper over a `Linear`.
///
/// Storage model:
///   - `host_weight_q4k` — Q4_K_M-formatted bytes, ~12.5% of fp32
///     (1 fp32 → 4.5 bits → ~9× compression for the weight tensor).
///   - `host_bias` — fp32, unchanged. Bias is small (one entry per
///     output neuron); not worth quantising.
///   - `weight_dev` / `bias_dev` — fp32 on device, allocated and
///     populated on `ensure_resident`, dropped on `evict`.
///
/// The on-device weight is fp32, not Q4_K, so the matvec dispatch can
/// reuse the existing `matvec_resident` path. The compression ratio
/// is on the *host* side: a 7B-param model that would not fit in
/// 8GB VRAM as fp32 (28 GB) sits in 3.5 GB as Q4_K, and the streaming
/// path rotates fp32 layers through VRAM as needed.
///
/// Lifecycle:
///   - `LinearResidentQuantized::from_linear(&lin)?` — quantises lin's
///     weight to Q4_K bytes, keeps bias fp32, no device allocation yet.
///   - `forward(batch, x_dev, &mut out_dev, auto_evict)` — uploads
///     Q4_K bytes, dequantises into a device fp32 buffer, dispatches
///     matvec, optionally evicts.
///   - `evict()` — drop device buffers (Q4_K weight + dequantised
///     fp32 weight + bias). Host bytes survive.
///
/// The Q4_K weight is uploaded once per resident lifetime; the
/// dequant kernel runs once per `ensure_resident` (idempotent for
/// `auto_evict=false`). The forward path itself is identical to
/// `LinearResident::forward` once the weight is dequantised.
///
/// Only available with `--features rocm`.
#[cfg(feature = "rocm")]
pub struct LinearResidentQuantized {
    /// Q4_K-quantised weight on HOST. Storage = ~12.5% of fp32.
    /// Length = `(out_dim * in_dim / 256) * 144` bytes; the caller
    /// is responsible for ensuring `out_dim * in_dim % 256 == 0`
    /// (Q4_K's block size is 256 elements).
    host_weight_q4k: Vec<u8>,
    /// fp32 bias on host (small — keep precision).
    host_bias: Vec<f32>,
    /// Q4_K-quantised weight on DEVICE — staged here once, the
    /// dequant kernel reads from it. Dropped on `evict`.
    weight_q4k_dev: Option<modgrad_device::backend::HipBuffer>,
    /// Device fp32 weight (dequantised from `weight_q4k_dev`).
    /// Allocated on first `ensure_resident`; dropped on `evict`.
    weight_dev: Option<modgrad_device::backend::HipBuffer>,
    /// Device fp32 bias. Same lifecycle as `weight_dev`.
    bias_dev: Option<modgrad_device::backend::HipBuffer>,
    in_dim: usize,
    out_dim: usize,
}

#[cfg(feature = "rocm")]
impl LinearResidentQuantized {
    /// Q4_K block size (in elements). One block packs 256 fp32 values
    /// into 144 bytes.
    pub const Q4K_BLOCK_ELEMS: usize = 256;

    /// Quantise `lin.weight` to Q4_K, store both on host. Bias kept
    /// fp32. Returns `WrongVariant` if the weight tensor's element
    /// count isn't a multiple of 256 — Q4_K's block size — because
    /// padding would either change the semantics (zero-fill ⇒ wrong
    /// dequant for the trailing elements) or silently truncate
    /// (lossy). Callers must size their `Linear` so the weight is a
    /// clean multiple of 256.
    pub fn from_linear(lin: &Linear) -> Result<Self, super::backend::ResidencyError> {
        use super::backend::ResidencyError;
        if lin.weight.len() % Self::Q4K_BLOCK_ELEMS != 0 {
            return Err(ResidencyError::WrongVariant {
                expected: "weight len multiple of 256",
                got: "weight len not aligned to Q4_K block",
            });
        }
        let n_blocks = lin.weight.len() / Self::Q4K_BLOCK_ELEMS;
        let mut q4k_bytes = vec![0u8; n_blocks * 144];
        modgrad_device::kfd::gguf::quantize_row_q4_k(&lin.weight, &mut q4k_bytes);
        Ok(Self {
            host_weight_q4k: q4k_bytes,
            host_bias: lin.bias.clone(),
            weight_q4k_dev: None,
            weight_dev: None,
            bias_dev: None,
            in_dim: lin.in_dim,
            out_dim: lin.out_dim,
        })
    }

    /// Are device buffers currently allocated + populated?
    pub fn is_resident(&self) -> bool {
        self.weight_dev.is_some()
            && self.bias_dev.is_some()
            && self.weight_q4k_dev.is_some()
    }

    /// Input dimension.
    pub fn in_dim(&self) -> usize { self.in_dim }
    /// Output dimension.
    pub fn out_dim(&self) -> usize { self.out_dim }

    /// Bytes occupied by the host-side state — this is the foundation
    /// model's actual on-disk / in-RAM cost. Q4_K weight + fp32 bias.
    /// Does NOT include the per-instance `Vec` allocator overhead;
    /// for a multi-GB weight that's negligible.
    pub fn host_size_bytes(&self) -> usize {
        self.host_weight_q4k.len() + self.host_bias.len() * 4
    }

    /// VRAM footprint when fully resident: Q4_K weight on device +
    /// dequantised fp32 weight + fp32 bias. The Q4_K bytes only need
    /// to live during the dequant kernel itself; a future
    /// dequant-direct-into-fp32 pipeline could drop them right after
    /// the kernel returns. Today they stay around for the duration
    /// of the resident period so a re-dequant doesn't need an extra
    /// host-to-device copy.
    pub fn dequant_size_bytes(&self) -> usize {
        self.host_weight_q4k.len() + self.in_dim * self.out_dim * 4 + self.host_bias.len() * 4
    }

    /// Allocate device buffers, upload Q4_K bytes, dispatch the
    /// dequant kernel, copy bias. No-op if already resident. The
    /// dequant kernel is dispatched *once per ensure_resident* — a
    /// follow-up `forward` call is just a `matvec_resident`, no
    /// additional dequant traffic.
    pub fn ensure_resident(
        &mut self,
        batch: &modgrad_device::backend::HipBatch,
    ) -> Result<(), super::backend::ResidencyError> {
        if self.is_resident() {
            return Ok(());
        }

        // Stage 1: Q4_K bytes onto device.
        let q4k_buf = modgrad_device::backend::HipBuffer::new(self.host_weight_q4k.len())?;
        // HipBuffer::copy_from_host expects an f32 slice; reinterpret
        // the byte slice. Q4_K block bytes (144) are a multiple of 4
        // and Vec<u8>'s allocator gives at least 4-byte alignment, so
        // this is safe.
        debug_assert!(self.host_weight_q4k.len() % 4 == 0,
            "Q4_K block bytes must be 4-byte multiple for f32-slice upload");
        let f32_view: &[f32] = unsafe {
            std::slice::from_raw_parts(
                self.host_weight_q4k.as_ptr() as *const f32,
                self.host_weight_q4k.len() / 4,
            )
        };
        q4k_buf.copy_from_host(f32_view)?;

        // Stage 2: allocate fp32 weight on device.
        let weight_fp32_buf = modgrad_device::backend::HipBuffer::new(
            self.in_dim * self.out_dim * 4,
        )?;

        // Stage 3: dispatch dequant kernel.
        let n_blocks = self.host_weight_q4k.len() / 144;
        unsafe {
            modgrad_device::backend::ops::dequant_q4k_resident(
                q4k_buf.device_ptr() as *const u8,
                weight_fp32_buf.device_ptr() as *mut f32,
                n_blocks,
            )?;
        }
        batch.note_dispatch()?;

        // Stage 4: bias on device.
        let bias_buf = modgrad_device::backend::HipBuffer::new(self.host_bias.len() * 4)?;
        bias_buf.copy_from_host(&self.host_bias)?;

        self.weight_q4k_dev = Some(q4k_buf);
        self.weight_dev = Some(weight_fp32_buf);
        self.bias_dev = Some(bias_buf);
        Ok(())
    }

    /// Drop every device buffer. Next `ensure_resident` (or `forward`)
    /// will re-allocate, re-upload, re-dequant from scratch. The
    /// host-side Q4_K bytes are unchanged so the redequantised fp32
    /// values are deterministically identical to the previous round.
    pub fn evict(&mut self) {
        self.weight_q4k_dev = None;
        self.weight_dev = None;
        self.bias_dev = None;
    }

    /// Streaming forward — `ensure_resident` then `matvec_resident`,
    /// then optionally `evict`. Same semantics as
    /// `LinearResidentStreaming::forward` but the resident weight
    /// comes from a Q4_K dequant pass instead of a host fp32 upload.
    ///
    /// **Requires a `&HipBatch`** — both the dequant kernel and the
    /// matvec count toward the batch's pending-dispatch tally, so the
    /// auto-sync at `DEFAULT_SYNC_EVERY = 256` keeps us bounded
    /// against the watchdog even when the streamer rotates many
    /// layers through VRAM.
    pub fn forward(
        &mut self,
        batch: &modgrad_device::backend::HipBatch,
        x_dev: &super::backend::GpuVec,
        out_dev: &mut super::backend::GpuVec,
        auto_evict: bool,
    ) -> Result<(), super::backend::ResidencyError> {
        use super::backend::{GpuVec, ResidencyError};
        debug_assert_eq!(x_dev.len(), self.in_dim);
        debug_assert_eq!(out_dev.len(), self.out_dim);

        // Stage 1: ensure both Q4_K bytes and dequantised fp32 are
        // on device.
        self.ensure_resident(batch)?;

        // Stage 2: unwrap GpuVec variants and dispatch matvec.
        let x_buf = match x_dev {
            GpuVec::Hip(b) => b,
            other => return Err(ResidencyError::WrongVariant {
                expected: "Hip", got: other.variant_name(),
            }),
        };
        let out_buf = match out_dev {
            GpuVec::Hip(b) => b,
            other => return Err(ResidencyError::WrongVariant {
                expected: "Hip", got: other.variant_name(),
            }),
        };
        let weight_dev = self.weight_dev.as_ref()
            .expect("ensure_resident postcondition: weight_dev is Some");
        let bias_dev = self.bias_dev.as_ref()
            .expect("ensure_resident postcondition: bias_dev is Some");

        unsafe {
            modgrad_device::backend::ops::matvec_resident(
                x_buf.device_ptr() as *const f32,
                weight_dev.device_ptr() as *const f32,
                bias_dev.device_ptr() as *const f32,
                out_buf.device_ptr() as *mut f32,
                self.out_dim,
                self.in_dim,
            )?;
        }
        batch.note_dispatch()?;

        if auto_evict {
            self.evict();
        }
        Ok(())
    }
}

/// Device-resident wrapper over a `SuperLinear`. Uploads the per-neuron
/// weight tensor (`[n_neurons × out_per × in_per]` flat, contiguous) and
/// bias tensor (`[n_neurons × out_per]`) to hipMalloc'd buffers once,
/// then `forward` issues `n_neurons` resident `matvec_resident` calls
/// with pointer offsets into the device buffers — zero PCIe transfers
/// per call.
///
/// Layout matches `SuperLinear`: each neuron's weight slice is
/// `weight_dev[n * out_per * in_per .. (n + 1) * out_per * in_per]`,
/// row-major within the neuron (one row per output, `in_per` cols).
/// The bias slice for neuron `n` is
/// `bias_dev[n * out_per .. (n + 1) * out_per]`.
///
/// Lifecycle mirrors `LinearResident`:
///   - `from_super_linear(&sl)?` — allocate + upload.
///   - `forward(&batch, x_dev, &mut out_dev)` — per-step dispatch.
///   - `sync_weights_from(&sl)?` — re-upload after an optimizer step.
///
/// Production `n_neurons` is ≤512, so the per-neuron loop dispatching
/// against the `HipBatch::DEFAULT_SYNC_EVERY = 256` cadence stays
/// well-behaved without coalescing the matvecs into a fused kernel.
///
/// Only available with `--features rocm`.
#[cfg(feature = "rocm")]
pub struct SuperLinearResident {
    pub weight_dev: modgrad_device::backend::HipBuffer,
    pub bias_dev: modgrad_device::backend::HipBuffer,
    pub n_neurons: usize,
    pub in_per: usize,
    pub out_per: usize,
}

#[cfg(feature = "rocm")]
impl SuperLinearResident {
    /// Allocate device buffers and upload weight + bias from a host-side
    /// `SuperLinear`. Returns `ResidencyError::Backend(_)` on hipMalloc /
    /// hipMemcpy failure; match the inner `BackendError` variant for
    /// typed recovery.
    pub fn from_super_linear(sl: &SuperLinear) -> Result<Self, super::backend::ResidencyError> {
        let weight_dev = modgrad_device::backend::HipBuffer::new(sl.weights.len() * 4)?;
        weight_dev.copy_from_host(&sl.weights)?;
        let bias_dev = modgrad_device::backend::HipBuffer::new(sl.biases.len() * 4)?;
        bias_dev.copy_from_host(&sl.biases)?;
        Ok(Self {
            weight_dev, bias_dev,
            n_neurons: sl.n_neurons,
            in_per: sl.in_per,
            out_per: sl.out_per,
        })
    }

    /// Re-upload weights + biases after an in-place optimizer step.
    pub fn sync_weights_from(&mut self, sl: &SuperLinear) -> Result<(), super::backend::ResidencyError> {
        debug_assert_eq!(sl.n_neurons, self.n_neurons);
        debug_assert_eq!(sl.in_per, self.in_per);
        debug_assert_eq!(sl.out_per, self.out_per);
        self.weight_dev.copy_from_host(&sl.weights)?;
        self.bias_dev.copy_from_host(&sl.biases)?;
        Ok(())
    }

    /// Resident forward: `x_dev` and `out_dev` are `GpuVec::Hip`. Issues
    /// one `matvec_resident` per neuron, with each call's pointers
    /// offset into the device buffers so the per-neuron weight matrix,
    /// bias vector, input slice, and output slice are addressed without
    /// any host staging.
    ///
    /// Input shape: `[n_neurons * in_per]` (per-neuron trace concatenated).
    /// Output shape: `[n_neurons * out_per]`.
    ///
    /// **Requires a `&HipBatch`.** See `LinearResident::forward` for
    /// rationale — the batch turns the queue-sync requirement into a
    /// compile-time obligation. Each per-neuron dispatch counts toward
    /// the batch's pending dispatch tally; the auto-sync at
    /// `DEFAULT_SYNC_EVERY = 256` keeps us bounded against the watchdog
    /// even for the larger production neuron counts.
    pub fn forward(
        &self,
        batch: &modgrad_device::backend::HipBatch,
        x_dev: &super::backend::GpuVec,
        out_dev: &mut super::backend::GpuVec,
    ) -> Result<(), super::backend::ResidencyError> {
        use super::backend::{GpuVec, ResidencyError};
        debug_assert_eq!(x_dev.len(), self.n_neurons * self.in_per);
        debug_assert_eq!(out_dev.len(), self.n_neurons * self.out_per);
        let x_buf = match x_dev {
            GpuVec::Hip(b) => b,
            other => return Err(ResidencyError::WrongVariant {
                expected: "Hip", got: other.variant_name(),
            }),
        };
        let out_buf = match out_dev {
            GpuVec::Hip(b) => b,
            other => return Err(ResidencyError::WrongVariant {
                expected: "Hip", got: other.variant_name(),
            }),
        };
        let x_base = x_buf.device_ptr() as *const f32;
        let out_base = out_buf.device_ptr() as *mut f32;
        let w_base = self.weight_dev.device_ptr() as *const f32;
        let b_base = self.bias_dev.device_ptr() as *const f32;
        let in_per = self.in_per;
        let out_per = self.out_per;
        let weight_stride = out_per * in_per;
        for n in 0..self.n_neurons {
            // Per-neuron pointer offsets — same layout the host
            // `SuperLinear::forward_cpu` uses, just expressed as
            // pointer arithmetic against the resident buffers.
            unsafe {
                modgrad_device::backend::ops::matvec_resident(
                    x_base.add(n * in_per),
                    w_base.add(n * weight_stride),
                    b_base.add(n * out_per),
                    out_base.add(n * out_per),
                    out_per,
                    in_per,
                )?;
            }
            batch.note_dispatch()?;
        }
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
        let mut x_dev = crate::backend::GpuVec::try_hip(64).expect("alloc x");
        x_dev.copy_from(&host_x);
        let mut out_dev = crate::backend::GpuVec::try_hip(128).expect("alloc out");

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
        let mut x_dev = crate::backend::GpuVec::try_hip(64).expect("x");
        x_dev.copy_from(&host_x);

        // Run 32 forwards in a row; output of every call should be
        // bit-identical (deterministic GPU execution, no PRNG).
        // The HipBatch ensures the queue stays bounded and a final
        // sync runs on Drop.
        let batch = modgrad_device::backend::HipBatch::new();
        let mut first_y: Option<Vec<f32>> = None;
        for _ in 0..32 {
            let mut out_dev = crate::backend::GpuVec::try_hip(64).expect("out");
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

    /// SuperLinearResident matches the CPU `SuperLinear::forward` output
    /// within FP tolerance on a small shape (8 neurons × 4 in × 3 out).
    /// Proves the per-neuron pointer-offset dispatch produces the same
    /// arithmetic as the host `dot`-based path. Tolerance is loose
    /// (1e-3) because rocBLAS uses a different accumulation order than
    /// the AVX-512 `dot` reduction.
    #[test]
    fn super_linear_resident_matches_host() {
        if !runtime_available() {
            eprintln!("hip runtime unavailable, skipping");
            return;
        }
        let n_neurons = 8;
        let in_per = 4;
        let out_per = 3;
        let mut sl = SuperLinear::new(n_neurons, in_per, out_per);
        // Randomize bias (the constructor zeros it) so the bias-add
        // in the resident path is exercised, not just the matmul.
        let mut rng = SimpleRng::new(0xABCD);
        for b in sl.biases.iter_mut() {
            *b = rng.next_normal() * 0.1;
        }
        let host_x: Vec<f32> = (0..n_neurons * in_per)
            .map(|_| rng.next_normal())
            .collect();

        // Host reference: route through `SuperLinear::forward`.
        let host_y = sl.forward(&host_x);

        // Device-resident: upload weights once, dispatch once.
        let resident = SuperLinearResident::from_super_linear(&sl)
            .expect("SuperLinearResident::from_super_linear");
        let mut x_dev = crate::backend::GpuVec::try_hip(n_neurons * in_per)
            .expect("alloc x");
        x_dev.copy_from(&host_x);
        let mut out_dev = crate::backend::GpuVec::try_hip(n_neurons * out_per)
            .expect("alloc out");

        let batch = modgrad_device::backend::HipBatch::new();
        resident.forward(&batch, &x_dev, &mut out_dev).expect("resident forward");
        batch.flush().expect("flush");

        let mut device_y = vec![0.0f32; n_neurons * out_per];
        out_dev.copy_to_host(&mut device_y);

        let max_diff = host_y.iter().zip(&device_y)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-3,
            "host vs resident mismatch: max |Δ| = {max_diff}");
    }

    /// `LinearResidentStreaming::forward` produces the same output as
    /// the one-shot `LinearResident::forward` on the same weights and
    /// the same input. This validates the on-demand upload path: the
    /// streaming wrapper must allocate, upload, and dispatch in a way
    /// that is bitwise-equivalent to the persistent path — any
    /// divergence here would mean the upload is corrupting weights or
    /// the dispatch is reading from the wrong pointer.
    #[test]
    fn streaming_basic() {
        if !runtime_available() {
            eprintln!("hip runtime unavailable, skipping");
            return;
        }
        let lin = Linear::new(64, 96);  // out_dim=96, in_dim=64
        let mut rng = SimpleRng::new(0xBABE);
        let host_x: Vec<f32> = (0..64).map(|_| rng.next_normal()).collect();

        // Reference: existing one-shot LinearResident.
        let resident = LinearResident::from_linear(&lin)
            .expect("LinearResident::from_linear");
        let mut x_dev = crate::backend::GpuVec::try_hip(64).expect("alloc x");
        x_dev.copy_from(&host_x);
        let mut ref_out_dev = crate::backend::GpuVec::try_hip(96).expect("alloc ref out");
        let batch = modgrad_device::backend::HipBatch::new();
        resident.forward(&batch, &x_dev, &mut ref_out_dev).expect("ref forward");
        batch.flush().expect("ref flush");
        let mut ref_y = vec![0.0f32; 96];
        ref_out_dev.copy_to_host(&mut ref_y);

        // Streaming: from_linear_arc → ensure_resident → forward.
        let host_arc = std::sync::Arc::new(lin);
        let mut streaming = LinearResidentStreaming::from_linear_arc(host_arc);
        assert!(!streaming.is_resident(),
            "streaming wrapper must not allocate at construction");
        streaming.ensure_resident().expect("ensure_resident");
        assert!(streaming.is_resident(),
            "ensure_resident must leave both device buffers populated");

        let mut stream_out_dev = crate::backend::GpuVec::try_hip(96)
            .expect("alloc stream out");
        let stream_batch = modgrad_device::backend::HipBatch::new();
        streaming.forward(&stream_batch, &x_dev, &mut stream_out_dev, false)
            .expect("streaming forward");
        stream_batch.flush().expect("stream flush");
        let mut stream_y = vec![0.0f32; 96];
        stream_out_dev.copy_to_host(&mut stream_y);

        // Bit-exact: same weights → same hipBLAS path → same output.
        // We allow a tiny FP slack (1e-6) because nothing in the
        // streaming path changes accumulation order, but a strict
        // equality assertion would be brittle to future hipBLAS
        // updates.
        let max_diff = ref_y.iter().zip(&stream_y)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-6,
            "LinearResident vs LinearResidentStreaming mismatch: max |Δ| = {max_diff}");
    }

    /// `auto_evict = true` must drop the device buffers right after
    /// the dispatch, and a subsequent `forward` must transparently
    /// re-upload and produce the same output. This is the
    /// foundation-model layer-rotation pattern: weights enter VRAM,
    /// run, leave VRAM, then come back later.
    #[test]
    fn streaming_evict_then_reupload() {
        if !runtime_available() {
            eprintln!("hip runtime unavailable, skipping");
            return;
        }
        let lin = Linear::new(48, 32);  // out_dim=32, in_dim=48
        let mut rng = SimpleRng::new(0xC0FFEE);
        let host_x: Vec<f32> = (0..48).map(|_| rng.next_normal()).collect();

        let host_arc = std::sync::Arc::new(lin);
        let mut streaming = LinearResidentStreaming::from_linear_arc(host_arc);

        let mut x_dev = crate::backend::GpuVec::try_hip(48).expect("alloc x");
        x_dev.copy_from(&host_x);

        // First forward with auto_evict=true: should upload, dispatch,
        // then immediately drop the device buffers.
        let mut out_a_dev = crate::backend::GpuVec::try_hip(32).expect("alloc out a");
        let batch_a = modgrad_device::backend::HipBatch::new();
        streaming.forward(&batch_a, &x_dev, &mut out_a_dev, true)
            .expect("first forward");
        batch_a.flush().expect("flush a");
        assert!(!streaming.is_resident(),
            "auto_evict=true must drop device buffers post-dispatch");
        let mut y_a = vec![0.0f32; 32];
        out_a_dev.copy_to_host(&mut y_a);

        // Second forward: must re-upload from host weights and produce
        // the same output. We're testing the round-trip through the
        // ensure_resident gate.
        let mut out_b_dev = crate::backend::GpuVec::try_hip(32).expect("alloc out b");
        let batch_b = modgrad_device::backend::HipBatch::new();
        streaming.forward(&batch_b, &x_dev, &mut out_b_dev, false)
            .expect("second forward");
        batch_b.flush().expect("flush b");
        assert!(streaming.is_resident(),
            "auto_evict=false must leave device buffers populated");
        let mut y_b = vec![0.0f32; 32];
        out_b_dev.copy_to_host(&mut y_b);

        // No PRNG, deterministic dispatch — outputs must agree exactly.
        assert_eq!(y_a, y_b,
            "evict+reupload changed output: bug in upload path");
    }

    /// After the caller mutates the host weights and calls `evict()`,
    /// the next `forward` must reflect the new weights. This is the
    /// AdamW-step contract: the optimizer mutates the shared
    /// `Arc<Linear>`, and the streaming wrapper picks up the change on
    /// the next upload. We use `Arc::get_mut` to do the mutation —
    /// requires no other strong references to be live, which is the
    /// realistic optimizer-driven pattern.
    #[test]
    fn streaming_after_host_mutation() {
        if !runtime_available() {
            eprintln!("hip runtime unavailable, skipping");
            return;
        }
        let lin = Linear::new(32, 16);  // small; mutation is cheap
        let mut rng = SimpleRng::new(0xDEADBEEF);
        let host_x: Vec<f32> = (0..32).map(|_| rng.next_normal()).collect();

        let host_arc = std::sync::Arc::new(lin);
        let mut streaming = LinearResidentStreaming::from_linear_arc(host_arc);

        let mut x_dev = crate::backend::GpuVec::try_hip(32).expect("alloc x");
        x_dev.copy_from(&host_x);

        // Forward with the original weights; record output.
        let mut out_orig_dev = crate::backend::GpuVec::try_hip(16)
            .expect("alloc out orig");
        let batch_orig = modgrad_device::backend::HipBatch::new();
        streaming.forward(&batch_orig, &x_dev, &mut out_orig_dev, false)
            .expect("forward orig");
        batch_orig.flush().expect("flush orig");
        let mut y_orig = vec![0.0f32; 16];
        out_orig_dev.copy_to_host(&mut y_orig);

        // Mutate the host weights — simulate an optimizer step that
        // overwrites every weight with a known different value.
        // `Arc::get_mut` succeeds because the streaming wrapper holds
        // the only strong reference at this point.
        {
            let host_mut = std::sync::Arc::get_mut(streaming.host_arc_mut())
                .expect("Arc::get_mut: only strong ref must be the streamer");
            // Set every weight to 0.5 and every bias to 1.0 — easy to
            // verify the new output reflects these values.
            for w in host_mut.weight.iter_mut() { *w = 0.5; }
            for b in host_mut.bias.iter_mut() { *b = 1.0; }
        }

        // Caller-side contract: after host mutation, evict so the next
        // forward re-uploads. (If we skipped this, the cached device
        // buffers would still hold the old weights — that's the
        // documented hazard.)
        streaming.evict();
        assert!(!streaming.is_resident(),
            "evict must drop device buffers");

        // Forward with mutated weights.
        let mut out_new_dev = crate::backend::GpuVec::try_hip(16)
            .expect("alloc out new");
        let batch_new = modgrad_device::backend::HipBatch::new();
        streaming.forward(&batch_new, &x_dev, &mut out_new_dev, false)
            .expect("forward new");
        batch_new.flush().expect("flush new");
        let mut y_new = vec![0.0f32; 16];
        out_new_dev.copy_to_host(&mut y_new);

        // Output must differ from y_orig. With weights=0.5 and bias=1.0,
        // y[i] = 1.0 + 0.5 * sum(host_x); compute that explicitly and
        // compare against the device output for a sharper check than
        // just inequality vs. y_orig.
        let expected_each = 1.0 + 0.5 * host_x.iter().sum::<f32>();
        let max_diff = y_new.iter()
            .map(|v| (v - expected_each).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-3,
            "post-mutation output not reflecting new weights: \
             expected each ≈ {expected_each}, got {y_new:?}");
        assert_ne!(y_orig, y_new,
            "post-mutation output identical to original — re-upload didn't happen");
    }

    /// Calling SuperLinearResident.forward many times with the same
    /// input must produce bit-identical output every iteration. No
    /// PRNG is in play so any drift would indicate a real bug
    /// (uninitialized accumulator, lingering state, queue overflow
    /// silently corrupting output, etc.).
    #[test]
    fn super_linear_resident_loop_no_drift() {
        if !runtime_available() {
            eprintln!("hip runtime unavailable, skipping");
            return;
        }
        let n_neurons = 8;
        let in_per = 4;
        let out_per = 3;
        let sl = SuperLinear::new(n_neurons, in_per, out_per);
        let mut rng = SimpleRng::new(0xFEED);
        let host_x: Vec<f32> = (0..n_neurons * in_per)
            .map(|_| rng.next_normal())
            .collect();

        let resident = SuperLinearResident::from_super_linear(&sl)
            .expect("resident");
        let mut x_dev = crate::backend::GpuVec::try_hip(n_neurons * in_per)
            .expect("x");
        x_dev.copy_from(&host_x);

        // 16 forwards on the same input; `n_neurons * 16 = 128`
        // dispatches stays under the 256 auto-sync threshold so we
        // verify both the in-batch and end-of-batch paths.
        let batch = modgrad_device::backend::HipBatch::new();
        let mut first_y: Option<Vec<f32>> = None;
        for _ in 0..16 {
            let mut out_dev = crate::backend::GpuVec::try_hip(n_neurons * out_per)
                .expect("out");
            resident.forward(&batch, &x_dev, &mut out_dev).expect("forward");
            // copy_to_host implicitly synchronises; safe mid-batch.
            let mut host_y = vec![0.0f32; n_neurons * out_per];
            out_dev.copy_to_host(&mut host_y);
            match &first_y {
                None => first_y = Some(host_y),
                Some(y0) => assert_eq!(*y0, host_y, "drift across calls"),
            }
        }
    }

    // ─── LinearResidentQuantized (Q4_K) ─────────────────────────────

    mod quantized {
        use super::*;

        /// Probe whether the hipcc-compiled Q4_K dequant kernel is
        /// linked into this build. Same gate the kernel test uses; lets
        /// the suite skip cleanly on a host without hipcc rather than
        /// fail with `Unsupported`.
        fn dequant_kernel_built() -> bool {
            use modgrad_device::backend::{Backend, Op, RocmBackend};
            let Some(be) = RocmBackend::try_new() else { return false; };
            let probe = Op::DequantQ4KResident {
                q4k_dev: std::ptr::null(),
                fp32_dev: std::ptr::null_mut(),
                n_blocks: 1,
            };
            be.supports(&probe)
        }

        /// `LinearResidentQuantized::forward` matches `LinearResident::forward`
        /// (uncompressed reference) within Q4_K's 1% error band.
        /// Validates the full pipeline:
        ///   - Host quantise → Q4_K bytes
        ///   - Upload → device dequant → fp32 weight
        ///   - Matvec dispatch via the fp32 weight
        /// produces output close to the lossless fp32 path.
        #[test]
        fn basic() {
            if !runtime_available() {
                eprintln!("hip runtime unavailable, skipping");
                return;
            }
            if !dequant_kernel_built() {
                eprintln!("dequant_q4k kernel not built, skipping");
                return;
            }
            // 1024 in × 256 out — clean multiple of 256 so quantize
            // accepts it without padding.
            let in_dim = 1024;
            let out_dim = 256;
            let lin = Linear::new(in_dim, out_dim);
            let mut rng = SimpleRng::new(0xCAFEBABE);
            let host_x: Vec<f32> = (0..in_dim).map(|_| rng.next_normal()).collect();

            // Reference: uncompressed fp32 LinearResident.
            let ref_resident = LinearResident::from_linear(&lin)
                .expect("LinearResident::from_linear");
            let mut x_dev = crate::backend::GpuVec::try_hip(in_dim).expect("alloc x");
            x_dev.copy_from(&host_x);
            let mut ref_out = crate::backend::GpuVec::try_hip(out_dim).expect("alloc ref out");
            let ref_batch = modgrad_device::backend::HipBatch::new();
            ref_resident.forward(&ref_batch, &x_dev, &mut ref_out).expect("ref forward");
            ref_batch.flush().expect("ref flush");
            let mut ref_y = vec![0.0f32; out_dim];
            ref_out.copy_to_host(&mut ref_y);

            // Quantised path.
            let mut quant = LinearResidentQuantized::from_linear(&lin)
                .expect("LinearResidentQuantized::from_linear");
            assert!(!quant.is_resident(),
                "from_linear must not allocate device buffers");
            let mut q_out = crate::backend::GpuVec::try_hip(out_dim).expect("alloc q out");
            let q_batch = modgrad_device::backend::HipBatch::new();
            quant.forward(&q_batch, &x_dev, &mut q_out, false).expect("quant forward");
            q_batch.flush().expect("q flush");
            let mut q_y = vec![0.0f32; out_dim];
            q_out.copy_to_host(&mut q_y);

            // RMS relative error — Q4_K's 1% band, with headroom for
            // the simpler reference (non-iterative) quantiser we ship.
            let signal_sq: f32 = ref_y.iter().map(|v| v * v).sum();
            let err_sq: f32 = ref_y.iter().zip(&q_y)
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            let rms_rel = (err_sq / signal_sq.max(1e-12)).sqrt();
            eprintln!("quantized_basic: rms_rel = {rms_rel}");
            assert!(rms_rel < 0.10,
                "Q4_K vs fp32 RMS rel = {rms_rel} exceeded 10% — quant pipeline regression");
        }

        /// `auto_evict = true` must drop device buffers right after
        /// the dispatch; the next `forward` re-dequantises from
        /// scratch and produces bit-identical output (host-side Q4_K
        /// bytes don't change, dequant kernel is a pure function of
        /// those bytes).
        #[test]
        fn evict_then_redequant() {
            if !runtime_available() {
                eprintln!("hip runtime unavailable, skipping");
                return;
            }
            if !dequant_kernel_built() {
                eprintln!("dequant_q4k kernel not built, skipping");
                return;
            }
            let in_dim = 512;
            let out_dim = 128;
            let lin = Linear::new(in_dim, out_dim);
            let mut rng = SimpleRng::new(0xC0FFEEFE);
            let host_x: Vec<f32> = (0..in_dim).map(|_| rng.next_normal()).collect();

            let mut quant = LinearResidentQuantized::from_linear(&lin)
                .expect("LinearResidentQuantized::from_linear");
            let mut x_dev = crate::backend::GpuVec::try_hip(in_dim).expect("alloc x");
            x_dev.copy_from(&host_x);

            // First forward, auto_evict=true: dequant, dispatch,
            // drop device buffers.
            let mut out_a = crate::backend::GpuVec::try_hip(out_dim).expect("alloc a");
            let batch_a = modgrad_device::backend::HipBatch::new();
            quant.forward(&batch_a, &x_dev, &mut out_a, true)
                .expect("first forward");
            batch_a.flush().expect("flush a");
            assert!(!quant.is_resident(),
                "auto_evict=true must drop device buffers after dispatch");
            let mut y_a = vec![0.0f32; out_dim];
            out_a.copy_to_host(&mut y_a);

            // Second forward: re-allocate, re-upload, re-dequant
            // from scratch and produce bit-identical output. The
            // dequant kernel is deterministic on identical inputs;
            // matvec dispatch is too; round trip must be exact.
            let mut out_b = crate::backend::GpuVec::try_hip(out_dim).expect("alloc b");
            let batch_b = modgrad_device::backend::HipBatch::new();
            quant.forward(&batch_b, &x_dev, &mut out_b, false)
                .expect("second forward");
            batch_b.flush().expect("flush b");
            assert!(quant.is_resident(),
                "auto_evict=false must leave device buffers populated");
            let mut y_b = vec![0.0f32; out_dim];
            out_b.copy_to_host(&mut y_b);

            // Strict equality: same bytes, same kernels, same input
            // ⇒ same output. Any drift is a real bug.
            assert_eq!(y_a, y_b,
                "evict + re-dequant produced different output — dequant kernel non-deterministic");
        }

        /// `host_size_bytes` reports the actual on-disk / in-RAM
        /// cost of a quantised Linear. The 4-bit-per-weight encoding
        /// (plus 144/256 byte block overhead) lands at ~12.5% of
        /// fp32 — the 8× compression that makes 30B-param models
        /// fit on consumer GPUs feasible.
        #[test]
        fn host_size_savings() {
            let in_dim = 1024;
            let out_dim = 1024;
            let lin = Linear::new(in_dim, out_dim);
            let fp32_bytes = lin.weight.len() * 4;

            // No GPU runtime needed — size calculation is host-side.
            let quant = LinearResidentQuantized::from_linear(&lin)
                .expect("LinearResidentQuantized::from_linear");
            let q_bytes = quant.host_size_bytes();
            let ratio = q_bytes as f32 / fp32_bytes as f32;
            eprintln!(
                "quantized_host_size_savings: fp32={fp32_bytes} bytes, \
                 q4k={q_bytes} bytes, ratio={ratio:.4} ({:.1}× compression)",
                fp32_bytes as f32 / q_bytes as f32,
            );
            // Spec gate: `host_size_bytes < lin.weight.len() * 4 / 4`
            // (≤ 25% of fp32). We tighten to 16% to catch regressions
            // in block-size overhead — the canonical Q4_K_M ratio is
            // 144 / (256 * 4) = 14.06%.
            assert!(q_bytes < fp32_bytes / 4,
                "Q4_K host bytes ({q_bytes}) >= 1/4 of fp32 ({fp32_bytes}) — \
                 quant pipeline lost compression");
            assert!(ratio < 0.16,
                "Q4_K compression ratio {ratio} > 16% of fp32 — \
                 expected ~14% (≈ 7× compression)");
        }
    }

    // ─── LinearResidentBf16 (mixed-precision) ──────────────────────

    mod bf16 {
        use super::*;
        use modgrad_device::backend::op::{bf16_to_f32, f32_to_bf16};

        /// `LinearResidentBf16::forward` matches the fp32 reference
        /// produced by `LinearResident::forward` within bf16's 1%
        /// relative tolerance band. Proves the full mixed-precision
        /// path: host fp32 → bf16 quantise → upload → bf16 GemmEx
        /// dispatch → bf16 download → fp32 dequantise.
        #[test]
        fn linear_resident_bf16_matches_fp32() {
            if !runtime_available() {
                eprintln!("hip runtime unavailable, skipping");
                return;
            }
            // Even dims for the bf16 upload helper (it pads odd
            // lengths but the matched-shape path is the common case).
            let in_dim = 64;
            let out_dim = 128;
            let lin = Linear::new(in_dim, out_dim);
            let mut rng = SimpleRng::new(0xBF16);
            let host_x: Vec<f32> = (0..in_dim).map(|_| rng.next_normal() * 0.5).collect();

            // fp32 reference, but operands are routed through bf16
            // round-trip so the comparison reflects the same
            // quantisation noise the device path saw.
            let lin_q = {
                let mut q = lin.clone();
                for w in q.weight.iter_mut() {
                    *w = bf16_to_f32(f32_to_bf16(*w));
                }
                for b in q.bias.iter_mut() {
                    *b = bf16_to_f32(f32_to_bf16(*b));
                }
                q
            };
            let host_xq: Vec<f32> = host_x.iter()
                .map(|&v| bf16_to_f32(f32_to_bf16(v))).collect();
            let mut ref_y = vec![0.0f32; out_dim];
            lin_q.forward_into(&host_xq, &mut ref_y);

            // Device-resident bf16 path.
            let resident = LinearResidentBf16::from_linear(&lin)
                .expect("LinearResidentBf16::from_linear");
            let mut x_dev = crate::backend::GpuVec::try_hip_bf16(in_dim)
                .expect("alloc x bf16");
            // Upload x as bf16 — copy raw bytes via the HipBuffer's
            // f32-typed copy_from_host, since the buffer was sized for
            // bf16 (in_dim * 2 bytes).
            let xq: Vec<u16> = host_x.iter().map(|&v| f32_to_bf16(v)).collect();
            if let crate::backend::GpuVec::Hip(buf) = &mut x_dev {
                let view = unsafe {
                    std::slice::from_raw_parts(xq.as_ptr() as *const f32, xq.len() / 2)
                };
                buf.copy_from_host(view).expect("upload x bf16");
            } else {
                panic!("expected Hip variant from try_hip_bf16");
            }
            let mut out_dev = crate::backend::GpuVec::try_hip_bf16(out_dim)
                .expect("alloc out bf16");

            let batch = modgrad_device::backend::HipBatch::new();
            resident.forward(&batch, &x_dev, &mut out_dev)
                .expect("bf16 forward");
            batch.flush().expect("flush");

            // Download as bf16 bytes, dequantise to fp32.
            let mut out_u16 = vec![0u16; out_dim];
            if let crate::backend::GpuVec::Hip(buf) = &out_dev {
                let view = unsafe {
                    std::slice::from_raw_parts_mut(
                        out_u16.as_mut_ptr() as *mut f32,
                        out_u16.len() / 2,
                    )
                };
                buf.copy_to_host(view).expect("download out bf16");
            } else {
                panic!("expected Hip variant");
            }
            let device_y: Vec<f32> = out_u16.iter().map(|&v| bf16_to_f32(v)).collect();

            // ~1% relative tolerance — bf16 has ~3 decimal digits of
            // precision and we accumulate `in_dim` terms in fp32.
            let mut max_rel = 0.0f32;
            for (a, b) in ref_y.iter().zip(&device_y) {
                let scale = a.abs().max(b.abs()).max(1e-3);
                let rel = (a - b).abs() / scale;
                if rel > max_rel { max_rel = rel; }
            }
            assert!(max_rel < 1e-2,
                "fp32 vs bf16 max rel = {max_rel}, expected < 1%");
        }

        /// `sync_from_master` round-trips correctly: an in-place edit
        /// to `host_master_weight` propagates to device on the next
        /// dispatch.
        #[test]
        fn linear_resident_bf16_sync_from_master() {
            if !runtime_available() {
                eprintln!("hip runtime unavailable, skipping");
                return;
            }
            let in_dim = 32;
            let out_dim = 16;
            let lin = Linear::new(in_dim, out_dim);
            let mut rng = SimpleRng::new(0xBADF00D);
            let host_x: Vec<f32> = (0..in_dim).map(|_| rng.next_normal() * 0.5).collect();

            let mut resident = LinearResidentBf16::from_linear(&lin)
                .expect("LinearResidentBf16::from_linear");

            // Mutate the master weights (simulate AdamW result), call
            // sync_from_master, dispatch, and assert the new weights
            // produced a different output than the original.
            let mut x_dev = crate::backend::GpuVec::try_hip_bf16(in_dim)
                .expect("alloc x");
            let xq: Vec<u16> = host_x.iter().map(|&v| f32_to_bf16(v)).collect();
            if let crate::backend::GpuVec::Hip(buf) = &mut x_dev {
                let view = unsafe {
                    std::slice::from_raw_parts(xq.as_ptr() as *const f32, xq.len() / 2)
                };
                buf.copy_from_host(view).expect("upload x");
            }
            let mut out_dev = crate::backend::GpuVec::try_hip_bf16(out_dim)
                .expect("alloc out");

            let batch = modgrad_device::backend::HipBatch::new();
            resident.forward(&batch, &x_dev, &mut out_dev).expect("forward 1");
            batch.flush().expect("flush 1");
            let mut out_u16_a = vec![0u16; out_dim];
            if let crate::backend::GpuVec::Hip(buf) = &out_dev {
                let view = unsafe {
                    std::slice::from_raw_parts_mut(
                        out_u16_a.as_mut_ptr() as *mut f32,
                        out_u16_a.len() / 2,
                    )
                };
                buf.copy_to_host(view).expect("download a");
            }

            // Mutate every weight + bias and re-sync.
            for w in resident.host_master_weight.iter_mut() { *w *= 2.0; }
            for b in resident.host_master_bias.iter_mut() { *b += 1.0; }
            resident.sync_from_master().expect("sync_from_master");

            let batch2 = modgrad_device::backend::HipBatch::new();
            resident.forward(&batch2, &x_dev, &mut out_dev).expect("forward 2");
            batch2.flush().expect("flush 2");
            let mut out_u16_b = vec![0u16; out_dim];
            if let crate::backend::GpuVec::Hip(buf) = &out_dev {
                let view = unsafe {
                    std::slice::from_raw_parts_mut(
                        out_u16_b.as_mut_ptr() as *mut f32,
                        out_u16_b.len() / 2,
                    )
                };
                buf.copy_to_host(view).expect("download b");
            }

            assert_ne!(out_u16_a, out_u16_b,
                "sync_from_master did not propagate weight mutation");
        }
    }
}

