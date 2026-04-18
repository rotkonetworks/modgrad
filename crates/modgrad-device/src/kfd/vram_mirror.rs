//! VRAM-resident training state for an FFN (or any model with an AdamW optimizer).
//!
//! The conventional `try_adamw` path uploads weights/grads/m/v to staging
//! VRAM, runs the kernel, and downloads the results — ~5 GB of PCIe/BAR
//! traffic per step for a 189 M parameter model. `VramMirror` holds all
//! four tensors as permanent VRAM allocations; `adamw_step` dispatches the
//! existing `adamw` kernel on those VA pointers directly, with zero
//! copies inside the training loop.
//!
//! # Ownership model
//! * Caller calls `new` with a list of tensor sizes (one entry per weight
//!   group — e.g. one per `Linear.weight`, one per `Linear.bias`, one per
//!   layer-norm gamma/beta). Mirror allocates 4 VRAM buffers per entry:
//!   weight, grad, m (AdamW first moment), v (AdamW second moment).
//! * Caller uploads initial weights via `upload_weight` at training start.
//!   `m` and `v` are zero-initialised by the allocator.
//! * Each training step:
//!   1. Caller computes gradients (CPU or GPU) and writes them into the
//!      mirror's grad buffer via `grad_as_mut_slice` or a zero-copy kernel
//!      that targets the VA.
//!   2. Caller invokes `adamw_step(...)`. Weights, m, v are updated
//!      in-place in VRAM; nothing crosses PCIe.
//! * At checkpoint boundaries: `download_weight` / `download_moments` pull
//!   a tensor back to CPU for bincode serialisation.
//!
//! # Safety / invariants
//! * All four `Vec<GpuBuffer>` vectors have the same length as `sizes`.
//! * All buffers are PUBLIC VRAM (resizable BAR), so `as_mut_slice` on any
//!   mirror tensor is safe CPU access through the BAR mapping.
//! * The mirror outlives any kernel dispatch it participates in — the GPU
//!   holds raw VAs, not lifetimes, so dropping the mirror while a kernel
//!   references its buffers would crash the device. Callers should keep
//!   the mirror alive for the full training run.

use super::HsaDevice;
use super::memory::GpuBuffer;

/// VRAM-resident training state.
pub struct VramMirror {
    /// Per-tensor permanent VRAM for weights. Length = `sizes.len()`.
    pub weights: Vec<GpuBuffer>,
    /// Per-tensor VRAM for gradients. Zeroed at start; written each step.
    pub grads: Vec<GpuBuffer>,
    /// AdamW first moments. Zero-initialised.
    pub moments_m: Vec<GpuBuffer>,
    /// AdamW second moments. Zero-initialised.
    pub moments_v: Vec<GpuBuffer>,
    /// Element count for each tensor (not byte count).
    pub sizes: Vec<usize>,
}

impl VramMirror {
    /// Allocate mirror storage for a set of tensors with the given element counts.
    /// Each tensor gets four VRAM buffers (weight/grad/m/v). Returns `None` if
    /// any allocation fails (caller should fall back to CPU AdamW).
    ///
    /// Every buffer is rounded up to the nearest 4 KiB page — the KFD allocator
    /// requires page-aligned sizes.
    pub fn new(dev: &HsaDevice, sizes: Vec<usize>) -> Option<Self> {
        let n_tensors = sizes.len();
        let mut weights = Vec::with_capacity(n_tensors);
        let mut grads = Vec::with_capacity(n_tensors);
        let mut moments_m = Vec::with_capacity(n_tensors);
        let mut moments_v = Vec::with_capacity(n_tensors);

        for &n in &sizes {
            let bytes = (n * 4) as u64;
            let cap = ((bytes + 4095) & !4095).max(4096);
            let w  = dev.alloc.alloc_vram(cap).ok()?;
            let g  = dev.alloc.alloc_vram(cap).ok()?;
            let mb = dev.alloc.alloc_vram(cap).ok()?;
            let vb = dev.alloc.alloc_vram(cap).ok()?;

            // Zero-init grads / m / v so the first AdamW step reads meaningful
            // values. Weights are written by the caller via `upload_weight`.
            // BAR writes are cheap and only run once per training start.
            let zeros = vec![0.0f32; n];
            g.write_f32(0, &zeros);
            mb.write_f32(0, &zeros);
            vb.write_f32(0, &zeros);

            weights.push(w);
            grads.push(g);
            moments_m.push(mb);
            moments_v.push(vb);
        }

        Some(Self { weights, grads, moments_m, moments_v, sizes })
    }

    /// Total VRAM consumed by this mirror (4× tensor bytes, page-aligned).
    pub fn vram_bytes(&self) -> u64 {
        self.weights.iter().chain(self.grads.iter())
            .chain(self.moments_m.iter()).chain(self.moments_v.iter())
            .map(|b| b.size).sum()
    }

    /// Upload `data` to the weights[idx] buffer. Typically called once at start.
    ///
    /// Returns false on length mismatch.
    pub fn upload_weight(&self, idx: usize, data: &[f32]) -> bool {
        if idx >= self.sizes.len() || data.len() != self.sizes[idx] { return false; }
        self.weights[idx].write_f32(0, data);
        true
    }

    /// Upload `data` to moments_m[idx]. Used when resuming from a checkpoint.
    pub fn upload_m(&self, idx: usize, data: &[f32]) -> bool {
        if idx >= self.sizes.len() || data.len() != self.sizes[idx] { return false; }
        self.moments_m[idx].write_f32(0, data);
        true
    }

    /// Upload `data` to moments_v[idx]. Used when resuming from a checkpoint.
    pub fn upload_v(&self, idx: usize, data: &[f32]) -> bool {
        if idx >= self.sizes.len() || data.len() != self.sizes[idx] { return false; }
        self.moments_v[idx].write_f32(0, data);
        true
    }

    /// Download the current weights[idx] for checkpointing.
    pub fn download_weight(&self, idx: usize) -> Vec<f32> {
        self.weights[idx].read_f32(0, self.sizes[idx])
    }

    /// Download moments_m[idx] for checkpointing.
    pub fn download_m(&self, idx: usize) -> Vec<f32> {
        self.moments_m[idx].read_f32(0, self.sizes[idx])
    }

    /// Download moments_v[idx] for checkpointing.
    pub fn download_v(&self, idx: usize) -> Vec<f32> {
        self.moments_v[idx].read_f32(0, self.sizes[idx])
    }

    /// Zero the grads buffer for tensor `idx`. Call at the start of each step.
    /// Uses BAR memset-through-pointer (not a kernel) — small compared to the
    /// gradient accumulation that follows.
    pub fn zero_grad(&self, idx: usize) {
        let n = self.sizes[idx];
        unsafe {
            let p = self.grads[idx].cpu_ptr as *mut f32;
            std::ptr::write_bytes(p, 0, n);
        }
    }

    /// Zero every grad buffer.
    pub fn zero_grads(&self) {
        for i in 0..self.sizes.len() { self.zero_grad(i); }
    }

    /// Mutable CPU view over `grads[idx]`. For writing gradients computed on CPU.
    /// The write goes straight to VRAM via BAR — no staging copy.
    ///
    /// # Safety
    /// The caller must not alias this slice across threads without external
    /// synchronisation; `GpuBuffer` is `Sync` but BAR writes are not lock-free.
    pub fn grad_as_mut_slice(&self, idx: usize) -> &mut [f32] {
        let n = self.sizes[idx];
        unsafe {
            std::slice::from_raw_parts_mut(self.grads[idx].cpu_ptr as *mut f32, n)
        }
    }

    /// Mutable CPU view over `weights[idx]`. For reading current weights from
    /// forward paths that haven't moved to GPU yet. Zero-copy — the slice is
    /// backed by BAR-mapped VRAM. Treat as a read-only slice during training
    /// (writes race with the AdamW kernel).
    pub fn weight_as_slice(&self, idx: usize) -> &[f32] {
        let n = self.sizes[idx];
        unsafe {
            std::slice::from_raw_parts(self.weights[idx].cpu_ptr as *const f32, n)
        }
    }

    /// Weights VA pointer for zero-copy kernel dispatch.
    pub fn weight_va(&self, idx: usize) -> u64 { self.weights[idx].va_addr }
    pub fn grad_va(&self, idx: usize) -> u64 { self.grads[idx].va_addr }
    pub fn m_va(&self, idx: usize) -> u64 { self.moments_m[idx].va_addr }
    pub fn v_va(&self, idx: usize) -> u64 { self.moments_v[idx].va_addr }
}
