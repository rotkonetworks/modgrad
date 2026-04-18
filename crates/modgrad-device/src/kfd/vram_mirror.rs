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

    // ─── Uploads: modify VRAM, require &mut self to prevent concurrent aliasing ───
    //
    // Previously took &self — which let a caller hold multiple simultaneous
    // &mut [f32] handles via other methods (safety audit finding). Upgrading
    // to &mut self means the borrow checker rejects any aliasing at compile
    // time, and removes a class of "the caller must synchronise" bugs.

    /// Upload `data` to the weights[idx] buffer. Typically called once at start.
    ///
    /// # Kernel-in-flight contract
    /// The caller must ensure no GPU kernel is currently reading `weights[idx]`.
    /// If a kernel was previously dispatched touching this buffer, call
    /// [`Self::wait_idle`] first — a concurrent BAR write + GPU read is a
    /// hardware-level data race even though the Rust types don't alias.
    ///
    /// Returns false on length mismatch.
    pub fn upload_weight(&mut self, idx: usize, data: &[f32]) -> bool {
        if idx >= self.sizes.len() || data.len() != self.sizes[idx] { return false; }
        self.weights[idx].write_f32(0, data);
        true
    }

    /// Upload `data` to moments_m[idx]. Used when resuming from a checkpoint.
    pub fn upload_m(&mut self, idx: usize, data: &[f32]) -> bool {
        if idx >= self.sizes.len() || data.len() != self.sizes[idx] { return false; }
        self.moments_m[idx].write_f32(0, data);
        true
    }

    /// Upload `data` to moments_v[idx]. Used when resuming from a checkpoint.
    pub fn upload_v(&mut self, idx: usize, data: &[f32]) -> bool {
        if idx >= self.sizes.len() || data.len() != self.sizes[idx] { return false; }
        self.moments_v[idx].write_f32(0, data);
        true
    }

    // ─── Downloads: read-only; &self is fine. See kernel-in-flight warning. ───

    /// Download the current weights[idx] for checkpointing.
    ///
    /// # Kernel-in-flight contract
    /// Caller must ensure no kernel is currently writing `weights[idx]`. In
    /// practice this means calling this only between training steps, never
    /// inside `step_update`.
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

    // ─── Mutating views: &mut self so the borrow checker enforces exclusivity. ───

    /// Zero the grads buffer for tensor `idx`. Call at the start of each step.
    /// Uses BAR memset through the CPU-mapped pointer (not a kernel dispatch).
    ///
    /// # Kernel-in-flight contract
    /// Racing this against a kernel reading `grads[idx]` is a hardware data
    /// race. `&mut self` prevents aliasing in Rust, but the GPU queue is an
    /// independent execution engine — callers must ensure the grad buffer
    /// is not being read by any dispatched kernel. In practice:
    ///     [last step's AdamW reads grads] → `flush()` → `zero_grads()` → fill.
    pub fn zero_grad(&mut self, idx: usize) {
        let n = self.sizes[idx];
        unsafe {
            let p = self.grads[idx].cpu_ptr as *mut f32;
            std::ptr::write_bytes(p, 0, n);
        }
    }

    /// Zero every grad buffer. Same kernel-in-flight contract as [`zero_grad`].
    pub fn zero_grads(&mut self) {
        for i in 0..self.sizes.len() { self.zero_grad(i); }
    }

    /// Mutable CPU view over `grads[idx]` through the BAR mapping.
    ///
    /// Upgraded to `&mut self` from the original `&self`: two simultaneous
    /// mutable slices to different tensor indices CAN be obtained without
    /// aliasing (each `GpuBuffer` has its own `cpu_ptr`), but the `&self`
    /// variant allowed safely-looking code that racing-y hardware could still
    /// corrupt. For batched writes use [`Self::grads_iter_mut`] which yields
    /// disjoint slices under a single `&mut self` borrow.
    ///
    /// # Kernel-in-flight contract
    /// See [`zero_grad`] — callers must sequence BAR writes against GPU
    /// reads of the same buffer externally (e.g. via `flush()`).
    pub fn grad_as_mut_slice(&mut self, idx: usize) -> &mut [f32] {
        let n = self.sizes[idx];
        unsafe {
            std::slice::from_raw_parts_mut(self.grads[idx].cpu_ptr as *mut f32, n)
        }
    }

    /// Iterator over every grad buffer as disjoint `&mut [f32]`. Exclusive
    /// borrow of `self` guarantees the collection is not aliased. Useful for
    /// sequentially staging per-tensor gradients into the mirror in one pass.
    pub fn grads_iter_mut(&mut self) -> impl Iterator<Item = &mut [f32]> {
        // SAFETY: Each `GpuBuffer` owns a disjoint `cpu_ptr`/`va_addr`; two
        // slices produced from different buffers cannot alias. The `&mut self`
        // receiver ensures no other path into the mirror's grads is active
        // for the iterator's lifetime.
        let sizes = self.sizes.clone();
        self.grads.iter_mut().zip(sizes.into_iter()).map(|(buf, n)| {
            unsafe { std::slice::from_raw_parts_mut(buf.cpu_ptr as *mut f32, n) }
        })
    }

    /// Immutable CPU view over `weights[idx]` through the BAR mapping.
    /// Safe while no kernel is writing `weights[idx]` — see kernel-in-flight
    /// contract on [`upload_weight`].
    pub fn weight_as_slice(&self, idx: usize) -> &[f32] {
        let n = self.sizes[idx];
        unsafe {
            std::slice::from_raw_parts(self.weights[idx].cpu_ptr as *const f32, n)
        }
    }

    /// Wait for all pending GPU dispatches on the mirror's tensors to finish.
    /// Call this before any BAR-side upload/download/zero of a buffer that
    /// might still be referenced by an in-flight kernel. Implemented as a
    /// no-op here — the underlying queue semantics already enforce ordering
    /// between `submit_wait` calls, and higher-level code (`step_update`
    /// etc.) drives the queue synchronously. If in future we switch to
    /// truly async queue submission this method is where the barrier goes.
    pub fn wait_idle(&self) {
        // Intentional no-op today. Keeping the entry point so callers can
        // express the intent; a later async backend can fill it in.
    }

    /// Weights VA pointer for zero-copy kernel dispatch.
    pub fn weight_va(&self, idx: usize) -> u64 { self.weights[idx].va_addr }
    pub fn grad_va(&self, idx: usize) -> u64 { self.grads[idx].va_addr }
    pub fn m_va(&self, idx: usize) -> u64 { self.moments_m[idx].va_addr }
    pub fn v_va(&self, idx: usize) -> u64 { self.moments_v[idx].va_addr }
}
