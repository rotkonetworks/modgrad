//! Device-agnostic training state for an AdamW-like optimizer.
//!
//! An `OptimizerState` holds — per tensor — permanent device storage for
//! (weight, grad, m, v). Training steps:
//!   1. stage gradients via `write_grad_scaled`
//!   2. dispatch `adamw_step` which updates weight / m / v in place
//!   3. optionally `sync_weights_to_cpu` between steps / at checkpoint
//!
//! The trait is the seam between architecture code (FFN, CTM, Transformer)
//! and backend code (KFD on AMD, cudarc on NVIDIA, a future Metal impl).
//! Architecture crates name `dyn OptimizerState`; device crates ship a
//! concrete impl + a factory that returns one.
//!
//! A CUDA impl of the same trait is the whole point — today `FfnAdamW`
//! hard-codes the KFD `VramMirror` type. With this trait it just holds
//! `Box<dyn OptimizerState>` and neither cares nor recompiles when CUDA
//! lands.

/// Permanent device-side storage for an AdamW training step.
///
/// All VRAM-typed operations are sequenced by the trait's `&mut self`
/// methods — callers cannot hold aliased mutable views across tensor
/// indices. VA accessors are `&self` and stable: a tensor's VA doesn't
/// move for the life of the state object.
pub trait OptimizerState: Send {
    /// How many (weight, grad, m, v) tuples this state holds.
    fn n_tensors(&self) -> usize;

    /// Element count of tensor `idx` (not byte count).
    fn tensor_len(&self, idx: usize) -> usize;

    /// Upload `data` into weight[idx]. Typically called once at start and
    /// again when loading a checkpoint. Returns false on length mismatch.
    fn upload_param(&mut self, idx: usize, data: &[f32]) -> bool;
    /// Upload `data` into moments_m[idx].
    fn upload_m(&mut self, idx: usize, data: &[f32]) -> bool;
    /// Upload `data` into moments_v[idx].
    fn upload_v(&mut self, idx: usize, data: &[f32]) -> bool;

    /// Download the current weight[idx] for checkpointing.
    fn download_param(&self, idx: usize) -> Vec<f32>;
    /// Download moments_m[idx].
    fn download_m(&self, idx: usize) -> Vec<f32>;
    /// Download moments_v[idx].
    fn download_v(&self, idx: usize) -> Vec<f32>;

    /// Zero the grad buffer for tensor `idx`.
    fn zero_grad(&mut self, idx: usize);
    /// Zero every grad buffer.
    fn zero_grads(&mut self) {
        for i in 0..self.n_tensors() { self.zero_grad(i); }
    }

    /// Write `grad * scale` into grad[idx]. Scale is typically the
    /// gradient-norm clip factor computed once per step across all
    /// gradients. Single-pass BAR write — faster than a CPU copy + scale
    /// for VRAM-backed state.
    fn write_grad_scaled(&mut self, idx: usize, grad: &[f32], scale: f32);

    /// Run one AdamW step across every tensor.
    ///
    /// * `wd_for(idx)` returns the weight-decay coefficient for tensor `idx`
    ///   (typically `wd` for weights and `0.0` for biases / layer-norm).
    /// * `bc1_inv` / `bc2_inv` are the pre-computed bias-correction terms
    ///   `1/(1-beta^t)` so the dispatch side stays branch-free.
    ///
    /// Returns true on success, false on any dispatch failure — caller
    /// falls back to a CPU AdamW.
    #[allow(clippy::too_many_arguments)]
    fn adamw_step(
        &mut self,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        bc1_inv: f32,
        bc2_inv: f32,
        wd_for: &dyn Fn(usize) -> f32,
    ) -> bool;
}

// ─── KFD impl ─────────────────────────────────────────────────────
//
// Orphan-rule safe: trait is local (this crate), type is foreign
// (modgrad-device). This keeps all device-specific glue inside the
// compute crate's boundary — FFN / CTM / any other architecture crate
// never names a `VramMirror` or the `kfd::` module path.
//
// Future CUDA impl: add an `impl OptimizerState for CudaVramMirror`
// here (or a cuda_backend.rs sibling), plus the appropriate branch
// in the `make_optimizer_state` factory in lib.rs.

use modgrad_device::kfd::vram_mirror::VramMirror;

impl OptimizerState for VramMirror {
    fn n_tensors(&self) -> usize { self.sizes.len() }
    fn tensor_len(&self, idx: usize) -> usize { self.sizes[idx] }

    fn upload_param(&mut self, idx: usize, data: &[f32]) -> bool {
        VramMirror::upload_weight(self, idx, data)
    }
    fn upload_m(&mut self, idx: usize, data: &[f32]) -> bool {
        VramMirror::upload_m(self, idx, data)
    }
    fn upload_v(&mut self, idx: usize, data: &[f32]) -> bool {
        VramMirror::upload_v(self, idx, data)
    }

    fn download_param(&self, idx: usize) -> Vec<f32> { self.download_weight(idx) }
    fn download_m(&self, idx: usize) -> Vec<f32>     { VramMirror::download_m(self, idx) }
    fn download_v(&self, idx: usize) -> Vec<f32>     { VramMirror::download_v(self, idx) }

    fn zero_grad(&mut self, idx: usize) { VramMirror::zero_grad(self, idx); }
    fn zero_grads(&mut self)             { VramMirror::zero_grads(self); }

    fn write_grad_scaled(&mut self, idx: usize, grad: &[f32], scale: f32) {
        VramMirror::write_grad_scaled(self, idx, grad, scale);
    }

    fn adamw_step(
        &mut self,
        lr: f32, beta1: f32, beta2: f32, eps: f32,
        bc1_inv: f32, bc2_inv: f32,
        wd_for: &dyn Fn(usize) -> f32,
    ) -> bool {
        modgrad_device::kfd::accel::try_adamw_vram_batch(
            self, lr, beta1, beta2, eps, bc1_inv, bc2_inv, wd_for,
        )
    }
}
