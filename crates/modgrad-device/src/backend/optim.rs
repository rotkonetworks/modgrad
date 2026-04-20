//! Batched-optimizer abstraction — one submit for N weight updates.
//!
//! `BatchedOptimizer` is a dispatch policy, not a device concern: it
//! answers "how does this backend turn a sequence of AdamW steps into
//! one round trip?" CPU has no batching to do, so the fallback iterates
//! one at a time through `ops::adamw`. KFD amortises N dispatches into
//! one `submit_wait` via `kfd::accel::try_adamw_vram_batch` when the
//! weights already live in a `VramMirror`.
//!
//! The trait lives under [`super::ops`] (the façade) at the layering
//! diagram's request — see `tasks/compute-device-unify.md` Stage 5. It
//! is **not** registered in [`BackendRegistry`]: registry-level dispatch
//! is per-`Op`, stateless, and `&[f32]`-slice only. Batched optimizer
//! state (device-resident weights, per-context hyperparams) is the
//! ComputeCtx concern, not the registry's.
//!
//! # KFD-variant semantic compromise
//!
//! The KFD `step_batch` signature accepts `impl Iterator<Item = AdamWArgs<'a>>`
//! but only reads **hyperparams + per-tensor `weight_decay`** from each
//! `AdamWArgs`. Weight/grad/m/v slices inside the args are ignored — on
//! the KFD path those tensors already live in the `VramMirror` that the
//! optimiser wraps, and the kernel reaches them through VA pointers, not
//! through the host-visible slices the args carry.
//!
//! This is the "simplify if the lifetime shape gets twisted" escape hatch
//! from the unification plan: we keep the trait signature uniform across
//! CPU and KFD (one call site works for both), at the cost of the KFD
//! impl silently ignoring the slice fields. An alternative design with a
//! dedicated KFD method would be more rigorous — and less useful at the
//! one call site that actually needs it (`modgrad-compute::optimizer_state`).
//!
//! [`BackendRegistry`]: super::BackendRegistry

use super::op::AdamWArgs;

/// A batched AdamW optimiser. Implementations decide how to turn
/// N updates into 1 dispatch (or N CPU-side passes).
///
/// Contract:
/// - `step_batch` consumes the iterator in one pass; the iterator's
///   order is the tensor-slot order (matters for the KFD impl, which
///   indexes per-tensor weight decay by iteration position).
/// - `flush` drains any backend-queued submits so subsequent reads of
///   the underlying state are coherent. CPU is always coherent → no-op.
pub trait BatchedOptimizer {
    /// Run one batched AdamW step. Returns `true` on success, `false` on
    /// dispatch failure — caller falls back to a CPU AdamW loop.
    fn step_batch<'a, I>(&mut self, slots: I) -> bool
    where
        I: IntoIterator<Item = AdamWArgs<'a>>;

    /// Drain any queued submits. No-op for CPU; GPU backends with async
    /// submit paths override. KFD today is synchronous inside
    /// `try_adamw_vram_batch` so this is a no-op on that backend too —
    /// kept here for forward-compat with an async submit refactor.
    fn flush(&mut self) {}
}

/// CPU fallback: iterate and dispatch `ops::adamw` per slot.
///
/// No device state, no submit batching — the trait is trivial on CPU,
/// and exists mostly so caller code can route through one API rather
/// than special-case CPU vs GPU backends.
pub struct CpuBatchedOptimizer;

impl CpuBatchedOptimizer {
    /// Construct. Zero-sized type — the constructor is purely documentary.
    pub fn new() -> Self { Self }
}

impl Default for CpuBatchedOptimizer {
    fn default() -> Self { Self::new() }
}

impl BatchedOptimizer for CpuBatchedOptimizer {
    fn step_batch<'a, I>(&mut self, slots: I) -> bool
    where
        I: IntoIterator<Item = AdamWArgs<'a>>,
    {
        for args in slots {
            // Panics on dispatch failure — matches every other ops:: fn.
            // We return `true` unconditionally because an op-level panic
            // is a programmer error (bad shapes / no backend supports
            // scalar AdamW), not a "fall back to CPU" condition.
            super::ops::adamw(args);
        }
        true
    }
}

// ─── KFD impl ─────────────────────────────────────────────────────
//
// Constructed with a mutable borrow of the mirror for the scope of one
// step — the caller (OptimizerState::adamw_step in modgrad-compute) is
// the mirror itself, so it can hand out `&mut *self` for this purpose.
//
// The KFD variant is always compiled (not gated on the `kfd` feature)
// because the underlying `kfd::vram_mirror::VramMirror` / `kfd::accel`
// are also always compiled — the feature only gates registry-level
// backend registration. Consumers that care about binary size get the
// benefit on the registration surface; they still link VramMirror's
// code, which is dominated by the buffer allocation pathway anyway.
pub use self::kfd_impls::KfdBatchedOptimizer;

mod kfd_impls {
    use super::{AdamWArgs, BatchedOptimizer};
    use crate::kfd::accel;
    use crate::kfd::vram_mirror::VramMirror;

    /// KFD batched AdamW — one `submit_wait` per step across every
    /// tensor in the wrapped `VramMirror`.
    ///
    /// The mirror is borrowed mutably for the duration of the step; the
    /// kernel writes VRAM in-place through the mirror's VA pointers.
    ///
    /// # Constructor
    /// Most callers use this through a `&mut VramMirror`:
    /// `KfdBatchedOptimizer::new(&mut mirror)`. Drop the optimiser (by
    /// letting it go out of scope) to release the borrow.
    pub struct KfdBatchedOptimizer<'m> {
        mirror: &'m mut VramMirror,
    }

    impl<'m> KfdBatchedOptimizer<'m> {
        /// Wrap a mirror for one batched step.
        pub fn new(mirror: &'m mut VramMirror) -> Self { Self { mirror } }
    }

    impl<'m> BatchedOptimizer for KfdBatchedOptimizer<'m> {
        fn step_batch<'a, I>(&mut self, slots: I) -> bool
        where
            I: IntoIterator<Item = AdamWArgs<'a>>,
        {
            // Collect per-tensor weight decays in iteration order +
            // hyperparams from the first slot. `try_adamw_vram_batch`
            // takes one shared set of hyperparams + `wd_for_idx`, which
            // matches AdamW's conventional "same lr for every tensor,
            // different wd for weights vs. biases/LN" semantics.
            //
            // The slice fields (w/g/m/v) inside each AdamWArgs are
            // **ignored on this path** — see module docs. The mirror's
            // VRAM is the real state; the slices carry only hyperparams.
            let mut iter = slots.into_iter();
            let first = match iter.next() {
                Some(a) => a,
                // Empty batch — nothing to do. Mirror unchanged.
                None => return true,
            };
            let (lr, beta1, beta2, eps, bc1_inv, bc2_inv) =
                (first.lr, first.beta1, first.beta2, first.eps, first.bc1_inv, first.bc2_inv);

            let mut wds: Vec<f32> = Vec::with_capacity(self.mirror.sizes.len());
            wds.push(first.weight_decay);
            for args in iter {
                wds.push(args.weight_decay);
            }
            // Iterator must supply one AdamWArgs per mirror tensor.
            // Mismatches are a caller bug — be strict rather than
            // silently padding or truncating.
            if wds.len() != self.mirror.sizes.len() {
                eprintln!(
                    "KfdBatchedOptimizer::step_batch: slot count {} ≠ mirror tensors {}",
                    wds.len(),
                    self.mirror.sizes.len(),
                );
                return false;
            }

            accel::try_adamw_vram_batch(
                self.mirror, lr, beta1, beta2, eps, bc1_inv, bc2_inv,
                |i| wds[i],
            )
        }

        // KFD submits synchronously inside `try_adamw_vram_batch`; no
        // queued work remains after `step_batch` returns. Kept as the
        // inherent no-op hook for a future async submit refactor.
        fn flush(&mut self) {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_batched_empty_is_ok() {
        let mut opt = CpuBatchedOptimizer::new();
        let empty: Vec<AdamWArgs<'_>> = Vec::new();
        assert!(opt.step_batch(empty));
        opt.flush();
    }

    #[test]
    fn cpu_batched_single_step_matches_direct_adamw() {
        // Drive one AdamW step via both the trait and the direct `ops::adamw`;
        // they must produce identical weights / m / v. The trait adds no
        // numeric behaviour of its own — this is a smoke test that the
        // iterator plumbing doesn't accidentally reorder or skip updates.
        let mut w_a = vec![0.25f32, -0.5, 0.75];
        let mut m_a = vec![0.0f32; 3];
        let mut v_a = vec![0.0f32; 3];
        let g = vec![0.1f32, 0.2, 0.3];

        let mut w_b = w_a.clone();
        let mut m_b = m_a.clone();
        let mut v_b = v_a.clone();

        let args_a = AdamWArgs {
            w: &mut w_a, g: &g, m: &mut m_a, v: &mut v_a,
            lr: 1e-3, beta1: 0.9, beta2: 0.999, eps: 1e-8,
            weight_decay: 0.0, bc1_inv: 1.0, bc2_inv: 1.0,
        };
        super::super::ops::adamw(args_a);

        let args_b = AdamWArgs {
            w: &mut w_b, g: &g, m: &mut m_b, v: &mut v_b,
            lr: 1e-3, beta1: 0.9, beta2: 0.999, eps: 1e-8,
            weight_decay: 0.0, bc1_inv: 1.0, bc2_inv: 1.0,
        };
        let mut opt = CpuBatchedOptimizer::new();
        assert!(opt.step_batch([args_b]));

        assert_eq!(w_a, w_b);
        assert_eq!(m_a, m_b);
        assert_eq!(v_a, v_b);
    }
}
