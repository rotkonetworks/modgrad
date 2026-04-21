//! KFD backend — thin wrapper around the existing `kfd::accel::try_*`
//! dispatch functions that turns them into a `Backend` trait impl.
//!
//! Phase 2 intent: wrap without rewriting. The 25 hand-written RDNA3
//! assembly kernels stay exactly where they are; this module only
//! translates between the `Op` variants and the pre-existing function
//! signatures, and lies about support where a kernel doesn't exist yet
//! on this path.
//!
//! Coverage on first landing (review-sized chunk): `matvec`, `matmul`,
//! `adamw`. Every other op falls through to CPU via `supports()`.
//! Follow-up commits extend to the full 25-kernel surface one op at a
//! time — each commit reviewable, each gated by the parity harness.

use crate::kfd::accel;

use super::{Backend, BackendError, BufferBackend, ComputeCtx, DeviceBuffer, DeviceInfo, DeviceKind, Op, QuantKind};

/// RDNA3 gfx1102 hand-written kernel backend. Available only when the
/// global KFD GPU state has successfully initialised (`accel::available`).
pub struct KfdBackend {
    /// Cached at construction — probing `accel::available()` on every
    /// dispatch would lock the global mutex for no reason.
    available: bool,
}

/// Turn a `try_*` bool return into our Result type. Every KFD dispatch
/// arm has the same shape: call the underlying function, translate
/// false → Runtime error named after the op. Centralizing the format
/// string also keeps error text consistent across arms.
#[inline]
fn try_result(ok: bool, op_name: &'static str) -> Result<(), BackendError> {
    if ok {
        Ok(())
    } else {
        Err(BackendError::Runtime(format!("kfd {op_name} dispatch failed")))
    }
}

impl KfdBackend {
    /// Probe the KFD runtime. Returns `None` when any of:
    ///   - The `kfd` Cargo feature is disabled.
    ///   - `MODGRAD_ENABLE_KFD` is not set (belt-and-suspenders guard
    ///     while the ring-stall bug is being investigated — see the
    ///     comment below).
    ///   - No supported GPU is present at runtime.
    ///
    /// Why the extra env-var gate: session 2026-04-21 found the KFD
    /// compute ring reaches a wedged state (put=4194313, read=9 —
    /// exact same values every process) after the boot-probe phase,
    /// even when the training code is configured to only dispatch
    /// single-workgroup matvecs to KFD. Ring state appears to
    /// persist across processes or the self-test dispatches are
    /// themselves filling the ring without the GPU draining.
    /// Root cause not yet narrowed. Until then, default-enabling KFD
    /// turns every `cargo build -p mazes --features kfd` binary into
    /// one that panics on first forward pass. Safer to require
    /// explicit opt-in: `MODGRAD_ENABLE_KFD=1 ./mazes ...`.
    pub fn try_new() -> Option<Self> {
        #[cfg(not(feature = "kfd"))]
        { return None; }
        #[cfg(feature = "kfd")]
        {
            match std::env::var("MODGRAD_ENABLE_KFD").as_deref() {
                Ok("1") | Ok("true") | Ok("yes") => {}
                _ => return None,
            }
            if accel::available() {
                Some(Self { available: true })
            } else {
                None
            }
        }
    }

    /// Construct without a probe — primarily for tests where the global
    /// mutex is awkward. Non-test callers should prefer `try_new`.
    pub fn new_unchecked() -> Self {
        Self { available: accel::available() }
    }
}

impl Backend for KfdBackend {
    fn name(&self) -> &'static str { "kfd" }

    fn invalidate_cache(&self) {
        accel::invalidate_cache();
    }

    fn device_info(&self) -> DeviceInfo {
        // Without extending `accel::` we don't have a direct handle on
        // total VRAM or arch string from here — leave placeholders and
        // fill in via a follow-up when we give `KfdDevice` a probe API.
        DeviceInfo {
            kind: DeviceKind::Kfd,
            name: "AMD RDNA3 (gfx1102)".into(),
            total_mem_bytes: 0,
            arch: Some("gfx1102".into()),
        }
    }

    fn supports(&self, op: &Op) -> bool {
        if !self.available { return false; }
        match op {
            // KFD matvec kernel:
            //   - proptest-found divergence at multiple small shapes
            //     including (6, 9).
            //   - boot-probe shapes 256x64 (1 WG) pass; 1024x64 (4 WG),
            //     4096x64 (16 WG), 8448x64 (33 WG) passed once per boot
            //     but multi-WG wedged the compute ring during real
            //     training (ring-buffer-full panic after ~4M dwords
            //     enqueued with only 9 consumed — GPU not draining).
            //   - Single-workgroup matvec (out_dim <= 256) has been
            //     reliable across every observed run.
            //
            // Conservative gate: claim only out_dim in [32, 256] so
            // multi-workgroup dispatches (which wedge the ring) fall
            // through to CPU. Raises the "is this safe yet?" bar for
            // any future widening — has to survive a full training
            // step without ring stall, not just a one-off kernel probe.
            Op::Matvec { quant: QuantKind::F32, out_dim, in_dim, .. }
                if *out_dim >= 32 && *out_dim <= 256 && *in_dim >= 32 => true,
            Op::MatvecT { out_dim, in_dim, .. }
                if *out_dim >= 32 && *out_dim <= 256 && *in_dim >= 32 => true,
            //
            // ── All other ops temporarily disabled ──────────────────
            //
            // Debugging session 2026-04-21: at size=21 d_model=256 the
            // compute ring wedges within one forward pass even with
            // single-WG matvec gated. Something among MatmulNN /
            // AdamW / SiluFwd{,Inplace} / SiluBwd / GluFwd /
            // SgdUpdate / TraceRotateInplace / OuterProductAcc /
            // SuperLinearFwd is either (a) a kernel bug the proptest
            // shape-gate didn't catch, or (b) the ring-fill pattern
            // from dispatching too many ops without GPU draining.
            //
            // Strategy: disable all of them for now so single-WG
            // matvec is the only KFD claim. Training works end-to-end
            // on CPU+KFD mix with no hang. Each op can be re-enabled
            // individually once (i) parity proptest passes at its
            // full claimed shape range AND (ii) a full training step
            // at our benchmark config runs without wedging the ring.
            // The second bar is new — kernel-parity is necessary but
            // not sufficient.
            //
            // Previous gates are preserved below as comments so the
            // existing shape analysis isn't lost. Re-enable one at a
            // time, verify with mazes end-to-end, then commit.

            // Op::MatmulNN { m, k, n, .. }
            //     if *n % 32 == 0 && *k % 8 == 0 && *m % 128 == 0 => true,
            // Op::AdamW(args) if args.w.len() >= 32 && args.w.len() % 32 == 0 => true,
            // Op::SiluFwd { x, .. } if x.len() >= 32 && x.len() % 32 == 0 => true,
            // Op::SiluFwdInplace { x } if x.len() >= 32 && x.len() % 32 == 0 => true,
            // Op::SiluBwd { x, .. } if x.len() >= 32 && x.len() % 32 == 0 => true,
            // Op::GluFwd { x, .. }  if x.len() >= 64 && x.len() % 64 == 0 => true,
            // Op::SgdUpdate { w, .. } if w.len() >= 32 && w.len() % 32 == 0 => true,
            // Op::TraceRotateInplace { d_model, memory_length, .. }
            //     if d_model * memory_length >= 32
            //         && (d_model * memory_length) % 32 == 0 => true,
            // Op::OuterProductAcc { m, n, .. }
            //     if *m >= 16 && n.is_power_of_two() && *n >= 32 => true,
            // Op::SuperLinearFwd { cache: None, d_model, out_per, memory_length, .. }
            //     if *d_model >= 16 && *out_per >= 2 && *memory_length >= 8 => true,
            // Fused synapse and standalone layer_norm_inplace:
            // **declined across the board in Stage 1.** Parity proptest
            // at committed tolerance (max_abs 2e-5, max_rel 1e-4) found
            // divergence at:
            //   - SynapseForward: out_dim=32, in_dim=96,
            //     seed=12390942540437353913 → |Δ|≈1.5e-4 (rel 1.5e-4).
            //   - LayerNormInplace: n_cols=96,
            //     seed=556123943721261328  → |Δ|≈3.96e-4 (rel 3.96e-4).
            // The KFD single-WG layer_norm_fwd kernel's parallel
            // reduction + approximate-division stack pushes drift past
            // the committed f32 tolerance at small output shapes.
            //
            // Per compute-device-unify plan failure-mode criterion:
            // tighten the supports() gate OR land with CPU-only fused
            // impl. Here we take the latter — the KFD kernels stay
            // callable via `kfd::accel::try_synapse_forward` /
            // `try_layer_norm_inplace` for existing modgrad-compute
            // callers (their tolerance decisions predate this audit),
            // but the `Op`-layer registry declines. Future kernel fix
            // re-enables fusion via this gate as a standalone commit.
            Op::SynapseForward { .. } => false,
            Op::LayerNormInplace { .. } => false,
            // Deferred — require Op/kernel alignment first. Tracked in
            // `tasks/kfd-kernel-bugs.md` ("Deferred" section). Each marker
            // below mirrors the SynapseForward precedent: inline reason,
            // pointer to the tracker for seed/shape detail (N/A here —
            // these fail the Op match before proptest reaches a dispatch).
            // FIXME(kfd-layer-norm-fwd): KFD's inplace kernel doesn't emit
            //   the `cache` tensor the Op variant expects; backward handoff
            //   blocked on kernel surface change. See
            //   tasks/kfd-kernel-bugs.md#deferred.
            // FIXME(kfd-sync-update-fwd): KFD's kernel expects
            //   alpha/beta/phases/dopamine args the Op struct doesn't
            //   carry; blocked on Op enum extension. See
            //   tasks/kfd-kernel-bugs.md#deferred.
            // FIXME(kfd-super-linear-bwd): Dw/Dx arg-shape mismatch between
            //   Op variants and KFD's existing dispatcher; blocked on
            //   dispatcher rework. See tasks/kfd-kernel-bugs.md#deferred.
            _ => false,
        }
    }

    fn dispatch(&self, op: &mut Op) -> Result<(), BackendError> {
        match op {
            Op::Matvec {
                x, weight, bias, out,
                out_dim, in_dim,
                quant: QuantKind::F32,
            } => try_result(
                accel::try_matvec(x, weight, bias, out, *out_dim as u32, *in_dim as u32),
                "matvec",
            ),

            Op::MatvecT { d_out, weight, d_input, out_dim, in_dim } => try_result(
                accel::try_matvec_t(d_out, weight, d_input, *out_dim as u32, *in_dim as u32),
                "matvec_t",
            ),

            Op::MatmulNN { a, b, out, bias, m, k, n } => {
                // KFD try_matmul does bias broadcast internally; pass
                // through our `bias` Option directly (zero-filled if None).
                let zero_bias;
                let bias_slice: &[f32] = match bias {
                    Some(b) => b,
                    None => { zero_bias = vec![0.0f32; *n]; &zero_bias },
                };
                try_result(
                    accel::try_matmul(a, b, bias_slice, out, *m as u32, *k as u32, *n as u32),
                    "matmul_nn",
                )
            }

            Op::AdamW(args) => try_result(
                accel::try_adamw(
                    args.w, args.g, args.m, args.v,
                    args.lr, args.beta1, args.beta2, args.eps, args.weight_decay,
                    args.bc1_inv, args.bc2_inv,
                ),
                "adamw",
            ),

            Op::SiluFwd { x, out } => {
                // KFD kernel is in-place; copy input over first, then
                // dispatch on the output buffer. One memcpy of size
                // `x.len()` — negligible compared to GPU dispatch cost.
                // Callers that don't need the pre-activation preserved
                // should prefer `Op::SiluFwdInplace` to skip this copy.
                out.copy_from_slice(x);
                try_result(accel::try_silu_inplace(out), "silu")
            }

            Op::SiluFwdInplace { x } => {
                // Native in-place path — the whole point of this variant.
                // No copy_from_slice: the kernel mutates `x` directly.
                try_result(accel::try_silu_inplace(x), "silu_inplace")
            }

            Op::SiluBwd { d_out, x, d_x } => try_result(
                accel::try_silu_backward(d_out, x, d_x),
                "silu_backward",
            ),

            Op::GluFwd { x, out } => try_result(
                // try_glu expects `n = out.len()` (the half-size).
                accel::try_glu(x, out, out.len() as u32),
                "glu",
            ),

            Op::OuterProductAcc { a, b, accum, m, n } => {
                // try_outer_product naming: (d_weight, d_out, input, out_dim, in_dim)
                //   d_weight accumulator of shape [out_dim × in_dim]
                //   d_out has out_dim rows; input has in_dim cols.
                // Our Op::OuterProductAcc has (a, b, accum) with:
                //   accum[i, j] += a[i] * b[j]  with shape [m × n]
                // Map: d_out=a (rows, size m), input=b (cols, size n),
                //      d_weight=accum, out_dim=m, in_dim=n.
                if accel::try_outer_product(accum, a, b, *m as u32, *n as u32) {
                    Ok(())
                } else {
                    Err(BackendError::Runtime("kfd outer_product dispatch failed".into()))
                }
            }

            Op::SgdUpdate { w, g, lr } => try_result(
                accel::try_sgd_update(w, g, *lr),
                "sgd_update",
            ),

            Op::TraceRotateInplace { trace, new_val, d_model, memory_length } => try_result(
                accel::try_trace_shift(trace, new_val, *d_model as u32, *memory_length as u32),
                "trace_shift",
            ),

            Op::SuperLinearFwd {
                trace, weights, biases, out,
                cache: None,
                d_model, memory_length, out_per,
            } => try_result(
                accel::try_superlinear(
                    trace, weights, biases, out,
                    *d_model as u32, *memory_length as u32, *out_per as u32,
                ),
                "superlinear",
            ),

            Op::SynapseForward { weight, bias, x, out, out_dim, in_dim } => {
                // `try_synapse_forward`'s `_scratch` parameter is unused
                // (the engine uses its own VRAM-resident scratch slot);
                // pass an empty slice to keep the signature honest.
                let mut scratch: [f32; 0] = [];
                try_result(
                    accel::try_synapse_forward(
                        x, weight, bias, out, &mut scratch,
                        *out_dim as u32, *in_dim as u32,
                    ),
                    "synapse_forward",
                )
            }

            Op::LayerNormInplace { x, n_rows: _, n_cols: _ } => try_result(
                // KFD kernel is single-row / single-WG — `supports()`
                // already gated `n_rows == 1`, so `x` is exactly one row.
                accel::try_layer_norm_inplace(x),
                "layer_norm_inplace",
            ),

            // Anything else should have been filtered by `supports()`.
            _ => Err(BackendError::Unsupported {
                op: op.name(),
                backend: "kfd",
            }),
        }
    }
}

/// KFD device-resident buffer: a thin `DeviceBuffer`-flavoured wrapper
/// around the existing `kfd::memory::GpuBuffer`. Allocates VRAM via
/// `accel::alloc_vram`, which rounds to 4 KiB pages internally.
///
/// The `GpuBuffer` lives as long as the `KfdBuffer` does — drop runs
/// the ioctl cleanup (unmap + free) automatically. `write_f32` and
/// `read_f32` are the underlying host-side copy operations; both go
/// through the BAR mapping (CPU-visible through resizable BAR on
/// gfx1102), so there's no explicit DMA submit.
///
/// `len` reports the **logical** f32 count requested at alloc time,
/// not the rounded-up page-aligned capacity of the underlying
/// `GpuBuffer`. Callers reason about their own sizing.
pub struct KfdBuffer {
    /// `None` only when the GPU was unavailable at alloc time — kept
    /// as an invariant to let `copy_*_host` return a clear runtime
    /// error rather than panic. In practice `alloc_buffer` refuses to
    /// construct a `KfdBuffer` without a live `GpuBuffer`, so this is
    /// belt-and-suspenders.
    inner: Option<crate::kfd::memory::GpuBuffer>,
    /// Logical element count, in f32s, as requested by the caller.
    len_f32: usize,
}

// SAFETY: GpuBuffer itself implements Send + Sync (see memory.rs); the
// Option wrapper doesn't change that, and `len_f32` is trivially safe.
unsafe impl Send for KfdBuffer {}
unsafe impl Sync for KfdBuffer {}

impl KfdBuffer {
    /// Destructively unwrap the underlying `GpuBuffer`. Returns `None`
    /// only if `alloc_buffer` returned a buffer with no backing GPU
    /// allocation (an invariant violation — kept here defensively).
    ///
    /// Escape hatch for Stage 5 of the compute-device unification
    /// chain: `modgrad_compute::tensor_device::VramTensor` holds a raw
    /// `GpuBuffer` + reaches into `cpu_ptr` / `va_addr` directly for
    /// zero-copy BAR access and kernel dispatch. Rather than pollute
    /// the public [`DeviceBuffer`] trait with those accessors, the
    /// tensor type keeps using its historical `GpuBuffer` view after
    /// unwrapping a freshly-allocated `KfdBuffer`. Stage 6 dissolves
    /// this when `VramTensor` either moves behind the `DeviceBuffer`
    /// contract itself or grows dedicated tensor-shape accessors.
    ///
    /// `#[doc(hidden)]` — this is an internal seam, not a stable API.
    #[doc(hidden)]
    pub fn into_inner(self) -> Option<crate::kfd::memory::GpuBuffer> {
        self.inner
    }
}

impl DeviceBuffer for KfdBuffer {
    fn backend_name(&self) -> &'static str { "kfd" }

    fn len(&self) -> usize { self.len_f32 }

    fn copy_from_host(&mut self, src: &[f32]) -> Result<(), BackendError> {
        if src.len() > self.len_f32 {
            return Err(BackendError::Runtime(format!(
                "KfdBuffer::copy_from_host: src.len()={} > buffer.len()={}",
                src.len(), self.len_f32,
            )));
        }
        let buf = self.inner.as_ref().ok_or_else(|| {
            BackendError::Runtime("KfdBuffer::copy_from_host: no backing GpuBuffer".into())
        })?;
        // write_f32 takes a byte offset; we always write from the start.
        buf.write_f32(0, src);
        Ok(())
    }

    fn copy_to_host(&self, dst: &mut [f32]) -> Result<(), BackendError> {
        if dst.len() > self.len_f32 {
            return Err(BackendError::Runtime(format!(
                "KfdBuffer::copy_to_host: dst.len()={} > buffer.len()={}",
                dst.len(), self.len_f32,
            )));
        }
        let buf = self.inner.as_ref().ok_or_else(|| {
            BackendError::Runtime("KfdBuffer::copy_to_host: no backing GpuBuffer".into())
        })?;
        // read_f32 allocates a fresh Vec; copy into the caller's slice
        // afterwards. Acceptable for Stage 2 — the hot path for KFD
        // still goes through zero-copy arena slices, not this API.
        let v = buf.read_f32(0, dst.len());
        dst.copy_from_slice(&v);
        Ok(())
    }
}

/// Wire KFD's VRAM allocator into the `BufferBackend` trait.
///
/// `accel::alloc_vram` takes bytes, returns `Option<GpuBuffer>`
/// (None when the GPU isn't available — matches our compile-time
/// invariant that the backend is only alive when `available = true`,
/// but we keep the None path reachable so a race with device loss
/// surfaces as a clean Runtime error).
impl BufferBackend for KfdBackend {
    type Buffer = KfdBuffer;

    fn alloc_buffer(&self, n_f32: usize) -> Result<KfdBuffer, BackendError> {
        let bytes = (n_f32 as u64).saturating_mul(4);
        let buf = accel::alloc_vram(bytes)
            .ok_or_else(|| BackendError::Runtime(
                "kfd alloc_buffer: accel::alloc_vram returned None".into()
            ))?;
        Ok(KfdBuffer { inner: Some(buf), len_f32: n_f32 })
    }
}

/// KFD-specific overrides for [`ComputeCtx`].
///
/// `arena_reset` is the entire point of this block: it scopes what used
/// to be a process-global `accel::arena_reset()` call to a
/// `ComputeCtx<KfdBackend>`. Callers that hold a context against a
/// specific backend now reset just that context's arena — matching the
/// "Committed stance: global mutable state stays" invariant while
/// beginning to move call sites away from the bare global fn.
///
/// `flush` is a no-op today (KFD submits synchronously in
/// `accel::try_*`). It exists here so that future changes migrating to
/// an async submit model land as one inherent-impl patch on
/// `ComputeCtx<KfdBackend>` rather than touching every caller.
impl ComputeCtx<KfdBackend> {
    /// Reset the KFD VRAM arena — forwards to `accel::arena_reset()`.
    ///
    /// See `tasks/compute-device-unify.md` under "Committed stance:
    /// global mutable state stays" for why this still goes through the
    /// global fn. Stage 3+ threads a real per-context arena through.
    pub fn arena_reset(&self) {
        accel::arena_reset();
    }

    /// No-op on KFD today — `accel::try_*` dispatch is already
    /// synchronous. Kept as an inherent override so migrations to an
    /// async dispatch path land here without touching callers.
    pub fn flush(&self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    /// try_new is runtime-conditional — on a CI host without gfx1102 it
    /// returns None. That's the defensive path we want the test to hit
    /// so we don't silently break CI on non-gfx1102 boxes.
    #[test]
    fn try_new_is_none_when_no_gpu_or_available_when_gfx1102() {
        let backend = KfdBackend::try_new();
        // Either is acceptable; this test documents that try_new is a
        // pure probe, not a side-effect constructor. If we're on
        // hardware, it returns Some; otherwise None. Both legal.
        match backend {
            Some(b) => assert_eq!(b.name(), "kfd"),
            None => { /* expected on non-gfx1102 CI */ }
        }
    }

    #[test]
    fn backend_declines_ops_not_yet_ported() {
        let be = KfdBackend { available: true };
        let x = [0.0f32; 4];
        let mut out = [0.0f32; 4];
        let mut cache = [0.0f32; 4];
        let gamma = [1.0f32; 4];
        let beta = [0.0f32; 4];
        // LayerNormFwd is intentionally deferred (KFD inplace kernel
        // doesn't emit cache). `supports` must return false so the
        // registry falls through to CPU for it.
        let op = Op::LayerNormFwd {
            x: &x, gamma: &gamma, beta: &beta,
            out: &mut out, cache: Some(&mut cache),
            n_rows: 1, n_cols: 4,
        };
        assert!(!be.supports(&op));
    }

    #[test]
    fn backend_declines_when_gpu_unavailable() {
        let be = KfdBackend { available: false };
        let x = [0.0f32; 4];
        let weight = [0.0f32; 16];
        let bias = [0.0f32; 4];
        let mut out = [0.0f32; 4];
        let op = Op::Matvec {
            x: &x, weight: &weight, bias: &bias, out: &mut out,
            out_dim: 4, in_dim: 4, quant: QuantKind::F32,
        };
        // Even though matvec is in our coverage list, an unavailable
        // GPU must make supports() return false so the registry skips
        // to the next backend.
        assert!(!be.supports(&op));
    }
}
