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

use super::{Backend, BackendError, DeviceInfo, DeviceKind, Op, QuantKind};

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
    /// Probe the KFD runtime. Returns `None` when:
    ///   - The `kfd` Cargo feature is disabled (KFD is opt-in, not a
    ///     default backend). Users running on gfx1102 who want the
    ///     hand-written fast path enable it explicitly.
    ///   - No supported GPU is present at runtime.
    pub fn try_new() -> Option<Self> {
        #[cfg(not(feature = "kfd"))]
        { return None; }
        #[cfg(feature = "kfd")]
        {
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
            // KFD matvec kernel proptest-found divergence at multiple
            // shapes including (6, 9). Production use in modgrad-ctm
            // runs at in_dim=64+ (kernel boot probes at 256x64, 1024x64,
            // 4096x64, 8448x64 all pass). Gate conservatively to that
            // regime; smaller shapes fall through to CPU. Needs kernel
            // audit before we can safely widen.
            Op::Matvec { quant: QuantKind::F32, out_dim, in_dim, .. }
                if *out_dim >= 32 && *in_dim >= 32 => true,
            Op::MatvecT { out_dim, in_dim, .. }
                if *out_dim >= 32 && *in_dim >= 32 => true,
            // KFD's matmul kernel has hard alignment preconditions:
            //   n % 32 == 0, k % 8 == 0, m % 128 == 0
            // Fail `supports()` for non-conforming shapes so registry
            // falls through to CPU (rather than claiming and erroring).
            Op::MatmulNN { m, k, n, .. }
                if *n % 32 == 0 && *k % 8 == 0 && *m % 128 == 0 => true,
            // Elementwise ops: proptest found tail-handling issues at
            // non-wavefront-aligned sizes. Gate conservatively to
            // multiples of 32 (RDNA3 wavefront) of reasonable size;
            // non-aligned cases fall through to CPU.
            Op::AdamW(args) if args.w.len() >= 32 && args.w.len() % 32 == 0 => true,
            Op::SiluFwd { x, .. } if x.len() >= 32 && x.len() % 32 == 0 => true,
            Op::SiluFwdInplace { x } if x.len() >= 32 && x.len() % 32 == 0 => true,
            Op::SiluBwd { x, .. } if x.len() >= 32 && x.len() % 32 == 0 => true,
            // GluFwd: input has length 2*half; gate on that.
            Op::GluFwd { x, .. }  if x.len() >= 64 && x.len() % 64 == 0 => true,
            Op::SgdUpdate { w, .. } if w.len() >= 32 && w.len() % 32 == 0 => true,
            // TraceRotateInplace: proptest found divergence at non-aligned
            // (d_model, memory_length) e.g. (3, 27). Gate on total
            // elements being a multiple of 32 (wavefront). Production
            // shapes (d_model × memory_length in {32×8, 512×64}) clear this.
            Op::TraceRotateInplace { d_model, memory_length, .. }
                if d_model * memory_length >= 32
                    && (d_model * memory_length) % 32 == 0 => true,
            // KFD outer_product: proptest finds divergence at many
            // mid-sized shapes (e.g. m=20, n=13). The float-div
            // row/col approximation likely fails when K isn't a power
            // of two. Gate to shapes where K is a clean power-of-two
            // multiple — these are what production actually uses.
            Op::OuterProductAcc { m, n, .. }
                if *m >= 16 && n.is_power_of_two() && *n >= 32 => true,
            // KFD try_superlinear: proptest found divergence at small
            // d_model/out_per combinations. Gate on shape bounds that
            // the hand-written parity case exercises successfully.
            Op::SuperLinearFwd { cache: None, d_model, out_per, memory_length, .. }
                if *d_model >= 16 && *out_per >= 2 && *memory_length >= 8 => true,
            // Deferred — require Op/kernel alignment first:
            //   LayerNormFwd         → KFD's inplace kernel doesn't emit cache
            //   SyncUpdateFwd        → KFD's kernel expects alpha/beta/phases/dopamine
            //   SuperLinearBwd{Dw,Dx}→ arg shape mismatch; TODO align
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

            // Anything else should have been filtered by `supports()`.
            _ => Err(BackendError::Unsupported {
                op: op.name(),
                backend: "kfd",
            }),
        }
    }
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
