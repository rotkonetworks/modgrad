//! `Tensor<D>` — JAX-style location-aware tensor (Path C of the
//! 2026-05-05 design discussion).
//!
//! ## Goal
//!
//! Make the SDK's UX feel like JAX: same code runs on CPU or GPU,
//! the type system says where the data lives, ops dispatch by
//! the tensor's location with **no hidden PCIe round-trips**.
//!
//! ```ignore
//! use modgrad_device::backend::tensor::{Tensor, Cpu, matvec};
//!
//! let x = Tensor::<Cpu>::from_slice(&[1.0, 2.0, 3.0]);
//! let w = Tensor::<Cpu>::from_slice(&[ /* 2×3 row-major */ ]);
//! let b = Tensor::<Cpu>::from_slice(&[0.0, 0.0]);
//! let mut y = Tensor::<Cpu>::zeros(2);
//! matvec(&x, &w, &b, &mut y, 2, 3)?;
//! ```
//!
//! Switching to ROCm is a single type swap:
//!
//! ```ignore
//! let x = Tensor::<Rocm>::from_slice(&[1.0, 2.0, 3.0])?;
//! // ... identical body, now runs on GPU with no PCIe round-trips per op.
//! ```
//!
//! ## Why a generic `Device`, not a runtime enum
//!
//! The committed policy at `buffer.rs:1` is **backend-affine**:
//! cross-backend mixing is a compile-time error. A runtime
//! `Tensor::Cpu | Tensor::Rocm` enum would silently support
//! `matvec(&t_cpu, &t_rocm, ...)` (with hidden host bouncing), which
//! is exactly what the policy rejected. With `Tensor<D>`, mixing is
//! a type error: `matvec<D>(...)` requires every argument to share
//! `D`. JAX UX, no hidden costs.
//!
//! ## v0 scope
//!
//! - `Cpu` device fully implemented (matvec only).
//! - `Rocm` device fully implemented (matvec only) under `feature = "rocm"`.
//! - Other ops (matmul_*, matvec_t, sync, AdamW, ...) added in
//!   follow-up commits.
//! - `Linear` / `SuperLinear` / `RegionalWeights` not yet ported;
//!   the cascade is scheduled per-container, one commit at a time.

use std::marker::PhantomData;

use super::BackendError;
use super::buffer::{DeviceBuffer, HostBuffer};

// ─── Device tag trait ────────────────────────────────────────

/// A device backend that owns a buffer type and knows how to run
/// ops on its native data layout. Implementors are zero-sized
/// marker types (`Cpu`, `Rocm`) used as a `PhantomData` parameter
/// on `Tensor`.
///
/// Methods on the trait are the **type-resolved kernels**: every
/// `D::matvec(...)` call routes directly to that device's native
/// implementation, with no runtime registry lookup. This is what
/// makes `Tensor<Cpu>` and `Tensor<Rocm>` strictly type-isolated.
pub trait Device: 'static + Send + Sync {
    /// The native buffer type for this device. Holds the raw storage
    /// (heap `Vec<f32>` for CPU, `HipBuffer` for ROCm, ...).
    type Buffer: DeviceBuffer;

    /// Short device name, matches `Backend::name()` for the
    /// equivalent runtime backend. Used in errors and telemetry.
    const NAME: &'static str;

    /// Allocate `n` zeroed f32s on this device.
    fn zeros(n: usize) -> Result<Self::Buffer, BackendError>;

    /// Upload host data to a fresh device buffer.
    fn from_host(xs: &[f32]) -> Result<Self::Buffer, BackendError>;

    /// `y = W·x + b` where W is row-major `[out_dim × in_dim]`.
    /// Caller must size buffers correctly.
    fn matvec(
        x: &Self::Buffer,
        weight: &Self::Buffer,
        bias: &Self::Buffer,
        out: &mut Self::Buffer,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<(), BackendError>;

    /// `C = A @ B`. `a` is `[m × k]` row-major, `b` is `[k × n]`,
    /// `out` is `[m × n]`. No bias: resident matmul has no bias
    /// argument; for Tensor parity, callers add bias via a separate
    /// op (cheap on either device).
    fn matmul_nn(
        a: &Self::Buffer,
        b: &Self::Buffer,
        out: &mut Self::Buffer,
        m: usize, k: usize, n: usize,
    ) -> Result<(), BackendError>;

    /// `C = A @ B^T`. `a` is `[m × k]` row-major, `b` is `[n × k]`
    /// row-major (transposed on the fly), `out` is `[m × n]`.
    fn matmul_nt(
        a: &Self::Buffer,
        b: &Self::Buffer,
        out: &mut Self::Buffer,
        m: usize, k: usize, n: usize,
    ) -> Result<(), BackendError>;

    /// `C = A^T @ B`. `a` is `[k × m]` row-major (transposed on the
    /// fly), `b` is `[k × n]` row-major, `out` is `[m × n]`.
    fn matmul_tn(
        a: &Self::Buffer,
        b: &Self::Buffer,
        out: &mut Self::Buffer,
        m: usize, k: usize, n: usize,
    ) -> Result<(), BackendError>;

    /// Per-neuron parallel matvec (SuperLinear forward).
    /// `out[n] = W[n]·x[n] + b[n]` for each `n` in `0..n_neurons`.
    /// Layout (all flat, row-major per neuron):
    ///   `x: [n_neurons × in_per]`,
    ///   `weight: [n_neurons × out_per × in_per]`,
    ///   `bias: [n_neurons × out_per]`,
    ///   `out: [n_neurons × out_per]`.
    fn super_linear_fwd(
        x: &Self::Buffer,
        weight: &Self::Buffer,
        bias: &Self::Buffer,
        out: &mut Self::Buffer,
        n_neurons: usize, in_per: usize, out_per: usize,
    ) -> Result<(), BackendError>;

    /// Row-wise layer normalisation: `y = γ ⊙ (x - μ)/√(σ² + ε) + β`.
    /// `x` and `out` are `[n_rows × n_cols]`; `gamma` and `beta` are
    /// `[n_cols]` (broadcast across rows). Epsilon is pinned at
    /// `1e-5` to match the CPU backend's hardcoded value (cpu.rs:430)
    /// — keeps cross-device parity bit-equal.
    fn layer_norm(
        x: &Self::Buffer,
        gamma: &Self::Buffer,
        beta: &Self::Buffer,
        out: &mut Self::Buffer,
        n_rows: usize, n_cols: usize,
    ) -> Result<(), BackendError>;

    /// Element-wise SiLU (sigmoid-linear unit): `y = x · σ(x)`.
    /// `x` and `out` are `[n]`. The CTM's go-to nonlinearity.
    fn silu(
        x: &Self::Buffer,
        out: &mut Self::Buffer,
    ) -> Result<(), BackendError>;

    /// Row-wise softmax: `y[r,j] = exp(x[r,j] - max(x[r,*])) / Σ`.
    /// Numerically stable (subtract per-row max). Pass `log=true`
    /// for log-softmax instead. Shapes:
    ///   `x: [n_rows × row_len]`, `out: [n_rows × row_len]`.
    fn softmax(
        x: &Self::Buffer,
        out: &mut Self::Buffer,
        n_rows: usize, row_len: usize,
        log: bool,
    ) -> Result<(), BackendError>;

    /// GLU (Gated Linear Unit): split `x: [n_rows × 2·half]` into
    /// `(value, gate)` halves, output `value ⊙ σ(gate)`.
    /// Output shape: `[n_rows × half]`.
    fn glu(
        x: &Self::Buffer,
        out: &mut Self::Buffer,
        n_rows: usize, half_size: usize,
    ) -> Result<(), BackendError>;

    /// Single-row embedding lookup (gather). Copy row `token_idx`
    /// from `table` (`[vocab_size × embed_dim]`) into `out`
    /// (`[embed_dim]`). Stays on device — for `Rocm`, this is a
    /// hipMemcpy D2D, no PCIe round-trip.
    fn embedding_lookup(
        table: &Self::Buffer,
        token_idx: usize,
        out: &mut Self::Buffer,
        embed_dim: usize,
    ) -> Result<(), BackendError>;

    /// Transposed matvec: `d_input = W^T · d_out`. The first
    /// backward primitive on the cascade — used by `Linear<D>::backward`
    /// to flow gradient through input. Layout matches `matvec`:
    /// `weight` is row-major `[out_dim × in_dim]`, `d_out` is
    /// `[out_dim]`, `d_input` is `[in_dim]`.
    ///
    /// **Output convention: OVERWRITE.** `d_input` is replaced;
    /// pre-existing contents are discarded. Use `add_assign` after
    /// the call if accumulation is needed.
    fn matvec_t(
        d_out: &Self::Buffer,
        weight: &Self::Buffer,
        d_input: &mut Self::Buffer,
        out_dim: usize, in_dim: usize,
    ) -> Result<(), BackendError>;

    /// Accumulating outer product: `accum += a ⊗ b`. `a` is `[m]`,
    /// `b` is `[n]`, `accum` is row-major `[m × n]`. Used by
    /// `Linear<D>::backward` for `d_W += d_y ⊗ x`.
    ///
    /// **Output convention: ACCUMULATE.** `accum` is read AND written;
    /// caller must zero it before the first sample of a fresh batch.
    /// Suffix `_acc` is the convention marker for accumulating ops.
    fn outer_product_acc(
        a: &Self::Buffer,
        b: &Self::Buffer,
        accum: &mut Self::Buffer,
        m: usize, n: usize,
    ) -> Result<(), BackendError>;

    /// Element-wise add (accumulate): `dst += src`. Used by
    /// `Linear<D>::backward` for `d_b += d_y` and gradient accumulation
    /// across batch.
    ///
    /// **Output convention: ACCUMULATE.** `dst` is read AND written;
    /// `dst` and `src` must not alias. Implemented via `axpy(α=1)`.
    fn add_assign(
        dst: &mut Self::Buffer,
        src: &Self::Buffer,
        n: usize,
    ) -> Result<(), BackendError>;

    /// SGD step: `w -= lr · g`. The minimal optimiser, used as the
    /// integration smoke until `AdamW<D>` lands.
    fn sgd_step(
        w: &mut Self::Buffer,
        g: &Self::Buffer,
        n: usize,
        lr: f32,
    ) -> Result<(), BackendError>;

    /// SuperLinear backward — gradient w.r.t. weights.
    /// `d_weights[n] += d_out[n] ⊗ trace[n]` for each neuron.
    ///
    /// **Output convention: ACCUMULATE.** `d_weights` is read AND
    /// written. Caller must zero before the first sample of a batch.
    /// (Matches the resident kernel's behaviour and the `_acc`
    /// convention used elsewhere — this fn keeps the legacy name for
    /// back-compat; treat as if it ended in `_acc`.)
    fn super_linear_bwd_dw(
        d_out: &Self::Buffer,
        trace: &Self::Buffer,
        d_weights: &mut Self::Buffer,
        n_neurons: usize, in_per: usize, out_per: usize,
    ) -> Result<(), BackendError>;

    /// SuperLinear backward — gradient w.r.t. input trace.
    /// `d_trace[n] = W[n]^T · d_out[n]`.
    ///
    /// **Output convention: OVERWRITE.** `d_trace` is replaced.
    fn super_linear_bwd_dx(
        d_out: &Self::Buffer,
        weights: &Self::Buffer,
        d_trace: &mut Self::Buffer,
        n_neurons: usize, in_per: usize, out_per: usize,
    ) -> Result<(), BackendError>;

    /// SiLU backward: `d_x = d_out · σ(x) · (1 + x · (1 − σ(x)))`.
    /// Stateless — reads `x` (saved from forward) plus upstream `d_out`.
    ///
    /// **Output convention: OVERWRITE.** `d_x` is replaced.
    fn silu_bwd(
        d_out: &Self::Buffer,
        x: &Self::Buffer,
        d_x: &mut Self::Buffer,
    ) -> Result<(), BackendError>;

    /// GLU backward. `x` is `[n_rows × 2·half]` (concat of value and
    /// gate halves saved at forward); `d_out` is `[n_rows × half]`;
    /// `d_x` is `[n_rows × 2·half]` (matches x layout). Computes:
    ///   `d_value = d_out · σ(gate)`
    ///   `d_gate  = d_out · value · σ(gate) · (1 − σ(gate))`
    ///
    /// **Output convention: OVERWRITE.** `d_x` is replaced.
    fn glu_bwd(
        d_out: &Self::Buffer,
        x: &Self::Buffer,
        d_x: &mut Self::Buffer,
        n_rows: usize, half_size: usize,
    ) -> Result<(), BackendError>;

    /// LayerNorm forward producing the cache needed for backward.
    /// Cache layout: `[2 * n_rows]` = `mean[n_rows]` then `rstd[n_rows]`.
    /// Backends that don't support the cache form must error.
    fn layer_norm_train(
        x: &Self::Buffer,
        gamma: &Self::Buffer,
        beta: &Self::Buffer,
        out: &mut Self::Buffer,
        cache: &mut Self::Buffer,
        n_rows: usize, n_cols: usize,
    ) -> Result<(), BackendError>;

    /// LayerNorm backward using the cache from `layer_norm_train`.
    ///
    /// **Mixed output convention** (audit this carefully when wiring):
    ///   - `d_x` is **overwritten** (per-row formula, no carry).
    ///   - `d_gamma` is **accumulated** (sums across rows; caller
    ///     must zero before a fresh batch).
    ///   - `d_beta` is **accumulated** (same).
    fn layer_norm_bwd(
        d_out: &Self::Buffer,
        x: &Self::Buffer,
        gamma: &Self::Buffer,
        cache: &Self::Buffer,
        d_x: &mut Self::Buffer,
        d_gamma: &mut Self::Buffer,
        d_beta: &mut Self::Buffer,
        n_rows: usize, n_cols: usize,
    ) -> Result<(), BackendError>;

    /// AdamW step: `w` is updated in place using `g` and the
    /// optimiser moments `m`, `v` (also updated in place).
    /// `bc1_inv = 1 / (1 - beta1^t)`, `bc2_inv = 1 / (1 - beta2^t)`.
    #[allow(clippy::too_many_arguments)]
    fn adamw_step(
        w: &mut Self::Buffer,
        g: &Self::Buffer,
        m: &mut Self::Buffer,
        v: &mut Self::Buffer,
        n: usize,
        lr: f32, beta1: f32, beta2: f32, eps: f32,
        weight_decay: f32,
        bc1_inv: f32, bc2_inv: f32,
    ) -> Result<(), BackendError>;
}

// ─── CPU device ──────────────────────────────────────────────

/// CPU device tag. `Tensor<Cpu>` wraps a `HostBuffer`; ops use
/// rayon-parallel routines from this module.
#[derive(Debug, Default, Clone, Copy)]
pub struct Cpu;

impl Device for Cpu {
    type Buffer = HostBuffer;
    const NAME: &'static str = "cpu";

    fn zeros(n: usize) -> Result<HostBuffer, BackendError> {
        Ok(HostBuffer::new(n))
    }

    fn from_host(xs: &[f32]) -> Result<HostBuffer, BackendError> {
        let mut buf = HostBuffer::new(xs.len());
        buf.copy_from_host(xs)?;
        Ok(buf)
    }

    fn matvec(
        x: &HostBuffer,
        weight: &HostBuffer,
        bias: &HostBuffer,
        out: &mut HostBuffer,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<(), BackendError> {
        // `HostBuffer` exposes `as_slice` / `as_mut_slice`; we use the
        // existing rayon-parallel matvec from the CPU backend by
        // inlining it here. Keeping the impl in one place per kernel
        // (cpu.rs vs tensor.rs) is a minor v0 duplication that the
        // cascade later removes.
        cpu_matvec(
            x.as_slice(),
            weight.as_slice(),
            bias.as_slice(),
            out.as_mut_slice(),
            out_dim,
            in_dim,
        );
        Ok(())
    }

    fn matmul_nn(
        a: &HostBuffer, b: &HostBuffer, out: &mut HostBuffer,
        m: usize, k: usize, n: usize,
    ) -> Result<(), BackendError> {
        // For matmul we route through `CpuBackend::dispatch` directly
        // rather than the registry. Going through `super::ops::matmul_nn`
        // would consult the registry priority list and could pick ROCm
        // (with PCIe bouncing) on hosts where ROCm is available — that
        // hidden host↔device round-trip is exactly what Path C forbids.
        // Constructing a `CpuBackend` per call is cheap (one
        // `rayon::current_num_threads` lookup).
        use super::Backend;
        let mut op = super::Op::MatmulNN {
            a: a.as_slice(), b: b.as_slice(), out: out.as_mut_slice(),
            bias: None, m, k, n,
        };
        super::CpuBackend::new().dispatch(&mut op)
    }

    fn matmul_nt(
        a: &HostBuffer, b: &HostBuffer, out: &mut HostBuffer,
        m: usize, k: usize, n: usize,
    ) -> Result<(), BackendError> {
        use super::Backend;
        let mut op = super::Op::MatmulNT {
            a: a.as_slice(), b: b.as_slice(), out: out.as_mut_slice(),
            bias: None, m, k, n,
        };
        super::CpuBackend::new().dispatch(&mut op)
    }

    fn matmul_tn(
        a: &HostBuffer, b: &HostBuffer, out: &mut HostBuffer,
        m: usize, k: usize, n: usize,
    ) -> Result<(), BackendError> {
        use super::Backend;
        let mut op = super::Op::MatmulTN {
            a: a.as_slice(), b: b.as_slice(), out: out.as_mut_slice(),
            bias: None, m, k, n,
        };
        super::CpuBackend::new().dispatch(&mut op)
    }

    fn super_linear_fwd(
        x: &HostBuffer, weight: &HostBuffer, bias: &HostBuffer,
        out: &mut HostBuffer,
        n_neurons: usize, in_per: usize, out_per: usize,
    ) -> Result<(), BackendError> {
        use super::Backend;
        // CPU dispatch directly, bypassing registry (same rationale
        // as matmul_nn: avoid hidden routing to a GPU backend).
        let mut op = super::Op::SuperLinearFwd {
            trace: x.as_slice(),
            weights: weight.as_slice(),
            biases: bias.as_slice(),
            out: out.as_mut_slice(),
            cache: None,
            d_model: n_neurons,
            memory_length: in_per,
            out_per,
        };
        super::CpuBackend::new().dispatch(&mut op)
    }

    fn layer_norm(
        x: &HostBuffer, gamma: &HostBuffer, beta: &HostBuffer,
        out: &mut HostBuffer,
        n_rows: usize, n_cols: usize,
    ) -> Result<(), BackendError> {
        use super::Backend;
        let mut op = super::Op::LayerNormFwd {
            x: x.as_slice(),
            gamma: gamma.as_slice(),
            beta: beta.as_slice(),
            out: out.as_mut_slice(),
            cache: None,
            n_rows, n_cols,
        };
        super::CpuBackend::new().dispatch(&mut op)
    }

    fn silu(
        x: &HostBuffer, out: &mut HostBuffer,
    ) -> Result<(), BackendError> {
        use super::Backend;
        let mut op = super::Op::SiluFwd {
            x: x.as_slice(),
            out: out.as_mut_slice(),
        };
        super::CpuBackend::new().dispatch(&mut op)
    }

    fn softmax(
        x: &HostBuffer, out: &mut HostBuffer,
        n_rows: usize, row_len: usize,
        log: bool,
    ) -> Result<(), BackendError> {
        // No host-slice softmax exists in the Op enum (CTM's MHA
        // does this inline). Implement it directly here — same
        // numerically-stable algorithm the resident kernel uses.
        // When the cascade ports CtmWeights and the inline softmax
        // disappears, this stays as the canonical CPU softmax.
        let xs = x.as_slice();
        let ys = out.as_mut_slice();
        if xs.len() < n_rows * row_len || ys.len() < n_rows * row_len {
            return Err(BackendError::Runtime(format!(
                "Cpu::softmax: buffer shorter than n_rows*row_len = {}*{}",
                n_rows, row_len,
            )));
        }
        for r in 0..n_rows {
            let row = &xs[r * row_len..(r + 1) * row_len];
            let out_row = &mut ys[r * row_len..(r + 1) * row_len];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            for (i, &v) in row.iter().enumerate() {
                let e = (v - max).exp();
                out_row[i] = e;
                sum_exp += e;
            }
            if log {
                let log_sum = sum_exp.ln() + max;
                for (i, &v) in row.iter().enumerate() {
                    out_row[i] = v - log_sum;
                }
            } else {
                let inv = 1.0 / sum_exp.max(1e-30);
                for v in out_row.iter_mut() { *v *= inv; }
            }
        }
        Ok(())
    }

    fn glu(
        x: &HostBuffer, out: &mut HostBuffer,
        _n_rows: usize, _half_size: usize,
    ) -> Result<(), BackendError> {
        use super::Backend;
        // `Op::GluFwd` only takes (x, out). Caller's _n_rows /
        // _half_size are inferred from buffer shape; the CPU kernel
        // already validates internally. Path-C signature still asks
        // for them so Rocm path can pass them along.
        let mut op = super::Op::GluFwd {
            x: x.as_slice(),
            out: out.as_mut_slice(),
        };
        super::CpuBackend::new().dispatch(&mut op)
    }

    fn embedding_lookup(
        table: &HostBuffer,
        token_idx: usize,
        out: &mut HostBuffer,
        embed_dim: usize,
    ) -> Result<(), BackendError> {
        let src_off = token_idx * embed_dim;
        let table = table.as_slice();
        let dst = out.as_mut_slice();
        if src_off + embed_dim > table.len() {
            return Err(BackendError::Runtime(format!(
                "embedding_lookup: token {} (offset {}, dim {}) exceeds table len {}",
                token_idx, src_off, embed_dim, table.len(),
            )));
        }
        if dst.len() < embed_dim {
            return Err(BackendError::Runtime(format!(
                "embedding_lookup: dst len {} < embed_dim {}",
                dst.len(), embed_dim,
            )));
        }
        dst[..embed_dim].copy_from_slice(&table[src_off..src_off + embed_dim]);
        Ok(())
    }

    fn matvec_t(
        d_out: &HostBuffer, weight: &HostBuffer, d_input: &mut HostBuffer,
        out_dim: usize, in_dim: usize,
    ) -> Result<(), BackendError> {
        use super::Backend;
        let mut op = super::Op::MatvecT {
            d_out: d_out.as_slice(),
            weight: weight.as_slice(),
            d_input: d_input.as_mut_slice(),
            out_dim, in_dim,
        };
        super::CpuBackend::new().dispatch(&mut op)
    }

    fn outer_product_acc(
        a: &HostBuffer, b: &HostBuffer, accum: &mut HostBuffer,
        m: usize, n: usize,
    ) -> Result<(), BackendError> {
        use super::Backend;
        let mut op = super::Op::OuterProductAcc {
            a: a.as_slice(),
            b: b.as_slice(),
            accum: accum.as_mut_slice(),
            m, n,
        };
        super::CpuBackend::new().dispatch(&mut op)
    }

    fn add_assign(
        dst: &mut HostBuffer, src: &HostBuffer, n: usize,
    ) -> Result<(), BackendError> {
        let s = src.as_slice();
        let d = dst.as_mut_slice();
        if s.len() < n || d.len() < n {
            return Err(BackendError::Runtime(format!(
                "Cpu::add_assign: buffer shorter than n={}", n,
            )));
        }
        for i in 0..n { d[i] += s[i]; }
        Ok(())
    }

    fn sgd_step(
        w: &mut HostBuffer, g: &HostBuffer, n: usize, lr: f32,
    ) -> Result<(), BackendError> {
        use super::Backend;
        let mut op = super::Op::SgdUpdate {
            w: &mut w.as_mut_slice()[..n],
            g: &g.as_slice()[..n],
            lr,
        };
        super::CpuBackend::new().dispatch(&mut op)
    }

    fn super_linear_bwd_dw(
        d_out: &HostBuffer, trace: &HostBuffer, d_weights: &mut HostBuffer,
        n_neurons: usize, in_per: usize, out_per: usize,
    ) -> Result<(), BackendError> {
        use super::Backend;
        let mut op = super::Op::SuperLinearBwdDw {
            d_out: d_out.as_slice(),
            trace: trace.as_slice(),
            d_weights: d_weights.as_mut_slice(),
            // CPU dispatch uses (d_model, memory_length, out_per) field
            // names but the semantics are (n_neurons, in_per, out_per).
            d_model: n_neurons,
            memory_length: in_per,
            out_per,
        };
        super::CpuBackend::new().dispatch(&mut op)
    }

    fn super_linear_bwd_dx(
        d_out: &HostBuffer, weights: &HostBuffer, d_trace: &mut HostBuffer,
        n_neurons: usize, in_per: usize, out_per: usize,
    ) -> Result<(), BackendError> {
        use super::Backend;
        let mut op = super::Op::SuperLinearBwdDx {
            d_out: d_out.as_slice(),
            weights: weights.as_slice(),
            d_trace: d_trace.as_mut_slice(),
            d_model: n_neurons,
            memory_length: in_per,
            out_per,
        };
        super::CpuBackend::new().dispatch(&mut op)
    }

    fn silu_bwd(
        d_out: &HostBuffer, x: &HostBuffer, d_x: &mut HostBuffer,
    ) -> Result<(), BackendError> {
        use super::Backend;
        let mut op = super::Op::SiluBwd {
            d_out: d_out.as_slice(),
            x: x.as_slice(),
            d_x: d_x.as_mut_slice(),
        };
        super::CpuBackend::new().dispatch(&mut op)
    }

    fn glu_bwd(
        d_out: &HostBuffer, x: &HostBuffer, d_x: &mut HostBuffer,
        n_rows: usize, half_size: usize,
    ) -> Result<(), BackendError> {
        use super::Backend;
        // Per-neuron GLU: each row of `x` is [half_size value | half_size gate].
        // The flat `Op::GluBwd` treats the whole buffer as ONE GLU (half = len/2),
        // which pairs neuron 0's values with neuron N/2's row as its gate and
        // scrambles the value/gate split across neurons. Dispatch the per-neuron
        // variant (the ROCm impl already does). Matches the per-neuron `glu` fwd.
        let mut op = super::Op::PerNeuronGluBwd {
            d_out: d_out.as_slice(),
            x: x.as_slice(),
            d_x: d_x.as_mut_slice(),
            n_neurons: n_rows,
            feat_per_neuron: half_size,
        };
        super::CpuBackend::new().dispatch(&mut op)
    }

    fn layer_norm_train(
        x: &HostBuffer, gamma: &HostBuffer, beta: &HostBuffer,
        out: &mut HostBuffer, cache: &mut HostBuffer,
        n_rows: usize, n_cols: usize,
    ) -> Result<(), BackendError> {
        use super::Backend;
        let mut op = super::Op::LayerNormFwd {
            x: x.as_slice(),
            gamma: gamma.as_slice(),
            beta: beta.as_slice(),
            out: out.as_mut_slice(),
            cache: Some(&mut cache.as_mut_slice()[..2 * n_rows]),
            n_rows, n_cols,
        };
        super::CpuBackend::new().dispatch(&mut op)
    }

    fn layer_norm_bwd(
        d_out: &HostBuffer, x: &HostBuffer, gamma: &HostBuffer, cache: &HostBuffer,
        d_x: &mut HostBuffer, d_gamma: &mut HostBuffer, d_beta: &mut HostBuffer,
        n_rows: usize, n_cols: usize,
    ) -> Result<(), BackendError> {
        use super::Backend;
        let mut op = super::Op::LayerNormBwd {
            d_out: d_out.as_slice(),
            x: x.as_slice(),
            gamma: gamma.as_slice(),
            cache: &cache.as_slice()[..2 * n_rows],
            d_x: d_x.as_mut_slice(),
            d_gamma: d_gamma.as_mut_slice(),
            d_beta: d_beta.as_mut_slice(),
            n_rows, n_cols,
        };
        super::CpuBackend::new().dispatch(&mut op)
    }

    fn adamw_step(
        w: &mut HostBuffer, g: &HostBuffer,
        m: &mut HostBuffer, v: &mut HostBuffer,
        n: usize,
        lr: f32, beta1: f32, beta2: f32, eps: f32,
        weight_decay: f32, bc1_inv: f32, bc2_inv: f32,
    ) -> Result<(), BackendError> {
        use super::Backend;
        let mut op = super::Op::AdamW(super::AdamWArgs {
            w: &mut w.as_mut_slice()[..n],
            g: &g.as_slice()[..n],
            m: &mut m.as_mut_slice()[..n],
            v: &mut v.as_mut_slice()[..n],
            lr, beta1, beta2, eps, weight_decay, bc1_inv, bc2_inv,
        });
        super::CpuBackend::new().dispatch(&mut op)
    }
}

/// Host-side LayerNorm backward formula, kept here so the Rocm path
/// can call it as a correctness-first fallback while MIOpen
/// `layer_norm_backward` parity is still being chased. Matches
/// `cpu.rs::layer_norm_bwd` line-for-line. Move to a shared helper
/// once MIOpen parity lands.
#[cfg(feature = "rocm")]
fn host_layer_norm_bwd(
    d_out: &[f32], x: &[f32], gamma: &[f32], cache: &[f32],
    d_x: &mut [f32], d_gamma: &mut [f32], d_beta: &mut [f32],
    n_rows: usize, n_cols: usize,
) {
    for v in d_gamma.iter_mut() { *v = 0.0; }
    for v in d_beta.iter_mut() { *v = 0.0; }
    for r in 0..n_rows {
        let mean = cache[r * 2];
        let rstd = cache[r * 2 + 1];
        let row = &x[r * n_cols..(r + 1) * n_cols];
        let d_row_in = &d_out[r * n_cols..(r + 1) * n_cols];
        let d_row_out = &mut d_x[r * n_cols..(r + 1) * n_cols];
        for c in 0..n_cols {
            let x_hat = (row[c] - mean) * rstd;
            d_gamma[c] += d_row_in[c] * x_hat;
            d_beta[c] += d_row_in[c];
        }
        let n = n_cols as f32;
        let sum_d: f32 = (0..n_cols).map(|c| d_row_in[c] * gamma[c]).sum();
        let sum_dx: f32 = (0..n_cols).map(|c| {
            let x_hat = (row[c] - mean) * rstd;
            d_row_in[c] * gamma[c] * x_hat
        }).sum();
        for c in 0..n_cols {
            let x_hat = (row[c] - mean) * rstd;
            d_row_out[c] = gamma[c] * rstd * (d_row_in[c] - (sum_d + x_hat * sum_dx) / n);
        }
    }
}

/// Inline rayon matvec, mirrors `cpu.rs::matvec` (private there).
/// Same threshold, same algorithm. When the cascade ports `Linear`
/// to `Tensor`, this duplication will collapse — only one matvec
/// stays.
fn cpu_matvec(
    x: &[f32], weight: &[f32], bias: &[f32], out: &mut [f32],
    _out_dim: usize, in_dim: usize,
) {
    use rayon::prelude::*;
    const PAR_THRESHOLD: usize = 16_384;
    let n_rows = out.len();
    if n_rows * in_dim < PAR_THRESHOLD {
        for i in 0..n_rows {
            let row = &weight[i * in_dim..(i + 1) * in_dim];
            let mut acc = bias[i];
            for j in 0..in_dim { acc += row[j] * x[j]; }
            out[i] = acc;
        }
    } else {
        out.par_iter_mut().enumerate().for_each(|(i, y)| {
            let row = &weight[i * in_dim..(i + 1) * in_dim];
            let mut acc = bias[i];
            for j in 0..in_dim { acc += row[j] * x[j]; }
            *y = acc;
        });
    }
}

// ─── ROCm device (feature-gated) ─────────────────────────────

#[cfg(feature = "rocm")]
mod rocm_impl {
    use super::*;
    use super::super::rocm::HipBuffer;

    // The existing `RocmBuffer` (rocm.rs) is a thin wrapper around
    // `HipBuffer` that implements `DeviceBuffer`. We want
    // `Tensor<Rocm>` to use `HipBuffer` directly — that's what the
    // rest of the resident codebase (`RegionalResidentCache`,
    // `RegionalGradientsResident`) holds. Adding the trait impl to
    // `HipBuffer` removes the wrapper indirection and unifies the
    // resident-path types under the device abstraction.
    impl DeviceBuffer for HipBuffer {
        fn backend_name(&self) -> &'static str { "rocm" }

        fn len(&self) -> usize { self.len_f32() }

        fn copy_from_host(&mut self, src: &[f32]) -> Result<(), BackendError> {
            // `HipBuffer::copy_from_host` exists as a public alias of
            // `upload_f32`; it takes `&self` because hipMemcpy only
            // touches device memory (no Rust-visible mutation). We
            // re-declare with `&mut self` here to satisfy the trait
            // signature; the underlying call is the same.
            HipBuffer::copy_from_host(&*self, src)
        }

        fn copy_to_host(&self, dst: &mut [f32]) -> Result<(), BackendError> {
            HipBuffer::copy_to_host(self, dst)
        }
    }

    /// ROCm device tag. `Tensor<Rocm>` wraps a `HipBuffer`; ops
    /// route through `hipblasSgemv` via the resident pointer path
    /// — zero PCIe round-trips per op.
    #[derive(Debug, Default, Clone, Copy)]
    pub struct Rocm;

    impl Device for Rocm {
        type Buffer = HipBuffer;
        const NAME: &'static str = "rocm";

        fn zeros(n: usize) -> Result<HipBuffer, BackendError> {
            // `HipBuffer::new` allocates uninitialised; zero-fill
            // via a host upload of zeros. Cheap for v0; a future
            // commit can plumb hipMemset for a single-launch zero.
            let buf = HipBuffer::new(n * 4)?;
            let zeros = vec![0.0f32; n];
            buf.copy_from_host(&zeros)?;
            Ok(buf)
        }

        fn from_host(xs: &[f32]) -> Result<HipBuffer, BackendError> {
            let buf = HipBuffer::new(xs.len() * 4)?;
            buf.copy_from_host(xs)?;
            Ok(buf)
        }

        fn matvec(
            x: &HipBuffer,
            weight: &HipBuffer,
            bias: &HipBuffer,
            out: &mut HipBuffer,
            out_dim: usize,
            in_dim: usize,
        ) -> Result<(), BackendError> {
            // Route through the resident-matvec façade so the
            // existing `Op::MatvecResident` registry path handles
            // dispatch. Pointers come from owned `HipBuffer`s
            // which outlive this call (we hold borrows).
            unsafe {
                super::super::ops::matvec_resident(
                    x.device_ptr() as *const f32,
                    weight.device_ptr() as *const f32,
                    bias.device_ptr() as *const f32,
                    out.device_ptr() as *mut f32,
                    out_dim,
                    in_dim,
                )
            }
        }

        fn matmul_nn(
            a: &HipBuffer, b: &HipBuffer, out: &mut HipBuffer,
            m: usize, k: usize, n: usize,
        ) -> Result<(), BackendError> {
            unsafe {
                super::super::ops::matmul_resident_nn(
                    a.device_ptr() as *const f32,
                    b.device_ptr() as *const f32,
                    out.device_ptr() as *mut f32,
                    m, k, n,
                )
            }
        }

        fn matmul_nt(
            a: &HipBuffer, b: &HipBuffer, out: &mut HipBuffer,
            m: usize, k: usize, n: usize,
        ) -> Result<(), BackendError> {
            unsafe {
                super::super::ops::matmul_resident_nt(
                    a.device_ptr() as *const f32,
                    b.device_ptr() as *const f32,
                    out.device_ptr() as *mut f32,
                    m, k, n,
                )
            }
        }

        fn matmul_tn(
            a: &HipBuffer, b: &HipBuffer, out: &mut HipBuffer,
            m: usize, k: usize, n: usize,
        ) -> Result<(), BackendError> {
            unsafe {
                super::super::ops::matmul_resident_tn(
                    a.device_ptr() as *const f32,
                    b.device_ptr() as *const f32,
                    out.device_ptr() as *mut f32,
                    m, k, n,
                )
            }
        }

        fn super_linear_fwd(
            x: &HipBuffer, weight: &HipBuffer, bias: &HipBuffer,
            out: &mut HipBuffer,
            n_neurons: usize, in_per: usize, out_per: usize,
        ) -> Result<(), BackendError> {
            unsafe {
                super::super::ops::super_linear_fwd_resident(
                    x.device_ptr() as *const f32,
                    weight.device_ptr() as *const f32,
                    bias.device_ptr() as *const f32,
                    out.device_ptr() as *mut f32,
                    n_neurons, in_per, out_per,
                )
            }
        }

        fn layer_norm(
            x: &HipBuffer, gamma: &HipBuffer, beta: &HipBuffer,
            out: &mut HipBuffer,
            n_rows: usize, n_cols: usize,
        ) -> Result<(), BackendError> {
            // Match CPU's hardcoded ε = 1e-5 (cpu.rs:430).
            unsafe {
                super::super::ops::layer_norm_resident(
                    x.device_ptr() as *const f32,
                    gamma.device_ptr() as *const f32,
                    beta.device_ptr() as *const f32,
                    out.device_ptr() as *mut f32,
                    n_rows, n_cols, 1e-5,
                )
            }
        }

        fn silu(
            x: &HipBuffer, out: &mut HipBuffer,
        ) -> Result<(), BackendError> {
            unsafe {
                super::super::ops::activation_resident(
                    x.device_ptr() as *const f32,
                    out.device_ptr() as *mut f32,
                    x.len(),
                    super::super::ActivationMode::Silu,
                )
            }
        }

        fn softmax(
            x: &HipBuffer, out: &mut HipBuffer,
            n_rows: usize, row_len: usize,
            log: bool,
        ) -> Result<(), BackendError> {
            unsafe {
                super::super::ops::softmax_resident(
                    x.device_ptr() as *const f32,
                    out.device_ptr() as *mut f32,
                    n_rows, row_len, log,
                )
            }
        }

        fn glu(
            x: &HipBuffer, out: &mut HipBuffer,
            n_rows: usize, half_size: usize,
        ) -> Result<(), BackendError> {
            unsafe {
                super::super::ops::glu_resident(
                    x.device_ptr() as *const f32,
                    out.device_ptr() as *mut f32,
                    n_rows, half_size,
                )
            }
        }

        fn embedding_lookup(
            table: &HipBuffer,
            token_idx: usize,
            out: &mut HipBuffer,
            embed_dim: usize,
        ) -> Result<(), BackendError> {
            // hipMemcpy D2D — table[token*embed_dim..(token+1)*embed_dim] → out[..embed_dim]
            out.copy_range_from(table, token_idx * embed_dim, 0, embed_dim)
        }

        fn matvec_t(
            d_out: &HipBuffer, weight: &HipBuffer, d_input: &mut HipBuffer,
            out_dim: usize, in_dim: usize,
        ) -> Result<(), BackendError> {
            unsafe {
                super::super::ops::matvec_t_resident(
                    d_out.device_ptr() as *const f32,
                    weight.device_ptr() as *const f32,
                    d_input.device_ptr() as *mut f32,
                    out_dim, in_dim,
                )
            }
        }

        fn outer_product_acc(
            a: &HipBuffer, b: &HipBuffer, accum: &mut HipBuffer,
            m: usize, n: usize,
        ) -> Result<(), BackendError> {
            unsafe {
                super::super::ops::outer_product_acc_resident(
                    a.device_ptr() as *const f32,
                    b.device_ptr() as *const f32,
                    accum.device_ptr() as *mut f32,
                    m, n,
                )
            }
        }

        fn add_assign(
            dst: &mut HipBuffer, src: &HipBuffer, n: usize,
        ) -> Result<(), BackendError> {
            // `dst += src` is `axpy(alpha=1, x=src, y=dst)`. Routes
            // through the proper `axpy_resident` primitive instead of
            // the historical `sgd_update_resident` with `lr=-1` hack.
            unsafe {
                super::super::ops::axpy_resident(
                    dst.device_ptr() as *mut f32,
                    src.device_ptr() as *const f32,
                    n,
                    1.0,
                )
            }
        }

        fn sgd_step(
            w: &mut HipBuffer, g: &HipBuffer, n: usize, lr: f32,
        ) -> Result<(), BackendError> {
            unsafe {
                super::super::ops::sgd_update_resident(
                    w.device_ptr() as *mut f32,
                    g.device_ptr() as *const f32,
                    n, lr,
                )
            }
        }

        fn super_linear_bwd_dw(
            d_out: &HipBuffer, trace: &HipBuffer, d_weights: &mut HipBuffer,
            n_neurons: usize, in_per: usize, out_per: usize,
        ) -> Result<(), BackendError> {
            unsafe {
                super::super::ops::super_linear_bwd_dw_resident(
                    d_out.device_ptr() as *const f32,
                    trace.device_ptr() as *const f32,
                    d_weights.device_ptr() as *mut f32,
                    n_neurons, in_per, out_per,
                )
            }
        }

        fn super_linear_bwd_dx(
            d_out: &HipBuffer, weights: &HipBuffer, d_trace: &mut HipBuffer,
            n_neurons: usize, in_per: usize, out_per: usize,
        ) -> Result<(), BackendError> {
            unsafe {
                super::super::ops::super_linear_bwd_dx_resident(
                    d_out.device_ptr() as *const f32,
                    weights.device_ptr() as *const f32,
                    d_trace.device_ptr() as *mut f32,
                    n_neurons, in_per, out_per,
                )
            }
        }

        fn glu_bwd(
            d_out: &HipBuffer, x: &HipBuffer, d_x: &mut HipBuffer,
            n_rows: usize, half_size: usize,
        ) -> Result<(), BackendError> {
            unsafe {
                super::super::ops::glu_backward_resident(
                    x.device_ptr() as *const f32,
                    d_out.device_ptr() as *const f32,
                    d_x.device_ptr() as *mut f32,
                    n_rows, half_size,
                )
            }
        }

        fn silu_bwd(
            d_out: &HipBuffer, x: &HipBuffer, d_x: &mut HipBuffer,
        ) -> Result<(), BackendError> {
            // Resident activation backward needs y = σ(x) · x — but
            // the SDK's ActivationBackwardResident takes y separately.
            // We forward through silu first to populate y, then call
            // backward. (Or: read existing forward output if we kept
            // it — caller does in production. v0 recomputes.)
            let n = x.len();
            let y_dev = HipBuffer::new(n * 4)?;
            // recompute y = silu(x)
            unsafe {
                super::super::ops::activation_resident(
                    x.device_ptr() as *const f32,
                    y_dev.device_ptr() as *mut f32,
                    n,
                    super::super::ActivationMode::Silu,
                )?;
                super::super::ops::activation_backward_resident(
                    x.device_ptr() as *const f32,
                    y_dev.device_ptr() as *const f32,
                    d_out.device_ptr() as *const f32,
                    d_x.device_ptr() as *mut f32,
                    n,
                    super::super::ActivationMode::Silu,
                )
            }
        }

        fn layer_norm_train(
            x: &HipBuffer, gamma: &HipBuffer, beta: &HipBuffer,
            out: &mut HipBuffer, cache: &mut HipBuffer,
            n_rows: usize, n_cols: usize,
        ) -> Result<(), BackendError> {
            // Run resident forward into `out`. Resident kernel doesn't
            // produce cache; we recompute mean/rstd on host as a cache
            // shim. This keeps the API parity-correct while avoiding
            // a brand-new kernel for a small bookkeeping op (cache is
            // O(n_rows), tiny vs the matmul work in the brain).
            unsafe {
                super::super::ops::layer_norm_resident(
                    x.device_ptr() as *const f32,
                    gamma.device_ptr() as *const f32,
                    beta.device_ptr() as *const f32,
                    out.device_ptr() as *mut f32,
                    n_rows, n_cols, 1e-5,
                )?;
            }
            // Compute cache (mean, rstd) on host. Cheap relative to
            // the GPU LayerNorm itself for typical CTM shapes. Layout
            // matches the CPU side: interleaved [m0, r0, m1, r1, ...]
            // — see cpu.rs::layer_norm_fwd.
            //
            // ⚠ HOST-FALLBACK: `miopenLayerNormForward` doesn't expose
            // the (mean, rstd) statistics it computes internally, so
            // we re-derive them on host. Costs one PCIe DtoH per call.
            // Tracked as a Path-C-policy violation; replace with
            // an op that returns cache when MIOpen exposes it (or
            // write a tiny custom kernel).
            let mut xv = vec![0.0f32; n_rows * n_cols];
            x.copy_to_host(&mut xv)?;
            let mut cache_h = vec![0.0f32; 2 * n_rows];
            const EPS: f32 = 1e-5;
            for r in 0..n_rows {
                let row = &xv[r * n_cols..(r + 1) * n_cols];
                let mean = row.iter().sum::<f32>() / n_cols as f32;
                let var = row.iter().map(|&v| (v - mean).powi(2))
                    .sum::<f32>() / n_cols as f32;
                let rstd = 1.0 / (var + EPS).sqrt();
                cache_h[r * 2] = mean;
                cache_h[r * 2 + 1] = rstd;
            }
            cache.copy_from_host(&cache_h)?;
            Ok(())
        }

        fn adamw_step(
            w: &mut HipBuffer, g: &HipBuffer,
            m: &mut HipBuffer, v: &mut HipBuffer,
            n: usize,
            lr: f32, beta1: f32, beta2: f32, eps: f32,
            weight_decay: f32, bc1_inv: f32, bc2_inv: f32,
        ) -> Result<(), BackendError> {
            unsafe {
                super::super::ops::adamw_resident(
                    w.device_ptr() as *mut f32,
                    g.device_ptr() as *const f32,
                    m.device_ptr() as *mut f32,
                    v.device_ptr() as *mut f32,
                    n, lr, beta1, beta2, eps, weight_decay, bc1_inv, bc2_inv,
                )
            }
        }

        fn layer_norm_bwd(
            d_out: &HipBuffer, x: &HipBuffer, gamma: &HipBuffer, cache: &HipBuffer,
            d_x: &mut HipBuffer, d_gamma: &mut HipBuffer, d_beta: &mut HipBuffer,
            n_rows: usize, n_cols: usize,
        ) -> Result<(), BackendError> {
            // ⚠ HOST-FALLBACK: MIOpen's `miopenLayerNormBackward`
            // produces results that disagree with the CPU formula —
            // likely a different formulation (undocumented mode flag
            // or weight axis). Until we resolve the MIOpen mismatch
            // (probably needs chasing through MIOpen source for the
            // exact PyTorch-compatible variant), fall back to host
            // computation: download x/gamma/cache/d_out, run the same
            // formula as `cpu.rs::layer_norm_bwd`, upload
            // d_x/d_gamma/d_beta. Four PCIe DtoH + three HtoD per call.
            //
            // Correctness-first: the backward is one call per training
            // step, and the matmul/super_linear backward dwarfs it.
            // Cost is acceptable until MIOpen parity lands. Tracked as
            // a Path-C-policy violation; this comment + the
            // `host_layer_norm_bwd` helper name make it discoverable.
            let total = n_rows * n_cols;
            let mut x_h = vec![0.0f32; total];
            let mut dy_h = vec![0.0f32; total];
            let mut g_h = vec![0.0f32; n_cols];
            let mut cache_h = vec![0.0f32; 2 * n_rows];
            x.copy_to_host(&mut x_h)?;
            d_out.copy_to_host(&mut dy_h)?;
            gamma.copy_to_host(&mut g_h)?;
            cache.copy_to_host(&mut cache_h)?;

            let mut dx_h = vec![0.0f32; total];
            let mut dg_h = vec![0.0f32; n_cols];
            let mut db_h = vec![0.0f32; n_cols];
            host_layer_norm_bwd(
                &dy_h, &x_h, &g_h, &cache_h,
                &mut dx_h, &mut dg_h, &mut db_h,
                n_rows, n_cols,
            );
            d_x.copy_from_host(&dx_h)?;
            d_gamma.copy_from_host(&dg_h)?;
            d_beta.copy_from_host(&db_h)?;
            Ok(())
        }
    }
}

#[cfg(feature = "rocm")]
pub use rocm_impl::Rocm;

// ─── Tensor<D> ───────────────────────────────────────────────

/// Location-aware tensor over device `D`. Holds `D`'s native
/// buffer, a shape vector, and the device tag (zero-cost via
/// `PhantomData`).
///
/// **Shape semantics (v1):** the `shape` field is *informational*
/// — `shape()` reports the rank-N view of the underlying buffer,
/// but most ops still take explicit `(out_dim, in_dim, ...)` args.
/// This is the seam for future per-op shape validation. Constructors
/// that don't take an explicit shape default to rank-1 (`shape = [len]`).
///
/// Once every op consumes shape from the tensor, the explicit
/// dimension args go away and "matvec on mismatched buffers compiles"
/// becomes "matvec on mismatched buffers is a runtime error at the
/// op boundary." See review notes for migration plan.
pub struct Tensor<D: Device> {
    buffer: D::Buffer,
    shape: Vec<usize>,
    _marker: PhantomData<D>,
}

impl<D: Device> Tensor<D> {
    /// Allocate `n` zeroed f32s as a rank-1 tensor (`shape = [n]`).
    pub fn zeros(n: usize) -> Result<Self, BackendError> {
        Ok(Self {
            buffer: D::zeros(n)?,
            shape: vec![n],
            _marker: PhantomData,
        })
    }

    /// Allocate a zeroed tensor with explicit shape. Total element
    /// count is the product of the shape dimensions.
    pub fn zeros_shape(shape: &[usize]) -> Result<Self, BackendError> {
        let n = shape.iter().product::<usize>();
        Ok(Self {
            buffer: D::zeros(n)?,
            shape: shape.to_vec(),
            _marker: PhantomData,
        })
    }

    /// Construct from host slice as a rank-1 tensor.
    pub fn from_slice(xs: &[f32]) -> Result<Self, BackendError> {
        Ok(Self {
            buffer: D::from_host(xs)?,
            shape: vec![xs.len()],
            _marker: PhantomData,
        })
    }

    /// Construct from host slice with explicit shape. Errors if
    /// `xs.len() != shape.product()`.
    pub fn from_slice_shape(xs: &[f32], shape: &[usize]) -> Result<Self, BackendError> {
        let expected: usize = shape.iter().product();
        if xs.len() != expected {
            return Err(BackendError::Runtime(format!(
                "Tensor::from_slice_shape: slice len {} != shape product {} for shape {:?}",
                xs.len(), expected, shape,
            )));
        }
        Ok(Self {
            buffer: D::from_host(xs)?,
            shape: shape.to_vec(),
            _marker: PhantomData,
        })
    }

    pub fn len(&self) -> usize { self.buffer.len() }
    pub fn is_empty(&self) -> bool { self.buffer.len() == 0 }

    /// Tensor shape — `[len]` for rank-1 tensors, `[m, n]` for rank-2,
    /// etc. **Currently informational** — most ops still take explicit
    /// dimension args. Use `shape_matches` to validate at op boundaries.
    pub fn shape(&self) -> &[usize] { &self.shape }

    /// Tensor rank (number of dimensions). 1 for vectors, 2 for
    /// matrices, etc.
    pub fn rank(&self) -> usize { self.shape.len() }

    /// Returns `true` iff `self.shape()` equals `expected`. Useful
    /// for op-level validation: `if !w.shape_matches(&[out, in_]) { error }`.
    pub fn shape_matches(&self, expected: &[usize]) -> bool {
        self.shape == expected
    }

    /// Reinterpret this tensor's shape (must preserve element count).
    /// Does not move data. Errors if the new shape's product doesn't
    /// match the buffer length.
    pub fn reshape(mut self, new_shape: &[usize]) -> Result<Self, BackendError> {
        let expected: usize = new_shape.iter().product();
        if expected != self.buffer.len() {
            return Err(BackendError::Runtime(format!(
                "Tensor::reshape: new shape {:?} has product {} but buffer holds {} elements",
                new_shape, expected, self.buffer.len(),
            )));
        }
        self.shape = new_shape.to_vec();
        Ok(self)
    }

    /// Read this tensor back to a host `Vec<f32>`. On `Cpu` this is
    /// a memcpy; on `Rocm` it's a `hipMemcpy D2H` (PCIe round-trip)
    /// — explicit in the caller's code, never hidden behind an op.
    pub fn to_vec(&self) -> Result<Vec<f32>, BackendError> {
        let mut out = vec![0.0f32; self.buffer.len()];
        self.buffer.copy_to_host(&mut out)?;
        Ok(out)
    }

    pub fn buffer(&self) -> &D::Buffer { &self.buffer }
    pub fn buffer_mut(&mut self) -> &mut D::Buffer { &mut self.buffer }

    /// Device tag short name (matches `Device::NAME`). Useful for
    /// error messages and tracing.
    pub fn device_name(&self) -> &'static str { D::NAME }
}

// ─── Top-level ops ───────────────────────────────────────────

/// `y = W·x + b`. Type system enforces all arguments share `D`;
/// no cross-device mixing without an explicit `to_vec` round-trip.
pub fn matvec<D: Device>(
    x: &Tensor<D>,
    weight: &Tensor<D>,
    bias: &Tensor<D>,
    out: &mut Tensor<D>,
    out_dim: usize,
    in_dim: usize,
) -> Result<(), BackendError> {
    D::matvec(
        &x.buffer, &weight.buffer, &bias.buffer, &mut out.buffer,
        out_dim, in_dim,
    )
}

/// `C = A @ B`, `[m×k] @ [k×n] = [m×n]` row-major.
pub fn matmul_nn<D: Device>(
    a: &Tensor<D>, b: &Tensor<D>, out: &mut Tensor<D>,
    m: usize, k: usize, n: usize,
) -> Result<(), BackendError> {
    D::matmul_nn(&a.buffer, &b.buffer, &mut out.buffer, m, k, n)
}

/// `C = A @ B^T`, `[m×k] @ [n×k]^T = [m×n]` row-major.
pub fn matmul_nt<D: Device>(
    a: &Tensor<D>, b: &Tensor<D>, out: &mut Tensor<D>,
    m: usize, k: usize, n: usize,
) -> Result<(), BackendError> {
    D::matmul_nt(&a.buffer, &b.buffer, &mut out.buffer, m, k, n)
}

/// `C = A^T @ B`, `[k×m]^T @ [k×n] = [m×n]` row-major.
pub fn matmul_tn<D: Device>(
    a: &Tensor<D>, b: &Tensor<D>, out: &mut Tensor<D>,
    m: usize, k: usize, n: usize,
) -> Result<(), BackendError> {
    D::matmul_tn(&a.buffer, &b.buffer, &mut out.buffer, m, k, n)
}

/// Per-neuron parallel matvec — see `Device::super_linear_fwd`.
pub fn super_linear_fwd<D: Device>(
    x: &Tensor<D>, weight: &Tensor<D>, bias: &Tensor<D>,
    out: &mut Tensor<D>,
    n_neurons: usize, in_per: usize, out_per: usize,
) -> Result<(), BackendError> {
    D::super_linear_fwd(
        &x.buffer, &weight.buffer, &bias.buffer, &mut out.buffer,
        n_neurons, in_per, out_per,
    )
}

/// Row-wise layer normalisation — see `Device::layer_norm`.
pub fn layer_norm<D: Device>(
    x: &Tensor<D>, gamma: &Tensor<D>, beta: &Tensor<D>,
    out: &mut Tensor<D>,
    n_rows: usize, n_cols: usize,
) -> Result<(), BackendError> {
    D::layer_norm(
        &x.buffer, &gamma.buffer, &beta.buffer, &mut out.buffer,
        n_rows, n_cols,
    )
}

/// Element-wise SiLU `y = x · σ(x)` — see `Device::silu`.
pub fn silu<D: Device>(
    x: &Tensor<D>, out: &mut Tensor<D>,
) -> Result<(), BackendError> {
    D::silu(&x.buffer, &mut out.buffer)
}

/// LayerNorm forward + cache (for backward). See `Device::layer_norm_train`.
pub fn layer_norm_train<D: Device>(
    x: &Tensor<D>, gamma: &Tensor<D>, beta: &Tensor<D>,
    out: &mut Tensor<D>, cache: &mut Tensor<D>,
    n_rows: usize, n_cols: usize,
) -> Result<(), BackendError> {
    D::layer_norm_train(&x.buffer, &gamma.buffer, &beta.buffer,
        &mut out.buffer, &mut cache.buffer, n_rows, n_cols)
}

/// LayerNorm backward — see `Device::layer_norm_bwd`.
pub fn layer_norm_bwd<D: Device>(
    d_out: &Tensor<D>, x: &Tensor<D>, gamma: &Tensor<D>, cache: &Tensor<D>,
    d_x: &mut Tensor<D>, d_gamma: &mut Tensor<D>, d_beta: &mut Tensor<D>,
    n_rows: usize, n_cols: usize,
) -> Result<(), BackendError> {
    D::layer_norm_bwd(&d_out.buffer, &x.buffer, &gamma.buffer, &cache.buffer,
        &mut d_x.buffer, &mut d_gamma.buffer, &mut d_beta.buffer, n_rows, n_cols)
}

/// `AdamW<D>` — JAX-style typed optimiser. Holds `m`, `v` moment
/// buffers per parameter group, applies the step in place. Single
/// flat parameter buffer; callers handle multi-parameter
/// orchestration.
pub struct AdamW<D: Device> {
    m: Tensor<D>,
    v: Tensor<D>,
    n_params: usize,
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    step: u64,
}

impl<D: Device> AdamW<D> {
    pub fn new(n_params: usize) -> Result<Self, BackendError> {
        Ok(Self {
            m: Tensor::<D>::zeros(n_params)?,
            v: Tensor::<D>::zeros(n_params)?,
            n_params,
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            step: 0,
        })
    }

    pub fn with_lr(mut self, lr: f32) -> Self { self.lr = lr; self }
    pub fn with_weight_decay(mut self, wd: f32) -> Self { self.weight_decay = wd; self }

    /// Apply one AdamW step in place: `w -= lr * (m_hat / (sqrt(v_hat) + eps) + wd * w)`.
    pub fn step(&mut self, w: &mut Tensor<D>, g: &Tensor<D>) -> Result<(), BackendError> {
        self.step += 1;
        let bc1_inv = 1.0 / (1.0 - self.beta1.powi(self.step as i32));
        let bc2_inv = 1.0 / (1.0 - self.beta2.powi(self.step as i32));
        D::adamw_step(
            &mut w.buffer, &g.buffer,
            &mut self.m.buffer, &mut self.v.buffer,
            self.n_params,
            self.lr, self.beta1, self.beta2, self.eps,
            self.weight_decay, bc1_inv, bc2_inv,
        )
    }
}

/// Row-wise softmax (log-softmax if `log=true`) — see `Device::softmax`.
pub fn softmax<D: Device>(
    x: &Tensor<D>, out: &mut Tensor<D>,
    n_rows: usize, row_len: usize, log: bool,
) -> Result<(), BackendError> {
    D::softmax(&x.buffer, &mut out.buffer, n_rows, row_len, log)
}

/// Gated Linear Unit — see `Device::glu`.
pub fn glu<D: Device>(
    x: &Tensor<D>, out: &mut Tensor<D>,
    n_rows: usize, half_size: usize,
) -> Result<(), BackendError> {
    D::glu(&x.buffer, &mut out.buffer, n_rows, half_size)
}

/// Single-token embedding lookup — see `Device::embedding_lookup`.
/// Copies row `token_idx` from `table` into `out`, staying on device.
pub fn embedding_lookup<D: Device>(
    table: &Tensor<D>,
    token_idx: usize,
    out: &mut Tensor<D>,
    embed_dim: usize,
) -> Result<(), BackendError> {
    D::embedding_lookup(&table.buffer, token_idx, &mut out.buffer, embed_dim)
}

/// Transposed matvec — `d_input = W^T · d_out`. The first backward
/// primitive in the cascade. See `Device::matvec_t`.
pub fn matvec_t<D: Device>(
    d_out: &Tensor<D>, weight: &Tensor<D>, d_input: &mut Tensor<D>,
    out_dim: usize, in_dim: usize,
) -> Result<(), BackendError> {
    D::matvec_t(&d_out.buffer, &weight.buffer, &mut d_input.buffer, out_dim, in_dim)
}

/// Accumulating outer product `accum += a ⊗ b` — see `Device::outer_product_acc`.
pub fn outer_product_acc<D: Device>(
    a: &Tensor<D>, b: &Tensor<D>, accum: &mut Tensor<D>,
    m: usize, n: usize,
) -> Result<(), BackendError> {
    D::outer_product_acc(&a.buffer, &b.buffer, &mut accum.buffer, m, n)
}

/// `dst += src` — see `Device::add_assign`.
pub fn add_assign<D: Device>(
    dst: &mut Tensor<D>, src: &Tensor<D>, n: usize,
) -> Result<(), BackendError> {
    D::add_assign(&mut dst.buffer, &src.buffer, n)
}

/// SGD weight update `w -= lr · g` — see `Device::sgd_step`.
pub fn sgd_step<D: Device>(
    w: &mut Tensor<D>, g: &Tensor<D>, n: usize, lr: f32,
) -> Result<(), BackendError> {
    D::sgd_step(&mut w.buffer, &g.buffer, n, lr)
}

/// SuperLinear backward — d_weights += d_out ⊗ trace per neuron.
pub fn super_linear_bwd_dw<D: Device>(
    d_out: &Tensor<D>, trace: &Tensor<D>, d_weights: &mut Tensor<D>,
    n_neurons: usize, in_per: usize, out_per: usize,
) -> Result<(), BackendError> {
    D::super_linear_bwd_dw(&d_out.buffer, &trace.buffer, &mut d_weights.buffer,
        n_neurons, in_per, out_per)
}

/// SuperLinear backward — d_trace = W^T · d_out per neuron (overwrites).
pub fn super_linear_bwd_dx<D: Device>(
    d_out: &Tensor<D>, weights: &Tensor<D>, d_trace: &mut Tensor<D>,
    n_neurons: usize, in_per: usize, out_per: usize,
) -> Result<(), BackendError> {
    D::super_linear_bwd_dx(&d_out.buffer, &weights.buffer, &mut d_trace.buffer,
        n_neurons, in_per, out_per)
}

/// SiLU backward — `d_x = d_out · silu'(x)`. See `Device::silu_bwd`.
pub fn silu_bwd<D: Device>(
    d_out: &Tensor<D>, x: &Tensor<D>, d_x: &mut Tensor<D>,
) -> Result<(), BackendError> {
    D::silu_bwd(&d_out.buffer, &x.buffer, &mut d_x.buffer)
}

/// GLU backward — see `Device::glu_bwd`.
pub fn glu_bwd<D: Device>(
    d_out: &Tensor<D>, x: &Tensor<D>, d_x: &mut Tensor<D>,
    n_rows: usize, half_size: usize,
) -> Result<(), BackendError> {
    D::glu_bwd(&d_out.buffer, &x.buffer, &mut d_x.buffer, n_rows, half_size)
}

// ─── Linear<D> ───────────────────────────────────────────────

/// JAX-style linear layer parameterised by device. Holds weight
/// + bias `Tensor<D>` and dispatches `forward_into` to the
/// device-native matvec. The same struct definition serves CPU
/// and ROCm; swap `D` and the call sites are identical.
///
/// This is the typed alternative to `modgrad_compute::neuron::Linear`
/// (which holds `Vec<f32>` directly and PCIe-bounces on GPU
/// dispatch). The cascade migrates callers from the untyped
/// `Linear` to `Linear<D>` one container at a time.
pub struct Linear<D: Device> {
    pub weight: Tensor<D>,
    pub bias: Tensor<D>,
    pub in_dim: usize,
    pub out_dim: usize,
}

impl<D: Device> Linear<D> {
    /// Construct from host-side weight + bias slices. Uploads to
    /// device on construction; subsequent `forward_into` calls
    /// have **zero PCIe round-trips** when `D = Rocm`.
    ///
    /// Caller owns weight initialisation: `weight` is row-major
    /// `[out_dim × in_dim]`, `bias` is `[out_dim]`. Random init
    /// lives in caller-side helpers (deliberately decoupled — the
    /// typed Linear is pure wiring, not policy).
    pub fn from_host(
        weight: &[f32],
        bias: &[f32],
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Self, BackendError> {
        if weight.len() != in_dim * out_dim {
            return Err(BackendError::Runtime(format!(
                "Linear::from_host: weight len {} != in_dim*out_dim ({} * {})",
                weight.len(), in_dim, out_dim,
            )));
        }
        if bias.len() != out_dim {
            return Err(BackendError::Runtime(format!(
                "Linear::from_host: bias len {} != out_dim {}",
                bias.len(), out_dim,
            )));
        }
        Ok(Self {
            // Weight gets rank-2 shape [out_dim, in_dim] — explicit so
            // future shape-validating ops can spot dimension mismatches.
            weight: Tensor::<D>::from_slice_shape(weight, &[out_dim, in_dim])?,
            // Bias is rank-1 [out_dim].
            bias:   Tensor::<D>::from_slice_shape(bias, &[out_dim])?,
            in_dim,
            out_dim,
        })
    }

    /// `y = W·x + b`. The op routes through `D::matvec`, which
    /// chose the kernel at type-instantiation time.
    pub fn forward_into(
        &self,
        x: &Tensor<D>,
        y: &mut Tensor<D>,
    ) -> Result<(), BackendError> {
        matvec(x, &self.weight, &self.bias, y, self.out_dim, self.in_dim)
    }

    /// Allocating forward — convenience for tests and one-shot
    /// callers. Production hot loops should prefer `forward_into`
    /// to reuse output buffers.
    pub fn forward(&self, x: &Tensor<D>) -> Result<Tensor<D>, BackendError> {
        let mut y = Tensor::<D>::zeros(self.out_dim)?;
        self.forward_into(x, &mut y)?;
        Ok(y)
    }

    /// Backward: given upstream `d_y` (`[out_dim]`) and the original
    /// input `x` (`[in_dim]`) saved during forward, *accumulate*:
    ///   `d_W += d_y ⊗ x`     (outer product)
    ///   `d_b += d_y`         (passthrough)
    ///   `d_x  = W^T · d_y`   (overwrite — d_x is fresh per-call)
    ///
    /// Accumulating into `d_W` and `d_b` matches the typical
    /// batched gradient pattern: caller zeros buffers once per
    /// batch and runs many `(forward, backward)` iterations
    /// summing into the same gradient buffers, then steps.
    pub fn backward(
        &self,
        d_y: &Tensor<D>,
        x: &Tensor<D>,
        d_w: &mut Tensor<D>,
        d_b: &mut Tensor<D>,
        d_x: &mut Tensor<D>,
    ) -> Result<(), BackendError> {
        // d_x = W^T · d_y
        matvec_t(d_y, &self.weight, d_x, self.out_dim, self.in_dim)?;
        // d_W += d_y ⊗ x
        outer_product_acc(d_y, x, d_w, self.out_dim, self.in_dim)?;
        // d_b += d_y
        add_assign(d_b, d_y, self.out_dim)?;
        Ok(())
    }

    /// Batched forward — the GEMV→GEMM lever. `x` is `[batch × in_dim]`,
    /// `y` is `[batch × out_dim]`, both row-major. Computes
    /// `Y = X · Wᵀ + b` as ONE matmul that reuses `W` across all `batch`
    /// rows (arithmetic intensity ≈ batch, vs ≈1 for the per-row matvec).
    /// Bias is broadcast over rows via a ones-vector outer product — no
    /// new kernel, just the already-wired NN/NT matmuls.
    pub fn forward_batched(
        &self,
        x: &Tensor<D>,
        y: &mut Tensor<D>,
        batch: usize,
    ) -> Result<(), BackendError> {
        // Y[B×out] = X[B×in] @ W[out×in]ᵀ
        matmul_nt(x, &self.weight, y, batch, self.in_dim, self.out_dim)?;
        // Broadcast bias to every row: Y += ones[B×1] @ bias[1×out].
        let ones = Tensor::<D>::from_slice(&vec![1.0f32; batch])?;
        let mut bias_bc = Tensor::<D>::zeros(batch * self.out_dim)?;
        matmul_nn(&ones, &self.bias, &mut bias_bc, batch, 1, self.out_dim)?;
        add_assign(y, &bias_bc, batch * self.out_dim)?;
        Ok(())
    }

    /// Batched backward. `d_y` is `[batch × out_dim]`, `x` the forward
    /// input `[batch × in_dim]`. `d_w`/`d_b` ACCUMULATE (batch-summed),
    /// `d_x` (`[batch × in_dim]`) overwrites. All three gradients are
    /// single GEMMs (plus a ones-matmul for `d_b`) — equivalent to, and
    /// far faster than, `batch` independent `backward` calls.
    pub fn backward_batched(
        &self,
        d_y: &Tensor<D>,
        x: &Tensor<D>,
        d_w: &mut Tensor<D>,
        d_b: &mut Tensor<D>,
        d_x: &mut Tensor<D>,
        batch: usize,
    ) -> Result<(), BackendError> {
        // d_X[B×in] = d_Y[B×out] @ W[out×in]
        matmul_nn(d_y, &self.weight, d_x, batch, self.out_dim, self.in_dim)?;
        // d_W += d_Yᵀ[out×B] @ X[B×in]  (matmul_tn overwrites → temp + add)
        let mut dw_batch = Tensor::<D>::zeros(self.out_dim * self.in_dim)?;
        matmul_tn(d_y, x, &mut dw_batch, self.out_dim, batch, self.in_dim)?;
        add_assign(d_w, &dw_batch, self.out_dim * self.in_dim)?;
        // d_b += Σ_rows d_Y = ones[1×B] @ d_Y[B×out]
        let ones = Tensor::<D>::from_slice(&vec![1.0f32; batch])?;
        let mut db_batch = Tensor::<D>::zeros(self.out_dim)?;
        matmul_nn(&ones, d_y, &mut db_batch, 1, batch, self.out_dim)?;
        add_assign(d_b, &db_batch, self.out_dim)?;
        Ok(())
    }
}

/// **Host-fallback** mean-squared-error loss: `loss = (1/n) Σ (pred - target)²`,
/// `d_pred = (2/n) (pred - target)`. Returns scalar loss; gradient
/// written to `d_pred` (overwrites).
///
/// # ⚠ PCIe round-trip on `Tensor<Rocm>`
///
/// This violates the Path C "no hidden PCIe round-trip" promise.
/// On `Rocm`, this fn does **three** PCIe crossings per call:
/// 1. download `pred` to host
/// 2. download `target` to host
/// 3. upload `d_pred` back to device
///
/// Acceptable for the integration smoke; **NOT** acceptable for
/// production training loops. The proper fix is `Op::MseLossResident`
/// + a tiny WGSL/HIP kernel that produces (loss, d_pred) on-device.
/// See `feedback_path_c_host_fallbacks` memory entry once it lands.
///
/// Function name kept stable for v0 ergonomics; rename to
/// `mse_loss_host_fallback` once the resident variant exists so
/// callers consciously choose.
pub fn mse_loss<D: Device>(
    pred: &Tensor<D>,
    target: &Tensor<D>,
    d_pred: &mut Tensor<D>,
    n: usize,
) -> Result<f32, BackendError> {
    // HOST-FALLBACK: download → compute → upload.
    // On Rocm, three PCIe crossings per call. See doc comment.
    let pv = pred.to_vec()?;
    let tv = target.to_vec()?;
    if pv.len() < n || tv.len() < n {
        return Err(BackendError::Runtime("mse_loss: buffer shorter than n".into()));
    }
    let inv_n = 1.0 / n as f32;
    let mut loss = 0.0f32;
    let mut grad = vec![0.0f32; n];
    for i in 0..n {
        let diff = pv[i] - tv[i];
        loss += diff * diff;
        grad[i] = 2.0 * diff * inv_n;
    }
    loss *= inv_n;
    let new_d_pred = Tensor::<D>::from_slice(&grad)?;
    // Replace d_pred contents
    let v = new_d_pred.to_vec()?;
    let dst = d_pred.buffer_mut();
    dst.copy_from_host(&v)?;
    Ok(loss)
}

// ─── SuperLinear<D> ──────────────────────────────────────────

/// JAX-style per-neuron parallel MLP, parameterised by device.
/// One weight matrix and bias per neuron — the workhorse for the
/// CTM's NLM stage1/stage2.
///
/// Same struct serves Cpu and Rocm; swap `D` and call sites are
/// identical. The `forward_into` op routes to
/// `Device::super_linear_fwd` which picks the kernel at type-
/// instantiation time (CPU: rayon-parallel host code; Rocm:
/// `hipblasSgemmStridedBatched` via the resident path).
pub struct SuperLinear<D: Device> {
    pub weights: Tensor<D>,
    pub biases: Tensor<D>,
    pub n_neurons: usize,
    pub in_per: usize,
    pub out_per: usize,
}

impl<D: Device> SuperLinear<D> {
    /// Construct from host-side weights + biases.
    /// `weights` is `[n_neurons × out_per × in_per]` row-major-per-neuron;
    /// `biases` is `[n_neurons × out_per]`.
    pub fn from_host(
        weights: &[f32],
        biases: &[f32],
        n_neurons: usize,
        in_per: usize,
        out_per: usize,
    ) -> Result<Self, BackendError> {
        let want_w = n_neurons * out_per * in_per;
        let want_b = n_neurons * out_per;
        if weights.len() != want_w {
            return Err(BackendError::Runtime(format!(
                "SuperLinear::from_host: weights len {} != n*out*in = {}",
                weights.len(), want_w,
            )));
        }
        if biases.len() != want_b {
            return Err(BackendError::Runtime(format!(
                "SuperLinear::from_host: biases len {} != n*out = {}",
                biases.len(), want_b,
            )));
        }
        Ok(Self {
            // Weights are rank-3 per neuron: [n_neurons, out_per, in_per].
            weights: Tensor::<D>::from_slice_shape(weights, &[n_neurons, out_per, in_per])?,
            // Biases are rank-2: [n_neurons, out_per].
            biases: Tensor::<D>::from_slice_shape(biases, &[n_neurons, out_per])?,
            n_neurons, in_per, out_per,
        })
    }

    /// Forward into pre-allocated output. `x` is `[n_neurons × in_per]`,
    /// `out` is `[n_neurons × out_per]`. Both must be allocated.
    pub fn forward_into(
        &self,
        x: &Tensor<D>,
        out: &mut Tensor<D>,
    ) -> Result<(), BackendError> {
        super_linear_fwd(
            x, &self.weights, &self.biases, out,
            self.n_neurons, self.in_per, self.out_per,
        )
    }

    /// Allocating forward.
    pub fn forward(&self, x: &Tensor<D>) -> Result<Tensor<D>, BackendError> {
        let mut out = Tensor::<D>::zeros(self.n_neurons * self.out_per)?;
        self.forward_into(x, &mut out)?;
        Ok(out)
    }

    /// Backward through a SuperLinear layer.
    ///   `d_W += d_out ⊗ trace` (accumulates into `d_weights`)
    ///   `d_b += d_out`         (per-neuron, accumulates into `d_biases`)
    ///   `d_trace  = W^T · d_out` (overwrites — fresh per call)
    pub fn backward(
        &self,
        d_out: &Tensor<D>,
        trace: &Tensor<D>,
        d_weights: &mut Tensor<D>,
        d_biases: &mut Tensor<D>,
        d_trace: &mut Tensor<D>,
    ) -> Result<(), BackendError> {
        super_linear_bwd_dw(d_out, trace, d_weights,
            self.n_neurons, self.in_per, self.out_per)?;
        super_linear_bwd_dx(d_out, &self.weights, d_trace,
            self.n_neurons, self.in_per, self.out_per)?;
        // d_b += d_out (per-neuron passthrough; bias is [n × out_per],
        // d_out is also [n × out_per], same shape so add_assign works).
        add_assign(d_biases, d_out, self.n_neurons * self.out_per)?;
        Ok(())
    }

    /// Batch-capable forward. `x` is `[batch × n_neurons × in_per]`,
    /// output `[batch × n_neurons × out_per]`. Makes the NLM usable in a
    /// batched tick loop.
    ///
    /// NOTE: this loops over the batch reusing the per-neuron kernel — it
    /// is batch-*capable*, not yet batch-*optimal*. The full GEMM win for
    /// the NLM (grow the strided-batched inner `n` from 1 to B so each
    /// neuron's RHS is `[in_per × B]`) is a device-kernel follow-up; the
    /// synapse — the dominant per-tick FLOP — already has the full GEMM.
    pub fn forward_batched(
        &self,
        x: &Tensor<D>,
        batch: usize,
    ) -> Result<Tensor<D>, BackendError> {
        let in_sz = self.n_neurons * self.in_per;
        let out_sz = self.n_neurons * self.out_per;
        let x_host = x.to_vec()?;
        let mut out_host = vec![0.0f32; batch * out_sz];
        for b in 0..batch {
            let xb = Tensor::<D>::from_slice(&x_host[b * in_sz..(b + 1) * in_sz])?;
            let ob = self.forward(&xb)?;
            out_host[b * out_sz..(b + 1) * out_sz].copy_from_slice(&ob.to_vec()?);
        }
        Tensor::<D>::from_slice(&out_host)
    }

    /// Batch-capable backward. `d_out`/`trace` are `[batch × n_neurons ×
    /// {out_per,in_per}]`; `d_weights`/`d_biases` ACCUMULATE (batch-summed,
    /// via the per-example backward); `d_trace` (`[batch × n_neurons ×
    /// in_per]`) overwritten.
    pub fn backward_batched(
        &self,
        d_out: &Tensor<D>,
        trace: &Tensor<D>,
        d_weights: &mut Tensor<D>,
        d_biases: &mut Tensor<D>,
        d_trace: &mut Tensor<D>,
        batch: usize,
    ) -> Result<(), BackendError> {
        let in_sz = self.n_neurons * self.in_per;
        let out_sz = self.n_neurons * self.out_per;
        let do_host = d_out.to_vec()?;
        let tr_host = trace.to_vec()?;
        let mut dtr_host = vec![0.0f32; batch * in_sz];
        for b in 0..batch {
            let dob = Tensor::<D>::from_slice(&do_host[b * out_sz..(b + 1) * out_sz])?;
            let trb = Tensor::<D>::from_slice(&tr_host[b * in_sz..(b + 1) * in_sz])?;
            let mut dtrb = Tensor::<D>::zeros(in_sz)?;
            self.backward(&dob, &trb, d_weights, d_biases, &mut dtrb)?;
            dtr_host[b * in_sz..(b + 1) * in_sz].copy_from_slice(&dtrb.to_vec()?);
        }
        *d_trace = Tensor::<D>::from_slice(&dtr_host)?;
        Ok(())
    }
}

// ─── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_tensor_roundtrip() {
        let t = Tensor::<Cpu>::from_slice(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(t.len(), 4);
        assert_eq!(t.device_name(), "cpu");
        let v = t.to_vec().unwrap();
        assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn cpu_tensor_zeros() {
        let t = Tensor::<Cpu>::zeros(8).unwrap();
        assert_eq!(t.to_vec().unwrap(), vec![0.0; 8]);
    }

    #[test]
    fn cpu_matvec_identity_plus_bias() {
        // 3×3 identity, bias = [10, 20, 30], x = [1, 2, 3]
        // → out = [1+10, 2+20, 3+30] = [11, 22, 33]
        let x = Tensor::<Cpu>::from_slice(&[1.0, 2.0, 3.0]).unwrap();
        #[rustfmt::skip]
        let w = Tensor::<Cpu>::from_slice(&[
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]).unwrap();
        let b = Tensor::<Cpu>::from_slice(&[10.0, 20.0, 30.0]).unwrap();
        let mut y = Tensor::<Cpu>::zeros(3).unwrap();
        matvec(&x, &w, &b, &mut y, 3, 3).unwrap();
        assert_eq!(y.to_vec().unwrap(), vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn cpu_matvec_general() {
        // 2×3 matrix W = [[1, 2, 3], [4, 5, 6]], x = [1, 0, -1], b = [0, 0]
        // → out = [1·1 + 2·0 + 3·(-1), 4·1 + 5·0 + 6·(-1)] = [-2, -2]
        let x = Tensor::<Cpu>::from_slice(&[1.0, 0.0, -1.0]).unwrap();
        let w = Tensor::<Cpu>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Tensor::<Cpu>::zeros(2).unwrap();
        let mut y = Tensor::<Cpu>::zeros(2).unwrap();
        matvec(&x, &w, &b, &mut y, 2, 3).unwrap();
        assert_eq!(y.to_vec().unwrap(), vec![-2.0, -2.0]);
    }

    #[test]
    fn cpu_matmul_nn_2x3x4() {
        // A = [[1,2,3],[4,5,6]] (2×3), B = [[1,0,0,1],[0,1,0,1],[0,0,1,1]] (3×4)
        // C = A·B = [[1,2,3,6],[4,5,6,15]]
        let a = Tensor::<Cpu>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Tensor::<Cpu>::from_slice(&[
            1.0, 0.0, 0.0, 1.0,
            0.0, 1.0, 0.0, 1.0,
            0.0, 0.0, 1.0, 1.0,
        ]).unwrap();
        let mut c = Tensor::<Cpu>::zeros(2 * 4).unwrap();
        matmul_nn(&a, &b, &mut c, 2, 3, 4).unwrap();
        assert_eq!(c.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 6.0, 4.0, 5.0, 6.0, 15.0]);
    }

    /// The whole point of Path C: same code, swap `Cpu` for `Rocm`,
    /// run identically. This test parameterises the matvec body and
    /// asserts CPU and ROCm produce bit-identical output. Skipped
    /// when `rocm` feature isn't compiled in.
    #[cfg(feature = "rocm")]
    #[test]
    fn cpu_and_rocm_matvec_match() {
        let x_data = vec![1.0, 0.5, -0.25, 0.125];
        let w_data: Vec<f32> = (0..3 * 4).map(|i| (i as f32) * 0.1).collect();
        let b_data = vec![0.1, 0.2, 0.3];

        let cpu_out = run_matvec::<Cpu>(&x_data, &w_data, &b_data, 3, 4).unwrap();

        // Skip ROCm portion if no GPU is available.
        let rocm_out = match run_matvec::<Rocm>(&x_data, &w_data, &b_data, 3, 4) {
            Ok(v) => v,
            Err(BackendError::Runtime(msg)) if msg.contains("hipMalloc") => return,
            Err(e) => panic!("rocm matvec failed: {:?}", e),
        };

        for (c, r) in cpu_out.iter().zip(&rocm_out) {
            assert!((c - r).abs() < 1e-4,
                "CPU and ROCm matvec disagree: cpu={} rocm={}", c, r);
        }
    }

    #[cfg(all(test, feature = "rocm"))]
    fn run_matvec<D: Device>(
        x: &[f32], w: &[f32], b: &[f32], out_dim: usize, in_dim: usize,
    ) -> Result<Vec<f32>, BackendError> {
        let xt = Tensor::<D>::from_slice(x)?;
        let wt = Tensor::<D>::from_slice(w)?;
        let bt = Tensor::<D>::from_slice(b)?;
        let mut yt = Tensor::<D>::zeros(out_dim)?;
        matvec(&xt, &wt, &bt, &mut yt, out_dim, in_dim)?;
        yt.to_vec()
    }

    /// Same-code-different-device test for matmul_nn. Confirms
    /// the resident hipBLAS path agrees with rayon CPU bit-by-bit
    /// (within float tolerance).
    #[cfg(feature = "rocm")]
    #[test]
    fn cpu_and_rocm_matmul_nn_match() {
        let a: Vec<f32> = (0..6).map(|i| (i as f32) * 0.1).collect(); // 2×3
        let b: Vec<f32> = (0..12).map(|i| (i as f32) * 0.05).collect(); // 3×4

        let cpu_out = run_matmul_nn::<Cpu>(&a, &b, 2, 3, 4).unwrap();
        let rocm_out = match run_matmul_nn::<Rocm>(&a, &b, 2, 3, 4) {
            Ok(v) => v,
            Err(BackendError::Runtime(msg)) if msg.contains("hipMalloc") => return,
            Err(e) => panic!("rocm matmul failed: {:?}", e),
        };

        for (c, r) in cpu_out.iter().zip(&rocm_out) {
            assert!((c - r).abs() < 1e-4,
                "CPU and ROCm matmul disagree: cpu={} rocm={}", c, r);
        }
    }

    #[cfg(all(test, feature = "rocm"))]
    fn run_matmul_nn<D: Device>(
        a: &[f32], b: &[f32], m: usize, k: usize, n: usize,
    ) -> Result<Vec<f32>, BackendError> {
        let at = Tensor::<D>::from_slice(a)?;
        let bt = Tensor::<D>::from_slice(b)?;
        let mut ct = Tensor::<D>::zeros(m * n)?;
        matmul_nn(&at, &bt, &mut ct, m, k, n)?;
        ct.to_vec()
    }

    // ─── Linear<D> tests ─────────────────────────────────

    #[test]
    fn linear_cpu_forward_matches_hand_calculation() {
        // Linear maps R^3 → R^2 with W=[[1,2,3],[4,5,6]], b=[10,20]
        // x=[1, 0, -1] → y=[1·1+2·0+3·(-1)+10, 4·1+5·0+6·(-1)+20]
        //              = [1-3+10, 4-6+20] = [8, 18]
        let w = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![10.0, 20.0];
        let lin = Linear::<Cpu>::from_host(&w, &b, 3, 2).unwrap();
        let x = Tensor::<Cpu>::from_slice(&[1.0, 0.0, -1.0]).unwrap();
        let y = lin.forward(&x).unwrap();
        assert_eq!(y.to_vec().unwrap(), vec![8.0, 18.0]);
    }

    #[test]
    fn linear_from_host_size_check() {
        let bad_weight = vec![0.0f32; 5]; // expected 6 (3×2)
        let bias = vec![0.0f32; 2];
        match Linear::<Cpu>::from_host(&bad_weight, &bias, 3, 2) {
            Err(BackendError::Runtime(_)) => {}
            Err(e) => panic!("expected Runtime error, got {:?}", e),
            Ok(_) => panic!("expected size mismatch error"),
        }
    }

    /// The cascade payoff: same Linear code runs on Cpu and Rocm,
    /// produces matching output. Demonstrates that swapping the
    /// device parameter is sufficient to switch hardware.
    #[cfg(feature = "rocm")]
    #[test]
    fn cpu_and_rocm_linear_match() {
        let w: Vec<f32> = (0..12).map(|i| (i as f32 - 6.0) * 0.1).collect(); // 4×3
        let b: Vec<f32> = vec![0.1, -0.2, 0.3, -0.4];
        let x_data = vec![0.5, -0.5, 0.25];

        let cpu_y = run_linear::<Cpu>(&w, &b, &x_data, 3, 4).unwrap();
        let rocm_y = match run_linear::<Rocm>(&w, &b, &x_data, 3, 4) {
            Ok(v) => v,
            Err(BackendError::Runtime(msg)) if msg.contains("hipMalloc") => return,
            Err(e) => panic!("rocm Linear failed: {:?}", e),
        };

        for (c, r) in cpu_y.iter().zip(&rocm_y) {
            assert!((c - r).abs() < 1e-4,
                "CPU and ROCm Linear disagree: cpu={} rocm={}", c, r);
        }
    }

    #[cfg(all(test, feature = "rocm"))]
    fn run_linear<D: Device>(
        w: &[f32], b: &[f32], x: &[f32],
        in_dim: usize, out_dim: usize,
    ) -> Result<Vec<f32>, BackendError> {
        let lin = Linear::<D>::from_host(w, b, in_dim, out_dim)?;
        let x_t = Tensor::<D>::from_slice(x)?;
        let y = lin.forward(&x_t)?;
        y.to_vec()
    }

    // ─── SuperLinear<D> tests ────────────────────────────

    #[test]
    fn super_linear_cpu_per_neuron_matvec() {
        // 2 neurons, in_per=3, out_per=2. Each neuron's W is the
        // 2×3 identity-ish matrix [[1,0,0],[0,1,0]]; biases zero.
        // Inputs differ per neuron: x[0]=[1,2,3], x[1]=[10,20,30].
        // Expected: out[0]=[1,2], out[1]=[10,20].
        let n = 2;
        let in_per = 3;
        let out_per = 2;
        let weights: Vec<f32> = (0..n).flat_map(|_| {
            vec![1.0, 0.0, 0.0,   // out_row 0
                 0.0, 1.0, 0.0]   // out_row 1
        }).collect();
        let biases = vec![0.0f32; n * out_per];
        let sl = SuperLinear::<Cpu>::from_host(
            &weights, &biases, n, in_per, out_per,
        ).unwrap();
        let x = Tensor::<Cpu>::from_slice(&[1.0, 2.0, 3.0, 10.0, 20.0, 30.0]).unwrap();
        let out = sl.forward(&x).unwrap();
        assert_eq!(out.to_vec().unwrap(), vec![1.0, 2.0, 10.0, 20.0]);
    }

    /// Same SuperLinear code on Cpu and Rocm produces matching
    /// output. Confirms the resident batched GEMM path agrees with
    /// the rayon-parallel host code.
    #[cfg(feature = "rocm")]
    #[test]
    fn cpu_and_rocm_super_linear_match() {
        let n = 3; let in_per = 4; let out_per = 2;
        let weights: Vec<f32> = (0..n * out_per * in_per)
            .map(|i| (i as f32 - 12.0) * 0.05).collect();
        let biases: Vec<f32> = (0..n * out_per).map(|i| i as f32 * 0.1).collect();
        let x_data: Vec<f32> = (0..n * in_per).map(|i| (i as f32) * 0.25).collect();

        let cpu_out = run_super::<Cpu>(&weights, &biases, &x_data, n, in_per, out_per).unwrap();
        let rocm_out = match run_super::<Rocm>(&weights, &biases, &x_data, n, in_per, out_per) {
            Ok(v) => v,
            Err(BackendError::Runtime(msg)) if msg.contains("hipMalloc") => return,
            Err(BackendError::Unsupported { .. }) => return, // no batched-GEMM on this backend
            Err(e) => panic!("rocm SuperLinear failed: {:?}", e),
        };

        for (c, r) in cpu_out.iter().zip(&rocm_out) {
            assert!((c - r).abs() < 1e-4,
                "CPU and ROCm SuperLinear disagree: cpu={} rocm={}", c, r);
        }
    }

    #[cfg(all(test, feature = "rocm"))]
    fn run_super<D: Device>(
        w: &[f32], b: &[f32], x: &[f32],
        n: usize, in_per: usize, out_per: usize,
    ) -> Result<Vec<f32>, BackendError> {
        let sl = SuperLinear::<D>::from_host(w, b, n, in_per, out_per)?;
        let x_t = Tensor::<D>::from_slice(x)?;
        let y = sl.forward(&x_t)?;
        y.to_vec()
    }

    // ─── Composition test: 2-layer MLP ───────────────────

    /// Two `Linear<D>` stacked: x → L1 → L2 → y. Proves that
    /// Tensor outputs from one layer feed cleanly into the next on
    /// the same device, with no host round-trip in between. The
    /// JAX UX promise materialised: write `mlp::<D>(...)` once, run
    /// on either backend.
    fn mlp<D: Device>(
        w1: &[f32], b1: &[f32],
        w2: &[f32], b2: &[f32],
        x: &[f32],
        in_dim: usize, hidden: usize, out_dim: usize,
    ) -> Result<Vec<f32>, BackendError> {
        let l1 = Linear::<D>::from_host(w1, b1, in_dim, hidden)?;
        let l2 = Linear::<D>::from_host(w2, b2, hidden, out_dim)?;
        let x_t = Tensor::<D>::from_slice(x)?;
        let h = l1.forward(&x_t)?;
        let y = l2.forward(&h)?;
        y.to_vec()
    }

    #[test]
    fn cpu_layer_norm_normalises_to_zero_mean_unit_var() {
        // Single row [1, 2, 3, 4]: mean=2.5, var=1.25
        // (x - μ)/√(var+ε) ≈ ([-1.5,-0.5,0.5,1.5] / 1.118)
        //                   ≈ [-1.342, -0.447, 0.447, 1.342]
        // gamma=1, beta=0 ⇒ identity affine.
        let x = Tensor::<Cpu>::from_slice(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let gamma = Tensor::<Cpu>::from_slice(&[1.0, 1.0, 1.0, 1.0]).unwrap();
        let beta = Tensor::<Cpu>::from_slice(&[0.0, 0.0, 0.0, 0.0]).unwrap();
        let mut out = Tensor::<Cpu>::zeros(4).unwrap();
        layer_norm(&x, &gamma, &beta, &mut out, 1, 4).unwrap();
        let v = out.to_vec().unwrap();
        // Approx; ε = 1e-5 perturbation.
        let expected = [-1.342, -0.447, 0.447, 1.342];
        for (a, e) in v.iter().zip(&expected) {
            assert!((a - e).abs() < 1e-2, "got {} vs {}", a, e);
        }
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn cpu_and_rocm_layer_norm_match() {
        let n_rows = 3; let n_cols = 4;
        let x_data: Vec<f32> = (0..n_rows * n_cols)
            .map(|i| (i as f32 - 5.0) * 0.5).collect();
        let gamma_data: Vec<f32> = (0..n_cols).map(|i| 1.0 + i as f32 * 0.1).collect();
        let beta_data: Vec<f32> = (0..n_cols).map(|i| i as f32 * 0.05 - 0.1).collect();

        let cpu_y = run_layer_norm::<Cpu>(&x_data, &gamma_data, &beta_data, n_rows, n_cols).unwrap();
        let rocm_y = match run_layer_norm::<Rocm>(&x_data, &gamma_data, &beta_data, n_rows, n_cols) {
            Ok(v) => v,
            Err(BackendError::Runtime(msg)) if msg.contains("hipMalloc") => return,
            Err(BackendError::Unsupported { .. }) => return,
            Err(e) => panic!("rocm layer_norm failed: {:?}", e),
        };
        for (c, r) in cpu_y.iter().zip(&rocm_y) {
            assert!((c - r).abs() < 1e-3,
                "CPU and ROCm LayerNorm disagree: cpu={} rocm={}", c, r);
        }
    }

    #[cfg(all(test, feature = "rocm"))]
    fn run_layer_norm<D: Device>(
        x: &[f32], g: &[f32], b: &[f32], n_rows: usize, n_cols: usize,
    ) -> Result<Vec<f32>, BackendError> {
        let xt = Tensor::<D>::from_slice(x)?;
        let gt = Tensor::<D>::from_slice(g)?;
        let bt = Tensor::<D>::from_slice(b)?;
        let mut yt = Tensor::<D>::zeros(n_rows * n_cols)?;
        layer_norm(&xt, &gt, &bt, &mut yt, n_rows, n_cols)?;
        yt.to_vec()
    }

    #[test]
    fn cpu_silu_zero_and_known_values() {
        // SiLU(0)=0, SiLU(1)≈0.7311, SiLU(-1)≈-0.2689
        let x = Tensor::<Cpu>::from_slice(&[0.0, 1.0, -1.0]).unwrap();
        let mut out = Tensor::<Cpu>::zeros(3).unwrap();
        silu(&x, &mut out).unwrap();
        let v = out.to_vec().unwrap();
        assert!((v[0] - 0.0).abs() < 1e-6);
        assert!((v[1] - 0.7311).abs() < 1e-3);
        assert!((v[2] - (-0.2689)).abs() < 1e-3);
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn cpu_and_rocm_silu_match() {
        let x_data: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.5).collect();
        let cpu_y = run_silu::<Cpu>(&x_data).unwrap();
        let rocm_y = match run_silu::<Rocm>(&x_data) {
            Ok(v) => v,
            Err(BackendError::Runtime(msg)) if msg.contains("hipMalloc") => return,
            Err(BackendError::Unsupported { .. }) => return,
            Err(e) => panic!("rocm silu failed: {:?}", e),
        };
        for (c, r) in cpu_y.iter().zip(&rocm_y) {
            assert!((c - r).abs() < 1e-4,
                "CPU and ROCm silu disagree: cpu={} rocm={}", c, r);
        }
    }

    #[cfg(all(test, feature = "rocm"))]
    fn run_silu<D: Device>(x: &[f32]) -> Result<Vec<f32>, BackendError> {
        let xt = Tensor::<D>::from_slice(x)?;
        let mut yt = Tensor::<D>::zeros(x.len())?;
        silu(&xt, &mut yt)?;
        yt.to_vec()
    }

    // ─── softmax ─────────────────────────────────────────

    #[test]
    fn cpu_softmax_uniform_row_is_uniform_distribution() {
        // Equal logits → uniform softmax = [1/4, 1/4, 1/4, 1/4]
        let x = Tensor::<Cpu>::from_slice(&[0.0, 0.0, 0.0, 0.0]).unwrap();
        let mut out = Tensor::<Cpu>::zeros(4).unwrap();
        softmax(&x, &mut out, 1, 4, false).unwrap();
        let v = out.to_vec().unwrap();
        for x in &v { assert!((x - 0.25).abs() < 1e-6); }
    }

    #[test]
    fn cpu_softmax_one_hot_logits_concentrates() {
        // [1000, 0, 0, 0] → near-one-hot
        let x = Tensor::<Cpu>::from_slice(&[1000.0, 0.0, 0.0, 0.0]).unwrap();
        let mut out = Tensor::<Cpu>::zeros(4).unwrap();
        softmax(&x, &mut out, 1, 4, false).unwrap();
        let v = out.to_vec().unwrap();
        assert!(v[0] > 0.999);
        for &x in &v[1..] { assert!(x < 1e-3); }
    }

    #[test]
    fn cpu_softmax_log_form_subtracts_logsumexp() {
        // log_softmax([0,0,0,0]) = log(0.25) ≈ -1.386
        let x = Tensor::<Cpu>::from_slice(&[0.0, 0.0, 0.0, 0.0]).unwrap();
        let mut out = Tensor::<Cpu>::zeros(4).unwrap();
        softmax(&x, &mut out, 1, 4, true).unwrap();
        let v = out.to_vec().unwrap();
        let expected = (0.25f32).ln();
        for x in &v { assert!((x - expected).abs() < 1e-6); }
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn cpu_and_rocm_softmax_match() {
        let n_rows = 3; let row_len = 5;
        let x_data: Vec<f32> = (0..n_rows * row_len).map(|i| (i as f32 - 7.0) * 0.3).collect();
        let cpu_y = run_softmax::<Cpu>(&x_data, n_rows, row_len, false).unwrap();
        let rocm_y = match run_softmax::<Rocm>(&x_data, n_rows, row_len, false) {
            Ok(v) => v,
            Err(BackendError::Runtime(msg)) if msg.contains("hipMalloc") => return,
            Err(BackendError::Unsupported { .. }) => return,
            Err(e) => panic!("rocm softmax failed: {:?}", e),
        };
        for (c, r) in cpu_y.iter().zip(&rocm_y) {
            assert!((c - r).abs() < 1e-4,
                "CPU and ROCm softmax disagree: cpu={} rocm={}", c, r);
        }
    }

    #[cfg(all(test, feature = "rocm"))]
    fn run_softmax<D: Device>(
        x: &[f32], n_rows: usize, row_len: usize, log: bool,
    ) -> Result<Vec<f32>, BackendError> {
        let xt = Tensor::<D>::from_slice(x)?;
        let mut yt = Tensor::<D>::zeros(x.len())?;
        softmax(&xt, &mut yt, n_rows, row_len, log)?;
        yt.to_vec()
    }

    // ─── glu ─────────────────────────────────────────────

    #[test]
    fn cpu_glu_value_times_sigmoid_of_gate() {
        // x = [1, 2, 3, 4] (1 row, half=2). value=[1,2], gate=[3,4].
        // out = [1·σ(3), 2·σ(4)]
        let x = Tensor::<Cpu>::from_slice(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let mut out = Tensor::<Cpu>::zeros(2).unwrap();
        glu(&x, &mut out, 1, 2).unwrap();
        let v = out.to_vec().unwrap();
        let s = |z: f32| 1.0 / (1.0 + (-z).exp());
        assert!((v[0] - 1.0 * s(3.0)).abs() < 1e-5);
        assert!((v[1] - 2.0 * s(4.0)).abs() < 1e-5);
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn cpu_and_rocm_glu_match() {
        let n_rows = 4; let half = 6;
        let x_data: Vec<f32> = (0..n_rows * 2 * half).map(|i| (i as f32 - 12.0) * 0.1).collect();
        let cpu_y = run_glu::<Cpu>(&x_data, n_rows, half).unwrap();
        let rocm_y = match run_glu::<Rocm>(&x_data, n_rows, half) {
            Ok(v) => v,
            Err(BackendError::Runtime(msg)) if msg.contains("hipMalloc") => return,
            Err(BackendError::Unsupported { .. }) => return,
            Err(e) => panic!("rocm glu failed: {:?}", e),
        };
        for (c, r) in cpu_y.iter().zip(&rocm_y) {
            assert!((c - r).abs() < 1e-4,
                "CPU and ROCm GLU disagree: cpu={} rocm={}", c, r);
        }
    }

    #[cfg(all(test, feature = "rocm"))]
    fn run_glu<D: Device>(
        x: &[f32], n_rows: usize, half_size: usize,
    ) -> Result<Vec<f32>, BackendError> {
        let xt = Tensor::<D>::from_slice(x)?;
        let mut yt = Tensor::<D>::zeros(n_rows * half_size)?;
        glu(&xt, &mut yt, n_rows, half_size)?;
        yt.to_vec()
    }

    // ─── embedding_lookup ────────────────────────────────

    #[test]
    fn cpu_embedding_lookup_copies_row() {
        // table = 3 tokens × 4 dims = [[0..3], [10..13], [20..23]]
        let table_data: Vec<f32> = vec![
            0.0, 1.0, 2.0, 3.0,
            10.0, 11.0, 12.0, 13.0,
            20.0, 21.0, 22.0, 23.0,
        ];
        let table = Tensor::<Cpu>::from_slice(&table_data).unwrap();
        let mut out = Tensor::<Cpu>::zeros(4).unwrap();
        embedding_lookup(&table, 1, &mut out, 4).unwrap();
        assert_eq!(out.to_vec().unwrap(), vec![10.0, 11.0, 12.0, 13.0]);
        embedding_lookup(&table, 2, &mut out, 4).unwrap();
        assert_eq!(out.to_vec().unwrap(), vec![20.0, 21.0, 22.0, 23.0]);
    }

    #[test]
    fn cpu_embedding_lookup_out_of_bounds() {
        let table = Tensor::<Cpu>::from_slice(&[0.0, 1.0, 2.0, 3.0]).unwrap();
        let mut out = Tensor::<Cpu>::zeros(2).unwrap();
        match embedding_lookup(&table, 5, &mut out, 2) {
            Err(BackendError::Runtime(_)) => {}
            other => panic!("expected Runtime error for OOB token, got {:?}",
                other.map(|_| "Ok")),
        }
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn cpu_and_rocm_embedding_lookup_match() {
        let vocab = 5; let dim = 6;
        let table_data: Vec<f32> = (0..vocab * dim).map(|i| (i as f32) * 0.5).collect();

        for token in [0usize, 2, 4] {
            let cpu_y = run_embedding::<Cpu>(&table_data, token, dim).unwrap();
            let rocm_y = match run_embedding::<Rocm>(&table_data, token, dim) {
                Ok(v) => v,
                Err(BackendError::Runtime(msg)) if msg.contains("hipMalloc") => return,
                Err(e) => panic!("rocm embedding_lookup failed: {:?}", e),
            };
            assert_eq!(cpu_y, rocm_y, "token {} mismatch", token);
        }
    }

    #[cfg(all(test, feature = "rocm"))]
    fn run_embedding<D: Device>(
        table: &[f32], token: usize, dim: usize,
    ) -> Result<Vec<f32>, BackendError> {
        let t = Tensor::<D>::from_slice(table)?;
        let mut out = Tensor::<D>::zeros(dim)?;
        embedding_lookup(&t, token, &mut out, dim)?;
        out.to_vec()
    }

    // ─── matvec_t (first backward primitive) ─────────────

    #[test]
    fn cpu_matvec_t_identity() {
        // W = 3×3 identity, d_out = [1,2,3] → d_input = W^T·d_out = [1,2,3]
        let d_out = Tensor::<Cpu>::from_slice(&[1.0, 2.0, 3.0]).unwrap();
        #[rustfmt::skip]
        let w = Tensor::<Cpu>::from_slice(&[
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]).unwrap();
        let mut d_input = Tensor::<Cpu>::zeros(3).unwrap();
        matvec_t(&d_out, &w, &mut d_input, 3, 3).unwrap();
        assert_eq!(d_input.to_vec().unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn cpu_matvec_t_general_2x3() {
        // W = [[1, 2, 3], [4, 5, 6]], d_out = [1, 1]
        // W^T·d_out = [[1+4], [2+5], [3+6]] = [5, 7, 9]
        let d_out = Tensor::<Cpu>::from_slice(&[1.0, 1.0]).unwrap();
        let w = Tensor::<Cpu>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let mut d_input = Tensor::<Cpu>::zeros(3).unwrap();
        matvec_t(&d_out, &w, &mut d_input, 2, 3).unwrap();
        assert_eq!(d_input.to_vec().unwrap(), vec![5.0, 7.0, 9.0]);
    }

    /// matvec then matvec_t with the same W gives the structure of
    /// the Linear forward + backward chain: `y = W·x; d_x = W^T·d_y`.
    /// Verifies the new resident kernel against CPU on identical W.
    #[cfg(feature = "rocm")]
    #[test]
    fn cpu_and_rocm_matvec_t_match() {
        let out_dim = 4;
        let in_dim = 5;
        let weight: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| (i as f32 - 9.0) * 0.1).collect();
        let d_out: Vec<f32> = (0..out_dim).map(|i| (i as f32 + 1.0) * 0.5).collect();

        let cpu_y = run_matvec_t::<Cpu>(&d_out, &weight, out_dim, in_dim).unwrap();
        let rocm_y = match run_matvec_t::<Rocm>(&d_out, &weight, out_dim, in_dim) {
            Ok(v) => v,
            Err(BackendError::Runtime(msg)) if msg.contains("hipMalloc") => return,
            Err(BackendError::Unsupported { .. }) => return,
            Err(e) => panic!("rocm matvec_t failed: {:?}", e),
        };
        for (c, r) in cpu_y.iter().zip(&rocm_y) {
            assert!((c - r).abs() < 1e-4,
                "CPU and ROCm matvec_t disagree: cpu={} rocm={}", c, r);
        }
    }

    #[cfg(all(test, feature = "rocm"))]
    fn run_matvec_t<D: Device>(
        d_out: &[f32], weight: &[f32], out_dim: usize, in_dim: usize,
    ) -> Result<Vec<f32>, BackendError> {
        let do_t = Tensor::<D>::from_slice(d_out)?;
        let w_t = Tensor::<D>::from_slice(weight)?;
        let mut di_t = Tensor::<D>::zeros(in_dim)?;
        matvec_t(&do_t, &w_t, &mut di_t, out_dim, in_dim)?;
        di_t.to_vec()
    }

    #[test]
    fn mlp_cpu_smoke() {
        let w1 = vec![1.0, 0.0, 0.0, 1.0];   // 2×2 identity
        let b1 = vec![0.0, 0.0];
        let w2 = vec![1.0, 1.0];             // 1×2: sum
        let b2 = vec![0.0];
        let x = vec![3.0, 4.0];
        let y = mlp::<Cpu>(&w1, &b1, &w2, &b2, &x, 2, 2, 1).unwrap();
        // h = x = [3,4]; y = [3+4] = [7]
        assert_eq!(y, vec![7.0]);
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn cpu_and_rocm_mlp_match() {
        let w1: Vec<f32> = (0..6).map(|i| (i as f32 - 2.5) * 0.1).collect();
        let b1: Vec<f32> = vec![0.1, -0.2];
        let w2: Vec<f32> = vec![0.5, -0.5];
        let b2: Vec<f32> = vec![0.05];
        let x = vec![0.4, -0.3, 0.7];

        let cpu_y  = mlp::<Cpu>(&w1, &b1, &w2, &b2, &x, 3, 2, 1).unwrap();
        let rocm_y = match mlp::<Rocm>(&w1, &b1, &w2, &b2, &x, 3, 2, 1) {
            Ok(v) => v,
            Err(BackendError::Runtime(msg)) if msg.contains("hipMalloc") => return,
            Err(e) => panic!("rocm MLP failed: {:?}", e),
        };
        for (c, r) in cpu_y.iter().zip(&rocm_y) {
            assert!((c - r).abs() < 1e-4,
                "CPU and ROCm MLP disagree: cpu={} rocm={}", c, r);
        }
    }

    // ─── shape ───────────────────────────────────────────

    #[test]
    fn cpu_tensor_default_shape_is_rank_1() {
        let t = Tensor::<Cpu>::from_slice(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(t.shape(), &[4]);
        assert_eq!(t.rank(), 1);
        assert!(t.shape_matches(&[4]));
        assert!(!t.shape_matches(&[2, 2]));
    }

    #[test]
    fn cpu_tensor_explicit_shape_construction() {
        let t = Tensor::<Cpu>::from_slice_shape(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
        ).unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.rank(), 2);
    }

    #[test]
    fn cpu_tensor_from_slice_shape_validates_product() {
        // 5 elements but shape [2, 3] = 6 — should error.
        match Tensor::<Cpu>::from_slice_shape(&[0.0; 5], &[2, 3]) {
            Err(BackendError::Runtime(_)) => {}
            other => panic!("expected Runtime error, got {:?}", other.map(|_| "Ok")),
        }
    }

    #[test]
    fn cpu_tensor_reshape_preserves_data() {
        let t = Tensor::<Cpu>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let t = t.reshape(&[2, 3]).unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn cpu_tensor_reshape_validates_product() {
        let t = Tensor::<Cpu>::from_slice(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        match t.reshape(&[3, 3]) {
            Err(BackendError::Runtime(_)) => {}
            other => panic!("expected reshape error, got {:?}", other.map(|_| "Ok")),
        }
    }

    #[test]
    fn cpu_linear_weight_carries_rank_2_shape() {
        let lin = Linear::<Cpu>::from_host(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[0.0, 0.0],
            3, 2,
        ).unwrap();
        assert_eq!(lin.weight.shape(), &[2, 3]); // [out_dim, in_dim]
        assert_eq!(lin.bias.shape(), &[2]);
    }

    // ─── outer_product_acc ───────────────────────────────

    #[test]
    fn cpu_outer_product_acc_first_call_overwrites_zero_init() {
        let a = Tensor::<Cpu>::from_slice(&[1.0, 2.0, 3.0]).unwrap();
        let b = Tensor::<Cpu>::from_slice(&[10.0, 20.0]).unwrap();
        let mut accum = Tensor::<Cpu>::zeros(6).unwrap();
        outer_product_acc(&a, &b, &mut accum, 3, 2).unwrap();
        assert_eq!(accum.to_vec().unwrap(), vec![10.0, 20.0, 20.0, 40.0, 30.0, 60.0]);
    }

    #[test]
    fn cpu_outer_product_acc_accumulates() {
        let a = Tensor::<Cpu>::from_slice(&[1.0, 2.0]).unwrap();
        let b = Tensor::<Cpu>::from_slice(&[3.0, 4.0, 5.0]).unwrap();
        let mut accum = Tensor::<Cpu>::from_slice(&[100.0; 6]).unwrap();
        outer_product_acc(&a, &b, &mut accum, 2, 3).unwrap();
        assert_eq!(accum.to_vec().unwrap(),
            vec![103.0, 104.0, 105.0, 106.0, 108.0, 110.0]);
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn cpu_and_rocm_outer_product_acc_match() {
        let m = 4; let n = 5;
        let a: Vec<f32> = (0..m).map(|i| (i as f32) * 0.3 - 0.5).collect();
        let b: Vec<f32> = (0..n).map(|i| (i as f32) * 0.2 + 0.1).collect();
        let init: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.05).collect();

        let cpu = run_outer::<Cpu>(&a, &b, &init, m, n).unwrap();
        let rocm = match run_outer::<Rocm>(&a, &b, &init, m, n) {
            Ok(v) => v,
            Err(BackendError::Runtime(msg)) if msg.contains("hipMalloc") => return,
            Err(BackendError::Unsupported { .. }) => return,
            Err(e) => panic!("rocm outer_product_acc failed: {:?}", e),
        };
        for (c, r) in cpu.iter().zip(&rocm) {
            assert!((c - r).abs() < 1e-4,
                "CPU and ROCm outer_product_acc disagree: cpu={} rocm={}", c, r);
        }
    }

    #[cfg(all(test, feature = "rocm"))]
    fn run_outer<D: Device>(
        a: &[f32], b: &[f32], init: &[f32], m: usize, n: usize,
    ) -> Result<Vec<f32>, BackendError> {
        let at = Tensor::<D>::from_slice(a)?;
        let bt = Tensor::<D>::from_slice(b)?;
        let mut accum = Tensor::<D>::from_slice(init)?;
        outer_product_acc(&at, &bt, &mut accum, m, n)?;
        accum.to_vec()
    }

    // ─── sgd_step ────────────────────────────────────────

    #[test]
    fn cpu_sgd_step_subtracts_lr_times_grad() {
        let mut w = Tensor::<Cpu>::from_slice(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let g = Tensor::<Cpu>::from_slice(&[0.5, 0.5, 0.5, 0.5]).unwrap();
        sgd_step(&mut w, &g, 4, 0.1).unwrap();
        let v = w.to_vec().unwrap();
        for (a, e) in v.iter().zip(&[0.95f32, 1.95, 2.95, 3.95]) {
            assert!((a - e).abs() < 1e-6);
        }
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn cpu_and_rocm_sgd_step_match() {
        let n = 8;
        let w_init: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5).collect();
        let g: Vec<f32> = (0..n).map(|i| (i as f32 - 4.0) * 0.1).collect();
        let lr = 0.05f32;

        let mut wc = Tensor::<Cpu>::from_slice(&w_init).unwrap();
        let gc = Tensor::<Cpu>::from_slice(&g).unwrap();
        sgd_step(&mut wc, &gc, n, lr).unwrap();
        let cpu = wc.to_vec().unwrap();

        let mut wr = match Tensor::<Rocm>::from_slice(&w_init) {
            Ok(v) => v,
            Err(BackendError::Runtime(msg)) if msg.contains("hipMalloc") => return,
            Err(e) => panic!("rocm tensor build failed: {:?}", e),
        };
        let gr = Tensor::<Rocm>::from_slice(&g).unwrap();
        sgd_step(&mut wr, &gr, n, lr).unwrap();
        let rocm = wr.to_vec().unwrap();

        for (c, r) in cpu.iter().zip(&rocm) {
            assert!((c - r).abs() < 1e-5,
                "CPU and ROCm sgd_step disagree: cpu={} rocm={}", c, r);
        }
    }

    // ─── Linear<D>::backward ─────────────────────────────

    #[test]
    fn cpu_linear_backward_matches_hand_calculation() {
        // W=[[1,2],[3,4]], b=[0,0], x=[1,1] → y=[3,7]
        // d_y=[1,1] →
        //   d_W += d_y ⊗ x = [[1,1],[1,1]]
        //   d_b += d_y     = [1,1]
        //   d_x  = W^T·d_y = [1+3, 2+4] = [4,6]
        let lin = Linear::<Cpu>::from_host(
            &[1.0, 2.0, 3.0, 4.0], &[0.0, 0.0], 2, 2,
        ).unwrap();
        let x = Tensor::<Cpu>::from_slice(&[1.0, 1.0]).unwrap();
        let d_y = Tensor::<Cpu>::from_slice(&[1.0, 1.0]).unwrap();
        let mut d_w = Tensor::<Cpu>::zeros(4).unwrap();
        let mut d_b = Tensor::<Cpu>::zeros(2).unwrap();
        let mut d_x = Tensor::<Cpu>::zeros(2).unwrap();
        lin.backward(&d_y, &x, &mut d_w, &mut d_b, &mut d_x).unwrap();
        assert_eq!(d_w.to_vec().unwrap(), vec![1.0, 1.0, 1.0, 1.0]);
        assert_eq!(d_b.to_vec().unwrap(), vec![1.0, 1.0]);
        assert_eq!(d_x.to_vec().unwrap(), vec![4.0, 6.0]);
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn cpu_and_rocm_linear_backward_match() {
        let in_d = 5; let out_d = 3;
        let w: Vec<f32> = (0..out_d * in_d).map(|i| (i as f32 - 7.0) * 0.1).collect();
        let b: Vec<f32> = vec![0.05, -0.1, 0.2];
        let x: Vec<f32> = (0..in_d).map(|i| (i as f32 - 2.0) * 0.3).collect();
        let d_y: Vec<f32> = (0..out_d).map(|i| (i as f32 + 1.0) * 0.5).collect();

        let (cpu_dw, cpu_db, cpu_dx) =
            run_linear_bwd::<Cpu>(&w, &b, &x, &d_y, in_d, out_d).unwrap();
        let (rocm_dw, rocm_db, rocm_dx) =
            match run_linear_bwd::<Rocm>(&w, &b, &x, &d_y, in_d, out_d) {
                Ok(t) => t,
                Err(BackendError::Runtime(msg)) if msg.contains("hipMalloc") => return,
                Err(BackendError::Unsupported { .. }) => return,
                Err(e) => panic!("rocm Linear::backward failed: {:?}", e),
            };
        assert_close("d_W", &cpu_dw, &rocm_dw, 1e-4);
        assert_close("d_b", &cpu_db, &rocm_db, 1e-4);
        assert_close("d_x", &cpu_dx, &rocm_dx, 1e-4);
    }

    #[cfg(all(test, feature = "rocm"))]
    fn run_linear_bwd<D: Device>(
        w: &[f32], b: &[f32], x: &[f32], d_y: &[f32],
        in_d: usize, out_d: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), BackendError> {
        let lin = Linear::<D>::from_host(w, b, in_d, out_d)?;
        let xt = Tensor::<D>::from_slice(x)?;
        let dyt = Tensor::<D>::from_slice(d_y)?;
        let mut dw = Tensor::<D>::zeros(out_d * in_d)?;
        let mut db = Tensor::<D>::zeros(out_d)?;
        let mut dx = Tensor::<D>::zeros(in_d)?;
        lin.backward(&dyt, &xt, &mut dw, &mut db, &mut dx)?;
        Ok((dw.to_vec()?, db.to_vec()?, dx.to_vec()?))
    }

    #[cfg(all(test, feature = "rocm"))]
    fn assert_close(label: &str, a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "{}: length mismatch", label);
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!((x - y).abs() < tol,
                "{} disagree at index {}: cpu={} rocm={}", label, i, x, y);
        }
    }

    // ─── End-to-end: Linear regression converges on both devices ──

    /// Train a `Linear<D>` to fit a simple known relationship.
    /// Synthetic data: `y = W*·x + b*` for fixed teacher parameters
    /// `(W*, b*)`. Initialise student `Linear<D>` at zero, run SGD,
    /// loss must drop below `tol` within `n_steps`. Same code on
    /// Cpu and Rocm — that's the whole point of Path C.
    fn train_linear_to_fit<D: Device>(
        teacher_w: &[f32], teacher_b: &[f32],
        in_d: usize, out_d: usize,
        n_steps: usize, lr: f32,
    ) -> Result<f32, BackendError> {
        // Student starts at zero (W=0, b=0). Inputs are deterministic.
        let zero_w = vec![0.0f32; out_d * in_d];
        let zero_b = vec![0.0f32; out_d];
        let mut lin = Linear::<D>::from_host(&zero_w, &zero_b, in_d, out_d)?;
        let teacher_lin = Linear::<D>::from_host(teacher_w, teacher_b, in_d, out_d)?;

        let mut d_w = Tensor::<D>::zeros(out_d * in_d)?;
        let mut d_b = Tensor::<D>::zeros(out_d)?;
        let mut d_x = Tensor::<D>::zeros(in_d)?;
        let mut d_pred = Tensor::<D>::zeros(out_d)?;

        let mut last_loss = f32::INFINITY;
        for step in 0..n_steps {
            // Generate one training sample deterministically.
            let xv: Vec<f32> = (0..in_d).map(|i| {
                ((step + i + 1) as f32 * 0.137).sin()
            }).collect();
            let x = Tensor::<D>::from_slice(&xv)?;
            // Teacher target (fresh buffer each step to keep code short).
            let y_target = teacher_lin.forward(&x)?;

            // Forward through student.
            let y = lin.forward(&x)?;

            // Compute MSE loss + d_pred. Reset d_w/d_b to zero each
            // step (true SGD; no batched accumulation here).
            let zero_dw = vec![0.0f32; out_d * in_d];
            let zero_db = vec![0.0f32; out_d];
            d_w.buffer_mut().copy_from_host(&zero_dw)?;
            d_b.buffer_mut().copy_from_host(&zero_db)?;

            last_loss = mse_loss(&y, &y_target, &mut d_pred, out_d)?;

            // Backward.
            lin.backward(&d_pred, &x, &mut d_w, &mut d_b, &mut d_x)?;

            // SGD step.
            sgd_step(&mut lin.weight, &d_w, out_d * in_d, lr)?;
            sgd_step(&mut lin.bias, &d_b, out_d, lr)?;
        }
        Ok(last_loss)
    }

    #[test]
    fn cpu_linear_regression_converges() {
        // Teacher: W=identity-scaled, b=small. Student should match.
        let teacher_w = vec![0.5, 0.0, 0.0, 0.5]; // 2×2 diag(0.5)
        let teacher_b = vec![0.1, -0.2];
        let final_loss = train_linear_to_fit::<Cpu>(
            &teacher_w, &teacher_b, 2, 2, 600, 0.05,
        ).unwrap();
        assert!(final_loss < 1e-3,
            "CPU Linear<Cpu> regression did not converge: final_loss={}", final_loss);
    }

    /// **The capability test.** Train `Linear<D>` end-to-end on
    /// both devices using identical code, identical hyperparameters,
    /// identical synthetic data. Both must converge below tol.
    #[cfg(feature = "rocm")]
    #[test]
    fn cpu_and_rocm_linear_regression_both_converge() {
        let teacher_w = vec![0.5, 0.0, 0.0, 0.5];
        let teacher_b = vec![0.1, -0.2];

        let cpu_loss = train_linear_to_fit::<Cpu>(
            &teacher_w, &teacher_b, 2, 2, 600, 0.05,
        ).unwrap();
        let rocm_loss = match train_linear_to_fit::<Rocm>(
            &teacher_w, &teacher_b, 2, 2, 600, 0.05,
        ) {
            Ok(v) => v,
            Err(BackendError::Runtime(msg)) if msg.contains("hipMalloc") => return,
            Err(BackendError::Unsupported { .. }) => return,
            Err(e) => panic!("rocm Linear regression failed: {:?}", e),
        };
        assert!(cpu_loss < 1e-3, "CPU did not converge: {}", cpu_loss);
        assert!(rocm_loss < 1e-3, "ROCm did not converge: {}", rocm_loss);
    }

    // ─── End-to-end MLP training (capability 4) ──────────

    /// Train a 2-layer MLP `(Linear → Linear)` end-to-end.
    /// Verifies that backward composes through stacked Linear
    /// layers without leaking gradient anywhere.
    fn train_mlp_to_fit<D: Device>(
        teacher_w1: &[f32], teacher_b1: &[f32],
        teacher_w2: &[f32], teacher_b2: &[f32],
        in_d: usize, hidden: usize, out_d: usize,
        n_steps: usize, lr: f32,
    ) -> Result<f32, BackendError> {
        // Student: zero init.
        let zw1 = vec![0.0f32; hidden * in_d];
        let zb1 = vec![0.0f32; hidden];
        let zw2 = vec![0.0f32; out_d * hidden];
        let zb2 = vec![0.0f32; out_d];
        let mut l1 = Linear::<D>::from_host(&zw1, &zb1, in_d, hidden)?;
        let mut l2 = Linear::<D>::from_host(&zw2, &zb2, hidden, out_d)?;

        let teacher1 = Linear::<D>::from_host(teacher_w1, teacher_b1, in_d, hidden)?;
        let teacher2 = Linear::<D>::from_host(teacher_w2, teacher_b2, hidden, out_d)?;

        // Reusable gradient buffers.
        let mut dw1 = Tensor::<D>::zeros(hidden * in_d)?;
        let mut db1 = Tensor::<D>::zeros(hidden)?;
        let mut dh = Tensor::<D>::zeros(hidden)?;
        let mut dw2 = Tensor::<D>::zeros(out_d * hidden)?;
        let mut db2 = Tensor::<D>::zeros(out_d)?;
        let mut d_pred = Tensor::<D>::zeros(out_d)?;
        let mut d_x = Tensor::<D>::zeros(in_d)?;

        let zw1z = zw1.clone();
        let zb1z = zb1.clone();
        let zw2z = zw2.clone();
        let zb2z = zb2.clone();

        let mut last_loss = f32::INFINITY;
        for step in 0..n_steps {
            let xv: Vec<f32> = (0..in_d).map(|i| {
                ((step + i + 1) as f32 * 0.137).sin()
            }).collect();
            let x = Tensor::<D>::from_slice(&xv)?;

            // Teacher target.
            let t1 = teacher1.forward(&x)?;
            let t_target = teacher2.forward(&t1)?;

            // Student forward (linear-only, no activation between
            // layers — keeps the math simple while still exercising
            // backward composition).
            let h = l1.forward(&x)?;
            let y = l2.forward(&h)?;

            // Reset gradient buffers for this step (true SGD).
            dw1.buffer_mut().copy_from_host(&zw1z)?;
            db1.buffer_mut().copy_from_host(&zb1z)?;
            dw2.buffer_mut().copy_from_host(&zw2z)?;
            db2.buffer_mut().copy_from_host(&zb2z)?;

            // Loss + d_pred at the output.
            last_loss = mse_loss(&y, &t_target, &mut d_pred, out_d)?;

            // Backward through l2 → produces d_h (gradient flowing into
            // l1's output, which IS the hidden activation).
            l2.backward(&d_pred, &h, &mut dw2, &mut db2, &mut dh)?;
            // Backward through l1 → consumes d_h.
            l1.backward(&dh, &x, &mut dw1, &mut db1, &mut d_x)?;

            // SGD on both layers.
            sgd_step(&mut l1.weight, &dw1, hidden * in_d, lr)?;
            sgd_step(&mut l1.bias,   &db1, hidden, lr)?;
            sgd_step(&mut l2.weight, &dw2, out_d * hidden, lr)?;
            sgd_step(&mut l2.bias,   &db2, out_d, lr)?;
        }
        Ok(last_loss)
    }

    /// Run `train_mlp_to_fit` for `n_steps` and report (initial_loss,
    /// final_loss). For the *capability* test we want "loss strictly
    /// decreases" — proving backward composes through stacked layers
    /// — without overcommitting to a specific convergence target on
    /// a degenerate (no-activation, 2-layer) topology.
    fn measure_mlp_loss_drop<D: Device>(
        teacher_w1: &[f32], teacher_b1: &[f32],
        teacher_w2: &[f32], teacher_b2: &[f32],
        in_d: usize, hidden: usize, out_d: usize,
        n_steps: usize, lr: f32,
    ) -> Result<(f32, f32), BackendError> {
        let zw1 = vec![0.0f32; hidden * in_d];
        let zb1 = vec![0.0f32; hidden];
        let zw2 = vec![0.0f32; out_d * hidden];
        let zb2 = vec![0.0f32; out_d];
        let mut l1 = Linear::<D>::from_host(&zw1, &zb1, in_d, hidden)?;
        let mut l2 = Linear::<D>::from_host(&zw2, &zb2, hidden, out_d)?;
        let teacher1 = Linear::<D>::from_host(teacher_w1, teacher_b1, in_d, hidden)?;
        let teacher2 = Linear::<D>::from_host(teacher_w2, teacher_b2, hidden, out_d)?;

        let mut dw1 = Tensor::<D>::zeros(hidden * in_d)?;
        let mut db1 = Tensor::<D>::zeros(hidden)?;
        let mut dh = Tensor::<D>::zeros(hidden)?;
        let mut dw2 = Tensor::<D>::zeros(out_d * hidden)?;
        let mut db2 = Tensor::<D>::zeros(out_d)?;
        let mut d_pred = Tensor::<D>::zeros(out_d)?;
        let mut d_x = Tensor::<D>::zeros(in_d)?;

        let mut initial_loss = 0.0f32;
        let mut last_loss = 0.0f32;

        for step in 0..n_steps {
            let xv: Vec<f32> = (0..in_d).map(|i| {
                ((step + i + 1) as f32 * 0.137).sin()
            }).collect();
            let x = Tensor::<D>::from_slice(&xv)?;
            let t1 = teacher1.forward(&x)?;
            let t_target = teacher2.forward(&t1)?;
            let h = l1.forward(&x)?;
            let y = l2.forward(&h)?;

            let zwd1 = vec![0.0f32; hidden * in_d];
            let zbd1 = vec![0.0f32; hidden];
            let zwd2 = vec![0.0f32; out_d * hidden];
            let zbd2 = vec![0.0f32; out_d];
            dw1.buffer_mut().copy_from_host(&zwd1)?;
            db1.buffer_mut().copy_from_host(&zbd1)?;
            dw2.buffer_mut().copy_from_host(&zwd2)?;
            db2.buffer_mut().copy_from_host(&zbd2)?;

            let loss = mse_loss(&y, &t_target, &mut d_pred, out_d)?;
            if step == 0 { initial_loss = loss; }
            last_loss = loss;

            l2.backward(&d_pred, &h, &mut dw2, &mut db2, &mut dh)?;
            l1.backward(&dh, &x, &mut dw1, &mut db1, &mut d_x)?;
            sgd_step(&mut l1.weight, &dw1, hidden * in_d, lr)?;
            sgd_step(&mut l1.bias, &db1, hidden, lr)?;
            sgd_step(&mut l2.weight, &dw2, out_d * hidden, lr)?;
            sgd_step(&mut l2.bias, &db2, out_d, lr)?;
        }
        Ok((initial_loss, last_loss))
    }

    #[test]
    fn cpu_mlp_loss_strictly_decreases() {
        // Capability test: backward composes through stacked layers.
        // Linear-only 2→2→1 is parameter-degenerate (many optimal
        // solutions), so we don't assert convergence to a specific
        // tolerance — only that loss DECREASES from the initial
        // forward to the final state. That alone proves backward,
        // gradient flow, and SGD step all compose correctly.
        let tw1 = vec![0.5, 0.0, 0.0, 0.5];
        let tb1 = vec![0.1, 0.0];
        let tw2 = vec![1.0, 1.0];
        let tb2 = vec![0.05];
        let (init, final_) = measure_mlp_loss_drop::<Cpu>(
            &tw1, &tb1, &tw2, &tb2, 2, 2, 1, 800, 0.03,
        ).unwrap();
        assert!(final_ < init * 0.5,
            "CPU MLP loss did not halve: init={} final={}", init, final_);
    }

    // ─── AdamW<D> ────────────────────────────────────────

    #[test]
    fn cpu_adamw_step_decreases_loss_on_quadratic() {
        // 1-param toy: minimise (w - 1)². Optimum at w=1.
        let mut w = Tensor::<Cpu>::from_slice(&[0.0]).unwrap();
        let mut opt = AdamW::<Cpu>::new(1).unwrap().with_lr(0.1);
        for _ in 0..200 {
            // grad of (w-1)² is 2(w-1).
            let wv = w.to_vec().unwrap();
            let g_v = vec![2.0 * (wv[0] - 1.0)];
            let g = Tensor::<Cpu>::from_slice(&g_v).unwrap();
            opt.step(&mut w, &g).unwrap();
        }
        let final_w = w.to_vec().unwrap()[0];
        assert!((final_w - 1.0).abs() < 1e-2, "AdamW failed to converge: w={}", final_w);
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn cpu_and_rocm_adamw_step_match() {
        // Identical sequence of (w_init, gradients, hyperparams) on
        // both devices must produce identical w trajectories.
        let n = 8;
        let w_init: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 - 0.4).collect();
        let g_seq: Vec<Vec<f32>> = (0..50).map(|step| {
            (0..n).map(|i| ((step + i) as f32 * 0.13).sin()).collect()
        }).collect();

        let cpu_final = run_adamw_seq::<Cpu>(&w_init, &g_seq).unwrap();
        let rocm_final = match run_adamw_seq::<Rocm>(&w_init, &g_seq) {
            Ok(v) => v,
            Err(BackendError::Runtime(msg)) if msg.contains("hipMalloc") => return,
            Err(BackendError::Unsupported { .. }) => return,
            Err(e) => panic!("rocm AdamW failed: {:?}", e),
        };
        for (c, r) in cpu_final.iter().zip(&rocm_final) {
            assert!((c - r).abs() < 1e-3,
                "CPU and ROCm AdamW disagree: cpu={} rocm={}", c, r);
        }
    }

    #[cfg(all(test, feature = "rocm"))]
    fn run_adamw_seq<D: Device>(
        w_init: &[f32], g_seq: &[Vec<f32>],
    ) -> Result<Vec<f32>, BackendError> {
        let n = w_init.len();
        let mut w = Tensor::<D>::from_slice(w_init)?;
        let mut opt = AdamW::<D>::new(n)?.with_lr(0.05);
        for g_step in g_seq {
            let g = Tensor::<D>::from_slice(g_step)?;
            opt.step(&mut w, &g)?;
        }
        w.to_vec()
    }

    // ─── layer_norm_train + layer_norm_bwd ───────────────

    #[test]
    fn cpu_layer_norm_train_and_bwd_roundtrip() {
        // Forward through layer_norm_train, then backward must
        // produce gradients consistent with forward output.
        let x = Tensor::<Cpu>::from_slice(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let gamma = Tensor::<Cpu>::from_slice(&[1.0, 1.0, 1.0, 1.0]).unwrap();
        let beta = Tensor::<Cpu>::from_slice(&[0.0, 0.0, 0.0, 0.0]).unwrap();
        let mut out = Tensor::<Cpu>::zeros(4).unwrap();
        let mut cache = Tensor::<Cpu>::zeros(2).unwrap();   // 2*n_rows = 2
        layer_norm_train(&x, &gamma, &beta, &mut out, &mut cache, 1, 4).unwrap();
        // backward
        let d_out = Tensor::<Cpu>::from_slice(&[1.0, 0.0, 0.0, -1.0]).unwrap();
        let mut d_x = Tensor::<Cpu>::zeros(4).unwrap();
        let mut d_gamma = Tensor::<Cpu>::zeros(4).unwrap();
        let mut d_beta = Tensor::<Cpu>::zeros(4).unwrap();
        layer_norm_bwd(&d_out, &x, &gamma, &cache,
            &mut d_x, &mut d_gamma, &mut d_beta, 1, 4).unwrap();
        // d_beta should equal d_out (sum across rows = single row).
        assert_eq!(d_beta.to_vec().unwrap(), vec![1.0, 0.0, 0.0, -1.0]);
        // d_x must have zero mean (LayerNorm backward preserves it).
        let dx = d_x.to_vec().unwrap();
        let mean_dx = dx.iter().sum::<f32>() / 4.0;
        assert!(mean_dx.abs() < 1e-5, "d_x mean should be ~0, got {}", mean_dx);
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn cpu_and_rocm_layer_norm_train_bwd_match() {
        let n_rows = 3; let n_cols = 4;
        let x_data: Vec<f32> = (0..n_rows * n_cols).map(|i| (i as f32 - 5.0) * 0.5).collect();
        let g_data: Vec<f32> = (0..n_cols).map(|i| 1.0 + i as f32 * 0.1).collect();
        let b_data: Vec<f32> = (0..n_cols).map(|i| i as f32 * 0.05 - 0.1).collect();
        let dy_data: Vec<f32> = (0..n_rows * n_cols).map(|i| (i as f32 - 5.0) * 0.2).collect();

        let cpu = run_ln_train_bwd::<Cpu>(&x_data, &g_data, &b_data, &dy_data, n_rows, n_cols).unwrap();
        let rocm = match run_ln_train_bwd::<Rocm>(&x_data, &g_data, &b_data, &dy_data, n_rows, n_cols) {
            Ok(t) => t,
            Err(BackendError::Runtime(msg)) if msg.contains("hipMalloc") => return,
            Err(BackendError::Unsupported { .. }) => return,
            Err(e) => panic!("rocm layer_norm_train+bwd failed: {:?}", e),
        };
        // out
        for (c, r) in cpu.0.iter().zip(&rocm.0) {
            assert!((c - r).abs() < 1e-3, "out: cpu={} rocm={}", c, r);
        }
        // d_x
        for (c, r) in cpu.1.iter().zip(&rocm.1) {
            assert!((c - r).abs() < 1e-3, "d_x: cpu={} rocm={}", c, r);
        }
        // d_gamma
        for (c, r) in cpu.2.iter().zip(&rocm.2) {
            assert!((c - r).abs() < 1e-3, "d_gamma: cpu={} rocm={}", c, r);
        }
        // d_beta
        for (c, r) in cpu.3.iter().zip(&rocm.3) {
            assert!((c - r).abs() < 1e-4, "d_beta: cpu={} rocm={}", c, r);
        }
    }

    #[cfg(all(test, feature = "rocm"))]
    fn run_ln_train_bwd<D: Device>(
        x: &[f32], gamma: &[f32], beta: &[f32], d_out: &[f32],
        n_rows: usize, n_cols: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>), BackendError> {
        let xt = Tensor::<D>::from_slice(x)?;
        let gt = Tensor::<D>::from_slice(gamma)?;
        let bt = Tensor::<D>::from_slice(beta)?;
        let dyt = Tensor::<D>::from_slice(d_out)?;
        let mut out = Tensor::<D>::zeros(n_rows * n_cols)?;
        let mut cache = Tensor::<D>::zeros(2 * n_rows)?;
        layer_norm_train(&xt, &gt, &bt, &mut out, &mut cache, n_rows, n_cols)?;
        let mut dx = Tensor::<D>::zeros(n_rows * n_cols)?;
        let mut dg = Tensor::<D>::zeros(n_cols)?;
        let mut db = Tensor::<D>::zeros(n_cols)?;
        layer_norm_bwd(&dyt, &xt, &gt, &cache, &mut dx, &mut dg, &mut db, n_rows, n_cols)?;
        Ok((out.to_vec()?, dx.to_vec()?, dg.to_vec()?, db.to_vec()?))
    }

    // ─── silu_bwd ────────────────────────────────────────

    #[test]
    fn cpu_silu_bwd_zero_at_minus_infinity_one_at_plus_infinity() {
        // silu'(x) → 0 as x → -∞, → 1 as x → +∞.
        let x = Tensor::<Cpu>::from_slice(&[-100.0, 0.0, 100.0]).unwrap();
        let d_out = Tensor::<Cpu>::from_slice(&[1.0, 1.0, 1.0]).unwrap();
        let mut d_x = Tensor::<Cpu>::zeros(3).unwrap();
        silu_bwd(&d_out, &x, &mut d_x).unwrap();
        let v = d_x.to_vec().unwrap();
        assert!(v[0].abs() < 1e-3, "silu'(-100) ≈ 0, got {}", v[0]);
        // silu'(0) = 0.5
        assert!((v[1] - 0.5).abs() < 1e-3, "silu'(0) = 0.5, got {}", v[1]);
        assert!((v[2] - 1.0).abs() < 1e-3, "silu'(100) ≈ 1, got {}", v[2]);
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn cpu_and_rocm_silu_bwd_match() {
        let n = 16;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 - 8.0) * 0.5).collect();
        let d_out: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.1).collect();

        let cpu = run_silu_bwd::<Cpu>(&d_out, &x).unwrap();
        let rocm = match run_silu_bwd::<Rocm>(&d_out, &x) {
            Ok(v) => v,
            Err(BackendError::Runtime(msg)) if msg.contains("hipMalloc") => return,
            Err(BackendError::Unsupported { .. }) => return,
            Err(e) => panic!("rocm silu_bwd failed: {:?}", e),
        };
        for (c, r) in cpu.iter().zip(&rocm) {
            assert!((c - r).abs() < 1e-3,
                "CPU and ROCm silu_bwd disagree: cpu={} rocm={}", c, r);
        }
    }

    #[cfg(all(test, feature = "rocm"))]
    fn run_silu_bwd<D: Device>(d_out: &[f32], x: &[f32]) -> Result<Vec<f32>, BackendError> {
        let do_t = Tensor::<D>::from_slice(d_out)?;
        let xt = Tensor::<D>::from_slice(x)?;
        let mut dx = Tensor::<D>::zeros(x.len())?;
        silu_bwd(&do_t, &xt, &mut dx)?;
        dx.to_vec()
    }

    // ─── glu_bwd ─────────────────────────────────────────

    #[test]
    fn cpu_glu_bwd_hand_calculation() {
        // x = [v=1, v=2, g=0, g=10] (1 row, half=2). σ(0)=0.5, σ(10)≈1.
        // Forward: y = [1·0.5, 2·1] = [0.5, 2.0].
        // For d_out = [1, 1]:
        //   d_value[0] = d_out[0]·σ(g[0])     = 1·0.5     = 0.5
        //   d_value[1] = d_out[1]·σ(g[1])     ≈ 1·1.0     = 1.0
        //   d_gate[0]  = d_out[0]·v[0]·σ·(1-σ)= 1·1·0.5·0.5 = 0.25
        //   d_gate[1]  = d_out[1]·v[1]·σ·(1-σ)≈ 1·2·1·0    = 0.0
        let x = Tensor::<Cpu>::from_slice(&[1.0, 2.0, 0.0, 10.0]).unwrap();
        let d_out = Tensor::<Cpu>::from_slice(&[1.0, 1.0]).unwrap();
        let mut d_x = Tensor::<Cpu>::zeros(4).unwrap();
        glu_bwd(&d_out, &x, &mut d_x, 1, 2).unwrap();
        let v = d_x.to_vec().unwrap();
        assert!((v[0] - 0.5).abs() < 1e-3);
        assert!((v[1] - 1.0).abs() < 1e-3);
        assert!((v[2] - 0.25).abs() < 1e-3);
        assert!(v[3].abs() < 1e-3);
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn cpu_and_rocm_glu_bwd_match() {
        let n_rows = 4; let half = 6;
        let x_data: Vec<f32> = (0..n_rows * 2 * half).map(|i| (i as f32 - 12.0) * 0.1).collect();
        let dy_data: Vec<f32> = (0..n_rows * half).map(|i| (i as f32 + 1.0) * 0.05).collect();

        let cpu = run_glu_bwd::<Cpu>(&dy_data, &x_data, n_rows, half).unwrap();
        let rocm = match run_glu_bwd::<Rocm>(&dy_data, &x_data, n_rows, half) {
            Ok(v) => v,
            Err(BackendError::Runtime(msg)) if msg.contains("hipMalloc") => return,
            Err(BackendError::Unsupported { .. }) => return,
            Err(e) => panic!("rocm glu_bwd failed: {:?}", e),
        };
        for (c, r) in cpu.iter().zip(&rocm) {
            assert!((c - r).abs() < 1e-3,
                "CPU and ROCm glu_bwd disagree: cpu={} rocm={}", c, r);
        }
    }

    #[cfg(all(test, feature = "rocm"))]
    fn run_glu_bwd<D: Device>(
        d_out: &[f32], x: &[f32], n_rows: usize, half: usize,
    ) -> Result<Vec<f32>, BackendError> {
        let do_t = Tensor::<D>::from_slice(d_out)?;
        let xt = Tensor::<D>::from_slice(x)?;
        let mut dx = Tensor::<D>::zeros(x.len())?;
        glu_bwd(&do_t, &xt, &mut dx, n_rows, half)?;
        dx.to_vec()
    }

    // ─── SuperLinear<D>::backward ────────────────────────

    #[test]
    fn cpu_super_linear_backward_shapes_and_accumulation() {
        // 2 neurons, in_per=3, out_per=2. W = identity-ish (per neuron):
        //   W[0] = [[1,0,0],[0,1,0]],  W[1] = [[1,0,0],[0,1,0]]
        // x[0]=[1,2,3], x[1]=[10,20,30]; d_out[0]=[1,1], d_out[1]=[1,1].
        // Then dx[n] = W[n]^T · d_out[n] = [1, 1, 0] for both.
        // dW[n] += d_out[n] ⊗ x[n] = [[1,2,3],[1,2,3]] for n=0
        //                          = [[10,20,30],[10,20,30]] for n=1
        let n = 2; let in_per = 3; let out_per = 2;
        let weights: Vec<f32> = (0..n).flat_map(|_| {
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        }).collect();
        let biases = vec![0.0f32; n * out_per];
        let sl = SuperLinear::<Cpu>::from_host(
            &weights, &biases, n, in_per, out_per,
        ).unwrap();
        let trace = Tensor::<Cpu>::from_slice(&[1.0, 2.0, 3.0, 10.0, 20.0, 30.0]).unwrap();
        let d_out = Tensor::<Cpu>::from_slice(&[1.0, 1.0, 1.0, 1.0]).unwrap();
        let mut d_w = Tensor::<Cpu>::zeros(n * out_per * in_per).unwrap();
        let mut d_b = Tensor::<Cpu>::zeros(n * out_per).unwrap();
        let mut d_trace = Tensor::<Cpu>::zeros(n * in_per).unwrap();
        sl.backward(&d_out, &trace, &mut d_w, &mut d_b, &mut d_trace).unwrap();

        // d_trace[n] = [1, 1, 0]
        assert_eq!(d_trace.to_vec().unwrap(),
            vec![1.0, 1.0, 0.0, 1.0, 1.0, 0.0]);
        // d_b += d_out
        assert_eq!(d_b.to_vec().unwrap(), vec![1.0, 1.0, 1.0, 1.0]);
        // d_w[n=0] = [[1,2,3],[1,2,3]] flat = [1,2,3,1,2,3]
        // d_w[n=1] = [[10,20,30],[10,20,30]] flat
        let expected: Vec<f32> = vec![
            1.0, 2.0, 3.0, 1.0, 2.0, 3.0,
            10.0, 20.0, 30.0, 10.0, 20.0, 30.0,
        ];
        assert_eq!(d_w.to_vec().unwrap(), expected);
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn cpu_and_rocm_super_linear_backward_match() {
        let n = 3; let in_per = 4; let out_per = 2;
        let weights: Vec<f32> = (0..n * out_per * in_per)
            .map(|i| (i as f32 - 12.0) * 0.05).collect();
        let biases: Vec<f32> = (0..n * out_per).map(|i| i as f32 * 0.1).collect();
        let trace: Vec<f32> = (0..n * in_per).map(|i| (i as f32) * 0.25).collect();
        let d_out: Vec<f32> = (0..n * out_per).map(|i| (i as f32 + 1.0) * 0.3).collect();

        let (cpu_dw, cpu_db, cpu_dt) = run_super_bwd::<Cpu>(
            &weights, &biases, &trace, &d_out, n, in_per, out_per,
        ).unwrap();
        let (rocm_dw, rocm_db, rocm_dt) =
            match run_super_bwd::<Rocm>(&weights, &biases, &trace, &d_out, n, in_per, out_per) {
                Ok(t) => t,
                Err(BackendError::Runtime(msg)) if msg.contains("hipMalloc") => return,
                Err(BackendError::Unsupported { .. }) => return,
                Err(e) => panic!("rocm SuperLinear::backward failed: {:?}", e),
            };
        assert_close("d_W", &cpu_dw, &rocm_dw, 1e-3);
        assert_close("d_b", &cpu_db, &rocm_db, 1e-4);
        assert_close("d_trace", &cpu_dt, &rocm_dt, 1e-3);
    }

    #[cfg(all(test, feature = "rocm"))]
    fn run_super_bwd<D: Device>(
        weights: &[f32], biases: &[f32], trace: &[f32], d_out: &[f32],
        n: usize, in_per: usize, out_per: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), BackendError> {
        let sl = SuperLinear::<D>::from_host(weights, biases, n, in_per, out_per)?;
        let trace_t = Tensor::<D>::from_slice(trace)?;
        let do_t = Tensor::<D>::from_slice(d_out)?;
        let mut dw = Tensor::<D>::zeros(n * out_per * in_per)?;
        let mut db = Tensor::<D>::zeros(n * out_per)?;
        let mut dt = Tensor::<D>::zeros(n * in_per)?;
        sl.backward(&do_t, &trace_t, &mut dw, &mut db, &mut dt)?;
        Ok((dw.to_vec()?, db.to_vec()?, dt.to_vec()?))
    }

    // ─── Typed deep block: Linear → SuperLinear → SiLU → Linear ──
    //
    // Exercises the full set of typed primitives: matvec (forward
    // through Linear), super_linear_fwd (per-neuron NLM forward),
    // silu (activation), then the corresponding backward chain.
    // This is the shape of a CTM cell without the temporal recurrence
    // — proving the cascade is sufficient for real CTM-style
    // computations before the brain-level container ports.

    /// Generate a synthetic "deep block" target using a fresh teacher
    /// with the same architecture. Train a zero-init student against
    /// the teacher's outputs. Both teacher and student run on the
    /// same `D` so the parity is end-to-end.
    fn measure_deep_block_loss_drop<D: Device>(
        in_d: usize, n_neurons: usize, out_d: usize,
        teacher_seed: u64,
        n_steps: usize, lr: f32,
    ) -> Result<(f32, f32), BackendError> {
        // Block dims: x[in_d] → l1 → trace[n_neurons × in_per] →
        // sl(in_per→out_per) → silu → l2[n_neurons × out_per → out_d]
        // For simplicity: in_per = in_d (one row), out_per = 1.
        let in_per = in_d;
        let out_per = 1;

        // Deterministic teacher params from seed. Using ranges
        // calibrated to produce O(1) outputs given sin-based inputs:
        // weights in [-1, 1], biases nonzero so the teacher's output
        // doesn't collapse to ~0 with zero-init student.
        let mut s = teacher_seed.wrapping_mul(2654435761);
        let mut nxt = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 32) as i32 as f32) / (i32::MAX as f32)  // [-1, 1]
        };
        let tw_l1: Vec<f32> = (0..n_neurons * in_per * in_d).map(|_| nxt()).collect();
        let tb_l1: Vec<f32> = (0..n_neurons * in_per).map(|_| nxt() * 0.3).collect();
        let tw_sl: Vec<f32> = (0..n_neurons * out_per * in_per).map(|_| nxt()).collect();
        let tb_sl: Vec<f32> = (0..n_neurons * out_per).map(|_| nxt() * 0.3).collect();
        let tw_l2: Vec<f32> = (0..out_d * (n_neurons * out_per)).map(|_| nxt()).collect();
        let tb_l2: Vec<f32> = (0..out_d).map(|_| nxt() * 0.5).collect();

        // Student: zero init.
        let zw_l1 = vec![0.0f32; n_neurons * in_per * in_d];
        let zb_l1 = vec![0.0f32; n_neurons * in_per];
        let zw_sl = vec![0.0f32; n_neurons * out_per * in_per];
        let zb_sl = vec![0.0f32; n_neurons * out_per];
        let zw_l2 = vec![0.0f32; out_d * (n_neurons * out_per)];
        let zb_l2 = vec![0.0f32; out_d];

        let mut s_l1 = Linear::<D>::from_host(&zw_l1, &zb_l1, in_d, n_neurons * in_per)?;
        let mut s_sl = SuperLinear::<D>::from_host(&zw_sl, &zb_sl, n_neurons, in_per, out_per)?;
        let mut s_l2 = Linear::<D>::from_host(&zw_l2, &zb_l2, n_neurons * out_per, out_d)?;

        let t_l1 = Linear::<D>::from_host(&tw_l1, &tb_l1, in_d, n_neurons * in_per)?;
        let t_sl = SuperLinear::<D>::from_host(&tw_sl, &tb_sl, n_neurons, in_per, out_per)?;
        let t_l2 = Linear::<D>::from_host(&tw_l2, &tb_l2, n_neurons * out_per, out_d)?;

        let n_post_silu = n_neurons * out_per;

        let mut dw_l1 = Tensor::<D>::zeros(n_neurons * in_per * in_d)?;
        let mut db_l1 = Tensor::<D>::zeros(n_neurons * in_per)?;
        let mut dw_sl = Tensor::<D>::zeros(n_neurons * out_per * in_per)?;
        let mut db_sl = Tensor::<D>::zeros(n_neurons * out_per)?;
        let mut dw_l2 = Tensor::<D>::zeros(out_d * n_post_silu)?;
        let mut db_l2 = Tensor::<D>::zeros(out_d)?;

        let mut d_y = Tensor::<D>::zeros(out_d)?;
        let mut d_post_silu = Tensor::<D>::zeros(n_post_silu)?;
        let mut d_pre_silu = Tensor::<D>::zeros(n_post_silu)?;
        let mut d_trace = Tensor::<D>::zeros(n_neurons * in_per)?;
        let mut d_x = Tensor::<D>::zeros(in_d)?;

        let mut initial_loss = 0.0f32;
        let mut final_loss = 0.0f32;

        for step in 0..n_steps {
            let xv: Vec<f32> = (0..in_d).map(|i| {
                ((step + i + 1) as f32 * 0.137).sin()
            }).collect();
            let x = Tensor::<D>::from_slice(&xv)?;

            // Teacher target.
            let t1 = t_l1.forward(&x)?;
            let t2 = t_sl.forward(&t1)?;
            let mut t2_silu = Tensor::<D>::zeros(n_post_silu)?;
            silu(&t2, &mut t2_silu)?;
            let target = t_l2.forward(&t2_silu)?;

            // Student forward (saving intermediates for backward).
            let trace = s_l1.forward(&x)?;          // [n_neurons*in_per]
            let pre_silu = s_sl.forward(&trace)?;    // [n_neurons*out_per]
            let mut post_silu = Tensor::<D>::zeros(n_post_silu)?;
            silu(&pre_silu, &mut post_silu)?;
            let y = s_l2.forward(&post_silu)?;       // [out_d]

            // Reset gradient buffers.
            dw_l1.buffer_mut().copy_from_host(&zw_l1)?;
            db_l1.buffer_mut().copy_from_host(&zb_l1)?;
            dw_sl.buffer_mut().copy_from_host(&zw_sl)?;
            db_sl.buffer_mut().copy_from_host(&zb_sl)?;
            dw_l2.buffer_mut().copy_from_host(&zw_l2)?;
            db_l2.buffer_mut().copy_from_host(&zb_l2)?;

            // Loss.
            let loss = mse_loss(&y, &target, &mut d_y, out_d)?;
            if step == 0 { initial_loss = loss; }
            final_loss = loss;

            // Backward through l2 → d_post_silu
            s_l2.backward(&d_y, &post_silu, &mut dw_l2, &mut db_l2, &mut d_post_silu)?;
            // Backward through silu → d_pre_silu
            silu_bwd(&d_post_silu, &pre_silu, &mut d_pre_silu)?;
            // Backward through SuperLinear → d_trace
            s_sl.backward(&d_pre_silu, &trace, &mut dw_sl, &mut db_sl, &mut d_trace)?;
            // Backward through l1 → d_x
            s_l1.backward(&d_trace, &x, &mut dw_l1, &mut db_l1, &mut d_x)?;

            // SGD on all params.
            sgd_step(&mut s_l1.weight, &dw_l1, n_neurons * in_per * in_d, lr)?;
            sgd_step(&mut s_l1.bias, &db_l1, n_neurons * in_per, lr)?;
            sgd_step(&mut s_sl.weights, &dw_sl, n_neurons * out_per * in_per, lr)?;
            sgd_step(&mut s_sl.biases, &db_sl, n_neurons * out_per, lr)?;
            sgd_step(&mut s_l2.weight, &dw_l2, out_d * n_post_silu, lr)?;
            sgd_step(&mut s_l2.bias, &db_l2, out_d, lr)?;
        }
        Ok((initial_loss, final_loss))
    }

    #[test]
    fn cpu_typed_deep_block_loss_strictly_decreases() {
        let (init, final_) = measure_deep_block_loss_drop::<Cpu>(
            3, 4, 2,        // in_d=3, n_neurons=4, out_d=2
            0xCAFE_F00D,    // teacher seed
            3000, 0.005,    // small lr — deep network needs care
        ).unwrap();
        assert!(final_ < init * 0.7,
            "CPU deep block did not reduce loss by 30%: init={} final={}", init, final_);
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn cpu_and_rocm_typed_deep_block_both_train() {
        // Capability proof: full primitive cascade
        // (Linear<D> + SuperLinear<D> + silu_fwd/bwd + sgd_step)
        // composes into a trainable network on BOTH devices, using
        // identical code parameterised only by `D`.
        let (cpu_init, cpu_final) = measure_deep_block_loss_drop::<Cpu>(
            3, 4, 2, 0xCAFE_F00D, 3000, 0.005,
        ).unwrap();
        let (rocm_init, rocm_final) =
            match measure_deep_block_loss_drop::<Rocm>(3, 4, 2, 0xCAFE_F00D, 3000, 0.005) {
                Ok(t) => t,
                Err(BackendError::Runtime(msg)) if msg.contains("hipMalloc") => return,
                Err(BackendError::Unsupported { .. }) => return,
                Err(e) => panic!("rocm typed_deep_block failed: {:?}", e),
            };
        assert!((cpu_init - rocm_init).abs() < 1e-3,
            "initial loss disagrees: cpu={} rocm={}", cpu_init, rocm_init);
        assert!(cpu_final < cpu_init * 0.7,
            "CPU deep block did not reduce 30%: init={} final={}", cpu_init, cpu_final);
        assert!(rocm_final < rocm_init * 0.7,
            "ROCm deep block did not reduce 30%: init={} final={}", rocm_init, rocm_final);
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn cpu_and_rocm_mlp_loss_drop_match() {
        // Both devices must produce loss-drop trajectories that
        // agree to tolerance. This is the cross-device composition
        // proof: same Linear<D>::backward chain works on both.
        let tw1 = vec![0.5, 0.0, 0.0, 0.5];
        let tb1 = vec![0.1, 0.0];
        let tw2 = vec![1.0, 1.0];
        let tb2 = vec![0.05];

        let (cpu_init, cpu_final) = measure_mlp_loss_drop::<Cpu>(
            &tw1, &tb1, &tw2, &tb2, 2, 2, 1, 800, 0.03,
        ).unwrap();
        let (rocm_init, rocm_final) =
            match measure_mlp_loss_drop::<Rocm>(&tw1, &tb1, &tw2, &tb2, 2, 2, 1, 800, 0.03) {
                Ok(t) => t,
                Err(BackendError::Runtime(msg)) if msg.contains("hipMalloc") => return,
                Err(BackendError::Unsupported { .. }) => return,
                Err(e) => panic!("rocm MLP regression failed: {:?}", e),
            };
        // Initial loss must agree (forward at zero-init is deterministic).
        assert!((cpu_init - rocm_init).abs() < 1e-4,
            "initial loss disagrees: cpu={} rocm={}", cpu_init, rocm_init);
        // After 800 steps both must drop substantially. The two
        // trajectories may diverge slightly because hipBLAS SGEMV
        // and rayon-CPU matvec aren't bit-equal, so we don't require
        // matching final losses — only that BOTH train.
        assert!(cpu_final < cpu_init * 0.5,
            "CPU loss did not halve: init={} final={}", cpu_init, cpu_final);
        assert!(rocm_final < rocm_init * 0.5,
            "ROCm loss did not halve: init={} final={}", rocm_init, rocm_final);
    }
}
