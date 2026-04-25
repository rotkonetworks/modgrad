//! Device-resident SwiGLU FFN block.
//!
//! Mirrors [`crate::FfnBlock`] / [`crate::FfnWeights`] but keeps weights
//! and intermediate activations on the GPU across the whole forward
//! pass. Zero PCIe transfers per dispatch — uploads happen once
//! (in `from_*`) and `sync_weights_from` re-uploads after an optimizer
//! step.
//!
//! # Composition
//!
//! Each [`SwigluResident`] block wraps three [`LinearResident`]s
//! (`gate`, `up`, `down`) plus device buffers for the pre-norm LayerNorm
//! parameters (`gamma`, `beta`). The forward composes:
//!
//!   1. `layer_norm_resident(x → normed)`         — pre-norm
//!   2. `gate.forward(normed → gate_out)`         — `[N × d] @ gate.W^T`
//!   3. `up.forward  (normed → up_out)`           — `[N × d] @ up.W^T`
//!   4. `activation_resident(Silu, gate_out)`     — in-place SiLU compose
//!   5. `op_tensor_resident(Mul, gate_out, up_out → hidden)`
//!   6. `down.forward(hidden → out)`
//!
//! Steps 2–3 and 4–5 are the win: gate+up share their input (one read of
//! `normed`), and SiLU+Mul stay device-resident through the MIOpen
//! activation/op-tensor chain.
//!
//! # Why a parallel struct, not a feature flag inside `FfnBlock`?
//!
//! The host `FfnBlock` is `Serialize + SchemaRead + SchemaWrite`. Adding
//! a `HipBuffer` field would either fail to compile under `--no-default-features`
//! (HipBuffer doesn't exist without `rocm`) or require ad-hoc skip
//! attributes everywhere. The pattern matches `LinearResident` next to
//! `Linear` in `modgrad-compute::neuron`: parallel types, additive only,
//! no widening of the host surface.
//!
//! # Lifecycle
//!
//! ```ignore
//! let host_block = FfnBlock::new(d_model, hidden);
//! let mut resident = FfnBlockResident::from_block(&host_block)?;
//!
//! // Per inference / training step:
//! resident.forward(&batch, n_tokens, &x_dev, &mut out_dev)?;
//!
//! // After optimizer step:
//! resident.sync_weights_from(&host_block)?;
//! ```
//!
//! Only available with `--features rocm`.

#![cfg(feature = "rocm")]

use crate::FfnBlock;
use modgrad_compute::backend::{GpuVec, ResidencyError};
use modgrad_compute::neuron::LinearResident;
use modgrad_device::backend::{HipBatch, HipBuffer};
use modgrad_device::backend::op::{ActivationMode, BinaryOpKind};
use modgrad_device::backend::ops::{
    activation_backward_resident, activation_resident,
    layer_norm_backward_resident, layer_norm_resident, matvec_resident,
    op_tensor_resident,
};

/// Device-resident SwiGLU FFN block. See module docs.
pub struct FfnBlockResident {
    /// LayerNorm gamma (scale): `[d_model]`.
    pub ln_gamma_dev: HipBuffer,
    /// LayerNorm beta (bias): `[d_model]`.
    pub ln_beta_dev: HipBuffer,
    /// Gate projection — `out_dim = hidden`, `in_dim = d_model`.
    pub gate: LinearResident,
    /// Up projection — same shape as gate.
    pub up: LinearResident,
    /// Down projection — `out_dim = d_model`, `in_dim = hidden`.
    pub down: LinearResident,
    /// Cached `d_model` for shape sanity-checking on `forward`.
    d_model: usize,
    /// Cached `hidden` for `forward`'s scratch sizing.
    hidden: usize,
    /// LayerNorm epsilon — matches `crate::layer_norm_forward_batched`.
    eps: f32,
}

impl FfnBlockResident {
    /// LayerNorm epsilon used by the host-side `layer_norm_forward_batched`
    /// (see `crate::lib.rs`). Pinned here so the resident path matches
    /// host arithmetic exactly.
    pub const LN_EPS: f32 = 1e-5;

    /// Allocate device buffers and upload weights from a host
    /// [`FfnBlock`]. Returns `ResidencyError::Backend(_)` on
    /// hipMalloc / hipMemcpy failure.
    pub fn from_block(block: &FfnBlock) -> Result<Self, ResidencyError> {
        let d_model = block.ln_gamma.len();
        let hidden = block.gate.out_dim;
        let ln_gamma_dev = HipBuffer::new(d_model * 4)?;
        ln_gamma_dev.copy_from_host(&block.ln_gamma)?;
        let ln_beta_dev = HipBuffer::new(d_model * 4)?;
        ln_beta_dev.copy_from_host(&block.ln_beta)?;
        let gate = LinearResident::from_linear(&block.gate)?;
        let up = LinearResident::from_linear(&block.up)?;
        let down = LinearResident::from_linear(&block.down)?;
        Ok(Self {
            ln_gamma_dev, ln_beta_dev,
            gate, up, down,
            d_model, hidden,
            eps: Self::LN_EPS,
        })
    }

    /// Re-upload all weights from a host [`FfnBlock`] after an optimizer
    /// step. Shapes must match.
    pub fn sync_weights_from(&mut self, block: &FfnBlock) -> Result<(), ResidencyError> {
        debug_assert_eq!(block.ln_gamma.len(), self.d_model);
        debug_assert_eq!(block.gate.out_dim, self.hidden);
        self.ln_gamma_dev.copy_from_host(&block.ln_gamma)?;
        self.ln_beta_dev.copy_from_host(&block.ln_beta)?;
        self.gate.sync_weights_from(&block.gate)?;
        self.up.sync_weights_from(&block.up)?;
        self.down.sync_weights_from(&block.down)?;
        Ok(())
    }

    /// `d_model` (input/output dimension).
    pub fn d_model(&self) -> usize { self.d_model }
    /// `hidden` (inner FFN dimension).
    pub fn hidden(&self) -> usize { self.hidden }

    /// Forward pass for a single token (`n_tokens = 1`) or a batch.
    ///
    /// `x_dev`: input `[n_tokens × d_model]` row-major, `GpuVec::Hip`.
    /// `out_dev`: output `[n_tokens × d_model]` row-major, `GpuVec::Hip`.
    /// Caller is responsible for residual addition (use
    /// `op_tensor_resident(Add)` or call from inside
    /// [`crate::resident::TransformerBlockResident`]). This function does
    /// NOT add `x_dev` back into `out_dev`.
    ///
    /// `n_tokens` parametrises the LayerNorm dispatch; matvec is single-
    /// token (we issue `n_tokens` `matvec_resident` calls per linear)
    /// because rocBLAS sgemv is what `LinearResident::forward` wraps.
    /// For prefill at large `n_tokens`, prefer the future
    /// `Op::MatmulResident*` pathway (separate slice per the survey).
    pub fn forward(
        &self,
        batch: &HipBatch,
        n_tokens: usize,
        x_dev: &GpuVec,
        out_dev: &mut GpuVec,
    ) -> Result<(), ResidencyError> {
        debug_assert_eq!(x_dev.len(), n_tokens * self.d_model);
        debug_assert_eq!(out_dev.len(), n_tokens * self.d_model);

        // Scratch buffers for the FFN intermediate activations. Allocated
        // per-call to avoid stashing GPU memory inside `&self`. For the
        // hot path the caller should keep an `FfnBlockScratch` alive
        // across calls (see [`FfnBlockScratch`]); this method takes
        // care of single-shot calls and the test path.
        let mut scratch = FfnBlockScratch::new(n_tokens, self.d_model, self.hidden)?;
        self.forward_with_scratch(batch, n_tokens, x_dev, &mut scratch, out_dev)
    }

    /// Forward variant that takes a caller-owned [`FfnBlockScratch`] for
    /// the intermediate activation buffers. Useful when several blocks
    /// chain together with the same `n_tokens` — allocating each
    /// scratch slab once and reusing it across blocks saves the
    /// per-call hipMalloc/hipFree round-trips.
    pub fn forward_with_scratch(
        &self,
        batch: &HipBatch,
        n_tokens: usize,
        x_dev: &GpuVec,
        scratch: &mut FfnBlockScratch,
        out_dev: &mut GpuVec,
    ) -> Result<(), ResidencyError> {
        debug_assert_eq!(x_dev.len(), n_tokens * self.d_model);
        debug_assert_eq!(out_dev.len(), n_tokens * self.d_model);
        debug_assert!(scratch.fits(n_tokens, self.d_model, self.hidden));

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

        // Stage 1: LayerNorm — `[n_tokens × d_model] → [n_tokens × d_model]`.
        unsafe {
            layer_norm_resident(
                x_buf.device_ptr() as *const f32,
                self.ln_gamma_dev.device_ptr() as *const f32,
                self.ln_beta_dev.device_ptr() as *const f32,
                scratch.normed.device_ptr() as *mut f32,
                n_tokens, self.d_model, self.eps,
            )?;
        }
        batch.note_dispatch()?;

        // Stage 2: gate + up matvecs — one matvec per row, in a tight
        // loop. The hot path on small `n_tokens` (decode regime); the
        // future MatmulResidentNT path will subsume this for prefill.
        let normed_base = scratch.normed.device_ptr() as *const f32;
        let gate_base = scratch.gate_out.device_ptr() as *mut f32;
        let up_base = scratch.up_out.device_ptr() as *mut f32;
        for t in 0..n_tokens {
            unsafe {
                matvec_resident(
                    normed_base.add(t * self.d_model),
                    self.gate.weight_dev.device_ptr() as *const f32,
                    self.gate.bias_dev.device_ptr() as *const f32,
                    gate_base.add(t * self.hidden),
                    self.hidden, self.d_model,
                )?;
            }
            batch.note_dispatch()?;

            unsafe {
                matvec_resident(
                    normed_base.add(t * self.d_model),
                    self.up.weight_dev.device_ptr() as *const f32,
                    self.up.bias_dev.device_ptr() as *const f32,
                    up_base.add(t * self.hidden),
                    self.hidden, self.d_model,
                )?;
            }
            batch.note_dispatch()?;
        }

        // Stage 3: SiLU(gate) — composes `Logistic(gate) → silu_scratch`,
        // then `Mul(gate, silu_scratch) → silu_scratch`. Two MIOpen
        // dispatches, both resident. We reuse `scratch.silu` for the
        // sigmoid output and the final `silu(gate)` value.
        let n_hidden = n_tokens * self.hidden;
        unsafe {
            activation_resident(
                gate_base as *const f32,
                scratch.silu.device_ptr() as *mut f32,
                n_hidden, ActivationMode::Logistic,
            )?;
        }
        batch.note_dispatch()?;
        unsafe {
            // silu = sigmoid(gate) * gate, written into silu_scratch.
            op_tensor_resident(
                scratch.silu.device_ptr() as *const f32,
                gate_base as *const f32,
                scratch.silu.device_ptr() as *mut f32,
                n_hidden,
                1.0, 1.0, 0.0,
                BinaryOpKind::Mul,
            )?;
        }
        batch.note_dispatch()?;

        // Stage 4: hidden = silu(gate) * up. Single MIOpen Mul.
        unsafe {
            op_tensor_resident(
                scratch.silu.device_ptr() as *const f32,
                up_base as *const f32,
                scratch.hidden.device_ptr() as *mut f32,
                n_hidden,
                1.0, 1.0, 0.0,
                BinaryOpKind::Mul,
            )?;
        }
        batch.note_dispatch()?;

        // Stage 5: down projection — same matvec-per-row pattern as gate/up.
        let hidden_base = scratch.hidden.device_ptr() as *const f32;
        let out_base = out_buf.device_ptr() as *mut f32;
        for t in 0..n_tokens {
            unsafe {
                matvec_resident(
                    hidden_base.add(t * self.hidden),
                    self.down.weight_dev.device_ptr() as *const f32,
                    self.down.bias_dev.device_ptr() as *const f32,
                    out_base.add(t * self.d_model),
                    self.d_model, self.hidden,
                )?;
            }
            batch.note_dispatch()?;
        }

        Ok(())
    }

    /// Backward pass through the FFN block. Mirrors `forward_with_scratch`'s
    /// dispatch chain in reverse.
    ///
    /// `dy_dev`: gradient w.r.t. output `[n_tokens × d_model]`.
    /// `dx_dev`: receives `d/d(x)` `[n_tokens × d_model]`.
    /// `scratch` must hold the activations populated by forward.
    /// `bwd_scratch` is per-call gradient temporary storage.
    /// `grads` accumulates dweight/dbias for gate/up/down + dgamma/dbeta
    /// for the LayerNorm.
    ///
    /// **Single-token only** (`n_tokens == 1`) for this slice. The
    /// matvec-per-row loop pattern in forward will need a matched
    /// row-loop backward + outer-product accumulator for `n_tokens > 1`;
    /// see slice 11 for the prefill backward.
    #[allow(clippy::too_many_arguments)]
    pub fn backward(
        &self,
        batch: &HipBatch,
        n_tokens: usize,
        x_dev: &GpuVec,
        dy_dev: &GpuVec,
        scratch: &FfnBlockScratch,
        bwd_scratch: &mut FfnBackwardScratch,
        grads: &mut FfnBlockResidentGrads,
        dx_dev: &mut GpuVec,
    ) -> Result<(), ResidencyError> {
        debug_assert_eq!(n_tokens, 1, "FfnBlockResident::backward currently only supports n_tokens=1");
        debug_assert_eq!(x_dev.len(), self.d_model);
        debug_assert_eq!(dy_dev.len(), self.d_model);
        debug_assert_eq!(dx_dev.len(), self.d_model);

        let _dy_buf = match dy_dev {
            GpuVec::Hip(b) => b,
            other => return Err(ResidencyError::WrongVariant {
                expected: "Hip", got: other.variant_name(),
            }),
        };
        let x_buf = match x_dev {
            GpuVec::Hip(b) => b,
            other => return Err(ResidencyError::WrongVariant {
                expected: "Hip", got: other.variant_name(),
            }),
        };

        // Stage 1: down backward — d_hidden = down.W^T · dy, dweight_down = dy ⊗ hidden.
        // The down LinearResident has bias_dev (zero in test); LinearResident::backward
        // writes dbias = dy regardless. We pre-allocate a hidden-sized
        // d_hidden buffer in bwd_scratch.
        let mut d_hidden = GpuVec::try_hip(self.hidden)?;
        let mut hidden_view = GpuVec::try_hip(self.hidden)?;
        // hidden_view is staged from scratch.hidden (device buffer not GpuVec).
        // We need a GpuVec wrapping the same memory. The simplest path:
        // D2D copy into a fresh GpuVec.
        unsafe {
            use std::os::raw::c_void;
            hip_memcpy_d2d_local(
                hip_buf_mut(&mut hidden_view).device_ptr(),
                scratch.hidden.device_ptr() as *const c_void,
                self.hidden * 4,
            )?;
        }
        batch.note_dispatch()?;

        self.down.backward(
            batch, &hidden_view, dy_dev,
            &mut d_hidden,
            &mut grads.dweight_down,
            &mut grads.dbias_down,
        )?;

        // Stage 2: SwiGLU split — d_silu = d_hidden * up_out, d_up = d_hidden * silu.
        let n = self.hidden;
        let d_hidden_buf = hip_buf_mut(&mut d_hidden).device_ptr() as *const f32;
        unsafe {
            op_tensor_resident(
                d_hidden_buf,
                scratch.up_out.device_ptr() as *const f32,
                bwd_scratch.d_silu.device_ptr() as *mut f32,
                n, 1.0, 1.0, 0.0, BinaryOpKind::Mul,
            )?;
        }
        batch.note_dispatch()?;
        unsafe {
            op_tensor_resident(
                d_hidden_buf,
                scratch.silu.device_ptr() as *const f32,
                bwd_scratch.d_up.device_ptr() as *mut f32,
                n, 1.0, 1.0, 0.0, BinaryOpKind::Mul,
            )?;
        }
        batch.note_dispatch()?;

        // Stage 3: SiLU backward.
        // y = silu(gate); we have x = gate_out (forward input) and y = silu.
        unsafe {
            activation_backward_resident(
                scratch.gate_out.device_ptr() as *const f32,
                scratch.silu.device_ptr() as *const f32,
                bwd_scratch.d_silu.device_ptr() as *const f32,
                bwd_scratch.d_gate.device_ptr() as *mut f32,
                n, ActivationMode::Silu,
            )?;
        }
        batch.note_dispatch()?;

        // Stage 4 + 5: up + gate backward.
        // forward: gate_out = gate.W · normed + gate.b, up_out = up.W · normed + up.b
        // We need normed (forward input to gate/up). Stage from scratch.normed.
        let mut normed_view = GpuVec::try_hip(self.d_model)?;
        unsafe {
            use std::os::raw::c_void;
            hip_memcpy_d2d_local(
                hip_buf_mut(&mut normed_view).device_ptr(),
                scratch.normed.device_ptr() as *const c_void,
                self.d_model * 4,
            )?;
        }
        batch.note_dispatch()?;

        let mut d_up_view = GpuVec::try_hip(self.hidden)?;
        let mut d_gate_view = GpuVec::try_hip(self.hidden)?;
        unsafe {
            use std::os::raw::c_void;
            hip_memcpy_d2d_local(
                hip_buf_mut(&mut d_up_view).device_ptr(),
                bwd_scratch.d_up.device_ptr() as *const c_void,
                self.hidden * 4,
            )?;
            hip_memcpy_d2d_local(
                hip_buf_mut(&mut d_gate_view).device_ptr(),
                bwd_scratch.d_gate.device_ptr() as *const c_void,
                self.hidden * 4,
            )?;
        }
        batch.note_dispatch()?;
        batch.note_dispatch()?;

        let mut dx_from_up = GpuVec::try_hip(self.d_model)?;
        let mut dx_from_gate = GpuVec::try_hip(self.d_model)?;
        self.up.backward(
            batch, &normed_view, &d_up_view,
            &mut dx_from_up,
            &mut grads.dweight_up,
            &mut grads.dbias_up,
        )?;
        self.gate.backward(
            batch, &normed_view, &d_gate_view,
            &mut dx_from_gate,
            &mut grads.dweight_gate,
            &mut grads.dbias_gate,
        )?;

        // d_normed = dx_from_up + dx_from_gate
        let mut d_normed = GpuVec::try_hip(self.d_model)?;
        unsafe {
            op_tensor_resident(
                hip_buf_mut(&mut dx_from_up).device_ptr() as *const f32,
                hip_buf_mut(&mut dx_from_gate).device_ptr() as *const f32,
                hip_buf_mut(&mut d_normed).device_ptr() as *mut f32,
                self.d_model, 1.0, 1.0, 0.0, BinaryOpKind::Add,
            )?;
        }
        batch.note_dispatch()?;

        // Stage 6: LayerNorm backward.
        // forward: normed = (x - mean) / sqrt(var + eps) * gamma + beta
        // We compute mean/rstd host-side from `x` (D2H), upload, then
        // dispatch layer_norm_backward_resident. dgamma and dbeta
        // accumulate into grads.dgamma_dev / dbeta_dev.
        let mut x_host = vec![0.0f32; self.d_model];
        x_buf.copy_to_host(&mut x_host).map_err(ResidencyError::Backend)?;

        let mut mean_host = vec![0.0f32; n_tokens];
        let mut rstd_host = vec![0.0f32; n_tokens];
        for r in 0..n_tokens {
            let row = &x_host[r * self.d_model..(r + 1) * self.d_model];
            let m: f32 = row.iter().sum::<f32>() / self.d_model as f32;
            let v: f32 = row.iter().map(|&x| (x - m).powi(2)).sum::<f32>() / self.d_model as f32;
            mean_host[r] = m;
            rstd_host[r] = 1.0 / (v + self.eps).sqrt();
        }
        bwd_scratch.mean.copy_from_host(&mean_host)?;
        bwd_scratch.rstd.copy_from_host(&rstd_host)?;

        // Zero dweight buffers (layer_norm_backward_resident accumulates).
        let zeros_d = vec![0.0f32; self.d_model];
        grads.dgamma_dev.copy_from_host(&zeros_d)?;
        grads.dbeta_dev.copy_from_host(&zeros_d)?;

        unsafe {
            layer_norm_backward_resident(
                x_buf.device_ptr() as *const f32,
                hip_buf_mut(&mut d_normed).device_ptr() as *const f32,
                self.ln_gamma_dev.device_ptr() as *const f32,
                bwd_scratch.mean.device_ptr() as *const f32,
                bwd_scratch.rstd.device_ptr() as *const f32,
                hip_buf_mut(dx_dev).device_ptr() as *mut f32,
                grads.dgamma_dev.device_ptr() as *mut f32,
                grads.dbeta_dev.device_ptr() as *mut f32,
                n_tokens, self.d_model,
            )?;
        }
        batch.note_dispatch()?;

        Ok(())
    }
}

/// Per-call backward scratch for [`FfnBlockResident::backward`].
pub struct FfnBackwardScratch {
    /// `[n_tokens × hidden]`.
    pub d_silu: HipBuffer,
    /// `[n_tokens × hidden]`.
    pub d_up: HipBuffer,
    /// `[n_tokens × hidden]`.
    pub d_gate: HipBuffer,
    /// `[n_tokens]` — per-row mean (uploaded from host).
    pub mean: HipBuffer,
    /// `[n_tokens]` — per-row reciprocal stddev (uploaded from host).
    pub rstd: HipBuffer,
    cap_n_tokens: usize,
    cap_d_model: usize,
    cap_hidden: usize,
}

impl FfnBackwardScratch {
    pub fn new(n_tokens: usize, d_model: usize, hidden: usize) -> Result<Self, ResidencyError> {
        Ok(Self {
            d_silu: HipBuffer::new(n_tokens * hidden * 4)?,
            d_up: HipBuffer::new(n_tokens * hidden * 4)?,
            d_gate: HipBuffer::new(n_tokens * hidden * 4)?,
            mean: HipBuffer::new(n_tokens * 4)?,
            rstd: HipBuffer::new(n_tokens * 4)?,
            cap_n_tokens: n_tokens,
            cap_d_model: d_model,
            cap_hidden: hidden,
        })
    }
    pub fn fits(&self, n_tokens: usize, d_model: usize, hidden: usize) -> bool {
        n_tokens <= self.cap_n_tokens && d_model <= self.cap_d_model && hidden <= self.cap_hidden
    }
}

/// Device-resident weight gradients for [`FfnBlockResident`].
pub struct FfnBlockResidentGrads {
    pub dgamma_dev: HipBuffer,
    pub dbeta_dev: HipBuffer,
    pub dweight_gate: GpuVec,
    pub dbias_gate: GpuVec,
    pub dweight_up: GpuVec,
    pub dbias_up: GpuVec,
    pub dweight_down: GpuVec,
    pub dbias_down: GpuVec,
}

impl FfnBlockResidentGrads {
    pub fn new(d_model: usize, hidden: usize) -> Result<Self, ResidencyError> {
        Ok(Self {
            dgamma_dev: HipBuffer::new(d_model * 4)?,
            dbeta_dev: HipBuffer::new(d_model * 4)?,
            dweight_gate: GpuVec::try_hip(hidden * d_model)?,
            dbias_gate: GpuVec::try_hip(hidden)?,
            dweight_up: GpuVec::try_hip(hidden * d_model)?,
            dbias_up: GpuVec::try_hip(hidden)?,
            dweight_down: GpuVec::try_hip(d_model * hidden)?,
            dbias_down: GpuVec::try_hip(d_model)?,
        })
    }
}

/// Helper: HIP D2D memcpy.
unsafe fn hip_memcpy_d2d_local(
    dst: *mut std::os::raw::c_void,
    src: *const std::os::raw::c_void,
    bytes: usize,
) -> Result<(), ResidencyError> {
    use modgrad_device::backend::rocm::ffi;
    const HIP_MEMCPY_DEVICE_TO_DEVICE: std::os::raw::c_int = 3;
    let err = unsafe {
        ffi::hipMemcpy(dst, src, bytes, HIP_MEMCPY_DEVICE_TO_DEVICE)
    };
    if err != 0 {
        return Err(ResidencyError::Backend(
            modgrad_device::backend::BackendError::Runtime(format!(
                "hipMemcpy D2D ({bytes} bytes): {}", ffi::hip_err_str(err),
            )),
        ));
    }
    Ok(())
}

#[inline]
fn hip_buf_mut(g: &mut GpuVec) -> &mut HipBuffer {
    match g {
        GpuVec::Hip(b) => b,
        _ => panic!("expected Hip variant"),
    }
}

/// Pre-allocated scratch buffers for [`FfnBlockResident::forward_with_scratch`].
/// Keeps the intermediate activation slabs (`normed`, `gate_out`, `up_out`,
/// `silu`, `hidden`) device-resident across calls — avoids hipMalloc/hipFree
/// round-trips on every forward.
///
/// **Sizing:** allocate for the maximum `n_tokens × max(d_model, hidden)`
/// you intend to call with. Smaller calls are fine (we only access the
/// `n_tokens * dim` prefix); larger calls panic at the
/// `debug_assert!(fits)` boundary.
pub struct FfnBlockScratch {
    /// Pre-norm output: `[n_tokens × d_model]`.
    normed: HipBuffer,
    /// Gate projection output: `[n_tokens × hidden]`.
    gate_out: HipBuffer,
    /// Up projection output: `[n_tokens × hidden]`.
    up_out: HipBuffer,
    /// SiLU(gate) intermediate: `[n_tokens × hidden]`. Reused for the
    /// sigmoid() output and the final silu() composite.
    silu: HipBuffer,
    /// silu(gate) * up: `[n_tokens × hidden]`. Down projection input.
    hidden: HipBuffer,
    /// Capacity sentinels — the buffers above were allocated for these
    /// shapes. `fits` is the runtime check.
    cap_n_tokens: usize,
    cap_d_model: usize,
    cap_hidden: usize,
}

impl FfnBlockScratch {
    /// Allocate scratch slabs sized to the given shape.
    pub fn new(n_tokens: usize, d_model: usize, hidden: usize) -> Result<Self, ResidencyError> {
        let normed = HipBuffer::new(n_tokens * d_model * 4)?;
        let gate_out = HipBuffer::new(n_tokens * hidden * 4)?;
        let up_out = HipBuffer::new(n_tokens * hidden * 4)?;
        let silu = HipBuffer::new(n_tokens * hidden * 4)?;
        let hidden_buf = HipBuffer::new(n_tokens * hidden * 4)?;
        Ok(Self {
            normed, gate_out, up_out, silu, hidden: hidden_buf,
            cap_n_tokens: n_tokens, cap_d_model: d_model, cap_hidden: hidden,
        })
    }

    /// True if these buffers are large enough for the given shape.
    pub fn fits(&self, n_tokens: usize, d_model: usize, hidden: usize) -> bool {
        n_tokens <= self.cap_n_tokens
            && d_model <= self.cap_d_model
            && hidden <= self.cap_hidden
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use modgrad_device::backend::rocm::ffi::runtime_available;
    use crate::FfnBlock;
    use std::sync::Mutex;

    /// HIP runtime tests must run serially — concurrent matvec_resident
    /// dispatches from multiple test threads share the default stream
    /// and can interleave intermediate buffers (the SiLU compose is
    /// three sequential dispatches; another test queuing matvec between
    /// stages corrupts the output). The other rocm-using tests in this
    /// crate (`tests::ffn_*` for AdamW path) already accept this risk;
    /// the cleanest local fix is a per-test mutex guard. Workspace-wide
    /// fix would be a `serial_test` dep.
    static HIP_TEST_LOCK: Mutex<()> = Mutex::new(());

    /// Reference: host-side single-token SwiGLU forward (no residual,
    /// no skip-add — matches what `FfnBlockResident::forward` returns).
    fn host_swiglu_one(block: &FfnBlock, x: &[f32]) -> Vec<f32> {
        let d = block.ln_gamma.len();
        let h = block.gate.out_dim;
        // LayerNorm
        let n = d as f32;
        let mean: f32 = x.iter().sum::<f32>() / n;
        let var: f32 = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
        let inv_std = 1.0 / (var + FfnBlockResident::LN_EPS).sqrt();
        let normed: Vec<f32> = (0..d).map(|i|
            block.ln_gamma[i] * (x[i] - mean) * inv_std + block.ln_beta[i]
        ).collect();
        // gate, up
        let mut gate = vec![0.0f32; h];
        block.gate.forward_into(&normed, &mut gate);
        let mut up = vec![0.0f32; h];
        block.up.forward_into(&normed, &mut up);
        // silu * up
        let hidden: Vec<f32> = (0..h).map(|i| {
            let g = gate[i];
            let s = 1.0 / (1.0 + (-g).exp());
            g * s * up[i]
        }).collect();
        // down
        let mut out = vec![0.0f32; d];
        block.down.forward_into(&hidden, &mut out);
        out
    }

    /// Build a SwiGLU block with **uncorrelated** gate/up/down weights.
    /// `FfnBlock::new` uses `SimpleRng::new(in ^ out)` for each `Linear`,
    /// which yields identical gate and up weights — fine for training
    /// where rotation breaks the symmetry, but we want each test
    /// dispatch to actually exercise the gate ≠ up code path.
    fn fresh_block(d: usize, h: usize, seed: u64) -> FfnBlock {
        let mut rng = modgrad_compute::neuron::SimpleRng::new(seed);
        let scale_d = (2.0 / d as f32).sqrt();
        let scale_h = (2.0 / h as f32).sqrt();
        FfnBlock {
            ln_gamma: (0..d).map(|_| 1.0 + 0.05 * rng.next_normal()).collect(),
            ln_beta: (0..d).map(|_| 0.05 * rng.next_normal()).collect(),
            gate: modgrad_compute::neuron::Linear {
                weight: (0..h*d).map(|_| rng.next_normal() * scale_d).collect(),
                bias: vec![0.0; h],
                in_dim: d, out_dim: h,
            },
            up: modgrad_compute::neuron::Linear {
                weight: (0..h*d).map(|_| rng.next_normal() * scale_d).collect(),
                bias: vec![0.0; h],
                in_dim: d, out_dim: h,
            },
            down: modgrad_compute::neuron::Linear {
                weight: (0..d*h).map(|_| rng.next_normal() * scale_h).collect(),
                bias: vec![0.0; d],
                in_dim: h, out_dim: d,
            },
        }
    }

    #[test]
    fn ffn_block_resident_matches_host_one_token() {
        let _guard = HIP_TEST_LOCK.lock().unwrap();
        if !runtime_available() {
            eprintln!("hip runtime unavailable, skipping");
            return;
        }
        let d = 64;
        let h = 128;
        let block = fresh_block(d, h, 0x9001);
        let mut rng = modgrad_compute::neuron::SimpleRng::new(0x9001);
        let host_x: Vec<f32> = (0..d).map(|_| rng.next_normal()).collect();
        let host_y = host_swiglu_one(&block, &host_x);

        let resident = FfnBlockResident::from_block(&block).expect("upload");
        let mut x_dev = GpuVec::try_hip(d).expect("alloc x");
        x_dev.copy_from(&host_x);
        let mut out_dev = GpuVec::try_hip(d).expect("alloc out");

        let batch = HipBatch::new();
        resident.forward(&batch, 1, &x_dev, &mut out_dev).expect("resident forward");
        batch.flush().expect("flush");

        let mut device_y = vec![0.0f32; d];
        out_dev.copy_to_host(&mut device_y);

        let max_diff = host_y.iter().zip(&device_y)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let rel = max_diff / host_y.iter().map(|v| v.abs()).fold(1.0f32, f32::max);
        assert!(rel < 1e-2,
            "ffn block resident vs host max |Δ| = {max_diff}, rel = {rel}");
    }

    #[test]
    fn ffn_block_resident_sync_weights() {
        let _guard = HIP_TEST_LOCK.lock().unwrap();
        if !runtime_available() {
            eprintln!("hip runtime unavailable, skipping");
            return;
        }
        let d = 32;
        let h = 64;
        let mut block = fresh_block(d, h, 0xC0DE);
        let mut resident = FfnBlockResident::from_block(&block).expect("upload");

        // Mutate the host weights and re-sync.
        for v in block.ln_gamma.iter_mut() { *v *= 1.5; }
        for v in block.gate.weight.iter_mut() { *v += 0.1; }
        resident.sync_weights_from(&block).expect("sync");

        // Forward with the synced weights and compare against host.
        let mut rng = modgrad_compute::neuron::SimpleRng::new(0x9002);
        let host_x: Vec<f32> = (0..d).map(|_| rng.next_normal()).collect();
        let host_y = host_swiglu_one(&block, &host_x);

        let mut x_dev = GpuVec::try_hip(d).expect("alloc x");
        x_dev.copy_from(&host_x);
        let mut out_dev = GpuVec::try_hip(d).expect("alloc out");
        let batch = HipBatch::new();
        resident.forward(&batch, 1, &x_dev, &mut out_dev).expect("forward");
        batch.flush().expect("flush");

        let mut device_y = vec![0.0f32; d];
        out_dev.copy_to_host(&mut device_y);
        let max_diff = host_y.iter().zip(&device_y)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let rel = max_diff / host_y.iter().map(|v| v.abs()).fold(1.0f32, f32::max);
        assert!(rel < 1e-2,
            "post-sync resident vs host max |Δ| = {max_diff}, rel = {rel}");
    }
}
