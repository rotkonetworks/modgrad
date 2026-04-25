//! Faithful Ctm SynapseUNet — the inter-neuron connection function.
//!
//! Matches `SynapseUNET` from continuous-thought-machines/models/modules.py.
//! A U-Net with skip connections: projects down to a bottleneck then back up,
//! with residual additions at each level. This is the deep, expressive synapse
//! that makes the CTM work — NOT a single linear layer.
//!
//! Architecture for depth=D, out_dims=O, min_width=16:
//!   widths = linspace(O, 16, D)   e.g. [4096, 2072, 16] for D=3
//!   first_projection: in_dims → widths[0], LayerNorm, SiLU
//!   down[i]: widths[i] → widths[i+1], LayerNorm, SiLU     (D-1 blocks)
//!   up[i]:   widths[i+1] → widths[i], LayerNorm, SiLU     (D-1 blocks, applied in reverse)
//!   skip_ln[i]: LayerNorm on (up_out + down_skip)          (D-1 norms)

use serde::{Deserialize, Serialize};
use modgrad_compute::neuron::Linear;
use wincode_derive::{SchemaRead, SchemaWrite};

// ─── Affine LayerNorm (now backend-dispatched) ─────────────

/// LayerNorm with learnable affine parameters (gamma, beta).
/// Matches PyTorch nn.LayerNorm default: elementwise_affine=True.
///
/// Dispatches through `modgrad_device::backend::ops::layer_norm_fwd`,
/// which routes to the fastest registered backend (CPU fallback always
/// available). Prior inline implementation is preserved as a fallback
/// inside the CPU backend; this function is now just a thin dispatch
/// shim so every hot-path call participates in GPU routing.
#[inline]
fn affine_layer_norm(x: &mut [f32], gamma: &[f32], beta: &[f32]) {
    let n = x.len();
    if n == 0 { return; }
    // Allocate a tmp output — layer_norm_fwd is not in-place by default.
    // The dispatched op writes into `out`, then we copy back into `x`.
    // Cheap on CPU (single memcpy); on GPU would be two passes but the
    // fused `ln_silu_fwd` path avoids this when the caller actually
    // wants LN+SiLU (see SynapseBlock::forward).
    let mut out = vec![0.0f32; n];
    modgrad_device::backend::ops::layer_norm_fwd(
        x, gamma, beta, &mut out, None, 1, n,
    ).expect("layer_norm_fwd dispatch");
    x.copy_from_slice(&out);
}

// ─── SynapseBlock ──────────────────────────────────────────

/// One block: Linear → LayerNorm(affine) → SiLU.
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub struct SynapseBlock {
    pub linear: Linear,
    pub ln_gamma: Vec<f32>,
    pub ln_beta: Vec<f32>,
}

impl SynapseBlock {
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        Self {
            linear: Linear::new(in_dim, out_dim),
            ln_gamma: vec![1.0; out_dim],
            ln_beta: vec![0.0; out_dim],
        }
    }

    /// Forward: Linear → LayerNorm → SiLU.
    ///
    /// Both the matmul (`linear.forward` → `ops::matvec`) and the
    /// fused LN+SiLU (`ops::ln_silu_fwd`) now dispatch through the
    /// `Backend` registry — the full block is GPU-ready when a
    /// capable backend is registered, CPU-fallback otherwise.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut y = self.linear.forward(x);
        let n = y.len();
        let mut out = vec![0.0f32; n];
        modgrad_device::backend::ops::ln_silu_fwd(
            &y, &self.ln_gamma, &self.ln_beta, &mut out, None, 1, n,
        ).expect("ln_silu_fwd dispatch");
        y = out;
        y
    }
}

// ─── SynapseUNet ───────────────────────────────────────────

/// Compute integer widths matching np.linspace(out_dims, min_width, depth).
fn compute_widths(out_dims: usize, min_width: usize, depth: usize) -> Vec<usize> {
    if depth <= 1 {
        return vec![out_dims];
    }
    (0..depth)
        .map(|i| {
            let t = i as f32 / (depth - 1) as f32;
            let w = out_dims as f32 + t * (min_width as f32 - out_dims as f32);
            (w.round() as usize).max(1)
        })
        .collect()
}

/// Faithful Ctm U-Net synapse.
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub struct SynapseUNet {
    pub widths: Vec<usize>,

    /// Initial projection: in_dims → widths[0].
    pub first_projection: SynapseBlock,

    /// Down blocks: widths[i] → widths[i+1]. Length = depth-1.
    pub down_blocks: Vec<SynapseBlock>,

    /// Up blocks: widths[i+1] → widths[i]. Length = depth-1.
    /// Stored in "down order" (index 0 = shallowest), applied in reverse.
    pub up_blocks: Vec<SynapseBlock>,

    /// Skip connection LayerNorm parameters. One per level.
    pub skip_ln_gamma: Vec<Vec<f32>>,
    pub skip_ln_beta: Vec<Vec<f32>>,
}

impl SynapseUNet {
    pub fn new(in_dims: usize, out_dims: usize, depth: usize, min_width: usize) -> Self {
        let widths = compute_widths(out_dims, min_width, depth);
        let n_blocks = widths.len().saturating_sub(1);

        let first_projection = SynapseBlock::new(in_dims, widths[0]);

        let mut down_blocks = Vec::with_capacity(n_blocks);
        let mut up_blocks = Vec::with_capacity(n_blocks);
        let mut skip_ln_gamma = Vec::with_capacity(n_blocks);
        let mut skip_ln_beta = Vec::with_capacity(n_blocks);

        for i in 0..n_blocks {
            down_blocks.push(SynapseBlock::new(widths[i], widths[i + 1]));
            up_blocks.push(SynapseBlock::new(widths[i + 1], widths[i]));
            skip_ln_gamma.push(vec![1.0; widths[i]]);
            skip_ln_beta.push(vec![0.0; widths[i]]);
        }

        Self { widths, first_projection, down_blocks, up_blocks, skip_ln_gamma, skip_ln_beta }
    }

    /// Forward pass matching Python SynapseUNET.forward exactly.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let n_blocks = self.down_blocks.len();

        // Initial projection
        let out_first = self.first_projection.forward(x);

        // Down path — store each level for skip connections
        let mut outs_down = Vec::with_capacity(n_blocks + 1);
        outs_down.push(out_first);
        for i in 0..n_blocks {
            let next = self.down_blocks[i].forward(outs_down.last().unwrap());
            outs_down.push(next);
        }

        // Up path — start from bottleneck, apply in reverse order
        let mut current = outs_down[n_blocks].clone(); // bottleneck

        for i in 0..n_blocks {
            let up_idx = n_blocks - 1 - i;

            // Up projection
            let mut up_out = self.up_blocks[up_idx].forward(&current);

            // Add skip connection from matching down level
            let skip = &outs_down[up_idx];
            debug_assert_eq!(up_out.len(), skip.len());
            for j in 0..up_out.len() {
                up_out[j] += skip[j];
            }

            // LayerNorm on the sum
            affine_layer_norm(
                &mut up_out,
                &self.skip_ln_gamma[up_idx],
                &self.skip_ln_beta[up_idx],
            );

            current = up_out;
        }

        current
    }

    /// Total number of trainable parameters.
    pub fn n_params(&self) -> usize {
        let mut n = self.first_projection.linear.weight.len()
            + self.first_projection.linear.bias.len()
            + self.first_projection.ln_gamma.len() * 2;
        for i in 0..self.down_blocks.len() {
            n += self.down_blocks[i].linear.weight.len()
                + self.down_blocks[i].linear.bias.len()
                + self.down_blocks[i].ln_gamma.len() * 2;
            n += self.up_blocks[i].linear.weight.len()
                + self.up_blocks[i].linear.bias.len()
                + self.up_blocks[i].ln_gamma.len() * 2;
            n += self.skip_ln_gamma[i].len() * 2;
        }
        n
    }
}

// ─── SynapseUNetResident ───────────────────────────────────
//
// Device-resident companion to `SynapseUNet`. Mirrors the host
// U-Net structure (first_projection + down + up + skip LNs) as a
// stack of `LinearResident`s with weights pinned in VRAM, plus
// `HipBuffer` copies of every LayerNorm `gamma`/`beta` so the LN
// + SiLU + skip-add chain stays fully device-resident. Build once
// via `from_synapse_unet`, refresh after optimizer steps via
// `sync_weights_from`, dispatch through `forward` with a
// `&HipBatch`.
//
// Activation flow: every block runs `matvec_resident` →
// `layer_norm_resident` → `activation_resident(Logistic)` into a
// SiLU scratch buffer → `op_tensor_resident(Mul, x, scratch, x)`
// to finish SiLU. The up path additionally does
// `op_tensor_resident(Add, current, skip, current)` for the skip
// connection followed by a second `layer_norm_resident` with the
// per-level `skip_ln_*_dev` weights. No D2H or H2D bounces remain
// inside the chain — every dispatch reads and writes hipMalloc'd
// device pointers. The `&HipBatch` parameter flows into every new
// dispatch via `note_dispatch` so the auto-sync cadence covers
// the new ops just like it covers the matvecs.
//
// MIOpen's `miopenLayerNormForward` requires non-null `mean` and
// `rstd` output pointers; the rocm backend honors that with two
// hipMalloc/hipFree pairs per dispatch (sub-µs on this hardware).
// We do not surface those scratch allocations here; they are
// internal to the resident dispatch.

#[cfg(feature = "rocm")]
pub struct SynapseUNetResident {
    /// Layer widths (mirrors the host `SynapseUNet`). Used to
    /// size scratch buffers per block and validate the
    /// `forward` input/output dims.
    pub widths: Vec<usize>,

    /// Initial projection: in_dims → widths[0]. The Linear weight
    /// + bias and the LN gamma/beta all live on-device.
    pub first_projection: modgrad_compute::neuron::LinearResident,
    pub first_ln_gamma_dev: modgrad_device::backend::HipBuffer,
    pub first_ln_beta_dev: modgrad_device::backend::HipBuffer,

    /// Down blocks: widths[i] → widths[i+1].
    pub down_blocks: Vec<modgrad_compute::neuron::LinearResident>,
    pub down_ln_gamma_dev: Vec<modgrad_device::backend::HipBuffer>,
    pub down_ln_beta_dev: Vec<modgrad_device::backend::HipBuffer>,

    /// Up blocks: widths[i+1] → widths[i] (stored in down-order,
    /// applied in reverse).
    pub up_blocks: Vec<modgrad_compute::neuron::LinearResident>,
    pub up_ln_gamma_dev: Vec<modgrad_device::backend::HipBuffer>,
    pub up_ln_beta_dev: Vec<modgrad_device::backend::HipBuffer>,

    /// Skip-connection LayerNorm parameters (one per up level).
    pub skip_ln_gamma_dev: Vec<modgrad_device::backend::HipBuffer>,
    pub skip_ln_beta_dev: Vec<modgrad_device::backend::HipBuffer>,

    /// Host-side dim cached for the input projection (= in_dim of
    /// `first_projection`). Used to validate `forward` arguments.
    pub in_dim: usize,
    /// Host-side dim cached for the U-Net output (= widths[0]).
    pub out_dim: usize,
}

#[cfg(feature = "rocm")]
impl SynapseUNetResident {
    /// Allocate a `HipBuffer` sized for `host.len()` floats and
    /// upload `host` into it. Used by `from_synapse_unet` and
    /// `sync_weights_from` to stage LN gamma/beta arrays onto the
    /// device. The caller is responsible for ensuring the buffer
    /// outlives every dispatch that reads from it (which here
    /// means: the buffer is owned by `Self`).
    fn upload_param_dev(
        host: &[f32],
    ) -> Result<modgrad_device::backend::HipBuffer, modgrad_compute::backend::ResidencyError> {
        let buf = modgrad_device::backend::HipBuffer::new(host.len() * 4)?;
        buf.copy_from_host(host)?;
        Ok(buf)
    }

    /// Allocate device buffers and upload every Linear's weight +
    /// bias and every LN gamma/beta. After this returns, every
    /// parameter the resident `forward` reads lives in VRAM and
    /// stays there until `Self` is dropped.
    pub fn from_synapse_unet(
        unet: &SynapseUNet,
    ) -> Result<Self, modgrad_compute::backend::ResidencyError> {
        use modgrad_compute::neuron::LinearResident;
        let first_projection = LinearResident::from_linear(&unet.first_projection.linear)?;
        let first_ln_gamma_dev = Self::upload_param_dev(&unet.first_projection.ln_gamma)?;
        let first_ln_beta_dev = Self::upload_param_dev(&unet.first_projection.ln_beta)?;

        let down_blocks = unet.down_blocks.iter()
            .map(|b| LinearResident::from_linear(&b.linear))
            .collect::<Result<Vec<_>, _>>()?;
        let down_ln_gamma_dev = unet.down_blocks.iter()
            .map(|b| Self::upload_param_dev(&b.ln_gamma))
            .collect::<Result<Vec<_>, _>>()?;
        let down_ln_beta_dev = unet.down_blocks.iter()
            .map(|b| Self::upload_param_dev(&b.ln_beta))
            .collect::<Result<Vec<_>, _>>()?;

        let up_blocks = unet.up_blocks.iter()
            .map(|b| LinearResident::from_linear(&b.linear))
            .collect::<Result<Vec<_>, _>>()?;
        let up_ln_gamma_dev = unet.up_blocks.iter()
            .map(|b| Self::upload_param_dev(&b.ln_gamma))
            .collect::<Result<Vec<_>, _>>()?;
        let up_ln_beta_dev = unet.up_blocks.iter()
            .map(|b| Self::upload_param_dev(&b.ln_beta))
            .collect::<Result<Vec<_>, _>>()?;

        let skip_ln_gamma_dev = unet.skip_ln_gamma.iter()
            .map(|g| Self::upload_param_dev(g))
            .collect::<Result<Vec<_>, _>>()?;
        let skip_ln_beta_dev = unet.skip_ln_beta.iter()
            .map(|g| Self::upload_param_dev(g))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            widths: unet.widths.clone(),
            first_projection,
            first_ln_gamma_dev,
            first_ln_beta_dev,
            down_blocks,
            down_ln_gamma_dev,
            down_ln_beta_dev,
            up_blocks,
            up_ln_gamma_dev,
            up_ln_beta_dev,
            skip_ln_gamma_dev,
            skip_ln_beta_dev,
            in_dim: unet.first_projection.linear.in_dim,
            out_dim: unet.widths[0],
        })
    }

    /// Re-upload every Linear's weight + bias and every LN
    /// gamma/beta. Call after every optimizer step that mutated
    /// the host `SynapseUNet`. Buffer sizes match the original
    /// upload (the U-Net topology is fixed at
    /// `from_synapse_unet`); we re-use the existing
    /// `HipBuffer`s rather than reallocating.
    pub fn sync_weights_from(
        &mut self, unet: &SynapseUNet,
    ) -> Result<(), modgrad_compute::backend::ResidencyError> {
        debug_assert_eq!(self.down_blocks.len(), unet.down_blocks.len());
        debug_assert_eq!(self.up_blocks.len(), unet.up_blocks.len());

        self.first_projection.sync_weights_from(&unet.first_projection.linear)?;
        self.first_ln_gamma_dev.copy_from_host(&unet.first_projection.ln_gamma)?;
        self.first_ln_beta_dev.copy_from_host(&unet.first_projection.ln_beta)?;

        for (r, b) in self.down_blocks.iter_mut().zip(&unet.down_blocks) {
            r.sync_weights_from(&b.linear)?;
        }
        for (g, b) in self.down_ln_gamma_dev.iter().zip(&unet.down_blocks) {
            g.copy_from_host(&b.ln_gamma)?;
        }
        for (g, b) in self.down_ln_beta_dev.iter().zip(&unet.down_blocks) {
            g.copy_from_host(&b.ln_beta)?;
        }

        for (r, b) in self.up_blocks.iter_mut().zip(&unet.up_blocks) {
            r.sync_weights_from(&b.linear)?;
        }
        for (g, b) in self.up_ln_gamma_dev.iter().zip(&unet.up_blocks) {
            g.copy_from_host(&b.ln_gamma)?;
        }
        for (g, b) in self.up_ln_beta_dev.iter().zip(&unet.up_blocks) {
            g.copy_from_host(&b.ln_beta)?;
        }

        for (g, src) in self.skip_ln_gamma_dev.iter().zip(&unet.skip_ln_gamma) {
            g.copy_from_host(src)?;
        }
        for (g, src) in self.skip_ln_beta_dev.iter().zip(&unet.skip_ln_beta) {
            g.copy_from_host(src)?;
        }
        Ok(())
    }

    /// Resident U-Net forward. `x_dev` and `out_dev` are
    /// `GpuVec::Hip`. Every Linear, every LayerNorm, every SiLU,
    /// and the up-path skip-add stay device-resident: zero D2H or
    /// H2D bounces from `forward` itself. The only host work is
    /// computing pointer offsets — every kernel reads and writes
    /// hipMalloc'd buffers.
    ///
    /// Activation chain per block: `matvec_resident` →
    /// `layer_norm_resident` (in-place) →
    /// `activation_resident(Logistic)` into `silu_scratch_dev` →
    /// `op_tensor_resident(Mul, x, silu_scratch_dev, x)`. The up
    /// path additionally inserts an
    /// `op_tensor_resident(Add, current, skip, current)` followed
    /// by the affine `layer_norm_resident` with the level's
    /// `skip_ln_*_dev` weights.
    ///
    /// The `silu_scratch_dev` buffer is allocated once at the
    /// largest block size (`max(widths)`) and re-used across every
    /// SiLU dispatch — the scratch buffer is only read inside the
    /// immediately following `OpTensor(Mul)`, so over-sizing it is
    /// safe and avoids per-block hipMalloc churn.
    ///
    /// `outs_down_dev` retains every down-path activation as a
    /// device buffer because the up path reads them as skip
    /// inputs. We pop the bottleneck out as the initial up-path
    /// `current_dev`; the remaining entries are addressed by
    /// `up_idx` during the up loop.
    ///
    /// Synchronisation: every resident dispatch counts toward the
    /// `HipBatch::DEFAULT_SYNC_EVERY = 256` cadence via
    /// `note_dispatch`, so the queue depth stays bounded for any
    /// reasonable `n_blocks`. The caller is responsible for
    /// flushing before reading `out_dev` back to host.
    pub fn forward(
        &self,
        batch: &modgrad_device::backend::HipBatch,
        x_dev: &modgrad_compute::backend::GpuVec,
        out_dev: &mut modgrad_compute::backend::GpuVec,
    ) -> Result<(), modgrad_compute::backend::ResidencyError> {
        use modgrad_compute::backend::{GpuVec, ResidencyError};
        use modgrad_device::backend::BinaryOpKind;
        debug_assert_eq!(x_dev.len(), self.in_dim);
        debug_assert_eq!(out_dev.len(), self.out_dim);

        // Boundary check: forward only consumes Hip-resident input
        // and writes Hip-resident output. A Heap or Vram GpuVec
        // slipping in is a caller bug; surface it as
        // `WrongVariant` rather than panicking deeper inside the
        // dispatch path.
        match x_dev {
            GpuVec::Hip(_) => {}
            other => return Err(ResidencyError::WrongVariant {
                expected: "Hip", got: other.variant_name(),
            }),
        }
        match out_dev {
            GpuVec::Hip(_) => {}
            other => return Err(ResidencyError::WrongVariant {
                expected: "Hip", got: other.variant_name(),
            }),
        }

        let n_blocks = self.down_blocks.len();

        // LayerNorm epsilon. Matches the host CPU LN reference
        // (`modgrad-device/backend/cpu.rs::layer_norm_fwd`) so
        // numerical equivalence holds modulo MIOpen's sqrt/exp
        // precision differences.
        const LN_EPS: f32 = 1e-5;

        // SiLU scratch buffer: sized to the widest block so it
        // can be reused across every block's SiLU dispatch
        // without reallocation. Reads/writes only the leading
        // `widths[i]` floats per call; the trailing capacity is
        // unused. Sub-µs hipMalloc cost is paid once per forward.
        let max_width = *self.widths.iter().max().unwrap_or(&self.out_dim);
        let silu_scratch_dev = modgrad_device::backend::HipBuffer::new(max_width * 4)?;
        let silu_scratch_ptr = silu_scratch_dev.device_ptr() as *mut f32;

        // Helper closure: run LN + SiLU in-place on a HipBuffer,
        // using the supplied LN gamma/beta device pointers. This
        // is the chain that replaces the previous D2H → host
        // `ln_silu_fwd` → H2D triple.
        //
        // Inlined manually below (closures over &HipBuffer are
        // ergonomic but obscure the dispatch sequence; explicit
        // inlining keeps the resident chain easy to audit).

        // Stage 1: first_projection (in_dim → widths[0]).
        //
        // When `n_blocks == 0` (depth-1 case, no down/up path),
        // the first projection IS the U-Net output — write
        // directly into `out_dev`. Otherwise, allocate a fresh
        // buffer, fill it via the projection, and stash it as
        // `outs_down_dev[0]` for the down loop.
        let mut outs_down_dev: Vec<GpuVec> = Vec::with_capacity(n_blocks + 1);

        if n_blocks == 0 {
            // Depth-1: first projection produces the final
            // activation directly into `out_dev`.
            self.first_projection.forward(batch, x_dev, out_dev)?;
            self.layer_norm_silu_resident_inplace(
                batch,
                out_dev,
                &self.first_ln_gamma_dev,
                &self.first_ln_beta_dev,
                self.widths[0],
                silu_scratch_ptr,
                LN_EPS,
            )?;
            return Ok(());
        }

        // n_blocks > 0: stage outs_down_dev[0] = first_act.
        let mut first_dev = GpuVec::try_hip(self.widths[0])?;
        self.first_projection.forward(batch, x_dev, &mut first_dev)?;
        self.layer_norm_silu_resident_inplace(
            batch,
            &mut first_dev,
            &self.first_ln_gamma_dev,
            &self.first_ln_beta_dev,
            self.widths[0],
            silu_scratch_ptr,
            LN_EPS,
        )?;
        outs_down_dev.push(first_dev);

        // Stage 2: down blocks. Each block reads
        // outs_down_dev[i], writes a fresh buffer of size
        // widths[i+1], runs LN + SiLU on it, and pushes the
        // result onto outs_down_dev. We do NOT reuse buffers
        // across levels because the up path needs every level's
        // activation as a skip input.
        for i in 0..n_blocks {
            let mut next_dev = GpuVec::try_hip(self.widths[i + 1])?;
            // borrow outs_down_dev[i] for matvec — split-borrow
            // pattern: the `forward` call only reads its first
            // arg and writes its second, and `next_dev` is a
            // disjoint allocation, so no aliasing.
            let prev_dev = outs_down_dev.last().expect("outs_down_dev[0] pushed above");
            self.down_blocks[i].forward(batch, prev_dev, &mut next_dev)?;
            self.layer_norm_silu_resident_inplace(
                batch,
                &mut next_dev,
                &self.down_ln_gamma_dev[i],
                &self.down_ln_beta_dev[i],
                self.widths[i + 1],
                silu_scratch_ptr,
                LN_EPS,
            )?;
            outs_down_dev.push(next_dev);
        }

        // Stage 3: up path. Pop the bottleneck out as the initial
        // `current_dev`; the remaining outs_down_dev entries
        // (indices 0..n_blocks) are still live as skip inputs.
        let mut current_dev = outs_down_dev.pop()
            .expect("bottleneck pushed in down loop");

        for i in 0..n_blocks {
            let up_idx = n_blocks - 1 - i;
            let is_last = i == n_blocks - 1;

            // Allocate the up-path activation buffer for this
            // level. matvec writes widths[up_idx+1] →
            // widths[up_idx]. On the LAST iteration we still
            // produce the activation in `up_out_dev`, then have
            // the final skip-LN write directly into the caller's
            // `out_dev`; on earlier iterations the skip-LN writes
            // back into `up_out_dev` and we roll it into
            // `current_dev`.
            let mut up_out_dev = GpuVec::try_hip(self.widths[up_idx])?;

            // matvec_resident widths[up_idx+1] → widths[up_idx].
            self.up_blocks[up_idx].forward(batch, &current_dev, &mut up_out_dev)?;

            // LN + SiLU in-place on up_out_dev (matches host
            // `SynapseBlock::forward`: Linear → LN → SiLU).
            self.layer_norm_silu_resident_inplace(
                batch,
                &mut up_out_dev,
                &self.up_ln_gamma_dev[up_idx],
                &self.up_ln_beta_dev[up_idx],
                self.widths[up_idx],
                silu_scratch_ptr,
                LN_EPS,
            )?;

            // Skip-add: up_out_dev += outs_down_dev[up_idx].
            // op_tensor_resident with alpha1=alpha2=1, beta=0:
            //   c = 1*a + 1*b + 0*c = a + b.
            // We aliase c=a (write back into up_out_dev). MIOpen
            // permits read-then-write aliasing per element.
            let skip_dev = &outs_down_dev[up_idx];
            let skip_ptr = match skip_dev {
                GpuVec::Hip(buf) => buf.device_ptr() as *const f32,
                _ => unreachable!("outs_down_dev populated with try_hip"),
            };
            let up_out_ptr_const = match &up_out_dev {
                GpuVec::Hip(buf) => buf.device_ptr() as *const f32,
                _ => unreachable!("we just allocated try_hip"),
            };
            let up_out_ptr_mut = match &up_out_dev {
                GpuVec::Hip(buf) => buf.device_ptr() as *mut f32,
                _ => unreachable!("we just allocated try_hip"),
            };
            unsafe {
                modgrad_device::backend::ops::op_tensor_resident(
                    up_out_ptr_const,
                    skip_ptr,
                    up_out_ptr_mut,
                    self.widths[up_idx],
                    1.0, 1.0, 0.0,
                    BinaryOpKind::Add,
                )?;
            }
            batch.note_dispatch()?;

            // Affine LN on the sum (matches host
            // `affine_layer_norm`). On the last up-iteration the
            // LN writes directly into the caller's `out_dev`;
            // otherwise it writes back into up_out_dev and we
            // roll that into current_dev for the next iteration.
            let gamma_ptr = self.skip_ln_gamma_dev[up_idx].device_ptr() as *const f32;
            let beta_ptr = self.skip_ln_beta_dev[up_idx].device_ptr() as *const f32;
            if is_last {
                let out_ptr = match out_dev {
                    GpuVec::Hip(buf) => buf.device_ptr() as *mut f32,
                    _ => unreachable!("checked above"),
                };
                unsafe {
                    modgrad_device::backend::ops::layer_norm_resident(
                        up_out_ptr_const,
                        gamma_ptr,
                        beta_ptr,
                        out_ptr,
                        1, self.widths[up_idx], LN_EPS,
                    )?;
                }
                batch.note_dispatch()?;
                // current_dev is dropped at scope end — its
                // allocation is reclaimed by hipFree.
            } else {
                // Run LN with up_out_dev as both input and
                // output; the rocm dispatch path reads x then
                // writes y, so element-wise self-aliasing is
                // safe (the kernel completes one (mean,rstd)
                // computation per row before writing the row).
                unsafe {
                    modgrad_device::backend::ops::layer_norm_resident(
                        up_out_ptr_const,
                        gamma_ptr,
                        beta_ptr,
                        up_out_ptr_mut,
                        1, self.widths[up_idx], LN_EPS,
                    )?;
                }
                batch.note_dispatch()?;
                current_dev = up_out_dev;
            }
        }

        Ok(())
    }

    /// Resident in-place LN + SiLU on a `GpuVec::Hip` buffer.
    ///
    /// Sequence:
    ///   1. `layer_norm_resident(x_dev, gamma_dev, beta_dev → x_dev)`
    ///   2. `activation_resident(Logistic, x_dev → silu_scratch_ptr)`
    ///   3. `op_tensor_resident(Mul, x_dev, silu_scratch_ptr → x_dev)`
    ///
    /// Caller supplies the LN parameter buffers and a scratch
    /// pointer of capacity ≥ `n` floats. Each dispatch is logged
    /// to the batch via `note_dispatch` for the auto-sync cadence.
    /// `n` is the number of floats to process (the buffer's
    /// effective length for this call; the underlying allocation
    /// may be larger if the scratch is shared across blocks).
    #[allow(clippy::too_many_arguments)]
    fn layer_norm_silu_resident_inplace(
        &self,
        batch: &modgrad_device::backend::HipBatch,
        x_dev: &mut modgrad_compute::backend::GpuVec,
        gamma_dev: &modgrad_device::backend::HipBuffer,
        beta_dev: &modgrad_device::backend::HipBuffer,
        n: usize,
        silu_scratch_ptr: *mut f32,
        epsilon: f32,
    ) -> Result<(), modgrad_compute::backend::ResidencyError> {
        use modgrad_compute::backend::GpuVec;
        use modgrad_device::backend::{ActivationMode, BinaryOpKind};

        let x_ptr_const = match x_dev {
            GpuVec::Hip(buf) => buf.device_ptr() as *const f32,
            _ => unreachable!("checked at forward entry"),
        };
        let x_ptr_mut = match x_dev {
            GpuVec::Hip(buf) => buf.device_ptr() as *mut f32,
            _ => unreachable!("checked at forward entry"),
        };

        // Step 1: in-place LN. MIOpen reads x and writes y; with
        // x == y the kernel completes per-row stats before
        // writing the row, so self-aliasing is safe.
        unsafe {
            modgrad_device::backend::ops::layer_norm_resident(
                x_ptr_const,
                gamma_dev.device_ptr() as *const f32,
                beta_dev.device_ptr() as *const f32,
                x_ptr_mut,
                1, n, epsilon,
            )?;
        }
        batch.note_dispatch()?;

        // Step 2: sigmoid(x) → silu_scratch. SiLU is
        // sigmoid(x) * x; we compute the gate into scratch so
        // the multiply step in step 3 still sees the original
        // (post-LN) x untouched until after the read.
        unsafe {
            modgrad_device::backend::ops::activation_resident(
                x_ptr_const,
                silu_scratch_ptr,
                n,
                ActivationMode::Logistic,
            )?;
        }
        batch.note_dispatch()?;

        // Step 3: x = x * sigmoid(x). MIOpen permits self-aliased
        // c = a in OpTensor (read-then-write per element).
        unsafe {
            modgrad_device::backend::ops::op_tensor_resident(
                x_ptr_const,
                silu_scratch_ptr as *const f32,
                x_ptr_mut,
                n,
                1.0, 1.0, 0.0,
                BinaryOpKind::Mul,
            )?;
        }
        batch.note_dispatch()?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn widths_match_linspace() {
        // np.linspace(256, 16, 4) = [256, 176, 96, 16]
        let w = compute_widths(256, 16, 4);
        assert_eq!(w.len(), 4);
        assert_eq!(w[0], 256);
        assert_eq!(w[3], 16);
        // Intermediate values should be monotonically decreasing
        assert!(w[1] > w[2]);
        assert!(w[2] > w[3]);
    }

    #[test]
    fn unet_forward_dims() {
        let unet = SynapseUNet::new(128, 64, 4, 16);
        let input = vec![0.1f32; 128];
        let out = unet.forward(&input);
        // Output should be widths[0] = out_dims = 64
        assert_eq!(out.len(), 64);
    }

    #[test]
    fn unet_depth1_is_single_block() {
        let unet = SynapseUNet::new(32, 16, 1, 16);
        assert_eq!(unet.down_blocks.len(), 0);
        assert_eq!(unet.up_blocks.len(), 0);
        let out = unet.forward(&vec![0.5; 32]);
        assert_eq!(out.len(), 16);
    }

    /// SynapseUNetResident matches the host `SynapseUNet` within
    /// 1e-3 FP tolerance (rocBLAS uses a different reduction
    /// order from the CPU dot kernels). Builds a small U-Net with
    /// random init, runs the same input through both paths.
    #[cfg(feature = "rocm")]
    #[test]
    fn synapse_unet_resident_matches_host() {
        use modgrad_compute::backend::GpuVec;
        use modgrad_device::backend::HipBatch;
        use modgrad_device::backend::rocm::ffi::runtime_available;
        use modgrad_compute::neuron::SimpleRng;

        if !runtime_available() {
            eprintln!("hip runtime unavailable, skipping");
            return;
        }
        // Small but non-trivial: in=64, out=32, depth=3, min_width=16.
        // widths = linspace(32,16,3) = [32, 24, 16].
        let unet = SynapseUNet::new(64, 32, 3, 16);
        let mut rng = SimpleRng::new(0xC7_5_3_7);
        let host_x: Vec<f32> = (0..64).map(|_| rng.next_normal()).collect();

        // Host path
        let host_y = unet.forward(&host_x);

        // Resident path
        let resident = SynapseUNetResident::from_synapse_unet(&unet)
            .expect("SynapseUNetResident::from_synapse_unet");
        let mut x_dev = GpuVec::try_hip(64).expect("alloc x");
        x_dev.copy_from(&host_x);
        let mut out_dev = GpuVec::try_hip(host_y.len()).expect("alloc out");
        let batch = HipBatch::new();
        resident.forward(&batch, &x_dev, &mut out_dev)
            .expect("resident forward");
        batch.flush().expect("flush");
        let mut device_y = vec![0.0f32; host_y.len()];
        out_dev.copy_to_host(&mut device_y);

        let max_diff = host_y.iter().zip(&device_y)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-3,
            "host vs resident mismatch: max |Δ| = {max_diff}");
    }

    /// Looping the same input through `SynapseUNetResident::forward`
    /// must produce bit-identical output every iteration. Weights
    /// stay resident, no PRNG fires, and the host-side activation
    /// passes are deterministic — so any drift signals a queue-state
    /// or sync bug, not legitimate FP variation.
    #[cfg(feature = "rocm")]
    #[test]
    fn synapse_unet_resident_loop_no_drift() {
        use modgrad_compute::backend::GpuVec;
        use modgrad_device::backend::HipBatch;
        use modgrad_device::backend::rocm::ffi::runtime_available;
        use modgrad_compute::neuron::SimpleRng;

        if !runtime_available() {
            eprintln!("hip runtime unavailable, skipping");
            return;
        }
        let unet = SynapseUNet::new(48, 24, 3, 12);
        let mut rng = SimpleRng::new(0xD7_1F7);
        let host_x: Vec<f32> = (0..48).map(|_| rng.next_normal()).collect();

        let resident = SynapseUNetResident::from_synapse_unet(&unet)
            .expect("resident");
        let mut x_dev = GpuVec::try_hip(48).expect("x");
        x_dev.copy_from(&host_x);

        let batch = HipBatch::new();
        let mut first_y: Option<Vec<f32>> = None;
        for _ in 0..16 {
            let mut out_dev = GpuVec::try_hip(unet.widths[0]).expect("out");
            resident.forward(&batch, &x_dev, &mut out_dev).expect("forward");
            // copy_to_host syncs against the default stream so the
            // read is safe mid-batch.
            let mut host_y = vec![0.0f32; unet.widths[0]];
            out_dev.copy_to_host(&mut host_y);
            match &first_y {
                None => first_y = Some(host_y),
                Some(y0) => assert_eq!(*y0, host_y, "drift across calls"),
            }
        }
    }
}
