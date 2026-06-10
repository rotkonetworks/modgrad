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

// ─── SynapseBlock<D> + SynapseUNet<D> — typed JAX-style ──────
//
// Path C ports of the U-Net synapse. Replaces the bifurcated host
// (`SynapseBlock`/`SynapseUNet`) + resident (`SynapseUNetResident`)
// pair with a single device-generic struct. `D = Cpu` runs on the
// host path, `D = Rocm` runs entirely device-resident, with the
// SAME forward code calling typed primitives that route to native
// kernels per device.
//
// v0 scope: forward only. Backward is a separate slice — needs
// matvec_t + outer_product_acc + layer_norm_bwd + silu_bwd composed
// in reverse-tick order with skip-connection adjoint. See review
// notes for the full backward composition.

use modgrad_device::backend::tensor as tensor_api;
use modgrad_device::backend::tensor::{Device, Tensor};
use modgrad_device::backend::BackendError;

/// One typed synapse block: `Linear → LayerNorm(γ,β) → SiLU`.
/// Same struct works on every `D: Device`.
pub struct SynapseBlockTyped<D: Device> {
    pub linear: tensor_api::Linear<D>,
    pub ln_gamma: Tensor<D>,
    pub ln_beta: Tensor<D>,
    pub out_dim: usize,
}

impl<D: Device> SynapseBlockTyped<D> {
    /// Construct from host-side weight/bias/gamma/beta buffers.
    /// Uploads to device on construction; subsequent `forward` calls
    /// have zero PCIe round-trips when `D = Rocm`.
    pub fn from_host(
        weight: &[f32],
        bias: &[f32],
        gamma: &[f32],
        beta: &[f32],
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Self, BackendError> {
        if gamma.len() != out_dim {
            return Err(BackendError::Runtime(format!(
                "SynapseBlockTyped::from_host: gamma len {} != out_dim {}",
                gamma.len(), out_dim,
            )));
        }
        if beta.len() != out_dim {
            return Err(BackendError::Runtime(format!(
                "SynapseBlockTyped::from_host: beta len {} != out_dim {}",
                beta.len(), out_dim,
            )));
        }
        Ok(Self {
            linear: tensor_api::Linear::<D>::from_host(weight, bias, in_dim, out_dim)?,
            ln_gamma: Tensor::<D>::from_slice(gamma)?,
            ln_beta: Tensor::<D>::from_slice(beta)?,
            out_dim,
        })
    }

    /// Convert from an existing untyped `SynapseBlock` (uploads to D).
    /// Useful for migrating a checkpoint loaded via the old serde path.
    pub fn from_untyped(block: &SynapseBlock) -> Result<Self, BackendError> {
        let in_dim = block.linear.weight.len() / block.linear.bias.len();
        let out_dim = block.linear.bias.len();
        Self::from_host(
            &block.linear.weight, &block.linear.bias,
            &block.ln_gamma, &block.ln_beta,
            in_dim, out_dim,
        )
    }

    /// Forward: `y = SiLU(LayerNorm_{γ,β}(W·x + b))`. All ops route
    /// through the typed cascade — same path on Cpu and Rocm.
    pub fn forward(&self, x: &Tensor<D>) -> Result<Tensor<D>, BackendError> {
        // Step 1: linear (matvec).
        let pre_ln = self.linear.forward(x)?;
        // Step 2: LayerNorm. Single-row apply (n_rows=1, n_cols=out_dim).
        let mut post_ln = Tensor::<D>::zeros(self.out_dim)?;
        tensor_api::layer_norm(&pre_ln, &self.ln_gamma, &self.ln_beta,
            &mut post_ln, 1, self.out_dim)?;
        // Step 3: SiLU.
        let mut out = Tensor::<D>::zeros(self.out_dim)?;
        tensor_api::silu(&post_ln, &mut out)?;
        Ok(out)
    }

    /// Forward with cache — returns `(output, cache)` where cache
    /// holds every intermediate the matched `backward` consumes.
    /// Used by `SynapseUNetTyped<D>::backward` to walk reverse.
    pub fn forward_with_cache(
        &self,
        x: &Tensor<D>,
    ) -> Result<(Tensor<D>, SynapseBlockCacheTyped<D>), BackendError> {
        // Stash the Linear input by reading back its host content
        // (small: in_dim per block, dwarfed by matmul cost).
        let lin_in_host: Vec<f32> = x.to_vec()?;
        let pre_ln = self.linear.forward(x)?;
        let pre_ln_host: Vec<f32> = pre_ln.to_vec()?;

        // LayerNorm with cache (mean + rstd, length 2*n_rows = 2).
        let mut post_ln = Tensor::<D>::zeros(self.out_dim)?;
        let mut ln_cache = Tensor::<D>::zeros(2)?;
        tensor_api::layer_norm_train(&pre_ln, &self.ln_gamma, &self.ln_beta,
            &mut post_ln, &mut ln_cache, 1, self.out_dim)?;
        let post_ln_host: Vec<f32> = post_ln.to_vec()?;

        // SiLU.
        let mut out = Tensor::<D>::zeros(self.out_dim)?;
        tensor_api::silu(&post_ln, &mut out)?;

        let cache = SynapseBlockCacheTyped {
            linear_in: Tensor::<D>::from_slice(&lin_in_host)?,
            ln_in: Tensor::<D>::from_slice(&pre_ln_host)?,
            ln_cache,
            silu_in: Tensor::<D>::from_slice(&post_ln_host)?,
        };
        Ok((out, cache))
    }

    /// Backward through one block: SiLU → LayerNorm → Linear (in
    /// reverse). Given upstream `d_out` (gradient w.r.t. block
    /// output), accumulate weight gradients (`d_W`, `d_b`,
    /// `d_gamma`, `d_beta`) and overwrite `d_input`.
    ///
    /// **Output convention:**
    ///   - `d_W`, `d_b`, `d_gamma`, `d_beta` ACCUMULATE
    ///   - `d_input` OVERWRITTEN
    pub fn backward(
        &self,
        d_out: &Tensor<D>,
        cache: &SynapseBlockCacheTyped<D>,
        d_w: &mut Tensor<D>,
        d_b: &mut Tensor<D>,
        d_gamma: &mut Tensor<D>,
        d_beta: &mut Tensor<D>,
        d_input: &mut Tensor<D>,
    ) -> Result<(), BackendError> {
        // Step 1: silu_bwd → d_post_ln.
        let mut d_post_ln = Tensor::<D>::zeros(self.out_dim)?;
        tensor_api::silu_bwd(d_out, &cache.silu_in, &mut d_post_ln)?;

        // Step 2: layer_norm_bwd → d_pre_ln. Accumulates into d_gamma/d_beta.
        let mut d_pre_ln = Tensor::<D>::zeros(self.out_dim)?;
        tensor_api::layer_norm_bwd(
            &d_post_ln, &cache.ln_in, &self.ln_gamma, &cache.ln_cache,
            &mut d_pre_ln, d_gamma, d_beta,
            1, self.out_dim,
        )?;

        // Step 3: Linear backward → d_input. Accumulates into d_W, d_b.
        self.linear.backward(&d_pre_ln, &cache.linear_in, d_w, d_b, d_input)?;

        Ok(())
    }

    /// Batched forward — `x` is `[batch × in_dim]`, output `[batch × out_dim]`.
    /// Same chain as `forward_with_cache`, but the Linear is a GEMM and the
    /// LayerNorm/SiLU run per-row over `batch` rows (they already take
    /// `n_rows`). The matmul win flows straight through.
    pub fn forward_batched(
        &self,
        x: &Tensor<D>,
        batch: usize,
    ) -> Result<(Tensor<D>, SynapseBlockCacheTyped<D>), BackendError> {
        let lin_in_host: Vec<f32> = x.to_vec()?;
        let mut pre_ln = Tensor::<D>::zeros(batch * self.out_dim)?;
        self.linear.forward_batched(x, &mut pre_ln, batch)?;
        let pre_ln_host: Vec<f32> = pre_ln.to_vec()?;

        let mut post_ln = Tensor::<D>::zeros(batch * self.out_dim)?;
        let mut ln_cache = Tensor::<D>::zeros(2 * batch)?; // mean+rstd per row
        tensor_api::layer_norm_train(&pre_ln, &self.ln_gamma, &self.ln_beta,
            &mut post_ln, &mut ln_cache, batch, self.out_dim)?;
        let post_ln_host: Vec<f32> = post_ln.to_vec()?;

        let mut out = Tensor::<D>::zeros(batch * self.out_dim)?;
        tensor_api::silu(&post_ln, &mut out)?;

        let cache = SynapseBlockCacheTyped {
            linear_in: Tensor::<D>::from_slice(&lin_in_host)?,
            ln_in: Tensor::<D>::from_slice(&pre_ln_host)?,
            ln_cache,
            silu_in: Tensor::<D>::from_slice(&post_ln_host)?,
        };
        Ok((out, cache))
    }

    /// Batched backward. `d_out` is `[batch × out_dim]`; `d_input`
    /// `[batch × in_dim]` overwritten; `d_w/d_b/d_gamma/d_beta` accumulate
    /// (batch-summed). Equivalent to `batch` scalar `backward` calls.
    pub fn backward_batched(
        &self,
        d_out: &Tensor<D>,
        cache: &SynapseBlockCacheTyped<D>,
        d_w: &mut Tensor<D>,
        d_b: &mut Tensor<D>,
        d_gamma: &mut Tensor<D>,
        d_beta: &mut Tensor<D>,
        d_input: &mut Tensor<D>,
        batch: usize,
    ) -> Result<(), BackendError> {
        let mut d_post_ln = Tensor::<D>::zeros(batch * self.out_dim)?;
        tensor_api::silu_bwd(d_out, &cache.silu_in, &mut d_post_ln)?;

        let mut d_pre_ln = Tensor::<D>::zeros(batch * self.out_dim)?;
        // `layer_norm_bwd` ZEROS d_gamma/d_beta internally (overwrite),
        // unlike the Linear grads which accumulate. Run it into temps and
        // add, so d_gamma/d_beta follow the same accumulate convention as
        // d_w/d_b (caller zeros once per batch; this batch's row-sum adds in).
        let mut dg_tmp = Tensor::<D>::zeros(self.out_dim)?;
        let mut dbe_tmp = Tensor::<D>::zeros(self.out_dim)?;
        tensor_api::layer_norm_bwd(
            &d_post_ln, &cache.ln_in, &self.ln_gamma, &cache.ln_cache,
            &mut d_pre_ln, &mut dg_tmp, &mut dbe_tmp,
            batch, self.out_dim,
        )?;
        tensor_api::add_assign(d_gamma, &dg_tmp, self.out_dim)?;
        tensor_api::add_assign(d_beta, &dbe_tmp, self.out_dim)?;

        self.linear.backward_batched(&d_pre_ln, &cache.linear_in, d_w, d_b, d_input, batch)?;
        Ok(())
    }
}

/// Cache produced by `SynapseBlockTyped::forward_with_cache`.
/// Holds every input that `backward` reads. The `ln_cache` is the
/// mean/rstd buffer in the same layout `Tensor::layer_norm_train` writes.
pub struct SynapseBlockCacheTyped<D: Device> {
    pub linear_in: Tensor<D>,
    pub ln_in: Tensor<D>,
    pub ln_cache: Tensor<D>,
    pub silu_in: Tensor<D>,
}

/// Typed U-Net synapse — generic over device. Mirrors the untyped
/// `SynapseUNet` structure: first projection → down path → up path
/// with skip connections + per-level LayerNorm.
pub struct SynapseUNetTyped<D: Device> {
    pub widths: Vec<usize>,
    pub first_projection: SynapseBlockTyped<D>,
    pub down_blocks: Vec<SynapseBlockTyped<D>>,
    pub up_blocks: Vec<SynapseBlockTyped<D>>,
    pub skip_ln_gamma: Vec<Tensor<D>>,
    pub skip_ln_beta: Vec<Tensor<D>>,
}

impl<D: Device> SynapseUNetTyped<D> {
    /// Convert from an existing untyped `SynapseUNet`. The host
    /// weights/biases are uploaded to D; subsequent forward calls
    /// stay on-device for `D = Rocm`.
    pub fn from_untyped(unet: &SynapseUNet) -> Result<Self, BackendError> {
        let first_projection = SynapseBlockTyped::<D>::from_untyped(&unet.first_projection)?;
        let down_blocks = unet.down_blocks.iter()
            .map(SynapseBlockTyped::<D>::from_untyped)
            .collect::<Result<Vec<_>, _>>()?;
        let up_blocks = unet.up_blocks.iter()
            .map(SynapseBlockTyped::<D>::from_untyped)
            .collect::<Result<Vec<_>, _>>()?;
        let skip_ln_gamma = unet.skip_ln_gamma.iter()
            .map(|g| Tensor::<D>::from_slice(g))
            .collect::<Result<Vec<_>, _>>()?;
        let skip_ln_beta = unet.skip_ln_beta.iter()
            .map(|b| Tensor::<D>::from_slice(b))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            widths: unet.widths.clone(),
            first_projection, down_blocks, up_blocks,
            skip_ln_gamma, skip_ln_beta,
        })
    }

    /// Forward pass — matches `SynapseUNet::forward` semantically.
    /// Algorithm:
    ///   1. first_projection(x) → outs_down[0]
    ///   2. down: outs_down[i+1] = down_blocks[i](outs_down[i])
    ///   3. up (in reverse): current = up_blocks[up_idx](current)
    ///                       current += outs_down[up_idx]   (skip add)
    ///                       current = LayerNorm_skip[up_idx](current)
    pub fn forward(&self, x: &Tensor<D>) -> Result<Tensor<D>, BackendError> {
        let n_blocks = self.down_blocks.len();
        let mut outs_down: Vec<Tensor<D>> = Vec::with_capacity(n_blocks + 1);
        outs_down.push(self.first_projection.forward(x)?);
        for i in 0..n_blocks {
            let next = self.down_blocks[i].forward(outs_down.last().unwrap())?;
            outs_down.push(next);
        }
        // Take the bottleneck out of outs_down (we won't need it as a skip).
        let mut current = outs_down.pop().expect("bottleneck present");
        for i in 0..n_blocks {
            let up_idx = n_blocks - 1 - i;
            // Up projection.
            let up_out = self.up_blocks[up_idx].forward(&current)?;
            // Skip add: up_out += outs_down[up_idx].
            // We swap into a fresh `accumulated` tensor so we can call
            // `add_assign` (which takes &mut + &). Simpler: copy up_out
            // into accumulated via reshape-to-self trick? Just allocate.
            let n = self.widths[up_idx];
            let mut accumulated = up_out;
            tensor_api::add_assign(&mut accumulated, &outs_down[up_idx], n)?;
            // LayerNorm on the sum.
            let mut normed = Tensor::<D>::zeros(n)?;
            tensor_api::layer_norm(&accumulated, &self.skip_ln_gamma[up_idx],
                &self.skip_ln_beta[up_idx], &mut normed, 1, n)?;
            current = normed;
        }
        Ok(current)
    }

    /// Output dimension (= widths[0]).
    pub fn out_dim(&self) -> usize { self.widths[0] }

    /// Forward with full cache — every intermediate the matched
    /// backward consumes (per-block caches, every level's down output
    /// for skip connections, every skip-LN cache).
    pub fn forward_with_cache(
        &self,
        x: &Tensor<D>,
    ) -> Result<(Tensor<D>, SynapseUNetCacheTyped<D>), BackendError> {
        let n_blocks = self.down_blocks.len();

        // First projection.
        let (first_out, first_cache) = self.first_projection.forward_with_cache(x)?;

        // Down path. Save each level's output (host shadow) for skip-conn replay.
        let mut down_caches: Vec<SynapseBlockCacheTyped<D>> = Vec::with_capacity(n_blocks);
        let mut down_outs_host: Vec<Vec<f32>> = Vec::with_capacity(n_blocks + 1);
        let first_out_host: Vec<f32> = first_out.to_vec()?;
        down_outs_host.push(first_out_host);
        let mut current = first_out;
        for i in 0..n_blocks {
            let (next, cache) = self.down_blocks[i].forward_with_cache(&current)?;
            let next_host: Vec<f32> = next.to_vec()?;
            down_outs_host.push(next_host);
            down_caches.push(cache);
            current = next;
        }

        // Up path. For each level (reverse): up_block(current) + skip → ln_skip.
        let mut up_caches: Vec<SynapseBlockCacheTyped<D>> = Vec::with_capacity(n_blocks);
        let mut skip_pre_ln_host: Vec<Vec<f32>> = Vec::with_capacity(n_blocks);
        let mut skip_ln_caches: Vec<Tensor<D>> = Vec::with_capacity(n_blocks);
        // up_caches and skip_*_caches are filled in REVERSE-loop order
        // (up_idx = n_blocks-1 first, then n_blocks-2, ...). We store in
        // that order for direct backward consumption.
        for i in 0..n_blocks {
            let up_idx = n_blocks - 1 - i;
            let (up_out, up_cache) = self.up_blocks[up_idx].forward_with_cache(&current)?;
            up_caches.push(up_cache);

            // Skip add: pre_skip_ln = up_out + down_outs[up_idx].
            let mut pre_skip = up_out;
            tensor_api::add_assign(
                &mut pre_skip,
                &Tensor::<D>::from_slice(&down_outs_host[up_idx])?,
                self.widths[up_idx],
            )?;
            let pre_skip_h: Vec<f32> = pre_skip.to_vec()?;
            skip_pre_ln_host.push(pre_skip_h);

            // Skip-LN: forward + cache.
            let mut post_ln = Tensor::<D>::zeros(self.widths[up_idx])?;
            let mut ln_cache = Tensor::<D>::zeros(2)?;   // 2*n_rows = 2
            tensor_api::layer_norm_train(
                &pre_skip,
                &self.skip_ln_gamma[up_idx], &self.skip_ln_beta[up_idx],
                &mut post_ln, &mut ln_cache, 1, self.widths[up_idx],
            )?;
            skip_ln_caches.push(ln_cache);
            current = post_ln;
        }

        let cache = SynapseUNetCacheTyped {
            first: first_cache,
            downs: down_caches,
            ups: up_caches,
            down_outs_host,
            skip_pre_ln_host,
            skip_ln_caches,
        };
        Ok((current, cache))
    }

    /// Backward through the full U-Net. Walk reverse:
    ///   1. For each up level (forward order = reverse of up loop):
    ///        layer_norm_bwd → d_pre_skip_ln (split into d_up_out + d_skip)
    ///        d_skip accumulates into the matching down level's gradient
    ///        up_block.backward → d_current_in
    ///   2. After up loop, add d_current_in to bottleneck slot.
    ///   3. Walk down path in reverse: down_block.backward propagates upstream.
    ///   4. first_projection.backward → d_x.
    ///
    /// **Output convention:**
    ///   - All weight grads (per-block + skip LN) ACCUMULATE.
    ///   - `d_x` OVERWRITTEN.
    pub fn backward(
        &self,
        d_out: &Tensor<D>,
        cache: &SynapseUNetCacheTyped<D>,
        grads: &mut UNetGradsTyped<D>,
        d_x: &mut Tensor<D>,
    ) -> Result<(), BackendError> {
        let n_blocks = self.down_blocks.len();

        // d_outs_down accumulates gradient flowing back into each
        // down level (skip connections + main path). Initialised to
        // zero per level; we accumulate via add_assign.
        let mut d_outs_down: Vec<Tensor<D>> = (0..=n_blocks)
            .map(|i| {
                let w = if i < self.widths.len() { self.widths[i] } else { *self.widths.last().unwrap() };
                Tensor::<D>::zeros(w)
            })
            .collect::<Result<Vec<_>, _>>()?;

        // ── Reverse up loop ─────────────────────────────────
        // Forward up loop pushed caches at step k (k=0..N-1) for
        // up_idx = N-1-k. Cache[0] = deepest up_idx (=N-1, reached
        // FIRST in forward). The output of forward is from forward
        // step k=N-1 (up_idx=0), so d_out's flow goes BACKWARD into
        // up_idx=0 first → must consume cache[N-1] first.
        //
        // We walk cache_idx_storage from N-1 down to 0; the
        // corresponding up_idx (= down_level) walks 0 up to N-1.
        let mut current_grad = Tensor::<D>::from_slice(&d_out.to_vec()?)?;
        for step in 0..n_blocks {
            let cache_idx = n_blocks - 1 - step;
            let down_level = step;

            // 1. layer_norm_bwd through skip-LN.
            // Split-borrow trick: lift the two mutable refs into local vars first.
            let pre_skip_ln = Tensor::<D>::from_slice(&cache.skip_pre_ln_host[cache_idx])?;
            let mut d_pre_skip = Tensor::<D>::zeros(self.widths[down_level])?;
            {
                let skip_grads = (
                    &mut grads.skip_d_gamma[down_level],
                    &mut grads.skip_d_beta[down_level],
                );
                tensor_api::layer_norm_bwd(
                    &current_grad, &pre_skip_ln,
                    &self.skip_ln_gamma[down_level],
                    &cache.skip_ln_caches[cache_idx],
                    &mut d_pre_skip,
                    skip_grads.0, skip_grads.1,
                    1, self.widths[down_level],
                )?;
            }

            // 2. Split d_pre_skip into d_up_out and d_skip (sum's adjoint).
            //    Both EQUAL d_pre_skip; we send d_skip into down level
            //    via add_assign and pass d_up_out into the up block.
            let d_skip = Tensor::<D>::from_slice(&d_pre_skip.to_vec()?)?;
            tensor_api::add_assign(
                &mut d_outs_down[down_level], &d_skip,
                self.widths[down_level],
            )?;

            // 3. Up block backward — split-borrow the per-block grads.
            let mut d_up_in = Tensor::<D>::zeros(
                self.up_blocks[down_level].linear.in_dim
            )?;
            {
                let ug = &mut grads.ups[down_level];
                self.up_blocks[down_level].backward(
                    &d_pre_skip, &cache.ups[cache_idx],
                    &mut ug.d_w, &mut ug.d_b, &mut ug.d_gamma, &mut ug.d_beta,
                    &mut d_up_in,
                )?;
            }
            current_grad = d_up_in;
        }

        // After up loop, current_grad is the gradient flowing into
        // the bottleneck (= outs_down[n_blocks]). Add it.
        tensor_api::add_assign(
            &mut d_outs_down[n_blocks], &current_grad,
            *self.widths.last().unwrap(),
        )?;

        // ── Reverse down loop ───────────────────────────────
        for i in (0..n_blocks).rev() {
            let upstream_host: Vec<f32> = d_outs_down[i + 1].to_vec()?;
            let upstream = Tensor::<D>::from_slice(&upstream_host)?;
            let mut d_block_in = Tensor::<D>::zeros(self.down_blocks[i].linear.in_dim)?;
            {
                let dg = &mut grads.downs[i];
                self.down_blocks[i].backward(
                    &upstream, &cache.downs[i],
                    &mut dg.d_w, &mut dg.d_b, &mut dg.d_gamma, &mut dg.d_beta,
                    &mut d_block_in,
                )?;
            }
            tensor_api::add_assign(
                &mut d_outs_down[i], &d_block_in,
                self.widths[i],
            )?;
        }

        // ── First projection backward ───────────────────────
        let first_upstream_host: Vec<f32> = d_outs_down[0].to_vec()?;
        let first_upstream = Tensor::<D>::from_slice(&first_upstream_host)?;
        {
            let fg = &mut grads.first;
            self.first_projection.backward(
                &first_upstream, &cache.first,
                &mut fg.d_w, &mut fg.d_b, &mut fg.d_gamma, &mut fg.d_beta,
                d_x,
            )?;
        }
        Ok(())
    }
}

/// Cache produced by `SynapseUNetTyped::forward_with_cache`.
pub struct SynapseUNetCacheTyped<D: Device> {
    pub first: SynapseBlockCacheTyped<D>,
    pub downs: Vec<SynapseBlockCacheTyped<D>>,
    pub ups: Vec<SynapseBlockCacheTyped<D>>,
    /// Each down-level output, host shadow. Used by backward to
    /// compute skip-conn adjoint without re-uploading.
    pub down_outs_host: Vec<Vec<f32>>,
    /// Pre-skip-LN values per up level (host shadow), for layer_norm_bwd.
    pub skip_pre_ln_host: Vec<Vec<f32>>,
    /// Per-up-level skip-LN cache (mean+rstd). Stored in reverse-of-up-idx
    /// order matching forward push order (cache_idx 0 came first).
    pub skip_ln_caches: Vec<Tensor<D>>,
}

/// Per-block typed gradients — d_w (Linear), d_b (Linear), d_gamma (LN), d_beta (LN).
pub struct SynapseBlockGradsTyped<D: Device> {
    pub d_w: Tensor<D>,
    pub d_b: Tensor<D>,
    pub d_gamma: Tensor<D>,
    pub d_beta: Tensor<D>,
}

impl<D: Device> SynapseBlockGradsTyped<D> {
    pub fn zeros(block: &SynapseBlockTyped<D>) -> Result<Self, BackendError> {
        Ok(Self {
            d_w: Tensor::<D>::zeros(block.linear.weight.len())?,
            d_b: Tensor::<D>::zeros(block.linear.bias.len())?,
            d_gamma: Tensor::<D>::zeros(block.ln_gamma.len())?,
            d_beta: Tensor::<D>::zeros(block.ln_beta.len())?,
        })
    }
}

/// Full U-Net typed gradients — first projection + down + up + skip LN.
pub struct UNetGradsTyped<D: Device> {
    pub first: SynapseBlockGradsTyped<D>,
    pub downs: Vec<SynapseBlockGradsTyped<D>>,
    pub ups: Vec<SynapseBlockGradsTyped<D>>,
    pub skip_d_gamma: Vec<Tensor<D>>,
    pub skip_d_beta: Vec<Tensor<D>>,
}

impl<D: Device> UNetGradsTyped<D> {
    pub fn zeros(unet: &SynapseUNetTyped<D>) -> Result<Self, BackendError> {
        Ok(Self {
            first: SynapseBlockGradsTyped::zeros(&unet.first_projection)?,
            downs: unet.down_blocks.iter()
                .map(SynapseBlockGradsTyped::zeros)
                .collect::<Result<Vec<_>, _>>()?,
            ups: unet.up_blocks.iter()
                .map(SynapseBlockGradsTyped::zeros)
                .collect::<Result<Vec<_>, _>>()?,
            skip_d_gamma: unet.skip_ln_gamma.iter()
                .map(|g| Tensor::<D>::zeros(g.len()))
                .collect::<Result<Vec<_>, _>>()?,
            skip_d_beta: unet.skip_ln_beta.iter()
                .map(|b| Tensor::<D>::zeros(b.len()))
                .collect::<Result<Vec<_>, _>>()?,
        })
    }
}

#[cfg(test)]
mod typed_synapse_tests {
    use super::*;
    use modgrad_device::backend::tensor::Cpu;
    #[cfg(feature = "rocm")]
    use modgrad_device::backend::tensor::Rocm;

    /// Build a small untyped SynapseUNet, run untyped forward + typed
    /// forward through it, assert outputs match within float tolerance.
    /// This is the parity proof: typed cascade reproduces the existing
    /// brain forward semantics exactly.
    #[test]
    fn cpu_typed_synapse_unet_matches_untyped() {
        let unet = SynapseUNet::new(8, 4, 3, 2);
        let x_data: Vec<f32> = (0..8).map(|i| (i as f32 - 3.0) * 0.2).collect();

        // Untyped reference.
        let ref_out = unet.forward(&x_data);

        // Typed.
        let typed = SynapseUNetTyped::<Cpu>::from_untyped(&unet)
            .expect("from_untyped");
        let xt = Tensor::<Cpu>::from_slice(&x_data).unwrap();
        let typed_out = typed.forward(&xt).expect("typed forward")
            .to_vec().unwrap();

        assert_eq!(ref_out.len(), typed_out.len());
        for (i, (a, b)) in ref_out.iter().zip(&typed_out).enumerate() {
            assert!((a - b).abs() < 1e-4,
                "synapse_unet typed/untyped disagree at {}: ref={} typed={}", i, a, b);
        }
    }

    /// Batched `SynapseBlockTyped` forward+backward must equal `batch`
    /// independent scalar calls — the GEMV→GEMM lever for the tick loop's
    /// synapse (its biggest FLOP consumer).
    #[test]
    fn cpu_synapse_block_batched_matches_scalar() {
        let (in_dim, out_dim, batch) = (6usize, 5usize, 4usize);
        let w: Vec<f32> = (0..in_dim * out_dim).map(|i| ((i * 11 % 13) as f32 - 6.0) * 0.1).collect();
        let b: Vec<f32> = (0..out_dim).map(|i| (i as f32 - 2.0) * 0.05).collect();
        let gamma: Vec<f32> = (0..out_dim).map(|i| 1.0 + (i as f32) * 0.05).collect();
        let beta: Vec<f32> = (0..out_dim).map(|i| (i as f32 - 2.0) * 0.03).collect();
        let blk = SynapseBlockTyped::<Cpu>::from_host(&w, &b, &gamma, &beta, in_dim, out_dim).unwrap();

        let x: Vec<f32> = (0..batch * in_dim).map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.2).collect();
        let dy: Vec<f32> = (0..batch * out_dim).map(|i| ((i * 5 % 9) as f32 - 4.0) * 0.15).collect();

        // Batched.
        let xt = Tensor::<Cpu>::from_slice(&x).unwrap();
        let (y_b, cache_b) = blk.forward_batched(&xt, batch).unwrap();
        let dyt = Tensor::<Cpu>::from_slice(&dy).unwrap();
        let mut dw_b = Tensor::<Cpu>::zeros(out_dim * in_dim).unwrap();
        let mut db_b = Tensor::<Cpu>::zeros(out_dim).unwrap();
        let mut dg_b = Tensor::<Cpu>::zeros(out_dim).unwrap();
        let mut dbe_b = Tensor::<Cpu>::zeros(out_dim).unwrap();
        let mut dx_b = Tensor::<Cpu>::zeros(batch * in_dim).unwrap();
        blk.backward_batched(&dyt, &cache_b, &mut dw_b, &mut db_b, &mut dg_b, &mut dbe_b, &mut dx_b, batch).unwrap();

        // Scalar reference. d_w/d_b accumulate (Linear), but scalar
        // block.backward OVERWRITES d_gamma/d_beta (layer_norm_bwd zeros
        // them), so accumulate those by hand over the batch.
        let mut y_s = vec![0.0f32; batch * out_dim];
        let mut dx_s = vec![0.0f32; batch * in_dim];
        let mut dw_s = Tensor::<Cpu>::zeros(out_dim * in_dim).unwrap();
        let mut db_s = Tensor::<Cpu>::zeros(out_dim).unwrap();
        let mut dg_acc = vec![0.0f32; out_dim];
        let mut dbe_acc = vec![0.0f32; out_dim];
        for bi in 0..batch {
            let xr = Tensor::<Cpu>::from_slice(&x[bi * in_dim..(bi + 1) * in_dim]).unwrap();
            let (yr, cr) = blk.forward_with_cache(&xr).unwrap();
            y_s[bi * out_dim..(bi + 1) * out_dim].copy_from_slice(&yr.to_vec().unwrap());
            let dyr = Tensor::<Cpu>::from_slice(&dy[bi * out_dim..(bi + 1) * out_dim]).unwrap();
            let mut dxr = Tensor::<Cpu>::zeros(in_dim).unwrap();
            let mut dg_tmp = Tensor::<Cpu>::zeros(out_dim).unwrap();
            let mut dbe_tmp = Tensor::<Cpu>::zeros(out_dim).unwrap();
            blk.backward(&dyr, &cr, &mut dw_s, &mut db_s, &mut dg_tmp, &mut dbe_tmp, &mut dxr).unwrap();
            dx_s[bi * in_dim..(bi + 1) * in_dim].copy_from_slice(&dxr.to_vec().unwrap());
            for (a, v) in dg_acc.iter_mut().zip(dg_tmp.to_vec().unwrap()) { *a += v; }
            for (a, v) in dbe_acc.iter_mut().zip(dbe_tmp.to_vec().unwrap()) { *a += v; }
        }

        let cmp = |a: &[f32], b: &[f32], n: &str| {
            assert_eq!(a.len(), b.len(), "{n} len");
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                let d = (x - y).abs();
                let s = x.abs().max(y.abs()).max(1.0);
                assert!(d < 1e-4 || d / s < 1e-4, "{n}[{i}]: {x} vs {y}");
            }
        };
        cmp(&y_b.to_vec().unwrap(), &y_s, "fwd");
        cmp(&dx_b.to_vec().unwrap(), &dx_s, "d_x");
        cmp(&dw_b.to_vec().unwrap(), &dw_s.to_vec().unwrap(), "d_w");
        cmp(&db_b.to_vec().unwrap(), &db_s.to_vec().unwrap(), "d_b");
        cmp(&dg_b.to_vec().unwrap(), &dg_acc, "d_gamma");
        cmp(&dbe_b.to_vec().unwrap(), &dbe_acc, "d_beta");
    }

    /// Smoke-test SynapseUNetTyped<D>::backward — full U-Net reverse
    /// composes: skip-LN backward, up block backwards (in reverse
    /// forward order), down block backwards, first projection
    /// backward, with skip-conn adjoint adding into matching down
    /// levels' gradient. Verifies wiring; confirms d_x and a sample
    /// of weight grads become non-zero.
    #[test]
    fn cpu_synapse_unet_backward_smoke() {
        // Small: in=8 out=4 depth=3 min=2 → widths = [4,3,2].
        let unet = SynapseUNet::new(8, 4, 3, 2);
        let typed = SynapseUNetTyped::<Cpu>::from_untyped(&unet).unwrap();

        let x_data: Vec<f32> = (0..8).map(|i| (i as f32 - 3.0) * 0.2).collect();
        let x = Tensor::<Cpu>::from_slice(&x_data).unwrap();
        let (out, cache) = typed.forward_with_cache(&x).unwrap();
        assert_eq!(out.shape(), &[4]);

        let d_out = Tensor::<Cpu>::from_slice(&[1.0, 0.0, -1.0, 0.5]).unwrap();
        let mut grads = UNetGradsTyped::<Cpu>::zeros(&typed).unwrap();
        let mut d_x = Tensor::<Cpu>::zeros(8).unwrap();
        typed.backward(&d_out, &cache, &mut grads, &mut d_x).expect("U-Net backward");

        // d_x should be non-zero (gradient flowed all the way through).
        let dx = d_x.to_vec().unwrap();
        assert!(dx.iter().any(|&v| v.abs() > 1e-6),
            "d_x all zero — backward did not flow through first_projection");

        // First projection weight grad must be non-zero.
        let dw = grads.first.d_w.to_vec().unwrap();
        assert!(dw.iter().any(|&v| v.abs() > 1e-6),
            "first.d_w all zero");

        // Skip-LN gamma grad on the deepest skip level — should be non-zero.
        let dg = grads.skip_d_gamma[0].to_vec().unwrap();
        assert!(dg.iter().any(|&v| v.abs() > 1e-6),
            "skip_d_gamma[0] all zero");
    }

    /// Smoke-test SynapseBlockTyped<D>::backward — verify gradient
    /// flow without exact numerical parity (full parity vs untyped
    /// block_backward needs careful cache-format alignment, deferred).
    /// Confirms d_W, d_b, d_gamma, d_beta all become non-zero and
    /// d_input has matching shape.
    #[test]
    fn cpu_synapse_block_backward_smoke() {
        // Build a small block: in_dim=4, out_dim=3.
        // Random-ish weights, bias zero, gamma=1, beta=0.
        let weights: Vec<f32> = (0..3 * 4).map(|i| (i as f32 - 5.5) * 0.1).collect();
        let bias = vec![0.0f32; 3];
        let gamma = vec![1.0f32; 3];
        let beta = vec![0.0f32; 3];
        let block = SynapseBlockTyped::<Cpu>::from_host(
            &weights, &bias, &gamma, &beta, 4, 3,
        ).unwrap();

        let x = Tensor::<Cpu>::from_slice(&[0.5, -0.5, 0.25, 0.75]).unwrap();
        let (out, cache) = block.forward_with_cache(&x).unwrap();
        assert_eq!(out.shape(), &[3]);

        // Hand-cooked d_out.
        let d_out = Tensor::<Cpu>::from_slice(&[1.0, 0.0, -1.0]).unwrap();
        let mut d_w = Tensor::<Cpu>::zeros(3 * 4).unwrap();
        let mut d_b = Tensor::<Cpu>::zeros(3).unwrap();
        let mut d_gamma = Tensor::<Cpu>::zeros(3).unwrap();
        let mut d_beta = Tensor::<Cpu>::zeros(3).unwrap();
        let mut d_input = Tensor::<Cpu>::zeros(4).unwrap();
        block.backward(&d_out, &cache, &mut d_w, &mut d_b,
            &mut d_gamma, &mut d_beta, &mut d_input).unwrap();

        // d_input non-zero (gradient flowed all the way).
        let di = d_input.to_vec().unwrap();
        assert!(di.iter().any(|&v| v.abs() > 1e-6),
            "d_input all zero — backward did not flow through Linear");
        // d_beta = sum of d_post_ln across rows — should equal d_post_ln
        // at this single-row config since silu_bwd preserves sign of d_out.
        let db = d_beta.to_vec().unwrap();
        assert!(db.iter().any(|&v| v.abs() > 1e-6),
            "d_beta all zero");
    }

    /// Same code on Rocm produces matching output (within float tol).
    /// The pay-off: the entire SynapseUNet now runs on AMD GPU using
    /// the same Rust source as Cpu.
    #[cfg(feature = "rocm")]
    #[test]
    fn cpu_and_rocm_typed_synapse_unet_match() {
        let unet = SynapseUNet::new(8, 4, 3, 2);
        let x_data: Vec<f32> = (0..8).map(|i| (i as f32 - 3.0) * 0.2).collect();

        let cpu_typed = SynapseUNetTyped::<Cpu>::from_untyped(&unet).unwrap();
        let xc = Tensor::<Cpu>::from_slice(&x_data).unwrap();
        let cpu_out = cpu_typed.forward(&xc).unwrap().to_vec().unwrap();

        let rocm_typed = match SynapseUNetTyped::<Rocm>::from_untyped(&unet) {
            Ok(t) => t,
            Err(BackendError::Runtime(msg)) if msg.contains("hipMalloc") => return,
            Err(e) => panic!("rocm SynapseUNet build failed: {:?}", e),
        };
        let xr = Tensor::<Rocm>::from_slice(&x_data).unwrap();
        let rocm_out = match rocm_typed.forward(&xr) {
            Ok(t) => t.to_vec().unwrap(),
            Err(BackendError::Runtime(msg)) if msg.contains("hipMalloc") => return,
            Err(BackendError::Unsupported { .. }) => return,
            Err(e) => panic!("rocm forward failed: {:?}", e),
        };

        for (a, b) in cpu_out.iter().zip(&rocm_out) {
            assert!((a - b).abs() < 1e-3,
                "Cpu and Rocm SynapseUNet disagree: cpu={} rocm={}", a, b);
        }
    }
}
