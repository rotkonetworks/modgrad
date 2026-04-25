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
// stack of `LinearResident`s with weights pinned in VRAM. Mirrors
// the lifecycle of `LinearResident`/`RegionalResidentCache`: build
// once via `from_synapse_unet`, refresh after optimizer steps via
// `sync_weights_from`, dispatch through `forward` with a `&HipBatch`.
//
// Activation roundtrip: `modgrad_device::backend::ops` exposes
// `MatvecResident` but no `LayerNormResident` or `LnSiluResident`.
// Per the brief: path (b) — between matvec dispatches we
// download into host scratch, apply `ln_silu_fwd` (and the affine
// LN + skip-add on the up path) on the host, then re-upload. Each
// matmul still runs on the GPU with weights resident in VRAM, but
// every block pays a D2H + H2D for the activation. This is the
// realistic floor until per-op resident kernels land. TODO: lift
// the LN+SiLU and skip-add+LN passes into resident kernels — see
// `memory/project_gpu_residency.md`.

#[cfg(feature = "rocm")]
pub struct SynapseUNetResident {
    /// Layer widths (mirrors the host `SynapseUNet`). Used to
    /// size scratch buffers per block.
    pub widths: Vec<usize>,

    /// Initial projection: in_dims → widths[0]. The Linear weight
    /// + bias live on-device; LN gamma/beta stay host-side because
    /// the LN+SiLU pass is not yet a resident kernel.
    pub first_projection: modgrad_compute::neuron::LinearResident,
    pub first_ln_gamma: Vec<f32>,
    pub first_ln_beta: Vec<f32>,

    /// Down blocks: widths[i] → widths[i+1].
    pub down_blocks: Vec<modgrad_compute::neuron::LinearResident>,
    pub down_ln_gamma: Vec<Vec<f32>>,
    pub down_ln_beta: Vec<Vec<f32>>,

    /// Up blocks: widths[i+1] → widths[i] (stored in down-order,
    /// applied in reverse).
    pub up_blocks: Vec<modgrad_compute::neuron::LinearResident>,
    pub up_ln_gamma: Vec<Vec<f32>>,
    pub up_ln_beta: Vec<Vec<f32>>,

    /// Skip-connection LayerNorm parameters (one per up level).
    pub skip_ln_gamma: Vec<Vec<f32>>,
    pub skip_ln_beta: Vec<Vec<f32>>,

    /// Host-side dim cached for the input projection (= in_dim of
    /// `first_projection`). Used to validate `forward` arguments.
    pub in_dim: usize,
    /// Host-side dim cached for the U-Net output (= widths[0]).
    pub out_dim: usize,
}

#[cfg(feature = "rocm")]
impl SynapseUNetResident {
    /// Allocate device buffers and upload every Linear's weight +
    /// bias. LN gamma/beta arrays are cloned to host-owned `Vec`s
    /// because the LN+SiLU pass is not yet a resident kernel — see
    /// the H2D/D2H roundtrip note in `forward`.
    pub fn from_synapse_unet(
        unet: &SynapseUNet,
    ) -> Result<Self, modgrad_compute::backend::ResidencyError> {
        use modgrad_compute::neuron::LinearResident;
        let first_projection = LinearResident::from_linear(&unet.first_projection.linear)?;
        let down_blocks = unet.down_blocks.iter()
            .map(|b| LinearResident::from_linear(&b.linear))
            .collect::<Result<Vec<_>, _>>()?;
        let up_blocks = unet.up_blocks.iter()
            .map(|b| LinearResident::from_linear(&b.linear))
            .collect::<Result<Vec<_>, _>>()?;
        let down_ln_gamma = unet.down_blocks.iter().map(|b| b.ln_gamma.clone()).collect();
        let down_ln_beta = unet.down_blocks.iter().map(|b| b.ln_beta.clone()).collect();
        let up_ln_gamma = unet.up_blocks.iter().map(|b| b.ln_gamma.clone()).collect();
        let up_ln_beta = unet.up_blocks.iter().map(|b| b.ln_beta.clone()).collect();
        Ok(Self {
            widths: unet.widths.clone(),
            first_projection,
            first_ln_gamma: unet.first_projection.ln_gamma.clone(),
            first_ln_beta: unet.first_projection.ln_beta.clone(),
            down_blocks,
            down_ln_gamma,
            down_ln_beta,
            up_blocks,
            up_ln_gamma,
            up_ln_beta,
            skip_ln_gamma: unet.skip_ln_gamma.clone(),
            skip_ln_beta: unet.skip_ln_beta.clone(),
            in_dim: unet.first_projection.linear.in_dim,
            out_dim: unet.widths[0],
        })
    }

    /// Re-upload every Linear's weight + bias and refresh the
    /// host-side LN parameter caches. Call after every optimizer
    /// step that mutated the host `SynapseUNet`.
    pub fn sync_weights_from(
        &mut self, unet: &SynapseUNet,
    ) -> Result<(), modgrad_compute::backend::ResidencyError> {
        debug_assert_eq!(self.down_blocks.len(), unet.down_blocks.len());
        debug_assert_eq!(self.up_blocks.len(), unet.up_blocks.len());
        self.first_projection.sync_weights_from(&unet.first_projection.linear)?;
        self.first_ln_gamma.copy_from_slice(&unet.first_projection.ln_gamma);
        self.first_ln_beta.copy_from_slice(&unet.first_projection.ln_beta);
        for (r, b) in self.down_blocks.iter_mut().zip(&unet.down_blocks) {
            r.sync_weights_from(&b.linear)?;
        }
        for (g, b) in self.down_ln_gamma.iter_mut().zip(&unet.down_blocks) {
            g.copy_from_slice(&b.ln_gamma);
        }
        for (g, b) in self.down_ln_beta.iter_mut().zip(&unet.down_blocks) {
            g.copy_from_slice(&b.ln_beta);
        }
        for (r, b) in self.up_blocks.iter_mut().zip(&unet.up_blocks) {
            r.sync_weights_from(&b.linear)?;
        }
        for (g, b) in self.up_ln_gamma.iter_mut().zip(&unet.up_blocks) {
            g.copy_from_slice(&b.ln_gamma);
        }
        for (g, b) in self.up_ln_beta.iter_mut().zip(&unet.up_blocks) {
            g.copy_from_slice(&b.ln_beta);
        }
        for (g, src) in self.skip_ln_gamma.iter_mut().zip(&unet.skip_ln_gamma) {
            g.copy_from_slice(src);
        }
        for (g, src) in self.skip_ln_beta.iter_mut().zip(&unet.skip_ln_beta) {
            g.copy_from_slice(src);
        }
        Ok(())
    }

    /// Resident U-Net forward. `x_dev` and `out_dev` are
    /// `GpuVec::Hip`. Weights stay device-resident across the full
    /// chain; activations roundtrip host-side once per block for
    /// LN+SiLU (and skip-add+LN on the up path).
    ///
    /// TODO(GPU residency follow-up): replace the per-block
    /// `copy_to_host` / `ln_silu_fwd` / `copy_from` triple with a
    /// resident `LnSiluResident` / `LayerNormResident` op. Same
    /// for the up-path skip-add (`hipblasSaxpy` would do it on
    /// device). Until those land, every block pays one D2H + one
    /// H2D — measurable but not load-bearing for correctness.
    pub fn forward(
        &self,
        batch: &modgrad_device::backend::HipBatch,
        x_dev: &modgrad_compute::backend::GpuVec,
        out_dev: &mut modgrad_compute::backend::GpuVec,
    ) -> Result<(), modgrad_compute::backend::ResidencyError> {
        use modgrad_compute::backend::{GpuVec, ResidencyError};
        debug_assert_eq!(x_dev.len(), self.in_dim);
        debug_assert_eq!(out_dev.len(), self.out_dim);

        // Boundary check: forward only consumes Hip-resident input.
        // A Heap or Vram GpuVec slipping in is a caller bug; surface
        // it as `WrongVariant` rather than panicking deeper inside
        // `LinearResident::forward`.
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

        // Stage 1: first_projection (in_dim → widths[0]).
        //
        // We always need outs_down[0] held host-side (it's the
        // shallowest skip target) and on-device (it's input to
        // down_blocks[0] when n_blocks > 0, or it's the final
        // output when n_blocks == 0). Stage activation on host
        // for LN+SiLU + skip retention; re-upload to a fresh
        // device buffer for the next dispatch.
        let mut first_dev = GpuVec::try_hip(self.widths[0])?;
        self.first_projection.forward(batch, x_dev, &mut first_dev)?;
        // D2H roundtrip — see TODO at fn-doc.
        batch.flush()?;
        let mut first_host = vec![0.0f32; self.widths[0]];
        match &first_dev {
            GpuVec::Hip(buf) => buf.copy_to_host(&mut first_host)?,
            _ => unreachable!("we just allocated Hip"),
        }
        // LN + SiLU on host (same dispatch path as
        // `SynapseBlock::forward`).
        let mut first_act = vec![0.0f32; self.widths[0]];
        modgrad_device::backend::ops::ln_silu_fwd(
            &first_host, &self.first_ln_gamma, &self.first_ln_beta,
            &mut first_act, None, 1, self.widths[0],
        )?;

        // Down-path skip cache. outs_down[0] = first_act,
        // outs_down[i+1] = down_blocks[i] applied to outs_down[i].
        // Host-side because every level is read again on the up
        // path for the skip-add.
        let mut outs_down: Vec<Vec<f32>> = Vec::with_capacity(n_blocks + 1);
        outs_down.push(first_act);

        // Stage 2: down blocks. Re-upload the previous level's
        // activation as the input to the next matvec, then
        // download for LN+SiLU.
        let mut current_dev = first_dev;
        for i in 0..n_blocks {
            // Upload outs_down[i] into current_dev (resize
            // unnecessary — current_dev was widths[i] from the
            // previous round, which matches outs_down[i].len()).
            current_dev.copy_from(outs_down.last().unwrap());

            // matvec_resident widths[i] → widths[i+1].
            let mut next_dev = GpuVec::try_hip(self.widths[i + 1])?;
            self.down_blocks[i].forward(batch, &current_dev, &mut next_dev)?;
            batch.flush()?;
            let mut next_host = vec![0.0f32; self.widths[i + 1]];
            match &next_dev {
                GpuVec::Hip(buf) => buf.copy_to_host(&mut next_host)?,
                _ => unreachable!("Hip alloc"),
            }
            let mut next_act = vec![0.0f32; self.widths[i + 1]];
            modgrad_device::backend::ops::ln_silu_fwd(
                &next_host, &self.down_ln_gamma[i], &self.down_ln_beta[i],
                &mut next_act, None, 1, self.widths[i + 1],
            )?;
            outs_down.push(next_act);
            current_dev = next_dev;
        }

        // Stage 3: up path. Start from the bottleneck
        // (outs_down[n_blocks]); each iteration takes
        // widths[up_idx+1] → widths[up_idx], adds the matching
        // skip from outs_down[up_idx], affine-LNs the sum, then
        // continues. The skip-add and LN are host-side for the
        // same reason LN+SiLU is.
        let mut current_host = outs_down[n_blocks].clone();
        for i in 0..n_blocks {
            let up_idx = n_blocks - 1 - i;
            // Upload current host activation onto a fresh device
            // buffer of the right size.
            let mut up_in_dev = GpuVec::try_hip(self.widths[up_idx + 1])?;
            up_in_dev.copy_from(&current_host);

            // matvec_resident widths[up_idx+1] → widths[up_idx].
            let mut up_out_dev = GpuVec::try_hip(self.widths[up_idx])?;
            self.up_blocks[up_idx].forward(batch, &up_in_dev, &mut up_out_dev)?;
            batch.flush()?;
            let mut up_out_host = vec![0.0f32; self.widths[up_idx]];
            match &up_out_dev {
                GpuVec::Hip(buf) => buf.copy_to_host(&mut up_out_host)?,
                _ => unreachable!("Hip alloc"),
            }
            // LN + SiLU on the up output (matches
            // `SynapseBlock::forward`).
            let mut up_act = vec![0.0f32; self.widths[up_idx]];
            modgrad_device::backend::ops::ln_silu_fwd(
                &up_out_host, &self.up_ln_gamma[up_idx], &self.up_ln_beta[up_idx],
                &mut up_act, None, 1, self.widths[up_idx],
            )?;
            // Skip-add: up_act += outs_down[up_idx]. Host-side;
            // see TODO at fn-doc.
            let skip = &outs_down[up_idx];
            debug_assert_eq!(up_act.len(), skip.len());
            for j in 0..up_act.len() {
                up_act[j] += skip[j];
            }
            // Affine LN on the sum (matches host
            // `affine_layer_norm`).
            let mut after_ln = vec![0.0f32; self.widths[up_idx]];
            modgrad_device::backend::ops::layer_norm_fwd(
                &up_act, &self.skip_ln_gamma[up_idx], &self.skip_ln_beta[up_idx],
                &mut after_ln, None, 1, self.widths[up_idx],
            )?;
            current_host = after_ln;
        }

        // Stage 4: stage the final activation back onto the
        // caller-provided `out_dev` buffer.
        out_dev.copy_from(&current_host);
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
