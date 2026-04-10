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

// ─── Affine LayerNorm ──────────────────────────────────────

/// LayerNorm with learnable affine parameters (gamma, beta).
/// Matches PyTorch nn.LayerNorm default: elementwise_affine=True.
#[inline]
fn affine_layer_norm(x: &mut [f32], gamma: &[f32], beta: &[f32]) {
    let n = x.len();
    if n == 0 { return; }
    let nf = n as f32;
    let mean: f32 = x.iter().sum::<f32>() / nf;
    let var: f32 = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / nf;
    let inv_std = 1.0 / (var + 1e-5).sqrt();
    for i in 0..n {
        x[i] = gamma[i] * (x[i] - mean) * inv_std + beta[i];
    }
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn silu_vec(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = silu(*v);
    }
}

// ─── SynapseBlock ──────────────────────────────────────────

/// One block: Linear → LayerNorm(affine) → SiLU.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut y = self.linear.forward(x);
        affine_layer_norm(&mut y, &self.ln_gamma, &self.ln_beta);
        silu_vec(&mut y);
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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
}
