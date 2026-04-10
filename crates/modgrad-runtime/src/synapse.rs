//! Inter-layer synapse connecting two brain regions.
//!
//! Extracted from ctm.rs for modularity.

use serde::{Deserialize, Serialize};
use modgrad_compute::neuron::{Linear, SimpleRng};
use modgrad_ctm::synapse::SynapseUNet;

/// Synapse connecting two brain regions.
///
/// Three modes:
///   - Depth-1: y = SiLU(W₁ × x)              (legacy linear)
///   - Depth-2: y = SiLU(W₂ × SiLU(W₁ × x))  (legacy 2-layer MLP)
///   - U-Net:   y = UNet(x)                     (Sakana-style deep synapse)
///
/// When `unet` is Some, `forward()` delegates to the U-Net and ignores
/// the linear layers. The linear fields are kept for serde backward compat.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Synapse {
    pub linear: Linear,
    /// Optional second layer for depth-2 synapses.
    pub linear2: Option<Linear>,
    /// Optional U-Net synapse (depth ≥ 3). When present, forward() uses this.
    #[serde(default)]
    pub unet: Option<SynapseUNet>,
}

/// SiLU activation: x * sigmoid(x)
#[inline]
fn silu(x: f32) -> f32 {
    let s = 1.0 / (1.0 + (-x).exp());
    x * s
}

/// SiLU derivative: σ(x)(1 + x(1 − σ(x)))
#[inline]
fn silu_deriv(x: f32) -> f32 {
    let s = 1.0 / (1.0 + (-x).exp());
    s + x * s * (1.0 - s)
}

impl Synapse {
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        Self { linear: Linear::new(in_dim, out_dim), linear2: None, unet: None }
    }

    /// Create a depth-2 synapse: in_dim → hidden_dim → out_dim.
    /// W₁ is Kaiming init (good random projection).
    /// W₂ is near-identity: if hidden == out, start as I + noise.
    /// Otherwise, Kaiming-scaled but small.
    pub fn new_depth2(in_dim: usize, hidden_dim: usize, out_dim: usize) -> Self {
        let linear = Linear::new(in_dim, hidden_dim);

        // W₂: identity + small perturbation when square, small Kaiming otherwise
        let scale = if hidden_dim == out_dim { 0.01 } else { (2.0 / hidden_dim as f32).sqrt() * 0.1 };
        let mut rng = SimpleRng::new(hidden_dim as u64 ^ out_dim as u64 ^ 0xd2d2d2);
        let weight2: Vec<f32> = (0..out_dim * hidden_dim)
            .map(|idx| {
                let i = idx / hidden_dim; // row (output neuron)
                let j = idx % hidden_dim; // col (input neuron)
                let identity = if i == j && hidden_dim == out_dim { 1.0 } else { 0.0 };
                identity + rng.next_normal() * scale
            })
            .collect();
        let linear2 = Linear {
            weight: weight2,
            bias: vec![0.0; out_dim],
            in_dim: hidden_dim,
            out_dim,
        };

        Self { linear, linear2: Some(linear2), unet: None }
    }

    /// Create a U-Net synapse.
    pub fn new_unet(in_dim: usize, out_dim: usize, depth: usize, min_width: usize) -> Self {
        Self {
            linear: Linear::new(in_dim, out_dim), // placeholder, unused when unet is Some
            linear2: None,
            unet: Some(SynapseUNet::new(in_dim, out_dim, depth, min_width)),
        }
    }

    pub fn depth(&self) -> usize {
        if self.unet.is_some() { self.unet.as_ref().unwrap().widths.len() }
        else if self.linear2.is_some() { 2 }
        else { 1 }
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        // U-Net path: deep synapse
        if let Some(ref unet) = self.unet {
            return unet.forward(x);
        }
        // Legacy path: linear + SiLU
        let mut h = self.linear.forward(x);
        // SiLU after layer 1
        for v in &mut h {
            *v = silu(*v);
        }

        // Depth-2: second layer + SiLU
        if let Some(ref l2) = self.linear2 {
            let mut out = l2.forward(&h);
            for v in &mut out {
                *v = silu(*v);
            }
            out
        } else {
            h
        }
    }

    /// Create a synapse respecting the config's synapse_depth.
    /// Depth 1: standard Linear + SiLU.
    /// Depth 2: Linear + SiLU + Linear + SiLU (W₂ = identity init).
    /// Depth ≥ 3: U-Net with skip connections (Sakana-style).
    pub fn new_with_depth(in_dim: usize, out_dim: usize, depth: usize) -> Self {
        Self::new_with_depth_and_min_width(in_dim, out_dim, depth, 16)
    }

    /// Create a synapse with explicit U-Net minimum bottleneck width.
    pub fn new_with_depth_and_min_width(
        in_dim: usize, out_dim: usize, depth: usize, min_width: usize,
    ) -> Self {
        if depth >= 3 {
            Synapse::new_unet(in_dim, out_dim, depth, min_width)
        } else if depth >= 2 {
            let hidden = out_dim;
            Synapse::new_depth2(in_dim, hidden, out_dim)
        } else {
            Synapse::new(in_dim, out_dim)
        }
    }

    /// Forward pass returning intermediate activations for alternating-LS.
    /// Returns (layer1_output, final_output) — layer1 is the hidden state
    /// between the two linear layers (post-SiLU).
    pub fn forward_with_intermediate(&self, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut h = self.linear.forward(x);
        for v in &mut h {
            *v = silu(*v);
        }

        if let Some(ref l2) = self.linear2 {
            let mut out = l2.forward(&h);
            for v in &mut out {
                *v = silu(*v);
            }
            (h, out)
        } else {
            let out = h.clone();
            (h, out)
        }
    }

    /// Forward pass that caches intermediates for backward.
    pub fn forward_cached(&self, x: &[f32]) -> (Vec<f32>, SynapseForwardCache) {
        let z1 = self.linear.forward(x);
        let mut h = z1.clone();
        for v in &mut h {
            *v = silu(*v);
        }

        if let Some(ref l2) = self.linear2 {
            let z2 = l2.forward(&h);
            let mut out = z2.clone();
            for v in &mut out {
                *v = silu(*v);
            }
            let cache = SynapseForwardCache {
                input: x.to_vec(), z1, h, z2: Some(z2), output: out.clone(),
            };
            (out, cache)
        } else {
            let cache = SynapseForwardCache {
                input: x.to_vec(), z1, h: h.clone(), z2: None, output: h.clone(),
            };
            (h, cache)
        }
    }

    /// Backward pass: compute gradients of loss w.r.t. weights.
    ///
    /// `d_output` is dL/d(output) — the gradient of loss w.r.t. this synapse's output.
    /// Returns (SynapseGradients, d_input) where d_input is dL/d(x) for upstream.
    pub fn backward(&self, d_output: &[f32], cache: &SynapseForwardCache) -> (SynapseGradients, Vec<f32>) {
        if let Some(ref l2) = self.linear2 {
            // ── Depth-2 backward ──
            let z2 = cache.z2.as_ref().unwrap();

            // d_z2 = d_output ⊙ SiLU'(z2)
            let d_z2: Vec<f32> = d_output.iter().zip(z2.iter()).map(|(&dy, &z)| {
                dy * silu_deriv(z)
            }).collect();

            // dW2 = d_z2 × h^T, db2 = d_z2
            let h = &cache.h;
            let mut dw2 = vec![0.0f32; l2.out_dim * l2.in_dim];
            for i in 0..l2.out_dim {
                for j in 0..l2.in_dim {
                    dw2[i * l2.in_dim + j] = d_z2[i] * h[j];
                }
            }
            let db2 = d_z2.clone();

            // d_h = W2^T × d_z2
            let mut d_h = vec![0.0f32; l2.in_dim];
            for j in 0..l2.in_dim {
                for i in 0..l2.out_dim {
                    d_h[j] += l2.weight[i * l2.in_dim + j] * d_z2[i];
                }
            }

            // d_z1 = d_h ⊙ SiLU'(z1)
            let z1 = &cache.z1;
            let d_z1: Vec<f32> = d_h.iter().zip(z1.iter()).map(|(&dh, &z)| {
                dh * silu_deriv(z)
            }).collect();

            // dW1 = d_z1 × x^T, db1 = d_z1
            let x = &cache.input;
            let l1 = &self.linear;
            let mut dw1 = vec![0.0f32; l1.out_dim * l1.in_dim];
            for i in 0..l1.out_dim {
                for j in 0..l1.in_dim {
                    dw1[i * l1.in_dim + j] = d_z1[i] * x[j];
                }
            }
            let db1 = d_z1.clone();

            // d_input = W1^T × d_z1
            let mut d_input = vec![0.0f32; l1.in_dim];
            for j in 0..l1.in_dim {
                for i in 0..l1.out_dim {
                    d_input[j] += l1.weight[i * l1.in_dim + j] * d_z1[i];
                }
            }

            (SynapseGradients { dw1, db1, dw2, db2 }, d_input)
        } else {
            // ── Depth-1 backward ──
            let z1 = &cache.z1;
            let d_z1: Vec<f32> = d_output.iter().zip(z1.iter()).map(|(&dy, &z)| {
                dy * silu_deriv(z)
            }).collect();

            let x = &cache.input;
            let l1 = &self.linear;
            let mut dw1 = vec![0.0f32; l1.out_dim * l1.in_dim];
            for i in 0..l1.out_dim {
                for j in 0..l1.in_dim {
                    dw1[i * l1.in_dim + j] = d_z1[i] * x[j];
                }
            }
            let db1 = d_z1.clone();

            let mut d_input = vec![0.0f32; l1.in_dim];
            for j in 0..l1.in_dim {
                for i in 0..l1.out_dim {
                    d_input[j] += l1.weight[i * l1.in_dim + j] * d_z1[i];
                }
            }

            (SynapseGradients { dw1, db1, dw2: vec![], db2: vec![] }, d_input)
        }
    }

    /// Apply gradients with learning rate (SGD step).
    pub fn apply_gradients(&mut self, grads: &SynapseGradients, lr: f32) {
        for (w, &g) in self.linear.weight.iter_mut().zip(grads.dw1.iter()) {
            *w -= lr * g;
        }
        for (b, &g) in self.linear.bias.iter_mut().zip(grads.db1.iter()) {
            *b -= lr * g;
        }
        if let Some(ref mut l2) = self.linear2 {
            for (w, &g) in l2.weight.iter_mut().zip(grads.dw2.iter()) {
                *w -= lr * g;
            }
            for (b, &g) in l2.bias.iter_mut().zip(grads.db2.iter()) {
                *b -= lr * g;
            }
        }
    }
}

/// Cached forward pass for backprop: stores all intermediates needed for backward.
#[derive(Clone)]
pub struct SynapseForwardCache {
    pub input: Vec<f32>,       // x
    pub z1: Vec<f32>,          // W1 @ x (pre-SiLU)
    pub h: Vec<f32>,           // SiLU(z1) (post layer 1)
    pub z2: Option<Vec<f32>>,  // W2 @ h (pre-SiLU, depth-2 only)
    pub output: Vec<f32>,      // final output
}

/// Gradients for one synapse (accumulated across a mini-batch).
pub struct SynapseGradients {
    pub dw1: Vec<f32>,   // [out1 × in1] same shape as linear.weight
    pub db1: Vec<f32>,   // [out1]
    pub dw2: Vec<f32>,   // [out2 × hidden] (empty if depth-1)
    pub db2: Vec<f32>,   // [out2] (empty if depth-1)
}
