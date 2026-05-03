//! Visual projector for VLM-style integration with a frozen LLM.
//!
//! 2026 small-VLM precedent (Idefics3 / SmolVLM / LLaVA-OneVision /
//! PaliGemma 2): convolutional vision features → **pixel-shuffle**
//! (spatial → channel reshape, reduces token count s² ×) → **2-layer
//! MLP** → LLM d_model. Visual tokens are then prepended to the text
//! context the LLM consumes.
//!
//! # Time axis from day one
//!
//! Input shape is `[T, C, H, W]` with `T = 1` for static images and
//! `T = N` for video. The projector iterates over `T` and emits
//! `T × (H/s) × (W/s)` tokens of `out_dim`. Adding video later is
//! data-loader work, not a re-architecture.
//!
//! # What this module provides
//!
//! Forward only. Backward / training comes in a follow-up commit when
//! we wire the projector into a training loop end-to-end. For initial
//! integration with a frozen Qwen + brain, projector weights are
//! random-init and the brain's logit-modulator does the learning —
//! this matches the bias-only-steering finding (arxiv 2505.18706).

use modgrad_compute::neuron::{Linear, SimpleRng};
use modgrad_device::backend::ops;

/// Idefics3-style 2-layer MLP visual projector with pixel-shuffle.
///
/// Construct via [`VisualProjector::new`]. Forward via
/// [`VisualProjector::forward`]. Input `[T, C, H, W]` flat row-major
/// (CHW per timestep, T timesteps concatenated). Output:
/// `[T × (H/s) × (W/s) × out_dim]` flat row-major (one token per
/// (t, h', w') triplet).
pub struct VisualProjector {
    /// Pixel-shuffle factor along each spatial axis. Must divide H
    /// and W. After shuffle: `C × s² channels, H/s × W/s spatial`.
    pub shuffle: usize,
    /// Input channels at the visual feature stage.
    pub in_channels: usize,
    /// Input spatial height per timestep.
    pub spatial_h: usize,
    /// Input spatial width per timestep.
    pub spatial_w: usize,
    /// MLP layer 1: `(C·s²) → mlp_hidden`.
    pub fc1: Linear,
    /// MLP layer 2: `mlp_hidden → out_dim`.
    pub fc2: Linear,
    pub mlp_hidden: usize,
    pub out_dim: usize,
}

impl VisualProjector {
    /// Build a fresh projector with random Linear weights (He-init via
    /// `Linear::new_he`-style fan-in scaling). Output token count per
    /// timestep is `(H/s) × (W/s)`.
    ///
    /// Pre: `H % s == 0` and `W % s == 0`.
    pub fn new(
        in_channels: usize,
        spatial_h: usize,
        spatial_w: usize,
        shuffle: usize,
        mlp_hidden: usize,
        out_dim: usize,
        seed: u64,
    ) -> Self {
        assert!(shuffle >= 1, "shuffle factor must be ≥1");
        assert_eq!(spatial_h % shuffle, 0,
            "spatial_h={spatial_h} not divisible by shuffle={shuffle}");
        assert_eq!(spatial_w % shuffle, 0,
            "spatial_w={spatial_w} not divisible by shuffle={shuffle}");
        let post_shuffle_dim = in_channels * shuffle * shuffle;
        let scale_1 = (2.0 / post_shuffle_dim as f32).sqrt();
        let scale_2 = (2.0 / mlp_hidden as f32).sqrt();
        let mut rng = SimpleRng::new(seed);
        let mut rng2 = SimpleRng::new(seed.wrapping_add(0x9E3779B97F4A7C15));

        let fc1_w: Vec<f32> = (0..mlp_hidden * post_shuffle_dim)
            .map(|_| rng.next_normal() * scale_1)
            .collect();
        let fc1 = Linear {
            weight: fc1_w,
            bias: vec![0.0f32; mlp_hidden],
            in_dim: post_shuffle_dim,
            out_dim: mlp_hidden,
        };

        let fc2_w: Vec<f32> = (0..out_dim * mlp_hidden)
            .map(|_| rng2.next_normal() * scale_2)
            .collect();
        let fc2 = Linear {
            weight: fc2_w,
            bias: vec![0.0f32; out_dim],
            in_dim: mlp_hidden,
            out_dim,
        };

        Self {
            shuffle, in_channels, spatial_h, spatial_w,
            fc1, fc2, mlp_hidden, out_dim,
        }
    }

    /// Number of output tokens per timestep.
    #[inline]
    pub fn tokens_per_timestep(&self) -> usize {
        (self.spatial_h / self.shuffle) * (self.spatial_w / self.shuffle)
    }

    /// Forward pass over `T` timesteps of `[C, H, W]` features.
    ///
    /// `x`: flat `[T × C × H × W]` row-major, CHW per timestep.
    /// Returns: flat `[T × tokens_per_timestep × out_dim]`.
    pub fn forward(&self, x: &[f32], t: usize) -> Vec<f32> {
        let c = self.in_channels;
        let h = self.spatial_h;
        let w = self.spatial_w;
        let s = self.shuffle;
        let h_out = h / s;
        let w_out = w / s;
        let tokens_per_t = h_out * w_out;
        let post_shuffle_c = c * s * s;
        debug_assert_eq!(x.len(), t * c * h * w,
            "x len {} != T·C·H·W = {}·{}·{}·{}", x.len(), t, c, h, w);

        // Pixel-shuffle: for each timestep, reshape [C, H, W] →
        // [C·s², H/s, W/s] by packing s×s spatial blocks into channels.
        // We assemble the post-shuffle tensor as a row-major sequence of
        // tokens: [tokens_per_t × post_shuffle_c] per timestep.
        // Pre-allocate the full post-shuffle output and the projection
        // outputs; per-token MLP is matvec on `post_shuffle_c → mlp_hidden
        // → out_dim`.
        let mut shuffled = vec![0.0f32; t * tokens_per_t * post_shuffle_c];
        for ti in 0..t {
            let xb = &x[ti * c * h * w..(ti + 1) * c * h * w];
            let yb = &mut shuffled[ti * tokens_per_t * post_shuffle_c
                                ..(ti + 1) * tokens_per_t * post_shuffle_c];
            // For each output token (yh, xh), pack s×s spatial block × C.
            for yh in 0..h_out {
                for xh in 0..w_out {
                    let token_idx = yh * w_out + xh;
                    let token_base = token_idx * post_shuffle_c;
                    for ci in 0..c {
                        for dy in 0..s {
                            for dx in 0..s {
                                let src_y = yh * s + dy;
                                let src_x = xh * s + dx;
                                let src = ci * h * w + src_y * w + src_x;
                                let dst = token_base + (ci * s * s + dy * s + dx);
                                yb[dst] = xb[src];
                            }
                        }
                    }
                }
            }
        }

        // 2-layer MLP per token: shuffled[token, :] → fc1 → GELU → fc2.
        // Batched matmul: shuffled is [N_tok × post_shuffle_c]; fc1.weight
        // is [mlp_hidden × post_shuffle_c]. Output of fc1 = shuffled · fc1ᵀ
        // = [N_tok × mlp_hidden]. Use matmul_nt (registry-dispatched).
        let n_tok = t * tokens_per_t;
        let mut h1 = vec![0.0f32; n_tok * self.mlp_hidden];
        ops::matmul_nt(
            &shuffled, &self.fc1.weight, &mut h1, None,
            n_tok, post_shuffle_c, self.mlp_hidden,
        ).expect("VisualProjector::forward fc1 matmul_nt");
        // Add fc1 bias + GELU activation in one host pass (small).
        for i in 0..n_tok {
            let row = &mut h1[i * self.mlp_hidden..(i + 1) * self.mlp_hidden];
            for k in 0..self.mlp_hidden {
                let v = row[k] + self.fc1.bias[k];
                // GELU approx (tanh form): 0.5 * v * (1 + tanh(√(2/π) · (v + 0.044715·v³)))
                let v3 = v * v * v;
                let inner = (2.0_f32 / std::f32::consts::PI).sqrt() * (v + 0.044715 * v3);
                row[k] = 0.5 * v * (1.0 + inner.tanh());
            }
        }

        let mut out = vec![0.0f32; n_tok * self.out_dim];
        ops::matmul_nt(
            &h1, &self.fc2.weight, &mut out, None,
            n_tok, self.mlp_hidden, self.out_dim,
        ).expect("VisualProjector::forward fc2 matmul_nt");
        for i in 0..n_tok {
            let row = &mut out[i * self.out_dim..(i + 1) * self.out_dim];
            for k in 0..self.out_dim {
                row[k] += self.fc2.bias[k];
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Static-image case (T=1): from VisualCortex::cifar() V4 features
    /// at [128, 4, 4] with shuffle=2 → 4 tokens × 512 post-shuffle →
    /// MLP into 896 (Qwen-0.5B d_model). Verifies output shape and
    /// non-trivial response.
    #[test]
    fn cifar_v4_to_qwen_dim_static_image() {
        let proj = VisualProjector::new(128, 4, 4, 2, 1024, 896, 0xC0FFEE);
        assert_eq!(proj.tokens_per_timestep(), 4);
        let x: Vec<f32> = (0..128 * 4 * 4).map(|i| ((i as f32 * 0.13).sin())).collect();
        let out = proj.forward(&x, 1);
        assert_eq!(out.len(), 1 * 4 * 896, "expected 4 tokens × 896 dim");
        // Non-trivial: output should not be all-zero with random weights.
        let nonzero = out.iter().filter(|&&v| v.abs() > 1e-6).count();
        assert!(nonzero > out.len() / 2,
            "expected most outputs non-zero, got {nonzero}/{}", out.len());
    }

    /// Video case (T=4): same shape per-frame but 4 timesteps. Should
    /// produce 4 × tokens_per_timestep tokens.
    #[test]
    fn video_path_t4_emits_per_frame_tokens() {
        let proj = VisualProjector::new(128, 4, 4, 2, 256, 896, 0xBEEF);
        let t = 4;
        let x: Vec<f32> = (0..t * 128 * 4 * 4).map(|i| ((i as f32 * 0.07).cos())).collect();
        let out = proj.forward(&x, t);
        assert_eq!(out.len(), t * proj.tokens_per_timestep() * 896);
    }

    /// Determinism: same seed, same forward → bit-identical output.
    #[test]
    fn forward_is_deterministic_for_same_seed_and_input() {
        let p1 = VisualProjector::new(64, 8, 8, 2, 256, 128, 0x42);
        let p2 = VisualProjector::new(64, 8, 8, 2, 256, 128, 0x42);
        let x: Vec<f32> = (0..64 * 8 * 8).map(|i| (i as f32) * 0.01).collect();
        let o1 = p1.forward(&x, 1);
        let o2 = p2.forward(&x, 1);
        assert_eq!(o1, o2);
    }
}
