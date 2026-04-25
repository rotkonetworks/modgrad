//! Retina: modality-specific input adapters.
//!
//! The brain has a universal cortical architecture with different input adapters
//! (retina for vision, cochlea for audio, skin receptors for touch). Each adapter
//! converts raw physical signals into a common format: neural spikes.
//!
//! Our retinas convert raw input (pixels, audio, bytes) into observation vectors
//! that the CTM can process. The CTM itself is modality-agnostic — it thinks
//! about whatever signals it receives.
//!
//! All retinas output the same format: Vec<f32> of dimension d_output.
//! The CTM doesn't know or care which sense produced the observation.
//!
//! Architecture follows the brain's sensory hierarchy:
//! - Each retina has multiple processing stages (V1→V2→V4 for vision)
//! - Each stage increases abstraction and decreases spatial resolution
//! - All stages run in parallel (read from previous tick's output)
//! - Learning is Hebbian (local, no backprop)

use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};
use wincode_derive::{SchemaRead, SchemaWrite};

/// Common interface for all sensory adapters.
pub trait Retina {
    /// Convert raw input into observation vector for the CTM.
    fn observe(&mut self, raw: &[f32]) -> Vec<f32>;

    /// Output dimension (must match CTM's d_input).
    fn d_output(&self) -> usize;

    /// Total parameters in this retina.
    fn param_count(&self) -> usize;
}

// ─── Simple RNG (reuse from ctm) ────────────────────────────

fn simple_rng_normal(state: &mut u64) -> f32 {
    // Box-Muller
    let u1 = {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*state >> 40) as f32 / (1u64 << 24) as f32
    }.max(1e-10);
    let u2 = {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*state >> 40) as f32 / (1u64 << 24) as f32
    };
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

// ─── im2col / col2im helpers ─────────────────────────────────
//
// Conv-as-matmul reformulation: unfold each k×k patch into one column of a
// `[patch_size × (n · n_patches)]` matrix where `n` is the batch size. Each
// batch's patches are concatenated along the column dim, so convolution over
// N images becomes a single matmul:
//   `output = weight @ im2col`       shape [out_ch × (n · n_patches)]
// The batch dim is always present; single-image callers pass `n = 1`.
//
// Patch-row order is `(ic, kh, kw)` — matches `Conv2d::weight` layout
// `[out_ch × in_ch × kh × kw]`, so weights map 1:1 to im2col rows without any
// reshape. Batches lay out contiguously by image: columns `[0..n_patches)`
// are image 0's patches, `[n_patches..2·n_patches)` are image 1's, etc.

/// Unfold `[n × in_ch × h × w]` into patch-column form
/// `[patch_size × (n · n_patches)]` row-major.
///
/// `input` must hold `n * in_ch * h * w` elements, laid out as N contiguous
/// `[in_ch × h × w]` images. Padding positions read zero.
fn im2col(
    input: &[f32], n: usize, in_ch: usize, h: usize, w: usize,
    k: usize, s: usize, p: usize, out_h: usize, out_w: usize,
) -> Vec<f32> {
    let patch_size = in_ch * k * k;
    let n_patches = out_h * out_w;
    let cols_per_img = n_patches;
    let total_cols = n * cols_per_img;
    debug_assert_eq!(input.len(), n * in_ch * h * w,
        "im2col input len mismatch: {} != {}·{}·{}·{}", input.len(), n, in_ch, h, w);
    let mut col = vec![0.0f32; patch_size * total_cols];
    let img_stride = in_ch * h * w;
    for b in 0..n {
        let img = &input[b * img_stride..(b + 1) * img_stride];
        let col_off = b * cols_per_img;
        for oh in 0..out_h {
            for ow in 0..out_w {
                let pidx = col_off + oh * out_w + ow;
                for ic in 0..in_ch {
                    for kh in 0..k {
                        for kw in 0..k {
                            let ih = (oh * s + kh) as isize - p as isize;
                            let iw = (ow * s + kw) as isize - p as isize;
                            if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                let in_idx = ic * h * w + ih as usize * w + iw as usize;
                                let row_idx = ic * k * k + kh * k + kw;
                                col[row_idx * total_cols + pidx] = img[in_idx];
                            }
                        }
                    }
                }
            }
        }
    }
    col
}

/// Adjoint of `im2col`: scatter a `[patch_size × (n · n_patches)]` matrix
/// into `[n × in_ch × in_h × in_w]` (accumulating where patches within an
/// image overlap). Images do not share gradient — image 0's columns scatter
/// into output batch index 0 only, etc.
fn col2im_acc(
    col: &[f32], n: usize, in_ch: usize, in_h: usize, in_w: usize,
    k: usize, s: usize, p: usize, out_h: usize, out_w: usize,
    output: &mut [f32],
) {
    let cols_per_img = out_h * out_w;
    let total_cols = n * cols_per_img;
    let img_stride = in_ch * in_h * in_w;
    debug_assert_eq!(output.len(), n * img_stride);
    for b in 0..n {
        let col_off = b * cols_per_img;
        let out_img = &mut output[b * img_stride..(b + 1) * img_stride];
        for oh in 0..out_h {
            for ow in 0..out_w {
                let pidx = col_off + oh * out_w + ow;
                for ic in 0..in_ch {
                    for kh in 0..k {
                        for kw in 0..k {
                            let ih = (oh * s + kh) as isize - p as isize;
                            let iw = (ow * s + kw) as isize - p as isize;
                            if ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize {
                                let row_idx = ic * k * k + kh * k + kw;
                                let out_idx = ic * in_h * in_w + ih as usize * in_w + iw as usize;
                                out_img[out_idx] += col[row_idx * total_cols + pidx];
                            }
                        }
                    }
                }
            }
        }
    }
}

// ─── Convolutional Layer ─────────────────────────────────────

/// 2D convolutional layer with Hebbian-compatible design.
/// Uses small kernels (3×3) like V1 simple cells.
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub struct Conv2d {
    /// Weights: [out_channels × in_channels × kh × kw]
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
}

impl Conv2d {
    pub fn new(in_ch: usize, out_ch: usize, kernel: usize, stride: usize, padding: usize) -> Self {
        let n = in_ch * kernel * kernel;
        let scale = (2.0 / n as f32).sqrt();
        let mut rng_state = (in_ch * out_ch * kernel) as u64 + 31337;
        let weight: Vec<f32> = (0..out_ch * in_ch * kernel * kernel)
            .map(|_| simple_rng_normal(&mut rng_state) * scale)
            .collect();
        let bias = vec![0.0; out_ch];
        Self { weight, bias, in_channels: in_ch, out_channels: out_ch,
               kernel_size: kernel, stride, padding }
    }

    /// Forward pass over a batch of `n` images of shape
    /// `[in_ch × h × w]`, packed contiguously in `input`.
    ///
    /// Output is `[n × out_ch × out_h × out_w]` row-major, laid out as
    /// `n` contiguous per-image chunks. For `n = 1` this matches the
    /// pre-batch signature bit-for-bit.
    ///
    /// Dispatched as one im2col + matmul via `modgrad_device::backend::ops`
    /// — amortizes hipBLAS dispatch overhead over the whole batch, which
    /// is what actually drives GPU utilization up on PCIe-bottlenecked
    /// hardware (7600M XT gfx1102, PCIe x4).
    ///
    /// Returns `(output, out_h, out_w)`.
    pub fn forward(
        &self, input: &[f32], n: usize, h: usize, w: usize,
    ) -> (Vec<f32>, usize, usize) {
        let k = self.kernel_size;
        let s = self.stride;
        let p = self.padding;
        let out_h = (h + 2 * p - k) / s + 1;
        let out_w = (w + 2 * p - k) / s + 1;
        let patch_size = self.in_channels * k * k;
        let n_patches = out_h * out_w;
        let total_cols = n * n_patches;

        let col = im2col(input, n, self.in_channels, h, w, k, s, p, out_h, out_w);
        let mut y = vec![0.0f32; self.out_channels * total_cols];

        // y[oc × (n·n_patches)] = W · col  (bias applied below; matmul's
        // built-in bias-add broadcasts per-column, not per-row which is
        // what we'd need here — so do it manually in one sweep).
        modgrad_device::backend::ops::matmul_nn(
            &self.weight, &col, &mut y, None,
            self.out_channels, patch_size, total_cols,
        ).expect("Conv2d::forward: matmul_nn dispatch");

        for oc in 0..self.out_channels {
            let b = self.bias[oc];
            let row = &mut y[oc * total_cols..(oc + 1) * total_cols];
            for v in row.iter_mut() { *v += b; }
        }

        // Re-layout from [out_ch × (n · n_patches)] to
        // [n × out_ch × n_patches] — per-image contiguous chunks, which
        // is what downstream leaky_relu / subsequent layers expect.
        if n == 1 {
            return (y, out_h, out_w);
        }
        let mut out = vec![0.0f32; n * self.out_channels * n_patches];
        for b in 0..n {
            for oc in 0..self.out_channels {
                let src_base = oc * total_cols + b * n_patches;
                let dst_base = b * self.out_channels * n_patches + oc * n_patches;
                out[dst_base..dst_base + n_patches]
                    .copy_from_slice(&y[src_base..src_base + n_patches]);
            }
        }
        (out, out_h, out_w)
    }

    pub fn param_count(&self) -> usize {
        self.weight.len() + self.bias.len()
    }

    /// Forward-with-patches for gradient-path callers (`backward` / GHL).
    /// Returns `y` in the **flat matmul layout** `[out_ch × (n · n_patches)]`
    /// (not per-image re-laid) plus the im2col `patches` buffer so the
    /// caller doesn't recompute them on backward.
    pub fn forward_with_patches(
        &self, input: &[f32], n: usize, h: usize, w: usize,
    ) -> (Vec<f32>, usize, usize, Vec<f32>) {
        let k = self.kernel_size;
        let s = self.stride;
        let p = self.padding;
        let out_h = (h + 2 * p - k) / s + 1;
        let out_w = (w + 2 * p - k) / s + 1;
        let patch_size = self.in_channels * k * k;
        let total_cols = n * out_h * out_w;

        let col = im2col(input, n, self.in_channels, h, w, k, s, p, out_h, out_w);
        let mut y = vec![0.0f32; self.out_channels * total_cols];
        modgrad_device::backend::ops::matmul_nn(
            &self.weight, &col, &mut y, None,
            self.out_channels, patch_size, total_cols,
        ).expect("Conv2d::forward_with_patches: matmul_nn dispatch");
        for oc in 0..self.out_channels {
            let b = self.bias[oc];
            let row = &mut y[oc * total_cols..(oc + 1) * total_cols];
            for v in row.iter_mut() { *v += b; }
        }
        (y, out_h, out_w, col)
    }

    /// Backward pass over a batch of `n` images.
    ///
    /// `d_output` is `[out_ch × (n · n_patches)]` in the flat matmul
    /// layout (matches what `forward_with_patches` returns for `y`).
    /// `patches` is the im2col from forward, same `n` batch.
    ///
    /// Returns:
    /// - `d_weight` `[out_ch × patch_size]` = `d_output · patchesᵀ`
    ///   (summed across all batch images — standard SGD gradient)
    /// - `d_bias`   `[out_ch]` = per-channel sum over all positions
    /// - `d_input`  `[n × in_ch × in_h × in_w]` via col2im per image
    pub fn backward(
        &self,
        d_output: &[f32],
        patches: &[f32],
        n: usize,
        in_h: usize, in_w: usize,
        out_h: usize, out_w: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let k = self.kernel_size;
        let s = self.stride;
        let p = self.padding;
        let patch_size = self.in_channels * k * k;
        let total_cols = n * out_h * out_w;
        let oc = self.out_channels;
        debug_assert_eq!(d_output.len(), oc * total_cols);
        debug_assert_eq!(patches.len(), patch_size * total_cols);

        let mut d_weight = vec![0.0f32; oc * patch_size];
        modgrad_device::backend::ops::matmul_nt(
            d_output, patches, &mut d_weight, None,
            oc, total_cols, patch_size,
        ).expect("Conv2d::backward: d_W matmul_nt dispatch");

        let mut d_bias = vec![0.0f32; oc];
        for o in 0..oc {
            let row = &d_output[o * total_cols..(o + 1) * total_cols];
            d_bias[o] = row.iter().sum();
        }

        let mut d_col = vec![0.0f32; patch_size * total_cols];
        modgrad_device::backend::ops::matmul_tn(
            &self.weight, d_output, &mut d_col, None,
            patch_size, oc, total_cols,
        ).expect("Conv2d::backward: d_col matmul_tn dispatch");

        let mut d_input = vec![0.0f32; n * self.in_channels * in_h * in_w];
        col2im_acc(&d_col, n, self.in_channels, in_h, in_w, k, s, p, out_h, out_w, &mut d_input);

        (d_weight, d_bias, d_input)
    }

    /// GHL (Global-guided Hebbian Learning) update.
    ///
    /// Implements the three-factor rule from Hebbian Learning with
    /// Global Direction (arXiv:2601.21367):
    ///
    /// ```text
    /// Δw_ik = lr · sign(∂L/∂w_ik) · | u_k · (x_i − y_k · w_ik) |
    /// ```
    ///
    /// where `u_k = softmax(y_k / τ)` across channels (per spatial
    /// position), `y = W · patches` is the pre-sparsity forward, and
    /// the sign comes from a full backward pass (supplied by the
    /// caller via `sign_d_w` of shape `[out_ch × patch_size]`).
    ///
    /// Locally this is Oja's rule + soft winner-take-all (stable,
    /// competitive); globally it's sign-modulated by the task
    /// gradient. The paper reports ~3% from BP on ResNet-50 ImageNet,
    /// where prior pure-local Hebbian methods collapse by 30%+.
    ///
    /// Caller supplies `patches` (im2col buffer) and `y_pre` (forward
    /// output, pre-bias) from a preceding `forward_with_patches` to
    /// avoid recomputation.
    pub fn ghl_update(
        &mut self,
        patches: &[f32],     // [patch_size × total_cols]
        y_pre: &[f32],       // [oc × total_cols], the W·patches output
        sign_d_w: &[f32],    // [oc × patch_size], values in {-1, 0, +1}
        total_cols: usize,   // n_batch · n_patches_per_image
        lr: f32,
        tau: f32,
    ) {
        let k = self.kernel_size;
        let patch_size = self.in_channels * k * k;
        let oc = self.out_channels;
        debug_assert_eq!(patches.len(), patch_size * total_cols);
        debug_assert_eq!(y_pre.len(), oc * total_cols);
        debug_assert_eq!(sign_d_w.len(), oc * patch_size);

        // 1. u = softmax(y / τ) column-wise (per spatial position,
        //    across output channels — the SWTA competition).
        let mut u = vec![0.0f32; oc * total_cols];
        for pidx in 0..total_cols {
            let mut max = f32::NEG_INFINITY;
            for o in 0..oc {
                let v = y_pre[o * total_cols + pidx] / tau;
                if v.is_finite() && v > max { max = v; }
            }
            let mut sum = 0.0f32;
            for o in 0..oc {
                let v = ((y_pre[o * total_cols + pidx] / tau) - max).exp();
                u[o * total_cols + pidx] = v;
                sum += v;
            }
            let inv = if sum > 1e-30 { 1.0 / sum } else { 0.0 };
            for o in 0..oc {
                u[o * total_cols + pidx] *= inv;
            }
        }

        // 2. Local Hebbian over all columns:
        //    Δ_local[k, i] = Σ_p u_kp · (x_ip − y_kp · w_ki)
        //                  = (u · patchesᵀ)[k, i] − α_k · W[k, i]
        //    where α_k = Σ_p u_kp · y_kp.
        let mut local = vec![0.0f32; oc * patch_size];
        modgrad_device::backend::ops::matmul_nt(
            &u, patches, &mut local, None,
            oc, total_cols, patch_size,
        ).expect("Conv2d::ghl_update: u·patchesᵀ matmul_nt dispatch");

        let mut alpha = vec![0.0f32; oc];
        for o in 0..oc {
            let row_u = &u[o * total_cols..(o + 1) * total_cols];
            let row_y = &y_pre[o * total_cols..(o + 1) * total_cols];
            alpha[o] = row_u.iter().zip(row_y).map(|(a, b)| a * b).sum();
        }

        for o in 0..oc {
            let a = alpha[o];
            let w_off = o * patch_size;
            for i in 0..patch_size {
                local[w_off + i] -= a * self.weight[w_off + i];
            }
        }

        // 3. Apply: W += lr · sign_d_w ⊙ |local| / total_cols.
        let scale = lr / total_cols.max(1) as f32;
        for i in 0..oc * patch_size {
            let mag = local[i].abs();
            let sgn = sign_d_w[i];
            let d = scale * sgn * mag;
            if d.is_finite() {
                self.weight[i] += d;
            }
        }
    }

    /// Transpose (adjoint) convolution.
    ///
    /// Inverts the spatial geometry of `forward`: given activations on
    /// the output grid `[out_ch × out_h × out_w]`, scatter them through
    /// the kernel weights back onto the input grid `[in_ch × in_h × in_w]`.
    ///
    /// This is the operation Hoel's "overfitted brain" hypothesis needs:
    /// a top-down pass where noise seeded at a high cortical layer is
    /// projected back down to pixel space, producing synthesized
    /// "dream" input that is sparse, hallucinatory, and coherent with
    /// the model's learned priors — the stochastic-dropout-like signal
    /// dreams are theorised to inject into the perceptual hierarchy.
    ///
    /// Callers must supply `in_h`, `in_w` — these are the spatial dims
    /// of the original forward-input, which `forward` collapsed to
    /// `out_h`, `out_w` according to `stride`/`padding`. Adjoint
    /// restores them exactly.
    pub fn transpose_forward(
        &self, input: &[f32], n: usize,
        out_h: usize, out_w: usize,
        in_h: usize, in_w: usize,
    ) -> Vec<f32> {
        let k = self.kernel_size;
        let s = self.stride;
        let p = self.padding;
        let patch_size = self.in_channels * k * k;
        let total_cols = n * out_h * out_w;
        assert_eq!(input.len(), self.out_channels * total_cols,
            "transpose_forward: input shape mismatch ({} vs {}·{})",
            input.len(), self.out_channels, total_cols);

        // col = W^T @ input_grid: [patch_size, total_cols]
        let mut col = vec![0.0f32; patch_size * total_cols];
        modgrad_device::backend::ops::matmul_tn(
            &self.weight, input, &mut col, None,
            patch_size, self.out_channels, total_cols,
        ).expect("Conv2d::transpose_forward: matmul_tn dispatch");

        // Scatter per-image. Overlapping patches within an image accumulate.
        let mut output = vec![0.0f32; n * self.in_channels * in_h * in_w];
        col2im_acc(&col, n, self.in_channels, in_h, in_w, k, s, p, out_h, out_w, &mut output);
        output
    }

    /// Sparse coding / dictionary learning update (Olshausen & Field 1996
    /// formulation; "Hebbian" for historical continuity).
    ///
    /// Mathematically this is one projected-gradient step of
    ///   min ½‖patches − Wᵀ · topK(W · patches)‖²
    /// with a unit-norm constraint on each filter row. Equivalent to
    /// backprop on a reconstruction loss with a straight-through estimator
    /// through the top-K non-linearity.
    ///
    /// Pipeline (all matmuls dispatched through `modgrad_device::backend::ops`,
    /// so this hits GPU when the backend feature is enabled):
    ///   1. im2col  → patches `[patch_size × n_patches]`
    ///   2. encode  → Z = W · patches + bias  (matmul_nn)
    ///   3. top-K sparsify per position (CPU, small)
    ///   4. recon   → R = Wᵀ · Z_sparse  (matmul_tn)
    ///   5. error   → E = patches − R
    ///   6. grad    → dW = Z_sparse · Eᵀ  (matmul_nt)
    ///   7. W += (lr / active_patches) · dW; row-normalize updated filters.
    ///
    /// # Semantic change vs prior implementation
    ///
    /// This is a **dense batch update** over all spatial positions in the
    /// input, not a sequential per-position update over a 64-position
    /// subsample. The previous version's "each position sees weights updated
    /// by earlier positions" behavior is gone; updates are computed against
    /// a single snapshot of W. In practice this is the standard dictionary-
    /// learning formulation and converges to the same solutions, just via
    /// a batch step instead of an online one. The `lr / active_patches`
    /// scaling keeps per-position update magnitude comparable.
    pub fn hebbian_update(
        &mut self,
        input: &[f32], n: usize, h: usize, w: usize,
        lr: f32, sparsity_k: usize,
    ) {
        let k = self.kernel_size;
        let s = self.stride;
        let p = self.padding;
        let out_h = (h + 2 * p - k) / s + 1;
        let out_w = (w + 2 * p - k) / s + 1;
        let patch_size = self.in_channels * k * k;
        let n_patches = out_h * out_w;
        let total_cols = n * n_patches;
        let oc_n = self.out_channels;

        if total_cols == 0 { return; }

        // 1. im2col: [patch_size, total_cols]
        let patches = im2col(input, n, self.in_channels, h, w, k, s, p, out_h, out_w);

        // 2. Z = W · patches + bias_broadcast: [oc_n, total_cols]
        //    (bias added manually — matmul's built-in bias is per-column,
        //    and our layout has bias per-row; same story as `forward`.)
        let mut z = vec![0.0f32; oc_n * total_cols];
        modgrad_device::backend::ops::matmul_nn(
            &self.weight, &patches, &mut z, None,
            oc_n, patch_size, total_cols,
        ).expect("Conv2d::hebbian_update: activations matmul_nn dispatch");
        for oc in 0..oc_n {
            let b = self.bias[oc];
            let row = &mut z[oc * total_cols..(oc + 1) * total_cols];
            for v in row.iter_mut() { *v += b; }
        }

        // 3. Per-position top-K sparsify. Skip near-zero patches so the
        //    update doesn't chase pure-padding columns. Also clamp finite/
        //    extreme values out of Z — matches prior impl's NaN-safety.
        let mut z_sparse = vec![0.0f32; oc_n * total_cols];
        let mut updated = vec![false; oc_n];
        let mut active_patches: usize = 0;
        let take = sparsity_k.min(oc_n);

        let mut scratch: Vec<(usize, f32)> = Vec::with_capacity(oc_n);
        for pidx in 0..total_cols {
            let mut energy = 0.0f32;
            for row in 0..patch_size {
                let v = patches[row * total_cols + pidx];
                energy += v * v;
            }
            if energy < 1e-6 { continue; }

            scratch.clear();
            for i in 0..oc_n {
                let v = z[i * total_cols + pidx];
                let v = if v.is_finite() { v.clamp(-100.0, 100.0) } else { 0.0 };
                scratch.push((i, v));
            }
            scratch.sort_unstable_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
            for &(i, v) in scratch.iter().take(take) {
                z_sparse[i * total_cols + pidx] = v;
                updated[i] = true;
            }
            active_patches += 1;
        }

        if active_patches == 0 { return; }

        // 4. R = Wᵀ · Z_sparse: [patch_size, total_cols]
        let mut recon = vec![0.0f32; patch_size * total_cols];
        modgrad_device::backend::ops::matmul_tn(
            &self.weight, &z_sparse, &mut recon, None,
            patch_size, oc_n, total_cols,
        ).expect("Conv2d::hebbian_update: recon matmul_tn dispatch");

        // 5. error = patches − recon (in-place; `recon` becomes `error`).
        let mut error = recon;
        for (err, &pat) in error.iter_mut().zip(patches.iter()) {
            *err = pat - *err;
        }

        // 6. dW = Z_sparse · errorᵀ: [oc_n, patch_size].
        //    error is physically [patch_size, total_cols]; matmul_nt reads it
        //    as B with shape [n=patch_size, k=total_cols] and computes A · Bᵀ.
        let mut dw = vec![0.0f32; oc_n * patch_size];
        modgrad_device::backend::ops::matmul_nt(
            &z_sparse, &error, &mut dw, None,
            oc_n, total_cols, patch_size,
        ).expect("Conv2d::hebbian_update: dW matmul_nt dispatch");

        // 7. Apply + row-normalize updated filters.
        let scale = lr / active_patches as f32;
        for i in 0..oc_n * patch_size {
            let d = scale * dw[i];
            if d.is_finite() {
                self.weight[i] += d;
            }
        }
        for o in 0..oc_n {
            if !updated[o] { continue; }
            let w_off = o * patch_size;
            let norm: f32 = (0..patch_size)
                .map(|i| self.weight[w_off + i].powi(2))
                .sum::<f32>()
                .sqrt();
            if norm > 1e-6 {
                for i in 0..patch_size {
                    self.weight[w_off + i] /= norm;
                }
            }
        }
    }
}

/// 1D convolutional layer for audio processing.
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub struct Conv1d {
    pub weight: Vec<f32>,  // [out_ch × in_ch × kernel]
    pub bias: Vec<f32>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
}

impl Conv1d {
    pub fn new(in_ch: usize, out_ch: usize, kernel: usize, stride: usize) -> Self {
        let n = in_ch * kernel;
        let scale = (2.0 / n as f32).sqrt();
        let mut rng_state = (in_ch * out_ch * kernel) as u64 + 7919;
        let weight: Vec<f32> = (0..out_ch * in_ch * kernel)
            .map(|_| simple_rng_normal(&mut rng_state) * scale)
            .collect();
        let bias = vec![0.0; out_ch];
        Self { weight, bias, in_channels: in_ch, out_channels: out_ch,
               kernel_size: kernel, stride }
    }

    /// Forward: [in_ch × length] → [out_ch × length']
    pub fn forward(&self, input: &[f32], length: usize) -> (Vec<f32>, usize) {
        let k = self.kernel_size;
        let s = self.stride;
        let out_len = (length - k) / s + 1;
        let mut output = vec![0.0f32; self.out_channels * out_len];

        for oc in 0..self.out_channels {
            for ol in 0..out_len {
                let mut sum = self.bias[oc];
                for ic in 0..self.in_channels {
                    for ki in 0..k {
                        let il = ol * s + ki;
                        if il < length {
                            let in_idx = ic * length + il;
                            let w_idx = oc * (self.in_channels * k) + ic * k + ki;
                            sum += input[in_idx] * self.weight[w_idx];
                        }
                    }
                }
                output[oc * out_len + ol] = sum;
            }
        }
        (output, out_len)
    }

    pub fn param_count(&self) -> usize {
        self.weight.len() + self.bias.len()
    }
}

// ─── Activation functions ────────────────────────────────────

/// Initialize a 12-channel retinal / LGN ganglion-cell filter bank.
///
/// This is the *true* retina layer: feedforward, fixed, evolutionarily
/// specified. No learning ever. 12 output channels covering the
/// well-characterized ganglion-cell types found in primate retina +
/// LGN — center-surround, color-opponent. No oriented edges or motion
/// filters; those are cortical (V1 simple/complex cells) and belong
/// in the learnable `v1` layer.
///
/// The 12 channels:
///   0: luminance ON-center DoG  (bright dot on dark background)
///   1: luminance OFF-center DoG (dark dot on bright)
///   2: red-only ON-center DoG
///   3: green-only ON-center DoG
///   4: blue-only ON-center DoG
///   5: red-ON / green-OFF opponent (spatially co-located)
///   6: green-ON / red-OFF opponent
///   7: blue-ON / yellow-OFF opponent (Y ≈ (R+G)/2)
///   8: yellow-ON / blue-OFF opponent
///   9: wide-field ON (parasol / magnocellular analogue — Gaussian)
///  10: wide-field OFF (parasol OFF)
///  11: luminance integrator (mean of R,G,B — no spatial structure)
pub fn init_retinal_ganglion_filters(conv: &mut Conv2d) {
    let k = conv.kernel_size;
    let in_ch = conv.in_channels;
    assert_eq!(in_ch, 3, "retinal ganglion expects 3 input channels (RGB)");
    assert_eq!(conv.out_channels, 12, "retinal ganglion is a 12-channel bank");
    assert_eq!(k, 3, "retinal DoG uses 3×3 kernels");

    conv.weight.fill(0.0);
    conv.bias.fill(0.0);

    let set_kernel = |conv: &mut Conv2d, oc: usize, ic: usize, kernel: &[f32; 9]| {
        let base = oc * (in_ch * k * k) + ic * (k * k);
        for i in 0..9 { conv.weight[base + i] = kernel[i]; }
    };

    // 3×3 DoG approximation: center cell +8, surround 8 cells -1 each
    // (integral = 0 → responds to contrast, not absolute brightness).
    let on_center: [f32; 9]  = [-1.0, -1.0, -1.0,
                                 -1.0,  8.0, -1.0,
                                 -1.0, -1.0, -1.0];
    let off_center: [f32; 9] = [ 1.0,  1.0,  1.0,
                                  1.0, -8.0,  1.0,
                                  1.0,  1.0,  1.0];
    let scale = 1.0 / 8.0;
    let scaled_on: [f32; 9]  = std::array::from_fn(|i| on_center[i]  * scale);
    let scaled_off: [f32; 9] = std::array::from_fn(|i| off_center[i] * scale);

    // 0: luminance ON-center (all RGB equally)
    for ic in 0..3 { set_kernel(conv, 0, ic, &scaled_on); }
    // 1: luminance OFF-center
    for ic in 0..3 { set_kernel(conv, 1, ic, &scaled_off); }
    // 2: red ON-center (R channel DoG, others 0)
    set_kernel(conv, 2, 0, &scaled_on);
    // 3: green ON-center
    set_kernel(conv, 3, 1, &scaled_on);
    // 4: blue ON-center
    set_kernel(conv, 4, 2, &scaled_on);
    // 5: red-ON / green-OFF opponent (spatially co-located, DoG per channel)
    set_kernel(conv, 5, 0, &scaled_on);
    set_kernel(conv, 5, 1, &scaled_off);
    // 6: green-ON / red-OFF
    set_kernel(conv, 6, 0, &scaled_off);
    set_kernel(conv, 6, 1, &scaled_on);
    // 7: blue-ON / yellow-OFF (yellow = (R+G)/2)
    let half_off: [f32; 9] = std::array::from_fn(|i| scaled_off[i] * 0.5);
    set_kernel(conv, 7, 2, &scaled_on);
    set_kernel(conv, 7, 0, &half_off);
    set_kernel(conv, 7, 1, &half_off);
    // 8: yellow-ON / blue-OFF
    let half_on: [f32; 9] = std::array::from_fn(|i| scaled_on[i] * 0.5);
    set_kernel(conv, 8, 0, &half_on);
    set_kernel(conv, 8, 1, &half_on);
    set_kernel(conv, 8, 2, &scaled_off);

    // Wide-field Gaussian blur (parasol cells are less spatially
    // selective than midget/parvo; this is a soft low-pass).
    let blur: [f32; 9] = [1.0/16.0, 2.0/16.0, 1.0/16.0,
                          2.0/16.0, 4.0/16.0, 2.0/16.0,
                          1.0/16.0, 2.0/16.0, 1.0/16.0];
    let blur_neg: [f32; 9] = std::array::from_fn(|i| -blur[i]);
    // 9: wide-field ON
    for ic in 0..3 { set_kernel(conv, 9, ic, &blur); }
    // 10: wide-field OFF
    for ic in 0..3 { set_kernel(conv, 10, ic, &blur_neg); }

    // 11: luminance integrator — no spatial structure (1×1 equivalent
    // via center-only weight), integrates RGB equally.
    let point: [f32; 9] = [0.0, 0.0, 0.0,
                            0.0, 1.0/3.0, 0.0,
                            0.0, 0.0, 0.0];
    for ic in 0..3 { set_kernel(conv, 11, ic, &point); }
}

/// Stamp a Gabor-like oriented-edge prior onto V1 in a
/// channel-selective way: for each V1 output channel, *replace* the
/// random init on ONE retinal input channel (luminance ON ch 0 or OFF
/// ch 1) with a unit-L2 Gabor kernel, and *preserve* the random init
/// on the other 11 channels. The result: per-filter viz shows clear
/// oriented Gabor structure (interpretable), while full feature
/// mixing across all 12 retinal channels is intact (so V1 can still
/// use color-opponent + wide-field ganglion signals).
///
/// Biology: V1 simple cells are orientation-selective from birth —
/// structure from prenatal retinal waves + genetic wiring, not
/// postnatal image experience. A pure Gabor init (zero all other
/// channels) over-biases and destroyed maze eval (0 % first-step). A
/// fully-additive 1/3-magnitude prior was invisible to the weight viz
/// even though it gave +5 pts eval. This hybrid gets both.
///
/// Expected shape: `[out_ch × in_ch × k × k]` with `in_ch = 12`,
/// `out_ch = 32`, `k = 3`.
pub fn init_v1_gabor_filters(conv: &mut Conv2d, seed: u64) {
    init_v1_gabor_filters_with_strength(conv, seed, 1.0 / std::f32::consts::SQRT_2)
}

/// Genome-friendly variant: explicit Gabor L2 magnitude per 9-weight
/// block. The default `init_v1_gabor_filters` uses 1/√2 ≈ 0.71.
pub fn init_v1_gabor_filters_with_strength(conv: &mut Conv2d, seed: u64, gabor_l2: f32) {
    let k = conv.kernel_size;
    let in_ch = conv.in_channels;
    let out_ch = conv.out_channels;
    assert_eq!(in_ch, 12, "V1 expects 12 retinal ganglion input channels");
    assert_eq!(out_ch, 32, "V1 Gabor bank targets 32 output channels");
    assert_eq!(k, 3, "V1 Gabor init only implemented for 3×3 kernels");

    // Keep existing random weights intact — Gabor is additive.

    // xorshift64 — deterministic per-seed noise without pulling a crate.
    let mut rng_state = seed.wrapping_add(0x9E3779B97F4A7C15);
    let mut rand01 = || {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        (rng_state as f64 / u64::MAX as f64) as f32
    };

    // Gabor hyper-params tuned for 3×3. σ controls envelope tightness,
    // λ the wavelength — both in the same units as the (x, y) grid.
    let sigma = 0.8f32;
    let lambda = 2.5f32;

    // Retinal input channels to build from (see init_retinal_ganglion_filters):
    //   ch 0: luminance ON-center   → edges where "light in the middle"
    //   ch 1: luminance OFF-center  → edges where "dark in the middle"
    let sources: [usize; 2] = [0, 1];

    for oc in 0..32 {
        let orientation_idx = oc % 8;             // 0..8
        let phase_idx = (oc / 8) % 2;             // 0..2
        let source_idx = (oc / 16) % 2;           // 0..2
        let ic = sources[source_idx];

        let theta = orientation_idx as f32 * std::f32::consts::PI / 8.0;
        let phase = if phase_idx == 0 { 0.0 } else { std::f32::consts::FRAC_PI_2 };

        let base = oc * (in_ch * k * k) + ic * (k * k);
        let (cos_t, sin_t) = (theta.cos(), theta.sin());

        let mut kernel = [0.0f32; 9];
        let mut sum = 0.0f32;
        for ky in 0..3usize {
            for kx in 0..3usize {
                let x = kx as f32 - 1.0;
                let y = ky as f32 - 1.0;
                let xp =  x * cos_t + y * sin_t;
                let yp = -x * sin_t + y * cos_t;
                let envelope = (-(xp*xp + yp*yp) / (2.0 * sigma * sigma)).exp();
                let carrier = (2.0 * std::f32::consts::PI * xp / lambda + phase).cos();
                let v = envelope * carrier;
                kernel[ky * 3 + kx] = v;
                sum += v;
            }
        }
        // Zero-mean the kernel so a uniform patch produces ~0 activation
        // (mirrors the DoG "respond to contrast not brightness" property).
        let mean = sum / 9.0;
        for v in kernel.iter_mut() { *v -= mean; }

        // Scale the Gabor to the caller-provided L2 magnitude per 9-weight
        // block. Default is 1/√2 ≈ 0.71 (energy 0.5) — random channel
        // energies span 0.05–0.33 so this cleanly dominates filter_viz
        // while leaving 11 random channels for feature mixing.
        let norm: f32 = kernel.iter().map(|v| v*v).sum::<f32>().sqrt().max(1e-6);
        let rescale = gabor_l2 / norm;
        for v in kernel.iter_mut() { *v *= rescale; }

        // REPLACE random weights on the source channel (clean Gabor
        // tile), leave other channels at their random init for
        // feature mixing.
        for i in 0..9 { conv.weight[base + i] = kernel[i]; }

        // Small noise on the Gabor itself for symmetry breaking.
        let noise_amp = 0.03 * gabor_l2;
        for i in 0..9 {
            conv.weight[base + i] += (rand01() - 0.5) * 2.0 * noise_amp;
        }
    }
}

/// Overlay contour + corner priors on V2 (additive, like V1).
///
/// V2 in biology codes for collinear contour integration ("this oriented
/// edge continues") and angle/junction detectors (pairs of V1 orientations
/// meeting). We seed a subset of V2's 64 channels with these patterns
/// and leave the rest untouched so feature-mixing across the full V1 feed
/// is preserved.
///
/// Layout (of the 64 V2 out channels):
///   0..8:   collinear-contour — 8 V1 orientations, stripe along axis
///   8..16:  orthogonal-cross  — paired V1 orientations (θ, θ+90°) crossing
///   16..64: left at their random init
///
/// Magnitude: ~1/3 of per-weight random init, same philosophy as V1.
/// V1 output channel layout from `init_v1_gabor_filters`:
///   0..8:   orientation θ, phase 0, lum-ON source
///   8..16:  orientation θ, phase 1, lum-ON
///   16..24: orientation θ, phase 0, lum-OFF
///   24..32: orientation θ, phase 1, lum-OFF
/// We pull from the phase-0 lum-ON bank (channels 0..8) as the canonical
/// "this position has orientation θ" signal.
/// V2 contour + orthogonal-cross priors. `cardinal_only`: restrict
/// to h / v / ±45° (axis-aligned tasks); otherwise full 8-orientation
/// bank (better for natural images).
pub fn init_v2_contour_filters(conv: &mut Conv2d, cardinal_only: bool) {
    let k = conv.kernel_size;
    let in_ch = conv.in_channels;
    let out_ch = conv.out_channels;
    assert_eq!(in_ch, 32, "V2 expects 32 V1 input channels");
    assert_eq!(out_ch, 64, "V2 contour bank targets 64 output channels");
    assert_eq!(k, 3, "V2 contour init only implemented for 3×3 kernels");

    let fan_in = (in_ch * k * k) as f32;
    let strength = 0.33 / fan_in.sqrt();

    // Given a 3×3 kernel and an orientation θ, return a 3-stripe
    // activation pattern: bright at center (1,1) and at two positions
    // displaced by ±(cos θ, sin θ). Quantised to the 3×3 grid.
    let stripe_pattern = |theta: f32| -> [f32; 9] {
        let (cos_t, sin_t) = (theta.cos(), theta.sin());
        let mut p = [0.0f32; 9];
        p[1 * 3 + 1] = 1.0; // center
        let dx = cos_t.round() as isize;
        let dy = sin_t.round() as isize;
        for sign in &[-1isize, 1] {
            let gx = 1 + sign * dx;
            let gy = 1 + sign * dy;
            if (0..3).contains(&gx) && (0..3).contains(&gy) {
                p[(gy as usize) * 3 + gx as usize] += 1.0;
            }
        }
        p
    };

    // Cardinal-only mode restricts priors to h / v / ±45° — the
    // orientations axis-aligned mazes actually contain. All-orientation
    // mode seeds 8 V1 orientations; useful for natural images.
    let orientation_indices: Vec<usize> = if cardinal_only {
        vec![0, 2, 4, 6]        // 0°, 45°, 90°, 135°
    } else {
        (0..8).collect()
    };

    // First bank: collinear contour detectors (one per selected θ).
    // V2[bank_idx]'s weight for V1 channel `ic` (same orientation) =
    // stripe along θ.
    for (bank_idx, &ic) in orientation_indices.iter().enumerate() {
        let theta = ic as f32 * std::f32::consts::PI / 8.0;
        let pat = stripe_pattern(theta);
        let oc_out = bank_idx;
        let base = oc_out * (in_ch * k * k) + ic * (k * k);
        for i in 0..9 { conv.weight[base + i] += strength * pat[i]; }
    }

    // Second bank: orthogonal-cross detectors. V2[bank_start + bank_idx]
    // pulls V1 channel `ic` at θ AND V1 channel `ic+4 mod 8` at θ+90°.
    let bank_start = orientation_indices.len();
    for (bank_idx, &ic) in orientation_indices.iter().enumerate() {
        let oc_out = bank_start + bank_idx;
        let theta_a = ic as f32 * std::f32::consts::PI / 8.0;
        let theta_b = theta_a + std::f32::consts::FRAC_PI_2;
        let pat_a = stripe_pattern(theta_a);
        let pat_b = stripe_pattern(theta_b);
        let ic_a = ic;
        let ic_b = (ic + 4) % 8;
        let base_a = oc_out * (in_ch * k * k) + ic_a * (k * k);
        let base_b = oc_out * (in_ch * k * k) + ic_b * (k * k);
        for j in 0..9 {
            conv.weight[base_a + j] += strength * pat_a[j];
            conv.weight[base_b + j] += strength * pat_b[j];
        }
    }
}

/// Initialize retinal layer with biologically-inspired FIXED filters.
/// These model the ~30+ ganglion cell types in the human retina.
/// NOT learned — equivalent to evolutionary specification of receptor types.
fn init_retinal_filters(conv: &mut Conv2d) {
    let k = conv.kernel_size; // 3
    let in_ch = conv.in_channels; // 3 (RGB)
    let out_ch = conv.out_channels; // 32

    // Zero all weights first
    conv.weight.fill(0.0);
    conv.bias.fill(0.0);

    let mut oc = 0;
    let set_kernel = |conv: &mut Conv2d, oc: usize, ic: usize, kernel: &[f32; 9]| {
        let base = oc * (in_ch * k * k) + ic * (k * k);
        for i in 0..9 { conv.weight[base + i] = kernel[i]; }
    };

    // --- Luminance ON center-surround (4 filters: per RGB + combined) ---
    // Models ON-center ganglion cells: excited by light in center, inhibited by surround
    let on_center: [f32; 9] = [-1.0/8.0, -1.0/8.0, -1.0/8.0,
                                -1.0/8.0,  1.0,     -1.0/8.0,
                                -1.0/8.0, -1.0/8.0, -1.0/8.0];
    for ic in 0..3 {
        set_kernel(conv, oc, ic, &on_center);
    }
    oc += 1;

    // --- Luminance OFF center-surround ---
    let off_center: [f32; 9] = [ 1.0/8.0,  1.0/8.0,  1.0/8.0,
                                  1.0/8.0, -1.0,      1.0/8.0,
                                  1.0/8.0,  1.0/8.0,  1.0/8.0];
    for ic in 0..3 {
        set_kernel(conv, oc, ic, &off_center);
    }
    oc += 1;

    // --- Color opponent: Red - Green ---
    let r_plus: [f32; 9] = [0.0, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, 0.0]; // center
    let g_minus: [f32; 9] = [0.0, 0.0, 0.0,  0.0, -1.0, 0.0,  0.0, 0.0, 0.0];
    set_kernel(conv, oc, 0, &r_plus);  // R channel +
    set_kernel(conv, oc, 1, &g_minus); // G channel -
    oc += 1;

    // --- Color opponent: Green - Red ---
    set_kernel(conv, oc, 0, &g_minus);
    set_kernel(conv, oc, 1, &r_plus);
    oc += 1;

    // --- Color opponent: Blue - Yellow (B - (R+G)/2) ---
    let half_minus: [f32; 9] = [0.0, 0.0, 0.0,  0.0, -0.5, 0.0,  0.0, 0.0, 0.0];
    set_kernel(conv, oc, 2, &r_plus);     // B +
    set_kernel(conv, oc, 0, &half_minus); // R -0.5
    set_kernel(conv, oc, 1, &half_minus); // G -0.5
    oc += 1;

    // --- Horizontal edge detector ---
    let h_edge: [f32; 9] = [-1.0, -1.0, -1.0,
                              0.0,  0.0,  0.0,
                              1.0,  1.0,  1.0];
    for ic in 0..3 { set_kernel(conv, oc, ic, &h_edge); }
    oc += 1;

    // --- Vertical edge detector ---
    let v_edge: [f32; 9] = [-1.0, 0.0, 1.0,
                             -1.0, 0.0, 1.0,
                             -1.0, 0.0, 1.0];
    for ic in 0..3 { set_kernel(conv, oc, ic, &v_edge); }
    oc += 1;

    // --- Diagonal edge detectors (2 orientations) ---
    let d1: [f32; 9] = [ 0.0, -1.0, -1.0,
                          1.0,  0.0, -1.0,
                          1.0,  1.0,  0.0];
    let d2: [f32; 9] = [-1.0, -1.0,  0.0,
                         -1.0,  0.0,  1.0,
                          0.0,  1.0,  1.0];
    for ic in 0..3 { set_kernel(conv, oc, ic, &d1); }
    oc += 1;
    for ic in 0..3 { set_kernel(conv, oc, ic, &d2); }
    oc += 1;

    // --- Motion/change detection (temporal difference approximated by spatial) ---
    // Horizontal motion-sensitive (ON-OFF direction selective)
    let motion_h: [f32; 9] = [-0.5, 0.0, 0.5,
                               -1.0, 0.0, 1.0,
                               -0.5, 0.0, 0.5];
    for ic in 0..3 { set_kernel(conv, oc, ic, &motion_h); }
    oc += 1;

    // Vertical motion-sensitive
    let motion_v: [f32; 9] = [-0.5, -1.0, -0.5,
                                0.0,  0.0,  0.0,
                                0.5,  1.0,  0.5];
    for ic in 0..3 { set_kernel(conv, oc, ic, &motion_v); }
    oc += 1;

    // --- Per-channel edge detectors (R/G/B independently) ---
    for ic in 0..3 {
        set_kernel(conv, oc, ic, &h_edge);
        oc += 1;
        set_kernel(conv, oc, ic, &v_edge);
        oc += 1;
    }
    // oc is now 17

    // --- Gaussian blur (low-pass, represents parasol/magno cells) ---
    let blur: [f32; 9] = [1.0/16.0, 2.0/16.0, 1.0/16.0,
                           2.0/16.0, 4.0/16.0, 2.0/16.0,
                           1.0/16.0, 2.0/16.0, 1.0/16.0];
    for ic in 0..3 { set_kernel(conv, oc, ic, &blur); }
    oc += 1;

    // --- Sharpen (high-pass) ---
    let sharp: [f32; 9] = [-1.0/8.0, -1.0/8.0, -1.0/8.0,
                            -1.0/8.0,  2.0,     -1.0/8.0,
                            -1.0/8.0, -1.0/8.0, -1.0/8.0];
    for ic in 0..3 { set_kernel(conv, oc, ic, &sharp); }
    oc += 1;

    // Fill remaining filters with DoG (Difference of Gaussians) at multiple scales.
    // NOT Gabors — retinal ganglion cells use center-surround, not oriented filters.
    // Gabors belong in V1 (cortex), not retina.
    let dogs: Vec<[f32; 9]> = vec![
        // Tight DoG (small receptive field, midget cells)
        [-0.05, -0.15, -0.05,  -0.15, 1.0, -0.15,  -0.05, -0.15, -0.05],
        // Wide DoG (large receptive field, parasol cells)
        [-0.11, -0.11, -0.11,  -0.11, 1.0, -0.11,  -0.11, -0.11, -0.11],
        // Inverted DoG (OFF-center cells)
        [0.05, 0.15, 0.05,  0.15, -1.0, 0.15,  0.05, 0.15, 0.05],
        [0.11, 0.11, 0.11,  0.11, -1.0, 0.11,  0.11, 0.11, 0.11],
    ];
    while oc < out_ch {
        let dog = &dogs[oc % dogs.len()];
        // Apply to different channel combinations
        let ic = oc % in_ch;
        set_kernel(conv, oc, ic, dog);
        oc += 1;
    }
}

/// Leaky ReLU: preserves weak negative signals (subthreshold activity).
/// Real neurons don't fully zero below threshold — membrane potential
/// still fluctuates. The leak (0.1) keeps information flowing through
/// the hierarchy so downstream layers can learn from it.
#[inline]
fn leaky_relu(x: &mut [f32]) {
    for v in x.iter_mut() {
        if *v < 0.0 { *v *= 0.1; }
    }
}

/// Backward pass through `leaky_relu` (slope 0.1). Multiplies the
/// upstream gradient in-place by 0.1 wherever the forward output was
/// negative — which means the pre-activation was negative, so the
/// local derivative is the leaky slope. Positive forward values came
/// through as-is (slope 1.0); they pass gradient unchanged.
fn leaky_relu_backward(d_out: &mut [f32], post: &[f32]) {
    debug_assert_eq!(d_out.len(), post.len());
    for (g, &p) in d_out.iter_mut().zip(post) {
        if p < 0.0 { *g *= 0.1; }
    }
}

#[inline]
fn adaptive_avg_pool(input: &[f32], channels: usize, h: usize, w: usize) -> Vec<f32> {
    // Global average pooling: [ch × h × w] → [ch]
    let mut out = vec![0.0f32; channels];
    let hw = h * w;
    for c in 0..channels {
        let sum: f32 = input[c * hw..(c + 1) * hw].iter().sum();
        out[c] = sum / hw as f32;
    }
    out
}

// ─── Visual Retina ───────────────────────────────────────────

/// Visual retina: hardcoded feature extraction (evolved, not learned).
///
/// The retina is NOT learned — it's specified by the genome (evolution).
/// ~30+ ganglion cell types each compute a fixed feature:
/// - ON/OFF cells (light increase/decrease)
/// - Center-surround (edge detection)
/// - Color-opponent (red-green, blue-yellow)
/// - Direction-selective (motion detection)
///
// ─── V4 Continuous Thought Machine ───────────────────────────
//
// V4 in real biology is *where* visual attention lives — the strongest
// attentional modulation in the ventral stream is here, per Desimone &
// Moran 1985 and decades of follow-ups. Modeling V4 as a plain Conv2d
// misses this: convs don't attend, don't iterate, don't have ticks.
//
// V4Ctm runs a CTM over the spatial tokens that come out of V4's
// Conv2d forward. Each tick attends over all spatial positions,
// updates the CTM's internal activated state + sync readout, and the
// final tick's `sync_out` is the distilled "what am I looking at"
// summary the downstream brain consumes.
//
// Design choice (Option A from retina-v2 design): V4Ctm currently
// *postprocesses* V4 Conv2d's spatial tokens rather than replacing
// the conv. This lets existing call sites (filter_viz, dream_gallery)
// that enumerate the conv layers keep working while we validate the
// CTM attachment. A future step will make V4Ctm consume V2 tokens
// directly and retire the V4 Conv2d.

/// A CTM instance wrapping the spatial token output of `VisualCortex.v4`.
/// Takes `[n_tokens × token_dim]` features, runs `iterations` internal
/// ticks of attention over positions, emits `[d_model]` sync_out.
///
/// Fresh CtmState each forward — we don't carry hidden state across
/// images. That matches the biological intuition that cortical
/// processing resets between saccades, and keeps the forward-call
/// API stateless for the caller.
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub struct V4Ctm {
    /// Learnable CTM weights (synapse, NLM, sync pairings, output proj).
    pub weights: modgrad_ctm::weights::CtmWeights,
}

impl V4Ctm {
    /// Construct a V4-CTM for input tokens of `token_dim` width,
    /// producing a `d_model`-wide summary via `iterations` ticks.
    /// Pool width and attention head count are picked sensibly from
    /// `d_model`; callers needing finer control can build a
    /// `CtmConfig` manually and call `V4Ctm::from_config`.
    pub fn new(token_dim: usize, d_model: usize, iterations: usize) -> Self {
        let heads = ((d_model / 16).max(1)).min(8);
        let config = modgrad_ctm::config::CtmConfig {
            iterations,
            d_model,
            d_input: d_model,       // kv_proj projects token_dim → d_model
            heads,
            n_synch_out: d_model,
            n_synch_action: d_model,
            synapse_depth: 2,
            memory_length: 8,
            deep_nlms: true,
            memory_hidden_dims: 4,
            out_dims: d_model,      // unused (we read sync_out, not predictions)
            n_random_pairing_self: 0,
            min_width: (d_model / 4).max(4),
            exit_strategy: modgrad_ctm::config::ExitStrategy::None,
            collect_trajectories: false,
        };
        let weights = modgrad_ctm::weights::CtmWeights::new(config, token_dim);
        Self { weights }
    }

    /// Forward: tokens `[n_tokens × token_dim]` → `sync_out [d_model]`.
    /// State is constructed fresh internally.
    pub fn forward(&self, tokens: &[f32], n_tokens: usize, token_dim: usize) -> Vec<f32> {
        debug_assert_eq!(tokens.len(), n_tokens * token_dim);
        let mut state = modgrad_ctm::weights::CtmState::new(&self.weights);
        let output = modgrad_ctm::forward::ctm_forward(
            &self.weights, &mut state,
            modgrad_ctm::forward::CtmInput::Raw {
                obs: tokens, n_tokens, raw_dim: token_dim,
            },
        );
        output.sync_out
    }

    pub fn d_model(&self) -> usize { self.weights.config.d_model }
    pub fn iterations(&self) -> usize { self.weights.config.iterations }
    pub fn param_count(&self) -> usize { self.weights.n_params() }
}

/// Four-stage visual front-end: retina → V1 → V2 → V4.
///
/// Biologically-honest split of what was previously conflated in a
/// single "v1" layer:
///
/// - `retina` (fixed)  : retinal ganglion / LGN analogue. 12 channels
///   of center-surround DoG + color-opponent filters. Evolutionarily
///   specified, never learns. This is the *actual* retina (eye).
/// - `v1` (learnable)  : V1 cortex — simple / complex cells. Starts
///   random, learns via Hebbian/GHL. Emerges oriented edges, motion
///   sensitivity, etc. from experience.
/// - `v2` (learnable)  : V2 cortex — contours, figure-ground, texture.
/// - `v4` (learnable)  : V4 cortex — shapes, curvature, attention-
///   modulated features. (Destined to become a CTM instance in the
///   retina-v2 architecture; currently still Conv2d.)
///
/// Missing vs. real cortex (known gaps, worth flagging):
/// - No attention in-layer (V1-V4 are heavily attention-modulated in
///   cortex; only the downstream CTM has attention in our current
///   architecture)
/// - No recurrence within a layer (one forward pass)
/// - No top-down feedback at inference (training uses it via
///   `train_dream` / `lsd`)
/// - No adaptive per-layer exit gate
///
/// Pool: global average over V4 → observation vector.
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub struct VisualCortex {
    /// Retinal ganglion / LGN analogue.  **FIXED.**  3 → 12 channels
    /// of DoG + color opponents. Center-surround, no orientation.
    #[serde(default = "default_retina")]
    pub retina: Conv2d,
    /// V1 cortex — learnable. 12 → 32 channels.
    pub v1: Conv2d,
    /// V2 cortex — learnable. 32 → 64 channels.
    pub v2: Conv2d,
    /// V4 cortex Conv2d — learnable. 64 → 128 channels. Kept as the
    /// feature-extraction frontend for the V4-CTM attention stage.
    pub v4: Conv2d,
    /// V4-CTM — attention + ticks + sync over V4 spatial tokens.
    /// Outputs a `[d_model]` summary vector consumed by the
    /// downstream brain. Constructed with `d_model = 128` to match V4
    /// channels.
    #[serde(default = "default_v4_ctm")]
    pub v4_ctm: V4Ctm,
    pub pool_dim: usize,
    pub input_h: usize,
    pub input_w: usize,
    /// Pharmacological state of V1/V2/V4 — modulates `lsd()` effect.
    #[serde(default)]
    pub receptors: ReceptorState,
}

fn default_v4_ctm() -> V4Ctm {
    // Default matches `new`/`preserve_spatial`: V4 Conv2d emits 128-dim
    // spatial tokens, V4-CTM runs 4 ticks producing a 128-dim summary.
    V4Ctm::new(128, 128, 4)
}

/// Serde default for the new `retina` field when loading older
/// checkpoints written before the retina/v1 split. Produces a
/// properly-initialized 12-channel DoG bank sized for 224×224 (a
/// convenient default — loaders should overwrite `input_h`/`input_w`
/// anyway). Downstream code re-reads `input_h` from the loaded
/// struct, so the retina's own `input_h` init here doesn't matter
/// beyond compilability.
fn default_retina() -> Conv2d {
    let mut r = Conv2d::new(3, 12, 3, 2, 1);
    init_retinal_ganglion_filters(&mut r);
    r
}

/// Receptor-level state of the visual cortex.
///
/// Models the availability of 5-HT2A receptors (the primary target of
/// classical psychedelics). In biology: each trip binds and
/// desensitises receptors for hours-to-days; full sensitivity recovers
/// over days. In code: `ht2a` is a scalar in [0, 1], consumed by
/// `VisualCortex::lsd` to scale effective plasticity, and restored by
/// `VisualCortex::tick(rate)`.
///
/// This is intentionally a separate abstraction from the transmitter-
/// level `modgrad_ctm::bio::Neuromodulators` — that models *how much
/// serotonin is in the synaptic cleft*, this models *how many
/// receptors are available to bind it*. A cortex can have normal
/// transmitter levels and desensitised receptors, which is exactly
/// what produces psychedelic tolerance.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub struct ReceptorState {
    /// 5-HT2A receptor availability. 1.0 = sober baseline, fully
    /// sensitive. Decreases with `lsd()` use, recovers with `tick()`.
    /// Floored at 0.05 so even a chronically-tripping cortex retains
    /// minimal responsiveness — matches biology: complete receptor
    /// wipe-out doesn't happen from drug exposure alone.
    pub ht2a: f32,
    /// Steps since last call to `lsd`. Useful for callers that gate
    /// re-dosing on a minimum recovery window. `u64::MAX` until the
    /// first trip.
    pub steps_since_trip: u64,
}

impl Default for ReceptorState {
    fn default() -> Self {
        Self { ht2a: 1.0, steps_since_trip: u64::MAX }
    }
}

impl ReceptorState {
    /// Consume receptors for a trip. `dose_fraction` is the effective
    /// dose in [0, 1+] (caller-normalised — a full-sensitisation
    /// dose is 1.0, smaller is sub-clinical, larger is heroic but not
    /// linearly more effective due to the floor and the cap below).
    /// Returns the effective potency after availability scaling:
    ///   potency = ht2a * min(dose_fraction, 1.0)
    /// Then desensitises: ht2a *= 1 - 0.25 * min(dose, 1.0).
    pub fn apply_dose(&mut self, dose_fraction: f32) -> f32 {
        let dose = dose_fraction.max(0.0).min(1.0);
        let potency = self.ht2a * dose;
        let desens = 0.25 * dose;
        self.ht2a = (self.ht2a * (1.0 - desens)).max(0.05);
        self.steps_since_trip = 0;
        potency
    }

    /// Advance time — exponential recovery toward full sensitivity.
    /// Typical rates: 1e-3 for step-level aging (full recovery over
    /// several thousand steps), 1e-2 for coarse-grained time.
    pub fn tick(&mut self, recovery_rate: f32) {
        self.steps_since_trip = self.steps_since_trip.saturating_add(1);
        let rate = recovery_rate.clamp(0.0, 1.0);
        self.ht2a += rate * (1.0 - self.ht2a);
    }
}

impl VisualCortex {
    /// Serialize the full cortex (V1/V2/V4 weights + receptor state
    /// + input dims) to `path`. Binary wincode with a magic header, or
    /// JSON if the path ends in `.json`.
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        modgrad_persist::persist::save(self, path)
    }

    /// Load a cortex from a file previously written with `save`.
    /// Shape compatibility is on the caller — if you pretrained on 96×96
    /// and want to use the weights on a different input size, set
    /// `input_h`/`input_w` after loading. Conv2d weights depend only on
    /// `(in_ch, out_ch, kernel_size)`, not on stride or input dims, so
    /// stride can also be mutated post-load to switch between the
    /// `::new` (stride-2 throughout) and `::preserve_spatial` (stride-1
    /// on V2/V4) geometries without retraining.
    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        modgrad_persist::persist::load(path)
    }

    /// One Hebbian update on a single `[3 × input_h × input_w]` image.
    ///
    /// Forwards the image through V1 (fixed), applies leaky-ReLU,
    /// Hebbian-updates V2 from those features, forwards V2, applies
    /// leaky-ReLU, Hebbian-updates V4 from those. Call once per image
    /// when streaming — `train_hebbian` wraps this over a fully-loaded
    /// batch, but datasets like STL-10 unlabeled (2.6 GB) need the
    /// streaming path instead.
    pub fn hebbian_step(&mut self, img: &[f32], lr: f32, sparsity: usize) {
        // Single-image convenience over the batched path.
        self.hebbian_step_batch(img, 1, lr, sparsity);
    }

    /// Batched Hebbian step over `n` images packed contiguously in
    /// `images`. Each image is `[3 × input_h × input_w]` row-major.
    ///
    /// One matmul per conv across the whole batch instead of one per
    /// image — amortizes hipBLAS dispatch overhead and lets the SMU
    /// sustain load long enough to boost clocks. At typical STL-10
    /// scales (96×96 input, batch 64), expected 3-5× throughput vs
    /// per-image on the 7600M XT.
    pub fn hebbian_step_batch(
        &mut self, images: &[f32], n: usize,
        lr: f32, sparsity: usize,
    ) {
        let h = self.input_h;
        let w = self.input_w;
        debug_assert_eq!(images.len(), n * 3 * h * w,
            "images len {} ≠ n({})·3·{}·{}", images.len(), n, h, w);

        // Retina (ganglion/LGN analogue): FIXED, batched forward + ReLU.
        let (mut r_out, rh, rw) = self.retina.forward(images, n, h, w);
        leaky_relu(&mut r_out);

        // V1 cortex: learns from retinal features.
        self.v1.hebbian_update(&r_out, n, rh, rw, lr, sparsity);
        let (mut v1_out, h1, w1) = self.v1.forward(&r_out, n, rh, rw);
        leaky_relu(&mut v1_out);

        // V2 cortex: learns from V1.
        self.v2.hebbian_update(&v1_out, n, h1, w1, lr * 0.75, sparsity);
        let (mut v2_out, h2, w2) = self.v2.forward(&v1_out, n, h1, w1);
        leaky_relu(&mut v2_out);

        // V4 cortex: learns from V2 at half rate — features further from
        // pixels should move more slowly.
        self.v4.hebbian_update(&v2_out, n, h2, w2, lr * 0.5, sparsity);
    }

    /// One GHL (sign-modulated three-factor) update over V1, V2, V4.
    /// Retina (ganglion/LGN) stays fixed — it's a feedforward filter
    /// bank, never learns, consistent with biology.
    ///
    /// Caller supplies `d_tokens` — the gradient of task loss w.r.t.
    /// the tokens this cortex emitted. We forward with intermediates,
    /// backprop through V4 → V2 → V1 to get per-layer `sign(∂L/∂W)`,
    /// then apply the GHL local rule at each cortical layer.
    ///
    /// `d_tokens` layout: `[n_tokens × token_dim]` row-major where
    /// `n_tokens = h4 * w4` and `token_dim = v4.out_channels`.
    pub fn ghl_step(&mut self, raw: &[f32], d_tokens: &[f32], lr: f32, tau: f32) {
        let h = self.input_h;
        let w = self.input_w;
        let k = self.v4.out_channels;

        // ── Forward, keeping intermediates ────────────────────────
        let (mut r_out, rh, rw) = self.retina.forward(raw, 1, h, w);
        leaky_relu(&mut r_out);

        let (v1_y_pre, h1, w1, v1_patches) =
            self.v1.forward_with_patches(&r_out, 1, rh, rw);
        let mut v1_out = v1_y_pre.clone();
        leaky_relu(&mut v1_out);

        let (v2_y_pre, h2, w2, v2_patches) =
            self.v2.forward_with_patches(&v1_out, 1, h1, w1);
        let mut v2_out = v2_y_pre.clone();
        leaky_relu(&mut v2_out);

        let (v4_y_pre, h4, w4, v4_patches) =
            self.v4.forward_with_patches(&v2_out, 1, h2, w2);
        let mut v4_out = v4_y_pre.clone();
        leaky_relu(&mut v4_out);

        let n_tokens = h4 * w4;
        debug_assert_eq!(d_tokens.len(), n_tokens * k,
            "d_tokens length {} ≠ n_tokens·token_dim ({}·{})",
            d_tokens.len(), n_tokens, k);

        // ── Backward: reshape tokens → CHW, chain through layers ──
        let mut d_v4_out = vec![0.0f32; k * n_tokens];
        for y in 0..h4 {
            for x in 0..w4 {
                let token_idx = y * w4 + x;
                for c in 0..k {
                    d_v4_out[c * n_tokens + (y * w4 + x)] =
                        d_tokens[token_idx * k + c];
                }
            }
        }
        leaky_relu_backward(&mut d_v4_out, &v4_out);
        let (d_w_v4, _d_b_v4, d_v2_post) = self.v4.backward(
            &d_v4_out, &v4_patches, 1, h2, w2, h4, w4);

        let mut d_v2_pre = d_v2_post;
        leaky_relu_backward(&mut d_v2_pre, &v2_out);
        let (d_w_v2, _d_b_v2, d_v1_post) = self.v2.backward(
            &d_v2_pre, &v2_patches, 1, h1, w1, h2, w2);

        let mut d_v1_pre = d_v1_post;
        leaky_relu_backward(&mut d_v1_pre, &v1_out);
        let (d_w_v1, _d_b_v1, _d_r) = self.v1.backward(
            &d_v1_pre, &v1_patches, 1, rh, rw, h1, w1);

        // ── GHL updates at each cortical layer ────────────────────
        let sign_of = |g: &[f32]| -> Vec<f32> {
            g.iter().map(|&v| {
                if v > 0.0 { 1.0 } else if v < 0.0 { -1.0 } else { 0.0 }
            }).collect()
        };

        let n_patches_v1 = h1 * w1;
        let n_patches_v2 = h2 * w2;
        let n_patches_v4 = h4 * w4;
        self.v1.ghl_update(&v1_patches, &v1_y_pre, &sign_of(&d_w_v1), n_patches_v1, lr, tau);
        self.v2.ghl_update(&v2_patches, &v2_y_pre, &sign_of(&d_w_v2), n_patches_v2, lr, tau);
        self.v4.ghl_update(&v4_patches, &v4_y_pre, &sign_of(&d_w_v4), n_patches_v4, lr, tau);
    }

    /// Create a cortex for images of size h×w×3. Default stride-2 on
    /// retina + V1 (aggressive early downsampling), stride-2 on V2/V4.
    /// Suited to large natural-image inputs (STL-10, ImageNet).
    ///
    /// - `retina` (3→12, stride 2): fixed DoG + color-opponent bank.
    ///   Evolutionarily specified, never learns.
    /// - `v1` (12→32, stride 1): learnable cortex — orientation, motion,
    ///   etc. emerge from training.
    /// - `v2` (32→64, stride 2): learnable cortex — contours, texture.
    /// - `v4` (64→128, stride 2): learnable cortex — shapes, curvature.
    ///
    /// Real biology has attention + recurrence + top-down feedback at
    /// every cortical stage; we model a purely feedforward subset here.
    /// The destined retina-v2 replaces `v4` with a CTM instance to get
    /// proper attention and ticks at that level.
    pub fn new(h: usize, w: usize) -> Self {
        // Production prior set: standard ganglion + V1 Gabor hybrid,
        // V2 random. Other variants live in `Genome`.
        let mut retina = Conv2d::new(3, 12, 3, 2, 1);
        init_retinal_ganglion_filters(&mut retina);
        let mut v1 = Conv2d::new(12, 32, 3, 1, 1);
        init_v1_gabor_filters(&mut v1, 0xC0FFEE);
        let v2 = Conv2d::new(32, 64, 3, 2, 1);
        let v4 = Conv2d::new(64, 128, 3, 2, 1);
        let v4_ctm = V4Ctm::new(128, 128, 4);
        Self { retina, v1, v2, v4, v4_ctm, pool_dim: 128,
               input_h: h, input_w: w, receptors: ReceptorState::default() }
    }

    /// For CIFAR-10 (32×32×3)
    pub fn cifar() -> Self { Self::new(32, 32) }

    /// For ImageNet-scale (224×224×3)
    pub fn imagenet() -> Self { Self::new(224, 224) }

    /// Preset for small inputs where default all-stride-2 would collapse
    /// the token grid to trivial sizes. Uses stride 1 on V1/V2/V4 so
    /// spatial dims are preserved past the retina downsample.
    ///
    /// Example — 39×39 input → retina(stride 2): 20×20 → V1(stride 1):
    /// 20×20 → V2(stride 1): 20×20 → V4(stride 1): 20×20 = 400 tokens.
    /// 13×13 input → 7×7 = 49 tokens.
    pub fn preserve_spatial(h: usize, w: usize) -> Self {
        // Production prior set: standard ganglion + V1 Gabor hybrid,
        // V2 random. Other variants live in `Genome`.
        let mut retina = Conv2d::new(3, 12, 3, 2, 1);
        init_retinal_ganglion_filters(&mut retina);
        let mut v1 = Conv2d::new(12, 32, 3, 1, 1);
        init_v1_gabor_filters(&mut v1, 0xC0FFEE);
        let v2 = Conv2d::new(32, 64, 3, 1, 1);
        let v4 = Conv2d::new(64, 128, 3, 1, 1);
        let v4_ctm = V4Ctm::new(128, 128, 4);
        Self { retina, v1, v2, v4, v4_ctm, pool_dim: 128,
               input_h: h, input_w: w, receptors: ReceptorState::default() }
    }

    /// Genome-friendly constructor — no priors, no env vars, just
    /// Conv2d random init. `Genome::express()` calls this and then
    /// applies whatever priors the Genome specifies. Use this when
    /// you need full control over what's "innate" vs left random.
    pub fn random(h: usize, w: usize) -> Self {
        let retina = Conv2d::new(3, 12, 3, 2, 1);
        let v1 = Conv2d::new(12, 32, 3, 1, 1);
        let v2 = Conv2d::new(32, 64, 3, 1, 1);
        let v4 = Conv2d::new(64, 128, 3, 1, 1);
        let v4_ctm = V4Ctm::new(128, 128, 4);
        Self { retina, v1, v2, v4, v4_ctm, pool_dim: 128,
               input_h: h, input_w: w, receptors: ReceptorState::default() }
    }

    /// Train the visual retina with Hebbian sparse coding.
    /// Each layer learns features from the previous layer's output,
    /// bottom-up: V1 trains on pixels, V2 on V1 features, V4 on V2 features.
    ///
    /// images: list of raw pixel arrays [3 × h × w]
    /// epochs: how many passes over the dataset
    /// Returns per-layer reconstruction error.
    pub fn train_hebbian(&mut self, images: &[&[f32]], epochs: usize, lr: f32) -> Vec<f32> {
        let h = self.input_h;
        let w = self.input_w;
        let sparsity = 8; // top-K winners (out of 32/64/128 channels)

        let mut errors = Vec::new();

        for epoch in 0..epochs {
            let mut v1_err = 0.0f32;
            let mut v2_err = 0.0f32;
            let mut v4_err = 0.0f32;

            for (i, &img) in images.iter().enumerate() {
                // V1 (retina) is FIXED — evolved, not learned.
                // Only cortical layers (V2, V4) learn via Hebbian.

                // Retina (fixed DoG + color opponents) → V1 input
                let (mut r_out, rh, rw) = self.retina.forward(img, 1, h, w);
                leaky_relu(&mut r_out);

                // V1 cortex learns from retinal features
                self.v1.hebbian_update(&r_out, 1, rh, rw, lr, sparsity);
                let (mut v1_out, h1, w1) = self.v1.forward(&r_out, 1, rh, rw);
                leaky_relu(&mut v1_out);

                // V2 cortex learns from V1 features
                self.v2.hebbian_update(&v1_out, 1, h1, w1, lr * 0.75, sparsity);
                let (mut v2_out, h2, w2) = self.v2.forward(&v1_out, 1, h1, w1);
                leaky_relu(&mut v2_out);

                // V4 cortex learns from V2 (half-rate for deepest layer)
                self.v4.hebbian_update(&v2_out, 1, h2, w2, lr * 0.5, sparsity);

                if i % 5000 == 0 && i > 0 {
                    eprintln!("    epoch {epoch} image {i}/{}", images.len());
                }
            }

            // Measure forward-pass variance per layer (proxy for
            // feature quality — good features have high variance).
            if let Some(&img) = images.first() {
                let (mut r_out, rh, rw) = self.retina.forward(img, 1, h, w);
                leaky_relu(&mut r_out);

                let (mut v1_out, h1, w1) = self.v1.forward(&r_out, 1, rh, rw);
                leaky_relu(&mut v1_out);
                v1_err = v1_out.iter().map(|x| x * x).sum::<f32>() / v1_out.len() as f32;

                let (mut v2_out, h2, w2) = self.v2.forward(&v1_out, 1, h1, w1);
                leaky_relu(&mut v2_out);
                v2_err = v2_out.iter().map(|x| x * x).sum::<f32>() / v2_out.len() as f32;

                let (v4_out, _, _) = self.v4.forward(&v2_out, 1, h2, w2);
                v4_err = v4_out.iter().map(|x| x * x).sum::<f32>() / v4_out.len() as f32;
            }

            eprintln!("  epoch {epoch}: V1 energy={v1_err:.4} V2={v2_err:.4} V4={v4_err:.4}");
            errors.push(v1_err + v2_err + v4_err);
        }

        errors
    }

    /// Dream pretraining (Hoel 2021, "The Overfitted Brain").
    ///
    /// Synthesizes `n_samples` pseudo-images by seeding V4 with sparse
    /// noise and projecting top-down through the cortex adjoint
    /// (V4^T→V2^T→V1^T), then runs the same Hebbian sparse-coding
    /// update on those synthesized images as `train_hebbian` does on
    /// real ones. No real task data is shown.
    ///
    /// # Deprecated — permanent integration
    ///
    /// This is the legacy permanent-dream path. Empirically it
    /// collapses OOD accuracy on the mazes benchmark (OOD first-step
    /// 2.5%, per-step 11.4% vs ~50%/~30% for the wear-off variant).
    /// New callers should use [`VisualCortex::lsd`] with
    /// `integration ≈ 0.7`, which is the cliff-side of the validated
    /// band and gave the best ID/OOD numbers in the sweep.
    /// `train_dream(...)` is equivalent to `lsd(integration = 1.0,
    /// plasticity_boost = 1.0)` and kept only for reproducibility of
    /// prior published experiments.
    pub fn train_dream(
        &mut self,
        n_samples: usize,
        epochs: usize,
        lr: f32,
        sparsity_k: usize,
        seed: u64,
    ) -> Vec<f32> {
        let bank: Vec<Vec<f32>> = (0..n_samples)
            .map(|i| {
                let s = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(0xD5EA_D5EA)
                    .wrapping_add(i as u64);
                self.dream_pixel(s, sparsity_k)
            })
            .collect();
        let refs: Vec<&[f32]> = bank.iter().map(|v| v.as_slice()).collect();
        self.train_hebbian(&refs, epochs, lr)
    }

    /// Enter an LSD-style cortical state: a temporary plasticity
    /// window driven by stochastic top-down activity, with a
    /// configurable wear-off.
    ///
    /// Mechanism — 5-HT2A agonism reduces the precision of top-down
    /// priors and raises entropy in higher visual cortex
    /// (Carhart-Harris & Friston 2019, REBUS). Stochastic V4 activity
    /// then propagates downward through learned feedback weights
    /// producing structured (not random) hallucinations. The cortex
    /// is simultaneously in a plasticity-enhanced state (Gu et al.
    /// 2023, psilocybin reopens critical-period plasticity in mouse
    /// visual cortex), so it learns from those hallucinations. When
    /// the drug metabolises (~8 hours for LSD, ~4 for psilocybin),
    /// receptors desensitise and the cortex partially integrates the
    /// trip's delta — controlled here by `integration ∈ [0, 1]`:
    ///
    ///   - `0.0` — full wear-off, a clean diagnostic probe (cortex
    ///     returns to pre-trip weights, useful for measuring what the
    ///     trip would have done without committing).
    ///   - `1.0` — full permanent integration (equivalent to the
    ///     legacy `train_dream`, which empirically overfits OOD:
    ///     see experiments in examples/mazes).
    ///   - `0.1..0.5` — biologically realistic: partial integration.
    ///     HPPD (Hallucinogen Persisting Perception Disorder) is the
    ///     failure mode where integration is too high and the cortex
    ///     cannot return to baseline perceptual function.
    ///
    /// Returns an `TripReport` with weight-delta diagnostics so the
    /// caller can verify the trip actually changed something (non-zero
    /// pre-integration delta) and that wear-off actually wore off
    /// (scaled post-integration delta).
    ///
    /// # Validated configurations
    ///
    /// As of the cliff sweep in examples/mazes (size=11, ood=21):
    ///
    ///   - `integration ∈ [0.1, 0.7]`: regularises cleanly — OOD
    ///     first-step 47–50%, per-step 30–39%, gap near zero. Best
    ///     observed: `integration=0.7` at OOD per-step 39.2% /
    ///     ID per-step 40.7% (gap 1.5pp).
    ///   - `integration >= ~0.95`: catastrophic OOD collapse — V2/V4
    ///     weights drift toward the dream's synthetic attractor far
    ///     enough that real-maze input produces near-zero activation.
    ///     Measured at integration=1.0: OOD first-step 2.5%, per-step
    ///     11.4%, gap +20pp. Callers hit this path will receive a
    ///     `log::warn!` at runtime.
    ///   - `integration` in `(0.7, 0.95)` not yet mapped empirically;
    ///     the cliff is somewhere in that range.
    pub fn lsd(&mut self, cfg: LsdConfig) -> TripReport {
        let integration = cfg.integration.clamp(0.0, 1.0);

        // Warn on the known-pathological regime. We leave it callable
        // for research reproducibility (someone reading the earlier
        // commit history may want to reproduce the collapse), but
        // silent acceptance would let naive callers foot-gun themselves.
        if integration >= 0.95 {
            tracing::warn!(
                integration = integration,
                "VisualCortex::lsd called with integration ≥ 0.95 — \
                 this is a known-broken regime (V2/V4 drift to the dream's \
                 synthetic attractor, OOD accuracy collapses). Recommended: \
                 integration ≈ 0.7 for maze-scale. See docstring for the \
                 empirical sweep."
            );
        }

        // Consume receptors — desensitises for future trips.
        // `dose` is top-K channels out of 128; we normalise into a
        // [0,1] dose fraction using 16 channels as "full dose."
        // Above that, more dose doesn't bind more receptors (saturation).
        let dose_fraction = (cfg.dose as f32 / 16.0).min(1.0);
        let potency = self.receptors.apply_dose(dose_fraction);

        // Snapshot the "sober" V2/V4 state. V1 is fixed; no snapshot.
        let v2_before = self.v2.weight.clone();
        let v4_before = self.v4.weight.clone();

        // Elevated plasticity during the trip, scaled by receptor
        // availability. A desensitised cortex receives less trip.
        // Floor at 0.01 of base lr so repeat callers still see *some*
        // movement (matches biology: tolerance ≠ zero-effect).
        let effective_boost = (cfg.plasticity_boost * potency).max(0.01);
        let tripping_lr = cfg.lr * effective_boost;
        let per_epoch_energies = self.train_dream(
            cfg.duration, cfg.epochs, tripping_lr, cfg.dose, cfg.seed,
        );

        // Measure the peak (pre-integration) delta — how far the
        // cortex strayed during the trip.
        let peak_v2_delta = l2_diff(&self.v2.weight, &v2_before);
        let peak_v4_delta = l2_diff(&self.v4.weight, &v4_before);

        // Wear-off: interpolate back toward the sober snapshot.
        //   W_final = integration * W_trip + (1 - integration) * W_sober
        // integration=0 fully reverts; integration=1 keeps the trip.
        if integration < 1.0 {
            let alpha = integration;
            let beta = 1.0 - alpha;
            for (w, &w_pre) in self.v2.weight.iter_mut().zip(&v2_before) {
                *w = alpha * *w + beta * w_pre;
            }
            for (w, &w_pre) in self.v4.weight.iter_mut().zip(&v4_before) {
                *w = alpha * *w + beta * w_pre;
            }
        }

        let post_v2_delta = l2_diff(&self.v2.weight, &v2_before);
        let post_v4_delta = l2_diff(&self.v4.weight, &v4_before);

        TripReport {
            peak_v2_delta,
            peak_v4_delta,
            post_v2_delta,
            post_v4_delta,
            integration,
            per_epoch_energies,
        }
    }
}

fn l2_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
}

/// Configuration for `VisualCortex::lsd`.
///
/// Field mapping to the pharmacological analog:
///   - `dose`        → top-K sparsity of the V4 seed. Higher K = more
///                     active channels per spatial cell = richer
///                     hallucination. Typical: 8–16 of 128.
///   - `duration`    → number of synthesized "dream" images processed
///                     during the trip.
///   - `epochs`      → passes over the dream bank (Hebbian update
///                     rounds).
///   - `lr`          → base Hebbian rate (sober baseline).
///   - `plasticity_boost` → multiplier on `lr` during the trip. The
///                     drug-induced elevation of cortical plasticity.
///                     1.0 = no boost; 2.0–4.0 = biologically modest;
///                     >10.0 = microdosing fantasy.
///   - `integration` → [0, 1]. Fraction of the trip's weight delta
///                     retained after wear-off. See `lsd` docs.
///   - `seed`        → RNG seed — deterministic trip for reproducibility.
#[derive(Debug, Clone, Copy)]
pub struct LsdConfig {
    pub dose: usize,
    pub duration: usize,
    pub epochs: usize,
    pub lr: f32,
    pub plasticity_boost: f32,
    pub integration: f32,
    pub seed: u64,
}

impl Default for LsdConfig {
    /// Validated default: `integration = 0.7`, the best point in
    /// the empirical sweep (OOD per-step 39.2%, ID per-step 40.7%,
    /// gap 1.5pp on maze-scale). Dose and duration match the
    /// configuration of the sweep so `VisualCortex::lsd(cfg)` with
    /// `LsdConfig::default()` reproduces the validated regime.
    fn default() -> Self {
        Self {
            dose: 8,
            duration: 500,
            epochs: 2,
            lr: 2e-4,
            plasticity_boost: 1.0,
            integration: 0.7,
            seed: 0,
        }
    }
}

impl VisualCortex {
    /// Advance pharmacological time: recover 5-HT2A receptor
    /// sensitivity toward baseline. Callers invoke this between
    /// trips to model the hours-to-days receptor recovery of real
    /// psychedelic pharmacology. For a typical training loop, one
    /// `tick_receptors` per gradient step with rate ~1e-3 gives
    /// ~thousand-step recovery to ≥95% availability.
    pub fn tick_receptors(&mut self, recovery_rate: f32) {
        self.receptors.tick(recovery_rate);
    }
}

/// Diagnostic report from `VisualCortex::lsd`.
///
/// `peak_*_delta` is the L2 distance between the trip's peak weights
/// and the pre-trip snapshot — how far the cortex moved during the
/// drug's action. `post_*_delta` is the same measurement after
/// wear-off — how far the cortex settled from baseline once the drug
/// metabolised. For a well-posed trip, `peak > post` and the ratio
/// approximates the `integration` parameter.
#[derive(Debug, Clone)]
pub struct TripReport {
    pub peak_v2_delta: f32,
    pub peak_v4_delta: f32,
    pub post_v2_delta: f32,
    pub post_v4_delta: f32,
    pub integration: f32,
    pub per_epoch_energies: Vec<f32>,
}

impl VisualCortex {
    /// Output spatial feature tokens for attention-based processing.
    ///
    /// Instead of global avg pool → single vector, this returns the V4 feature
    /// map as a sequence of spatial tokens: each (x,y) position becomes one token.
    ///
    /// Returns (tokens, n_tokens, token_dim) where:
    ///   tokens: flattened [n_tokens × token_dim]
    ///   n_tokens: h4 × w4 (spatial positions in V4 output)
    ///   token_dim: V4 out_channels (128)
    ///
    /// This is the equivalent of ResNet backbone → flatten(2) → transpose(1,2)
    /// in the Python CTM. The CTM attends over these spatial tokens.
    pub fn spatial_tokens(&self, raw: &[f32]) -> (Vec<f32>, usize, usize) {
        let h = self.input_h;
        let w = self.input_w;

        let (mut r_out, rh, rw) = self.retina.forward(raw, 1, h, w);
        leaky_relu(&mut r_out);

        let (mut v1_out, h1, w1) = self.v1.forward(&r_out, 1, rh, rw);
        leaky_relu(&mut v1_out);

        let (mut v2_out, h2, w2) = self.v2.forward(&v1_out, 1, h1, w1);
        leaky_relu(&mut v2_out);

        let (mut v4_out, h4, w4) = self.v4.forward(&v2_out, 1, h2, w2);
        leaky_relu(&mut v4_out);

        let channels = self.v4.out_channels;
        let n_tokens = h4 * w4;

        // CHW → tokens [n_tokens × channels]
        let mut tokens = vec![0.0f32; n_tokens * channels];
        for y in 0..h4 {
            for x in 0..w4 {
                let token_idx = y * w4 + x;
                for c in 0..channels {
                    tokens[token_idx * channels + c] =
                        v4_out[c * h4 * w4 + y * w4 + x];
                }
            }
        }

        (tokens, n_tokens, channels)
    }

    /// Multi-scale forward — emits V1, V2, V4 spatial tokens as
    /// independent streams plus the raw retinal-ganglion (DoG +
    /// color-opponent) output. Mirrors biology where V1 has direct
    /// projections to many areas, AND a subcortical fast path
    /// (superior colliculus → pulvinar → amygdala/insula) routes
    /// raw salience signal to limbic regions before cortical V1
    /// even finishes processing.
    ///
    /// Returns four (tokens, n_tokens, channels) tuples in order:
    /// V1, V2, V4, GANGLION. Token layout for each:
    /// `[n_tokens × channels]` row-major.
    pub fn spatial_tokens_multiscale(
        &self,
        raw: &[f32],
    ) -> [(Vec<f32>, usize, usize); 4] {
        let h = self.input_h;
        let w = self.input_w;

        let (mut r_out, rh, rw) = self.retina.forward(raw, 1, h, w);
        leaky_relu(&mut r_out);
        // Snapshot ganglion output for subcortical fast path before
        // cortex processes it further.
        let r_snapshot = r_out.clone();

        let (mut v1_out, h1, w1) = self.v1.forward(&r_out, 1, rh, rw);
        leaky_relu(&mut v1_out);

        let (mut v2_out, h2, w2) = self.v2.forward(&v1_out, 1, h1, w1);
        leaky_relu(&mut v2_out);

        let (mut v4_out, h4, w4) = self.v4.forward(&v2_out, 1, h2, w2);
        leaky_relu(&mut v4_out);

        let chw_to_tokens = |chw: &[f32], ch: usize, hh: usize, ww: usize| {
            let n = hh * ww;
            let mut tok = vec![0.0f32; n * ch];
            for y in 0..hh {
                for x in 0..ww {
                    let ti = y * ww + x;
                    for c in 0..ch {
                        tok[ti * ch + c] = chw[c * n + y * ww + x];
                    }
                }
            }
            (tok, n, ch)
        };

        [
            chw_to_tokens(&v1_out, self.v1.out_channels, h1, w1),
            chw_to_tokens(&v2_out, self.v2.out_channels, h2, w2),
            chw_to_tokens(&v4_out, self.v4.out_channels, h4, w4),
            chw_to_tokens(&r_snapshot, self.retina.out_channels, rh, rw),
        ]
    }

    /// V4-CTM summary path: runs `spatial_tokens`, then feeds the
    /// result through the V4-CTM stage, emitting a single
    /// `[d_model]` summary vector.
    ///
    /// Opt-in (not used by default encoding) — the brain currently
    /// expects spatial tokens. When V4-CTM gains proper backward
    /// (for joint end-to-end training), callers can switch to this
    /// path and the brain's attention degenerates to one-token.
    pub fn spatial_tokens_v4ctm_summary(&self, raw: &[f32])
        -> (Vec<f32>, usize, usize) {
        let (v4_tokens, n_spatial, channels) = self.spatial_tokens(raw);
        let sync = self.v4_ctm.forward(&v4_tokens, n_spatial, channels);
        let d_model = self.v4_ctm.d_model();
        (sync, 1, d_model)
    }

    /// Spatial dims of each stage's output. Returns
    /// `(rh, rw, h1, w1, h2, w2, h4, w4)` — retina + 3 cortical stages.
    /// Useful for shape plumbing (dream-pixel seeding, receptive-field
    /// visualization) without running a probe forward.
    pub fn stage_dims(&self) -> (usize, usize, usize, usize, usize, usize, usize, usize) {
        let (h, w) = (self.input_h, self.input_w);
        let out_dim = |in_h: usize, k: usize, s: usize, p: usize| (in_h + 2 * p - k) / s + 1;
        let (kr, sr, pr) = (self.retina.kernel_size, self.retina.stride, self.retina.padding);
        let (k1, s1, p1) = (self.v1.kernel_size, self.v1.stride, self.v1.padding);
        let (k2, s2, p2) = (self.v2.kernel_size, self.v2.stride, self.v2.padding);
        let (k4, s4, p4) = (self.v4.kernel_size, self.v4.stride, self.v4.padding);
        let rh = out_dim(h, kr, sr, pr);
        let rw = out_dim(w, kr, sr, pr);
        let h1 = out_dim(rh, k1, s1, p1);
        let w1 = out_dim(rw, k1, s1, p1);
        let h2 = out_dim(h1, k2, s2, p2);
        let w2 = out_dim(w1, k2, s2, p2);
        let h4 = out_dim(h2, k4, s4, p4);
        let w4 = out_dim(w2, k4, s4, p4);
        (rh, rw, h1, w1, h2, w2, h4, w4)
    }

    /// Dream-pixel synthesis (Hoel 2021).
    ///
    /// Seeds V4 with sparse noise, projects top-down through the
    /// learned cortex via the transpose operator of each conv layer,
    /// and returns a synthesized `[3 × input_h × input_w]` pixel
    /// tensor. Output is the adjoint image of the seed — sparse,
    /// hallucinatory, and statistically coherent with the model's
    /// learned V2/V4 weights, which is exactly the "corrupted,
    /// top-down generated sensory input" the overfitted-brain
    /// hypothesis posits as the regularization signal of sleep.
    ///
    /// `sparsity_k`: number of active (non-zero) V4 channels per
    /// spatial position — the Hoel-style dropout/sparsity factor.
    /// Typical: 8 of 128 for maze-scale retinas.
    ///
    /// `seed`: RNG seed so the same seed produces the same dream.
    pub fn dream_pixel(&self, seed: u64, sparsity_k: usize) -> Vec<f32> {
        let (rh, rw, h1, w1, h2, w2, h4, w4) = self.stage_dims();

        // 1. Seed V4 with sparse Gaussian noise. K winners per spatial
        //    cell, rest zero — matches V2/V4's Hebbian sparse code.
        let ch4 = self.v4.out_channels;
        let mut v4_seed = vec![0.0f32; ch4 * h4 * w4];
        let mut rng = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        for y in 0..h4 {
            for x in 0..w4 {
                let spatial = y * w4 + x;
                // draw ch4 noise samples
                let mut vals: Vec<(usize, f32)> = (0..ch4)
                    .map(|c| {
                        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        let u = ((rng >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0;
                        (c, u)
                    })
                    .collect();
                // keep top-K by magnitude
                vals.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
                for &(c, v) in vals.iter().take(sparsity_k) {
                    v4_seed[c * h4 * w4 + spatial] = v;
                }
            }
        }

        // 2. V4 adjoint: [128 × h4 × w4] → [64 × h2 × w2]
        let v2_grid = self.v4.transpose_forward(&v4_seed, 1, h4, w4, h2, w2);
        // ReLU-ish in reverse: nonlinearity was forward leaky_relu;
        // for synthesis, we keep the linear adjoint as-is — adding a
        // nonlinearity here would break the adjoint identity and
        // invent distribution-shifted outputs.

        // 3. V2 adjoint: [64 × h2 × w2] → [32 × h1 × w1]
        let v1_grid = self.v2.transpose_forward(&v2_grid, 1, h2, w2, h1, w1);

        // 4. V1 adjoint: [32 × h1 × w1] → [12 × rh × rw]
        let retina_grid = self.v1.transpose_forward(
            &v1_grid, 1, h1, w1, rh, rw,
        );

        // 5. Retina adjoint: [12 × rh × rw] → [3 × input_h × input_w]
        let pixel_grid = self.retina.transpose_forward(
            &retina_grid, 1, rh, rw, self.input_h, self.input_w,
        );

        pixel_grid
    }
}

impl VisualCortex {
    /// Encode with a per-token spatial gain mask applied *after* the
    /// V4 forward. `gain.len()` must equal `n_tokens` (the `spatial_tokens`
    /// output count); each token vector is scaled by its corresponding
    /// gain scalar.
    ///
    /// Use case: top-down attention from a downstream region (e.g. the
    /// brain CTM's `attention` area) driving which spatial locations
    /// the cortex emphasizes. The gain is applied post-V4 — the conv
    /// stack runs normally; only the tokens the brain sees are scaled.
    ///
    /// Gradient path is clean: upstream gradient on tokens splits into
    /// (a) `d_tokens · (1 - gain)` back to the raw v4 output → conv
    /// backward, and (b) `d_tokens · v4_output` back to the gain
    /// itself → caller's top-down projection weights.
    pub fn encode_gated(&self, raw: &[f32], gain: &[f32]) -> modgrad_traits::TokenInput {
        let (mut tokens, n_tokens, token_dim) = self.spatial_tokens(raw);
        assert_eq!(gain.len(), n_tokens,
            "encode_gated: gain len {} ≠ n_tokens {}", gain.len(), n_tokens);
        for t in 0..n_tokens {
            let g = gain[t];
            let base = t * token_dim;
            for c in 0..token_dim {
                tokens[base + c] *= g;
            }
        }
        modgrad_traits::TokenInput { tokens, n_tokens, token_dim }
    }

    /// Multi-stage top-down modulation: applies a per-spatial-position
    /// `gain` mask after V1, after V2, and after V4. Mirrors the
    /// biological back-projections from V4 → V2 → V1 that let frontal
    /// attention areas pre-emphasize spatial locations early in the
    /// visual stream rather than only post-V4.
    ///
    /// The gain mask is the same across stages because our
    /// `preserve_spatial` encoder keeps spatial dims constant. For
    /// stride-2 encoders, callers would need per-stage masks of
    /// matching shape.
    ///
    /// Returns a `MultiScaleTokens` bundle so a multi-scale brain
    /// (different regions consuming V1/V2/V4) sees the same
    /// attention modulation applied at the level it consumes.
    pub fn encode_gated_multistage(
        &self,
        raw: &[f32],
        gain: &[f32],
    ) -> modgrad_traits::MultiScaleTokens {
        let h = self.input_h;
        let w = self.input_w;

        let (mut r_out, rh, rw) = self.retina.forward(raw, 1, h, w);
        leaky_relu(&mut r_out);
        // Subcortical fast path: snapshot raw ganglion output BEFORE
        // top-down modulation. Biological subcortical bypass is
        // independent of cortical attention by design.
        let r_snapshot = r_out.clone();

        let (mut v1_out, h1, w1) = self.v1.forward(&r_out, 1, rh, rw);
        leaky_relu(&mut v1_out);
        apply_spatial_gain_chw(&mut v1_out, gain, self.v1.out_channels, h1, w1);

        let (mut v2_out, h2, w2) = self.v2.forward(&v1_out, 1, h1, w1);
        leaky_relu(&mut v2_out);
        apply_spatial_gain_chw(&mut v2_out, gain, self.v2.out_channels, h2, w2);

        let (mut v4_out, h4, w4) = self.v4.forward(&v2_out, 1, h2, w2);
        leaky_relu(&mut v4_out);
        apply_spatial_gain_chw(&mut v4_out, gain, self.v4.out_channels, h4, w4);

        let chw_to_tokens = |chw: &[f32], ch: usize, hh: usize, ww: usize| {
            let n = hh * ww;
            let mut tok = vec![0.0f32; n * ch];
            for y in 0..hh {
                for x in 0..ww {
                    let ti = y * ww + x;
                    for c in 0..ch {
                        tok[ti * ch + c] = chw[c * n + y * ww + x];
                    }
                }
            }
            modgrad_traits::TokenInput {
                n_tokens: n, token_dim: ch, tokens: tok,
            }
        };

        modgrad_traits::MultiScaleTokens {
            scales: vec![
                chw_to_tokens(&v4_out, self.v4.out_channels, h4, w4),
                chw_to_tokens(&v2_out, self.v2.out_channels, h2, w2),
                chw_to_tokens(&v1_out, self.v1.out_channels, h1, w1),
                chw_to_tokens(&r_snapshot, self.retina.out_channels, rh, rw),
            ],
        }
    }
}

/// In-place per-spatial-position multiplicative gain on a CHW tensor.
/// `gain` length must equal `h*w`; same scalar is applied to every
/// channel at that position.
fn apply_spatial_gain_chw(chw: &mut [f32], gain: &[f32], ch: usize, h: usize, w: usize) {
    let n = h * w;
    debug_assert_eq!(gain.len(), n,
        "gain len {} ≠ h*w ({}*{}={})", gain.len(), h, w, n);
    debug_assert_eq!(chw.len(), ch * n,
        "chw len {} ≠ ch*h*w ({}*{}*{}={})", chw.len(), ch, h, w, ch * n);
    for c in 0..ch {
        for p in 0..n {
            chw[c * n + p] *= gain[p];
        }
    }
}

impl modgrad_traits::Encoder for VisualCortex {
    type Raw = [f32]; // pixels: [3 × H × W] flattened

    fn encode(&self, raw: &[f32]) -> modgrad_traits::TokenInput {
        let (tokens, n_tokens, token_dim) = self.spatial_tokens(raw);
        modgrad_traits::TokenInput { tokens, n_tokens, token_dim }
    }

    fn token_dim(&self) -> usize { self.v4.out_channels }

    /// Multi-scale tap points in cortical→subcortical order.
    /// scale 0: V4 (high-level semantic, "primary")
    /// scale 1: V2 (mid contour)
    /// scale 2: V1 (fine cortical edges)
    ///
    /// Subcortical fast path (raw ganglion) is available via
    /// `spatial_tokens_multiscale` directly but not exposed as a
    /// trait scale until `Genome` declares an amygdala-style
    /// consumer for it (INSULA wiring hurt eval per A/B).
    fn encode_multiscale(&self, raw: &[f32]) -> modgrad_traits::MultiScaleTokens {
        let [v1, v2, v4, _gang] = self.spatial_tokens_multiscale(raw);
        let mk = |(tokens, n_tokens, token_dim): (Vec<f32>, usize, usize)|
            modgrad_traits::TokenInput { tokens, n_tokens, token_dim };
        modgrad_traits::MultiScaleTokens { scales: vec![mk(v4), mk(v2), mk(v1)] }
    }

    fn token_dims(&self) -> Vec<usize> {
        vec![
            self.v4.out_channels,
            self.v2.out_channels,
            self.v1.out_channels,
        ]
    }
}

impl Retina for VisualCortex {
    fn observe(&mut self, raw: &[f32]) -> Vec<f32> {
        // Input: [3 × h × w] flattened
        let h = self.input_h;
        let w = self.input_w;

        // Retina (fixed DoG + color opponents)
        let (mut r_out, rh, rw) = self.retina.forward(raw, 1, h, w);
        leaky_relu(&mut r_out);

        // V1 cortex (learnable): edges / orientations from retinal features
        let (mut v1_out, h1, w1) = self.v1.forward(&r_out, 1, rh, rw);
        leaky_relu(&mut v1_out);

        // V2 cortex: contours from V1
        let (mut v2_out, h2, w2) = self.v2.forward(&v1_out, 1, h1, w1);
        leaky_relu(&mut v2_out);

        // V4 cortex: shapes from V2
        let (mut v4_out, h4, w4) = self.v4.forward(&v2_out, 1, h2, w2);
        leaky_relu(&mut v4_out);

        // Multi-scale pooling over V1 and V4 (fine + coarse).
        let v1_pool = adaptive_avg_pool(&v1_out, 32, h1, w1);
        let _v2_pool = adaptive_avg_pool(&v2_out, 64, h2, w2);
        let v4_pool = adaptive_avg_pool(&v4_out, 128, h4, w4);
        let mut out = v4_pool;
        for i in 0..32.min(out.len()) {
            out[i] = 0.7 * out[i] + 0.3 * v1_pool[i];
        }
        out
    }

    fn d_output(&self) -> usize { self.pool_dim }

    fn param_count(&self) -> usize {
        self.retina.param_count() + self.v1.param_count()
            + self.v2.param_count() + self.v4.param_count()
    }
}

// ─── Auditory Retina ─────────────────────────────────────────

/// Auditory retina: converts raw audio waveform to features.
/// Implements cochlea → A1 → A2 hierarchy.
///
/// Cochlea: frequency decomposition (like mel filterbank)
/// A1: temporal patterns (1D conv on frequency bands)
/// A2: abstract audio features (1D conv)
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub struct AuditoryRetina {
    pub cochlea: Conv1d,  // raw waveform → frequency bands
    pub a1: Conv1d,       // frequency → temporal patterns
    pub a2: Conv1d,       // patterns → features
    pub d_out: usize,
}

impl AuditoryRetina {
    /// Create auditory retina for audio at given sample rate.
    /// window_size: samples per analysis window (e.g., 400 for 25ms at 16kHz)
    pub fn new(d_output: usize) -> Self {
        // Cochlea: 1 channel → 64 frequency bands, kernel=64 (4ms at 16kHz), stride=32
        let cochlea = Conv1d::new(1, 64, 64, 32);
        // A1: 64 bands → 128 temporal patterns, kernel=8, stride=4
        let a1 = Conv1d::new(64, 128, 8, 4);
        // A2: 128 → d_output, kernel=4, stride=2
        let a2 = Conv1d::new(128, d_output, 4, 2);
        Self { cochlea, a1, a2, d_out: d_output }
    }
}

impl Retina for AuditoryRetina {
    fn observe(&mut self, raw: &[f32]) -> Vec<f32> {
        let len = raw.len();

        // Cochlea: waveform → frequency bands
        let (mut c_out, c_len) = self.cochlea.forward(raw, len);
        leaky_relu(&mut c_out);

        // A1: frequency → temporal patterns
        let (mut a1_out, a1_len) = self.a1.forward(&c_out, c_len);
        leaky_relu(&mut a1_out);

        // A2: patterns → features
        let (mut a2_out, a2_len) = self.a2.forward(&a1_out, a1_len);
        leaky_relu(&mut a2_out);

        // Global average pooling over time
        let mut out = vec![0.0f32; self.d_out];
        if a2_len > 0 {
            for c in 0..self.d_out {
                let sum: f32 = (0..a2_len).map(|t| a2_out[c * a2_len + t]).sum();
                out[c] = sum / a2_len as f32;
            }
        }
        out
    }

    fn d_output(&self) -> usize { self.d_out }

    fn param_count(&self) -> usize {
        self.cochlea.param_count() + self.a1.param_count() + self.a2.param_count()
    }
}

// ─── Text Retina ─────────────────────────────────────────────

/// Text retina: converts byte stream to observation vectors.
/// This is our existing embedding table — just wrapped as a Retina.
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub struct TextRetina {
    pub embeddings: Vec<f32>,  // [vocab_size × d_output]
    pub vocab_size: usize,
    pub d_out: usize,
}

impl TextRetina {
    pub fn new(vocab_size: usize, d_output: usize) -> Self {
        let scale = (1.0 / d_output as f32).sqrt();
        let mut rng_state = (vocab_size * d_output) as u64;
        let embeddings: Vec<f32> = (0..vocab_size * d_output)
            .map(|_| simple_rng_normal(&mut rng_state) * scale)
            .collect();
        Self { embeddings, vocab_size, d_out: d_output }
    }
}

impl Retina for TextRetina {
    fn observe(&mut self, raw: &[f32]) -> Vec<f32> {
        // raw[0] is the byte value (cast to f32)
        let byte_id = (raw[0] as usize).min(self.vocab_size - 1);
        let off = byte_id * self.d_out;
        self.embeddings[off..off + self.d_out].to_vec()
    }

    fn d_output(&self) -> usize { self.d_out }

    fn param_count(&self) -> usize { self.embeddings.len() }
}

// ─── Proprioceptive Retina ────────────────────────────────
//
// The organism's sense of its own body. Like the insular cortex:
// discovers hardware sensors at birth, reads them each tick as raw floats.
// No labels, no interpretation — just numbers the organism learns to use.
//
// Scans at init:
//   /sys/class/hwmon/*/  — temps, fans, power, voltage, frequency
//   /proc/stat           — CPU utilization (jiffies)
//   /proc/meminfo        — memory pressure
//   /proc/loadavg        — system load
//
// Each sensor becomes a slot in the raw observation vector.
// A random projection maps N sensors → d_output dimensions.
// The organism learns what these numbers mean through correlation
// with its own performance (surprise, sleep pressure, etc.)

/// A discovered hardware sensor.
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub struct Sensor {
    /// File path to read (e.g. "/sys/class/hwmon/hwmon2/temp1_input").
    pub path: String,
    /// Running exponential moving average (for normalization).
    pub ema: f32,
    /// Running variance estimate (for normalization).
    pub var: f32,
    /// How many times this sensor has been read.
    pub n: u32,
}

impl Sensor {
    fn new(path: String) -> Self {
        Self { path, ema: 0.0, var: 1.0, n: 0 }
    }

    /// Read current value. Returns None if unreadable.
    fn read(&self) -> Option<f32> {
        std::fs::read_to_string(&self.path).ok()
            .and_then(|s| s.trim().parse::<f64>().ok())
            .map(|v| v as f32)
    }

    /// Read and normalize to approximately zero-mean unit-variance.
    /// Uses online Welford's algorithm so no precomputation needed.
    fn read_normalized(&mut self) -> f32 {
        let raw = self.read().unwrap_or(self.ema);
        self.n += 1;
        let alpha = if self.n < 100 { 1.0 / self.n as f32 } else { 0.01 };
        let delta = raw - self.ema;
        self.ema += alpha * delta;
        self.var = (1.0 - alpha) * self.var + alpha * delta * (raw - self.ema);
        let std = self.var.sqrt().max(1e-6);
        (raw - self.ema) / std
    }
}

/// Proprioceptive retina: senses the organism's own hardware.
///
/// At birth, scans the system for readable sensors (temperature, fan speed,
/// memory, CPU load, etc.). Each tick, reads all sensors as raw floats,
/// normalizes them online, and projects through a random matrix to d_output
/// dimensions. The organism learns what they mean.
///
/// This is analogous to interoception: the brain doesn't know what a
/// thermoreceptor IS, it just learns that when signal X rises, performance
/// drops and sleep pressure increases.

/// Per-region compute metrics injected by the harness.
/// Each region feels its own processing state independently.
#[derive(Debug, Clone, Default)]
pub struct RegionMetrics {
    /// Matmul latency for this region (0.0 = instant, 1.0 = very slow).
    pub latency: f32,
    /// Core utilization for this region's core group (0.0 = idle, 1.0 = saturated).
    pub utilization: f32,
    /// Spike input queue pressure (0.0 = empty, 1.0 = dropping spikes).
    pub queue_pressure: f32,
    /// Whether this region is on GPU (0.0 = CPU, 1.0 = GPU).
    pub on_gpu: f32,
}

/// Compute interoception: the organism's sense of its own processing.
/// Not a fixed size — scales with the number of regions.
/// The harness fills this per forward pass from runtime metrics.
#[derive(Debug, Clone, Default)]
pub struct ComputeInteroception {
    /// Per-region metrics. Length = number of regions (dynamic).
    pub regions: Vec<RegionMetrics>,
    /// Global sync coherence across all regions (0.0–1.0).
    pub coherence: f32,
    /// Overall deliberation load (fraction of max gamma cycles used).
    pub deliberation_load: f32,
    /// Overall memory pressure from compute buffers (0.0–1.0).
    pub memory_pressure: f32,
}

impl ComputeInteroception {
    /// Flatten to a f32 vector for projection through the retina.
    /// Layout: [region_0_latency, region_0_util, region_0_queue, region_0_gpu,
    ///          region_1_latency, ..., coherence, deliberation_load, memory_pressure]
    pub fn to_vec(&self) -> Vec<f32> {
        let mut v = Vec::with_capacity(self.regions.len() * 4 + 3);
        for r in &self.regions {
            v.push(r.latency);
            v.push(r.utilization);
            v.push(r.queue_pressure);
            v.push(r.on_gpu);
        }
        v.push(self.coherence);
        v.push(self.deliberation_load);
        v.push(self.memory_pressure);
        v
    }

    /// Number of sensor channels.
    pub fn n_channels(&self) -> usize {
        self.regions.len() * 4 + 3
    }
}

#[derive(Debug, Clone)]
pub struct ProprioceptiveRetina {
    /// Discovered sensors (file paths + running stats).
    pub sensors: Vec<Sensor>,
    /// Random projection: [d_output × n_sensors].
    /// Fixed at birth (genome-tier). The organism can't choose what to attend to,
    /// it can only learn which projected dimensions correlate with what.
    projection: Vec<f32>,
    /// Synthetic sensors computed from /proc (CPU%, mem%, load).
    /// These are appended to the hwmon sensors.
    n_proc_sensors: usize,
    /// Compute interoception sensors injected by harness.
    n_compute_sensors: usize,
    d_out: usize,
    /// Previous /proc/stat CPU jiffies (for computing delta = utilization).
    prev_cpu_total: u64,
    prev_cpu_idle: u64,
    /// Latest compute metrics from harness. Updated each forward pass.
    pub compute: ComputeInteroception,
}

impl ProprioceptiveRetina {
    /// Discover all readable sensors on this system.
    pub fn new(d_output: usize) -> Self {
        let mut sensors = Vec::new();

        // Scan /sys/class/hwmon/
        if let Ok(entries) = std::fs::read_dir("/sys/class/hwmon") {
            let mut dirs: Vec<_> = entries.filter_map(|e| e.ok()).collect();
            dirs.sort_by_key(|e| e.file_name());
            for entry in dirs {
                let dir = entry.path();
                if let Ok(files) = std::fs::read_dir(&dir) {
                    let mut sensor_files: Vec<_> = files
                        .filter_map(|f| f.ok())
                        .filter(|f| {
                            let name = f.file_name().to_string_lossy().to_string();
                            name.ends_with("_input")
                        })
                        .collect();
                    sensor_files.sort_by_key(|f| f.file_name());
                    for sf in sensor_files {
                        let path = sf.path().to_string_lossy().to_string();
                        // Verify readable
                        if Sensor::new(path.clone()).read().is_some() {
                            sensors.push(Sensor::new(path));
                        }
                    }
                }
            }
        }

        // /proc synthetic sensors: CPU%, mem%, load1, load5, load15
        let n_proc_sensors = 5;
        // Compute interoception: dynamic, depends on region count.
        // At boot, assume 0 compute sensors. The projection matrix will be
        // rebuilt on first observe() if compute channels are injected.
        let n_compute_sensors = 0;
        let n_total = sensors.len() + n_proc_sensors + n_compute_sensors;

        // Random projection matrix (fixed at birth)
        let mut rng_state = (n_total * d_output + 7919) as u64;
        let scale = (1.0 / n_total as f32).sqrt();
        let projection: Vec<f32> = (0..d_output * n_total)
            .map(|_| simple_rng_normal(&mut rng_state) * scale)
            .collect();

        eprintln!("[proprioception] discovered {} hwmon sensors + {} proc sensors = {} total → {} dim",
            sensors.len(), n_proc_sensors, n_total, d_output);
        for s in &sensors {
            if let Some(v) = s.read() {
                eprintln!("  {}: {v}", s.path);
            }
        }

        Self {
            sensors,
            projection,
            n_proc_sensors,
            n_compute_sensors,
            d_out: d_output,
            prev_cpu_total: 0,
            prev_cpu_idle: 0,
            compute: ComputeInteroception::default(),
        }
    }

    /// Read /proc/stat CPU utilization as a 0.0-1.0 fraction.
    fn read_cpu_util(&mut self) -> f32 {
        let line = match std::fs::read_to_string("/proc/stat") {
            Ok(s) => s,
            Err(_) => return 0.5,
        };
        let first = match line.lines().next() {
            Some(l) => l,
            None => return 0.5,
        };
        // cpu  user nice system idle iowait irq softirq steal guest guest_nice
        let vals: Vec<u64> = first.split_whitespace()
            .skip(1) // skip "cpu"
            .filter_map(|s| s.parse().ok())
            .collect();
        if vals.len() < 4 { return 0.5; }
        let total: u64 = vals.iter().sum();
        let idle = vals[3]; // 4th field is idle

        let dtotal = total.saturating_sub(self.prev_cpu_total).max(1);
        let didle = idle.saturating_sub(self.prev_cpu_idle);
        self.prev_cpu_total = total;
        self.prev_cpu_idle = idle;

        1.0 - (didle as f32 / dtotal as f32)
    }

    /// Read memory pressure as fraction used (0.0 = all free, 1.0 = all used).
    fn read_mem_pressure() -> f32 {
        let info = match std::fs::read_to_string("/proc/meminfo") {
            Ok(s) => s,
            Err(_) => return 0.5,
        };
        let mut total = 0u64;
        let mut avail = 0u64;
        for line in info.lines() {
            if line.starts_with("MemTotal:") {
                total = line.split_whitespace().nth(1)
                    .and_then(|s| s.parse().ok()).unwrap_or(1);
            } else if line.starts_with("MemAvailable:") {
                avail = line.split_whitespace().nth(1)
                    .and_then(|s| s.parse().ok()).unwrap_or(0);
            }
        }
        if total == 0 { return 0.5; }
        1.0 - (avail as f32 / total as f32)
    }

    /// Read load averages (1, 5, 15 min).
    fn read_loadavg() -> (f32, f32, f32) {
        let s = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
        let mut parts = s.split_whitespace();
        let l1 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
        let l5 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
        let l15 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
        (l1, l5, l15)
    }

    /// Read all sensors into a raw vector.
    fn read_all(&mut self) -> Vec<f32> {
        // Check if compute channels changed (first inject from harness)
        let current_compute = self.compute.n_channels();
        if current_compute != self.n_compute_sensors {
            self.n_compute_sensors = current_compute;
            let n_total = self.sensors.len() + self.n_proc_sensors + current_compute;
            // Rebuild projection matrix with new total sensor count
            let mut rng_state = (n_total * self.d_out + 7919) as u64;
            let scale = (1.0 / n_total as f32).sqrt();
            self.projection = (0..self.d_out * n_total)
                .map(|_| simple_rng_normal(&mut rng_state) * scale)
                .collect();
        }

        let mut raw = Vec::with_capacity(
            self.sensors.len() + self.n_proc_sensors + self.n_compute_sensors);

        // Hardware sensors (normalized online)
        for sensor in &mut self.sensors {
            raw.push(sensor.read_normalized());
        }

        // Synthetic /proc sensors (already 0-1 or small positive)
        let cpu = self.read_cpu_util();
        let mem = Self::read_mem_pressure();
        let (l1, l5, l15) = Self::read_loadavg();
        // Normalize load by CPU count for ~0-1 range
        let ncpu = rayon::current_num_threads() as f32;
        raw.push(cpu * 2.0 - 1.0);           // center around 0
        raw.push(mem * 2.0 - 1.0);           // center around 0
        raw.push(l1 / ncpu * 2.0 - 1.0);
        raw.push(l5 / ncpu * 2.0 - 1.0);
        raw.push(l15 / ncpu * 2.0 - 1.0);

        // Compute interoception (injected by harness, centered around 0)
        // Per-region: each region feels its own latency, utilization, queue, GPU status
        // Plus global: coherence, deliberation load, memory pressure
        for v in self.compute.to_vec() {
            raw.push(v * 2.0 - 1.0);
        }

        raw
    }
}

impl Retina for ProprioceptiveRetina {
    /// Observe the organism's own body.
    /// Input `raw` is ignored — proprioception reads from the OS directly.
    fn observe(&mut self, _raw: &[f32]) -> Vec<f32> {
        let sensor_vals = self.read_all();
        let n = sensor_vals.len();

        // Project through random matrix: out[i] = sum_j projection[i*n + j] * sensor[j]
        let mut out = vec![0.0f32; self.d_out];
        for i in 0..self.d_out {
            let row_start = i * n;
            let mut sum = 0.0f32;
            for j in 0..n.min(self.projection.len().saturating_sub(row_start)) {
                sum += self.projection[row_start + j] * sensor_vals[j];
            }
            out[i] = sum;
        }
        out
    }

    fn d_output(&self) -> usize { self.d_out }

    fn param_count(&self) -> usize { self.projection.len() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    fn fill_rand(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// The adjoint identity: for any x, y of matching shapes,
    ///   <forward(x), y> == <x, transpose_forward(y)>
    /// This is *the* correctness test for a transpose operator — it
    /// holds iff the two operations are true adjoints of each other
    /// under the Euclidean inner product. Failing this means the
    /// Hoel dream-pixel synthesis is not inverting what encode did.
    #[test]
    fn transpose_is_forward_adjoint_stride1() {
        let conv = Conv2d::new(3, 5, 3, 1, 1);
        let (in_h, in_w) = (7usize, 7usize);
        let (out_h, out_w) = (7usize, 7usize); // stride 1, same padding

        let x = fill_rand(3 * in_h * in_w, 42);
        let y = fill_rand(5 * out_h * out_w, 1337);

        let fx = {
            let (out, h, w) = conv.forward(&x, 1, in_h, in_w);
            assert_eq!((h, w), (out_h, out_w));
            // forward() adds bias — subtract it to isolate the linear op
            let mut out = out;
            for oc in 0..conv.out_channels {
                for hw in 0..out_h * out_w {
                    out[oc * out_h * out_w + hw] -= conv.bias[oc];
                }
            }
            out
        };
        let ty = conv.transpose_forward(&y, 1, out_h, out_w, in_h, in_w);

        let lhs = dot(&fx, &y);
        let rhs = dot(&x, &ty);
        let rel = (lhs - rhs).abs() / lhs.abs().max(rhs.abs()).max(1e-6);
        assert!(rel < 1e-5, "adjoint identity failed: <Ax,y>={lhs} <x,A*y>={rhs} rel={rel}");
    }

    #[test]
    fn lsd_integration_zero_fully_reverts() {
        let mut retina = VisualCortex::preserve_spatial(11, 11);
        let v4_sober: Vec<f32> = retina.v4.weight.clone();
        let v2_sober: Vec<f32> = retina.v2.weight.clone();

        let report = retina.lsd(LsdConfig {
            dose: 8, duration: 50, epochs: 1, lr: 1e-4,
            plasticity_boost: 2.0, integration: 0.0, seed: 42,
        });

        // The trip did something — peak deltas must be non-zero, else
        // we aren't testing wear-off, we're testing a no-op.
        assert!(report.peak_v4_delta > 1e-4,
            "trip didn't move V4: peak_v4_delta={}", report.peak_v4_delta);

        // Post-integration delta must be ~0 — cortex fully returned
        // to the sober snapshot. Float roundoff tolerance only.
        assert!(report.post_v4_delta < 1e-5,
            "integration=0 left V4 drifted: post_v4_delta={}", report.post_v4_delta);
        assert!(report.post_v2_delta < 1e-5,
            "integration=0 left V2 drifted: post_v2_delta={}", report.post_v2_delta);

        // Weights themselves match the snapshot bit-for-bit under tolerance.
        let max_v4_diff: f32 = retina.v4.weight.iter().zip(&v4_sober)
            .map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        let max_v2_diff: f32 = retina.v2.weight.iter().zip(&v2_sober)
            .map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        assert!(max_v4_diff < 1e-6, "V4 max residual = {max_v4_diff}");
        assert!(max_v2_diff < 1e-6, "V2 max residual = {max_v2_diff}");
    }

    #[test]
    fn lsd_tachyphylaxis_second_trip_smaller() {
        // The biological invariant: without recovery between trips,
        // the second trip must move the cortex LESS than the first.
        // This is what the receptor state buys us — a natural
        // self-limiting mechanism against spam dosing.
        let mut retina = VisualCortex::preserve_spatial(11, 11);
        let cfg = LsdConfig {
            dose: 12, duration: 50, epochs: 1, lr: 1e-4,
            plasticity_boost: 4.0, integration: 1.0, seed: 42,
        };

        let r1 = retina.lsd(cfg);
        let availability_after_1 = retina.receptors.ht2a;

        // Second trip — no tick_receptors between, so receptors are
        // still desensitised.
        let r2 = retina.lsd(cfg);
        let availability_after_2 = retina.receptors.ht2a;

        // Receptors must have desensitised further.
        assert!(availability_after_2 < availability_after_1,
            "second trip didn't desensitise further: {availability_after_1} → {availability_after_2}");
        // The second trip's cortical delta must be smaller.
        // peak_v4_delta is cumulative here since integration=1 keeps
        // each trip's changes, so compare the per-trip movement via
        // post - peak on each report's snapshot — but since both
        // reports were taken against different baselines, the cleaner
        // comparison is: r2.peak_v4_delta is measured from BEFORE
        // the second trip (already containing trip 1's delta),
        // so r2 < r1 directly.
        assert!(r2.peak_v4_delta < r1.peak_v4_delta,
            "second trip moved V4 more than first: r1={} r2={}",
            r1.peak_v4_delta, r2.peak_v4_delta);
    }

    #[test]
    fn lsd_receptor_recovery_via_tick() {
        // After a trip, `tick_receptors` should restore sensitivity
        // toward 1.0 asymptotically. High rate should hit ≥95% fast.
        let mut retina = VisualCortex::preserve_spatial(11, 11);
        retina.lsd(LsdConfig {
            dose: 16, duration: 20, epochs: 1, lr: 1e-4,
            plasticity_boost: 1.0, integration: 0.0, seed: 1,
        });
        let after_trip = retina.receptors.ht2a;
        assert!(after_trip < 0.9, "heavy dose didn't desensitise much: {after_trip}");

        // 500 ticks at rate=0.01 should asymptotically approach 1.0:
        // 1 - (1 - a)(1 - r)^n → ~0.993 for a=0.75, r=0.01, n=500.
        for _ in 0..500 { retina.tick_receptors(0.01); }
        assert!(retina.receptors.ht2a > 0.95,
            "receptors didn't recover: {}", retina.receptors.ht2a);
    }

    #[test]
    fn lsd_availability_floor() {
        // Even with many heavy trips and no recovery, availability
        // never hits zero — biology doesn't wipe out receptors from
        // agonist exposure alone.
        let mut retina = VisualCortex::preserve_spatial(11, 11);
        for i in 0..20 {
            retina.lsd(LsdConfig {
                dose: 16, duration: 10, epochs: 1, lr: 1e-5,
                plasticity_boost: 1.0, integration: 0.0, seed: i,
            });
        }
        assert!(retina.receptors.ht2a >= 0.05,
            "availability went below floor: {}", retina.receptors.ht2a);
    }

    #[test]
    fn train_dream_runs_and_modifies_cortex() {
        let mut retina = VisualCortex::preserve_spatial(11, 11);
        // Snapshot V4 weights (V2/V4 = learned, V1 fixed).
        let v4_before: Vec<f32> = retina.v4.weight.clone();
        let energies = retina.train_dream(50, 1, 1e-4, 8, 42);
        assert_eq!(energies.len(), 1);
        // Cortex must have moved. If V4 is unchanged the dream loop
        // isn't actually training anything.
        let diff: f32 = retina.v4.weight.iter().zip(&v4_before)
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1e-6, "train_dream left V4 unchanged (diff={diff})");
    }

    #[test]
    fn dream_pixel_shape_and_nonzero() {
        let retina = VisualCortex::preserve_spatial(11, 11);
        let pixels = retina.dream_pixel(1234, 8);
        assert_eq!(pixels.len(), 3 * 11 * 11);
        // With random seeding and a learned-but-random cortex, the
        // synthesized image should be non-degenerate — not all zero,
        // not all one. Variance > 0 is the cheapest sanity check.
        let mean: f32 = pixels.iter().sum::<f32>() / pixels.len() as f32;
        let var: f32 = pixels.iter().map(|x| (x - mean).powi(2)).sum::<f32>()
            / pixels.len() as f32;
        assert!(var > 1e-6, "dream_pixel produced a near-constant image (var={var})");
    }

    #[test]
    fn transpose_is_forward_adjoint_stride2() {
        let conv = Conv2d::new(2, 4, 3, 2, 1);
        let (in_h, in_w) = (8usize, 8usize);
        let (out_h, out_w) = (4usize, 4usize); // stride 2

        let x = fill_rand(2 * in_h * in_w, 42);
        let y = fill_rand(4 * out_h * out_w, 1337);

        let fx = {
            let (out, h, w) = conv.forward(&x, 1, in_h, in_w);
            assert_eq!((h, w), (out_h, out_w));
            let mut out = out;
            for oc in 0..conv.out_channels {
                for hw in 0..out_h * out_w {
                    out[oc * out_h * out_w + hw] -= conv.bias[oc];
                }
            }
            out
        };
        let ty = conv.transpose_forward(&y, 1, out_h, out_w, in_h, in_w);

        let lhs = dot(&fx, &y);
        let rhs = dot(&x, &ty);
        let rel = (lhs - rhs).abs() / lhs.abs().max(rhs.abs()).max(1e-6);
        assert!(rel < 1e-5, "adjoint identity failed (stride 2): <Ax,y>={lhs} <x,A*y>={rhs} rel={rel}");
    }

    // ─── GPU-shape smoke tests ────────────────────────────────
    //
    // These deliberately use dims ≥ 64 in m, k, n for all three matmul
    // variants so the ROCm backend's `should_run_on_gpu` gate routes them
    // to hipBLAS under `--features rocm`. On default (CPU) builds they
    // exercise the same code path through the CPU matmul — passing on
    // both configs is the correctness signal that our im2col layout
    // and SGEMM arg derivations are consistent between backends.

    /// Adjoint identity at GPU-capable shapes. Exercises matmul_nn in
    /// `forward` and matmul_tn in `transpose_forward`. Conv shapes:
    /// 64→64 channels, 3×3 kernel, stride 1, 10×10 input.
    ///   forward: m=64, k=576, n=100
    ///   transpose: m=576, k=64, n=100
    /// All clear the ≥64 gate.
    #[test]
    fn transpose_is_forward_adjoint_gpu_shape() {
        let conv = Conv2d::new(64, 64, 3, 1, 1);
        let (in_h, in_w) = (10usize, 10usize);
        let (out_h, out_w) = (10usize, 10usize);

        let x = fill_rand(64 * in_h * in_w, 42);
        let y = fill_rand(64 * out_h * out_w, 1337);

        let fx = {
            let (out, h, w) = conv.forward(&x, 1, in_h, in_w);
            assert_eq!((h, w), (out_h, out_w));
            let mut out = out;
            for oc in 0..conv.out_channels {
                for hw in 0..out_h * out_w {
                    out[oc * out_h * out_w + hw] -= conv.bias[oc];
                }
            }
            out
        };
        let ty = conv.transpose_forward(&y, 1, out_h, out_w, in_h, in_w);

        let lhs = dot(&fx, &y);
        let rhs = dot(&x, &ty);
        let rel = (lhs - rhs).abs() / lhs.abs().max(rhs.abs()).max(1e-6);
        // Slightly looser tolerance than the small-shape tests: f32
        // matmul accumulates more rounding across k=576 contributions.
        assert!(rel < 1e-4,
            "gpu-shape adjoint: <Ax,y>={lhs} <x,A*y>={rhs} rel={rel}");
    }

    /// End-to-end `VisualCortex::ghl_step` smoke test.
    #[test]
    fn cortex_ghl_step_smoke() {
        let mut cortex = VisualCortex::preserve_spatial(16, 16);
        let retina_before = cortex.retina.weight.clone();
        let v1_before = cortex.v1.weight.clone();
        let v2_before = cortex.v2.weight.clone();
        let v4_before = cortex.v4.weight.clone();

        let img = fill_rand(3 * 16 * 16, 42);
        let (tokens, n_tokens, token_dim) = cortex.spatial_tokens(&img);
        let d_tokens = fill_rand(tokens.len(), 1337);
        assert_eq!(d_tokens.len(), n_tokens * token_dim);

        cortex.ghl_step(&img, &d_tokens, 1e-3, 1.0);

        // Retina is fixed — must not have moved at all.
        let retina_diff: f32 = cortex.retina.weight.iter().zip(&retina_before)
            .map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        assert_eq!(retina_diff, 0.0, "retina must be fixed (DoG bank)");

        // V1 / V2 / V4 are learnable cortex — all should move under GHL.
        let v1_diff: f32 = cortex.v1.weight.iter().zip(&v1_before)
            .map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        let v2_diff: f32 = cortex.v2.weight.iter().zip(&v2_before)
            .map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        let v4_diff: f32 = cortex.v4.weight.iter().zip(&v4_before)
            .map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        assert!(v1_diff > 1e-8, "V1 weights should have moved: max|Δ|={v1_diff}");
        assert!(v2_diff > 1e-8, "V2 weights should have moved: max|Δ|={v2_diff}");
        assert!(v4_diff > 1e-8, "V4 weights should have moved: max|Δ|={v4_diff}");

        for &w in &cortex.v1.weight { assert!(w.is_finite()); }
        for &w in &cortex.v2.weight { assert!(w.is_finite()); }
        for &w in &cortex.v4.weight { assert!(w.is_finite()); }
    }

    /// GHL update structural test: runs `ghl_update` with a known sign
    /// mask and verifies (a) weights moved, (b) all deltas stayed
    /// finite, (c) a negated sign mask produces the opposite-direction
    /// update of the same magnitude. The third is the paper's core
    /// promise — the sign modulates direction, local rule sets magnitude.
    #[test]
    fn ghl_update_sign_flips_direction() {
        let mut conv = Conv2d::new(4, 8, 3, 1, 1);
        let w_before: Vec<f32> = conv.weight.clone();

        let (h, w_) = (6usize, 6usize);
        let input = fill_rand(4 * h * w_, 42);
        let (y, _oh, _ow, patches) = conv.forward_with_patches(&input, 1, h, w_);
        let n_patches = y.len() / conv.out_channels;

        // Sign mask = all +1.
        let patch_size = conv.in_channels * conv.kernel_size * conv.kernel_size;
        let ones = vec![1.0f32; conv.out_channels * patch_size];

        let mut conv_pos = conv.clone();
        conv_pos.ghl_update(&patches, &y, &ones, n_patches, 1e-2, 1.0);
        let delta_pos: Vec<f32> = conv_pos.weight.iter().zip(&w_before)
            .map(|(a, b)| a - b).collect();

        // Sign mask = all -1.
        let neg = vec![-1.0f32; conv.out_channels * patch_size];
        let mut conv_neg = conv.clone();
        conv_neg.ghl_update(&patches, &y, &neg, n_patches, 1e-2, 1.0);
        let delta_neg: Vec<f32> = conv_neg.weight.iter().zip(&w_before)
            .map(|(a, b)| a - b).collect();

        // Sanity: something changed.
        let max_abs: f32 = delta_pos.iter().map(|d| d.abs()).fold(0.0, f32::max);
        assert!(max_abs > 1e-8, "GHL update didn't move weights: max|Δw|={max_abs}");

        // Core property: negating sign flips delta exactly (since
        // local magnitude |...| is unchanged by sign of gradient).
        for (i, (&dp, &dn)) in delta_pos.iter().zip(&delta_neg).enumerate() {
            let s = dp + dn;
            assert!(s.abs() < 1e-6,
                "delta[{i}] should be negated by flipped sign: +={dp} -={dn}");
        }

        // Finiteness / unit-norm not enforced — GHL doesn't normalize rows.
        for &w in &conv_pos.weight { assert!(w.is_finite()); }
        for &w in &conv_neg.weight { assert!(w.is_finite()); }
    }

    /// Numerical-gradient check of `Conv2d::backward`. Picks a random
    /// output dim, perturbs W and input at that coordinate, finite-
    /// difference compares against analytic `d_weight` / `d_input`.
    /// A failure here breaks GHL (needs sign(d_W)) and any downstream
    /// BP through the retina.
    #[test]
    fn conv2d_backward_matches_numerical_gradient() {
        let mut conv = Conv2d::new(3, 4, 3, 1, 1);
        // Scale down weights so finite-diff stays in linear regime.
        for w in conv.weight.iter_mut() { *w *= 0.1; }
        let (h, w_) = (6usize, 6usize);
        let input = fill_rand(3 * h * w_, 42);

        // Forward once to materialize patches, then pick a random
        // upstream gradient for a surrogate loss L = <d_out, y>.
        let (y, out_h, out_w, patches) = conv.forward_with_patches(&input, 1, h, w_);
        let d_out = fill_rand(y.len(), 1337);
        let (d_w, _d_b, d_in) = conv.backward(&d_out, &patches, 1, h, w_, out_h, out_w);

        // Verify a few random weight entries by finite difference.
        let eps = 1e-3f32;
        let mut rng_state = 7u64;
        for _ in 0..8 {
            // pick random weight index
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let idx = (rng_state as usize) % conv.weight.len();
            let w0 = conv.weight[idx];
            conv.weight[idx] = w0 + eps;
            let (y_p, _, _) = conv.forward(&input, 1, h, w_);
            conv.weight[idx] = w0 - eps;
            let (y_m, _, _) = conv.forward(&input, 1, h, w_);
            conv.weight[idx] = w0;
            let lp: f32 = y_p.iter().zip(&d_out).map(|(a, b)| a * b).sum();
            let lm: f32 = y_m.iter().zip(&d_out).map(|(a, b)| a * b).sum();
            let num = (lp - lm) / (2.0 * eps);
            let ana = d_w[idx];
            let rel = (num - ana).abs() / num.abs().max(ana.abs()).max(1e-4);
            assert!(rel < 5e-3,
                "weight[{idx}] grad mismatch: num={num} ana={ana} rel={rel}");
        }

        // Verify a few input entries.
        for _ in 0..8 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let idx = (rng_state as usize) % input.len();
            let mut inp_p = input.clone();
            inp_p[idx] += eps;
            let mut inp_m = input.clone();
            inp_m[idx] -= eps;
            let (y_p, _, _) = conv.forward(&inp_p, 1, h, w_);
            let (y_m, _, _) = conv.forward(&inp_m, 1, h, w_);
            let lp: f32 = y_p.iter().zip(&d_out).map(|(a, b)| a * b).sum();
            let lm: f32 = y_m.iter().zip(&d_out).map(|(a, b)| a * b).sum();
            let num = (lp - lm) / (2.0 * eps);
            let ana = d_in[idx];
            let rel = (num - ana).abs() / num.abs().max(ana.abs()).max(1e-4);
            assert!(rel < 5e-3,
                "input[{idx}] grad mismatch: num={num} ana={ana} rel={rel}");
        }
    }

    /// Save → load → verify weights and state are bit-identical.
    #[test]
    fn save_load_roundtrip() {
        let mut retina = VisualCortex::preserve_spatial(16, 16);
        // Train a little so weights differ from init, to make sure the
        // test isn't just checking that deterministic inits round-trip.
        let bank = fill_rand(3 * 16 * 16, 42);
        retina.hebbian_step(&bank, 1e-3, 4);

        let tmp = std::env::temp_dir().join("modgrad_retina_roundtrip.bin");
        retina.save(&tmp).expect("save");
        let loaded = VisualCortex::load(&tmp).expect("load");
        std::fs::remove_file(&tmp).ok();

        assert_eq!(loaded.v1.weight, retina.v1.weight, "V1 weights differ after roundtrip");
        assert_eq!(loaded.v2.weight, retina.v2.weight, "V2 weights differ after roundtrip");
        assert_eq!(loaded.v4.weight, retina.v4.weight, "V4 weights differ after roundtrip");
        assert_eq!(loaded.input_h, retina.input_h);
        assert_eq!(loaded.input_w, retina.input_w);
        assert!((loaded.receptors.ht2a - retina.receptors.ht2a).abs() < 1e-9);
    }

    /// Hebbian update at GPU-capable shapes. Exercises all three matmul
    /// variants in a single call (NN for activations, TN for recon, NT
    /// for dW). Asserts weights changed, unit-norm maintained, and no
    /// NaNs. Doesn't assert specific values — backends may differ at
    /// f32 precision, but the structural invariants must hold.
    #[test]
    fn hebbian_update_gpu_shape_structural() {
        let mut conv = Conv2d::new(64, 64, 3, 1, 1);
        let w_before: Vec<f32> = conv.weight.clone();
        let input = fill_rand(64 * 10 * 10, 42);

        conv.hebbian_update(&input, 1, 10, 10, 1e-2, 8);

        let max_diff: f32 = conv.weight.iter().zip(&w_before)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(max_diff > 1e-6,
            "hebbian_update left weights unchanged: max_diff={max_diff}");

        let patch_size = 64 * 9;
        for oc in 0..conv.out_channels {
            let w_off = oc * patch_size;
            let norm: f32 = (0..patch_size)
                .map(|i| conv.weight[w_off + i].powi(2))
                .sum::<f32>()
                .sqrt();
            // Either unit-norm (row was updated and normalized) or
            // legitimately tiny (unreachable; untouched rows preserve
            // init norm). Reject values in between — that would mean
            // normalization ran on a half-updated row.
            assert!((norm - 1.0).abs() < 1e-3 || norm < 1e-6,
                "filter {oc} norm = {norm}, expected ≈1.0 or ≈0");
        }
        for (i, &w) in conv.weight.iter().enumerate() {
            assert!(w.is_finite(), "weight[{i}] = {w} not finite after hebbian");
        }
    }
}
