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

    /// Forward pass: [in_ch × h × w] → [out_ch × h' × w']
    /// Returns (output, output_h, output_w)
    pub fn forward(&self, input: &[f32], h: usize, w: usize) -> (Vec<f32>, usize, usize) {
        let k = self.kernel_size;
        let s = self.stride;
        let p = self.padding;
        let out_h = (h + 2 * p - k) / s + 1;
        let out_w = (w + 2 * p - k) / s + 1;
        let mut output = vec![0.0f32; self.out_channels * out_h * out_w];

        for oc in 0..self.out_channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut sum = self.bias[oc];
                    for ic in 0..self.in_channels {
                        for kh in 0..k {
                            for kw in 0..k {
                                let ih = oh * s + kh;
                                let iw = ow * s + kw;
                                let ih = ih as isize - p as isize;
                                let iw = iw as isize - p as isize;
                                if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                    let in_idx = ic * h * w + ih as usize * w + iw as usize;
                                    let w_idx = oc * (self.in_channels * k * k) + ic * (k * k) + kh * k + kw;
                                    sum += input[in_idx] * self.weight[w_idx];
                                }
                            }
                        }
                    }
                    output[oc * out_h * out_w + oh * out_w + ow] = sum;
                }
            }
        }
        (output, out_h, out_w)
    }

    pub fn param_count(&self) -> usize {
        self.weight.len() + self.bias.len()
    }

    /// Hebbian sparse coding update (Olshausen & Field 1996).
    ///
    /// For each spatial position, extract the input patch, encode through
    /// the conv weights, sparsify (top-K winners), then update weights
    /// to reduce reconstruction error.
    ///
    /// This is how V1 learns Gabor filters from natural images:
    /// - Encode: activations = W @ patch
    /// - Sparsify: zero out all but top-K activations
    /// - Reconstruct: recon = W^T @ sparse_activations
    /// - Update: W += lr * (patch - recon) ⊗ sparse_activations
    ///
    /// Fully local (Hebbian): each weight updates from pre/post activity only.
    ///
    /// Serial online update: each position sees the weights updated by
    /// prior positions within the same call. A rayon-parallel batch
    /// variant was tried but changes semantics non-trivially (sum-over-
    /// positions ≠ online, average-over-positions underscales) — needs
    /// calibration (likely 1/sqrt(N) scaling) before it can ship.
    /// Leaving serial here; use GPU conv2d for real-scale speedups.
    pub fn hebbian_update(&mut self, input: &[f32], h: usize, w: usize, lr: f32, sparsity_k: usize) {
        let k = self.kernel_size;
        let s = self.stride;
        let p = self.padding;
        let out_h = (h + 2 * p - k) / s + 1;
        let out_w = (w + 2 * p - k) / s + 1;
        let patch_size = self.in_channels * k * k;

        // Sample random spatial positions (don't process ALL — too slow)
        let max_positions = 64;
        let step_h = (out_h / max_positions.min(out_h).max(1)).max(1);
        let step_w = (out_w / max_positions.min(out_w).max(1)).max(1);

        let mut updated_filters = vec![false; self.out_channels];

        for oh in (0..out_h).step_by(step_h) {
            for ow in (0..out_w).step_by(step_w) {
                // Extract patch
                let mut patch = vec![0.0f32; patch_size];
                for ic in 0..self.in_channels {
                    for kh in 0..k {
                        for kw in 0..k {
                            let ih = (oh * s + kh) as isize - p as isize;
                            let iw = (ow * s + kw) as isize - p as isize;
                            if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                patch[ic * k * k + kh * k + kw] =
                                    input[ic * h * w + ih as usize * w + iw as usize];
                            }
                        }
                    }
                }

                // Skip near-zero patches (V1 output after ReLU is often sparse)
                let patch_energy: f32 = patch.iter().map(|x| x * x).sum();
                if patch_energy < 1e-6 { continue; }

                // Encode: activations[oc] = W[oc] · patch (clamped to prevent NaN)
                let mut activations = vec![0.0f32; self.out_channels];
                for oc in 0..self.out_channels {
                    let w_off = oc * patch_size;
                    let dot: f32 = (0..patch_size)
                        .map(|i| self.weight[w_off + i] * patch[i])
                        .sum();
                    activations[oc] = (dot + self.bias[oc]).clamp(-100.0, 100.0);
                }

                // Sparsify: keep top-K by absolute value (winner-take-all)
                let mut sorted_indices: Vec<usize> = (0..self.out_channels).collect();
                sorted_indices.sort_by(|&a, &b| {
                    let va = if activations[a].is_finite() { activations[a].abs() } else { 0.0 };
                    let vb = if activations[b].is_finite() { activations[b].abs() } else { 0.0 };
                    vb.total_cmp(&va)
                });
                let mut sparse = vec![0.0f32; self.out_channels];
                for &idx in sorted_indices.iter().take(sparsity_k) {
                    sparse[idx] = activations[idx];
                }

                // Reconstruct: recon = W^T @ sparse
                let mut recon = vec![0.0f32; patch_size];
                for oc in 0..self.out_channels {
                    if sparse[oc] == 0.0 { continue; }
                    let w_off = oc * patch_size;
                    for i in 0..patch_size {
                        recon[i] += self.weight[w_off + i] * sparse[oc];
                    }
                }

                // Error: patch - recon
                let error: Vec<f32> = patch.iter().zip(&recon).map(|(p, r)| p - r).collect();

                // Update: W[oc] += lr * sparse[oc] * error
                for oc in 0..self.out_channels {
                    if sparse[oc] == 0.0 { continue; }
                    updated_filters[oc] = true;
                    let w_off = oc * patch_size;
                    for i in 0..patch_size {
                        let delta = lr * sparse[oc] * error[i];
                        if delta.is_finite() {
                            self.weight[w_off + i] += delta;
                        }
                    }
                }
            }
        }

        // Normalize only filters that were updated (don't destroy untouched weights)
        for oc in 0..self.out_channels {
            if !updated_filters[oc] { continue; }
            let w_off = oc * patch_size;
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
/// Our retina layer uses FIXED Gabor-like filters (not random, not trained).
/// These are the "evolved" feature extractors.
///
/// V1+ cortex (the CTM) is where LEARNING happens — via predictive coding
/// and Hebbian plasticity. The retina just hands it preprocessed features.
///
/// Architecture:
/// - Retina (fixed): edge detection, color opponency, contrast normalization
/// - V1 (learnable via Hebbian): oriented edges, spatial frequency
/// - V2 (learnable): contours, texture boundaries
/// - V4 (learnable): shapes, curvature
/// Pool: global average → observation vector
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub struct VisualRetina {
    pub v1: Conv2d,     // edges: 3 → 32 channels
    pub v2: Conv2d,     // contours: 32 → 64 channels
    pub v4: Conv2d,     // shapes: 64 → 128 channels
    pub pool_dim: usize, // output of global avg pool = 128
    pub input_h: usize,
    pub input_w: usize,
}

impl VisualRetina {
    /// Create visual retina for images of size h×w×3.
    ///
    /// The retinal layer (v1) uses FIXED biologically-inspired filters:
    /// - Horizontal/vertical/diagonal edge detectors (center-surround)
    /// - Color opponent channels (R-G, B-Y)
    /// - Luminance channels
    /// These are "evolved" not learned — equivalent to ganglion cell types.
    ///
    /// V2/V4 are randomly initialized — these represent CORTEX, not retina.
    /// Cortex learns via Hebbian sparse coding during experience.
    pub fn new(h: usize, w: usize) -> Self {
        // Retinal layer: FIXED Gabor-like filters (evolved, not learned)
        let mut v1 = Conv2d::new(3, 32, 3, 2, 1);
        init_retinal_filters(&mut v1);

        // Cortical layers: random init, will learn via Hebbian
        let v2 = Conv2d::new(32, 64, 3, 2, 1);
        let v4 = Conv2d::new(64, 128, 3, 2, 1);

        Self { v1, v2, v4, pool_dim: 128, input_h: h, input_w: w }
    }

    /// For CIFAR-10 (32×32×3)
    pub fn cifar() -> Self { Self::new(32, 32) }

    /// For ImageNet-scale (224×224×3)
    pub fn imagenet() -> Self { Self::new(224, 224) }

    /// For mazes and small grid-world images: preserves spatial resolution.
    ///
    /// Uses stride 1 on V2/V4 so spatial dimensions are not aggressively
    /// reduced. A 39×39 maze → V1(stride 2): 20×20 → V2(stride 1): 20×20
    /// → V4(stride 1): 20×20 = 400 spatial tokens of 128-dim.
    /// An 11×11 maze → 6×6 = 36 tokens. Much better than 4 with all stride-2.
    pub fn maze(h: usize, w: usize) -> Self {
        let mut v1 = Conv2d::new(3, 32, 3, 2, 1); // stride 2: halve spatial
        init_retinal_filters(&mut v1);
        let v2 = Conv2d::new(32, 64, 3, 1, 1);    // stride 1: preserve spatial
        let v4 = Conv2d::new(64, 128, 3, 1, 1);   // stride 1: preserve spatial
        Self { v1, v2, v4, pool_dim: 128, input_h: h, input_w: w }
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

                // Forward V1 (fixed retinal filters) to get V2's input
                let (mut v1_out, h1, w1) = self.v1.forward(img, h, w);
                leaky_relu(&mut v1_out);

                // V2 (cortex) learns from retinal features
                self.v2.hebbian_update(&v1_out, h1, w1, lr, sparsity);

                // Forward V2 to get V4's input
                let (mut v2_out, h2, w2) = self.v2.forward(&v1_out, h1, w1);
                leaky_relu(&mut v2_out);

                // V4 (cortex) learns from V2 features
                self.v4.hebbian_update(&v2_out, h2, w2, lr * 0.5, sparsity);

                if i % 5000 == 0 && i > 0 {
                    eprintln!("    epoch {epoch} image {i}/{}", images.len());
                }
            }

            // Measure reconstruction quality by computing forward pass variance
            // (proxy for feature quality — good features have high variance)
            if let Some(&img) = images.first() {
                let (mut v1_out, h1, w1) = self.v1.forward(img, h, w);
                leaky_relu(&mut v1_out);
                v1_err = v1_out.iter().map(|x| x * x).sum::<f32>() / v1_out.len() as f32;

                let (mut v2_out, h2, w2) = self.v2.forward(&v1_out, h1, w1);
                leaky_relu(&mut v2_out);
                v2_err = v2_out.iter().map(|x| x * x).sum::<f32>() / v2_out.len() as f32;

                let (v4_out, _, _) = self.v4.forward(&v2_out, h2, w2);
                v4_err = v4_out.iter().map(|x| x * x).sum::<f32>() / v4_out.len() as f32;
            }

            eprintln!("  epoch {epoch}: V1 energy={v1_err:.4} V2={v2_err:.4} V4={v4_err:.4}");
            errors.push(v1_err + v2_err + v4_err);
        }

        errors
    }
}

impl VisualRetina {
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

        let (mut v1_out, h1, w1) = self.v1.forward(raw, h, w);
        leaky_relu(&mut v1_out);

        let (mut v2_out, h2, w2) = self.v2.forward(&v1_out, h1, w1);
        leaky_relu(&mut v2_out);

        let (mut v4_out, h4, w4) = self.v4.forward(&v2_out, h2, w2);
        leaky_relu(&mut v4_out);

        let channels = self.v4.out_channels; // 128
        let n_tokens = h4 * w4;

        // Transpose from CHW [channels × h4 × w4] to token sequence [n_tokens × channels]
        // Each spatial position (y, x) becomes one token
        let mut tokens = vec![0.0f32; n_tokens * channels];
        for y in 0..h4 {
            for x in 0..w4 {
                let token_idx = y * w4 + x;
                for c in 0..channels {
                    tokens[token_idx * channels + c] = v4_out[c * h4 * w4 + y * w4 + x];
                }
            }
        }

        (tokens, n_tokens, channels)
    }
}

impl modgrad_traits::Encoder for VisualRetina {
    type Raw = [f32]; // pixels: [3 × H × W] flattened

    fn encode(&self, raw: &[f32]) -> modgrad_traits::TokenInput {
        let (tokens, n_tokens, token_dim) = self.spatial_tokens(raw);
        modgrad_traits::TokenInput { tokens, n_tokens, token_dim }
    }

    fn token_dim(&self) -> usize { self.v4.out_channels }
}

impl Retina for VisualRetina {
    fn observe(&mut self, raw: &[f32]) -> Vec<f32> {
        // Input: [3 × h × w] flattened
        let h = self.input_h;
        let w = self.input_w;

        // V1 (retina, fixed): edge detection, color opponency
        let (mut v1_out, h1, w1) = self.v1.forward(raw, h, w);
        leaky_relu(&mut v1_out);

        // V2 (cortex): contours from edges
        let (mut v2_out, h2, w2) = self.v2.forward(&v1_out, h1, w1);
        leaky_relu(&mut v2_out);

        // V4 (cortex): shapes from contours
        let (mut v4_out, h4, w4) = self.v4.forward(&v2_out, h2, w2);
        leaky_relu(&mut v4_out);

        // Multi-scale pooling: combine features from all levels.
        // V1 captures fine edges, V4 captures coarse shapes.
        // Pool each level and concatenate for richer representation.
        let v1_pool = adaptive_avg_pool(&v1_out, 32, h1, w1);   // 32-dim
        let _v2_pool = adaptive_avg_pool(&v2_out, 64, h2, w2);   // 64-dim
        let v4_pool = adaptive_avg_pool(&v4_out, 128, h4, w4);  // 128-dim
        // Total: 32 + 64 + 128 = 224... but we need pool_dim.
        // Use V4 pool as primary (matches pool_dim=128)
        // but add V1 summary as bonus signal
        let mut out = v4_pool;
        // Blend in V1 edge statistics (first 32 dims get V1 contribution)
        for i in 0..32.min(out.len()) {
            out[i] = 0.7 * out[i] + 0.3 * v1_pool[i];
        }
        out
    }

    fn d_output(&self) -> usize { self.pool_dim }

    fn param_count(&self) -> usize {
        self.v1.param_count() + self.v2.param_count() + self.v4.param_count()
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
