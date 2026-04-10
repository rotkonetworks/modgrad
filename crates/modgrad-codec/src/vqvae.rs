//! VQ-VAE image tokenizer: pixels → discrete codes → pixels.
//!
//! Converts images into sequences of discrete codes from a learned codebook.
//! These codes can be embedded like bytes and fed into the CTM alongside text.
//!
//! Architecture (Chameleon-inspired):
//!   Encoder: Conv2d stack → downsample → latent vectors [H'×W'×D]
//!   VQ:      nearest codebook lookup → discrete codes [H'×W']
//!   Decoder: upsample from codebook vectors → reconstruct pixels
//!
//! For CIFAR-10 (32×32×3): encoder produces 8×8 latent map = 64 codes per image.

use serde::{Deserialize, Serialize};
use super::retina::Conv2d;

// ─── Vector Quantization ───────────────────────────────────

/// Vector Quantization layer with codebook.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorQuantizer {
    /// Codebook: [n_codes × code_dim]
    pub codebook: Vec<f32>,
    pub n_codes: usize,
    pub code_dim: usize,
    /// EMA decay for codebook update (0.99 typical)
    pub ema_decay: f32,
    /// Usage count per code (for dead code revival)
    pub usage: Vec<f32>,
    /// EMA cluster means
    pub ema_means: Vec<f32>,
    /// EMA cluster counts
    pub ema_counts: Vec<f32>,
}

impl VectorQuantizer {
    pub fn new(n_codes: usize, code_dim: usize) -> Self {
        // Initialize codebook with uniform random in [-1/n_codes, 1/n_codes]
        let mut rng_state = (n_codes * code_dim) as u64 + 42;
        let scale = 1.0 / n_codes as f32;
        let codebook: Vec<f32> = (0..n_codes * code_dim)
            .map(|_| {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let u = (rng_state >> 40) as f32 / (1u64 << 24) as f32;
                (u * 2.0 - 1.0) * scale
            })
            .collect();

        Self {
            codebook: codebook.clone(),
            n_codes,
            code_dim,
            ema_decay: 0.99,
            usage: vec![0.0; n_codes],
            ema_means: codebook,
            ema_counts: vec![1.0; n_codes], // avoid div by zero
        }
    }

    /// Quantize: find nearest codebook entry for each vector.
    /// Input: [n_vectors × code_dim] flat.
    /// Returns: (quantized vectors [n_vectors × code_dim], code indices [n_vectors])
    pub fn quantize(&self, z: &[f32]) -> (Vec<f32>, Vec<usize>) {
        let n = z.len() / self.code_dim;
        let d = self.code_dim;
        let mut quantized = Vec::with_capacity(n * d);
        let mut indices = Vec::with_capacity(n);

        for i in 0..n {
            let vec = &z[i * d..(i + 1) * d];
            let mut best_idx = 0;
            let mut best_dist = f32::MAX;

            for c in 0..self.n_codes {
                let code = &self.codebook[c * d..(c + 1) * d];
                let dist: f32 = vec.iter().zip(code).map(|(&a, &b)| (a - b).powi(2)).sum();
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = c;
                }
            }

            indices.push(best_idx);
            quantized.extend_from_slice(&self.codebook[best_idx * d..(best_idx + 1) * d]);
        }

        (quantized, indices)
    }

    /// Decode code indices back to vectors using codebook lookup.
    pub fn decode_indices(&self, indices: &[usize]) -> Vec<f32> {
        let d = self.code_dim;
        let mut out = Vec::with_capacity(indices.len() * d);
        for &idx in indices {
            out.extend_from_slice(&self.codebook[idx * d..(idx + 1) * d]);
        }
        out
    }

    /// EMA codebook update (during training).
    /// z_flat: encoder outputs [n × code_dim]
    /// indices: which code each vector mapped to
    pub fn update_codebook(&mut self, z_flat: &[f32], indices: &[usize]) {
        let d = self.code_dim;
        let n = indices.len();
        let decay = self.ema_decay;

        // Count assignments and accumulate means
        let mut new_counts = vec![0.0f32; self.n_codes];
        let mut new_sums = vec![0.0f32; self.n_codes * d];

        for i in 0..n {
            let idx = indices[i];
            new_counts[idx] += 1.0;
            let vec = &z_flat[i * d..(i + 1) * d];
            for j in 0..d {
                new_sums[idx * d + j] += vec[j];
            }
        }

        // EMA update
        for c in 0..self.n_codes {
            self.ema_counts[c] = decay * self.ema_counts[c] + (1.0 - decay) * new_counts[c];
            for j in 0..d {
                self.ema_means[c * d + j] = decay * self.ema_means[c * d + j]
                    + (1.0 - decay) * new_sums[c * d + j];
            }
            // Update codebook entry
            let count = self.ema_counts[c].max(1e-5);
            for j in 0..d {
                self.codebook[c * d + j] = self.ema_means[c * d + j] / count;
            }
            self.usage[c] = decay * self.usage[c] + (1.0 - decay) * new_counts[c];
        }

        // Random restart for dead codes
        let mut rng_state = (n as u64).wrapping_mul(2654435761);
        for c in 0..self.n_codes {
            if self.usage[c] < 0.01 && n > 0 {
                // Replace with random input vector
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let src = (rng_state as usize) % n;
                let vec = &z_flat[src * d..(src + 1) * d];
                self.codebook[c * d..(c + 1) * d].copy_from_slice(vec);
                self.ema_means[c * d..(c + 1) * d].copy_from_slice(vec);
                self.ema_counts[c] = 1.0;
                self.usage[c] = 1.0;
            }
        }
    }

    /// Codebook utilization: fraction of codes used at least once.
    pub fn utilization(&self) -> f32 {
        let used = self.usage.iter().filter(|&&u| u > 0.1).count();
        used as f32 / self.n_codes as f32
    }
}

// ─── Transposed Conv2d (for decoder upsampling) ────────────

/// Transposed 2D convolution for upsampling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvTranspose2d {
    /// Weights: [in_channels × out_channels × kh × kw]
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
}

impl ConvTranspose2d {
    pub fn new(in_ch: usize, out_ch: usize, kernel: usize, stride: usize, padding: usize) -> Self {
        let n = out_ch * kernel * kernel;
        let scale = (2.0 / n as f32).sqrt();
        let mut rng_state = (in_ch * out_ch * kernel + 7777) as u64;
        let weight: Vec<f32> = (0..in_ch * out_ch * kernel * kernel)
            .map(|_| {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let u1 = ((rng_state >> 40) as f32 / (1u64 << 24) as f32).max(1e-10);
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let u2 = (rng_state >> 40) as f32 / (1u64 << 24) as f32;
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() * scale
            })
            .collect();
        Self { weight, bias: vec![0.0; out_ch], in_channels: in_ch, out_channels: out_ch,
               kernel_size: kernel, stride, padding }
    }

    /// Forward: [in_ch × h × w] → [out_ch × h' × w']
    /// h' = (h - 1) * stride - 2*padding + kernel
    pub fn forward(&self, input: &[f32], h: usize, w: usize) -> (Vec<f32>, usize, usize) {
        let k = self.kernel_size;
        let s = self.stride;
        let p = self.padding;
        let out_h = (h - 1) * s - 2 * p + k;
        let out_w = (w - 1) * s - 2 * p + k;
        let mut output = vec![0.0f32; self.out_channels * out_h * out_w];

        // Initialize with bias
        for oc in 0..self.out_channels {
            for i in 0..out_h * out_w {
                output[oc * out_h * out_w + i] = self.bias[oc];
            }
        }

        // Transposed convolution: scatter input values
        for ic in 0..self.in_channels {
            for ih in 0..h {
                for iw in 0..w {
                    let val = input[ic * h * w + ih * w + iw];
                    for kh in 0..k {
                        for kw in 0..k {
                            let oh = ih * s + kh;
                            let ow = iw * s + kw;
                            if oh >= p && ow >= p && oh - p < out_h && ow - p < out_w {
                                let oi = (oh - p) * out_w + (ow - p);
                                for oc in 0..self.out_channels {
                                    let w_idx = ic * (self.out_channels * k * k)
                                        + oc * (k * k) + kh * k + kw;
                                    output[oc * out_h * out_w + oi] += val * self.weight[w_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
        (output, out_h, out_w)
    }
}

// ─── VQ-VAE ────────────────────────────────────────────────

/// Output of a VQ-VAE forward pass.
pub struct VqVaeOutput {
    /// Reconstructed pixels [3 × H × W].
    pub reconstruction: Vec<f32>,
    /// Discrete codebook indices (one per spatial position).
    pub codes: Vec<usize>,
    /// Commitment loss: ||z_e - sg(z_q)||².
    pub commitment_loss: f32,
}

/// VQ-VAE image tokenizer.
///
/// Encoder: 3 → 32 → 64 → 128 → code_dim (conv + relu + downsample)
/// VQ: codebook lookup
/// Decoder: code_dim → 128 → 64 → 32 → 3 (conv + relu + upsample)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VqVae {
    // Encoder
    pub enc1: Conv2d,   // 3 → 32, stride 2 (32×32 → 16×16)
    pub enc2: Conv2d,   // 32 → 64, stride 2 (16×16 → 8×8)
    pub enc3: Conv2d,   // 64 → 128, stride 1 (8×8 → 8×8)
    pub enc_proj: Conv2d, // 128 → code_dim, stride 1, kernel 1

    // Vector quantizer
    pub vq: VectorQuantizer,

    // Decoder
    pub dec_proj: Conv2d,         // code_dim → 128, stride 1, kernel 1
    pub dec1: ConvTranspose2d,    // 128 → 64, stride 1 (8×8 → 8×8)
    pub dec2: ConvTranspose2d,    // 64 → 32, stride 2 (8×8 → 16×16)
    pub dec3: ConvTranspose2d,    // 32 → 3, stride 2 (16×16 → 32×32)

    pub code_dim: usize,
    pub n_codes: usize,
}

fn relu_vec(v: &mut [f32]) {
    for x in v.iter_mut() { *x = x.max(0.0); }
}

impl VqVae {
    /// Create a VQ-VAE for images of size 32×32×3.
    /// `n_codes`: codebook size (4096 recommended, matching WavTokenizer).
    /// `code_dim`: latent dimension per code (64 typical).
    pub fn new(n_codes: usize, code_dim: usize) -> Self {
        Self {
            enc1: Conv2d::new(3, 32, 4, 2, 1),     // 32×32 → 16×16
            enc2: Conv2d::new(32, 64, 4, 2, 1),    // 16×16 → 8×8
            enc3: Conv2d::new(64, 128, 3, 1, 1),   // 8×8 → 8×8
            enc_proj: Conv2d::new(128, code_dim, 1, 1, 0), // project to code_dim

            vq: VectorQuantizer::new(n_codes, code_dim),

            dec_proj: Conv2d::new(code_dim, 128, 1, 1, 0),
            dec1: ConvTranspose2d::new(128, 64, 3, 1, 1),    // 8×8 → 8×8
            dec2: ConvTranspose2d::new(64, 32, 4, 2, 1),     // 8×8 → 16×16
            dec3: ConvTranspose2d::new(32, 3, 4, 2, 1),      // 16×16 → 32×32

            code_dim,
            n_codes,
        }
    }

    /// Encode image to latent vectors.
    /// Input: [3 × 32 × 32] flat (CHW).
    /// Output: [64 × code_dim] flat (8×8 spatial positions, each code_dim dims).
    pub fn encode(&self, pixels: &[f32]) -> Vec<f32> {
        let (mut h1, h, w) = self.enc1.forward(pixels, 32, 32);
        relu_vec(&mut h1);
        let (mut h2, h, w) = self.enc2.forward(&h1, h, w);
        relu_vec(&mut h2);
        let (mut h3, h, w) = self.enc3.forward(&h2, h, w);
        relu_vec(&mut h3);
        let (z, _, _) = self.enc_proj.forward(&h3, h, w);
        z
    }

    /// Result of a VQ-VAE forward pass.
    pub fn forward(&self, pixels: &[f32]) -> VqVaeOutput {
        let z = self.encode(pixels);
        let (quantized, indices) = self.vq.quantize(&z);

        // Commitment loss: ||z - sg(quantized)||²
        let commitment: f32 = z.iter().zip(&quantized)
            .map(|(&a, &b)| (a - b).powi(2)).sum::<f32>()
            / z.len() as f32;

        // Straight-through estimator: use quantized in forward, gradient flows to z
        // In the actual backward pass, d_z = d_quantized (straight-through)
        let reconstruction = self.decode(&quantized);

        VqVaeOutput { reconstruction, codes: indices, commitment_loss: commitment }
    }

    /// Decode from quantized latent vectors.
    /// Input: [64 × code_dim] flat (8×8 spatial, code_dim channels).
    /// Need to reshape to [code_dim × 8 × 8] for conv processing.
    pub fn decode(&self, quantized: &[f32]) -> Vec<f32> {
        // quantized is [n_positions × code_dim], need [code_dim × h × w]
        let h = 8;
        let w = 8;
        let d = self.code_dim;
        let n = h * w;

        // Transpose: [n × d] → [d × h × w]
        let mut latent = vec![0.0f32; d * h * w];
        for pos in 0..n {
            for c in 0..d {
                latent[c * n + pos] = quantized[pos * d + c];
            }
        }

        let (mut h1, oh, ow) = self.dec_proj.forward(&latent, h, w);
        relu_vec(&mut h1);
        let (mut h2, oh, ow) = self.dec1.forward(&h1, oh, ow);
        relu_vec(&mut h2);
        let (mut h3, oh, ow) = self.dec2.forward(&h2, oh, ow);
        relu_vec(&mut h3);
        let (recon, _, _) = self.dec3.forward(&h3, oh, ow);
        // No activation on final layer — output is pixel values
        recon
    }

    /// Tokenize: image → code indices (what the CTM sees).
    pub fn tokenize(&self, pixels: &[f32]) -> Vec<usize> {
        let z = self.encode(pixels);
        let (_, indices) = self.vq.quantize(&z);
        indices
    }

    /// Detokenize: code indices → reconstructed image.
    pub fn detokenize(&self, indices: &[usize]) -> Vec<f32> {
        let quantized = self.vq.decode_indices(indices);
        self.decode(&quantized)
    }

    /// Reconstruction loss: MSE between original and reconstructed pixels.
    pub fn reconstruction_loss(original: &[f32], reconstructed: &[f32]) -> f32 {
        original.iter().zip(reconstructed)
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>() / original.len() as f32
    }

    /// Total loss = reconstruction + beta * commitment.
    pub fn loss(original: &[f32], reconstructed: &[f32], commitment: f32, beta: f32) -> f32 {
        Self::reconstruction_loss(original, reconstructed) + beta * commitment
    }

    /// Save. Format by extension: `.bin` → bincode, `.json` → JSON.
    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        modgrad_persist::persist::save(self, path).map_err(|e| e.into())
    }

    /// Load. Format by extension, with JSON fallback.
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        modgrad_persist::persist::load(path).map_err(|e| e.into())
    }

    /// Number of spatial codes per image.
    pub fn codes_per_image(&self) -> usize { 64 } // 8×8 for 32×32 input

    /// Total parameters.
    pub fn n_params(&self) -> usize {
        self.enc1.param_count() + self.enc2.param_count()
            + self.enc3.param_count() + self.enc_proj.param_count()
            + self.vq.codebook.len()
            + self.dec_proj.param_count() + self.dec1.weight.len() + self.dec1.bias.len()
            + self.dec2.weight.len() + self.dec2.bias.len()
            + self.dec3.weight.len() + self.dec3.bias.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vq_quantize_roundtrip() {
        let vq = VectorQuantizer::new(256, 16);
        let z = vec![0.1f32; 4 * 16]; // 4 vectors of dim 16
        let (quantized, indices) = vq.quantize(&z);
        assert_eq!(quantized.len(), 4 * 16);
        assert_eq!(indices.len(), 4);
        // Decode should give same as quantized
        let decoded = vq.decode_indices(&indices);
        assert_eq!(decoded, quantized);
    }

    #[test]
    fn vqvae_forward_dims() {
        let vae = VqVae::new(256, 32);
        let pixels = vec![0.5f32; 3 * 32 * 32];
        let out = vae.forward(&pixels);
        assert_eq!(out.codes.len(), 64, "8×8 = 64 codes");
        assert_eq!(out.reconstruction.len(), 3 * 32 * 32, "reconstruction should match input dims");
        assert!(out.commitment_loss >= 0.0);
        eprintln!("  vqvae params: {}", vae.n_params());
        eprintln!("  codes per image: {}", out.codes.len());
        eprintln!("  commitment loss: {:.4}", out.commitment_loss);
        eprintln!("  recon loss: {:.4}", VqVae::reconstruction_loss(&pixels, &out.reconstruction));
    }

    #[test]
    fn tokenize_detokenize() {
        let vae = VqVae::new(512, 16);
        let pixels = vec![0.3f32; 3 * 32 * 32];
        let codes = vae.tokenize(&pixels);
        assert_eq!(codes.len(), 64);
        let recon = vae.detokenize(&codes);
        assert_eq!(recon.len(), 3 * 32 * 32);
    }
}
