//! Audio codec: waveform → discrete codes → waveform.
//!
//! Simplified WavTokenizer-style architecture:
//!   Encoder: Conv1d stack → downsample 320× → latent vectors
//!   VQ:      single codebook (4096 entries), EMA update
//!   Decoder: upsample → Conv1d stack → reconstruct waveform
//!
//! At 24kHz sample rate with 320× downsampling: 75 codes/second.
//! Single quantizer — codes are directly embeddable like bytes.

use serde::{Deserialize, Serialize};
use super::vqvae::VectorQuantizer;
use wincode_derive::{SchemaRead, SchemaWrite};

// ─── 1D Convolution ────────────────────────────────────────

/// 1D causal convolution layer.
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub struct Conv1d {
    /// Weights: [out_channels × in_channels × kernel_size]
    pub weight: Vec<f32>,
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
        let mut state = (in_ch * out_ch * kernel + 9999) as u64;
        let weight: Vec<f32> = (0..out_ch * in_ch * kernel)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let u1 = ((state >> 40) as f32 / (1u64 << 24) as f32).max(1e-10);
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let u2 = (state >> 40) as f32 / (1u64 << 24) as f32;
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() * scale
            })
            .collect();
        Self { weight, bias: vec![0.0; out_ch], in_channels: in_ch, out_channels: out_ch,
               kernel_size: kernel, stride }
    }

    /// Forward: [in_ch × length] → [out_ch × out_length]
    /// out_length = (length - kernel_size) / stride + 1
    pub fn forward(&self, input: &[f32], length: usize) -> (Vec<f32>, usize) {
        let k = self.kernel_size;
        let s = self.stride;
        let out_len = (length.saturating_sub(k)) / s + 1;
        let mut output = vec![0.0f32; self.out_channels * out_len];

        for oc in 0..self.out_channels {
            for t in 0..out_len {
                let mut sum = self.bias[oc];
                for ic in 0..self.in_channels {
                    for ki in 0..k {
                        let idx = t * s + ki;
                        if idx < length {
                            sum += input[ic * length + idx]
                                * self.weight[oc * (self.in_channels * k) + ic * k + ki];
                        }
                    }
                }
                output[oc * out_len + t] = sum;
            }
        }
        (output, out_len)
    }

    pub fn param_count(&self) -> usize { self.weight.len() + self.bias.len() }
}

/// 1D transposed convolution for upsampling.
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub struct ConvTranspose1d {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
}

impl ConvTranspose1d {
    pub fn new(in_ch: usize, out_ch: usize, kernel: usize, stride: usize) -> Self {
        let n = out_ch * kernel;
        let scale = (2.0 / n as f32).sqrt();
        let mut state = (in_ch * out_ch * kernel + 7777) as u64;
        let weight: Vec<f32> = (0..in_ch * out_ch * kernel)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let u1 = ((state >> 40) as f32 / (1u64 << 24) as f32).max(1e-10);
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let u2 = (state >> 40) as f32 / (1u64 << 24) as f32;
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() * scale
            })
            .collect();
        Self { weight, bias: vec![0.0; out_ch], in_channels: in_ch, out_channels: out_ch,
               kernel_size: kernel, stride }
    }

    /// Forward: [in_ch × length] → [out_ch × out_length]
    /// out_length = (length - 1) * stride + kernel_size
    pub fn forward(&self, input: &[f32], length: usize) -> (Vec<f32>, usize) {
        let k = self.kernel_size;
        let s = self.stride;
        let out_len = (length - 1) * s + k;
        let mut output = vec![0.0f32; self.out_channels * out_len];

        // Init with bias
        for oc in 0..self.out_channels {
            for t in 0..out_len {
                output[oc * out_len + t] = self.bias[oc];
            }
        }

        for ic in 0..self.in_channels {
            for t in 0..length {
                let val = input[ic * length + t];
                for ki in 0..k {
                    let ot = t * s + ki;
                    for oc in 0..self.out_channels {
                        let w_idx = ic * (self.out_channels * k) + oc * k + ki;
                        output[oc * out_len + ot] += val * self.weight[w_idx];
                    }
                }
            }
        }
        (output, out_len)
    }
}

// ─── Audio Codec ───────────────────────────────────────────

fn relu_1d(v: &mut [f32]) { for x in v.iter_mut() { *x = x.max(0.0); } }

/// Audio codec output.
pub struct AudioCodecOutput {
    /// Reconstructed waveform samples.
    pub reconstruction: Vec<f32>,
    /// Discrete codebook indices (one per frame).
    pub codes: Vec<usize>,
    /// Commitment loss.
    pub commitment_loss: f32,
}

/// Simplified WavTokenizer-style audio codec.
///
/// Encoder downsamples by 320× (factors: 2, 4, 5, 8).
/// At 24kHz: 75 frames/second, each → 1 codebook entry.
/// Single quantizer with 4096 codes.
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub struct AudioCodec {
    // Encoder: 1 → 32 → 64 → 128 → 256 → code_dim
    pub enc1: Conv1d,  // stride 2
    pub enc2: Conv1d,  // stride 4
    pub enc3: Conv1d,  // stride 5
    pub enc4: Conv1d,  // stride 8
    pub enc_proj: Conv1d, // 256 → code_dim, stride 1, kernel 1

    // VQ
    pub vq: VectorQuantizer,

    // Decoder: code_dim → 256 → 128 → 64 → 32 → 1
    pub dec_proj: Conv1d,          // code_dim → 256, kernel 1
    pub dec1: ConvTranspose1d,     // stride 8
    pub dec2: ConvTranspose1d,     // stride 5
    pub dec3: ConvTranspose1d,     // stride 4
    pub dec4: ConvTranspose1d,     // stride 2

    pub code_dim: usize,
    pub n_codes: usize,
    pub sample_rate: usize,
    pub downsample_factor: usize,  // 2 * 4 * 5 * 8 = 320
}

impl AudioCodec {
    /// Create audio codec for given sample rate.
    /// Default: 24kHz, 4096 codebook, 64-dim codes → 75 codes/sec.
    pub fn new(n_codes: usize, code_dim: usize, sample_rate: usize) -> Self {
        // Downsampling: 2 × 4 × 5 × 8 = 320
        Self {
            enc1: Conv1d::new(1, 32, 4, 2),
            enc2: Conv1d::new(32, 64, 8, 4),
            enc3: Conv1d::new(64, 128, 10, 5),
            enc4: Conv1d::new(128, 256, 16, 8),
            enc_proj: Conv1d::new(256, code_dim, 1, 1),

            vq: VectorQuantizer::new(n_codes, code_dim),

            dec_proj: Conv1d::new(code_dim, 256, 1, 1),
            dec1: ConvTranspose1d::new(256, 128, 16, 8),
            dec2: ConvTranspose1d::new(128, 64, 10, 5),
            dec3: ConvTranspose1d::new(64, 32, 8, 4),
            dec4: ConvTranspose1d::new(32, 1, 4, 2),

            code_dim,
            n_codes,
            sample_rate,
            downsample_factor: 320,
        }
    }

    /// Default 24kHz codec.
    pub fn new_24khz() -> Self { Self::new(4096, 64, 24000) }

    /// Encode waveform to latent vectors.
    /// Input: mono waveform samples [n_samples] as &[f32].
    /// Output: [n_frames × code_dim] flat.
    pub fn encode(&self, waveform: &[f32]) -> (Vec<f32>, usize) {
        let n = waveform.len();
        // Reshape to [1 × n] (mono channel)
        let (mut h, len) = self.enc1.forward(waveform, n);
        relu_1d(&mut h);
        let (mut h, len) = self.enc2.forward(&h, len);
        relu_1d(&mut h);
        let (mut h, len) = self.enc3.forward(&h, len);
        relu_1d(&mut h);
        let (mut h, len) = self.enc4.forward(&h, len);
        relu_1d(&mut h);
        let (z, len) = self.enc_proj.forward(&h, len);
        (z, len)
    }

    /// Full forward: encode → quantize → decode.
    pub fn forward(&self, waveform: &[f32]) -> AudioCodecOutput {
        let (z, n_frames) = self.encode(waveform);

        // Reshape z from [code_dim × n_frames] to [n_frames × code_dim] for VQ
        let mut z_flat = vec![0.0f32; n_frames * self.code_dim];
        for c in 0..self.code_dim {
            for t in 0..n_frames {
                z_flat[t * self.code_dim + c] = z[c * n_frames + t];
            }
        }

        let (quantized, codes) = self.vq.quantize(&z_flat);
        let commitment: f32 = z_flat.iter().zip(&quantized)
            .map(|(&a, &b)| (a - b).powi(2)).sum::<f32>()
            / z_flat.len().max(1) as f32;

        // Reshape quantized back to [code_dim × n_frames]
        let mut q_chan = vec![0.0f32; self.code_dim * n_frames];
        for c in 0..self.code_dim {
            for t in 0..n_frames {
                q_chan[c * n_frames + t] = quantized[t * self.code_dim + c];
            }
        }

        let reconstruction = self.decode(&q_chan, n_frames);
        AudioCodecOutput { reconstruction, codes, commitment_loss: commitment }
    }

    /// Decode from channel-first latent [code_dim × n_frames].
    pub fn decode(&self, quantized: &[f32], n_frames: usize) -> Vec<f32> {
        let (mut h, len) = self.dec_proj.forward(quantized, n_frames);
        relu_1d(&mut h);
        let (mut h, len) = self.dec1.forward(&h, len);
        relu_1d(&mut h);
        let (mut h, len) = self.dec2.forward(&h, len);
        relu_1d(&mut h);
        let (mut h, len) = self.dec3.forward(&h, len);
        relu_1d(&mut h);
        let (recon, _) = self.dec4.forward(&h, len);
        recon
    }

    /// Tokenize: waveform → code indices.
    pub fn tokenize(&self, waveform: &[f32]) -> Vec<usize> {
        self.forward(waveform).codes
    }

    /// Codes per second at the configured sample rate.
    pub fn codes_per_second(&self) -> f32 {
        self.sample_rate as f32 / self.downsample_factor as f32
    }

    /// Save. Format by extension: `.bin` → bincode, `.json` → JSON.
    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        modgrad_persist::persist::save(self, path).map_err(|e| e.into())
    }

    /// Load. Format by extension, with JSON fallback.
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        modgrad_persist::persist::load(path).map_err(|e| e.into())
    }

    pub fn n_params(&self) -> usize {
        self.enc1.param_count() + self.enc2.param_count()
            + self.enc3.param_count() + self.enc4.param_count()
            + self.enc_proj.param_count()
            + self.vq.codebook.len()
            + self.dec_proj.param_count()
            + self.dec1.weight.len() + self.dec1.bias.len()
            + self.dec2.weight.len() + self.dec2.bias.len()
            + self.dec3.weight.len() + self.dec3.bias.len()
            + self.dec4.weight.len() + self.dec4.bias.len()
    }
}

// Wire into the Tokenizer trait.
// For now, all codes are AudioSemantic (single codebook).
// When we add a proper semantic/acoustic split codec, semantic codes
// come from a separate VQ and acoustic from FSQ.
// NOTE: `impl Tokenizer for AudioCodec` lives in the main modgrad crate
// because it bridges modgrad-codec and modgrad-data (cross-crate integration).

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv1d_dims() {
        let conv = Conv1d::new(1, 32, 4, 2);
        let input = vec![0.1f32; 1000]; // 1 channel × 1000 samples
        let (out, len) = conv.forward(&input, 1000);
        assert_eq!(len, (1000 - 4) / 2 + 1); // = 499
        assert_eq!(out.len(), 32 * len);
    }

    #[test]
    fn audio_codec_forward() {
        let codec = AudioCodec::new(256, 16, 24000);
        // 1 second of 24kHz audio
        let waveform: Vec<f32> = (0..24000).map(|i| (i as f32 * 0.01).sin()).collect();
        let out = codec.forward(&waveform);

        let expected_codes = 24000 / 320; // = 75
        eprintln!("  audio codec params: {}", codec.n_params());
        eprintln!("  codes per second: {:.0}", codec.codes_per_second());
        eprintln!("  actual codes: {}", out.codes.len());
        eprintln!("  commitment loss: {:.4}", out.commitment_loss);
        eprintln!("  recon length: {}", out.reconstruction.len());

        // Should produce approximately 75 codes per second
        assert!((out.codes.len() as i64 - expected_codes as i64).abs() < 5,
            "expected ~{} codes, got {}", expected_codes, out.codes.len());
    }

    // audio_tokenizer_trait test lives in main modgrad crate (cross-crate integration)
}
