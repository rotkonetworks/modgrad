//! RMS normalization.
//!
//! `y = x / sqrt(mean(x²) + eps)`
//!
//! No mean subtraction, no learnable parameters.
//! Distinct from the existing `layer_norm_inplace` which does mean subtraction.

use super::tensor::Tensor1;
use super::dims::ModelDim;

/// RMS normalization (no learnable params).
pub struct RmsNorm {
    eps: f32,
}

impl RmsNorm {
    pub fn new(eps: f32) -> Self {
        Self { eps }
    }

    /// Normalize `src` into `dst`. Separate buffers to avoid aliasing.
    #[inline]
    pub fn forward(&self, src: &[f32], dst: &mut [f32]) {
        debug_assert_eq!(src.len(), dst.len());
        let n = src.len() as f32;
        let ss: f32 = src.iter().map(|x| x * x).sum();
        let inv_rms = 1.0 / (ss / n + self.eps).sqrt();
        for (d, s) in dst.iter_mut().zip(src.iter()) {
            *d = s * inv_rms;
        }
    }

    /// In-place variant — only use when you are certain the buffer
    /// is not simultaneously read elsewhere (e.g., residual add already done).
    #[inline]
    pub fn forward_inplace(&self, x: &mut [f32]) {
        let n = x.len() as f32;
        let ss: f32 = x.iter().map(|v| v * v).sum();
        let inv_rms = 1.0 / (ss / n + self.eps).sqrt();
        for v in x.iter_mut() {
            *v *= inv_rms;
        }
    }
}

/// RMS norm with a learnable scale vector (for final norm before lm_head).
pub struct ScaledRmsNorm {
    pub norm: RmsNorm,
    pub scale: Tensor1<ModelDim>,
}

impl ScaledRmsNorm {
    pub fn new(scale: Vec<f32>, eps: f32) -> Self {
        Self {
            norm: RmsNorm::new(eps),
            scale: Tensor1::new(scale),
        }
    }

    /// Normalize and scale: `y = (x / rms(x)) * scale`.
    pub fn forward(&self, src: &[f32], dst: &mut [f32]) {
        debug_assert_eq!(src.len(), dst.len());
        debug_assert_eq!(src.len(), self.scale.len());
        let n = src.len() as f32;
        let ss: f32 = src.iter().map(|x| x * x).sum();
        let inv_rms = 1.0 / (ss / n + self.norm.eps).sqrt();
        let scale = self.scale.as_slice();
        for i in 0..src.len() {
            dst[i] = src[i] * inv_rms * scale[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm_unit_variance() {
        let norm = RmsNorm::new(1e-5);
        let src = vec![1.0, 2.0, 3.0, 4.0];
        let mut dst = vec![0.0; 4];
        norm.forward(&src, &mut dst);

        // After RMS norm, sum(x²)/n should be ~1.0
        let ss: f32 = dst.iter().map(|x| x * x).sum::<f32>() / dst.len() as f32;
        assert!((ss - 1.0).abs() < 1e-4, "rms² = {ss}");
    }

    #[test]
    fn test_rms_norm_inplace_matches() {
        let norm = RmsNorm::new(1e-5);
        let src = vec![0.5, -1.0, 2.0, -0.3];
        let mut dst = vec![0.0; 4];
        norm.forward(&src, &mut dst);

        let mut inplace = src.clone();
        norm.forward_inplace(&mut inplace);

        for (a, b) in dst.iter().zip(inplace.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
