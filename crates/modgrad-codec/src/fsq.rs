//! Finite Scalar Quantization (FSQ) — VQ-VAE made simple.
//!
//! From Mentzer et al. 2023 and adapted from VoxCPM's internal bottleneck.
//! Each dimension is independently quantized to N uniform levels.
//! No codebook, no codebook collapse, no EMA updates.
//!
//! Use cases:
//! - Audio codec (Voxtral: 36 dims × 21 levels)
//! - Internal bottleneck (VoxCPM: 256 dims × 9 levels)
//! - Image tokenizer (alternative to VQ)
//!
//! The straight-through estimator passes gradients through the
//! rounding operation as if it weren't there.

/// Apply FSQ in-place: tanh → scale → round → rescale.
/// Each element is quantized to one of `levels` uniform values in [-1, 1].
///
/// Forward: x_q = round(tanh(x) * L/2) / (L/2)
/// Backward (STE): d_input = d_output (gradient passes through round)
#[inline]
pub fn fsq_forward(x: &mut [f32], levels: usize) {
    let half_l = (levels / 2) as f32;
    for v in x.iter_mut() {
        let bounded = v.tanh();
        *v = (bounded * half_l).round() / half_l;
    }
}

/// FSQ forward with caching for backward pass.
/// Returns the pre-quantization values (after tanh, before round).
pub fn fsq_forward_cached(x: &mut [f32], levels: usize) -> Vec<f32> {
    let half_l = (levels / 2) as f32;
    let mut pre_round = Vec::with_capacity(x.len());
    for v in x.iter_mut() {
        let bounded = v.tanh();
        pre_round.push(bounded);
        *v = (bounded * half_l).round() / half_l;
    }
    pre_round
}

/// FSQ backward (straight-through estimator + tanh derivative).
/// d_input = d_output * (1 - tanh(x)²)
pub fn fsq_backward(d_output: &[f32], pre_tanh: &[f32]) -> Vec<f32> {
    d_output.iter().zip(pre_tanh).map(|(&d, &x)| {
        let t = x.tanh();
        d * (1.0 - t * t) // tanh derivative
    }).collect()
}

/// Compute the quantization residual: what FSQ throws away.
/// residual = pre_quantized - post_quantized
/// This is what VoxCPM's RALM learns to recover.
pub fn fsq_residual(pre: &[f32], post: &[f32]) -> Vec<f32> {
    pre.iter().zip(post).map(|(&a, &b)| a - b).collect()
}

/// Compute the discrete code index for each FSQ value.
/// Maps the quantized value back to an integer in [0, levels).
pub fn fsq_to_index(quantized: f32, levels: usize) -> usize {
    let half_l = (levels / 2) as f32;
    let level = (quantized * half_l).round() as i32 + (levels / 2) as i32;
    level.clamp(0, levels as i32 - 1) as usize
}

/// Convert index back to FSQ value.
pub fn index_to_fsq(index: usize, levels: usize) -> f32 {
    let half_l = (levels / 2) as f32;
    (index as f32 - half_l) / half_l
}

/// Multi-dimensional FSQ: quantize a vector and return indices.
/// Each dimension gets its own level index.
pub fn fsq_encode(x: &[f32], levels: usize) -> (Vec<f32>, Vec<usize>) {
    let mut quantized = x.to_vec();
    fsq_forward(&mut quantized, levels);
    let indices: Vec<usize> = quantized.iter()
        .map(|&v| fsq_to_index(v, levels))
        .collect();
    (quantized, indices)
}

/// Decode FSQ indices back to values.
pub fn fsq_decode(indices: &[usize], levels: usize) -> Vec<f32> {
    indices.iter().map(|&i| index_to_fsq(i, levels)).collect()
}

/// Dithered FSQ (VoxCPM training recipe):
/// 50% quantize with FSQ, 25% add uniform noise, 25% pass through.
/// This schedule balances learning discrete and continuous representations.
pub fn fsq_dithered(x: &mut [f32], levels: usize, rng_state: &mut u64) -> bool {
    *rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
    let r = (*rng_state >> 48) as f32 / 65536.0;

    if r < 0.5 {
        // 50%: quantize
        fsq_forward(x, levels);
        true
    } else if r < 0.75 {
        // 25%: dither (add uniform noise of magnitude 1/L)
        let noise_scale = 1.0 / levels as f32;
        for v in x.iter_mut() {
            let bounded = v.tanh();
            *rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise = ((*rng_state >> 48) as f32 / 65536.0 * 2.0 - 1.0) * noise_scale;
            *v = bounded + noise;
        }
        false
    } else {
        // 25%: pass through (no quantization)
        for v in x.iter_mut() {
            *v = v.tanh();
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fsq_quantizes_to_levels() {
        let mut x = vec![0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0];
        fsq_forward(&mut x, 9); // 9 levels: -4/4, -3/4, ..., 0, ..., 3/4, 4/4

        // All values should be multiples of 1/4 (since half_l = 4)
        for &v in &x {
            let scaled = v * 4.0;
            assert!((scaled - scaled.round()).abs() < 1e-5,
                "FSQ output {} not a valid level", v);
        }
    }

    #[test]
    fn fsq_encode_decode_roundtrip() {
        let x = vec![0.3, -0.7, 0.0, 0.95];
        let (quantized, indices) = fsq_encode(&x, 21);
        let decoded = fsq_decode(&indices, 21);
        assert_eq!(quantized.len(), decoded.len());
        for (q, d) in quantized.iter().zip(&decoded) {
            assert!((q - d).abs() < 1e-5, "roundtrip failed: {} vs {}", q, d);
        }
    }

    #[test]
    fn fsq_residual_recovers_info() {
        let original = vec![0.3, -0.7, 0.123];
        let (quantized, _) = fsq_encode(&original, 5);
        let bounded: Vec<f32> = original.iter().map(|x| x.tanh()).collect();
        let residual = fsq_residual(&bounded, &quantized);

        // bounded = quantized + residual
        for i in 0..original.len() {
            assert!((bounded[i] - quantized[i] - residual[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn voxtral_config() {
        // Voxtral: 36 acoustic dims × 21 levels
        let x = vec![0.5f32; 36];
        let (_, indices) = fsq_encode(&x, 21);
        assert_eq!(indices.len(), 36);
        assert!(indices.iter().all(|&i| i < 21));
    }

    #[test]
    fn voxcpm_config() {
        // VoxCPM: 256 semantic dims × 9 levels
        let x = vec![0.1f32; 256];
        let (_, indices) = fsq_encode(&x, 9);
        assert_eq!(indices.len(), 256);
        assert!(indices.iter().all(|&i| i < 9));
    }
}
