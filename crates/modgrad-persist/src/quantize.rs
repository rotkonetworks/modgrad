//! Key quantization: f32 ↔ f16 ↔ i8
//!
//! f32: reference (lossless)
//! f16: half-precision (lossless cosine similarity for normalized keys)
//! i8:  8-bit scaled (±0.001 sim error, 4× compression)

use half::f16;

/// Quantize f32 key to f16 (IEEE 754 half-precision).
pub fn f32_to_f16(key: &[f32]) -> Vec<u16> {
    key.iter().map(|&v| f16::from_f32(v).to_bits()).collect()
}

/// Dequantize f16 key back to f32.
pub fn f16_to_f32(data: &[u16]) -> Vec<f32> {
    data.iter().map(|&bits| f16::from_bits(bits).to_f32()).collect()
}

/// Quantize f32 key to i8 with symmetric scaling.
/// Returns (quantized_data, scale_factor).
/// To dequantize: real_value = i8_value * scale
pub fn f32_to_i8(key: &[f32]) -> (Vec<i8>, f32) {
    let max_abs = key.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    if max_abs == 0.0 {
        return (vec![0i8; key.len()], 1.0);
    }
    let scale = max_abs / 127.0;
    let data = key
        .iter()
        .map(|&v| (v / scale).round().clamp(-127.0, 127.0) as i8)
        .collect();
    (data, scale)
}

/// Dequantize i8 key back to f32.
pub fn i8_to_f32(data: &[i8], scale: f32) -> Vec<f32> {
    data.iter().map(|&v| v as f32 * scale).collect()
}

/// Cosine similarity between two f32 keys.
pub fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 { return 0.0; }
    dot / (na * nb)
}

/// Cosine similarity directly on i8 keys (no dequantization needed).
/// Scale factors cancel out in cosine similarity.
pub fn cosine_i8(a: &[i8], b: &[i8]) -> f32 {
    let dot: i64 = a.iter().zip(b).map(|(&x, &y)| x as i64 * y as i64).sum();
    let na: i64 = a.iter().map(|&x| x as i64 * x as i64).sum();
    let nb: i64 = b.iter().map(|&x| x as i64 * x as i64).sum();
    let denom = (na as f64).sqrt() * (nb as f64).sqrt();
    if denom == 0.0 { return 0.0; }
    (dot as f64 / denom) as f32
}

/// Key format for serialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyFormat {
    F32,
    F16,
    I8,
}

impl KeyFormat {
    pub fn bytes_per_key(&self, dim: usize) -> usize {
        match self {
            KeyFormat::F32 => dim * 4,
            KeyFormat::F16 => dim * 2,
            KeyFormat::I8 => dim + 4, // data + scale
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_roundtrip() {
        let key: Vec<f32> = (0..896).map(|i| (i as f32 / 896.0) - 0.5).collect();
        let quantized = f32_to_f16(&key);
        let restored = f16_to_f32(&quantized);
        let sim = cosine_f32(&key, &restored);
        assert!(sim > 0.9999, "f16 roundtrip sim = {sim}");
    }

    #[test]
    fn i8_roundtrip() {
        let key: Vec<f32> = (0..896).map(|i| (i as f32 / 896.0) - 0.5).collect();
        let (quantized, scale) = f32_to_i8(&key);
        let restored = i8_to_f32(&quantized, scale);
        let sim = cosine_f32(&key, &restored);
        assert!(sim > 0.999, "i8 roundtrip sim = {sim}");
    }

    #[test]
    fn i8_direct_cosine() {
        let a: Vec<f32> = (0..896).map(|i| (i as f32 / 896.0) - 0.5).collect();
        let b: Vec<f32> = (0..896).map(|i| (i as f32 / 896.0) - 0.3).collect();

        let sim_f32 = cosine_f32(&a, &b);
        let (qa, _) = f32_to_i8(&a);
        let (qb, _) = f32_to_i8(&b);
        let sim_i8 = cosine_i8(&qa, &qb);

        assert!((sim_f32 - sim_i8).abs() < 0.002, "i8 sim error = {}", (sim_f32 - sim_i8).abs());
    }
}
