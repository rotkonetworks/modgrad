//! Quantized vector dot products matching ggml's implementation.
//!
//! Q5_K × Q8_K vec_dot: the core operation for quantized inference.
//! Operates on quantized blocks directly — no dequantization to f32.
//! This is what makes llama.cpp/ollama produce correct logits where
//! our dequant→f32→dot approach failed.
//!
//! Reference: llamafile/llama.cpp/ggml/src/ggml-cpu/quants.c

/// Q8_K block: 256 int8 values + scale + block sums.
pub struct Q8KBlock {
    pub d: f32,           // scale
    pub qs: [i8; 256],    // quantized values
    pub bsums: [i16; 16], // block sums (sum of 16 consecutive qs)
}

/// Quantize f32 vector to Q8_K blocks.
/// Input length must be multiple of 256.
pub fn quantize_f32_to_q8k(x: &[f32]) -> Vec<Q8KBlock> {
    assert!(x.len() % 256 == 0);
    let nb = x.len() / 256;
    let mut blocks = Vec::with_capacity(nb);

    for i in 0..nb {
        let block = &x[i * 256..(i + 1) * 256];

        // Find max absolute value
        let mut amax: f32 = 0.0;
        let mut max: f32 = 0.0;
        for &v in block {
            let av = v.abs();
            if av > amax { amax = av; max = v; }
        }

        if amax == 0.0 {
            blocks.push(Q8KBlock { d: 0.0, qs: [0; 256], bsums: [0; 16] });
            continue;
        }

        let iscale = -127.0 / max;
        let mut qs = [0i8; 256];
        for j in 0..256 {
            let v = (iscale * block[j]).round() as i32;
            qs[j] = v.clamp(-127, 127) as i8;
        }

        // Block sums (sum of 16 consecutive values)
        let mut bsums = [0i16; 16];
        for j in 0..16 {
            let mut sum = 0i32;
            for k in 0..16 {
                sum += qs[j * 16 + k] as i32;
            }
            bsums[j] = sum as i16;
        }

        blocks.push(Q8KBlock { d: 1.0 / iscale, qs, bsums });
    }
    blocks
}

/// Compute dot product: Q5_K weight row × Q8_K quantized input.
/// `q5k_block_data`: raw Q5_K block bytes (176 bytes per block)
/// `q8k`: Q8_K blocks of the input vector
/// `n_blocks`: number of 256-element blocks
///
/// Returns the scalar dot product.
///
/// Matches ggml_vec_dot_q5_K_q8_K_generic exactly.
pub fn vec_dot_q5k_q8k(q5k_data: &[u8], q8k: &[Q8KBlock], n_blocks: usize) -> f32 {
    let kmask1: u32 = 0x3f3f3f3f;
    let kmask2: u32 = 0x0f0f0f0f;
    let kmask3: u32 = 0x03030303;

    let mut sums = [0.0f32; 8];
    let mut sumf: f32 = 0.0;

    for i in 0..n_blocks {
        let xb = &q5k_data[i * 176..];
        let yb = &q8k[i];

        // Dequantize Q5_K to int8 values (0-31 range)
        let q4 = &xb[48..176]; // qs[128]
        let hm = &xb[16..48];  // qh[32]
        let q8 = &yb.qs;

        let mut aux8 = [0i8; 256];
        let mut a_idx = 0usize;
        let mut m: u8 = 1;
        let mut q4_off = 0usize;
        for _j in 0..4 { // QK_K/64 = 4
            for l in 0..32 {
                aux8[a_idx + l] = (q4[q4_off + l] & 0xF) as i8;
            }
            for l in 0..32 {
                aux8[a_idx + l] += if hm[l] & m != 0 { 16 } else { 0 };
            }
            a_idx += 32;
            m <<= 1;
            for l in 0..32 {
                aux8[a_idx + l] = (q4[q4_off + l] >> 4) as i8;
            }
            for l in 0..32 {
                aux8[a_idx + l] += if hm[l] & m != 0 { 16 } else { 0 };
            }
            a_idx += 32;
            m <<= 1;
            q4_off += 32;
        }

        // Unpack scales and mins from the 12-byte packed format
        let mut utmp = [0u32; 4];
        utmp[0] = u32::from_le_bytes([xb[4], xb[5], xb[6], xb[7]]);
        utmp[1] = u32::from_le_bytes([xb[8], xb[9], xb[10], xb[11]]);
        utmp[2] = u32::from_le_bytes([xb[12], xb[13], xb[14], xb[15]]);

        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        let uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        let scales: &[u8; 16] = unsafe { &*(utmp.as_ptr() as *const [u8; 16]) };
        let mins: &[u8; 16] = unsafe { &*((utmp.as_ptr() as *const u8).add(8) as *const [u8; 16]) };

        // Compute sum of bsums * mins
        let mut sumi = 0i32;
        for j in 0..16 { // QK_K/16
            sumi += yb.bsums[j] as i32 * mins[j / 2] as i32;
        }

        // Integer dot product with scales
        let mut aux32 = [0i32; 8];
        let mut a_off = 0usize;
        let mut q8_off = 0usize;
        let mut is = 0usize;
        for _j in 0..8 { // QK_K/32
            let scale = scales[is] as i32;
            is += 1;
            for l in 0..8 { aux32[l] += scale * (q8[q8_off + l] as i32 * aux8[a_off + l] as i32); }
            q8_off += 8; a_off += 8;
            for l in 0..8 { aux32[l] += scale * (q8[q8_off + l] as i32 * aux8[a_off + l] as i32); }
            q8_off += 8; a_off += 8;
            for l in 0..8 { aux32[l] += scale * (q8[q8_off + l] as i32 * aux8[a_off + l] as i32); }
            q8_off += 8; a_off += 8;
            for l in 0..8 { aux32[l] += scale * (q8[q8_off + l] as i32 * aux8[a_off + l] as i32); }
            q8_off += 8; a_off += 8;
        }

        // Accumulate with float scales (clamp NaN block headers)
        let mut d_q5 = f16_to_f32(u16::from_le_bytes([xb[0], xb[1]]));
        let mut dmin_q5 = f16_to_f32(u16::from_le_bytes([xb[2], xb[3]]));
        if d_q5.is_nan() || d_q5.is_infinite() { d_q5 = 0.0; }
        if dmin_q5.is_nan() || dmin_q5.is_infinite() { dmin_q5 = 0.0; }
        let d = d_q5 * yb.d;
        for l in 0..8 { sums[l] += d * aux32[l] as f32; }
        let dmin = dmin_q5 * yb.d;
        sumf -= dmin * sumi as f32;
    }

    for l in 0..8 { sumf += sums[l]; }
    sumf
}

/// Q4_K × Q8_K vec_dot. Same pattern as Q5_K but simpler (no high bits).
pub fn vec_dot_q4k_q8k(q4k_data: &[u8], q8k: &[Q8KBlock], n_blocks: usize) -> f32 {
    let kmask1: u32 = 0x3f3f3f3f;
    let kmask2: u32 = 0x0f0f0f0f;
    let kmask3: u32 = 0x03030303;

    let mut sums = [0.0f32; 8];
    let mut sumf: f32 = 0.0;

    for i in 0..n_blocks {
        let xb = &q4k_data[i * 144..];
        let yb = &q8k[i];

        let q4 = &xb[16..144]; // qs[128]
        let q8 = &yb.qs;

        // Unpack scales
        let mut utmp = [0u32; 4];
        utmp[0] = u32::from_le_bytes([xb[4], xb[5], xb[6], xb[7]]);
        utmp[1] = u32::from_le_bytes([xb[8], xb[9], xb[10], xb[11]]);
        utmp[2] = u32::from_le_bytes([xb[12], xb[13], xb[14], xb[15]]);

        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        let uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        let scales: &[u8; 16] = unsafe { &*(utmp.as_ptr() as *const [u8; 16]) };
        let mins: &[u8; 16] = unsafe { &*((utmp.as_ptr() as *const u8).add(8) as *const [u8; 16]) };

        let mut sumi = 0i32;
        for j in 0..16 { sumi += yb.bsums[j] as i32 * mins[j / 2] as i32; }

        let mut aux32 = [0i32; 8];
        let mut q4_off = 0usize;
        let mut q8_off = 0usize;
        let mut is = 0usize;

        // Low nibbles (elements 0-127)
        for _j in 0..4 {
            let scale = scales[is] as i32; is += 1;
            for l in 0..8 { aux32[l] += scale * (q8[q8_off + l] as i32 * (q4[q4_off + l] & 0xF) as i32); }
            q8_off += 8;
            for l in 0..8 { aux32[l] += scale * (q8[q8_off + l] as i32 * (q4[q4_off + 8 + l] & 0xF) as i32); }
            q8_off += 8;
            for l in 0..8 { aux32[l] += scale * (q8[q8_off + l] as i32 * (q4[q4_off + 16 + l] & 0xF) as i32); }
            q8_off += 8;
            for l in 0..8 { aux32[l] += scale * (q8[q8_off + l] as i32 * (q4[q4_off + 24 + l] & 0xF) as i32); }
            q8_off += 8;
            q4_off += 32;
        }
        // High nibbles (elements 128-255)
        q4_off = 0;
        for _j in 0..4 {
            let scale = scales[is] as i32; is += 1;
            for l in 0..8 { aux32[l] += scale * (q8[q8_off + l] as i32 * (q4[q4_off + l] >> 4) as i32); }
            q8_off += 8;
            for l in 0..8 { aux32[l] += scale * (q8[q8_off + l] as i32 * (q4[q4_off + 8 + l] >> 4) as i32); }
            q8_off += 8;
            for l in 0..8 { aux32[l] += scale * (q8[q8_off + l] as i32 * (q4[q4_off + 16 + l] >> 4) as i32); }
            q8_off += 8;
            for l in 0..8 { aux32[l] += scale * (q8[q8_off + l] as i32 * (q4[q4_off + 24 + l] >> 4) as i32); }
            q8_off += 8;
            q4_off += 32;
        }

        let mut d_q4 = f16_to_f32(u16::from_le_bytes([xb[0], xb[1]]));
        let mut dmin_q4 = f16_to_f32(u16::from_le_bytes([xb[2], xb[3]]));
        if d_q4.is_nan() || d_q4.is_infinite() { d_q4 = 0.0; }
        if dmin_q4.is_nan() || dmin_q4.is_infinite() { dmin_q4 = 0.0; }
        let d = d_q4 * yb.d;
        for l in 0..8 { sums[l] += d * aux32[l] as f32; }
        let dmin = dmin_q4 * yb.d;
        sumf -= dmin * sumi as f32;
    }

    for l in 0..8 { sumf += sums[l]; }
    sumf
}

/// Q6_K × Q8_K vec_dot. Matches `ggml_vec_dot_q6_K_q8_K_generic`.
///
/// Q6_K block (210 bytes / 256 elems): `ql[128] | qh[64] | scales[16] (i8) | d (f16)`.
/// Unlike Q4_K/Q5_K there is **no** min/dmin — the 6-bit quants are signed
/// (offset −32) and each 16-element sub-block has its own signed int8 scale.
pub fn vec_dot_q6k_q8k(q6k_data: &[u8], q8k: &[Q8KBlock], n_blocks: usize) -> f32 {
    let mut sums = [0.0f32; 8];
    let mut sumf = 0.0f32;

    for i in 0..n_blocks {
        let xb = &q6k_data[i * 210..];
        let ql = &xb[0..128];
        let qh = &xb[128..192];
        let scales = &xb[192..208]; // 16 signed int8 scales
        let q8 = &q8k[i].qs;

        // Dequantize 256 6-bit quants to signed int8 (offset −32). Two
        // groups of 128, mirroring ggml's `for j in 0..QK_K step 128`.
        let mut aux8 = [0i8; 256];
        for grp in 0..2 {
            let ab = grp * 128;
            let qlb = grp * 64;
            let qhb = grp * 32;
            for l in 0..32 {
                let q4l0 = ql[qlb + l] as i32;
                let q4l32 = ql[qlb + l + 32] as i32;
                let h = qh[qhb + l] as i32;
                aux8[ab + l] = (((q4l0 & 0xF) | (((h) & 3) << 4)) - 32) as i8;
                aux8[ab + l + 32] = (((q4l32 & 0xF) | (((h >> 2) & 3) << 4)) - 32) as i8;
                aux8[ab + l + 64] = (((q4l0 >> 4) | (((h >> 4) & 3) << 4)) - 32) as i8;
                aux8[ab + l + 96] = (((q4l32 >> 4) | (((h >> 6) & 3) << 4)) - 32) as i8;
            }
        }

        // Integer dot with per-16-element signed scales (16 sub-blocks).
        let mut aux32 = [0i32; 8];
        let mut a_off = 0usize;
        let mut q8_off = 0usize;
        for j in 0..16 {
            let scale = (scales[j] as i8) as i32;
            for l in 0..8 { aux32[l] += scale * (q8[q8_off + l] as i32 * aux8[a_off + l] as i32); }
            q8_off += 8; a_off += 8;
            for l in 0..8 { aux32[l] += scale * (q8[q8_off + l] as i32 * aux8[a_off + l] as i32); }
            q8_off += 8; a_off += 8;
        }

        let mut d_q6 = f16_to_f32(u16::from_le_bytes([xb[208], xb[209]]));
        if d_q6.is_nan() || d_q6.is_infinite() { d_q6 = 0.0; }
        let d = d_q6 * q8k[i].d;
        for l in 0..8 { sums[l] += d * aux32[l] as f32; }
    }

    for l in 0..8 { sumf += sums[l]; }
    sumf
}

/// Quantized matvec: y = Q_weight @ x_f32.
/// Quantizes x to Q8_K first, then uses quantized vec_dot per row.
/// This produces correct results where dequant→f32→dot fails.
pub fn qmatvec(
    weight_data: &[u8],     // raw K-quant data, [out_dim * blocks_per_row * block_size] bytes
    x: &[f32],              // f32 input vector [in_dim]
    y: &mut [f32],          // output [out_dim]
    out_dim: usize,
    in_dim: usize,
    block_size: usize,      // 144 Q4_K, 176 Q5_K, 210 Q6_K
    is_q5k: bool,
) {
    let blocks_per_row = in_dim / 256;
    let row_bytes = blocks_per_row * block_size;

    // Pad input to multiple of 256 and quantize to Q8_K
    let padded_len = (in_dim + 255) & !255;
    let mut x_padded = vec![0.0f32; padded_len];
    x_padded[..in_dim].copy_from_slice(x);
    let q8k = quantize_f32_to_q8k(&x_padded);

    for row in 0..out_dim {
        let row_data = &weight_data[row * row_bytes..row * row_bytes + row_bytes];
        y[row] = if block_size == 210 {
            vec_dot_q6k_q8k(row_data, &q8k, blocks_per_row)
        } else if is_q5k {
            vec_dot_q5k_q8k(row_data, &q8k, blocks_per_row)
        } else {
            vec_dot_q4k_q8k(row_data, &q8k, blocks_per_row)
        };
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;
    if exp == 0 {
        if mant == 0 { return if sign == 1 { -0.0 } else { 0.0 }; }
        let m = mant as f32 / 1024.0 * 2.0f32.powi(-14);
        return if sign == 1 { -m } else { m };
    }
    if exp == 31 { return if mant != 0 { f32::NAN } else if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY }; }
    f32::from_bits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q8k_roundtrip() {
        let x: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();
        let q8k = quantize_f32_to_q8k(&x);
        assert_eq!(q8k.len(), 1);
        assert!(q8k[0].d.abs() > 0.0);
        // Check bsums
        let sum: i32 = q8k[0].qs.iter().map(|&v| v as i32).sum();
        let bsum: i32 = q8k[0].bsums.iter().map(|&v| v as i32).sum();
        assert_eq!(sum, bsum);
    }

    /// Q6_K dot, uniform case: ql=qh=0 → every 6-bit quant = (0)−32 = −32,
    /// scales=1, d=1. Dotted with x = all 1.0 → 256·(−32)·1 = −8192.
    #[test]
    fn test_q6k_vec_dot_uniform() {
        let mut block = vec![0u8; 210];
        for s in &mut block[192..208] { *s = 1; } // scales = 1
        block[208] = 0x00; // f16(1.0) = 0x3C00
        block[209] = 0x3C;
        let x = vec![1.0f32; 256];
        let q8k = quantize_f32_to_q8k(&x);
        let dot = vec_dot_q6k_q8k(&block, &q8k, 1);
        assert!((dot + 8192.0).abs() < 1.0, "q6k uniform dot = {dot}, expected -8192");
    }

    /// Q6_K dot exercising the low-nibble + high-2-bit unpacking for one
    /// element: ql[0]=5, qh[0]=1 → quant[0] = (5 | (1<<4)) − 32 = −11; the
    /// other 255 quants stay −32. Dot with x=all 1.0 → −11 + 255·(−32) = −8171.
    #[test]
    fn test_q6k_vec_dot_unpack_one() {
        let mut block = vec![0u8; 210];
        block[0] = 0x05; // ql[0] low nibble = 5
        block[128] = 0x01; // qh[0] bits 0-1 = 1
        for s in &mut block[192..208] { *s = 1; }
        block[208] = 0x00;
        block[209] = 0x3C;
        let x = vec![1.0f32; 256];
        let q8k = quantize_f32_to_q8k(&x);
        let dot = vec_dot_q6k_q8k(&block, &q8k, 1);
        assert!((dot + 8171.0).abs() < 1.0, "q6k unpack dot = {dot}, expected -8171");
    }
}
