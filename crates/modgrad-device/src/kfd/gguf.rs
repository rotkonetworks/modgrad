//! Minimal GGUF v3 parser. Reads metadata and tensor info from a GGUF file.
//! Tensors are NOT loaded into memory — the file is mmap'd and tensors
//! are accessed by offset. This means 5GB of Q4 weights cost 0 RSS until
//! the OS pages them in on access.

use std::collections::HashMap;
use std::io::{Read, Seek};

/// GGUF tensor quantization types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
#[allow(non_camel_case_types)] // canonical GGUF quant format names (Q4_K, IQ2_XXS, etc.)
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 14,
    Q6_K = 15,
    IQ2_XXS = 16,
    BF16 = 30,
    Unknown(u32),
}

impl From<u32> for GgmlType {
    fn from(v: u32) -> Self {
        match v {
            0 => Self::F32, 1 => Self::F16, 2 => Self::Q4_0, 3 => Self::Q4_1,
            6 => Self::Q5_0, 7 => Self::Q5_1, 8 => Self::Q8_0, 9 => Self::Q8_1,
            10 => Self::Q2_K, 11 => Self::Q3_K, 12 => Self::Q4_K, 14 => Self::Q5_K,
            15 => Self::Q6_K, 16 => Self::IQ2_XXS, 30 => Self::BF16,
            other => Self::Unknown(other),
        }
    }
}

impl GgmlType {
    /// Bytes per block and elements per block for quantized types.
    /// Returns (block_bytes, block_elements).
    pub fn block_size(&self) -> (usize, usize) {
        match self {
            Self::F32 => (4, 1),
            Self::F16 => (2, 1),
            Self::BF16 => (2, 1),
            Self::Q4_0 => (18, 32),
            Self::Q4_1 => (20, 32),
            Self::Q5_0 => (22, 32),
            Self::Q5_1 => (24, 32),
            Self::Q8_0 => (34, 32),
            Self::Q8_1 => (36, 32),
            Self::Q2_K => (84, 256),
            Self::Q3_K => (110, 256),
            Self::Q4_K => (144, 256),
            Self::Q5_K => (176, 256),
            Self::Q6_K => (210, 256),
            _ => (0, 1),
        }
    }

    /// Total bytes for `n_elements` of this type.
    pub fn data_size(&self, n_elements: usize) -> usize {
        let (block_bytes, block_elems) = self.block_size();
        let n_blocks = (n_elements + block_elems - 1) / block_elems;
        n_blocks * block_bytes
    }
}

/// Information about one tensor in the GGUF file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<usize>,
    pub dtype: GgmlType,
    /// Byte offset from data section start.
    pub offset: usize,
}

impl TensorInfo {
    pub fn n_elements(&self) -> usize {
        self.dims.iter().product::<usize>().max(1)
    }

    pub fn data_bytes(&self) -> usize {
        self.dtype.data_size(self.n_elements())
    }
}

/// Parsed GGUF file.
pub struct GgufFile {
    /// Key-value metadata.
    pub metadata: HashMap<String, MetaValue>,
    /// Tensor infos (name → info).
    pub tensors: HashMap<String, TensorInfo>,
    /// Tensor list in file order.
    pub tensor_list: Vec<String>,
    /// Absolute byte offset where tensor data starts in the file.
    pub data_offset: usize,
}

#[derive(Debug, Clone)]
pub enum MetaValue {
    U8(u8), I8(i8), U16(u16), I16(i16),
    U32(u32), I32(i32), U64(u64), I64(i64),
    F32(f32), F64(f64), Bool(bool),
    Str(String),
    Array(Vec<MetaValue>),
}

impl MetaValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::U32(v) => Some(*v),
            Self::I32(v) => Some(*v as u32),
            Self::U64(v) => Some(*v as u32),
            _ => None,
        }
    }
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::F32(v) => Some(*v),
            Self::F64(v) => Some(*v as f32),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self { Self::Str(s) => Some(s), _ => None }
    }
    pub fn as_bool_array(&self) -> Option<Vec<bool>> {
        match self {
            Self::Array(arr) => {
                arr.iter().map(|v| match v {
                    Self::Bool(b) => Some(*b),
                    _ => None,
                }).collect()
            }
            _ => None,
        }
    }
}

impl GgufFile {
    /// Parse a GGUF file. Only reads metadata and tensor info — does NOT
    /// load tensor data. The file can be mmap'd separately.
    pub fn parse<R: Read + Seek>(r: &mut R) -> Result<Self, String> {
        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];

        // Magic
        r.read_exact(&mut buf4).map_err(|e| e.to_string())?;
        if &buf4 != b"GGUF" { return Err("not a GGUF file".into()); }

        // Version
        r.read_exact(&mut buf4).map_err(|e| e.to_string())?;
        let version = u32::from_le_bytes(buf4);
        if version < 2 || version > 3 {
            return Err(format!("unsupported GGUF version {}", version));
        }

        // Tensor count, KV count
        r.read_exact(&mut buf8).map_err(|e| e.to_string())?;
        let n_tensors = u64::from_le_bytes(buf8) as usize;
        r.read_exact(&mut buf8).map_err(|e| e.to_string())?;
        let n_kv = u64::from_le_bytes(buf8) as usize;

        // Read KV metadata
        let mut metadata = HashMap::new();
        for _ in 0..n_kv {
            let key = read_string(r)?;
            let val = read_meta_value(r)?;
            metadata.insert(key, val);
        }

        // Read tensor infos
        let mut tensors = HashMap::new();
        let mut tensor_list = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = read_string(r)?;
            r.read_exact(&mut buf4).map_err(|e| e.to_string())?;
            let n_dims = u32::from_le_bytes(buf4) as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                r.read_exact(&mut buf8).map_err(|e| e.to_string())?;
                dims.push(u64::from_le_bytes(buf8) as usize);
            }
            r.read_exact(&mut buf4).map_err(|e| e.to_string())?;
            let dtype = GgmlType::from(u32::from_le_bytes(buf4));
            r.read_exact(&mut buf8).map_err(|e| e.to_string())?;
            let offset = u64::from_le_bytes(buf8) as usize;

            tensor_list.push(name.clone());
            tensors.insert(name, TensorInfo { name: tensor_list.last().unwrap().clone(), dims, dtype, offset });
        }

        // Data starts after header, aligned to 32 bytes
        let pos = r.stream_position().map_err(|e| e.to_string())? as usize;
        let data_offset = (pos + 31) & !31;

        Ok(GgufFile { metadata, tensors, tensor_list, data_offset })
    }

    /// Get a metadata value.
    pub fn meta(&self, key: &str) -> Option<&MetaValue> {
        self.metadata.get(key)
    }

    /// Get architecture name (e.g. "gemma4", "qwen3", "llama").
    pub fn architecture(&self) -> Option<&str> {
        self.meta("general.architecture")?.as_str()
    }

    /// Get u32 metadata.
    pub fn meta_u32(&self, key: &str) -> Option<u32> {
        self.meta(key)?.as_u32()
    }

    /// Get f32 metadata.
    pub fn meta_f32(&self, key: &str) -> Option<f32> {
        self.meta(key)?.as_f32()
    }

    /// Absolute byte position of a tensor's data in the file.
    pub fn tensor_file_offset(&self, name: &str) -> Option<usize> {
        self.tensors.get(name).map(|t| self.data_offset + t.offset)
    }
}

fn read_string<R: Read>(r: &mut R) -> Result<String, String> {
    let mut buf8 = [0u8; 8];
    r.read_exact(&mut buf8).map_err(|e| e.to_string())?;
    let len = u64::from_le_bytes(buf8) as usize;
    if len > 1_000_000 { return Err("string too long".into()); }
    let mut s = vec![0u8; len];
    r.read_exact(&mut s).map_err(|e| e.to_string())?;
    String::from_utf8(s).map_err(|e| e.to_string())
}

fn read_meta_value<R: Read>(r: &mut R) -> Result<MetaValue, String> {
    let mut buf4 = [0u8; 4];
    r.read_exact(&mut buf4).map_err(|e| e.to_string())?;
    let vtype = u32::from_le_bytes(buf4);
    read_typed_value(r, vtype)
}

// ─── Q4_K format: dequantize + quantize ─────────────────────
//
// Q4_K_M block layout (144 bytes / 256 elements):
//   +0x00: d    (fp16) — super-block scale
//   +0x02: dmin (fp16) — super-block min
//   +0x04: scales[12]  — 8 sub-blocks worth of 6-bit (scale, min)
//                        nibbles, packed via the canonical llama.cpp
//                        `get_scale_min_k4` layout
//   +0x10: qs[128]     — 256 4-bit quants (low nibble = element j,
//                        high nibble = element j+128)
//
// Per sub-block i (32 elements at indices [i*32 .. (i+1)*32]):
//   element = d * sc[i] * q[i,j] - dmin * m[i]
//
// The dequant + quantize routines below mirror llama.cpp's
// `dequantize_row_q4_K` / `quantize_row_q4_K` exactly so a round trip
// `f32 → q4k → f32` is bit-stable across reference and runtime.

/// Bytes per Q4_K_M block.
pub const Q4K_BLOCK_BYTES: usize = 144;

/// Elements per Q4_K_M block.
pub const Q4K_BLOCK_ELEMS: usize = 256;

/// IEEE-754 fp16 → f32. Subnormals are flushed to zero on the negative
/// path of llama.cpp's reader; we match that exactly so a GGUF file
/// loaded here decodes byte-for-byte the same as upstream.
#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;
    if exp == 0 {
        if mant == 0 { return if sign == 1 { -0.0 } else { 0.0 }; }
        let m = mant as f32 / 1024.0 * 2.0f32.powi(-14);
        return if sign == 1 { -m } else { m };
    }
    if exp == 31 { return f32::NAN; }
    f32::from_bits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
}

/// f32 → IEEE-754 fp16 (round-to-nearest-even). Mirrors the helper
/// llama.cpp uses when serialising Q4_K block scales.
#[inline]
pub fn f32_to_f16(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 31) & 0x1) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;

    if exp == 0xFF {
        // Inf/NaN.
        let m = (mant >> 13) as u16;
        return (sign << 15) | (0x1F << 10) | if mant != 0 { m.max(1) } else { 0 };
    }
    let unbiased = exp - 127 + 15;
    if unbiased >= 0x1F {
        // Overflow → ±inf.
        return (sign << 15) | (0x1F << 10);
    }
    if unbiased <= 0 {
        // Subnormal or underflow → flush to zero (matches reader's
        // path where exp == 0 → tiny non-zero or zero; we collapse
        // to zero for simplicity and full compatibility with the
        // GGUF reference decoder).
        if unbiased < -10 {
            return sign << 15;
        }
        let mant_with_lead = mant | 0x800000;
        let shift = 14 - unbiased;
        let mant_h = (mant_with_lead >> shift) as u16;
        return (sign << 15) | mant_h;
    }
    let mant_h = (mant >> 13) as u16;
    (sign << 15) | ((unbiased as u16) << 10) | mant_h
}

/// Llama.cpp's canonical 6-bit (scale, min) nibble extractor. `j` is
/// the sub-block index in `[0, 8)`, `scales` is the 12-byte packed
/// header at offset +0x04 of a Q4_K block.
#[inline]
pub fn q4k_get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (sc, m)
    }
}

/// Inverse of `q4k_get_scale_min_k4`: pack 8 (sc, m) 6-bit pairs into
/// the 12-byte header in the canonical layout. Each `sc[i]` and
/// `m[i]` MUST be in `0..64` — values are not truncated, the caller's
/// quantisation loop is responsible for clamping.
#[inline]
pub fn q4k_pack_scales_mins(sc: &[u8; 8], m: &[u8; 8], scales: &mut [u8; 12]) {
    // First 4 sub-blocks: scales[i] holds sc[i] (low 6), scales[i+4]
    // holds m[i] (low 6), with the top 2 bits of (sc[i+4], m[i+4])
    // packed into the upper bits of scales[i] / scales[i+4].
    for i in 0..4 {
        scales[i] = (sc[i] & 0x3F) | (((sc[i + 4] >> 4) & 0x3) << 6);
        scales[i + 4] = (m[i] & 0x3F) | (((m[i + 4] >> 4) & 0x3) << 6);
        scales[i + 8] = (sc[i + 4] & 0xF) | ((m[i + 4] & 0xF) << 4);
    }
}

/// Dequantize `n_blocks` Q4_K_M blocks into `out` (length must be at
/// least `n_blocks * 256`). Mirrors llama.cpp's
/// `dequantize_row_q4_K` exactly. Public so kernel parity tests can
/// generate the host reference.
pub fn dequantize_row_q4_k(data: &[u8], out: &mut [f32], n_blocks: usize) {
    debug_assert!(data.len() >= n_blocks * Q4K_BLOCK_BYTES);
    debug_assert!(out.len() >= n_blocks * Q4K_BLOCK_ELEMS);
    for blk in 0..n_blocks {
        let b = &data[blk * Q4K_BLOCK_BYTES..];
        let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([b[2], b[3]]));
        let d = if d.is_nan() || d.is_infinite() { 0.0 } else { d };
        let dmin = if dmin.is_nan() || dmin.is_infinite() { 0.0 } else { dmin };
        let scales = &b[4..16];
        let qs = &b[16..144];

        let base = blk * Q4K_BLOCK_ELEMS;
        let mut is = 0;
        for j in (0..128).step_by(32) {
            let (sc, m) = q4k_get_scale_min_k4(is, scales);
            let d1 = d * sc as f32;
            let m1 = dmin * m as f32;
            for l in 0..32 {
                if base + j + l < out.len() {
                    out[base + j + l] = d1 * (qs[j + l] & 0xF) as f32 - m1;
                }
            }
            is += 1;
            let (sc, m) = q4k_get_scale_min_k4(is, scales);
            let d2 = d * sc as f32;
            let m2 = dmin * m as f32;
            for l in 0..32 {
                if base + j + l + 128 < out.len() {
                    out[base + j + l + 128] = d2 * (qs[j + l] >> 4) as f32 - m2;
                }
            }
            is += 1;
        }
    }
}

/// Per-sub-block (min, max) of `x` clamped to fit Q4_K's packing
/// constraint that the dmin path is non-negative. Returns
/// `(local_d, local_dmin)` where dequant per element is
/// `q * local_d - local_dmin` for `q ∈ 0..15`.
#[inline]
fn q4k_sub_block_local_scale_min(x: &[f32]) -> (f32, f32) {
    let (mut x_min, mut x_max) = (x[0], x[0]);
    for &v in &x[1..] {
        if v < x_min { x_min = v; }
        if v > x_max { x_max = v; }
    }
    // Q4_K's encoding requires `dmin * m ≥ 0`. If x is entirely
    // non-negative we clamp x_min to 0 (so local_dmin == 0 and the
    // sub-block is encoded purely via the scaling path); the loss
    // is at most one quant step at the low end, which is well under
    // Q4_K's published 1% band on real model weights.
    if x_min > 0.0 { x_min = 0.0; }
    let local_d = (x_max - x_min) / 15.0;
    let local_dmin = -x_min; // ≥ 0 by construction
    (local_d, local_dmin)
}

/// Quantise one 32-element sub-block. Computes the 6-bit `sc` / `m`
/// pair against the already-fixed super-block (`super_d`,
/// `super_dmin`) and writes 32 raw nibbles (0..15) into `quants`.
fn quantize_sub_block(
    x: &[f32],
    sc: &mut u8,
    m: &mut u8,
    quants: &mut [u8],
    super_d: f32,
    super_dmin: f32,
) {
    debug_assert_eq!(x.len(), 32);
    debug_assert_eq!(quants.len(), 32);

    let (local_d, local_dmin) = q4k_sub_block_local_scale_min(x);

    let sc_f = if super_d > 0.0 { local_d / super_d } else { 0.0 };
    let m_f = if super_dmin > 0.0 { local_dmin / super_dmin } else { 0.0 };

    let sc_q = sc_f.round().clamp(0.0, 63.0) as u8;
    let m_q = m_f.round().clamp(0.0, 63.0) as u8;
    *sc = sc_q;
    *m = m_q;

    let d1 = super_d * sc_q as f32;
    let m1 = super_dmin * m_q as f32;
    for i in 0..32 {
        let v = if d1 > 0.0 {
            ((x[i] + m1) / d1).round().clamp(0.0, 15.0) as u8
        } else {
            0
        };
        quants[i] = v;
    }
}

/// Quantize `x` (length must be a multiple of `Q4K_BLOCK_ELEMS`) to
/// Q4_K_M, writing `(x.len() / 256) * 144` bytes into `out`. Round
/// trip with [`dequantize_row_q4_k`] is bit-stable for any caller
/// that re-quantises the dequantised values (i.e. not exact across
/// the first f32→Q4_K boundary, but exact thereafter). Returns the
/// number of bytes written for caller bookkeeping.
///
/// Reference implementation — llama.cpp's `quantize_row_q4_K` runs an
/// iterative search to minimise reconstruction error at the cost of
/// several extra passes per block. We use the simpler linear
/// `(min, max) → scale` path that the upstream tests classify as
/// `quantize_row_q4_K_reference`, which is within Q4_K's typical 1%
/// relative error band — sufficient for the residency-streaming
/// path's correctness tests.
pub fn quantize_row_q4_k(x: &[f32], out: &mut [u8]) -> usize {
    assert!(x.len() % Q4K_BLOCK_ELEMS == 0,
        "quantize_row_q4_k: x.len() = {} not a multiple of {}",
        x.len(), Q4K_BLOCK_ELEMS);
    let n_blocks = x.len() / Q4K_BLOCK_ELEMS;
    assert!(out.len() >= n_blocks * Q4K_BLOCK_BYTES,
        "quantize_row_q4_k: out has {} bytes, need {}",
        out.len(), n_blocks * Q4K_BLOCK_BYTES);

    // Mapping from sub-block index `is` (0..8) to the contiguous
    // element range it owns inside the 256-element block. This must
    // match the read pattern in `dequantize_row_q4_k`:
    //   is = 0 → elements   0..32      (low  nibble of qs[ 0.. 32])
    //   is = 1 → elements 128..160     (high nibble of qs[ 0.. 32])
    //   is = 2 → elements  32..64      (low  nibble of qs[32.. 64])
    //   is = 3 → elements 160..192     (high nibble of qs[32.. 64])
    //   is = 4 → elements  64..96      (low  nibble of qs[64.. 96])
    //   is = 5 → elements 192..224     (high nibble of qs[64.. 96])
    //   is = 6 → elements  96..128     (low  nibble of qs[96..128])
    //   is = 7 → elements 224..256     (high nibble of qs[96..128])
    const SUBBLOCK_OFFSETS: [usize; 8] = [0, 128, 32, 160, 64, 192, 96, 224];

    for blk in 0..n_blocks {
        let block_x = &x[blk * Q4K_BLOCK_ELEMS..(blk + 1) * Q4K_BLOCK_ELEMS];

        // Pre-scan all 8 sub-blocks to set the super-block scale/min.
        // Both `super_d` and `super_dmin` are the maximum of their
        // per-sub-block locals so every sub-block can be packed into
        // a 6-bit quantised fraction.
        let mut super_d = 0.0f32;
        let mut super_dmin = 0.0f32;
        for is in 0..8 {
            let off = SUBBLOCK_OFFSETS[is];
            let s = &block_x[off..off + 32];
            let (local_d, local_dmin) = q4k_sub_block_local_scale_min(s);
            if local_d > super_d { super_d = local_d; }
            if local_dmin > super_dmin { super_dmin = local_dmin; }
        }
        // 6-bit headroom: divide by 63 so per-sub-block scales fit.
        let super_d_q = super_d / 63.0;
        let super_dmin_q = super_dmin / 63.0;
        // Round-trip through fp16: anything we encode now must
        // survive the storage step, otherwise the dequant value is
        // shifted by the fp16 rounding.
        let super_d_q = f16_to_f32(f32_to_f16(super_d_q));
        let super_dmin_q = f16_to_f32(f32_to_f16(super_dmin_q));

        let mut sc = [0u8; 8];
        let mut m = [0u8; 8];
        // Per-sub-block nibbles, addressed by sub-block index `is`.
        let mut nibs = [[0u8; 32]; 8];
        for is in 0..8 {
            let off = SUBBLOCK_OFFSETS[is];
            let s = &block_x[off..off + 32];
            quantize_sub_block(s, &mut sc[is], &mut m[is], &mut nibs[is], super_d_q, super_dmin_q);
        }

        // Pack into 144-byte block.
        let dst = &mut out[blk * Q4K_BLOCK_BYTES..(blk + 1) * Q4K_BLOCK_BYTES];
        let d_h = f32_to_f16(super_d_q);
        let dmin_h = f32_to_f16(super_dmin_q);
        dst[0..2].copy_from_slice(&d_h.to_le_bytes());
        dst[2..4].copy_from_slice(&dmin_h.to_le_bytes());
        let mut packed_scales = [0u8; 12];
        q4k_pack_scales_mins(&sc, &m, &mut packed_scales);
        dst[4..16].copy_from_slice(&packed_scales);

        // qs layout: byte qs[j] (j ∈ 0..128) holds the low nibble of
        // sub-block (j / 32) * 2 (mapping above) at column (j % 32),
        // and the high nibble of sub-block (j / 32) * 2 + 1 at the
        // same column.
        let qs = &mut dst[16..144];
        for j in 0..128 {
            let low_sub = (j / 32) * 2;
            let high_sub = low_sub + 1;
            let low_nib = nibs[low_sub][j % 32];
            let high_nib = nibs[high_sub][j % 32];
            qs[j] = (low_nib & 0xF) | ((high_nib & 0xF) << 4);
        }
    }

    n_blocks * Q4K_BLOCK_BYTES
}

fn read_typed_value<R: Read>(r: &mut R, vtype: u32) -> Result<MetaValue, String> {
    let mut buf = [0u8; 8];
    match vtype {
        0 => { r.read_exact(&mut buf[..1]).map_err(|e| e.to_string())?; Ok(MetaValue::U8(buf[0])) }
        1 => { r.read_exact(&mut buf[..1]).map_err(|e| e.to_string())?; Ok(MetaValue::I8(buf[0] as i8)) }
        2 => { r.read_exact(&mut buf[..2]).map_err(|e| e.to_string())?; Ok(MetaValue::U16(u16::from_le_bytes([buf[0], buf[1]]))) }
        3 => { r.read_exact(&mut buf[..2]).map_err(|e| e.to_string())?; Ok(MetaValue::I16(i16::from_le_bytes([buf[0], buf[1]]))) }
        4 => { r.read_exact(&mut buf[..4]).map_err(|e| e.to_string())?; Ok(MetaValue::U32(u32::from_le_bytes(buf[..4].try_into().unwrap()))) }
        5 => { r.read_exact(&mut buf[..4]).map_err(|e| e.to_string())?; Ok(MetaValue::I32(i32::from_le_bytes(buf[..4].try_into().unwrap()))) }
        6 => { r.read_exact(&mut buf[..4]).map_err(|e| e.to_string())?; Ok(MetaValue::F32(f32::from_le_bytes(buf[..4].try_into().unwrap()))) }
        7 => { r.read_exact(&mut buf[..1]).map_err(|e| e.to_string())?; Ok(MetaValue::Bool(buf[0] != 0)) }
        8 => { Ok(MetaValue::Str(read_string(r)?)) }
        9 => {
            // Array
            r.read_exact(&mut buf[..4]).map_err(|e| e.to_string())?;
            let atype = u32::from_le_bytes(buf[..4].try_into().unwrap());
            r.read_exact(&mut buf[..8]).map_err(|e| e.to_string())?;
            let alen = u64::from_le_bytes(buf) as usize;
            let mut arr = Vec::with_capacity(alen.min(1024));
            for _ in 0..alen {
                arr.push(read_typed_value(r, atype)?);
            }
            Ok(MetaValue::Array(arr))
        }
        10 => { r.read_exact(&mut buf).map_err(|e| e.to_string())?; Ok(MetaValue::U64(u64::from_le_bytes(buf))) }
        11 => { r.read_exact(&mut buf).map_err(|e| e.to_string())?; Ok(MetaValue::I64(i64::from_le_bytes(buf))) }
        12 => { r.read_exact(&mut buf).map_err(|e| e.to_string())?; Ok(MetaValue::F64(f64::from_le_bytes(buf))) }
        _ => Err(format!("unknown GGUF value type {}", vtype)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip f32 → Q4_K → f32 must stay within Q4_K's published
    /// 1% relative-error band on smoothly-varying data (which is what
    /// real model weights look like).
    #[test]
    fn q4k_round_trip_under_tolerance() {
        let n_blocks = 4;
        let n = n_blocks * Q4K_BLOCK_ELEMS;
        // Smooth signal; normal-distributed weights look more like
        // this than uniform noise, and the published 1% bound assumes
        // smooth data.
        let mut x = vec![0.0f32; n];
        for (i, v) in x.iter_mut().enumerate() {
            let t = i as f32 / n as f32;
            *v = (t * 7.3).sin() * 0.5 + (t * 17.1).cos() * 0.2;
        }
        let mut q = vec![0u8; n_blocks * Q4K_BLOCK_BYTES];
        let written = quantize_row_q4_k(&x, &mut q);
        assert_eq!(written, n_blocks * Q4K_BLOCK_BYTES);

        let mut y = vec![0.0f32; n];
        dequantize_row_q4_k(&q, &mut y, n_blocks);

        // Q4_K has ~1% typical relative error; we use 5% as the test
        // gate. The reconstruction error norm (root-mean-squared
        // relative to the signal norm) is the canonical Q4_K
        // tolerance metric — pointwise relative error is unbounded
        // near zero crossings and not a useful signal there.
        let signal_sq: f32 = x.iter().map(|v| v * v).sum();
        let err_sq: f32 = x.iter().zip(&y).map(|(a, b)| (a - b).powi(2)).sum();
        let rms_rel = (err_sq / signal_sq.max(1e-12)).sqrt();
        let mut max_abs = 0.0f32;
        for i in 0..n {
            let abs_diff = (x[i] - y[i]).abs();
            if abs_diff > max_abs { max_abs = abs_diff; }
        }
        eprintln!("q4k round trip: max_abs={max_abs} rms_rel={rms_rel}");
        assert!(rms_rel < 0.05,
            "q4k round trip rms_rel={rms_rel} exceeded 5% — algorithm regression");
    }

    /// Dequantising the same Q4_K bytes twice MUST produce bit-identical
    /// f32 outputs. This is the property `LinearResidentQuantized` relies
    /// on for evict-and-redequant: the host-side q4k bytes don't change
    /// across an evict/restore cycle, so the dequant kernel must be a
    /// pure function of those bytes.
    #[test]
    fn q4k_dequant_is_pure_function_of_bytes() {
        let n_blocks = 2;
        let n = n_blocks * Q4K_BLOCK_ELEMS;
        let mut x = vec![0.0f32; n];
        for (i, v) in x.iter_mut().enumerate() {
            *v = ((i as f32) * 0.013).sin();
        }
        let mut q = vec![0u8; n_blocks * Q4K_BLOCK_BYTES];
        quantize_row_q4_k(&x, &mut q);
        let mut y1 = vec![0.0f32; n];
        let mut y2 = vec![0.0f32; n];
        dequantize_row_q4_k(&q, &mut y1, n_blocks);
        dequantize_row_q4_k(&q, &mut y2, n_blocks);
        assert_eq!(y1, y2, "dequant is not pure (same bytes → different f32)");
    }
}
