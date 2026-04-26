//! Minimal pure-Rust safetensors reader.
//!
//! Format (per safetensors spec):
//!   - `[u64 LE]` header byte length
//!   - `[u8; header_len]` JSON map: `{ "<name>": { "dtype": "...",
//!     "shape": [...], "data_offsets": [start, end] }, ... }` plus an
//!     optional `__metadata__` key (ignored here).
//!   - `[u8; ...]` raw tensor bytes; tensor `name` lives at byte offset
//!     `8 + header_len + start` (length `end - start`).
//!
//! Only the dtypes we need (`F32`, `BF16`, `F16`) are supported; other
//! dtypes return an error. Reads memory-map the file via plain Rust
//! `File::read_at` so we do not pull in a memmap2 dep.
//!
//! All `read_tensor_*` helpers return `Vec<f32>` — bf16/f16 inputs are
//! upconverted on load. The expected workflow is "load fp32 weights into
//! `LinearResident`" (see `crates/modgrad-io/src/qwen2.rs`).

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::backend::BoxErr;

/// One tensor entry from the JSON header.
#[derive(Debug, Clone, Deserialize)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [u64; 2],
}

/// Parsed safetensors file. Holds the file handle + header so each
/// `read_tensor` is a single seek + read.
pub struct SafetensorsFile {
    file: File,
    header_end: u64,
    tensors: BTreeMap<String, TensorInfo>,
    path: PathBuf,
}

impl SafetensorsFile {
    /// Open and parse the header. Tensor data is *not* read yet — the
    /// header only memoizes (offset, dtype, shape) so per-tensor
    /// `read_tensor` is one seek + read.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, BoxErr> {
        let path_ref = path.as_ref();
        let mut file = File::open(path_ref)
            .map_err(|e| format!("safetensors open {}: {e}", path_ref.display()))?;

        let mut len_bytes = [0u8; 8];
        file.read_exact(&mut len_bytes)
            .map_err(|e| format!("safetensors header-len read: {e}"))?;
        let header_len = u64::from_le_bytes(len_bytes);

        let mut header_bytes = vec![0u8; header_len as usize];
        file.read_exact(&mut header_bytes)
            .map_err(|e| format!("safetensors header read ({} bytes): {e}", header_len))?;

        // The header is `BTreeMap<String, TensorInfo>` plus an optional
        // `__metadata__` map. We deserialise into a generic JSON map and
        // pluck out the named tensors so the unknown `__metadata__` key
        // doesn't trip serde's strict mode.
        let raw: serde_json::Value = serde_json::from_slice(&header_bytes)
            .map_err(|e| format!("safetensors header json: {e}"))?;
        let obj = raw.as_object()
            .ok_or_else(|| "safetensors header: expected JSON object")?;

        let mut tensors: BTreeMap<String, TensorInfo> = BTreeMap::new();
        for (name, value) in obj {
            if name == "__metadata__" { continue; }
            let info: TensorInfo = serde_json::from_value(value.clone())
                .map_err(|e| format!("safetensors tensor `{name}` info: {e}"))?;
            tensors.insert(name.clone(), info);
        }

        let header_end = 8 + header_len;
        Ok(Self { file, header_end, tensors, path: path_ref.to_path_buf() })
    }

    /// All tensor names in the file, sorted (BTreeMap order).
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }

    /// Look up a tensor's metadata (shape / dtype / offsets). Returns
    /// `None` if the name is not in the header.
    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// Number of tensors in the header.
    pub fn len(&self) -> usize { self.tensors.len() }

    /// True if the header has zero tensors.
    pub fn is_empty(&self) -> bool { self.tensors.is_empty() }

    /// Path the file was opened from (useful for diagnostics).
    pub fn path(&self) -> &Path { &self.path }

    /// Read a tensor and convert to fp32. Supports `F32`, `BF16`, `F16`.
    /// Returns the flattened row-major data (numel = product of shape).
    pub fn read_tensor(&mut self, name: &str) -> Result<Vec<f32>, BoxErr> {
        let info = self.tensors.get(name)
            .ok_or_else(|| format!("safetensors tensor `{name}` not found in {}",
                self.path.display()))?
            .clone();
        let numel: usize = info.shape.iter().product();
        let nbytes = (info.data_offsets[1] - info.data_offsets[0]) as usize;

        match info.dtype.as_str() {
            "F32" => {
                if nbytes != numel * 4 {
                    return Err(format!(
                        "safetensors tensor `{name}` F32: byte len {nbytes} \
                         vs numel*4 {} (shape {:?})", numel * 4, info.shape).into());
                }
                let mut bytes = vec![0u8; nbytes];
                self.read_at(info.data_offsets[0], &mut bytes)?;
                let mut out = vec![0.0f32; numel];
                for i in 0..numel {
                    let chunk = &bytes[i * 4..i * 4 + 4];
                    out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                }
                Ok(out)
            }
            "BF16" => {
                if nbytes != numel * 2 {
                    return Err(format!(
                        "safetensors tensor `{name}` BF16: byte len {nbytes} \
                         vs numel*2 {} (shape {:?})", numel * 2, info.shape).into());
                }
                let mut bytes = vec![0u8; nbytes];
                self.read_at(info.data_offsets[0], &mut bytes)?;
                let mut out = vec![0.0f32; numel];
                for i in 0..numel {
                    // bf16 stores the upper 16 bits of an fp32. Pad zeros
                    // in the low 16 bits to reconstruct fp32 bits.
                    let lo = bytes[i * 2] as u32;
                    let hi = bytes[i * 2 + 1] as u32;
                    let bits = ((hi << 8) | lo) << 16;
                    out[i] = f32::from_bits(bits);
                }
                Ok(out)
            }
            "F16" => {
                if nbytes != numel * 2 {
                    return Err(format!(
                        "safetensors tensor `{name}` F16: byte len {nbytes} \
                         vs numel*2 {} (shape {:?})", numel * 2, info.shape).into());
                }
                let mut bytes = vec![0u8; nbytes];
                self.read_at(info.data_offsets[0], &mut bytes)?;
                let mut out = vec![0.0f32; numel];
                for i in 0..numel {
                    let bits = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
                    out[i] = half::f16::from_bits(bits).to_f32();
                }
                Ok(out)
            }
            other => Err(format!(
                "safetensors tensor `{name}`: unsupported dtype `{other}` \
                 (expected F32, BF16, or F16)").into()),
        }
    }

    /// Seek + read at a tensor-relative offset (offset is relative to
    /// the start of the data block, *not* the file).
    fn read_at(&mut self, rel_offset: u64, buf: &mut [u8]) -> Result<(), BoxErr> {
        let abs = self.header_end + rel_offset;
        self.file.seek(SeekFrom::Start(abs))
            .map_err(|e| format!("safetensors seek to {abs}: {e}"))?;
        self.file.read_exact(buf)
            .map_err(|e| format!("safetensors read {} bytes at {abs}: {e}", buf.len()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Hand-craft a tiny safetensors file with one F32 and one BF16
    /// tensor. Verifies the parser picks up shapes, offsets, and that
    /// BF16 → f32 conversion is bit-correct (upper 16 bits, low zeros).
    #[test]
    fn parse_and_read_round_trip() {
        let dir = std::env::temp_dir().join("modgrad_safetensors_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tiny.safetensors");

        // F32: [1.0, 2.0, 3.0]   12 bytes
        // BF16: [1.0, -2.0]       4 bytes
        // Build bf16 deliberately: bf16 = upper 16 bits of fp32, low 16 bits zero.
        let f32_bytes: Vec<u8> = [1.0f32, 2.0, 3.0].iter()
            .flat_map(|f| f.to_le_bytes()).collect();
        let bf16_bytes: Vec<u8> = [1.0f32, -2.0f32].iter()
            .flat_map(|&f| {
                let bits = f.to_bits();
                let hi = (bits >> 16) as u16;
                hi.to_le_bytes()
            }).collect();
        assert_eq!(bf16_bytes.len(), 4);

        let header_obj = serde_json::json!({
            "f32_t": {
                "dtype": "F32",
                "shape": [3],
                "data_offsets": [0, 12],
            },
            "bf16_t": {
                "dtype": "BF16",
                "shape": [2],
                "data_offsets": [12, 16],
            },
        });
        let header_bytes = serde_json::to_vec(&header_obj).unwrap();
        let header_len = header_bytes.len() as u64;

        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&header_len.to_le_bytes()).unwrap();
        file.write_all(&header_bytes).unwrap();
        file.write_all(&f32_bytes).unwrap();
        file.write_all(&bf16_bytes).unwrap();
        drop(file);

        let mut sf = SafetensorsFile::load(&path).unwrap();
        let mut names = sf.tensor_names();
        names.sort();
        assert_eq!(names, vec!["bf16_t", "f32_t"]);

        let f32_t = sf.read_tensor("f32_t").unwrap();
        assert_eq!(f32_t, vec![1.0, 2.0, 3.0]);

        let bf16_t = sf.read_tensor("bf16_t").unwrap();
        assert_eq!(bf16_t, vec![1.0, -2.0]);
    }
}
