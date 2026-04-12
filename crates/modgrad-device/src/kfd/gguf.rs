//! Minimal GGUF v3 parser. Reads metadata and tensor info from a GGUF file.
//! Tensors are NOT loaded into memory — the file is mmap'd and tensors
//! are accessed by offset. This means 5GB of Q4 weights cost 0 RSS until
//! the OS pages them in on access.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

/// GGUF tensor quantization types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
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
