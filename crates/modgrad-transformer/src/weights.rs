//! Weight storage for the transformer model.
//!
//! Uses FlatBuffers for serialization (already a project dependency).
//! NOT serde_json — these are potentially multi-GB tensors.
//!
//! In-memory layout: raw f32 arrays with shape metadata.
//! On-disk: flatbuffers or memory-mapped raw pages with a header.

use super::dims::*;
use super::error::{TransformerError, Result};

/// Per-block (layer) weights.
pub struct BlockWeights {
    // Attention
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,

    // MLP
    pub mlp_fc: Vec<f32>,
    pub mlp_proj: Vec<f32>,

    // Value embedding (optional — only on VE layers)
    pub ve_table: Option<Vec<f32>>,
    pub ve_gate: Option<Vec<f32>>,
}

/// All model weights.
pub struct GptWeights {
    /// Token embedding: [vocab_size, model_dim].
    pub token_embed: Vec<f32>,
    /// LM head (output projection): [vocab_size, model_dim].
    /// May be tied to token_embed.
    pub lm_head: Vec<f32>,
    /// Final RMS norm scale: [model_dim].
    pub final_norm_scale: Vec<f32>,

    /// Smear gate weights: [model_dim, smear_channels].
    pub smear_gate: Vec<f32>,

    /// Per-layer weights.
    pub blocks: Vec<BlockWeights>,
}

impl GptWeights {
    /// Validate that all weight shapes match the config.
    pub fn validate(&self, config: &super::config::GptConfig) -> Result<()> {
        let md = config.model_dim.get();
        let vocab = config.vocab_size.get();
        let n_layers = config.num_layers.get();
        let kv_dim = config.num_kv_heads.get() * config.head_dim.get();
        let mlp_dim = config.mlp_dim.get();

        check_shape("token_embed", self.token_embed.len(), vocab * md)?;
        check_shape("lm_head", self.lm_head.len(), vocab * md)?;
        check_shape("final_norm_scale", self.final_norm_scale.len(), md)?;
        check_shape("smear_gate", self.smear_gate.len(), md * config.smear.gate_channels)?;

        if self.blocks.len() != n_layers {
            return Err(TransformerError::ShapeMismatch {
                context: "num_blocks",
                expected: n_layers,
                actual: self.blocks.len(),
            });
        }

        for (i, b) in self.blocks.iter().enumerate() {
            let _ctx = |s: &'static str| s; // Layer context would need alloc, skip for now
            check_shape("wq", b.wq.len(), md * md)?;
            check_shape("wk", b.wk.len(), kv_dim * md)?;
            check_shape("wv", b.wv.len(), kv_dim * md)?;
            check_shape("wo", b.wo.len(), md * md)?;
            check_shape("mlp_fc", b.mlp_fc.len(), mlp_dim * md)?;
            check_shape("mlp_proj", b.mlp_proj.len(), md * mlp_dim)?;

            let has_ve = config.has_value_embed(
                LayerIdx::new_unchecked(i, config.num_layers)
            );
            if has_ve {
                if let Some(ref t) = b.ve_table {
                    check_shape("ve_table", t.len(), vocab * kv_dim)?;
                } else {
                    return Err(TransformerError::BuilderMissing("ve_table for VE layer"));
                }
                if let Some(ref g) = b.ve_gate {
                    check_shape("ve_gate", g.len(), kv_dim * config.value_embed.gate_channels)?;
                } else {
                    return Err(TransformerError::BuilderMissing("ve_gate for VE layer"));
                }
            }
        }

        Ok(())
    }

    /// Save weights to a raw binary file with a minimal header.
    ///
    /// Format: [magic: u32][version: u32][n_tensors: u32]
    ///         [tensor_header: name_len, name, shape_len, shape, data_offset, data_len]*
    ///         [raw f32 data]*
    pub fn save_raw(&self, path: &std::path::Path) -> Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(path)?;

        // Collect all tensors as (name, data) pairs
        let mut tensors: Vec<(&str, &[f32])> = vec![
            ("token_embed", &self.token_embed),
            ("lm_head", &self.lm_head),
            ("final_norm_scale", &self.final_norm_scale),
            ("smear_gate", &self.smear_gate),
        ];
        for (i, b) in self.blocks.iter().enumerate() {
            // We'll use index-based naming
            let _ = i; // Names are positional in the block array
            tensors.push(("wq", &b.wq));
            tensors.push(("wk", &b.wk));
            tensors.push(("wv", &b.wv));
            tensors.push(("wo", &b.wo));
            tensors.push(("mlp_fc", &b.mlp_fc));
            tensors.push(("mlp_proj", &b.mlp_proj));
            if let Some(ref t) = b.ve_table { tensors.push(("ve_table", t)); }
            if let Some(ref g) = b.ve_gate { tensors.push(("ve_gate", g)); }
        }

        // Magic + version + count
        file.write_all(&0x49534953u32.to_le_bytes())?; // "ISIS"
        file.write_all(&1u32.to_le_bytes())?;
        file.write_all(&(tensors.len() as u32).to_le_bytes())?;

        // Write tensor sizes (for seeking)
        for (_, data) in &tensors {
            file.write_all(&(data.len() as u64).to_le_bytes())?;
        }

        // Write raw data
        for (_, data) in &tensors {
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
            };
            file.write_all(bytes)?;
        }

        Ok(())
    }

    /// Load weights from raw binary file.
    pub fn load_raw(path: &std::path::Path, config: &super::config::GptConfig) -> Result<Self> {
        use std::io::Read;
        let mut file = std::fs::File::open(path)?;

        let mut buf4 = [0u8; 4];
        file.read_exact(&mut buf4)?;
        let magic = u32::from_le_bytes(buf4);
        if magic != 0x49534953 {
            return Err(TransformerError::ShapeMismatch {
                context: "magic",
                expected: 0x49534953 as usize,
                actual: magic as usize,
            });
        }

        file.read_exact(&mut buf4)?; // version
        file.read_exact(&mut buf4)?;
        let n_tensors = u32::from_le_bytes(buf4) as usize;

        // Read sizes
        let mut sizes = Vec::with_capacity(n_tensors);
        let mut buf8 = [0u8; 8];
        for _ in 0..n_tensors {
            file.read_exact(&mut buf8)?;
            sizes.push(u64::from_le_bytes(buf8) as usize);
        }

        // Read tensors
        let mut tensors: Vec<Vec<f32>> = Vec::with_capacity(n_tensors);
        for &sz in &sizes {
            let mut data = vec![0.0f32; sz];
            let bytes: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, sz * 4)
            };
            file.read_exact(bytes)?;
            tensors.push(data);
        }

        // Destructure into GptWeights
        let mut it = tensors.into_iter();
        let token_embed = it.next().unwrap();
        let lm_head = it.next().unwrap();
        let final_norm_scale = it.next().unwrap();
        let smear_gate = it.next().unwrap();

        let n_layers = config.num_layers.get();
        let mut blocks = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let has_ve = config.has_value_embed(
                LayerIdx::new_unchecked(i, config.num_layers)
            );
            let wq = it.next().unwrap();
            let wk = it.next().unwrap();
            let wv = it.next().unwrap();
            let wo = it.next().unwrap();
            let mlp_fc = it.next().unwrap();
            let mlp_proj = it.next().unwrap();
            let ve_table = if has_ve { Some(it.next().unwrap()) } else { None };
            let ve_gate = if has_ve { Some(it.next().unwrap()) } else { None };

            blocks.push(BlockWeights {
                wq, wk, wv, wo, mlp_fc, mlp_proj, ve_table, ve_gate,
            });
        }

        let weights = Self {
            token_embed, lm_head, final_norm_scale, smear_gate, blocks,
        };
        weights.validate(config)?;
        Ok(weights)
    }
}

fn check_shape(context: &'static str, actual: usize, expected: usize) -> Result<()> {
    if actual != expected {
        Err(TransformerError::ShapeMismatch { context, expected, actual })
    } else {
        Ok(())
    }
}
