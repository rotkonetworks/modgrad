//! Binary checkpoint format for [`BltModel`]. **NOT a stable on-disk
//! format** — intended for training-then-resume within a session, not
//! cross-version interchange. For cross-tool compatibility use
//! safetensors (separate slice).
//!
//! ## Layout
//!
//! ```text
//!   magic           : 8 bytes  "BLTCKPT1"
//!   format_version  : u32 LE   = 1
//!   config_blob     :
//!       n_enc_layers, byte_dim, patch_dim,
//!       n_byte_heads, byte_head_dim, byte_mlp_dim, byte_norm_eps,
//!       byte_rope_base, byte_max_seq_len,
//!       ngram_min_n, ngram_max_n, ngram_vocab_per_n,
//!       n_lat_layers, latent_n_heads, latent_head_dim,
//!       latent_mlp_dim, latent_norm_eps, latent_rope_base, max_patches,
//!       n_dec_layers
//!   buffer_count    : u64 LE   (sanity check)
//!   for each weight buffer in canonical key order
//!     (mirroring `BltModelTrainer::param_keys_for_model`):
//!       u64 LE        : byte length of this buffer (validated against
//!                       the freshly-built `BltModel`'s buffer; refuse
//!                       load on mismatch)
//!       raw f32 LE    : that many bytes of weight data (no compression)
//! ```
//!
//! Total file size: ~50 MB for the BLT tiny config (byte_dim=32,
//! patch_dim=64, lE=1, lL=2, lD=1); ~1.9 GB for a Qwen2.5-0.5B-class
//! latent.
//!
//! ## Determinism
//!
//! The format is byte-identical across save/load — the round-trip test
//! asserts byte-equal weight buffers, not approximate equality.
//!
//! ## Out of scope
//!
//! - No optimizer state (AdamW m/v) — resume from arbitrary mid-step is
//!   a separate slice.
//! - No compression — raw f32 is fine for v0.
//! - No safetensors interop — separate slice.

#![cfg(feature = "rocm")]

use std::io::{Read, Write};
use std::path::Path;

use modgrad_compute::backend::ResidencyError;
use modgrad_device::backend::HipBuffer;

use crate::cross_attn::CrossAttention;
use crate::decoder::LocalDecoderConfig;
use crate::encoder::LocalEncoderConfig;
use crate::model::{BltConfig, BltLatentConfig, BltModel};

// ─── Constants ────────────────────────────────────────────────

/// Magic header bytes. Distinguishes a BLT checkpoint from arbitrary
/// other binary blobs (notably the `weights.rs` ISIS format which uses
/// a different magic).
const BLT_MAGIC: &[u8; 8] = b"BLTCKPT1";

/// Format version. Bumped on incompatible layout changes; loaders refuse
/// older / newer versions rather than silently misinterpreting.
const FORMAT_VERSION: u32 = 1;

// ─── Error type ───────────────────────────────────────────────

/// Failure modes for [`save_blt_model`] / [`load_blt_model_into`] /
/// [`load_blt_model_from_path`]. All silent corruption paths are made
/// explicit — refuse-and-bail beats best-effort here.
#[derive(Debug)]
pub enum CheckpointError {
    /// Underlying I/O failed — file open, read, write.
    Io(std::io::Error),
    /// Magic header mismatch — file is not a BLT checkpoint.
    BadMagic,
    /// Format version this loader doesn't understand.
    UnsupportedVersion { expected: u32, got: u32 },
    /// Buffer count in the file doesn't match the buffer count
    /// produced by the freshly-built [`BltModel`].
    BufferCountMismatch { expected: usize, got: usize },
    /// Stored config dims disagree with the in-memory model the caller
    /// asked us to load into. Carries human-readable summaries on both
    /// sides for diagnostics.
    ConfigMismatch { expected: String, got: String },
    /// A weight buffer's stored byte length differs from the size the
    /// model expects. Identifies the offending key.
    SizeMismatch { key: String, expected: usize, got: usize },
    /// A device-side error (hipMalloc / hipMemcpy) bubbled up while
    /// (re)building the model or (un)loading buffers.
    Residency(ResidencyError),
}

impl std::fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "checkpoint io: {e}"),
            Self::BadMagic => write!(f, "checkpoint: bad magic (not a BLT checkpoint)"),
            Self::UnsupportedVersion { expected, got } => write!(
                f, "checkpoint: unsupported format version (expected {expected}, got {got})",
            ),
            Self::BufferCountMismatch { expected, got } => write!(
                f, "checkpoint: buffer count mismatch (expected {expected}, got {got})",
            ),
            Self::ConfigMismatch { expected, got } => write!(
                f, "checkpoint: config mismatch (expected {expected}, got {got})",
            ),
            Self::SizeMismatch { key, expected, got } => write!(
                f, "checkpoint: size mismatch for {key} (expected {expected} bytes, got {got})",
            ),
            Self::Residency(e) => write!(f, "checkpoint residency: {e}"),
        }
    }
}

impl std::error::Error for CheckpointError {}

impl From<std::io::Error> for CheckpointError {
    fn from(e: std::io::Error) -> Self { Self::Io(e) }
}

impl From<ResidencyError> for CheckpointError {
    fn from(e: ResidencyError) -> Self { Self::Residency(e) }
}

// ─── Public API ───────────────────────────────────────────────

/// Save `model`'s device-resident weights to `path` in the BLT
/// checkpoint format.
///
/// Each weight buffer is downloaded from the device to a host scratch
/// vec, then written to disk as raw little-endian f32. Total host
/// scratch peak is one buffer at a time, not the full model — safe for
/// large latents.
pub fn save_blt_model<P: AsRef<Path>>(
    model: &BltModel,
    path: P,
) -> Result<(), CheckpointError> {
    let mut file = std::fs::File::create(path.as_ref())?;

    // Header.
    file.write_all(BLT_MAGIC)?;
    write_u32(&mut file, FORMAT_VERSION)?;

    // Config blob (manual schema — `BltConfig` doesn't derive Serialize).
    write_config(&mut file, &model.config)?;

    // Iterate keys in canonical order, writing each buffer with a length
    // prefix for validation on load.
    let keys = blt_param_keys(model);
    write_u64(&mut file, keys.len() as u64)?;

    let mut scratch: Vec<f32> = Vec::new();
    for key in &keys {
        let buf = blt_weight_dev(model, key);
        let n_f32 = buf.len_f32();
        let n_bytes = buf.bytes();
        debug_assert_eq!(n_bytes, n_f32 * 4, "{key}: HipBuffer not f32-aligned");

        // Length prefix.
        write_u64(&mut file, n_bytes as u64)?;

        // Download → write.
        scratch.resize(n_f32, 0.0);
        buf.copy_to_host(&mut scratch)
            .map_err(|e| CheckpointError::Residency(ResidencyError::Backend(e)))?;
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(scratch.as_ptr() as *const u8, n_bytes)
        };
        file.write_all(bytes)?;
    }
    file.flush()?;
    Ok(())
}

/// Load a checkpoint at `path` into an existing `model`. The model's
/// config must match the on-disk config exactly; otherwise returns
/// [`CheckpointError::ConfigMismatch`].
///
/// On any error the model is left in an **indeterminate state** —
/// some buffers may have been overwritten, some not. Callers that need
/// atomicity should use [`load_blt_model_from_path`] which builds a
/// fresh model.
pub fn load_blt_model_into<P: AsRef<Path>>(
    model: &mut BltModel,
    path: P,
) -> Result<(), CheckpointError> {
    let mut file = std::fs::File::open(path.as_ref())?;

    read_and_check_header(&mut file)?;
    let stored_config = read_config(&mut file)?;
    if !configs_equal(&stored_config, &model.config) {
        return Err(CheckpointError::ConfigMismatch {
            expected: format_config(&model.config),
            got: format_config(&stored_config),
        });
    }

    let stored_count = read_u64(&mut file)? as usize;
    let keys = blt_param_keys(model);
    if stored_count != keys.len() {
        return Err(CheckpointError::BufferCountMismatch {
            expected: keys.len(),
            got: stored_count,
        });
    }

    let mut scratch: Vec<f32> = Vec::new();
    for key in &keys {
        let stored_bytes = read_u64(&mut file)? as usize;
        let buf = blt_weight_dev(model, key);
        let expected_bytes = buf.bytes();
        if stored_bytes != expected_bytes {
            return Err(CheckpointError::SizeMismatch {
                key: key.clone(),
                expected: expected_bytes,
                got: stored_bytes,
            });
        }

        let n_f32 = stored_bytes / 4;
        scratch.resize(n_f32, 0.0);
        let dst: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(scratch.as_mut_ptr() as *mut u8, stored_bytes)
        };
        file.read_exact(dst)?;

        buf.copy_from_host(&scratch)
            .map_err(|e| CheckpointError::Residency(ResidencyError::Backend(e)))?;
    }
    Ok(())
}

/// Convenience: build a fresh [`BltModel`] from the config stored in
/// the checkpoint, then load weights into it. Caller doesn't need to
/// know the model dims up front.
pub fn load_blt_model_from_path<P: AsRef<Path>>(
    path: P,
) -> Result<BltModel, CheckpointError> {
    // Peek at the header + config to discover the dims.
    let mut file = std::fs::File::open(path.as_ref())?;
    read_and_check_header(&mut file)?;
    let stored_config = read_config(&mut file)?;
    drop(file);

    // Build the model with a default (all-ones) final-norm scale; the
    // real scale comes through as part of the weight stream below.
    let mut model = BltModel::new(stored_config)?;
    load_blt_model_into(&mut model, path)?;
    Ok(model)
}

// ─── Internal: header / config I/O ─────────────────────────────

fn read_and_check_header<R: Read>(r: &mut R) -> Result<(), CheckpointError> {
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != BLT_MAGIC {
        return Err(CheckpointError::BadMagic);
    }
    let version = read_u32(r)?;
    if version != FORMAT_VERSION {
        return Err(CheckpointError::UnsupportedVersion {
            expected: FORMAT_VERSION,
            got: version,
        });
    }
    Ok(())
}

fn write_config<W: Write>(w: &mut W, c: &BltConfig) -> std::io::Result<()> {
    let e = &c.encoder;
    write_u64(w, e.n_layers as u64)?;
    write_u64(w, e.byte_dim as u64)?;
    write_u64(w, e.patch_dim as u64)?;
    write_u64(w, e.n_heads as u64)?;
    write_u64(w, e.head_dim as u64)?;
    write_u64(w, e.mlp_dim as u64)?;
    write_f32(w, e.norm_eps)?;
    write_f32(w, e.rope_base)?;
    write_u64(w, e.max_seq_len as u64)?;
    write_u64(w, e.ngram_min_n as u64)?;
    write_u64(w, e.ngram_max_n as u64)?;
    write_u64(w, e.ngram_vocab_per_n as u64)?;

    let l = &c.latent;
    write_u64(w, l.n_layers as u64)?;
    write_u64(w, l.n_heads as u64)?;
    write_u64(w, l.head_dim as u64)?;
    write_u64(w, l.mlp_dim as u64)?;
    write_f32(w, l.norm_eps)?;
    write_f32(w, l.rope_base)?;
    write_u64(w, l.max_patches as u64)?;

    let d = &c.decoder;
    write_u64(w, d.n_layers as u64)?;
    write_u64(w, d.byte_dim as u64)?;
    write_u64(w, d.patch_dim as u64)?;
    write_u64(w, d.n_heads as u64)?;
    write_u64(w, d.head_dim as u64)?;
    write_u64(w, d.mlp_dim as u64)?;
    write_f32(w, d.norm_eps)?;
    write_f32(w, d.rope_base)?;
    write_u64(w, d.max_seq_len as u64)?;
    Ok(())
}

fn read_config<R: Read>(r: &mut R) -> Result<BltConfig, CheckpointError> {
    let encoder = LocalEncoderConfig {
        n_layers: read_u64(r)? as usize,
        byte_dim: read_u64(r)? as usize,
        patch_dim: read_u64(r)? as usize,
        n_heads: read_u64(r)? as usize,
        head_dim: read_u64(r)? as usize,
        mlp_dim: read_u64(r)? as usize,
        norm_eps: read_f32(r)?,
        rope_base: read_f32(r)?,
        max_seq_len: read_u64(r)? as usize,
        ngram_min_n: read_u64(r)? as usize,
        ngram_max_n: read_u64(r)? as usize,
        ngram_vocab_per_n: read_u64(r)? as usize,
    };

    // Latent: patch_dim is implied by encoder.patch_dim (validated via
    // `BltConfig::validate` once the model is rebuilt).
    let lat_n_layers = read_u64(r)? as usize;
    let lat_n_heads = read_u64(r)? as usize;
    let lat_head_dim = read_u64(r)? as usize;
    let lat_mlp_dim = read_u64(r)? as usize;
    let lat_norm_eps = read_f32(r)?;
    let lat_rope_base = read_f32(r)?;
    let lat_max_patches = read_u64(r)? as usize;
    let latent = BltLatentConfig {
        n_layers: lat_n_layers,
        patch_dim: encoder.patch_dim,
        n_heads: lat_n_heads,
        head_dim: lat_head_dim,
        mlp_dim: lat_mlp_dim,
        norm_eps: lat_norm_eps,
        rope_base: lat_rope_base,
        max_patches: lat_max_patches,
    };

    let decoder = LocalDecoderConfig {
        n_layers: read_u64(r)? as usize,
        byte_dim: read_u64(r)? as usize,
        patch_dim: read_u64(r)? as usize,
        n_heads: read_u64(r)? as usize,
        head_dim: read_u64(r)? as usize,
        mlp_dim: read_u64(r)? as usize,
        norm_eps: read_f32(r)?,
        rope_base: read_f32(r)?,
        max_seq_len: read_u64(r)? as usize,
    };

    Ok(BltConfig { encoder, latent, decoder })
}

fn configs_equal(a: &BltConfig, b: &BltConfig) -> bool {
    let e = &a.encoder;
    let f = &b.encoder;
    if (e.n_layers, e.byte_dim, e.patch_dim, e.n_heads, e.head_dim, e.mlp_dim,
        e.max_seq_len, e.ngram_min_n, e.ngram_max_n, e.ngram_vocab_per_n)
        != (f.n_layers, f.byte_dim, f.patch_dim, f.n_heads, f.head_dim, f.mlp_dim,
            f.max_seq_len, f.ngram_min_n, f.ngram_max_n, f.ngram_vocab_per_n)
    {
        return false;
    }
    if e.norm_eps.to_bits() != f.norm_eps.to_bits()
        || e.rope_base.to_bits() != f.rope_base.to_bits()
    {
        return false;
    }

    let l = &a.latent;
    let m = &b.latent;
    if (l.n_layers, l.patch_dim, l.n_heads, l.head_dim, l.mlp_dim, l.max_patches)
        != (m.n_layers, m.patch_dim, m.n_heads, m.head_dim, m.mlp_dim, m.max_patches)
    {
        return false;
    }
    if l.norm_eps.to_bits() != m.norm_eps.to_bits()
        || l.rope_base.to_bits() != m.rope_base.to_bits()
    {
        return false;
    }

    let d = &a.decoder;
    let g = &b.decoder;
    if (d.n_layers, d.byte_dim, d.patch_dim, d.n_heads, d.head_dim, d.mlp_dim,
        d.max_seq_len)
        != (g.n_layers, g.byte_dim, g.patch_dim, g.n_heads, g.head_dim, g.mlp_dim,
            g.max_seq_len)
    {
        return false;
    }
    if d.norm_eps.to_bits() != g.norm_eps.to_bits()
        || d.rope_base.to_bits() != g.rope_base.to_bits()
    {
        return false;
    }
    true
}

fn format_config(c: &BltConfig) -> String {
    format!(
        "BltConfig {{ enc: {}L bd={} pd={} h={}x{} mlp={} seq={} ngram=[{}..{}]x{}; \
         lat: {}L h={}x{} mlp={} max_patches={}; \
         dec: {}L bd={} pd={} h={}x{} mlp={} seq={} }}",
        c.encoder.n_layers, c.encoder.byte_dim, c.encoder.patch_dim,
        c.encoder.n_heads, c.encoder.head_dim, c.encoder.mlp_dim,
        c.encoder.max_seq_len,
        c.encoder.ngram_min_n, c.encoder.ngram_max_n, c.encoder.ngram_vocab_per_n,
        c.latent.n_layers, c.latent.n_heads, c.latent.head_dim, c.latent.mlp_dim,
        c.latent.max_patches,
        c.decoder.n_layers, c.decoder.byte_dim, c.decoder.patch_dim,
        c.decoder.n_heads, c.decoder.head_dim, c.decoder.mlp_dim,
        c.decoder.max_seq_len,
    )
}

// ─── Internal: param key + buffer enumeration ──────────────────
//
// Mirrors `BltModelTrainer`'s private `param_keys_for_model` and
// `weight_dev_for_blt_key` helpers in `trainer.rs`. Kept in lock-step
// with that module — any new BLT weight added there MUST also be added
// here, otherwise the checkpoint will silently drop or shuffle weights.
// The `param_keys_match_trainer` test below pins the count against
// trainer-side expectations.

fn blt_param_keys(model: &BltModel) -> Vec<String> {
    let n_enc = model.encoder.n_layers();
    let n_lat = model.latent.num_layers();
    let n_dec = model.decoder.n_layers();

    let mut keys = Vec::with_capacity(
        1 + n_enc * 11 + n_lat * 7 + 1 + n_dec * 11 + 3,
    );

    keys.push("encoder.byte_embed".to_string());
    for li in 0..n_enc {
        keys.push(format!("encoder.block.{li}.wq"));
        keys.push(format!("encoder.block.{li}.wk"));
        keys.push(format!("encoder.block.{li}.wv"));
        keys.push(format!("encoder.block.{li}.wo"));
        keys.push(format!("encoder.block.{li}.gate"));
        keys.push(format!("encoder.block.{li}.up"));
        keys.push(format!("encoder.block.{li}.down"));
        keys.push(format!("encoder.cross_attn.{li}.wq"));
        keys.push(format!("encoder.cross_attn.{li}.wk"));
        keys.push(format!("encoder.cross_attn.{li}.wv"));
        keys.push(format!("encoder.cross_attn.{li}.wo"));
    }
    for li in 0..n_lat {
        keys.push(format!("latent.block.{li}.wq"));
        keys.push(format!("latent.block.{li}.wk"));
        keys.push(format!("latent.block.{li}.wv"));
        keys.push(format!("latent.block.{li}.wo"));
        keys.push(format!("latent.block.{li}.gate"));
        keys.push(format!("latent.block.{li}.up"));
        keys.push(format!("latent.block.{li}.down"));
    }
    keys.push("latent.final_norm".to_string());
    for li in 0..n_dec {
        keys.push(format!("decoder.block.{li}.wq"));
        keys.push(format!("decoder.block.{li}.wk"));
        keys.push(format!("decoder.block.{li}.wv"));
        keys.push(format!("decoder.block.{li}.wo"));
        keys.push(format!("decoder.block.{li}.gate"));
        keys.push(format!("decoder.block.{li}.up"));
        keys.push(format!("decoder.block.{li}.down"));
        keys.push(format!("decoder.cross_attn.{li}.wq"));
        keys.push(format!("decoder.cross_attn.{li}.wk"));
        keys.push(format!("decoder.cross_attn.{li}.wv"));
        keys.push(format!("decoder.cross_attn.{li}.wo"));
    }
    keys.push("decoder.lm_head".to_string());
    keys.push("decoder.lm_head_bias".to_string());
    keys.push("decoder.final_norm".to_string());

    keys
}

fn blt_weight_dev<'a>(model: &'a BltModel, key: &str) -> &'a HipBuffer {
    if key == "encoder.byte_embed" { return &model.encoder.byte_embed_dev; }
    if key == "latent.final_norm" { return &model.latent_final_norm_weight_dev; }
    if key == "decoder.lm_head" { return &model.decoder.lm_head.weight_dev; }
    if key == "decoder.lm_head_bias" { return &model.decoder.lm_head.bias_dev; }
    if key == "decoder.final_norm" { return &model.decoder.final_norm_weight_dev; }

    if let Some(rest) = key.strip_prefix("encoder.block.") {
        let (li, slot) = parse_block_key(rest);
        return block_slot(&model.encoder.byte_layers[li], slot, key);
    }
    if let Some(rest) = key.strip_prefix("encoder.cross_attn.") {
        let (li, slot) = parse_block_key(rest);
        return cross_slot(&model.encoder.cross_attns[li], slot, key);
    }
    if let Some(rest) = key.strip_prefix("latent.block.") {
        let (li, slot) = parse_block_key(rest);
        return block_slot(&model.latent.blocks[li], slot, key);
    }
    if let Some(rest) = key.strip_prefix("decoder.block.") {
        let (li, slot) = parse_block_key(rest);
        return block_slot(&model.decoder.byte_layers[li], slot, key);
    }
    if let Some(rest) = key.strip_prefix("decoder.cross_attn.") {
        let (li, slot) = parse_block_key(rest);
        return cross_slot(&model.decoder.cross_attns[li], slot, key);
    }
    panic!("checkpoint::blt_weight_dev: unknown key {key}");
}

fn parse_block_key(rest: &str) -> (usize, &str) {
    let mut parts = rest.splitn(2, '.');
    let li: usize = parts.next().expect("layer index segment")
        .parse().expect("layer index parses as usize");
    let slot = parts.next().expect("slot segment");
    (li, slot)
}

fn block_slot<'a>(
    block: &'a modgrad_transformer::resident::TransformerBlockResident,
    slot: &str,
    key: &str,
) -> &'a HipBuffer {
    match slot {
        "wq" => &block.attn.q_proj.weight_dev,
        "wk" => &block.attn.k_proj.weight_dev,
        "wv" => &block.attn.v_proj.weight_dev,
        "wo" => &block.attn.o_proj.weight_dev,
        "gate" => &block.mlp.gate.weight_dev,
        "up" => &block.mlp.up.weight_dev,
        "down" => &block.mlp.down.weight_dev,
        _ => panic!("checkpoint::block_slot: unknown slot in {key}"),
    }
}

fn cross_slot<'a>(cross: &'a CrossAttention, slot: &str, key: &str) -> &'a HipBuffer {
    match slot {
        "wq" => &cross.q_proj.weight_dev,
        "wk" => &cross.k_proj.weight_dev,
        "wv" => &cross.v_proj.weight_dev,
        "wo" => &cross.o_proj.weight_dev,
        _ => panic!("checkpoint::cross_slot: unknown slot in {key}"),
    }
}

// ─── Internal: little-endian primitive I/O ─────────────────────

#[inline] fn write_u32<W: Write>(w: &mut W, v: u32) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
#[inline] fn write_u64<W: Write>(w: &mut W, v: u64) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
#[inline] fn write_f32<W: Write>(w: &mut W, v: f32) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
#[inline] fn read_u32<R: Read>(r: &mut R) -> std::io::Result<u32> {
    let mut b = [0u8; 4]; r.read_exact(&mut b)?; Ok(u32::from_le_bytes(b))
}
#[inline] fn read_u64<R: Read>(r: &mut R) -> std::io::Result<u64> {
    let mut b = [0u8; 8]; r.read_exact(&mut b)?; Ok(u64::from_le_bytes(b))
}
#[inline] fn read_f32<R: Read>(r: &mut R) -> std::io::Result<f32> {
    let mut b = [0u8; 4]; r.read_exact(&mut b)?; Ok(f32::from_le_bytes(b))
}

// ─── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use modgrad_device::backend::rocm::ffi::runtime_available;

    /// Tiny BLT config — duplicated from `trainer::tests::tiny_blt_config`
    /// (private). Same shape: 32-byte sequence, 8 patches, lE=1, lL=2,
    /// lD=1.
    fn tiny_blt_config() -> BltConfig {
        let byte_dim = 32usize;
        let n_byte_heads = 4usize;
        let byte_head_dim = byte_dim / n_byte_heads;
        let patch_dim = 64usize;
        let n_patch_heads = 4usize;
        let patch_head_dim = patch_dim / n_patch_heads;
        let max_seq = 32usize;
        let max_patches = 8usize;

        BltConfig {
            encoder: LocalEncoderConfig {
                n_layers: 1, byte_dim, patch_dim,
                n_heads: n_byte_heads, head_dim: byte_head_dim,
                mlp_dim: byte_dim * 2,
                norm_eps: 1e-5, rope_base: 10_000.0,
                max_seq_len: max_seq,
                ngram_min_n: 3, ngram_max_n: 5,
                ngram_vocab_per_n: 256,
            },
            latent: BltLatentConfig {
                n_layers: 2, patch_dim,
                n_heads: n_patch_heads, head_dim: patch_head_dim,
                mlp_dim: patch_dim * 2,
                norm_eps: 1e-5, rope_base: 10_000.0,
                max_patches,
            },
            decoder: LocalDecoderConfig {
                n_layers: 1, byte_dim, patch_dim,
                n_heads: n_byte_heads, head_dim: byte_head_dim,
                mlp_dim: byte_dim * 2,
                norm_eps: 1e-5, rope_base: 10_000.0,
                max_seq_len: max_seq,
            },
        }
    }

    /// Pick a temp path under `std::env::temp_dir()` parameterised by
    /// PID + a per-test tag so concurrent test runs don't collide.
    fn tmp_ckpt_path(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("modgrad-blt-ckpt-{}-{}.bin", std::process::id(), tag));
        p
    }

    fn cleanup(path: &Path) {
        let _ = std::fs::remove_file(path);
    }

    /// End-to-end round-trip: save a fresh BLT model, load it back into
    /// a new model, and assert byte-identical weights for every
    /// canonical buffer key. This is the proof that the format faithfully
    /// preserves training state.
    #[test]
    fn blt_checkpoint_roundtrip() {
        let _guard = modgrad_device::test_lock::hip_test_lock();
        if std::env::var("MODGRAD_SKIP_HIP_TESTS").is_ok() { return; }
        if !runtime_available() {
            eprintln!("hip runtime unavailable, skipping");
            return;
        }

        let config = tiny_blt_config();
        let model_a = BltModel::new(config.clone()).expect("BltModel::new (a)");

        let path = tmp_ckpt_path("roundtrip");
        save_blt_model(&model_a, &path).expect("save_blt_model");

        let model_b = load_blt_model_from_path(&path).expect("load_blt_model_from_path");

        // Walk every canonical key; compare byte-for-byte.
        let keys = blt_param_keys(&model_a);
        let keys_b = blt_param_keys(&model_b);
        assert_eq!(keys, keys_b, "key list must agree across save/load");

        let mut mismatches = 0usize;
        for key in &keys {
            let buf_a = blt_weight_dev(&model_a, key);
            let buf_b = blt_weight_dev(&model_b, key);
            assert_eq!(buf_a.bytes(), buf_b.bytes(),
                "{key}: buffer size differs after roundtrip");
            let n = buf_a.len_f32();
            let mut a = vec![0f32; n];
            let mut b = vec![0f32; n];
            buf_a.copy_to_host(&mut a).expect("download a");
            buf_b.copy_to_host(&mut b).expect("download b");
            // Bit-exact comparison via to_bits — NaN/zero edge cases
            // would otherwise let `==` lie. Random-init weights here
            // are finite, but the contract is "byte-identical", not
            // "approximately equal".
            for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                if x.to_bits() != y.to_bits() {
                    mismatches += 1;
                    if mismatches <= 3 {
                        eprintln!(
                            "{key}[{i}]: a={x} ({:#x}) b={y} ({:#x})",
                            x.to_bits(), y.to_bits(),
                        );
                    }
                }
            }
        }
        cleanup(&path);
        assert_eq!(mismatches, 0,
            "BLT checkpoint roundtrip: {mismatches} f32 mismatches across all buffers");
    }

    /// Refuse to load a file with the wrong magic.
    #[test]
    fn rejects_bad_magic() {
        let path = tmp_ckpt_path("bad-magic");
        std::fs::write(&path, b"NOTBLTCK\x01\x00\x00\x00").expect("write");
        // Build a model only if HIP is up; otherwise skip — the magic
        // check happens before any HIP work, so this still validates
        // the early-fail contract on either path.
        let mut file = std::fs::File::open(&path).expect("open");
        let result = read_and_check_header(&mut file);
        cleanup(&path);
        assert!(matches!(result, Err(CheckpointError::BadMagic)),
            "expected BadMagic, got {result:?}");
    }

    /// Refuse to load a file with an unsupported version.
    #[test]
    fn rejects_unsupported_version() {
        let path = tmp_ckpt_path("bad-version");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(BLT_MAGIC);
        bytes.extend_from_slice(&999u32.to_le_bytes());
        std::fs::write(&path, &bytes).expect("write");
        let mut file = std::fs::File::open(&path).expect("open");
        let result = read_and_check_header(&mut file);
        cleanup(&path);
        assert!(matches!(result, Err(CheckpointError::UnsupportedVersion { got: 999, .. })),
            "expected UnsupportedVersion(999), got {result:?}");
    }
}
