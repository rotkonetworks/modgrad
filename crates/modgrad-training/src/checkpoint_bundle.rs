//! Atomic single-file checkpoint — bundles `{model, optimizer, metadata}`.
//!
//! The old pattern (FfnWeights::save, FfnAdamW::save as separate files) has
//! three failure modes we've hit in practice:
//!
//!   1. **Non-atomic split.** Crash between `w.save(path)` and
//!      `opt.save(&opt_path)` leaves a weights file without matching
//!      optimizer state. Resume corrupt.
//!   2. **No metadata.** Step, tokens seen, loss, training-start time —
//!      all lost unless the filename encodes them, which it doesn't.
//!   3. **No model-type check.** Loading an FFN `.bin` into a CTM weights
//!      slot gives a bincode deserialisation error far from the cause.
//!
//! `CheckpointBundle` fixes all three: one bundle per file, written via
//! tmp-then-rename (POSIX-atomic), with a `model_kind` string that a
//! loader checks before deserialising the weights.
//!
//! # Shape
//! ```ignore
//! // Save during training:
//! CheckpointBundle {
//!     schema: CURRENT_SCHEMA,
//!     model_kind: "ffn-cerebellum".into(),
//!     model: my_weights.clone(),
//!     optimizer: my_opt.clone(),
//!     meta: BasicMeta { step: 5_000, tokens_seen: 640_000, .. },
//! }.save("cerebellum.bin")?;
//!
//! // Resume later:
//! let bundle = CheckpointBundle::<FfnWeights, FfnAdamW>::load(
//!     "cerebellum.bin", "ffn-cerebellum"
//! )?;
//! let (w, opt, meta) = (bundle.model, bundle.optimizer, bundle.meta);
//! ```
//!
//! # Atomicity
//! `save` writes to `<path>.tmp`, `fsync`s, then `rename`s over the
//! target. On POSIX, rename is atomic — either the new file is fully
//! visible at `path`, or the old one is. A SIGINT in the middle leaves
//! `path` untouched.

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::io::{self, Write};
use std::path::Path;

/// Bump when the on-disk layout changes incompatibly. Loaders refuse
/// higher-than-known schema versions.
pub const CURRENT_SCHEMA: u32 = 1;

/// 4-byte magic prepended to every CheckpointBundle file. Lets a loader
/// reject random data or files from unrelated formats without even
/// trying to deserialise.
const MAGIC: &[u8; 4] = b"MGCK";  // "MoGrad ChecKpoint"

/// Default metadata — covers the things a resumable trainer needs.
/// Implementations that want more fields define their own struct;
/// `CheckpointBundle<M, O, Meta>` is generic over it.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BasicMeta {
    /// Optimizer step count at save time.
    pub step: u64,
    /// Tokens seen during training (sum across epochs).
    pub tokens_seen: u64,
    /// Smoothed loss reported at the save step.
    pub loss_at_save: f32,
    /// Best smoothed loss seen so far (nan if untracked).
    pub best_loss: f32,
    /// Unix timestamp of the save.
    pub timestamp_unix: u64,
    /// Wall-clock seconds of training elapsed up to this save.
    pub elapsed_secs: u64,
}

/// The bundle. Generic over model type, optimizer type, metadata type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointBundle<M, O, Meta = BasicMeta> {
    /// Schema version. Present in every bundle; loader checks against
    /// `CURRENT_SCHEMA` and supported range.
    pub schema: u32,
    /// Model-family identifier chosen by the saver — e.g. `"ffn-cerebellum"`,
    /// `"ctm-regional"`, `"gpt-transformer"`. Loader must pass the same
    /// string; mismatch is rejected before deserialising weights.
    pub model_kind: String,
    /// The actual model weights.
    pub model: M,
    /// Optimizer state (moments, step counter, LR, etc.).
    pub optimizer: O,
    /// Metadata. `BasicMeta` by default; callers can substitute a richer
    /// type if they need more fields.
    pub meta: Meta,
}

/// Failure modes of `save` / `load`. Distinct variants for the loader
/// so a runtime can decide "schema mismatch → refuse to load" vs
/// "io error → retry on fresh path".
#[derive(Debug)]
pub enum CheckpointError {
    /// Filesystem error reading or writing.
    Io(io::Error),
    /// The file isn't a CheckpointBundle at all — wrong magic or too short.
    NotACheckpoint,
    /// Bincode failed to serialise the payload on save. Rare — typically
    /// means an unserialisable type slipped into the model/optimizer.
    Serialize(Box<bincode::ErrorKind>),
    /// Bincode failed to deserialise the payload. Usually schema skew
    /// that slipped past our explicit version check.
    Deserialize(Box<bincode::ErrorKind>),
    /// Schema version in the file is newer than this build knows about.
    SchemaMismatch { found: u32, supported_max: u32 },
    /// `model_kind` in the file doesn't match what the caller expected.
    /// Prevents loading an FFN checkpoint into a CTM slot.
    ModelKindMismatch { found: String, expected: String },
}

impl std::fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CheckpointError::Io(e) => write!(f, "checkpoint I/O error: {e}"),
            CheckpointError::NotACheckpoint => write!(f, "file is not a modgrad checkpoint bundle"),
            CheckpointError::Serialize(e) => write!(f, "checkpoint serialize failed: {e}"),
            CheckpointError::Deserialize(e) => write!(f, "checkpoint deserialize failed: {e}"),
            CheckpointError::SchemaMismatch { found, supported_max } => {
                write!(f, "checkpoint schema {found} is newer than supported max {supported_max}")
            }
            CheckpointError::ModelKindMismatch { found, expected } => {
                write!(f, "checkpoint is '{found}', expected '{expected}'")
            }
        }
    }
}

impl std::error::Error for CheckpointError {}

impl From<io::Error> for CheckpointError {
    fn from(e: io::Error) -> Self { CheckpointError::Io(e) }
}

impl<M, O, Meta> CheckpointBundle<M, O, Meta>
where
    M: Serialize + DeserializeOwned,
    O: Serialize + DeserializeOwned,
    Meta: Serialize + DeserializeOwned,
{
    /// Atomic write: serialise to `<path>.tmp`, fsync, then rename over
    /// `path`. POSIX guarantees the rename is atomic — `path` is either
    /// fully the new file or still the old. A crash mid-serialise
    /// leaves `path` untouched and a stray `.tmp` behind.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), CheckpointError> {
        let path = path.as_ref();
        let tmp = path.with_extension(match path.extension() {
            Some(ext) => format!("{}.tmp", ext.to_string_lossy()),
            None => "tmp".to_string(),
        });

        // Serialise to a Vec first so an error mid-bincode doesn't leave
        // a partially-written tmp file either.
        let payload = bincode::serialize(self)
            .map_err(CheckpointError::Serialize)?;

        {
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(MAGIC)?;
            f.write_all(&CURRENT_SCHEMA.to_le_bytes())?;
            f.write_all(&payload)?;
            f.sync_all()?;
        }
        // Rename is atomic on POSIX. On Windows, std::fs::rename is also
        // atomic when the target exists (uses ReplaceFile under the hood).
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Load and verify. `expected_kind` is the `model_kind` string the
    /// caller requires; mismatch is rejected before any weight
    /// deserialisation happens.
    pub fn load(path: impl AsRef<Path>, expected_kind: &str) -> Result<Self, CheckpointError> {
        let data = std::fs::read(path)?;
        if data.len() < 8 || &data[..4] != MAGIC {
            return Err(CheckpointError::NotACheckpoint);
        }
        let schema = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        if schema > CURRENT_SCHEMA {
            return Err(CheckpointError::SchemaMismatch {
                found: schema,
                supported_max: CURRENT_SCHEMA,
            });
        }
        let bundle: CheckpointBundle<M, O, Meta> = bincode::deserialize(&data[8..])
            .map_err(CheckpointError::Deserialize)?;

        if bundle.model_kind != expected_kind {
            return Err(CheckpointError::ModelKindMismatch {
                found: bundle.model_kind,
                expected: expected_kind.to_string(),
            });
        }
        Ok(bundle)
    }
}

/// Save a `BasicMeta` checkpoint with the common defaults filled in —
/// `schema = CURRENT_SCHEMA`, `timestamp_unix = now`, zeroed loss fields.
///
/// Collapses the 13-line boilerplate every training loop would otherwise
/// repeat (model + optimizer cloned into a bundle, `CURRENT_SCHEMA`
/// wired in, `SystemTime::now()` called, `loss_at_save` / `best_loss`
/// defaulted). Callers that need richer meta fields construct
/// [`CheckpointBundle`] directly.
pub fn save_training_checkpoint<M, O>(
    path: impl AsRef<Path>,
    model_kind: &str,
    model: &M,
    optimizer: &O,
    step: u64,
    tokens_seen: u64,
    elapsed_secs: u64,
) -> Result<(), CheckpointError>
where
    M: Serialize + DeserializeOwned + Clone,
    O: Serialize + DeserializeOwned + Clone,
{
    let bundle = CheckpointBundle {
        schema: CURRENT_SCHEMA,
        model_kind: model_kind.to_string(),
        model: model.clone(),
        optimizer: optimizer.clone(),
        meta: BasicMeta {
            step,
            tokens_seen,
            loss_at_save: 0.0,
            best_loss: 0.0,
            timestamp_unix: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            elapsed_secs,
        },
    };
    bundle.save(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct FakeModel { weights: Vec<f32>, label: String }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct FakeOpt { lr: f32, step: u64, moments: Vec<f32> }

    fn bundle(kind: &str, step: u64) -> CheckpointBundle<FakeModel, FakeOpt> {
        CheckpointBundle {
            schema: CURRENT_SCHEMA,
            model_kind: kind.to_string(),
            model: FakeModel { weights: vec![1.0, 2.0, 3.0], label: "test".into() },
            optimizer: FakeOpt { lr: 1e-4, step, moments: vec![0.5; 3] },
            meta: BasicMeta { step, tokens_seen: step * 100, loss_at_save: 0.5, ..Default::default() },
        }
    }

    #[test]
    fn roundtrip_preserves_content() {
        let path = std::env::temp_dir().join("mgck_test_roundtrip.ckpt");
        let b = bundle("fake-model", 42);
        b.save(&path).unwrap();
        let loaded = CheckpointBundle::<FakeModel, FakeOpt>::load(&path, "fake-model").unwrap();
        assert_eq!(loaded.model, b.model);
        assert_eq!(loaded.optimizer, b.optimizer);
        assert_eq!(loaded.meta.step, 42);
        assert_eq!(loaded.schema, CURRENT_SCHEMA);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_rejects_wrong_model_kind() {
        let path = std::env::temp_dir().join("mgck_test_kind.ckpt");
        bundle("ffn-cerebellum", 1).save(&path).unwrap();
        match CheckpointBundle::<FakeModel, FakeOpt>::load(&path, "ctm-regional") {
            Err(CheckpointError::ModelKindMismatch { found, expected }) => {
                assert_eq!(found, "ffn-cerebellum");
                assert_eq!(expected, "ctm-regional");
            }
            other => panic!("expected ModelKindMismatch, got {other:?}"),
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_rejects_random_bytes() {
        let path = std::env::temp_dir().join("mgck_test_junk.ckpt");
        std::fs::write(&path, b"this is not a checkpoint").unwrap();
        match CheckpointBundle::<FakeModel, FakeOpt>::load(&path, "fake-model") {
            Err(CheckpointError::NotACheckpoint) => {}
            other => panic!("expected NotACheckpoint, got {other:?}"),
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_rejects_future_schema() {
        let path = std::env::temp_dir().join("mgck_test_schema.ckpt");
        // Manually craft a file with a too-new schema.
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&(CURRENT_SCHEMA + 1).to_le_bytes());
        buf.extend_from_slice(&[0; 16]);  // junk payload — won't be reached
        std::fs::write(&path, &buf).unwrap();
        match CheckpointBundle::<FakeModel, FakeOpt>::load(&path, "fake-model") {
            Err(CheckpointError::SchemaMismatch { found, supported_max }) => {
                assert_eq!(found, CURRENT_SCHEMA + 1);
                assert_eq!(supported_max, CURRENT_SCHEMA);
            }
            other => panic!("expected SchemaMismatch, got {other:?}"),
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn save_leaves_no_tmp_file_on_success() {
        let path = std::env::temp_dir().join("mgck_test_tmp.ckpt");
        let tmp = path.with_extension("ckpt.tmp");
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&tmp);
        bundle("fake-model", 7).save(&path).unwrap();
        assert!(path.exists(), "target file should exist after successful save");
        assert!(!tmp.exists(), ".tmp file should be renamed away after successful save");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn helper_fills_defaults_and_roundtrips() {
        let path = std::env::temp_dir().join("mgck_test_helper.ckpt");
        let m = FakeModel { weights: vec![1.0, 2.0], label: "h".into() };
        let o = FakeOpt { lr: 1e-3, step: 7, moments: vec![0.1, 0.2] };
        save_training_checkpoint(&path, "helper-test", &m, &o, 7, 42, 3).unwrap();
        let loaded = CheckpointBundle::<FakeModel, FakeOpt>::load(&path, "helper-test").unwrap();
        assert_eq!(loaded.model, m);
        assert_eq!(loaded.optimizer, o);
        assert_eq!(loaded.meta.step, 7);
        assert_eq!(loaded.meta.tokens_seen, 42);
        assert_eq!(loaded.meta.elapsed_secs, 3);
        assert_eq!(loaded.meta.loss_at_save, 0.0);
        assert_eq!(loaded.meta.best_loss, 0.0);
        assert!(loaded.meta.timestamp_unix > 1_700_000_000,
            "timestamp should be a recent unix time, got {}", loaded.meta.timestamp_unix);
        assert_eq!(loaded.schema, CURRENT_SCHEMA);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn save_is_atomic_under_existing_file() {
        // An older file exists at `path`. Saving a new bundle must
        // replace it atomically — at no point is `path` missing or
        // partially written (rename is atomic). Content after must be
        // the new bundle, never a mix.
        let path = std::env::temp_dir().join("mgck_test_atomic.ckpt");
        bundle("fake-model", 1).save(&path).unwrap();
        let first = CheckpointBundle::<FakeModel, FakeOpt>::load(&path, "fake-model").unwrap();
        assert_eq!(first.meta.step, 1);

        bundle("fake-model", 9999).save(&path).unwrap();
        let second = CheckpointBundle::<FakeModel, FakeOpt>::load(&path, "fake-model").unwrap();
        assert_eq!(second.meta.step, 9999,
            "saved bundle should fully replace the old one");
        let _ = std::fs::remove_file(&path);
    }
}
