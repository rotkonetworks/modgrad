//! Persistent brain: load from disk, learn, auto-save on drop.
//!
//! Enforces the invariant that learned weights survive across runs.
//! `Brain` is the sole owner of `CtmWeights` backed by a file path.
//! Modification through `weights_mut()` marks the brain dirty;
//! `Drop` flushes to disk so forgetting to save is not possible.

use std::io;
use std::path::{Path, PathBuf};

use super::config::CtmConfig;
use super::session::CtmSession;
use super::tick_state::CtmTickState;
use super::weights::{Ctm, CtmWeights};

/// A brain backed by persistent storage.
///
/// Load an existing brain or create one from config. Weights are saved
/// to disk on explicit `save()` or automatically when the `Brain` is
/// dropped after mutation.
pub struct Brain {
    weights: CtmWeights,
    path: PathBuf,
    dirty: bool,
}

impl Brain {
    /// Open an existing brain, or create a new one if the file does not exist.
    ///
    /// When loading, `cfg` is ignored — the stored config wins.
    /// When creating, saves the initial (random) weights immediately.
    pub fn open(path: impl AsRef<Path>, cfg: &CtmConfig) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        if path.exists() {
            return Self::load(&path);
        }
        let ctm = Ctm::new(cfg.clone());
        let (weights, _) = ctm.into_split();
        let mut brain = Self { weights, path, dirty: true };
        brain.save()?;
        Ok(brain)
    }

    /// Load a brain from disk. Fails if the file does not exist.
    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let weights = CtmWeights::load(path.to_str().unwrap_or(""))
            .map_err(|e| io::Error::other(e.to_string()))?;
        Ok(Self { weights, path, dirty: false })
    }

    /// Flush weights to disk.
    pub fn save(&mut self) -> io::Result<()> {
        self.weights
            .save(self.path.to_str().unwrap_or(""))
            .map_err(|e| io::Error::other(e.to_string()))?;
        self.dirty = false;
        Ok(())
    }

    /// Immutable access to weights (forward pass, eval).
    #[inline]
    pub fn weights(&self) -> &CtmWeights { &self.weights }

    /// Mutable access to weights. Marks the brain dirty so it will be
    /// saved on the next `save()` call or on `Drop`.
    #[inline]
    pub fn weights_mut(&mut self) -> &mut CtmWeights {
        self.dirty = true;
        &mut self.weights
    }

    /// The config baked into these weights.
    #[inline]
    pub fn config(&self) -> &CtmConfig { &self.weights.config }

    /// Fresh session (ephemeral per-run state, not persisted).
    pub fn session(&self) -> CtmSession { CtmSession::new(&self.weights.config) }

    /// Fresh tick state.
    pub fn tick_state(&self) -> CtmTickState { self.weights.init_tick_state() }

    /// File path backing this brain.
    pub fn path(&self) -> &Path { &self.path }

    /// Whether weights have been mutated since the last save.
    pub fn is_dirty(&self) -> bool { self.dirty }
}

impl Drop for Brain {
    fn drop(&mut self) {
        if self.dirty {
            if let Err(e) = self.save() {
                eprintln!("brain: failed to save to {}: {e}", self.path.display());
            }
        }
    }
}
