//! Checkpoint manager: periodic save, best model tracking, resume.
//!
//! Replaces ad-hoc brain.save() calls with structured checkpoint
//! management. Keeps last K checkpoints, tracks best by metric,
//! enables training resume after crash.

use serde::{Serialize, de::DeserializeOwned};
use std::path::{Path, PathBuf};
use std::io;

/// Metadata stored alongside each checkpoint.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CheckpointMeta {
    pub step: usize,
    pub loss: f32,
    pub metrics: Vec<(String, f32)>,
    pub timestamp: u64,
}

/// Manages checkpoints in a directory.
pub struct CheckpointManager {
    dir: PathBuf,
    /// Save every N steps.
    save_every: usize,
    /// Keep at most K checkpoints (oldest deleted).
    keep_last: usize,
    /// Best metric value seen (for best-model tracking).
    best_metric: f32,
    best_step: Option<usize>,
    /// Metric name to track for "best" (e.g., "loss").
    track_metric: String,
    /// Lower is better (true for loss, false for accuracy).
    lower_is_better: bool,
}

impl CheckpointManager {
    pub fn new(
        dir: impl AsRef<Path>,
        save_every: usize,
        keep_last: usize,
        track_metric: &str,
        lower_is_better: bool,
    ) -> io::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;
        Ok(Self {
            dir,
            save_every,
            keep_last,
            best_metric: if lower_is_better { f32::MAX } else { f32::MIN },
            best_step: None,
            track_metric: track_metric.to_string(),
            lower_is_better,
        })
    }

    /// Should we save at this step?
    pub fn should_save(&self, step: usize) -> bool {
        step > 0 && step % self.save_every == 0
    }

    /// Save a checkpoint. Returns true if this is the new best.
    pub fn save<T: Serialize>(
        &mut self,
        weights: &T,
        step: usize,
        loss: f32,
        metrics: &[(String, f32)],
    ) -> io::Result<bool> {
        let path = self.dir.join(format!("step_{:08}.bin", step));
        modgrad_persist::persist::save(weights, &path)?;

        // Save metadata
        let meta = CheckpointMeta {
            step,
            loss,
            metrics: metrics.to_vec(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };
        let meta_path = self.dir.join(format!("step_{:08}.meta.json", step));
        let meta_json = serde_json::to_string_pretty(&meta)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        std::fs::write(&meta_path, meta_json)?;

        // Check if this is the best
        let tracked = metrics.iter()
            .find(|(k, _)| k == &self.track_metric)
            .map(|(_, v)| *v)
            .unwrap_or(loss);

        let is_best = if self.lower_is_better {
            tracked < self.best_metric
        } else {
            tracked > self.best_metric
        };

        if is_best {
            self.best_metric = tracked;
            self.best_step = Some(step);
            // Symlink best
            let best_path = self.dir.join("best.bin");
            let _ = std::fs::remove_file(&best_path);
            // Copy instead of symlink for portability
            std::fs::copy(&path, &best_path)?;
        }

        // Prune old checkpoints
        self.prune()?;

        Ok(is_best)
    }

    /// Load the latest checkpoint.
    pub fn load_latest<T: DeserializeOwned>(&self) -> io::Result<(T, CheckpointMeta)> {
        let (path, meta_path) = self.find_latest()?;
        let weights: T = modgrad_persist::persist::load(&path)?;
        let meta_str = std::fs::read_to_string(&meta_path)?;
        let meta: CheckpointMeta = serde_json::from_str(&meta_str)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        Ok((weights, meta))
    }

    /// Load the best checkpoint.
    pub fn load_best<T: DeserializeOwned>(&self) -> io::Result<T> {
        let path = self.dir.join("best.bin");
        modgrad_persist::persist::load(&path)
    }

    /// The step of the best checkpoint.
    pub fn best_step(&self) -> Option<usize> { self.best_step }

    /// Find the latest checkpoint files.
    fn find_latest(&self) -> io::Result<(PathBuf, PathBuf)> {
        let mut checkpoints: Vec<PathBuf> = std::fs::read_dir(&self.dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("bin"))
            .filter(|e| e.file_name().to_string_lossy().starts_with("step_"))
            .map(|e| e.path())
            .collect();
        checkpoints.sort();
        let latest = checkpoints.last()
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "no checkpoints found"))?;
        let meta = latest.with_extension("meta.json");
        Ok((latest.clone(), meta))
    }

    /// Delete old checkpoints, keeping only the last `keep_last`.
    fn prune(&self) -> io::Result<()> {
        let mut checkpoints: Vec<PathBuf> = std::fs::read_dir(&self.dir)?
            .filter_map(|e| e.ok())
            .filter(|e| {
                let name = e.file_name().to_string_lossy().to_string();
                name.starts_with("step_") && name.ends_with(".bin") && !name.contains("meta")
            })
            .map(|e| e.path())
            .collect();
        checkpoints.sort();

        while checkpoints.len() > self.keep_last {
            if let Some(oldest) = checkpoints.first() {
                let meta = oldest.with_extension("meta.json");
                let _ = std::fs::remove_file(oldest);
                let _ = std::fs::remove_file(&meta);
                checkpoints.remove(0);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_save_load_prune() {
        let dir = "/tmp/isis_checkpoint_test";
        let _ = std::fs::remove_dir_all(dir);

        let mut mgr = CheckpointManager::new(dir, 10, 3, "loss", true).unwrap();

        let weights = vec![1.0f32, 2.0, 3.0];

        // Save 5 checkpoints
        for step in (10..=50).step_by(10) {
            let loss = 1.0 - step as f32 / 100.0;
            mgr.save(&weights, step, loss, &[("loss".into(), loss)]).unwrap();
        }

        // Should have kept only last 3
        let bins: Vec<_> = std::fs::read_dir(dir).unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("bin"))
            .filter(|e| e.file_name().to_string_lossy().starts_with("step_"))
            .collect();
        assert!(bins.len() <= 3, "should keep at most 3, got {}", bins.len());

        // Load latest
        let (loaded, meta): (Vec<f32>, _) = mgr.load_latest().unwrap();
        assert_eq!(loaded, weights);
        assert_eq!(meta.step, 50);

        // Best should be step 50 (lowest loss)
        assert_eq!(mgr.best_step(), Some(50));

        let _ = std::fs::remove_dir_all(dir);
    }
}
