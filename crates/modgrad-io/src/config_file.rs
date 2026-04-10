//! Load model and training configuration from TOML files.
//!
//! A single TOML file IS the experiment specification.
//! `isis train --config train.toml` should be enough to start training.
//!
//! Example train.toml:
//! ```toml
//! [model]
//! d_model = 512
//! d_input = 128
//! iterations = 8
//! synapse_depth = 4
//! heads = 4
//!
//! [training]
//! total_steps = 50000
//! micro_batch = 16
//! lr = 0.001
//! optimizer = "adamw"
//! weight_decay = 0.01
//! warmup_steps = 1000
//! save_every = 5000
//!
//! [data]
//! text = ["train_climbmix.txt"]
//! images = ["cifar10_train_pixels.feat"]
//! audio = []
//! text_weight = 0.7
//! image_weight = 0.3
//! context_len = 64
//! ```

use serde::{Deserialize, Serialize};
use std::io;
use std::path::Path;

/// Full experiment configuration loaded from TOML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentConfig {
    pub model: ModelConfig,
    pub training: TrainingConfig,
    pub data: DataConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    #[serde(default = "default_d_model")]
    pub d_model: usize,
    #[serde(default = "default_d_input")]
    pub d_input: usize,
    #[serde(default = "default_iterations")]
    pub iterations: usize,
    #[serde(default = "default_synapse_depth")]
    pub synapse_depth: usize,
    #[serde(default = "default_heads")]
    pub heads: usize,
    #[serde(default = "default_memory_length")]
    pub memory_length: usize,
    #[serde(default = "default_deep_nlms")]
    pub deep_nlms: bool,
    #[serde(default = "default_memory_hidden")]
    pub memory_hidden_dims: usize,
    #[serde(default = "default_n_synch")]
    pub n_synch_out: usize,
    #[serde(default = "default_n_synch")]
    pub n_synch_action: usize,
    #[serde(default)]
    pub early_exit: bool,
    #[serde(default = "default_certainty")]
    pub certainty_threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    #[serde(default = "default_steps")]
    pub total_steps: usize,
    #[serde(default = "default_micro_batch")]
    pub micro_batch: usize,
    #[serde(default = "default_accum")]
    pub accum_steps: usize,
    #[serde(default = "default_lr")]
    pub lr: f32,
    #[serde(default = "default_optimizer")]
    pub optimizer: String,
    #[serde(default)]
    pub weight_decay: f32,
    #[serde(default = "default_warmup")]
    pub warmup_steps: usize,
    #[serde(default = "default_save_every")]
    pub save_every: usize,
    #[serde(default = "default_log_every")]
    pub log_every: usize,
    #[serde(default = "default_grad_clip")]
    pub grad_clip: f32,
    #[serde(default)]
    pub checkpoint_dir: String,
    #[serde(default)]
    pub resume_from: String,
    #[serde(default = "default_loss")]
    pub loss: String,
    #[serde(default = "default_seed")]
    pub seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    #[serde(default)]
    pub text: Vec<String>,
    #[serde(default)]
    pub images: Vec<String>,
    #[serde(default)]
    pub audio: Vec<String>,
    #[serde(default = "default_text_weight")]
    pub text_weight: f32,
    #[serde(default)]
    pub image_weight: f32,
    #[serde(default)]
    pub audio_weight: f32,
    #[serde(default = "default_context")]
    pub context_len: usize,
}

// Defaults
fn default_d_model() -> usize { 128 }
fn default_d_input() -> usize { 128 }
fn default_iterations() -> usize { 8 }
fn default_synapse_depth() -> usize { 4 }
fn default_heads() -> usize { 4 }
fn default_memory_length() -> usize { 8 }
fn default_deep_nlms() -> bool { true }
fn default_memory_hidden() -> usize { 4 }
fn default_n_synch() -> usize { 64 }
fn default_certainty() -> f32 { 0.95 }
fn default_steps() -> usize { 10000 }
fn default_micro_batch() -> usize { 8 }
fn default_accum() -> usize { 1 }
fn default_lr() -> f32 { 0.001 }
fn default_optimizer() -> String { "adamw".into() }
fn default_warmup() -> usize { 1000 }
fn default_save_every() -> usize { 1000 }
fn default_log_every() -> usize { 100 }
fn default_grad_clip() -> f32 { 1.0 }
fn default_loss() -> String { "ctm".into() }
fn default_seed() -> u64 { 42 }
fn default_text_weight() -> f32 { 1.0 }
fn default_context() -> usize { 64 }

impl ExperimentConfig {
    /// Load from a TOML file.
    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        toml::from_str(&content)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    /// Save to a TOML file (for reproducibility).
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let content = toml::to_string_pretty(self)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, content)
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_minimal_toml() {
        let toml = r#"
[model]
d_model = 256

[training]
total_steps = 5000
lr = 0.0003

[data]
text = ["train.txt"]
"#;
        let cfg: ExperimentConfig = toml::from_str(toml).unwrap();
        assert_eq!(cfg.model.d_model, 256);
        assert_eq!(cfg.training.total_steps, 5000);
        assert_eq!(cfg.data.text, vec!["train.txt"]);
        // Defaults fill in
        assert_eq!(cfg.model.heads, 4);
        assert_eq!(cfg.training.optimizer, "adamw");
    }

    #[test]
    fn roundtrip_save_load() {
        let cfg = ExperimentConfig {
            model: ModelConfig {
                d_model: 512, d_input: 128, iterations: 8,
                synapse_depth: 4, heads: 8, memory_length: 16,
                deep_nlms: true, memory_hidden_dims: 8,
                n_synch_out: 256, n_synch_action: 256,
                early_exit: true, certainty_threshold: 0.9,
            },
            training: TrainingConfig {
                total_steps: 50000, micro_batch: 16, accum_steps: 4,
                lr: 0.0003, optimizer: "adamw".into(), weight_decay: 0.01,
                warmup_steps: 2000, save_every: 5000, log_every: 100,
                grad_clip: 1.0, checkpoint_dir: "checkpoints".into(),
                resume_from: String::new(), loss: "thinking".into(),
                seed: 42,
            },
            data: DataConfig {
                text: vec!["train_climbmix.txt".into()],
                images: vec!["cifar10_train.feat".into()],
                audio: vec![],
                text_weight: 0.7, image_weight: 0.3, audio_weight: 0.0,
                context_len: 64,
            },
        };

        let path = "/tmp/isis_config_test.toml";
        cfg.save(path).unwrap();
        let loaded = ExperimentConfig::load(path).unwrap();
        assert_eq!(loaded.model.d_model, 512);
        assert_eq!(loaded.training.lr, 0.0003);
        assert_eq!(loaded.data.text_weight, 0.7);

        // Verify the TOML is human-readable
        let content = std::fs::read_to_string(path).unwrap();
        assert!(content.contains("[model]"));
        assert!(content.contains("d_model = 512"));

        std::fs::remove_file(path).ok();
    }
}
