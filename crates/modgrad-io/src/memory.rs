//! Memory bank: the persistent, JSON-serializable brain state.
//!
//! This is pure data — no behavior. Filters read/write it.
//! Serialize to JSON for persistence, IPFS, or on-chain storage.

use serde::{Deserialize, Serialize};
use crate::types::*;
use crate::episode::{find_competitors, effective_strength};

use std::time::{SystemTime, UNIX_EPOCH};

fn now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64()
}

/// The complete memory bank — all brain state in one serializable struct.
/// Model identity — exactly which weights produced these keys.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelId {
    /// HuggingFace model name (e.g. "Qwen/Qwen2.5-0.5B")
    pub model: String,
    /// Backend used to produce keys (e.g. "onnx", "gguf", "transformers")
    pub backend: String,
    /// Quantization format (e.g. "f32", "f16", "q4_k_m", "q8_0")
    pub quant: String,
    /// Hidden dimension (e.g. 896)
    pub hidden_dim: u32,
    /// Which hidden state extraction point (e.g. "pre_mlp_layer23", "post_norm")
    pub extraction: String,
    /// EOS token ID for this model (e.g. 151643 for Qwen, 2 for Llama)
    #[serde(default = "default_eos")]
    pub eos_token_id: i64,
}

fn default_eos() -> i64 { 151643 }

impl Default for ModelId {
    fn default() -> Self {
        Self {
            model: "Qwen/Qwen2.5-0.5B".into(),
            backend: "onnx".into(),
            quant: "f32".into(),
            hidden_dim: 896,
            extraction: "pre_mlp_layer23".into(),
            eos_token_id: 151643,
        }
    }
}

impl ModelId {
    /// Compact string for filenames: "qwen2.5-0.5b.onnx.f32"
    pub fn file_stem(&self) -> String {
        let short_model = self.model
            .rsplit('/')
            .next()
            .unwrap_or(&self.model)
            .to_lowercase();
        format!("{}.{}.{}", short_model, self.backend, self.quant)
    }

    /// Check compatibility with another ModelId.
    /// Returns None if compatible, or a reason string.
    pub fn check_compatible(&self, other: &ModelId) -> Option<String> {
        if self.model != other.model {
            return Some(format!("model mismatch: '{}' vs '{}'", self.model, other.model));
        }
        if self.backend != other.backend {
            return Some(format!("backend mismatch: '{}' vs '{}'", self.backend, other.backend));
        }
        if self.quant != other.quant {
            return Some(format!("quant mismatch: '{}' vs '{}'", self.quant, other.quant));
        }
        if self.extraction != other.extraction {
            return Some(format!("extraction point mismatch: '{}' vs '{}'", self.extraction, other.extraction));
        }
        None
    }
}

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}:{}:{}", self.model, self.backend, self.quant, self.extraction)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryBank {
    pub version: u32,
    /// Exact model identity — keys are ONLY valid for this exact config.
    pub model_id: ModelId,
    pub threshold: f32,
    pub alters: Vec<Alter>,
    pub rules: Vec<Rule>,
    pub avoidances: Vec<Avoidance>,
}

impl Default for MemoryBank {
    fn default() -> Self {
        Self {
            version: 2,
            model_id: ModelId::default(),
            threshold: 0.70,
            alters: Vec::new(),
            rules: Vec::new(),
            avoidances: Vec::new(),
        }
    }
}

/// Common function words to skip when creating content keys.
const SKIP_TOKENS: &[&str] = &[
    "The", "the", " the", " The", " a", " A", " an", " is", " are",
    " was", " were", " for", " of", " in", " on", " at", " to",
    " and", " or", " it", " that", " this", " with", " from", " by",
    " not", " but", " if", " so", " as", " has", " have", " had",
    " do", " does", " did", ".", ",", "!", "?", ":", ";",
];

impl MemoryBank {
    pub fn is_skip_token(token: &str) -> bool {
        SKIP_TOKENS.contains(&token)
    }

    /// Check if this memory bank is compatible with the given model.
    /// Returns None if compatible, or a warning message.
    pub fn check_compatible(&self, expected: &ModelId) -> Option<String> {
        self.model_id.check_compatible(expected)
    }

    /// Store an episodic memory with engram competition.
    pub fn teach(
        &mut self,
        prompt: &str,
        answer: &str,
        alter_name: &str,
        content_keys: Vec<(Vec<f32>, String, i32)>,
        prompt_end_key: Vec<f32>,
        logit_biases: Vec<LogitBias>,
        importance: f32,
        surprise: f32,
    ) {
        let strength = (surprise * importance).clamp(0.1, 10.0);

        let mut keys: Vec<ContentKey> = content_keys
            .into_iter()
            .map(|(key, token, pos)| ContentKey { key, token, position: pos })
            .collect();

        keys.push(ContentKey {
            key: prompt_end_key.clone(),
            token: "[PROMPT_END]".into(),
            position: -1,
        });

        let episode = Episode {
            prompt: prompt.into(),
            answer: answer.into(),
            alter: alter_name.into(),
            keys,
            logit_biases,
            strength,
            recall_count: 0,
            created_at: now(),
            consolidated: false,
            consolidation_score: 0.0,
            sleep_cycles: 0,
            valence: Valence::Neutral,
            last_recalled_at: 0.0,
            visible_to: Vec::new(),
        };

        // Engram competition
        let competitors = find_competitors(self, &prompt_end_key, 0.80);
        for &(ai, ei, _) in &competitors {
            let existing = &self.alters[ai].episodes[ei];
            if existing.answer.to_lowercase() == answer.to_lowercase() {
                // Reinforcement
                let existing = &mut self.alters[ai].episodes[ei];
                existing.recall_count += 3;
                existing.strength = existing.strength.max(strength);
                return;
            } else if effective_strength(existing, now()) > strength {
                // Old wins — store weakly
                let mut weak = episode;
                weak.strength *= 0.3;
                self.ensure_alter(alter_name);
                self.get_alter_mut(alter_name).unwrap().episodes.push(weak);
                return;
            } else {
                // New wins — suppress old
                let existing = &mut self.alters[ai].episodes[ei];
                existing.strength *= 0.1;
            }
        }

        self.ensure_alter(alter_name);
        self.get_alter_mut(alter_name).unwrap().episodes.push(episode);
    }

    pub fn add_rule(&mut self, instruction: &str, priority: f32, trigger: &str) {
        self.rules.push(Rule {
            instruction: instruction.into(),
            priority,
            trigger: trigger.into(),
            active: true,
        });
    }

    pub fn add_avoidance(&mut self, pattern: &str, reason: &str, key: Vec<f32>,
                          suppress_token_ids: Vec<u32>, strength: f32) {
        self.avoidances.push(Avoidance {
            pattern: pattern.into(),
            reason: reason.into(),
            key,
            suppress_token_ids,
            strength,
            active: true,
        });
    }

    fn ensure_alter(&mut self, name: &str) {
        if !self.alters.iter().any(|a| a.name == name) {
            self.alters.push(Alter {
                name: name.into(),
                episodes: Vec::new(),
                attention_bias: Vec::new(),
                can_see: Vec::new(),
            });
        }
    }

    fn get_alter_mut(&mut self, name: &str) -> Option<&mut Alter> {
        self.alters.iter_mut().find(|a| a.name == name)
    }

    /// Save as JSON.
    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let data = serde_json::to_string_pretty(self)?;
        std::fs::write(path, data)?;
        Ok(())
    }

    /// Load from JSON.
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&data)?)
    }

    /// Save as FlatBuffers with the given key quantization format.
    pub fn save_fb(&self, path: &str, format: modgrad_persist::quantize::KeyFormat) -> Result<(), Box<dyn std::error::Error>> {
        let bytes = crate::flatbuf::serialize(self, format);
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Load from FlatBuffers. Keys are dequantized back to f32.
    pub fn load_fb(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path)?;
        crate::flatbuf::deserialize(&bytes).map_err(|e| e.into())
    }

    /// Smart load: picks format based on file extension (.fb → FlatBuffers, else JSON).
    pub fn open(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        if path.ends_with(".fb") {
            Self::load_fb(path)
        } else {
            Self::load(path)
        }
    }

    /// Smart save: picks format based on file extension.
    /// FlatBuffers defaults to f32; use save_fb() for other quantization.
    pub fn write(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        if path.ends_with(".fb") {
            self.save_fb(path, modgrad_persist::quantize::KeyFormat::F32)
        } else {
            self.save(path)
        }
    }
}
