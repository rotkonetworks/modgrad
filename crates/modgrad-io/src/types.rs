use serde::{Deserialize, Serialize};

/// A memory key — the hidden state fingerprint of a prompt context.
/// 896 floats for Qwen 2.5-0.5B, stored as f32 or quantized to i8.
pub type Key = Vec<f32>;

/// Logit bias for one answer token position.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogitBias {
    pub token_id: u32,
    pub token: String,
    pub strength: f32,
    /// Tokens to suppress (negative bias)
    #[serde(default)]
    pub suppress: Vec<(u32, f32)>,
}

/// Emotional valence of a memory.
/// Fear memories resist consolidation (PTSD model).
/// Joy memories consolidate faster.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Valence {
    /// Neutral: normal consolidation dynamics
    Neutral,
    /// Positive: consolidates faster, recalled with positive bias
    Positive,
    /// Negative: resists consolidation, triggers avoidance-like suppression
    Negative,
    /// Fear: strongly resists consolidation, can block other recalls
    Fear,
}

impl Default for Valence {
    fn default() -> Self { Valence::Neutral }
}

/// One episodic memory — a fact with multi-key retrieval and consolidation strength.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    pub prompt: String,
    pub answer: String,
    pub alter: String,
    /// Content-word keys — any fragment retrieves this memory
    pub keys: Vec<ContentKey>,
    /// Sequential logit biases — one per answer subword token
    pub logit_biases: Vec<LogitBias>,
    /// Consolidation strength: dopamine × serotonin × norepinephrine
    pub strength: f32,
    /// Recall count — strengthens with use (reconsolidation)
    pub recall_count: u32,
    /// Unix timestamp of creation
    pub created_at: f64,
    /// Whether this memory has been fully consolidated into CTM weights.
    #[serde(default)]
    pub consolidated: bool,
    /// Consolidation score: how well the CTM can reproduce this memory.
    #[serde(default)]
    pub consolidation_score: f32,
    /// Number of sleep cycles this episode has been through.
    #[serde(default)]
    pub sleep_cycles: u32,
    /// Emotional valence — affects consolidation and recall dynamics.
    #[serde(default)]
    pub valence: Valence,
    /// Reconsolidation window: unix timestamp when last recalled.
    /// For ~6h after recall, the memory is labile (can be modified/weakened).
    /// Reference: Nader (2003) "Memory traces unbound"
    #[serde(default)]
    pub last_recalled_at: f64,
    /// Which alters can see this memory. Empty = visible to all.
    /// Enables asymmetric inter-alter amnesia barriers.
    /// Reference: Reinders et al. (2006) DID fMRI studies.
    #[serde(default)]
    pub visible_to: Vec<String>,
}

/// A key associated with a content word at a specific position.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentKey {
    pub key: Key,
    pub token: String,
    pub position: i32, // -1 = PROMPT_END special key
}

/// One self-state with its own memory bank and CTM configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alter {
    pub name: String,
    pub episodes: Vec<Episode>,
    /// Per-alter attention region bias — different alters route through
    /// the thalamic gate differently. This is what makes alters genuinely
    /// different ways of THINKING, not just different memory banks.
    /// Reference: Reinders et al. (2006) — different alters show
    /// different thalamic activation patterns on fMRI.
    #[serde(default)]
    pub attention_bias: Vec<f32>,
    /// Which other alters this alter can see memories from.
    /// Empty = can see own memories only (amnesic barrier).
    /// ["*"] = can see all alters (co-conscious).
    #[serde(default)]
    pub can_see: Vec<String>,
}

/// A behavioral rule (cerebellar procedural memory).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rule {
    pub instruction: String,
    pub priority: f32,
    /// Optional: only active when this keyword is in context
    #[serde(default)]
    pub trigger: String,
    pub active: bool,
}

/// An avoidance pattern (amygdala).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Avoidance {
    pub pattern: String,
    pub reason: String,
    pub key: Key,
    pub suppress_token_ids: Vec<u32>,
    pub strength: f32,
    pub active: bool,
}

/// Working memory entry.
#[derive(Debug, Clone)]
pub struct WorkingMemoryEntry {
    pub hidden: Vec<f32>,
    pub text: String,
    pub timestamp: f64,
}

// MemoryBank is defined in memory.rs (the canonical location).
// Re-exported from lib.rs.
