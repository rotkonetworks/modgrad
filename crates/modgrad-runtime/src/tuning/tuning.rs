//! Capability-based parameter governance.
//!
//! Every tunable number in the organism flows through this system.
//! Parameters have PROVENANCE — you can always read them, but to write
//! you need sufficient authority.
//!
//! Four permission tiers (like biological control levels):
//!
//! ```text
//!   Genome (immutable)        → architecture, topology
//!     set by: user/developer, requires restart
//!
//!   Epigenome (developmental) → NLM depth, memory length
//!     set by: parent model during teaching
//!
//!   Hormonal (slow tuning)    → learning rates, noise, thresholds
//!     set by: organism (homeostasis), autoresearcher, parent
//!     bounded by: genome-tier ranges
//!
//!   Neuromodulatory (fast)    → DA, 5HT, NE, ACh levels
//!     set by: organism automatic dynamics
//!     bounded by: hormonal-tier ranges
//! ```
//!
//! Design follows pledge/seccomp philosophy applied to ML:
//! the organism can tune itself within bounds, but can't change its own bounds.
//!
//! Hot-reloadable from JSON. Daemon checks mtime each idle cycle.

use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::SystemTime;

// ─── Permission Tiers ──────────────────────────────────────

/// Authority level required to modify a parameter.
/// Ordered: Genome > Epigenome > Hormonal > Neuromodulatory.
/// A caller with tier T can modify any param at tier ≤ T.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Tier {
    /// Fast modulation — changes per tick.
    /// Set by organism dynamics (neuromodulator release).
    Neuromodulatory,
    /// Slow modulation — changes over hours/days.
    /// Set by homeostasis, autoresearcher, parent.
    Hormonal,
    /// Developmental — set during early training, rarely changed.
    /// Set by parent model during teaching.
    Epigenome,
    /// Immutable at runtime — architecture, topology.
    /// Set by user/developer, requires restart.
    Genome,
}

/// Who made the change. Audit trail, not access control —
/// access control is tier-based.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Changer {
    /// The CTM itself (homeostasis, neuromod dynamics).
    Organism,
    /// Parent model during supervised teaching.
    Parent,
    /// External optimization agent.
    Autoresearcher,
    /// Human developer.
    User,
}

// ─── TuningParam ───────────────────────────────────────────

/// A single tunable value with bounds, tier, and audit trail.
///
/// Reading is free — `.get()` returns `T`.
/// Writing is capability-gated — `.set()` checks tier and clamps to bounds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Param<T: Clone> {
    value: T,
    lo: T,
    hi: T,
    default: T,
    tier: Tier,
    #[serde(default = "default_changer")]
    last_changed_by: Changer,
    #[serde(default)]
    last_changed_at: u64,
}

fn default_changer() -> Changer { Changer::User }

/// Why a `.set()` was rejected.
#[derive(Debug)]
pub enum TuneError {
    /// Caller's tier is below the param's tier.
    InsufficientAuthority { need: Tier, have: Tier },
}

impl std::fmt::Display for TuneError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TuneError::InsufficientAuthority { need, have } =>
                write!(f, "need {:?} authority, have {:?}", need, have),
        }
    }
}

impl<T: Clone + PartialOrd> Param<T> {
    /// Create a new param. Value is clamped to [lo, hi].
    pub fn new(value: T, lo: T, hi: T, tier: Tier) -> Self {
        let clamped = clamp(value.clone(), lo.clone(), hi.clone());
        Self {
            default: clamped.clone(),
            value: clamped,
            lo, hi, tier,
            last_changed_by: Changer::User,
            last_changed_at: now_secs(),
        }
    }

    /// Read the current value. Always succeeds, zero cost.
    #[inline]
    pub fn get(&self) -> &T { &self.value }

    /// Read as owned value.
    #[inline]
    pub fn val(&self) -> T { self.value.clone() }

    /// Attempt to set. Checks tier authority, clamps to bounds.
    /// Returns the actual value set (may be clamped).
    pub fn set(&mut self, value: T, caller_tier: Tier, who: Changer) -> Result<T, TuneError> {
        if caller_tier < self.tier {
            return Err(TuneError::InsufficientAuthority {
                need: self.tier,
                have: caller_tier,
            });
        }
        let clamped = clamp(value, self.lo.clone(), self.hi.clone());
        self.value = clamped.clone();
        self.last_changed_by = who;
        self.last_changed_at = now_secs();
        Ok(clamped)
    }

    /// Widen bounds (only a higher tier can do this).
    pub fn set_bounds(&mut self, lo: T, hi: T, caller_tier: Tier) -> Result<(), TuneError> {
        // Must be strictly above param's tier to change bounds.
        if caller_tier <= self.tier {
            return Err(TuneError::InsufficientAuthority {
                need: next_tier(self.tier),
                have: caller_tier,
            });
        }
        self.lo = lo;
        self.hi = hi;
        // Re-clamp current value
        self.value = clamp(self.value.clone(), self.lo.clone(), self.hi.clone());
        Ok(())
    }

    /// Reset to default.
    pub fn reset(&mut self) {
        self.value = self.default.clone();
    }

    pub fn tier(&self) -> Tier { self.tier }
    pub fn bounds(&self) -> (&T, &T) { (&self.lo, &self.hi) }
}

fn clamp<T: PartialOrd + Clone>(v: T, lo: T, hi: T) -> T {
    if v < lo { lo } else if v > hi { hi } else { v }
}

fn next_tier(t: Tier) -> Tier {
    match t {
        Tier::Neuromodulatory => Tier::Hormonal,
        Tier::Hormonal => Tier::Epigenome,
        Tier::Epigenome => Tier::Genome,
        Tier::Genome => Tier::Genome, // can't go higher
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ─── TuningConfig ──────────────────────────────────────────
//
// The struct IS the schema. Each field is a Param<f32>.
// Grouped by subsystem. Serde handles JSON. No string lookups.

/// All tunable parameters, grouped by subsystem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuningConfig {
    pub noise: NoiseParams,
    pub learning: LearningParams,
    pub sleep: SleepParams,
    pub homeostasis: HomeostasisParams,
    pub neuromod: NeuromodParams,
    pub diversity: DiversityParams,
    pub basal_ganglia: BasalGangliaParams,
    pub memory: MemoryParams,
    /// Gamma cycle: async region synchronization rhythm.
    /// Controlled by harness (hypothalamus), NOT fixed in DNA.
    #[serde(default)]
    pub gamma: GammaParams,
}

/// Gamma cycle parameters — how async regions synchronize.
/// Tuned at runtime by the harness based on arousal, surprise, task demands.
/// Like JAM's slot timing: async execution within window, commit at boundary.
///
/// The BOUNDS are set here. The actual cycle speed is computed each cycle from:
///   ACh (acetylcholine/precision) → speeds up gamma (more firings/window)
///   Inhibition → slows down gamma (fewer firings, longer per-cycle compute)
///   NE (norepinephrine/arousal) → shortens window (reactive mode)
///   5HT (serotonin/gate) → lengthens window (contemplative mode)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GammaParams {
    /// Minimum region firings per gamma window.
    pub min_firings: Param<f32>,
    /// Maximum region firings per window (safety cap).
    pub max_firings: Param<f32>,
    /// Sync coherence threshold for early commit (0.0–1.0).
    pub coherence_threshold: Param<f32>,
    /// ACh sensitivity: how strongly acetylcholine speeds up gamma.
    /// effective_firings = base + ach_gain * ACh_level
    pub ach_gain: Param<f32>,
    /// NE sensitivity: how strongly norepinephrine shortens the window.
    /// effective_max = max_firings / (1 + ne_gain * NE_level)
    pub ne_gain: Param<f32>,
    /// 5HT sensitivity: how strongly serotonin lengthens the window.
    /// effective_min = min_firings * (1 + serotonin_gain * 5HT_level)
    pub serotonin_gain: Param<f32>,
}

/// Biological noise injection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseParams {
    /// Base amplitude of neural noise.
    pub base_amplitude: Param<f32>,
    /// Base vesicle depletion fraction (activity-dependent dropout).
    pub base_dropout: Param<f32>,
    /// NREM noise scaling factor.
    pub nrem_scale: Param<f32>,
    /// REM noise scaling factor (creative exploration).
    pub rem_scale: Param<f32>,
    /// Correlated column noise mixing ratio (vs independent).
    pub column_mix: Param<f32>,
    /// Spontaneous firing amplitude.
    pub spontaneous_amplitude: Param<f32>,
    /// Homeostatic scaling: amplify quiet neurons below this threshold.
    pub quiet_threshold: Param<f32>,
    /// Homeostatic scaling: suppress hyperactive neurons above this threshold.
    pub active_threshold: Param<f32>,
}

/// Learning rates and Hebbian dynamics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningParams {
    /// Surprise modulation range for learning rate.
    pub surprise_lr_min: Param<f32>,
    pub surprise_lr_max: Param<f32>,
    /// Embedding co-occurrence update scale.
    pub embedding_delta_scale: Param<f32>,
    /// Output projection target token LR scale.
    pub output_target_lr_scale: Param<f32>,
    /// Output projection predicted token (contrastive) LR scale.
    pub output_predicted_lr_scale: Param<f32>,
    /// Parent imprint blend per exposure.
    pub parent_blend_rate: Param<f32>,
    /// Sensory LS consolidation blend.
    pub sensory_ls_blend: Param<f32>,
    /// Minimum traces before LS consolidation.
    pub min_ls_traces: Param<f32>,
    /// Hebbian baseline variance floor.
    pub baseline_var_floor: Param<f32>,
    /// Hebbian rebalancing rate.
    pub rebalance_rate: Param<f32>,
}

/// Sleep and consolidation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SleepParams {
    /// NREM synapse consolidation blend.
    pub nrem_blend: Param<f32>,
    /// Initial sleep blend (ramps up with cycles).
    pub blend_initial: Param<f32>,
    /// Maximum sleep blend after ramping.
    pub blend_max: Param<f32>,
    /// Sleep cycles to reach max blend.
    pub blend_ramp_cycles: Param<f32>,
    /// Fear consolidation rate (slow — resists change).
    pub fear_rate: Param<f32>,
    /// Negative consolidation rate.
    pub negative_rate: Param<f32>,
    /// Neutral consolidation rate.
    pub neutral_rate: Param<f32>,
    /// Positive consolidation rate (fast).
    pub positive_rate: Param<f32>,
    /// Replay prune age in seconds (default: 7 days).
    pub prune_age_secs: Param<f32>,
    /// Replay prune surprise threshold.
    pub prune_surprise: Param<f32>,
    /// Minimum sleep quality.
    pub min_sleep_quality: Param<f32>,
}

/// Homeostasis pressure weights and thresholds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HomeostasisParams {
    /// Weight of activation pressure in total.
    pub w_activation: Param<f32>,
    /// Weight of divergence pressure.
    pub w_divergence: Param<f32>,
    /// Weight of drift pressure.
    pub w_drift: Param<f32>,
    /// Weight of buffer pressure.
    pub w_buffer: Param<f32>,
    /// Weight of emotional pressure.
    pub w_emotional: Param<f32>,
    /// Weight of surprise EMA.
    pub w_surprise: Param<f32>,
    /// Activation EMA smoothing.
    pub activation_ema: Param<f32>,
    /// Divergence increment per non-converge.
    pub divergence_increment: Param<f32>,
    /// Divergence decrement per converge.
    pub divergence_decrement: Param<f32>,
    /// Yellow zone threshold.
    pub yellow_threshold: Param<f32>,
    /// Red zone threshold.
    pub red_threshold: Param<f32>,
    /// Forced sleep threshold.
    pub forced_threshold: Param<f32>,
    /// Output quality degradation factor.
    pub quality_degradation: Param<f32>,
}

/// Neuromodulator dynamics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuromodParams {
    /// Dopamine range min.
    pub da_min: Param<f32>,
    /// Dopamine range max.
    pub da_max: Param<f32>,
    /// Serotonin range min.
    pub serotonin_min: Param<f32>,
    /// Serotonin range max.
    pub serotonin_max: Param<f32>,
    /// Serotonin baseline (floor even at zero energy).
    pub serotonin_baseline: Param<f32>,
    /// Serotonin modulation scale.
    pub serotonin_mod_scale: Param<f32>,
    /// NE decay per step.
    pub ne_decay: Param<f32>,
    /// NE floor (tonic arousal).
    pub ne_floor: Param<f32>,
    /// NE maximum.
    pub ne_max: Param<f32>,
    /// Surprise EMA smoothing for dopamine.
    pub surprise_ema_alpha: Param<f32>,
    /// ACh EMA smoothing with attention sync.
    pub ach_ema_alpha: Param<f32>,
}

/// Diversity gating.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiversityParams {
    /// Cosine similarity threshold — above this, representations are collapsing.
    pub collapse_threshold: Param<f32>,
    /// Minimum output traces before LS consolidation.
    pub min_output_traces: Param<f32>,
}

/// Basal ganglia reward learning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasalGangliaParams {
    /// Dopamine baseline for BG three-factor learning.
    pub da_baseline: Param<f32>,
    /// Minimum BG updates before three-factor learning kicks in.
    pub min_updates: Param<f32>,
    /// Teach correction weight toward target.
    pub teach_toward: Param<f32>,
    /// Teach bias additive.
    pub teach_bias: Param<f32>,
}

/// Memory and episode parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryParams {
    /// Recall similarity threshold.
    pub recall_threshold: Param<f32>,
    /// Engram competition similarity threshold.
    pub competition_threshold: Param<f32>,
    /// Replay buffer capacity.
    pub replay_capacity: Param<f32>,
    /// Replay surprise threshold for storage.
    pub replay_surprise: Param<f32>,
    /// Labile window hours (reconsolidation vulnerability).
    pub labile_hours: Param<f32>,
    /// Labile weakness factor.
    pub labile_factor: Param<f32>,
}

// ─── Defaults ──────────────────────────────────────────────

impl Default for TuningConfig {
    fn default() -> Self {
        Self {
            noise: NoiseParams::default(),
            learning: LearningParams::default(),
            sleep: SleepParams::default(),
            homeostasis: HomeostasisParams::default(),
            neuromod: NeuromodParams::default(),
            diversity: DiversityParams::default(),
            basal_ganglia: BasalGangliaParams::default(),
            memory: MemoryParams::default(),
            gamma: GammaParams::default(),
        }
    }
}

impl Default for GammaParams {
    fn default() -> Self {
        Self {
            min_firings:         Param::new(1.0,  1.0, 8.0,  Tier::Hormonal),
            max_firings:         Param::new(4.0,  1.0, 16.0, Tier::Hormonal),
            coherence_threshold: Param::new(0.8,  0.0, 1.0,  Tier::Hormonal),
            ach_gain:            Param::new(2.0,  0.0, 8.0,  Tier::Hormonal),
            ne_gain:             Param::new(1.0,  0.0, 4.0,  Tier::Hormonal),
            serotonin_gain:      Param::new(0.5,  0.0, 4.0,  Tier::Hormonal),
        }
    }
}

impl GammaParams {
    /// Compute effective gamma cycle parameters from current neuromodulator state.
    /// Called by the harness each gamma cycle to set the window timing.
    ///
    /// ACh speeds up (more firings per window).
    /// NE shortens window (fewer max firings — reactive).
    /// 5HT lengthens window (higher minimum — contemplative).
    /// Inhibition is handled per-region (slows individual NLM steps).
    pub fn effective_firings(&self, ach: f32, ne: f32, serotonin: f32) -> (usize, usize) {
        // ACh boosts the base firing count
        let boosted_min = self.min_firings.val() + self.ach_gain.val() * ach;
        // 5HT raises the minimum (contemplative — don't rush)
        let calm_min = boosted_min * (1.0 + self.serotonin_gain.val() * serotonin);
        // NE lowers the maximum (reactive — decide fast)
        let reactive_max = self.max_firings.val() / (1.0 + self.ne_gain.val() * ne);

        let min = (calm_min.round() as usize).max(1);
        let max = (reactive_max.round() as usize).max(min);
        (min, max)
    }
}

impl Default for NoiseParams {
    fn default() -> Self {
        Self {
            base_amplitude:       Param::new(0.1,  0.0, 1.0,  Tier::Hormonal),
            base_dropout:         Param::new(0.15, 0.0, 0.8,  Tier::Hormonal),
            nrem_scale:           Param::new(0.3,  0.0, 1.0,  Tier::Hormonal),
            rem_scale:            Param::new(2.0,  0.5, 5.0,  Tier::Hormonal),
            column_mix:           Param::new(0.7,  0.0, 1.0,  Tier::Hormonal),
            spontaneous_amplitude: Param::new(0.3, 0.0, 1.0,  Tier::Hormonal),
            quiet_threshold:      Param::new(0.1,  0.0, 1.0,  Tier::Hormonal),
            active_threshold:     Param::new(0.5,  0.0, 2.0,  Tier::Hormonal),
        }
    }
}

impl Default for LearningParams {
    fn default() -> Self {
        Self {
            surprise_lr_min:         Param::new(0.1,   0.01, 1.0,  Tier::Hormonal),
            surprise_lr_max:         Param::new(2.0,   0.5,  5.0,  Tier::Hormonal),
            embedding_delta_scale:   Param::new(0.01,  0.0,  0.1,  Tier::Hormonal),
            output_target_lr_scale:  Param::new(0.01,  0.0,  0.1,  Tier::Hormonal),
            output_predicted_lr_scale: Param::new(0.005, 0.0, 0.1, Tier::Hormonal),
            parent_blend_rate:       Param::new(0.05,  0.0,  0.5,  Tier::Epigenome),
            sensory_ls_blend:        Param::new(0.05,  0.0,  0.3,  Tier::Hormonal),
            min_ls_traces:           Param::new(100.0, 10.0, 1000.0, Tier::Hormonal),
            baseline_var_floor:      Param::new(0.01,  0.001, 0.1, Tier::Hormonal),
            rebalance_rate:          Param::new(0.05,  0.0,  0.5,  Tier::Hormonal),
        }
    }
}

impl Default for SleepParams {
    fn default() -> Self {
        Self {
            nrem_blend:         Param::new(0.5,   0.1,  1.0,    Tier::Hormonal),
            blend_initial:      Param::new(0.05,  0.01, 0.3,    Tier::Hormonal),
            blend_max:          Param::new(0.3,   0.05, 0.8,    Tier::Hormonal),
            blend_ramp_cycles:  Param::new(100.0, 10.0, 1000.0, Tier::Hormonal),
            fear_rate:          Param::new(0.05,  0.01, 0.3,    Tier::Hormonal),
            negative_rate:      Param::new(0.15,  0.05, 0.5,    Tier::Hormonal),
            neutral_rate:       Param::new(0.3,   0.1,  0.8,    Tier::Hormonal),
            positive_rate:      Param::new(0.5,   0.1,  1.0,    Tier::Hormonal),
            prune_age_secs:     Param::new(604800.0, 3600.0, 2592000.0, Tier::Hormonal), // 7d, 1h-30d
            prune_surprise:     Param::new(0.5,   0.0,  2.0,    Tier::Hormonal),
            min_sleep_quality:  Param::new(0.3,   0.1,  1.0,    Tier::Hormonal),
        }
    }
}

impl Default for HomeostasisParams {
    fn default() -> Self {
        Self {
            w_activation:          Param::new(0.25,  0.0, 1.0,  Tier::Hormonal),
            w_divergence:          Param::new(0.15,  0.0, 1.0,  Tier::Hormonal),
            w_drift:               Param::new(0.20,  0.0, 1.0,  Tier::Hormonal),
            w_buffer:              Param::new(0.20,  0.0, 1.0,  Tier::Hormonal),
            w_emotional:           Param::new(0.10,  0.0, 1.0,  Tier::Hormonal),
            w_surprise:            Param::new(0.10,  0.0, 1.0,  Tier::Hormonal),
            activation_ema:        Param::new(0.95,  0.5, 0.999, Tier::Hormonal),
            divergence_increment:  Param::new(0.02,  0.001, 0.1, Tier::Hormonal),
            divergence_decrement:  Param::new(0.005, 0.001, 0.05, Tier::Hormonal),
            yellow_threshold:      Param::new(0.4,   0.1, 0.8,  Tier::Hormonal),
            red_threshold:         Param::new(0.7,   0.3, 0.95, Tier::Hormonal),
            forced_threshold:      Param::new(1.0,   0.5, 1.5,  Tier::Hormonal),
            quality_degradation:   Param::new(0.6,   0.1, 1.0,  Tier::Hormonal),
        }
    }
}

impl Default for NeuromodParams {
    fn default() -> Self {
        Self {
            da_min:              Param::new(0.5,  0.0, 1.0,  Tier::Hormonal),
            da_max:              Param::new(1.0,  0.5, 2.0,  Tier::Hormonal),
            serotonin_min:       Param::new(0.1,  0.0, 0.5,  Tier::Hormonal),
            serotonin_max:       Param::new(1.0,  0.5, 2.0,  Tier::Hormonal),
            serotonin_baseline:  Param::new(0.4,  0.0, 0.8,  Tier::Hormonal),
            serotonin_mod_scale: Param::new(0.6,  0.0, 2.0,  Tier::Hormonal),
            ne_decay:            Param::new(0.95, 0.8, 0.999, Tier::Hormonal),
            ne_floor:            Param::new(0.3,  0.0, 1.0,  Tier::Hormonal),
            ne_max:              Param::new(3.0,  1.0, 10.0, Tier::Hormonal),
            surprise_ema_alpha:  Param::new(0.9,  0.5, 0.999, Tier::Hormonal),
            ach_ema_alpha:       Param::new(0.8,  0.5, 0.999, Tier::Neuromodulatory),
        }
    }
}

impl Default for DiversityParams {
    fn default() -> Self {
        Self {
            collapse_threshold: Param::new(0.9, 0.5, 1.0, Tier::Hormonal),
            min_output_traces:  Param::new(200.0, 50.0, 1000.0, Tier::Hormonal),
        }
    }
}

impl Default for BasalGangliaParams {
    fn default() -> Self {
        Self {
            da_baseline:  Param::new(0.7,  0.0, 1.0,  Tier::Hormonal),
            min_updates:  Param::new(32.0, 1.0, 100.0, Tier::Hormonal),
            teach_toward: Param::new(0.7,  0.1, 1.0,  Tier::Hormonal),
            teach_bias:   Param::new(0.5,  0.0, 2.0,  Tier::Hormonal),
        }
    }
}

impl Default for MemoryParams {
    fn default() -> Self {
        Self {
            recall_threshold:     Param::new(0.70, 0.3, 0.95, Tier::Hormonal),
            competition_threshold: Param::new(0.80, 0.5, 0.99, Tier::Hormonal),
            replay_capacity:      Param::new(64.0, 16.0, 1024.0, Tier::Epigenome),
            replay_surprise:      Param::new(1.5,  0.0,  5.0,  Tier::Hormonal),
            labile_hours:         Param::new(6.0,  1.0,  24.0, Tier::Hormonal),
            labile_factor:        Param::new(0.95, 0.5,  1.0,  Tier::Hormonal),
        }
    }
}

// ─── Hot Reload ────────────────────────────────────────────

/// Registry: wraps TuningConfig with hot-reload state.
#[derive(Debug, Clone)]
pub struct TuningRegistry {
    pub config: TuningConfig,
    /// Path to JSON file for hot-reload.
    file_path: Option<String>,
    /// Last known mtime of the JSON file.
    last_mtime: Option<u64>,
}

impl TuningRegistry {
    /// Create with defaults, no file.
    pub fn new() -> Self {
        Self {
            config: TuningConfig::default(),
            file_path: None,
            last_mtime: None,
        }
    }

    /// Create from a JSON file. Falls back to defaults if file doesn't exist.
    pub fn from_file(path: &str) -> Self {
        let config = match std::fs::read_to_string(path) {
            Ok(json) => match serde_json::from_str(&json) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("[tuning] parse error in {path}: {e}, using defaults");
                    TuningConfig::default()
                }
            },
            Err(_) => TuningConfig::default(),
        };
        let mtime = file_mtime(path);
        Self {
            config,
            file_path: Some(path.to_string()),
            last_mtime: mtime,
        }
    }

    /// Check if the JSON file has changed. If so, reload.
    /// Returns true if config was reloaded.
    pub fn check_reload(&mut self) -> bool {
        let path = match &self.file_path {
            Some(p) => p.clone(),
            None => return false,
        };
        let current_mtime = file_mtime(&path);
        if current_mtime == self.last_mtime {
            return false;
        }
        match std::fs::read_to_string(&path) {
            Ok(json) => match serde_json::from_str::<TuningConfig>(&json) {
                Ok(new_config) => {
                    eprintln!("[tuning] reloaded from {path}");
                    self.config = new_config;
                    self.last_mtime = current_mtime;
                    true
                }
                Err(e) => {
                    eprintln!("[tuning] parse error on reload: {e}");
                    self.last_mtime = current_mtime; // don't retry every tick
                    false
                }
            },
            Err(e) => {
                eprintln!("[tuning] read error on reload: {e}");
                false
            }
        }
    }

    /// Save current config to file.
    pub fn save(&self) -> Result<(), String> {
        let path = self.file_path.as_ref().ok_or("no file path set")?;
        let json = serde_json::to_string_pretty(&self.config)
            .map_err(|e| format!("serialize: {e}"))?;
        std::fs::write(path, json).map_err(|e| format!("write {path}: {e}"))
    }

    /// Set the file path for hot-reload.
    pub fn set_file(&mut self, path: &str) {
        self.file_path = Some(path.to_string());
        self.last_mtime = file_mtime(path);
    }

    /// Dump a default config to a file (for bootstrapping).
    pub fn write_defaults(path: &str) -> Result<(), String> {
        let config = TuningConfig::default();
        let json = serde_json::to_string_pretty(&config)
            .map_err(|e| format!("serialize: {e}"))?;
        std::fs::write(path, json).map_err(|e| format!("write: {e}"))
    }
}

fn file_mtime(path: &str) -> Option<u64> {
    Path::new(path).metadata().ok()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
}

// ─── Summary / Inspection ──────────────────────────────────

impl TuningConfig {
    /// Human-readable summary of all params that differ from defaults.
    pub fn diff_from_default(&self) -> Vec<String> {
        let default = TuningConfig::default();
        let self_json = serde_json::to_value(self).unwrap_or_default();
        let default_json = serde_json::to_value(&default).unwrap_or_default();
        let mut diffs = Vec::new();
        collect_diffs("", &self_json, &default_json, &mut diffs);
        diffs
    }
}

fn collect_diffs(prefix: &str, a: &serde_json::Value, b: &serde_json::Value, out: &mut Vec<String>) {
    // Skip audit fields — only compare tuning values and bounds
    let skip = ["last_changed_at", "last_changed_by"];
    if let Some(leaf) = prefix.rsplit('.').next() {
        if skip.contains(&leaf) { return; }
    }
    match (a, b) {
        (serde_json::Value::Object(am), serde_json::Value::Object(bm)) => {
            for (k, av) in am {
                let key = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
                match bm.get(k) {
                    Some(bv) => collect_diffs(&key, av, bv, out),
                    None => out.push(format!("{key}: {av} (new)")),
                }
            }
        }
        (serde_json::Value::Number(an), serde_json::Value::Number(bn)) if an != bn => {
            out.push(format!("{prefix}: {:.4} → {:.4}",
                bn.as_f64().unwrap_or(0.0), an.as_f64().unwrap_or(0.0)));
        }
        _ if a != b => {
            out.push(format!("{prefix}: {b} → {a}"));
        }
        _ => {}
    }
}

// ─── Tests ─────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn param_clamps_on_create() {
        let p = Param::new(5.0_f32, 0.0, 1.0, Tier::Hormonal);
        assert_eq!(*p.get(), 1.0);
    }

    #[test]
    fn param_set_respects_tier() {
        let mut p = Param::new(0.5_f32, 0.0, 1.0, Tier::Hormonal);
        // Neuromodulatory < Hormonal → rejected
        assert!(p.set(0.9, Tier::Neuromodulatory, Changer::Organism).is_err());
        // Hormonal = Hormonal → accepted
        assert!(p.set(0.9, Tier::Hormonal, Changer::Organism).is_ok());
        assert_eq!(*p.get(), 0.9);
        // Genome > Hormonal → accepted
        assert!(p.set(0.1, Tier::Genome, Changer::User).is_ok());
        assert_eq!(*p.get(), 0.1);
    }

    #[test]
    fn param_set_clamps() {
        let mut p = Param::new(0.5_f32, 0.0, 1.0, Tier::Hormonal);
        let v = p.set(999.0, Tier::Hormonal, Changer::Autoresearcher).unwrap();
        assert_eq!(v, 1.0);
        assert_eq!(*p.get(), 1.0);
    }

    #[test]
    fn bounds_require_higher_tier() {
        let mut p = Param::new(0.5_f32, 0.0, 1.0, Tier::Hormonal);
        // Hormonal can't change Hormonal bounds
        assert!(p.set_bounds(0.0, 2.0, Tier::Hormonal).is_err());
        // Epigenome can change Hormonal bounds
        assert!(p.set_bounds(0.0, 2.0, Tier::Epigenome).is_ok());
        // Now value can go to 2.0
        assert!(p.set(1.5, Tier::Hormonal, Changer::Organism).is_ok());
        assert_eq!(*p.get(), 1.5);
    }

    #[test]
    fn default_config_roundtrips_json() {
        let config = TuningConfig::default();
        let json = serde_json::to_string_pretty(&config).unwrap();
        let parsed: TuningConfig = serde_json::from_str(&json).unwrap();
        // Check a few values survived
        assert_eq!(*parsed.noise.base_amplitude.get(), 0.1);
        assert_eq!(*parsed.diversity.collapse_threshold.get(), 0.9);
        assert_eq!(*parsed.basal_ganglia.da_baseline.get(), 0.7);
    }

    #[test]
    fn diff_from_default_empty_for_defaults() {
        let config = TuningConfig::default();
        let diffs = config.diff_from_default();
        assert!(diffs.is_empty(), "expected no diffs, got: {:?}", diffs);
    }
}
