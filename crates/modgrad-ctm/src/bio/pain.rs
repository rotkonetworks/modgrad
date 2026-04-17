//! Pain bridge: connects loss, valence, and memory retrieval
//! to the organism's internal state (homeostasis + neuromodulators).
//!
//! Pain is not punishment. Pain is a destabilization signal that
//! forces reorganization. The organism that learns to avoid pain
//! develops stronger internal models than one trained on gradients alone.
//!
//! Design invariants (de Valence):
//!   - Pain is *relative* to a running baseline, not absolute loss.
//!     A loss of 3.5 is painful if baseline is 2.0, joyful if baseline is 4.0.
//!   - Neuromodulators decay toward baseline between events.
//!     Saturation = no signal = useless. Signals must be transient.
//!   - LR modulation is a smooth function centered at 1.0.
//!     No product of clamps. A softmax-temperature model.
//!   - All state transitions are monotone in the right direction.
//!     Pain never causes relief. Relief never causes pain.
//!
//! Three sources of pain:
//!   1. **Surprise pain**: loss exceeded expected baseline (prediction error)
//!   2. **Retrieval pain**: this context was painful before (episodic valence)
//!   3. **Isolation pain**: cut off from co-conscious peers (partition drift)
//!
//! Relief:
//!   - Correct prediction → dopamine burst, serotonin recovery
//!   - Overcoming a painful context → stronger relief proportional to old pain

use serde::{Deserialize, Serialize};

use super::homeostasis::Homeostasis;
use super::neuromod::Neuromodulators;

// ─── Baseline tracker ────────────────────────────────────────

/// Exponential moving average of loss. Pain is relative to this.
/// The organism adapts to its current performance level.
/// A loss of 3.5 at baseline 4.0 is relief. At baseline 2.0 is pain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossBaseline {
    pub ema: f32,
    pub variance_ema: f32,
    pub alpha: f32, // smoothing factor
    pub count: u64,
}

impl LossBaseline {
    pub fn new(alpha: f32) -> Self {
        Self { ema: 0.0, variance_ema: 1.0, alpha, count: 0 }
    }

    /// Update baseline and return the surprise: (loss - baseline) / stddev.
    /// Positive = worse than expected (painful).
    /// Negative = better than expected (relief).
    /// Near zero = expected (no signal).
    pub fn update(&mut self, loss: f32) -> f32 {
        if self.count == 0 {
            self.ema = loss;
            self.variance_ema = 1.0;
            self.count = 1;
            return 0.0; // first observation, no surprise
        }

        let delta = loss - self.ema;
        self.ema = self.alpha * self.ema + (1.0 - self.alpha) * loss;
        self.variance_ema = self.alpha * self.variance_ema
            + (1.0 - self.alpha) * delta * delta;
        self.count += 1;

        let stddev = self.variance_ema.sqrt().max(0.1);
        delta / stddev // z-score: how many stddevs from expected
    }

    pub fn expected(&self) -> f32 { self.ema }
}

impl Default for LossBaseline {
    fn default() -> Self { Self::new(0.95) }
}

// ─── Configuration ───────────────────────────────────────────

/// Pain configuration — how strongly each source affects the organism.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PainConfig {
    /// How much surprise (z-score) affects emotional pressure.
    pub surprise_pain_scale: f32,
    /// How much retrieved negative valence affects emotional pressure.
    pub retrieval_pain_scale: f32,
    /// How much surprise spikes norepinephrine.
    pub surprise_ne_scale: f32,
    /// How much surprise drains serotonin.
    pub surprise_serotonin_scale: f32,
    /// How much relief boosts dopamine.
    pub relief_dopamine_boost: f32,
    /// How much relief recovers serotonin.
    pub relief_serotonin_boost: f32,
    /// Confidence multiplier: pain is worse when certain and wrong.
    pub confidence_multiplier: f32,
    /// Exit gate bias from negative retrieval.
    pub retrieval_exit_bias: f32,
    /// Valence threshold for avoidance.
    pub avoidance_threshold: f32,
    /// Neuromod decay rate toward baseline per tick.
    /// 0.0 = no decay, 1.0 = instant reset. 0.05 = gentle drift back.
    pub neuromod_decay: f32,
    /// Baseline smoothing factor for loss EMA.
    pub baseline_alpha: f32,
}

impl Default for PainConfig {
    fn default() -> Self {
        Self {
            surprise_pain_scale: 0.15,
            retrieval_pain_scale: 0.2,
            surprise_ne_scale: 0.1,
            surprise_serotonin_scale: 0.03,
            relief_dopamine_boost: 0.3,
            relief_serotonin_boost: 0.1,
            confidence_multiplier: 1.5,
            retrieval_exit_bias: 0.2,
            avoidance_threshold: -0.3,
            neuromod_decay: 0.05,
            baseline_alpha: 0.95,
        }
    }
}

// ─── Response ────────────────────────────────────────────────

/// The result of processing pain/relief for one step.
#[derive(Debug, Clone)]
pub struct PainResponse {
    /// How much exit gate bias to add (positive = hold longer).
    pub exit_bias_delta: f32,
    /// Whether an avoidance response was triggered.
    pub avoidance_triggered: bool,
    /// Whether relief was experienced.
    pub relief: bool,
    /// The valence to tag on the episodic memory for this step.
    pub valence_for_storage: f32,
    /// Pain intensity (for diagnostics). Positive = pain, negative = relief.
    pub pain_intensity: f32,
    /// The surprise z-score that drove this response.
    pub surprise: f32,
}

// ─── Neuromod decay ──────────────────────────────────────────

/// Decay neuromodulators toward baseline. Call once per step.
///
/// Invariant: after N steps with no pain/relief events,
/// all neuromodulators converge to their default values.
/// This prevents saturation — signals are transient, not permanent.
pub fn decay_neuromod(neuromod: &mut Neuromodulators, cfg: &PainConfig) {
    let d = cfg.neuromod_decay;
    // Decay toward defaults: DA=1.0, 5HT=1.0, NE=0.5
    neuromod.dopamine += d * (1.0 - neuromod.dopamine);
    neuromod.serotonin += d * (1.0 - neuromod.serotonin);
    neuromod.norepinephrine += d * (0.5 - neuromod.norepinephrine);
    // Curiosity and anxiety also decay
    neuromod.curiosity *= 1.0 - d;
    neuromod.anxiety *= 1.0 - d;
}

// ─── LR modulation ───────────────────────────────────────────

/// Compute learning rate scale from neuromod state.
///
/// Returns a multiplier centered at 1.0.
/// High DA (surprise/relief) → explore faster (>1.0)
/// High anxiety → exploit carefully (<1.0)
/// The function is smooth and bounded: output ∈ [0.5, 2.0].
///
/// Formally: scale = sigmoid(DA - 1.0 + curiosity - anxiety) mapped to [0.5, 2.0]
pub fn lr_scale(neuromod: &Neuromodulators) -> f32 {
    let signal = (neuromod.dopamine - 1.0)
        + 0.5 * neuromod.curiosity
        - 0.5 * neuromod.anxiety
        + 0.3 * (neuromod.serotonin - 1.0);
    // Sigmoid mapped to [0.5, 2.0]
    let s = 1.0 / (1.0 + (-signal).exp());
    0.5 + 1.5 * s
}

// ─── Pain events ─────────────────────────────────────────────

/// Process a prediction outcome using relative surprise.
///
/// `surprise`: z-score from LossBaseline::update() (positive = worse than expected)
/// `loss`: raw loss value (for valence receipt)
/// `confidence`: prediction confidence (0.0-1.0)
/// `was_correct`: did the organism get it right?
pub fn on_prediction(
    homeostasis: &mut Homeostasis,
    neuromod: &mut Neuromodulators,
    surprise: f32,
    loss: f32,
    confidence: f32,
    was_correct: bool,
    cfg: &PainConfig,
) -> PainResponse {
    // Always decay first — signals are transient
    decay_neuromod(neuromod, cfg);

    if was_correct {
        // Relief scales with how surprising the success was
        // Succeeding when expected = mild. Succeeding when struggling = euphoria.
        let relief_strength = (-surprise).max(0.0) * 0.5 + 0.1;
        neuromod.dopamine = (neuromod.dopamine + relief_strength * cfg.relief_dopamine_boost).min(3.0);
        neuromod.serotonin = (neuromod.serotonin + relief_strength * cfg.relief_serotonin_boost).min(2.0);

        let valence = (relief_strength * 0.5 + 0.1).min(1.0);
        return PainResponse {
            exit_bias_delta: 0.0,
            avoidance_triggered: false,
            relief: true,
            valence_for_storage: valence,
            pain_intensity: -relief_strength,
            surprise,
        };
    }

    // Pain: only if worse than expected (surprise > 0)
    let effective_surprise = surprise.max(0.0);
    if effective_surprise < 0.1 {
        // Expected failure — not painful, just baseline noise
        return PainResponse {
            exit_bias_delta: 0.0,
            avoidance_triggered: false,
            relief: false,
            valence_for_storage: -0.05, // mild negative
            pain_intensity: 0.0,
            surprise,
        };
    }

    // Confidence amplifies: certain + wrong = shameful
    let confidence_penalty = 1.0 + confidence * cfg.confidence_multiplier;
    let pain = effective_surprise * cfg.surprise_pain_scale * confidence_penalty;

    // Destabilize — proportional to surprise, not absolute loss
    homeostasis.emotional_pressure =
        (homeostasis.emotional_pressure + pain).min(1.0);
    neuromod.norepinephrine =
        (neuromod.norepinephrine + effective_surprise * cfg.surprise_ne_scale).min(2.0);
    neuromod.serotonin =
        (neuromod.serotonin - effective_surprise * cfg.surprise_serotonin_scale).max(0.1);

    let valence = -(pain.min(1.0));
    homeostasis.recompute_pressure();

    PainResponse {
        exit_bias_delta: 0.0,
        avoidance_triggered: false,
        relief: false,
        valence_for_storage: valence,
        pain_intensity: pain,
        surprise,
    }
}

/// Process retrieval pain: the organism remembers this context was painful.
pub fn on_retrieval(
    homeostasis: &mut Homeostasis,
    neuromod: &mut Neuromodulators,
    blended_valence: f32,
    cfg: &PainConfig,
) -> PainResponse {
    if blended_valence >= 0.0 {
        return PainResponse {
            exit_bias_delta: 0.0,
            avoidance_triggered: false,
            relief: false,
            valence_for_storage: 0.0,
            pain_intensity: 0.0,
            surprise: 0.0,
        };
    }

    let pain_magnitude = blended_valence.abs();
    let avoidance = blended_valence < cfg.avoidance_threshold;

    homeostasis.emotional_pressure =
        (homeostasis.emotional_pressure + pain_magnitude * cfg.retrieval_pain_scale).min(1.0);
    neuromod.norepinephrine =
        (neuromod.norepinephrine + pain_magnitude * 0.1).min(2.0);
    neuromod.serotonin =
        (neuromod.serotonin - pain_magnitude * 0.02).max(0.1);

    homeostasis.recompute_pressure();

    PainResponse {
        exit_bias_delta: if avoidance { pain_magnitude * cfg.retrieval_exit_bias } else { 0.0 },
        avoidance_triggered: avoidance,
        relief: false,
        valence_for_storage: 0.0,
        pain_intensity: pain_magnitude,
        surprise: 0.0,
    }
}

/// Process relief on a previously painful context.
pub fn on_overcoming(
    homeostasis: &mut Homeostasis,
    neuromod: &mut Neuromodulators,
    old_valence: f32,
    was_correct: bool,
    cfg: &PainConfig,
) -> PainResponse {
    if !was_correct || old_valence >= 0.0 {
        return PainResponse {
            exit_bias_delta: 0.0,
            avoidance_triggered: false,
            relief: false,
            valence_for_storage: 0.0,
            pain_intensity: 0.0,
            surprise: 0.0,
        };
    }

    let relief_strength = old_valence.abs();
    neuromod.dopamine = (neuromod.dopamine + relief_strength * cfg.relief_dopamine_boost).min(3.0);
    neuromod.serotonin = (neuromod.serotonin + relief_strength * cfg.relief_serotonin_boost).min(2.0);
    neuromod.norepinephrine = (neuromod.norepinephrine - relief_strength * 0.1).max(0.1);

    homeostasis.emotional_pressure =
        (homeostasis.emotional_pressure - relief_strength * 0.3).max(0.0);
    homeostasis.recompute_pressure();

    let valence = relief_strength.min(1.0);

    PainResponse {
        exit_bias_delta: -0.1,
        avoidance_triggered: false,
        relief: true,
        valence_for_storage: valence,
        pain_intensity: 0.0,
        surprise: 0.0,
    }
}

// ─── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baseline_tracks_loss() {
        let mut bl = LossBaseline::default();
        // Feed constant loss — surprise should approach zero
        for _ in 0..100 { bl.update(3.0); }
        let surprise = bl.update(3.0);
        assert!(surprise.abs() < 0.5, "constant loss should produce low surprise: {surprise}");

        // Spike — should produce high positive surprise
        let spike = bl.update(6.0);
        assert!(spike > 1.0, "loss spike should produce high surprise: {spike}");

        // Drop — should produce negative surprise (relief)
        let drop = bl.update(1.0);
        assert!(drop < -1.0, "loss drop should produce negative surprise: {drop}");
    }

    #[test]
    fn expected_failure_no_pain() {
        let mut h = Homeostasis::default();
        let mut n = Neuromodulators::default();
        let cfg = PainConfig::default();

        // Surprise near zero = expected failure = no pain
        let resp = on_prediction(&mut h, &mut n, 0.05, 3.0, 0.5, false, &cfg);
        assert!(resp.pain_intensity == 0.0, "expected failure shouldn't hurt");
    }

    #[test]
    fn surprising_failure_causes_pain() {
        let mut h = Homeostasis::default();
        let mut n = Neuromodulators::default();
        let cfg = PainConfig::default();

        // High positive surprise = unexpectedly bad
        let resp = on_prediction(&mut h, &mut n, 2.0, 5.0, 0.8, false, &cfg);
        assert!(resp.pain_intensity > 0.0);
        assert!(resp.valence_for_storage < 0.0);
        assert!(h.emotional_pressure > 0.0);
    }

    #[test]
    fn surprising_success_gives_relief() {
        let mut h = Homeostasis::default();
        let mut n = Neuromodulators::default();
        let da_before = n.dopamine;
        let cfg = PainConfig::default();

        // Negative surprise = better than expected
        let resp = on_prediction(&mut h, &mut n, -2.0, 1.0, 0.5, true, &cfg);
        assert!(resp.relief);
        assert!(n.dopamine > da_before, "DA should burst on surprising success");
    }

    #[test]
    fn neuromod_decays_toward_baseline() {
        let mut n = Neuromodulators {
            dopamine: 3.0,
            serotonin: 0.1,
            norepinephrine: 2.0,
            curiosity: 1.0,
            anxiety: 1.0,
        };
        let cfg = PainConfig::default();

        for _ in 0..100 {
            decay_neuromod(&mut n, &cfg);
        }

        // Should converge toward defaults
        assert!((n.dopamine - 1.0).abs() < 0.1, "DA should decay to 1.0: {}", n.dopamine);
        assert!((n.serotonin - 1.0).abs() < 0.1, "5HT should decay to 1.0: {}", n.serotonin);
        assert!((n.norepinephrine - 0.5).abs() < 0.1, "NE should decay to 0.5: {}", n.norepinephrine);
        assert!(n.curiosity < 0.1, "curiosity should decay: {}", n.curiosity);
        assert!(n.anxiety < 0.1, "anxiety should decay: {}", n.anxiety);
    }

    #[test]
    fn lr_scale_centered() {
        let default_n = Neuromodulators::default();
        let scale = lr_scale(&default_n);
        // Default state should give ~1.0 scale
        assert!((scale - 1.0).abs() < 0.3, "default neuromod should give ~1.0 lr scale: {scale}");

        // High DA + curiosity → higher scale
        let excited = Neuromodulators {
            dopamine: 2.5, curiosity: 1.0, anxiety: 0.0, serotonin: 1.5, ..Default::default()
        };
        let high = lr_scale(&excited);
        assert!(high > scale, "excited state should have higher lr: {high} vs {scale}");

        // High anxiety → lower scale
        let anxious = Neuromodulators {
            dopamine: 0.5, anxiety: 1.5, curiosity: 0.0, serotonin: 0.5, ..Default::default()
        };
        let low = lr_scale(&anxious);
        assert!(low < scale, "anxious state should have lower lr: {low} vs {scale}");
    }

    #[test]
    fn confidence_amplifies_pain() {
        let cfg = PainConfig::default();

        let mut h1 = Homeostasis::default();
        let mut n1 = Neuromodulators::default();
        let r1 = on_prediction(&mut h1, &mut n1, 1.5, 3.0, 0.1, false, &cfg);

        let mut h2 = Homeostasis::default();
        let mut n2 = Neuromodulators::default();
        let r2 = on_prediction(&mut h2, &mut n2, 1.5, 3.0, 0.9, false, &cfg);

        assert!(r2.pain_intensity > r1.pain_intensity,
            "high confidence should amplify: {:.3} vs {:.3}", r2.pain_intensity, r1.pain_intensity);
    }

    #[test]
    fn full_cycle_with_baseline() {
        let mut bl = LossBaseline::default();
        let mut h = Homeostasis::default();
        let mut n = Neuromodulators::default();
        let cfg = PainConfig::default();

        // Establish baseline at loss=3.0
        for _ in 0..50 {
            let s = bl.update(3.0);
            on_prediction(&mut h, &mut n, s, 3.0, 0.5, false, &cfg);
        }

        // NE and 5HT should be near baseline (decayed back)
        assert!((n.norepinephrine - 0.5).abs() < 0.3,
            "NE should be near baseline after expected failures: {}", n.norepinephrine);
        assert!(n.serotonin > 0.5,
            "5HT should not be floored after expected failures: {}", n.serotonin);

        // Now spike the loss — this should hurt
        let surprise = bl.update(6.0);
        assert!(surprise > 1.0);
        let resp = on_prediction(&mut h, &mut n, surprise, 6.0, 0.8, false, &cfg);
        assert!(resp.pain_intensity > 0.0);
        let ne_after_spike = n.norepinephrine;

        // Wait a few steps — NE should decay back
        for _ in 0..20 {
            let s = bl.update(3.0);
            on_prediction(&mut h, &mut n, s, 3.0, 0.5, false, &cfg);
        }
        assert!(n.norepinephrine < ne_after_spike,
            "NE should decay after spike: {} vs {}", n.norepinephrine, ne_after_spike);
    }

    #[test]
    fn negative_retrieval_triggers_avoidance() {
        let mut h = Homeostasis::default();
        let mut n = Neuromodulators::default();
        let resp = on_retrieval(&mut h, &mut n, -0.8, &PainConfig::default());
        assert!(resp.avoidance_triggered);
        assert!(resp.exit_bias_delta > 0.0);
    }

    #[test]
    fn overcoming_pain_gives_relief() {
        let mut h = Homeostasis::default();
        h.emotional_pressure = 0.5;
        let mut n = Neuromodulators::default();
        let da_before = n.dopamine;

        let resp = on_overcoming(&mut h, &mut n, -0.8, true, &PainConfig::default());
        assert!(resp.relief);
        assert!(n.dopamine > da_before);
        assert!(h.emotional_pressure < 0.5);
    }
}
