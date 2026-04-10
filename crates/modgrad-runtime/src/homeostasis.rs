//! Homeostasis: self-monitoring and sleep pressure system.
//!
//! The organism tracks its own wellbeing through measurable signals,
//! not abstract feelings. Like checking your own vital signs.
//!
//! Sleep pressure accumulates during waking and can only be
//! cleared by sleep. The organism can observe its pressure level
//! and decide when to sleep — but past a threshold, sleep is forced.
//!
//! Three zones:
//!   GREEN  (0.0-0.5): healthy, can work indefinitely
//!   YELLOW (0.5-0.8): degrading, should sleep soon
//!   RED    (0.8-1.0): critical, forced sleep imminent
//!   FORCED (>1.0):    cannot process, must sleep

use super::tuning::HomeostasisParams;
use serde::{Deserialize, Serialize};

/// Sleep pressure state. Accumulates from neural activity, cleared by sleep.
///
/// Not tied to tokens — tied to how hard the brain is working.
/// A boring repetitive input barely raises pressure.
/// A complex surprising input exhausts the system fast.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Homeostasis {
    /// Sleep pressure: 0.0 = fully rested, 1.0 = must sleep.
    pub pressure: f32,

    /// Components of pressure (each 0.0 - 1.0, weighted sum = pressure)

    /// Activation energy EMA: how hard the neurons are firing.
    /// High = complex processing = more metabolic waste.
    pub activation_pressure: f32,

    /// Sync divergence: when CTM ticks don't converge, the brain is struggling.
    /// Like trying to make a decision but going in circles.
    pub divergence_pressure: f32,

    /// Hebbian drift: how far running means have moved from calibrated baseline.
    /// High drift = outputs becoming noisy = consolidation overdue.
    pub drift_pressure: f32,

    /// Buffer pressure: sensory traces and replay buffer filling up.
    /// Full buffers = consolidation MUST happen.
    pub buffer_pressure: f32,

    /// Emotional pressure: unprocessed fear/negative memories accumulating.
    /// The subconscious needs REM time.
    pub emotional_pressure: f32,

    /// Surprise EMA: sustained high surprise = sustained stress.
    pub surprise_ema: f32,

    /// Output quality: degrades as pressure increases.
    pub output_quality: f32,

    /// Steps since last sleep (for reporting, not for pressure).
    pub steps_since_sleep: u64,
}

impl Default for Homeostasis {
    fn default() -> Self {
        Self {
            pressure: 0.0,
            activation_pressure: 0.0,
            divergence_pressure: 0.0,
            drift_pressure: 0.0,
            buffer_pressure: 0.0,
            emotional_pressure: 0.0,
            surprise_ema: 0.0,
            output_quality: 1.0,
            steps_since_sleep: 0,
        }
    }
}

/// What the organism observes about itself.
#[derive(Debug, Clone)]
pub enum SleepZone {
    /// Fully rested. Can work indefinitely.
    Green,
    /// Getting tired. Should consider sleeping.
    Yellow,
    /// Critically tired. Must sleep very soon.
    Red,
    /// Cannot function. Sleep is forced.
    Forced,
}

impl Homeostasis {
    /// Update from a CTM forward pass result.
    /// This is the primary pressure source — how hard the brain worked.
    ///
    /// `activation_energy`: L2 norm of all region activations (how hard neurons fired)
    /// `sync_converged`: did the sync accumulators converge? (false = struggling)
    /// `surprise`: prediction error for this step
    pub fn tick_from_ctm(
        &mut self,
        activation_energy: f32,
        sync_converged: bool,
        surprise: f32,
    ) {
        self.steps_since_sleep += 1;

        self.tick_from_ctm_with(activation_energy, sync_converged, surprise, &HomeostasisParams::default());
    }

    /// Update from CTM forward pass with tuning params.
    pub fn tick_from_ctm_with(
        &mut self,
        activation_energy: f32,
        sync_converged: bool,
        surprise: f32,
        p: &HomeostasisParams,
    ) {
        self.steps_since_sleep += 1;

        let act_signal = (activation_energy / 5.0).clamp(0.0, 1.0);
        let ema = p.activation_ema.val();
        self.activation_pressure = ema * self.activation_pressure + (1.0 - ema) * act_signal;

        if !sync_converged {
            self.divergence_pressure = (self.divergence_pressure + p.divergence_increment.val()).min(1.0);
        } else {
            self.divergence_pressure = (self.divergence_pressure - p.divergence_decrement.val()).max(0.0);
        }

        self.surprise_ema = 0.9 * self.surprise_ema + 0.1 * surprise;

        self.recompute_pressure_with(p);
    }

    /// Update from organism state (buffers, drift, emotional).
    /// Called periodically, not every step.
    pub fn observe_state(
        &mut self,
        sensory_buffer_fill: f32,   // 0.0 - 1.0
        replay_buffer_fill: f32,    // 0.0 - 1.0
        hebbian_drift: f32,         // L2 of drift from baseline
        unprocessed_fears: u32,
    ) {
        // Buffer pressure: full buffers = consolidation overdue
        self.buffer_pressure = ((sensory_buffer_fill + replay_buffer_fill) / 2.0).clamp(0.0, 1.0);

        // Drift pressure: how noisy have the outputs become?
        // Typical drift ~0.1, concerning ~0.5, critical ~1.0
        self.drift_pressure = (hebbian_drift / 1.0).clamp(0.0, 1.0);

        // Emotional pressure: unprocessed fears need REM time
        self.emotional_pressure = (unprocessed_fears as f32 / 5.0).clamp(0.0, 1.0);

        self.recompute_pressure();
    }

    /// Recompute total pressure from all components.
    pub fn recompute_pressure(&mut self) {
        self.recompute_pressure_with(&HomeostasisParams::default());
    }

    /// Recompute total pressure using tuning params.
    pub fn recompute_pressure_with(&mut self, p: &HomeostasisParams) {
        self.pressure = (
            p.w_activation.val() * self.activation_pressure
            + p.w_divergence.val() * self.divergence_pressure
            + p.w_drift.val() * self.drift_pressure
            + p.w_buffer.val() * self.buffer_pressure
            + p.w_emotional.val() * self.emotional_pressure
            + p.w_surprise.val() * (self.surprise_ema / 5.0).clamp(0.0, 1.0)
        ).clamp(0.0, 1.5);

        self.output_quality = (1.0 - self.pressure * p.quality_degradation.val()).max(0.1);
    }

    /// What zone are we in? Based purely on pressure, not arbitrary token counts.
    pub fn zone(&self) -> SleepZone {
        self.zone_with(&HomeostasisParams::default())
    }

    /// Zone check with tuning params.
    pub fn zone_with(&self, p: &HomeostasisParams) -> SleepZone {
        if self.pressure >= p.forced_threshold.val() {
            SleepZone::Forced
        } else if self.pressure >= p.red_threshold.val() {
            SleepZone::Red
        } else if self.pressure >= p.yellow_threshold.val() {
            SleepZone::Yellow
        } else {
            SleepZone::Green
        }
    }

    /// Should the organism sleep? True if Red or Forced.
    /// Yellow = organism can push through (its choice).
    pub fn should_sleep(&self) -> bool {
        matches!(self.zone(), SleepZone::Red | SleepZone::Forced)
    }

    /// Must the organism sleep? True only if Forced.
    /// Cannot continue — quality too degraded.
    pub fn must_sleep(&self) -> bool {
        matches!(self.zone(), SleepZone::Forced)
    }

    /// Reset after sleeping. Each component cleared by its respective sleep phase.
    pub fn on_sleep(&mut self, sleep_quality: f32) {
        let q = sleep_quality.clamp(0.3, 1.0);

        // NREM clears: activation, drift, buffers
        self.activation_pressure *= 1.0 - q * 0.8;
        self.drift_pressure *= 1.0 - q * 0.9;
        self.buffer_pressure *= 1.0 - q * 0.95;

        // REM clears: divergence, emotional, surprise
        self.divergence_pressure *= 1.0 - q * 0.7;
        self.emotional_pressure *= 1.0 - q * 0.8;
        self.surprise_ema *= 1.0 - q * 0.5;

        self.steps_since_sleep = 0;
        self.recompute_pressure();
    }

    /// Self-report: what the organism observes about its own state.
    /// This is introspection — the organism can read this and decide.
    pub fn self_report(&self) -> String {
        let zone = match self.zone() {
            SleepZone::Green => "rested",
            SleepZone::Yellow => "getting tired",
            SleepZone::Red => "need sleep",
            SleepZone::Forced => "MUST SLEEP NOW",
        };
        format!(
            "pressure={:.2} [{zone}] quality={:.2} | act={:.2} div={:.2} drift={:.2} buf={:.2} emo={:.2} surprise={:.2}",
            self.pressure, self.output_quality,
            self.activation_pressure, self.divergence_pressure,
            self.drift_pressure, self.buffer_pressure,
            self.emotional_pressure, self.surprise_ema
        )
    }

    /// Which component is the biggest contributor to pressure?
    /// The organism can use this for self-diagnosis.
    pub fn biggest_pressure_source(&self) -> &'static str {
        let components = [
            (self.activation_pressure * 0.25, "neural activation (thinking too hard)"),
            (self.divergence_pressure * 0.15, "sync divergence (can't reach decisions)"),
            (self.drift_pressure * 0.20, "hebbian drift (outputs getting noisy)"),
            (self.buffer_pressure * 0.20, "buffer overflow (consolidation overdue)"),
            (self.emotional_pressure * 0.10, "unprocessed emotions (need REM)"),
            (self.surprise_ema / 5.0 * 0.10, "sustained surprise (stress)"),
        ];
        components.iter()
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .map(|(_, name)| *name)
            .unwrap_or("unknown")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pressure_from_activation() {
        let mut h = Homeostasis::default();
        assert!(matches!(h.zone(), SleepZone::Green));

        // Intense processing: high activation, no convergence, high surprise
        for _ in 0..200 {
            h.tick_from_ctm(5.0, false, 8.0);
        }
        // Should be in at least yellow
        assert!(h.pressure > 0.3, "pressure={:.2}", h.pressure);
    }

    #[test]
    fn test_easy_input_low_pressure() {
        let mut h = Homeostasis::default();

        // Boring input: low activation, converges, no surprise
        for _ in 0..200 {
            h.tick_from_ctm(0.1, true, 0.1);
        }
        // Should still be green — easy work doesn't tire you
        assert!(matches!(h.zone(), SleepZone::Green),
            "easy input should stay green, pressure={:.2}", h.pressure);
    }

    #[test]
    fn test_sleep_clears_pressure() {
        let mut h = Homeostasis::default();
        h.activation_pressure = 1.0;
        h.buffer_pressure = 1.0;
        h.drift_pressure = 1.0;
        h.divergence_pressure = 1.0;
        h.recompute_pressure();
        assert!(h.should_sleep(), "pressure={:.2}", h.pressure);

        h.on_sleep(1.0); // full sleep
        assert!(h.pressure < 0.2, "post-sleep pressure={:.2}", h.pressure);
        assert!(!h.should_sleep());
    }

    #[test]
    fn test_forced_sleep() {
        let mut h = Homeostasis::default();
        // Max out all components
        h.activation_pressure = 1.0;
        h.divergence_pressure = 1.0;
        h.drift_pressure = 1.0;
        h.buffer_pressure = 1.0;
        h.emotional_pressure = 1.0;
        h.surprise_ema = 10.0;
        h.recompute_pressure();
        assert!(h.must_sleep(), "pressure={:.2}", h.pressure);
    }

    #[test]
    fn test_fear_increases_pressure() {
        let mut h = Homeostasis::default();
        h.observe_state(0.0, 0.0, 0.0, 10); // 10 unprocessed fears
        assert!(h.emotional_pressure > 0.0);
        assert!(h.pressure > 0.0);
    }

    #[test]
    fn test_biggest_source() {
        let mut h = Homeostasis::default();
        h.emotional_pressure = 0.9; // dominant
        h.recompute_pressure();
        let source = h.biggest_pressure_source();
        assert!(source.contains("emotion"), "expected emotional, got: {source}");
    }
}
