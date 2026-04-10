//! Neuromodulation dynamics: dopamine, serotonin, norepinephrine.
//!
//! Models the chemical signals that gate learning and attention:
//!   - Dopamine: prediction error → surprise signal
//!   - Serotonin: mood/energy → consolidation priority
//!   - Norepinephrine: arousal → attention/urgency
//!   - Curiosity: pred_error × DA × calm (active inference)
//!   - Anxiety: pred_error × DA × stress

use serde::{Deserialize, Serialize};

/// Neuromodulator state — updated every tick.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Neuromodulators {
    pub dopamine: f32,        // prediction error signal [0.1, 3.0]
    pub serotonin: f32,       // mood/energy [0.1, 2.0]
    pub norepinephrine: f32,  // arousal [0.1, 2.0]
    pub curiosity: f32,       // intrinsic motivation [0, ∞)
    pub anxiety: f32,         // urgency signal [0, ∞)
}

impl Default for Neuromodulators {
    fn default() -> Self {
        Self {
            dopamine: 1.0,
            serotonin: 1.0,
            norepinephrine: 0.5,
            curiosity: 0.0,
            anxiety: 0.0,
        }
    }
}

impl Neuromodulators {
    /// Update all neuromodulators for one tick.
    ///
    /// `pred_error`: prediction error magnitude from cerebellar module
    /// `learning_progress`: how much the model improved (Schmidhuber's "fun")
    pub fn update(&mut self, pred_error: f32, learning_progress: f32) {
        let da = self.dopamine;
        let ne = self.norepinephrine;
        let calm = (1.0 - ne / 2.0).clamp(0.0, 1.0);
        let stress = (ne / 2.0).min(1.0);

        // Active inference (Friston's free energy)
        let curiosity = pred_error * da * calm;
        let anxiety = pred_error * da * stress;

        // EMA updates
        self.curiosity = 0.7 * self.curiosity + 0.3 * (curiosity + 0.5 * learning_progress);
        self.anxiety = 0.7 * self.anxiety + 0.3 * anxiety;

        // Arousal: curiosity → engagement, anxiety → fight-or-flight
        self.norepinephrine = (ne + 0.05 * curiosity + 0.15 * anxiety).clamp(0.1, 2.0);

        // Serotonin: learning feels good, anxiety feels bad
        self.serotonin = (self.serotonin + 0.02 * learning_progress - 0.01 * anxiety).clamp(0.1, 2.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn high_error_increases_curiosity() {
        let mut nm = Neuromodulators::default();
        nm.update(1.0, 0.0);
        assert!(nm.curiosity > 0.0);
    }

    #[test]
    fn learning_increases_serotonin() {
        let mut nm = Neuromodulators::default();
        let before = nm.serotonin;
        nm.update(0.0, 1.0);
        assert!(nm.serotonin > before);
    }

    #[test]
    fn stays_in_bounds() {
        let mut nm = Neuromodulators::default();
        for _ in 0..1000 { nm.update(10.0, 10.0); }
        assert!(nm.norepinephrine <= 2.0);
        assert!(nm.serotonin >= 0.1);
        assert!(nm.serotonin <= 2.0);
    }
}
