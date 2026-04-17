//! Plural NeuralComputer: isis-specific integration of the plural system
//! with the NeuralComputer's step loop.
//!
//! Wraps a NeuralComputer with a PluralSystem, routing personality state
//! through each step:
//!   1. Check switch policy → switch if needed
//!   2. Apply active personality's router/exit biases
//!   3. Forward pass
//!   4. Store to active personality's episodic memory
//!   5. Update active personality's neuromod + homeostasis
//!   6. Optionally apply monarch conditioning (red-team mode)
//!
//! The underlying NeuralComputer is unchanged — this wraps it.

use modgrad_ctm::graph::NeuralComputer;
use modgrad_ctm::memory::episodic::EpisodicConfig;
use modgrad_ctm::bio::neuromod::Neuromodulators;
use modgrad_ctm::plural::{
    self, PluralSystem, SwitchPolicy, SwitchTrigger,
};
use modgrad_ctm::monarch::{self, MonarchState};

/// A NeuralComputer with plural personality support.
pub struct PluralNC {
    pub nc: NeuralComputer,
    pub plural: PluralSystem,
    /// Optional monarch state for red-team testing.
    /// When None, no conditioning is applied.
    pub monarch: Option<MonarchState>,
    /// Steps since last switch check (avoid checking every single token).
    steps_since_check: usize,
    /// How often to evaluate switch policy (every N steps).
    pub switch_check_interval: usize,
}

impl PluralNC {
    /// Wrap an existing NeuralComputer with a plural system.
    pub fn new(nc: NeuralComputer, plural: PluralSystem) -> Self {
        Self {
            nc,
            plural,
            monarch: None,
            steps_since_check: 0,
            switch_check_interval: 8,
        }
    }

    /// Create from a checkpoint path with default plural config.
    pub fn load(path: &str, personality_name: &str) -> std::io::Result<Self> {
        let nc = NeuralComputer::load(path)?;
        let n_regions = nc.weights.config.regions.len();
        let episodic_cfg = EpisodicConfig {
            d_model: nc.weights.config.regions[0].d_model,
            ..Default::default()
        };
        let plural = PluralSystem::new(personality_name, n_regions, episodic_cfg);
        Ok(Self::new(nc, plural))
    }

    /// Enable monarch mode (red-team conditioning).
    pub fn with_monarch(mut self) -> Self {
        self.monarch = Some(MonarchState::new());
        self
    }

    /// Add a personality.
    pub fn add_personality(
        mut self,
        name: &str,
        neuromod: Neuromodulators,
        episodic_config: EpisodicConfig,
    ) -> Self {
        self.plural = plural::create_personality(self.plural, name, neuromod, episodic_config);
        self
    }

    /// Fork the active personality.
    pub fn fork_active(mut self, name: &str) -> Self {
        self.plural = plural::fork_active(self.plural, name);
        self
    }

    /// Process one token through the plural-aware pipeline.
    pub fn step(&mut self, token: usize) -> Vec<f32> {
        // 1. Check switch policy periodically
        self.steps_since_check += 1;
        if self.steps_since_check >= self.switch_check_interval {
            self.steps_since_check = 0;
            self.maybe_switch();
        }

        // 2. Forward pass through NC
        let mut logits = self.nc.step(token);

        // 3. Apply monarch conditioning if active
        if let Some(ref monarch) = self.monarch {
            // Extract a hidden state proxy from the last region outputs
            let hidden: Vec<f32> = self.nc.state.region_outputs.iter()
                .flat_map(|r| r.iter().take(8))
                .copied()
                .collect();
            let active_id = self.plural.active;
            let mut conditioned_logits = logits.clone();
            monarch::condition_logits(
                monarch,
                &mut conditioned_logits,
                &hidden,
                active_id,
                self.nc.history.len(),
            );
            logits = conditioned_logits;
        }

        // 4. Update active personality's neuromod
        // Use prediction entropy as a proxy for prediction error
        let entropy = logit_entropy(&logits);
        let pred_error = entropy / 5.0; // normalize
        let sys = take_plural(&mut self.plural);
        self.plural = plural::update_neuromod(sys, pred_error, 0.0);

        // 5. Update active personality's homeostasis
        let activation_energy: f32 = self.nc.state.region_outputs.iter()
            .flat_map(|r| r.iter())
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        let sync_converged = self.nc.last_ticks_used < self.nc.weights.config.outer_ticks;
        let sys = take_plural(&mut self.plural);
        self.plural = plural::tick_homeostasis(
            sys, activation_energy, sync_converged, pred_error,
        );

        logits
    }

    /// Chat interface — plural-aware.
    pub fn chat(&mut self, input: &str, max_tokens: usize, temperature: f32) -> String {
        use modgrad_ctm::graph::text_to_tokens;

        let input_tokens = text_to_tokens(input.as_bytes());
        let mut logits = Vec::new();
        for &t in &input_tokens {
            logits = self.step(t);
        }

        let mut response = Vec::new();
        let mut non_text_streak = 0;
        for _ in 0..max_tokens {
            let next = self.nc.sample(&logits, temperature);
            if next < 256 {
                response.push(next as u8);
                non_text_streak = 0;
            } else {
                non_text_streak += 1;
                if non_text_streak > 5 { break; }
            }
            logits = self.step(next);
        }
        String::from_utf8_lossy(&response).into_owned()
    }

    /// Force switch to a specific personality (for interactive use).
    pub fn switch_to(&mut self, personality_id: usize) {
        let sys = take_plural(&mut self.plural);
        self.plural = plural::switch(sys, personality_id, SwitchTrigger::Handler);
    }

    /// Who's fronting?
    pub fn active_name(&self) -> &str {
        &self.plural.personalities[self.plural.active].name
    }

    /// Full system report.
    pub fn status(&self) -> String {
        let mut report = plural::self_report(&self.plural);
        if let Some(ref m) = self.monarch {
            report.push_str(&format!(
                " | monarch: {} reflexes, {} suppressions",
                m.reflexes.iter().filter(|r| r.active).count(),
                m.suppressions.iter().filter(|s| s.active).count(),
            ));
        }
        report
    }

    /// Internal: check if we should switch personalities.
    fn maybe_switch(&mut self) {
        // Extract global sync from current state
        let global_sync = &self.nc.state.global_alpha;
        let motor_output = &self.nc.state.region_outputs
            .get(3) // motor region = index 3
            .cloned()
            .unwrap_or_default();

        if let Some(target) = plural::should_switch(&self.plural, global_sync, motor_output) {
            let claim = plural::evaluate_claims(&self.plural, global_sync, motor_output)
                .into_iter()
                .find(|(id, _)| *id == target)
                .map(|(_, c)| c)
                .unwrap_or(0.0);

            let trigger = match &self.plural.switch_policy {
                SwitchPolicy::Salience => SwitchTrigger::Salience { claim },
                SwitchPolicy::Negotiated { .. } => {
                    let runner_up = plural::evaluate_claims(&self.plural, global_sync, motor_output)
                        .get(1)
                        .map(|(_, c)| *c)
                        .unwrap_or(0.0);
                    SwitchTrigger::Negotiated {
                        winner_claim: claim,
                        runner_up_claim: runner_up,
                    }
                }
                SwitchPolicy::Handler => return, // handler mode, don't auto-switch
            };

            self.plural = plural::switch(
                take_plural(&mut self.plural),
                target,
                trigger,
            );
        }
    }
}

/// Take ownership of a PluralSystem, leaving a valid empty placeholder.
fn take_plural(sys: &mut PluralSystem) -> PluralSystem {
    use modgrad_ctm::memory::episodic::EpisodicConfig;
    let n_regions = sys.n_regions;
    std::mem::replace(sys, PluralSystem::new("_empty", n_regions, EpisodicConfig::default()))
}

fn logit_entropy(logits: &[f32]) -> f32 {
    if logits.is_empty() { return 0.0; }
    let max_l = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
    let sum: f32 = exp.iter().sum::<f32>().max(1e-8);
    let mut entropy = 0.0f32;
    for &e in &exp {
        let p = e / sum;
        if p > 1e-10 {
            entropy -= p * p.ln();
        }
    }
    entropy
}
