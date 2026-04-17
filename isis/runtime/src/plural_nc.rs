//! Organism-aware NeuralComputer wrapper for isis runtime.
//!
//! Wraps a NeuralComputer with an Organism (pain, memory, dream,
//! plural, homeostasis) — the same SDK composition used in training.
//!
//! For inference, the organism runs per-token:
//!   1. begin_step() → plural switching check
//!   2. before_sample(hidden) → episodic retrieval + valence
//!   3. NC forward pass
//!   4. after_position(0, ...) → pain signal from prediction entropy
//!   5. after_sample(entropy, ...) → store to episodic memory
//!   6. after_batch(entropy) → LR scale (unused at inference), sleep/dream
//!   7. Optionally apply monarch conditioning (red-team mode)
//!
//! The underlying NeuralComputer is unchanged — this wraps it.

use modgrad_ctm::graph::NeuralComputer;
use modgrad_ctm::memory::episodic::EpisodicConfig;
use modgrad_ctm::bio::pain::PainConfig;
use modgrad_ctm::organism::{Organism, OrganismConfig};
use modgrad_ctm::monarch::{self, MonarchState};

/// A NeuralComputer with full organism support (pain, memory, dream, plural).
pub struct OrganismNC {
    pub nc: NeuralComputer,
    pub organism: Organism,
    /// Optional monarch state for red-team testing.
    pub monarch: Option<MonarchState>,
    /// Steps since last organism tick (batch inference tokens before ticking).
    steps_since_tick: usize,
    /// How many tokens to batch before an organism tick.
    pub tick_interval: usize,
    /// Running entropy accumulator for organism feedback.
    entropy_acc: f32,
    entropy_count: usize,
}

impl OrganismNC {
    /// Wrap an existing NeuralComputer with an organism.
    pub fn new(nc: NeuralComputer, organism: Organism) -> Self {
        Self {
            nc,
            organism,
            monarch: None,
            steps_since_tick: 0,
            tick_interval: 8,
            entropy_acc: 0.0,
            entropy_count: 0,
        }
    }

    /// Create from a checkpoint path with default organism config.
    pub fn load(path: &str, plural: bool) -> std::io::Result<Self> {
        let nc = NeuralComputer::load(path)?;
        let n_regions = nc.weights.config.regions.len();
        let d_model = nc.weights.config.regions[0].d_model;
        let max_ticks = nc.weights.config.outer_ticks;
        let organism = Organism::new(OrganismConfig {
            n_regions,
            d_model,
            max_ticks,
            pain: PainConfig::default(),
            episodic: EpisodicConfig {
                d_model,
                max_ticks,
                ..Default::default()
            },
            warmup_steps: 0, // no warmup at inference
            n_positions: 1,
            plural,
            max_personalities: 4,
            red_threshold_for_split: 50,
        });
        Ok(Self::new(nc, organism))
    }

    /// Enable monarch mode (red-team conditioning).
    pub fn with_monarch(mut self) -> Self {
        self.monarch = Some(MonarchState::new());
        self
    }

    /// Process one token through the organism-aware pipeline.
    pub fn step(&mut self, token: usize) -> Vec<f32> {
        self.steps_since_tick += 1;

        // Organism tick at interval
        if self.steps_since_tick >= self.tick_interval {
            self.organism_tick();
        }

        // Forward pass through NC
        let mut logits = self.nc.step(token);

        // Track prediction entropy for organism feedback
        let entropy = logit_entropy(&logits);
        self.entropy_acc += entropy;
        self.entropy_count += 1;

        // Apply monarch conditioning if active
        if let Some(ref monarch) = self.monarch {
            let hidden: Vec<f32> = self.nc.state.region_outputs.iter()
                .flat_map(|r| r.iter().take(8))
                .copied()
                .collect();
            let active_id = self.organism.plural_system()
                .map_or(0, |sys| sys.active);
            monarch::condition_logits(
                monarch,
                &mut logits,
                &hidden,
                active_id,
                self.nc.history.len(),
            );
        }

        logits
    }

    /// Chat interface — organism-aware.
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
        if let Some(sys) = self.organism.plural_system_mut() {
            use modgrad_ctm::plural::{self as p, SwitchTrigger};
            let n_regions = sys.n_regions;
            let temp = std::mem::replace(
                sys,
                modgrad_ctm::plural::PluralSystem::new("_", n_regions, EpisodicConfig::default()),
            );
            *sys = p::switch(temp, personality_id, SwitchTrigger::Handler);
        }
    }

    /// Who's fronting?
    pub fn active_name(&self) -> &str {
        self.organism.plural_system()
            .map_or("single", |sys| &sys.personalities[sys.active].name)
    }

    /// Full system report.
    pub fn status(&self) -> String {
        let mut report = self.organism.report();
        if let Some(ref m) = self.monarch {
            report.push_str(&format!(
                " | monarch: {} reflexes, {} suppressions",
                m.reflexes.iter().filter(|r| r.active).count(),
                m.suppressions.iter().filter(|s| s.active).count(),
            ));
        }
        report
    }

    /// Internal: flush accumulated entropy into organism lifecycle.
    fn organism_tick(&mut self) {
        let avg_entropy = if self.entropy_count > 0 {
            self.entropy_acc / self.entropy_count as f32
        } else {
            0.0
        };
        let pred_error = avg_entropy / 5.0;

        self.organism.begin_step();

        // Use region outputs as query for episodic retrieval
        let query: Vec<f32> = self.nc.state.region_outputs
            .first()
            .cloned()
            .unwrap_or_default();
        let before = self.organism.before_sample(&query);
        let _ = before; // retrieval priming would need state injection — future work

        // Feed back prediction quality
        let correct = pred_error < 0.5;
        self.organism.after_position(0, correct, 1.0 - pred_error, 0.0);
        self.organism.after_sample(pred_error, if correct { 1 } else { 0 }, 1, &query);
        let _batch = self.organism.after_batch(pred_error, None);

        self.steps_since_tick = 0;
        self.entropy_acc = 0.0;
        self.entropy_count = 0;
    }
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
