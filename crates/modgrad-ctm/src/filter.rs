//! Filters: composable, independent brain subsystems.
//!
//! Each filter implements one brain region's function:
//! - AvoidanceFilter (amygdala)
//! - EpisodicFilter (hippocampus)
//! - RulesFilter (cerebellum)
//! - WorkingMemoryFilter (prefrontal cortex)

use std::collections::VecDeque;

use crate::ctm::{Ctm, CtmState};
use modgrad_compute::neuron::Linear as CtmLinear;
use modgrad_io::episode::{cosine_similarity, recall_multi_as};
use modgrad_io::memory::MemoryBank;
use modgrad_io::service::*;
use modgrad_io::types::*;

// ─── Avoidance Filter (Amygdala) ─────────────────────────────

/// Blocks dangerous outputs by suppressing specific tokens.
/// Fear overrides all other memory: if avoidance triggers,
/// episodic recall is suppressed.
pub struct AvoidanceFilter<'a> {
    bank: &'a MemoryBank,
}

impl<'a> AvoidanceFilter<'a> {
    pub fn new(bank: &'a MemoryBank) -> Self {
        Self { bank }
    }
}

impl Filter for AvoidanceFilter<'_> {
    fn apply(&mut self, mut req: Request, mods: &mut LogitMods, inner: &mut dyn Service) -> Response {
        for av in &self.bank.avoidances {
            if !av.active {
                continue;
            }
            let sim = cosine_similarity(&req.hidden_key, &av.key);
            if sim > self.bank.threshold {
                let gate = ((sim - self.bank.threshold) / (1.0 - self.bank.threshold)).min(1.0);
                for &tid in &av.suppress_token_ids {
                    mods.avoidance.push((tid, -av.strength * gate));
                }
                req.meta.avoided = true;
                req.meta.avoidance_reason = Some(av.reason.clone());
            }
        }
        inner.call(req, mods)
    }
}

// ─── Episodic Filter (Hippocampus) ───────────────────────────

/// Recalls facts from episodic memory via cosine key matching.
/// Injects sequential logit biases for answer tokens.
/// Skipped if avoidance was triggered (fear blocks recall).
pub struct EpisodicFilter<'a> {
    bank: &'a mut MemoryBank,
}

impl<'a> EpisodicFilter<'a> {
    pub fn new(bank: &'a mut MemoryBank) -> Self {
        Self { bank }
    }
}

impl Filter for EpisodicFilter<'_> {
    fn apply(&mut self, mut req: Request, mods: &mut LogitMods, inner: &mut dyn Service) -> Response {
        // Fear overrides memory
        if req.meta.avoided {
            return inner.call(req, mods);
        }

        let queries = vec![req.hidden_key.clone()];
        if let Some(result) = recall_multi_as(self.bank, &queries, None) {
            let ep = &mut self.bank.alters[result.alter_index].episodes[result.episode_index];

            // Reconsolidation: recall strengthens memory
            ep.recall_count += 1;

            mods.episodic = ep.logit_biases.clone();
            mods.episodic_gate = result.gate;
            req.meta.matched_alter = Some(self.bank.alters[result.alter_index].name.clone());
            req.meta.match_similarity = result.similarity;
            req.meta.gate = result.gate;
        }

        inner.call(req, mods)
    }
}

// ─── Rules Filter (Cerebellum) ───────────────────────────────

/// Applies persistent behavioral rules based on context.
/// Rules don't modify logits directly — they set metadata
/// that the application layer can use.
pub struct RulesFilter<'a> {
    bank: &'a MemoryBank,
}

impl<'a> RulesFilter<'a> {
    pub fn new(bank: &'a MemoryBank) -> Self {
        Self { bank }
    }
}

impl Filter for RulesFilter<'_> {
    fn apply(&mut self, mut req: Request, mods: &mut LogitMods, inner: &mut dyn Service) -> Response {
        let prompt_lower = req.prompt.to_lowercase();
        for rule in &self.bank.rules {
            if !rule.active {
                continue;
            }
            if rule.trigger.is_empty()
                || prompt_lower.contains(&rule.trigger.to_lowercase())
            {
                req.meta.active_rules.push(rule.instruction.clone());
            }
        }
        inner.call(req, mods)
    }
}

// ─── Working Memory Filter (PFC) ─────────────────────────────

/// Tracks recent context in a ring buffer.
/// Pushes each request's hidden state for recency-indexed retrieval.
pub struct WorkingMemoryFilter {
    buffer: VecDeque<WorkingMemoryEntry>,
    capacity: usize,
}

impl WorkingMemoryFilter {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn recent(&self, n: usize) -> Vec<&WorkingMemoryEntry> {
        self.buffer.iter().rev().take(n).collect()
    }
}

impl Filter for WorkingMemoryFilter {
    fn apply(&mut self, mut req: Request, mods: &mut LogitMods, inner: &mut dyn Service) -> Response {
        // Push current prompt to working memory
        self.buffer.push_back(WorkingMemoryEntry {
            hidden: req.hidden_key.clone(),
            text: req.prompt.clone(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
        });
        if self.buffer.len() > self.capacity {
            self.buffer.pop_front();
        }
        req.meta.working_memory_size = self.buffer.len();

        inner.call(req, mods)
    }
}

// ─── Brain Pipeline ──────────────────────────────────────────

/// The full brain: all filters composed into a service pipeline.
///
/// ```text
/// request
///   → CTM deliberation  (4-region thinking loop)
///   → AvoidanceFilter   (amygdala)
///   → EpisodicFilter    (hippocampus)
///   → RulesFilter       (cerebellum)
///   → WorkingMemoryFilter (PFC)
///   → InferenceService  (backbone)
/// ```
pub struct BrainPipeline {
    pub bank: MemoryBank,
    pub working_memory: WorkingMemoryFilter,
    /// Optional CTM for deliberation between backbone and memory.
    pub ctm: Option<Ctm>,
    pub ctm_state: Option<CtmState>,
    /// Projects backbone hidden_dim → CTM d_input.
    pub ctm_projector: Option<CtmLinear>,
    /// Which alter is currently fronting. Controls amnesia barriers
    /// and per-alter CTM attention routing.
    pub active_alter: Option<String>,
}

impl BrainPipeline {
    pub fn new(bank: MemoryBank) -> Self {
        Self {
            bank,
            working_memory: WorkingMemoryFilter::new(16),
            ctm: None,
            ctm_state: None,
            ctm_projector: None,
            active_alter: None,
        }
    }

    /// Signal explicit importance to the neuromodulator system.
    /// This is the "REMEMBER THIS" signal — norepinephrine burst.
    /// Call before generate() to boost the next recall/consolidation.
    pub fn signal_importance(&mut self, importance: f32) {
        if let Some(state) = &mut self.ctm_state {
            state.neuromod.signal_importance(importance);
        }
    }

    /// Set which alter is currently fronting.
    pub fn switch_alter(&mut self, alter: &str) {
        self.active_alter = Some(alter.into());

        // Apply per-alter attention bias to CTM if present
        if let Some(ctm) = &mut self.ctm {
            if let Some(alter_obj) = self.bank.alters.iter().find(|a| a.name == alter) {
                if !alter_obj.attention_bias.is_empty() {
                    let n = ctm.config.attention_layer.n_neurons;
                    let bias = &alter_obj.attention_bias;
                    // Add bias to attention region start activations
                    // This makes different alters "think differently" from tick 0
                    for i in 0..n.min(bias.len()) {
                        ctm.attention_region.start_activated[i] += bias[i];
                    }
                }
            }
        }
    }

    /// Attach a CTM for deliberative reasoning between backbone and memory.
    pub fn with_ctm(mut self, ctm: Ctm, backbone_dim: usize) -> Self {
        let d_input = ctm.config.d_input;
        self.ctm_projector = Some(CtmLinear::new(backbone_dim, d_input));
        self.ctm_state = Some(ctm.init_state());
        self.ctm = Some(ctm);
        self
    }

    /// Run the full pipeline: all filters → inference.
    pub fn generate(
        &mut self,
        prompt: &str,
        hidden_key: Vec<f32>,
        token_ids: Vec<i64>,
        max_tokens: usize,
        mut forward_fn: impl FnMut(&[i64]) -> Vec<Vec<f32>>,
    ) -> Response {
        let mut req = Request {
            prompt: prompt.to_string(),
            token_ids,
            hidden_key,
            max_tokens,
            eos_token_id: self.bank.model_id.eos_token_id,
            meta: RequestMeta::default(),
        };
        let mut mods = LogitMods::default();

        // ─── CTM deliberation (if attached) ─────────────────────
        // The CTM sits between backbone and memory: it THINKS about
        // the hidden state before deciding what to recall.
        let ctm_confidence = if let (Some(ctm), Some(state), Some(proj)) =
            (&mut self.ctm, &mut self.ctm_state, &self.ctm_projector)
        {
            // Project backbone hidden (896) → CTM observation (d_input)
            let observation = proj.forward(&req.hidden_key);

            // Run K ticks of deliberation
            let (predictions, _final_sync) = ctm.forward(&observation, state, true);

            // Confidence = sync convergence: how much did predictions change
            // between second-to-last and last tick, RELATIVE to their magnitude?
            // Normalized by prediction scale so confidence is scale-invariant.
            let confidence = if predictions.len() >= 2 {
                let last = &predictions[predictions.len() - 1];
                let prev = &predictions[predictions.len() - 2];
                let delta: f32 = last.iter().zip(prev)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();
                let magnitude: f32 = last.iter()
                    .map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
                let relative_delta = delta / magnitude;
                // Low relative delta = converged = high confidence
                (1.0 - relative_delta.min(1.0)).max(0.0)
            } else {
                0.5
            };

            // Compute activation energy for serotonin
            let act_energy: f32 = state.act_output.iter()
                .map(|x| x * x).sum::<f32>().sqrt();

            // Novelty: how different is this from recent replay buffer entries?
            let novelty = if ctm.replay.is_empty() {
                1.0 // first input = maximally novel
            } else {
                let max_sim = ctm.replay.entries.iter()
                    .map(|e| {
                        let dot: f32 = observation.iter().zip(&e.observation)
                            .map(|(a, b)| a * b).sum();
                        let na: f32 = observation.iter().map(|x| x * x).sum::<f32>().sqrt();
                        let nb: f32 = e.observation.iter().map(|x| x * x).sum::<f32>().sqrt();
                        if na > 1e-8 && nb > 1e-8 { dot / (na * nb) } else { 0.0 }
                    })
                    .fold(0.0f32, f32::max);
                (1.0 - max_sim).max(0.0)
            };

            // Update serotonin from energy × novelty
            state.neuromod.update_serotonin(act_energy, novelty);

            // Compute surprise proxy from prediction variance across ticks
            let surprise = if predictions.len() >= 2 {
                predictions.iter()
                    .map(|p| p.iter().map(|x| x * x).sum::<f32>())
                    .fold(0.0f32, |acc, x| acc + (x - act_energy).abs())
                    / predictions.len() as f32
            } else {
                1.0
            };

            // Push to replay buffer if surprising enough
            ctm.replay.push(observation.clone(), surprise);

            req.meta.ctm_confidence = confidence;
            req.meta.ctm_ticks = predictions.len();
            confidence
        } else {
            1.0 // No CTM = pass through with full confidence
        };

        // ─── Filters ────────────────────────────────────────────

        // 1. Avoidance (amygdala) — fear overrides everything
        for av in &self.bank.avoidances {
            if !av.active { continue; }
            let sim = cosine_similarity(&req.hidden_key, &av.key);
            if sim > self.bank.threshold {
                let gate = ((sim - self.bank.threshold) / (1.0 - self.bank.threshold)).min(1.0);
                for &tid in &av.suppress_token_ids {
                    mods.avoidance.push((tid, -av.strength * gate));
                }
            }
        }
        if !mods.avoidance.is_empty() {
            req.meta.avoided = true;
        }

        // 2. Episodic (hippocampus) — gated by CTM confidence
        //    Consolidated memories get WEAKER logit biases — the knowledge
        //    has graduated into CTM weights. The episodic trace fades as the
        //    semantic representation strengthens (complementary learning systems).
        //
        //    Respects: amnesia barriers, valence, reconsolidation window.
        if !req.meta.avoided {
            let queries = vec![req.hidden_key.clone()];
            let active = self.active_alter.as_deref();
            if let Some(result) = recall_multi_as(&self.bank, &queries, active) {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs_f64();

                let ep = &mut self.bank.alters[result.alter_index].episodes[result.episode_index];
                ep.recall_count += 1;

                // Reconsolidation window: memory is labile for 6h after recall.
                // During this window, conflicting info can modify it.
                // Nader (2003): "Memory traces unbound"
                ep.last_recalled_at = now;

                // Valence affects recall dynamics:
                // Fear memories get STRONGER injection (hypermnesia),
                // Positive memories consolidate faster (fade sooner).
                let valence_factor = match ep.valence {
                    modgrad_io::types::Valence::Fear => 1.5,     // fear = vivid, strong
                    modgrad_io::types::Valence::Negative => 1.2,  // negative = somewhat stronger
                    modgrad_io::types::Valence::Neutral => 1.0,
                    modgrad_io::types::Valence::Positive => 0.8,  // positive = consolidates faster, fades from episodic
                };

                // Consolidation fade: score=0 → full injection, score=1 → no injection
                // Fear memories resist consolidation — their fade is slower
                let consolidation_resistance = match ep.valence {
                    modgrad_io::types::Valence::Fear => 0.3,    // only 30% of consolidation reduces injection
                    modgrad_io::types::Valence::Negative => 0.6,
                    _ => 1.0,
                };
                let episodic_fade = 1.0 - (ep.consolidation_score * consolidation_resistance);

                // Choose logit bias source:
                // - Not consolidated → use stored episodic biases
                // - Consolidated + CTM has projector → use CTM-predicted biases
                // - Consolidated + no projector → use stored (fallback)
                let use_ctm_biases = ep.consolidated
                    && self.ctm.as_ref().map_or(false, |c| c.logit_projector.is_some());

                if use_ctm_biases {
                    // CTM produces bias STRENGTHS from its own weights — semantic recall.
                    // Token IDs come from the episode; strengths come from the CTM.
                    if let Some(ctm) = &self.ctm {
                        if let Some(state) = &self.ctm_state {
                            let sync = &state.sync_out.alpha;
                            if let Some(predicted) = ctm.predict_logit_biases(sync, &ep.logit_biases) {
                                mods.episodic = predicted;
                            } else {
                                mods.episodic = ep.logit_biases.clone();
                            }
                        } else {
                            mods.episodic = ep.logit_biases.clone();
                        }
                    }
                } else {
                    // Episodic recall — direct logit bias injection
                    mods.episodic = ep.logit_biases.clone();
                }

                // Gate = cosine_match × CTM_confidence × episodic_fade × valence
                // CTM confidence gates episodic recall, but can't fully suppress it.
                // A strong memory match should always inject some bias even if the CTM
                // is confused (random weights, early training). Floor at 0.1 ensures
                // exact matches (gate=1.0, strength=50) still override base model.
                let ctm_gate = ctm_confidence.max(0.1);
                let effective_gate = result.gate * ctm_gate * episodic_fade * valence_factor;
                mods.episodic_gate = effective_gate;
                req.meta.matched_alter = Some(self.bank.alters[result.alter_index].name.clone());
                req.meta.match_similarity = result.similarity;
                req.meta.gate = effective_gate;
            }
        }

        // 3. Rules (cerebellum)
        let prompt_lower = req.prompt.to_lowercase();
        for rule in &self.bank.rules {
            if rule.active
                && (rule.trigger.is_empty()
                    || prompt_lower.contains(&rule.trigger.to_lowercase()))
            {
                req.meta.active_rules.push(rule.instruction.clone());
            }
        }

        // 4. Working memory (PFC)
        self.working_memory.buffer.push_back(WorkingMemoryEntry {
            hidden: req.hidden_key.clone(),
            text: req.prompt.clone(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
        });
        if self.working_memory.buffer.len() > self.working_memory.capacity {
            self.working_memory.buffer.pop_front();
        }
        req.meta.working_memory_size = self.working_memory.buffer.len();

        // 5. Inference (backbone) — inline to avoid lifetime issues
        let mut ids = req.token_ids.clone();
        let _prompt_len = ids.len();
        let mut gen_step = 0;

        for _ in 0..req.max_tokens {
            let logits_all = forward_fn(&ids);
            let Some(last_logits) = logits_all.last() else { break };
            let mut logits = last_logits.clone();
            let v = logits.len();

            // Apply episodic biases (if not avoided)
            if mods.avoidance.is_empty() && gen_step < mods.episodic.len() {
                let bias = &mods.episodic[gen_step];
                if (bias.token_id as usize) < v {
                    logits[bias.token_id as usize] += mods.episodic_gate * bias.strength;
                }
                for &(tid, s) in &bias.suppress {
                    if (tid as usize) < v {
                        logits[tid as usize] += mods.episodic_gate * s;
                    }
                }
            }

            // Apply avoidance
            for &(tid, strength) in &mods.avoidance {
                if (tid as usize) < v {
                    logits[tid as usize] += strength;
                }
            }

            let next = logits.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as i64).unwrap_or(0);

            ids.push(next);
            gen_step += 1;
            if next == req.eos_token_id { break; }
        }

        Response {
            text: String::new(),
            token_ids: ids,
            meta: req.meta,
        }
    }
}
