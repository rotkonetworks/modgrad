//! Service abstraction: the symmetric async function at the core.
//!
//! A Service is `Req → Response`. Clients and servers are both Services.
//! Filters compose with Services to modify behavior.

use crate::types::LogitBias;

/// A generation request flowing through the pipeline.
#[derive(Debug, Clone)]
pub struct Request {
    /// The prompt text
    pub prompt: String,
    /// Token IDs (populated by tokenizer)
    pub token_ids: Vec<i64>,
    /// Hidden state at last token (populated by backbone)
    pub hidden_key: Vec<f32>,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// EOS token ID for this model
    pub eos_token_id: i64,
    /// Metadata flowing through the pipeline
    pub meta: RequestMeta,
}

/// Metadata accumulated as the request passes through filters.
#[derive(Debug, Clone, Default)]
pub struct RequestMeta {
    /// Which alter matched (set by EpisodicFilter)
    pub matched_alter: Option<String>,
    /// Similarity of best match (set by EpisodicFilter)
    pub match_similarity: f32,
    /// Gate value (set by EpisodicFilter)
    pub gate: f32,
    /// Whether avoidance was triggered (set by AvoidanceFilter)
    pub avoided: bool,
    /// Avoidance reason (set by AvoidanceFilter)
    pub avoidance_reason: Option<String>,
    /// Active rules (set by RulesFilter)
    pub active_rules: Vec<String>,
    /// Working memory size (set by WorkingMemoryFilter)
    pub working_memory_size: usize,
    /// CTM confidence from sync convergence (0=uncertain, 1=converged)
    pub ctm_confidence: f32,
    /// CTM ticks used
    pub ctm_ticks: usize,
}

/// A generation response from the pipeline.
#[derive(Debug, Clone)]
pub struct Response {
    /// Generated text
    pub text: String,
    /// Full token IDs (prompt + generated)
    pub token_ids: Vec<i64>,
    /// Pipeline metadata
    pub meta: RequestMeta,
}

/// Logit modifications to apply during generation.
#[derive(Debug, Clone, Default)]
pub struct LogitMods {
    /// Sequential biases from episodic memory (one per gen step)
    pub episodic: Vec<LogitBias>,
    /// Gate strength for episodic biases
    pub episodic_gate: f32,
    /// Suppression biases from amygdala (applied every step)
    pub avoidance: Vec<(u32, f32)>,
}

/// The core service trait. A function from Request to Response.
pub trait Service {
    fn call(&mut self, req: Request, mods: &mut LogitMods) -> Response;
}

/// A filter wraps a service, modifying its behavior.
/// Filters compose: `filter.and_then(service)` produces a new service.
pub trait Filter {
    /// Process the request, potentially modifying it and the logit mods,
    /// then delegate to the inner service.
    fn apply(&mut self, req: Request, mods: &mut LogitMods, inner: &mut dyn Service) -> Response;
}

/// A filter composed with a service, producing a new service.
pub struct Filtered<F, S> {
    pub filter: F,
    pub inner: S,
}

impl<F: Filter, S: Service> Service for Filtered<F, S> {
    fn call(&mut self, req: Request, mods: &mut LogitMods) -> Response {
        self.filter.apply(req, mods, &mut self.inner)
    }
}

/// Extension trait for composing filters with services.
pub trait ServiceExt: Service + Sized {
    fn and_then<F: Filter>(self, filter: F) -> Filtered<F, Self> {
        Filtered { filter, inner: self }
    }
}

impl<S: Service> ServiceExt for S {}

/// The bottom-level inference service: runs the frozen backbone.
/// The bottom-level service in the composable pipeline.
pub struct InferenceService {
    /// Callback that runs ONNX inference: token_ids → logits
    forward_fn: Box<dyn FnMut(&[i64]) -> Vec<Vec<f32>>>,
    /// Vocabulary size
    vocab_size: usize,
}

impl InferenceService {
    pub fn new(
        forward_fn: impl FnMut(&[i64]) -> Vec<Vec<f32>> + 'static,
        vocab_size: usize,
    ) -> Self {
        Self {
            forward_fn: Box::new(forward_fn),
            vocab_size,
        }
    }
}

impl Service for InferenceService {
    fn call(&mut self, req: Request, mods: &mut LogitMods) -> Response {
        let mut ids = req.token_ids.clone();
        let _prompt_len = ids.len();

        let mut gen_step = 0;
        for _ in 0..req.max_tokens {
            let logits_all = (self.forward_fn)(&ids);
            let Some(last_logits) = logits_all.last() else { break };
            let mut logits = last_logits.clone();
            let v = logits.len();

            // Apply episodic memory biases (if avoidance not triggered)
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

            // Apply avoidance suppressions
            for &(tid, strength) in &mods.avoidance {
                if (tid as usize) < v {
                    logits[tid as usize] += strength;
                }
            }

            // Argmax
            let next = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as i64)
                .unwrap_or(0);

            ids.push(next);
            gen_step += 1;

            // EOS
            if next == req.eos_token_id {
                break;
            }
        }

        Response {
            text: String::new(), // decoded by caller
            token_ids: ids,
            meta: req.meta,
        }
    }
}
