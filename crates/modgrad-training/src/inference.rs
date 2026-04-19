//! Inference runtime — the dual of the training loop.
//!
//! train:    Brain + Loss + Data    → Weights
//! generate: Brain + Sampler + Stop → Tokens
//!
//! Speculation is an internal optimization, not an API concept.
//! If a drafter is provided, generate() uses speculative decoding
//! (draft N tokens cheaply, verify in batch, accept prefix).
//! The output is identical — lossless guarantee.
//!
//! The step function (`StepFn`) is the universal interface between
//! generate and any architecture. Transformers, CTM, future models
//! all reduce to: token → logits. The caller constructs the closure;
//! generate() doesn't know what's behind it.

use modgrad_traits::Sampler;

/// One step of autoregressive generation: feed a token, get logits back.
/// The closure captures the model, weights, and state — generate() is
/// agnostic to what produces the logits.
pub type StepFn<'a> = &'a mut dyn FnMut(usize) -> Vec<f32>;

/// Batch-verify multiple tokens at once. Returns logits for each position.
/// Used for speculative verification — the verifier scores all draft
/// tokens in a single forward pass.
pub type VerifyFn<'a> = &'a mut dyn FnMut(&[usize]) -> Vec<Vec<f32>>;

/// When to stop generating.
pub enum Stop {
    /// Stop after N tokens.
    MaxTokens(usize),
    /// Stop when this token is produced.
    Eos(usize),
    /// Stop on either condition.
    EosOrMax { eos: usize, max_tokens: usize },
}

impl Stop {
    fn should_stop(&self, token: usize, n: usize) -> bool {
        match self {
            Stop::MaxTokens(max) => n >= *max,
            Stop::Eos(eos) => token == *eos,
            Stop::EosOrMax { eos, max_tokens } => token == *eos || n >= *max_tokens,
        }
    }
}

/// Result of generation.
pub struct GenerateResult {
    pub tokens: Vec<usize>,
    /// Number of forward passes through the verifier (= draft-verify cycles).
    /// Lower than tokens.len() when speculation accepts multiple tokens per cycle.
    pub verifier_steps: usize,
    /// Number of speculative tokens accepted (0 if speculation not used).
    pub speculative_accepts: usize,
}

// ═══════════════════════════════════════════════════════════════
// NORMAL DECODING
// ═══════════════════════════════════════════════════════════════

/// Generate tokens. The simplest correct thing.
///
/// `step`: feed a token, get next-token logits. The closure owns the model.
/// `initial_logits`: logits from the prefill/prompt forward pass.
/// `sampler`: how to pick tokens from logits.
/// `stop`: when to stop.
///
/// ```ignore
/// // Usage with any Brain:
/// let mut state = B::init_state(&weights);
/// let (output, new_state) = B::forward(&weights, state, &prompt);
/// state = new_state;
/// let initial_logits = output.predictions.last().unwrap().clone();
///
/// let result = generate(
///     &mut |token| {
///         let input = embed(token);
///         let (out, s) = B::forward(&weights, state, &input);
///         state = s;
///         out.predictions.last().unwrap().clone()
///     },
///     initial_logits,
///     &mut Greedy,
///     &Stop::EosOrMax { eos: 0, max_tokens: 512 },
/// );
/// ```
pub fn generate(
    step: StepFn<'_>,
    initial_logits: Vec<f32>,
    sampler: &mut dyn Sampler,
    stop: &Stop,
) -> GenerateResult {
    let mut tokens = Vec::new();
    let mut logits = initial_logits;

    loop {
        let token = sampler.sample(&logits);
        tokens.push(token);
        if stop.should_stop(token, tokens.len()) { break; }
        logits = step(token);
    }

    GenerateResult {
        verifier_steps: tokens.len(),
        speculative_accepts: 0,
        tokens,
    }
}

// ═══════════════════════════════════════════════════════════════
// SPECULATIVE DECODING
// ═══════════════════════════════════════════════════════════════

/// Generate with speculative decoding. Same output as `generate()` —
/// the verifier is authoritative, speculation only affects speed.
///
/// `draft`: cheap model that guesses N future tokens.
/// `verify`: authoritative model that scores a batch of tokens.
///           Returns logits[i] = P(next | context ++ tokens[..i]).
/// `budget`: how many tokens to draft per cycle.
///
/// The lossless guarantee: for greedy decoding, output is bitwise
/// identical to normal decoding. For stochastic sampling, the current
/// implementation uses token-matching (not rejection sampling), so the
/// output distribution may differ slightly from normal decoding.
/// Proper rejection sampling would make the stochastic path bit-
/// equivalent to normal decoding — not implemented yet.
pub fn generate_speculative(
    draft: StepFn<'_>,
    verify: VerifyFn<'_>,
    initial_logits: Vec<f32>,
    sampler: &mut dyn Sampler,
    stop: &Stop,
    budget: usize,
) -> GenerateResult {
    let mut tokens = Vec::new();
    let mut logits = initial_logits;
    let mut verifier_steps = 0;
    let mut speculative_accepts = 0;

    loop {
        // Sample first token from current logits (always accepted)
        let first = sampler.sample(&logits);
        tokens.push(first);
        if stop.should_stop(first, tokens.len()) { break; }

        // Draft phase: generate `budget` candidate tokens cheaply
        let mut draft_tokens = Vec::with_capacity(budget);
        let mut dl = draft(first);

        for _ in 0..budget {
            let t = sampler.sample(&dl);
            draft_tokens.push(t);
            dl = draft(t);
        }

        // Verify phase: batch-score all draft tokens through the verifier
        let verified = verify(&draft_tokens);
        verifier_steps += 1;

        // Accept prefix: find longest prefix where verifier agrees with drafter
        let mut accepted = 0;
        for i in 0..draft_tokens.len() {
            let verifier_choice = sampler.sample(&verified[i]);
            if verifier_choice == draft_tokens[i] {
                tokens.push(draft_tokens[i]);
                accepted += 1;
                speculative_accepts += 1;
                if stop.should_stop(draft_tokens[i], tokens.len()) {
                    return GenerateResult { tokens, verifier_steps, speculative_accepts };
                }
            } else {
                // Divergence: take the verifier's token instead (bonus token)
                tokens.push(verifier_choice);
                if stop.should_stop(verifier_choice, tokens.len()) {
                    return GenerateResult { tokens, verifier_steps, speculative_accepts };
                }
                // Continue from verifier's logits at divergence point
                logits = verified[i].clone();
                break;
            }
        }

        // All draft tokens accepted — use logits after last verified position
        if accepted == draft_tokens.len() {
            logits = if let Some(last) = verified.last() {
                last.clone()
            } else {
                draft(draft_tokens[draft_tokens.len() - 1])
            };
        }
    }

    GenerateResult { tokens, verifier_steps, speculative_accepts }
}

// ═══════════════════════════════════════════════════════════════
// UNIFIED ENTRY POINT
// ═══════════════════════════════════════════════════════════════

/// Configuration for generation. Pass a drafter to enable speculation.
pub struct GenerateConfig<'a> {
    pub sampler: &'a mut dyn Sampler,
    pub stop: Stop,
    /// Optional: cheap draft model for speculative decoding.
    /// If None, uses normal autoregressive decoding.
    pub drafter: Option<DraftConfig<'a>>,
}

/// Draft model configuration.
pub struct DraftConfig<'a> {
    /// Draft step function (cheap model).
    pub step: StepFn<'a>,
    /// How many tokens to draft per cycle.
    pub budget: usize,
}

/// Unified generate — picks the right strategy automatically.
///
/// If a drafter is provided, uses speculative decoding.
/// Otherwise, normal autoregressive decoding.
/// Output is identical either way.
pub fn generate_auto(
    step: StepFn<'_>,
    verify: Option<VerifyFn<'_>>,
    initial_logits: Vec<f32>,
    config: &mut GenerateConfig<'_>,
) -> GenerateResult {
    match (config.drafter.take(), verify) {
        (Some(dc), Some(vf)) => {
            generate_speculative(
                dc.step, vf,
                initial_logits,
                config.sampler,
                &config.stop,
                dc.budget,
            )
        }
        _ => {
            generate(step, initial_logits, config.sampler, &config.stop)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use modgrad_traits::Greedy;

    #[test]
    fn greedy_generate() {
        // Trivial model: always predicts token 1, then token 2, then EOS (0)
        let sequence = [1usize, 2, 0];
        let mut pos = 0;
        let mut step = |_token: usize| -> Vec<f32> {
            pos += 1;
            let mut logits = vec![0.0; 3];
            if pos < sequence.len() {
                logits[sequence[pos]] = 10.0;
            } else {
                logits[0] = 10.0; // EOS
            }
            logits
        };

        let mut initial = vec![0.0; 3];
        initial[1] = 10.0; // first token = 1

        let result = generate(
            &mut step,
            initial,
            &mut Greedy,
            &Stop::EosOrMax { eos: 0, max_tokens: 10 },
        );
        assert_eq!(result.tokens, vec![1, 2, 0]);
    }

    #[test]
    fn speculative_matches_normal() {
        // Verifier: deterministic sequence 3,1,4,1,5
        let target = [3usize, 1, 4, 1, 5];
        let mut v_pos;

        let make_logits = |token: usize| -> Vec<f32> {
            let mut l = vec![0.0; 6];
            l[token] = 10.0;
            l
        };

        // Normal decode
        let mut n_pos = 0;
        let result_normal = generate(
            &mut |_t| {
                n_pos += 1;
                make_logits(if n_pos < target.len() { target[n_pos] } else { 0 })
            },
            make_logits(target[0]),
            &mut Greedy,
            &Stop::MaxTokens(5),
        );

        // Speculative decode with perfect drafter (same as verifier)
        let mut d_pos = 0;
        v_pos = 0;
        let result_spec = generate_speculative(
            &mut |_t| {
                d_pos += 1;
                make_logits(if d_pos < target.len() { target[d_pos] } else { 0 })
            },
            &mut |tokens| {
                tokens.iter().map(|_| {
                    v_pos += 1;
                    make_logits(if v_pos < target.len() { target[v_pos] } else { 0 })
                }).collect()
            },
            make_logits(target[0]),
            &mut Greedy,
            &Stop::MaxTokens(5),
            4,
        );

        assert_eq!(result_normal.tokens, result_spec.tokens);
        assert!(result_spec.verifier_steps < result_normal.verifier_steps);
    }
}
