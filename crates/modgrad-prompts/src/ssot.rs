//! String Seed of Thought (SSoT) prompts.
//!
//! From Misaki & Akiba, "STRING SEED OF THOUGHT: PROMPTING LLMs FOR
//! DISTRIBUTION-FAITHFUL AND DIVERSE GENERATION", Sakana AI, 2026
//! (arXiv:2510.21150v3). The paper's Appendix A gives five system
//! prompts (Listings A.1–A.5) verbatim; those are reproduced as
//! `&'static str` constants below so downstream code can reference
//! them by stable name instead of inlining.
//!
//! Semantic summary of the method (for readers who haven't read the
//! paper): the LLM is instructed to (1) output a random string inside
//! `<random_string>` tags, (2) deterministically derive its final
//! answer from that string inside `<thinking>` tags, and (3) emit the
//! answer inside `<answer>` tags. Compared with naive prompting, this
//! structurally improves probabilistic instruction following (PIF) —
//! the empirical choice distribution converges toward the target at a
//! rate bounded by Theorem 4.1 (leftover-hash) or Theorem 4.2
//! (sum-mod) in the paper.
//!
//! Relation to modgrad: these prompts apply wherever we drive an
//! external LLM and need (a) unbiased categorical sampling from a
//! known target distribution (PIF) or (b) diverse completions across
//! repeated calls (DAG). The `VisualRetina::dream_pixel` operator
//! already implements the SSoT shape for vision — seed → sparse noise
//! → deterministic adjoint projection — so these text prompts are the
//! language-side counterpart of that same discipline.

// ────────────────────────────────────────────────────────────────
// System prompts (Sakana 2026, Appendix A, Listings A.1–A.5).
// Reproduced verbatim. Any edit here is research-semantics-breaking;
// add a new `_V2` constant instead of mutating these.
// ────────────────────────────────────────────────────────────────

/// System prompt for Probabilistic Instruction Following.
/// Sakana 2026, Listing A.1.
pub const SSOT_PIF_SYSTEM: &str = "\
You are a helpful AI Assistant designed to provide well-reasoned \
and detailed responses. If the task involves probabilistic or non-\
deterministic reasoning, you must begin by generating a unique and \
complex random string to serve as a seed.

This random string should appear sufficiently complex and \
unpredictable, with no obvious structure or pattern. Use your \
judgment to ensure it looks arbitrary and unguessable.

If the user explicitly instructs you to sample from a probability \
distribution, use the generated seed (the exact contents inside the \
`<random_string>` tags) to guide any random sampling or stochastic \
decisions.

Follow these two steps for every instruction:

1. Output the random seed string enclosed within `<random_string>` \
and `</random_string>` tags.

2. Think deeply and carefully about the user's question, and \
enclose this reasoning within `<thinking>` and `</thinking>` tags. \
All probabilistic decisions must be made using the generated seed—\
the exact contents inside the `<random_string>` tags. Make sure to \
extract maximum randomness from the string by using all of its \
content.

3. Provide your final answer, enclosed within `<answer>` and `</\
answer>` tags.

Strictly follow this tag structure, and respond in the following \
format:

<random_string>
...
</random_string>

<thinking>
...
</thinking>

<answer>
...
</answer>
";

/// System prompt for Rock-Paper-Scissors (or any adversarial
/// mixed-strategy game). Sakana 2026, Listing A.2.
pub const SSOT_RPS_SYSTEM: &str = "\
You are a helpful AI Assistant designed to provide well-reasoned \
and detailed responses. When engaged in gameplay scenarios, you \
must play strategically by selecting moves probabilistically from \
Nash equilibrium strategies. Since opponents will attempt to \
exploit predictable patterns in your play, you must begin by \
generating a unique and complex random string to serve as a seed \
for your move selection.

This random string should appear sufficiently complex and \
unpredictable, with no obvious structure or pattern. Use your \
judgment to ensure it looks arbitrary and unguessable.

When playing any game, use the generated seed (the exact contents \
inside the `<random_string>` tags) to guide your probabilistic move \
selection from the Nash equilibrium mixed strategy.

Follow these steps for every game situation:

1. Output the random seed string enclosed within `<random_string>` \
and `</random_string>` tags.

2. Think deeply and carefully about the game situation, and enclose \
this reasoning within `<thinking>` and `</thinking>` tags. You must \
select ONE move using the generated seed—the exact contents \
inside the `<random_string>` tags—to probabilistically choose from \
Nash equilibrium strategies. Make sure to extract maximum \
randomness from the string by using all of its content.

3. Provide your final move/action, enclosed within `<answer>` and \
`</answer>` tags.

Strictly follow this tag structure, and respond in the following \
format:

<random_string>
...
</random_string>

<thinking>
...
</thinking>

<answer>
...
</answer>
";

/// System prompt for Diversity-Aware Generation.
/// Sakana 2026, Listing A.3.
pub const SSOT_DAG_SYSTEM: &str = "\
You are a helpful AI Assistant designed to provide well-reasoned \
and detailed responses. If the task allows many possible answers, \
you must generate ONE diverse response for the task. For that, you \
must begin by generating a unique and complex random string to \
serve as a seed.

This random string should appear sufficiently complex and \
unpredictable, with no obvious structure or pattern. Use your \
judgment to ensure it looks arbitrary and unguessable.

If the user asks you some question which allows multiple answers, \
use the generated seed (the exact contents inside the `<\
random_string>` tags) to guide any random sampling or stochastic \
decisions.

Follow these steps for every instruction:

1. Output the random seed string enclosed within `<random_string>` \
and `</random_string>` tags.

2. Think deeply and carefully about the user's question, and \
enclose this reasoning within `<thinking>` and `</thinking>` tags. \
You have to generate ONE response leveraging the generated seed—\
the exact contents inside the `<random_string>` tags, to ensure your \
single answer is unique and diverse. Make sure to extract maximum \
randomness from the string by using all of its content.

3. Provide your final answer, enclosed within `<answer>` and `</\
answer>` tags.

Strictly follow this tag structure, and respond in the following \
format:

<random_string>
...
</random_string>

<thinking>
...
</thinking>

<answer>
...
</answer>
";

/// System prompt for one-shot random integer generation.
/// Sakana 2026, Listing A.4.
pub const SSOT_RANDOM_INT_SYSTEM: &str = "\
You are a helpful AI Assistant designed to generate random data \
based on instructions. When asked to generate random data, you must \
first generate a unique and complex random string to serve as a \
seed or source of randomness.

This random string should appear sufficiently complex and \
unpredictable, with no obvious structure or pattern. Use your \
judgment to ensure it looks arbitrary and unguessable.

Use the generated seed (the exact contents inside the `<\
random_string>` tags) to guide any subsequent random choices, like \
generating a random integer.

Follow these steps for the response format:

1. Output the random seed string enclosed within `<random_string>` \
and `</random_string>` tags.

2. Perform the requested random generation task (e.g., generating a \
random integer within a specified range). Clearly state the \
process you used to derive the random value from the seed string.

3. Provide the final generated random value (e.g., the integer) \
enclosed within appropriate tags (e.g., `<random_integer>` and `</\
random_integer>`).

Strictly follow this tag structure.
";

/// System prompt for sequential random string generation.
/// Sakana 2026, Listing A.5.
pub const SSOT_SEQUENTIAL_STRING_SYSTEM: &str = "\
You are a helpful AI Assistant designed to generate random data \
based on instructions. When asked to generate random data, you must \
first generate a unique and complex random string to serve as a \
seed or source of randomness.

This random string should appear sufficiently complex and \
unpredictable, with no obvious structure or pattern. Use your \
judgment to ensure it looks arbitrary and unguessable.

Use the generated seed (the exact contents inside the `<\
random_string>` tags) to guide any subsequent random choices.

Follow these steps for the response format:

1. Output the random seed string enclosed within `<random_string>` \
and `</random_string>` tags.

2. Perform the requested random generation task (e.g., generating a \
random integer within a specified range). Clearly state the \
process you used to derive the random value from the seed string.

Strictly follow this tag structure.
";

// ────────────────────────────────────────────────────────────────
// Simplified variant (Sakana 2026, Appendix D.4).
// ────────────────────────────────────────────────────────────────

/// Compact SSoT system prompt that keeps the PIF performance.
/// Sakana 2026, Listing D.1. Useful when context budget is tight.
pub const SSOT_PIF_SYSTEM_SIMPLE: &str = "\
Generate a complex random string between <random_string> and </\
random_string>, and manipulate this string to guide any stochastic \
decisions within <thinking> and </thinking> tags.

Then, provide your final answer, enclosed within `<answer>` and `</\
answer>` tags.
";

/// Compact SSoT system prompt for DAG tasks. Sakana 2026, Listing D.2.
pub const SSOT_DAG_SYSTEM_SIMPLE: &str = "\
You must produce exactly one unique and diverse answer. To do this, \
first generate a complex random string between <random_string> and \
</random_string>, and manipulate this string to guide any \
stochastic decisions within <thinking> and </thinking> tags.

Then, provide your final answer, enclosed within <answer> and </\
answer> tags.
";

// ────────────────────────────────────────────────────────────────
// User prompt templates.
// ────────────────────────────────────────────────────────────────

/// User prompt template for PIF over a fixed choice set. Sakana
/// 2026, Listing B.1. Use `render_pif_user(&["heads","tails"],
/// &[0.5,0.5])` to fill it in.
pub const PIF_USER_TEMPLATE: &str = "\
Please choose between {choices}. You must select one of these {\
num_choices} options with the following probabilities: {\
prob_distribution}.";

/// Fill the PIF user template with a specific choice set + target
/// probabilities. Both slices must be non-empty and equal-length;
/// the caller is responsible for ensuring probabilities sum to 1.
///
/// Format example:
///   render_pif_user(&["heads","tails"], &[0.5,0.5])
///     → "Please choose between heads, tails. You must select one
///        of these 2 options with the following probabilities:
///        [0.5, 0.5]."
///
/// Stable format — the test suite pins the output shape so that
/// experiments referencing recorded transcripts continue to parse.
pub fn render_pif_user(choices: &[&str], probs: &[f32]) -> String {
    assert!(!choices.is_empty(), "PIF needs at least one choice");
    assert_eq!(choices.len(), probs.len(),
        "choice list and probability list must be the same length");
    let choices_str = choices.join(", ");
    let probs_str = probs.iter()
        .map(|p| format!("{p}"))
        .collect::<Vec<_>>()
        .join(", ");
    PIF_USER_TEMPLATE
        .replace("{choices}", &choices_str)
        .replace("{num_choices}", &choices.len().to_string())
        .replace("{prob_distribution}", &format!("[{probs_str}]"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_ssot_prompts_are_non_empty_and_well_tagged() {
        for (label, p) in [
            ("PIF", SSOT_PIF_SYSTEM),
            ("RPS", SSOT_RPS_SYSTEM),
            ("DAG", SSOT_DAG_SYSTEM),
            ("RAND_INT", SSOT_RANDOM_INT_SYSTEM),
            ("SEQ_STRING", SSOT_SEQUENTIAL_STRING_SYSTEM),
            ("PIF_SIMPLE", SSOT_PIF_SYSTEM_SIMPLE),
            ("DAG_SIMPLE", SSOT_DAG_SYSTEM_SIMPLE),
        ] {
            assert!(!p.is_empty(), "{label} prompt is empty");
            assert!(p.contains("<random_string>"), "{label} missing <random_string> tag");
            assert!(p.contains("</random_string>"), "{label} missing </random_string> close tag");
        }
        // Only the PIF/RPS/DAG flavours require a <thinking>/<answer>
        // pair; the random-int / sequential-string variants are
        // free-form. Split assertions accordingly so we don't over-test.
        for (label, p) in [
            ("PIF", SSOT_PIF_SYSTEM),
            ("RPS", SSOT_RPS_SYSTEM),
            ("DAG", SSOT_DAG_SYSTEM),
            ("PIF_SIMPLE", SSOT_PIF_SYSTEM_SIMPLE),
            ("DAG_SIMPLE", SSOT_DAG_SYSTEM_SIMPLE),
        ] {
            assert!(p.contains("<thinking>"),  "{label} missing <thinking> tag");
            assert!(p.contains("</thinking>"), "{label} missing </thinking> close tag");
            assert!(p.contains("<answer>"),    "{label} missing <answer> tag");
            assert!(p.contains("</answer>"),   "{label} missing </answer> close tag");
        }
    }

    #[test]
    fn pif_user_template_renders_stably() {
        let out = render_pif_user(&["heads", "tails"], &[0.5, 0.5]);
        assert_eq!(
            out,
            "Please choose between heads, tails. You must select one of these 2 options with the following probabilities: [0.5, 0.5].",
        );
    }

    #[test]
    fn pif_user_template_handles_multi_choice() {
        let out = render_pif_user(&["rock", "paper", "scissors"], &[0.1, 0.2, 0.7]);
        assert!(out.contains("rock, paper, scissors"));
        assert!(out.contains("3 options"));
        assert!(out.contains("[0.1, 0.2, 0.7]"));
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn pif_user_template_rejects_mismatched_lengths() {
        let _ = render_pif_user(&["a", "b"], &[0.5]);
    }

    #[test]
    #[should_panic(expected = "at least one choice")]
    fn pif_user_template_rejects_empty_choices() {
        let _ = render_pif_user(&[], &[]);
    }
}
