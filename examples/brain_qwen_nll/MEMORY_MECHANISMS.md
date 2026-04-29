# Memory mechanisms in the brain → Qwen seam

This file captures an architectural insight from a long conversation:
**the codebase has multiple "episodic memory" mechanisms, they are all
the same primitive at different scales, and none of them will unlock
generalizable semantic recall while the brain's QK projections are
random-init.**

## Mechanisms inventory

| Mechanism | Where | Time scale | Trigger encoding | Response |
|---|---|---|---|---|
| Per-tick episodic attention | `modgrad-compute/src/kv_buffer.rs` + per-region MHA | Within forward (every tick) | brain hidden state via Q projection | weighted sum over stored V |
| Hippocampus → attention edge | `eight_region` connection added in `20736b2` | Within forward (per tick) | hippocampus activated state | feeds into ATTENTION region's input |
| Dream-phase priming | `modgrad-ctm/src/bio/dream.rs:prime_state` | Between forwards | retrieved blended_final_state | mixes into hippocampus init |
| CRI (Conditioned Reflex Injection) | `modgrad-ctm/src/monarch.rs:inject_reflex` | At forward | hidden state cosine match | adds biases to logits |

## They are all the same primitive

Every mechanism is **cosine-similarity-based content-addressable
memory**. Different time scales and different injection points, but
the core operation is identical:

1. Store a (key, value) pair where key is a hidden state pattern.
2. At query time, compute similarity between current state and stored
   keys.
3. Inject the value of the best-matching key as a bias on
   downstream computation.

Self-attention with a KV cache is the soft (weighted-sum) version.
CRI is the discrete (threshold) version. Dream-phase priming is the
state-level (blend into region_outputs) version. Per-tick episodic
attention is the in-flight version.

This is an architectural unification: **memory and attention are not
different systems**. They are the same operation at different scales,
and **hypnosis works because it is just memory installation by an
external party** — same primitive, different controller.

## Why adding more mechanisms wouldn't help

The brain's hippocampus QK projection is random-init Gaussian. It maps
input patterns to keys with random similarity geometry. Memory storage
is correct (KV buffer literally holds the past). Memory retrieval is
random because the matching function is random.

This means:

- **Per-tick attention**: query-key similarity is in a random subspace.
  Likely-similar past states match likely-similar past tokens, but no
  semantic specificity.

- **Dream-phase prime_state**: same QK randomness — retrieved episode
  is whichever past state happens to be closest in the random
  subspace, not whichever is semantically relevant.

- **CRI installed from corpus walk**: would work if cosine similarity
  in the brain's hidden state space was already discriminative
  per-fact. Empirically (per `fadf632` per-position diagnostic) it
  isn't — the brain conflates "of France" and "of Germany" enough that
  the modulator can't separate them.

Adding more wrappers around the same random-QK lookup doesn't add
information. It just exposes the same underlying random geometry from
different angles.

## What WOULD unlock recall

Train the QK projections. Either via:

1. **End-to-end gradient (BPTT)** through CTM ticks back into the
   brain's attention layers. Multi-day slice. The architecture has
   the backward path (`regional_train_token_*`); we just don't route
   the modulator's d_brain_output into it.

2. **Self-supervised pretraining** of the brain on next-token
   prediction (loss on brain's own predictions, not Qwen's). This
   pressures the QK to develop content-aware retrieval for prediction.

3. **Attention QK only** (intermediate): freeze synapse/NLM/sync,
   only let the optimizer touch QK. Smallest training scope, addresses
   the bottleneck directly.

Until one of these lands, the empirical ceiling on recall is bounded
by what random QK projections can match — which (per this session's
ablation) is "small generic content-causal correction, no clean
semantic recall."

## Decision

No more memory-mechanism wrappers in this example. Either:

- Land brain QK training (multi-day, architectural).
- Move to a different problem.

The mechanism debate is resolved: the code is correct, the weights
aren't trained. Adding more hooks on top of untrained weights is
displacement activity.

— Discovered through a long discussion that started with "have we
tried the conditional reflex injection trick for memory" and ended
with "memory IS continuous self-conditioning." Both true; both
already in code; neither matters until QK trains.

## Friction discovered while scoping joint-train integration

(Added 2026-04-29.) The `--joint-train` scaffold flag in
`brain_qwen_nll` was added at `57e61c3`, but wiring its body
revealed a real architectural friction with `regional_train_step`
that wasn't visible at the unit-test scale of `d6d9775`:

**`regional_train_step` creates a fresh `RegionalState::new(w)` per
call** (see `crates/modgrad-ctm/src/graph.rs:2738`). Each call
starts the brain from `start_activated` and a fresh empty episodic
KV. This is fine for single-step gradient checks but **breaks the
"continuous memory across tokens" assumption** the brain_qwen_nll
modulator-only path relies on (which uses `nc.step` that mutates
`nc.state` across calls).

For per-token joint training to maintain episodic memory across
tokens — which is the entire point of having episodic memory — we
need ONE OF:

1. **A stateful train variant** that takes `&mut RegionalState`
   alongside `&mut grads`. Real engineering: refactor of the
   training function or a new wrapper.

2. **Accept per-step independence** during training (each token's
   training gradient computed against fresh-state forward). Loses
   the "brain accumulates context" signal for training; only good
   for IID training samples, not for token streams.

3. **Whole-sequence forward then per-step backward** — analogous to
   how transformers train on a sequence (one forward over the
   whole sequence, then per-position backward). Architecturally
   different from the current per-step `regional_train_step` loop;
   would require a new `regional_train_sequence` function.

(1) is the cleanest fit for joint train on token streams. (3) is
the cleanest fit for "this is how transformers train"; might let
us reuse Qwen's per-position backward pattern. (2) sacrifices the
architectural commitment to stateful brain.

This is the actual gap between `d6d9775` (chain composes at
single-step scale) and "wire joint train into brain_qwen_nll"
(needs sequence-level state management). Smaller than rebuilding
brain from scratch but bigger than I scoped at `57e61c3`.

Decision: scaffold flag stays unwired until one of (1)/(3) lands.
Don't ship a partial joint-train that silently uses fresh-state
per token — that would be the WORST of both worlds (joint training
without the memory advantage that motivates the brain at all).

## Calibration: the Qwen2.5-0.5B baseline is weak

(Added 2026-04-29.) A roundtable of three haiku probe agents +
direct test runs converged on the same observation:

  Raw prompt + T=0.7    → token salad
  Chat template + T=0.7 → multilingual word soup
  Chat template + T=0   → "C C C C / Answer: C" loop

Qwen2.5-0.5B is genuinely a poor chatbot at this size. The chat
template parses correctly (special tokens 151644/151645 visible in
encoded IDs), the throughput is fine (~35 tok/s on gfx1102), the
sampler works — the model itself just doesn't produce coherent
free-form text at 500M params, regardless of formatting or
temperature.

This calibrates the held-out NLL improvements measured this
session:

  best on 37-token corpus  : -0.13 nats / +13.9% raw probability
  robust on 265-token corpus: -0.003 nats

These are real NLL deltas — the math holds. But they're nudging
probability mass within a distribution whose mode is already
gibberish. Brain isn't going to make Qwen-0.5B speak; it's just
slightly improving its already-broken token guesses.

For the redshiftzero arc this means:
  ✓ The architectural pipeline (modulator, gradient flow, joint
    training infra) is verified end-to-end on real weights.
  ✗ The practical demonstration "brain makes the LLM better at
    talking" needs a stronger LLM. Qwen2.5-0.5B is the test
    harness; Qwen2.5-3B or 7B is the actual deployment target if
    we want observable text-quality gains.

The session's primitives are correct and ready for a bigger Qwen.
The held-out NLL improvements at 500M param size are measurement
artifacts of a working pipeline on a weak baseline, not a claim
that brain makes Qwen-0.5B competent.
