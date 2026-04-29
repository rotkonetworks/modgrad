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
