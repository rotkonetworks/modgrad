# Brain Architecture — Relation Graph

This document is the source of truth for **what connects to what** in the
isis CTM brain, and **how big each region should be**. The current preset
is legacy (toy-sized cerebellum); the target architecture mounts a real
LLM (Qwen2.5 / BLT-byte-ified) as the cerebellum.

Read this in tandem with:
- `crates/modgrad-ctm/src/graph.rs` — the `RegionalConfig` presets.
- `crates/modgrad-ctm/src/cerebellum.rs` — the `FrozenCerebellum` trait.
- `crates/modgrad-blt/` — the BLT byte-level model that replaces the
  existing frozen cerebellum.

---

## 1. The eight regions

```
                              OBSERVATION
                                   │
                                   ▼
   ┌──────────────────────────────────────────────────────────────┐
   │                         CORTICAL LOOP                        │
   │                                                              │
   │   input ─→ attention ─→ output ─→ motor ─┐                   │
   │     ▲                                    │                   │
   │     └────────────────────────────────────┘                   │
   └────────┬───────────────────────────┬─────────────────────────┘
            │                           │ + observation
            │                           ▼
            │                       cerebellum  ←── THE LLM
            │                       (Qwen2.5 / BLT)
            │
            │     ┌──────────────┐         ┌──────────────┐
            │     │ basal_ganglia│ ◀── output                       │
            │     └──────────────┘                                  │
            │                                                       │
            │     ┌──────────────┐         ┌──────────────┐         │
            └────▶│ hippocampus  │────────▶│   insula     │         │
                  │ (sees all 4  │         │              │         │
                  │  cortical)   │         └──────────────┘         │
                  └──────────────┘                                  │
```

| # | Region | Role | Receives from |
|---|---|---|---|
| 0 | **input** | Sensory encoder (bytes, pixels). Embeds raw observation. | motor, observation |
| 1 | **attention** | Salience routing. Selects what to attend to. | input |
| 2 | **output** | Action / language decoder. Picks the next byte / next move. | attention |
| 3 | **motor** | Action effector. Emits the actual action signal. | output |
| 4 | **cerebellum** | **World model / language model.** Reads motor's action plan + observation, predicts future world state. **Mounting point for the LLM (Qwen2.5 → BLT).** | motor, observation |
| 5 | **basal_ganglia** | Decision gating. Modulates which output activates. | output |
| 6 | **insula** | Interoception. Salience / motivation. | hippocampus |
| 7 | **hippocampus** | Episodic memory. KV cache + recall. | input, attention, output, motor |

---

## 2. Connection graph (the relation table)

This is the source of truth for the `Connection { from, to, … }` list in
`RegionalConfig::eight_region`. Each row is one directed edge. Cycles
are intentional — the cortical loop is closed.

| edge # | from | to | carries observation? | notes |
|---|---|---|---|---|
| 0 | motor | input | ✓ | Action-conditioned next-frame perception |
| 1 | input | attention | — | Salience selection |
| 2 | attention | output | — | Selected representation → decision |
| 3 | output | motor | — | Decision → effector |
| 4 | motor | **cerebellum** | ✓ | **Action + observation → LLM input.** This is the prompt seam. |
| 5 | output | basal_ganglia | — | Decision gating |
| 6 | hippocampus | insula | — | Memory → salience |
| 7 | input + attention + output + motor | hippocampus | — | All cortical activity → episodic memory |

The cerebellum's outputs flow back to the cortex through whatever
`Connection { from: vec![CEREBELLUM], … }` edges we add — currently
**none in the default preset**, which is why the LLM is "wired in but
unread" today. The first concrete cerebellum-output edge to add is
likely `cerebellum → attention` (LLM prediction biases what to look at)
or `cerebellum → output` (LLM directly proposes the next action).

---

## 3. Current parameter budget (toy / legacy)

The existing `RegionalConfig::eight_region(obs_dim=512)` preset:

| Region | Neurons | d_model | Params (approx) |
|---|---|---|---|
| input | 512 | 64 | 32K |
| attention | 512 | 64 | 32K |
| output | 512 | 64 | 32K |
| motor | 512 | 64 | 32K |
| **cerebellum** | **64** | **32** | **2K** ← tiny |
| basal_ganglia | 64 | 32 | 2K |
| insula | 64 | 32 | 2K |
| hippocampus | 64 | 64 | 4K |
| **total** | 2304 | — | **~140K** |

**Problem:** the cerebellum is 1.4% of total params. In biology it's
~80% of neurons. In our intended architecture it hosts the LLM, so it
should dominate. The current preset is left over from when "cerebellum"
meant a small frozen transformer; with Qwen2.5-0.5B (and later 7B+)
mounted there, the proportions invert.

---

## 4. Target parameter budget (cerebellum-dominant)

The right shape, given Qwen2.5-0.5B today and the 8 GB VRAM budget:

| Region | Role-driven sizing | Params |
|---|---|---|
| input | Byte/sensory embedder; small Conv/Linear stack | ~10M |
| attention | Light routing MLP + salience matvec | ~10M |
| output | Action decoder; small head | ~10M |
| motor | Action effector | ~10M |
| **cerebellum** | **Qwen2.5-0.5B GptModelResident, frozen or 1/10 LR** | **~494M** |
| basal_ganglia | Gating MLP | ~5M |
| insula | Interoception scalar head | ~5M |
| hippocampus | Episodic KV cache + small recall MLP | ~50M |
| **total** | — | **~600M** |

**Cerebellum share: ~82%.** Matches biological proportion and matches
the "cerebellum is the LLM" architectural commitment.

When we eventually mount a 7B-class LLM via streaming + Q4_K residency
(the foundation-model machinery from yesterday), the share goes to
~98%. Non-cerebellum stays small.

---

## 5. Cerebellum integration: how the LLM mounts

`crates/modgrad-ctm/src/cerebellum.rs:104` already has the right shape:

```rust
pub trait FrozenCerebellum: Send {
    fn encode_context_layers(&mut self, token_ids: &[i64]) -> CerebellumCache;
}
```

`CerebellumCache` is `[n_layers × n_positions × hidden_dim]` — every
intermediate hidden state from the LLM, available for the cortex
regions to attend over. Existing impls: `OnnxCerebellum`,
`FrozenTransformer`, `RandomExpansion`. Adding **`BltCerebellum`** (or
`QwenCerebellum`) is the next slice — it just wraps a
`GptModelResident` / `BltModel` and exposes the layer cache.

**The seam:**

| Cortex side | Cerebellum side |
|---|---|
| motor's action vector | becomes part of the LLM's input prompt |
| observation bytes | the rest of the prompt |
| cortex regions' Q-vectors | cross-attend over `CerebellumCache.hidden_states` |
| LLM's per-layer hidden states | are *the* world-model representation the cortex reads |

The cortex regions never call the LLM forward themselves — the runtime
calls `FrozenCerebellum::encode_context_layers` once per context window
(per the architecture comment in `frozen_transformer.rs`), and the
cortex reads from the cache via `CerebellumCache::blend_layers` (a
learned softmax over which depth of representation to use).

---

## 6. What needs to change to ship this

| Item | Where | Effort |
|---|---|---|
| `RegionalConfig::eight_region_v2` (asymmetric, cerebellum-dominant) | `crates/modgrad-ctm/src/graph.rs` | small (~80 LOC) |
| `BltCerebellum: FrozenCerebellum` adapter | `crates/modgrad-blt/` or `crates/modgrad-ctm/` | small (~150 LOC) |
| `QwenCerebellum: FrozenCerebellum` adapter (frozen-only path; useful for inference-as-cerebellum without BLT scaffolding) | same | smaller (~80 LOC) |
| `Connection` edges from cerebellum back to cortex (e.g., `cerebellum → attention`) | `RegionalConfig::eight_region_v2` | trivial |
| Smoke test: bytes in → BLT cerebellum → cortex → action out | `examples/` or `tests/` | ~150 LOC |

---

## 7. Open questions

1. **Cerebellum mount: special region or sibling service?**
   Two clean shapes:
   - (a) The cerebellum stays a region in `RegionalConfig.regions` but
     its forward/backward is dispatched to a `FrozenCerebellum` trait
     object instead of the regular CTM `forward_cached_resident`.
   - (b) The cerebellum becomes a *sibling* of the regional brain — a
     separate `FrozenCerebellum` instance that the cortex regions read
     from via existing connection topology. The 8-region brain
     becomes 7 cortical/subcortical + 1 frozen LLM service.

   (b) is cleaner separation; (a) keeps the connection graph uniform.
   This file leans toward (b) but the call hasn't been made.

2. **What does `motor → cerebellum` look like as an LLM prompt?**
   Concretely: motor emits an action vector. The cerebellum's input is
   `[observation_bytes, action_token_or_embedding]`. Open: do we
   tokenize the action, embed it as a dense vector concatenated to the
   byte stream, or both?

3. **Backward path through the cerebellum.**
   With Qwen2.5-0.5B as cerebellum, three options:
   - frozen (no backward through cerebellum — cortex learns to use
     fixed representations)
   - low-LR (BLT §6.2 byte-ification recipe — 1/10 LR on cerebellum,
     full LR on cortex + local components)
   - LoRA-only (cerebellum frozen, LoRA adapters trainable — fits 8 GB
     trivially)

   The default for the v2 preset should probably be **LoRA-only** at
   first for the 8 GB hardware reality, with **low-LR** as the upgrade
   path once we have Q4_K residency for the cerebellum and a bf16
   master path.
