# isis

**intermediate system to intermediate system**

gradient-free neuroplasticity for language models. a 4-region continuous thought machine that learns through hebbian plasticity and sleep consolidation — no backpropagation.

## what it does

isis bolts onto a frozen language model (or grows from scratch) and adds:

- **episodic memory** — teach facts, recall them via cosine key matching
- **deliberation** — 4-region ctm thinks across multiple ticks before responding
- **sleep consolidation** — least-squares weight optimization during nrem, emotional processing during rem
- **plural identity** — per-alter ctm weights, amnesia barriers, asymmetric memory access
- **self-monitoring** — homeostasis tracks sleep pressure from native neural signals, not timers

## architecture

```
frozen backbone (onnx/gguf)
  → hidden state
    → ctm v2 (4 regions, K ticks, hebbian plasticity)
      → input region  (v1/s1, fast, wide)
      → attention region (thalamus, gating, no broadcast)
      → output region (association cortex, evidence accumulation)
      → motor region  (m1/bg, drift-diffusion threshold)
    → sync convergence → confidence signal
    → episodic memory (cosine recall, valence, reconsolidation)
    → logit injection → output token
```

or without a backbone:

```
raw bytes
  → learned embeddings (dna: randomly initialized)
  → sensory layer (trained via least-squares from parent hidden states)
  → ctm v2 (same architecture)
  → output projection → next byte
```

## two modes

### backbone mode

attach isis to any frozen language model. the backbone handles perception and language. isis handles memory, deliberation, and identity.

```bash
# teach facts through the backbone
isis chat models isis.json

# full pipeline test: backbone → ctm → memory → sleep → recall
isis e2e models
```

### tabula rasa mode

no backbone. the organism starts blank and learns from experience. like a newborn — right architecture, zero knowledge.

```bash
# develop from raw text (self-supervised)
isis develop train_data.txt organism.bin

# develop with parent guidance (backbone teaches the child)
isis parent models child.bin

# learn from precomputed parent hidden states (fast)
isis learn parent_states.bin child.bin

# probe what it understands
isis probe child.bin
```

## brain regions

| region | neurons | memory | ticks | role |
|--------|---------|--------|-------|------|
| input (v1/s1) | 31% | short | few | perceive, fast feature detection |
| attention (thalamus) | 27% | long | many | route, gate, no broadcast |
| output (assoc cortex) | 27% | long | many | accumulate evidence, understand |
| motor (m1/bg) | 14% | short | few | decide, drift-diffusion threshold |

20% of neurons per region are inhibitory (gaba), creating competition and winner-take-all dynamics.

## sleep

two phases, like real sleep:

**nrem (slow-wave):**
- least-squares synapse weight optimization
- sensory layer consolidation from parent traces
- graduation testing: episodic → semantic
- logit projector training

**rem (dreaming):**
- high-surprise experience replay
- subconscious fear memory processing (auto-tuned plasticity)
- avoidance pattern pruning (hate prevention)
- emotional health diagnosis

sleep is not on a timer. it's driven by native pressure signals:
- activation energy (how hard neurons fired)
- sync divergence (can't reach decisions)
- hebbian drift (outputs getting noisy)
- buffer overflow (consolidation overdue)
- unprocessed emotions (rem needed)

the organism observes its own pressure and can choose to sleep or push through. past threshold, sleep is forced.

## memory lifecycle

```
teach → episodic snapshot (isis.json/isis.fb)
  → recall via cosine match + ctm confidence gating
  → sleep → least-squares consolidation into ctm weights
  → graduation: consolidation_score crosses threshold
  → episodic injection fades, ctm produces from own weights
  → eventual pruning: knowledge lives only in ctm
```

fear memories resist consolidation (slow decay, high threshold). the autonomic system processes them during rem with elevated plasticity — the psilocybin analogue. valence shifts gradually: fear → negative → neutral.

## emotional health

the autonomic system monitors:
- **ptsd risk**: fear memories > 30% of total
- **depressive risk**: negative+fear > 50%
- **hate risk**: avoidance patterns over-generalizing (cosine > 0.7)
- **hypervigilance**: too many active avoidances

`safe_to_deploy()` returns false if any diagnosis is present.

## neuromodulators

| signal | source | modulates | range |
|--------|--------|-----------|-------|
| dopamine | surprise (prediction error) | sync accumulation | [0.5, 1.0] |
| serotonin | energy × novelty | consolidation priority | [0.1, 1.0] |
| norepinephrine | explicit importance | memory strength | [0.0, 3.0] |

## formats

| format | extension | use case |
|--------|-----------|----------|
| json | .json | human-readable memory banks |
| flatbuffers | .fb | production (f32/f16/i8 quantized keys) |
| binary | .bin | organism checkpoints (named weight sections) |
| prometheus | .prom | metrics scraping |
| jsonl | .jsonl | time series telemetry |

memory banks are model-specific. the `model_id` field tracks exactly which backbone produced the keys:
```json
{
  "model_id": {
    "model": "Qwen/Qwen2.5-0.5B",
    "backend": "onnx",
    "quant": "f32",
    "hidden_dim": 896,
    "extraction": "pre_mlp_layer23",
    "eos_token_id": 151643
  }
}
```

## parent-child teaching

the backbone is the parent. the organism is the child. the parent demonstrates understanding, the child learns to match.

```
parent processes "the cat sat" → hidden state (meaning)
child processes same text → its own representation (initially random)
teaching signal = cosine similarity between parent and child
hebbian update: blend child embedding toward parent representation
sleep: least-squares sensory layer training against parent targets
```

angeris bound diagnostic: 78% of parent representation is linearly recoverable from child embeddings after training. the remaining 22% is nonlinear/contextual — handled by the ctm's multi-tick deliberation.

## performance

- avx-512 optimized dot product (16-wide unroll)
- rayon parallel superlinear (per-neuron mlp) for large configs
- vulkan compute shaders for gpu dispatch (superlinear + matvec)
- parallel least-squares solver (row-parallel xtx/xty + column-parallel cholesky solve)
- adaptive flop threshold: gpu → rayon → sequential fallback
- binary organism format: ~5x smaller than json, ~100x faster save/load

## building

```bash
cargo build --release                    # cpu only
cargo build --release --features gpu     # with vulkan compute
cargo test                               # 61 tests
```

requires:
- rust 2024 edition
- onnxruntime (for backbone mode)
- libvulkan.so (for gpu feature, optional)
- glslangvalidator (to recompile shaders, optional)

## dna configs

| config | neurons | params | context | ticks | use |
|--------|---------|--------|---------|-------|-----|
| tiny | 64 | 74k | 64 | 8 | testing |
| small | 256 | 705k | 128 | 4 | fast experiments |
| medium | 256 | ~1m | 64 | 16 | deep thinking |
| large | 4096 | ~137m | 256 | 12 | language learning |
| child_of(n) | 256 | varies | 128 | 8 | matched to parent dim |

## references

- sakana ai ctm paper (arxiv 2505.05522)
- angeris (2022) "a note on generalizing power bounds for physical design"
- eriksen (2013) "your server as a function"
- complementary learning systems (mcclelland et al.)
- nader (2003) "memory traces unbound" (reconsolidation)
- reinders et al. (2006) did fmri studies (plural systems)

## license

mit
