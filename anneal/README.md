# Anneal

**A crowd slowly annealing a shared model out of noise.**

Anneal is a decentralized, verifiable training network: contributors train
individual model blocks on their own hardware (WebGPU, in-browser) and earn when
their block *verifiably* improves the shared model. `anneal.network`.

## Why it can exist at all

Every prior "SETI@home for LLMs" dies on the same rock — gradient all-reduce needs
every node talking to every other node, every step, over the internet. Anneal
sidesteps it entirely by training with **DiffusionBlocks** (Shing et al., ICLR
2026), where the network is partitioned into blocks that each denoise one noise
range. Two structural facts fall out:

1. **Zero inter-block communication.** Blocks are independent — a volunteer
   downloads {shared embedding + one block + its noise range + data}, trains
   locally, uploads the block. Nodes never talk to each other. *This* is what
   makes disconnected, heterogeneous, even temporally-spread contribution possible.
2. **Per-block verifiability.** Each block has an objective, cheap quality signal —
   its denoising loss on a held-out shard. So "useful work" is directly checkable,
   which is exactly what a fraud-resistant reward market needs.

## The stack

| Layer | Tool | Role |
|---|---|---|
| Trainable decomposition | **DiffusionBlocks** | independent, communication-free blocks (the enabler) |
| Off-chain training | **WebGPU / WASM** | contributors train a block in the browser (Rust → `wgpu`) |
| Data availability | **ZODA** | block weights + val shards available + correctly encoded, ~zero overhead |
| Authenticated state | **NOMT** | canonical registry: `block-id → {weight-commitment, val-loss, owner, stake}` |
| Succinct outcome proofs | **Ligerito** | proof of "block X → loss L on shard S"; cheap on-chain verify |
| Secured compute + economics | **JAM** (ELVES + stake/slash) | runs/verifies the outcome check, accepts blocks, pays DOT |

Anti-gaming is **economic, not cryptographic**: a submission posts a stake;
acceptance requires beating the current best by a margin on a **fresh secret shard**
(selected by JAM randomness, so it can't be overfit); failed submissions are
slashed. Garbage, replay, plagiarism, and val-grinding all become −EV. We verify
the **outcome**, not the training process — proving the process is both
impractical (proving cost dwarfs the training) and unnecessary.

## Build order — core first, on purpose

The entire architecture above is *infrastructure*. The one assumption holding it up
is **not** a crypto question and no JAM/ZODA/NOMT/Ligerito component can answer it:

> **Does val-gated async block training actually converge to a good model?**

So the code grows from that question outward, cheapest-decisive-experiment first:

1. **Slice 1 — noise primitives.** ✅ EDM schedule + equi-prob partitioning, in
   `modgrad-transformer::diffusion`.
2. **Slice 2 — the trainer.** AdaLN noise conditioning on the residual block +
   the block-wise denoising trainer. Target the masked-diffusion LM variant → a
   real **bits-per-char**, compared against the existing 3.88 bits/byte baseline.
   *Proves the method works at all.* (Lives in `modgrad-transformer`.)
3. **Slice 3 — async-volunteer simulation.** Train blocks independently/async with
   the val-gated accept rule, in-process (no browser, no chain), and check the
   assembled model matches end-to-end. *Proves the distributed dynamics converge.*
   (Lives here, in `anneal`.)
4. **Then, and only then** — WebGPU client, and the JAM/ZODA/NOMT/Ligerito network.

If slices 2–3 don't converge, the network is moot and we've saved ourselves
building a beautiful vault around an empty room.

## Layout

- ML engine (DiffusionBlocks primitives + trainer) → `../crates/modgrad-transformer`
- Distributed/verification layer (this crate) → `anneal/`
- Extract to its own repo once the network layer outgrows the workspace.
