# Planning in the brain: folding value iteration into the regions

**Status:** design direction (not yet implemented). Captures the plan to make
the maze planner *part of* the 8-region brain instead of a separate
`VinReadout` module bolted alongside it.

## The problem with the current setup

The `/play` maze demo solves with a **VIN** (`modgrad-ctm::vin::VinReadout`): a
separate learned module that runs explicit value iteration (K Bellman backups
with a highway gate) over the decoded grid and reads the move **ego-centrically**
at the agent's own cell. The agent drives on the VIN's move logits. The 8-region
brain runs *alongside* it (vision/retina, telemetry, the 3D viz) but is **not the
thing choosing the move**.

This is honest in one sense — no solver/BFS in the loop, it's learned, it
generalises to bigger grids — but the planner is *next to* the brain, not *in*
it. The two ingredients that actually make it work are:

1. **ego-centric readout** — read value at the agent's own grid cell, not a
   pooled global vector (pooling destroys local wall info — see
   `BRAIN_ARCHITECTURE.md` / the maze root-cause notes);
2. **local iterative propagation** — value spreads cell→neighbour→neighbour,
   one Bellman sweep at a time, until it reaches the agent from the goal.

Both are *generic computations*. There's no reason they must live in a bespoke
module.

## Value iteration is already what the brain does — distributed

Value iteration is model-based planning: propagate value over a map. Biology
does this messily across three systems we already model as regions:

- **Hippocampus** — the cognitive map (place cells; O'Keefe / Tolman). Entorhinal
  **grid cells** give it a metric.
- **Hippocampal replay / preplay** — sequential reactivation of place-cell
  trajectories during pauses and sleep: literally rolling out paths and
  propagating value backward from the goal. The biological analog of Bellman
  backups. (Our "sleep consolidation" already gestures at this.)
- **Basal ganglia + dopamine** — the value/reward machinery; dopamine = reward
  prediction error (TD learning).

So "value iteration" is a tidy abstraction of **hippocampal replay over a
cognitive map, scored by the striatal value system.**

## The design: value iteration *as region dynamics*

Each VIN ingredient already has a home in the 8 regions:

| VIN ingredient | Becomes |
|---|---|
| per-cell value **map** | **hippocampus** = a *spatial* region (`spatial: Some((n_tok, raw_dim))`), one token per grid cell — a place-cell cognitive map |
| **local 4-neighbour** Bellman backup, iterated | the hippocampus region's **recurrent tick update**, restricted to grid-adjacent cells: one tick ≈ one value-iteration sweep |
| per-cell **reward / traversability** | **basal-ganglia** region projecting a reward/value scalar into the map |
| **ego-centric** move readout | **motor** region gathers the hippocampus map **at the agent's own cell** → move logits |

Flow: basal-ganglia supplies reward → hippocampus propagates value locally over
its ticks → motor reads ego-centrically → move. No external module.

## What already supports it

- Regions can be **spatial** (per-cell tokens) today.
- The CTM's **ticks are the iterations**, and **adaptive compute** can learn to
  run *more* ticks on bigger mazes — which is exactly how value iteration should
  scale with grid diameter.
- **BPTT** through the recurrence exists (`forward_train` / `backward`), so it is
  trainable end-to-end.

## The hard parts (be honest)

1. **Local vs global attention.** CTM regions attend *globally*; value iteration
   needs *grid-local* (4-neighbour) propagation. Give the hippocampus region a
   grid-adjacency-structured update — porting the VIN's neighbour offsets in as
   an architectural prior.
2. **Tick budget.** Propagation needs ~grid-diameter sweeps (≈20 for 11×11);
   outer-ticks are few. The hippocampus needs enough inner iterations, or
   adaptive ticks that scale with grid size.
3. **Learned-vs-explicit risk.** The explicit Bellman structure is a *strong*
   prior; a free-form region may not rediscover it and could underperform.

## Pragmatic path

Don't learn it from scratch. **Warm-start the hippocampus region from the
explicit VIN** — initialise its update toward a Bellman backup, or distill the
trained `VinReadout` into it — then fine-tune end-to-end so the planner *becomes*
part of the brain while keeping the VIN's competence. This is the
AlphaGo→AlphaZero arc: imitate the solver first, then absorb it into the learned
system. The standalone `VinReadout` stays as the **ablation/baseline** to measure
against.

## Eval protocol

- Train on 9×9, test zero-shot on 11×11 / 13×13 / larger — the planner must keep
  generalising (the whole point of the ego-centric + iterative design).
- Probe the hippocampus map: is local wall info linearly decodable at the agent
  cell (the metric that collapses under pooling)?
- Compare solve-rate / agreement / wall-hit-rate vs the standalone VIN baseline.
