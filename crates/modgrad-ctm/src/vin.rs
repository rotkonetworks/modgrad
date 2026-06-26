//! Value-Iteration-Network (VIN) readout over a 2D grid of spatial tokens.
//!
//! Reference: Tamar et al., "Value Iteration Networks", arXiv:1602.02867.
//!
//! ## Why this exists
//!
//! The maze brain's legacy readout reads the move from a POOLED global-sync
//! vector. Pooling destroys local position: the wall signal is ~94%
//! decodable in the agent's-own-cell V4 token but collapses to ~60% (near
//! chance) once sum-pooled. The fix (deep-research, adversarially verified):
//! propagate value *convolutionally* across the grid and read the decision
//! at the AGENT'S OWN cell — ego-centric, never pooled.
//!
//! ## What it does (forward only)
//!
//! Input: a flat `[n_cells × raw_dim]` buffer of per-cell V4 spatial tokens
//! on a `grid_h × grid_w` grid (row-major).
//!
//! 1. Per-cell projection of the token → (local reward `r`, a
//!    traversability gate `t ∈ [0,1]`, and an initial value-feature
//!    vector). The gate is the learned "wall mask from the input": a cell
//!    with `t ≈ 0` is a wall and blocks value flowing *into* it.
//! 2. K capped rounds of a 3×3 "Bellman backup": each cell aggregates a
//!    (soft)max over its 4 neighbours' values, masked by the neighbour's
//!    traversability gate, plus its own local reward. An optional
//!    Highway-style gate `h = g·update + (1-g)·prev` stabilises deep
//!    propagation (Highway-VIN, arXiv:2406.03485). K is bounded/modest
//!    (8–20): "more iterations always helps" was REFUTED.
//! 3. Locate the agent cell (max along a learned "agent-ness" projection,
//!    or an explicit `(row,col)`), gather its post-propagation state and
//!    its 4 neighbours' values, and emit `n_dirs` move logits from THAT
//!    local readout — no global pooling.
//!
//! ## Scope
//!
//! FORWARD ONLY. This is built to make the wall signal decodable at the
//! agent cell for the wall-probe. Training/backward of the VIN readout is
//! DEFERRED (noted at `VinReadout::forward`). It is strongly typed,
//! self-contained, and does not touch the existing `RegionalWeights`
//! backward path.

use modgrad_compute::neuron::Linear;
use serde::{Deserialize, Serialize};
use wincode_derive::{SchemaRead, SchemaWrite};

/// Neighbour offsets in canonical direction order: Up, Down, Left, Right.
/// Matches the wall-probe's `[U, D, L, R]` convention.
pub const DIR_OFFSETS: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

/// Configuration for the VIN readout. All dims are small by design.
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub struct VinConfig {
    /// Per-cell value-feature width propagated across the grid.
    pub value_dim: usize,
    /// Number of value-iteration (Bellman backup) rounds. Bounded/modest
    /// (8–20). Capped at `max_iters` regardless of request.
    pub iters: usize,
    /// Hard cap on `iters` — deep propagation collapses on long paths.
    pub max_iters: usize,
    /// Aggregation softness for the neighbour backup. `0.0` → hard max
    /// (argmax over neighbours); `>0` → softmax with this temperature
    /// (smoother, differentiable). Modest temperatures keep it max-like.
    pub softmax_temp: f32,
    /// Use a Highway-style gated update for stability across iterations.
    pub highway_gate: bool,
    /// Number of output move directions (logits) the head emits.
    pub n_dirs: usize,
}

impl Default for VinConfig {
    fn default() -> Self {
        Self {
            value_dim: 16,
            iters: 10,
            max_iters: 20,
            softmax_temp: 0.5,
            highway_gate: true,
            n_dirs: 4,
        }
    }
}

impl VinConfig {
    /// Effective iteration count after the cap.
    #[inline]
    pub fn effective_iters(&self) -> usize {
        self.iters.min(self.max_iters.max(1))
    }
}

/// Self-contained VIN readout weights. Independent of `RegionalWeights`;
/// owns its own small linear projections.
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub struct VinReadout {
    pub config: VinConfig,
    /// Per-cell token width this readout consumes.
    pub raw_dim: usize,

    /// token → local reward scalar. `Linear(raw_dim → 1)`.
    pub reward_proj: Linear,
    /// token → traversability logit (sigmoid → gate ∈ [0,1]). `Linear(raw_dim → 1)`.
    pub gate_proj: Linear,
    /// token → initial value features. `Linear(raw_dim → value_dim)`.
    pub value_proj: Linear,
    /// token → "agent-ness" logit, used to localise the agent cell when no
    /// explicit cell is supplied. `Linear(raw_dim → 1)`.
    pub agent_proj: Linear,

    /// Per-iteration Highway update gate from `[prev_value | candidate]`.
    /// `Linear(2*value_dim → value_dim)`. Unused if `highway_gate=false`
    /// (still allocated for a stable serialised shape).
    pub highway_proj: Linear,

    /// Final ego-centric move head. Consumes the agent cell's
    /// post-propagation readout (see `agent_cell_readout_dim`).
    /// `Linear(readout_dim → n_dirs)`.
    pub move_head: Linear,
}

impl VinReadout {
    /// Build a VIN readout for `raw_dim` per-cell tokens.
    pub fn new(config: VinConfig, raw_dim: usize) -> Self {
        let v = config.value_dim.max(1);
        let readout_dim = Self::agent_cell_readout_dim_for(v);
        Self {
            reward_proj: Linear::new(raw_dim, 1),
            gate_proj: Linear::new(raw_dim, 1),
            value_proj: Linear::new(raw_dim, v),
            agent_proj: Linear::new(raw_dim, 1),
            highway_proj: Linear::new(2 * v, v),
            move_head: Linear::new(readout_dim, config.n_dirs),
            config,
            raw_dim,
        }
    }

    /// Dimension of the agent-cell readout fed to the move head:
    /// the agent cell's own value vector (`value_dim`) concatenated with
    /// each of its 4 neighbours' value vectors (`4 * value_dim`).
    /// Off-grid / wall neighbours contribute a zero block.
    #[inline]
    pub fn agent_cell_readout_dim(&self) -> usize {
        Self::agent_cell_readout_dim_for(self.config.value_dim.max(1))
    }

    #[inline]
    fn agent_cell_readout_dim_for(value_dim: usize) -> usize {
        // agent cell's own PRE-propagation value (the clean per-cell signal)
        // ++ agent cell's POST-propagation value ++ 4 neighbour post-values.
        value_dim * 6
    }

    /// FORWARD ONLY. Backward/training of the VIN readout is DEFERRED — the
    /// wall-probe needs only the forward path. Run the value propagation and
    /// produce the ego-centric output.
    ///
    /// - `tokens`: flat `[n_cells × raw_dim]`, `n_cells == grid_h * grid_w`,
    ///   row-major.
    /// - `agent_cell`: explicit `(row, col)`; if `None`, located as the
    ///   argmax of the learned agent-ness projection.
    ///
    /// Returns the populated [`VinOutput`].
    pub fn forward(
        &self,
        tokens: &[f32],
        grid_h: usize,
        grid_w: usize,
        agent_cell: Option<(usize, usize)>,
    ) -> VinOutput {
        let n_cells = grid_h * grid_w;
        let v = self.config.value_dim.max(1);
        debug_assert_eq!(
            tokens.len(),
            n_cells * self.raw_dim,
            "VinReadout::forward: token buffer size mismatch"
        );

        // ── 1. Per-cell projections: reward, traversability gate, value ──
        let mut reward = vec![0.0f32; n_cells];
        let mut gate = vec![0.0f32; n_cells]; // traversability ∈ [0,1]
        let mut value = vec![0.0f32; n_cells * v]; // [n_cells × value_dim]
        let mut agent_score = vec![0.0f32; n_cells];

        for cell in 0..n_cells {
            let tok = &tokens[cell * self.raw_dim..(cell + 1) * self.raw_dim];
            reward[cell] = self.reward_proj.forward(tok)[0];
            gate[cell] = sigmoid(self.gate_proj.forward(tok)[0]);
            agent_score[cell] = self.agent_proj.forward(tok)[0];
            let vproj = self.value_proj.forward(tok);
            value[cell * v..(cell + 1) * v].copy_from_slice(&vproj);
        }

        // Keep the clean PRE-propagation per-cell value (the per-cell signal
        // that decodes walls at the agent cell); propagation mixes neighbour
        // values in and dilutes it.
        let value_init = value.clone();

        // ── 2. K capped Bellman backups over the 4-neighbour grid ────────
        let iters = self.config.effective_iters();
        let mut next = value.clone();
        for _ in 0..iters {
            for r in 0..grid_h {
                for c in 0..grid_w {
                    let cell = r * grid_w + c;
                    // Candidate value = local reward (broadcast) + (soft)max
                    // over traversable neighbours' value vectors.
                    let cand = self.backup_cell(
                        r, c, grid_h, grid_w, &value, &gate, reward[cell], v,
                    );
                    let dst = &mut next[cell * v..(cell + 1) * v];
                    if self.config.highway_gate {
                        // h = g·cand + (1-g)·prev, g from [prev | cand].
                        let prev = &value[cell * v..(cell + 1) * v];
                        let mut hin = Vec::with_capacity(2 * v);
                        hin.extend_from_slice(prev);
                        hin.extend_from_slice(&cand);
                        let g = self.highway_proj.forward(&hin);
                        for k in 0..v {
                            let gk = sigmoid(g[k]);
                            dst[k] = gk * cand[k] + (1.0 - gk) * prev[k];
                        }
                    } else {
                        dst.copy_from_slice(&cand);
                    }
                }
            }
            std::mem::swap(&mut value, &mut next);
        }
        // `value` now holds the post-propagation per-cell value grid.

        // ── 3. Locate the agent cell ─────────────────────────────────────
        let (ar, ac) = agent_cell.unwrap_or_else(|| {
            let mut best = 0usize;
            let mut best_s = f32::NEG_INFINITY;
            for cell in 0..n_cells {
                if agent_score[cell] > best_s {
                    best_s = agent_score[cell];
                    best = cell;
                }
            }
            (best / grid_w, best % grid_w)
        });
        let agent_cell_idx = ar * grid_w + ac;

        // Ego-centric readout: agent cell PRE-value ++ agent cell POST-value
        // ++ its 4 neighbours' POST-values (zero block for off-grid /
        // fully-blocked neighbours).
        let readout = self.gather_agent_readout(
            ar, ac, grid_h, grid_w, &value, &value_init, &gate, v,
        );
        let move_logits = self.move_head.forward(&readout);

        VinOutput {
            move_logits,
            agent_readout: readout,
            agent_value: value[agent_cell_idx * v..(agent_cell_idx + 1) * v].to_vec(),
            agent_cell: (ar, ac),
            value_grid: value,
            gate,
            reward,
            grid_h,
            grid_w,
            value_dim: v,
        }
    }

    /// One cell's Bellman backup: `reward + (soft)max_neighbour gate·value`.
    /// Returns a `value_dim` vector.
    fn backup_cell(
        &self,
        r: usize,
        c: usize,
        grid_h: usize,
        grid_w: usize,
        value: &[f32],
        gate: &[f32],
        local_reward: f32,
        v: usize,
    ) -> Vec<f32> {
        // Collect traversable neighbours and a scalar "preference" per
        // neighbour for the (soft)max weighting: the neighbour's gate times
        // the L1 magnitude of its value vector (a cheap scalar summary).
        let mut nbr_idx: Vec<usize> = Vec::with_capacity(4);
        let mut nbr_pref: Vec<f32> = Vec::with_capacity(4);
        for (dr, dc) in DIR_OFFSETS {
            let nr = r as i32 + dr;
            let nc = c as i32 + dc;
            if nr < 0 || nc < 0 || nr >= grid_h as i32 || nc >= grid_w as i32 {
                continue; // off-grid acts as a wall (blocks propagation)
            }
            let ncell = nr as usize * grid_w + nc as usize;
            let g = gate[ncell];
            let mag: f32 = value[ncell * v..(ncell + 1) * v]
                .iter()
                .map(|x| x.abs())
                .sum::<f32>();
            nbr_idx.push(ncell);
            nbr_pref.push(g * mag); // wall (g≈0) → near-zero preference
        }

        let mut out = vec![0.0f32; v];
        if nbr_idx.is_empty() {
            for k in 0..v {
                out[k] = local_reward;
            }
            return out;
        }

        if self.config.softmax_temp <= 0.0 {
            // Hard max: pick the single best neighbour, gate-scaled.
            let mut best = 0usize;
            let mut best_p = f32::NEG_INFINITY;
            for (i, &p) in nbr_pref.iter().enumerate() {
                if p > best_p {
                    best_p = p;
                    best = i;
                }
            }
            let ncell = nbr_idx[best];
            let g = gate[ncell];
            for k in 0..v {
                out[k] = local_reward + g * value[ncell * v + k];
            }
        } else {
            // Softmax over neighbour preferences, then gate-weighted blend.
            let t = self.config.softmax_temp;
            let maxp = nbr_pref.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut wsum = 0.0f32;
            let mut w = vec![0.0f32; nbr_idx.len()];
            for (i, &p) in nbr_pref.iter().enumerate() {
                let e = ((p - maxp) / t).exp();
                w[i] = e;
                wsum += e;
            }
            let inv = 1.0 / wsum.max(1e-8);
            for k in 0..v {
                let mut acc = 0.0f32;
                for (i, &ncell) in nbr_idx.iter().enumerate() {
                    let g = gate[ncell];
                    acc += w[i] * inv * g * value[ncell * v + k];
                }
                out[k] = local_reward + acc;
            }
        }
        out
    }

    /// Gather the agent cell's value vector plus its 4 neighbours' values
    /// (each gate-scaled; zero block for off-grid neighbours), in
    /// `[U, D, L, R]` order. Length == `agent_cell_readout_dim`.
    #[allow(clippy::too_many_arguments)]
    fn gather_agent_readout(
        &self,
        ar: usize,
        ac: usize,
        grid_h: usize,
        grid_w: usize,
        value: &[f32],
        value_init: &[f32],
        gate: &[f32],
        v: usize,
    ) -> Vec<f32> {
        let mut out = Vec::with_capacity(v * 6);
        let acell = ar * grid_w + ac;
        // Agent cell's clean pre-propagation value, then its post value.
        out.extend_from_slice(&value_init[acell * v..(acell + 1) * v]);
        out.extend_from_slice(&value[acell * v..(acell + 1) * v]);
        for (dr, dc) in DIR_OFFSETS {
            let nr = ar as i32 + dr;
            let nc = ac as i32 + dc;
            if nr < 0 || nc < 0 || nr >= grid_h as i32 || nc >= grid_w as i32 {
                out.extend(std::iter::repeat(0.0f32).take(v));
                continue;
            }
            let ncell = nr as usize * grid_w + nc as usize;
            let g = gate[ncell];
            for k in 0..v {
                out.push(g * value[ncell * v + k]);
            }
        }
        out
    }

    /// Parameter count (for telemetry).
    pub fn num_params(&self) -> usize {
        let p = |l: &Linear| l.weight.len() + l.bias.len();
        p(&self.reward_proj)
            + p(&self.gate_proj)
            + p(&self.value_proj)
            + p(&self.agent_proj)
            + p(&self.highway_proj)
            + p(&self.move_head)
    }
}

/// Output of one VIN readout forward pass.
pub struct VinOutput {
    /// Ego-centric move logits `[n_dirs]` from the agent cell.
    pub move_logits: Vec<f32>,
    /// The exact ego-centric readout fed to the move head — the
    /// representation the wall-probe decodes. Length `value_dim * 5`.
    pub agent_readout: Vec<f32>,
    /// The agent cell's own post-propagation value vector `[value_dim]`.
    pub agent_value: Vec<f32>,
    /// The (row, col) of the agent cell used (explicit or argmax-located).
    pub agent_cell: (usize, usize),
    /// Full post-propagation per-cell value grid `[n_cells × value_dim]`.
    pub value_grid: Vec<f32>,
    /// Per-cell traversability gate `[n_cells]` (≈1 free, ≈0 wall).
    pub gate: Vec<f32>,
    /// Per-cell local reward `[n_cells]`.
    pub reward: Vec<f32>,
    pub grid_h: usize,
    pub grid_w: usize,
    pub value_dim: usize,
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ════════════════════════════════════════════════════════════════════════
//  Backward / training of the VIN readout.
//
//  The forward path above (`VinReadout::forward`) is UNCHANGED — the
//  wall-probe depends on it. Here we add a cached forward (`forward_train`)
//  that records every intermediate, and a `backward` that backpropagates
//  through the K Jacobi value-iteration sweeps (BPTT) plus the highway gate,
//  the soft-max neighbour backup, and the per-cell projections.
//
//  Only the softmax_temp > 0 path is differentiable here; the hard-max path
//  (softmax_temp <= 0) is non-differentiable and NOT supported in backward.
//  The agent cell is always supplied explicitly during training, so
//  `agent_proj` carries no gradient (left zero).
// ════════════════════════════════════════════════════════════════════════

/// Accumulated gradient for one [`Linear`]: same shapes as `weight`/`bias`.
#[derive(Debug, Clone)]
pub struct LinearGrad {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

impl LinearGrad {
    fn zeros(l: &Linear) -> Self {
        Self {
            weight: vec![0.0f32; l.weight.len()],
            bias: vec![0.0f32; l.bias.len()],
        }
    }

    fn add(&mut self, other: &Self) {
        for (a, b) in self.weight.iter_mut().zip(&other.weight) {
            *a += *b;
        }
        for (a, b) in self.bias.iter_mut().zip(&other.bias) {
            *a += *b;
        }
    }

    /// Accumulate the gradient of one `y = W x + b` application.
    /// `dy` is `[out_dim]`, `x` is `[in_dim]`. Optionally back-propagates
    /// into `dx` (`[in_dim]`, accumulated).
    #[inline]
    fn accum(&mut self, l: &Linear, x: &[f32], dy: &[f32], dx: Option<&mut [f32]>) {
        let (out_dim, in_dim) = (l.out_dim, l.in_dim);
        for o in 0..out_dim {
            self.bias[o] += dy[o];
            let row = o * in_dim;
            for i in 0..in_dim {
                self.weight[row + i] += dy[o] * x[i];
            }
        }
        if let Some(dx) = dx {
            for i in 0..in_dim {
                let mut acc = 0.0f32;
                for o in 0..out_dim {
                    acc += dy[o] * l.weight[o * in_dim + i];
                }
                dx[i] += acc;
            }
        }
    }
}

/// Gradients for all trainable Linears of a [`VinReadout`].
/// `agent_proj` is intentionally absent — it never gets gradient in training.
#[derive(Debug, Clone)]
pub struct VinGradients {
    pub reward_proj: LinearGrad,
    pub gate_proj: LinearGrad,
    pub value_proj: LinearGrad,
    pub highway_proj: LinearGrad,
    pub move_head: LinearGrad,
}

impl VinGradients {
    pub fn zeros(vin: &VinReadout) -> Self {
        Self {
            reward_proj: LinearGrad::zeros(&vin.reward_proj),
            gate_proj: LinearGrad::zeros(&vin.gate_proj),
            value_proj: LinearGrad::zeros(&vin.value_proj),
            highway_proj: LinearGrad::zeros(&vin.highway_proj),
            move_head: LinearGrad::zeros(&vin.move_head),
        }
    }

    pub fn add(&mut self, other: &Self) {
        self.reward_proj.add(&other.reward_proj);
        self.gate_proj.add(&other.gate_proj);
        self.value_proj.add(&other.value_proj);
        self.highway_proj.add(&other.highway_proj);
        self.move_head.add(&other.move_head);
    }
}

/// Per-(sweep, cell) backup intermediates needed for backward.
#[derive(Debug, Clone)]
#[allow(dead_code)] // some fields are recorded for completeness / debugging
struct BackupRecord {
    /// In-grid neighbour cell indices (in DIR_OFFSETS scan order).
    nbr_idx: Vec<usize>,
    /// Soft-max weights `w_i * inv` (sum to 1 over neighbours).
    sw: Vec<f32>,
    /// Soft-max preferences `pref_i = gate[ncell] * mag_i` (for backward).
    pref: Vec<f32>,
    /// L1 magnitude `mag_i = sum_k |value[ncell*v+k]|` at sweep start.
    mag: Vec<f32>,
    /// Candidate value vector (pre-highway) `[v]`.
    cand: Vec<f32>,
    /// Previous value vector (this cell, start of sweep) `[v]`.
    prev: Vec<f32>,
    /// Highway gate `g[k]` (post-sigmoid) `[v]` — empty if no highway.
    hg: Vec<f32>,
}

/// Everything the backward pass needs from a cached forward. Holds borrowed
/// nothing — all owned, so it can outlive the call.
#[allow(dead_code)] // gate_logit/reward/value_init recorded for completeness
pub struct VinCache {
    grid_h: usize,
    grid_w: usize,
    n_cells: usize,
    v: usize,
    iters: usize,
    agent_cell: (usize, usize),
    /// Per-cell gate logits (pre-sigmoid) `[n_cells]`.
    gate_logit: Vec<f32>,
    /// Per-cell gate values (post-sigmoid) `[n_cells]`.
    gate: Vec<f32>,
    /// Per-cell local reward `[n_cells]`.
    reward: Vec<f32>,
    /// Clean pre-propagation value grid `[n_cells*v]`.
    value_init: Vec<f32>,
    /// Value grid at the START of every sweep, plus the final grid:
    /// `grids[0]` = value_init (sweep-0 input), `grids[K]` = final value.
    grids: Vec<Vec<f32>>,
    /// Backup records, indexed `[sweep][cell]`.
    backups: Vec<Vec<BackupRecord>>,
    /// The exact readout fed to the move head `[6v]`.
    readout: Vec<f32>,
}

impl VinReadout {
    /// Cached forward for training. Mirrors `forward` exactly (softmax path,
    /// explicit agent cell) but records every intermediate into a
    /// [`VinCache`] so `backward` recomputes nothing.
    ///
    /// Requires `softmax_temp > 0` (the differentiable path). Panics in debug
    /// if the hard-max path is configured.
    pub fn forward_train(
        &self,
        tokens: &[f32],
        grid_h: usize,
        grid_w: usize,
        agent_cell: (usize, usize),
    ) -> (VinOutput, VinCache) {
        let n_cells = grid_h * grid_w;
        let v = self.config.value_dim.max(1);
        let t = self.config.softmax_temp;
        debug_assert!(t > 0.0, "forward_train requires softmax_temp > 0");
        debug_assert_eq!(tokens.len(), n_cells * self.raw_dim);

        let mut reward = vec![0.0f32; n_cells];
        let mut gate = vec![0.0f32; n_cells];
        let mut gate_logit = vec![0.0f32; n_cells];
        let mut value = vec![0.0f32; n_cells * v];

        for cell in 0..n_cells {
            let tok = &tokens[cell * self.raw_dim..(cell + 1) * self.raw_dim];
            reward[cell] = self.reward_proj.forward(tok)[0];
            let gl = self.gate_proj.forward(tok)[0];
            gate_logit[cell] = gl;
            gate[cell] = sigmoid(gl);
            let vproj = self.value_proj.forward(tok);
            value[cell * v..(cell + 1) * v].copy_from_slice(&vproj);
        }

        let value_init = value.clone();
        let iters = self.config.effective_iters();

        let mut grids: Vec<Vec<f32>> = Vec::with_capacity(iters + 1);
        let mut backups: Vec<Vec<BackupRecord>> = Vec::with_capacity(iters);
        grids.push(value.clone()); // grids[0] = sweep-0 input

        let mut next = value.clone();
        for _ in 0..iters {
            let mut recs: Vec<BackupRecord> = Vec::with_capacity(n_cells);
            for r in 0..grid_h {
                for c in 0..grid_w {
                    let cell = r * grid_w + c;
                    let (cand, rec) = self.backup_cell_cached(
                        r, c, grid_h, grid_w, &value, &gate, reward[cell], v, t,
                    );
                    let dst = &mut next[cell * v..(cell + 1) * v];
                    let mut rec = rec;
                    rec.cand = cand.clone();
                    rec.prev = value[cell * v..(cell + 1) * v].to_vec();
                    if self.config.highway_gate {
                        let prev = &value[cell * v..(cell + 1) * v];
                        let mut hin = Vec::with_capacity(2 * v);
                        hin.extend_from_slice(prev);
                        hin.extend_from_slice(&cand);
                        let g = self.highway_proj.forward(&hin);
                        let mut hg = vec![0.0f32; v];
                        for k in 0..v {
                            let gk = sigmoid(g[k]);
                            hg[k] = gk;
                            dst[k] = gk * cand[k] + (1.0 - gk) * prev[k];
                        }
                        rec.hg = hg;
                    } else {
                        dst.copy_from_slice(&cand);
                    }
                    recs.push(rec);
                }
            }
            std::mem::swap(&mut value, &mut next);
            grids.push(value.clone()); // value AT START of next sweep
            backups.push(recs);
        }

        let (ar, ac) = agent_cell;
        let agent_cell_idx = ar * grid_w + ac;
        let readout = self.gather_agent_readout(
            ar, ac, grid_h, grid_w, &value, &value_init, &gate, v,
        );
        let move_logits = self.move_head.forward(&readout);

        let out = VinOutput {
            move_logits,
            agent_readout: readout.clone(),
            agent_value: value[agent_cell_idx * v..(agent_cell_idx + 1) * v].to_vec(),
            agent_cell: (ar, ac),
            value_grid: value,
            gate: gate.clone(),
            reward: reward.clone(),
            grid_h,
            grid_w,
            value_dim: v,
        };

        let cache = VinCache {
            grid_h,
            grid_w,
            n_cells,
            v,
            iters,
            agent_cell: (ar, ac),
            gate_logit,
            gate,
            reward,
            value_init,
            grids,
            backups,
            readout,
        };

        (out, cache)
    }

    /// Softmax backup that also returns the record needed for backward.
    #[allow(clippy::too_many_arguments)]
    fn backup_cell_cached(
        &self,
        r: usize,
        c: usize,
        grid_h: usize,
        grid_w: usize,
        value: &[f32],
        gate: &[f32],
        local_reward: f32,
        v: usize,
        t: f32,
    ) -> (Vec<f32>, BackupRecord) {
        let mut nbr_idx: Vec<usize> = Vec::with_capacity(4);
        let mut nbr_pref: Vec<f32> = Vec::with_capacity(4);
        let mut nbr_mag: Vec<f32> = Vec::with_capacity(4);
        for (dr, dc) in DIR_OFFSETS {
            let nr = r as i32 + dr;
            let nc = c as i32 + dc;
            if nr < 0 || nc < 0 || nr >= grid_h as i32 || nc >= grid_w as i32 {
                continue;
            }
            let ncell = nr as usize * grid_w + nc as usize;
            let g = gate[ncell];
            let mag: f32 = value[ncell * v..(ncell + 1) * v]
                .iter()
                .map(|x| x.abs())
                .sum::<f32>();
            nbr_idx.push(ncell);
            nbr_pref.push(g * mag);
            nbr_mag.push(mag);
        }

        let mut out = vec![0.0f32; v];
        if nbr_idx.is_empty() {
            for k in 0..v {
                out[k] = local_reward;
            }
            return (
                out,
                BackupRecord {
                    nbr_idx,
                    sw: vec![],
                    pref: vec![],
                    mag: vec![],
                    cand: vec![],
                    prev: vec![],
                    hg: vec![],
                },
            );
        }

        let maxp = nbr_pref.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut wsum = 0.0f32;
        let mut w = vec![0.0f32; nbr_idx.len()];
        for (i, &p) in nbr_pref.iter().enumerate() {
            let e = ((p - maxp) / t).exp();
            w[i] = e;
            wsum += e;
        }
        let inv = 1.0 / wsum.max(1e-8);
        let sw: Vec<f32> = w.iter().map(|wi| wi * inv).collect();
        for k in 0..v {
            let mut acc = 0.0f32;
            for (i, &ncell) in nbr_idx.iter().enumerate() {
                let g = gate[ncell];
                acc += sw[i] * g * value[ncell * v + k];
            }
            out[k] = local_reward + acc;
        }
        let rec = BackupRecord {
            nbr_idx,
            sw,
            pref: nbr_pref,
            mag: nbr_mag,
            cand: vec![],
            prev: vec![],
            hg: vec![],
        };
        (out, rec)
    }

    /// Backpropagate `d_move_logits` (the upstream gradient on the move
    /// logits) through the cached forward, returning all trainable grads.
    pub fn backward(
        &self,
        cache: &VinCache,
        tokens: &[f32],
        grid_h: usize,
        grid_w: usize,
        d_move_logits: &[f32],
    ) -> VinGradients {
        debug_assert_eq!(grid_h, cache.grid_h);
        debug_assert_eq!(grid_w, cache.grid_w);
        let v = cache.v;
        let n_cells = cache.n_cells;
        let t = self.config.softmax_temp;
        let mut g = VinGradients::zeros(self);

        // ── move_head: d_readout from d_move_logits ──────────────────────
        let mut d_readout = vec![0.0f32; self.move_head.in_dim];
        g.move_head
            .accum(&self.move_head, &cache.readout, d_move_logits, Some(&mut d_readout));

        // Accumulators across the whole grid.
        // d_value for the *current* sweep's output grid (post-propagation
        // at the readout, then back through sweeps).
        let mut d_value = vec![0.0f32; n_cells * v];
        let mut d_value_init = vec![0.0f32; n_cells * v];
        // d_gate accumulates contributions at neighbour cells from the
        // readout AND from every backup; finally pushed through sigmoid.
        let mut d_gate = vec![0.0f32; n_cells];
        let mut d_reward = vec![0.0f32; n_cells];

        let (ar, ac) = cache.agent_cell;
        let acell = ar * grid_w + ac;

        // readout layout: [0]=value_init[acell] (v) | [1]=value[acell] (v) |
        //                 [2..6]= gate[ncell]*value[ncell] for 4 nbrs (v each)
        // block 0 → d_value_init[acell]
        for k in 0..v {
            d_value_init[acell * v + k] += d_readout[k];
        }
        // block 1 → d_value[acell] (post-propagation)
        for k in 0..v {
            d_value[acell * v + k] += d_readout[v + k];
        }
        // blocks 2..6 → neighbours, each = gate[ncell] * value[ncell]
        for (di, (dr, dc)) in DIR_OFFSETS.iter().enumerate() {
            let base = (2 + di) * v;
            let nr = ar as i32 + dr;
            let nc = ac as i32 + dc;
            if nr < 0 || nc < 0 || nr >= grid_h as i32 || nc >= grid_w as i32 {
                continue; // off-grid → zero block, no grad
            }
            let ncell = nr as usize * grid_w + nc as usize;
            let gate_n = cache.gate[ncell];
            for k in 0..v {
                let dr_k = d_readout[base + k];
                // d(gate*value)/dvalue = gate ; d/dgate = value
                d_value[ncell * v + k] += dr_k * gate_n;
                d_gate[ncell] += dr_k * cache.grids[cache.iters][ncell * v + k];
            }
        }

        // ── BPTT through the K sweeps, reverse order ─────────────────────
        // For sweep s (0-indexed), input grid = cache.grids[s], output grid =
        // cache.grids[s+1]. We hold `d_value` = grad on the OUTPUT grid of
        // sweep s (= grids[s+1]); after processing we transform it into grad
        // on the INPUT grid (grids[s]) and continue.
        for s in (0..cache.iters).rev() {
            let recs = &cache.backups[s];
            let in_grid = &cache.grids[s];
            // grad accumulator for the input grid of this sweep
            let mut d_in = vec![0.0f32; n_cells * v];

            for cell in 0..n_cells {
                let rec = &recs[cell];
                let d_next = &d_value[cell * v..(cell + 1) * v];

                // d_cand and d_prev (prev = in_grid[cell]) from highway blend.
                let mut d_cand = vec![0.0f32; v];
                if self.config.highway_gate {
                    // next = g*cand + (1-g)*prev
                    let hg = &rec.hg;
                    let mut d_g = vec![0.0f32; v];
                    for k in 0..v {
                        d_cand[k] += d_next[k] * hg[k];
                        d_in[cell * v + k] += d_next[k] * (1.0 - hg[k]);
                        d_g[k] = d_next[k] * (rec.cand[k] - rec.prev[k]);
                    }
                    // d_g through sigmoid: dz = d_g * g*(1-g)
                    let mut dz = vec![0.0f32; v];
                    for k in 0..v {
                        dz[k] = d_g[k] * hg[k] * (1.0 - hg[k]);
                    }
                    // highway_proj input = [prev | cand] (len 2v)
                    let mut hin = Vec::with_capacity(2 * v);
                    hin.extend_from_slice(&rec.prev);
                    hin.extend_from_slice(&rec.cand);
                    let mut d_hin = vec![0.0f32; 2 * v];
                    g.highway_proj
                        .accum(&self.highway_proj, &hin, &dz, Some(&mut d_hin));
                    // d_hin[0..v] → prev (= in_grid[cell]); d_hin[v..2v] → cand
                    for k in 0..v {
                        d_in[cell * v + k] += d_hin[k];
                        d_cand[k] += d_hin[v + k];
                    }
                } else {
                    // next = cand
                    for k in 0..v {
                        d_cand[k] += d_next[k];
                    }
                }

                // ── backup_cell backward ──────────────────────────────────
                // cand[k] = local_reward + sum_i sw_i * gate[ni] * value[ni*v+k]
                if rec.nbr_idx.is_empty() {
                    // cand[k] = local_reward (broadcast) for all k
                    for k in 0..v {
                        d_reward[cell] += d_cand[k];
                    }
                    continue;
                }

                // d_local_reward = sum_k d_cand[k]
                for k in 0..v {
                    d_reward[cell] += d_cand[k];
                }

                let m = rec.nbr_idx.len();
                // Path A — direct value term: for each neighbour i, each k:
                //   d_value[ni,k] += d_cand[k] * sw_i * gate[ni]
                //   d_gate[ni]    += d_cand[k] * sw_i * value[ni,k]
                // Also build d_sw[i] = sum_k d_cand[k] * gate[ni] * value[ni,k]
                //   (this feeds the softmax-weight path).
                let mut d_sw = vec![0.0f32; m];
                for (i, &ncell) in rec.nbr_idx.iter().enumerate() {
                    let gate_n = cache.gate[ncell];
                    let swi = rec.sw[i];
                    let mut dsw_i = 0.0f32;
                    for k in 0..v {
                        let val = in_grid[ncell * v + k];
                        let dck = d_cand[k];
                        d_in[ncell * v + k] += dck * swi * gate_n;
                        d_gate[ncell] += dck * swi * val;
                        dsw_i += dck * gate_n * val;
                    }
                    d_sw[i] = dsw_i;
                }

                // Path B — softmax: sw_i = softmax_i(pref/t), pref_i = gate*mag.
                // d_pref_j = (1/t) * sw_j * ( d_sw_j - sum_i sw_i d_sw_i ).
                let mut dot = 0.0f32;
                for i in 0..m {
                    dot += rec.sw[i] * d_sw[i];
                }
                for j in 0..m {
                    let d_pref_j = (rec.sw[j] * (d_sw[j] - dot)) / t;
                    let ncell = rec.nbr_idx[j];
                    let gate_n = cache.gate[ncell];
                    // pref_j = gate[nj] * mag_j
                    //   d_gate[nj] += d_pref_j * mag_j
                    //   d_mag_j     = d_pref_j * gate[nj]
                    d_gate[ncell] += d_pref_j * rec.mag[j];
                    let d_mag = d_pref_j * gate_n;
                    // mag_j = sum_k |value[nj,k]| → d_value[nj,k] += d_mag*sign
                    for k in 0..v {
                        let val = in_grid[ncell * v + k];
                        let sign = if val > 0.0 {
                            1.0
                        } else if val < 0.0 {
                            -1.0
                        } else {
                            0.0
                        };
                        d_in[ncell * v + k] += d_mag * sign;
                    }
                }
            }

            // d_in is now grad on grids[s] (input of this sweep) → becomes
            // d_value for the previous sweep's output. Note: highway already
            // routed the prev-passthrough into d_in[cell] above.
            d_value = d_in;
        }

        // After the loop, d_value holds grad on grids[0] = value_init grid
        // (the sweep-0 input == the projected `value`). value_init is also
        // read directly by the readout (block 0). Both flow through
        // value_proj. The sweep-0 input grid and value_init are the SAME
        // projected values, so combine.
        for idx in 0..n_cells * v {
            d_value_init[idx] += d_value[idx];
        }

        // ── Per-cell projection backward ─────────────────────────────────
        for cell in 0..n_cells {
            let tok = &tokens[cell * self.raw_dim..(cell + 1) * self.raw_dim];
            // value_proj: token → value[cell] (v outputs)
            let dv = &d_value_init[cell * v..(cell + 1) * v];
            g.value_proj.accum(&self.value_proj, tok, dv, None);
            // reward_proj: token → reward[cell] (1 output)
            let dr = [d_reward[cell]];
            g.reward_proj.accum(&self.reward_proj, tok, &dr, None);
            // gate_proj: token → gate_logit → sigmoid → gate
            let gv = cache.gate[cell];
            let dz = d_gate[cell] * gv * (1.0 - gv);
            g.gate_proj.accum(&self.gate_proj, tok, &[dz], None);
        }

        g
    }

    /// Plain SGD update: `w -= lr * dw` on every trainable Linear.
    pub fn apply_grads(&mut self, g: &VinGradients, lr: f32) {
        let step = |l: &mut Linear, grad: &LinearGrad| {
            for (w, dw) in l.weight.iter_mut().zip(&grad.weight) {
                *w -= lr * dw;
            }
            for (b, db) in l.bias.iter_mut().zip(&grad.bias) {
                *b -= lr * db;
            }
        };
        step(&mut self.reward_proj, &g.reward_proj);
        step(&mut self.gate_proj, &g.gate_proj);
        step(&mut self.value_proj, &g.value_proj);
        step(&mut self.highway_proj, &g.highway_proj);
        step(&mut self.move_head, &g.move_head);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> VinReadout {
        let cfg = VinConfig {
            value_dim: 4,
            iters: 6,
            max_iters: 20,
            softmax_temp: 0.5,
            highway_gate: true,
            n_dirs: 4,
        };
        VinReadout::new(cfg, 3)
    }

    #[test]
    fn forward_shapes_are_consistent() {
        let vin = tiny();
        let (h, w) = (3usize, 3usize);
        let tokens = vec![0.1f32; h * w * vin.raw_dim];
        let out = vin.forward(&tokens, h, w, Some((1, 1)));
        assert_eq!(out.move_logits.len(), 4);
        assert_eq!(out.agent_readout.len(), vin.agent_cell_readout_dim());
        assert_eq!(out.agent_value.len(), 4);
        assert_eq!(out.value_grid.len(), h * w * 4);
        assert_eq!(out.gate.len(), h * w);
        assert_eq!(out.agent_cell, (1, 1));
    }

    #[test]
    fn iters_are_capped() {
        let cfg = VinConfig { iters: 999, max_iters: 12, ..Default::default() };
        assert_eq!(cfg.effective_iters(), 12);
    }

    #[test]
    fn agent_localisation_picks_max_agentness() {
        // Force a token that scores high on agent_proj at cell 5.
        let mut vin = tiny();
        // Zero agent_proj, then set a bias so all equal; tie-break = first.
        vin.agent_proj.weight.fill(0.0);
        vin.agent_proj.bias = vec![0.0];
        let (h, w) = (3usize, 3usize);
        let mut tokens = vec![0.0f32; h * w * vin.raw_dim];
        // Make cell 5's token large and agent_proj positive on dim 0.
        vin.agent_proj.weight = vec![1.0, 0.0, 0.0];
        tokens[5 * vin.raw_dim] = 10.0;
        let out = vin.forward(&tokens, h, w, None);
        assert_eq!(out.agent_cell, (5 / w, 5 % w));
    }

    // A tiny deterministic PRNG so the gradient check is reproducible.
    struct Lcg(u64);
    impl Lcg {
        fn new(seed: u64) -> Self {
            Lcg(seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1))
        }
        fn next_f32(&mut self) -> f32 {
            // xorshift64*
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            let r = x.wrapping_mul(0x2545F4914F6CDD1D);
            // map to [-1, 1)
            ((r >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
    }

    #[test]
    fn vin_backward_gradient_check() {
        let cfg = VinConfig {
            value_dim: 3,
            iters: 3,
            max_iters: 20,
            softmax_temp: 0.5,
            highway_gate: true,
            n_dirs: 4,
        };
        let raw_dim = 3;
        let mut vin = VinReadout::new(cfg, raw_dim);

        // Fixed pseudo-random params for all trainable Linears.
        let mut rng = Lcg::new(0xC0FFEE);
        let fill = |l: &mut Linear, rng: &mut Lcg| {
            for w in l.weight.iter_mut() {
                *w = rng.next_f32() * 0.5;
            }
            for b in l.bias.iter_mut() {
                *b = rng.next_f32() * 0.5;
            }
        };
        fill(&mut vin.reward_proj, &mut rng);
        fill(&mut vin.gate_proj, &mut rng);
        fill(&mut vin.value_proj, &mut rng);
        fill(&mut vin.highway_proj, &mut rng);
        fill(&mut vin.move_head, &mut rng);
        // agent_proj left at init (irrelevant — explicit agent cell).

        let (h, w) = (3usize, 3usize);
        let n_cells = h * w;
        let mut tokens = vec![0.0f32; n_cells * raw_dim];
        for t in tokens.iter_mut() {
            *t = rng.next_f32();
        }
        let agent = (1usize, 1usize);

        // Fixed upstream gradient on the move logits.
        let mut d_move = vec![0.0f32; vin.config.n_dirs];
        for d in d_move.iter_mut() {
            *d = rng.next_f32();
        }

        // L = sum_d d_move[d] * move_logits[d]  ⇒ dL/dlogit = d_move.
        let loss = |vin: &VinReadout| -> f32 {
            let out = vin.forward(&tokens, h, w, Some(agent));
            out.move_logits
                .iter()
                .zip(&d_move)
                .map(|(l, d)| l * d)
                .sum()
        };

        // Analytic grads.
        let (_out, cache) = vin.forward_train(&tokens, h, w, agent);
        let grads = vin.backward(&cache, &tokens, h, w, &d_move);

        // Finite-difference each parameter element.
        let eps = 1e-3f32;
        let mut max_rel = 0.0f32;
        let mut worst = String::new();

        // (name, accessor to (weight, bias) on vin, accessor to grad)
        macro_rules! check_linear {
            ($name:literal, $lin:ident, $grad:ident) => {{
                let n_w = vin.$lin.weight.len();
                let n_b = vin.$lin.bias.len();
                for idx in 0..(n_w + n_b) {
                    // perturb
                    let mut vp = vin.clone();
                    let mut vm = vin.clone();
                    if idx < n_w {
                        vp.$lin.weight[idx] += eps;
                        vm.$lin.weight[idx] -= eps;
                    } else {
                        vp.$lin.bias[idx - n_w] += eps;
                        vm.$lin.bias[idx - n_w] -= eps;
                    }
                    let numeric = (loss(&vp) - loss(&vm)) / (2.0 * eps);
                    let analytic = if idx < n_w {
                        grads.$grad.weight[idx]
                    } else {
                        grads.$grad.bias[idx - n_w]
                    };
                    let rel = (analytic - numeric).abs() / (numeric.abs() + 1e-3);
                    if rel > max_rel {
                        max_rel = rel;
                        worst = format!(
                            "{} [{}] analytic={:.6} numeric={:.6}",
                            $name, idx, analytic, numeric
                        );
                    }
                }
            }};
        }

        check_linear!("reward_proj", reward_proj, reward_proj);
        check_linear!("gate_proj", gate_proj, gate_proj);
        check_linear!("value_proj", value_proj, value_proj);
        check_linear!("highway_proj", highway_proj, highway_proj);
        check_linear!("move_head", move_head, move_head);

        println!("vin gradient check: max rel error = {max_rel:.6e}  at  {worst}");
        assert!(
            max_rel < 2e-2,
            "VIN backward gradient check failed: max rel error {max_rel:.6e} at {worst}"
        );
    }

    /// Deterministic builder for the cross-repo golden vector. Fills every
    /// trainable Linear's weight[i] and bias[i] with a fixed integer-hash
    /// pattern so the WEB engine port can reconstruct the EXACT same readout
    /// from the serialized JSON and assert a bit-exact forward.
    fn golden_fill(l: &mut Linear) {
        for (i, w) in l.weight.iter_mut().enumerate() {
            *w = ((i as u64).wrapping_mul(2654435761) % 1000) as f32 / 1000.0 - 0.5;
        }
        for (i, b) in l.bias.iter_mut().enumerate() {
            *b = ((i as u64).wrapping_mul(2654435761) % 1000) as f32 / 1000.0 - 0.5;
        }
    }

    fn golden_vin() -> VinReadout {
        let cfg = VinConfig {
            value_dim: 4,
            iters: 5,
            max_iters: 20,
            softmax_temp: 0.5,
            highway_gate: true,
            n_dirs: 4,
        };
        let mut vin = VinReadout::new(cfg, 3);
        golden_fill(&mut vin.reward_proj);
        golden_fill(&mut vin.gate_proj);
        golden_fill(&mut vin.value_proj);
        golden_fill(&mut vin.agent_proj);
        golden_fill(&mut vin.highway_proj);
        golden_fill(&mut vin.move_head);
        vin
    }

    fn golden_tokens() -> Vec<f32> {
        // 3x3 grid, raw_dim=3 → 27 token elements.
        (0..27)
            .map(|i| ((i as u64).wrapping_mul(40503) % 100) as f32 / 100.0)
            .collect()
    }

    /// Prints the golden VinReadout JSON, tokens, and resulting move_logits at
    /// full f32 precision. Run with:
    ///   cargo test -p modgrad-ctm vin::tests::print_golden_vector -- --nocapture
    #[test]
    fn print_golden_vector() {
        let vin = golden_vin();
        let tokens = golden_tokens();
        let out = vin.forward(&tokens, 3, 3, Some((1, 1)));
        let json = serde_json::to_string(&vin).unwrap();
        println!("GOLDEN_JSON_BEGIN");
        println!("{json}");
        println!("GOLDEN_JSON_END");
        println!("GOLDEN_TOKENS {:?}", tokens);
        println!("GOLDEN_MOVE_LOGITS {:?}", out.move_logits);
    }

    #[test]
    fn hard_max_path_runs() {
        let cfg = VinConfig {
            value_dim: 4,
            iters: 4,
            max_iters: 20,
            softmax_temp: 0.0, // hard max
            highway_gate: false,
            n_dirs: 4,
        };
        let vin = VinReadout::new(cfg, 3);
        let (h, w) = (4usize, 4usize);
        let tokens = vec![0.2f32; h * w * vin.raw_dim];
        let out = vin.forward(&tokens, h, w, Some((0, 0)));
        assert_eq!(out.move_logits.len(), 4);
        assert!(out.value_grid.iter().all(|x| x.is_finite()));
    }
}
