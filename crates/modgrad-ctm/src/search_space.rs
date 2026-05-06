//! Brain NAS search space — replaces the hand-picked
//! `eight_region_{small,medium,large,billion,mega}` presets with a
//! sampleable, encoded representation. Presets become *output points*
//! of this space, not inputs.
//!
//! Three layers, smallest → largest scope:
//!
//! ```text
//! Layer 0: PerRegionShape   — scalars per region (d_model, memory_length,
//!                             deep_nlms, beta, iterations)
//! Layer 1: ConnectivityMask — 8×8 source→target adjacency + per-target
//!                             recv_obs flag + per-target obs_scale
//! Layer 2: Globals          — outer_ticks, outer beta, aux_losses
//! ```
//!
//! Region indices are fixed at canonical roles
//! (input/attention/output/motor/cerebellum/basal_ganglia/insula/
//! hippocampus); searching `n_regions` would be a different refactor.
//!
//! The connectivity is NOT required to be a DAG: the brain's outer
//! tick loop reads from `prev_outputs` across ticks, so cycles
//! through cortex are intentional (see `eight_region_billion`'s
//! MOTOR→INPUT→ATTENTION→OUTPUT→MOTOR loop).
//!
//! Hardware-lesson constraint baked into `is_valid`: cortical
//! `d_model` must lie in `{≤256} ∪ {≥1024}` to skip the measured
//! "dead zone" where neither the host AVX-512 path nor the resident
//! MIOpen path wins (see `feedback_residency_dispatch_overhead`).

use modgrad_compute::neuron::SimpleRng;
use serde::{Deserialize, Serialize};

use crate::config::{CtmConfig, ExitStrategy};
use crate::graph::{AuxLossConfig, Connection, RegionalConfig, RouterConfig};

/// Number of typed regions (canonical 8-region brain).
pub const N_REGIONS: usize = 8;

/// Canonical region indices (must match `eight_region*` builders).
pub const INPUT: usize = 0;
pub const ATTENTION: usize = 1;
pub const OUTPUT: usize = 2;
pub const MOTOR: usize = 3;
pub const CEREBELLUM: usize = 4;
pub const BASAL_GANGLIA: usize = 5;
pub const INSULA: usize = 6;
pub const HIPPOCAMPUS: usize = 7;

/// Stable role names; fixed because forward/backward indexes by role.
pub const REGION_NAMES: [&str; N_REGIONS] = [
    "input", "attention", "output", "motor",
    "cerebellum", "basal_ganglia", "insula", "hippocampus",
];

/// Cortical regions (input/attention/output/motor) — held to the
/// hardware crossover discipline. Subcortical regions are exempt
/// because their per-dispatch compute is small by design.
pub const CORTICAL: [usize; 4] = [INPUT, ATTENTION, OUTPUT, MOTOR];

// ─── Search-space grids ──────────────────────────────────────

/// Log-spaced d_model grid spanning `eight_region_small` (32) up
/// through `eight_region_billion` (1024). The dead-zone constraint
/// in `is_valid` will further reject 512 for cortical regions.
pub const D_MODEL_GRID: &[usize] = &[
    8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024,
];

/// NLM history window length (M in paper).
pub const MEMORY_GRID: &[usize] = &[4, 8, 16, 32, 64, 128];

/// Per-region inner-tick budget.
pub const ITERATIONS_GRID: &[usize] = &[2, 4, 6, 8];

/// Adaptive-gate β (entropy regulariser strength).
pub const BETA_GRID: &[f32] = &[0.05, 0.1, 0.15];

/// Outer-tick budget (whole-brain).
pub const OUTER_TICKS_GRID: &[usize] = &[2, 4, 6, 8, 12];

/// Auxiliary-loss weight when any aux is enabled.
pub const AUX_WEIGHT_GRID: &[f32] = &[0.05, 0.1, 0.2];

/// Observation-scale grid (V1 / V2 / V4 indices for visual cortex).
/// Held at `[0]` for v1 because `to_regional_config` builds a
/// single-scale obs vector — a multi-scale axis only makes sense
/// when the upstream encoder also emits multiple scales.
pub const OBS_SCALE_GRID: &[usize] = &[0];

// ─── Constraints ─────────────────────────────────────────────

/// Maximum total parameters (8GB VRAM with f32 weights+grads+Adam).
pub const MAX_PARAMS: usize = 200_000_000;

/// Maximum sum of `d_model` across regions (residency dispatch headroom).
pub const MAX_TOTAL_D_MODEL: usize = 8192;

/// Cortical d_model must be in this set: ≤256 (host wins) or ≥1024
/// (resident wins). See `feedback_residency_dispatch_overhead`.
pub fn cortical_d_model_in_band(d: usize) -> bool {
    d <= 256 || d >= 1024
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArchError {
    SelfEdge { region: usize },
    NoObservation,
    TotalDModelTooLarge { sum: usize },
    CorticalDModelInDeadZone { region: usize, d_model: usize },
    InvalidGridValue { field: &'static str },
}

// ─── PerRegionShape ──────────────────────────────────────────

/// Scalars searched per region. Everything else in `CtmConfig`
/// (synapse_depth, min_width, heads, n_synch_*, memory_hidden_dims)
/// is *derived* by `CtmConfig::region`; we treat that derivation
/// as a fixed search-space prior, not as additional axes.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PerRegionShape {
    pub d_model: u16,
    pub memory_length: u16,
    pub deep_nlms: bool,
    /// Adaptive-gate β; AdaptiveGate is the only exit strategy in the
    /// space (matches every preset). Threshold is fixed at 0.99.
    pub beta: f32,
    pub iterations: u16,
}

impl PerRegionShape {
    pub fn into_ctm_config(self, name: &str, d_input: usize) -> CtmConfig {
        CtmConfig::region(
            name,
            self.d_model as usize,
            d_input,
            self.memory_length as usize,
            self.deep_nlms,
            self.iterations as usize,
            ExitStrategy::AdaptiveGate { beta: self.beta, threshold: 0.99 },
        )
    }
}

// ─── ConnectivityMask ────────────────────────────────────────

/// Per-target source bag + per-target observation flags.
///
/// `sources[target]` is a bitmask over 8 source regions (bit `s` set
/// means region `s` feeds into `target`). Diagonal bits (self-edges)
/// are forced to 0 by `is_valid`.
///
/// The 4-source merge in `eight_region_billion`'s hippocampus edge
/// (`{INPUT, ATTENTION, OUTPUT, MOTOR} → HIPPOCAMPUS`) is naturally
/// expressible: bits 0,1,2,3 set in `sources[HIPPOCAMPUS]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConnectivityMask {
    /// Bit `s` of `sources[t]` set ⇔ region `s` feeds region `t`.
    pub sources: [u8; N_REGIONS],
    /// `recv_obs[t]` ⇒ target `t` reads raw observation.
    pub recv_obs: [bool; N_REGIONS],
    /// `obs_scale[t]` ∈ `OBS_SCALE_GRID` (only meaningful when `recv_obs[t]`).
    pub obs_scale: [u8; N_REGIONS],
}

impl ConnectivityMask {
    pub fn empty() -> Self {
        Self {
            sources: [0; N_REGIONS],
            recv_obs: [false; N_REGIONS],
            obs_scale: [0; N_REGIONS],
        }
    }

    pub fn has_edge(&self, src: usize, tgt: usize) -> bool {
        self.sources[tgt] & (1u8 << src) != 0
    }

    pub fn set_edge(&mut self, src: usize, tgt: usize, on: bool) {
        let bit = 1u8 << src;
        if on {
            self.sources[tgt] |= bit;
        } else {
            self.sources[tgt] &= !bit;
        }
    }

    pub fn into_connections(self) -> Vec<Connection> {
        let mut out = Vec::new();
        for tgt in 0..N_REGIONS {
            let mut from: Vec<usize> = (0..N_REGIONS)
                .filter(|&s| self.sources[tgt] & (1u8 << s) != 0)
                .collect();
            // Single-source edges that also receive obs and edges that
            // receive obs without sources still need an entry; skip
            // only when there's nothing to project at all (region falls
            // back to obs_projected in regional_forward).
            let recv = self.recv_obs[tgt];
            if from.is_empty() && !recv {
                continue;
            }
            // If recv-only, the synapse still needs a source list; an
            // empty `from` is allowed by the Connection schema and the
            // forward concatenates `observation` onto an empty source
            // list cleanly.
            from.shrink_to_fit();
            out.push(Connection {
                from,
                to: tgt,
                receives_observation: recv,
                observation_scale: self.obs_scale[tgt] as usize,
            });
        }
        out
    }
}

// ─── Globals ─────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Globals {
    pub outer_ticks: u16,
    pub outer_beta: f32,
    pub aux: [bool; 3], // [cerebellar_pred, hipp_contrastive, bg_td]
    pub aux_weight: f32,
    pub router_enabled: bool,
}

impl Globals {
    pub fn default_preset_compatible() -> Self {
        Self {
            outer_ticks: 4,
            outer_beta: 0.1,
            aux: [false; 3],
            aux_weight: 0.1,
            router_enabled: true,
        }
    }

    pub fn into_aux_config(self) -> AuxLossConfig {
        AuxLossConfig {
            cerebellar_prediction: self.aux[0],
            hippocampal_contrastive: self.aux[1],
            bg_temporal_difference: self.aux[2],
            aux_weight: self.aux_weight,
        }
    }
}

// ─── BrainArch ───────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainArch {
    pub regions: [PerRegionShape; N_REGIONS],
    pub mask: ConnectivityMask,
    pub globals: Globals,
}

impl BrainArch {
    /// Serialise this architecture to a pretty-printed JSON file.
    /// Used by `brain_nas_smoke` to persist top-K NAS results across
    /// runs so they can be re-evaluated or trained later.
    pub fn save_json(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    pub fn load_json(path: &str) -> std::io::Result<Self> {
        let s = std::fs::read_to_string(path)?;
        serde_json::from_str(&s)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

impl BrainArch {
    /// Constructs a `RegionalConfig` for given observation/output
    /// dimensions. Uses `CtmConfig::region` for every per-region
    /// derivation — same rules as the hand-picked presets.
    pub fn to_regional_config(&self, obs_dim: usize, out_dims: usize) -> RegionalConfig {
        let regions: Vec<CtmConfig> = (0..N_REGIONS)
            .map(|i| self.regions[i].into_ctm_config(REGION_NAMES[i], obs_dim))
            .collect();
        let region_names: Vec<String> =
            REGION_NAMES.iter().map(|s| s.to_string()).collect();
        let connections = self.mask.into_connections();

        let total_neurons: usize = regions.iter().map(|r| r.d_model).sum();
        let n_global_sync = total_neurons.min(2048);

        RegionalConfig {
            regions,
            region_names,
            connections,
            outer_ticks: self.globals.outer_ticks as usize,
            exit_strategy: ExitStrategy::AdaptiveGate {
                beta: self.globals.outer_beta,
                threshold: 0.99,
            },
            n_global_sync,
            out_dims,
            raw_obs_dim: obs_dim,
            obs_scale_dims: vec![obs_dim],
            aux_losses: self.globals.into_aux_config(),
            router: if self.globals.router_enabled {
                Some(RouterConfig::default())
            } else {
                None
            },
            cereb_mode: Default::default(),
        }
    }

    /// Cheap structural validity check. Does not allocate weights.
    pub fn is_valid(&self) -> Result<(), ArchError> {
        for t in 0..N_REGIONS {
            if self.mask.sources[t] & (1u8 << t) != 0 {
                return Err(ArchError::SelfEdge { region: t });
            }
        }
        if !self.mask.recv_obs.iter().any(|&b| b) {
            return Err(ArchError::NoObservation);
        }
        let total: usize = self.regions.iter().map(|r| r.d_model as usize).sum();
        if total > MAX_TOTAL_D_MODEL {
            return Err(ArchError::TotalDModelTooLarge { sum: total });
        }
        for &i in &CORTICAL {
            let d = self.regions[i].d_model as usize;
            if !cortical_d_model_in_band(d) {
                return Err(ArchError::CorticalDModelInDeadZone { region: i, d_model: d });
            }
        }
        Ok(())
    }

    /// Uniform random draw from the discrete grids. Caller filters
    /// with `is_valid` and resamples on rejection.
    pub fn random_sample(rng: &mut SimpleRng) -> Self {
        Self::random_sample_biased(rng, &SamplerBias::uniform())
    }

    /// Biased random draw. With `SamplerBias::uniform` reduces to
    /// `random_sample`; with non-zero `cortex_large_prob`, cortical
    /// regions are forced to `d_model=1024` with that probability,
    /// otherwise drawn uniformly from the lower-band slice
    /// (≤256). Subcortical regions are always uniform across the
    /// full grid since the dead-zone constraint doesn't apply to them.
    ///
    /// This is the lever that turns random search from a
    /// small-cortex-dominated walk into a search that actually
    /// hits the resident regime — under uniform sampling, four
    /// cortical regions all landing at 1024 has p≈10⁻⁴; under
    /// `cortex_large_prob=1.0`, p=1.
    pub fn random_sample_biased(rng: &mut SimpleRng, bias: &SamplerBias) -> Self {
        let regions = std::array::from_fn(|i| {
            let is_cortex = CORTICAL.contains(&i);
            sample_region(rng, is_cortex, bias)
        });
        let mask = sample_mask(rng, bias);
        let globals = sample_globals(rng);
        Self { regions, mask, globals }
    }

    /// Rejection-sample until `is_valid` passes or `max_attempts`
    /// runs out. Returns the architecture and the attempt count.
    pub fn sample_valid(
        rng: &mut SimpleRng,
        max_attempts: usize,
    ) -> Option<(Self, usize)> {
        Self::sample_valid_biased(rng, &SamplerBias::uniform(), max_attempts)
    }

    pub fn sample_valid_biased(
        rng: &mut SimpleRng,
        bias: &SamplerBias,
        max_attempts: usize,
    ) -> Option<(Self, usize)> {
        for k in 1..=max_attempts {
            let arch = Self::random_sample_biased(rng, bias);
            if arch.is_valid().is_ok() {
                return Some((arch, k));
            }
        }
        None
    }
}

/// Knobs that bias the random sampler away from uniform without
/// expanding the search space itself. `uniform()` reproduces the
/// default behaviour of `random_sample`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SamplerBias {
    /// Probability (0..=1) that each cortical region's `d_model` is
    /// forced to 1024. With probability `1 - cortex_large_prob` the
    /// cortical d_model is drawn uniformly from the lower band only
    /// (≤256), so dead-zone resampling never trips.
    pub cortex_large_prob: f32,
    /// Per-target probability that `recv_obs` is set.
    pub recv_obs_prob: f32,
    /// Per non-self edge probability.
    pub edge_prob: f32,
}

impl SamplerBias {
    pub fn uniform() -> Self {
        Self {
            cortex_large_prob: 0.0,
            recv_obs_prob: 0.25,
            edge_prob: 0.35,
        }
    }

    /// Bias toward the resident-likely regime (cortex ≥ 1024 with
    /// probability 0.5). Subcortical and structural axes unchanged
    /// from `uniform()`.
    pub fn resident_friendly() -> Self {
        Self { cortex_large_prob: 0.5, ..Self::uniform() }
    }
}

/// Lower-band slice of `D_MODEL_GRID` (cortical regions, when not
/// forced to 1024). All values ≤ 256.
fn d_model_lower_band() -> &'static [usize] {
    // D_MODEL_GRID is `[8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]`
    // Lower band = first 10 entries (≤256).
    &D_MODEL_GRID[..10]
}

fn pick<T: Copy>(rng: &mut SimpleRng, grid: &[T]) -> T {
    let idx = (rng.next_u64() as usize) % grid.len();
    grid[idx]
}

fn sample_region(rng: &mut SimpleRng, is_cortex: bool, bias: &SamplerBias) -> PerRegionShape {
    let d_model = if is_cortex {
        if rng.next_f32() < bias.cortex_large_prob {
            1024usize
        } else {
            pick(rng, d_model_lower_band())
        }
    } else {
        pick(rng, D_MODEL_GRID)
    };
    PerRegionShape {
        d_model: d_model as u16,
        memory_length: pick(rng, MEMORY_GRID) as u16,
        deep_nlms: rng.next_u64() & 1 == 0,
        beta: pick(rng, BETA_GRID),
        iterations: pick(rng, ITERATIONS_GRID) as u16,
    }
}

fn sample_mask(rng: &mut SimpleRng, bias: &SamplerBias) -> ConnectivityMask {
    let mut sources = [0u8; N_REGIONS];
    for t in 0..N_REGIONS {
        let mut row = 0u8;
        for s in 0..N_REGIONS {
            if s == t { continue; }
            if rng.next_f32() < bias.edge_prob {
                row |= 1u8 << s;
            }
        }
        sources[t] = row;
    }
    let mut recv_obs = [false; N_REGIONS];
    for t in 0..N_REGIONS {
        recv_obs[t] = rng.next_f32() < bias.recv_obs_prob;
    }
    let mut obs_scale = [0u8; N_REGIONS];
    for t in 0..N_REGIONS {
        obs_scale[t] = pick(rng, OBS_SCALE_GRID) as u8;
    }
    ConnectivityMask { sources, recv_obs, obs_scale }
}

fn sample_globals(rng: &mut SimpleRng) -> Globals {
    Globals {
        outer_ticks: pick(rng, OUTER_TICKS_GRID) as u16,
        outer_beta: pick(rng, BETA_GRID),
        aux: [
            rng.next_u64() & 1 == 0,
            rng.next_u64() & 1 == 0,
            rng.next_u64() & 1 == 0,
        ],
        aux_weight: pick(rng, AUX_WEIGHT_GRID),
        router_enabled: rng.next_u64() & 1 == 0,
    }
}

// ─── Preset → BrainArch encodings (round-trip targets) ───────

impl BrainArch {
    /// Encode `eight_region_billion`'s shape as a `BrainArch` point.
    /// Cortex d_model=1024 / memory=128, subcortical d_model=128 /
    /// memory=64 (hippocampus memory=128).
    pub fn eight_region_billion_arch() -> Self {
        let cortex = PerRegionShape {
            d_model: 1024,
            memory_length: 128,
            deep_nlms: true,
            beta: 0.05,
            iterations: 4,
        };
        let cortex_router = PerRegionShape { beta: 0.1, ..cortex };
        let subcort = PerRegionShape {
            d_model: 128,
            memory_length: 64,
            deep_nlms: true,
            beta: 0.05,
            iterations: 4,
        };
        let bg = PerRegionShape { beta: 0.1, ..subcort };
        let hippo = PerRegionShape {
            d_model: 128, memory_length: 128, deep_nlms: true, beta: 0.15, iterations: 4,
        };
        let regions = [
            cortex,        // input
            cortex_router, // attention
            cortex_router, // output
            cortex,        // motor
            subcort,       // cerebellum
            bg,            // basal_ganglia
            subcort,       // insula
            hippo,         // hippocampus
        ];

        let mut mask = ConnectivityMask::empty();
        // MOTOR → INPUT (recv_obs)
        mask.set_edge(MOTOR, INPUT, true);
        mask.recv_obs[INPUT] = true;
        // INPUT → ATTENTION
        mask.set_edge(INPUT, ATTENTION, true);
        // ATTENTION → OUTPUT
        mask.set_edge(ATTENTION, OUTPUT, true);
        // OUTPUT → MOTOR
        mask.set_edge(OUTPUT, MOTOR, true);
        // MOTOR → CEREBELLUM (recv_obs)
        mask.set_edge(MOTOR, CEREBELLUM, true);
        mask.recv_obs[CEREBELLUM] = true;
        // OUTPUT → BASAL_GANGLIA
        mask.set_edge(OUTPUT, BASAL_GANGLIA, true);
        // HIPPOCAMPUS → INSULA
        mask.set_edge(HIPPOCAMPUS, INSULA, true);
        // {INPUT, ATTENTION, OUTPUT, MOTOR} → HIPPOCAMPUS
        for &s in &[INPUT, ATTENTION, OUTPUT, MOTOR] {
            mask.set_edge(s, HIPPOCAMPUS, true);
        }

        Self {
            regions,
            mask,
            globals: Globals::default_preset_compatible(),
        }
    }

    /// Encode `eight_region_small`'s shape (d=16, mem=8 cortex; d=8 subcort).
    /// Cortex d_model=16 — passes the dead-zone band (≤256).
    pub fn eight_region_small_arch() -> Self {
        let cortex_no_deep = PerRegionShape {
            d_model: 16, memory_length: 8, deep_nlms: false, beta: 0.05, iterations: 4,
        };
        let cortex_deep = PerRegionShape { deep_nlms: true, beta: 0.1, ..cortex_no_deep };
        let subcort = PerRegionShape {
            d_model: 8, memory_length: 4, deep_nlms: false, beta: 0.05, iterations: 4,
        };
        let bg = PerRegionShape { beta: 0.1, ..subcort };
        let hippo = PerRegionShape {
            d_model: 8, memory_length: 8, deep_nlms: true, beta: 0.15, iterations: 4,
        };
        let regions = [
            cortex_no_deep, // input
            cortex_deep,    // attention
            cortex_deep,    // output
            cortex_no_deep, // motor
            subcort,        // cerebellum
            bg,             // basal_ganglia
            subcort,        // insula
            hippo,          // hippocampus
        ];

        let mut mask = ConnectivityMask::empty();
        mask.set_edge(MOTOR, INPUT, true);
        mask.recv_obs[INPUT] = true;
        mask.set_edge(INPUT, ATTENTION, true);
        mask.set_edge(ATTENTION, OUTPUT, true);
        mask.set_edge(OUTPUT, MOTOR, true);
        mask.set_edge(MOTOR, CEREBELLUM, true);
        mask.recv_obs[CEREBELLUM] = true;
        mask.set_edge(OUTPUT, BASAL_GANGLIA, true);
        mask.set_edge(HIPPOCAMPUS, INSULA, true);
        for &s in &[INPUT, ATTENTION, OUTPUT, MOTOR] {
            mask.set_edge(s, HIPPOCAMPUS, true);
        }

        Self {
            regions,
            mask,
            globals: Globals::default_preset_compatible(),
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn billion_arch_encodes_to_billion_shapes() {
        let arch = BrainArch::eight_region_billion_arch();
        let cfg = arch.to_regional_config(64, 16);
        assert_eq!(cfg.regions.len(), N_REGIONS);
        // Cortex regions all d_model=1024 with memory=128.
        for &i in &CORTICAL {
            assert_eq!(cfg.regions[i].d_model, 1024, "cortex {} d_model", REGION_NAMES[i]);
            assert_eq!(cfg.regions[i].memory_length, 128, "cortex {} memory", REGION_NAMES[i]);
        }
        // Hippocampus has memory=128 by spec.
        assert_eq!(cfg.regions[HIPPOCAMPUS].memory_length, 128);
        // Subcortical d_model=128.
        for &i in &[CEREBELLUM, BASAL_GANGLIA, INSULA, HIPPOCAMPUS] {
            assert_eq!(cfg.regions[i].d_model, 128);
        }
        // 8 region names in canonical order.
        for i in 0..N_REGIONS {
            assert_eq!(cfg.region_names[i], REGION_NAMES[i]);
        }
    }

    #[test]
    fn billion_arch_connections_match_preset() {
        let arch = BrainArch::eight_region_billion_arch();
        let conns = arch.mask.into_connections();
        // Same edge count as the hand-picked preset (8 connections).
        assert_eq!(conns.len(), 8);
        // The 4-source merge into hippocampus must be present.
        let hippo_conn = conns.iter().find(|c| c.to == HIPPOCAMPUS).unwrap();
        assert_eq!(hippo_conn.from.len(), 4);
        assert!(hippo_conn.from.contains(&INPUT));
        assert!(hippo_conn.from.contains(&ATTENTION));
        assert!(hippo_conn.from.contains(&OUTPUT));
        assert!(hippo_conn.from.contains(&MOTOR));
    }

    #[test]
    fn small_arch_passes_dead_zone_band() {
        let arch = BrainArch::eight_region_small_arch();
        // Cortex d=16 ≤ 256 — passes lower band.
        assert_eq!(arch.is_valid(), Ok(()));
    }

    #[test]
    fn billion_arch_passes_dead_zone_band() {
        let arch = BrainArch::eight_region_billion_arch();
        // Cortex d=1024 ≥ 1024 — passes upper band.
        assert_eq!(arch.is_valid(), Ok(()));
    }

    #[test]
    fn d512_cortex_rejected_by_dead_zone() {
        let mut arch = BrainArch::eight_region_billion_arch();
        // Inject d_model=512 into cortex — measured dead zone.
        for &i in &CORTICAL {
            arch.regions[i].d_model = 512;
        }
        assert!(matches!(
            arch.is_valid(),
            Err(ArchError::CorticalDModelInDeadZone { d_model: 512, .. })
        ));
    }

    #[test]
    fn self_edge_rejected() {
        let mut arch = BrainArch::eight_region_billion_arch();
        arch.mask.sources[OUTPUT] |= 1u8 << OUTPUT;
        assert!(matches!(arch.is_valid(), Err(ArchError::SelfEdge { region: OUTPUT })));
    }

    #[test]
    fn no_observation_rejected() {
        let mut arch = BrainArch::eight_region_billion_arch();
        arch.mask.recv_obs = [false; N_REGIONS];
        assert_eq!(arch.is_valid(), Err(ArchError::NoObservation));
    }

    #[test]
    fn random_sample_validity_fraction() {
        // Under `SamplerBias::uniform`, cortex regions are drawn from
        // the lower band only (so the dead-zone constraint can't fire),
        // and total d_model is bounded ≤ 4×256 + 4×1024 = 5120 < 8192
        // (so the size cap can't fire). Only `NoObservation` filters
        // samples — at recv_obs_prob=0.25 across 8 regions,
        // P(no recv_obs) = 0.75^8 ≈ 10%. So expect ~90% valid.
        let mut rng = SimpleRng::new(0xBEEFu64);
        let mut valid = 0usize;
        let n = 1000;
        for _ in 0..n {
            let arch = BrainArch::random_sample(&mut rng);
            if arch.is_valid().is_ok() { valid += 1; }
        }
        let frac = valid as f32 / n as f32;
        assert!(frac > 0.80, "valid fraction {} too low — uniform sampler should rarely reject", frac);
        assert!(frac < 0.98, "valid fraction {} too high — NoObservation should still fire", frac);
    }

    #[test]
    fn cortex_large_prob_one_forces_resident_regime() {
        let mut rng = SimpleRng::new(0xCAFE);
        let bias = SamplerBias { cortex_large_prob: 1.0, ..SamplerBias::uniform() };
        for _ in 0..100 {
            let arch = BrainArch::random_sample_biased(&mut rng, &bias);
            for &i in &CORTICAL {
                assert_eq!(arch.regions[i].d_model, 1024,
                    "cortex_large_prob=1 must force d_model=1024");
            }
            // No dead-zone error possible — every cortex region is 1024.
            // (Other constraints like NoObservation may still fire.)
        }
    }

    #[test]
    fn cortex_large_prob_lifts_resident_likely_fraction() {
        // Under uniform sampling, "any cortex ≥ 1024" hits ~35% post-band.
        // Under cortex_large_prob=0.5, every cortex region is 1024 with
        // p=0.5 independently, so "any cortex ≥ 1024" should hit 1-0.5^4 ≈ 94%.
        let mut rng = SimpleRng::new(0xDEAF);
        let bias = SamplerBias::resident_friendly();
        let mut resident_likely = 0;
        let n = 500;
        for _ in 0..n {
            let arch = BrainArch::random_sample_biased(&mut rng, &bias);
            if CORTICAL.iter().any(|&i| arch.regions[i].d_model >= 1024) {
                resident_likely += 1;
            }
        }
        let frac = resident_likely as f32 / n as f32;
        assert!(frac > 0.85,
            "resident_friendly bias should lift any-cortex≥1024 above 85%, got {:.2}", frac);
    }

    #[test]
    fn json_round_trip_preserves_billion_arch() {
        let arch = BrainArch::eight_region_billion_arch();
        let json = serde_json::to_string(&arch).unwrap();
        let restored: BrainArch = serde_json::from_str(&json).unwrap();
        for i in 0..N_REGIONS {
            assert_eq!(arch.regions[i], restored.regions[i]);
        }
        assert_eq!(arch.mask, restored.mask);
        assert_eq!(arch.globals, restored.globals);
        // The decoded arch must produce an identical RegionalConfig shape.
        let cfg_a = arch.to_regional_config(64, 16);
        let cfg_b = restored.to_regional_config(64, 16);
        assert_eq!(cfg_a.regions.len(), cfg_b.regions.len());
        for i in 0..cfg_a.regions.len() {
            assert_eq!(cfg_a.regions[i].d_model, cfg_b.regions[i].d_model);
            assert_eq!(cfg_a.regions[i].memory_length, cfg_b.regions[i].memory_length);
        }
    }

    #[test]
    fn sample_valid_terminates() {
        let mut rng = SimpleRng::new(0xC0FFEE);
        let result = BrainArch::sample_valid(&mut rng, 200);
        assert!(result.is_some(), "200 attempts insufficient — unexpected");
        let (arch, _attempts) = result.unwrap();
        assert_eq!(arch.is_valid(), Ok(()));
    }

    #[test]
    fn random_sample_total_d_model_within_cap() {
        // Even the largest possible draw must respect MAX_TOTAL_D_MODEL
        // when reported via is_valid; check that invalid-by-d_model is
        // raised cleanly when constructed pathologically.
        let big = PerRegionShape {
            d_model: 1024, memory_length: 64, deep_nlms: true,
            beta: 0.1, iterations: 4,
        };
        let mut arch = BrainArch::eight_region_billion_arch();
        arch.regions = [big; N_REGIONS];
        // 8 × 1024 = 8192 — exactly at the limit, should pass.
        assert_eq!(arch.is_valid(), Ok(()));

        // Bump one to 1024+1 (off-grid but still tests the cap)
        arch.regions[INSULA].d_model = 1025;
        assert!(matches!(arch.is_valid(), Err(ArchError::TotalDModelTooLarge { .. })));
    }
}
