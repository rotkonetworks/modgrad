//! Genome — the serializable, reproducible specification of a brain.
//!
//! Real biology ships humans with a genome that encodes:
//!   - **Connectome topology** (which cortical regions exist, how they
//!     connect via genetic axon-guidance molecules)
//!   - **Innate filter priors** (retinal cell types, V1 orientation
//!     columns from prenatal retinal waves)
//!   - **Plasticity schedules** (critical periods for each region)
//!   - **Pathway gates** (zero-init scalars for new pathways so they
//!     start silent and grow only when experience proves them useful)
//!
//! Our codebase has all of these scattered across init functions,
//! `RegionalConfig` constructors, env-var gates, and hardcoded
//! constants. `Genome` formalizes them into one serializable artifact:
//! shipping a `Genome` is shipping the developmental program; same
//! Genome bytes → same `express()` output → same training trajectory
//! given the same RNG seeding.
//!
//! ## Reproducibility caveat
//!
//! `Genome::express()` is deterministic. Subsequent training is
//! deterministic *only* if `RAYON_NUM_THREADS=1` — `rayon`'s
//! parallel-for floating-point accumulation order changes with the
//! thread count, producing tiny per-step weight drift. This affects
//! all neural-net code under rayon, not just genome-derived brains.

use modgrad_ctm::graph::RegionalConfig;
use serde::{Deserialize, Serialize};
use wincode_derive::{SchemaRead, SchemaWrite};

use crate::retina::{
    Conv2d, VisualCortex, init_retinal_ganglion_filters,
    init_v1_gabor_filters_with_strength, init_v2_contour_filters,
};

/// The full innate-state specification of a brain.
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub struct Genome {
    pub visual: VisualPriors,
    pub connectome: ConnectomeSpec,
    pub plasticity: PlasticitySchedule,
    pub pathway_gates: PathwayGates,
}

/// Visual cortex initialization — what's "wired at birth" before
/// any image experience.
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub struct VisualPriors {
    pub input_h: usize,
    pub input_w: usize,
    pub retina: RetinaPriorKind,
    pub v1: V1PriorKind,
    pub v2: V2PriorKind,
}

/// Retinal ganglion layer init. The retina layer is fixed in our
/// model — biology wires it via cell-type genetics, never learns it.
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub enum RetinaPriorKind {
    /// 12-channel: 2 luminance ON/OFF + 3 single-color ON-center +
    /// 4 color opponents + 2 wide-field + 1 luminance integrator.
    /// Models the ~30+ ganglion cell types in the human retina.
    StandardGanglion,
}

/// V1 layer init.
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub enum V1PriorKind {
    /// Pure random init — no orientation columns at birth.
    Random,
    /// Channel-selective Gabor hybrid. For each V1 output channel,
    /// REPLACE one retinal-input channel's weights with a unit-L2
    /// Gabor kernel; leave random init on the other 11. Mirrors the
    /// biological "V1 has Gabor priors from prenatal retinal waves
    /// + cell-type genetics, not from postnatal image experience".
    GaborHybrid {
        seed: u64,
        /// Per-9-weight-block L2 magnitude. Default 1/√2 ≈ 0.707.
        /// Stronger Gabor → cleaner orientation selectivity but
        /// risk of saturating V2.
        gabor_l2: f32,
    },
}

/// V2 layer init.
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub enum V2PriorKind {
    /// Pure random — V2 learns contours from experience.
    Random,
    /// 8-orientation collinear-contour + cross detectors.
    AllOrientations,
    /// Cardinal-only (h / v / ±45°) — for axis-aligned tasks.
    CardinalOnly,
}

/// Brain wiring topology — which regions exist, how they connect,
/// and which scales they consume.
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub struct ConnectomeSpec {
    pub topology: TopologyKind,
    pub out_dims: usize,
    pub ticks: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub enum TopologyKind {
    /// 8-region single-scale: every observation-receiving region
    /// consumes V4 tokens. Backward-compatible with pre-multi-scale.
    EightRegionSmall,
    /// 8-region multi-scale: INPUT ← V4, ATTENTION ← V2, MOTOR/
    /// CEREBELLUM ← V1. Skip projections matching real cortex.
    EightRegionSmallMultiscale,
    /// 8-region multi-scale at d_model=512 cortical / 64 subcortical.
    /// Same skip-projection wiring as `EightRegionSmallMultiscale` but
    /// scaled up so SuperLinear stays above the GPU dispatch threshold
    /// — the topology of choice for `--features rocm` benchmarks.
    EightRegionMediumMultiscale,
}

/// Per-region plasticity schedule — how much each region learns.
/// Critical-period biology: not all regions are equally plastic
/// throughout life.
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub struct PlasticitySchedule {
    /// Length must match `n_regions` of the expressed connectome.
    /// `1.0` = full LR (default), `0.0` = frozen, `0.5` = half-rate.
    pub region_lr_mult: Vec<f32>,
    /// Multiplicative LR decay applied each gradient step. `1.0` =
    /// no decay (current default). `<1.0` = critical-period-style
    /// closing — early plasticity drops over training.
    pub critical_period_decay: f32,
}

impl PlasticitySchedule {
    pub fn full(n_regions: usize) -> Self {
        Self {
            region_lr_mult: vec![1.0; n_regions],
            critical_period_decay: 1.0,
        }
    }
}

/// Zero-init scalar gates for "experimental" pathways. From the
/// meditation-ring lesson: any new pathway must start at zero
/// contribution and grow only via verified gradient signal.
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub struct PathwayGates {
    /// Top-down V4→V2→V1 modulation: `gain = 1 + α · (sigmoid(attn) − 0.5)`.
    /// `α = 0` → no top-down (current behavior). `α = 1` → full gating.
    /// Read at the `--topdown` site in `examples/mazes` (CLI override:
    /// `--topdown-alpha`). Save/load round-trip verified by
    /// `pathway_gates_roundtrip`.
    pub topdown_alpha: f32,
    /// Subcortical fast-path → INSULA strength. `0.0` = silent
    /// (default — INSULA stays on its hippocampus-driven path).
    ///
    /// **Reserved — no consumer wired today.** The INSULA-as-amygdala
    /// path was removed from `encode_multiscale` after A/B showed it
    /// hurt eval; this scalar is the gate that future amygdala-region
    /// wiring will read. Today it only round-trips through Genome
    /// save/load (see `pathway_gates_roundtrip`) so that genomes
    /// written with a non-zero value preserve it across reload — once
    /// a consumer lands, no schema migration is needed.
    pub subcortical_alpha: f32,
}

impl Default for PathwayGates {
    fn default() -> Self {
        Self { topdown_alpha: 0.0, subcortical_alpha: 0.0 }
    }
}

/// Result of expressing a Genome — the realized brain.
pub struct ExpressedBrain {
    pub cortex: VisualCortex,
    pub config: RegionalConfig,
}

impl Genome {
    /// Reference "human visual" genome reproducing the codebase's
    /// production setup as of 2026-04-25:
    ///   - Standard ganglion DoG + color-opponent retina
    ///   - V1 Gabor hybrid (seed 0xC0FFEE, gabor_l2 = 1/√2)
    ///   - V2 random (priors hurt on narrow tasks per A/B)
    ///   - Multi-scale 8-region brain (V1/V2/V4 skip wiring)
    ///   - Full plasticity, no pathway-gate experiments
    pub fn default_human_visual(
        input_h: usize,
        input_w: usize,
        out_dims: usize,
        ticks: usize,
    ) -> Self {
        Self {
            visual: VisualPriors {
                input_h,
                input_w,
                retina: RetinaPriorKind::StandardGanglion,
                v1: V1PriorKind::GaborHybrid {
                    seed: 0xC0FFEE,
                    gabor_l2: 1.0 / std::f32::consts::SQRT_2,
                },
                v2: V2PriorKind::Random,
            },
            connectome: ConnectomeSpec {
                topology: TopologyKind::EightRegionSmallMultiscale,
                out_dims,
                ticks,
            },
            plasticity: PlasticitySchedule::full(8),
            pathway_gates: PathwayGates::default(),
        }
    }

    /// Larger sibling of `default_human_visual` — same visual priors
    /// (standard ganglion retina, V1 Gabor hybrid, V2 random) but
    /// expressed onto the medium multi-scale 8-region topology
    /// (cortical d_model=512, subcortical d_model=64). Use this when
    /// you want a brain that actually exercises the GPU under
    /// `--features rocm`; the small variant is too tiny to escape
    /// the matvec/launch-overhead regime.
    pub fn default_human_visual_medium(
        input_h: usize,
        input_w: usize,
        out_dims: usize,
        ticks: usize,
    ) -> Self {
        Self {
            visual: VisualPriors {
                input_h,
                input_w,
                retina: RetinaPriorKind::StandardGanglion,
                v1: V1PriorKind::GaborHybrid {
                    seed: 0xC0FFEE,
                    gabor_l2: 1.0 / std::f32::consts::SQRT_2,
                },
                v2: V2PriorKind::Random,
            },
            connectome: ConnectomeSpec {
                topology: TopologyKind::EightRegionMediumMultiscale,
                out_dims,
                ticks,
            },
            plasticity: PlasticitySchedule::full(8),
            pathway_gates: PathwayGates::default(),
        }
    }

    /// "Blank slate" baseline — random V1, random V2, single-scale
    /// brain. Useful for ablations.
    pub fn blank_slate(
        input_h: usize,
        input_w: usize,
        out_dims: usize,
        ticks: usize,
    ) -> Self {
        Self {
            visual: VisualPriors {
                input_h,
                input_w,
                retina: RetinaPriorKind::StandardGanglion, // retina is always-on DNA
                v1: V1PriorKind::Random,
                v2: V2PriorKind::Random,
            },
            connectome: ConnectomeSpec {
                topology: TopologyKind::EightRegionSmall,
                out_dims,
                ticks,
            },
            plasticity: PlasticitySchedule::full(8),
            pathway_gates: PathwayGates::default(),
        }
    }

    /// Express the genome into a runnable brain. Deterministic
    /// given the genome bytes (modulo `rayon` thread-count).
    pub fn express(&self) -> ExpressedBrain {
        let cortex = self.express_cortex();
        let config = self.express_connectome(&cortex);
        ExpressedBrain { cortex, config }
    }

    fn express_cortex(&self) -> VisualCortex {
        // Build cortex with Conv2d random init (deterministic via
        // fan-in seeded SimpleRng — see Conv2d::new). Then overwrite
        // per the visual priors.
        let mut cortex = VisualCortex::random(
            self.visual.input_h,
            self.visual.input_w,
        );

        // Retina is always genetically wired in this version.
        match self.visual.retina {
            RetinaPriorKind::StandardGanglion => {
                init_retinal_ganglion_filters(&mut cortex.retina);
            }
        }

        match self.visual.v1 {
            V1PriorKind::Random => { /* leave Conv2d random init */ }
            V1PriorKind::GaborHybrid { seed, gabor_l2 } => {
                init_v1_gabor_filters_with_strength(&mut cortex.v1, seed, gabor_l2);
            }
        }

        match self.visual.v2 {
            V2PriorKind::Random => { /* leave Conv2d random init */ }
            V2PriorKind::AllOrientations => {
                init_v2_contour_filters(&mut cortex.v2, false);
            }
            V2PriorKind::CardinalOnly => {
                init_v2_contour_filters(&mut cortex.v2, true);
            }
        }

        cortex
    }

    fn express_connectome(&self, cortex: &VisualCortex) -> RegionalConfig {
        match self.connectome.topology {
            TopologyKind::EightRegionSmall => {
                RegionalConfig::eight_region_small(
                    cortex.v4.out_channels,
                    self.connectome.out_dims,
                    self.connectome.ticks,
                )
            }
            TopologyKind::EightRegionSmallMultiscale => {
                let scale_dims = vec![
                    cortex.v4.out_channels,
                    cortex.v2.out_channels,
                    cortex.v1.out_channels,
                ];
                RegionalConfig::eight_region_small_multiscale(
                    &scale_dims,
                    self.connectome.out_dims,
                    self.connectome.ticks,
                )
            }
            TopologyKind::EightRegionMediumMultiscale => {
                let scale_dims = vec![
                    cortex.v4.out_channels,
                    cortex.v2.out_channels,
                    cortex.v1.out_channels,
                ];
                RegionalConfig::eight_region_medium_multiscale(
                    &scale_dims,
                    self.connectome.out_dims,
                    self.connectome.ticks,
                )
            }
        }
    }

    /// Wincode persistence — same pattern as `VisualCortex::save`.
    pub fn save(&self, path: &str) -> Result<(), String> {
        modgrad_persist::persist::save(self, path).map_err(|e| e.to_string())
    }

    pub fn load(path: &str) -> Result<Self, String> {
        modgrad_persist::persist::load(path).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_human_visual_expresses() {
        let genome = Genome::default_human_visual(21, 21, 25, 3);
        let brain = genome.express();

        // Retina is always 12-channel ganglion bank.
        assert_eq!(brain.cortex.retina.out_channels, 12);
        assert_eq!(brain.cortex.v1.out_channels, 32);

        // Multi-scale topology → obs_scale_dims has 3 entries.
        assert_eq!(brain.config.obs_scale_dims.len(), 3);
        assert_eq!(brain.config.obs_scale_dims[0], 128); // V4
        assert_eq!(brain.config.obs_scale_dims[1], 64);  // V2
        assert_eq!(brain.config.obs_scale_dims[2], 32);  // V1
    }

    #[test]
    fn default_human_visual_medium_expresses() {
        let genome = Genome::default_human_visual_medium(21, 21, 25, 3);
        let brain = genome.express();

        // Retina is always 12-channel ganglion bank.
        assert_eq!(brain.cortex.retina.out_channels, 12);
        assert_eq!(brain.cortex.v1.out_channels, 32);

        // Multi-scale topology → obs_scale_dims has 3 entries
        // exposing the V4/V2/V1 token dims.
        assert_eq!(brain.config.obs_scale_dims.len(), 3);
        assert_eq!(brain.config.obs_scale_dims[0], 128); // V4
        assert_eq!(brain.config.obs_scale_dims[1], 64);  // V2
        assert_eq!(brain.config.obs_scale_dims[2], 32);  // V1

        // Medium topology: 8 regions, cortical d_model=512.
        assert_eq!(brain.config.regions.len(), 8);
        assert_eq!(brain.config.regions[0].d_model, 512); // INPUT
        assert_eq!(brain.config.regions[3].d_model, 512); // MOTOR
        assert_eq!(brain.config.regions[4].d_model, 64);  // CEREBELLUM
        assert_eq!(brain.config.regions[7].d_model, 64);  // HIPPOCAMPUS
    }

    #[test]
    fn blank_slate_uses_single_scale() {
        let genome = Genome::blank_slate(21, 21, 25, 3);
        let brain = genome.express();
        assert_eq!(brain.config.obs_scale_dims, vec![128]); // V4 only
    }

    #[test]
    fn determinism_same_genome_same_weights() {
        // Two expressions of identical genome bytes must produce
        // identical cortex weight values.
        let g1 = Genome::default_human_visual(21, 21, 25, 3);
        let g2 = Genome::default_human_visual(21, 21, 25, 3);
        let b1 = g1.express();
        let b2 = g2.express();

        // V1 weights bit-for-bit equal (Gabor seed is fixed).
        assert_eq!(b1.cortex.v1.weight, b2.cortex.v1.weight);
        // Retina weights fully fixed.
        assert_eq!(b1.cortex.retina.weight, b2.cortex.retina.weight);
    }

    #[test]
    fn save_load_roundtrip() {
        use std::path::PathBuf;
        let genome = Genome::default_human_visual(21, 21, 25, 3);
        let path = std::env::temp_dir()
            .join(format!("genome_test_{}.bin", std::process::id()));
        let path_str = path.to_str().unwrap().to_string();
        genome.save(&path_str).unwrap();
        let loaded = Genome::load(&path_str).unwrap();

        // Express both — weights must match bit-for-bit.
        let original = genome.express();
        let restored = loaded.express();
        assert_eq!(original.cortex.v1.weight, restored.cortex.v1.weight);
        assert_eq!(original.cortex.retina.weight, restored.cortex.retina.weight);
        assert_eq!(original.config.obs_scale_dims, restored.config.obs_scale_dims);

        let _ = std::fs::remove_file(&PathBuf::from(path_str));
    }

    #[test]
    fn pathway_gates_roundtrip() {
        // Default scalars must be zero — the meditation-ring zero-init
        // contract for any new pathway.
        let g0 = Genome::default_human_visual(21, 21, 25, 3);
        assert_eq!(g0.pathway_gates.topdown_alpha, 0.0);
        assert_eq!(g0.pathway_gates.subcortical_alpha, 0.0);
        let gb = Genome::blank_slate(21, 21, 25, 3);
        assert_eq!(gb.pathway_gates.topdown_alpha, 0.0);
        assert_eq!(gb.pathway_gates.subcortical_alpha, 0.0);

        // Custom values must survive a save/load round-trip so a future
        // amygdala consumer for `subcortical_alpha` doesn't need a
        // schema migration when it lands.
        let mut g = Genome::default_human_visual(21, 21, 25, 3);
        g.pathway_gates.topdown_alpha = 0.7;
        g.pathway_gates.subcortical_alpha = 0.42;

        use std::path::PathBuf;
        let path = std::env::temp_dir()
            .join(format!("genome_gates_test_{}.bin", std::process::id()));
        let path_str = path.to_str().unwrap().to_string();
        g.save(&path_str).unwrap();
        let loaded = Genome::load(&path_str).unwrap();
        assert_eq!(loaded.pathway_gates.topdown_alpha, 0.7);
        assert_eq!(loaded.pathway_gates.subcortical_alpha, 0.42);

        let _ = std::fs::remove_file(&PathBuf::from(path_str));
    }
}
