//! QEC-localized learning: use error correction to route learning signals.
//!
//! The brain's 8 regions form a computation graph. When the brain makes an error,
//! we treat the per-region activation anomalies as a "syndrome" and use MWPM
//! (via fusion-blossom) to localize which synapses are most responsible.
//!
//! This replaces uniform dopamine broadcasting with targeted error correction:
//!   - Matched synapses get high learning rate (they caused the error)
//!   - Unmatched synapses get low/zero learning rate (leave them alone)
//!
//! Biologically: this models the VTA's differential dopamine projections to
//! cortical areas based on which area's prediction was most violated.

use fusion_blossom::util::{SolverInitializer, SyndromePattern, VertexIndex, Weight};
use fusion_blossom::fusion_mwpm;

/// The brain's computation graph for error localization.
/// 8 regions (vertices), 8 synapses (edges).
pub struct BrainGraph {
    /// Number of vertices (regions)
    n_regions: usize,
    /// Edges: (source_region, target_region, synapse_index)
    edges: Vec<(usize, usize, usize)>,
    /// Solver initializer (cached, reusable across forward passes)
    initializer: SolverInitializer,
}

/// Per-synapse learning weight from error localization.
/// 0.0 = don't update this synapse, 1.0 = full update.
pub type ErrorLocalization = [f32; 8];

impl BrainGraph {
    /// Build the brain's computation graph.
    /// Edges follow the synapse connectivity:
    ///   0: motor → input (syn_motor_input)
    ///   1: input → attention (syn_input_attn)
    ///   2: attention → output (syn_attn_output)
    ///   3: output → motor (syn_output_motor)
    ///   4: motor → cerebellum (syn_cerebellum)
    ///   5: output → basal_ganglia (syn_basal_ganglia)
    ///   6: proprio → insula (syn_insula)
    ///   7: cortical → hippocampus (syn_hippocampus)
    pub fn new() -> Self {
        let n_regions = 8;
        // Region indices: 0=input, 1=attention, 2=output, 3=motor,
        //                 4=cerebellum, 5=basal_ganglia, 6=insula, 7=hippocampus
        let edges = vec![
            (3, 0, 0), // motor → input
            (0, 1, 1), // input → attention
            (1, 2, 2), // attention → output
            (2, 3, 3), // output → motor
            (3, 4, 4), // motor → cerebellum
            (2, 5, 5), // output → basal_ganglia
            (3, 6, 6), // motor → insula (via proprioception path)
            (0, 7, 7), // input → hippocampus
        ];

        // Build fusion-blossom initializer
        // Weighted edges: weight = 1 (uniform initially, refined by activation magnitudes)
        let weighted_edges: Vec<(VertexIndex, VertexIndex, Weight)> = edges.iter()
            .map(|&(src, dst, _)| (src as VertexIndex, dst as VertexIndex, 2 as Weight))
            .collect();

        let initializer = SolverInitializer::new(
            n_regions as VertexIndex,
            weighted_edges,
            vec![], // no virtual vertices in the brain graph
        );

        Self { n_regions, edges, initializer }
    }

    /// Given per-region activation anomalies (defects), find which synapses
    /// are most responsible for the error using MWPM.
    ///
    /// `region_anomaly[i]` = how anomalous region i's activation was this tick.
    /// High anomaly = that region's prediction was violated.
    ///
    /// Returns per-synapse learning weight: 1.0 = matched (update this synapse),
    /// 0.0 = not matched (leave it alone).
    pub fn localize_error(&self, region_anomaly: &[f32; 8]) -> ErrorLocalization {
        // Find defect vertices: regions with anomaly above threshold
        let threshold = 0.1;
        let defects: Vec<VertexIndex> = region_anomaly.iter().enumerate()
            .filter(|(_, a)| **a > threshold)
            .map(|(i, _)| i as VertexIndex)
            .collect();

        // MWPM needs even number of defects
        if defects.len() < 2 {
            // Not enough defects to match — spread learning uniformly
            return [1.0; 8];
        }

        // Rebuild initializer with weighted edges based on activation magnitudes
        // Higher anomaly on endpoints → lower edge weight → more likely to be matched
        let weighted_edges: Vec<(VertexIndex, VertexIndex, Weight)> = self.edges.iter()
            .map(|&(src, dst, _syn_idx)| {
                // Edge weight = inverse of endpoint anomalies
                // Low weight = both endpoints anomalous = likely error path
                let src_anom = region_anomaly[src].max(0.01);
                let dst_anom = region_anomaly[dst].max(0.01);
                let weight = (1000.0 / (src_anom * dst_anom)) as Weight;
                // fusion-blossom requires even weights
                let w_even = (weight.max(2) / 2) * 2;
                (src as VertexIndex, dst as VertexIndex, w_even)
            })
            .collect();

        let init = SolverInitializer::new(
            self.n_regions as VertexIndex,
            weighted_edges,
            vec![],
        );

        // Run MWPM via fusion-blossom
        // Returns matched vertex pairs (not edges)
        let syndrome = SyndromePattern::new_vertices(defects);
        let matched_vertices = fusion_mwpm(&init, &syndrome);

        // Map matched vertex pairs back to synapse indices.
        // matched_vertices is a flat list where pairs are (matched[0], matched[1]), etc.
        let mut localization = [0.1f32; 8]; // baseline: low learning for all
        for pair in matched_vertices.chunks(2) {
            if pair.len() == 2 {
                let v1 = pair[0] as usize;
                let v2 = pair[1] as usize;
                // Find which synapse connects v1 and v2
                for &(src, dst, syn_idx) in &self.edges {
                    if (src == v1 && dst == v2) || (src == v2 && dst == v1) {
                        localization[syn_idx] = 1.0;
                    }
                }
            }
        }

        localization
    }

    /// Compute per-region activation anomaly from tick state.
    /// Anomaly = deviation from running mean (surprise signal per region).
    pub fn compute_anomaly(
        activations: &[f32],
        act_offsets: &[usize; 8],
        act_sizes: &[usize; 8],
        baseline_means: &[f32; 8],
    ) -> [f32; 8] {
        let mut anomaly = [0.0f32; 8];
        for r in 0..8 {
            let off = act_offsets[r];
            let sz = act_sizes[r];
            if sz == 0 { continue; }
            let mean: f32 = activations[off..off + sz].iter().sum::<f32>() / sz as f32;
            let var: f32 = activations[off..off + sz].iter()
                .map(|x| (x - mean).powi(2)).sum::<f32>() / sz as f32;
            // Anomaly = how different this region is from its baseline
            anomaly[r] = (mean - baseline_means[r]).abs() + var.sqrt();
        }
        anomaly
    }
}
