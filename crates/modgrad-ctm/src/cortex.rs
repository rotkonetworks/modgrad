//! Hierarchical cortex: many small 1-tick regions in a pipeline.
//!
//! The neocortex is a sheet of cortical columns. Each column does ONE
//! feedforward pass. Depth comes from the pipeline, not from ticking
//! the same column repeatedly.
//!
//! Architecture:
//!   Level 0: Primary sensory (many small columns, parallel)
//!   Level 1: Secondary processing (fewer, combine level 0)
//!   Level 2: Association (fewer still, integrate)
//!   Level 3: Executive (few, decide)
//!   Level 4: Motor (output)
//!   Feedback: executive → sensory (top-down prediction)
//!
//! Each level fires in parallel. Levels fire sequentially.
//! One forward pass through all levels = one gamma cycle.
//! Feedback creates the next cycle's context.
//! Deliberation depth = number of gamma cycles.

use modgrad_compute::neuron::Linear;
use crate::config::LayerConfig;
use crate::neuron::NeuronLayerWeights;

// ─── Column: the basic compute unit ──────────────────────

/// A cortical column: one feedforward pass.
/// Equivalent to NeuronLayerWeights but with a synapse built in.
#[derive(Debug, Clone)]
pub struct Column {
    pub name: String,
    pub level: usize,
    pub synapse: Linear,    // input → 2×n_neurons (for GLU)
    pub nlm: NeuronLayerWeights,
    pub n_in: usize,
    pub n_out: usize,
}

impl Column {
    pub fn new(name: &str, level: usize, n_in: usize, n_neurons: usize, memory_length: usize) -> Self {
        let config = LayerConfig {
            n_neurons,
            memory_length,
            nlm_depth: 1,
            inhibitory_fraction: 0.3,
            ..Default::default()
        };
        Self {
            name: name.into(),
            level,
            synapse: Linear::new(n_in, n_neurons * 2), // ×2 for GLU
            nlm: NeuronLayerWeights::new(&config),
            n_in,
            n_out: n_neurons,
        }
    }

    /// One tick: synapse → GLU → SiLU → NLM → activation.
    pub fn forward(&self, input: &[f32], trace: &mut Vec<f32>) -> Vec<f32> {
        let raw = self.synapse.forward(input);
        let signal = glu_silu(&raw);
        self.nlm.step(&signal, trace)
    }
}

/// GLU + SiLU activation (same as ctm.rs Synapse::forward).
fn glu_silu(raw: &[f32]) -> Vec<f32> {
    let half = raw.len() / 2;
    let mut out = Vec::with_capacity(half);
    for i in 0..half {
        let gate = 1.0 / (1.0 + (-raw[half + i]).exp());
        let glu = raw[i] * gate;
        let silu_sig = 1.0 / (1.0 + (-glu).exp());
        out.push(glu * silu_sig);
    }
    out
}

// ─── Cortex: hierarchical pipeline ───────────────────────

/// Connection between columns.
#[derive(Debug, Clone)]
pub struct Projection {
    pub from: usize,  // column index
    pub to: usize,    // column index
}

/// Feedback connection (higher level → lower level).
#[derive(Debug, Clone)]
pub struct Feedback {
    pub from: usize,
    pub to: usize,
    pub transform: Linear,  // project feedback dimensions
}

/// The cortex: a hierarchical pipeline of 1-tick columns.
#[derive(Debug, Clone)]
pub struct Cortex {
    pub columns: Vec<Column>,
    pub projections: Vec<Projection>,
    pub feedback: Vec<Feedback>,
    pub n_levels: usize,
    /// Which columns receive external input.
    pub input_columns: Vec<usize>,
    /// Which columns produce the final output.
    pub output_columns: Vec<usize>,
}

/// Per-column mutable state (traces).
#[derive(Debug, Clone)]
pub struct CortexState {
    pub traces: Vec<Vec<f32>>,
    pub activations: Vec<Vec<f32>>,
}

impl Cortex {
    /// Build a QEC-optimized cortex for syndrome decoding.
    /// Hierarchical: syndrome_bits → local_parity → patterns → hypothesis → classify
    pub fn for_qec(syndrome_dim: usize, n_classes: usize) -> Self {
        let mut columns = Vec::new();
        let mut projections = Vec::new();

        // Level 0: one column per syndrome bit (or pair)
        let n_level0 = syndrome_dim;
        let col_size = 8;
        let mem_len = 4;
        for i in 0..n_level0 {
            columns.push(Column::new(
                &format!("s{i}"), 0, 1, col_size, mem_len));
        }

        // Level 1: local parity — each sees 2-3 adjacent syndrome columns
        let n_level1 = (n_level0 + 1) / 2;
        for i in 0..n_level1 {
            let col_idx = columns.len();
            columns.push(Column::new(
                &format!("p{i}"), 1, col_size * 2, col_size, mem_len));
            // Connect from 2 adjacent level-0 columns
            let from1 = i * 2;
            let from2 = (i * 2 + 1).min(n_level0 - 1);
            projections.push(Projection { from: from1, to: col_idx });
            projections.push(Projection { from: from2, to: col_idx });
        }

        // Level 2: pattern recognition — each sees 2-3 level-1 columns
        let n_level2 = (n_level1 + 1) / 2;
        let level1_start = n_level0;
        for i in 0..n_level2 {
            let col_idx = columns.len();
            columns.push(Column::new(
                &format!("r{i}"), 2, col_size * 2, col_size * 2, mem_len));
            let from1 = level1_start + i * 2;
            let from2 = level1_start + (i * 2 + 1).min(n_level1 - 1);
            projections.push(Projection { from: from1, to: col_idx });
            projections.push(Projection { from: from2, to: col_idx });
        }

        // Level 3: hypothesis — sees all level-2 columns
        let n_level3 = 3;
        let level2_start = n_level0 + n_level1;
        let level2_total_out = n_level2 * col_size * 2;
        for i in 0..n_level3 {
            let col_idx = columns.len();
            columns.push(Column::new(
                &format!("h{i}"), 3, level2_total_out, col_size * 2, mem_len));
            // Connect from ALL level-2 columns
            for j in 0..n_level2 {
                projections.push(Projection { from: level2_start + j, to: col_idx });
            }
        }

        // Level 4: classify — sees all hypothesis columns
        let level3_start = n_level0 + n_level1 + n_level2;
        let level3_total_out = n_level3 * col_size * 2;
        let classify_idx = columns.len();
        columns.push(Column::new(
            "classify", 4, level3_total_out, n_classes * 4, mem_len));
        for j in 0..n_level3 {
            projections.push(Projection { from: level3_start + j, to: classify_idx });
        }

        let n_levels = 5;
        let input_columns: Vec<usize> = (0..n_level0).collect();
        let output_columns = vec![classify_idx];

        // Feedback: hypothesis → level 1 (top-down prediction)
        let mut feedback = Vec::new();
        for i in 0..n_level3 {
            for j in 0..n_level1 {
                feedback.push(Feedback {
                    from: level3_start + i,
                    to: level1_start + j,
                    transform: Linear::new(col_size * 2, col_size * 2),
                });
            }
        }

        let total = columns.len();
        eprintln!("[cortex] {} columns across {} levels: {}→{}→{}→{}→1",
            total, n_levels, n_level0, n_level1, n_level2, n_level3);

        Self { columns, projections, feedback, n_levels, input_columns, output_columns }
    }

    /// Initialize mutable state for one forward pass.
    pub fn init_state(&self) -> CortexState {
        CortexState {
            traces: self.columns.iter()
                .map(|c| c.nlm.start_trace.clone())
                .collect(),
            activations: self.columns.iter()
                .map(|c| vec![0.0f32; c.n_out])
                .collect(),
        }
    }

    /// Run one gamma cycle: forward through all levels.
    /// Returns the output columns' activations concatenated.
    pub fn forward(&self, observation: &[f32], state: &mut CortexState) -> Vec<f32> {
        // Level 0: each input column gets a slice of observation
        // Split observation evenly across input columns
        let _n_inputs = self.input_columns.len().max(1);
        let chunk_size = self.columns.get(0).map(|c| c.n_in).unwrap_or(1);
        for (i, &col_idx) in self.input_columns.iter().enumerate() {
            let start = i * chunk_size;
            let end = (start + chunk_size).min(observation.len());
            let input: Vec<f32> = if start < observation.len() {
                let mut v = observation[start..end].to_vec();
                while v.len() < chunk_size { v.push(0.0); }
                v
            } else {
                vec![0.0; chunk_size]
            };
            state.activations[col_idx] = self.columns[col_idx].forward(
                &input, &mut state.traces[col_idx]);
        }

        // Levels 1+: each column concatenates its upstream activations
        for level in 1..self.n_levels {
            // Collect columns at this level
            let level_cols: Vec<usize> = self.columns.iter().enumerate()
                .filter(|(_, c)| c.level == level)
                .map(|(i, _)| i)
                .collect();

            for &col_idx in &level_cols {
                // Concatenate upstream activations
                let mut input = Vec::new();
                for proj in &self.projections {
                    if proj.to == col_idx {
                        input.extend_from_slice(&state.activations[proj.from]);
                    }
                }

                // Add feedback if any (from previous cycle)
                for fb in &self.feedback {
                    if fb.to == col_idx {
                        let fb_act = fb.transform.forward(&state.activations[fb.from]);
                        // Additive: feedback modulates, doesn't replace
                        for (j, v) in fb_act.iter().enumerate() {
                            if j < input.len() {
                                input[j] += v * 0.1; // soft feedback
                            }
                        }
                    }
                }

                state.activations[col_idx] = self.columns[col_idx].forward(
                    &input, &mut state.traces[col_idx]);
            }
        }

        // Collect output
        let mut output = Vec::new();
        for &col_idx in &self.output_columns {
            output.extend_from_slice(&state.activations[col_idx]);
        }
        output
    }

    /// Run N gamma cycles (with feedback refinement).
    pub fn forward_cycles(&self, observation: &[f32], state: &mut CortexState, n_cycles: usize) -> Vec<f32> {
        let mut output = Vec::new();
        for _ in 0..n_cycles {
            output = self.forward(observation, state);
        }
        output
    }

    /// Two-phase forward: feedforward sweep + sustained binding with sync.
    ///
    /// Phase 1 (feedforward sweep):
    ///   Level-by-level pipeline. Each column fires once. Fast.
    ///   Gives every column an initial activation.
    ///
    /// Phase 2 (sustained binding):
    ///   All columns re-fire simultaneously for N binding ticks.
    ///   Each column reads its upstream's CURRENT activation (not waiting).
    ///   Sync accumulator integrates pairwise correlations across columns.
    ///   This is where temporal binding happens — cofiring creates sync.
    ///
    /// Returns the sync signal (not raw activations).
    pub fn forward_with_binding(
        &self,
        observation: &[f32],
        state: &mut CortexState,
        sync: &mut SyncState,
        binding_ticks: usize,
    ) -> Vec<f32> {
        // Phase 1: feedforward sweep (one pass through levels)
        self.forward(observation, state);

        // Phase 2: sustained binding — all columns active simultaneously
        for _tick in 0..binding_ticks {
            // Every column re-fires using current activations from its upstreams
            // (not level-by-level — ALL columns fire in this tick)
            let prev_activations = state.activations.clone();

            for col_idx in 0..self.columns.len() {
                let col = &self.columns[col_idx];

                if col.level == 0 {
                    // Input columns re-read observation + feedback
                    let chunk_size = col.n_in;
                    let i = self.input_columns.iter().position(|&c| c == col_idx).unwrap_or(0);
                    let start = i * chunk_size;
                    let end = (start + chunk_size).min(observation.len());
                    let mut input: Vec<f32> = if start < observation.len() {
                        let mut v = observation[start..end].to_vec();
                        while v.len() < chunk_size { v.push(0.0); }
                        v
                    } else {
                        vec![0.0; chunk_size]
                    };

                    // Add feedback from sustained activity
                    for fb in &self.feedback {
                        if fb.to == col_idx {
                            let fb_act = fb.transform.forward(&prev_activations[fb.from]);
                            for (j, v) in fb_act.iter().enumerate() {
                                if j < input.len() { input[j] += v * 0.1; }
                            }
                        }
                    }

                    state.activations[col_idx] = col.forward(&input, &mut state.traces[col_idx]);
                } else {
                    // Non-input columns read from upstream's CURRENT activation
                    let mut input = Vec::new();
                    for proj in &self.projections {
                        if proj.to == col_idx {
                            input.extend_from_slice(&prev_activations[proj.from]);
                        }
                    }

                    // Feedback
                    for fb in &self.feedback {
                        if fb.to == col_idx {
                            let fb_act = fb.transform.forward(&prev_activations[fb.from]);
                            for (j, v) in fb_act.iter().enumerate() {
                                if j < input.len() { input[j] += v * 0.1; }
                            }
                        }
                    }

                    // Pad/truncate to expected input size
                    while input.len() < col.n_in { input.push(0.0); }
                    input.truncate(col.n_in);

                    state.activations[col_idx] = col.forward(&input, &mut state.traces[col_idx]);
                }
            }

            // Sync accumulator: measure cofiring across ALL columns
            // Concatenate all activations into one vector
            let all_acts: Vec<f32> = state.activations.iter().flat_map(|a| a.iter().copied()).collect();
            sync.update(&all_acts);
        }

        // Return sync signal
        sync.output()
    }
}

/// Sync state for the cortex: tracks cofiring across binding ticks.
/// Uses the same exponential decay pairwise product as the CTM.
#[derive(Debug, Clone)]
pub struct SyncState {
    pub n_pairs: usize,
    pub indices_left: Vec<usize>,
    pub indices_right: Vec<usize>,
    pub alpha: Vec<f32>,
    pub beta: Vec<f32>,
    pub decay: Vec<f32>,
    pub initialized: bool,
}

impl SyncState {
    pub fn new(n_pairs: usize, n_neurons: usize) -> Self {
        let mut rng_state = (n_pairs as u64).wrapping_mul(7919);
        let simple_rand = |state: &mut u64| -> u64 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *state
        };
        let indices_left: Vec<usize> = (0..n_pairs).map(|_| simple_rand(&mut rng_state) as usize % n_neurons).collect();
        let indices_right: Vec<usize> = (0..n_pairs).map(|_| simple_rand(&mut rng_state) as usize % n_neurons).collect();
        let decay: Vec<f32> = (0..n_pairs).map(|i| {
            0.5 + 0.4 * ((i as f32 / n_pairs as f32) * std::f32::consts::PI).sin()
        }).collect();

        Self {
            n_pairs,
            indices_left,
            indices_right,
            alpha: vec![0.0; n_pairs],
            beta: vec![0.0; n_pairs],
            decay,
            initialized: false,
        }
    }

    /// Update sync from all column activations (concatenated).
    pub fn update(&mut self, all_activations: &[f32]) {
        let n = all_activations.len();
        for i in 0..self.n_pairs {
            let l = self.indices_left[i] % n;
            let r = self.indices_right[i] % n;
            let pairwise = all_activations[l] * all_activations[r];
            let r_decay = (-self.decay[i].clamp(0.0, 15.0)).exp();

            if !self.initialized {
                self.alpha[i] = pairwise;
                self.beta[i] = 1.0;
            } else {
                self.alpha[i] = r_decay * self.alpha[i] + pairwise;
                self.beta[i] = r_decay * self.beta[i] + 1.0;
            }
        }
        self.initialized = true;
    }

    /// Read the sync signal: alpha / sqrt(beta).
    pub fn output(&self) -> Vec<f32> {
        self.alpha.iter().zip(&self.beta)
            .map(|(a, b)| a / b.sqrt().max(1e-8))
            .collect()
    }
}

impl Cortex {
    /// Total number of parameters across all columns.
    pub fn param_count(&self) -> usize {
        self.columns.iter().map(|c| {
            c.synapse.weight.len() + c.synapse.bias.len()
            + c.nlm.nlm_stage1.weights.len() + c.nlm.nlm_stage1.biases.len()
        }).sum::<usize>()
        + self.feedback.iter().map(|f| f.transform.weight.len() + f.transform.bias.len()).sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cortex_qec() {
        let cortex = Cortex::for_qec(20, 2);
        assert!(cortex.columns.len() > 30); // should have ~38 columns
        assert_eq!(cortex.n_levels, 5);

        let mut state = cortex.init_state();
        let obs = vec![0.0f32; 20];
        let output = cortex.forward(&obs, &mut state);
        assert!(output.len() > 0);

        // With cycles
        let mut state2 = cortex.init_state();
        let output2 = cortex.forward_cycles(&obs, &mut state2, 3);
        assert!(output2.len() > 0);
        // Multiple cycles should give different output than 1 (feedback)
        // (might be same if feedback is zero — that's ok for this test)
    }
}
