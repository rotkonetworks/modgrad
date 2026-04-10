//! Head-to-head: zeratul CTM vs isis brain on 7×7 maze.
//!
//! Both use frozen random weights + LS readout (reservoir computing).
//! This shows how much the architecture matters, holding the readout constant.
//!
//! cargo test --release --test ctm_comparison -- --nocapture

use modgrad::ctm::{Ctm, CtmConfig, CtmWeights, CtmSession, LayerConfig, forward_split, SimpleRng};
use modgrad::tasks;
use modgrad::accuracy::ls_accuracy;
use modgrad::linalg;
use rayon::prelude::*;

// ─── Minimal CTM (from zeratul/poker-pvm/cfr/ctm_native.rs) ───────────

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

struct MiniCTM {
    // step_net: concat(input, hidden) → Linear → GELU → Linear → GELU
    step0_w: Vec<f32>, step0_b: Vec<f32>,
    step1_w: Vec<f32>, step1_b: Vec<f32>,
    init_hidden: Vec<f32>,
    // sync accumulator
    sync_left: Vec<usize>,
    sync_right: Vec<usize>,
    decay: Vec<f32>,
    // dims
    input_dim: usize,
    hidden_dim: usize,
    n_sync: usize,
    k: usize,
}

struct MiniCTMOutput {
    hidden: Vec<f32>,
    sync: Vec<f32>,
}

impl MiniCTM {
    fn new(input_dim: usize, hidden_dim: usize, n_sync: usize, k: usize, seed: u64) -> Self {
        let mut rng = seed;
        let mut next = || -> f32 {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            ((rng as u32) as f32 / u32::MAX as f32 - 0.5) * 0.1
        };

        let concat_dim = input_dim + hidden_dim;
        let step0_w: Vec<f32> = (0..concat_dim * hidden_dim).map(|_| next()).collect();
        let step0_b: Vec<f32> = (0..hidden_dim).map(|_| next() * 0.01).collect();
        let step1_w: Vec<f32> = (0..hidden_dim * hidden_dim).map(|_| next()).collect();
        let step1_b: Vec<f32> = (0..hidden_dim).map(|_| next() * 0.01).collect();
        let init_hidden: Vec<f32> = (0..hidden_dim).map(|_| next() * 0.01).collect();
        let decay: Vec<f32> = (0..n_sync).map(|_| 1.0 + next()).collect();

        let sync_left: Vec<usize> = (0..n_sync).map(|i| {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            (rng as usize) % hidden_dim
        }).collect();
        let sync_right: Vec<usize> = (0..n_sync).map(|i| {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            (rng as usize) % hidden_dim
        }).collect();

        Self { step0_w, step0_b, step1_w, step1_b, init_hidden,
               sync_left, sync_right, decay, input_dim, hidden_dim, n_sync, k }
    }

    fn forward(&self, input: &[f32]) -> MiniCTMOutput {
        let mut h = self.init_hidden.clone();
        let r: Vec<f32> = self.decay.iter().map(|d| (-d.clamp(0.0, 15.0)).exp()).collect();
        let mut alpha = vec![0.0f32; self.n_sync];
        let mut beta = vec![1.0f32; self.n_sync];

        for _step in 0..self.k {
            // concat(input, h)
            let mut concat = Vec::with_capacity(self.input_dim + self.hidden_dim);
            concat.extend_from_slice(&input[..self.input_dim.min(input.len())]);
            concat.resize(self.input_dim, 0.0);
            concat.extend_from_slice(&h);

            // Linear0 → GELU
            let mut h_new = self.step0_b.clone();
            for i in 0..self.hidden_dim {
                let row = i * (self.input_dim + self.hidden_dim);
                for j in 0..concat.len().min(self.input_dim + self.hidden_dim) {
                    h_new[i] += self.step0_w[row + j] * concat[j];
                }
                h_new[i] = gelu(h_new[i]);
            }

            // Linear1 → GELU
            let h_mid = h_new.clone();
            h_new = self.step1_b.clone();
            for i in 0..self.hidden_dim {
                let row = i * self.hidden_dim;
                for j in 0..self.hidden_dim {
                    h_new[i] += self.step1_w[row + j] * h_mid[j];
                }
                h_new[i] = gelu(h_new[i]);
            }
            h = h_new;

            // Sync accumulation
            for i in 0..self.n_sync {
                let left = h[self.sync_left[i] % self.hidden_dim];
                let right = h[self.sync_right[i] % self.hidden_dim];
                alpha[i] = r[i] * alpha[i] + left * right;
                beta[i] = r[i] * beta[i] + 1.0;
            }
        }

        let sync: Vec<f32> = alpha.iter().zip(beta.iter())
            .map(|(a, b)| a / b.sqrt().max(1e-8))
            .collect();

        MiniCTMOutput { hidden: h, sync }
    }

    /// Total feature dimension: hidden + sync (what LS readout trains on)
    fn feature_dim(&self) -> usize { self.hidden_dim + self.n_sync }
}

#[test]
fn ctm_vs_isis_maze() {
    let d_input = 64;
    let data = tasks::maze_examples(7, d_input, 4000);
    let train = &data[..3000];
    let test = &data[3000..];

    eprintln!("\n  === 7×7 MAZE: Zeratul CTM vs isis Brain (LS readout) ===\n");

    // ── Zeratul-style CTM (single hidden state + sync) ──
    // Match isis total neuron count: 128+64+64+4 = 260 → hidden=256, sync=64
    for &(hidden, sync, k, label) in &[
        (256, 64, 8, "zeratul CTM (256h, 64sync, 8 ticks)"),
        (128, 64, 8, "zeratul CTM (128h, 64sync, 8 ticks)"),
        (512, 128, 8, "zeratul CTM (512h, 128sync, 8 ticks)"),
    ] {
        let ctm = MiniCTM::new(d_input, hidden, sync, k, 42);
        let t0 = std::time::Instant::now();

        let train_feats: Vec<(Vec<f32>, usize)> = train.par_iter().map(|ex| {
            let out = ctm.forward(&ex.input);
            let mut feat = out.hidden;
            feat.extend_from_slice(&out.sync);
            (feat, ex.target)
        }).collect();

        let test_feats: Vec<(Vec<f32>, usize)> = test.par_iter().map(|ex| {
            let out = ctm.forward(&ex.input);
            let mut feat = out.hidden;
            feat.extend_from_slice(&out.sync);
            (feat, ex.target)
        }).collect();

        let train_acc = ls_accuracy(&train_feats, 4);
        let test_acc = ls_accuracy(&test_feats, 4);
        let elapsed = t0.elapsed().as_secs_f64();
        let n_params: usize = ctm.step0_w.len() + ctm.step0_b.len()
            + ctm.step1_w.len() + ctm.step1_b.len() + ctm.init_hidden.len();

        eprintln!("  {}", label);
        eprintln!("    features: {}  params: {}  time: {:.1}s", ctm.feature_dim(), n_params, elapsed);
        eprintln!("    train LS: {:.1}%  test LS: {:.1}%\n", train_acc * 100.0, test_acc * 100.0);
    }

    // ── isis brain (8 regions, NLM, salience) ──
    for &(neurons, ticks, label) in &[
        (64, 8, "isis brain (4×64, 8 ticks)"),
        (128, 8, "isis brain (128+3×64, 8 ticks)"),
    ] {
        let cfg = CtmConfig {
            iterations: ticks, d_input, n_sync_out: 64,
            input_layer: LayerConfig { n_neurons: if neurons == 128 { 128 } else { 64 }, ..Default::default() },
            attention_layer: LayerConfig { n_neurons: 64, ..Default::default() },
            output_layer: LayerConfig { n_neurons: 64, ..Default::default() },
            motor_layer: LayerConfig { n_neurons: 4, ..Default::default() },
            ..CtmConfig::default()
        };

        let ctm = Ctm::new(cfg.clone());
        let (weights, _) = ctm.into_split();
        let proprio = vec![0.0f32; d_input];
        let t0 = std::time::Instant::now();

        let act_dim = weights.init_tick_state().activations.len();

        let train_feats: Vec<(Vec<f32>, usize)> = train.par_iter().map(|ex| {
            let mut s = CtmSession::new(&weights.config);
            let mut t = weights.init_tick_state();
            let _ = forward_split(&weights, &mut s, &mut t, &ex.input, &proprio, false);
            (t.activations.clone(), ex.target)
        }).collect();

        let test_feats: Vec<(Vec<f32>, usize)> = test.par_iter().map(|ex| {
            let mut s = CtmSession::new(&weights.config);
            let mut t = weights.init_tick_state();
            let _ = forward_split(&weights, &mut s, &mut t, &ex.input, &proprio, false);
            (t.activations.clone(), ex.target)
        }).collect();

        let train_acc = ls_accuracy(&train_feats, 4);
        let test_acc = ls_accuracy(&test_feats, 4);
        let elapsed = t0.elapsed().as_secs_f64();
        let n_params: usize = weights.synapse_refs().iter()
            .map(|s| s.linear.weight.len()).sum();

        eprintln!("  {}", label);
        eprintln!("    features: {}  params: {}  time: {:.1}s", act_dim, n_params, elapsed);
        eprintln!("    train LS: {:.1}%  test LS: {:.1}%\n", train_acc * 100.0, test_acc * 100.0);
    }

    // ── isis brain with wake/sleep motor learning ──
    eprintln!("  isis brain (128+3×64, 8 ticks) + wake/sleep motor learning");
    eprintln!("    (see maze_learn::wake_sleep_maze for full results)");
    eprintln!("    best motor accuracy: 56.0% (+27.8pp from random)");
    eprintln!("    note: motor learning uses reward only, no labels\n");

    eprintln!("  Random baseline: 25.0%\n");
}
