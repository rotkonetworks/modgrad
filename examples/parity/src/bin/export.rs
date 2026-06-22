//! Exporter for the in-browser modgrad demo (modgrad.com `/play`).
//!
//! Trains a tiny adaptive-exit parity CTM with the real SDK, then dumps:
//!   - `parity_weights.json` — the full `CtmWeights` (serde), loaded by the
//!     wasm reimplementation.
//!   - `parity_reference.json` — per-tick forward traces on a fixed set of
//!     sequences. The wasm forward is validated bit-for-bit against these
//!     so the browser demo is provably the same math as `ctm_forward`.
//!
//! Usage: `cargo run -p parity --bin export --release -- <out_dir>`

use modgrad_ctm::config::{CtmConfig, ExitStrategy};
use modgrad_ctm::weights::{CtmWeights, CtmState};
use modgrad_ctm::train::{train_step, CtmGradients};
use modgrad_ctm::forward::{ctm_forward, CtmInput};
use serde::Serialize;

const SEQ_LEN: usize = 8;
const TICKS: usize = 8;
const D_MODEL: usize = 64;
const STEPS: usize = 16000;
const LR: f32 = 1.2e-3;
const SEED: u64 = 42;

/// Reproducible LCG, identical to the parity benchmark.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Self(seed) }
    fn next(&mut self) -> u64 {
        self.0 = self.0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn bit(&mut self) -> u8 { (self.next() >> 33) as u8 & 1 }
}

/// Parity sample: bits encoded +1/-1, target = (#ones) mod 2.
fn gen_sample(rng: &mut Rng, seq_len: usize) -> (Vec<f32>, usize) {
    let mut bits = Vec::with_capacity(seq_len);
    let mut ones = 0usize;
    for _ in 0..seq_len {
        let b = rng.bit();
        bits.push(if b == 1 { 1.0 } else { -1.0 });
        ones += b as usize;
    }
    (bits, ones % 2)
}

fn make_config() -> CtmConfig {
    let d_input = D_MODEL / 2;
    CtmConfig {
        iterations: TICKS,
        d_model: D_MODEL,
        d_input,
        heads: 4.min(d_input / 2).max(1),
        n_synch_out: D_MODEL,
        n_synch_action: D_MODEL,
        synapse_depth: 2,
        memory_length: 8,
        deep_nlms: true,
        memory_hidden_dims: 4,
        out_dims: 2,
        n_random_pairing_self: 0,
        min_width: 8,
        exit_strategy: ExitStrategy::None,
        collect_trajectories: true, // populate per-tick activations for the viz
        spatial: None,
    }
}

#[derive(Serialize)]
struct TickTrace {
    prediction: Vec<f32>,    // logits over [even, odd]
    certainty: [f32; 2],     // [normalized_entropy, 1 - it]
    exit_lambda: f32,        // per-tick halting probability (0 if no gate fired)
    activations: Vec<f32>,   // [d_model] activated neuron state at this tick
}

#[derive(Serialize)]
struct SampleTrace {
    bits: Vec<f32>,
    target: usize,
    predicted: usize,
    commit_tick: usize,   // the max-certainty tick the readout is taken from
    ticks_used: usize,
    ticks: Vec<TickTrace>,
}

/// Faithful CTM readout: the tick where the brain is most certain.
fn commit_tick(certainties: &[[f32; 2]]) -> usize {
    (0..certainties.len())
        .max_by(|&a, &b| certainties[a][1].partial_cmp(&certainties[b][1]).unwrap())
        .unwrap_or(0)
}

#[derive(Serialize)]
struct Reference {
    task: String,
    seq_len: usize,
    d_model: usize,
    iterations: usize,
    out_dims: usize,
    eval_acc: f32,
    samples: Vec<SampleTrace>,
}

fn forward_trace(w: &CtmWeights, obs: &[f32], target: usize) -> SampleTrace {
    let d = w.config.d_model;
    let mut state = CtmState::new(w);
    let out = ctm_forward(w, &mut state, CtmInput::Raw {
        obs, n_tokens: SEQ_LEN, raw_dim: 1,
    });
    let mut ticks = Vec::with_capacity(out.ticks_used);
    for t in 0..out.ticks_used {
        let activations = if out.trajectory.len() >= (t + 1) * d {
            out.trajectory[t * d..(t + 1) * d].to_vec()
        } else {
            Vec::new()
        };
        ticks.push(TickTrace {
            prediction: out.predictions[t].clone(),
            certainty: out.certainties[t],
            exit_lambda: out.exit_lambdas.get(t).copied().unwrap_or(0.0),
            activations,
        });
    }
    let ct = commit_tick(&out.certainties);
    let predicted = argmax(&out.predictions[ct]);
    SampleTrace { bits: obs.to_vec(), target, predicted, commit_tick: ct,
        ticks_used: out.ticks_used, ticks }
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn main() {
    let out_dir = std::env::args().nth(1).unwrap_or_else(|| ".".to_string());

    eprintln!("Training parity CTM: seq_len={SEQ_LEN} ticks={TICKS} d_model={D_MODEL} steps={STEPS}");
    let cfg = make_config();
    let mut w = CtmWeights::new(cfg, 1);
    let mut grads = CtmGradients::zeros(&w);
    let mut rng = Rng::new(SEED);

    let mut losses: Vec<f32> = Vec::new();
    let mut corrects: Vec<f32> = Vec::new();
    for step in 0..STEPS {
        let (obs, target) = gen_sample(&mut rng, SEQ_LEN);
        grads.zero();
        let r = train_step(&w, &mut grads, &obs, SEQ_LEN, 1, target);
        // Linear lr decay to 10% — settles the late-training accuracy bounce.
        let lr = LR * (1.0 - 0.9 * step as f32 / STEPS as f32);
        grads.apply(&mut w, lr, 5.0);
        losses.push(r.loss);
        corrects.push(if r.prediction == target { 1.0 } else { 0.0 });
        if step % 1000 == 0 || step == STEPS - 1 {
            let win = 200.min(losses.len());
            let l: f32 = losses[losses.len() - win..].iter().sum::<f32>() / win as f32;
            let a: f32 = corrects[corrects.len() - win..].iter().sum::<f32>() / win as f32;
            eprintln!("  step {step:5}: loss={l:.4} acc={:.1}%", a * 100.0);
        }
    }

    // Held-out eval.
    let mut erng = Rng::new(SEED.wrapping_add(999));
    let mut correct = 0usize;
    let n_eval = 500usize;
    for _ in 0..n_eval {
        let (obs, target) = gen_sample(&mut erng, SEQ_LEN);
        let mut st = CtmState::new(&w);
        let out = ctm_forward(&w, &mut st, CtmInput::Raw { obs: &obs, n_tokens: SEQ_LEN, raw_dim: 1 });
        if argmax(&out.predictions[commit_tick(&out.certainties)]) == target { correct += 1; }
    }
    let eval_acc = correct as f32 / n_eval as f32;
    eprintln!("eval acc: {:.1}%", eval_acc * 100.0);

    // Reference traces on a fixed, reproducible set (the parity oracle).
    let mut trng = Rng::new(7);
    let samples: Vec<SampleTrace> = (0..24)
        .map(|_| { let (obs, target) = gen_sample(&mut trng, SEQ_LEN); forward_trace(&w, &obs, target) })
        .collect();

    let reference = Reference {
        task: "parity".to_string(),
        seq_len: SEQ_LEN,
        d_model: D_MODEL,
        iterations: TICKS,
        out_dims: 2,
        eval_acc,
        samples,
    };

    let weights_path = format!("{out_dir}/parity_weights.json");
    let reference_path = format!("{out_dir}/parity_reference.json");
    std::fs::write(&weights_path, serde_json::to_string(&w).unwrap())
        .expect("write weights");
    std::fs::write(&reference_path, serde_json::to_string_pretty(&reference).unwrap())
        .expect("write reference");
    eprintln!("wrote {weights_path}");
    eprintln!("wrote {reference_path}");
}
