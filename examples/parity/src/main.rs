//! Parity benchmark for CTM — measures adaptive exit vs fixed ticks.
//!
//! Task: given a sequence of '0' and '1' bytes, predict the parity of the
//! entire sequence (even=0, odd=1). Requires scanning all bits — more ticks
//! should help with longer sequences.
//!
//! Usage:
//!   parity --mode baseline --len 8 --ticks 8 --steps 5000
//!   parity --mode adaptive --len 8 --ticks 8 --steps 5000
//!   parity --mode compare --len 8 --ticks 8 --steps 5000

use modgrad_ctm::config::{CtmConfig, ExitStrategy};
use modgrad_ctm::weights::CtmWeights;
use modgrad_ctm::train::{train_step, CtmGradients};
use modgrad_ctm::forward::{ctm_forward, CtmInput};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut mode = "compare".to_string();
    let mut seq_len = 8usize;
    let mut ticks = 8usize;
    let mut steps = 5000usize;
    let mut d_model = 64usize;
    let mut lr = 1e-3f32;
    let mut seed = 42u64;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--mode" => { mode = args[i+1].clone(); i += 2; }
            "--len" => { seq_len = args[i+1].parse().unwrap(); i += 2; }
            "--ticks" => { ticks = args[i+1].parse().unwrap(); i += 2; }
            "--steps" => { steps = args[i+1].parse().unwrap(); i += 2; }
            "--d-model" => { d_model = args[i+1].parse().unwrap(); i += 2; }
            "--lr" => { lr = args[i+1].parse().unwrap(); i += 2; }
            "--seed" => { seed = args[i+1].parse().unwrap(); i += 2; }
            _ => { i += 1; }
        }
    }

    eprintln!("Parity benchmark: seq_len={seq_len} ticks={ticks} d_model={d_model} steps={steps}");

    match mode.as_str() {
        "baseline" => {
            let stats = run_training(seq_len, ticks, d_model, lr, steps, seed, false);
            print_stats("BASELINE", &stats);
        }
        "adaptive" => {
            let stats = run_training(seq_len, ticks, d_model, lr, steps, seed, true);
            print_stats("ADAPTIVE", &stats);
        }
        "compare" => {
            eprintln!("\n═══ RUN 1: BASELINE (ExitStrategy::None) ═══");
            let s1 = run_training(seq_len, ticks, d_model, lr, steps, seed, false);
            print_stats("BASELINE", &s1);

            eprintln!("\n═══ RUN 2: ADAPTIVE (ExitStrategy::AdaptiveGate) ═══");
            let s2 = run_training(seq_len, ticks, d_model, lr, steps, seed, true);
            print_stats("ADAPTIVE", &s2);

            eprintln!("\n═══ COMPARISON ═══");
            eprintln!("  final loss:  baseline={:.4}  adaptive={:.4}",
                s1.final_loss, s2.final_loss);
            eprintln!("  final acc:   baseline={:.1}%  adaptive={:.1}%",
                s1.final_acc * 100.0, s2.final_acc * 100.0);
            eprintln!("  eval acc:    baseline={:.1}%  adaptive={:.1}%",
                s1.eval_acc * 100.0, s2.eval_acc * 100.0);
            if s2.avg_ticks_used > 0.0 {
                eprintln!("  avg ticks:   baseline={:.1}/{ticks}  adaptive={:.1}/{ticks}",
                    ticks as f32, s2.avg_ticks_used);
            }
        }
        _ => { eprintln!("Unknown mode: {mode}. Use baseline, adaptive, or compare."); }
    }
}

struct RunStats {
    final_loss: f32,
    final_acc: f32,
    eval_acc: f32,
    avg_ticks_used: f32,
}

fn print_stats(label: &str, s: &RunStats) {
    eprintln!("  [{label}] final_loss={:.4} final_acc={:.1}% eval_acc={:.1}%",
        s.final_loss, s.final_acc * 100.0, s.eval_acc * 100.0);
    if s.avg_ticks_used > 0.0 {
        eprintln!("  [{label}] avg_ticks_used={:.2}", s.avg_ticks_used);
    }
}

fn make_config(ticks: usize, d_model: usize, adaptive: bool) -> CtmConfig {
    let d_input = d_model / 2;
    CtmConfig {
        iterations: ticks,
        d_model,
        d_input,
        heads: 4.min(d_input / 2).max(1),
        n_synch_out: d_model,
        n_synch_action: d_model,
        synapse_depth: 2,
        memory_length: 8,
        deep_nlms: true,
        memory_hidden_dims: 4,
        out_dims: 2, // even or odd
        n_random_pairing_self: 0,
        min_width: 8,
        exit_strategy: if adaptive {
            ExitStrategy::AdaptiveGate { beta: 0.1, threshold: 0.5 }
        } else {
            ExitStrategy::None
        },
        collect_trajectories: false,
    }
}

/// Simple LCG RNG for reproducibility.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Self(seed) }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    fn bit(&mut self) -> u8 { (self.next() >> 33) as u8 & 1 }
}

/// Generate a parity sample: sequence of 0/1 bytes, target = parity.
fn gen_sample(rng: &mut Rng, seq_len: usize) -> (Vec<f32>, usize) {
    let mut bits = Vec::with_capacity(seq_len);
    let mut count_ones = 0usize;
    for _ in 0..seq_len {
        let b = rng.bit();
        // Encode as +1/-1 (like the Python version)
        bits.push(if b == 1 { 1.0 } else { -1.0 });
        count_ones += b as usize;
    }
    let parity = count_ones % 2; // 0=even, 1=odd
    (bits, parity)
}

fn run_training(
    seq_len: usize, ticks: usize, d_model: usize,
    lr: f32, steps: usize, seed: u64, adaptive: bool,
) -> RunStats {
    let cfg = make_config(ticks, d_model, adaptive);
    let raw_dim = 1; // each token is a single f32
    let n_tokens = seq_len;
    let w_init = CtmWeights::new(cfg, raw_dim);
    let mut w = w_init;
    let mut grads = CtmGradients::zeros(&w);

    let mut rng = Rng::new(seed);
    let mut losses = Vec::new();
    let mut corrects = Vec::new();

    for step in 0..steps {
        let (obs, target) = gen_sample(&mut rng, seq_len);

        grads.zero();
        let result = train_step(&w, &mut grads, &obs, n_tokens, raw_dim, target);
        grads.apply(&mut w, lr, 5.0);

        losses.push(result.loss);
        corrects.push(if result.prediction == target { 1.0f32 } else { 0.0 });

        if step % 500 == 0 || step == steps - 1 {
            let window = 200.min(losses.len());
            let avg_loss: f32 = losses[losses.len()-window..].iter().sum::<f32>() / window as f32;
            let avg_acc: f32 = corrects[corrects.len()-window..].iter().sum::<f32>() / window as f32;
            eprintln!("  step {step:5}: loss={avg_loss:.4} acc={avg_acc:.1}%",
                avg_acc = avg_acc * 100.0);
        }
    }

    // Eval on 500 fresh samples
    let mut eval_rng = Rng::new(seed.wrapping_add(999));
    let mut eval_correct = 0usize;
    let mut total_ticks = 0usize;
    let n_eval = 500;

    for _ in 0..n_eval {
        let (obs, target) = gen_sample(&mut eval_rng, seq_len);
        let mut state = modgrad_ctm::weights::CtmState::new(&w);
        let output = ctm_forward(&w, &mut state, CtmInput::Raw {
            obs: &obs, n_tokens, raw_dim,
        });

        // Use last prediction
        let pred = output.predictions.last().unwrap();
        let pred_class = pred.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
        if pred_class == target { eval_correct += 1; }
        total_ticks += output.ticks_used;
    }

    let window = 200.min(losses.len());
    RunStats {
        final_loss: losses[losses.len()-window..].iter().sum::<f32>() / window as f32,
        final_acc: corrects[corrects.len()-window..].iter().sum::<f32>() / window as f32,
        eval_acc: eval_correct as f32 / n_eval as f32,
        avg_ticks_used: if adaptive { total_ticks as f32 / n_eval as f32 } else { 0.0 },
    }
}
