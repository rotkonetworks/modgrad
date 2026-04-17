//! Benchmark: does pain-driven training improve learning?
//!
//! Trains two identical models on the same data:
//!   A) baseline: standard next-token prediction
//!   B) pain: next-token + pain bridge (loss→homeostasis, valence on memory)
//!
//! Compares loss curves, accuracy, and convergence speed.

use modgrad_ctm::graph::*;
use modgrad_ctm::bio::homeostasis::Homeostasis;
use modgrad_ctm::bio::neuromod::Neuromodulators;
use modgrad_ctm::bio::pain::{self, PainConfig};
use modgrad_ctm::memory::episodic::{self, EpisodicConfig, EpisodicMemory, ValenceReceipt};

const STEPS: usize = 300;
const LOG_EVERY: usize = 50;

fn main() {
    // Training data: repeating patterns the model should learn
    let data: Vec<Vec<usize>> = vec![
        // Pattern: byte sequences with structure
        b"the cat sat on the mat ".iter().map(|&b| b as usize).collect(),
        b"the dog ran in the park ".iter().map(|&b| b as usize).collect(),
        b"one two three four five ".iter().map(|&b| b as usize).collect(),
        b"aaabbbcccdddeeefffggg ".iter().map(|&b| b as usize).collect(),
        b"hello world hello world ".iter().map(|&b| b as usize).collect(),
        b"0123456789 0123456789 ".iter().map(|&b| b as usize).collect(),
    ];

    eprintln!("=== Pain Training Benchmark ===");
    eprintln!("  {} steps, {} sequences", STEPS, data.len());
    eprintln!();

    // Train baseline
    let (baseline_losses, baseline_acc) = train_baseline(&data);

    eprintln!();

    // Train with pain
    let (pain_losses, pain_acc) = train_with_pain(&data);

    // Compare
    eprintln!("\n=== Results ===");

    let bl_final = avg_last_n(&baseline_losses, 50);
    let pain_final = avg_last_n(&pain_losses, 50);
    let bl_acc_final = avg_last_n(&baseline_acc, 50);
    let pain_acc_final = avg_last_n(&pain_acc, 50);

    eprintln!("  Baseline  — final avg loss: {:.4}, final avg acc: {:.2}%", bl_final, bl_acc_final * 100.0);
    eprintln!("  Pain      — final avg loss: {:.4}, final avg acc: {:.2}%", pain_final, pain_acc_final * 100.0);
    eprintln!();

    if pain_final < bl_final {
        let improvement = ((bl_final - pain_final) / bl_final * 100.0).abs();
        eprintln!("  Pain training improved loss by {:.1}%", improvement);
    } else {
        let degradation = ((pain_final - bl_final) / bl_final * 100.0).abs();
        eprintln!("  Pain training degraded loss by {:.1}%", degradation);
    }

    if pain_acc_final > bl_acc_final {
        let improvement = ((pain_acc_final - bl_acc_final) * 100.0).abs();
        eprintln!("  Pain training improved accuracy by {:.1}pp", improvement);
    } else {
        let degradation = ((bl_acc_final - pain_acc_final) * 100.0).abs();
        eprintln!("  Pain training degraded accuracy by {:.1}pp", degradation);
    }

    // Convergence speed: steps to reach baseline's final loss
    let target = bl_final;
    let steps_pain = pain_losses.iter().position(|&l| l <= target);
    let steps_base = baseline_losses.iter().position(|&l| l <= target);
    if let (Some(sp), Some(sb)) = (steps_pain, steps_base) {
        eprintln!("  Steps to reach {:.3} loss — baseline: {}, pain: {} ({} faster)",
            target, sb, sp, if sp < sb { format!("{}x", sb/sp.max(1)) } else { "not faster".to_string() });
    }
}

fn make_model() -> (RegionalWeights, RegionalGradients) {
    let cfg = RegionalConfig::eight_region_small(32, 256, 4);
    let w = RegionalWeights::new(cfg);
    let grads = RegionalGradients::zeros(&w);
    (w, grads)
}

fn train_baseline(data: &[Vec<usize>]) -> (Vec<f32>, Vec<f32>) {
    eprintln!("--- Training: BASELINE (no pain) ---");
    let (mut w, mut grads) = make_model();
    let mut opt = RegionalAdamW::new(&w).with_lr(3e-4);

    let mut losses = Vec::with_capacity(STEPS);
    let mut accs = Vec::with_capacity(STEPS);

    for step in 0..STEPS {
        let seq = &data[step % data.len()];
        grads.zero();

        let mut step_loss = 0.0f32;
        let mut correct = 0usize;
        let n = seq.len() - 1;

        for pos in 0..n {
            let (loss, pred) = regional_train_token(&w, &mut grads, seq[pos], seq[pos + 1]);
            step_loss += loss;
            if pred == seq[pos + 1] { correct += 1; }
        }

        opt.step(&mut w, &mut grads);
        let avg_loss = step_loss / n as f32;
        let acc = correct as f32 / n as f32;
        losses.push(avg_loss);
        accs.push(acc);

        if step % LOG_EVERY == 0 || step == STEPS - 1 {
            eprintln!("  step {:4}: loss={:.4} acc={}/{} ({:.0}%)",
                step, avg_loss, correct, n, acc * 100.0);
        }
    }

    (losses, accs)
}

fn train_with_pain(data: &[Vec<usize>]) -> (Vec<f32>, Vec<f32>) {
    eprintln!("--- Training: WITH PAIN ---");
    let (mut w, mut grads) = make_model();
    let mut opt = RegionalAdamW::new(&w).with_lr(3e-4);

    let pain_cfg = PainConfig::default();
    let mut homeostasis = Homeostasis::default();
    let mut neuromod = Neuromodulators::default();
    let mut memory = EpisodicMemory::new(EpisodicConfig {
        capacity: 128,
        max_ticks: 4,
        d_model: 32, // matches eight_region_small d_model
        min_ticks_for_storage: 1,
        min_surprise: 0.0,
        retrieval_threshold: 0.5,
        ..Default::default()
    });

    let mut losses = Vec::with_capacity(STEPS);
    let mut accs = Vec::with_capacity(STEPS);

    for step in 0..STEPS {
        let seq = &data[step % data.len()];
        grads.zero();

        let mut step_loss = 0.0f32;
        let mut correct = 0usize;
        let n = seq.len() - 1;

        for pos in 0..n {
            let token = seq[pos];
            let target = seq[pos + 1];

            let (loss, pred) = regional_train_token(&w, &mut grads, token, target);
            let was_correct = pred == target;
            step_loss += loss;
            if was_correct { correct += 1; }

            // Confidence from logit distribution (proxy: inverse loss)
            let confidence = (-loss).exp().clamp(0.0, 1.0);

            // Pain bridge: loss → homeostasis + neuromod
            let response = pain::on_prediction(
                &mut homeostasis, &mut neuromod,
                loss, confidence, was_correct, &pain_cfg,
            );

            // Store with valence receipt
            // Use a simple trajectory proxy (token embedding as flat vector)
            let traj: Vec<f32> = (0..4 * 32).map(|i| ((i + token) as f32 * 0.01).sin()).collect();
            let cert = vec![[1.0 - confidence, confidence]; 4];
            let receipt = ValenceReceipt {
                valence: response.valence_for_storage,
                loss,
                confidence,
                correct: was_correct,
            };
            let (m, _) = episodic::store_with_valence(
                memory, &traj, &cert, &[], 4, loss, Some(receipt),
            );
            memory = m;

            // Retrieve: check if this context was painful before
            let query: Vec<f32> = (0..32).map(|i| ((i + token) as f32 * 0.01).sin()).collect();
            let retrieval = episodic::retrieve(&memory, &query);
            if retrieval.n_matches > 0 {
                let _retrieval_response = pain::on_retrieval(
                    &mut homeostasis, &mut neuromod,
                    retrieval.blended_valence, &pain_cfg,
                );

                // If we overcame a previously painful context
                if was_correct && retrieval.blended_valence < -0.2 {
                    pain::on_overcoming(
                        &mut homeostasis, &mut neuromod,
                        retrieval.blended_valence, true, &pain_cfg,
                    );
                }
            }

            // Neuromod affects effective learning rate
            // High dopamine (surprise/relief) → learn faster
            // Low serotonin (sustained pain) → learn slower (protective)
            let neuromod_lr_scale = (neuromod.dopamine / 1.0).clamp(0.5, 2.0)
                * (neuromod.serotonin / 1.0).clamp(0.5, 1.5);
            opt.lr = 3e-4 * neuromod_lr_scale;
        }

        opt.step(&mut w, &mut grads);
        let avg_loss = step_loss / n as f32;
        let acc = correct as f32 / n as f32;
        losses.push(avg_loss);
        accs.push(acc);

        // Homeostasis tick
        homeostasis.tick_from_ctm(avg_loss, true, avg_loss);

        // Periodic sleep consolidation with reappraisal
        if step > 0 && step % 100 == 0 {
            let (m, outcomes) = episodic::reappraise(memory, &|_| Some(avg_loss), 0.5);
            memory = m;
            let reappraised = outcomes.iter().filter(|o| o.reappraised).count();
            if reappraised > 0 {
                eprintln!("    [sleep] reappraised {} painful memories", reappraised);
            }
            homeostasis.on_sleep(0.8);
        }

        if step % LOG_EVERY == 0 || step == STEPS - 1 {
            eprintln!("  step {:4}: loss={:.4} acc={}/{} ({:.0}%) | pressure={:.2} DA={:.2} 5HT={:.2} NE={:.2}",
                step, avg_loss, correct, n, acc * 100.0,
                homeostasis.pressure, neuromod.dopamine, neuromod.serotonin, neuromod.norepinephrine);
        }
    }

    (losses, accs)
}

fn avg_last_n(v: &[f32], n: usize) -> f32 {
    let start = v.len().saturating_sub(n);
    let slice = &v[start..];
    if slice.is_empty() { return 0.0; }
    slice.iter().sum::<f32>() / slice.len() as f32
}
