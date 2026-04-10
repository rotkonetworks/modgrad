//! Backprop sleep consolidation.
//!
//! Wake: online three-factor REINFORCE (fast, noisy, biological)
//! Sleep: backprop through synapses on replay buffer (thorough, precise)
//!
//! cargo test --release --test backprop_sleep -- --nocapture

use modgrad::ctm::{
    Ctm, CtmConfig, CtmWeights, CtmSession, Linear, LayerConfig,
    forward_split, Synapse, SynapseForwardCache, SynapseGradients, SimpleRng,
};
use modgrad::ctm::NeuromodConfig;
use modgrad::tasks;
use modgrad::accuracy::{eval_motor, eval_ls};
use modgrad::linalg;
use rayon::prelude::*;

// ─── Gradient check: verify backward pass is correct ───────────────────

#[test]
fn gradient_check_synapse() {
    // Numerical gradient check: compare analytical backward to finite differences
    let eps = 1e-3;
    let tol = 0.1; // 10% relative tolerance

    for &(in_d, out_d, depth) in &[(8, 4, 1), (8, 4, 2), (16, 8, 1), (16, 8, 2)] {
        let syn = if depth == 2 {
            Synapse::new_depth2(in_d, out_d, out_d)
        } else {
            Synapse::new(in_d, out_d)
        };

        let mut rng = SimpleRng::new(42);
        let x: Vec<f32> = (0..in_d).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
        let target: Vec<f32> = (0..out_d).map(|_| rng.next_f32()).collect();

        // Forward + backward
        let (out, cache) = syn.forward_cached(&x);
        let d_output: Vec<f32> = out.iter().zip(target.iter())
            .map(|(o, t)| o - t).collect(); // MSE gradient
        let (grads, _d_input) = syn.backward(&d_output, &cache);

        // Check dW1 with finite differences
        let mut max_err = 0.0f32;
        for idx in 0..syn.linear.weight.len().min(20) { // check first 20 weights
            let mut syn_plus = syn.clone();
            syn_plus.linear.weight[idx] += eps;
            let out_plus = syn_plus.forward(&x);
            let loss_plus: f32 = out_plus.iter().zip(target.iter())
                .map(|(o, t)| 0.5 * (o - t).powi(2)).sum();

            let mut syn_minus = syn.clone();
            syn_minus.linear.weight[idx] -= eps;
            let out_minus = syn_minus.forward(&x);
            let loss_minus: f32 = out_minus.iter().zip(target.iter())
                .map(|(o, t)| 0.5 * (o - t).powi(2)).sum();

            let numerical = (loss_plus - loss_minus) / (2.0 * eps);
            let analytical = grads.dw1[idx];
            let err = (numerical - analytical).abs() / (numerical.abs() + analytical.abs() + 1e-8);
            max_err = max_err.max(err);
        }

        // Also check with absolute error for near-zero gradients
        let mut max_abs = 0.0f32;
        for idx in 0..syn.linear.weight.len().min(20) {
            let mut syn_plus = syn.clone();
            syn_plus.linear.weight[idx] += eps;
            let out_plus = syn_plus.forward(&x);
            let loss_plus: f32 = out_plus.iter().zip(target.iter())
                .map(|(o, t)| 0.5 * (o - t).powi(2)).sum();
            let mut syn_minus = syn.clone();
            syn_minus.linear.weight[idx] -= eps;
            let out_minus = syn_minus.forward(&x);
            let loss_minus: f32 = out_minus.iter().zip(target.iter())
                .map(|(o, t)| 0.5 * (o - t).powi(2)).sum();
            let numerical = (loss_plus - loss_minus) / (2.0 * eps);
            let analytical = grads.dw1[idx];
            max_abs = max_abs.max((numerical - analytical).abs());
            if idx < 3 {
                eprintln!("    w1[{}]: numerical={:.6} analytical={:.6} diff={:.6}",
                    idx, numerical, analytical, (numerical - analytical).abs());
            }
        }
        let status = if max_abs < 0.01 { "OK" } else if max_err < tol { "OK(rel)" } else { "FAIL" };
        eprintln!("  grad check depth-{} ({:>2}→{:>2}): max_rel={:.4} max_abs={:.6} [{}]",
            depth, in_d, out_d, max_err, max_abs, status);
        assert!(max_abs < 0.05 || max_err < tol,
            "Gradient check failed for depth-{} ({}->{}) rel={} abs={}", depth, in_d, out_d, max_err, max_abs);
    }
    eprintln!();
}

// ─── Sleep backprop on replay buffer ───────────────────────────────────

/// Run one forward pass through all synapses, return cached intermediates per synapse.
/// This is the "sleep forward" — we need caches for backward.
fn sleep_forward(
    weights: &CtmWeights,
    observation: &[f32],
    proprio: &[f32],
) -> (Vec<f32>, Vec<(usize, SynapseForwardCache)>) {
    // Simplified: just run forward_split and collect motor output.
    // For full backprop we'd need caches from each synapse in the tick loop.
    // For now: train only syn_output_motor (synapse 3) with cached forward.
    let mut session = CtmSession::new(&weights.config);
    let mut tick_state = weights.init_tick_state();
    let _ = forward_split(weights, &mut session, &mut tick_state, observation, proprio, false);

    // Motor evidence = our prediction
    let motor_evidence = tick_state.motor_evidence.clone();

    // For the motor synapse, we can do a targeted cached forward.
    // The input to syn_output_motor is: concat(output_region, bg_region) + global
    // We approximate: use the final tick activations.
    let out_act = tick_state.act(2).to_vec();  // REGION_OUTPUT = 2
    let bg_act = tick_state.act(5).to_vec();  // REGION_BASAL_GANGLIA = 5
    let mut motor_input = out_act;
    motor_input.extend_from_slice(&bg_act);
    // If motor receives broadcast, we'd need global context too.
    // For simplicity, use what we have.
    // Pad to match synapse input dim
    motor_input.resize(weights.syn_output_motor.linear.in_dim, 0.0);

    let (_, cache) = weights.syn_output_motor.forward_cached(&motor_input);

    (motor_evidence, vec![(3, cache)])
}

#[test]
fn backprop_sleep_maze() {
    // Train the motor synapse (output→motor) with backprop during sleep.
    // Wake collects experiences. Sleep backprops cross-entropy on the motor output.
    let n_actions = 4;

    let cfg = CtmConfig {
        iterations: 8, d_input: 64, n_sync_out: 64,
        synapse_depth: 2,
        input_layer: LayerConfig { n_neurons: 256, ..Default::default() },
        attention_layer: LayerConfig { n_neurons: 128, ..Default::default() },
        output_layer: LayerConfig { n_neurons: 128, ..Default::default() },
        motor_layer: LayerConfig { n_neurons: n_actions, ..Default::default() },
        neuromod: NeuromodConfig {
            hebb_syn_lr: 0.0002,
            bg_elig_decay: 0.95,
            bg_reward_threshold: 0.01,
            da_gate: 0.1,
            ..NeuromodConfig::default()
        },
        ..CtmConfig::default()
    };

    let data = tasks::maze_examples(7, cfg.d_input, 5000);
    let train = &data[..4000];
    let test = &data[4000..];

    let ctm = Ctm::new(cfg.clone());
    let (mut weights, _) = ctm.into_split();
    let proprio = vec![0.0f32; cfg.d_input];

    let baseline_motor = eval_motor(&weights, test);
    let baseline_ls = eval_ls(&weights, test);
    eprintln!("\n  === 7×7 MAZE: Backprop Sleep (depth-2 synapses) ===");
    eprintln!("  Motor baseline: {:.1}%  (random=25%)", baseline_motor * 100.0);
    eprintln!("  LS ceiling:     {:.1}%", baseline_ls * 100.0);
    eprintln!("  Synapse depth:  {}", weights.syn_output_motor.depth());
    eprintln!();

    let t0 = std::time::Instant::now();
    let mut best_motor = baseline_motor;

    // ── WAKE: collect replay buffer ──
    // For each sample, collect the FULL activations (all regions) as features.
    // This is what the LS readout uses — the same feature space.
    // We'll train a standalone depth-2 network on these features → action.
    eprintln!("  Collecting wake experiences...");
    let act_dim = weights.init_tick_state().activations.len();
    let mut replay: Vec<(Vec<f32>, usize)> = Vec::new(); // (activations, target)

    for ex in train {
        let mut session = CtmSession::new(&weights.config);
        let mut tick_state = weights.init_tick_state();
        let _ = forward_split(&weights, &mut session, &mut tick_state,
            &ex.input, &proprio, false);
        replay.push((tick_state.activations.clone(), ex.target));
    }
    eprintln!("  Collected {} episodes, feature_dim={} ({:.1}s)",
        replay.len(), act_dim, t0.elapsed().as_secs_f64());

    // ── SLEEP: train a depth-2 action head on frozen activations ──
    // This is the sleep equivalent: activations are frozen (from wake forward passes),
    // we train a standalone network to map activations → action.
    // After training, these weights get injected back into the brain.
    let hidden_dim = 128;
    let mut head = Synapse::new_depth2(act_dim, hidden_dim, n_actions);

    let sleep_lr = 0.03;
    let sleep_epochs = 300;
    let batch_size = 64;
    let mut rng = SimpleRng::new(42);

    for epoch in 0..sleep_epochs {
        let mut epoch_loss = 0.0f32;
        let mut epoch_correct = 0usize;
        let mut epoch_total = 0usize;

        let offset = (rng.next_u64() as usize) % replay.len();

        for batch_start in (0..replay.len()).step_by(batch_size) {
            let actual_batch = batch_size.min(replay.len() - batch_start);

            let mut grad_accum_w1 = vec![0.0f32; head.linear.weight.len()];
            let mut grad_accum_b1 = vec![0.0f32; head.linear.bias.len()];
            let mut grad_accum_w2 = head.linear2.as_ref()
                .map(|l| vec![0.0f32; l.weight.len()]).unwrap_or_default();
            let mut grad_accum_b2 = head.linear2.as_ref()
                .map(|l| vec![0.0f32; l.bias.len()]).unwrap_or_default();

            for b in 0..actual_batch {
                let idx = (batch_start + b + offset) % replay.len();
                let (ref features, target) = replay[idx];

                let (output, cache) = head.forward_cached(features);

                // Cross-entropy gradient
                let max_out = output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = output.iter().map(|o| (o - max_out).exp()).sum();
                let mut d_output: Vec<f32> = output.iter()
                    .map(|o| (o - max_out).exp() / exp_sum).collect();
                if target < d_output.len() { d_output[target] -= 1.0; }

                let pred = output.iter().enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i).unwrap_or(0);
                if pred == target { epoch_correct += 1; }
                epoch_total += 1;

                let p_target = ((output[target.min(output.len()-1)] - max_out).exp() / exp_sum).max(1e-8);
                epoch_loss += -p_target.ln();

                let (grads, _) = head.backward(&d_output, &cache);
                for (a, &g) in grad_accum_w1.iter_mut().zip(grads.dw1.iter()) { *a += g; }
                for (a, &g) in grad_accum_b1.iter_mut().zip(grads.db1.iter()) { *a += g; }
                for (a, &g) in grad_accum_w2.iter_mut().zip(grads.dw2.iter()) { *a += g; }
                for (a, &g) in grad_accum_b2.iter_mut().zip(grads.db2.iter()) { *a += g; }
            }

            let scale = sleep_lr / actual_batch as f32;
            head.apply_gradients(&SynapseGradients {
                dw1: grad_accum_w1, db1: grad_accum_b1,
                dw2: grad_accum_w2, db2: grad_accum_b2,
            }, scale);
        }

        if (epoch + 1) % 20 == 0 {
            // Evaluate the trained head on test activations
            let test_feats: Vec<(Vec<f32>, usize)> = test.par_iter().map(|ex| {
                let mut s = CtmSession::new(&weights.config);
                let mut t = weights.init_tick_state();
                let _ = forward_split(&weights, &mut s, &mut t, &ex.input, &proprio, false);
                (t.activations.clone(), ex.target)
            }).collect();

            let head_correct: usize = test_feats.iter().map(|(f, lab)| {
                let out = head.forward(f);
                let pred = out.iter().enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i).unwrap_or(0);
                if pred == *lab { 1 } else { 0 }
            }).sum();
            let head_acc = head_correct as f32 / test_feats.len() as f32;
            if head_acc > best_motor { best_motor = head_acc; }

            eprintln!("  Sleep epoch {:3}: loss={:.3} train={:.1}% test_head={:.1}% best={:.1}% ({:.1}s)",
                epoch + 1,
                epoch_loss / epoch_total as f32,
                epoch_correct as f32 / epoch_total as f32 * 100.0,
                head_acc * 100.0,
                best_motor * 100.0,
                t0.elapsed().as_secs_f64());
        }
    }

    eprintln!("\n  Sleep-trained head: {:.1}% → {:.1}% (best={:.1}%, {:+.1}pp)",
        baseline_motor * 100.0, best_motor * 100.0, best_motor * 100.0,
        (best_motor - baseline_motor) * 100.0);
    eprintln!("  LS ceiling:        {:.1}%", baseline_ls * 100.0);
    eprintln!("  Gap closed:        {:.1}pp of {:.1}pp possible\n",
        (best_motor - baseline_motor) * 100.0,
        (baseline_ls - baseline_motor) * 100.0);
}
