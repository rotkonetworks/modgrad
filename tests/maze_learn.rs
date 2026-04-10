//! Maze learning via three-factor REINFORCE (no backprop, no SPSA).
//!
//! The brain learns from experience: forward → check → reward → repeat.
//! Dopamine reward injection modulates Hebbian eligibility traces.
//!
//! cargo test --release --test maze_learn -- --nocapture

use modgrad::ctm::{Ctm, CtmConfig, CtmWeights, CtmSession, LayerConfig, forward_split, SimpleRng};
use modgrad::ctm::NeuromodConfig;
use modgrad::ops::encode_label;
use modgrad::tasks;
use modgrad::accuracy::{eval_ls, quick_eval};

/// Total weight magnitude across all synapses.
fn total_weight_norm(weights: &CtmWeights) -> f32 {
    weights.synapse_refs().iter()
        .map(|s| s.linear.weight.iter().map(|w| w * w).sum::<f32>())
        .sum::<f32>()
        .sqrt()
}

#[test]
fn three_factor_maze_7x7() {
    // 7×7 maze: 49 cells + 4 pos/goal = 53 features encoded in d_input=64
    // 4 output classes (up/right/down/left)
    // Random baseline ~25%, LS readout on frozen weights should be moderate
    let cfg = CtmConfig {
        iterations: 8, d_input: 64, n_sync_out: 64,
        input_layer: LayerConfig { n_neurons: 128, ..Default::default() },
        attention_layer: LayerConfig { n_neurons: 128, ..Default::default() },
        output_layer: LayerConfig { n_neurons: 128, ..Default::default() },
        motor_layer: LayerConfig { n_neurons: 128, ..Default::default() },
        neuromod: NeuromodConfig {
            hebb_syn_lr: 0.000005,   // very conservative — don't destroy features
            bg_elig_decay: 0.95,     // faster trace decay for shorter episodes
            bg_reward_threshold: 0.02, // lower threshold, more updates
            da_gate: 0.3,            // lower gate so learning fires more often
            ..NeuromodConfig::default()
        },
        ..CtmConfig::default()
    };

    let data = tasks::maze_examples(7, cfg.d_input, 4000);
    let train = &data[..3000];
    let test = &data[3000..];

    let ctm = Ctm::new(cfg.clone());
    let (mut weights, _) = ctm.into_split();

    // Measure baseline
    let baseline = eval_ls(&weights, test);
    let quick_base = quick_eval(&weights, test);
    let norm0 = total_weight_norm(&weights);
    eprintln!("\n  === 7×7 MAZE: Three-Factor REINFORCE ===");
    eprintln!("  Baseline LS readout: {:.1}%  (random=25%)", baseline * 100.0);
    eprintln!("  Baseline quick_eval: {:.1}%", quick_base * 100.0);
    eprintln!("  Weight norm: {:.2}", norm0);
    eprintln!("  Synapses: {}", weights.synapse_refs().iter()
        .map(|s| s.linear.weight.len()).sum::<usize>());
    eprintln!();

    // Pre-compute label proprioception vectors (one per class).
    // During training the brain hears the label; during eval it doesn't.
    let labels: Vec<Vec<f32>> = (0..4).map(|c| encode_label(c, cfg.d_input)).collect();

    let mut session = CtmSession::new(&weights.config);
    session.init_syn_deltas(&weights);
    session.hebbian_enabled = true;
    session.last_reward = 1.0; // neutral start

    let t0 = std::time::Instant::now();
    let batch_size = 32;
    let n_epochs = 10;
    let apply_every = 32; // apply weight deltas every N samples

    for epoch in 0..n_epochs {
        let mut epoch_correct = 0usize;
        let mut epoch_total = 0usize;
        let mut total_delta_mag = 0.0f32;

        for (i, ex) in train.iter().enumerate() {
            let mut tick_state = weights.init_tick_state();

            // Forward pass — label as proprioception drives co-activation
            let (preds, _sync, _signals) = forward_split(
                &weights, &mut session, &mut tick_state,
                &ex.input, &labels[ex.target], false,
            );

            // Decode prediction from output region activations
            let n_classes = ex.n_classes;
            let out_start = tick_state.act_offsets[2];
            let out_size = tick_state.act_sizes[2];
            let bin_size = out_size / n_classes.max(1);
            let pred = (0..n_classes).max_by(|&a, &b| {
                let sum_a: f32 = tick_state.activations[out_start + a * bin_size..out_start + (a+1).min(n_classes) * bin_size]
                    .iter().sum();
                let sum_b: f32 = tick_state.activations[out_start + b * bin_size..out_start + (b+1).min(n_classes) * bin_size]
                    .iter().sum();
                sum_a.partial_cmp(&sum_b).unwrap()
            }).unwrap_or(0);

            let correct = pred == ex.target;
            if correct { epoch_correct += 1; }
            epoch_total += 1;

            // Reward signal: positive for correct, negative for wrong
            session.last_reward = if correct { 2.0 } else { 0.2 };

            // Apply weight deltas periodically
            if (i + 1) % apply_every == 0 {
                // Measure delta magnitude before applying
                for deltas in &session.syn_deltas {
                    total_delta_mag += deltas.iter().map(|d| d.abs()).sum::<f32>();
                }
                session.apply_syn_deltas(&mut weights);
                session.apply_bg_weight_delta(&mut weights);
            }
        }

        let norm = total_weight_norm(&weights);
        let test_acc = eval_ls(&weights, test);
        let test_quick = quick_eval(&weights, test);
        let elapsed = t0.elapsed().as_secs_f64();
        eprintln!("  Epoch {:2}: train_quick={:.1}% test_LS={:.1}% test_quick={:.1}% |ΔW|={:.6} ||W||={:.2} ({:.1}s)",
            epoch + 1,
            epoch_correct as f32 / epoch_total as f32 * 100.0,
            test_acc * 100.0,
            test_quick * 100.0,
            total_delta_mag,
            norm,
            elapsed);
    }

    let final_acc = eval_ls(&weights, test);
    let norm_final = total_weight_norm(&weights);
    eprintln!("\n  RESULT: {:.1}% → {:.1}% ({:+.1}pp)", baseline * 100.0, final_acc * 100.0,
        (final_acc - baseline) * 100.0);
    eprintln!("  Weight norm: {:.2} → {:.2} (Δ={:.4})", norm0, norm_final, norm_final - norm0);
    eprintln!();
}

#[test]
fn motor_action_selection_maze() {
    // KEY INSIGHT: make motor_layer.n_neurons = n_classes.
    // Each motor neuron IS an action. Drift-diffusion picks the winner.
    // Reward strengthens the specific synapses that drove the winning neuron.
    //
    // Softmax exploration: instead of deterministic argmax, sample action
    // from softmax distribution. Temperature decays over training (explore→exploit).
    // This is biologically analogous to norepinephrine-driven randomness.

    let n_actions = 4; // up, right, down, left

    let cfg = CtmConfig {
        iterations: 8, d_input: 64, n_sync_out: 32,
        input_layer: LayerConfig { n_neurons: 128, ..Default::default() },
        attention_layer: LayerConfig { n_neurons: 64, ..Default::default() },
        output_layer: LayerConfig { n_neurons: 64, ..Default::default() },
        motor_layer: LayerConfig { n_neurons: n_actions, ..Default::default() },
        neuromod: NeuromodConfig {
            hebb_syn_lr: 0.0001,
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
    let eval_proprio = vec![0.0f32; cfg.d_input]; // zeros for eval — no answer leak
    let labels: Vec<Vec<f32>> = (0..n_actions).map(|c| encode_label(c, cfg.d_input)).collect();

    // Eval always uses deterministic argmax (proprio = zeros)
    let test_motor = |w: &CtmWeights, data: &[tasks::Example]| -> f32 {
        let mut correct = 0usize;
        for ex in data {
            let mut s = CtmSession::new(&w.config);
            let mut t = w.init_tick_state();
            let _ = forward_split(w, &mut s, &mut t, &ex.input, &eval_proprio, false);
            let pred = t.motor_evidence.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i).unwrap_or(0);
            if pred == ex.target { correct += 1; }
        }
        correct as f32 / data.len() as f32
    };

    let baseline = test_motor(&weights, test);
    let ls_baseline = eval_ls(&weights, test);
    eprintln!("\n  === 7×7 MAZE: Motor Action Selection + Softmax Exploration ===");
    eprintln!("  Motor argmax baseline:  {:.1}%  (random=25%)", baseline * 100.0);
    eprintln!("  LS readout baseline:    {:.1}%", ls_baseline * 100.0);
    eprintln!();

    let mut session = CtmSession::new(&weights.config);
    session.init_syn_deltas(&weights);
    session.hebbian_enabled = true;
    session.last_reward = 1.0;
    let mut rng = SimpleRng::new(7);

    let t0 = std::time::Instant::now();
    let n_epochs = 30;
    let mut best_test = baseline;
    let mut best_weights: Option<CtmWeights> = None;
    let mut no_improve = 0u32;

    for epoch in 0..n_epochs {
        let mut correct = 0usize;
        let mut total = 0usize;

        // Temperature: high early (explore), low late (exploit)
        let temperature = 2.0 * (1.0 - epoch as f32 / n_epochs as f32).max(0.1);

        // Learning rate decay: cosine schedule
        let lr_scale = (1.0 + (std::f32::consts::PI * epoch as f32 / n_epochs as f32).cos()) / 2.0;
        weights.config.neuromod.hebb_syn_lr = 0.0002 * lr_scale;

        // Reset eligibility traces each epoch to prevent interference
        for elig in &mut session.syn_eligibility {
            elig.fill(0.0);
        }

        for (i, ex) in train.iter().enumerate() {
            let mut tick_state = weights.init_tick_state();
            let _ = forward_split(&weights, &mut session, &mut tick_state,
                &ex.input, &labels[ex.target], false);

            // Softmax action selection over motor evidence
            let evidence = &tick_state.motor_evidence;
            let max_ev = evidence.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = evidence.iter()
                .map(|e| ((e - max_ev) / temperature).exp())
                .sum();
            let probs: Vec<f32> = evidence.iter()
                .map(|e| ((e - max_ev) / temperature).exp() / exp_sum)
                .collect();

            // Sample action from softmax distribution
            let r = rng.next_f32();
            let mut cumsum = 0.0;
            let mut action = 0;
            for (a, &p) in probs.iter().enumerate() {
                cumsum += p;
                if r < cumsum { action = a; break; }
            }

            let got_right = action == ex.target;
            if got_right { correct += 1; }
            total += 1;

            // Reward contrast: dopamine burst on correct, dip on wrong.
            session.last_reward = if got_right { 2.0 } else { 0.2 };

            if (i + 1) % 32 == 0 {
                session.apply_syn_deltas(&mut weights);
                session.apply_bg_weight_delta(&mut weights);
            }
        }

        let test_acc = test_motor(&weights, test);
        if test_acc > best_test {
            best_test = test_acc;
            best_weights = Some(weights.clone());
            no_improve = 0;
        } else {
            no_improve += 1;
            if no_improve >= 8 {
                eprintln!("  Early stop: no improvement for 8 epochs");
                break;
            }
        }
        eprintln!("  Epoch {:2}: train={:.1}% test={:.1}% best={:.1}% T={:.2} lr={:.6} ({:.1}s)",
            epoch + 1,
            correct as f32 / total as f32 * 100.0,
            test_acc * 100.0,
            best_test * 100.0,
            temperature,
            weights.config.neuromod.hebb_syn_lr,
            t0.elapsed().as_secs_f64());
    }

    // Restore best weights
    if let Some(bw) = best_weights {
        weights = bw;
    }
    let final_motor = test_motor(&weights, test);
    let final_ls = eval_ls(&weights, test);
    eprintln!("\n  Motor (best checkpoint): {:.1}% → {:.1}% ({:+.1}pp)",
        baseline * 100.0, final_motor * 100.0, (final_motor - baseline) * 100.0);
    eprintln!("  LS:    {:.1}% → {:.1}% ({:+.1}pp)\n",
        ls_baseline * 100.0, final_ls * 100.0, (final_ls - ls_baseline) * 100.0);
}

#[test]
fn cri_episodic_maze() {
    // CRI-style episodic memory: store (brain state → correct action) reflexes.
    // No weight modification. No gradients. No SPSA.
    // Pure Pavlovian conditioning: see pattern → recall correct action.
    //
    // This is how hippocampal episodic memory works:
    //   1. Experience maze → cortical activation pattern
    //   2. Get feedback (correct direction)
    //   3. Store (pattern, direction) in episodic memory
    //   4. Next time: similar pattern → retrieve → inject motor bias
    //
    // Adapted from: Conditioned Reflex Injection (CRI), Niemi 2026
    //   ~/rotko/epimem — Pavlovian conditioning for frozen transformers

    let n_actions = 4;

    let cfg = CtmConfig {
        iterations: 8, d_input: 64, n_sync_out: 32,
        input_layer: LayerConfig { n_neurons: 128, ..Default::default() },
        attention_layer: LayerConfig { n_neurons: 64, ..Default::default() },
        output_layer: LayerConfig { n_neurons: 64, ..Default::default() },
        motor_layer: LayerConfig { n_neurons: n_actions, ..Default::default() },
        ..CtmConfig::default()
    };

    let data = tasks::maze_examples(7, cfg.d_input, 5000);
    let train = &data[..4000];
    let test = &data[4000..];

    let ctm = Ctm::new(cfg.clone());
    let (weights, _) = ctm.into_split();
    let eval_proprio = vec![0.0f32; cfg.d_input];
    let labels: Vec<Vec<f32>> = (0..n_actions).map(|c| encode_label(c, cfg.d_input)).collect();

    // Eval: forward + motor evidence argmax (CRI injects bias during forward)
    let eval_motor = |w: &CtmWeights, session: &CtmSession, data: &[tasks::Example]| -> f32 {
        let mut correct = 0usize;
        for ex in data {
            let mut s = session.clone();
            let mut t = w.init_tick_state();
            let _ = forward_split(w, &mut s, &mut t, &ex.input, &eval_proprio, false);
            let pred = t.motor_evidence.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i).unwrap_or(0);
            if pred == ex.target { correct += 1; }
        }
        correct as f32 / data.len() as f32
    };

    // Baseline: no memories stored
    let session = CtmSession::new(&weights.config);
    let baseline = eval_motor(&weights, &session, test);
    let ls_baseline = eval_ls(&weights, test);
    eprintln!("\n  === 7×7 MAZE: CRI Episodic Memory (no weight modification) ===");
    eprintln!("  Motor baseline:     {:.1}%  (random=25%)", baseline * 100.0);
    eprintln!("  LS readout:         {:.1}%", ls_baseline * 100.0);
    eprintln!("  Motor memory cap:   {}", session.motor_memory.capacity);
    eprintln!();

    // Phase 1: Conditioning — store (brain state, correct action) episodes
    let mut session = CtmSession::new(&weights.config);
    let t0 = std::time::Instant::now();

    for (i, ex) in train.iter().enumerate() {
        let mut tick_state = weights.init_tick_state();
        let _ = forward_split(&weights, &mut session, &mut tick_state,
            &ex.input, &labels[ex.target], false);

        // Create motor bias: one-hot for correct action
        let mut motor_bias = vec![0.0f32; n_actions];
        motor_bias[ex.target] = 5.0; // strong bias toward correct action

        // Store the conditioned reflex
        session.condition_motor(&tick_state, &motor_bias, 1.0);

        if (i + 1) % 1000 == 0 {
            let acc = eval_motor(&weights, &session, test);
            eprintln!("  Conditioned {:4} episodes: test_motor={:.1}% ({:.1}s)",
                i + 1, acc * 100.0, t0.elapsed().as_secs_f64());
        }
    }

    // Phase 2: Evaluate — brain state triggers stored reflexes
    let final_motor = eval_motor(&weights, &session, test);
    let final_ls = eval_ls(&weights, test); // LS should be unchanged (no weight mod)
    eprintln!("\n  Motor: {:.1}% → {:.1}% ({:+.1}pp) [{} memories stored]",
        baseline * 100.0, final_motor * 100.0,
        (final_motor - baseline) * 100.0,
        session.motor_memory.count);
    eprintln!("  LS:    {:.1}% → {:.1}% ({:+.1}pp) (should be 0 — no weight change)\n",
        ls_baseline * 100.0, final_ls * 100.0, (final_ls - ls_baseline) * 100.0);
}

#[test]
fn cri_plus_reinforce_maze() {
    // Best of both: CRI for fast episodic recall + three-factor for slow weight learning.
    // CRI handles "seen before" cases instantly.
    // Three-factor gradually tunes motor weights for novel cases.
    // This is dual-process theory: System 1 (fast, episodic) + System 2 (slow, learned).

    let n_actions = 4;

    let cfg = CtmConfig {
        iterations: 8, d_input: 64, n_sync_out: 32,
        input_layer: LayerConfig { n_neurons: 128, ..Default::default() },
        attention_layer: LayerConfig { n_neurons: 64, ..Default::default() },
        output_layer: LayerConfig { n_neurons: 64, ..Default::default() },
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
    let eval_proprio = vec![0.0f32; cfg.d_input];
    let labels: Vec<Vec<f32>> = (0..n_actions).map(|c| encode_label(c, cfg.d_input)).collect();

    let eval_motor = |w: &CtmWeights, session: &CtmSession, data: &[tasks::Example]| -> f32 {
        let mut correct = 0usize;
        for ex in data {
            let mut s = session.clone();
            let mut t = w.init_tick_state();
            let _ = forward_split(w, &mut s, &mut t, &ex.input, &eval_proprio, false);
            let pred = t.motor_evidence.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i).unwrap_or(0);
            if pred == ex.target { correct += 1; }
        }
        correct as f32 / data.len() as f32
    };

    let session = CtmSession::new(&weights.config);
    let baseline = eval_motor(&weights, &session, test);
    eprintln!("\n  === 7×7 MAZE: CRI + Three-Factor (dual-process) ===");
    eprintln!("  Motor baseline: {:.1}%  (random=25%)", baseline * 100.0);
    eprintln!();

    let mut session = CtmSession::new(&weights.config);
    session.init_syn_deltas(&weights);
    session.hebbian_enabled = true;
    session.last_reward = 1.0;

    let t0 = std::time::Instant::now();

    for epoch in 0..10 {
        let mut correct = 0usize;
        let mut total = 0usize;

        for (i, ex) in train.iter().enumerate() {
            let mut tick_state = weights.init_tick_state();
            let _ = forward_split(&weights, &mut session, &mut tick_state,
                &ex.input, &labels[ex.target], false);

            let pred = tick_state.motor_evidence.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i).unwrap_or(0);

            let got_right = pred == ex.target;
            if got_right { correct += 1; }
            total += 1;

            // CRI: store correct action as conditioned reflex
            let mut motor_bias = vec![0.0f32; n_actions];
            motor_bias[ex.target] = 5.0;
            session.condition_motor(&tick_state, &motor_bias, 1.0);

            // Three-factor reward for weight learning
            session.last_reward = if got_right { 2.0 } else { 0.2 };

            if (i + 1) % 32 == 0 {
                session.apply_syn_deltas(&mut weights);
            }
        }

        let test_acc = eval_motor(&weights, &session, test);
        eprintln!("  Epoch {:2}: train={:.1}% test={:.1}% memories={} ({:.1}s)",
            epoch + 1,
            correct as f32 / total as f32 * 100.0,
            test_acc * 100.0,
            session.motor_memory.count,
            t0.elapsed().as_secs_f64());
    }

    let final_acc = eval_motor(&weights, &session, test);
    eprintln!("\n  Motor: {:.1}% → {:.1}% ({:+.1}pp)\n",
        baseline * 100.0, final_acc * 100.0, (final_acc - baseline) * 100.0);
}

#[test]
fn wake_sleep_maze() {
    // Wake/sleep learning cycle:
    //   WAKE:  experience mazes, store (observation, reward, salience) episodes
    //   SLEEP: replay high-salience episodes through forward_split, three-factor learns
    //
    // Why this works: during wake, conflicting samples cause catastrophic interference.
    // During sleep, we replay ONLY the surprising/important episodes, multiple times,
    // without new conflicting data. The three-factor rule consolidates these into weights.
    //
    // This is hippocampal replay → cortical consolidation.
    // Biological basis: sharp-wave ripples during NREM sleep replay recent episodes
    // at 5-20× speed. VTA dopamine during replay reinforces the same synapses.

    let n_actions = 4;

    let cfg = CtmConfig {
        iterations: 8, d_input: 64, n_sync_out: 32,
        input_layer: LayerConfig { n_neurons: 128, ..Default::default() },
        attention_layer: LayerConfig { n_neurons: 64, ..Default::default() },
        output_layer: LayerConfig { n_neurons: 64, ..Default::default() },
        motor_layer: LayerConfig { n_neurons: n_actions, ..Default::default() },
        neuromod: NeuromodConfig {
            hebb_syn_lr: 0.0002,
            bg_elig_decay: 0.95,
            bg_reward_threshold: 0.01,
            da_gate: 0.1,
            forget_alpha: 0.001,
            ..NeuromodConfig::default()
        },
        ..CtmConfig::default()
    };

    let data = tasks::maze_examples(7, cfg.d_input, 5000);
    let train = &data[..4000];
    let test = &data[4000..];

    let ctm = Ctm::new(cfg.clone());
    let (mut weights, _) = ctm.into_split();
    let eval_proprio = vec![0.0f32; cfg.d_input];
    let labels: Vec<Vec<f32>> = (0..n_actions).map(|c| encode_label(c, cfg.d_input)).collect();

    let eval_motor = |w: &CtmWeights, data: &[tasks::Example]| -> f32 {
        let mut correct = 0usize;
        for ex in data {
            let mut s = CtmSession::new(&w.config);
            let mut t = w.init_tick_state();
            let _ = forward_split(w, &mut s, &mut t, &ex.input, &eval_proprio, false);
            let pred = t.motor_evidence.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i).unwrap_or(0);
            if pred == ex.target { correct += 1; }
        }
        correct as f32 / data.len() as f32
    };

    let baseline = eval_motor(&weights, test);
    eprintln!("\n  === 7×7 MAZE: Wake/Sleep Consolidation ===");
    eprintln!("  Motor baseline: {:.1}%  (random=25%)", baseline * 100.0);
    eprintln!();

    let mut session = CtmSession::new(&weights.config);
    session.init_syn_deltas(&weights);
    session.hebbian_enabled = true;
    session.last_reward = 1.0;
    // Replay buffer: 512 episodes, store anything with salience > 0.3
    session.replay = modgrad::ctm::ReplayBuffer::new(512, 0.3);

    let mut rng = SimpleRng::new(7);
    let t0 = std::time::Instant::now();
    let mut best_test = baseline;

    for cycle in 0..10 {
        // ── WAKE: experience mazes, collect episodes ──
        let mut wake_correct = 0usize;
        let mut wake_total = 0usize;
        let temperature = 1.5 * (1.0 - cycle as f32 / 10.0).max(0.2);

        for ex in train.iter() {
            let mut tick_state = weights.init_tick_state();
            let _ = forward_split(&weights, &mut session, &mut tick_state,
                &ex.input, &labels[ex.target], false);

            // Softmax action selection
            let evidence = &tick_state.motor_evidence;
            let max_ev = evidence.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = evidence.iter()
                .map(|e| ((e - max_ev) / temperature).exp()).sum();
            let probs: Vec<f32> = evidence.iter()
                .map(|e| ((e - max_ev) / temperature).exp() / exp_sum).collect();
            let r = rng.next_f32();
            let mut cumsum = 0.0;
            let mut action = 0;
            for (a, &p) in probs.iter().enumerate() {
                cumsum += p;
                if r < cumsum { action = a; break; }
            }

            let correct = action == ex.target;
            if correct { wake_correct += 1; }
            wake_total += 1;

            let reward = if correct { 2.0 } else { 0.2 };
            session.last_reward = reward;

            // Compute salience for replay storage
            let rpe = (reward - session.syn_reward_baseline.get(3).copied().unwrap_or(1.0)).abs();
            let salience = rpe;

            // Store episode in replay buffer (only if surprising enough)
            session.replay.push_episode(
                ex.input.clone(), salience, reward, action, correct,
            );

            // Apply deltas during wake too (online learning)
            if wake_total % 64 == 0 {
                session.apply_syn_deltas(&mut weights);
            }
        }
        session.apply_syn_deltas(&mut weights);

        let post_wake = eval_motor(&weights, test);

        // ── SLEEP: replay high-salience episodes ──
        let n_replays = session.replay.entries.len().min(256);
        let replay_episodes: Vec<_> = session.replay.prioritized()
            .into_iter().take(n_replays).cloned().collect();

        let mut sleep_replays = 0;
        // Replay each episode 3× (sharp-wave ripple compression)
        for _rep in 0..3 {
            for entry in &replay_episodes {
                let mut tick_state = weights.init_tick_state();
                // Inject stored reward to drive three-factor during replay
                session.last_reward = entry.reward;
                let _ = forward_split(&weights, &mut session, &mut tick_state,
                    &entry.observation, &eval_proprio, false);
                sleep_replays += 1;

                if sleep_replays % 64 == 0 {
                    session.apply_syn_deltas(&mut weights);
                }
            }
        }
        session.apply_syn_deltas(&mut weights);

        let post_sleep = eval_motor(&weights, test);
        if post_sleep > best_test { best_test = post_sleep; }

        eprintln!("  Cycle {:2}: wake={:.1}% post_wake={:.1}% post_sleep={:.1}% best={:.1}% buf={} replayed={} ({:.1}s)",
            cycle + 1,
            wake_correct as f32 / wake_total as f32 * 100.0,
            post_wake * 100.0,
            post_sleep * 100.0,
            best_test * 100.0,
            session.replay.entries.len(),
            sleep_replays,
            t0.elapsed().as_secs_f64());
    }

    let final_acc = eval_motor(&weights, test);
    eprintln!("\n  Motor: {:.1}% → {:.1}% (best={:.1}%, {:+.1}pp)\n",
        baseline * 100.0, final_acc * 100.0, best_test * 100.0,
        (best_test - baseline) * 100.0);
}

#[test]
fn three_factor_maze_15x15() {
    // 15×15 maze: 225 cells, needs d_input=256 to encode grid + position
    // This should be hard enough that frozen random weights don't trivially solve it
    let cfg = CtmConfig {
        iterations: 8, d_input: 256, n_sync_out: 64,
        input_layer: LayerConfig { n_neurons: 128, ..Default::default() },
        attention_layer: LayerConfig { n_neurons: 128, ..Default::default() },
        output_layer: LayerConfig { n_neurons: 128, ..Default::default() },
        motor_layer: LayerConfig { n_neurons: 128, ..Default::default() },
        neuromod: NeuromodConfig {
            hebb_syn_lr: 0.001,
            bg_elig_decay: 0.95,
            bg_reward_threshold: 0.05,
            da_gate: 0.5,
            ..NeuromodConfig::default()
        },
        ..CtmConfig::default()
    };

    let data = tasks::maze_examples(15, cfg.d_input, 3000);
    let train = &data[..2000];
    let test = &data[2000..];

    let ctm = Ctm::new(cfg.clone());
    let (mut weights, _) = ctm.into_split();

    let baseline = eval_ls(&weights, test);
    let norm0 = total_weight_norm(&weights);
    eprintln!("\n  === 15×15 MAZE: Three-Factor REINFORCE ===");
    eprintln!("  Baseline LS readout: {:.1}%  (random=25%)", baseline * 100.0);
    eprintln!("  Weight norm: {:.2}", norm0);
    eprintln!("  Synapses: {}", weights.synapse_refs().iter()
        .map(|s| s.linear.weight.len()).sum::<usize>());
    eprintln!();

    let labels: Vec<Vec<f32>> = (0..4).map(|c| encode_label(c, cfg.d_input)).collect();
    let mut session = CtmSession::new(&weights.config);
    session.init_syn_deltas(&weights);
    session.hebbian_enabled = true;
    session.last_reward = 1.0;

    let t0 = std::time::Instant::now();

    for epoch in 0..10 {
        let mut epoch_correct = 0usize;
        let mut epoch_total = 0usize;
        let mut total_delta_mag = 0.0f32;

        for (i, ex) in train.iter().enumerate() {
            let mut tick_state = weights.init_tick_state();
            let _ = forward_split(&weights, &mut session, &mut tick_state,
                &ex.input, &labels[ex.target], false);

            let n_classes = ex.n_classes;
            let out_start = tick_state.act_offsets[2];
            let out_size = tick_state.act_sizes[2];
            let bin_size = out_size / n_classes.max(1);
            let pred = (0..n_classes).max_by(|&a, &b| {
                let sum_a: f32 = tick_state.activations[out_start + a * bin_size..out_start + (a+1).min(n_classes) * bin_size]
                    .iter().sum();
                let sum_b: f32 = tick_state.activations[out_start + b * bin_size..out_start + (b+1).min(n_classes) * bin_size]
                    .iter().sum();
                sum_a.partial_cmp(&sum_b).unwrap()
            }).unwrap_or(0);

            if pred == ex.target { epoch_correct += 1; }
            epoch_total += 1;
            session.last_reward = if pred == ex.target { 2.0 } else { 0.2 };

            if (i + 1) % 32 == 0 {
                for deltas in &session.syn_deltas {
                    total_delta_mag += deltas.iter().map(|d| d.abs()).sum::<f32>();
                }
                session.apply_syn_deltas(&mut weights);
                session.apply_bg_weight_delta(&mut weights);
            }
        }

        let test_acc = eval_ls(&weights, test);
        let norm = total_weight_norm(&weights);
        eprintln!("  Epoch {:2}: train_quick={:.1}% test_LS={:.1}% |ΔW|={:.6} ||W||={:.2} ({:.1}s)",
            epoch + 1,
            epoch_correct as f32 / epoch_total as f32 * 100.0,
            test_acc * 100.0,
            total_delta_mag, norm,
            t0.elapsed().as_secs_f64());
    }

    let final_acc = eval_ls(&weights, test);
    eprintln!("\n  RESULT: {:.1}% → {:.1}% ({:+.1}pp)\n",
        baseline * 100.0, final_acc * 100.0, (final_acc - baseline) * 100.0);
}

#[test]
fn three_factor_parity() {
    // 8-bit parity — the classic nonlinear test
    let cfg = CtmConfig {
        iterations: 8, d_input: 8, n_sync_out: 32,
        input_layer: LayerConfig { n_neurons: 64, ..Default::default() },
        attention_layer: LayerConfig { n_neurons: 64, ..Default::default() },
        output_layer: LayerConfig { n_neurons: 64, ..Default::default() },
        motor_layer: LayerConfig { n_neurons: 64, ..Default::default() },
        neuromod: NeuromodConfig {
            hebb_syn_lr: 0.001,
            bg_elig_decay: 0.95,
            bg_reward_threshold: 0.05,
            da_gate: 0.5,
            ..NeuromodConfig::default()
        },
        ..CtmConfig::default()
    };

    let data = tasks::parity_examples(8, cfg.d_input);
    let train = &data[..2000];
    let test = &data[2000..3000];

    let ctm = Ctm::new(cfg.clone());
    let (mut weights, _) = ctm.into_split();

    let baseline = eval_ls(&weights, test);
    eprintln!("\n  === 8-BIT PARITY: Three-Factor REINFORCE ===");
    eprintln!("  Baseline LS readout: {:.1}%  (random=50%)", baseline * 100.0);

    let labels: Vec<Vec<f32>> = (0..2).map(|c| encode_label(c, cfg.d_input)).collect();
    let mut session = CtmSession::new(&weights.config);
    session.init_syn_deltas(&weights);
    session.hebbian_enabled = true;
    session.last_reward = 1.0;

    let t0 = std::time::Instant::now();

    for epoch in 0..20 {
        let mut correct = 0usize;
        let mut total = 0usize;

        for (i, ex) in train.iter().enumerate() {
            let mut tick_state = weights.init_tick_state();
            let _ = forward_split(&weights, &mut session, &mut tick_state,
                &ex.input, &labels[ex.target], false);

            let out_start = tick_state.act_offsets[2];
            let out_size = tick_state.act_sizes[2];
            let half = out_size / 2;
            let s0: f32 = tick_state.activations[out_start..out_start+half].iter().sum();
            let s1: f32 = tick_state.activations[out_start+half..out_start+out_size].iter().sum();
            let pred = if s1 > s0 { 1 } else { 0 };

            if pred == ex.target { correct += 1; }
            total += 1;
            session.last_reward = if pred == ex.target { 2.0 } else { 0.2 };

            if (i + 1) % 32 == 0 {
                session.apply_syn_deltas(&mut weights);
            }
        }

        let test_acc = eval_ls(&weights, test);
        eprintln!("  Epoch {:2}: train_quick={:.1}% test_LS={:.1}% ({:.1}s)",
            epoch + 1,
            correct as f32 / total as f32 * 100.0,
            test_acc * 100.0,
            t0.elapsed().as_secs_f64());
    }

    let final_acc = eval_ls(&weights, test);
    eprintln!("\n  RESULT: {:.1}% → {:.1}% ({:+.1}pp)\n",
        baseline * 100.0, final_acc * 100.0, (final_acc - baseline) * 100.0);
}
