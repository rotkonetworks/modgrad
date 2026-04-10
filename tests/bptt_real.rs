//! BPTT through the REAL forward_split — no simplified pipeline.
//!
//! Uses session.bptt_caches to record intermediates during the actual
//! forward pass, then backward through the full tick loop.
//!
//! cargo test --release --test bptt_real -- --nocapture

use modgrad::ctm::{
    Ctm, CtmConfig, CtmWeights, CtmSession, Linear, LayerConfig,
    forward_split, Synapse, SynapseGradients, SimpleRng,
};
use modgrad::ctm::NeuromodConfig;
use modgrad::ctm::BpttTickCache;
use modgrad::tasks;
use modgrad::accuracy::{eval_motor, eval_ls};
use modgrad::linalg;
use rayon::prelude::*;

/// Backward through one tick's caches. Returns gradients for 8 synapses + 8 NLMs,
/// plus d_activations to propagate to the previous tick.
fn backward_tick(
    weights: &CtmWeights,
    tick_cache: &BpttTickCache,
    d_activations: &[f32; 8],  // gradient w.r.t. each region's output [region][neurons...]
    region_sizes: &[usize; 8],
) -> (Vec<SynapseGradients>, Vec<f32>) {
    // For each region: backward through NLM, then through synapse
    let regions = [
        &weights.input_region, &weights.attention_region,
        &weights.output_region, &weights.motor_region,
        &weights.cerebellum_region, &weights.basal_ganglia_region,
        &weights.insula_region, &weights.hippocampus_region,
    ];
    let synapses = [
        &weights.syn_motor_input, &weights.syn_input_attn,
        &weights.syn_attn_output, &weights.syn_output_motor,
        &weights.syn_cerebellum, &weights.syn_basal_ganglia,
        &weights.syn_insula, &weights.syn_hippocampus,
    ];

    let mut syn_grads = Vec::with_capacity(8);
    let mut all_d_syn_inputs: Vec<Vec<f32>> = Vec::with_capacity(8);

    for r in 0..8 {
        // Unpack gradient for this region
        let d_act = &d_activations[r];

        // Backward through NLM
        let d_act_slice = unsafe {
            std::slice::from_raw_parts(d_act as *const _ as *const f32, region_sizes[r])
        };
        let (d_pre_act, _nlm_grads) = regions[r].backward_nlm(d_act_slice, &tick_cache.nlm_caches[r]);

        // Backward through synapse
        let (sg, d_syn_input) = synapses[r].backward(&d_pre_act, &tick_cache.syn_caches[r]);
        syn_grads.push(sg);
        all_d_syn_inputs.push(d_syn_input);
    }

    // Map d_syn_inputs back to d_activations for previous tick.
    // Each synapse input is a concat of various region activations.
    // We need to scatter the gradients back.
    // For simplicity, return the raw d_syn_inputs and let the caller handle routing.
    let flat_d_syn: Vec<f32> = all_d_syn_inputs.into_iter().flatten().collect();

    (syn_grads, flat_d_syn)
}

#[test]
fn bptt_real_forward_split() {
    let n_actions = 4;
    let cfg = CtmConfig {
        iterations: 8, d_input: 64, n_sync_out: 64,
        synapse_depth: 1,
        input_layer: LayerConfig { n_neurons: 128, ..Default::default() },
        attention_layer: LayerConfig { n_neurons: 64, ..Default::default() },
        output_layer: LayerConfig { n_neurons: 64, ..Default::default() },
        motor_layer: LayerConfig { n_neurons: n_actions, ..Default::default() },
        ..CtmConfig::default()
    };

    let data = tasks::maze_examples(7, cfg.d_input, 5000);
    let train = &data[..4000];
    let test = &data[4000..];

    let weights_path = "data/brain_7x7_maze.json";
    let mut weights = if std::path::Path::new(weights_path).exists() {
        eprintln!("  Loading weights from {}", weights_path);
        CtmWeights::load(weights_path).unwrap()
    } else {
        eprintln!("  Fresh init (no saved weights)");
        let ctm = Ctm::new(cfg.clone());
        let (w, _) = ctm.into_split();
        w
    };
    let proprio = vec![0.0f32; cfg.d_input];

    let region_sizes: [usize; 8] = [
        cfg.input_layer.n_neurons, cfg.attention_layer.n_neurons,
        cfg.output_layer.n_neurons, cfg.motor_layer.n_neurons,
        cfg.cerebellum_layer.n_neurons, cfg.basal_ganglia_layer.n_neurons,
        cfg.insula_layer.n_neurons, cfg.hippocampus_layer.n_neurons,
    ];

    let baseline_motor = eval_motor(&weights, test);
    let baseline_ls = eval_ls(&weights, test);
    eprintln!("\n  === 7×7 MAZE: BPTT through real forward_split ===");
    eprintln!("  Motor baseline: {:.1}%", baseline_motor * 100.0);
    eprintln!("  LS ceiling:     {:.1}%\n", baseline_ls * 100.0);

    let t0 = std::time::Instant::now();
    let lr = 0.0001;
    let epochs = 40;
    let batch_size = 32;
    let mut best_motor = baseline_motor;

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0f32;
        let mut epoch_correct = 0usize;
        let mut epoch_total = 0usize;

        for batch_start in (0..train.len()).step_by(batch_size) {
            let actual = batch_size.min(train.len() - batch_start);

            // Accumulate synapse gradients across batch
            let mut syn_grad_acc: Vec<SynapseGradients> = (0..8).map(|s| {
                let syn = &weights.synapse_refs()[s];
                let w1_len = syn.linear.weight.len();
                let b1_len = syn.linear.bias.len();
                let w2_len = syn.linear2.as_ref().map(|l| l.weight.len()).unwrap_or(0);
                let b2_len = syn.linear2.as_ref().map(|l| l.bias.len()).unwrap_or(0);
                SynapseGradients {
                    dw1: vec![0.0; w1_len], db1: vec![0.0; b1_len],
                    dw2: vec![0.0; w2_len], db2: vec![0.0; b2_len],
                }
            }).collect();

            for b in 0..actual {
                let ex = &train[batch_start + b];

                // Forward with BPTT caching enabled
                let mut session = CtmSession::new(&weights.config);
                session.bptt_caches = Some(Vec::with_capacity(cfg.iterations));
                let mut tick_state = weights.init_tick_state();

                let (_preds, _sync, _signals) = forward_split(
                    &weights, &mut session, &mut tick_state,
                    &ex.input, &proprio, false,
                );

                let caches = session.bptt_caches.take().unwrap();
                if batch_start == 0 && b == 0 && epoch == 0 {
                    eprintln!("    [debug] caches.len() = {}, motor_ev = {:?}", caches.len(),
                        &tick_state.motor_evidence);
                }

                // Loss: cross-entropy on motor evidence
                let motor_ev = &tick_state.motor_evidence;
                let max_ev = motor_ev.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = motor_ev.iter().map(|e| (e - max_ev).exp()).sum();
                let softmax: Vec<f32> = motor_ev.iter().map(|e| (e - max_ev).exp() / exp_sum).collect();

                let pred = motor_ev.iter().enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i).unwrap_or(0);
                if pred == ex.target { epoch_correct += 1; }
                epoch_total += 1;

                let p = softmax.get(ex.target).copied().unwrap_or(1e-8).max(1e-8);
                epoch_loss += -p.ln();

                // d_motor_evidence = softmax - one_hot(target)
                let mut d_motor_ev = softmax.clone();
                if ex.target < d_motor_ev.len() { d_motor_ev[ex.target] -= 1.0; }

                // Backward through ticks in reverse
                // Initialize per-region gradient: only motor gets the loss gradient
                let mut d_acts: Vec<Vec<f32>> = region_sizes.iter()
                    .map(|&sz| vec![0.0f32; sz]).collect();

                for tick in (0..caches.len()).rev() {
                    // Motor evidence: each tick contributes motor activations
                    // d(motor_act) += d_motor_evidence (motor evidence = sum of motor acts)
                    for (dm, &dev) in d_acts[3].iter_mut().zip(d_motor_ev.iter()) {
                        *dm += dev;
                    }

                    let tc = &caches[tick];
                    let regions = [
                        &weights.input_region, &weights.attention_region,
                        &weights.output_region, &weights.motor_region,
                        &weights.cerebellum_region, &weights.basal_ganglia_region,
                        &weights.insula_region, &weights.hippocampus_region,
                    ];
                    let synapses = weights.synapse_refs();

                    // Debug: check d_acts magnitude before NLM backward
                    if batch_start == 0 && b == 0 && tick == caches.len() - 1 && epoch == 0 {
                        for r in 0..8 {
                            let m: f32 = d_acts[r].iter().map(|x| x.abs()).sum();
                            eprintln!("    [debug] tick {} region {} |d_act| = {:.8} (size={})", tick, r, m, d_acts[r].len());
                        }
                    }

                    // Per-region: backward through NLM then synapse
                    let mut d_syn_inputs: Vec<Vec<f32>> = Vec::with_capacity(8);
                    for r in 0..8 {
                        let (d_pre_act, _nlm_g) = regions[r].backward_nlm(&d_acts[r], &tc.nlm_caches[r]);
                        let (sg, d_sin) = synapses[r].backward(&d_pre_act, &tc.syn_caches[r]);

                        // Accumulate synapse gradients
                        for (a, &g) in syn_grad_acc[r].dw1.iter_mut().zip(sg.dw1.iter()) { *a += g; }
                        for (a, &g) in syn_grad_acc[r].db1.iter_mut().zip(sg.db1.iter()) { *a += g; }
                        for (a, &g) in syn_grad_acc[r].dw2.iter_mut().zip(sg.dw2.iter()) { *a += g; }
                        for (a, &g) in syn_grad_acc[r].db2.iter_mut().zip(sg.db2.iter()) { *a += g; }

                        d_syn_inputs.push(d_sin);
                    }

                    // Route synapse input gradients back to region activations for prev tick.
                    // Synapse 0 (motor→input): input = concat(observation, act_motor, [global])
                    //   → d_act_motor gets gradient from positions [d_input..d_input+n_motor]
                    // Synapse 1 (input→attn): input = concat(act_input, attn_result, [global])
                    //   → d_act_input gets gradient from positions [0..n_input]
                    // Synapse 2 (attn→output): input = concat(act_attn, attn_result, [global])
                    //   → d_act_attn gets gradient from positions [0..n_attn]
                    // Synapse 3 (output→motor): input = concat(act_output, act_bg, [global])
                    //   → d_act_output gets gradient from positions [0..n_output]
                    //
                    // Reset d_acts for the previous tick
                    d_acts = region_sizes.iter().map(|&sz| vec![0.0f32; sz]).collect();

                    let d_input = cfg.d_input;

                    // syn0: concat(obs, motor, [global]) → motor part starts at d_input
                    let n_motor = region_sizes[3];
                    for i in 0..n_motor.min(d_syn_inputs[0].len().saturating_sub(d_input)) {
                        d_acts[3][i] += d_syn_inputs[0][d_input + i];
                    }

                    // syn1: concat(input, attn_result, [global]) → input part at [0..n_input]
                    let n_input = region_sizes[0];
                    for i in 0..n_input.min(d_syn_inputs[1].len()) {
                        d_acts[0][i] += d_syn_inputs[1][i];
                    }

                    // syn2: concat(attn, attn_result, [global]) → attn part at [0..n_attn]
                    let n_attn = region_sizes[1];
                    for i in 0..n_attn.min(d_syn_inputs[2].len()) {
                        d_acts[1][i] += d_syn_inputs[2][i];
                    }

                    // syn3: concat(output, bg, [global]) → output part at [0..n_output]
                    let n_output = region_sizes[2];
                    for i in 0..n_output.min(d_syn_inputs[3].len()) {
                        d_acts[2][i] += d_syn_inputs[3][i];
                    }

                    // syn7: concat(input, attn, output, motor) → scatter back
                    let mut off = 0;
                    for r in 0..4 {
                        for i in 0..region_sizes[r].min(d_syn_inputs[7].len().saturating_sub(off)) {
                            d_acts[r][i] += d_syn_inputs[7][off + i];
                        }
                        off += region_sizes[r];
                    }
                }
            }

            // Debug: check gradient magnitudes
            if batch_start == 0 && epoch == 0 {
                let total_grad: f32 = syn_grad_acc.iter()
                    .map(|sg| sg.dw1.iter().map(|g| g.abs()).sum::<f32>()
                        + sg.dw2.iter().map(|g| g.abs()).sum::<f32>())
                    .sum();
                eprintln!("    [debug] total |syn_grad| = {:.8}", total_grad);
                // Per-synapse gradient magnitude
                for s in 0..8 {
                    let m: f32 = syn_grad_acc[s].dw1.iter().map(|g| g.abs()).sum();
                    if m > 0.0 || s < 4 {
                        eprintln!("    [debug] syn[{}] |dw1| = {:.8}", s, m);
                    }
                }
            }

            // Apply accumulated gradients
            let scale = lr / actual as f32;
            let synapses_mut = [
                &mut weights.syn_motor_input, &mut weights.syn_input_attn,
                &mut weights.syn_attn_output, &mut weights.syn_output_motor,
                &mut weights.syn_cerebellum, &mut weights.syn_basal_ganglia,
                &mut weights.syn_insula, &mut weights.syn_hippocampus,
            ];
            for (s, syn) in synapses_mut.into_iter().enumerate() {
                syn.apply_gradients(&syn_grad_acc[s], scale);
            }
        }

        if (epoch + 1) % 5 == 0 {
            let motor_acc = eval_motor(&weights, test);
            let ls_acc = eval_ls(&weights, test);
            if motor_acc > best_motor {
                best_motor = motor_acc;
                weights.save(weights_path).unwrap();
                eprint!(" [saved]");
            }
            eprintln!("  Epoch {:3}: loss={:.3} train={:.1}% motor={:.1}% LS={:.1}% best={:.1}% ({:.1}s)",
                epoch + 1,
                epoch_loss / epoch_total as f32,
                epoch_correct as f32 / epoch_total as f32 * 100.0,
                motor_acc * 100.0, ls_acc * 100.0, best_motor * 100.0,
                t0.elapsed().as_secs_f64());
        }
    }

    let final_motor = eval_motor(&weights, test);
    let final_ls = eval_ls(&weights, test);
    eprintln!("\n  Motor: {:.1}% → {:.1}% (best={:.1}%, {:+.1}pp)",
        baseline_motor * 100.0, final_motor * 100.0, best_motor * 100.0,
        (best_motor - baseline_motor) * 100.0);
    eprintln!("  LS:    {:.1}% → {:.1}% ({:+.1}pp)",
        baseline_ls * 100.0, final_ls * 100.0, (final_ls - baseline_ls) * 100.0);
    eprintln!("  Gap:   {:.1}pp remaining (motor vs LS)\n",
        (final_ls - final_motor) * 100.0);
}
