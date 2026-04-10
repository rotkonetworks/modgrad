//! Task suite for testing general intelligence capabilities.
//!
//! Each task tests a specific cognitive ability:
//! - XOR/parity: nonlinear computation (can the CTM compute at all?)
//! - Maze: planning and spatial reasoning (does multi-tick deliberation help?)
//! - Classification: feature extraction (can it learn categories?)
//!
//! Tasks provide (input, target) pairs that the organism trains on.
//! The Angeris bound on each task tells us the ceiling for linear readout.

use crate::weights::Ctm;
use super::linalg;

/// A training example: input tokens → target class.
pub struct Example {
    pub input: Vec<f32>,   // observation vector (fed to CTM)
    pub target: usize,     // target class ID
    pub n_classes: usize,  // total number of classes
}

/// XOR task: the simplest test of nonlinear computation.
/// If the CTM can't learn XOR, nothing else will work.
///
/// Input: 2 bits (as floats: 0.0 or 1.0, embedded in d_input dims)
/// Output: XOR of the bits (0 or 1)
/// Requires exactly 1 nonlinear transformation — a linear model CANNOT solve this.
pub fn xor_examples(d_input: usize) -> Vec<Example> {
    let mut examples = Vec::new();
    // Massive repetition for Hebbian learning
    for _ in 0..500 {
        for &(a, b, target) in &[(0.0f32, 0.0, 0usize), (0.0, 1.0, 1), (1.0, 0.0, 1), (1.0, 1.0, 0)] {
            let mut input = vec![0.0f32; d_input];
            input[0] = a;
            input[1] = b;
            // Spread the bits across more dimensions for richer representation
            if d_input > 2 {
                input[2] = a * 2.0 - 1.0; // [-1, 1] encoding
                input[3] = b * 2.0 - 1.0;
            }
            if d_input > 4 {
                input[4] = a * b;          // AND feature
                input[5] = (a + b).min(1.0); // OR feature
            }
            examples.push(Example { input, target, n_classes: 2 });
        }
    }
    examples
}

/// Parity task: XOR generalized to N bits.
/// Tests if the CTM can chain nonlinear operations.
pub fn parity_examples(n_bits: usize, d_input: usize) -> Vec<Example> {
    let mut examples = Vec::new();
    let n_patterns = 1 << n_bits.min(8); // up to 256 patterns
    for _ in 0..100 {
        for pattern in 0..n_patterns {
            let mut input = vec![0.0f32; d_input];
            let mut parity = 0usize;
            for bit in 0..n_bits {
                let val = ((pattern >> bit) & 1) as f32;
                if bit < d_input { input[bit] = val; }
                parity ^= (pattern >> bit) & 1;
            }
            examples.push(Example { input, target: parity, n_classes: 2 });
        }
    }
    examples
}

/// Maze task: spatial planning.
/// Input: flattened maze grid (0=open, 1=wall) + current position + goal position
/// Output: next move (0=up, 1=right, 2=down, 3=left)
///
/// This tests multi-tick deliberation: the CTM should "think" about the path
/// before the motor region decides.
/// BFS shortest path on a grid. Returns next move (0=up, 1=right, 2=down, 3=left) or None.
fn bfs_next_move(grid: &[u8], size: usize, start: usize, goal: usize) -> Option<usize> {
    if start == goal { return Some(2); } // arbitrary if already there
    let mut visited = vec![false; size * size];
    let mut parent = vec![usize::MAX; size * size];
    let mut queue = std::collections::VecDeque::new();
    visited[start] = true;
    queue.push_back(start);

    let dirs: [(i32, i32, usize); 4] = [(-1, 0, 0), (0, 1, 1), (1, 0, 2), (0, -1, 3)];

    while let Some(pos) = queue.pop_front() {
        let r = (pos / size) as i32;
        let c = (pos % size) as i32;
        for &(dr, dc, _) in &dirs {
            let nr = r + dr;
            let nc = c + dc;
            if nr < 0 || nr >= size as i32 || nc < 0 || nc >= size as i32 { continue; }
            let npos = nr as usize * size + nc as usize;
            if visited[npos] || grid[npos] == 1 { continue; }
            visited[npos] = true;
            parent[npos] = pos;
            if npos == goal {
                // Trace back to find first step from start
                let mut cur = goal;
                while parent[cur] != start { cur = parent[cur]; }
                // cur is the cell adjacent to start on the shortest path
                let cr = cur / size;
                let cc = cur % size;
                let sr = start / size;
                let sc = start % size;
                if cr < sr { return Some(0); }      // up
                if cc > sc { return Some(1); }      // right
                if cr > sr { return Some(2); }      // down
                return Some(3);                      // left
            }
            queue.push_back(npos);
        }
    }
    None // no path
}

pub fn maze_examples(size: usize, d_input: usize, count: usize) -> Vec<Example> {
    let mut rng = modgrad_compute::neuron::SimpleRng::new(42);
    let mut examples = Vec::new();
    let goal = size * size - 1;

    while examples.len() < count {
        // Generate maze with ~25% walls
        let mut grid = vec![0u8; size * size];
        for cell in &mut grid {
            if rng.next_f32() < 0.25 { *cell = 1; }
        }
        grid[0] = 0;
        grid[goal] = 0;

        // Pick random open position
        let mut pos = 0;
        for _ in 0..200 {
            let p = (rng.next_u64() as usize) % (size * size);
            if grid[p] == 0 && p != goal { pos = p; break; }
        }

        // BFS for ground-truth next move
        let target = match bfs_next_move(&grid, size, pos, goal) {
            Some(t) => t,
            None => continue, // no path, skip this maze
        };

        // Encode: grid cells + normalized position + normalized goal
        let mut input = vec![0.0f32; d_input];
        for (i, &cell) in grid.iter().enumerate().take(d_input.min(size * size)) {
            input[i] = cell as f32;
        }
        let grid_end = (size * size).min(d_input);
        if grid_end + 4 <= d_input {
            input[grid_end] = (pos / size) as f32 / size as f32;
            input[grid_end + 1] = (pos % size) as f32 / size as f32;
            input[grid_end + 2] = (size - 1) as f32 / size as f32;
            input[grid_end + 3] = (size - 1) as f32 / size as f32;
        }

        examples.push(Example { input, target, n_classes: 4 });
    }
    examples
}

/// Simple classification: is the input "mostly ones" or "mostly zeros"?
/// Tests basic feature extraction.
pub fn majority_examples(n_bits: usize, d_input: usize) -> Vec<Example> {
    let mut rng = modgrad_compute::neuron::SimpleRng::new(7777);
    let mut examples = Vec::new();
    for _ in 0..2000 {
        let mut input = vec![0.0f32; d_input];
        let mut ones = 0;
        for i in 0..n_bits.min(d_input) {
            let bit = if rng.next_f32() > 0.5 { 1.0 } else { 0.0 };
            input[i] = bit;
            if bit > 0.5 { ones += 1; }
        }
        let target = if ones > n_bits / 2 { 1 } else { 0 };
        examples.push(Example { input, target, n_classes: 2 });
    }
    examples
}

/// Train a CTM directly on a task (bypass the Organism's text-oriented pipeline).
/// Returns (final_loss, accuracy, angeris_optimal_loss, angeris_optimal_accuracy).
pub fn train_ctm_on_task(
    ctm: &mut Ctm,
    examples: &[Example],
    epochs: usize,
    report_every: usize,
) -> (f32, f32, f32, f32) {
    if examples.is_empty() { return (f32::MAX, 0.0, f32::MAX, 0.0); }

    let n_classes = examples[0].n_classes;
    let sync_dim = ctm.config.n_sync_out;

    // Output projection: sync → n_classes
    let mut out_weights = vec![0.0f32; n_classes * sync_dim];
    let mut out_bias = vec![0.0f32; n_classes];
    // Small random init
    let mut rng = modgrad_compute::neuron::SimpleRng::new(12345);
    for w in &mut out_weights { *w = rng.next_normal() * 0.1; }

    let mut all_syncs: Vec<(Vec<f32>, usize)> = Vec::new();

    for epoch in 0..epochs {
        let mut total_loss = 0.0f32;
        let mut correct = 0;
        let mut count = 0;
        all_syncs.clear();

        for ex in examples {
            // CTM forward
            let mut state = ctm.init_state();
            let (_preds, sync) = ctm.forward(&ex.input, &mut state, epoch > 0);

            // Linear readout: logits = W @ sync + bias
            let mut logits = vec![0.0f32; n_classes];
            for c in 0..n_classes {
                let row = c * sync_dim;
                let mut dot = out_bias[c];
                for k in 0..sync_dim.min(sync.len()) {
                    dot += out_weights[row + k] * sync[k];
                }
                logits[c] = dot;
            }

            // Softmax + CE loss
            let max_l: f32 = logits.iter().fold(f32::MIN, |a, &b| a.max(b));
            let exps: Vec<f32> = logits.iter().map(|&x| (x - max_l).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let loss = (sum.ln() + max_l) - logits[ex.target];
            total_loss += loss;

            // Accuracy
            let pred = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i).unwrap_or(0);
            if pred == ex.target { correct += 1; }
            count += 1;

            // Collect for LS solve
            all_syncs.push((sync.clone(), ex.target));

            // Simple online gradient-free weight update:
            // Increase weight for correct class, decrease for predicted class
            let lr = 0.01;
            if pred != ex.target {
                for k in 0..sync_dim.min(sync.len()) {
                    out_weights[ex.target * sync_dim + k] += lr * sync[k];
                    out_weights[pred * sync_dim + k] -= lr * sync[k];
                }
                out_bias[ex.target] += lr;
                out_bias[pred] -= lr;
            }
        }

        let avg_loss = total_loss / count as f32;
        let accuracy = correct as f32 / count as f32;

        if epoch % report_every == 0 || epoch == epochs - 1 {
            eprintln!("  epoch {:4}: loss={:.4} acc={:.1}% ({}/{})", epoch, avg_loss, accuracy * 100.0, correct, count);
        }

        // Periodic LS consolidation of output weights
        if epoch > 0 && epoch % 10 == 0 && all_syncs.len() >= 50 {
            let xs: Vec<&[f32]> = all_syncs.iter().map(|(s, _)| s.as_slice()).collect();
            let mut xtx = vec![0.0f32; sync_dim * sync_dim];
            linalg::accumulate_xtx(&mut xtx, &xs, sync_dim);
            // Adaptive regularization: prevents Cholesky explosion on low-rank problems.
    let lambda = (1.0 / xs.len().max(1) as f32).max(1e-3);
    for i in 0..sync_dim { xtx[i * sync_dim + i] += lambda; }

            if let Some(l) = linalg::cholesky(&xtx, sync_dim) {
                for c in 0..n_classes {
                    let mut xty = vec![0.0f32; sync_dim];
                    for (sync, target) in &all_syncs {
                        if *target == c {
                            for k in 0..sync_dim { xty[k] += sync[k]; }
                        }
                    }
                    let z = linalg::forward_solve(&l, &xty, sync_dim);
                    let w = linalg::backward_solve(&l, &z, sync_dim);
                    for k in 0..sync_dim {
                        out_weights[c * sync_dim + k] = 0.7 * out_weights[c * sync_dim + k] + 0.3 * w[k];
                    }
                }
            }
        }
    }

    // Final evaluation
    let mut final_loss = 0.0f32;
    let mut final_correct = 0;
    let n = examples.len();
    for ex in examples {
        let mut state = ctm.init_state();
        let (_, sync) = ctm.forward(&ex.input, &mut state, false);
        let mut logits = vec![0.0f32; n_classes];
        for c in 0..n_classes {
            let row = c * sync_dim;
            logits[c] = out_bias[c] + (0..sync_dim.min(sync.len()))
                .map(|k| out_weights[row + k] * sync[k]).sum::<f32>();
        }
        let max_l: f32 = logits.iter().fold(f32::MIN, |a, &b| a.max(b));
        let exps: Vec<f32> = logits.iter().map(|&x| (x - max_l).exp()).collect();
        let sum: f32 = exps.iter().sum();
        final_loss += (sum.ln() + max_l) - logits[ex.target];
        let pred = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
        if pred == ex.target { final_correct += 1; }
    }
    let final_avg = final_loss / n as f32;
    let final_acc = final_correct as f32 / n as f32;

    // Angeris optimal: LS-solve the entire output projection
    let xs: Vec<&[f32]> = all_syncs.iter().map(|(s, _)| s.as_slice()).collect();
    let mut xtx = vec![0.0f32; sync_dim * sync_dim];
    linalg::accumulate_xtx(&mut xtx, &xs, sync_dim);
    // Regularization: scale with dimension and sample count.
    // High sync_dim with few unique inputs → severely ill-conditioned.
    let lambda = ((sync_dim as f32) / xs.len().max(1) as f32).max(0.01);
    for i in 0..sync_dim { xtx[i * sync_dim + i] += lambda; }

    let (opt_loss, opt_acc) = if let Some(l) = linalg::cholesky(&xtx, sync_dim) {
        let mut opt_w = vec![0.0f32; n_classes * sync_dim];
        for c in 0..n_classes {
            let mut xty = vec![0.0f32; sync_dim];
            for (sync, target) in &all_syncs {
                if *target == c {
                    for k in 0..sync_dim { xty[k] += sync[k]; }
                }
            }
            let z = linalg::forward_solve(&l, &xty, sync_dim);
            let w = linalg::backward_solve(&l, &z, sync_dim);
            for k in 0..sync_dim { opt_w[c * sync_dim + k] = w[k]; }
        }

        let mut loss = 0.0f32;
        let mut correct = 0;
        for (sync, target) in &all_syncs {
            let mut logits = vec![0.0f32; n_classes];
            for c in 0..n_classes {
                logits[c] = (0..sync_dim.min(sync.len()))
                    .map(|k| opt_w[c * sync_dim + k] * sync[k]).sum();
            }
            let max_l: f32 = logits.iter().fold(f32::MIN, |a, &b| a.max(b));
            let exps: Vec<f32> = logits.iter().map(|&x| (x - max_l).exp()).collect();
            let sum: f32 = exps.iter().sum();
            loss += (sum.ln() + max_l) - logits[*target];
            let pred = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i).unwrap_or(0);
            if pred == *target { correct += 1; }
        }
        (loss / all_syncs.len() as f32, correct as f32 / all_syncs.len() as f32)
    } else {
        (f32::MAX, 0.0)
    };

    (final_avg, final_acc, opt_loss, opt_acc)
}

/// Compute Angeris bound diagnostic for collected (sync, target) pairs.
/// Returns (current_loss, optimal_loss, optimal_accuracy, effective_rank).
pub fn angeris_diagnostic(
    syncs: &[(Vec<f32>, usize)],
    weights: &[f32],
    bias: &[f32],
    sync_dim: usize,
    n_classes: usize,
) -> (f32, f32, f32, usize) {
    if syncs.is_empty() { return (0.0, 0.0, 0.0, 0); }

    // Current loss
    let mut current_loss = 0.0f32;
    for (sync, target) in syncs {
        let mut logits = vec![0.0f32; n_classes];
        for c in 0..n_classes {
            logits[c] = bias.get(c).copied().unwrap_or(0.0)
                + (0..sync_dim.min(sync.len()))
                    .map(|k| weights.get(c * sync_dim + k).copied().unwrap_or(0.0) * sync[k])
                    .sum::<f32>();
        }
        let max_l: f32 = logits.iter().fold(f32::MIN, |a, &b| a.max(b));
        let log_sum_exp = logits.iter().map(|&x| (x - max_l).exp()).sum::<f32>().ln() + max_l;
        current_loss += log_sum_exp - logits[*target];
    }

    // LS optimal
    let xs: Vec<&[f32]> = syncs.iter().map(|(s, _)| s.as_slice()).collect();
    let mut xtx = vec![0.0f32; sync_dim * sync_dim];
    linalg::accumulate_xtx(&mut xtx, &xs, sync_dim);
    // Adaptive regularization: prevents Cholesky explosion on low-rank problems.
    let lambda = (1.0 / xs.len().max(1) as f32).max(1e-3);
    for i in 0..sync_dim { xtx[i * sync_dim + i] += lambda; }

    // Effective rank from diagonal
    let mut diag: Vec<f32> = (0..sync_dim).map(|i| xtx[i * sync_dim + i]).collect();
    diag.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let total: f32 = diag.iter().sum();
    let mut cum = 0.0f32;
    let mut rank = sync_dim;
    for (i, &d) in diag.iter().enumerate() {
        cum += d;
        if cum > total * 0.95 { rank = i + 1; break; }
    }

    let (opt_loss, opt_acc) = if let Some(l) = linalg::cholesky(&xtx, sync_dim) {
        let mut opt_w = vec![0.0f32; n_classes * sync_dim];
        for c in 0..n_classes {
            let mut xty = vec![0.0f32; sync_dim];
            for (sync, target) in syncs {
                if *target == c {
                    for k in 0..sync_dim { xty[k] += sync[k]; }
                }
            }
            let z = linalg::forward_solve(&l, &xty, sync_dim);
            let w = linalg::backward_solve(&l, &z, sync_dim);
            for k in 0..sync_dim { opt_w[c * sync_dim + k] = w[k]; }
        }
        let mut loss = 0.0f32;
        let mut correct = 0;
        for (sync, target) in syncs {
            let mut logits = vec![0.0f32; n_classes];
            for c in 0..n_classes {
                logits[c] = (0..sync_dim.min(sync.len()))
                    .map(|k| opt_w[c * sync_dim + k] * sync[k]).sum();
            }
            let max_l: f32 = logits.iter().fold(f32::MIN, |a, &b| a.max(b));
            let exps: Vec<f32> = logits.iter().map(|&x| (x - max_l).exp()).collect();
            let sum: f32 = exps.iter().sum();
            loss += (sum.ln() + max_l) - logits[*target];
            let pred = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i).unwrap_or(0);
            if pred == *target { correct += 1; }
        }
        (loss / syncs.len() as f32, correct as f32 / syncs.len() as f32)
    } else {
        (f32::MAX, 0.0)
    };

    (current_loss / syncs.len() as f32, opt_loss, opt_acc, rank)
}

// ─── MLP readout: nonlinear 2-layer classifier ─────────────

/// Train a 2-layer MLP on collected (sync, target) pairs via mini-batch SGD.
/// Architecture: sync_dim → hidden → n_classes with ReLU activation.
/// Returns (loss, accuracy) on the training set.
pub fn train_mlp_readout(
    syncs: &[(Vec<f32>, usize)],
    sync_dim: usize,
    n_classes: usize,
    hidden: usize,
    epochs: usize,
    lr: f32,
) -> (f32, f32) {
    if syncs.is_empty() { return (f32::MAX, 0.0); }
    let mut rng = modgrad_compute::neuron::SimpleRng::new(7777);

    // Kaiming init
    let scale1 = (2.0 / sync_dim as f32).sqrt();
    let scale2 = (2.0 / hidden as f32).sqrt();
    let mut w1 = vec![0.0f32; hidden * sync_dim]; // [hidden × sync_dim]
    let mut b1 = vec![0.0f32; hidden];
    let mut w2 = vec![0.0f32; n_classes * hidden]; // [n_classes × hidden]
    let mut b2 = vec![0.0f32; n_classes];
    for w in &mut w1 { *w = rng.next_normal() * scale1; }
    for w in &mut w2 { *w = rng.next_normal() * scale2; }

    for _epoch in 0..epochs {
        for (sync, target) in syncs {
            // Forward: h = relu(W1 @ x + b1)
            let mut h = vec![0.0f32; hidden];
            for j in 0..hidden {
                let mut dot = b1[j];
                for k in 0..sync_dim.min(sync.len()) {
                    dot += w1[j * sync_dim + k] * sync[k];
                }
                h[j] = dot.max(0.0); // ReLU
            }

            // logits = W2 @ h + b2
            let mut logits = vec![0.0f32; n_classes];
            for c in 0..n_classes {
                let mut dot = b2[c];
                for j in 0..hidden { dot += w2[c * hidden + j] * h[j]; }
                logits[c] = dot;
            }

            // Softmax
            let max_l: f32 = logits.iter().fold(f32::MIN, |a, &b| a.max(b));
            let exps: Vec<f32> = logits.iter().map(|&x| (x - max_l).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();

            // Backward: d_logits = probs - one_hot(target)
            let mut d_logits = probs.clone();
            d_logits[*target] -= 1.0;

            // Gradient for W2, b2
            for c in 0..n_classes {
                for j in 0..hidden {
                    w2[c * hidden + j] -= lr * d_logits[c] * h[j];
                }
                b2[c] -= lr * d_logits[c];
            }

            // Gradient for h (backprop through ReLU)
            let mut d_h = vec![0.0f32; hidden];
            for j in 0..hidden {
                let mut grad = 0.0f32;
                for c in 0..n_classes { grad += d_logits[c] * w2[c * hidden + j]; }
                d_h[j] = if h[j] > 0.0 { grad } else { 0.0 }; // ReLU gradient
            }

            // Gradient for W1, b1
            for j in 0..hidden {
                for k in 0..sync_dim.min(sync.len()) {
                    w1[j * sync_dim + k] -= lr * d_h[j] * sync[k];
                }
                b1[j] -= lr * d_h[j];
            }
        }
    }

    // Evaluate
    let mut loss = 0.0f32;
    let mut correct = 0;
    for (sync, target) in syncs {
        let mut h = vec![0.0f32; hidden];
        for j in 0..hidden {
            let mut dot = b1[j];
            for k in 0..sync_dim.min(sync.len()) { dot += w1[j * sync_dim + k] * sync[k]; }
            h[j] = dot.max(0.0);
        }
        let mut logits = vec![0.0f32; n_classes];
        for c in 0..n_classes {
            let mut dot = b2[c];
            for j in 0..hidden { dot += w2[c * hidden + j] * h[j]; }
            logits[c] = dot;
        }
        let max_l: f32 = logits.iter().fold(f32::MIN, |a, &b| a.max(b));
        let exps: Vec<f32> = logits.iter().map(|&x| (x - max_l).exp()).collect();
        let sum: f32 = exps.iter().sum();
        loss += (sum.ln() + max_l) - logits[*target];
        let pred = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
        if pred == *target { correct += 1; }
    }
    (loss / syncs.len() as f32, correct as f32 / syncs.len() as f32)
}

/// Train CTM with MLP readout instead of linear. Same interface as train_ctm_on_task.
/// Collects sync features first (1 epoch), then trains MLP on them.
pub fn train_ctm_mlp(
    ctm: &mut Ctm,
    examples: &[Example],
    mlp_epochs: usize,
) -> (f32, f32) {
    if examples.is_empty() { return (f32::MAX, 0.0); }
    let n_classes = examples[0].n_classes;
    let sync_dim = ctm.config.n_sync_out;

    // Collect sync features
    let syncs: Vec<(Vec<f32>, usize)> = examples.iter().map(|ex| {
        let mut state = ctm.init_state();
        let (_, sync) = ctm.forward(&ex.input, &mut state, false);
        (sync, ex.target)
    }).collect();

    let hidden = 128.min(sync_dim);
    train_mlp_readout(&syncs, sync_dim, n_classes, hidden, mlp_epochs, 0.005)
}

/// 64-bit parity: randomly sampled bit vectors (exhaustive is impossible for 64 bits).
pub fn parity_examples_large(n_bits: usize, d_input: usize, n_samples: usize) -> Vec<Example> {
    let mut rng = modgrad_compute::neuron::SimpleRng::new(42);
    let mut examples = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let mut input = vec![0.0f32; d_input];
        let mut parity = 0usize;
        for bit in 0..n_bits {
            let val = (rng.next_u64() & 1) as f32;
            if bit < d_input { input[bit] = val; }
            parity ^= val as usize;
        }
        examples.push(Example { input, target: parity, n_classes: 2 });
    }
    examples
}

// ─── Tier 1: Primitive byte operations ─────────────────────
// Tests: can the CTM manipulate bytes? Copy, reverse, classify, convert.
// Should pass with 16-64 neurons, 4 ticks.

/// Echo/copy: input byte → same byte.
/// The simplest possible task. If this fails, nothing else will work.
/// Tests: can the CTM preserve information through the tick loop?
pub fn echo_examples(d_input: usize) -> Vec<Example> {
    let mut examples = Vec::new();
    for _ in 0..5 {
        for byte in 0..128u8 {
            let mut input = vec![0.0f32; d_input];
            // Encode byte value in multiple ways for robustness
            if d_input > 0 { input[0] = byte as f32 / 128.0 - 1.0; }
            // One-hot in available dims
            let slot = (byte as usize) % (d_input.saturating_sub(1).max(1));
            if d_input > 1 { input[1 + slot] = 1.0; }
            examples.push(Example { input, target: byte as usize, n_classes: 128 });
        }
    }
    examples
}

/// Hex digit classification: byte → hex value (0-15).
/// "0"-"9" → 0-9, "a"-"f" → 10-15, "A"-"F" → 10-15.
/// Tests: can it learn a lookup table with case-insensitive aliasing?
pub fn hex_classify_examples(d_input: usize) -> Vec<Example> {
    let mut examples = Vec::new();
    let hex_chars: Vec<(u8, usize)> = vec![
        (b'0',0),(b'1',1),(b'2',2),(b'3',3),(b'4',4),(b'5',5),(b'6',6),(b'7',7),
        (b'8',8),(b'9',9),(b'a',10),(b'b',11),(b'c',12),(b'd',13),(b'e',14),(b'f',15),
        (b'A',10),(b'B',11),(b'C',12),(b'D',13),(b'E',14),(b'F',15),
    ];
    for _ in 0..30 {
        for &(byte, val) in &hex_chars {
            let mut input = vec![0.0f32; d_input];
            if d_input > 0 { input[0] = byte as f32 / 128.0 - 1.0; }
            let slot = (byte as usize) % (d_input.saturating_sub(1).max(1));
            if d_input > 1 { input[1 + slot] = 1.0; }
            examples.push(Example { input, target: val, n_classes: 16 });
        }
    }
    examples
}

/// Binary value: "0"/"1" byte → 0/1 class.
/// Trivial binary classification — the atom of computation.
pub fn binary_classify_examples(d_input: usize) -> Vec<Example> {
    let mut examples = Vec::new();
    for _ in 0..100 {
        for &(byte, val) in &[(b'0', 0usize), (b'1', 1)] {
            let mut input = vec![0.0f32; d_input];
            if d_input > 0 { input[0] = byte as f32 / 128.0 - 1.0; }
            if d_input > 1 { input[1] = val as f32; }
            examples.push(Example { input, target: val, n_classes: 2 });
        }
    }
    examples
}

/// Bit NOT: 0→1, 1→0. Tests: can the CTM invert a signal?
pub fn bit_not_examples(d_input: usize) -> Vec<Example> {
    let mut examples = Vec::new();
    for _ in 0..100 {
        for &(val, target) in &[(0usize, 1usize), (1, 0)] {
            let mut input = vec![0.0f32; d_input];
            if d_input > 0 { input[0] = val as f32 * 2.0 - 1.0; }
            if d_input > 1 { input[1] = val as f32; }
            examples.push(Example { input, target, n_classes: 2 });
        }
    }
    examples
}

// ─── Tier 2: Compositional / sequence operations ───────────
// Tests: can the CTM chain operations, follow rules, generalize?
// Needs 64+ neurons, 8+ ticks.

/// Hex pair → byte value: "4","1" → 65 (0x41 = 'A').
/// Two sequential hex digits → combined value.
/// Tests: can it combine two inputs into one output?
/// Input: two hex values encoded as [high_nibble, low_nibble, ...]
pub fn hex_pair_examples(d_input: usize) -> Vec<Example> {
    let mut examples = Vec::new();
    for hi in 0..16u8 {
        for lo in 0..16u8 {
            let val = (hi << 4) | lo;
            if val >= 128 { continue; } // keep n_classes manageable
            let mut input = vec![0.0f32; d_input];
            if d_input > 0 { input[0] = hi as f32 / 8.0 - 1.0; }
            if d_input > 1 { input[1] = lo as f32 / 8.0 - 1.0; }
            // Also encode as bits
            for bit in 0..4 {
                if d_input > 2 + bit { input[2 + bit] = ((hi >> bit) & 1) as f32; }
                if d_input > 6 + bit { input[6 + bit] = ((lo >> bit) & 1) as f32; }
            }
            for _ in 0..10 {
                examples.push(Example {
                    input: input.clone(),
                    target: val as usize,
                    n_classes: 128,
                });
            }
        }
    }
    examples
}

/// 4-bit addition: a + b mod 16.
/// Tests: can the CTM do arithmetic? Requires carry logic.
pub fn nibble_add_examples(d_input: usize) -> Vec<Example> {
    let mut examples = Vec::new();
    for a in 0..16u8 {
        for b in 0..16u8 {
            let sum = (a + b) % 16;
            let mut input = vec![0.0f32; d_input];
            if d_input > 0 { input[0] = a as f32 / 8.0 - 1.0; }
            if d_input > 1 { input[1] = b as f32 / 8.0 - 1.0; }
            for bit in 0..4 {
                if d_input > 2 + bit { input[2 + bit] = ((a >> bit) & 1) as f32; }
                if d_input > 6 + bit { input[6 + bit] = ((b >> bit) & 1) as f32; }
            }
            for _ in 0..20 {
                examples.push(Example {
                    input: input.clone(),
                    target: sum as usize,
                    n_classes: 16,
                });
            }
        }
    }
    examples
}

/// 3-bit binary to decimal: "101" → 5.
/// Input: 3 bits encoded as floats. Output: decimal value 0-7.
/// Tests: base conversion (positional value system).
pub fn bin_to_dec_examples(d_input: usize) -> Vec<Example> {
    let mut examples = Vec::new();
    for val in 0..8u8 {
        let mut input = vec![0.0f32; d_input];
        for bit in 0..3 {
            if d_input > bit { input[bit] = ((val >> (2 - bit)) & 1) as f32 * 2.0 - 1.0; }
        }
        for _ in 0..200 {
            examples.push(Example { input: input.clone(), target: val as usize, n_classes: 8 });
        }
    }
    examples
}

// ─── Tier 3: BLC and abstract reasoning ────────────────────
// Tests: can the CTM evaluate programs? Learn rules from examples?
// Needs 256+ neurons, 16+ ticks. The AGI frontier.

/// BLC token classification: given a BLC bitstring prefix,
/// classify the next construct as ABS(00), APP(01), or VAR(1*0).
/// Tests: can it parse a formal grammar from byte sequences?
pub fn blc_parse_examples(d_input: usize) -> Vec<Example> {
    let mut examples = Vec::new();
    // BLC: 00=abstraction, 01=application, 1*0=variable
    // Generate random valid BLC prefixes and classify the construct
    let mut rng = 42u64;
    let mut next = || -> u8 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng >> 32) as u8
    };

    for _ in 0..200 {
        // Abstraction: 00
        let mut input = vec![0.0f32; d_input];
        if d_input > 0 { input[0] = 0.0; } // bit 1 = 0
        if d_input > 1 { input[1] = 0.0; } // bit 2 = 0
        examples.push(Example { input, target: 0, n_classes: 3 }); // ABS

        // Application: 01
        let mut input = vec![0.0f32; d_input];
        if d_input > 0 { input[0] = 0.0; }
        if d_input > 1 { input[1] = 1.0; }
        examples.push(Example { input, target: 1, n_classes: 3 }); // APP

        // Variable index 0: 10
        let mut input = vec![0.0f32; d_input];
        if d_input > 0 { input[0] = 1.0; }
        if d_input > 1 { input[1] = 0.0; }
        // Add noise in remaining dims for diversity
        for j in 2..d_input { input[j] = (next() as f32 / 255.0) * 0.1; }
        examples.push(Example { input, target: 2, n_classes: 3 }); // VAR
    }
    examples
}

/// BLC evaluation: given a BLC expression applied to TRUE/FALSE,
/// predict the output (TRUE=1, FALSE=0).
/// Expressions: identity (λ0), TRUE (λλ1), FALSE (λλ0), NOT (λ[[0 F]T]).
/// Tests: can the CTM evaluate lambda calculus?
pub fn blc_eval_examples(d_input: usize) -> Vec<Example> {
    let mut examples = Vec::new();

    // Encode BLC programs as fixed-length feature vectors.
    // Each program is a known expression; we encode its structure + argument.
    // 0 = identity applied to X → X
    // 1 = TRUE applied to X Y → X
    // 2 = FALSE applied to X Y → Y
    // 3 = NOT applied to X → ¬X

    for _ in 0..200 {
        // identity(TRUE) → TRUE(1)
        let mut input = vec![0.0f32; d_input];
        if d_input > 0 { input[0] = 0.0; } // program = identity
        if d_input > 1 { input[1] = 1.0; } // arg = TRUE
        examples.push(Example { input, target: 1, n_classes: 2 });

        // identity(FALSE) → FALSE(0)
        let mut input = vec![0.0f32; d_input];
        if d_input > 0 { input[0] = 0.0; }
        if d_input > 1 { input[1] = 0.0; }
        examples.push(Example { input, target: 0, n_classes: 2 });

        // NOT(TRUE) → FALSE(0)
        let mut input = vec![0.0f32; d_input];
        if d_input > 0 { input[0] = 1.0; } // program = NOT
        if d_input > 1 { input[1] = 1.0; } // arg = TRUE
        examples.push(Example { input, target: 0, n_classes: 2 });

        // NOT(FALSE) → TRUE(1)
        let mut input = vec![0.0f32; d_input];
        if d_input > 0 { input[0] = 1.0; }
        if d_input > 1 { input[1] = 0.0; }
        examples.push(Example { input, target: 1, n_classes: 2 });
    }
    examples
}

/// Church numeral successor: given n (0-7), predict n+1 mod 8.
/// Tests: can it learn the successor function (Peano arithmetic)?
pub fn church_succ_examples(d_input: usize) -> Vec<Example> {
    let mut examples = Vec::new();
    for _ in 0..200 {
        for n in 0..8u8 {
            let mut input = vec![0.0f32; d_input];
            // Encode as church numeral features
            if d_input > 0 { input[0] = n as f32 / 4.0 - 1.0; }
            // Binary encoding
            for bit in 0..3 {
                if d_input > 1 + bit { input[1 + bit] = ((n >> bit) & 1) as f32 * 2.0 - 1.0; }
            }
            // Unary encoding (thermometer)
            for i in 0..(n as usize).min(d_input.saturating_sub(4)) {
                if d_input > 4 + i { input[4 + i] = 1.0; }
            }
            examples.push(Example {
                input: input.clone(),
                target: ((n + 1) % 8) as usize,
                n_classes: 8,
            });
        }
    }
    examples
}

/// Run the extended task suite (Tier 1 + 2 + 3).
pub fn run_extended_suite(d_model: usize) {
    let d_input = d_model; // observation dimension matches model
    let _ticks = 8;

    eprintln!("\n=== EXTENDED TASK SUITE ===\n");

    // ── Tier 1: Primitives ──
    eprintln!("─── Tier 1: Primitive Operations ───\n");

    let tier1 = [
        ("Binary classify (0/1)", binary_classify_examples(d_input)),
        ("Bit NOT", bit_not_examples(d_input)),
        ("Hex classify (0-F→0-15)", hex_classify_examples(d_input)),
        ("3-bit binary→decimal", bin_to_dec_examples(d_input)),
    ];

    // Helper: run a task with properly-sized CTM
    let run = |name: &str, examples: &[Example], epochs: usize, pass_thresh: f32| {
        let n_classes = examples[0].n_classes;
        let mut ctm = crate::ctm::build_ctm("poker", d_model, n_classes); // 8 ticks, faster
        ctm.enable_hebbian();
        let (_, final_acc, _, opt_acc) = train_ctm_on_task(&mut ctm, examples, epochs, 100);
        let status = if final_acc >= pass_thresh { "PASS" }
            else if opt_acc >= pass_thresh { "REPR OK" }
            else { "FAIL" };
        eprintln!("  [{status}] {name}: {:.1}% (optimal {:.1}%)", final_acc * 100.0, opt_acc * 100.0);
    };

    run("Binary classify (0/1)", &tier1[0].1, 20, 0.9);
    run("Bit NOT", &tier1[1].1, 20, 0.9);
    run("Hex classify (0-F→0-15)", &tier1[2].1, 20, 0.9);
    run("3-bit binary→decimal", &tier1[3].1, 20, 0.9);

    // ── Tier 2: Compositional ──
    eprintln!("\n─── Tier 2: Compositional Reasoning ───\n");

    let tier2: Vec<(&str, Vec<Example>)> = vec![
        ("Nibble addition (a+b mod 16)", nibble_add_examples(d_input)),
        ("Hex pair→byte (hi,lo→val)", hex_pair_examples(d_input)),
    ];

    run("Nibble addition (a+b mod 16)", &tier2[0].1, 50, 0.8);
    run("Hex pair→byte (hi,lo→val)", &tier2[1].1, 50, 0.8);

    // ── Tier 3: BLC / Abstract ──
    eprintln!("\n─── Tier 3: BLC & Abstract Reasoning ───\n");

    let tier3: Vec<(&str, Vec<Example>)> = vec![
        ("BLC token parse (00/01/1*0)", blc_parse_examples(d_input)),
        ("BLC eval (identity/NOT on T/F)", blc_eval_examples(d_input)),
        ("Church successor (n→n+1 mod 8)", church_succ_examples(d_input)),
    ];

    run("BLC token parse (00/01/1*0)", &tier3[0].1, 50, 0.8);
    run("BLC eval (identity/NOT on T/F)", &tier3[1].1, 50, 0.8);
    run("Church successor (n→n+1 mod 8)", &tier3[2].1, 50, 0.8);

    // ── Tier 3+: Abstract Reasoning ──
    eprintln!("\n─── Tier 3+: Abstract & Symbolic ───\n");

    let t3b_analogy = analogy_examples(d_input);
    let t3b_varbind = variable_binding_examples(d_input);

    run("Analogy (a:A::m:? → case flip)", &t3b_analogy, 50, 0.8);
    run("Variable binding (x=5, x? → 5)", &t3b_varbind, 50, 0.8);

    eprintln!("\n=== END EXTENDED SUITE ===");
}

/// Analogy task: A:B::C:? — learn a transformation from one pair, apply to another.
/// Tests abstract relational reasoning (Hofstadter/Chollet core test).
///
/// Relationships:
///   - Case flip: a→A, b→B (offset = -32 in ASCII)
///   - Successor: 3→4, 7→8 (offset = +1)
///   - Predecessor: 5→4, 9→8 (offset = -1)
///
/// Input encodes: [a, b, c, relationship_hint]
/// Target: d such that relationship(c) = d
pub fn analogy_examples(d_input: usize) -> Vec<Example> {
    let mut examples = Vec::new();

    for _ in 0..50 {
        // Case flip lowercase→uppercase: a:A::c:C
        for c in b'a'..=b'z' {
            let a = b'a' + (c - b'a' + 7) % 26; // different from c
            let b_val = a - 32; // uppercase of a
            let d = c - 32; // uppercase of c
            let mut input = vec![0.0f32; d_input];
            if d_input > 0 { input[0] = a as f32 / 128.0 - 1.0; }
            if d_input > 1 { input[1] = b_val as f32 / 128.0 - 1.0; }
            if d_input > 2 { input[2] = c as f32 / 128.0 - 1.0; }
            if d_input > 3 { input[3] = 0.5; } // relationship hint: case flip
            examples.push(Example { input, target: d as usize, n_classes: 128 });
        }

        // Successor: 3:4::7:8
        for n in b'0'..b'9' {
            let a = b'0' + (n - b'0' + 3) % 9;
            let b_val = a + 1;
            let d = n + 1;
            let mut input = vec![0.0f32; d_input];
            if d_input > 0 { input[0] = a as f32 / 128.0 - 1.0; }
            if d_input > 1 { input[1] = b_val as f32 / 128.0 - 1.0; }
            if d_input > 2 { input[2] = n as f32 / 128.0 - 1.0; }
            if d_input > 3 { input[3] = -0.5; } // relationship hint: successor
            examples.push(Example { input, target: d as usize, n_classes: 128 });
        }
    }
    examples
}

/// Variable binding: given "x=5", query "x?" → 5.
/// Tests associative memory (hippocampal CAM).
///
/// Input encodes: [var_name, value, query_name, match_indicator]
/// Target: digit value (0-9) if query matches, else 0
/// Reduced to 10 classes (digits) for tractability at small scale.
pub fn variable_binding_examples(d_input: usize) -> Vec<Example> {
    let mut examples = Vec::new();

    // 4 variable names × 10 digit values = 40 unique bindings
    let names: &[u8] = b"xyzw";

    for _ in 0..50 {
        for (ni, &_name) in names.iter().enumerate() {
            for val in 0..10u8 {
                let mut input = vec![0.0f32; d_input];
                // Encode var name as one-hot in first 4 dims
                if d_input > ni { input[ni] = 1.0; }
                // Encode value clearly
                if d_input > 4 { input[4] = val as f32 / 5.0 - 1.0; }
                // Binary encoding of value
                for bit in 0..4 {
                    if d_input > 5 + bit { input[5 + bit] = ((val >> bit) & 1) as f32 * 2.0 - 1.0; }
                }
                // Query name = same (one-hot)
                if d_input > 9 + ni { input[9 + ni] = 1.0; }
                // Match indicator
                if d_input > 13 { input[13] = 1.0; }

                examples.push(Example { input, target: val as usize, n_classes: 10 });
            }
        }
    }
    examples
}
