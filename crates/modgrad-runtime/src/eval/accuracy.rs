//! Canonical evaluation functions for CTM accuracy measurement.
//!
//! Consolidates `eval_motor`, `eval_ls`, `quick_eval`, and `ls_accuracy`
//! that were duplicated across 6+ test files.

use crate::weights::CtmWeights;
use crate::session::CtmSession;
use modgrad_compute::neuron::Linear;
use crate::forward::forward_split;
use super::tasks::Example;
use super::linalg;
use rayon::prelude::*;

/// Motor accuracy: forward pass then argmax on motor_evidence.
pub fn eval_motor(weights: &CtmWeights, data: &[Example]) -> f32 {
    let proprio = vec![0.0; weights.config.d_input];
    let mut correct = 0usize;
    for ex in data {
        let mut s = CtmSession::new(&weights.config);
        let mut t = weights.init_tick_state();
        let _ = forward_split(weights, &mut s, &mut t, &ex.input, &proprio, false);
        let pred = t.motor_evidence.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
        if pred == ex.target { correct += 1; }
    }
    correct as f32 / data.len() as f32
}

/// LS readout accuracy: forward pass, collect activations, Cholesky LS, accuracy.
/// Sweeps lambda in [1e-4, 1e-2, 0.1, 1.0, 10.0].
/// Evaluate with proper train/test split.
/// Fits LS readout on first 80% of data, evaluates on last 20%.
pub fn eval_ls(weights: &CtmWeights, data: &[Example]) -> f32 {
    let n_classes = data[0].n_classes;
    let proprio = vec![0.0; weights.config.d_input];
    let features: Vec<(Vec<f32>, usize)> = data.par_iter().map(|ex| {
        let mut s = CtmSession::new(&weights.config);
        let mut t = weights.init_tick_state();
        let _ = forward_split(weights, &mut s, &mut t, &ex.input, &proprio, false);
        (t.activations.clone(), ex.target)
    }).collect();

    // Split: train on 80%, test on 20%
    let split = features.len() * 4 / 5;
    let (train_feat, test_feat) = features.split_at(split);
    ls_accuracy_split(train_feat, test_feat, n_classes)
}

/// Fit LS readout on train features, evaluate on test features.
pub fn ls_accuracy_split(train: &[(Vec<f32>, usize)], test: &[(Vec<f32>, usize)], n_classes: usize) -> f32 {
    if train.is_empty() || test.is_empty() { return 0.0; }
    let fd = train[0].0.len();
    let mut xtx = vec![0.0f32; fd * fd];
    let mut xty = vec![0.0f32; fd * n_classes];
    for (f, l) in train {
        for r in 0..fd {
            for c in 0..fd { xtx[r * fd + c] += f[r] * f[c]; }
            xty[r * n_classes + *l] += f[r];
        }
    }
    let mut best = 0.0f32;
    for &lam in &[1e-4, 1e-2, 0.1, 1.0, 10.0] {
        let mut xr = xtx.clone();
        for i in 0..fd { xr[i * fd + i] += lam; }
        if let Some(l) = linalg::cholesky(&xr, fd) {
            let mut rd = Linear::new(fd, n_classes);
            for c in 0..n_classes {
                let rhs: Vec<f32> = (0..fd).map(|r| xty[r * n_classes + c]).collect();
                let z = linalg::forward_solve(&l, &rhs, fd);
                let w = linalg::backward_solve(&l, &z, fd);
                for r in 0..fd { rd.weight[c * rd.in_dim + r] = w[r]; }
            }
            let ok: usize = test.iter()
                .map(|(f, lab)| {
                    let logits = rd.forward(f);
                    let pred = logits.iter().enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .map(|(i, _)| i).unwrap_or(0);
                    if pred == *lab { 1 } else { 0 }
                }).sum();
            best = best.max(ok as f32 / test.len() as f32);
        }
    }
    best
}

/// LS readout accuracy on pre-computed feature vectors.
/// Sweeps lambda in [1e-4, 1e-2, 0.1, 1.0, 10.0].
pub fn ls_accuracy(features: &[(Vec<f32>, usize)], n_classes: usize) -> f32 {
    if features.is_empty() { return 0.0; }
    let fd = features[0].0.len();
    let mut xtx = vec![0.0f32; fd * fd];
    let mut xty = vec![0.0f32; fd * n_classes];
    for (f, l) in features {
        for r in 0..fd {
            for c in 0..fd { xtx[r * fd + c] += f[r] * f[c]; }
            xty[r * n_classes + *l] += f[r];
        }
    }
    let mut best = 0.0f32;
    for &lam in &[1e-4, 1e-2, 0.1, 1.0, 10.0] {
        let mut xr = xtx.clone();
        for i in 0..fd { xr[i * fd + i] += lam; }
        if let Some(l) = linalg::cholesky(&xr, fd) {
            let mut rd = Linear::new(fd, n_classes);
            for c in 0..n_classes {
                let rhs: Vec<f32> = (0..fd).map(|r| xty[r * n_classes + c]).collect();
                let z = linalg::forward_solve(&l, &rhs, fd);
                let w = linalg::backward_solve(&l, &z, fd);
                for r in 0..fd { rd.weight[c * rd.in_dim + r] = w[r]; }
            }
            let ok: usize = features.iter()
                .map(|(f, lab)| {
                    let logits = rd.forward(f);
                    let pred = logits.iter().enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .map(|(i, _)| i).unwrap_or(0);
                    if pred == *lab { 1 } else { 0 }
                }).sum();
            best = best.max(ok as f32 / features.len() as f32);
        }
    }
    best
}

/// Quick eval: argmax on output region bins (no LS training).
pub fn quick_eval(weights: &CtmWeights, batch: &[Example]) -> f32 {
    if batch.is_empty() { return 0.0; }
    let n_classes = batch[0].n_classes;
    let proprio = vec![0.0; weights.config.d_input];
    let mut correct = 0usize;
    for ex in batch {
        let mut s = CtmSession::new(&weights.config);
        let mut t = weights.init_tick_state();
        let _ = forward_split(weights, &mut s, &mut t, &ex.input, &proprio, false);
        let out_start = t.act_offsets[2];
        let out_size = t.act_sizes[2];
        let bin_size = out_size / n_classes.max(1);
        let pred = (0..n_classes).max_by(|&a, &b| {
            let sum_a: f32 = t.activations[out_start + a * bin_size..out_start + (a + 1).min(n_classes) * bin_size]
                .iter().sum();
            let sum_b: f32 = t.activations[out_start + b * bin_size..out_start + (b + 1).min(n_classes) * bin_size]
                .iter().sum();
            sum_a.partial_cmp(&sum_b).unwrap()
        }).unwrap_or(0);
        if pred == ex.target { correct += 1; }
    }
    correct as f32 / batch.len() as f32
}
