//! Smoke test: does the retina produce visually useful V4 features?
//!
//! Loads a 10-class ImageNet subset (preprocessed at 32×32 by
//! `examples/retina_viz/imagenet10_to_cifarfeat.py` to .feat format
//! the existing `cifar` loader reads). Forwards every image through
//! `VisualCortex`, mean-pools V4 spatially → 128-D feature vector
//! per image. Then trains a linear classifier (softmax + cross-
//! entropy + plain SGD) on the train split and reports top-1 accuracy
//! on the eval split.
//!
//! Compares two cortex configurations, controlled by argv[1]:
//!   `cifar`   — `VisualCortex::cifar()`: standard ganglion DoG +
//!               V1 Gabor priors + random V2/V4 (the production
//!               default for 32×32 inputs)
//!   `random`  — `VisualCortex::random()`: random init throughout
//!               (no priors anywhere).
//!
//! The relevant question this answers: do the visual priors (DoG +
//! Gabor) plus untrained-but-structured V2/V4 produce features that
//! a plain linear classifier can read, or are they no better than
//! random projections? Audit recommendation 6.5.

use modgrad_codec::cifar::load_feat;
use modgrad_codec::retina::VisualCortex;

const TRAIN_PATH: &str = "/tmp/retina_imagenet10_train.feat";
const EVAL_PATH:  &str = "/tmp/retina_imagenet10_eval.feat";
const N_CLASSES: usize = 10;
const FEAT_DIM:  usize = 128;
const EPOCHS:    usize = 30;
const LR:        f32 = 0.05;

fn extract_features(cortex: &VisualCortex, images: &[modgrad_codec::cifar::CifarImage])
    -> (Vec<f32>, Vec<usize>)
{
    let n = images.len();
    let mut feats = vec![0.0f32; n * FEAT_DIM];
    let mut labels = Vec::with_capacity(n);
    for (i, img) in images.iter().enumerate() {
        let scales = cortex.spatial_tokens_multiscale(&img.pixels);
        // scales[2] is V4 (per-token feature, n_tokens × channels)
        let (v4_tok, n_tok, channels) = (&scales[2].0, scales[2].1, scales[2].2);
        debug_assert_eq!(channels, FEAT_DIM);
        // Mean-pool spatially → 128-D vector
        let f = &mut feats[i * FEAT_DIM..(i + 1) * FEAT_DIM];
        for t in 0..n_tok {
            for c in 0..channels {
                f[c] += v4_tok[t * channels + c];
            }
        }
        let inv = 1.0 / n_tok.max(1) as f32;
        for c in 0..channels { f[c] *= inv; }
        labels.push(img.label);
    }
    (feats, labels)
}

/// Plain softmax classifier with cross-entropy loss + SGD.
fn train_linear_probe(
    train_feat: &[f32], train_lab: &[usize],
    eval_feat: &[f32],  eval_lab: &[usize],
) -> (f32, f32) {
    let n_train = train_lab.len();
    let n_eval = eval_lab.len();
    // Weight matrix [N_CLASSES × FEAT_DIM] + bias [N_CLASSES].
    let mut w = vec![0.0f32; N_CLASSES * FEAT_DIM];
    let mut b = vec![0.0f32; N_CLASSES];

    let forward = |w: &[f32], b: &[f32], f: &[f32]| -> Vec<f32> {
        // logits[k] = sum_j w[k, j] * f[j] + b[k]
        let mut logits = vec![0.0f32; N_CLASSES];
        for k in 0..N_CLASSES {
            let mut s = b[k];
            for j in 0..FEAT_DIM {
                s += w[k * FEAT_DIM + j] * f[j];
            }
            logits[k] = s;
        }
        // log-softmax for stability
        let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for k in 0..N_CLASSES { sum += (logits[k] - mx).exp(); }
        let lse = mx + sum.ln();
        for k in 0..N_CLASSES { logits[k] -= lse; }   // log-prob
        logits
    };

    let acc_at = |w: &[f32], b: &[f32], feats: &[f32], labs: &[usize]| -> f32 {
        let mut correct = 0;
        for i in 0..labs.len() {
            let f = &feats[i * FEAT_DIM..(i + 1) * FEAT_DIM];
            let lp = forward(w, b, f);
            let pred = (0..N_CLASSES).max_by(|&a, &b|
                lp[a].partial_cmp(&lp[b]).unwrap_or(std::cmp::Ordering::Equal)
            ).unwrap();
            if pred == labs[i] { correct += 1; }
        }
        correct as f32 / labs.len() as f32
    };

    let mut shuffled: Vec<usize> = (0..n_train).collect();
    let mut rng_state: u64 = 0xC0FFEE;
    for ep in 0..EPOCHS {
        // Shuffle indices
        for i in (1..n_train).rev() {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let r = ((rng_state >> 33) as usize) % (i + 1);
            shuffled.swap(i, r);
        }
        let mut total_loss = 0.0f32;
        for &i in &shuffled {
            let f = &train_feat[i * FEAT_DIM..(i + 1) * FEAT_DIM];
            let y = train_lab[i];
            let lp = forward(&w, &b, f);
            // cross-entropy loss: -log p[y]
            total_loss += -lp[y];
            // gradient: dL/dlogit[k] = softmax[k] - one_hot[y][k]
            for k in 0..N_CLASSES {
                let p = lp[k].exp();
                let g = p - if k == y { 1.0 } else { 0.0 };
                b[k] -= LR * g;
                for j in 0..FEAT_DIM {
                    w[k * FEAT_DIM + j] -= LR * g * f[j];
                }
            }
        }
        if ep == 0 || (ep + 1) % 5 == 0 || ep == EPOCHS - 1 {
            let train_acc = acc_at(&w, &b, train_feat, train_lab);
            let eval_acc  = acc_at(&w, &b, eval_feat,  eval_lab);
            eprintln!(
                "  epoch {:>2}/{}  loss={:.3}  train_acc={:.1}%  eval_acc={:.1}%",
                ep + 1, EPOCHS, total_loss / n_train as f32,
                train_acc * 100.0, eval_acc * 100.0,
            );
        }
    }
    let train_acc = acc_at(&w, &b, train_feat, train_lab);
    let eval_acc = acc_at(&w, &b, eval_feat, eval_lab);
    (train_acc, eval_acc)
}

/// k-NN classification in V4 space. For each query, find k nearest
/// references by Euclidean distance, return majority-vote label.
/// Used for the held-out-class transfer test: if priors-on V4 clusters
/// images of UNSEEN classes more semantically than random V4 does, the
/// priors generalise beyond the classes any classifier was trained on.
fn knn_accuracy(
    refs: &[f32], ref_labs: &[usize],
    queries: &[f32], query_labs: &[usize],
    k: usize,
) -> f32 {
    let n_ref = ref_labs.len();
    let n_q = query_labs.len();
    let mut correct = 0usize;
    for q in 0..n_q {
        let qf = &queries[q * FEAT_DIM..(q + 1) * FEAT_DIM];
        let mut dists: Vec<(f32, usize)> = (0..n_ref).map(|r| {
            let rf = &refs[r * FEAT_DIM..(r + 1) * FEAT_DIM];
            let d2: f32 = qf.iter().zip(rf).map(|(a, b)| (a - b).powi(2)).sum();
            (d2, ref_labs[r])
        }).collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut votes = [0usize; N_CLASSES];
        for &(_, lab) in dists.iter().take(k) { votes[lab] += 1; }
        let pred = (0..N_CLASSES).max_by_key(|&c| votes[c]).unwrap();
        if pred == query_labs[q] { correct += 1; }
    }
    correct as f32 / n_q as f32
}

fn run_one(mode: &str) -> (f32, f32, f32) {
    eprintln!("\n────────── mode = {mode} ──────────");
    let cortex = match mode {
        "cifar"  => { eprintln!("  cortex: VisualCortex::cifar() (standard priors)"); VisualCortex::cifar() }
        "random" => { eprintln!("  cortex: VisualCortex::random(32, 32) (no priors)"); VisualCortex::random(32, 32) }
        other    => panic!("unknown mode '{other}', want cifar|random"),
    };

    let train = load_feat(TRAIN_PATH).expect("load train .feat");
    let eval  = load_feat(EVAL_PATH).expect("load eval .feat");

    let (train_feat, train_lab) = extract_features(&cortex, &train);
    let (eval_feat,  eval_lab)  = extract_features(&cortex, &eval);

    // ── Test 1: linear probe over all 10 classes (in-distribution) ──
    eprintln!("test 1 — linear probe, all 10 classes:");
    let (_, eval_acc_all) = train_linear_probe(
        &train_feat, &train_lab, &eval_feat, &eval_lab,
    );

    // ── Test 2: k-NN classification on UNSEEN classes (held-out, generalization) ──
    // Split classes into "seen" (0..5) and "unseen" (5..10). The probe
    // never saw the unseen-class images at all; we ask whether V4
    // features cluster the unseen classes by semantic content. If yes,
    // the visual representation generalises beyond the classes any
    // classifier was trained on.
    let unseen_lo = 5;
    let unseen_hi = 10;
    let pick = |feats: &[f32], labs: &[usize]| -> (Vec<f32>, Vec<usize>) {
        let mut f_sub = Vec::new();
        let mut l_sub = Vec::new();
        for (i, &l) in labs.iter().enumerate() {
            if l >= unseen_lo && l < unseen_hi {
                f_sub.extend_from_slice(&feats[i * FEAT_DIM..(i + 1) * FEAT_DIM]);
                l_sub.push(l);
            }
        }
        (f_sub, l_sub)
    };
    let (un_train_f, un_train_l) = pick(&train_feat, &train_lab);
    let (un_eval_f,  un_eval_l)  = pick(&eval_feat,  &eval_lab);
    eprintln!(
        "test 2 — k-NN(k=5) within unseen classes [{}..{}): {} train refs, {} eval queries",
        unseen_lo, unseen_hi, un_train_l.len(), un_eval_l.len(),
    );
    let knn_unseen = knn_accuracy(&un_train_f, &un_train_l, &un_eval_f, &un_eval_l, 5);
    let unseen_chance = 1.0 / (unseen_hi - unseen_lo) as f32;

    // ── Test 3: k-NN within SEEN classes for comparison (in-distribution baseline for k-NN) ──
    let pick_seen = |feats: &[f32], labs: &[usize]| -> (Vec<f32>, Vec<usize>) {
        let mut f_sub = Vec::new();
        let mut l_sub = Vec::new();
        for (i, &l) in labs.iter().enumerate() {
            if l < unseen_lo {
                f_sub.extend_from_slice(&feats[i * FEAT_DIM..(i + 1) * FEAT_DIM]);
                l_sub.push(l);
            }
        }
        (f_sub, l_sub)
    };
    let (s_train_f, s_train_l) = pick_seen(&train_feat, &train_lab);
    let (s_eval_f,  s_eval_l)  = pick_seen(&eval_feat,  &eval_lab);
    let knn_seen = knn_accuracy(&s_train_f, &s_train_l, &s_eval_f, &s_eval_l, 5);

    eprintln!(
        "  results: linear-probe 10cls eval={:.1}% | k-NN seen[0..5)={:.1}% | k-NN unseen[5..10)={:.1}% (chance={:.1}%)",
        eval_acc_all * 100.0, knn_seen * 100.0, knn_unseen * 100.0, unseen_chance * 100.0,
    );

    (eval_acc_all, knn_seen, knn_unseen)
}

fn main() {
    let mode_arg = std::env::args().nth(1).unwrap_or_else(|| "compare".to_string());

    if mode_arg == "compare" {
        let (l_c, ks_c, ku_c) = run_one("cifar");
        let (l_r, ks_r, ku_r) = run_one("random");

        eprintln!("\n══════════════════════════════════════════════════════════");
        eprintln!("                           cifar    random   Δ");
        eprintln!("──────────────────────────────────────────────────────────");
        eprintln!("  linear probe, 10cls       {:5.1}%   {:5.1}%   {:+5.1} pp",
                  l_c * 100.0, l_r * 100.0, (l_c - l_r) * 100.0);
        eprintln!("  k-NN, seen [0..5)         {:5.1}%   {:5.1}%   {:+5.1} pp",
                  ks_c * 100.0, ks_r * 100.0, (ks_c - ks_r) * 100.0);
        eprintln!("  k-NN, UNSEEN [5..10)      {:5.1}%   {:5.1}%   {:+5.1} pp   ← generalization",
                  ku_c * 100.0, ku_r * 100.0, (ku_c - ku_r) * 100.0);
        eprintln!("══════════════════════════════════════════════════════════");
    } else {
        run_one(&mode_arg);
    }
}
