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

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "cifar".to_string());
    eprintln!("cifar10_probe: mode={mode}");

    let cortex = match mode.as_str() {
        "cifar"  => { eprintln!("  cortex: VisualCortex::cifar() (standard priors)"); VisualCortex::cifar() }
        "random" => { eprintln!("  cortex: VisualCortex::random(32, 32) (no priors)"); VisualCortex::random(32, 32) }
        other    => panic!("unknown mode '{other}', want cifar|random"),
    };

    let train = load_feat(TRAIN_PATH).expect("load train .feat");
    let eval  = load_feat(EVAL_PATH).expect("load eval .feat");
    eprintln!("  loaded {} train + {} eval", train.len(), eval.len());

    let (train_feat, train_lab) = extract_features(&cortex, &train);
    let (eval_feat,  eval_lab)  = extract_features(&cortex, &eval);
    eprintln!("  extracted V4 features: {} train, {} eval, dim={FEAT_DIM}",
              train_lab.len(), eval_lab.len());

    eprintln!("\n=== linear probe training ===");
    let (train_acc, eval_acc) = train_linear_probe(
        &train_feat, &train_lab, &eval_feat, &eval_lab,
    );
    eprintln!(
        "\n=== {} ===  final train_acc={:.1}%  eval_acc={:.1}%  (chance={:.1}%)",
        mode, train_acc * 100.0, eval_acc * 100.0, 100.0 / N_CLASSES as f32,
    );
}
