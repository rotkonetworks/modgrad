//! CTC loss wrapped in the modgrad `LossFn` trait + a tiny smoke-train
//! driver that plugs into the existing `train_step_composed` pipeline.
//!
//! Architecture (phase 0):
//!   image: LINE_H × W grayscale
//!     ↓ each column = one token of dim LINE_H (no retina yet)
//!   tokens: W × LINE_H
//!     ↓ kv_proj → d_input
//!   CTM (T ticks)
//!     ↓ per-tick output_proj → out_dims = ALPHABET_SIZE
//!   predictions: T × ALPHABET_SIZE
//!     ↓ CTC vs target class sequence
//!   loss + d_predictions: T × ALPHABET_SIZE
//!
//! "Retina" is omitted in phase 0 by design (per ocr-demo.md §"Open
//! questions" 2.): proves the CTC/CTM loop on raw columns first. A
//! pretrained retina frontend can be slotted in by replacing the
//! "image → tokens" step.

use modgrad_traits::LossFn;
use modgrad_training::ctc::{ctc_loss_grad, ctc_greedy_decode};

use crate::render::ALPHABET_SIZE;

/// CTC loss as a `LossFn` over per-tick predictions.
///
/// `target` is a CTC label sequence — class indices in `[1, ALPHABET_SIZE)`.
/// Class 0 is blank. The wrapped loss is identical to the underlying
/// `ctc_loss_grad`; this adapter only handles the `Vec<Vec<f32>>` ↔
/// flat conversion the modgrad pipeline needs.
///
/// `smoothing` (0.0 default): label-smoothing strength. Mixes the CTC
/// gradient with `eps * (softmax(y) - uniform)`, which pulls the
/// predicted distribution gently toward uniform. Helps escape the
/// classic CTC "predict mostly blank" attractor by penalising
/// over-confident wrong predictions. Values around 0.05–0.1 are
/// typical; pure CTC = 0.0.
pub struct CtcLossFn {
    pub alphabet_size: usize,
    pub smoothing: f32,
}

impl CtcLossFn {
    pub fn new() -> Self { Self { alphabet_size: ALPHABET_SIZE, smoothing: 0.0 } }
    pub fn with_smoothing(eps: f32) -> Self {
        Self { alphabet_size: ALPHABET_SIZE, smoothing: eps }
    }
}

impl Default for CtcLossFn {
    fn default() -> Self { Self::new() }
}

impl LossFn for CtcLossFn {
    // Sized target so train_step_composed (which has an implicit
    // `Sized` bound on T) accepts &Vec<usize>.
    type Target = Vec<usize>;

    fn compute(
        &self,
        predictions: &[Vec<f32>],
        _certainties: &[[f32; 2]],
        target: &Self::Target,
    ) -> (f32, Vec<Vec<f32>>) {
        let t = predictions.len();
        if t == 0 {
            return (f32::INFINITY, Vec::new());
        }
        let a = self.alphabet_size;
        // Flatten into [T × A].
        let mut flat = Vec::with_capacity(t * a);
        for row in predictions {
            debug_assert_eq!(row.len(), a, "prediction width != alphabet_size");
            flat.extend_from_slice(row);
        }
        let (nll, mut grad_flat) = ctc_loss_grad(&flat, a, target);

        // Label smoothing: add eps * (softmax(logits) - uniform) at
        // every (t, k). This is the gradient of the entropy-pull term
        //   eps * KL(softmax(logits) || uniform)
        // w.r.t. the logits — analytically clean and cheap.
        if self.smoothing > 0.0 {
            let eps = self.smoothing;
            let inv_a = 1.0 / a as f32;
            for ti in 0..t {
                // Row-wise softmax for numerical stability.
                let off = ti * a;
                let row = &flat[off..off + a];
                let max = row.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
                let mut sum = 0.0f32;
                let mut sm = vec![0.0f32; a];
                for k in 0..a {
                    let e = (row[k] - max).exp();
                    sm[k] = e;
                    sum += e;
                }
                let inv = 1.0 / sum;
                for k in 0..a {
                    let pk = sm[k] * inv;
                    grad_flat[off + k] += eps * (pk - inv_a);
                }
            }
        }

        // Reshape grad back into Vec<Vec<f32>>.
        let mut d_preds = Vec::with_capacity(t);
        for ti in 0..t {
            d_preds.push(grad_flat[ti * a..(ti + 1) * a].to_vec());
        }
        (nll, d_preds)
    }
}

/// Greedy CTC decode of per-tick predictions back to a class sequence.
///
/// Thin wrapper that flattens predictions and forwards to
/// `modgrad_training::ctc::ctc_greedy_decode`. Drops blanks and
/// repeats; the result is suitable for `classes_to_string`.
pub fn greedy_decode_predictions(predictions: &[Vec<f32>]) -> Vec<usize> {
    let a = ALPHABET_SIZE;
    if predictions.is_empty() { return Vec::new(); }
    let mut flat = Vec::with_capacity(predictions.len() * a);
    for row in predictions { flat.extend_from_slice(row); }
    let (decoded, _conf) = ctc_greedy_decode(&flat, a);
    decoded
}

/// Pack one rendered line into a [W × LINE_H] token stream.
///
/// Each column becomes one token of dim LINE_H. Returns (tokens_flat,
/// n_tokens=W, token_dim=LINE_H).
///
/// Column = token because OCR text reads left-to-right and adjacent
/// columns carry adjacent stroke evidence. Phase 0 design (§"Open
/// questions" 4.(a)) picked the simplest column→token route; a retina
/// can replace this step later.
pub fn line_to_tokens(line: &crate::render::RenderedLine) -> (Vec<f32>, usize, usize) {
    let h = line.h;
    let w = line.w;
    let mut tokens = vec![0.0f32; w * h];
    for x in 0..w {
        for y in 0..h {
            tokens[x * h + y] = line.pixels[y * w + x];
        }
    }
    (tokens, w, h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ctc_lossfn_returns_finite_grad_on_matching_target() {
        // 4 ticks × 3 classes (blank + 2). Target [1, 2].
        let a = 3;
        let predictions = vec![
            vec![0.1, 0.8, 0.1],  // class 1
            vec![0.1, 0.8, 0.1],
            vec![0.1, 0.1, 0.8],  // class 2
            vec![0.1, 0.1, 0.8],
        ];
        let target: Vec<usize> = vec![1, 2];
        let loss_fn = CtcLossFn { alphabet_size: a, smoothing: 0.0 };
        let (loss, d_preds) = loss_fn.compute(&predictions, &[], &target);
        assert!(loss.is_finite() && loss > 0.0);
        assert_eq!(d_preds.len(), 4);
        for row in &d_preds {
            assert_eq!(row.len(), a);
            for &v in row {
                assert!(v.is_finite());
            }
        }
    }
}
