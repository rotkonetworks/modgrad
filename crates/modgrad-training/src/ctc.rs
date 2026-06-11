//! Connectionist Temporal Classification — loss + gradient + greedy decode.
//!
//! CTC trains a sequence model against an unaligned target label sequence.
//! The model emits one logit vector per timestep over an alphabet that
//! includes a special "blank" class; CTC sums the probability of every
//! alignment of blanks-and-labels that collapses to the target sequence.
//!
//! Convention: blank is class index 0. Matches PyTorch `nn.CTCLoss`,
//! TensorFlow `tf.nn.ctc_loss`, and PaddleOCR's `CTCLabelDecode`.
//!
//! Reference: Graves et al. 2006, *Connectionist Temporal Classification:
//! Labelling Unsegmented Sequence Data with Recurrent Neural Networks*.
//! Greedy decode mirrors `paddlex/.../text_recognition/processors.py`
//! `BaseRecLabelDecode.decode`.
//!
//! Pure Rust, no GPU dispatch — wasm32 builds drop straight in.

const NEG_INF: f32 = f32::NEG_INFINITY;

/// Conventional blank index. Don't change this — every reference
/// implementation (PyTorch, TF, PaddleOCR) puts blank at 0; aligning
/// with that lets us interoperate with their checkpoints later.
pub const BLANK: usize = 0;

// ─── Stable log-sum-exp helpers ───────────────────────────────

#[inline]
fn logsumexp2(a: f32, b: f32) -> f32 {
    if a == NEG_INF { return b; }
    if b == NEG_INF { return a; }
    let (m, x) = if a > b { (a, b) } else { (b, a) };
    m + (-((m - x).abs())).exp().ln_1p()
}

/// In-place log-softmax of one row.
fn log_softmax_row(logits: &[f32], out: &mut [f32]) {
    let max_l = logits.iter().fold(NEG_INF, |a, &b| a.max(b));
    let mut sum_exp = 0.0f32;
    for &l in logits { sum_exp += (l - max_l).exp(); }
    let lse = max_l + sum_exp.ln();
    for (o, &l) in out.iter_mut().zip(logits.iter()) {
        *o = l - lse;
    }
}

// ─── Forward (loss only) ──────────────────────────────────────

/// CTC negative log-likelihood.
///
/// `logits` is row-major `[T × alphabet_size]` of raw pre-softmax scores.
/// `target` is a sequence of class indices in `[1, alphabet_size)` — must
/// not contain blank (index 0).
///
/// Returns `f32::INFINITY` for impossible alignments (T smaller than the
/// minimum required to emit `target`).
pub fn ctc_loss(
    logits: &[f32],
    alphabet_size: usize,
    target: &[usize],
) -> f32 {
    let (nll, _) = ctc_loss_grad_inner(logits, alphabet_size, target, false);
    nll
}

/// CTC NLL + gradient w.r.t. `logits` (same shape).
///
/// Gradient is zero-filled for impossible alignments (loss = +∞). Caller
/// is expected to check `.is_finite()` and skip the batch.
pub fn ctc_loss_grad(
    logits: &[f32],
    alphabet_size: usize,
    target: &[usize],
) -> (f32, Vec<f32>) {
    ctc_loss_grad_inner(logits, alphabet_size, target, true)
}

fn ctc_loss_grad_inner(
    logits: &[f32],
    a: usize,
    target: &[usize],
    want_grad: bool,
) -> (f32, Vec<f32>) {
    assert!(a >= 2, "alphabet must include at least blank + 1 class");
    let t = logits.len() / a;
    assert_eq!(logits.len(), t * a, "logits length not divisible by alphabet_size");

    let grad_init = || if want_grad { vec![0.0f32; t * a] } else { Vec::new() };
    if t == 0 { return (f32::INFINITY, grad_init()); }

    // Log-softmax once. `y` = softmax probs, needed for gradient.
    let mut log_y = vec![0.0f32; t * a];
    for ti in 0..t {
        log_softmax_row(&logits[ti * a..(ti + 1) * a], &mut log_y[ti * a..(ti + 1) * a]);
    }

    // Empty target: only blank-blank-...-blank alignment is valid.
    if target.is_empty() {
        let mut nll = 0.0f32;
        for ti in 0..t { nll -= log_y[ti * a + BLANK]; }
        if !want_grad { return (nll, Vec::new()); }
        let mut grad = vec![0.0f32; t * a];
        for ti in 0..t {
            for k in 0..a { grad[ti * a + k] = log_y[ti * a + k].exp(); }
            grad[ti * a + BLANK] -= 1.0;
        }
        return (nll, grad);
    }

    let l = target.len();
    let s_len = 2 * l + 1;
    // Extended label: [blank, L[0], blank, L[1], …, L[n-1], blank].
    let lp: Vec<usize> = (0..s_len).map(|s| if s % 2 == 0 { BLANK } else { target[s / 2] }).collect();

    // Minimum T = |L| + number of adjacent duplicate labels (each pair
    // needs a separating blank). Anything below this is unreachable.
    let mut min_t = l;
    for i in 1..l { if target[i] == target[i - 1] { min_t += 1; } }
    if t < min_t { return (f32::INFINITY, grad_init()); }

    // ── Forward pass (log_alpha) ──
    let mut log_alpha = vec![NEG_INF; t * s_len];
    log_alpha[0] = log_y[BLANK];
    log_alpha[1] = log_y[target[0]];
    for ti in 1..t {
        for s in 0..s_len {
            let mut acc = log_alpha[(ti - 1) * s_len + s];
            if s >= 1 {
                acc = logsumexp2(acc, log_alpha[(ti - 1) * s_len + s - 1]);
            }
            if s >= 2 && lp[s] != BLANK && lp[s] != lp[s - 2] {
                acc = logsumexp2(acc, log_alpha[(ti - 1) * s_len + s - 2]);
            }
            log_alpha[ti * s_len + s] = acc + log_y[ti * a + lp[s]];
        }
    }

    // Total log P = logsumexp(last position, second-to-last position).
    let log_p = logsumexp2(
        log_alpha[(t - 1) * s_len + s_len - 1],
        log_alpha[(t - 1) * s_len + s_len - 2],
    );
    let nll = -log_p;

    if !want_grad || !log_p.is_finite() {
        return (nll, grad_init());
    }

    // ── Backward pass (log_beta) ──
    let mut log_beta = vec![NEG_INF; t * s_len];
    log_beta[(t - 1) * s_len + s_len - 1] = log_y[(t - 1) * a + BLANK];
    log_beta[(t - 1) * s_len + s_len - 2] = log_y[(t - 1) * a + lp[s_len - 2]];
    for ti in (0..t - 1).rev() {
        for s in 0..s_len {
            let mut acc = log_beta[(ti + 1) * s_len + s];
            if s + 1 < s_len {
                acc = logsumexp2(acc, log_beta[(ti + 1) * s_len + s + 1]);
            }
            if s + 2 < s_len && lp[s] != BLANK && lp[s] != lp[s + 2] {
                acc = logsumexp2(acc, log_beta[(ti + 1) * s_len + s + 2]);
            }
            log_beta[ti * s_len + s] = acc + log_y[ti * a + lp[s]];
        }
    }

    // ── Gradient: dL/d_logits[t][k] = y[t][k] - p_k[t][k] where
    //    p_k[t][k] = (1/Z) * Σ_{s: lp[s]=k} α[t][s] β[t][s] / y[t][k]
    // In log space: log_p_k[t][k] = logsumexp_{s: lp[s]=k}
    //   (log_alpha[t][s] + log_beta[t][s] - log_y[t][k]) - log_p
    let mut log_pk = vec![NEG_INF; t * a];
    for ti in 0..t {
        for s in 0..s_len {
            let k = lp[s];
            let gamma_s =
                log_alpha[ti * s_len + s] + log_beta[ti * s_len + s] - log_y[ti * a + k] - log_p;
            let slot = &mut log_pk[ti * a + k];
            *slot = logsumexp2(*slot, gamma_s);
        }
    }

    let mut grad = vec![0.0f32; t * a];
    for ti in 0..t {
        for k in 0..a {
            let y = log_y[ti * a + k].exp();
            let p_k = log_pk[ti * a + k].exp();
            grad[ti * a + k] = y - p_k;
        }
    }
    (nll, grad)
}

// ─── Greedy decode ────────────────────────────────────────────

/// Greedy CTC decode: argmax per timestep, collapse consecutive
/// duplicates, drop blanks. Returns `(decoded_indices, mean_confidence)`.
///
/// Confidence is the softmax probability of the argmax class at each
/// kept timestep, averaged. Empty decode → confidence 0.
///
/// Mirrors `BaseRecLabelDecode.decode` in PaddleOCR.
pub fn ctc_greedy_decode(logits: &[f32], alphabet_size: usize) -> (Vec<usize>, f32) {
    let t = logits.len() / alphabet_size;
    let mut indices = Vec::new();
    let mut conf_sum = 0.0f32;
    let mut conf_n = 0usize;
    let mut prev: Option<usize> = None;
    for ti in 0..t {
        let row = &logits[ti * alphabet_size..(ti + 1) * alphabet_size];
        let (best_k, &best_v) = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        if best_k != BLANK && prev != Some(best_k) {
            // Softmax probability of best_k for confidence.
            let max_l = row.iter().fold(NEG_INF, |a, &b| a.max(b));
            let lse = max_l + row.iter().map(|&l| (l - max_l).exp()).sum::<f32>().ln();
            conf_sum += (best_v - lse).exp();
            conf_n += 1;
            indices.push(best_k);
        }
        prev = Some(best_k);
    }
    let mean_conf = if conf_n == 0 { 0.0 } else { conf_sum / conf_n as f32 };
    (indices, mean_conf)
}

// ─── Adapter for the per-tick CTM loss API ───────────────────

/// CTC loss in the `(predictions, certainties) → (loss, d_predictions)`
/// shape used by `modgrad_ctm::graph::regional_train_step_generic`.
///
/// `regional_forward` already returns per-tick predictions as
/// `Vec<Vec<f32>>` (one row per outer tick, `out_dims` wide). Wire
/// up CTC by setting the CTM's `out_dims` to your `alphabet_size`
/// (e.g. 96 = printable ascii + blank) and passing this closure to
/// the generic train step.
///
/// Pure adapter; the math is in `ctc_loss_grad`. Lives here rather
/// than in `modgrad-traits` because it needs the CTC math, which
/// pulls in nothing else.
pub struct CtcLoss<'a> {
    pub alphabet_size: usize,
    pub target: &'a [usize],
}

impl<'a> CtcLoss<'a> {
    pub fn new(alphabet_size: usize, target: &'a [usize]) -> Self {
        Self { alphabet_size, target }
    }

    /// Compute loss + per-tick gradients. `predictions` is `[T][A]` —
    /// rows are softmax logits at each outer tick.
    pub fn compute(&self, predictions: &[Vec<f32>]) -> (f32, Vec<Vec<f32>>) {
        let t = predictions.len();
        let a = self.alphabet_size;
        if t == 0 {
            return (f32::INFINITY, Vec::new());
        }
        let mut flat = Vec::with_capacity(t * a);
        for row in predictions {
            assert_eq!(row.len(), a, "prediction width {} != alphabet_size {}", row.len(), a);
            flat.extend_from_slice(row);
        }
        let (loss, grad_flat) = ctc_loss_grad(&flat, a, self.target);
        let grads: Vec<Vec<f32>> = (0..t)
            .map(|ti| grad_flat[ti * a..(ti + 1) * a].to_vec())
            .collect();
        (loss, grads)
    }
}

// ─── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_logits(t: usize, a: usize) -> Vec<f32> { vec![0.0f32; t * a] }

    #[test]
    fn empty_target_uniform_is_t_times_log_a() {
        // T=4, alphabet=3, uniform logits → P(blank)^T per timestep.
        // NLL = -T * log(1/A) = T * ln(A).
        let logits = uniform_logits(4, 3);
        let nll = ctc_loss(&logits, 3, &[]);
        let expected = 4.0 * (3.0f32).ln();
        assert!((nll - expected).abs() < 1e-4, "got {nll} expected {expected}");
    }

    #[test]
    fn t_eq_2_target_a_uniform() {
        // Hand-derived: T=2, alphabet={blank, a}, uniform.
        // Three alignments collapse to "a": (b,a), (a,b), (a,a).
        // Each path probability = 0.25 → P(L) = 0.75 → NLL = -log(0.75).
        let logits = uniform_logits(2, 2);
        let nll = ctc_loss(&logits, 2, &[1]);
        let expected = -(0.75f32).ln();
        assert!((nll - expected).abs() < 1e-5, "got {nll} expected {expected}");
    }

    #[test]
    fn impossible_alignment_returns_inf() {
        // L=[a,a] needs at least 3 frames (a, blank, a). T=2 is impossible.
        let logits = uniform_logits(2, 2);
        let nll = ctc_loss(&logits, 2, &[1, 1]);
        assert!(nll.is_infinite() && nll > 0.0);
    }

    #[test]
    fn confident_correct_alignment_collapses_to_zero() {
        // T=3, alphabet={blank, a, b}. Force alignment (a, blank, b)
        // with near-1 probability. NLL should be ~0.
        let mut logits = vec![0.0f32; 3 * 3];
        for (ti, k) in [(0, 1usize), (1, 0), (2, 2)].iter().copied() {
            logits[ti * 3 + k] = 20.0;
        }
        let nll = ctc_loss(&logits, 3, &[1, 2]);
        assert!(nll < 1e-3, "got {nll}");
    }

    #[test]
    fn gradient_matches_finite_differences() {
        // Gold-standard CTC test: numerical gradient by finite differences,
        // compare to analytical gradient. Small, deterministic logits.
        let t = 4;
        let a = 4;
        let target = vec![1usize, 2, 1];
        // Fixed pseudo-random logits — same seed every run.
        let mut rng = 0xC0FFEEu64;
        let mut logits = vec![0.0f32; t * a];
        for x in logits.iter_mut() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            *x = ((rng >> 32) as i32 as f32) / (i32::MAX as f32);
        }

        let (nll0, grad) = ctc_loss_grad(&logits, a, &target);
        assert!(nll0.is_finite(), "nll0 = {nll0}");

        let eps = 1e-3f32;
        let mut max_err = 0.0f32;
        for i in 0..(t * a) {
            let mut lp = logits.clone();
            let mut lm = logits.clone();
            lp[i] += eps;
            lm[i] -= eps;
            let fd = (ctc_loss(&lp, a, &target) - ctc_loss(&lm, a, &target)) / (2.0 * eps);
            let an = grad[i];
            let err = (fd - an).abs() / (fd.abs().max(an.abs()).max(1e-3));
            max_err = max_err.max(err);
        }
        // f32 finite differences accumulate roundoff through the log_alpha
        // / log_beta logsumexp passes; 1.5% relative tolerance is standard
        // for f32 gradcheck (PyTorch uses 1e-2 for f32 by default).
        assert!(max_err < 1.5e-2, "max relative grad error {max_err}");
    }

    #[test]
    fn empty_target_gradient_matches_finite_differences() {
        // Empty-target path has its own gradient code; check separately.
        let t = 3;
        let a = 4;
        let mut rng = 0xDEADBEEFu64;
        let mut logits = vec![0.0f32; t * a];
        for x in logits.iter_mut() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            *x = ((rng >> 32) as i32 as f32) / (i32::MAX as f32);
        }

        let (_, grad) = ctc_loss_grad(&logits, a, &[]);
        let eps = 1e-3f32;
        let mut max_err = 0.0f32;
        for i in 0..(t * a) {
            let mut lp = logits.clone();
            let mut lm = logits.clone();
            lp[i] += eps;
            lm[i] -= eps;
            let fd = (ctc_loss(&lp, a, &[]) - ctc_loss(&lm, a, &[])) / (2.0 * eps);
            let err = (fd - grad[i]).abs() / (fd.abs().max(grad[i].abs()).max(1e-3));
            max_err = max_err.max(err);
        }
        assert!(max_err < 5e-3, "empty-target max grad err {max_err}");
    }

    #[test]
    fn gradient_step_decreases_loss() {
        // Sanity: one step of gradient descent on logits should reduce NLL.
        let t = 5;
        let a = 5;
        let target = vec![1usize, 2, 3];
        let mut logits = vec![0.0f32; t * a];
        let (nll0, grad) = ctc_loss_grad(&logits, a, &target);
        let lr = 0.1f32;
        for i in 0..(t * a) { logits[i] -= lr * grad[i]; }
        let nll1 = ctc_loss(&logits, a, &target);
        assert!(nll1 < nll0, "nll did not decrease: {nll0} -> {nll1}");
    }

    #[test]
    fn greedy_decode_collapses_duplicates_and_drops_blanks() {
        // argmax sequence [1,1,0,1,2,2] → decode "1 1 2" (the blank
        // between the two 1s preserves the boundary; trailing 2 dups drop).
        let a = 3;
        let t = 6;
        let mut logits = vec![0.0f32; t * a];
        for (ti, k) in [(0, 1usize), (1, 1), (2, 0), (3, 1), (4, 2), (5, 2)].iter().copied() {
            logits[ti * a + k] = 10.0;
        }
        let (out, conf) = ctc_greedy_decode(&logits, a);
        assert_eq!(out, vec![1, 1, 2]);
        assert!(conf > 0.99, "confidence {conf}");
    }

    #[test]
    fn ctc_loss_adapter_matches_flat_form() {
        // CtcLoss::compute should produce identical loss + per-tick gradients
        // as ctc_loss_grad on the flattened logits.
        let t = 4;
        let a = 4;
        let target = vec![1usize, 2, 1];
        let preds: Vec<Vec<f32>> = (0..t)
            .map(|ti| (0..a).map(|k| ((ti * a + k) as f32 * 0.3).sin()).collect())
            .collect();
        let flat: Vec<f32> = preds.iter().flatten().copied().collect();

        let (loss_flat, grad_flat) = ctc_loss_grad(&flat, a, &target);
        let (loss_adapt, grad_adapt) = CtcLoss::new(a, &target).compute(&preds);

        assert!((loss_flat - loss_adapt).abs() < 1e-6);
        for ti in 0..t {
            for k in 0..a {
                let g_flat = grad_flat[ti * a + k];
                let g_adapt = grad_adapt[ti][k];
                assert!((g_flat - g_adapt).abs() < 1e-6, "tick {ti} class {k}");
            }
        }
    }

    #[test]
    fn greedy_decode_empty_when_all_blank() {
        let logits = uniform_logits(5, 3);
        let mut force_blank = logits.clone();
        for ti in 0..5 { force_blank[ti * 3 + 0] = 10.0; }
        let (out, _conf) = ctc_greedy_decode(&force_blank, 3);
        assert!(out.is_empty());
    }
}
