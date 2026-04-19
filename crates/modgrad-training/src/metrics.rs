//! Evaluation metrics: accuracy, perplexity, bits-per-byte, loss tracking.
//!
//! Each metric implements the Metric trait: update with predictions,
//! compute the scalar result, reset for next epoch.
//! Composable — pass a Vec<Box<dyn Metric>> to an eval loop.

/// A metric that accumulates over predictions and computes a scalar.
pub trait Metric: Send {
    /// Name of this metric (for logging).
    fn name(&self) -> &str;
    /// Update with one sample's prediction and target.
    fn update(&mut self, prediction: &[f32], target: usize);
    /// Compute the metric value from accumulated data.
    fn compute(&self) -> f32;
    /// Reset for next epoch.
    fn reset(&mut self);
}

// ─── Accuracy ──────────────────────────────────────────────

/// Classification accuracy: fraction of correct argmax predictions.
pub struct Accuracy {
    correct: usize,
    total: usize,
}

impl Accuracy {
    pub fn new() -> Self { Self { correct: 0, total: 0 } }
}

impl Metric for Accuracy {
    fn name(&self) -> &str { "accuracy" }

    fn update(&mut self, prediction: &[f32], target: usize) {
        let pred = prediction.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
        if pred == target { self.correct += 1; }
        self.total += 1;
    }

    fn compute(&self) -> f32 {
        if self.total == 0 { 0.0 } else { self.correct as f32 / self.total as f32 }
    }

    fn reset(&mut self) { self.correct = 0; self.total = 0; }
}

// ─── Top-K Accuracy ────────────────────────────────────────

/// Top-K accuracy: target is in the top K predictions.
pub struct TopKAccuracy {
    k: usize,
    correct: usize,
    total: usize,
}

impl TopKAccuracy {
    pub fn new(k: usize) -> Self { Self { k, correct: 0, total: 0 } }
}

impl Metric for TopKAccuracy {
    fn name(&self) -> &str { "top_k_accuracy" }

    fn update(&mut self, prediction: &[f32], target: usize) {
        let mut indexed: Vec<(usize, f32)> = prediction.iter()
            .enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let in_top_k = indexed.iter().take(self.k).any(|(i, _)| *i == target);
        if in_top_k { self.correct += 1; }
        self.total += 1;
    }

    fn compute(&self) -> f32 {
        if self.total == 0 { 0.0 } else { self.correct as f32 / self.total as f32 }
    }

    fn reset(&mut self) { self.correct = 0; self.total = 0; }
}

// ─── Mean Loss ─────────────────────────────────────────────

/// Tracks average loss (any scalar).
pub struct MeanLoss {
    sum: f64,
    count: usize,
}

impl MeanLoss {
    pub fn new() -> Self { Self { sum: 0.0, count: 0 } }

    /// Update with a raw loss value (doesn't need prediction/target).
    pub fn update_loss(&mut self, loss: f32) {
        self.sum += loss as f64;
        self.count += 1;
    }
}

impl Metric for MeanLoss {
    fn name(&self) -> &str { "mean_loss" }

    fn update(&mut self, prediction: &[f32], target: usize) {
        // Cross-entropy loss inline
        let max_l = prediction.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_sum: f32 = prediction.iter().map(|&l| (l - max_l).exp()).sum();
        let log_prob = (prediction.get(target).copied().unwrap_or(0.0) - max_l) - exp_sum.ln();
        self.update_loss(-log_prob);
    }

    fn compute(&self) -> f32 {
        if self.count == 0 { 0.0 } else { (self.sum / self.count as f64) as f32 }
    }

    fn reset(&mut self) { self.sum = 0.0; self.count = 0; }
}

// ─── Bits Per Byte ─────────────────────────────────────────

/// Bits per byte: cross-entropy loss / ln(2).
/// Tokenizer-independent measure of compression quality.
pub struct BitsPerByte {
    total_ce: f64,
    total_bytes: usize,
}

impl BitsPerByte {
    pub fn new() -> Self { Self { total_ce: 0.0, total_bytes: 0 } }
}

impl Metric for BitsPerByte {
    fn name(&self) -> &str { "bits_per_byte" }

    fn update(&mut self, prediction: &[f32], target: usize) {
        let max_l = prediction.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_sum: f32 = prediction.iter().map(|&l| (l - max_l).exp()).sum();
        let log_prob = (prediction.get(target).copied().unwrap_or(0.0) - max_l) - exp_sum.ln();
        self.total_ce += (-log_prob) as f64;
        self.total_bytes += 1;
    }

    fn compute(&self) -> f32 {
        if self.total_bytes == 0 { 0.0 }
        else { (self.total_ce / (self.total_bytes as f64 * 2.0f64.ln())) as f32 }
    }

    fn reset(&mut self) { self.total_ce = 0.0; self.total_bytes = 0; }
}

// ─── Perplexity ────────────────────────────────────────────

/// Perplexity: exp(mean cross-entropy).
pub struct Perplexity {
    inner: MeanLoss,
}

impl Perplexity {
    pub fn new() -> Self { Self { inner: MeanLoss::new() } }
}

impl Metric for Perplexity {
    fn name(&self) -> &str { "perplexity" }

    fn update(&mut self, prediction: &[f32], target: usize) {
        self.inner.update(prediction, target);
    }

    fn compute(&self) -> f32 {
        self.inner.compute().exp()
    }

    fn reset(&mut self) { self.inner.reset(); }
}

/// Windowed bits-per-byte over a token stream, forward-only.
///
/// Slides over `tokens` in non-overlapping `context_len`-sized windows
/// and accumulates cross-entropy into a `BitsPerByte` metric. `forward`
/// is supplied by the caller — it gets a slice of length `context_len`
/// and must return per-position logits, one row per input position.
///
/// Generic over the model: pass any closure that does a forward pass.
/// Tests use a fake uniform forward; isis passes `ffn_forward`. Keeps
/// the metric logic in one place (no per-architecture copies) and out
/// of a circular dep (training doesn't depend on ffn).
///
/// Returns `Err` on too-short data rather than a silent `0.0` — a
/// zero bpb looks like a perfect score to an autoresearch agent and
/// would cause it to keep a mutation that actually can't be evaluated.
pub fn compute_bpb<F>(
    tokens: &[usize],
    context_len: usize,
    mut forward: F,
) -> Result<(f32, usize), String>
where
    F: FnMut(&[usize]) -> Vec<Vec<f32>>,
{
    if tokens.len() < context_len + 1 {
        return Err(format!(
            "val data too short: {} tokens, need at least context_len + 1 = {}",
            tokens.len(),
            context_len + 1,
        ));
    }
    let mut bpb = BitsPerByte::new();
    let mut offset = 0usize;
    let mut n_positions = 0usize;
    while offset + context_len + 1 <= tokens.len() {
        let chunk = &tokens[offset..offset + context_len + 1];
        let logits = forward(&chunk[..context_len]);
        for (pos, row) in logits.iter().enumerate() {
            bpb.update(row, chunk[pos + 1]);
            n_positions += 1;
        }
        offset += context_len;
    }
    if n_positions == 0 {
        return Err(format!(
            "no target positions evaluated ({} tokens, ctx {context_len})",
            tokens.len(),
        ));
    }
    Ok((bpb.compute(), n_positions))
}

/// Run evaluation: given a Brain, data, and metrics, compute all metrics.
/// Returns a vec of (metric_name, value) pairs.
pub fn evaluate(
    metrics: &mut [Box<dyn Metric>],
    predictions_and_targets: &[(Vec<f32>, usize)],
) -> Vec<(String, f32)> {
    for m in metrics.iter_mut() { m.reset(); }
    for (pred, target) in predictions_and_targets {
        for m in metrics.iter_mut() {
            m.update(pred, *target);
        }
    }
    metrics.iter().map(|m| (m.name().to_string(), m.compute())).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accuracy_works() {
        let mut acc = Accuracy::new();
        acc.update(&[0.1, 0.9], 1); // correct
        acc.update(&[0.9, 0.1], 1); // wrong
        acc.update(&[0.1, 0.9], 1); // correct
        assert!((acc.compute() - 2.0/3.0).abs() < 1e-5);
    }

    #[test]
    fn bits_per_byte_range() {
        let mut bpb = BitsPerByte::new();
        // Uniform prediction over 256 classes → BPB = 8.0
        let uniform = vec![0.0f32; 256];
        for _ in 0..100 {
            bpb.update(&uniform, 0);
        }
        assert!((bpb.compute() - 8.0).abs() < 0.1, "uniform 256 should be ~8 BPB, got {}", bpb.compute());
    }

    #[test]
    fn perplexity_of_uniform() {
        let mut ppl = Perplexity::new();
        let uniform = vec![0.0f32; 10]; // 10 classes
        for _ in 0..100 {
            ppl.update(&uniform, 0);
        }
        // Perplexity of uniform 10-class = 10.0
        assert!((ppl.compute() - 10.0).abs() < 0.5, "uniform 10-class PPL should be ~10, got {}", ppl.compute());
    }

    #[test]
    fn compute_bpb_rejects_too_short_data() {
        // Context 8 needs at least 9 tokens. 5 tokens should fail
        // loudly, not return a bogus 0.0.
        let tokens = vec![0usize; 5];
        let err = compute_bpb(&tokens, 8, |_| vec![])
            .expect_err("should refuse to evaluate on too-short data");
        assert!(err.contains("too short"), "err = {err}");
    }

    #[test]
    fn compute_bpb_uniform_forward_hits_log2_vocab() {
        // A model that predicts a uniform distribution over 256 symbols
        // should score exactly 8 bpb (= log2(256)). The closure returns
        // all-zero logits, which softmax to uniform. Any deviation in
        // this test means either the windowing or the softmax is wrong.
        const VOCAB: usize = 256;
        const CTX: usize = 8;
        let tokens: Vec<usize> = (0..100).map(|i| i % VOCAB).collect();

        let (bpb, n) = compute_bpb(&tokens, CTX, |chunk| {
            chunk.iter().map(|_| vec![0.0f32; VOCAB]).collect()
        }).unwrap();

        // Expect exactly (100 / 8) windows * 8 positions = 96 targets.
        assert_eq!(n, 96);
        assert!((bpb - 8.0).abs() < 0.05, "expected ~8.0 bpb, got {bpb}");
    }

    #[test]
    fn compute_bpb_confident_correct_forward_is_near_zero() {
        // Opposite end of the range from the uniform test: a forward
        // that always predicts token 0 with huge confidence, evaluated
        // on a stream where every token is 0, should score ~0 bpb.
        // Confirms cross-entropy is correctly signed and targets line
        // up with the logit rows.
        const VOCAB: usize = 16;
        const CTX: usize = 4;
        let tokens = vec![0usize; 25];   // every target will be token 0

        let (bpb, n) = compute_bpb(&tokens, CTX, |chunk| {
            chunk.iter().map(|_| {
                let mut row = vec![0.0f32; VOCAB];
                row[0] = 20.0;   // near-1 softmax mass on token 0
                row
            }).collect()
        }).unwrap();

        assert!(n > 0);
        assert!(bpb < 0.01,
            "confident correct predictions should collapse bpb to ~0, got {bpb}");
    }

    #[test]
    fn compute_bpb_count_matches_window_math() {
        // n_positions should equal floor(len / CTX) * CTX once the
        // length gate passes. Regression guard against an off-by-one
        // in the windowing.
        const CTX: usize = 8;
        let tokens = vec![0usize; 100];
        let (_, n) = compute_bpb(&tokens, CTX, |chunk| {
            chunk.iter().map(|_| vec![0.0f32; 4]).collect()
        }).unwrap();
        // 100 tokens, windows advance by CTX=8. Last window starts at
        // offset where offset + CTX + 1 <= 100 ⇒ offset <= 91. Offsets
        // are 0, 8, 16, …, 88 — that's 12 windows × 8 positions = 96.
        assert_eq!(n, 96);
    }

    #[test]
    fn evaluate_multiple_metrics() {
        let data = vec![
            (vec![0.1f32, 0.9], 1usize),
            (vec![0.9, 0.1], 0),
            (vec![0.3, 0.7], 1),
        ];
        let mut metrics: Vec<Box<dyn Metric>> = vec![
            Box::new(Accuracy::new()),
            Box::new(MeanLoss::new()),
        ];
        let results = evaluate(&mut metrics, &data);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "accuracy");
        assert!((results[0].1 - 1.0).abs() < 1e-5); // all correct
    }
}
