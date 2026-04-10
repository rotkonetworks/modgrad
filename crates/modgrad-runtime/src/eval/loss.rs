//! Loss functions for transformer training.
//!
//! Cross-entropy loss with gradient computation.
//! No autograd — just explicit forward + backward for each loss type.

/// Cross-entropy loss for language model training.
///
/// Takes logits [seq_len × vocab_size] and target token IDs [seq_len].
/// Returns (scalar loss, gradient w.r.t. logits [seq_len × vocab_size]).
///
/// The gradient is softmax(logits) - one_hot(target), averaged over seq_len.
pub fn cross_entropy(
    logits: &[f32],     // [seq_len * vocab_size], row-major
    targets: &[i64],    // [seq_len]
    vocab_size: usize,
) -> (f32, Vec<f32>) {
    let seq_len = targets.len();
    debug_assert_eq!(logits.len(), seq_len * vocab_size);

    let mut total_loss = 0.0f32;
    let mut grad = vec![0.0f32; seq_len * vocab_size];

    for t in 0..seq_len {
        let offset = t * vocab_size;
        let row = &logits[offset..offset + vocab_size];
        let target = targets[t] as usize;

        // Numerically stable softmax
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        for &v in row {
            sum_exp += (v - max).exp();
        }
        let log_sum_exp = max + sum_exp.ln();

        // Loss: -log P(target)
        total_loss += log_sum_exp - row[target];

        // Gradient: softmax(logits) - one_hot(target)
        let inv_sum = 1.0 / sum_exp;
        for i in 0..vocab_size {
            let prob = (row[i] - max).exp() * inv_sum;
            grad[offset + i] = prob;
        }
        grad[offset + target] -= 1.0;
    }

    // Average over sequence length
    let inv_seq = 1.0 / seq_len as f32;
    total_loss *= inv_seq;
    for g in &mut grad {
        *g *= inv_seq;
    }

    (total_loss, grad)
}

/// Cross-entropy loss for a single position (single token prediction).
///
/// Useful for decode-mode training or evaluation.
pub fn cross_entropy_single(
    logits: &[f32],    // [vocab_size]
    target: usize,
) -> (f32, Vec<f32>) {
    let vocab_size = logits.len();

    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum_exp = 0.0f32;
    for &v in logits {
        sum_exp += (v - max).exp();
    }
    let log_sum_exp = max + sum_exp.ln();

    let loss = log_sum_exp - logits[target];

    let inv_sum = 1.0 / sum_exp;
    let mut grad = Vec::with_capacity(vocab_size);
    for (i, &v) in logits.iter().enumerate() {
        let prob = (v - max).exp() * inv_sum;
        if i == target {
            grad.push(prob - 1.0);
        } else {
            grad.push(prob);
        }
    }

    (loss, grad)
}

/// Perplexity from average cross-entropy loss.
#[inline]
pub fn perplexity(avg_loss: f32) -> f32 {
    avg_loss.exp()
}

/// Top-k accuracy: is the correct token in the top-k predictions?
pub fn top_k_accuracy(logits: &[f32], targets: &[i64], vocab_size: usize, k: usize) -> f32 {
    let seq_len = targets.len();
    let mut correct = 0usize;

    for t in 0..seq_len {
        let row = &logits[t * vocab_size..(t + 1) * vocab_size];
        let target = targets[t] as usize;

        // Find top-k indices
        let mut indexed: Vec<(usize, f32)> = row.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        if indexed[..k.min(indexed.len())].iter().any(|&(i, _)| i == target) {
            correct += 1;
        }
    }

    correct as f32 / seq_len as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_entropy_perfect() {
        // If logits strongly predict the correct token, loss should be near 0
        let mut logits = vec![-10.0; 100];
        logits[42] = 10.0; // Strong prediction for token 42
        let targets = vec![42i64];

        let (loss, grad) = cross_entropy(&logits, &targets, 100);
        assert!(loss < 0.001, "loss = {}", loss);
        // Gradient at correct token should be near 0 (softmax ≈ 1, minus 1 = ~0)
        assert!(grad[42].abs() < 0.001, "grad[42] = {}", grad[42]);
    }

    #[test]
    fn test_cross_entropy_uniform() {
        // Uniform logits → loss = ln(vocab_size)
        let logits = vec![0.0; 100];
        let targets = vec![0i64];

        let (loss, _) = cross_entropy(&logits, &targets, 100);
        let expected = (100.0f32).ln();
        assert!((loss - expected).abs() < 0.01, "loss = {}, expected = {}", loss, expected);
    }

    #[test]
    fn test_gradient_sums_to_zero() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let targets = vec![2i64];
        let (_, grad) = cross_entropy(&logits, &targets, 4);

        let sum: f32 = grad.iter().sum();
        assert!(sum.abs() < 1e-6, "gradient sum = {} (should be ~0)", sum);
    }

    #[test]
    fn test_perplexity() {
        // loss=0 → perplexity=1 (perfect prediction)
        assert!((perplexity(0.0) - 1.0).abs() < 1e-6);
        // loss=ln(V) → perplexity=V (random guessing)
        assert!((perplexity(100.0f32.ln()) - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_top_k_accuracy() {
        let logits = vec![
            1.0, 5.0, 3.0, 2.0,  // token 1 is top
            4.0, 1.0, 2.0, 3.0,  // token 0 is top
        ];
        let targets = vec![1, 0];
        assert_eq!(top_k_accuracy(&logits, &targets, 4, 1), 1.0);
        assert_eq!(top_k_accuracy(&logits, &targets, 4, 3), 1.0);
    }
}
