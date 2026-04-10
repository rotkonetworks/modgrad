//! Loss functions for transformer training.

/// Cross-entropy loss for language model training.
///
/// Takes logits [seq_len * vocab_size] and target token IDs [seq_len].
/// Returns (scalar loss, gradient w.r.t. logits).
pub fn cross_entropy(
    logits: &[f32],
    targets: &[i64],
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

        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        for &v in row {
            sum_exp += (v - max).exp();
        }
        let log_sum_exp = max + sum_exp.ln();

        total_loss += log_sum_exp - row[target];

        let inv_sum = 1.0 / sum_exp;
        for i in 0..vocab_size {
            let prob = (row[i] - max).exp() * inv_sum;
            grad[offset + i] = prob;
        }
        grad[offset + target] -= 1.0;
    }

    let inv_seq = 1.0 / seq_len as f32;
    total_loss *= inv_seq;
    for g in &mut grad {
        *g *= inv_seq;
    }

    (total_loss, grad)
}
