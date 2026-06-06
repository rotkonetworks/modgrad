//! Held-out evaluation helpers — byte-level character error rate (CER)
//! against a greedy CTC decode.

/// Byte-level Levenshtein distance between two ascii strings.
///
/// O(|a| × |b|) time, O(min) space. Smoke-test scale is small enough
/// that this is fine — no need for a fancier algorithm.
pub fn levenshtein(a: &[u8], b: &[u8]) -> usize {
    if a.is_empty() { return b.len(); }
    if b.is_empty() { return a.len(); }
    let (a, b) = if a.len() < b.len() { (a, b) } else { (b, a) };
    let mut prev: Vec<usize> = (0..=a.len()).collect();
    let mut cur = vec![0usize; a.len() + 1];
    for (i, &bi) in b.iter().enumerate() {
        cur[0] = i + 1;
        for (j, &aj) in a.iter().enumerate() {
            let cost = if aj == bi { 0 } else { 1 };
            cur[j + 1] = (prev[j + 1] + 1).min(cur[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[a.len()]
}

/// Character error rate = sum of edit distances / sum of target lengths.
///
/// Standard ASR/OCR metric. 0.0 = perfect, 1.0 = at least as many edits
/// as characters. Values above 1.0 are possible (the model produces
/// more spurious characters than the ground truth has).
pub fn cer(pairs: &[(String, String)]) -> f32 {
    let mut edits = 0usize;
    let mut chars = 0usize;
    for (target, predicted) in pairs {
        edits += levenshtein(target.as_bytes(), predicted.as_bytes());
        chars += target.len();
    }
    if chars == 0 { return 0.0; }
    edits as f32 / chars as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levenshtein_basic() {
        assert_eq!(levenshtein(b"hello", b"hello"), 0);
        assert_eq!(levenshtein(b"hello", b"hallo"), 1);
        assert_eq!(levenshtein(b"kitten", b"sitting"), 3);
        assert_eq!(levenshtein(b"", b"abc"), 3);
        assert_eq!(levenshtein(b"abc", b""), 3);
    }

    #[test]
    fn cer_basic() {
        let pairs = vec![
            ("hello".to_string(), "hello".to_string()),
            ("world".to_string(), "wprld".to_string()),
        ];
        // 0 + 1 edits over 5 + 5 chars = 0.1
        assert!((cer(&pairs) - 0.1).abs() < 1e-6);
    }
}
