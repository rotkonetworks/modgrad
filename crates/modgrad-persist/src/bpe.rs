//! Byte-Pair Encoding tokenizer for transformer training.
//!
//! Two modes:
//!   - Byte-level (vocab=256+specials): zero training, works on anything
//!   - BPE (vocab=N): trained on corpus, better compression
//!
//! Usage:
//!   let tok = Bpe::byte_level();           // instant, vocab=259
//!   let tok = Bpe::train(text, 4096);      // train BPE, vocab=4096
//!   let ids = tok.encode("hello world");
//!   let text = tok.decode(&ids);

use std::collections::HashMap;
use std::io::{self, BufRead, BufReader, Write as IoWrite};
use std::fs::File;
use std::path::Path;

/// Special token IDs (always the last 3 in vocab).
const NUM_SPECIALS: u32 = 3;

/// Byte-Pair Encoding tokenizer.
pub struct Bpe {
    /// Token ID → byte sequence.
    vocab: Vec<Vec<u8>>,
    /// Byte sequence → token ID (for base tokens + merges).
    encoder: HashMap<Vec<u8>, u32>,
    /// Ordered merge rules: (left_id, right_id) applied in sequence during encoding.
    merges: Vec<(u32, u32)>,
    /// Fast merge lookup: (left_id, right_id) → rank (lower = higher priority).
    merge_rank: HashMap<(u32, u32), usize>,
    /// Special token offsets (set after vocab is built).
    pub bos_id: u32,
    pub eos_id: u32,
    pub pad_id: u32,
}

impl Bpe {
    /// Create a byte-level tokenizer (no training needed).
    ///
    /// Vocab = 256 byte tokens + 3 specials (BOS, EOS, PAD) = 259 total.
    /// Good for quick experiments. Every byte maps to exactly one token.
    pub fn byte_level() -> Self {
        let mut vocab: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
        let mut encoder = HashMap::new();
        for (i, v) in vocab.iter().enumerate() {
            encoder.insert(v.clone(), i as u32);
        }

        // Add specials
        let bos_id = vocab.len() as u32;
        vocab.push(b"<BOS>".to_vec());
        let eos_id = vocab.len() as u32;
        vocab.push(b"<EOS>".to_vec());
        let pad_id = vocab.len() as u32;
        vocab.push(b"<PAD>".to_vec());

        Self {
            vocab,
            encoder,
            merges: Vec::new(),
            merge_rank: HashMap::new(),
            bos_id,
            eos_id,
            pad_id,
        }
    }

    /// Train BPE on a text corpus.
    ///
    /// Starts with 256 byte tokens, greedily merges most frequent pairs
    /// until `vocab_size` is reached. `vocab_size` includes the 3 special tokens.
    pub fn train(text: &str, vocab_size: usize) -> Self {
        assert!(vocab_size > 256 + NUM_SPECIALS as usize,
            "vocab_size must be > {} (256 bytes + specials)", 256 + NUM_SPECIALS);

        let num_merges = vocab_size - 256 - NUM_SPECIALS as usize;
        let bytes = text.as_bytes();

        // Start: each byte is its own token
        let mut ids: Vec<u32> = bytes.iter().map(|&b| b as u32).collect();

        // Build base vocab
        let mut vocab: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
        let mut encoder: HashMap<Vec<u8>, u32> = HashMap::new();
        for (i, v) in vocab.iter().enumerate() {
            encoder.insert(v.clone(), i as u32);
        }

        let mut merges = Vec::with_capacity(num_merges);
        let mut merge_rank = HashMap::new();

        for step in 0..num_merges {
            if ids.len() < 2 { break; }

            // Count adjacent pairs
            let mut pair_counts: HashMap<(u32, u32), usize> = HashMap::new();
            for w in ids.windows(2) {
                *pair_counts.entry((w[0], w[1])).or_insert(0) += 1;
            }

            // Find most frequent pair
            let best = pair_counts.into_iter()
                .max_by_key(|&(_, count)| count);

            let (best_pair, count) = match best {
                Some(b) if b.1 >= 2 => b,
                _ => break, // No pair appears more than once
            };

            // Create merged token
            let new_id = vocab.len() as u32;
            let mut merged_bytes = vocab[best_pair.0 as usize].clone();
            merged_bytes.extend_from_slice(&vocab[best_pair.1 as usize]);
            encoder.insert(merged_bytes.clone(), new_id);
            vocab.push(merged_bytes);

            merges.push(best_pair);
            merge_rank.insert(best_pair, step);

            // Apply merge to the token sequence
            let mut new_ids = Vec::with_capacity(ids.len());
            let mut i = 0;
            while i < ids.len() {
                if i + 1 < ids.len() && ids[i] == best_pair.0 && ids[i + 1] == best_pair.1 {
                    new_ids.push(new_id);
                    i += 2;
                } else {
                    new_ids.push(ids[i]);
                    i += 1;
                }
            }
            ids = new_ids;

            if (step + 1) % 500 == 0 || step + 1 == num_merges {
                eprintln!("  BPE merge {}/{}: vocab={}, pair count={}, tokens={}",
                    step + 1, num_merges, vocab.len(), count, ids.len());
            }
        }

        // Add specials
        let bos_id = vocab.len() as u32;
        vocab.push(b"<BOS>".to_vec());
        let eos_id = vocab.len() as u32;
        vocab.push(b"<EOS>".to_vec());
        let pad_id = vocab.len() as u32;
        vocab.push(b"<PAD>".to_vec());

        Self { vocab, encoder, merges, merge_rank, bos_id, eos_id, pad_id }
    }

    /// Vocab size (including specials).
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> Vec<i64> {
        let bytes = text.as_bytes();
        if bytes.is_empty() { return Vec::new(); }

        // Start with byte-level tokens
        let mut ids: Vec<u32> = bytes.iter().map(|&b| b as u32).collect();

        // Apply merges in order (greedy left-to-right)
        for &(left, right) in &self.merges {
            let merged = *self.encoder
                .get(&[self.vocab[left as usize].as_slice(),
                       self.vocab[right as usize].as_slice()].concat())
                .expect("merge target missing from encoder");

            let mut new_ids = Vec::with_capacity(ids.len());
            let mut i = 0;
            while i < ids.len() {
                if i + 1 < ids.len() && ids[i] == left && ids[i + 1] == right {
                    new_ids.push(merged);
                    i += 2;
                } else {
                    new_ids.push(ids[i]);
                    i += 1;
                }
            }
            ids = new_ids;
        }

        ids.into_iter().map(|id| id as i64).collect()
    }

    /// Encode with BOS/EOS tokens.
    pub fn encode_with_specials(&self, text: &str) -> Vec<i64> {
        let mut ids = vec![self.bos_id as i64];
        ids.extend(self.encode(text));
        ids.push(self.eos_id as i64);
        ids
    }

    /// Decode token IDs to text.
    pub fn decode(&self, ids: &[i64]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            let id = id as u32 as usize;
            if id < self.vocab.len() {
                let token = &self.vocab[id];
                // Skip specials in decode output
                if id as u32 == self.bos_id || id as u32 == self.eos_id || id as u32 == self.pad_id {
                    continue;
                }
                bytes.extend_from_slice(token);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Save tokenizer to a file (merges + vocab).
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let file = File::create(path)?;
        let mut w = io::BufWriter::new(file);

        // Header
        writeln!(w, "bpe v1")?;
        writeln!(w, "vocab_size {}", self.vocab.len())?;
        writeln!(w, "num_merges {}", self.merges.len())?;
        writeln!(w, "bos_id {}", self.bos_id)?;
        writeln!(w, "eos_id {}", self.eos_id)?;
        writeln!(w, "pad_id {}", self.pad_id)?;

        // Merges (one per line: left_id right_id)
        for &(l, r) in &self.merges {
            writeln!(w, "{} {}", l, r)?;
        }

        Ok(())
    }

    /// Load tokenizer from a file.
    pub fn load(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        // Parse header
        let header = lines.next().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "empty file"))??;
        if header != "bpe v1" {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad header"));
        }

        let parse_field = |line: &str, prefix: &str| -> io::Result<usize> {
            line.strip_prefix(prefix)
                .and_then(|s| s.trim().parse().ok())
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, format!("bad field: {}", prefix)))
        };

        let vocab_size = parse_field(&lines.next().ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, ""))??, "vocab_size ")?;
        let num_merges = parse_field(&lines.next().ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, ""))??, "num_merges ")?;
        let bos_id = parse_field(&lines.next().ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, ""))??, "bos_id ")? as u32;
        let eos_id = parse_field(&lines.next().ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, ""))??, "eos_id ")? as u32;
        let pad_id = parse_field(&lines.next().ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, ""))??, "pad_id ")? as u32;

        // Build base vocab (256 bytes)
        let mut vocab: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
        let mut encoder: HashMap<Vec<u8>, u32> = HashMap::new();
        for (i, v) in vocab.iter().enumerate() {
            encoder.insert(v.clone(), i as u32);
        }

        // Read merges and rebuild vocab
        let mut merges = Vec::with_capacity(num_merges);
        let mut merge_rank = HashMap::new();

        for step in 0..num_merges {
            let line = lines.next().ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, ""))??;
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() != 2 {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "bad merge line"));
            }
            let left: u32 = parts[0].parse().map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "bad merge id"))?;
            let right: u32 = parts[1].parse().map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "bad merge id"))?;

            // Reconstruct merged token
            let mut merged_bytes = vocab[left as usize].clone();
            merged_bytes.extend_from_slice(&vocab[right as usize]);
            let new_id = vocab.len() as u32;
            encoder.insert(merged_bytes.clone(), new_id);
            vocab.push(merged_bytes);

            merges.push((left, right));
            merge_rank.insert((left, right), step);
        }

        // Add specials
        while vocab.len() < vocab_size {
            vocab.push(Vec::new());
        }

        Ok(Self { vocab, encoder, merges, merge_rank, bos_id, eos_id, pad_id })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_level_roundtrip() {
        let tok = Bpe::byte_level();
        assert_eq!(tok.vocab_size(), 259); // 256 + 3 specials

        let text = "Hello, world! 🦀";
        let ids = tok.encode(text);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn byte_level_specials() {
        let tok = Bpe::byte_level();
        let ids = tok.encode_with_specials("hi");
        assert_eq!(ids[0], tok.bos_id as i64);
        assert_eq!(*ids.last().unwrap(), tok.eos_id as i64);
        // Decode should strip specials
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, "hi");
    }

    #[test]
    fn bpe_train_reduces_tokens() {
        let text = "abababababababababababababababab cdcdcdcdcdcdcdcdcdcdcdcdcdcdcd";
        let tok = Bpe::train(text, 260); // 256 + 1 merge + 3 specials
        assert!(tok.vocab_size() <= 260);

        // "ab" should be merged → fewer tokens than bytes
        let ids = tok.encode("abababab");
        assert!(ids.len() < 8, "expected merge to reduce token count, got {}", ids.len());
    }

    #[test]
    fn bpe_roundtrip() {
        let text = "the cat sat on the mat. the cat sat on the mat.";
        let tok = Bpe::train(text, 280);
        let ids = tok.encode(text);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text);
    }
}
