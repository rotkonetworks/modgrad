//! Training data pipeline for language model training.
//!
//! Reads text → tokenizes → packs into fixed-length sequences → batches.
//!
//! Usage:
//!   let tok = Bpe::byte_level();
//!   let loader = DataLoader::from_file("data.txt", &tok, 512)?;
//!   for (input, target) in loader.batches(4) {
//!       // input: [batch_size][seq_len] token IDs
//!       // target: [batch_size][seq_len] shifted by 1
//!   }

use std::fs;
use std::io;
use std::path::Path;
use modgrad_persist::bpe::Bpe;

/// Training data: a flat array of token IDs packed into sequences.
pub struct DataLoader {
    /// All tokens concatenated.
    tokens: Vec<i64>,
    /// Sequence length for training.
    pub seq_len: usize,
    /// Total number of complete sequences.
    pub num_sequences: usize,
}

/// A single training batch: inputs and targets (shifted by 1).
pub struct Batch {
    /// Input token IDs: [batch_size][seq_len].
    pub input: Vec<Vec<i64>>,
    /// Target token IDs: [batch_size][seq_len] (input shifted right by 1).
    pub target: Vec<Vec<i64>>,
}

impl DataLoader {
    /// Load and tokenize a text file.
    ///
    /// Reads the entire file, tokenizes, and packs into sequences of `seq_len + 1`
    /// (the +1 is for the target shift).
    pub fn from_file(path: &Path, tokenizer: &Bpe, seq_len: usize) -> io::Result<Self> {
        let text = fs::read_to_string(path)?;
        Ok(Self::from_text(&text, tokenizer, seq_len))
    }

    /// Tokenize text and prepare for training.
    pub fn from_text(text: &str, tokenizer: &Bpe, seq_len: usize) -> Self {
        let tokens = tokenizer.encode(text);
        let num_sequences = if tokens.len() > seq_len {
            tokens.len() - seq_len
        } else {
            0
        };

        Self { tokens, seq_len, num_sequences }
    }

    /// Load from pre-tokenized token IDs.
    pub fn from_tokens(tokens: Vec<i64>, seq_len: usize) -> Self {
        let num_sequences = if tokens.len() > seq_len {
            tokens.len() - seq_len
        } else {
            0
        };
        Self { tokens, seq_len, num_sequences }
    }

    /// Total tokens in the dataset.
    pub fn total_tokens(&self) -> usize {
        self.tokens.len()
    }

    /// Create an iterator over non-overlapping batches.
    ///
    /// Each batch has `batch_size` sequences. Sequences are packed tightly
    /// (no overlap, no padding). Last incomplete batch is dropped.
    pub fn batches(&self, batch_size: usize) -> BatchIter<'_> {
        // Compute stride: non-overlapping chunks
        let chunk_size = self.tokens.len() / batch_size;
        BatchIter {
            loader: self,
            batch_size,
            chunk_size,
            position: 0,
        }
    }

    /// Create a shuffled batch iterator (random sequence positions).
    pub fn shuffled_batches(&self, batch_size: usize, seed: u64) -> ShuffledBatchIter<'_> {
        // Generate shuffled indices
        let _num_batches = self.num_sequences / batch_size;
        let mut indices: Vec<usize> = (0..self.num_sequences).collect();

        // Fisher-Yates shuffle with simple LCG
        let mut rng = seed;
        for i in (1..indices.len()).rev() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let j = (rng >> 33) as usize % (i + 1);
            indices.swap(i, j);
        }

        ShuffledBatchIter {
            loader: self,
            indices,
            batch_size,
            position: 0,
        }
    }

    /// Get a single (input, target) pair at position `idx`.
    fn get_pair(&self, idx: usize) -> (Vec<i64>, Vec<i64>) {
        let input = self.tokens[idx..idx + self.seq_len].to_vec();
        let target = self.tokens[idx + 1..idx + self.seq_len + 1].to_vec();
        (input, target)
    }
}

/// Iterator over non-overlapping batches (streaming, no random access).
pub struct BatchIter<'a> {
    loader: &'a DataLoader,
    batch_size: usize,
    chunk_size: usize,
    position: usize,
}

impl<'a> Iterator for BatchIter<'a> {
    type Item = Batch;

    fn next(&mut self) -> Option<Batch> {
        if self.position + self.loader.seq_len >= self.chunk_size {
            return None;
        }

        let mut input = Vec::with_capacity(self.batch_size);
        let mut target = Vec::with_capacity(self.batch_size);

        for b in 0..self.batch_size {
            let start = b * self.chunk_size + self.position;
            if start + self.loader.seq_len >= self.loader.tokens.len() {
                return None;
            }
            let (inp, tgt) = self.loader.get_pair(start);
            input.push(inp);
            target.push(tgt);
        }

        self.position += self.loader.seq_len;
        Some(Batch { input, target })
    }
}

/// Iterator over shuffled batches.
pub struct ShuffledBatchIter<'a> {
    loader: &'a DataLoader,
    indices: Vec<usize>,
    batch_size: usize,
    position: usize,
}

impl<'a> Iterator for ShuffledBatchIter<'a> {
    type Item = Batch;

    fn next(&mut self) -> Option<Batch> {
        if self.position + self.batch_size > self.indices.len() {
            return None;
        }

        let mut input = Vec::with_capacity(self.batch_size);
        let mut target = Vec::with_capacity(self.batch_size);

        for i in 0..self.batch_size {
            let idx = self.indices[self.position + i];
            let (inp, tgt) = self.loader.get_pair(idx);
            input.push(inp);
            target.push(tgt);
        }

        self.position += self.batch_size;
        Some(Batch { input, target })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_loading() {
        let tok = Bpe::byte_level();
        let text = "Hello, world! This is a test of the data loader.";
        let loader = DataLoader::from_text(text, &tok, 8);

        assert!(loader.total_tokens() > 0);
        assert!(loader.num_sequences > 0);
    }

    #[test]
    fn test_batch_shapes() {
        let tok = Bpe::byte_level();
        // Need enough text for batching
        let text = "a".repeat(1000);
        let loader = DataLoader::from_text(&text, &tok, 16);

        let batch = loader.batches(4).next().unwrap();
        assert_eq!(batch.input.len(), 4);
        assert_eq!(batch.input[0].len(), 16);
        assert_eq!(batch.target[0].len(), 16);
    }

    #[test]
    fn test_target_is_shifted() {
        let tok = Bpe::byte_level();
        let text = "abcdefghijklmnop";
        let loader = DataLoader::from_text(&text, &tok, 4);

        let batch = loader.batches(1).next().unwrap();
        // target should be input shifted by 1
        let inp = &batch.input[0];
        let tgt = &batch.target[0];
        // inp[1] == tgt[0] (shifted)
        assert_eq!(inp[1], tgt[0]);
        assert_eq!(inp[2], tgt[1]);
        assert_eq!(inp[3], tgt[2]);
    }

    #[test]
    fn test_shuffled_batches() {
        let tok = Bpe::byte_level();
        let text = "a".repeat(2000);
        let loader = DataLoader::from_text(&text, &tok, 8);

        let batches: Vec<_> = loader.shuffled_batches(4, 42).collect();
        assert!(!batches.is_empty());
    }
}
