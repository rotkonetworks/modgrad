//! Hash n-gram embeddings for byte-level models.
//!
//! From BLT (Pagnoni et al. 2024): instead of embedding each byte alone,
//! hash the preceding 3-8 bytes and add their embeddings. This gives
//! sub-word context without a tokenizer.
//!
//! embed(byte_i) = embed_table[byte_i] + Σ hash_embed_n[hash(bytes[i-n+1..i])]
//!                                         n=3..8
//!
//! The hash n-gram embeddings are the single most impactful component
//! for matching tokenizer-based model performance at byte level.

use serde::{Deserialize, Serialize};
use modgrad_compute::neuron::SimpleRng;

/// Rolling polynomial hash for byte n-grams.
/// Uses a large prime base for good distribution.
#[inline]
fn rolling_poly_hash(bytes: &[u8]) -> u64 {
    const BASE: u64 = 2654435761; // golden ratio prime
    let mut h: u64 = 0;
    for (j, &b) in bytes.iter().enumerate() {
        h = h.wrapping_add((b as u64).wrapping_mul(BASE.wrapping_pow(j as u32)));
    }
    h
}

/// Hash n-gram embedding tables for byte sequences.
///
/// For each n in [min_n..max_n], maintains an embedding table of `vocab_per_n`
/// entries. Each byte position looks up its preceding n-gram, hashes it,
/// and adds the resulting embedding to the byte embedding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NgramHashEmbeddings {
    /// Embedding tables: one per n-gram size. tables[0] is for n=min_n.
    pub tables: Vec<Vec<f32>>,
    /// Embedding dimension (must match byte embed dim).
    pub embed_dim: usize,
    /// Vocabulary size per n-gram table.
    pub vocab_per_n: usize,
    /// Minimum n-gram size (inclusive).
    pub min_n: usize,
    /// Maximum n-gram size (inclusive).
    pub max_n: usize,
}

impl NgramHashEmbeddings {
    /// Create new hash n-gram embedding tables.
    ///
    /// `embed_dim`: must match the byte embedding dimension.
    /// `vocab_per_n`: hash table size per n-gram length (BLT uses 500K).
    /// `min_n`, `max_n`: n-gram range (BLT uses 3..=8).
    pub fn new(embed_dim: usize, vocab_per_n: usize, min_n: usize, max_n: usize) -> Self {
        let n_tables = max_n - min_n + 1;
        let mut rng = SimpleRng::new((embed_dim * vocab_per_n * n_tables) as u64);
        let scale = 1.0 / ((n_tables + 1) as f32 * embed_dim as f32).sqrt();

        let tables: Vec<Vec<f32>> = (0..n_tables)
            .map(|_| {
                (0..vocab_per_n * embed_dim)
                    .map(|_| rng.next_normal() * scale)
                    .collect()
            })
            .collect();

        Self { tables, embed_dim, vocab_per_n, min_n, max_n }
    }

    /// Augment a byte embedding with n-gram hash embeddings.
    ///
    /// `embedding`: the base byte embedding [embed_dim], modified in place.
    /// `context`: the byte sequence up to and including the current byte.
    /// `pos`: position of the current byte in the context.
    ///
    /// For each n in [min_n..max_n], if pos >= n-1, hashes bytes[pos-n+1..=pos]
    /// and adds the corresponding embedding.
    #[inline]
    pub fn augment(&self, embedding: &mut [f32], context: &[u8], pos: usize) {
        let d = self.embed_dim;
        debug_assert_eq!(embedding.len(), d);

        for n_idx in 0..self.tables.len() {
            let n = self.min_n + n_idx;
            if pos + 1 < n { continue; } // not enough context

            let start = pos + 1 - n;
            let ngram = &context[start..=pos];
            let hash = rolling_poly_hash(ngram) as usize % self.vocab_per_n;
            let offset = hash * d;
            let table = &self.tables[n_idx];

            for j in 0..d {
                embedding[j] += table[offset + j];
            }
        }
    }

    /// Embed an entire byte sequence with n-gram augmentation.
    ///
    /// Returns [n_bytes × embed_dim] flat token vector.
    /// Each position gets: embed_table[byte] + Σ hash_embed_n[hash(ngram)].
    pub fn embed_bytes(
        &self,
        bytes: &[u8],
        embed_table: &[f32],
        embed_dim: usize,
    ) -> Vec<f32> {
        let n = bytes.len();
        let mut tokens = Vec::with_capacity(n * embed_dim);
        let norm = 1.0 / (self.tables.len() + 1) as f32; // normalize by n_sources

        for (i, &b) in bytes.iter().enumerate() {
            // Base byte embedding
            let base_offset = b as usize * embed_dim;
            let mut emb: Vec<f32> = embed_table[base_offset..base_offset + embed_dim].to_vec();

            // Add n-gram hash embeddings
            self.augment(&mut emb, bytes, i);

            // Normalize (BLT divides by n_gram_sizes + 1)
            for v in &mut emb { *v *= norm; }

            tokens.extend_from_slice(&emb);
        }
        tokens
    }

    /// Total parameters in the hash tables.
    pub fn n_params(&self) -> usize {
        self.tables.len() * self.vocab_per_n * self.embed_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ngram_augments_embedding() {
        let nge = NgramHashEmbeddings::new(32, 1000, 3, 5);
        let mut emb = vec![0.0f32; 32];
        let context = b"hello world";

        // Position 0,1: too short for any n-gram (min_n=3)
        nge.augment(&mut emb, context, 0);
        assert!(emb.iter().all(|&v| v == 0.0), "pos 0 should have no n-gram");

        nge.augment(&mut emb, context, 1);
        assert!(emb.iter().all(|&v| v == 0.0), "pos 1 should have no n-gram");

        // Position 2: has 3-gram "hel"
        nge.augment(&mut emb, context, 2);
        assert!(emb.iter().any(|&v| v != 0.0), "pos 2 should have 3-gram");

        // Position 5: has 3,4,5-gram
        let mut emb2 = vec![0.0f32; 32];
        nge.augment(&mut emb2, context, 5);
        let mag: f32 = emb2.iter().map(|v| v.abs()).sum();
        assert!(mag > 0.0, "pos 5 should have multiple n-grams");
    }

    #[test]
    fn embed_bytes_dims() {
        let nge = NgramHashEmbeddings::new(16, 500, 3, 6);
        let embed_table = vec![0.1f32; 256 * 16];
        let tokens = nge.embed_bytes(b"test", &embed_table, 16);
        assert_eq!(tokens.len(), 4 * 16);
    }

    #[test]
    fn deterministic() {
        let nge = NgramHashEmbeddings::new(8, 100, 3, 4);
        let embed_table = vec![0.5f32; 256 * 8];
        let t1 = nge.embed_bytes(b"abc", &embed_table, 8);
        let t2 = nge.embed_bytes(b"abc", &embed_table, 8);
        assert_eq!(t1, t2, "same input must produce same output");
    }
}
