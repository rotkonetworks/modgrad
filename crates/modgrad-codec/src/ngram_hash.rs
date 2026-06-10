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

    /// Flatten the hash tables into one row-major `[n_params]` buffer,
    /// **table-major** (table 0's rows first, then table 1, …). This is
    /// the canonical layout the device mirror (`ngram_tables_dev`) and the
    /// gradient slab (`d_ngram_tables`) both use, so the weight, the grad
    /// scatter, and the post-AdamW sync all agree. Inverse of
    /// [`load_flat_tables`](Self::load_flat_tables).
    pub fn flat_tables(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.n_params());
        for t in &self.tables {
            out.extend_from_slice(t);
        }
        out
    }

    /// Overwrite the hash tables from a flat buffer produced by
    /// [`flat_tables`](Self::flat_tables) (e.g. the AdamW-updated device
    /// mirror downloaded back to host). Same table-major layout.
    pub fn load_flat_tables(&mut self, flat: &[f32]) {
        debug_assert_eq!(flat.len(), self.n_params());
        let stride = self.vocab_per_n * self.embed_dim;
        for (i, t) in self.tables.iter_mut().enumerate() {
            t.copy_from_slice(&flat[i * stride..(i + 1) * stride]);
        }
    }

    /// Backward of [`augment`] w.r.t. the hash tables, for a single
    /// position. `d_emb` is the upstream gradient on the *final*
    /// (post-`norm`) augmented embedding at `pos`; it is scattered into
    /// every hash row that position hit.
    ///
    /// The `1 / (n_tables + 1)` factor mirrors the forward `* norm` in
    /// [`embed_bytes`] (∂augmented/∂table_row = norm), exactly as the
    /// encoder's `scatter_byte_embed_grad` applies it to the base byte
    /// embedding. `table_grads[n_idx]` must match `tables[n_idx]`'s shape
    /// (`vocab_per_n * embed_dim`); contributions are *added* (the caller
    /// zeroes).
    pub fn accumulate_table_grads(
        &self,
        d_emb: &[f32],
        context: &[u8],
        pos: usize,
        table_grads: &mut [Vec<f32>],
    ) {
        let d = self.embed_dim;
        debug_assert_eq!(d_emb.len(), d);
        debug_assert_eq!(table_grads.len(), self.tables.len());
        let norm = 1.0 / (self.tables.len() + 1) as f32;

        for n_idx in 0..self.tables.len() {
            let n = self.min_n + n_idx;
            if pos + 1 < n { continue; } // not enough context (matches `augment`)

            let start = pos + 1 - n;
            let ngram = &context[start..=pos];
            let hash = rolling_poly_hash(ngram) as usize % self.vocab_per_n;
            let offset = hash * d;
            let g = &mut table_grads[n_idx];
            for j in 0..d {
                g[offset + j] += d_emb[j] * norm;
            }
        }
    }

    /// Sequence-level backward of [`embed_bytes`]: scatter the per-position
    /// upstream gradient `d_augmented` (`[n_bytes × embed_dim]`, i.e. the
    /// encoder's layer-0 input gradient) into both the base byte-embed
    /// gradient (`[256 × embed_dim]`) and the n-gram table gradients.
    ///
    /// Mirrors [`embed_bytes`] exactly: both the base byte row and every
    /// hit hash row receive `d_emb * norm`. Both grad slabs are *added*
    /// into (the caller zeroes). This is the host-side counterpart the
    /// encoder backward needs so the hash tables — currently frozen at
    /// init because nothing scatters into them — actually train.
    pub fn embed_bytes_backward(
        &self,
        d_augmented: &[f32],
        bytes: &[u8],
        d_byte_embed: &mut [f32],
        table_grads: &mut [Vec<f32>],
    ) {
        let d = self.embed_dim;
        let n = bytes.len();
        debug_assert_eq!(d_augmented.len(), n * d);
        let norm = 1.0 / (self.tables.len() + 1) as f32;

        for (i, &b) in bytes.iter().enumerate() {
            let d_emb = &d_augmented[i * d..(i + 1) * d];
            // Base byte embedding row.
            let base = b as usize * d;
            for j in 0..d {
                d_byte_embed[base + j] += d_emb[j] * norm;
            }
            // n-gram hash table rows.
            self.accumulate_table_grads(d_emb, bytes, i, table_grads);
        }
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

    /// Pure-host gradcheck of [`NgramHashEmbeddings::embed_bytes_backward`]
    /// against finite differences of [`embed_bytes`]. No GPU needed — the
    /// whole augmentation path is host code. Validates that the table
    /// scatter (and the base byte-embed scatter) carry the correct
    /// `1/(n_tables+1)` factor and hit the right rows.
    ///
    /// Loss `L = Σ_{t,j} w[t,j] · augmented[t,j]`, so `dL/d(augmented) = w`
    /// and `embed_bytes_backward(w, …)` yields `dL/d(table)` directly.
    #[test]
    fn flat_tables_round_trip() {
        let mut nge = NgramHashEmbeddings::new(4, 8, 2, 4); // 3 tables
        let original = nge.flat_tables();
        assert_eq!(original.len(), nge.n_params());
        // Perturb host tables, then restore from the flat snapshot.
        for t in &mut nge.tables {
            for v in t.iter_mut() { *v += 1.0; }
        }
        assert_ne!(nge.flat_tables(), original);
        nge.load_flat_tables(&original);
        assert_eq!(nge.flat_tables(), original, "flatten/unflatten must round-trip");
    }

    #[test]
    fn embed_bytes_backward_matches_finite_difference() {
        let d = 4usize;
        let vocab = 16usize;
        let mut nge = NgramHashEmbeddings::new(d, vocab, 2, 3); // 2 tables
        // Spread the (initially random) tables so no row is degenerate.
        let bytes: &[u8] = &[3, 1, 4, 1, 5, 9]; // hits n=2 (pos≥1) and n=3 (pos≥2)
        let embed_table = vec![0.05f32; 256 * d];

        // Fixed non-uniform upstream weights w[t,j].
        let n = bytes.len();
        let w: Vec<f32> = (0..n * d).map(|k| 0.3 + 0.11 * (k % 7) as f32).collect();
        let loss = |nge: &NgramHashEmbeddings| -> f32 {
            let aug = nge.embed_bytes(bytes, &embed_table, d);
            aug.iter().zip(&w).map(|(a, wi)| a * wi).sum()
        };

        // Analytic table grads.
        let mut d_byte_embed = vec![0.0f32; 256 * d];
        let mut table_grads: Vec<Vec<f32>> =
            nge.tables.iter().map(|t| vec![0.0f32; t.len()]).collect();
        nge.embed_bytes_backward(&w, bytes, &mut d_byte_embed, &mut table_grads);

        // Finite-difference every table entry; compare the ones that are
        // actually reachable (non-zero analytic) and assert the zeros are
        // genuinely unreachable (FD also ~0).
        const EPS: f32 = 1e-3;
        let mut checked = 0usize;
        for n_idx in 0..nge.tables.len() {
            for entry in 0..nge.tables[n_idx].len() {
                let orig = nge.tables[n_idx][entry];
                nge.tables[n_idx][entry] = orig + EPS;
                let lp = loss(&nge);
                nge.tables[n_idx][entry] = orig - EPS;
                let lm = loss(&nge);
                nge.tables[n_idx][entry] = orig;
                let num = (lp - lm) / (2.0 * EPS);
                let analytic = table_grads[n_idx][entry];

                if analytic.abs() < 1e-9 {
                    assert!(
                        num.abs() < 1e-6,
                        "table[{n_idx}][{entry}]: analytic 0 but FD={num:+e} \
                         (a reachable row was missed by the scatter)",
                    );
                    continue;
                }
                let rel = (num - analytic).abs() / num.abs().max(analytic.abs());
                assert!(
                    rel < 1e-3,
                    "table[{n_idx}][{entry}]: analytic={analytic:+e} FD={num:+e} \
                     rel_err={rel:.3e}",
                );
                checked += 1;
            }
        }
        assert!(checked > 0, "no reachable table entries — test is vacuous");

        // Also spot-check the base byte-embed scatter for one present byte.
        let mut d_byte_embed2 = vec![0.0f32; 256 * d];
        let mut tg2: Vec<Vec<f32>> =
            nge.tables.iter().map(|t| vec![0.0f32; t.len()]).collect();
        nge.embed_bytes_backward(&w, bytes, &mut d_byte_embed2, &mut tg2);
        let b0 = bytes[0] as usize;
        let mut embed_table_p = embed_table.clone();
        embed_table_p[b0 * d] += EPS;
        let lp = {
            let aug = nge.embed_bytes(bytes, &embed_table_p, d);
            aug.iter().zip(&w).map(|(a, wi)| a * wi).sum::<f32>()
        };
        embed_table_p[b0 * d] -= 2.0 * EPS;
        let lm = {
            let aug = nge.embed_bytes(bytes, &embed_table_p, d);
            aug.iter().zip(&w).map(|(a, wi)| a * wi).sum::<f32>()
        };
        let num = (lp - lm) / (2.0 * EPS);
        let analytic = d_byte_embed2[b0 * d];
        let rel = (num - analytic).abs() / num.abs().max(analytic.abs()).max(1e-9);
        assert!(
            rel < 1e-3,
            "byte_embed[{b0}][0]: analytic={analytic:+e} FD={num:+e} rel_err={rel:.3e}",
        );
    }
}
