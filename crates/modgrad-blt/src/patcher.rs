//! Offline byte-stream → patch-boundary segmenter.
//!
//! Per BLT paper §2.3, identifies patch boundaries from per-byte
//! entropies under a small byte-LM:
//!
//! - **Global threshold:** `H(x_t) > θ_g` starts a new patch.
//! - **Approx. monotonic:** `H(x_t) − H(x_{t-1}) > θ_r` starts a new
//!   patch (handles "entropy drift" in long contexts; reset entropy
//!   model context on newlines per §4.4).
//!
//! Boundaries are computed once per dataloading step, then the
//! tokenizer-equivalent step in BLT training is "look up which patch
//! this byte belongs to."
//!
//! ## Output contract
//!
//! [`EntropyPatcher::boundaries`] returns a `Vec<usize>` of length
//! `n_patches + 1` such that patch `i` covers
//! `bytes[boundaries[i]..boundaries[i+1]]`. The first element is
//! always `0` and the last is always `bytes.len()`. A boundary at
//! position `t` (`0 < t < bytes.len()`) means a new patch starts at
//! byte `t` — the byte at `t` belongs to the new patch, not the
//! previous one. This matches the convention in the paper's Figure 2
//! and is the natural shape for an `Iterator::windows`-style patch
//! enumeration in the encoder.
//!
//! Pure host-side logic — no GPU dependency. Owner: agent leah.

use serde::{Deserialize, Serialize};

/// Patch-boundary criterion. The entropy model emits a per-byte
/// entropy estimate `H(x_t)`; the patcher consumes that array and
/// fires boundaries under one of two rules.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PatchMode {
    /// Fire a boundary at position `t` when `H(x_t) > θ_g`.
    /// Paper's "global" rule (§2.3).
    GlobalThreshold(f32),
    /// Fire a boundary at position `t` when
    /// `H(x_t) − H(x_{t-1}) > θ_r`. Paper's "approximate monotonic"
    /// rule. Robust to entropy-baseline drift in long contexts.
    ApproxMonotonic(f32),
}

/// Offline entropy-driven patcher.
///
/// Stateless — `boundaries` is a pure function of `(bytes, entropies,
/// mode, reset_on_newline)`. Construct once and reuse across batches;
/// it is `Copy + Send + Sync`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EntropyPatcher {
    /// Boundary criterion.
    pub mode: PatchMode,
    /// Per paper §4.4: forcibly start a new patch right after every
    /// `b'\n'` byte. The entropy model's context is reset on newlines
    /// for the same reason — drift in the byte-LM's running entropy
    /// baseline, which would otherwise make the monotonic rule fire
    /// erratically across paragraph boundaries.
    pub reset_on_newline: bool,
}

impl EntropyPatcher {
    /// New patcher. Default is `reset_on_newline = true` per paper §4.4.
    pub fn new(mode: PatchMode) -> Self {
        Self { mode, reset_on_newline: true }
    }

    /// Disable the §4.4 newline reset. Boundaries are then governed
    /// purely by the entropy criterion.
    pub fn without_newline_reset(mut self) -> Self {
        self.reset_on_newline = false;
        self
    }

    /// Compute patch boundaries from a byte sequence and its per-byte
    /// entropies. Returns `boundaries` such that patch `i` covers
    /// `bytes[boundaries[i]..boundaries[i+1]]`. Always starts with
    /// `0` and ends with `bytes.len()`.
    ///
    /// `entropies` must have the same length as `bytes`. Empty input
    /// returns `[0]` (no patches).
    pub fn boundaries(
        &self,
        bytes: &[u8],
        entropies: &[f32],
    ) -> Vec<usize> {
        debug_assert_eq!(
            bytes.len(), entropies.len(),
            "EntropyPatcher::boundaries: bytes and entropies must \
             have equal length (bytes={}, entropies={})",
            bytes.len(), entropies.len(),
        );

        if bytes.is_empty() {
            return vec![0];
        }

        let mut boundaries = Vec::with_capacity(bytes.len() / 4 + 2);
        boundaries.push(0);

        // `reset_at` is the position where the previous "context"
        // begins — used by the monotonic rule's prev-entropy lookup
        // and by the newline-reset semantics. A boundary fired at
        // position `t` means the new patch starts at `t`; the
        // monotonic rule should not compare across that boundary
        // (the prev-entropy at `t-1` is in the previous patch and
        // its baseline may not match).
        let mut prev_entropy_valid_from = 0usize;

        for t in 1..bytes.len() {
            // Newline reset: byte at `t-1` was '\n' ⇒ start a new
            // patch at `t` regardless of the entropy criterion. The
            // patch break sits *after* the newline so the newline is
            // the last byte of its patch (matches the paper's intent
            // that paragraphs are natural patch boundaries).
            let newline_break = self.reset_on_newline && bytes[t - 1] == b'\n';

            let entropy_break = match self.mode {
                PatchMode::GlobalThreshold(theta_g) => {
                    entropies[t] > theta_g
                }
                PatchMode::ApproxMonotonic(theta_r) => {
                    if t <= prev_entropy_valid_from {
                        // No valid prev-entropy in the current
                        // context window — can't fire monotonic.
                        false
                    } else {
                        entropies[t] - entropies[t - 1] > theta_r
                    }
                }
            };

            if newline_break || entropy_break {
                // Avoid emitting duplicate boundaries (would yield
                // an empty patch). With the §4.4 reset, an entropy
                // spike right at the byte after a newline would
                // otherwise produce `boundaries = [.., t, t, ..]`.
                if *boundaries.last().unwrap() != t {
                    boundaries.push(t);
                }
                prev_entropy_valid_from = t;
            }
        }

        if *boundaries.last().unwrap() != bytes.len() {
            boundaries.push(bytes.len());
        }
        boundaries
    }

    /// Convenience: number of patches produced for a given input.
    pub fn n_patches(&self, bytes: &[u8], entropies: &[f32]) -> usize {
        let b = self.boundaries(bytes, entropies);
        b.len().saturating_sub(1)
    }
}

// ─── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_returns_single_zero() {
        let p = EntropyPatcher::new(PatchMode::GlobalThreshold(1.0));
        assert_eq!(p.boundaries(&[], &[]), vec![0]);
        assert_eq!(p.n_patches(&[], &[]), 0);
    }

    #[test]
    fn boundaries_always_start_zero_end_len() {
        let p = EntropyPatcher::new(PatchMode::GlobalThreshold(0.5));
        let bytes = b"hello world";
        let entropies = vec![0.1f32; bytes.len()];
        let b = p.boundaries(bytes, &entropies);
        assert_eq!(b.first().copied(), Some(0));
        assert_eq!(b.last().copied(), Some(bytes.len()));
    }

    #[test]
    fn global_threshold_below_theta_yields_one_patch() {
        // Disable newline reset so we can isolate the entropy rule.
        let p = EntropyPatcher::new(PatchMode::GlobalThreshold(5.0))
            .without_newline_reset();
        let bytes = b"abcdefgh";
        let entropies = vec![1.0f32; bytes.len()];
        let b = p.boundaries(bytes, &entropies);
        assert_eq!(b, vec![0, bytes.len()]);
        assert_eq!(p.n_patches(bytes, &entropies), 1);
    }

    #[test]
    fn global_threshold_fires_on_high_entropy_byte() {
        let p = EntropyPatcher::new(PatchMode::GlobalThreshold(2.0))
            .without_newline_reset();
        let bytes = b"abcdef";
        // Spike entropies at positions 2 and 4.
        let entropies = vec![1.0, 1.0, 3.0, 1.0, 3.0, 1.0];
        let b = p.boundaries(bytes, &entropies);
        // Boundaries at positions 2 and 4 ⇒ patches [0..2], [2..4], [4..6].
        assert_eq!(b, vec![0, 2, 4, 6]);
    }

    #[test]
    fn approx_monotonic_fires_on_jump() {
        let p = EntropyPatcher::new(PatchMode::ApproxMonotonic(0.5))
            .without_newline_reset();
        let bytes = b"abcdef";
        // Jumps: 1.0 → 2.0 (Δ=1.0, fires), 2.0 → 2.1 (Δ=0.1, no),
        // 2.1 → 4.0 (Δ=1.9, fires), 4.0 → 3.0 (Δ=-1.0, no),
        // 3.0 → 3.1 (Δ=0.1, no).
        let entropies = vec![1.0, 2.0, 2.1, 4.0, 3.0, 3.1];
        let b = p.boundaries(bytes, &entropies);
        assert_eq!(b, vec![0, 1, 3, 6]);
    }

    #[test]
    fn approx_monotonic_does_not_fire_on_smooth_increase() {
        let p = EntropyPatcher::new(PatchMode::ApproxMonotonic(1.0))
            .without_newline_reset();
        let bytes = b"abcdef";
        // Each delta is 0.5 — under threshold 1.0.
        let entropies = vec![1.0, 1.5, 2.0, 2.5, 3.0, 3.5];
        let b = p.boundaries(bytes, &entropies);
        assert_eq!(b, vec![0, bytes.len()]);
    }

    #[test]
    fn newline_reset_forces_boundary_after_newline() {
        let p = EntropyPatcher::new(PatchMode::GlobalThreshold(99.0));
        // Threshold so high it never fires by entropy — only newline
        // resets can produce boundaries.
        let bytes = b"abc\ndef\nghi";
        let entropies = vec![0.1f32; bytes.len()];
        let b = p.boundaries(bytes, &entropies);
        // Newlines at indices 3 and 7 ⇒ breaks at 4 and 8.
        assert_eq!(b, vec![0, 4, 8, bytes.len()]);
    }

    #[test]
    fn newline_reset_disabled_keeps_one_patch_with_high_threshold() {
        let p = EntropyPatcher::new(PatchMode::GlobalThreshold(99.0))
            .without_newline_reset();
        let bytes = b"abc\ndef\nghi";
        let entropies = vec![0.1f32; bytes.len()];
        let b = p.boundaries(bytes, &entropies);
        assert_eq!(b, vec![0, bytes.len()]);
    }

    #[test]
    fn no_duplicate_boundaries_when_newline_and_entropy_coincide() {
        // Newline at index 2; spike at index 3. Both rules want a
        // break at index 3 — must produce only one boundary.
        let p = EntropyPatcher::new(PatchMode::GlobalThreshold(2.0));
        let bytes = b"ab\ncd";
        let entropies = vec![0.1, 0.1, 0.1, 5.0, 0.1];
        let b = p.boundaries(bytes, &entropies);
        // Newline at index 2 ⇒ break at 3. Entropy spike at 3 also
        // wants break at 3. Must be one boundary at 3.
        assert_eq!(b, vec![0, 3, bytes.len()]);
    }

    #[test]
    fn boundaries_are_strictly_increasing() {
        let p = EntropyPatcher::new(PatchMode::GlobalThreshold(0.5));
        let bytes = b"a\nb\nc\nd\ne\nf";
        // Force lots of newlines + some spikes.
        let mut entropies = vec![0.1f32; bytes.len()];
        entropies[5] = 1.0;
        entropies[7] = 1.0;
        let b = p.boundaries(bytes, &entropies);
        for w in b.windows(2) {
            assert!(w[0] < w[1], "boundaries not strictly increasing: {b:?}");
        }
        assert_eq!(b.first().copied(), Some(0));
        assert_eq!(b.last().copied(), Some(bytes.len()));
    }

    #[test]
    fn covers_all_bytes_no_gaps() {
        let p = EntropyPatcher::new(PatchMode::ApproxMonotonic(0.5));
        let bytes = b"the quick brown\nfox jumps";
        let entropies = (0..bytes.len()).map(|i| (i as f32 * 0.3) % 2.0).collect::<Vec<_>>();
        let b = p.boundaries(bytes, &entropies);
        // Patches must tile [0, bytes.len()) with no overlap and no gap.
        let mut covered = 0usize;
        for w in b.windows(2) {
            assert!(w[1] > w[0]);
            covered += w[1] - w[0];
        }
        assert_eq!(covered, bytes.len());
    }
}
