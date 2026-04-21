//! Edge-of-chaos diagnostic for cortex / dream / retina grids.
//!
//! Measures whether a 2D activation grid is sitting at the "edge of
//! chaos" — the Gell-Mann & Lloyd effective-complexity regime where
//! patterns are simultaneously varied (high Shannon entropy) and
//! structurally compressible (low redundancy under a basic compressor):
//!
//!     C_eff = H(x) * (1 - compress_ratio(x))
//!
//! - Pure order:  H ≈ 0 → C_eff ≈ 0  (frozen / monoculture)
//! - Pure noise:  compress_ratio ≈ 1 → C_eff ≈ 0  (uniform-looking entropy)
//! - Edge-of-chaos: both terms non-trivial → C_eff peaks
//!
//! Cited chain:
//!   Langton 1990 ("computation at the edge of chaos")
//!   Gell-Mann & Lloyd 1996 ("effective complexity")
//!   Hoel 2021 cites these for the dream-as-regulariser hypothesis
//!   Berdica et al. 2026 ("evolving many worlds") uses the same
//!     formula to gate the PBT-NCA population selection
//!
//! No external compression crate dependency: we use run-length
//! encoding on an 8-bit quantised grid as the compressibility proxy.
//! RLE is not LZ77, but it captures the same "how much of the signal
//! is repeated neighbours" property cheaply and works well on the
//! kinds of grids modgrad produces (cortex activations after
//! normalisation, maze cell bitmaps, retina output magnitudes). If a
//! future caller needs sharper compressibility, a zlib-backed variant
//! can be added behind a `zlib` feature.

/// Normalised Shannon entropy of a 2D grid in [0, 1].
///
/// Quantises the input into 256 bins over `[min, max]`, then computes
/// `H = -Σ p(b) log2 p(b) / log2(256)`. Returns 0 on empty input or
/// single-value input (where every sample lands in the same bin).
pub fn shannon_entropy(values: &[f32]) -> f32 {
    if values.is_empty() { return 0.0; }
    let (mn, mx) = values.iter().fold((f32::INFINITY, f32::NEG_INFINITY),
        |(a, b), &v| (a.min(v), b.max(v)));
    let span = mx - mn;
    if span <= 1e-9 { return 0.0; }
    let mut hist = [0u32; 256];
    for &v in values {
        let b = ((v - mn) / span * 255.0).clamp(0.0, 255.0) as usize;
        hist[b] += 1;
    }
    let n = values.len() as f32;
    let mut h = 0.0f32;
    for &c in &hist {
        if c == 0 { continue; }
        let p = c as f32 / n;
        h -= p * p.log2();
    }
    // log2(256) = 8; normalise to [0, 1].
    h / 8.0
}

/// Run-length encoding ratio: compressed_bytes / raw_bytes, where
/// raw_bytes is the 8-bit quantised grid and compressed_bytes is
/// `(byte, run_length)` pairs (2 bytes each). A fully-constant grid
/// compresses to ~2 bytes per 65536 raw bytes → near-zero ratio.
/// Fully random bytes compress to ~2× raw (every byte becomes a
/// (byte, 1) pair) but we clamp the ratio to [0, 1] — we're measuring
/// "how much worse than uncompressed" as a [0, 1] fraction, which
/// matches the "1 - ratio" interpretation in `effective_complexity`.
///
/// Run-length cap at 255 per pair prevents integer overflow on
/// uniform grids and doesn't affect the ratio beyond a small constant.
pub fn rle_ratio(values: &[f32]) -> f32 {
    if values.is_empty() { return 0.0; }
    let (mn, mx) = values.iter().fold((f32::INFINITY, f32::NEG_INFINITY),
        |(a, b), &v| (a.min(v), b.max(v)));
    let span = (mx - mn).max(1e-9);
    let bytes: Vec<u8> = values.iter()
        .map(|&v| ((v - mn) / span * 255.0).clamp(0.0, 255.0) as u8)
        .collect();

    let mut compressed = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        let b = bytes[i];
        let mut run = 1usize;
        while i + run < bytes.len() && bytes[i + run] == b && run < 255 {
            run += 1;
        }
        compressed += 2;
        i += run;
    }

    let ratio = compressed as f32 / bytes.len().max(1) as f32;
    ratio.clamp(0.0, 1.0)
}

/// Gell-Mann & Lloyd-style effective complexity.
///
///     C_eff = H(x) * (1 - compress_ratio(x))
///
/// Returns a value in [0, 1]. Near 0 for both frozen and noisy
/// grids; peaks in the "edge of chaos" regime where the signal has
/// broad histogram but strong local redundancy.
pub fn effective_complexity(values: &[f32]) -> f32 {
    let h = shannon_entropy(values);
    let r = rle_ratio(values);
    (h * (1.0 - r)).clamp(0.0, 1.0)
}

/// Effective complexity on a discrete-label grid derived from a
/// multi-channel activation tensor `[channels × h × w]` by taking the
/// per-spatial winning channel (argmax-over-channels).
///
/// This is the natural EoC measurement for layered vision
/// activations: RLE on raw float channels gives uninformative
/// near-1.0 ratios because continuous values rarely repeat;
/// reducing to per-position winners gives a discrete label map
/// that actually has runs.
///
/// Matches the PBT-NCA paper's "winner-map entropy" style
/// behavioral descriptor and Langton 1990's edge-of-chaos regime
/// on cellular automata (discrete states per cell). Returns 0 when
/// inputs are ill-formed (length mismatch).
/// `(C_eff, unique_winner_count)` report for a multi-channel
/// activation tensor. The second element exposes whether a zero
/// `C_eff` means frozen monoculture (unique=1) or is saturated
/// some other way. Callers that don't need the count can drop it.
pub fn effective_complexity_argmax_detail(
    data: &[f32],
    channels: usize,
    h: usize,
    w: usize,
) -> (f32, usize) {
    if data.len() != channels * h * w || channels == 0 || h * w == 0 {
        return (0.0, 0);
    }
    let mut winners = vec![0u8; h * w];
    for idx in 0..h * w {
        let mut best_c = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for c in 0..channels {
            let v = data[c * h * w + idx];
            if v > best_v { best_v = v; best_c = c; }
        }
        winners[idx] = (best_c & 0xFF) as u8;
    }
    let mut hist = [0u32; 256];
    for &b in &winners { hist[b as usize] += 1; }
    let unique = hist.iter().filter(|&&c| c > 0).count();
    let n = winners.len() as f32;
    let mut h_raw = 0.0f32;
    for &c in &hist {
        if c == 0 { continue; }
        let p = c as f32 / n;
        h_raw -= p * p.log2();
    }
    let h_norm = (h_raw / (unique as f32).max(1.0).log2().max(1.0)).clamp(0.0, 1.0);
    let mut compressed = 0usize;
    let mut i = 0usize;
    while i < winners.len() {
        let b = winners[i];
        let mut run = 1usize;
        while i + run < winners.len() && winners[i + run] == b && run < 255 {
            run += 1;
        }
        compressed += 2;
        i += run;
    }
    let r = (compressed as f32 / winners.len().max(1) as f32).clamp(0.0, 1.0);
    let c_eff = (h_norm * (1.0 - r)).clamp(0.0, 1.0);
    (c_eff, unique)
}

pub fn effective_complexity_argmax(
    data: &[f32],
    channels: usize,
    h: usize,
    w: usize,
) -> f32 {
    if data.len() != channels * h * w || channels == 0 || h * w == 0 {
        return 0.0;
    }
    // Build per-position winning channel.
    let mut winners = vec![0u8; h * w];
    for idx in 0..h * w {
        let mut best_c = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for c in 0..channels {
            let v = data[c * h * w + idx];
            if v > best_v { best_v = v; best_c = c; }
        }
        // Narrow to u8 — at most 256 channels (retina V4 = 128). For
        // wider layers we clip with modulo, which is a behavioral
        // lossy-but-stable hash appropriate for a diagnostic.
        winners[idx] = (best_c & 0xFF) as u8;
    }
    // Entropy of the winner distribution.
    let mut hist = [0u32; 256];
    for &b in &winners { hist[b as usize] += 1; }
    let n = winners.len() as f32;
    let mut h_raw = 0.0f32;
    for &c in &hist {
        if c == 0 { continue; }
        let p = c as f32 / n;
        h_raw -= p * p.log2();
    }
    // Normalise entropy by log2(unique-channels-observed), not log2(256);
    // otherwise a 3-channel layer caps at ~log2(3)/8 ≈ 0.2 and always
    // looks "low entropy" even when it's varied across its own labels.
    let unique = hist.iter().filter(|&&c| c > 0).count().max(1) as f32;
    let h_norm = (h_raw / unique.log2().max(1.0)).clamp(0.0, 1.0);

    // RLE on the winners byte-array directly. Reuse internal logic
    // via a lightweight inline — we don't want to expose a `rle_bytes`
    // helper publicly.
    let mut compressed = 0usize;
    let mut i = 0usize;
    while i < winners.len() {
        let b = winners[i];
        let mut run = 1usize;
        while i + run < winners.len() && winners[i + run] == b && run < 255 {
            run += 1;
        }
        compressed += 2;
        i += run;
    }
    let r = (compressed as f32 / winners.len().max(1) as f32).clamp(0.0, 1.0);
    (h_norm * (1.0 - r)).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_grid_has_zero_complexity() {
        let g = vec![0.5f32; 1024];
        assert_eq!(shannon_entropy(&g), 0.0);
        // A fully-uniform grid RLEs to ceil(N/255) pairs × 2 bytes.
        // For N=1024 that's 5 pairs × 2 = 10 bytes / 1024 = ~0.01.
        assert!(rle_ratio(&g) < 0.05);
        assert_eq!(effective_complexity(&g), 0.0);
    }

    #[test]
    fn random_noise_has_low_complexity() {
        // Pseudo-random bytes uniformly across [0, 1].
        let mut s = 42u64;
        let g: Vec<f32> = (0..4096)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                (s >> 32) as f32 / u32::MAX as f32
            })
            .collect();
        let h = shannon_entropy(&g);
        let r = rle_ratio(&g);
        let c = effective_complexity(&g);
        // Noise has high H (close to 1) and RLE cannot compress it:
        // most runs are length 1, so compressed ≈ 2 × N. Ratio clamps
        // at 1.0 → c = H × 0 = 0. Low complexity confirmed.
        assert!(h > 0.9, "random bytes should have near-maximal entropy, got {h}");
        assert!(r > 0.9, "random bytes shouldn't compress via RLE, got {r}");
        assert!(c < 0.1, "random noise should have low effective complexity, got {c}");
    }

    #[test]
    fn argmax_frozen_monoculture_has_zero_complexity() {
        // 4 channels, 8×8 grid; channel 0 wins everywhere.
        let (ch, h, w) = (4, 8, 8);
        let mut data = vec![0.0f32; ch * h * w];
        for idx in 0..h * w { data[idx] = 1.0; } // ch 0 > others
        assert_eq!(effective_complexity_argmax(&data, ch, h, w), 0.0);
    }

    #[test]
    fn argmax_edge_of_chaos_is_nonzero() {
        // 4 channels, 8×8 grid with striped winner pattern:
        // channel (idx / 4) % 4 wins each cell → runs of length 4, all
        // 4 labels used.
        let (ch, h, w) = (4, 8, 8);
        let mut data = vec![0.0f32; ch * h * w];
        for idx in 0..h * w {
            let winner = (idx / 4) % 4;
            data[winner * h * w + idx] = 1.0;
        }
        let c = effective_complexity_argmax(&data, ch, h, w);
        assert!(c > 0.1,
            "striped winner pattern should have non-trivial C_eff, got {c}");
    }

    #[test]
    fn structured_pattern_has_mid_complexity() {
        // Repeating stripes — broad histogram, strong local redundancy.
        let g: Vec<f32> = (0..4096)
            .map(|i| if (i / 16) % 2 == 0 { 0.1 } else { 0.9 })
            .collect();
        let c = effective_complexity(&g);
        // Stripes: H is low-to-medium (only two distinct bins), but
        // RLE compresses extremely well (~1% of raw). So C_eff sits
        // in the middle region — non-zero but not saturating.
        assert!(c > 0.05 && c < 0.5,
            "striped pattern should have mid complexity, got {c}");
    }
}
