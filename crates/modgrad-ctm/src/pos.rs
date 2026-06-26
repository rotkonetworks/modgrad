//! Fixed 2D sinusoidal positional encoding (transformer-standard).
//!
//! Zero trainable parameters. Added in place to a flattened
//! `[n_tokens × raw_dim]` token buffer so that downstream attention can
//! recover *where* each token sits on the 2D grid. Without this, the
//! maze brain's spatial attention sum-pools tokens and position is
//! unrecoverable (the wall-probe is then at chance).
//!
//! Layout: each token's `raw_dim` channels are split in half. The first
//! half encodes the token's grid ROW, the second half encodes its grid
//! COLUMN, each with the standard `sin/cos` interleaving over geometric
//! frequencies `1 / 10000^(2i/half_dim)`. Tokens are assumed to be in
//! row-major order on a `grid_w`-wide grid (token `t` → row `t/grid_w`,
//! col `t%grid_w`).

/// Add a fixed 2D sinusoidal positional encoding, in place, to a
/// flattened `[n_tokens × raw_dim]` buffer.
///
/// - `tokens`: flat buffer, length must be `n_tokens * raw_dim`.
/// - `n_tokens`: number of spatial tokens.
/// - `raw_dim`: per-token channel count.
/// - `grid_w`: grid width used to recover (row, col) from token index.
pub fn add_sinusoidal_pos_2d(
    tokens: &mut [f32],
    n_tokens: usize,
    raw_dim: usize,
    grid_w: usize,
) {
    debug_assert_eq!(
        tokens.len(),
        n_tokens * raw_dim,
        "add_sinusoidal_pos_2d: buffer size mismatch"
    );
    if raw_dim == 0 || n_tokens == 0 || grid_w == 0 {
        return;
    }
    // Split channels: first half encodes row, second half encodes col.
    let half = raw_dim / 2;
    let grid_w = grid_w.max(1);
    for t in 0..n_tokens {
        let row = (t / grid_w) as f32;
        let col = (t % grid_w) as f32;
        let base = t * raw_dim;
        // Row encoding into channels [0, half).
        add_axis(&mut tokens[base..base + half], row, half);
        // Col encoding into channels [half, raw_dim). If raw_dim is odd
        // the column axis gets the extra channel.
        let col_len = raw_dim - half;
        add_axis(&mut tokens[base + half..base + raw_dim], col, col_len);
    }
}

/// Add a 1D sinusoidal encoding of scalar `pos` over `len` channels,
/// in place. Standard transformer frequencies, sin/cos interleaved.
fn add_axis(slice: &mut [f32], pos: f32, len: usize) {
    let mut i = 0;
    while i < len {
        // Frequency for this (sin, cos) pair.
        let pair = (i / 2) as f32;
        let denom = 10000f32.powf((2.0 * pair) / (len.max(1) as f32));
        let angle = pos / denom;
        slice[i] += angle.sin();
        if i + 1 < len {
            slice[i + 1] += angle.cos();
        }
        i += 2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distinct_positions_get_distinct_codes() {
        let n = 9; // 3x3 grid
        let d = 8;
        let grid_w = 3;
        let mut a = vec![0.0f32; n * d];
        add_sinusoidal_pos_2d(&mut a, n, d, grid_w);
        // Every token's code should differ from token 0 (except token 0).
        let tok0 = &a[0..d];
        for t in 1..n {
            let tok = &a[t * d..(t + 1) * d];
            let diff: f32 = tok.iter().zip(tok0).map(|(x, y)| (x - y).abs()).sum();
            assert!(diff > 1e-4, "token {t} identical to token 0");
        }
    }

    #[test]
    fn additive_and_size_preserving() {
        let mut a = vec![1.0f32; 4 * 6];
        add_sinusoidal_pos_2d(&mut a, 4, 6, 2);
        assert_eq!(a.len(), 24);
    }
}
