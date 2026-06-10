//! Typed RMSNorm on Tensor<D> (I2) — known-value + weight-scaling checks.

use modgrad_device::backend::tensor::{rms_norm, Cpu, Tensor};

#[test]
fn rms_norm_normalizes_to_unit_rms() {
    // x = [2,2,2,2], weight=1, eps=0 → RMS = 2, y = x/2 = [1,1,1,1].
    let x = vec![2.0f32; 4];
    let w = vec![1.0f32; 4];
    let xt = Tensor::<Cpu>::from_slice(&x).unwrap();
    let wt = Tensor::<Cpu>::from_slice(&w).unwrap();
    let mut out = Tensor::<Cpu>::zeros(4).unwrap();
    rms_norm(&xt, &wt, &mut out, 1, 4, 0.0).unwrap();
    for v in out.to_vec().unwrap() {
        assert!((v - 1.0).abs() < 1e-5, "expected unit RMS, got {v}");
    }
}

#[test]
fn rms_norm_applies_weight_and_rows() {
    // Two rows, per-column weight. Validate against the definition directly.
    let n_rows = 2;
    let n_cols = 4;
    let eps = 1e-6f32;
    let x: Vec<f32> = (0..n_rows * n_cols).map(|i| (i as f32 - 3.0) * 0.5).collect();
    let w: Vec<f32> = (0..n_cols).map(|j| 1.0 + 0.25 * j as f32).collect();
    let xt = Tensor::<Cpu>::from_slice(&x).unwrap();
    let wt = Tensor::<Cpu>::from_slice(&w).unwrap();
    let mut out = Tensor::<Cpu>::zeros(n_rows * n_cols).unwrap();
    rms_norm(&xt, &wt, &mut out, n_rows, n_cols, eps).unwrap();
    let o = out.to_vec().unwrap();
    for r in 0..n_rows {
        let row = &x[r * n_cols..(r + 1) * n_cols];
        let ms = row.iter().map(|v| v * v).sum::<f32>() / n_cols as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for j in 0..n_cols {
            let expect = row[j] * inv * w[j];
            assert!((o[r * n_cols + j] - expect).abs() < 1e-5,
                "row {r} col {j}: got {} expected {expect}", o[r * n_cols + j]);
        }
    }
}
