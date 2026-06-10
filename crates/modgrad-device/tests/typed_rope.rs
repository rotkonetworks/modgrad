//! Typed RoPE on Tensor<D> (I2) — identity at pos 0, norm-preserving, known angle.

use modgrad_device::backend::tensor::{rope, Cpu, Tensor};

#[test]
fn rope_position_zero_is_identity() {
    let x: Vec<f32> = (0..8).map(|i| i as f32 * 0.3 - 1.0).collect();
    let xt = Tensor::<Cpu>::from_slice(&x).unwrap();
    let out = rope(&xt, &[0], 1, 1, 8, 10_000.0, None).unwrap();
    for (a, b) in out.to_vec().unwrap().iter().zip(&x) {
        assert!((a - b).abs() < 1e-6, "position 0 must be identity");
    }
}

#[test]
fn rope_preserves_norm() {
    // Rotation is orthogonal → ‖rope(x)‖ == ‖x‖.
    let x: Vec<f32> = (0..16).map(|i| (i as f32 * 0.7).sin()).collect();
    let xt = Tensor::<Cpu>::from_slice(&x).unwrap();
    let out = rope(&xt, &[5], 1, 2, 8, 10_000.0, None).unwrap(); // 2 heads × head_dim 8
    let n0: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    let n1: f32 = out.to_vec().unwrap().iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!((n0 - n1).abs() < 1e-4, "rope changed the norm: {n0} vs {n1}");
}

#[test]
fn rope_known_rotation() {
    // head_dim=2, 1 head, base=10000, pos=1, i=0: freq=1, θ=1; x=[1,0] → [cos1, sin1].
    let x = vec![1.0f32, 0.0];
    let xt = Tensor::<Cpu>::from_slice(&x).unwrap();
    let o = rope(&xt, &[1], 1, 1, 2, 10_000.0, None).unwrap().to_vec().unwrap();
    assert!((o[0] - 1.0f32.cos()).abs() < 1e-5, "got {}", o[0]);
    assert!((o[1] - 1.0f32.sin()).abs() < 1e-5, "got {}", o[1]);
}

#[test]
fn rope_freq_factor_freezes_dimension() {
    // A huge freq factor → ~zero angle → that pair is left ~unchanged.
    let x = vec![1.0f32, 0.5];
    let xt = Tensor::<Cpu>::from_slice(&x).unwrap();
    let ff = [1.0e30f32];
    let o = rope(&xt, &[7], 1, 1, 2, 10_000.0, Some(&ff)).unwrap().to_vec().unwrap();
    assert!((o[0] - 1.0).abs() < 1e-4 && (o[1] - 0.5).abs() < 1e-4, "frozen dim moved: {o:?}");
}
