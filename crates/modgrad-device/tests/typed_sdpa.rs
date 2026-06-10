//! Scaled dot-product attention (I3) — causal + GQA + uniform-weight invariants.

use modgrad_device::backend::tensor::{sdpa, Cpu, Tensor};

#[test]
fn sdpa_token0_attends_only_to_itself() {
    // Causal: token 0 sees only token 0 → output row 0 == v row 0, for any q/k.
    let n_tokens = 3;
    let head_dim = 2;
    let q: Vec<f32> = (0..n_tokens * head_dim).map(|i| i as f32 * 0.3).collect();
    let k: Vec<f32> = (0..n_tokens * head_dim).map(|i| (i as f32 * 0.2).cos()).collect();
    let v: Vec<f32> = (0..n_tokens * head_dim).map(|i| i as f32 + 1.0).collect();
    let qt = Tensor::<Cpu>::from_slice(&q).unwrap();
    let kt = Tensor::<Cpu>::from_slice(&k).unwrap();
    let vt = Tensor::<Cpu>::from_slice(&v).unwrap();
    let out = sdpa(&qt, &kt, &vt, n_tokens, 1, 1, head_dim, None, None).unwrap();
    let o = out.to_vec().unwrap();
    assert!((o[0] - v[0]).abs() < 1e-6 && (o[1] - v[1]).abs() < 1e-6,
        "token 0 output {:?} != v row 0 {:?}", &o[0..2], &v[0..2]);
}

#[test]
fn sdpa_uniform_scores_average_values() {
    // If all keys are equal, every score is equal → token 1 attends uniformly
    // to {0,1} → output row 1 == mean(v row 0, v row 1).
    let n_tokens = 2;
    let head_dim = 2;
    let q = vec![0.5f32, 0.5, 0.5, 0.5];
    let k = vec![1.0f32, 1.0, 1.0, 1.0]; // identical keys ⇒ identical scores
    let v = vec![2.0f32, 4.0, 6.0, 10.0];
    let qt = Tensor::<Cpu>::from_slice(&q).unwrap();
    let kt = Tensor::<Cpu>::from_slice(&k).unwrap();
    let vt = Tensor::<Cpu>::from_slice(&v).unwrap();
    let o = sdpa(&qt, &kt, &vt, n_tokens, 1, 1, head_dim, None, None).unwrap().to_vec().unwrap();
    // row 1 = mean of v rows: ([2,4]+[6,10])/2 = [4,7].
    assert!((o[2] - 4.0).abs() < 1e-5 && (o[3] - 7.0).abs() < 1e-5, "row1 = {:?}", &o[2..4]);
}

#[test]
fn sdpa_gqa_shares_kv_heads() {
    // 2 query heads, 1 kv head: both query heads read the same K/V. Runs +
    // produces the right shape (sanity for GQA index math).
    let n_tokens = 2;
    let head_dim = 2;
    let n_heads = 2;
    let n_kv = 1;
    let q = vec![0.1f32; n_tokens * n_heads * head_dim];
    let k = vec![0.2f32; n_tokens * n_kv * head_dim];
    let v: Vec<f32> = (0..n_tokens * n_kv * head_dim).map(|i| i as f32).collect();
    let qt = Tensor::<Cpu>::from_slice(&q).unwrap();
    let kt = Tensor::<Cpu>::from_slice(&k).unwrap();
    let vt = Tensor::<Cpu>::from_slice(&v).unwrap();
    let o = sdpa(&qt, &kt, &vt, n_tokens, n_heads, n_kv, head_dim, None, None).unwrap().to_vec().unwrap();
    assert_eq!(o.len(), n_tokens * n_heads * head_dim);
    assert!(o.iter().all(|x| x.is_finite()));
    // token 0, both heads attend only to kv token 0 ⇒ == v row 0 = [0,1].
    assert!((o[0] - 0.0).abs() < 1e-6 && (o[1] - 1.0).abs() < 1e-6, "head0 t0 {:?}", &o[0..2]);
    assert!((o[2] - 0.0).abs() < 1e-6 && (o[3] - 1.0).abs() < 1e-6, "head1 t0 {:?}", &o[2..4]);
}
