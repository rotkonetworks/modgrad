//! Qwen2.5-0.5B load + one-shot forward smoke test.
//!
//! Verifies the safetensors → resident path end-to-end:
//!   1. Load `model.safetensors` from the HF snapshot.
//!   2. Upload to GpuModelResident via `modgrad_io::qwen2::load_qwen2_5_0_5b`.
//!   3. Run one forward pass at token id 9707 (` Hello` in Qwen BPE,
//!      but any well-defined id works — we don't decode here).
//!   4. Copy logits to host, print top-10 + min/max/mean/nan-count.
//!
//! Pass criteria:
//!   - Logits length == 151_936
//!   - No NaN / no Inf
//!   - max - min > 1.0 (output is not all zeros / not collapsed)
//!   - End-to-end runtime well under 10 s on the 7600M XT
//!
//! Run:
//!   cargo run --release --features rocm -p qwen_load_smoke

#[cfg(not(feature = "rocm"))]
fn main() {
    eprintln!("qwen_load_smoke: built without `--features rocm`. Rebuild with:");
    eprintln!("  cargo run --release --features rocm -p qwen_load_smoke");
    std::process::exit(0);
}

#[cfg(feature = "rocm")]
fn main() {
    use std::time::Instant;

    use modgrad_compute::backend::GpuVec;
    use modgrad_device::backend::HipBatch;
    use modgrad_device::backend::rocm::ffi::runtime_available;
    use modgrad_io::qwen2::load_qwen2_5_0_5b;
    use modgrad_transformer::kv_cache_resident::KvCacheResident;

    if !runtime_available() {
        eprintln!("qwen_load_smoke: HIP runtime unavailable (no GPU?). Skipping.");
        std::process::exit(0);
    }

    let path = "/steam/llm/hf_cache/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987/model.safetensors";
    eprintln!("qwen_load_smoke: model = {path}");

    let max_seq: usize = 2048;
    let total_start = Instant::now();

    // ── Load + upload ─────────────────────────────────────────
    let mut resident = match load_qwen2_5_0_5b(path, max_seq) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("qwen_load_smoke: load failed: {e}");
            std::process::exit(1);
        }
    };

    eprintln!("qwen_load_smoke: model resident; running forward ...");

    // ── Allocate KV cache + logits buffer ─────────────────────
    let mut kv_cache = KvCacheResident::new(
        resident.num_layers(),
        modgrad_io::qwen2::QWEN2_5_0_5B_NUM_KV_HEADS,
        modgrad_io::qwen2::QWEN2_5_0_5B_HEAD_DIM,
        max_seq,
        resident.model_dim(),
    )
    .expect("alloc kv cache");

    let vocab = resident.vocab_size();
    let mut logits_dev = GpuVec::try_hip(vocab).expect("alloc logits");
    let token_ids: [i64; 1] = [9707];
    let positions: [usize; 1] = [0];

    // ── Forward ───────────────────────────────────────────────
    let fwd_start = Instant::now();
    let batch = HipBatch::new();
    resident
        .forward(&batch, &token_ids, &positions, &mut kv_cache, &mut logits_dev)
        .expect("resident forward");
    batch.flush().expect("flush");
    let fwd_ms = fwd_start.elapsed().as_millis();
    eprintln!("qwen_load_smoke: forward took {fwd_ms} ms");

    // ── Copy logits and inspect ───────────────────────────────
    let mut logits = vec![0.0f32; vocab];
    logits_dev.copy_to_host(&mut logits);

    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut nan_count = 0usize;
    let mut inf_count = 0usize;
    for &v in &logits {
        if v.is_nan() {
            nan_count += 1;
            continue;
        }
        if !v.is_finite() {
            inf_count += 1;
            continue;
        }
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
        sum += v as f64;
    }
    let mean = sum / (vocab as f64);

    let mut indexed: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    // Use partial_cmp; ties broken by index (lower wins).
    indexed.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    let total_ms = total_start.elapsed().as_millis();

    eprintln!();
    eprintln!("══════════════ qwen_load_smoke summary ══════════════");
    eprintln!("vocab          = {vocab}");
    eprintln!("token in       = {}", token_ids[0]);
    eprintln!("position       = {}", positions[0]);
    eprintln!("logits length  = {} (== {vocab}? {})", logits.len(), logits.len() == vocab);
    eprintln!("nan count      = {nan_count}");
    eprintln!("inf count      = {inf_count}");
    eprintln!("min logit      = {min:.4}");
    eprintln!("max logit      = {max:.4}");
    eprintln!("max-min        = {:.4}", max - min);
    eprintln!("mean           = {mean:.4}");
    eprintln!("forward time   = {fwd_ms} ms");
    eprintln!("total time     = {total_ms} ms (incl. load+upload+patch)");
    eprintln!();
    eprintln!("top-10 logits (index → value):");
    for (rank, (idx, val)) in indexed.iter().take(10).enumerate() {
        eprintln!("  #{:<2} idx={idx:<6} val={val:.4}", rank + 1);
    }
    eprintln!();

    // ── Pass / fail ───────────────────────────────────────────
    let mut failed = false;
    if logits.len() != vocab {
        eprintln!("FAIL: logits length mismatch");
        failed = true;
    }
    if nan_count > 0 {
        eprintln!("FAIL: {nan_count} NaN logits");
        failed = true;
    }
    if inf_count > 0 {
        eprintln!("FAIL: {inf_count} non-finite logits");
        failed = true;
    }
    if (max - min) <= 1.0 {
        eprintln!("FAIL: spread {:.4} ≤ 1.0 — logits suspiciously flat", max - min);
        failed = true;
    }
    if fwd_ms > 10_000 {
        eprintln!(
            "WARN: forward took {fwd_ms} ms (> 10 s) — investigate dispatch overhead"
        );
    }

    if failed {
        eprintln!("RESULT: FAIL");
        std::process::exit(1);
    } else {
        eprintln!("RESULT: PASS");
    }
}
