//! Hebbian / sparse-coding pretraining of `VisualCortex`'s V2/V4 layers
//! on STL-10 unlabeled natural images.
//!
//! # Why
//!
//! The retina's V2/V4 learn by sparse-coding reconstruction (see
//! `Conv2d::hebbian_update`). That objective needs natural-image
//! statistics to converge to Gabor-like / shape-like filters — train
//! it on maze pixels (3 discrete values, no texture) and it collapses
//! to gray mush. STL-10's unlabeled split (100k × 96×96 RGB, ~2.6 GB)
//! is purpose-built for this kind of pretraining.
//!
//! # Usage
//!
//! ```text
//! # one-time: download the data (~2.6 GB)
//! wget https://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz
//! tar xzf stl10_binary.tar.gz
//!
//! # pretrain (default: 2 epochs, save to retina_stl10.bin)
//! cargo run --release --features rocm -p pretrain_retina -- \
//!     --data stl10_binary/unlabeled_X.bin \
//!     --out  retina_stl10.bin
//! ```
//!
//! `--features rocm` routes the conv matmuls to hipBLAS on ROCm-
//! capable hardware; without it everything runs on CPU. Expect
//! roughly 10–100× slowdown on CPU.
//!
//! # Output
//!
//! The saved file is a wincode-encoded `VisualCortex` including V1
//! (fixed retinal filters), V2 + V4 (learned cortical filters), and
//! receptor state. Load it in another binary via
//! `VisualCortex::load(path)` and, if the downstream input size
//! differs, set `input_h` / `input_w` on the loaded struct — Conv2d
//! weights depend only on channels and kernel size, so the same
//! pretrained weights work at any input resolution.

use std::path::PathBuf;
use std::time::Instant;

use modgrad_codec::retina::VisualCortex;
use modgrad_codec::stl10::Stl10UnlabeledReader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    let mut data_path: Option<PathBuf> = None;
    let mut out_path: PathBuf = PathBuf::from("retina_stl10.bin");
    let mut epochs: usize = 2;
    let mut lr: f32 = 2e-4;
    let mut sparsity_k: usize = 8;
    let mut seed: u64 = 42;
    let mut limit: Option<usize> = None;
    let mut log_every: usize = 5000;
    // Batch size: images per Hebbian update. Larger amortizes hipBLAS
    // dispatch overhead and sustains GPU load long enough for SMU to
    // boost. 64 is a good default for 8 GB VRAM; 128 still fits
    // comfortably. batch=1 reproduces the pre-batch behavior exactly.
    let mut batch_size: usize = 64;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--data" => { data_path = Some(PathBuf::from(&args[i + 1])); i += 2; }
            "--out" => { out_path = PathBuf::from(&args[i + 1]); i += 2; }
            "--epochs" => { epochs = args[i + 1].parse()?; i += 2; }
            "--lr" => { lr = args[i + 1].parse()?; i += 2; }
            "--sparsity-k" => { sparsity_k = args[i + 1].parse()?; i += 2; }
            "--seed" => { seed = args[i + 1].parse()?; i += 2; }
            "--limit" => { limit = Some(args[i + 1].parse()?); i += 2; }
            "--log-every" => { log_every = args[i + 1].parse()?; i += 2; }
            "--batch-size" => { batch_size = args[i + 1].parse()?; i += 2; }
            "--help" | "-h" => {
                print_usage();
                return Ok(());
            }
            _ => { i += 1; }
        }
    }

    let data_path = data_path.ok_or_else(|| -> Box<dyn std::error::Error> {
        "missing --data PATH (path to stl10_binary/unlabeled_X.bin). \
         Pass --help for setup instructions.".into()
    })?;

    // ── Load dataset ──────────────────────────────────────────
    let reader = Stl10UnlabeledReader::open(&data_path)?;
    let total = reader.len();
    let n_train = limit.map_or(total, |lim| lim.min(total));
    eprintln!("stl10: {} images available, training on {}", total, n_train);
    if n_train == 0 {
        return Err("no images to train on".into());
    }

    // ── Retina ────────────────────────────────────────────────
    //
    // `preserve_spatial` matches the geometry used at maze inference
    // (stride-1 on V2/V4), so the weights drop straight into the maze
    // retina with no reshape. Conv2d weights are stride-independent;
    // if later we want to reuse these on a different stride layout,
    // just set `stride` on the loaded retina's v2/v4 fields.
    let mut retina = VisualCortex::preserve_spatial(96, 96);
    eprintln!(
        "retina: V1(fixed) {}→{} k={}  V2(learn) {}→{} k={}  V4(learn) {}→{} k={}",
        retina.v1.in_channels, retina.v1.out_channels, retina.v1.kernel_size,
        retina.v2.in_channels, retina.v2.out_channels, retina.v2.kernel_size,
        retina.v4.in_channels, retina.v4.out_channels, retina.v4.kernel_size,
    );
    eprintln!("config: epochs={epochs} batch={batch_size} lr={lr} sparsity_k={sparsity_k} seed={seed}");

    // ── Training loop ─────────────────────────────────────────
    let t0 = Instant::now();
    let mut rng = seed.wrapping_mul(6364136223846793005) | 1;
    let img_bytes = 3 * Stl10UnlabeledReader::IMAGE_H * Stl10UnlabeledReader::IMAGE_W;

    for epoch in 0..epochs {
        let t_epoch = Instant::now();
        // Deterministic shuffle of 0..n_train (Fisher–Yates on xorshift64*).
        let mut order: Vec<usize> = (0..n_train).collect();
        for j in (1..order.len()).rev() {
            rng = xorshift64_star(rng);
            let k = (rng as usize) % (j + 1);
            order.swap(j, k);
        }

        let mut step = 0usize;
        while step < n_train {
            let batch_end = (step + batch_size).min(n_train);
            let actual_batch = batch_end - step;

            // Pack batch into contiguous buffer [actual_batch × 3 × H × W].
            let mut batch_buf = Vec::with_capacity(actual_batch * img_bytes);
            for &idx in &order[step..batch_end] {
                let img = reader.get(idx)?;
                batch_buf.extend_from_slice(&img);
            }

            retina.hebbian_step_batch(&batch_buf, actual_batch, lr, sparsity_k);

            step = batch_end;
            if step % log_every < batch_size {
                let rate = step as f64 / t_epoch.elapsed().as_secs_f64();
                eprintln!(
                    "  epoch {}/{}  step {}/{}  {:.1} img/s",
                    epoch + 1, epochs, step, n_train, rate,
                );
            }
        }

        let dt = t_epoch.elapsed();
        let rate = n_train as f64 / dt.as_secs_f64();
        eprintln!(
            "  epoch {}/{} done in {:.1}s  ({:.1} img/s)",
            epoch + 1, epochs, dt.as_secs_f64(), rate,
        );
    }

    let dt_total = t0.elapsed();
    eprintln!("training: {:.1}s total", dt_total.as_secs_f64());

    // ── Save ──────────────────────────────────────────────────
    retina.save(&out_path)?;
    eprintln!("saved: {}", out_path.display());
    Ok(())
}

fn print_usage() {
    eprintln!(
"Usage: pretrain_retina --data PATH [options]

Required:
  --data PATH          Path to stl10_binary/unlabeled_X.bin (~2.6 GB)

Options:
  --out PATH           Output weights file [retina_stl10.bin]
  --epochs N           Passes over the dataset [2]
  --lr F               Hebbian learning rate [2e-4]
  --sparsity-k K       Top-K winners per position [8]
  --seed N             Shuffle seed [42]
  --limit N            Train on first N images (fast smoke test)
  --log-every N        Progress log cadence in images [5000]
  --batch-size N       Images per Hebbian update [64]. Bigger = better
                       GPU utilization; ~64 fits comfortably in 8 GB.

Setup:
  wget https://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz
  tar xzf stl10_binary.tar.gz

Build with --features rocm for GPU dispatch on AMD ROCm hardware
(≥64-per-dim matmuls; smaller ops stay on CPU)."
    );
}

/// xorshift64* — tiny non-crypto RNG for shuffle ordering.
fn xorshift64_star(x: u64) -> u64 {
    let mut x = x;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    x.wrapping_mul(0x2545F4914F6CDD1D)
}
