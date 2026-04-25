//! Does modgrad's hebbian_update actually produce Gabor-like filters?
//!
//! Load ~2000 grayscale tiny-imagenet images (64×64), run sparse-coding
//! Hebbian updates against a fresh Conv2d layer, dump filters as a PGM
//! grid. If we see oriented edge detectors the rule works; if we see
//! noise or collapse the rule (or our use of it) is broken.
//!
//! Run:
//!   python3 examples/hebbian_sanity/preprocess.py
//!   cargo run -p hebbian_sanity --release

use modgrad_codec::retina::Conv2d;
use std::fs::File;
use std::io::{Read, Write};

const BIN: &str = "/tmp/hebbian_sanity_images.bin";
const OUT_PGM: &str = "/tmp/hebbian_sanity_filters.pgm";

const N_FILTERS: usize = 64;
const KERNEL: usize = 8;
const EPOCHS: usize = 3;
const LR: f32 = 0.0002;
const SPARSITY_K: usize = 20;

fn load_images() -> (Vec<f32>, usize, usize, usize) {
    let mut f = File::open(BIN).expect("run preprocess.py first");
    let mut hdr = [0u8; 12];
    f.read_exact(&mut hdr).unwrap();
    let n = u32::from_le_bytes(hdr[0..4].try_into().unwrap()) as usize;
    let h = u32::from_le_bytes(hdr[4..8].try_into().unwrap()) as usize;
    let w = u32::from_le_bytes(hdr[8..12].try_into().unwrap()) as usize;
    let mut raw = vec![0u8; n * h * w * 4];
    f.read_exact(&mut raw).unwrap();
    let data: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    (data, n, h, w)
}

/// L2 norm of a slice.
fn l2(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Orientation-selectivity proxy: per filter, compute gradient along
/// +x and +y; a Gabor-like filter has strong anisotropy (|gx| very
/// different from |gy| at some orientation after rotating), an
/// isotropic / random filter has |gx| ≈ |gy|. We use the cheap proxy
/// |<|gx|> - <|gy|>| / (<|gx|> + <|gy|>) averaged over all kernel
/// positions. Near 0 = isotropic, near 1 = oriented.
fn orientation_score(filter: &[f32], k: usize) -> f32 {
    let mut sgx = 0.0f32;
    let mut sgy = 0.0f32;
    let mut n = 0.0f32;
    for y in 0..k {
        for x in 0..k {
            let c = filter[y * k + x];
            if x + 1 < k {
                sgx += (filter[y * k + x + 1] - c).abs();
            }
            if y + 1 < k {
                sgy += (filter[(y + 1) * k + x] - c).abs();
            }
            n += 1.0;
        }
    }
    let gx = sgx / n;
    let gy = sgy / n;
    if gx + gy < 1e-9 { return 0.0; }
    (gx - gy).abs() / (gx + gy)
}

/// Write the filter bank as a PGM grid. Each filter is 8×8 bumped to
/// upscale × 8 pixels with nearest-neighbor so it's eyeball-readable.
fn save_filters_pgm(conv: &Conv2d, path: &str, upscale: usize) {
    let k = conv.kernel_size;
    let n = conv.out_channels;
    let cols = (n as f32).sqrt().ceil() as usize;
    let rows = (n + cols - 1) / cols;
    let cell = k * upscale;
    let pad = 2;
    let width = cols * (cell + pad) + pad;
    let height = rows * (cell + pad) + pad;
    let mut img = vec![128u8; width * height];

    for f in 0..n {
        let filt = &conv.weight[f * k * k..(f + 1) * k * k];
        let mn = filt.iter().cloned().fold(f32::INFINITY, f32::min);
        let mx = filt.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let span = (mx - mn).max(1e-9);
        let r = f / cols;
        let c = f % cols;
        let ox = pad + c * (cell + pad);
        let oy = pad + r * (cell + pad);
        for fy in 0..k {
            for fx in 0..k {
                let v = filt[fy * k + fx];
                let norm = ((v - mn) / span * 255.0).clamp(0.0, 255.0) as u8;
                for dy in 0..upscale {
                    for dx in 0..upscale {
                        let px = ox + fx * upscale + dx;
                        let py = oy + fy * upscale + dy;
                        img[py * width + px] = norm;
                    }
                }
            }
        }
    }

    let mut f = File::create(path).unwrap();
    writeln!(f, "P5").unwrap();
    writeln!(f, "{} {}", width, height).unwrap();
    writeln!(f, "255").unwrap();
    f.write_all(&img).unwrap();
}

fn main() {
    let (data, n_imgs, h, w) = load_images();
    // Use f64 accumulator: f32 saturates around sum=2^24 ≈ 16.7M and
    // silently drops further additions, which on 142M values yields a
    // wildly wrong mean. Classic de-Valence "precision matters" trap.
    let sum_f64: f64 = data.iter().map(|&x| x as f64).sum();
    let mean = (sum_f64 / data.len() as f64) as f32;
    let var_f64: f64 = data.iter().map(|&x| { let d = x as f64 - mean as f64; d * d }).sum::<f64>() / data.len() as f64;
    let std = var_f64.sqrt() as f32;
    println!("loaded {n_imgs} images of {h}×{w} (mean={mean:.3}, std={std:.3})");

    let mut conv = Conv2d::new(1, N_FILTERS, KERNEL, 2, 0);
    let w0_norm = l2(&conv.weight);
    let score0: f32 = (0..N_FILTERS)
        .map(|i| orientation_score(&conv.weight[i * KERNEL * KERNEL..(i + 1) * KERNEL * KERNEL], KERNEL))
        .sum::<f32>() / N_FILTERS as f32;
    println!("init: |W|₂={w0_norm:.3}, mean orientation score={score0:.3}");

    for epoch in 0..EPOCHS {
        let w_before = conv.weight.clone();
        for img_idx in 0..n_imgs {
            let img = &data[img_idx * h * w..(img_idx + 1) * h * w];
            conv.hebbian_update(img, 1, h, w, LR, SPARSITY_K);
        }
        let w_now_norm = l2(&conv.weight);
        let delta: f32 = conv.weight.iter().zip(&w_before)
            .map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();
        let score_now: f32 = (0..N_FILTERS)
            .map(|i| orientation_score(&conv.weight[i * KERNEL * KERNEL..(i + 1) * KERNEL * KERNEL], KERNEL))
            .sum::<f32>() / N_FILTERS as f32;
        let has_nan = conv.weight.iter().any(|x| !x.is_finite());
        println!("epoch {epoch}: |W|₂={w_now_norm:.3}, Δ={delta:.3}, orient={score_now:.3}, nan={has_nan}");
    }

    // Final per-filter orientation histogram + per-filter norm dump —
    // want to know: of the filters that survived, what's the
    // distribution? (helps distinguish "all filters died" from
    // "some filters are Gabor-like, others near-init").
    let mut oriented = 0usize;
    let mut dead = 0usize;
    let mut near_unit = 0usize;
    let mut norms = Vec::with_capacity(N_FILTERS);
    for f in 0..N_FILTERS {
        let filt = &conv.weight[f * KERNEL * KERNEL..(f + 1) * KERNEL * KERNEL];
        let s = orientation_score(filt, KERNEL);
        let energy = l2(filt);
        norms.push(energy);
        if energy < 1e-3 { dead += 1; }
        else if s > 0.3 { oriented += 1; }
        if (energy - 1.0).abs() < 0.05 { near_unit += 1; }
    }
    norms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let nmin = norms[0];
    let nmed = norms[N_FILTERS / 2];
    let nmax = norms[N_FILTERS - 1];
    println!(
        "final: {oriented}/{N_FILTERS} oriented (score>0.3), {dead}/{N_FILTERS} dead (|f|<1e-3), \
         {near_unit}/{N_FILTERS} near-unit norm"
    );
    println!("filter norms: min={nmin:.3}, median={nmed:.3}, max={nmax:.3}");

    save_filters_pgm(&conv, OUT_PGM, 4);
    println!("wrote {OUT_PGM} — `feh {OUT_PGM}` or `display {OUT_PGM}` to eyeball it");
}
