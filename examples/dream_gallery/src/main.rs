//! Visualise what a `VisualCortex` dreams — adjoint-projected samples
//! from sparse V4 noise, rendered as PPM images.
//!
//! Two modes run by default, side-by-side:
//!   - `sober/`    — a fresh retina with random V2/V4 (the baseline
//!                   cortex; dreams are statistics of the random init).
//!   - `lsd_0.7/`  — the same retina after a single `lsd()` trip on
//!                   a bank of real maze pixels (integration=0.7, the
//!                   sweep-validated regime).
//!
//! Each sub-directory holds N per-image PPM files, an `NxN` montage
//! grid PPM, and a `meta.txt` provenance block (cortex type, seeds,
//! config, per-dream summary stats). Provenance lives in data, not
//! types — fine for a research example. If two runs of the gallery
//! are invoked with the same args, output bytes must be identical.
//!
//! Usage:
//!   cargo run -p dream_gallery --release
//!   cargo run -p dream_gallery --release -- --count 100 --size 21
//!   cargo run -p dream_gallery --release -- --sparsity-k 16 --seed 9
//!
//! Output default: /tmp/dream_gallery/{sober,lsd_0.7,summary.txt}

use modgrad_codec::retina::{LsdConfig, VisualCortex};
use std::io::Write;
use std::path::Path;

// ────────────────────────────────────────────────────────────────
// PPM writing — P6, 8-bit RGB. No dep; 40 lines.
// ────────────────────────────────────────────────────────────────

/// Write a `[3 × h × w]` float buffer as a P6 PPM file.
/// Per-image min-max normalised to [0,255]. No gamma correction.
fn write_ppm(path: &Path, pixels: &[f32], h: usize, w: usize) -> std::io::Result<()> {
    assert_eq!(pixels.len(), 3 * h * w);
    let (mn, mx) = pixels.iter().fold((f32::INFINITY, f32::NEG_INFINITY),
        |(a, b), &v| (a.min(v), b.max(v)));
    let span = (mx - mn).max(1e-6);
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "P6\n{w} {h}\n255")?;
    let mut rgb = vec![0u8; 3 * h * w];
    // Source layout is CHW (channels × h × w). PPM wants interleaved RGB.
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let v = pixels[c * h * w + y * w + x];
                let n = ((v - mn) / span * 255.0).clamp(0.0, 255.0) as u8;
                rgb[(y * w + x) * 3 + c] = n;
            }
        }
    }
    f.write_all(&rgb)?;
    Ok(())
}

/// Write an m×m montage of count images into a single PPM.
fn write_grid_ppm(path: &Path, dreams: &[Vec<f32>], h: usize, w: usize) -> std::io::Result<()> {
    let m = (dreams.len() as f32).sqrt().ceil() as usize;
    let gh = m * h;
    let gw = m * w;
    let mut grid = vec![0.0f32; 3 * gh * gw];
    for (i, d) in dreams.iter().enumerate() {
        let row = i / m;
        let col = i % m;
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    let src = c * h * w + y * w + x;
                    let dst = c * gh * gw + (row * h + y) * gw + (col * w + x);
                    grid[dst] = d[src];
                }
            }
        }
    }
    write_ppm(path, &grid, gh, gw)
}

// ────────────────────────────────────────────────────────────────
// Dream stats — compute, don't pretend to cite biology.
// ────────────────────────────────────────────────────────────────

struct DreamStats {
    mean: f32,
    std: f32,
    energy: f32,
    min: f32,
    max: f32,
}

fn stats(pixels: &[f32]) -> DreamStats {
    let n = pixels.len() as f32;
    let mean = pixels.iter().sum::<f32>() / n;
    let var = pixels.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    let std = var.sqrt();
    let energy = pixels.iter().map(|x| x * x).sum::<f32>() / n;
    let (mn, mx) = pixels.iter().fold((f32::INFINITY, f32::NEG_INFINITY),
        |(a, b), &v| (a.min(v), b.max(v)));
    DreamStats { mean, std, energy, min: mn, max: mx }
}

// ────────────────────────────────────────────────────────────────
// Pixel buffer for the LSD-pretrain path.
// Deterministic checkerboard-ish mazes without pulling in the maze
// generator crate — the goal is to give V2/V4 *some* structured input
// distribution to shape from, not to reproduce the maze benchmark.
// ────────────────────────────────────────────────────────────────

// Deterministic LCG used by the maze generator and the top-K seeder.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Self(seed.wrapping_add(1)) }
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn range(&mut self, n: usize) -> usize { (self.next_u32() as usize) % n.max(1) }
    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() { v.swap(i, self.range(i + 1)); }
    }
}

/// Recursive-backtracker maze on an odd-side grid. Returns the cell
/// bitmap (row-major, `true` = path, `false` = wall).
fn generate_maze_cells(size: usize, seed: u64) -> Vec<bool> {
    assert!(size % 2 == 1 && size >= 5);
    let mut cells = vec![false; size * size];
    let mut rng = Rng::new(seed);
    let mut stack = vec![(1usize, 1usize)];
    cells[size + 1] = true;
    while let Some(&(y, x)) = stack.last() {
        let mut dirs = [(0isize, -2isize), (0, 2), (-2, 0), (2, 0)];
        rng.shuffle(&mut dirs);
        let mut carved = false;
        for &(dy, dx) in &dirs {
            let ny = y as isize + dy;
            let nx = x as isize + dx;
            if ny < 1 || nx < 1 || ny >= (size - 1) as isize || nx >= (size - 1) as isize { continue; }
            let (ny, nx) = (ny as usize, nx as usize);
            if !cells[ny * size + nx] {
                cells[ny * size + nx] = true;
                let my = (y as isize + dy / 2) as usize;
                let mx = (x as isize + dx / 2) as usize;
                cells[my * size + mx] = true;
                stack.push((ny, nx));
                carved = true;
                break;
            }
        }
        if !carved { stack.pop(); }
    }
    cells
}

/// Build a bank of real mazes as [3 × size × size] pixel tensors
/// (wall = 0.1, path = 0.9). This replaces the old synthetic-noise
/// bank — prior versions trained the cortex on pixels biased to 0/1
/// but otherwise random, which shaped V2/V4 on non-maze statistics.
/// Now the bank is actual recursive-backtracker mazes, so any
/// Hebbian/LSD result is grounded in maze-structure learning.
fn maze_pixel_bank(n: usize, size: usize, seed: u64) -> Vec<Vec<f32>> {
    (0..n).map(|i| {
        let s = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(0xBED_CAFE)
            .wrapping_add(i as u64);
        let cells = generate_maze_cells(size, s);
        let mut pixels = vec![0.0f32; 3 * size * size];
        for idx in 0..cells.len() {
            let v = if cells[idx] { 0.9 } else { 0.1 };
            for c in 0..3 { pixels[c * size * size + idx] = v; }
        }
        pixels
    }).collect()
}

// ────────────────────────────────────────────────────────────────
// Gallery: render N dreams from a retina, write images + meta.
// ────────────────────────────────────────────────────────────────

struct Gallery<'a> {
    label: &'a str,
    retina: &'a VisualCortex,
    count: usize,
    base_seed: u64,
    sparsity_k: usize,
    out_dir: &'a Path,
}

fn run_gallery(g: &Gallery) -> std::io::Result<Vec<DreamStats>> {
    std::fs::create_dir_all(g.out_dir)?;
    let (h, w) = (g.retina.input_h, g.retina.input_w);
    let mut dreams = Vec::with_capacity(g.count);
    let mut per_image_stats = Vec::with_capacity(g.count);
    for i in 0..g.count {
        let seed = g.base_seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(0xDEAD_BEEF)
            .wrapping_add(i as u64);
        let pixels = g.retina.dream_pixel(seed, g.sparsity_k);
        per_image_stats.push(stats(&pixels));
        let path = g.out_dir.join(format!("dream_{i:03}.ppm"));
        write_ppm(&path, &pixels, h, w)?;
        dreams.push(pixels);
    }
    write_grid_ppm(&g.out_dir.join("grid.ppm"), &dreams, h, w)?;
    write_meta(g, &per_image_stats)?;
    Ok(per_image_stats)
}

fn write_meta(g: &Gallery, s: &[DreamStats]) -> std::io::Result<()> {
    let mut f = std::fs::File::create(g.out_dir.join("meta.txt"))?;
    writeln!(f, "# dream_gallery provenance")?;
    writeln!(f, "cortex_label   = {}", g.label)?;
    writeln!(f, "input_h        = {}", g.retina.input_h)?;
    writeln!(f, "input_w        = {}", g.retina.input_w)?;
    writeln!(f, "count          = {}", g.count)?;
    writeln!(f, "base_seed      = {}", g.base_seed)?;
    writeln!(f, "sparsity_k     = {}", g.sparsity_k)?;
    writeln!(f, "receptors_ht2a = {:.3}", g.retina.receptors.ht2a)?;
    let n = s.len() as f32;
    let mean_mean = s.iter().map(|x| x.mean).sum::<f32>() / n;
    let mean_std = s.iter().map(|x| x.std).sum::<f32>() / n;
    let mean_energy = s.iter().map(|x| x.energy).sum::<f32>() / n;
    let span_min = s.iter().map(|x| x.min).fold(f32::INFINITY, f32::min);
    let span_max = s.iter().map(|x| x.max).fold(f32::NEG_INFINITY, f32::max);
    writeln!(f, "\n# summary across {} dreams", s.len())?;
    writeln!(f, "mean_of_means    = {mean_mean:.4}")?;
    writeln!(f, "mean_of_stds     = {mean_std:.4}")?;
    writeln!(f, "mean_of_energies = {mean_energy:.4}")?;
    writeln!(f, "global_min       = {span_min:.4}")?;
    writeln!(f, "global_max       = {span_max:.4}")?;
    Ok(())
}

// ────────────────────────────────────────────────────────────────
// Main: sober vs LSD side-by-side by default.
// ────────────────────────────────────────────────────────────────

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut size = 11usize;
    let mut count = 64usize;
    let mut seed = 42u64;
    let mut sparsity_k = 8usize;
    let mut integration = 0.7f32;
    let mut pretrain_samples = 500usize;
    let mut pretrain_epochs = 2usize;
    let mut pretrain_lr = 2e-4f32;
    let mut out_root = String::from("/tmp/dream_gallery");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--size" => { size = args[i+1].parse().unwrap(); i += 2; }
            "--count" => { count = args[i+1].parse().unwrap(); i += 2; }
            "--seed" => { seed = args[i+1].parse().unwrap(); i += 2; }
            "--sparsity-k" => { sparsity_k = args[i+1].parse().unwrap(); i += 2; }
            "--integration" => { integration = args[i+1].parse().unwrap(); i += 2; }
            "--pretrain-samples" => { pretrain_samples = args[i+1].parse().unwrap(); i += 2; }
            "--pretrain-epochs" => { pretrain_epochs = args[i+1].parse().unwrap(); i += 2; }
            "--pretrain-lr" => { pretrain_lr = args[i+1].parse().unwrap(); i += 2; }
            "--out-dir" => { out_root = args[i+1].clone(); i += 2; }
            "--help" | "-h" => {
                eprintln!(
"Usage: dream_gallery [--size N] [--count N] [--seed N] [--sparsity-k N]
                     [--integration F (default 0.7)]
                     [--pretrain-samples N] [--pretrain-epochs N] [--pretrain-lr F]
                     [--out-dir PATH]

Dumps two side-by-side galleries:
  <out>/sober/      - dreams from a fresh (untrained) cortex
  <out>/lsd_<I>/    - dreams after VisualCortex::lsd(integration=<I>)
                      applied to a synthetic maze-like pixel bank

Each sub-directory contains:
  dream_NNN.ppm     — per-image PPM
  grid.ppm          — ceil(sqrt(count))^2 montage
  meta.txt          — provenance (cortex state, seeds, stats)

Provenance is data-level. The gallery is reproducible bit-for-bit
from identical args (deterministic RNG + serialized retina weights).
"); return Ok(());
            }
            _ => { i += 1; }
        }
    }

    let size = size | 1;
    eprintln!("dream_gallery: size={size} count={count} seed={seed} sparsity_k={sparsity_k}");
    let root = Path::new(&out_root);
    std::fs::create_dir_all(root)?;

    // ── Condition 1: sober cortex ──
    let sober = VisualCortex::preserve_spatial(size, size);
    let sober_dir = root.join("sober");
    eprintln!("[sober] random V2/V4 — no pretraining");
    let sober_stats = run_gallery(&Gallery {
        label: "sober",
        retina: &sober,
        count, base_seed: seed, sparsity_k,
        out_dir: &sober_dir,
    })?;

    // ── Condition 2: LSD-pretrained cortex ──
    let mut trained = VisualCortex::preserve_spatial(size, size);
    let bank = maze_pixel_bank(pretrain_samples, size, seed);
    let bank_refs: Vec<&[f32]> = bank.iter().map(|v| v.as_slice()).collect();
    // First shape the cortex on real mazes (Hebbian prior), then
    // apply one LSD trip at the validated integration. This is the
    // "refinement" mode from the train_hebbian / lsd docs.
    eprintln!("[lsd] Hebbian prior on {pretrain_samples} real mazes...");
    trained.train_hebbian(&bank_refs, pretrain_epochs, pretrain_lr);
    eprintln!("[lsd] trip: integration={integration}, duration={pretrain_samples}");
    let report = trained.lsd(LsdConfig {
        dose: sparsity_k,
        duration: pretrain_samples,
        epochs: pretrain_epochs,
        lr: pretrain_lr,
        plasticity_boost: 1.0,
        integration,
        seed,
    });
    eprintln!("[lsd] peak V4 delta={:.3}, post V4 delta={:.3}, receptors.ht2a={:.3}",
        report.peak_v4_delta, report.post_v4_delta, trained.receptors.ht2a);

    let lsd_label = format!("lsd_{integration:.2}");
    let lsd_dir = root.join(&lsd_label);
    let lsd_stats = run_gallery(&Gallery {
        label: &lsd_label,
        retina: &trained,
        count, base_seed: seed, sparsity_k,
        out_dir: &lsd_dir,
    })?;

    // ── Side-by-side summary ──
    let mut f = std::fs::File::create(root.join("summary.txt"))?;
    writeln!(f, "# dream_gallery summary  (size={size}, count={count}, seed={seed}, sparsity_k={sparsity_k})")?;
    writeln!(f, "")?;
    writeln!(f, "{:<18} {:>12} {:>12}", "metric", "sober", lsd_label)?;
    writeln!(f, "{}", "-".repeat(44))?;
    let s_mean = avg(&sober_stats, |x| x.mean);
    let l_mean = avg(&lsd_stats, |x| x.mean);
    let s_std = avg(&sober_stats, |x| x.std);
    let l_std = avg(&lsd_stats, |x| x.std);
    let s_en = avg(&sober_stats, |x| x.energy);
    let l_en = avg(&lsd_stats, |x| x.energy);
    writeln!(f, "{:<18} {:>12.4} {:>12.4}", "mean of means", s_mean, l_mean)?;
    writeln!(f, "{:<18} {:>12.4} {:>12.4}", "mean of stds",  s_std,  l_std)?;
    writeln!(f, "{:<18} {:>12.4} {:>12.4}", "mean of energies", s_en, l_en)?;
    writeln!(f, "")?;
    writeln!(f, "Interpret with care:")?;
    writeln!(f, "  - If LSD std/energy are *lower* than sober: the trip narrowed V2/V4")?;
    writeln!(f, "    onto a concentrated region of filter space. Consistent with filter")?;
    writeln!(f, "    rotation toward the dream's synthetic attractor (the zero-collapse")?;
    writeln!(f, "    tendency, partially integrated — the expected behaviour at integ<1).")?;
    writeln!(f, "  - If LSD std/energy are *higher*: the trip injected more dispersion")?;
    writeln!(f, "    without closing onto a narrow manifold. Less common.")?;
    writeln!(f, "  - If columns are indistinguishable: cortex hasn't internalised structure;")?;
    writeln!(f, "    dreams reflect the random init only.")?;
    writeln!(f, "Visual inspection of grid.ppm is ultimately what matters.")?;

    eprintln!("\ndone.  grids:");
    eprintln!("  {}", sober_dir.join("grid.ppm").display());
    eprintln!("  {}", lsd_dir.join("grid.ppm").display());
    eprintln!("  {}", root.join("summary.txt").display());
    Ok(())
}

fn avg<T>(xs: &[T], f: impl Fn(&T) -> f32) -> f32 {
    xs.iter().map(&f).sum::<f32>() / xs.len().max(1) as f32
}
