//! Render VisualRetina's Conv2d filter weights as tiled images —
//! the Olshausen & Field 1996 "did Gabors emerge?" plot, ported.
//!
//! This is the most direct test of "did training shape the cortex".
//! `dream_gallery` shows adjoint-of-noise (derivative). `retina_viz`
//! shows forward activations (derivative). This shows the weights
//! themselves (primary).
//!
//! Per layer, per filter:
//!   - V1 (in=3, k=3):    render kernel as 3×3 RGB tile (channels = colour)
//!   - V2 (in=32, k=3):   collapse in-channel axis via per-position
//!                        L2 norm → 3×3 grayscale tile
//!   - V4 (in=64, k=3):   same grayscale aggregation → 3×3 tile
//!
//! Per-filter tiles are min-max normalised locally and upscaled by a
//! factor (default 32×) so structure is visible to the eye. Filters
//! that learned edge-like structure show up as oriented light/dark
//! patches. Unshaped random filters look like noise.
//!
//! Three conditions rendered side-by-side, so you can compare
//! sober / Hebbian / LSD weight shape directly.
//!
//! Usage:
//!   cargo run -p filter_viz --release
//!   cargo run -p filter_viz --release -- --size 21 --pretrain-samples 500

use modgrad_codec::retina::{Conv2d, LsdConfig, VisualRetina};
use std::io::Write;
use std::path::Path;

// ────────────────────────────────────────────────────────────────
// Maze gen (duplicated from maze_viz to keep binary standalone).
// ────────────────────────────────────────────────────────────────

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

fn maze_pixel_bank(n: usize, size: usize, seed: u64) -> Vec<Vec<f32>> {
    (0..n).map(|i| {
        let s = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(0xBED_CAFE)
            .wrapping_add(i as u64);
        let cells = generate_maze_cells(size, s);
        let mut px = vec![0.0f32; 3 * size * size];
        for idx in 0..cells.len() {
            let v = if cells[idx] { 0.9 } else { 0.1 };
            for c in 0..3 { px[c * size * size + idx] = v; }
        }
        px
    }).collect()
}

// ────────────────────────────────────────────────────────────────
// PPM helpers.
// ────────────────────────────────────────────────────────────────

fn write_ppm(path: &Path, rgb: &[u8], h: usize, w: usize) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "P6\n{w} {h}\n255")?;
    f.write_all(rgb)?;
    Ok(())
}

/// Min-max normalise a slice to [0, 255] bytes.
fn normalise_to_u8(vals: &[f32]) -> Vec<u8> {
    let (mn, mx) = vals.iter().fold((f32::INFINITY, f32::NEG_INFINITY),
        |(a, b), &v| (a.min(v), b.max(v)));
    let span = (mx - mn).max(1e-6);
    vals.iter().map(|v| (((v - mn) / span) * 255.0).clamp(0.0, 255.0) as u8).collect()
}

fn upscale_rgb(tile_rgb: &[u8], h: usize, w: usize, factor: usize) -> (Vec<u8>, usize, usize) {
    let oh = h * factor;
    let ow = w * factor;
    let mut out = vec![0u8; 3 * oh * ow];
    for y in 0..h {
        for x in 0..w {
            let src = (y * w + x) * 3;
            for dy in 0..factor {
                for dx in 0..factor {
                    let dst = ((y * factor + dy) * ow + (x * factor + dx)) * 3;
                    out[dst] = tile_rgb[src];
                    out[dst + 1] = tile_rgb[src + 1];
                    out[dst + 2] = tile_rgb[src + 2];
                }
            }
        }
    }
    (out, oh, ow)
}

// ────────────────────────────────────────────────────────────────
// Per-filter tiles.
// ────────────────────────────────────────────────────────────────

/// V1 has in_channels=3 — render each filter as a proper RGB 3×3 tile.
/// Weights are [out_ch × 3 × kh × kw]. Extract filter `oc`, reshape,
/// min-max normalise per-channel, emit RGB. If the fixed Gabor-ish
/// init is working, tiles will show centre-surround / edge patterns.
fn v1_tile_rgb(conv: &Conv2d, oc: usize) -> Vec<u8> {
    let k = conv.kernel_size;
    assert_eq!(conv.in_channels, 3);
    let mut rgb_f = vec![0.0f32; 3 * k * k];
    for ic in 0..3 {
        for y in 0..k {
            for x in 0..k {
                let w_idx = oc * (3 * k * k) + ic * (k * k) + y * k + x;
                rgb_f[(y * k + x) * 3 + ic] = conv.weight[w_idx];
            }
        }
    }
    normalise_to_u8(&rgb_f)
}

/// V2/V4 have many in channels — collapse with per-position L2 norm
/// across in_channels so a 3×3 tile summarises filter `oc`. If Hebbian
/// shaped the filter, positions will differ (structured); random init
/// produces uniform-ish norms (noise).
fn vn_tile_rgb(conv: &Conv2d, oc: usize) -> Vec<u8> {
    let k = conv.kernel_size;
    let mut norms = vec![0.0f32; k * k];
    for y in 0..k {
        for x in 0..k {
            let mut s = 0.0f32;
            for ic in 0..conv.in_channels {
                let w_idx = oc * (conv.in_channels * k * k) + ic * (k * k) + y * k + x;
                s += conv.weight[w_idx].powi(2);
            }
            norms[y * k + x] = s.sqrt();
        }
    }
    let u8 = normalise_to_u8(&norms);
    // Grey: replicate to 3 channels.
    u8.into_iter().flat_map(|v| [v, v, v]).collect()
}

// ────────────────────────────────────────────────────────────────
// Tiled grid.
// ────────────────────────────────────────────────────────────────

/// Render a Conv2d layer as a tiled grid of per-filter tiles. Each
/// tile is `k × k` upscaled by `factor`. `cols` tiles per row; rows
/// derived from filter count.
fn render_layer_grid(
    conv: &Conv2d,
    is_v1: bool,
    cols: usize,
    factor: usize,
) -> (Vec<u8>, usize, usize) {
    let k = conv.kernel_size;
    let tile_h = k * factor;
    let tile_w = k * factor;
    let n = conv.out_channels;
    let rows = (n + cols - 1) / cols;
    let border = 2usize;
    let gh = rows * tile_h + (rows + 1) * border;
    let gw = cols * tile_w + (cols + 1) * border;
    let mut grid = vec![60u8; 3 * gh * gw]; // dark grey background
    for oc in 0..n {
        let raw_rgb = if is_v1 { v1_tile_rgb(conv, oc) } else { vn_tile_rgb(conv, oc) };
        let (tile, _, _) = upscale_rgb(&raw_rgb, k, k, factor);
        let r = oc / cols;
        let c = oc % cols;
        let oy = border + r * (tile_h + border);
        let ox = border + c * (tile_w + border);
        for y in 0..tile_h {
            for x in 0..tile_w {
                let src = (y * tile_w + x) * 3;
                let dst = ((oy + y) * gw + (ox + x)) * 3;
                grid[dst] = tile[src];
                grid[dst + 1] = tile[src + 1];
                grid[dst + 2] = tile[src + 2];
            }
        }
    }
    (grid, gh, gw)
}

// ────────────────────────────────────────────────────────────────
// Per-condition dump.
// ────────────────────────────────────────────────────────────────

fn dump_condition(label: &str, retina: &VisualRetina, out_dir: &Path, factor: usize) -> std::io::Result<()> {
    std::fs::create_dir_all(out_dir)?;
    let (v1, h1, w1) = render_layer_grid(&retina.v1, true, 8, factor);
    let (v2, h2, w2) = render_layer_grid(&retina.v2, false, 8, factor);
    let (v4, h4, w4) = render_layer_grid(&retina.v4, false, 16, factor);
    write_ppm(&out_dir.join("v1_filters.ppm"), &v1, h1, w1)?;
    write_ppm(&out_dir.join("v2_filters.ppm"), &v2, h2, w2)?;
    write_ppm(&out_dir.join("v4_filters.ppm"), &v4, h4, w4)?;

    // Filter magnitude stats per layer — helps spot dead / saturated filters.
    let mut meta = std::fs::File::create(out_dir.join("meta.txt"))?;
    writeln!(meta, "# filter_viz :: {label}")?;
    for (name, conv) in [("V1", &retina.v1), ("V2", &retina.v2), ("V4", &retina.v4)] {
        let k = conv.kernel_size;
        let per_filter = conv.in_channels * k * k;
        let n = conv.out_channels;
        let mut norms = Vec::with_capacity(n);
        let mut sparse = 0usize;
        for oc in 0..n {
            let base = oc * per_filter;
            let l2 = conv.weight[base..base + per_filter].iter()
                .map(|w| w * w).sum::<f32>().sqrt();
            if l2 < 1e-6 { sparse += 1; }
            norms.push(l2);
        }
        let mean = norms.iter().sum::<f32>() / n as f32;
        let var = norms.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
        let mn = norms.iter().cloned().fold(f32::INFINITY, f32::min);
        let mx = norms.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        writeln!(meta, "\n{name}: out={n}, in={}, k={k}", conv.in_channels)?;
        writeln!(meta, "  filter ||W||₂ : mean={mean:.4} std={:.4} min={mn:.4} max={mx:.4}", var.sqrt())?;
        writeln!(meta, "  near-zero filters: {sparse}/{n}")?;
    }
    Ok(())
}

// ────────────────────────────────────────────────────────────────
// Main.
// ────────────────────────────────────────────────────────────────

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut size = 21usize;
    let mut seed = 42u64;
    let mut factor = 32usize;
    let mut pretrain_samples = 500usize;
    let mut pretrain_epochs = 2usize;
    let mut integration = 0.7f32;
    let mut out_root = String::from("/tmp/filter_viz");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--size" => { size = args[i+1].parse().unwrap(); i += 2; }
            "--seed" => { seed = args[i+1].parse().unwrap(); i += 2; }
            "--factor" => { factor = args[i+1].parse().unwrap(); i += 2; }
            "--pretrain-samples" => { pretrain_samples = args[i+1].parse().unwrap(); i += 2; }
            "--pretrain-epochs" => { pretrain_epochs = args[i+1].parse().unwrap(); i += 2; }
            "--integration" => { integration = args[i+1].parse().unwrap(); i += 2; }
            "--out-dir" => { out_root = args[i+1].clone(); i += 2; }
            "--help" | "-h" => {
                eprintln!(
"Usage: filter_viz [--size N] [--seed N] [--factor N]
                  [--pretrain-samples N] [--pretrain-epochs N] [--integration F]
                  [--out-dir PATH]

Dumps Conv2d filter grids for three conditions:
  <out>/sober/           fresh random V2/V4
  <out>/hebbian/         train_hebbian on N real mazes
  <out>/lsd_<int>/       hebbian + lsd(integration=<int>)

Each condition writes v1_filters.ppm / v2_filters.ppm / v4_filters.ppm
(32 / 64 / 128 tiles respectively) plus meta.txt with per-layer filter
norm stats. If Hebbian shaped the filters, `mean ||W||₂` will shift,
tile variance will grow, and tiles will show visible spatial structure.
Pure noise = training did nothing.
"); return Ok(());
            }
            _ => { i += 1; }
        }
    }
    if size % 2 == 0 { size += 1; }
    let root = Path::new(&out_root);
    std::fs::create_dir_all(root)?;
    eprintln!("filter_viz: size={size} seed={seed} factor={factor}");

    // Build the shared maze bank once.
    let bank = maze_pixel_bank(pretrain_samples, size, seed);
    let bank_refs: Vec<&[f32]> = bank.iter().map(|v| v.as_slice()).collect();

    // Sober.
    let sober = VisualRetina::maze(size, size);
    eprintln!("[sober]   dumping weights");
    dump_condition("sober", &sober, &root.join("sober"), factor)?;

    // Hebbian.
    let mut hebbian = VisualRetina::maze(size, size);
    eprintln!("[hebbian] training on {pretrain_samples} real mazes");
    hebbian.train_hebbian(&bank_refs, pretrain_epochs, 2e-4);
    dump_condition("hebbian", &hebbian, &root.join("hebbian"), factor)?;

    // LSD.
    let mut lsd = VisualRetina::maze(size, size);
    eprintln!("[lsd_{integration:.2}]   hebbian + lsd(integration={integration:.2})");
    lsd.train_hebbian(&bank_refs, pretrain_epochs, 2e-4);
    lsd.lsd(LsdConfig {
        dose: 8,
        duration: pretrain_samples,
        epochs: pretrain_epochs,
        lr: 2e-4,
        plasticity_boost: 1.0,
        integration,
        seed,
    });
    dump_condition(
        &format!("lsd_{integration:.2}"),
        &lsd,
        &root.join(format!("lsd_{integration:.2}")),
        factor,
    )?;

    eprintln!("\ndone.");
    eprintln!("  look: {}/*/v1_filters.ppm  — V1 should show RGB edge patterns (fixed Gabor init)", root.display());
    eprintln!("  look: {}/*/v2_filters.ppm  — V2 norm tiles; Hebbian vs sober should differ", root.display());
    eprintln!("  look: {}/*/v4_filters.ppm  — V4 norm tiles; 128 filters", root.display());
    Ok(())
}
