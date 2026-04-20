//! Does the retina actually see? This binary runs one real maze
//! through `VisualRetina`'s forward pass and dumps:
//!
//!   input.ppm    — the [3 × h × w] maze pixels, upscaled for viewing.
//!   v1.ppm       — 32 V1 activation heatmaps after `leaky_relu`.
//!                  If our fixed Gabor-ish filters work, these should
//!                  show oriented edges tracing the maze walls.
//!   v2.ppm       — 64 V2 activation heatmaps. Higher-level contours.
//!   v4.ppm       — 128 V4 activation heatmaps. Shape / texture.
//!
//! The three activation grids answer the "does each layer produce
//! structured, input-dependent output" question directly. Grey noise
//! tiles = broken layer. Structured patterns that change per-maze =
//! the retina is doing what it claims.
//!
//! Runs three conditions:
//!   sober/     — fresh random V2/V4
//!   hebbian/   — after train_hebbian on synthetic maze-bias pixels
//!   lsd_0.7/   — after lsd(integration=0.7)
//!
//! Usage:
//!   cargo run -p retina_viz --release
//!   cargo run -p retina_viz --release -- --size 21 --seed 7

use modgrad_codec::retina::{LsdConfig, VisualRetina};
use std::io::Write;
use std::path::Path;

// ────────────────────────────────────────────────────────────────
// Minimal deterministic maze generator (same algorithm as maze_viz).
// Duplicated intentionally to keep this binary standalone.
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

fn generate_maze(size: usize, seed: u64) -> Vec<bool> {
    assert!(size % 2 == 1 && size >= 5);
    let mut cells = vec![false; size * size];
    let mut rng = Rng::new(seed);
    let mut stack = vec![(1usize, 1usize)];
    cells[1 * size + 1] = true;
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

/// Convert `cells: bool[size*size]` into a `[3 × size × size]` float
/// tensor (wall = 0.1, path = 0.9 for a slight nonzero floor — matches
/// how the retina benchmark typically feeds mazes).
fn maze_pixels(cells: &[bool], size: usize) -> Vec<f32> {
    let mut px = vec![0.0f32; 3 * size * size];
    for i in 0..cells.len() {
        let v = if cells[i] { 0.9 } else { 0.1 };
        for c in 0..3 { px[c * size * size + i] = v; }
    }
    px
}

// ────────────────────────────────────────────────────────────────
// PPM rendering helpers.
// ────────────────────────────────────────────────────────────────

fn write_ppm(path: &Path, rgb: &[u8], h: usize, w: usize) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "P6\n{w} {h}\n255")?;
    f.write_all(rgb)?;
    Ok(())
}

/// Nearest-neighbour upscale a single-channel f32 heatmap into an RGB
/// block. Values normalised per-tile to [0, 255]; positive values blue,
/// negative red, so sign is visible at a glance (pixel-level activations
/// after leaky_relu are mostly positive, but let's be honest about
/// anything negative that slips through).
fn heatmap_tile(vals: &[f32], h: usize, w: usize, upscale: usize) -> Vec<u8> {
    let (mn, mx) = vals.iter().fold((f32::INFINITY, f32::NEG_INFINITY),
        |(a, b), &v| (a.min(v), b.max(v)));
    let span = (mx - mn).max(1e-6);
    let out_h = h * upscale;
    let out_w = w * upscale;
    let mut rgb = vec![0u8; 3 * out_h * out_w];
    for y in 0..h {
        for x in 0..w {
            let v = (vals[y * w + x] - mn) / span; // [0,1]
            // Diverging palette: 0 → deep red, 0.5 → grey, 1 → deep blue
            let r = ((1.0 - v) * 255.0).clamp(0.0, 255.0) as u8;
            let b = (v * 255.0).clamp(0.0, 255.0) as u8;
            let g = ((0.5 - (v - 0.5).abs()) * 200.0).clamp(0.0, 255.0) as u8;
            for dy in 0..upscale {
                for dx in 0..upscale {
                    let py = y * upscale + dy;
                    let px = x * upscale + dx;
                    let off = (py * out_w + px) * 3;
                    rgb[off] = r;
                    rgb[off + 1] = g;
                    rgb[off + 2] = b;
                }
            }
        }
    }
    rgb
}

/// Render a multi-channel activation `[channels × h × w]` as a tiled
/// grid of heatmaps. `cols` tiles per row, per-tile upscale factor.
fn channels_to_ppm(
    path: &Path,
    activations: &[f32],
    channels: usize,
    h: usize,
    w: usize,
    cols: usize,
    upscale: usize,
) -> std::io::Result<()> {
    let rows = (channels + cols - 1) / cols;
    let tile_h = h * upscale;
    let tile_w = w * upscale;
    let border = 2usize;
    let gh = rows * tile_h + (rows + 1) * border;
    let gw = cols * tile_w + (cols + 1) * border;
    let mut grid = vec![50u8; 3 * gh * gw];
    for c in 0..channels {
        let r = c / cols;
        let col = c % cols;
        let oy = border + r * (tile_h + border);
        let ox = border + col * (tile_w + border);
        let tile = heatmap_tile(
            &activations[c * h * w..(c + 1) * h * w], h, w, upscale,
        );
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
    write_ppm(path, &grid, gh, gw)
}

/// Render the [3 × h × w] input pixel tensor at scale.
fn render_input(path: &Path, pixels: &[f32], h: usize, w: usize, upscale: usize) -> std::io::Result<()> {
    let out_h = h * upscale;
    let out_w = w * upscale;
    let mut rgb = vec![0u8; 3 * out_h * out_w];
    for y in 0..h {
        for x in 0..w {
            let r = (pixels[0 * h * w + y * w + x] * 255.0).clamp(0.0, 255.0) as u8;
            let g = (pixels[1 * h * w + y * w + x] * 255.0).clamp(0.0, 255.0) as u8;
            let b = (pixels[2 * h * w + y * w + x] * 255.0).clamp(0.0, 255.0) as u8;
            for dy in 0..upscale {
                for dx in 0..upscale {
                    let py = y * upscale + dy;
                    let px = x * upscale + dx;
                    let off = (py * out_w + px) * 3;
                    rgb[off] = r;
                    rgb[off + 1] = g;
                    rgb[off + 2] = b;
                }
            }
        }
    }
    write_ppm(path, &rgb, out_h, out_w)
}

// ────────────────────────────────────────────────────────────────
// Retina forward pass — mirror of VisualRetina::spatial_tokens up
// through each layer, but keeping intermediates.
// ────────────────────────────────────────────────────────────────

fn leaky_relu(x: &mut [f32]) {
    for v in x.iter_mut() { if *v < 0.0 { *v *= 0.1; } }
}

struct LayerOut {
    data: Vec<f32>,
    channels: usize,
    h: usize,
    w: usize,
}

fn retina_forward(retina: &VisualRetina, pixels: &[f32]) -> (LayerOut, LayerOut, LayerOut) {
    let (h, w) = (retina.input_h, retina.input_w);
    let (mut v1, h1, w1) = retina.v1.forward(pixels, h, w);
    leaky_relu(&mut v1);
    let v1_out = LayerOut { data: v1.clone(), channels: retina.v1.out_channels, h: h1, w: w1 };
    let (mut v2, h2, w2) = retina.v2.forward(&v1, h1, w1);
    leaky_relu(&mut v2);
    let v2_out = LayerOut { data: v2.clone(), channels: retina.v2.out_channels, h: h2, w: w2 };
    let (mut v4, h4, w4) = retina.v4.forward(&v2, h2, w2);
    leaky_relu(&mut v4);
    let v4_out = LayerOut { data: v4, channels: retina.v4.out_channels, h: h4, w: w4 };
    (v1_out, v2_out, v4_out)
}

// ────────────────────────────────────────────────────────────────
// Pretraining helpers (same synthetic bank as dream_gallery).
// ────────────────────────────────────────────────────────────────

fn synthetic_maze_bank(n: usize, size: usize, seed: u64) -> Vec<Vec<f32>> {
    (0..n).map(|i| {
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(i as u64);
        let mut pixels = vec![0.0f32; 3 * size * size];
        for y in 0..size {
            for x in 0..size {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let u = (s >> 32) as f32 / u32::MAX as f32;
                let pv = if u < 0.35 { 0.1 } else if u > 0.65 { 0.9 } else { u };
                for c in 0..3 { pixels[c * size * size + y * size + x] = pv; }
            }
        }
        pixels
    }).collect()
}

// ────────────────────────────────────────────────────────────────
// Per-condition dump.
// ────────────────────────────────────────────────────────────────

fn dump_condition(
    label: &str,
    retina: &VisualRetina,
    pixels: &[f32],
    out_dir: &Path,
    upscale: usize,
) -> std::io::Result<()> {
    std::fs::create_dir_all(out_dir)?;
    let (v1, v2, v4) = retina_forward(retina, pixels);
    render_input(&out_dir.join("input.ppm"), pixels, retina.input_h, retina.input_w, upscale)?;
    // Tile layout choices: 32 = 8×4, 64 = 8×8, 128 = 16×8.
    // Use smaller per-tile upscale for deep layers so files stay manageable.
    channels_to_ppm(&out_dir.join("v1.ppm"), &v1.data, v1.channels, v1.h, v1.w, 8,  upscale.max(8))?;
    channels_to_ppm(&out_dir.join("v2.ppm"), &v2.data, v2.channels, v2.h, v2.w, 8,  upscale.max(6))?;
    channels_to_ppm(&out_dir.join("v4.ppm"), &v4.data, v4.channels, v4.h, v4.w, 16, upscale.max(4))?;

    // Summary stats per layer — cheap way to spot a dead layer.
    let stats = |x: &[f32]| {
        let n = x.len() as f32;
        let mean = x.iter().sum::<f32>() / n;
        let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
        let mx = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let nonzero = x.iter().filter(|v| v.abs() > 1e-4).count();
        (mean, var.sqrt(), mx, nonzero, x.len())
    };
    let mut meta = std::fs::File::create(out_dir.join("meta.txt"))?;
    writeln!(meta, "# retina_viz :: {label}")?;
    writeln!(meta, "input_hw   = {} × {}", retina.input_h, retina.input_w)?;
    for (name, o) in [("V1", &v1), ("V2", &v2), ("V4", &v4)] {
        let (m, s, mx, nz, n) = stats(&o.data);
        writeln!(meta, "\n{name}: channels={}, {}×{} per channel", o.channels, o.h, o.w)?;
        writeln!(meta, "  mean={m:.4}, std={s:.4}, max={mx:.4}")?;
        writeln!(meta, "  nonzero={nz}/{n} ({:.1}%)", 100.0 * nz as f32 / n as f32)?;
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
    let mut upscale = 16usize;
    let mut pretrain_samples = 500usize;
    let mut pretrain_epochs = 2usize;
    let mut integration = 0.7f32;
    let mut out_root = String::from("/tmp/retina_viz");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--size" => { size = args[i+1].parse().unwrap(); i += 2; }
            "--seed" => { seed = args[i+1].parse().unwrap(); i += 2; }
            "--upscale" => { upscale = args[i+1].parse().unwrap(); i += 2; }
            "--pretrain-samples" => { pretrain_samples = args[i+1].parse().unwrap(); i += 2; }
            "--pretrain-epochs" => { pretrain_epochs = args[i+1].parse().unwrap(); i += 2; }
            "--integration" => { integration = args[i+1].parse().unwrap(); i += 2; }
            "--out-dir" => { out_root = args[i+1].clone(); i += 2; }
            "--help" | "-h" => {
                eprintln!(
"Usage: retina_viz [--size N (odd, ≥5)] [--seed N] [--upscale N]
                  [--pretrain-samples N] [--pretrain-epochs N] [--integration F]
                  [--out-dir PATH]

Runs one real maze through VisualRetina's forward pass and dumps:
  <out>/<cond>/input.ppm     — the maze pixels (upscaled for viewing)
  <out>/<cond>/v1.ppm        — 32 V1 activation tiles (leaky_relu'd)
  <out>/<cond>/v2.ppm        — 64 V2 activation tiles
  <out>/<cond>/v4.ppm        — 128 V4 activation tiles
  <out>/<cond>/meta.txt      — layer summary stats (mean/std/nonzero%)

Three conditions rendered side by side:
  sober/        random V2/V4 (no training)
  hebbian/      train_hebbian on synthetic maze-bias pixel bank
  lsd_<int>/    train_hebbian then lsd(integration=<int>)

Structured heatmaps that change between channels = retina is seeing.
Flat or pure noise tiles = broken layer.
"); return Ok(());
            }
            _ => { i += 1; }
        }
    }
    if size % 2 == 0 { size += 1; }
    let root = Path::new(&out_root);
    std::fs::create_dir_all(root)?;

    // One maze shared across all three conditions.
    let cells = generate_maze(size, seed);
    let pixels = maze_pixels(&cells, size);

    eprintln!("retina_viz: size={size} seed={seed} upscale={upscale}");

    // Sober.
    let sober = VisualRetina::maze(size, size);
    eprintln!("[sober]   random cortex");
    dump_condition("sober", &sober, &pixels, &root.join("sober"), upscale)?;

    // Hebbian.
    let mut hebbian = VisualRetina::maze(size, size);
    let bank = synthetic_maze_bank(pretrain_samples, size, seed);
    let refs: Vec<&[f32]> = bank.iter().map(|v| v.as_slice()).collect();
    eprintln!("[hebbian] train_hebbian on {pretrain_samples} synthetic mazes");
    hebbian.train_hebbian(&refs, pretrain_epochs, 2e-4);
    dump_condition("hebbian", &hebbian, &pixels, &root.join("hebbian"), upscale)?;

    // LSD (hebbian prior + lsd trip).
    let mut lsd = VisualRetina::maze(size, size);
    eprintln!("[lsd_{integration:.2}]   hebbian + lsd(integration={integration:.2})");
    lsd.train_hebbian(&refs, pretrain_epochs, 2e-4);
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
        &pixels,
        &root.join(format!("lsd_{integration:.2}")),
        upscale,
    )?;

    eprintln!("\ndone.");
    eprintln!("  open {}/sober/v1.ppm — if Gabors work, these show edges.", root.display());
    eprintln!("  open {}/sober/v4.ppm — 128 channels; same maze should give same output.", root.display());
    eprintln!("  diff sober vs hebbian vs lsd_{integration:.2} to see training effect.");
    Ok(())
}
