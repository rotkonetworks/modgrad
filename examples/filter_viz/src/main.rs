//! Render VisualCortex's Conv2d filter weights as tiled images —
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

use modgrad_codec::retina::{Conv2d, LsdConfig, VisualCortex};
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

/// V1 has in_channels=12 (retinal ganglion output) — pick the retinal
/// input channel where this filter has the most energy and render that
/// one as a 3×3 grayscale tile. The Gabor init places each V1 filter
/// onto exactly one of retinal channels 0 (luminance ON) or 1
/// (luminance OFF); GHL training can spread weight across channels.
/// The max-energy channel is the most informative view.
fn v1_tile_rgb(conv: &Conv2d, oc: usize) -> Vec<u8> {
    let k = conv.kernel_size;
    let in_ch = conv.in_channels;
    let mut best_ic = 0usize;
    let mut best_energy = 0.0f32;
    for ic in 0..in_ch {
        let mut e = 0.0f32;
        for i in 0..(k * k) {
            let w_idx = oc * (in_ch * k * k) + ic * (k * k) + i;
            let v = conv.weight[w_idx];
            e += v * v;
        }
        if e > best_energy { best_energy = e; best_ic = ic; }
    }
    let mut gray_f = vec![0.0f32; 3 * k * k];
    for y in 0..k {
        for x in 0..k {
            let w_idx = oc * (in_ch * k * k) + best_ic * (k * k) + y * k + x;
            let v = conv.weight[w_idx];
            let p = (y * k + x) * 3;
            gray_f[p] = v;
            gray_f[p + 1] = v;
            gray_f[p + 2] = v;
        }
    }
    normalise_to_u8(&gray_f)
}

/// Pull a single V2 or V4 filter's weights back through V1 (and V2 if
/// going from V4) to pixel space. This is the "effective receptive
/// field" visualization — what pixel pattern maximally activates this
/// filter, assuming the downstream layers are passthrough.
///
/// For V2 filter `oc`: treat its `[32 × 3 × 3]` weights as a one-hot
/// activation grid at the V2-output layer, then `v1.transpose_forward`
/// scatters it to `[3 × 5 × 5]` pixel-space (one extra cell per
/// 3×3 kernel on each side due to the convolution's receptive field).
/// That's a 3-channel RGB patch we can display directly.
///
/// For V4 filter `oc`: two transpose_forward passes — V2 first then V1.
/// Resulting pixel patch is larger (receptive field compounds).
fn pixelspace_receptive_field(
    cortex: &VisualCortex,
    layer: &'static str,
    oc: usize,
) -> (Vec<f32>, usize, usize) {
    let v1 = &cortex.v1;
    let v2 = &cortex.v2;
    let v4 = &cortex.v4;
    let k1 = v1.kernel_size;
    let k2 = v2.kernel_size;
    let k4 = v4.kernel_size;
    let s1 = v1.stride;
    let s2 = v2.stride;
    let s4 = v4.stride;
    let p1 = v1.padding;
    let p2 = v2.padding;
    let p4 = v4.padding;

    // Seed: the filter weights of this oc, treated as an activation
    // map at the matching layer's output. Shape [in_ch × k × k].
    match layer {
        "v2" => {
            // V2 weights shape [oc × in_ch=32 × k2 × k2]. Extract this filter.
            let w2_off = oc * v2.in_channels * k2 * k2;
            let seed_v1 = &v2.weight[w2_off..w2_off + v2.in_channels * k2 * k2];
            // Pull V1 adjoint: input "activation" of shape [32 × k2 × k2]
            // scatters to [3 × h1_pre × w1_pre] pixel space.
            // The forward V1: (h + 2p - k) / s + 1 = k2; solve for
            // pre-V1 h: h_pre = (k2 - 1)*s1 + k1 - 2*p1.
            let h_pre = (k2 - 1) * s1 + k1 - 2 * p1;
            let w_pre = h_pre;
            let pixels = v1.transpose_forward(seed_v1, 1, k2, k2, h_pre, w_pre);
            (pixels, h_pre, w_pre)
        }
        "v4" => {
            // V4 weights [oc × in_ch=64 × k4 × k4]. Treat as activations
            // at V4-output, pull through V2 then V1.
            let w4_off = oc * v4.in_channels * k4 * k4;
            let seed_v2 = &v4.weight[w4_off..w4_off + v4.in_channels * k4 * k4];
            // Through V2 adjoint: [64 × k4 × k4] → [32 × h2 × w2]
            let h2 = (k4 - 1) * s2 + k2 - 2 * p2;
            let w2d = h2;
            let via_v2 = v2.transpose_forward(seed_v2, 1, k4, k4, h2, w2d);
            // Through V1 adjoint: [32 × h2 × w2] → [3 × h1 × w1]
            let h_pre = (h2 - 1) * s1 + k1 - 2 * p1;
            let w_pre = h_pre;
            let pixels = v1.transpose_forward(&via_v2, 1, h2, w2d, h_pre, w_pre);
            (pixels, h_pre, w_pre)
        }
        _ => panic!("pixelspace_receptive_field: layer must be v2 or v4"),
    }
}

/// Render one cortex filter as its pixel-space receptive field tile
/// (RGB). Normalises per-tile so each filter's pattern is visible
/// regardless of absolute magnitude.
fn rf_tile_rgb(cortex: &VisualCortex, layer: &'static str, oc: usize) -> (Vec<u8>, usize, usize) {
    let (pixels, h, w) = pixelspace_receptive_field(cortex, layer, oc);
    // Re-layout from CHW to HWC, per-tile min-max over all channels.
    let mut hwc = vec![0.0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                hwc[(y * w + x) * 3 + c] = pixels[c * h * w + y * w + x];
            }
        }
    }
    (normalise_to_u8(&hwc), h, w)
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

/// Render a grid of pixel-space receptive fields for V2 or V4. Each
/// tile shows one filter's preferred stimulus (pulled back through V1
/// adjoint, or V2→V1 for V4). `cols` tiles per row; per-tile upscale
/// applied for visibility.
fn write_rf_grid(
    cortex: &VisualCortex, layer: &'static str,
    cols: usize, upscale: usize,
    path: &Path,
) -> std::io::Result<()> {
    let n = match layer {
        "v2" => cortex.v2.out_channels,
        "v4" => cortex.v4.out_channels,
        _ => unreachable!(),
    };
    // Probe tile size off filter 0.
    let (_, th, tw) = rf_tile_rgb(cortex, layer, 0);
    let tile_h = th * upscale;
    let tile_w = tw * upscale;
    let rows = (n + cols - 1) / cols;
    let border = 2usize;
    let gh = rows * tile_h + (rows + 1) * border;
    let gw = cols * tile_w + (cols + 1) * border;
    let mut grid = vec![60u8; 3 * gh * gw];
    for oc in 0..n {
        let (rgb, h, w) = rf_tile_rgb(cortex, layer, oc);
        let (tile, _, _) = upscale_rgb(&rgb, h, w, upscale);
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
    write_ppm(path, &grid, gh, gw)
}

fn dump_condition(label: &str, retina: &VisualCortex, out_dir: &Path, factor: usize) -> std::io::Result<()> {
    std::fs::create_dir_all(out_dir)?;
    let (v1, h1, w1) = render_layer_grid(&retina.v1, true, 8, factor);
    let (v2, h2, w2) = render_layer_grid(&retina.v2, false, 8, factor);
    let (v4, h4, w4) = render_layer_grid(&retina.v4, false, 16, factor);
    write_ppm(&out_dir.join("v1_filters.ppm"), &v1, h1, w1)?;
    write_ppm(&out_dir.join("v2_filters.ppm"), &v2, h2, w2)?;
    write_ppm(&out_dir.join("v4_filters.ppm"), &v4, h4, w4)?;

    // Pixel-space receptive fields: what pixel pattern would maximally
    // activate each V2/V4 filter. Much more informative than per-position
    // norm tiles for deep filters — you're literally seeing Olshausen/
    // Field-style "preferred stimulus" patches.
    write_rf_grid(retina, "v2", 8, 12, &out_dir.join("v2_rf.ppm"))?;
    write_rf_grid(retina, "v4", 16, 8, &out_dir.join("v4_rf.ppm"))?;

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
    // --load PATH: skip the maze-based conditions and just dump filters
    // from a pretrained cortex file (e.g. retina_stl10.bin from
    // pretrain_retina). Only produces `<out-dir>/loaded/{v1,v2,v4}.ppm`.
    let mut load_path: Option<String> = None;

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
            "--load" => { load_path = Some(args[i+1].clone()); i += 2; }
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

    // ── --load mode: dump filters from a pretrained file, no training. ──
    if let Some(path) = &load_path {
        let cortex = VisualCortex::load(path)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData,
                format!("failed to load {path}: {e}")))?;
        eprintln!(
            "[loaded from {path}] V1 {}→{}  V2 {}→{}  V4 {}→{}  input {}×{}",
            cortex.v1.in_channels, cortex.v1.out_channels,
            cortex.v2.in_channels, cortex.v2.out_channels,
            cortex.v4.in_channels, cortex.v4.out_channels,
            cortex.input_h, cortex.input_w,
        );
        dump_condition("loaded", &cortex, &root.join("loaded"), factor)?;
        eprintln!("\ndone.  look at {}/loaded/*.ppm", root.display());
        return Ok(());
    }

    // Build the shared maze bank once.
    let bank = maze_pixel_bank(pretrain_samples, size, seed);
    let bank_refs: Vec<&[f32]> = bank.iter().map(|v| v.as_slice()).collect();

    // Sober.
    let sober = VisualCortex::preserve_spatial(size, size);
    eprintln!("[sober]   dumping weights");
    dump_condition("sober", &sober, &root.join("sober"), factor)?;

    // Hebbian.
    let mut hebbian = VisualCortex::preserve_spatial(size, size);
    eprintln!("[hebbian] training on {pretrain_samples} real mazes");
    hebbian.train_hebbian(&bank_refs, pretrain_epochs, 2e-4);
    dump_condition("hebbian", &hebbian, &root.join("hebbian"), factor)?;

    // LSD.
    let mut lsd = VisualCortex::preserve_spatial(size, size);
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
