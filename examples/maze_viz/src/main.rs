//! Render real mazes as PPM — the structured counterpart to
//! `dream_gallery`'s noise.
//!
//! What this shows: the actual input distribution the cortex would
//! face on the maze benchmark. Not an adjoint of anything, not a
//! downstream derivative — just the pixel tensor that gets fed into
//! `VisualRetina::encode`, upscaled for visibility and with the
//! optimal route drawn on top.
//!
//! The visual contrast with `dream_gallery` is the point: real mazes
//! are strongly structured (walls / paths / topology); adjoint
//! dreams are near-gray noise. That gap is what a learned decoder
//! (Ha/Schmidhuber VAE, Deperrois adversarial dreaming, etc.) would
//! be designed to close — but we aren't there yet.
//!
//! Usage:
//!   cargo run -p maze_viz --release
//!   cargo run -p maze_viz --release -- --size 21 --count 16
//!   cargo run -p maze_viz --release -- --no-route
//!
//! Output: /tmp/maze_viz/maze_NN.ppm + grid.ppm + meta.txt

use std::io::Write;
use std::path::Path;

// ────────────────────────────────────────────────────────────────
// Deterministic RNG — same seed → bit-identical maze.
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

// ────────────────────────────────────────────────────────────────
// Maze: grid of cells, `false` = wall, `true` = path.
// ────────────────────────────────────────────────────────────────

struct Maze {
    size: usize,
    cells: Vec<bool>,           // row-major, size*size
    route: Vec<(usize, usize)>, // path from (1,1) to (size-2, size-2)
}

impl Maze {
    fn new(size: usize) -> Self {
        Self { size, cells: vec![false; size * size], route: Vec::new() }
    }
    fn at(&self, y: usize, x: usize) -> bool { self.cells[y * self.size + x] }
    fn set(&mut self, y: usize, x: usize, v: bool) { self.cells[y * self.size + x] = v; }
}

/// Recursive-backtracker maze on an odd-side grid. Start carves from
/// (1,1); neighbours are checked two cells away (standard wall-between
/// -cells construction).
fn generate_maze(size: usize, seed: u64) -> Maze {
    assert!(size % 2 == 1 && size >= 5, "maze size must be odd and >= 5");
    let mut m = Maze::new(size);
    let mut rng = Rng::new(seed);
    let mut stack = vec![(1usize, 1usize)];
    m.set(1, 1, true);
    while let Some(&(y, x)) = stack.last() {
        // Neighbours two cells away.
        let mut dirs = [(0isize, -2isize), (0, 2), (-2, 0), (2, 0)];
        rng.shuffle(&mut dirs);
        let mut carved = false;
        for &(dy, dx) in &dirs {
            let ny = y as isize + dy;
            let nx = x as isize + dx;
            if ny < 1 || nx < 1 || ny >= (size - 1) as isize || nx >= (size - 1) as isize {
                continue;
            }
            let (ny, nx) = (ny as usize, nx as usize);
            if !m.at(ny, nx) {
                m.set(ny, nx, true);
                // Carve the wall between (y,x) and (ny,nx).
                m.set((y as isize + dy / 2) as usize, (x as isize + dx / 2) as usize, true);
                stack.push((ny, nx));
                carved = true;
                break;
            }
        }
        if !carved { stack.pop(); }
    }
    // Compute shortest path from (1,1) to (size-2, size-2) by BFS.
    m.route = bfs_route(&m);
    m
}

fn bfs_route(m: &Maze) -> Vec<(usize, usize)> {
    let size = m.size;
    let start = (1, 1);
    let goal = (size - 2, size - 2);
    let mut prev = vec![None::<(usize, usize)>; size * size];
    let mut seen = vec![false; size * size];
    let idx = |y: usize, x: usize| y * size + x;
    let mut q = std::collections::VecDeque::new();
    q.push_back(start);
    seen[idx(start.0, start.1)] = true;
    while let Some((y, x)) = q.pop_front() {
        if (y, x) == goal { break; }
        for (dy, dx) in [(0isize, -1isize), (0, 1), (-1, 0), (1, 0)] {
            let ny = y as isize + dy;
            let nx = x as isize + dx;
            if ny < 0 || nx < 0 || ny >= size as isize || nx >= size as isize { continue; }
            let (ny, nx) = (ny as usize, nx as usize);
            if !m.at(ny, nx) || seen[idx(ny, nx)] { continue; }
            seen[idx(ny, nx)] = true;
            prev[idx(ny, nx)] = Some((y, x));
            q.push_back((ny, nx));
        }
    }
    let mut cur = goal;
    let mut out = Vec::new();
    if !seen[idx(goal.0, goal.1)] { return out; } // disconnected — shouldn't happen
    loop {
        out.push(cur);
        match prev[idx(cur.0, cur.1)] {
            Some(p) => cur = p,
            None => break,
        }
    }
    out.reverse();
    out
}

// ────────────────────────────────────────────────────────────────
// Rendering.
// ────────────────────────────────────────────────────────────────

const COL_WALL: [u8; 3] = [30, 30, 40];      // nearly black
const COL_PATH: [u8; 3] = [240, 240, 240];   // near white
const COL_ROUTE: [u8; 3] = [90, 160, 255];   // light blue overlay
const COL_START: [u8; 3] = [90, 220, 110];   // green
const COL_GOAL: [u8; 3] = [255, 90, 90];     // red

fn render(m: &Maze, upscale: usize, draw_route: bool) -> (Vec<u8>, usize, usize) {
    let size = m.size;
    let h = size * upscale;
    let w = size * upscale;
    let mut rgb = vec![0u8; 3 * h * w];

    let mut on_route = vec![false; size * size];
    if draw_route {
        for &(ry, rx) in &m.route { on_route[ry * size + rx] = true; }
    }
    let start = (1usize, 1usize);
    let goal = (size - 2, size - 2);

    for cy in 0..size {
        for cx in 0..size {
            let base = if (cy, cx) == start {
                COL_START
            } else if (cy, cx) == goal {
                COL_GOAL
            } else if on_route[cy * size + cx] {
                COL_ROUTE
            } else if m.at(cy, cx) {
                COL_PATH
            } else {
                COL_WALL
            };
            for dy in 0..upscale {
                for dx in 0..upscale {
                    let py = cy * upscale + dy;
                    let px = cx * upscale + dx;
                    let off = (py * w + px) * 3;
                    rgb[off] = base[0];
                    rgb[off + 1] = base[1];
                    rgb[off + 2] = base[2];
                }
            }
        }
    }
    (rgb, h, w)
}

fn write_ppm(path: &Path, rgb: &[u8], h: usize, w: usize) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "P6\n{w} {h}\n255")?;
    f.write_all(rgb)?;
    Ok(())
}

/// Tile n images into a sqrt(n) × sqrt(n) grid with a 2-pixel border.
fn write_grid(path: &Path, tiles: &[(Vec<u8>, usize, usize)]) -> std::io::Result<()> {
    let n = tiles.len();
    let cols = (n as f32).sqrt().ceil() as usize;
    let rows = (n + cols - 1) / cols;
    let (_, th, tw) = tiles[0];
    for (_, h, w) in tiles { assert_eq!((*h, *w), (th, tw), "tile size mismatch"); }
    let border = 2usize;
    let gh = rows * th + (rows + 1) * border;
    let gw = cols * tw + (cols + 1) * border;
    let mut grid = vec![60u8; 3 * gh * gw]; // dark grey background
    for (i, (rgb, _, _)) in tiles.iter().enumerate() {
        let r = i / cols;
        let c = i % cols;
        let oy = border + r * (th + border);
        let ox = border + c * (tw + border);
        for y in 0..th {
            for x in 0..tw {
                let src = (y * tw + x) * 3;
                let dst = ((oy + y) * gw + (ox + x)) * 3;
                grid[dst] = rgb[src];
                grid[dst + 1] = rgb[src + 1];
                grid[dst + 2] = rgb[src + 2];
            }
        }
    }
    write_ppm(path, &grid, gh, gw)
}

// ────────────────────────────────────────────────────────────────
// Main.
// ────────────────────────────────────────────────────────────────

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut size = 21usize;
    let mut count = 16usize;
    let mut seed = 42u64;
    let mut upscale = 20usize;
    let mut draw_route = true;
    let mut out_dir = String::from("/tmp/maze_viz");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--size" => { size = args[i+1].parse().unwrap(); i += 2; }
            "--count" => { count = args[i+1].parse().unwrap(); i += 2; }
            "--seed" => { seed = args[i+1].parse().unwrap(); i += 2; }
            "--upscale" => { upscale = args[i+1].parse().unwrap(); i += 2; }
            "--out-dir" => { out_dir = args[i+1].clone(); i += 2; }
            "--no-route" => { draw_route = false; i += 1; }
            "--help" | "-h" => {
                eprintln!(
"Usage: maze_viz [--size N] [--count N] [--seed N] [--upscale N]
                [--out-dir PATH] [--no-route]

Renders `count` real mazes via recursive-backtracker generation at
odd-side `size`, with the BFS-optimal route drawn in blue (start green,
goal red). Each cell is drawn as an `upscale × upscale` block of pixels
so the structure is visible in an image viewer.

Default: size=21, count=16, upscale=20, route drawn.

Output:
  <out>/maze_NN.ppm  — individual mazes
  <out>/grid.ppm     — sqrt(count)×sqrt(count) montage with borders
  <out>/meta.txt     — provenance (size, seeds, route lengths)
"); return Ok(());
            }
            _ => { i += 1; }
        }
    }
    if size % 2 == 0 { size += 1; }

    let root = Path::new(&out_dir);
    std::fs::create_dir_all(root)?;
    eprintln!("maze_viz: size={size} count={count} upscale={upscale} seed={seed} route={draw_route}");

    let mut tiles = Vec::with_capacity(count);
    let mut route_lengths = Vec::with_capacity(count);
    for i in 0..count {
        let s = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(0x_DEAD_F00D)
            .wrapping_add(i as u64);
        let m = generate_maze(size, s);
        route_lengths.push(m.route.len());
        let (rgb, h, w) = render(&m, upscale, draw_route);
        write_ppm(&root.join(format!("maze_{i:02}.ppm")), &rgb, h, w)?;
        tiles.push((rgb, h, w));
    }
    write_grid(&root.join("grid.ppm"), &tiles)?;

    let mut f = std::fs::File::create(root.join("meta.txt"))?;
    writeln!(f, "# maze_viz provenance")?;
    writeln!(f, "size         = {size}")?;
    writeln!(f, "count        = {count}")?;
    writeln!(f, "seed         = {seed}")?;
    writeln!(f, "upscale      = {upscale}")?;
    writeln!(f, "draw_route   = {draw_route}")?;
    let (px_h, px_w) = (size * upscale, size * upscale);
    writeln!(f, "image_hw     = {px_h} × {px_w}")?;
    let mean_route = route_lengths.iter().sum::<usize>() as f32 / route_lengths.len() as f32;
    writeln!(f, "\n# route lengths (BFS optimal, start→goal inclusive)")?;
    writeln!(f, "mean         = {mean_route:.1}")?;
    writeln!(f, "min          = {}", route_lengths.iter().min().unwrap_or(&0))?;
    writeln!(f, "max          = {}", route_lengths.iter().max().unwrap_or(&0))?;

    eprintln!("\ndone.");
    eprintln!("  grid:  {}", root.join("grid.ppm").display());
    eprintln!("  meta:  {}", root.join("meta.txt").display());
    Ok(())
}
