//! Maze generation: random solvable mazes with BFS path solutions.
//!
//! Generates grid mazes using randomized DFS (recursive backtracker).
//! Each maze has a start (S) and end (E) position with a known shortest path.
//! Outputs: pixel array [3 × H × W] + direction sequence [Up, Down, Left, Right, Wait].

/// Directions the CTM must predict at each route step.
/// Matches Python CTM: [Up=0, Down=1, Left=2, Right=3, Wait=4]
pub const DIR_UP: usize = 0;
pub const DIR_DOWN: usize = 1;
pub const DIR_LEFT: usize = 2;
pub const DIR_RIGHT: usize = 3;
pub const DIR_WAIT: usize = 4;
pub const N_DIRECTIONS: usize = 5;

/// A generated maze with its solution.
#[derive(Clone)]
pub struct Maze {
    /// Grid: true = wall, false = path. Size: grid_size × grid_size.
    pub grid: Vec<bool>,
    pub grid_size: usize,
    /// Start position (row, col).
    pub start: (usize, usize),
    /// End position (row, col).
    pub end: (usize, usize),
    /// Solution path as direction sequence. Length = path_length.
    pub route: Vec<usize>,
    /// Path length (number of steps from start to end).
    pub path_length: usize,
}

/// Simple LCG RNG.
pub struct MazeRng(u64);

impl MazeRng {
    pub fn new(seed: u64) -> Self { Self(seed) }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }

    pub fn range(&mut self, max: usize) -> usize {
        (self.next() >> 33) as usize % max
    }

    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = self.range(i + 1);
            v.swap(i, j);
        }
    }
}

/// Generate a random solvable maze using randomized DFS (recursive backtracker).
///
/// `size` must be odd (grid cells are at odd coordinates, walls at even).
/// Returns a Maze with guaranteed path from start to end.
pub fn generate_maze(size: usize, rng: &mut MazeRng) -> Maze {
    let size = size | 1; // force odd
    let mut grid = vec![true; size * size]; // all walls

    // Carve passages using iterative DFS with stack
    let start_r = 1;
    let start_c = 1;
    grid[start_r * size + start_c] = false;

    let mut stack = vec![(start_r, start_c)];
    while let Some(&(r, c)) = stack.last() {
        let mut neighbors = Vec::new();
        // Check 4 directions, 2 cells away
        for &(dr, dc) in &[(-2i32, 0), (2, 0), (0, -2), (0, 2)] {
            let nr = r as i32 + dr;
            let nc = c as i32 + dc;
            if nr > 0 && nr < size as i32 - 1 && nc > 0 && nc < size as i32 - 1 {
                let nr = nr as usize;
                let nc = nc as usize;
                if grid[nr * size + nc] {
                    neighbors.push((nr, nc));
                }
            }
        }

        if neighbors.is_empty() {
            stack.pop();
        } else {
            let idx = rng.range(neighbors.len());
            let (nr, nc) = neighbors[idx];
            // Carve wall between current and neighbor
            let wr = (r + nr) / 2;
            let wc = (c + nc) / 2;
            grid[wr * size + wc] = false;
            grid[nr * size + nc] = false;
            stack.push((nr, nc));
        }
    }

    // Pick start and end — opposite corners for decent path length
    let start = (1, 1);
    let end = (size - 2, size - 2);

    // BFS to find shortest path
    let route = bfs_path(&grid, size, start, end);
    let path_length = route.len();

    Maze { grid, grid_size: size, start, end, route, path_length }
}

/// Load a bank of mazes from the binary format written by
/// `/tmp/export_sakana_mazes.py`. Each maze's route is recomputed
/// via BFS (we don't trust any pre-drawn solution).
pub fn load_maze_bank(path: &str) -> std::io::Result<Vec<Maze>> {
    use std::io::Read;
    let mut buf = Vec::new();
    std::fs::File::open(path)?.read_to_end(&mut buf)?;
    if buf.len() < 12 {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "bank too short"));
    }
    let n = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
    let h = u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;
    let w = u32::from_le_bytes(buf[8..12].try_into().unwrap()) as usize;
    if h != w {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
            format!("non-square maze {}x{}", h, w)));
    }
    let size = h;
    let wall_bytes = size * size;
    let rec = wall_bytes + 16;
    if buf.len() < 12 + n * rec {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "bank truncated"));
    }
    let mut mazes = Vec::with_capacity(n);
    for i in 0..n {
        let off = 12 + i * rec;
        let walls = &buf[off..off + wall_bytes];
        let grid: Vec<bool> = walls.iter().map(|&b| b != 0).collect();
        let sy = u32::from_le_bytes(buf[off + wall_bytes..off + wall_bytes + 4].try_into().unwrap()) as usize;
        let sx = u32::from_le_bytes(buf[off + wall_bytes + 4..off + wall_bytes + 8].try_into().unwrap()) as usize;
        let ey = u32::from_le_bytes(buf[off + wall_bytes + 8..off + wall_bytes + 12].try_into().unwrap()) as usize;
        let ex = u32::from_le_bytes(buf[off + wall_bytes + 12..off + wall_bytes + 16].try_into().unwrap()) as usize;
        let start = (sy, sx);
        let end = (ey, ex);
        let route = bfs_path(&grid, size, start, end);
        let path_length = route.len();
        mazes.push(Maze { grid, grid_size: size, start, end, route, path_length });
    }
    Ok(mazes)
}

/// BFS shortest path, returns direction sequence.
fn bfs_path(grid: &[bool], size: usize, start: (usize, usize), end: (usize, usize)) -> Vec<usize> {
    let mut visited = vec![false; size * size];
    let mut parent: Vec<Option<(usize, usize, usize)>> = vec![None; size * size]; // (prev_r, prev_c, direction)
    let mut queue = std::collections::VecDeque::new();

    visited[start.0 * size + start.1] = true;
    queue.push_back(start);

    let dirs: [(i32, i32, usize); 4] = [(-1, 0, DIR_UP), (1, 0, DIR_DOWN), (0, -1, DIR_LEFT), (0, 1, DIR_RIGHT)];

    while let Some((r, c)) = queue.pop_front() {
        if (r, c) == end { break; }
        for &(dr, dc, dir) in &dirs {
            let nr = r as i32 + dr;
            let nc = c as i32 + dc;
            if nr >= 0 && nr < size as i32 && nc >= 0 && nc < size as i32 {
                let nr = nr as usize;
                let nc = nc as usize;
                let idx = nr * size + nc;
                if !grid[idx] && !visited[idx] {
                    visited[idx] = true;
                    parent[idx] = Some((r, c, dir));
                    queue.push_back((nr, nc));
                }
            }
        }
    }

    // Reconstruct path
    let mut path = Vec::new();
    let mut cur = end;
    while cur != start {
        let idx = cur.0 * size + cur.1;
        if let Some((pr, pc, dir)) = parent[idx] {
            path.push(dir);
            cur = (pr, pc);
        } else {
            break; // no path (shouldn't happen with valid maze)
        }
    }
    path.reverse();
    path
}

/// Render maze to pixel array [3 × H × W] with values in [0, 1].
///
/// - Walls: black (0, 0, 0)
/// - Paths: white (1, 1, 1)
/// - Start: red (1, 0, 0)
/// - End: green (0, 1, 0)
/// Process-wide toggle for SDF input encoding. Set once at startup
/// from `main` via `set_render_mode_sdf(true)`; every call to
/// `render_input(&maze)` then dispatches to the SDF variant without
/// threading a bool through every function signature. Use-case is
/// narrow (one flag, set once, never toggled), so a static Atomic is
/// fine — not production architecture, just example wiring.
static RENDER_SDF: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub fn set_render_mode_sdf(on: bool) {
    RENDER_SDF.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Process-wide toggle for retina bypass. Same atomic-static pattern
/// as `RENDER_SDF`. When set, the training loop replaces
/// `encoder.encode(&pixels)` with `bypass_tokens(...)` — raw pixel
/// average-pool in place of the VisualRetina conv stack. Answers "is
/// vision net-positive vs neutral vs net-negative" on the maze task.
static RETINA_BYPASS: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub fn set_retina_bypass(on: bool) {
    RETINA_BYPASS.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn retina_bypass_enabled() -> bool {
    RETINA_BYPASS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Average-pool raw [3 × in_h × in_w] pixels into an
/// [out_h × out_w × 3] grid, then broadcast each RGB triple into a
/// `token_dim`-dim vector (first 3 dims are the colour, rest zero).
/// Produces a TokenInput-shaped `Vec<f32>` of length
/// `out_h * out_w * token_dim`.
///
/// This is the deliberately-minimal "no learning" encoder: the brain
/// sees the exact pixel colours at the spatial resolution the retina
/// would have produced, but with no filters, no non-linearities, no
/// Hebbian shaping. Any performance it achieves is credit to the
/// CTM, not the encoder.
pub fn bypass_tokens(
    pixels: &[f32],
    in_h: usize, in_w: usize,
    out_h: usize, out_w: usize,
    token_dim: usize,
) -> Vec<f32> {
    assert_eq!(pixels.len(), 3 * in_h * in_w,
        "bypass_tokens: pixel buffer size mismatch");
    assert!(token_dim >= 3, "token_dim must hold at least RGB");
    let n_tokens = out_h * out_w;
    let mut out = vec![0.0f32; n_tokens * token_dim];
    // Integer-div pool: each output cell averages the `stride_h × stride_w`
    // block of input pixels that maps to it. For in=21,out=11: mostly
    // 2×2 blocks with edge handling. Exact weighting doesn't matter —
    // this is a *probe* encoder, not a production one.
    for oy in 0..out_h {
        for ox in 0..out_w {
            // Map output cell (oy, ox) to input rectangle.
            let y0 = (oy * in_h) / out_h;
            let y1 = ((oy + 1) * in_h) / out_h;
            let x0 = (ox * in_w) / out_w;
            let x1 = ((ox + 1) * in_w) / out_w;
            let n = ((y1 - y0) * (x1 - x0)).max(1) as f32;
            let mut rgb = [0.0f32; 3];
            for y in y0..y1 {
                for x in x0..x1 {
                    for c in 0..3 {
                        rgb[c] += pixels[c * in_h * in_w + y * in_w + x];
                    }
                }
            }
            let tok_off = (oy * out_w + ox) * token_dim;
            for c in 0..3 {
                out[tok_off + c] = rgb[c] / n;
            }
            // remainder of token_dim stays zero
        }
    }
    out
}

pub fn render_input(maze: &Maze) -> Vec<f32> {
    if RENDER_SDF.load(std::sync::atomic::Ordering::Relaxed) {
        render_maze_sdf(maze)
    } else {
        render_maze(maze)
    }
}

pub fn render_maze(maze: &Maze) -> Vec<f32> {
    let s = maze.grid_size;
    let mut pixels = vec![0.0f32; 3 * s * s];

    for r in 0..s {
        for c in 0..s {
            if !maze.grid[r * s + c] {
                // Path pixel — white
                for ch in 0..3 {
                    pixels[ch * s * s + r * s + c] = 1.0;
                }
            }
            // Walls stay black (0.0)
        }
    }

    // Start: red
    let (sr, sc) = maze.start;
    pixels[0 * s * s + sr * s + sc] = 1.0; // R
    pixels[1 * s * s + sr * s + sc] = 0.0; // G
    pixels[2 * s * s + sr * s + sc] = 0.0; // B

    // End: green
    let (er, ec) = maze.end;
    pixels[0 * s * s + er * s + ec] = 0.0; // R
    pixels[1 * s * s + er * s + ec] = 1.0; // G
    pixels[2 * s * s + er * s + ec] = 0.0; // B

    pixels
}

/// Render maze with a signed-distance-field encoding instead of flat
/// wall/path pixels. BFS wall-distance is computed for every cell;
/// path cells carry their normalized distance as luminance, wall cells
/// stay at 0. Start/end still painted red/green as single-channel markers
/// so the brain can find goal locations. Shape is still [3 × H × W] —
/// the retina is unchanged, only the pixel distribution changes.
///
/// Motivation: the standard renderer wastes V1's bio-filters by making
/// all path cells identical. SDF gives every path cell a distinct value
/// proportional to wall-gap, so center-surround filters in V1 extract
/// navigational structure directly rather than leaving it entirely to
/// the learned V2/V4 cortex.
///
/// Analog: peripheral-vision distance cues in biology — parasol cells,
/// stereo disparity — the retina has evolved priors beyond simple RGB.
/// The 3rd channel here is "wall-distance sense" painted into the same
/// tensor shape so no retina surgery is required for an A/B test.
pub fn render_maze_sdf(maze: &Maze) -> Vec<f32> {
    let s = maze.grid_size;
    let n = s * s;
    let mut pixels = vec![0.0f32; 3 * n];

    // BFS wall-distance: seed all walls at 0, expand outward.
    let mut dist = vec![u32::MAX; n];
    let mut q = std::collections::VecDeque::new();
    for r in 0..s {
        for c in 0..s {
            if maze.grid[r * s + c] {
                dist[r * s + c] = 0;
                q.push_back((r, c));
            }
        }
    }
    while let Some((r, c)) = q.pop_front() {
        let d = dist[r * s + c];
        for (dr, dc) in [(-1isize, 0isize), (1, 0), (0, -1), (0, 1)] {
            let nr = r as isize + dr;
            let nc = c as isize + dc;
            if nr < 0 || nc < 0 || nr >= s as isize || nc >= s as isize { continue; }
            let (nr, nc) = (nr as usize, nc as usize);
            if dist[nr * s + nc] == u32::MAX {
                dist[nr * s + nc] = d + 1;
                q.push_back((nr, nc));
            }
        }
    }
    let max_d = dist.iter().filter(|&&d| d != u32::MAX).copied().max().unwrap_or(1).max(1);

    for r in 0..s {
        for c in 0..s {
            let idx = r * s + c;
            if !maze.grid[idx] {
                // Path cell — normalized wall-distance as luminance.
                // 0.1 floor so it never equals wall=0 and stays bio-plausible.
                let v = 0.1 + 0.9 * (dist[idx] as f32 / max_d as f32);
                for ch in 0..3 {
                    pixels[ch * n + idx] = v;
                }
            }
        }
    }

    let (sr, sc) = maze.start;
    pixels[sr * s + sc] = 1.0; // R
    pixels[n + sr * s + sc] = 0.0;
    pixels[2 * n + sr * s + sc] = 0.0;

    let (er, ec) = maze.end;
    pixels[er * s + ec] = 0.0;
    pixels[n + er * s + ec] = 1.0;
    pixels[2 * n + er * s + ec] = 0.0;

    pixels
}
