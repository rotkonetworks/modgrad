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

    fn range(&mut self, max: usize) -> usize {
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
