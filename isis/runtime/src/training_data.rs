// ── Training data generators ──────────────────────────────
//
// Each stage gets optimized training data that teaches exactly
// the capability being tested. No wasted tokens.

/// Stage 1 data: ASCII character class transitions.
/// Teaches: uppercase→lowercase, space→letter, period→space, digit contexts.
pub fn generate_byte_class_data(n_bytes: usize) -> Vec<u8> {
    let patterns: &[&[u8]] = &[
        b"Hello World This Is Text ",
        b"The Quick Brown Fox Jumps ",
        b"age 25, score 100, year 2024 ",
        b"count 1, count 2, count 3 ",
        b"yes. no. wait! really? ok. ",
        b"first, second, third. ",
        b"line one.\nline two.\nline three.\n",
        b"Name: John, Age: 30.\nCity: NYC.\n",
        b"Status: OK. Code: 200.\n",
    ];
    cycle_patterns(patterns, n_bytes)
}

/// Stage 2 data: common English bigrams.
/// Teaches: th→e, he→ , in→g, an→d, er→ , on→ , re→ , at→ , ou→r, is→ , to→ .
pub fn generate_bigram_data(n_bytes: usize) -> Vec<u8> {
    let patterns: &[&[u8]] = &[
        b"the then them there these through ",
        b"and another ancient android ",
        b"thing being ring sing string ",
        b"our out outer ours around ",
        b"there are here are more ",
        b"better after under over ever ",
        b"at that cat bat mat hat ",
        b"to the to them to this ",
        b"enter entire went bent sent ",
        b"is it is there is then ",
        b"on one stone honor alone ",
        b"the thing that the theory ",
        b"he then he them he there ",
    ];
    cycle_patterns(patterns, n_bytes)
}

/// Stage 3 data: common English word transitions.
/// Teaches: "the "→letter, "is "→letter, "and "→letter, "of "→"t", etc.
pub fn generate_word_data(n_bytes: usize) -> Vec<u8> {
    let patterns: &[&[u8]] = &[
        b"the cat sat on the mat. the dog ran to the park. ",
        b"it is a good day. she is not here. he is very tall. ",
        b"and then and after and before and again. ",
        b"of the best of a kind of my own. ",
        b"to the store to a friend to be or not to be. ",
        b"in the morning in a house in my room. ",
        b"for the first time for a moment for the best. ",
        b"the bird sat on the tree. the fish swam in the sea. ",
        b"i can see the sun. you can hear the wind. we can feel the rain. ",
        b"she said hello. he said goodbye. they said nothing. ",
        b"the old man sat by the fire. the young girl read a book. ",
        b"it was cold and dark. the snow fell on the ground. ",
    ];
    cycle_patterns(patterns, n_bytes)
}

/// Stage 4 data: coherent multi-sentence text.
/// Teaches: sentence structure, paragraph flow, narrative.
pub fn generate_coherent_data(n_bytes: usize) -> Vec<u8> {
    let patterns: &[&[u8]] = &[
        b"the cat sat on the mat. it was a warm day. the sun was bright and the sky was blue. ",
        b"the house was old and grey. it had a red door and two small windows. the garden was green. ",
        b"hello said the boy. how are you asked the girl. i am fine he said. that is good she said. ",
        b"water is wet. fire is hot. ice is cold. the sky is blue. grass is green. snow is white. ",
        b"the dog ran fast. it ran down the hill and into the field. the boy ran after it. ",
        b"she went to the store. she bought some bread and milk. then she went home. ",
        b"the bird sang a song. it was a beautiful morning. the flowers were in bloom. ",
        b"he opened the door and looked outside. it was raining. he closed the door and sat down. ",
        b"the children played in the park. they ran and jumped and laughed. it was a good day. ",
        b"the moon rose over the hill. the stars came out one by one. the night was quiet and still. ",
        b"once there was a small town by the sea. the people were kind and the food was good. ",
        b"the teacher asked a question. the student thought for a moment. then she gave the answer. ",
    ];
    cycle_patterns(patterns, n_bytes)
}

/// Stage 5 data: ASCII maze solving.
/// Teaches: spatial reasoning from sequential byte input.
///
/// Format: maze grid (# = wall, . = path, S = start, E = end) followed by
/// a newline and the solution route as direction characters (U/D/L/R).
///
/// Example:
///   #####
///   #S..#
///   #.#.#
///   #..E#
///   #####
///   RRDDRD
///
/// The CTM must learn to predict the route bytes given the maze layout.
pub fn generate_maze_data(n_bytes: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n_bytes);
    let mut rng = 42u64;
    let size = 11; // small mazes for byte-level training

    while out.len() < n_bytes {
        let maze = gen_text_maze(size, &mut rng);
        out.extend_from_slice(maze.as_bytes());
        out.push(b'\n');
    }
    out.truncate(n_bytes);
    out
}

/// Generate one ASCII maze with its solution route.
fn gen_text_maze(size: usize, rng: &mut u64) -> String {
    let size = size | 1; // force odd
    let mut grid = vec![true; size * size]; // all walls

    let next_rng = |rng: &mut u64| -> u64 {
        *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *rng
    };

    // Carve passages: iterative DFS
    let start = (1, 1);
    grid[start.0 * size + start.1] = false;
    let mut stack = vec![start];

    while let Some(&(r, c)) = stack.last() {
        let mut neighbors = Vec::new();
        for &(dr, dc) in &[(-2i32, 0), (2, 0), (0, -2), (0, 2)] {
            let nr = r as i32 + dr;
            let nc = c as i32 + dc;
            if nr > 0 && nr < size as i32 - 1 && nc > 0 && nc < size as i32 - 1 {
                let (nr, nc) = (nr as usize, nc as usize);
                if grid[nr * size + nc] { neighbors.push((nr, nc)); }
            }
        }
        if neighbors.is_empty() {
            stack.pop();
        } else {
            let idx = (next_rng(rng) >> 33) as usize % neighbors.len();
            let (nr, nc) = neighbors[idx];
            grid[(r + nr) / 2 * size + (c + nc) / 2] = false;
            grid[nr * size + nc] = false;
            stack.push((nr, nc));
        }
    }

    let end = (size - 2, size - 2);

    // BFS shortest path
    let mut visited = vec![false; size * size];
    let mut parent: Vec<Option<(usize, usize, u8)>> = vec![None; size * size];
    let mut queue = std::collections::VecDeque::new();
    visited[start.0 * size + start.1] = true;
    queue.push_back(start);

    let dirs: [(i32, i32, u8); 4] = [(-1, 0, b'U'), (1, 0, b'D'), (0, -1, b'L'), (0, 1, b'R')];
    while let Some((r, c)) = queue.pop_front() {
        if (r, c) == end { break; }
        for &(dr, dc, ch) in &dirs {
            let nr = r as i32 + dr;
            let nc = c as i32 + dc;
            if nr >= 0 && nr < size as i32 && nc >= 0 && nc < size as i32 {
                let (nr, nc) = (nr as usize, nc as usize);
                let idx = nr * size + nc;
                if !grid[idx] && !visited[idx] {
                    visited[idx] = true;
                    parent[idx] = Some((r, c, ch));
                    queue.push_back((nr, nc));
                }
            }
        }
    }

    // Reconstruct route
    let mut route = Vec::new();
    let mut cur = end;
    while cur != start {
        let idx = cur.0 * size + cur.1;
        if let Some((pr, pc, ch)) = parent[idx] {
            route.push(ch);
            cur = (pr, pc);
        } else { break; }
    }
    route.reverse();

    // Render as ASCII
    let mut s = String::with_capacity(size * (size + 1) + route.len() + 2);
    for r in 0..size {
        for c in 0..size {
            if (r, c) == start { s.push('S'); }
            else if (r, c) == end { s.push('E'); }
            else if grid[r * size + c] { s.push('#'); }
            else { s.push('.'); }
        }
        s.push('\n');
    }
    s.push_str(std::str::from_utf8(&route).unwrap_or(""));
    s.push('\n');
    s
}

/// Helper: cycle through patterns to fill n_bytes.
fn cycle_patterns(patterns: &[&[u8]], n_bytes: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n_bytes);
    let mut i = 0;
    while out.len() < n_bytes {
        let p = patterns[i % patterns.len()];
        let remaining = n_bytes - out.len();
        if remaining >= p.len() {
            out.extend_from_slice(p);
        } else {
            out.extend_from_slice(&p[..remaining]);
        }
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_generators_produce_correct_size() {
        assert_eq!(generate_byte_class_data(10000).len(), 10000);
        assert_eq!(generate_bigram_data(10000).len(), 10000);
        assert_eq!(generate_word_data(10000).len(), 10000);
        assert_eq!(generate_coherent_data(10000).len(), 10000);
    }

    #[test]
    fn maze_data_generates_valid_mazes() {
        let data = generate_maze_data(2000);
        let text = std::str::from_utf8(&data).unwrap();
        // Should contain maze chars and route chars
        assert!(text.contains('#'));
        assert!(text.contains('S'));
        assert!(text.contains('E'));
        assert!(text.contains('.'));
        // Should contain at least one route direction
        assert!(text.contains('U') || text.contains('D') || text.contains('L') || text.contains('R'));
        eprintln!("Maze sample:\n{}", &text[..text.len().min(400)]);
    }

    #[test]
    fn data_is_valid_utf8() {
        let _ = std::str::from_utf8(&generate_byte_class_data(1000)).unwrap();
        let _ = std::str::from_utf8(&generate_bigram_data(1000)).unwrap();
        let _ = std::str::from_utf8(&generate_word_data(1000)).unwrap();
        let _ = std::str::from_utf8(&generate_coherent_data(1000)).unwrap();
    }

    #[test]
    fn data_is_deterministic() {
        assert_eq!(generate_bigram_data(5000), generate_bigram_data(5000));
    }

    #[test]
    fn bigram_data_contains_targets() {
        let text = String::from_utf8(generate_bigram_data(10000)).unwrap();
        assert!(text.contains("the"));
        assert!(text.contains("and"));
        assert!(text.contains("ing"));
        assert!(text.contains("our"));
    }
}
