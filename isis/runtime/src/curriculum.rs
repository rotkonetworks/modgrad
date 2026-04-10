//! Byte-first staged curriculum for computational organisms.
//!
//! Unlike human pedagogy, this curriculum starts from the organism's
//! native substrate: raw bytes (0-255). Each phase builds on the
//! previous, earning the right to higher abstractions.
//!
//! Phase 0: Byte identity — learn to distinguish individual byte values
//! Phase 1: Byte classification — ASCII structure (letters, digits, space, punctuation)
//! Phase 2: Byte sequences — common bigrams and trigrams in English
//! Phase 3: Word boundaries — space (32) separates words, newline (10) separates lines
//! Phase 4: Simple words — 2-4 letter common English words
//! Phase 5: Word pairs — article+noun, adjective+noun
//! Phase 6: Sentences — subject-verb-object patterns
//! Phase 7: Stories — full narrative structure
//!
//! Mastery criteria per phase:
//! - Loss below threshold for N consecutive steps
//! - The organism must EARN advancement

/// How many phases there are.
pub const NUM_PHASES: usize = 8;

/// Phase names.
pub const PHASE_NAMES: [&str; NUM_PHASES] = [
    "byte_identity",
    "byte_classify",
    "byte_sequence",
    "word_boundary",
    "simple_words",
    "word_pairs",
    "sentences",
    "stories",
];

/// Loss threshold to advance from each phase.
/// Tuned for RegionalCtm with BPTT — converges slower on simple tasks
/// but generalizes better across phases.
pub const MASTERY_THRESHOLDS: [f32; NUM_PHASES] = [
    3.5,   // Phase 0: byte identity — learn byte distributions
    3.5,   // Phase 1: byte classification — ASCII structure
    3.5,   // Phase 2: byte sequences — bigram prediction
    3.5,   // Phase 3: word boundaries — space prediction
    3.5,   // Phase 4: simple words — small vocab
    4.0,   // Phase 5: word pairs — combinatorics
    4.5,   // Phase 6: sentences — structure
    5.0,   // Phase 7: stories — narratives
];

/// How many consecutive steps below threshold to advance.
pub const MASTERY_STREAK: usize = 10;

/// Generate curriculum bytes for a given phase.
/// Returns (phase_name, data).
pub fn generate(phase: usize, repetitions: usize) -> (&'static str, Vec<u8>) {
    let p = phase.min(NUM_PHASES - 1);
    (PHASE_NAMES[p], match p {
        0 => phase_byte_identity(repetitions),
        1 => phase_byte_classify(repetitions),
        2 => phase_byte_sequence(repetitions),
        3 => phase_word_boundary(repetitions),
        4 => phase_simple_words(repetitions),
        5 => phase_word_pairs(repetitions),
        6 => phase_sentences(repetitions),
        7 => phase_stories(repetitions),
        _ => phase_stories(repetitions),
    })
}

/// Load curriculum from external JSON file.
/// Format: [{"phase": N, "data": "..."}, ...]
pub fn load_external(path: &str) -> Result<Vec<(usize, Vec<u8>)>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read {path}: {e}"))?;
    let items: Vec<serde_json::Value> = serde_json::from_str(&text)
        .map_err(|e| format!("invalid JSON: {e}"))?;

    let mut result = Vec::new();
    for item in &items {
        let phase = item["phase"].as_u64().unwrap_or(0) as usize;
        if let Some(data) = item["data"].as_str() {
            result.push((phase, data.as_bytes().to_vec()));
        }
    }
    Ok(result)
}

// ─── Phase 0: Byte Identity ─────────────────────────────────
// Teach: given byte X, the next byte is X.
// "Learn to see" — each byte value must produce a distinct embedding.
// Training data: repeated single bytes.
// After this phase, the 256 embeddings should be meaningfully different.

fn phase_byte_identity(reps: usize) -> Vec<u8> {
    let mut data = Vec::new();
    for _ in 0..reps {
        // All printable ASCII bytes, repeated
        for b in 32u8..=126 {
            // Repeat same byte 8 times: the organism learns "X predicts X"
            for _ in 0..8 {
                data.push(b);
            }
            data.push(b'\n');
        }
        // Also digits and letters in runs
        // "aaaa" "bbbb" "cccc" — massive repetition per byte
        for b in b'a'..=b'z' {
            for _ in 0..16 { data.push(b); }
            data.push(b'\n');
        }
        for b in b'A'..=b'Z' {
            for _ in 0..16 { data.push(b); }
            data.push(b'\n');
        }
        for b in b'0'..=b'9' {
            for _ in 0..16 { data.push(b); }
            data.push(b'\n');
        }
    }
    data
}

// ─── Phase 1: Byte Classification ───────────────────────────
// Teach: ASCII has structure. Letters form a range. Digits form a range.
// Space is special. Punctuation is special.
// Training data: alternating category blocks.
// "aeiou" then "01234" then ".,;:!" — learn that categories exist.

fn phase_byte_classify(reps: usize) -> Vec<u8> {
    let mut data = Vec::new();

    // Vowels block
    let vowels = b"aeiouAEIOU";
    // Consonants block
    let consonants = b"bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ";
    // Digits block
    let digits = b"0123456789";
    // Punctuation block
    let punct = b".,;:!?-'\"()";
    // Whitespace block
    let spaces = b"    \t\n  \n  ";

    for _ in 0..reps {
        // Category runs: organism sees "vowels predict vowels", "digits predict digits"
        for _ in 0..8 {
            data.extend_from_slice(vowels);
        }
        data.push(b'\n');
        for _ in 0..8 {
            data.extend_from_slice(consonants);
        }
        data.push(b'\n');
        for _ in 0..12 {
            data.extend_from_slice(digits);
        }
        data.push(b'\n');
        for _ in 0..12 {
            data.extend_from_slice(punct);
        }
        data.push(b'\n');
        for _ in 0..12 {
            data.extend_from_slice(spaces);
        }
        data.push(b'\n');

        // Alternating categories: "aaa111bbb222" — learn transitions
        for b in b'a'..=b'z' {
            data.push(b);
            data.push(b);
            data.push(b);
            let d = b'0' + (b - b'a') % 10;
            data.push(d);
            data.push(d);
            data.push(d);
        }
        data.push(b'\n');

        // Case pairs: "aAbBcCdD" — learn upper/lower relationship
        for b in b'a'..=b'z' {
            data.push(b);
            data.push(b - 32); // uppercase
        }
        data.push(b'\n');
    }
    data
}

// ─── Phase 2: Byte Sequences ────────────────────────────────
// Teach: common byte transitions in English.
// "th" "he" "in" "er" "an" — the most frequent bigrams.
// After this, the organism can predict likely next bytes.

fn phase_byte_sequence(reps: usize) -> Vec<u8> {
    let mut data = Vec::new();

    // Top English bigrams by frequency
    let bigrams = [
        "th", "he", "in", "er", "an", "re", "on", "at",
        "en", "nd", "ti", "es", "or", "te", "of", "ed",
        "is", "it", "al", "ar", "st", "to", "nt", "ng",
        "se", "ha", "as", "ou", "io", "le", "ve", "co",
        "me", "de", "hi", "ri", "ro", "ic", "ne", "ea",
        "ra", "ce", "li", "ch", "ll", "be", "ma", "si",
    ];

    // Top trigrams
    let trigrams = [
        "the", "and", "ing", "her", "hat", "his", "tha",
        "ere", "for", "ent", "ion", "ter", "was", "you",
        "ith", "ver", "all", "wit", "thi", "tio", "tion",
        "are", "not", "rea", "eve", "ess", "ons", "ome",
    ];

    for _ in 0..reps {
        // Bigram drills: massive repetition
        for bg in &bigrams {
            for _ in 0..20 {
                data.extend_from_slice(bg.as_bytes());
            }
            data.push(b'\n');
        }

        // Trigram drills
        for tg in &trigrams {
            for _ in 0..16 {
                data.extend_from_slice(tg.as_bytes());
            }
            data.push(b'\n');
        }

        // Common byte transitions with space
        // "t h" appears because "t" often ends a word before "h" starts the next
        let spaced = [
            "t h", "s t", "n t", "d t", "e a", "e i",
            "f t", "n a", "s a", "d a", "e t", "y t",
        ];
        for sp in &spaced {
            for _ in 0..16 {
                data.extend_from_slice(sp.as_bytes());
            }
            data.push(b'\n');
        }
    }
    data
}

// ─── Phase 3: Word Boundaries ───────────────────────────────
// Teach: space (32) separates words. Newline (10) separates lines.
// After letters come spaces. After spaces come letters.
// The organism learns the RHYTHM of text: letters-space-letters-space.

fn phase_word_boundary(reps: usize) -> Vec<u8> {
    let mut data = Vec::new();

    for _ in 0..reps {
        // Simple: "xxx yyy zzz" patterns where x/y/z are consistent letter runs
        // Teaches: after N letters, expect a space
        for len in 1..=6 {
            for base in [b'a', b'b', b'c', b'd', b'e', b'f'] {
                for _ in 0..10 {
                    for _ in 0..len {
                        data.push(base);
                    }
                    data.push(b' ');
                }
                data.push(b'\n');
            }
        }

        // "ab cd ef gh" — 2-letter chunks separated by space
        for _ in 0..8 {
            for pair in ["ab", "cd", "ef", "gh", "ij", "kl", "mn", "op"] {
                data.extend_from_slice(pair.as_bytes());
                data.push(b' ');
            }
            data.push(b'\n');
        }

        // Punctuation as boundaries: "abc. def. ghi."
        for _ in 0..8 {
            for word in ["abc", "def", "ghi", "jkl", "mno"] {
                data.extend_from_slice(word.as_bytes());
                data.extend_from_slice(b". ");
            }
            data.push(b'\n');
        }

        // Real word-length patterns (without real words yet)
        // 3-letter runs: most common English word length
        let three = [
            "aaa bbb ccc ddd eee fff ggg hhh\n",
            "xxx yyy zzz aaa bbb ccc ddd eee\n",
        ];
        for t in &three {
            for _ in 0..6 {
                data.extend_from_slice(t.as_bytes());
            }
        }
    }
    data
}

// ─── Phase 4: Simple Words ──────────────────────────────────
// First real words. 1-4 letters, extremely common.
// Massive repetition per word. The organism learns
// "t-h-e" is a unit, "c-a-t" is a unit.

fn phase_simple_words(reps: usize) -> Vec<u8> {
    let mut data = Vec::new();

    // Ordered by frequency and length
    let words_1 = ["a", "i"];
    let words_2 = ["is", "it", "in", "on", "to", "he", "at", "an", "no", "so",
                    "do", "we", "or", "up", "if", "my", "go", "me", "be"];
    let words_3 = ["the", "and", "for", "are", "but", "not", "you", "all",
                    "can", "had", "her", "was", "one", "our", "out", "has",
                    "his", "how", "its", "may", "new", "now", "old", "see",
                    "way", "who", "did", "get", "let", "say", "she", "too",
                    "use", "cat", "dog", "big", "red", "run", "sat", "hat"];
    let words_4 = ["that", "with", "have", "this", "will", "your", "from",
                    "they", "been", "said", "each", "make", "like", "just",
                    "over", "such", "good", "very", "when", "come", "more",
                    "some", "them", "than", "word", "what"];

    for _ in 0..reps {
        // Single repetition blocks: "the the the the the the\n"
        for word in words_1.iter()
            .chain(words_2.iter())
            .chain(words_3.iter())
            .chain(words_4.iter())
        {
            for _ in 0..16 {
                data.extend_from_slice(word.as_bytes());
                data.push(b' ');
            }
            data.push(b'\n');
        }
    }
    data
}

// ─── Phase 5: Word Pairs ────────────────────────────────────

fn phase_word_pairs(reps: usize) -> Vec<u8> {
    let pairs = [
        "the cat", "the dog", "the bird", "the fish",
        "a cat", "a dog", "a bird", "a fish",
        "the boy", "the girl", "the tree", "the house",
        "big cat", "small dog", "red bird", "old tree",
        "blue ball", "new book", "big house", "small fish",
        "is good", "is bad", "is big", "is old", "is new",
        "can run", "can see", "can go", "can fly", "can swim",
        "to the", "in the", "on the", "at the", "for the",
        "he said", "she said", "it was", "we are", "they had",
    ];

    let mut data = Vec::new();
    for _ in 0..reps {
        for pair in &pairs {
            for _ in 0..12 {
                data.extend_from_slice(pair.as_bytes());
                data.extend_from_slice(b". ");
            }
            data.push(b'\n');
        }
    }
    data
}

// ─── Phase 6: Sentences ─────────────────────────────────────

fn phase_sentences(reps: usize) -> Vec<u8> {
    let sentences = [
        "the cat sat",
        "the dog ran",
        "the bird flew",
        "the fish swam",
        "the boy walked",
        "the girl jumped",
        "the cat sat on the mat",
        "the dog ran in the park",
        "the bird flew over the tree",
        "the big cat sat on the mat",
        "the small dog ran in the park",
        "a red bird flew over the house",
        "the old fish swam in the river",
        "the boy and the girl walked",
        "the cat and the dog sat",
        "he said it was good",
        "she said it was not good",
        "they had a big dog",
        "we can see the sun",
        "i can go to the house",
    ];

    let mut data = Vec::new();
    for _ in 0..reps {
        for s in &sentences {
            for _ in 0..8 {
                data.extend_from_slice(s.as_bytes());
                data.extend_from_slice(b". ");
            }
            data.push(b'\n');
        }
    }
    data
}

// ─── Phase 7: Stories ───────────────────────────────────────

fn phase_stories(reps: usize) -> Vec<u8> {
    let stories = [
        "once upon a time there was a cat. the cat was big. the cat sat on the mat. the cat was happy. the end.",
        "once upon a time there was a dog. the dog was small. the dog ran in the park. the dog was happy. the end.",
        "once upon a time there was a bird. the bird was red. the bird flew over the tree. the bird was happy. the end.",
        "the boy had a cat. the cat was his friend. one day the cat sat on the boy. the boy said no. the cat jumped. the end.",
        "the girl had a dog. the dog was her friend. one day the dog ran to the girl. the girl was happy. the end.",
        "there was a big tree. a bird sat in the tree. the bird sang a song. the sun was bright. it was a good day.",
        "the cat and the dog were friends. one day they went to the park. the cat sat and the dog ran. they were happy.",
    ];

    let mut data = Vec::new();
    for _ in 0..reps {
        for s in &stories {
            data.extend_from_slice(s.as_bytes());
            data.push(b'\n');
        }
    }
    data
}
