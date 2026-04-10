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
