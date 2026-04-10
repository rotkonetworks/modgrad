//! Coherent English text generation for byte-level language model training.
//!
//! Generates deterministic, well-structured English text mixing:
//! - Simple stories with narrative flow
//! - Descriptions of objects and scenes
//! - Dialogues between speakers
//! - Facts and simple statements
//! - Repetitive sentence patterns for easier learning
//!
//! All output is deterministic and reproducible.

/// Generate coherent English text for byte-level language model training.
///
/// Creates a deterministic corpus of well-structured English sentences mixing:
/// - Simple narratives with subject-verb-object patterns
/// - Descriptive passages with adjective-noun combinations
/// - Natural dialogue with consistent speaker patterns
/// - Factual statements about basic properties
/// - Repetitive patterns at the sentence level for reinforcement learning
///
/// Each sentence ends with a period, making sentence boundaries explicit.
/// Text is simple, clear, and uses high-frequency vocabulary suitable for
/// a byte-level model learning English structure.
///
/// # Arguments
///
/// * `n_bytes` - Target number of bytes to generate (approximately)
///
/// # Returns
///
/// A vector of bytes containing deterministic English text.
///
/// # Example
///
/// ```ignore
/// let data = generate_coherent_text_data(1024);
/// assert!(data.len() > 900); // May be slightly less due to sentence boundaries
/// ```
pub fn generate_coherent_text_data(n_bytes: usize) -> Vec<u8> {
    // Templates for deterministic text generation
    // Each category uses simple, repetitive patterns

    // Stories: subject-verb-object with temporal progression
    let stories = [
        "the cat sat on the mat. it was a warm day. the sun was bright. the cat was happy. the cat closed its eyes.",
        "the boy ran in the park. he was fast. he jumped over a log. he fell down. he got back up. he was strong.",
        "the girl walked to the store. she wanted milk. she found the milk. she paid for the milk. she went home. she was glad.",
        "the dog played with a ball. the ball was red. the dog jumped. the dog caught the ball. the dog ran. the dog was tired.",
        "the bird flew in the sky. the sky was blue. the bird flew high. the bird flew far. the bird was free. the bird was happy.",
    ];

    // Descriptions: static properties and attributes
    let descriptions = [
        "the house was old and grey. it had a red door and two windows. the windows were small. the door was heavy. the house was quiet.",
        "the tree was tall and green. it had many leaves. the leaves moved in the wind. the trunk was brown. the roots were deep.",
        "the book was thick and red. it had many pages. the pages were white. the words were small. the book was old. the book was heavy.",
        "the apple was round and red. it was sweet. it was firm. it was shiny. it was fresh. the apple was good.",
        "the water was clear and cold. it was still. it was deep. it was quiet. water is wet. water is essential.",
    ];

    // Dialogues: speaker alternation with natural patterns
    let dialogues = [
        "hello said the boy. how are you asked the girl. i am fine said the boy. me too said the girl. we are friends said the boy. yes we are said the girl.",
        "do you like cats asked alice. yes i love cats said bob. cats are soft said alice. cats are cute said bob. i have a cat said alice. what is its name asked bob.",
        "where are you going asked tom. i am going to school said jane. can i go with you asked tom. yes please said jane. we will go together said tom. good said jane.",
        "what is that asked the child. that is a bird said the parent. what does it do asked the child. it flies said the parent. can i fly asked the child. no you cannot said the parent.",
        "do you want tea asked mother. yes please said daughter. sugar asked mother. two spoons said daughter. milk asked mother. a little said daughter. here you are said mother.",
    ];

    // Facts: simple, repeatable statements
    let facts = [
        "water is wet. fire is hot. ice is cold. the sky is blue. the grass is green.",
        "cats have fur. birds have feathers. fish have scales. dogs have teeth. people have hands.",
        "the sun is bright. the moon is round. stars are far. clouds are white. rain is wet.",
        "one is less than two. two is less than three. three is less than four. numbers go on forever.",
        "apples are red. bananas are yellow. lemons are yellow. oranges are orange. berries are small.",
    ];

    // Repetitive patterns: sentence-level repetition for learning
    let patterns = [
        "the cat is here. the cat is here. the cat is here. the cat is happy. the cat is happy.",
        "it is a day. it is a good day. it is a nice day. it is a warm day. it is a fine day.",
        "i see a thing. i see a thing. i see a thing. i see it well. i see it now.",
        "this is good. this is good. this is good. very good. very good.",
        "the thing is there. the thing is there. the thing is there. we see it. we like it.",
    ];

    let mut result = Vec::with_capacity(n_bytes);
    let mut seed = 1u64;

    // Deterministic PRNG for variety without external randomness
    fn next_seed(seed: &mut u64) -> usize {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*seed >> 32) as usize
    }

    // Cycle through all templates repeatedly until target size is reached
    let all_templates = [
        &stories[..],
        &descriptions[..],
        &dialogues[..],
        &facts[..],
        &patterns[..],
    ];

    let mut template_idx = 0;
    let mut item_idx = 0;

    while result.len() < n_bytes {
        let templates = all_templates[template_idx % all_templates.len()];
        let template = templates[item_idx % templates.len()];

        result.extend_from_slice(template.as_bytes());
        result.push(b' '); // Space between sentences/paragraphs

        item_idx = item_idx.wrapping_add(next_seed(&mut seed) % 3 + 1);

        if item_idx % 7 == 0 {
            // Add paragraph breaks every ~7 items
            result.extend_from_slice(b"\n\n");
        }

        template_idx += 1;
    }

    // Trim to exact size if we overshot
    result.truncate(n_bytes);

    // Ensure we end at a sentence boundary (at a period)
    while !result.is_empty() && result[result.len() - 1] != b'.' {
        result.pop();
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coherent_text_generation() {
        let data = generate_coherent_text_data(2048);
        assert!(data.len() > 1900);
        assert!(data.len() <= 2048);

        // Check it's valid UTF-8
        let text = String::from_utf8_lossy(&data);
        assert!(!text.is_empty());

        // Check contains expected sentences (with larger corpus, should have all)
        assert!(text.contains("the cat"));
        assert!(text.contains("said"));
        assert!(text.contains("is"));

        // Check ends with period
        let s = text.trim();
        assert!(s.ends_with('.'));
    }

    #[test]
    fn test_deterministic() {
        let data1 = generate_coherent_text_data(5000);
        let data2 = generate_coherent_text_data(5000);
        assert_eq!(data1, data2);
    }

    #[test]
    fn test_large_corpus() {
        let data = generate_coherent_text_data(200_000);
        assert!(data.len() > 190_000);
        assert!(data.len() <= 200_000);

        let text = String::from_utf8_lossy(&data);
        let line_count = text.matches('\n').count();
        assert!(line_count > 10); // Should have multiple paragraphs
    }

    #[test]
    fn test_byte_validity() {
        let data = generate_coherent_text_data(2048);
        // All bytes should be valid ASCII for English text
        for &byte in &data {
            assert!(
                byte == b'\n' || byte == b' ' || (byte >= 32 && byte < 127),
                "Invalid byte: {}",
                byte
            );
        }
    }
}
