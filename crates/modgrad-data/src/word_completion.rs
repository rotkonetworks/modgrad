//! Byte-level training data for word completion tasks.
//!
//! Generates high-frequency English word transitions suitable for training
//! byte-level language models to predict common continuations.

/// Generate deterministic training data for word completion tasks.
///
/// Creates dense text with very high frequency of common English word transitions:
/// - "the cat", "the dog", "the mat", etc.
/// - "is a", "is good", "is very", etc.
/// - "of the", "of a", "of my", etc.
/// - Simple SVO sentences with common transitional patterns
///
/// # Arguments
/// * `n_bytes` - Target size in bytes (approximately, may be slightly less)
///
/// # Returns
/// A `Vec<u8>` containing UTF-8 encoded text with deterministic repetition of target patterns.
///
/// # Example
/// ```no_run
/// let data = generate_word_completion_data(1024);
/// assert!(data.len() <= 1024);
/// ```
pub fn generate_word_completion_data(n_bytes: usize) -> Vec<u8> {
    let mut output = Vec::with_capacity(n_bytes);

    // Core transitional patterns - each is carefully chosen to provide
    // dense training signal for byte-level models
    let patterns = [
        // "the" patterns (most frequent word in English)
        "the cat sat on the mat. ",
        "the dog ran to the park. ",
        "the bird flew over the house. ",
        "the tree swayed in the wind. ",
        "the sun warmed the earth. ",
        "the moon lit the night. ",
        "the river flowed to the sea. ",
        "the mountain rose to the sky. ",

        // "is" patterns (high frequency copula)
        "it is a good day. ",
        "she is not here. ",
        "he is very tall. ",
        "this is a test. ",
        "that is not right. ",
        "what is your name. ",
        "who is at the door. ",
        "where is the key. ",

        // "and" patterns (conjunction)
        "and then and after and before ",
        "and so and now and here ",
        "and when and where and why ",
        "and if and but and nor ",
        "and all and some and many ",
        "and yes and no and maybe ",

        // "of" patterns (preposition)
        "of the of a of my of our ",
        "of his of her of their of its ",
        "of time of place of way of life ",
        "of all of some of many of none ",
        "of this of that of what of which ",

        // "to" patterns (preposition/infinitive)
        "to the to a to be to do ",
        "to go to see to know to try ",
        "to find to make to take to give ",
        "to work to rest to play to stay ",
        "to here to there to now to then ",

        // "in" patterns (preposition)
        "in the in a in this in that ",
        "in here in there in now in time ",
        "in way in place in life in world ",
        "in all in some in many in few ",

        // "for" patterns (preposition)
        "for the for a for my for you ",
        "for him for her for them for us ",
        "for day for time for life for all ",
        "for good for bad for best for worst ",

        // "cat" as example noun with common patterns
        "cat sat cat ran cat slept cat ate ",
        "cat and dog cat with mouse cat by fire ",

        // Simple SVO sentences with common transitions
        "they walk in the park. ",
        "we sit by the fire. ",
        "you read in the sun. ",
        "i dream of the sea. ",
        "he runs to the store. ",
        "she waits at the gate. ",
        "they play in the garden. ",
        "we sing at the show. ",

        // More complex patterns with multiple transitions
        "the cat and the dog are good. ",
        "the sun is warm and bright. ",
        "it is a day of joy and peace. ",
        "in the land of the free and brave. ",
        "to be or not to be is the question. ",
        "of all the things of this world. ",

        // Repeated short patterns for signal density
        "a a a a a is is is is is ",
        "the the the the the to to to to to ",
        "and and and and of of of of in in ",
        "for for for in in the the cat cat ",

        // Transitional drilling
        "the is and of to in for at be do ",
        "go see know try find make take give ",
        "work rest play stay sit run walk talk ",
        "come go here there now then why what ",

        // Dense byte-level signal with word boundaries
        "cat dog rat bat hat sat mat fat ",
        "man pan fan can ban tan ran ",
        "it is at as we be ",
        "no go so to do ",
    ];

    // Fill output by cycling through patterns
    while output.len() < n_bytes {
        for pattern in &patterns {
            output.extend_from_slice(pattern.as_bytes());
            if output.len() >= n_bytes {
                break;
            }
        }
    }

    // Trim to exact size requested
    output.truncate(n_bytes);
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generates_correct_size() {
        let sizes = [100, 1024, 10240, 102400];
        for size in sizes.iter() {
            let data = generate_word_completion_data(*size);
            assert!(data.len() <= *size);
            assert!(data.len() > 0);
        }
    }

    #[test]
    fn test_is_valid_utf8() {
        let data = generate_word_completion_data(10000);
        assert!(std::str::from_utf8(&data).is_ok());
    }

    #[test]
    fn test_deterministic() {
        let data1 = generate_word_completion_data(5000);
        let data2 = generate_word_completion_data(5000);
        assert_eq!(data1, data2);
    }

    #[test]
    fn test_contains_target_patterns() {
        let data = generate_word_completion_data(10000);
        let text = std::str::from_utf8(&data).unwrap();

        assert!(text.contains("the cat"));
        assert!(text.contains("is a"));
        assert!(text.contains("of the"));
        assert!(text.contains("and then"));
        assert!(text.contains("to the"));
    }

    #[test]
    fn test_hundred_kb_generation() {
        let data = generate_word_completion_data(102400);
        assert!(data.len() > 100000);
        assert!(data.len() <= 102400);
    }
}
