//! Word-level vocabulary: the organism's learned tokenizer.
//!
//! Instead of byte-by-byte processing, the organism works with whole words.
//! Starts with the most common words from training data.
//! Can grow during development (learn new words through repetition).
//!
//! Each word gets ONE embedding and ONE CTM processing step.
//! "the cat sat" = 3 steps, not 15 bytes.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Word-level vocabulary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vocab {
    /// Word → token ID mapping
    pub word_to_id: HashMap<String, usize>,
    /// Token ID → word mapping
    pub id_to_word: Vec<String>,
    /// Special tokens
    pub unk_id: usize,   // unknown word
    pub space_id: usize, // explicit space (word separator)
}

impl Vocab {
    /// Build vocabulary from the N most common words in text.
    pub fn from_text(text: &str, max_words: usize) -> Self {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for word in text.split_whitespace() {
            let w = word.to_lowercase();
            *counts.entry(w).or_insert(0) += 1;
        }

        // Sort by frequency
        let mut sorted: Vec<(String, usize)> = counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));

        // Build vocab: reserve 0 for UNK, 1 for SPACE
        let mut id_to_word = vec!["<UNK>".to_string(), " ".to_string()];
        let mut word_to_id = HashMap::new();
        word_to_id.insert("<UNK>".to_string(), 0);
        word_to_id.insert(" ".to_string(), 1);

        for (word, _count) in sorted.iter().take(max_words - 2) {
            let id = id_to_word.len();
            word_to_id.insert(word.clone(), id);
            id_to_word.push(word.clone());
        }

        Self {
            word_to_id,
            id_to_word,
            unk_id: 0,
            space_id: 1,
        }
    }

    pub fn size(&self) -> usize {
        self.id_to_word.len()
    }

    /// Tokenize text into word IDs.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.split_whitespace()
            .map(|w| {
                let lower = w.to_lowercase();
                *self.word_to_id.get(&lower).unwrap_or(&self.unk_id)
            })
            .collect()
    }

    /// Decode word IDs back to text.
    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .map(|&id| {
                self.id_to_word.get(id)
                    .map(|s| s.as_str())
                    .unwrap_or("<UNK>")
            })
            .collect::<Vec<&str>>()
            .join(" ")
    }

    /// Get word for an ID.
    pub fn get_word(&self, id: usize) -> &str {
        self.id_to_word.get(id).map(|s| s.as_str()).unwrap_or("<UNK>")
    }
}
