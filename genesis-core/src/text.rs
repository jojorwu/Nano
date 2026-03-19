use std::collections::HashMap;

pub struct SpikingTextModule {
    pub vocabulary: HashMap<String, usize>,
    pub embeddings: Vec<Vec<bool>>,
}

impl SpikingTextModule {
    pub fn new(vocab_size: usize, _pattern_length: usize) -> Self {
        Self {
            vocabulary: HashMap::with_capacity(vocab_size),
            embeddings: Vec::with_capacity(vocab_size),
        }
    }
    pub fn tokenize(&mut self, text: &str) -> Vec<usize> {
        text.split_whitespace().map(|word| {
            let next_idx = self.vocabulary.len();
            *self.vocabulary.entry(word.to_string()).or_insert(next_idx)
        }).collect()
    }
    pub fn encode(&self, token: usize, pattern_length: usize) -> Vec<bool> {
        let mut pattern = vec![false; pattern_length];
        let hash = token as u64;
        for i in 0..pattern_length {
            if (hash >> (i % 64)) & 1 == 1 {
                pattern[i] = true;
            }
        }
        pattern
    }
}
