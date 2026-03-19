use std::collections::HashMap;

pub struct SpikingTextModule<'a> {
    pub vocabulary: &'a mut HashMap<String, usize>,
}

impl<'a> SpikingTextModule<'a> {
    pub fn new(vocab: &'a mut HashMap<String, usize>) -> Self {
        Self {
            vocabulary: vocab,
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
        // Improved deterministic hash-based encoding
        let mut h = token as u64;
        h = h.wrapping_mul(0x517cc1b727220a95);
        for i in 0..pattern_length {
            if (h.rotate_right(i as u32) & 1) == 1 {
                pattern[i] = true;
            }
        }
        pattern
    }
}
