use std::collections::HashMap;
use crate::{NanoModule, NeuronsSoA, SynapsesSoA, IValue};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TextProcessorModule {
    pub vocabulary: HashMap<String, usize>,
    pub pattern_length: usize,
    pub last_tokens: Vec<usize>,
}

impl TextProcessorModule {
    pub fn new(pattern_length: usize) -> Self {
        Self {
            vocabulary: HashMap::new(),
            pattern_length,
            last_tokens: Vec::new(),
        }
    }

    pub fn tokenize_and_queue(&mut self, text: &str) {
        let tokens = text.split_whitespace().map(|word| {
            let next_idx = self.vocabulary.len();
            *self.vocabulary.entry(word.to_string()).or_insert(next_idx)
        }).collect();
        self.last_tokens = tokens;
    }

    pub fn encode_token(&self, token: usize) -> Vec<bool> {
        let mut pattern = vec![false; self.pattern_length];
        let mut h = token as u64;
        h = h.wrapping_mul(0x517cc1b727220a95);
        for i in 0..self.pattern_length {
            if (h.rotate_right(i as u32) & 1) == 1 {
                pattern[i] = true;
            }
        }
        pattern
    }
}

impl NanoModule for TextProcessorModule {
    fn name(&self) -> &str { "text_processor" }

    fn on_tick(&mut self, neurons: &mut NeuronsSoA, _previous_spikes: &[bool], _tick: u32) {
        if let Some(token) = self.last_tokens.get(0) {
            let pattern = self.encode_token(*token);
            for (i, &spiked) in pattern.iter().enumerate() {
                if spiked && i < neurons.len() {
                    neurons.proximal_potential[i] = neurons.proximal_potential[i].saturating_add(1024);
                }
            }
            // Consume token
            self.last_tokens.remove(0);
        }
    }

    fn on_update_weights(&mut self, _neurons: &mut NeuronsSoA, _previous_spikes: &[bool], _current_spikes: &[bool], _tick: u32, _reward: Option<IValue>) {}
    fn on_night_phase(&mut self, _synapses: &mut SynapsesSoA, _reward: Option<IValue>) {}

    fn box_clone(&self) -> Box<dyn NanoModule> {
        Box::new(self.clone())
    }

    fn get_state(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap_or_default()
    }

    fn set_state(&mut self, state: &[u8]) {
        if let Ok(new_self) = bincode::deserialize::<Self>(state) {
            *self = new_self;
        }
    }
}

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

pub struct ByteSpikingModule;

impl ByteSpikingModule {
    pub fn encode_byte(byte: u8, pattern_length: usize) -> Vec<bool> {
        let mut pattern = vec![false; pattern_length];
        let mut h = byte as u64;
        h = h.wrapping_mul(0x517cc1b727220a95);

        for i in 0..pattern_length {
            // Improve entropy for long patterns by mixing in the bit index
            let mut bit_h = h.wrapping_add(i as u64);
            bit_h = bit_h.wrapping_mul(0xbf58476d1ce4e5b9);
            bit_h = bit_h ^ (bit_h >> 31);

            if (bit_h & 1) == 1 {
                pattern[i] = true;
            }
        }
        pattern
    }

    pub fn encode_text(text: &str, pattern_length: usize) -> Vec<Vec<bool>> {
        text.as_bytes().iter().map(|&b| Self::encode_byte(b, pattern_length)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byte_spiking() {
        let pattern = ByteSpikingModule::encode_byte(b'A', 64);
        assert_eq!(pattern.len(), 64);
        let pattern2 = ByteSpikingModule::encode_byte(b'A', 64);
        assert_eq!(pattern, pattern2);
        let pattern_b = ByteSpikingModule::encode_byte(b'B', 64);
        assert_ne!(pattern, pattern_b);
    }

    #[test]
    fn test_encode_text() {
        let text = "Hello";
        let patterns = ByteSpikingModule::encode_text(text, 64);
        assert_eq!(patterns.len(), 5);
    }
}
