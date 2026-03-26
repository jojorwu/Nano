use std::collections::HashMap;
use crate::{NanoModule, NeuronsSoA, SynapsesSoA, IValue, SCALE};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TextProcessorModule {
    pub vocabulary: HashMap<String, usize>,
    pub pattern_length: usize,
    pub last_tokens: Vec<usize>,
    pub mode: TextMode,
    /// Spiking Auto-Encoder: trainable weights mapping tokens to patterns
    pub embedding_weights: HashMap<usize, Vec<IValue>>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum TextMode {
    Vocabulary,
    ByteSpiking,
}

impl TextProcessorModule {
    pub fn new(pattern_length: usize) -> Self {
        Self {
            vocabulary: HashMap::new(),
            pattern_length,
            last_tokens: Vec::new(),
            mode: TextMode::Vocabulary,
            embedding_weights: HashMap::new(),
        }
    }

    pub fn set_mode(&mut self, mode: TextMode) {
        self.mode = mode;
    }

    pub fn tokenize_and_queue(&mut self, text: &str) {
        if self.mode == TextMode::Vocabulary {
            let tokens = text.split_whitespace().map(|word| {
                let next_idx = self.vocabulary.len();
                *self.vocabulary.entry(word.to_string()).or_insert(next_idx)
            }).collect();
            self.last_tokens = tokens;
        } else {
             // In ByteSpikingMode, we store raw bytes as "tokens"
             self.last_tokens = text.as_bytes().iter().map(|&b| b as usize).collect();
        }
    }

    pub fn encode_token(&mut self, token: usize) -> Vec<bool> {
        // Trainable Embedding: use learned weights if available
        if let Some(weights) = self.embedding_weights.get(&token) {
            let mut pattern = vec![false; self.pattern_length];
            for (i, &w) in weights.iter().enumerate().take(self.pattern_length) {
                if w > 512 { pattern[i] = true; }
            }
            return pattern;
        }

        if self.mode == TextMode::Vocabulary {
            let mut pattern = vec![false; self.pattern_length];
            let mut h = token as u64;
            h = h.wrapping_mul(0x517cc1b727220a95);
            for i in 0..self.pattern_length {
                if (h.rotate_right(i as u32) & 1) == 1 {
                    pattern[i] = true;
                }
            }
            pattern
        } else {
            ByteSpikingModule::encode_byte(token as u8, self.pattern_length)
        }
    }
}

impl NanoModule for TextProcessorModule {
    fn name(&self) -> &str { "text_processor" }
    fn tier(&self) -> u32 { 0 }
    fn outputs(&self) -> Vec<String> { vec!["modality:text".to_string(), "proximal".to_string()] }
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }

    fn handle_input(&mut self, input: &crate::ModuleInput) {
        if let crate::ModuleInput::Text(text) = input {
            self.tokenize_and_queue(text);
        }
    }

    fn on_tick(&mut self, bus: &crate::InputBus, _previous_spikes: &[bool], _tick: u32) {
        // In burst mode, we might process multiple tokens if the bus is cleared between them.
        // For now, we keep the 1-token-per-tick cadence but allow the Runtime's process_burst
        // to handle the sequential extraction.
        if let Some(&token) = self.last_tokens.get(0) {
            let pattern = self.encode_token(token);
            let prox = bus.proximal();
            for (i, &spiked) in pattern.iter().enumerate() {
                if spiked {
                    bus.set_modality(crate::Modality::Text, i, SCALE);
                    crate::InputBus::atomic_saturating_add(&prox[i], SCALE);
                }
            }
            // Consume token
            self.last_tokens.remove(0);
        }
    }

    fn on_update_weights(&mut self, neurons: &mut NeuronsSoA, _previous_spikes: &[bool], current_spikes: &[bool], _tick: u32, reward: Option<IValue>) {
        // Spiking Auto-Encoder (SAE) Training Logic:
        // Align token patterns with the neural representations they evoke.
        // If a token is active, we reinforce embedding bits that correspond to firing neurons.
        if let Some(&token) = self.last_tokens.get(0) {
            let weights = self.embedding_weights.entry(token).or_insert_with(|| vec![512; self.pattern_length]);
            let r = reward.unwrap_or(0);

            for i in 0..self.pattern_length {
                if i < current_spikes.len() {
                    let w = &mut weights[i];

                    // SAE Reconstruction Error:
                    // current_spikes[i] is reality, *w > 512 is prediction
                    let predicted = *w > 512;
                    let actual = current_spikes[i];

                    if actual && predicted {
                        // Correct Prediction: strengthen (SAE LTP)
                        *w = w.saturating_add(if r >= 0 { 20 } else { 5 });
                    } else if actual && !predicted {
                        // Missing Feature: strengthen (Learning)
                        *w = w.saturating_add(30);
                    } else if !actual && predicted {
                        // False Positive: weaken (SAE LTD)
                        *w = w.saturating_sub(15);
                    } else {
                        // Sparse decay
                        *w = w.saturating_sub(1);
                    }

                    // Top-down feedback: neurons that consistently help reconstruction get a specialization boost
                    if actual {
                        neurons.specialization_score[i] = neurons.specialization_score[i] * 0.99 + 0.05;
                    }

                    *w = (*w).clamp(0, SCALE);
                }
            }
        }
    }
    fn on_night_phase(&mut self, _neurons: &mut NeuronsSoA, _synapses: &mut SynapsesSoA, _reward: Option<IValue>) {}

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
    fn test_text_processor_byte_mode() {
        let mut tp = TextProcessorModule::new(64);
        tp.set_mode(TextMode::ByteSpiking);
        tp.tokenize_and_queue("ABC");
        assert_eq!(tp.last_tokens.len(), 3);
        assert_eq!(tp.last_tokens[0], b'A' as usize);
    }
}
