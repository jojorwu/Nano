use serde::{Deserialize, Serialize};
use crate::IValue;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TitanMemory {
    pub weights: Vec<IValue>,
    pub learning_rate: IValue,
    pub surprise_threshold: IValue,
}

impl TitanMemory {
    pub fn new(size: usize, lr: IValue) -> Self {
        Self {
            weights: vec![0; size],
            learning_rate: lr,
            surprise_threshold: 100, // 0.1
        }
    }

    /// Step updates the long-term memory weights based on the current surprise (prediction error).
    pub fn step(&mut self, input_pattern: &[bool], surprise: IValue) {
        if surprise > self.surprise_threshold {
            for (i, &spiked) in input_pattern.iter().enumerate() {
                if spiked && i < self.weights.len() {
                    // Update weights with scaled surprise
                    let delta = (surprise * self.learning_rate) / 1000;
                    self.weights[i] += delta;

                    // Clamp memory weights
                    if self.weights[i] > 10000 { self.weights[i] = 10000; }
                }
            }
        }
    }

    /// Retrieve returns the aggregated memory signal for the given input pattern.
    pub fn retrieve(&self, input_pattern: &[bool]) -> IValue {
        let mut sum = 0;
        for (i, &spiked) in input_pattern.iter().enumerate() {
            if spiked && i < self.weights.len() {
                sum += self.weights[i];
            }
        }
        sum
    }
}
