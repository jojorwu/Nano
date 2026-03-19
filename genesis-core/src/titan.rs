use serde::{Deserialize, Serialize};
use crate::IValue;

#[derive(Serialize, Deserialize)]
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
            surprise_threshold: 500, // 0.5
        }
    }
    pub fn step(&mut self, input_pattern: &[bool], prediction_error: IValue) {
        if prediction_error > self.surprise_threshold {
            for (i, &spiked) in input_pattern.iter().enumerate() {
                if spiked && i < self.weights.len() {
                    self.weights[i] += (prediction_error * self.learning_rate) / 1000;
                }
            }
        }
    }
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
