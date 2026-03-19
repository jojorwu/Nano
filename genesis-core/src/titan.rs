use serde::{Deserialize, Serialize};
use crate::IValue;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TitanMemory {
    pub weights: Vec<IValue>,
    pub learning_rate: IValue,
    pub surprise_threshold: IValue,
    pub moment: Vec<IValue>,
}

impl TitanMemory {
    pub fn new(size: usize, lr: IValue) -> Self {
        Self {
            weights: vec![0; size],
            learning_rate: lr,
            surprise_threshold: 100,
            moment: vec![0; size],
        }
    }

    pub fn step(&mut self, input_pattern: &[bool], error: IValue) {
        if error.abs() > self.surprise_threshold {
            for (i, &spiked) in input_pattern.iter().enumerate() {
                if spiked && i < self.weights.len() {
                    let grad = error;
                    self.moment[i] = (self.moment[i] * 9 + grad) / 10;
                    let update = (self.moment[i] * self.learning_rate) / 1000;
                    self.weights[i] = self.weights[i].saturating_add(update);
                    if self.weights[i] > 10000 { self.weights[i] = 10000; }
                    if self.weights[i] < -10000 { self.weights[i] = -10000; }
                }
            }
        }
    }

    pub fn retrieve(&self, input_pattern: &[bool]) -> IValue {
        let mut sum: IValue = 0;
        for (i, &spiked) in input_pattern.iter().enumerate() {
            if spiked && i < self.weights.len() {
                sum = sum.saturating_add(self.weights[i]);
            }
        }
        sum
    }
}
