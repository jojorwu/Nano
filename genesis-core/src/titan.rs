use serde::{Deserialize, Serialize};
use crate::IValue;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TitanMemory {
    pub weights: Vec<IValue>,
    pub learning_rate: IValue,
    pub surprise_threshold: IValue,
    pub moment: Vec<IValue>,
    pub decay_rate: IValue,
}

impl TitanMemory {
    pub fn new(size: usize, lr: IValue) -> Self {
        Self {
            weights: vec![0; size],
            learning_rate: lr,
            surprise_threshold: 100,
            moment: vec![0; size],
            decay_rate: 1, // Default 0.1% decay (1/1000)
        }
    }

    pub fn step(&mut self, input_pattern: &[bool], error: IValue) {
        // Surprise-Modulated Dynamic Gating:
        // Decay is inversely proportional to surprise.
        // High surprise (error) reduces decay to preserve new important information.
        let dynamic_decay = if error.abs() > self.surprise_threshold {
            self.decay_rate / 2
        } else {
            self.decay_rate
        };

        // Apply Weight Decay (Gating) as forgetting mechanism
        if dynamic_decay > 0 {
            for w in self.weights.iter_mut() {
                let decay = (*w * dynamic_decay) / 1000;
                *w = w.saturating_sub(decay);
            }
        }

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_titan_decay() {
        let mut titan = TitanMemory::new(10, 100);
        titan.weights[0] = 1000;
        titan.decay_rate = 100; // 10% decay
        titan.step(&[false; 10], 0);
        assert!(titan.weights[0] < 1000);
        assert_eq!(titan.weights[0], 900);
    }

    #[test]
    fn test_surprise_modulation() {
        let mut titan = TitanMemory::new(10, 100);
        titan.weights[0] = 1000;
        titan.decay_rate = 100;
        titan.surprise_threshold = 50;

        // High surprise (error 100) -> half decay (50)
        titan.step(&[false; 10], 100);
        assert_eq!(titan.weights[0], 950);
    }
}
