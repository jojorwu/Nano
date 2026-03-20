use serde::{Deserialize, Serialize};
use crate::IValue;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TitanMemory {
    pub weights: Vec<IValue>,
    pub permanent_weights: Vec<IValue>, // Hierarchical Memory
    pub learning_rate: IValue,
    pub surprise_threshold: IValue,
    pub moment: Vec<IValue>,
    pub decay_rate: IValue,
}

impl TitanMemory {
    pub fn new(size: usize, lr: IValue) -> Self {
        Self {
            weights: vec![0; size],
            permanent_weights: vec![0; size],
            learning_rate: lr,
            surprise_threshold: 100,
            moment: vec![0; size],
            decay_rate: 1, // Default ~0.1% decay
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
                let decay = ((*w as i64 * dynamic_decay as i64) >> 10) as i32;
                *w = w.saturating_sub(decay);
            }
        }

        if error.abs() > self.surprise_threshold {
            for (i, &spiked) in input_pattern.iter().enumerate() {
                if spiked && i < self.weights.len() {
                    let grad = error;
                    self.moment[i] = (self.moment[i] * 9 + grad) / 10;

                    // Surprise-Modulated Learning Rate (SMLR)
                    // Boost LR if surprise is extremely high
                    let lr_boost = if error.abs() > self.surprise_threshold * 2 { 2 } else { 1 };
                    let update = ((self.moment[i] as i64 * self.learning_rate as i64 * lr_boost as i64) >> 10) as i32;

                    self.weights[i] = self.weights[i].saturating_add(update);

                    // Permanent weight update (slow consolidation)
                    let slow_update = update / 10;
                    self.permanent_weights[i] = self.permanent_weights[i].saturating_add(slow_update);

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
                // Combine temporary and permanent memory
                sum = sum.saturating_add(self.weights[i]);
                sum = sum.saturating_add(self.permanent_weights[i]);
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
        titan.weights[0] = 1024;
        titan.decay_rate = 128; // ~12.5% decay
        titan.step(&[false; 10], 0);
        assert!(titan.weights[0] < 1024);
        assert_eq!(titan.weights[0], 1024 - 128);
    }

    #[test]
    fn test_surprise_modulation() {
        let mut titan = TitanMemory::new(10, 100);
        titan.weights[0] = 1024;
        titan.decay_rate = 128;
        titan.surprise_threshold = 50;

        // High surprise (error 100) -> half decay (64)
        titan.step(&[false; 10], 100);
        assert_eq!(titan.weights[0], 1024 - 64);
    }

    #[test]
    fn test_titan_hierarchical_memory() {
        let mut titan = TitanMemory::new(10, 500); // Higher learning rate
        // Surprise-driven learning should update both temp and permanent weights
        titan.step(&[true; 10], 500); // High error
        assert!(titan.weights[0] != 0);
        assert!(titan.permanent_weights[0] != 0);

        let retrieved = titan.retrieve(&[true; 10]);
        assert!(retrieved != 0);
    }
}
