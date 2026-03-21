use serde::{Deserialize, Serialize};
use crate::{IValue, NanoModule, NeuronsSoA, SynapsesSoA};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TitanMemory {
    /// Associative memory weights [input_neuron_count * output_neuron_count]
    pub weights: Vec<IValue>,
    pub permanent_weights: Vec<IValue>,
    pub learning_rate: IValue,
    pub surprise_threshold: IValue,
    pub moment: Vec<IValue>,
    pub decay_rate: IValue,
    pub associative_size: usize,
}

impl TitanMemory {
    pub fn new(size: usize, lr: IValue) -> Self {
        // Associative size is the number of neurons it interacts with.
        // For simplicity, we assume the same count for input/output.
        Self {
            weights: vec![0; size * size],
            permanent_weights: vec![0; size * size],
            learning_rate: lr,
            surprise_threshold: 100,
            moment: vec![0; size * size],
            decay_rate: 1,
            associative_size: size,
        }
    }

    pub fn step(&mut self, pre_pattern: &[bool], post_pattern: &[bool], error: IValue) {
        let dynamic_decay = if error.abs() > self.surprise_threshold {
            self.decay_rate / 2
        } else {
            self.decay_rate
        };

        if dynamic_decay > 0 {
            for w in self.weights.iter_mut() {
                let decay = ((*w as i64 * dynamic_decay as i64) >> 10) as i32;
                *w = w.saturating_sub(decay);
            }
        }

        if error.abs() > self.surprise_threshold {
            for i in 0..self.associative_size {
                if pre_pattern[i] {
                    for j in 0..self.associative_size {
                        if post_pattern[j] {
                            let idx = i * self.associative_size + j;
                            let grad = error;
                            self.moment[idx] = (self.moment[idx] * 9 + grad) / 10;

                            let lr_boost = if error.abs() > self.surprise_threshold * 2 { 2 } else { 1 };
                            let update = ((self.moment[idx] as i64 * self.learning_rate as i64 * lr_boost as i64) >> 10) as i32;

                            self.weights[idx] = self.weights[idx].saturating_add(update);
                            let slow_update = update / 10;
                            self.permanent_weights[idx] = self.permanent_weights[idx].saturating_add(slow_update);

                            if self.weights[idx] > 10000 { self.weights[idx] = 10000; }
                            if self.weights[idx] < -10000 { self.weights[idx] = -10000; }
                        }
                    }
                }
            }
        }
    }

    pub fn retrieve(&self, input_pattern: &[bool], output_potentials: &mut [IValue]) {
        for i in 0..self.associative_size {
            if input_pattern[i] {
                for j in 0..self.associative_size {
                    let idx = i * self.associative_size + j;
                    let val = self.weights[idx].saturating_add(self.permanent_weights[idx]);
                    output_potentials[j] = output_potentials[j].saturating_add(val);
                }
            }
        }
    }
}

impl NanoModule for TitanMemory {
    fn name(&self) -> &str { "titan" }

    fn on_tick(&mut self, neurons: &mut NeuronsSoA, previous_spikes: &[bool], _tick: u32) {
        // Map spikes from previous tick to current potential bias
        let mut bias = vec![0; self.associative_size.min(neurons.len())];
        self.retrieve(&previous_spikes[..bias.len()], &mut bias);

        for i in 0..bias.len() {
            neurons.proximal_potential[i] = neurons.proximal_potential[i].saturating_add(bias[i]);
        }
    }

    fn on_update_weights(&mut self, _neurons: &mut NeuronsSoA, previous_spikes: &[bool], current_spikes: &[bool], _tick: u32, reward: Option<IValue>) {
        let activity = current_spikes.iter().filter(|&&s| s).count() as i32;
        let base_error = (10 - activity) * 10;
        let final_error = if let Some(r) = reward { base_error + r } else { base_error };

        let min_len = self.associative_size.min(previous_spikes.len()).min(current_spikes.len());
        self.step(&previous_spikes[..min_len], &current_spikes[..min_len], final_error);
    }

    fn on_night_phase(&mut self, _synapses: &mut SynapsesSoA, reward: Option<IValue>) {
        if let Some(r) = reward {
             log::debug!("Titan Night Phase with reward: {}", r);
        }
    }

    fn get_state(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap_or_default()
    }

    fn set_state(&mut self, state: &[u8]) {
        if let Ok(new_self) = bincode::deserialize::<Self>(state) {
            *self = new_self;
        }
    }

    fn box_clone(&self) -> Box<dyn NanoModule> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_titan_associative() {
        let mut titan = TitanMemory::new(4, 100);
        // Pre-spikes at [0], Post-spikes at [1]
        titan.step(&[true, false, false, false], &[false, true, false, false], 500);

        let mut potentials = vec![0; 4];
        titan.retrieve(&[true, false, false, false], &mut potentials);
        assert!(potentials[1] > 0);
        assert_eq!(potentials[0], 0);
    }
}
