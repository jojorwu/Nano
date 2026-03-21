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

    /// Three-Factor STDP: update = (pre * post * modulator)
    /// Here 'modulator' is the 'surprise' signal.
    pub fn step_three_factor(&mut self, pre_pattern: &[bool], post_pattern: &[bool], surprise: IValue) {
        // Gating Mechanism: Forgetting is modulated by surprise.
        // If surprise is low, we decay weights faster (forget irrelevant details).
        // If surprise is high, we preserve weights (important context).
        let dynamic_decay = if surprise < self.surprise_threshold {
            self.decay_rate * 2
        } else {
            self.decay_rate / 2
        };

        if dynamic_decay > 0 {
            for w in self.weights.iter_mut() {
                let decay = ((*w as i64 * dynamic_decay as i64) >> 10) as i32;
                *w = w.saturating_sub(decay);
            }
        }

        // Three-Factor Learning: Only update if there is significant surprise (neuromodulation)
        let pre_len = pre_pattern.len().min(self.associative_size);
        let post_len = post_pattern.len().min(self.associative_size);

        if surprise > self.surprise_threshold {
            for i in 0..pre_len {
                if pre_pattern[i] {
                    for j in 0..post_len {
                        if post_pattern[j] {
                            let idx = i * self.associative_size + j;

                            // Correlation (pre * post) modulated by surprise (3rd factor)
                            let update_base = (surprise as i64 * self.learning_rate as i64) >> 10;
                            let update = update_base as i32;

                            self.moment[idx] = (self.moment[idx] * 8 + update * 2) / 10;
                            self.weights[idx] = self.weights[idx].saturating_add(self.moment[idx]);

                            // Slow consolidation into permanent memory
                            let slow_update = self.moment[idx] / 10;
                            self.permanent_weights[idx] = self.permanent_weights[idx].saturating_add(slow_update);

                            // Clamp
                            if self.weights[idx] > 10000 { self.weights[idx] = 10000; }
                            if self.weights[idx] < -10000 { self.weights[idx] = -10000; }
                        }
                    }
                }
            }
        }
    }

    pub fn retrieve(&self, input_pattern: &[bool], output_potentials: &mut [IValue]) {
        let in_len = input_pattern.len().min(self.associative_size);
        let out_len = output_potentials.len().min(self.associative_size);

        for i in 0..in_len {
            if input_pattern[i] {
                for j in 0..out_len {
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
        let mut bias = vec![0; self.associative_size.min(neurons.len())];
        self.retrieve(&previous_spikes[..bias.len()], &mut bias);

        for i in 0..bias.len() {
            // Memory as Context (S-MAC): Targeted injection into distal dendrites.
            // This allows the memory context to be gated by local somatic activity,
            // implementing a more sophisticated biological feedback mechanism.
            neurons.distal_potential[i] = neurons.distal_potential[i].saturating_add(bias[i]);
        }
    }

    fn on_update_weights(&mut self, _neurons: &mut NeuronsSoA, previous_spikes: &[bool], current_spikes: &[bool], _tick: u32, surprise: Option<IValue>) {
        if let Some(s) = surprise {
            let min_len = self.associative_size.min(previous_spikes.len()).min(current_spikes.len());
            self.step_three_factor(&previous_spikes[..min_len], &current_spikes[..min_len], s);
        }
    }

    fn on_night_phase(&mut self, _synapses: &mut SynapsesSoA, reward: Option<IValue>) {
        if let Some(r) = reward {
             // Reward-modulated consolidation or pruning could go here
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
    fn test_titan_three_factor() {
        let mut titan = TitanMemory::new(4, 100);
        // High surprise (500 > 100 threshold) triggers learning
        titan.step_three_factor(&[true, false, false, false], &[false, true, false, false], 500);

        let mut potentials = vec![0; 4];
        titan.retrieve(&[true, false, false, false], &mut potentials);
        assert!(potentials[1] > 0);
    }

    #[test]
    fn test_titan_forgetting() {
        let mut titan = TitanMemory::new(4, 100);
        titan.weights[0] = 1000;
        titan.decay_rate = 10;
        // Low surprise (50 < 100 threshold) triggers faster forgetting
        titan.step_three_factor(&[false; 4], &[false; 4], 50);
        assert!(titan.weights[0] < 1000);
    }
}
