use crate::IValue;

/// Trait defining a Reinforcement Learning Environment.
pub trait Environment {
    /// Return the dimensionality of the observation space.
    fn observation_space(&self) -> usize;
    /// Return the dimensionality of the action space.
    fn action_space(&self) -> usize;
    /// Reset the environment and return the initial observation.
    fn reset(&mut self) -> Vec<IValue>;
    /// Take a step in the environment.
    /// Returns (new_observation, reward, done).
    fn step(&mut self, actions: &[bool]) -> (Vec<IValue>, IValue, bool);
}

/// Interface for the SNN Agent to interact with an RL Environment.
pub struct RLAgent {
    pub input_indices: Vec<usize>,
    pub output_indices: Vec<usize>,
}

impl RLAgent {
    pub fn new(input_count: usize, output_count: usize, _total_neurons: usize) -> Self {
        // Simple mapping: first neurons are inputs, next are outputs.
        Self {
            input_indices: (0..input_count).collect(),
            output_indices: (input_count..input_count + output_count).collect(),
        }
    }

    pub fn encode_observation(&self, observation: &[IValue], neuron_count: usize) -> Vec<IValue> {
        let mut inputs = vec![0; neuron_count];
        for (i, &obs) in observation.iter().enumerate() {
            if i < self.input_indices.len() {
                inputs[self.input_indices[i]] = obs;
            }
        }
        inputs
    }

    pub fn decode_action(&self, spikes: &[bool]) -> Vec<bool> {
        self.output_indices.iter().map(|&idx| spikes[idx]).collect()
    }
}
