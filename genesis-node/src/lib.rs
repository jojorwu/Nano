use genesis_core::BakedModel;
use genesis_compute::{ComputeBackend, CpuBackend};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct SpikePacket {
    pub tick: u64,
    pub active_indices: Vec<usize>,
}

pub struct Runtime {
    pub model: BakedModel,
    pub backend: Box<dyn ComputeBackend + Send + Sync>,
    pub previous_spikes: Vec<bool>,
    pub tick_counter: u64,
    pub spikes_history: Vec<Vec<bool>>,
}

impl Runtime {
    pub fn load(path: &str) -> std::io::Result<Self> {
        let model = BakedModel::load(path)?;
        let n_count = model.neurons.len();
        Ok(Self {
            model,
            backend: Box::new(CpuBackend::default()),
            previous_spikes: vec![false; n_count],
            tick_counter: 0,
            spikes_history: Vec::new(),
        })
    }

    pub fn tick_with_reward(&mut self, external_inputs: &[i32], reward: Option<i32>) -> Vec<bool> {
        self.tick_counter += 1;
        // 1. Day Phase: Inference
        let current_spikes = self.backend.day_phase(&mut self.model, external_inputs, &self.previous_spikes, self.tick_counter);

        self.spikes_history.push(current_spikes.clone());

        // 2. Night Phase: Learning (every 100 ticks)
        if self.tick_counter % 100 == 0 {
            // Replay history for learning
            let mut prev = self.previous_spikes.clone(); // Correct batch boundary
            for (i, current) in self.spikes_history.iter().enumerate() {
                let tick = self.tick_counter - (self.spikes_history.len() as u64) + (i as u64) + 1;

                // Update neuron last_spike_tick for the replayed tick
                for (n_idx, &spiked) in current.iter().enumerate() {
                    if spiked { self.model.neurons.last_spike_tick[n_idx] = tick; }
                }

                self.backend.night_phase(&mut self.model, &prev, current, tick, reward);
                prev = current.clone();
            }
            self.spikes_history.clear();
        }

        self.previous_spikes = current_spikes.clone();
        current_spikes
    }

    pub fn tick(&mut self, external_inputs: &[i32]) -> Vec<bool> {
        self.tick_with_reward(external_inputs, None)
    }
}

#[cfg(test)]
mod tests;
