use genesis_core::BakedModel;
use genesis_compute::{ComputeBackend, CpuBackend};

pub struct Runtime {
    pub model: BakedModel,
    pub backend: Box<dyn ComputeBackend>,
    pub previous_spikes: Vec<bool>,
    pub tick_counter: u64,
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
        })
    }

    pub fn tick(&mut self, external_inputs: &[i32]) -> Vec<bool> {
        // 1. Day Phase: Inference
        let current_spikes = self.backend.day_phase(&mut self.model, external_inputs, &self.previous_spikes);

        // 2. Night Phase: Learning (every 100 ticks)
        self.tick_counter += 1;
        if self.tick_counter % 100 == 0 {
            self.backend.night_phase(&mut self.model, &self.previous_spikes, &current_spikes);
        }

        self.previous_spikes = current_spikes.clone();
        current_spikes
    }
}

#[cfg(test)]
mod tests;
