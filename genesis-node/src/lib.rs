use genesis_core::{BakedModel, update_weight_gsop, SCALE};
use genesis_core::plasticity::{prune_synapses, grow_synapse, StructuralPlasticityConfig};

pub struct Runtime {
    pub model: BakedModel,
    pub previous_spikes: Vec<bool>,
    pub learning_rate: i32,
    pub tick_counter: u64,
    pub structural_config: StructuralPlasticityConfig,
}

impl Runtime {
    pub fn load(path: &str) -> std::io::Result<Self> {
        let model = BakedModel::load(path)?;
        let n_count = model.neurons.potential.len();
        Ok(Self {
            model,
            previous_spikes: vec![false; n_count],
            learning_rate: 10,
            tick_counter: 0,
            structural_config: StructuralPlasticityConfig::default(),
        })
    }

    pub fn tick(&mut self, external_inputs: &[i32]) -> Vec<bool> {
        let n_count = self.model.neurons.potential.len();
        let mut current_inputs = vec![0i32; n_count];

        for (i, &val) in external_inputs.iter().enumerate() {
            if i < n_count { current_inputs[i] = current_inputs[i].saturating_add(val); }
        }

        // Synaptic propagation from SoA
        for i in 0..self.model.synapses.len() {
            let src = self.model.synapses.source_index[i] as usize;
            if self.previous_spikes[src] {
                let target = self.model.synapses.target_index[i] as usize;
                current_inputs[target] = current_inputs[target].saturating_add(self.model.synapses.weight[i]);
            }
        }

        #[cfg(feature = "titan")]
        if let Some(ref titan) = self.model.titan_memory {
            let memory_input = titan.retrieve(&self.previous_spikes);
            let dist_input = memory_input / (n_count as i32).max(1);
            for i in 0..n_count {
                current_inputs[i] = current_inputs[i].saturating_add(dist_input);
            }
        }

        // Tick neurons SoA way
        let mut new_spikes = vec![false; n_count];
        for i in 0..n_count {
            if self.model.neurons.refractory_timer[i] > 0 {
                self.model.neurons.refractory_timer[i] -= 1;
                self.model.neurons.potential[i] = 0;
            } else {
                self.model.neurons.potential[i] = self.model.neurons.potential[i].saturating_add(current_inputs[i]);

                // Decay
                let decay = self.model.neurons.decay[i];
                self.model.neurons.potential[i] = (self.model.neurons.potential[i] * (SCALE - decay)) / SCALE;

                if self.model.neurons.potential[i] >= self.model.neurons.threshold[i] {
                    self.model.neurons.potential[i] = 0;
                    self.model.neurons.refractory_timer[i] = 2;
                    new_spikes[i] = true;
                }
            }
        }

        // Plasticity (GSOP) SoA
        for i in 0..self.model.synapses.len() {
            let pre_spiked = self.previous_spikes[self.model.synapses.source_index[i] as usize];
            let post_spiked = new_spikes[self.model.synapses.target_index[i] as usize];
            update_weight_gsop(&mut self.model.synapses.weight[i], pre_spiked, post_spiked, self.learning_rate);
        }

        #[cfg(feature = "titan")]
        if let Some(ref mut titan) = self.model.titan_memory {
            let activity = new_spikes.iter().filter(|&&s| s).count() as i32;
            titan.step(&self.previous_spikes, (10 - activity) * 10);
        }

        self.tick_counter += 1;
        if self.tick_counter % 100 == 0 {
            self.night_phase();
        }

        self.previous_spikes = new_spikes.clone();
        new_spikes
    }

    pub fn night_phase(&mut self) {
        prune_synapses(&mut self.model.synapses, self.structural_config.prune_threshold);

        let active_indices: Vec<usize> = self.previous_spikes.iter().enumerate()
            .filter(|&(_, &s)| s)
            .map(|(i, _)| i)
            .collect();

        if active_indices.len() > 1 && self.model.synapses.len() < self.structural_config.max_synapses {
            for &i in active_indices.iter().take(10) {
                for &j in active_indices.iter().take(10) {
                    if i != j {
                        grow_synapse(&mut self.model.synapses, i as u32, j as u32, 50, &self.structural_config);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
