use genesis_core::{BakedModel, SpikingNeuron, update_weight_gsop};
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
        let n_count = model.neurons.len();
        Ok(Self {
            model,
            previous_spikes: vec![false; n_count],
            learning_rate: 10,
            tick_counter: 0,
            structural_config: StructuralPlasticityConfig::default(),
        })
    }

    pub fn tick(&mut self, external_inputs: &[i32]) -> Vec<bool> {
        let n_count = self.model.neurons.len();
        let mut current_inputs = vec![0i32; n_count];

        for (i, &val) in external_inputs.iter().enumerate() {
            if i < n_count { current_inputs[i] = current_inputs[i].saturating_add(val); }
        }

        for synapse in &self.model.synapses {
            if self.previous_spikes[synapse.source_index as usize] {
                current_inputs[synapse.target_index as usize] = current_inputs[synapse.target_index as usize].saturating_add(synapse.weight);
            }
        }

        #[cfg(feature = "titan")]
        if let Some(ref titan) = self.model.titan_memory {
            let memory_input = titan.retrieve(&self.previous_spikes);
            for i in 0..n_count {
                current_inputs[i] = current_inputs[i].saturating_add(memory_input / (n_count as i32).max(1));
            }
        }

        let mut new_spikes = Vec::with_capacity(n_count);
        for (i, neuron) in self.model.neurons.iter_mut().enumerate() {
            let s = neuron.tick(current_inputs[i]);
            new_spikes.push(s);
        }

        for synapse in self.model.synapses.iter_mut() {
            let pre_spiked = self.previous_spikes[synapse.source_index as usize];
            let post_spiked = new_spikes[synapse.target_index as usize];
            update_weight_gsop(&mut synapse.weight, pre_spiked, post_spiked, self.learning_rate);
        }

        #[cfg(feature = "titan")]
        if let Some(ref mut titan) = self.model.titan_memory {
            let activity = new_spikes.iter().filter(|&&s| s).count() as i32;
            let target_activity = 10;
            let error = target_activity - activity; // Simple error
            titan.step(&self.previous_spikes, error * 10);
        }

        self.tick_counter += 1;
        if self.tick_counter % 100 == 0 {
            self.night_phase();
        }

        self.previous_spikes = new_spikes.clone();
        new_spikes
    }

    /// Night Phase: Optimized structural evolution.
    pub fn night_phase(&mut self) {
        let pruned = prune_synapses(&mut self.model.synapses, self.structural_config.prune_threshold);
        if pruned > 0 {
            // Synapses pruned
        }

        // Optimized Growth: Fire together -> wire together
        // Instead of $O(N^2)$, we only consider active neurons.
        let active_indices: Vec<usize> = self.previous_spikes.iter().enumerate()
            .filter(|&(_, &s)| s)
            .map(|(i, _)| i)
            .collect();

        if active_indices.len() > 1 && self.model.synapses.len() < self.structural_config.max_synapses {
            for &i in active_indices.iter().take(10) { // Limit to avoid burst growth
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
