use genesis_core::{BakedModel, SpikingNeuron, update_weight_gsop};

pub struct Runtime {
    pub model: BakedModel,
    pub previous_spikes: Vec<bool>,
    pub learning_rate: i32,
}

impl Runtime {
    pub fn load(path: &str) -> std::io::Result<Self> {
        let model = BakedModel::load(path)?;
        let n_count = model.neurons.len();
        Ok(Self {
            model,
            previous_spikes: vec![false; n_count],
            learning_rate: 10,
        })
    }

    pub fn tick(&mut self, external_inputs: &[i32]) -> Vec<bool> {
        let n_count = self.model.neurons.len();
        let mut current_inputs = vec![0i32; n_count];

        for (i, &val) in external_inputs.iter().enumerate() {
            if i < n_count { current_inputs[i] += val; }
        }

        for synapse in &self.model.synapses {
            if self.previous_spikes[synapse.source_index as usize] {
                current_inputs[synapse.target_index as usize] += synapse.weight;
            }
        }

        #[cfg(feature = "titan")]
        if let Some(ref titan) = self.model.titan_memory {
            let memory_input = titan.retrieve(&self.previous_spikes);
            for i in 0..n_count {
                current_inputs[i] += memory_input / (n_count as i32).max(1);
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
            titan.step(&self.previous_spikes, activity * 10);
        }

        self.previous_spikes = new_spikes.clone();
        new_spikes
    }
}

#[cfg(test)]
mod tests;
