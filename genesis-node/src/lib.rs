use genesis_core::{BakedModel, SpikingNeuron};

pub struct Runtime {
    pub model: BakedModel,
}

impl Runtime {
    pub fn load(path: &str) -> std::io::Result<Self> {
        let model = BakedModel::load(path)?;
        Ok(Self { model })
    }

    pub fn tick(&mut self, inputs: &[i32]) -> Vec<bool> {
        let n_count = self.model.neurons.len();
        let mut current_inputs = vec![0i32; n_count];

        // Add external inputs
        for (i, &val) in inputs.iter().enumerate() {
            if i < n_count {
                current_inputs[i] += val;
            }
        }

        // Add inputs from previous spikes through synapses
        // In a real Axicor engine, we'd use the spikes from the *previous* tick
        // But for this simple implementation, we'll collect spikes from neurons
        // and apply synapse weights for the next iteration.

        let mut spikes = Vec::with_capacity(n_count);
        for (i, neuron) in self.model.neurons.iter_mut().enumerate() {
            let s = neuron.tick(current_inputs[i]);
            spikes.push(s);
        }

        // Apply synaptic propagation for next time (simplified)
        // This is where real Axicor logic happens.

        // Titan memory update if available
        #[cfg(feature = "titan")]
        if let Some(ref mut titan) = self.model.titan_memory {
            // Simplified prediction error based on spike activity
            let activity: i32 = spikes.iter().filter(|&&s| s).count() as i32;
            let prediction_error = (activity * 10).abs();
            titan.step(&spikes, prediction_error);
        }

        spikes
    }
}
