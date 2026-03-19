#[cfg(test)]
mod tests {
    use crate::Runtime;
    use genesis_core::{BakedModel, NeuronState, Synapse};

    #[test]
    fn test_runtime_synapse_propagation() {
        let mut model = BakedModel {
            neurons: vec![NeuronState::default(); 2],
            synapses: vec![Synapse {
                source_index: 0,
                target_index: 1,
                weight: 1500, // Enough to spike neuron 1
            }],
            #[cfg(feature = "titan")]
            titan_memory: None,
            has_text: false,
        };
        model.neurons[1].threshold = 1000;

        let mut runtime = Runtime {
            model,
            previous_spikes: vec![true, false], // Neuron 0 fired
            learning_rate: 0,
        };

        let spikes = runtime.tick(&[0, 0]);
        // Neuron 1 should fire due to synapse from neuron 0
        assert!(spikes[1]);
    }

    #[test]
    #[cfg(feature = "titan")]
    fn test_titan_memory_influence() {
        use genesis_core::titan::TitanMemory;
        let mut titan = TitanMemory::new(2, 500);
        titan.weights[0] = 2000; // Strong memory signal

        let mut model = BakedModel {
            neurons: vec![NeuronState::default(); 2],
            synapses: vec![],
            titan_memory: Some(titan),
            has_text: false,
        };
        model.neurons[0].threshold = 500;
        model.neurons[1].threshold = 500;

        let mut runtime = Runtime {
            model,
            previous_spikes: vec![true, false], // Stimulate with neuron 0
            learning_rate: 0,
        };

        let spikes = runtime.tick(&[0, 0]);
        // Distributed input = 2000 / 2 = 1000. Threshold = 500.
        assert!(spikes[0]);
        assert!(spikes[1]);
    }
}
