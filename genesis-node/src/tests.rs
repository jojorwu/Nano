#[cfg(test)]
mod tests {
    use crate::Runtime;
    use genesis_core::{BakedModel, NeuronState, Synapse};
    use std::collections::HashMap;

    #[test]
    fn test_runtime_synapse_propagation() {
        let mut model = BakedModel {
            neurons: vec![NeuronState::default(); 2],
            synapses: vec![Synapse {
                source_index: 0,
                target_index: 1,
                weight: 1500,
            }],
            #[cfg(feature = "titan")]
            titan_memory: None,
            has_text: false,
            vocabulary: HashMap::new(),
        };
        model.neurons[1].threshold = 1000;

        let mut runtime = Runtime {
            model,
            previous_spikes: vec![true, false],
            learning_rate: 0,
            tick_counter: 0,
            structural_config: Default::default(),
        };

        let spikes = runtime.tick(&[0, 0]);
        assert!(spikes[1]);
    }

    #[test]
    fn test_night_phase_pruning() {
        let model = BakedModel {
            neurons: vec![NeuronState::default(); 2],
            synapses: vec![Synapse {
                source_index: 0,
                target_index: 1,
                weight: 5,
            }],
            #[cfg(feature = "titan")]
            titan_memory: None,
            has_text: false,
            vocabulary: HashMap::new(),
        };

        let mut runtime = Runtime {
            model,
            previous_spikes: vec![false; 2],
            learning_rate: 0,
            tick_counter: 99,
            structural_config: Default::default(),
        };

        runtime.tick(&[0, 0]);
        assert_eq!(runtime.model.synapses.len(), 0);
    }
}
