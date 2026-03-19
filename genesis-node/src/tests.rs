#[cfg(test)]
mod tests {
    use crate::Runtime;
    use genesis_core::{BakedModel, NeuronsSoA, SynapsesSoA};
    use genesis_compute::CpuBackend;
    use std::collections::HashMap;

    #[test]
    fn test_runtime_synapse_propagation() {
        let mut model = BakedModel {
            node_id: 0,
            local_range: (0, 2),
            neurons: NeuronsSoA::new(2),
            synapses: {
                let mut s = SynapsesSoA::with_capacity(1);
                s.push(0, 1, 1500);
                s
            },
            #[cfg(feature = "titan")]
            titan_memory: None,
            has_text: false,
            has_vision: false,
            has_audio: false,
            vocabulary: HashMap::new(),
        };
        model.neurons.threshold[1] = 1000;

        let mut runtime = Runtime {
            model,
            backend: Box::new(CpuBackend::default()),
            previous_spikes: vec![false; 2],
            tick_counter: 0,
            spikes_history: Vec::new(),
            network_manager: None,
        };
        runtime.previous_spikes[0] = true;

        let spikes = runtime.tick(&[0, 0]);
        assert!(spikes[1]);
    }

    #[test]
    fn test_night_phase_pruning() {
        let model = BakedModel {
            node_id: 0,
            local_range: (0, 2),
            neurons: NeuronsSoA::new(2),
            synapses: {
                let mut s = SynapsesSoA::with_capacity(1);
                s.push(0, 1, 5);
                s
            },
            #[cfg(feature = "titan")]
            titan_memory: None,
            has_text: false,
            has_vision: false,
            has_audio: false,
            vocabulary: HashMap::new(),
        };

        let mut runtime = Runtime {
            model,
            backend: Box::new(CpuBackend::default()),
            previous_spikes: vec![false; 2],
            tick_counter: 99,
            spikes_history: Vec::new(),
            network_manager: None,
        };

        runtime.tick(&[0, 0]);
        assert_eq!(runtime.model.synapses.len(), 0);
    }
}
