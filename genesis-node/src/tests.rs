#[cfg(test)]
mod tests {
    use crate::{Runtime, Observer};
    use genesis_core::{BakedModel, NeuronsSoA, SynapsesSoA};
    use genesis_compute::CpuBackend;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    #[test]
    fn test_runtime_synapse_propagation() {
        let mut model = BakedModel {
            config: genesis_core::NetworkConfig::default(),
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
            has_robotics: false,
            has_fusion: false,
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
            observer: Observer { max_spikes_per_tick: 10, total_energy_consumed: 0 },
            remote_spike_queue: Arc::new(Mutex::new(Vec::new())),
            telemetry: crate::Telemetry { spike_counts: Vec::new(), episode_rewards: Vec::new() },
        };
        runtime.previous_spikes[0] = true;

        let spikes = runtime.tick(&[0, 0]);
        assert!(spikes[1]);
    }

    #[test]
    fn test_night_phase_pruning() {
        let model = BakedModel {
            config: genesis_core::NetworkConfig::default(),
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
            has_robotics: false,
            has_fusion: false,
            vocabulary: HashMap::new(),
        };

        let mut runtime = Runtime {
            model,
            backend: Box::new(CpuBackend::default()),
            previous_spikes: vec![false; 2],
            tick_counter: 99,
            spikes_history: Vec::new(),
            network_manager: None,
            observer: Observer { max_spikes_per_tick: 10, total_energy_consumed: 0 },
            remote_spike_queue: Arc::new(Mutex::new(Vec::new())),
            telemetry: crate::Telemetry { spike_counts: Vec::new(), episode_rewards: Vec::new() },
        };

        runtime.tick(&[0, 0]);
        assert_eq!(runtime.model.synapses.len(), 0);
    }
}
