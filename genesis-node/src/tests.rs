#[cfg(test)]
mod tests {
    use crate::{Runtime, Observer, Telemetry};
    use genesis_core::{BakedModel, NeuronsSoA, SynapsesSoA, ModuleManager};
    use genesis_compute::CpuBackend;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    #[test]
    fn test_runtime_synapse_propagation() {
        let mut model = BakedModel {
            version: "3.7".to_string(),
            config: genesis_core::NetworkConfig::default(),
            node_id: 0,
            local_range: (0, 2),
            neurons: NeuronsSoA::new(2),
            synapses: {
                let mut s = SynapsesSoA::with_capacity(1);
                s.push(0, 1, 1500);
                s
            },
            module_states: HashMap::new(),
            #[cfg(feature = "titan")]
            titan_memory: None,
            has_text: false,
            has_vision: false,
            has_audio: false,
            has_robotics: false,
            has_fusion: false,
            vocabulary: HashMap::new(),
        };
        model.neurons.threshold[1] = genesis_core::SCALE;

        let mut runtime = Runtime {
            model,
            modules: {
                let mut mm = ModuleManager::new();
                mm.register_factory("titan", || Box::new(genesis_core::titan::TitanMemory::new(10, 100)));
                mm
            },
            settings: crate::SimulationSettings::default(),
            backend: Box::new(CpuBackend::default()),
            previous_spikes: vec![false; 2],
            current_spikes_buffer: vec![false; 2],
            merged_inputs_buffer: vec![0; 2],
            tick_counter: 0,
            spikes_history: vec![crate::SpikeData::Sparse(Vec::new()); 100],
            history_ptr: 0,
            episode_reward_history: Vec::new(),
            global_modulators: genesis_core::NeuromodulationState::default(),
            rolling_spike_count: 0.0,
            network_manager: None,
            observer: Observer { max_spikes_per_tick: 10, total_energy_consumed: 0, energy_budget_per_tick: 100, current_energy_usage: 0 },
            remote_spike_queue: Arc::new(Mutex::new(Vec::new())),
            telemetry: Telemetry::default(),
        };
        runtime.previous_spikes[0] = true;

        let spikes = runtime.tick(&[0, 0]);
        assert!(spikes[1]);
    }

    #[test]
    fn test_night_phase_pruning() {
        let model = BakedModel {
            version: "3.7".to_string(),
            config: genesis_core::NetworkConfig::default(),
            node_id: 0,
            local_range: (0, 2),
            neurons: NeuronsSoA::new(2),
            synapses: {
                let mut s = SynapsesSoA::with_capacity(1);
                s.push(0, 1, 5);
                s
            },
            module_states: HashMap::new(),
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
            modules: {
                let mut mm = ModuleManager::new();
                mm.register_factory("titan", || Box::new(genesis_core::titan::TitanMemory::new(10, 100)));
                mm
            },
            settings: crate::SimulationSettings::default(),
            backend: Box::new(CpuBackend::default()),
            previous_spikes: vec![false; 2],
            current_spikes_buffer: vec![false; 2],
            merged_inputs_buffer: vec![0; 2],
            tick_counter: 99,
            spikes_history: vec![crate::SpikeData::Sparse(Vec::new()); 100],
            history_ptr: 0,
            episode_reward_history: Vec::new(),
            global_modulators: genesis_core::NeuromodulationState::default(),
            rolling_spike_count: 0.0,
            network_manager: None,
            observer: Observer { max_spikes_per_tick: 10, total_energy_consumed: 0, energy_budget_per_tick: 100, current_energy_usage: 0 },
            remote_spike_queue: Arc::new(Mutex::new(Vec::new())),
            telemetry: Telemetry::default(),
        };

        runtime.tick(&[0, 0]);
        assert_eq!(runtime.model.synapses.len(), 0);
    }

    #[test]
    fn test_observer_graceful_degradation() {
        let mut model = BakedModel {
            version: "3.7".to_string(),
            config: genesis_core::NetworkConfig::default(),
            node_id: 0,
            local_range: (0, 10),
            neurons: NeuronsSoA::new(10),
            synapses: SynapsesSoA::default(),
            module_states: HashMap::new(),
            #[cfg(feature = "titan")]
            titan_memory: None,
            has_text: false,
            has_vision: false,
            has_audio: false,
            has_robotics: false,
            has_fusion: false,
            vocabulary: HashMap::new(),
        };
        let initial_threshold = model.neurons.threshold[0];

        let mut observer = Observer {
            max_spikes_per_tick: 100,
            energy_budget_per_tick: 2, // Very low budget
            current_energy_usage: 0,
            total_energy_consumed: 0,
        };

        let mut spikes = vec![true; 5]; // 5 spikes > 2 budget
        spikes.extend(vec![false; 5]);

        observer.process_spikes(&mut spikes, &mut model);

        // Threshold should have increased due to budget violation
        assert!(model.neurons.threshold[0] > initial_threshold);
        assert_eq!(observer.current_energy_usage, 5);
    }

    #[test]
    fn test_elias_fano_roundtrip() {
        use crate::{SpikePacket, SpikeData};
        let indices = vec![10, 50, 100, 250];
        let universe = 1000;

        let compressed = SpikePacket::compress_indices(&indices, universe);
        let packet = SpikePacket { tick: 1, data: SpikeData::Compressed(compressed) };

        let queue = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        // Simulate network run logic manually
        match packet.data {
            SpikeData::Compressed(data) => {
                let mut q = queue.lock().unwrap();
                let count = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
                let low_bits = data[4] as u32;
                let mut bit_ptr = 40usize;
                let mut current_high = 0u32;
                for _ in 0..count {
                    while bit_ptr / 8 < data.len() && (data[bit_ptr / 8] >> (bit_ptr % 8)) & 1 == 0 {
                        current_high += 1;
                        bit_ptr += 1;
                    }
                    bit_ptr += 1;
                    let mut low = 0u32;
                    for i in 0..low_bits {
                        if bit_ptr / 8 < data.len() && (data[bit_ptr / 8] >> (bit_ptr % 8)) & 1 == 1 {
                            low |= 1 << i;
                        }
                        bit_ptr += 1;
                    }
                    q.push(((current_high << low_bits) | low) as usize);
                }
            }
            _ => panic!("Expected compressed data"),
        }

        let mut result = queue.lock().unwrap();
        result.sort_unstable();
        assert_eq!(*result, indices);
    }
}
