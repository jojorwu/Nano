#[cfg(test)]
mod tests_advanced {
    use crate::{Runtime, SimulationObserver, SimulationEngine};
    use genesis_core::{BakedModel, NeuronsSoA, SynapsesSoA, ModuleManager, SpikeData, SCALE};
    use genesis_compute::CpuBackend;
    use std::collections::HashMap;

    fn create_advanced_test_runtime(n_count: usize) -> Runtime {
        let mut config = genesis_core::NetworkConfig::default();
        config.noise_amplitude = 0;

        let mut neurons = NeuronsSoA::new(n_count);
        // Assign neurons to blocks
        for i in 0..n_count {
            neurons.block_id[i] = (i / 4) as u32; // 4 neurons per block
        }

        let model = BakedModel {
            version: "4.2".to_string(),
            config,
            node_id: 0,
            local_range: (0, n_count),
            neurons,
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

        let settings = crate::SimulationSettings::default();
        let mut modules = ModuleManager::new();

        // Add AttnRes and Titan modules
        modules.add_module(Box::new(genesis_core::AttnResModule::new((n_count + 3) / 4)));
        modules.add_module(Box::new(genesis_core::titan::BitWiseTitan::new(100)));

        let backend = Box::new(CpuBackend::default());
        let observers: Vec<Box<dyn SimulationObserver>> = Vec::new();

        Runtime {
            engine: SimulationEngine::new(model, modules, backend, &settings),
            settings,
            episode_reward_history: Vec::new(),
            network_manager: None,
            observers,
        }
    }

    #[test]
    fn test_bitpacked_history_integrity() {
        let mut rt = create_advanced_test_runtime(64);

        // Trigger some spikes
        let inputs = vec![2000; 64];
        rt.tick(&inputs);

        let hist_ptr = rt.engine.state.history_ptr;
        let last_idx = if hist_ptr == 0 { rt.engine.state.spikes_history.len() - 1 } else { hist_ptr - 1 };
        let history = &rt.engine.state.spikes_history[last_idx];
        if let SpikeData::BitPacked(packed) = history {
            assert!(packed.len() >= 1);
            assert!(packed[0] > 0, "Spikes should be recorded in bitpacked history");
        } else {
            panic!("History should be BitPacked: {:?}", history);
        }
    }

    #[test]
    fn test_bitwise_titan_associative_learning() {
        let mut rt = create_advanced_test_runtime(16);

        // Titan learning requires surprise and specific history
        // Phase 1: Co-activate Block 0 (Input) and Block 1 (Output)
        // Step 1: Spike Block 0
        let mut inputs = vec![0; 16];
        for i in 0..4 { inputs[i] = 2000; }
        rt.tick(&inputs);

        // Step 2: Spike Block 1 immediately after with high surprise
        let mut inputs = vec![0; 16];
        for i in 4..8 { inputs[i] = 2000; }
        // Ensure surprise is high by changing activity levels drastically
        rt.tick(&inputs);

        // Force Titan learn_from_history manually for test if pipeline didn't trigger
        // Need to use the PREVIOUS history index since we just ticked
        let history = rt.engine.state.spikes_history.clone();
        let h_ptr = if rt.engine.state.history_ptr == 0 { history.len() - 1 } else { rt.engine.state.history_ptr - 1 };

        for m in &mut rt.engine.modules.modules {
            if m.name() == "titan" {
                if let Ok(mut titan) = bincode::deserialize::<genesis_core::titan::BitWiseTitan>(&m.get_state()) {
                    titan.learn_from_history(&history, h_ptr, &rt.engine.model.neurons, 1000);
                    m.set_state(&bincode::serialize(&titan).unwrap());
                }
            }
        }

        // Phase 2: Check retrieval
        // Clear activity
        for _ in 0..5 { rt.tick(&[0; 16]); }

        // Ensure indices are updated (CpuBackend)
        rt.engine.backend.rebuild_index(&rt.engine.model);

        // Trigger Block 0 again
        let mut inputs = vec![0; 16];
        for i in 0..4 { inputs[i] = 2000; }
        rt.tick(&inputs);

        // Next tick should have distal potential in Block 1
        rt.tick(&[0; 16]);

        let mut retrieved = false;
        for i in 4..8 {
            if rt.engine.model.neurons.distal_potential[i] > 0 { retrieved = true; }
        }
        assert!(retrieved, "Titan should have retrieved association for Block 1 neurons");
    }

    #[test]
    fn test_predictive_coding_reward() {
        let mut rt = create_advanced_test_runtime(8);
        // Connect 0 -> 4 via Titan association manually for test
        for m in &mut rt.engine.modules.modules {
            if m.name() == "titan" {
                let mut titan = bincode::deserialize::<genesis_core::titan::BitWiseTitan>(&m.get_state()).unwrap();
                titan.update_block(0, vec![genesis_core::titan::Association { target: 4, weight: 10 }]);
                m.set_state(&bincode::serialize(&titan).unwrap());
            }
        }

        // 1. Prediction Success:
        // Tick 1: Block 0 fires (it's in block_id 0)
        let mut inputs = vec![0; 8];
        for i in 0..4 { inputs[i] = 2000; }
        rt.tick(&inputs);

        // Tick 2: Neuron 4 fires via external input (Proximal match)
        let mut inputs = vec![0; 8];
        inputs[4] = 2000;
        rt.tick(&inputs);

        // Titan learning needs to be forced for test if pipeline didn't trigger (needs surprise)
        let history = rt.engine.state.spikes_history.clone();
        let h_ptr = if rt.engine.state.history_ptr == 0 { history.len() - 1 } else { rt.engine.state.history_ptr - 1 };

        for m in &mut rt.engine.modules.modules {
            if m.name() == "titan" {
                if let Ok(mut titan) = bincode::deserialize::<genesis_core::titan::BitWiseTitan>(&m.get_state()) {
                    titan.learn_from_history(&history, h_ptr, &rt.engine.model.neurons, 1000);
                    m.set_state(&bincode::serialize(&titan).unwrap());
                }
            }
        }

        // Verify weight of association 0 -> 4 increased in Titan (Predictive Success)
        for m in &rt.engine.modules.modules {
            if m.name() == "titan" {
                let titan = bincode::deserialize::<genesis_core::titan::BitWiseTitan>(&m.get_state()).unwrap();
                let start = titan.block_offsets[0] as usize;
                let assoc = &titan.associations_flat[start];
                assert!(assoc.weight > 10, "Weight was {}, should increase on successful prediction", assoc.weight);
            }
        }
    }

    #[test]
    fn test_prediction_error_ltd() {
        use genesis_core::{PlasticityRule, PlasticityContext, Compartment};
        use genesis_core::GsopRule;
        let gsop = GsopRule { learning_rate: 100 };
        let neurons = NeuronsSoA::new(1);

        let mut weight = 1000;
        // Case: Memory triggered (pre_spiked=true) but neuron DID NOT fire (post_spiked=false)
        // This is a prediction error that should trigger LTD.
        let ctx = PlasticityContext {
            pre_spiked: true, post_spiked: false, backprop_signal: 0,
            compartment: Compartment::Distal, // Memory
            reward: None, neuromodulation: Default::default(),
            pre_last_spike: 5, post_last_spike: 0, current_tick: 10,
            post_index: 0, neurons: &neurons,
        };

        gsop.apply(&mut weight, &ctx);
        assert!(weight < 1000, "Weight should decrease (LTD) on prediction error, got {}", weight);
    }

    #[test]
    fn test_burst_mode_convergence() {
        let mut rt = create_advanced_test_runtime(16);
        let mut inputs = vec![0; 16];
        inputs[0] = 2000;

        // Should finish processing burst eventually
        let result = rt.process_burst(&inputs, 10);
        assert!(result.iter().any(|&s| s), "Burst should produce at least some spikes");
    }

    #[test]
    fn test_scaling_memory_efficiency() {
        let n_count = 1_000_000;
        // Just verify we can allocate 1M neurons with blocks and bit-history
        let mut rt = create_advanced_test_runtime(n_count);

        // Run one tick
        let inputs = vec![0; 0];
        rt.tick(&inputs);

        let packed_size = if let SpikeData::BitPacked(p) = &rt.engine.state.spikes_history[0] {
            p.len() * 8 // bytes
        } else { 0 };

        // 1M bits = 125,000 bytes. With 64-bit packing and 16 history slots.
        assert!(packed_size < 130_000, "Packed history should be efficient, got {} bytes", packed_size);
    }

    #[test]
    fn test_consolidation_phase_strengthening() {
        let mut rt = create_advanced_test_runtime(16);
        // Block 0 is neurons 0-3. Neuron 8 is in Block 2.
        // Add a weak association manually: Block 0 -> Neuron 8
        for m in &mut rt.engine.modules.modules {
            if m.name() == "titan" {
                let mut titan = bincode::deserialize::<genesis_core::titan::BitWiseTitan>(&m.get_state()).unwrap();
                titan.update_block(0, vec![genesis_core::titan::Association { target: 8, weight: 1 }]);
                m.set_state(&bincode::serialize(&titan).unwrap());
            }
        }

        // Fill history with Block 0 firing (Step 1)
        let mut inputs = vec![0; 16];
        for i in 0..4 { inputs[i] = 2000; }
        rt.tick(&inputs);

        // Followed by Neuron 8 firing (Step 2)
        let mut inputs = vec![0; 16];
        inputs[8] = 2000;
        rt.tick(&inputs);

        // Run consolidation (it will force learning from history)
        rt.consolidate_memory(5);

        // Association 0 -> 8 should be strengthened
        for m in &rt.engine.modules.modules {
            if m.name() == "titan" {
                let titan = bincode::deserialize::<genesis_core::titan::BitWiseTitan>(&m.get_state()).unwrap();
                assert!(titan.block_counts[0] > 0, "Association list for Block 0 should exist");
                let start = titan.block_offsets[0] as usize;
                let count = titan.block_counts[0] as usize;
                let assoc = &titan.associations_flat[start..start+count];
                let match_assoc = assoc.iter().find(|a| a.target == 8).expect("Association 0 -> 8 should exist");
                assert!(match_assoc.weight > 1, "Consolidation should strengthen associations, got {}", match_assoc.weight);
            }
        }
    }
}
