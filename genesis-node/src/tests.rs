#[cfg(test)]
mod tests {
    use crate::{Runtime, Observer, Telemetry, SimulationEngine, SimulationObserver};
    use genesis_core::{BakedModel, NeuronsSoA, SynapsesSoA, ModuleManager, SpikeData};
    use genesis_compute::CpuBackend;
    use std::collections::HashMap;

    #[allow(dead_code)]
    fn create_test_runtime(n_count: usize) -> Runtime {
        create_test_runtime_with_backend(n_count, "cpu")
    }

    #[allow(dead_code)]
    fn create_test_runtime_with_backend(n_count: usize, backend_name: &str) -> Runtime {
        create_test_runtime_internal(n_count, backend_name, true)
    }

    fn create_test_runtime_internal(n_count: usize, backend_name: &str, with_observer: bool) -> Runtime {
        let mut config = genesis_core::NetworkConfig::default();
        config.physics.noise_amplitude = 0; // High precision parity requires deterministic environment

        let model = BakedModel {
            version: "4.2".to_string(),
            config,
            node_id: 0,
            local_range: (0, n_count),
            neurons: NeuronsSoA::new(n_count),
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

        let mut settings = crate::SimulationSettings::default();
        settings.preferred_backend = Some(backend_name.to_string());

        let modules = ModuleManager::new();
        let registry = genesis_compute::BackendRegistry::new();
        let backend = registry.create(backend_name).unwrap_or_else(|| Box::new(CpuBackend::default()));

        let mut observers: Vec<Box<dyn SimulationObserver>> = Vec::new();
        if with_observer {
            observers.push(Box::new(Observer::new(n_count)));
        }
        observers.push(Box::new(Telemetry::default()));

        Runtime {
            engine: SimulationEngine::new(model, modules, backend, &settings).unwrap(),
            settings,
            episode_reward_history: Vec::new(),
            network_manager: None,
            observers,
            last_surprise: 0,
            surprise_history: Vec::new(),
            titan_rx: None,
        }
    }

    #[test]
    fn test_stp_dynamics_parity() {
        #[cfg(feature = "cpp")]
        {
            let _n_count = 10;
            let mut rt_cpu = create_test_runtime_with_backend(n_count, "cpu");
            let mut rt_cpp = create_test_runtime_with_backend(n_count, "cpp");

            for i in 0..n_count {
                // High weight to trigger strong STP
                rt_cpu.engine.model.synapses.push(i as u32, ((i + 1) % n_count) as u32, 2048);
                rt_cpp.engine.model.synapses.push(i as u32, ((i + 1) % n_count) as u32, 2048);
            }

            // Burst input to trigger depression
            let inputs = vec![2000; n_count];

            for t in 0..20 {
                rt_cpu.tick(&inputs);
                rt_cpp.tick(&inputs);

                for i in 0..rt_cpu.engine.model.synapses.len() {
                    let res_cpu = rt_cpu.engine.model.synapses.stp_resources[i];
                    let res_cpp = rt_cpp.engine.model.synapses.stp_resources[i];
                    assert_eq!(res_cpu, res_cpp, "STP Resource mismatch at synapse {} tick {}", i, t);

                    let calc_cpu = rt_cpu.engine.model.synapses.stp_calcium[i];
                    let calc_cpp = rt_cpp.engine.model.synapses.stp_calcium[i];
                    assert_eq!(calc_cpu, calc_cpp, "STP Calcium mismatch at synapse {} tick {}", i, t);
                }
            }
        }
    }

    #[test]
    fn test_dendritic_gating_parity() {
        #[cfg(feature = "cpp")]
        {
            let n_count = 5;
            let mut rt_cpu = create_test_runtime_with_backend(n_count, "cpu");
            let mut rt_cpp = create_test_runtime_with_backend(n_count, "cpp");

            for i in 0..n_count {
                // Connect 0 -> 1 Distally
                rt_cpu.engine.model.synapses.push_to_compartment(0, 1, 1000, 1, genesis_core::Compartment::Distal);
                rt_cpp.engine.model.synapses.push_to_compartment(0, 1, 1000, 1, genesis_core::Compartment::Distal);
            }

            // Manually set dendritic gate to fully open for neuron 1
            rt_cpu.engine.model.neurons.dendritic_gate[1] = genesis_core::SCALE;
            rt_cpp.engine.model.neurons.dendritic_gate[1] = genesis_core::SCALE;

            // Stimulate 0 (source) to fire
            rt_cpu.engine.state.previous_spikes[0] = true;
            rt_cpp.engine.state.previous_spikes[0] = true;

            // Tick: Source 0 spikes. Target 1 receives 1500 proximally via external inputs,
            // which should open its somatic gate for the distal input from 0.
            let inputs = vec![0, 1500, 0, 0, 0];
            rt_cpu.tick(&inputs);
            rt_cpp.tick(&inputs);

            let pot_cpu = rt_cpu.engine.model.neurons.potential[1];
            let pot_cpp = rt_cpp.engine.model.neurons.potential[1];
            assert_eq!(pot_cpu, pot_cpp, "Gated potential mismatch");
            // 1500 (prox) + 1000 (dist) = 2500. After decay and firing reset it might be 0.
            // Let's check if it fired.
            assert!(rt_cpu.engine.state.previous_spikes[1], "Target neuron should have fired");
        }
    }

    #[test]
    fn test_runtime_synapse_propagation_high_precision() {
        let mut runtime = create_test_runtime_internal(2, "cpu", false);
        // Precise weight: 1.5 * SCALE
        runtime.engine.model.synapses.push(0, 1, 1536);
        runtime.engine.model.neurons.threshold[1] = 10000; // high threshold to prevent firing
        runtime.engine.state.previous_spikes[0] = true;

        // Ensure indices are rebuilt
        runtime.engine.backend.rebuild_index(&runtime.engine.model);

        let _spikes = runtime.tick(&[0, 0]);

        // Calculation:
        // Weight = 1536
        // Gate = 1024 (default)
        // Gated weight = (1536 * 1024) >> 10 = 1536
        // Pot_after_prop = prox_pot = 1536
        // Decay = 50
        // liquid_mod = (abs(1536) + 0) * 10 >> 10 = 15360 >> 10 = 15
        // final_decay = 50 - 15 = 35
        // final_pot = (1536 * (1024 - 35)) >> 10 = (1536 * 989) >> 10 = 1519104 >> 10 = 1483

        let expected_pot = (1536i64 * (1024 - 35)) >> 10;
        assert_eq!(runtime.engine.model.neurons.potential[1], expected_pot as i32);
    }

    #[test]
    fn test_backend_numerical_parity_high_precision() {
        #[cfg(feature = "cpp")]
        {
            let n_count = 100;
            let mut rt_cpu = create_test_runtime_with_backend(n_count, "cpu");
            let mut rt_cpp = create_test_runtime_with_backend(n_count, "cpp");

            // Seed both with same complex topology
            for i in 0..n_count {
                let i = i as u32;
                let n = n_count as u32;
                rt_cpu.engine.model.synapses.push(i, (i + 1) % n, 1000 + i as i32);
                rt_cpp.engine.model.synapses.push(i, (i + 1) % n, 1000 + i as i32);

                rt_cpu.engine.model.synapses.push_to_compartment(i, (i + 5) % n, 500, 1, genesis_core::Compartment::Distal);
                rt_cpp.engine.model.synapses.push_to_compartment(i, (i + 5) % n, 500, 1, genesis_core::Compartment::Distal);
            }

            let inputs = (0..n_count).map(|i| (i * 10) as i32).collect::<Vec<_>>();

            for t in 0..50 {
                let spikes_cpu = rt_cpu.tick(&inputs);
                let spikes_cpp = rt_cpp.tick(&inputs);

                assert_eq!(spikes_cpu, spikes_cpp, "Spike parity failed at tick {}", t);

                for i in 0..n_count {
                    let pot_cpu = rt_cpu.engine.model.neurons.potential[i];
                    let pot_cpp = rt_cpp.engine.model.neurons.potential[i];
                    assert_eq!(pot_cpu, pot_cpp, "Potential mismatch at neuron {} tick {}", i, t);

                    let thr_cpu = rt_cpu.engine.model.neurons.threshold[i];
                    let thr_cpp = rt_cpp.engine.model.neurons.threshold[i];
                    assert_eq!(thr_cpu, thr_cpp, "Threshold mismatch at neuron {} tick {}", i, t);
                }
            }
        }
    }

    #[test]
    fn test_stdp_homeostasis_high_precision() {
        use genesis_core::{PlasticityRule, PlasticityContext, Compartment};
        use genesis_core::plasticity::StdpRule;
        let stdp = StdpRule { tau: 10, a_plus: 100, a_minus: 100, reward_scale: 1024 };
        let mut neurons = NeuronsSoA::new(1);

        let config = genesis_core::NetworkConfig::default();
        // Low activity -> should result in larger weight increase
        neurons.activity_ema[0] = 50;
        let mut weight_low = 1000;
        let ctx_low = PlasticityContext {
            pre_spiked: true, post_spiked: true, backprop_signal: 0,
            prediction_error: 0,
            compartment: Compartment::Proximal,
            reward: None,
            neuromodulation: Default::default(),
            pre_last_spike: 5, post_last_spike: 7, current_tick: 10,
            post_index: 0,
            neurons: &neurons,
            config: &config,
        };
        stdp.apply(&mut weight_low, &ctx_low);
        let delta_low = weight_low - 1000;

        // High activity -> Homeostasis should dampen LTP
        neurons.activity_ema[0] = 500;
        let mut weight_high = 1000;
        let ctx_high = PlasticityContext {
            pre_spiked: true, post_spiked: true, backprop_signal: 0,
            prediction_error: 0,
            compartment: Compartment::Proximal,
            reward: None,
            neuromodulation: Default::default(),
            pre_last_spike: 5, post_last_spike: 7, current_tick: 10,
            post_index: 0,
            neurons: &neurons,
            config: &config,
        };
        stdp.apply(&mut weight_high, &ctx_high);
        let delta_high = weight_high - 1000;

        assert!(delta_high < delta_low, "Homeostatic precision check failed: delta_high ({}) >= delta_low ({})", delta_high, delta_low);
    }

    #[test]
    fn test_elias_fano_roundtrip_precision() {
        use crate::SpikePacket;
        let indices = vec![1, 10, 100, 500, 999];
        let universe = 1000;

        let compressed = SpikePacket::compress_indices(&indices, universe);
        // High precision: compressed size should be significantly smaller than raw indices
        assert!(compressed.len() < indices.len() * 4);

        let packet = SpikePacket { tick: 1, data: SpikeData::Compressed(compressed) };
        let queue = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));

        // Use real NetworkManager logic (extracted)
        if let SpikeData::Compressed(data) = packet.data {
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

        let mut result = queue.lock().unwrap();
        result.sort_unstable();
        assert_eq!(*result, indices, "Elias-Fano precision roundtrip failed");
    }

    #[test]
    fn test_sfa_high_precision() {
        let mut runtime = create_test_runtime_internal(1, "cpu", false);
        runtime.engine.model.neurons.threshold[0] = 500;
        runtime.engine.model.neurons.base_threshold[0] = 500;
        runtime.engine.model.neurons.decay[0] = 0;
        runtime.engine.model.config.plasticity.ip_increment = 0;
        runtime.engine.model.neurons.dendritic_gate[0] = 1024;

        // Tick 1: Input 2000 -> Fires. Adaptation -> 100.
        let spikes1 = runtime.tick(&[2000]);
        assert!(spikes1[0], "Neuron 0 should have fired in tick 1.");
        assert_eq!(runtime.engine.model.neurons.adaptation_current[0], 100);

        // Tick 2: Input 2000.
        // In the tick loop, previous_spikes[0] is TRUE.
        // But the neuron just fired, so it should be in REFRACTORY state.
        // Refractory mult = 1 + (1 << 4) = 17.
        // Effective threshold = 500 * 17 = 8500.
        // Potential 2000 < 8500 -> NO FIRE.

        let spikes2 = runtime.tick(&[2000]);
        assert!(!spikes2[0], "Neuron 0 should be refractory in tick 2.");

        // Tick 3, 4, 5, 6: still refractory (timer counts down 4, 3, 2, 1, 0)
        for _ in 0..4 { runtime.tick(&[2000]); }

        // Tick 7: Should fire again.
        let spikes7 = runtime.tick(&[2000]);
        assert!(spikes7[0], "Neuron 0 should fire again after refractory period.");
        assert!(runtime.engine.model.neurons.adaptation_current[0] > 100);
    }
}
