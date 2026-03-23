#[cfg(test)]
mod tests {
    use crate::{Runtime, Observer, Telemetry, SimulationEngine};
    use genesis_core::{BakedModel, NeuronsSoA, SynapsesSoA, ModuleManager, SpikeData};
    use genesis_compute::CpuBackend;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    fn create_test_runtime(n_count: usize) -> Runtime {
        let model = BakedModel {
            version: "4.2".to_string(),
            config: genesis_core::NetworkConfig::default(),
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

        let settings = crate::SimulationSettings::default();
        let modules = ModuleManager::new();
        let backend = Box::new(CpuBackend::default());

        Runtime {
            engine: SimulationEngine::new(model, modules, backend, &settings),
            settings,
            episode_reward_history: Vec::new(),
            network_manager: None,
            observers: vec![Box::new(Observer::new(n_count))],
            telemetry: Telemetry::default(),
        }
    }

    #[test]
    fn test_runtime_synapse_propagation() {
        let mut runtime = create_test_runtime(2);
        runtime.engine.model.synapses.push(0, 1, 1500);
        runtime.engine.model.neurons.threshold[1] = genesis_core::SCALE;
        runtime.engine.previous_spikes[0] = true;

        let spikes = runtime.tick(&[0, 0]);
        assert!(spikes[1]);
    }

    #[test]
    fn test_night_phase_pruning() {
        let mut runtime = create_test_runtime(2);
        runtime.engine.model.synapses.push(0, 1, 5);
        runtime.engine.tick_counter = 99;

        runtime.tick(&[0, 0]);
        assert_eq!(runtime.engine.model.synapses.len(), 0);
    }

    #[test]
    fn test_observer_graceful_degradation() {
        let mut runtime = create_test_runtime(10);
        let initial_threshold = runtime.engine.model.neurons.threshold[0];

        for obs in &mut runtime.observers {
            if let Some(o) = obs.as_any_mut().downcast_mut::<Observer>() {
                o.energy_budget_per_tick = 2; // Very low budget
            }
        }

        let mut spikes = vec![true; 5]; // 5 spikes > 2 budget
        spikes.extend(vec![false; 5]);

        for obs in &mut runtime.observers {
            if let Some(o) = obs.as_any_mut().downcast_mut::<Observer>() {
                o.process_spikes(&mut spikes, &mut runtime.engine.model);
            }
        }

        // Threshold should have increased due to budget violation
        assert!(runtime.engine.model.neurons.threshold[0] > initial_threshold);

        for obs in &mut runtime.observers {
            if let Some(o) = obs.as_any_mut().downcast_mut::<Observer>() {
                assert_eq!(o.current_energy_usage, 5);
            }
        }
    }

    #[test]
    fn test_elias_fano_roundtrip() {
        use crate::SpikePacket;
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

    #[test]
    fn test_temporal_delay_consistency() {
        let mut runtime = create_test_runtime(2);
        // Delay of 3 ticks
        runtime.engine.model.synapses.push_delayed(0, 1, 1500, 3);
        runtime.engine.model.neurons.threshold[1] = genesis_core::SCALE;

        // Fire neuron 0 at T=1
        runtime.tick(&[2000, 0]);
        assert!(runtime.engine.previous_spikes[0]);
        assert!(!runtime.engine.previous_spikes[1]);

        // T=2: signal in transit
        runtime.tick(&[0, 0]);
        assert!(!runtime.engine.previous_spikes[1]);

        // T=3: signal in transit
        runtime.tick(&[0, 0]);
        assert!(!runtime.engine.previous_spikes[1]);

        // T=4: signal arrives (Delay=3 means t+3)
        runtime.tick(&[0, 0]);
        assert!(runtime.engine.previous_spikes[1], "Signal should have arrived at T=4");
    }

    #[test]
    fn test_stdp_homeostasis() {
        use genesis_core::{PlasticityRule, PlasticityContext, Compartment};
        use genesis_core::plasticity::StdpRule;
        let stdp = StdpRule { tau: 10, a_plus: 100, a_minus: 100, reward_scale: 1024 };
        let mut neurons = NeuronsSoA::new(1);

        // Scenario 1: Low activity -> Strong LTP
        neurons.activity_ema[0] = 50;
        let mut weight_low = 1000;
        let ctx_low = PlasticityContext {
            pre_spiked: true, post_spiked: true, backprop_signal: 0,
            compartment: Compartment::Proximal,
            reward: None,
            neuromodulation: Default::default(),
            pre_last_spike: 5, post_last_spike: 7, current_tick: 10,
            post_index: 0,
            neurons: &neurons,
        };
        stdp.apply(&mut weight_low, &ctx_low);
        let delta_low = weight_low - 1000;

        // Scenario 2: High activity -> Weak LTP (Homeostasis)
        neurons.activity_ema[0] = 500;
        let mut weight_high = 1000;
        let ctx_high = PlasticityContext {
            pre_spiked: true, post_spiked: true, backprop_signal: 0,
            compartment: Compartment::Proximal,
            reward: None,
            neuromodulation: Default::default(),
            pre_last_spike: 5, post_last_spike: 7, current_tick: 10,
            post_index: 0,
            neurons: &neurons,
        };
        stdp.apply(&mut weight_high, &ctx_high);
        let delta_high = weight_high - 1000;

        assert!(delta_high < delta_low, "LTP should be suppressed by high activity");
    }

    #[test]
    fn test_multimodal_concurrent_injection() {
        let n_count = 10;
        let mut runtime = create_test_runtime(n_count);
        // Tier 0 modules
        runtime.engine.modules.add_module(Box::new(genesis_core::text::TextProcessorModule::new(n_count)));
        runtime.engine.modules.add_module(Box::new(genesis_core::vision::VisionModule::new(1, n_count as u32)));
        // Tier 1 module
        runtime.engine.modules.add_module(Box::new(genesis_core::fusion::SpikingFusionModule::new(vec![5])));

        runtime.inject_text("hello");
        #[cfg(feature = "vision")]
        runtime.inject_image(&vec![255; n_count]);

        runtime.tick(&vec![0; n_count]);

        // Fusion neuron (index 5) should have received combined signals
        assert!(runtime.engine.input_bus.proximal[5].load(std::sync::atomic::Ordering::Relaxed) > 0);
    }

    #[test]
    fn test_sfa_dynamics() {
        let mut runtime = create_test_runtime(1);
        runtime.engine.model.neurons.decay[0] = 0; // No decay for simpler tracking
        runtime.engine.model.neurons.threshold[0] = 500;
        runtime.engine.model.neurons.base_threshold[0] = 500;

        // Stimulate constantly
        let mut spikes = 0;
        for _ in 0..50 {
            let res = runtime.tick(&[1000]);
            if res[0] { spikes += 1; }
        }

        // Without SFA, it would fire every tick (since potential=1000, thresh=500+adaptation).
        // With SFA, adaptation current builds up and slows down the firing.
        assert!(spikes < 50, "SFA should have reduced the firing rate");
        assert!(runtime.engine.model.neurons.adaptation_current[0] > 0);
    }

    #[test]
    fn test_think_command() {
        let mut runtime = create_test_runtime(1);
        runtime.engine.modules.instantiate("think");

        // Use handle_command
        let res = runtime.handle_command("set_think ticks 10");
        assert_eq!(res, "Think settings updated");

        // Verify state via serialize/deserialize (standard flow in sync_modules_to_model)
        runtime.sync_modules_to_model();
        let state = runtime.engine.model.module_states.get("think").unwrap();
        let think: genesis_core::ThinkModule = bincode::deserialize(state).unwrap();
        assert_eq!(think.extra_ticks, 10);
    }

    #[test]
    fn test_temporal_delay_consistency_multi() {
        let mut runtime = create_test_runtime(3);
        runtime.engine.model.synapses.push_delayed(0, 1, 1500, 2); // 0 -> 1, delay 2
        runtime.engine.model.synapses.push_delayed(0, 2, 1500, 5); // 0 -> 2, delay 5
        runtime.engine.model.neurons.threshold.fill(genesis_core::SCALE);

        // Tick 1: Fire neuron 0
        runtime.tick(&[2000, 0, 0]);
        assert!(runtime.engine.previous_spikes[0]);

        // Tick 2: Signal 0->1 in transit
        runtime.tick(&[0, 0, 0]);
        assert!(!runtime.engine.previous_spikes[1]);

        // Tick 3: Signal 0->1 arrives (delay 2 means t+2)
        runtime.tick(&[0, 0, 0]);
        assert!(runtime.engine.previous_spikes[1], "Neuron 1 should fire at T=3");
        assert!(!runtime.engine.previous_spikes[2]);

        // Tick 4, 5: Signal 0->2 in transit
        runtime.tick(&[0, 0, 0]);
        runtime.tick(&[0, 0, 0]);
        assert!(!runtime.engine.previous_spikes[2]);

        // Tick 6: Signal 0->2 arrives (delay 5 means t+5)
        runtime.tick(&[0, 0, 0]);
        assert!(runtime.engine.previous_spikes[2], "Neuron 2 should fire at T=6");
    }
}
