use genesis_core::{BakedModel, NeuronsSoA, SynapsesSoA, NetworkConfig, Compartment};
use genesis_node::{Runtime, SimulationSettings, Observer, Telemetry, SimulationEngine};
use genesis_compute::{CpuBackend, ComputeBackend};
use std::collections::HashMap;

#[test]
fn test_module_persistence_and_restoration() {
    let model = BakedModel {
        version: "4.0".to_string(),
        config: NetworkConfig::default(),
        node_id: 0,
        local_range: (0, 10),
        neurons: NeuronsSoA::new(10),
        synapses: SynapsesSoA::default(),
        module_states: HashMap::new(),
        #[cfg(feature = "titan")]
        titan_memory: Some(genesis_core::titan::BitWiseTitan::new(500)),
        has_text: false,
        has_vision: false,
        has_audio: false,
        has_robotics: false,
        has_fusion: false,
        vocabulary: HashMap::new(),
    };

    let mut runtime = Runtime {
        engine: SimulationEngine::new(
            model,
            {
                let mut mm = genesis_core::ModuleManager::new();
                #[cfg(feature = "titan")]
                mm.register_factory("titan", || Box::new(genesis_core::titan::BitWiseTitan::new(500)));
                mm
            },
            Box::new(CpuBackend::default()),
            &SimulationSettings::default()
        ),
        settings: SimulationSettings {
            night_phase_interval: 1, // Learning every tick for test
            ..SimulationSettings::default()
        },
        episode_reward_history: Vec::new(),
        network_manager: None,
        observers: vec![Box::new(Observer::new(10)), Box::new(Telemetry::default())],
        last_surprise: 0,
        surprise_history: Vec::new(),
    };

    // 1. Manually instantiate titan (since it wasn't in module_states initially)
    runtime.engine.modules.instantiate("titan");

    // 2. Perform some learning (Day/Night cycle)
    let input = [1024; 10];
    runtime.tick_with_reward(&input, Some(1000)); // High positive reward

    // 3. Force state synchronization
    for module in &runtime.engine.modules.modules {
        runtime.engine.model.module_states.insert(module.name().to_string(), module.get_state());
    }

    let saved_state = runtime.engine.model.module_states.get("titan").expect("Titan state should be saved");
    let saved_data = saved_state.clone();

    // 4. Create new runtime and restore
    let model2 = runtime.engine.model.clone();
    let mut runtime2 = Runtime {
        engine: SimulationEngine::new(
            model2,
            {
                let mut mm = genesis_core::ModuleManager::new();
                #[cfg(feature = "titan")]
                mm.register_factory("titan", || Box::new(genesis_core::titan::BitWiseTitan::new(500)));
                mm
            },
            Box::new(CpuBackend::default()),
            &SimulationSettings::default()
        ),
        settings: SimulationSettings::default(),
        episode_reward_history: Vec::new(),
        network_manager: None,
        observers: vec![Box::new(Observer::new(10)), Box::new(Telemetry::default())],
        last_surprise: 0,
        surprise_history: Vec::new(),
    };

    // Restoration logic (usually in load_with_settings, testing manually here)
    for (name, state) in &runtime2.engine.model.module_states {
        if runtime2.engine.modules.instantiate(name) {
            runtime2.engine.modules.modules.last_mut().unwrap().set_state(state);
        }
    }

    let restored_state = runtime2.engine.modules.modules[0].get_state();
    assert_eq!(saved_data, restored_state);
}

#[test]
fn test_multi_compartment_gating_physics() {
    let mut model = BakedModel {
        version: "4.0".to_string(),
        config: NetworkConfig {
            dendritic_coincidence_threshold: 512, // 0.5 * SCALE
            ..NetworkConfig::default()
        },
        node_id: 0,
        local_range: (0, 3),
        neurons: NeuronsSoA::new(3),
        synapses: {
            let mut s = SynapsesSoA::with_capacity(2);
            s.push_to_compartment(0, 2, 2000, 1, Compartment::Proximal);
            s.push_to_compartment(1, 2, 2000, 1, Compartment::Distal);
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

    let mut backend = CpuBackend::default();

    // Case 1: ONLY Distal input (Gated/Attenuated)
    // Neuron 1 spikes -> Distal input to Neuron 2
    model.neurons.proximal_potential[2] = 0;
    model.neurons.distal_potential[2] = 0;
    let mut prev_spikes = vec![false, true, false];
    let spike_data = backend.day_phase(&mut model, &[0, 0, 0], &prev_spikes, &[], 1, Default::default());
    let mut spikes = vec![false; 3];
    if let genesis_core::SpikeData::Sparse(indices) = spike_data {
        for i in indices { if i < 3 { spikes[i] = true; } }
    }
    // Should NOT spike because distal is attenuated (2000 / 16 = 125 < 1024 threshold)
    assert!(!spikes[2], "Neuron 2 should not spike with only distal input");

    // Case 2: ONLY Proximal input (Direct)
    // Neuron 0 spikes -> Proximal input to Neuron 2
    model.neurons.potential[2] = 0;
    model.neurons.proximal_potential[2] = 0;
    model.neurons.distal_potential[2] = 0;
    model.neurons.refractory_timer[2] = 0;
    prev_spikes = vec![true, false, false];
    let spike_data = backend.day_phase(&mut model, &[0, 0, 0], &prev_spikes, &[], 2, Default::default());
    let mut spikes = vec![false; 3];
    if let genesis_core::SpikeData::Sparse(indices) = spike_data {
        for i in indices { if i < 3 { spikes[i] = true; } }
    }
    // Should spike because proximal is direct (2000 > 1024 threshold)
    assert!(spikes[2], "Neuron 2 should spike with strong proximal input");

    // Case 3: Coincidence (Proximal opens Distal)
    model.neurons.potential[2] = 0;
    model.neurons.proximal_potential[2] = 0;
    model.neurons.distal_potential[2] = 0;
    model.neurons.refractory_timer[2] = 0;
    // We need some proximal potential to reach threshold 512.
    // Let's inject external input to proximal.
    prev_spikes = vec![false, true, false]; // Distal only via synapse
    let spike_data = backend.day_phase(&mut model, &[0, 0, 600], &prev_spikes, &[], 3, Default::default());
    let mut spikes = vec![false; 3];
    if let genesis_core::SpikeData::Sparse(indices) = spike_data {
        for i in indices { if i < 3 { spikes[i] = true; } }
    }
    // Proximal 600 > 512 threshold -> Distal 2000 fully integrated.
    // 600 + 2000 = 2600 > 1024 threshold.
    assert!(spikes[2], "Neuron 2 should spike with coincident proximal and distal input");
}

#[test]
fn test_evolutionary_structural_growth() {
    let model = BakedModel {
        version: "4.0".to_string(),
        config: NetworkConfig::default(),
        node_id: 0,
        local_range: (0, 100),
        neurons: NeuronsSoA::new(100),
        synapses: SynapsesSoA::with_capacity(10),
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
        engine: SimulationEngine::new(
            model,
            genesis_core::ModuleManager::new(),
            Box::new(CpuBackend::default()),
            &SimulationSettings::default()
        ),
        settings: SimulationSettings {
            night_phase_interval: 10,
            ..SimulationSettings::default()
        },
        episode_reward_history: Vec::new(),
        network_manager: None,
        observers: vec![Box::new(Observer::new(100)), Box::new(Telemetry::default())],
        last_surprise: 0,
        surprise_history: Vec::new(),
    };

    // Simulate high reward and some activity to trigger growth
    for _ in 0..10 {
        let mut inputs = vec![0; 100];
        inputs[0] = 5000;
        inputs[1] = 5000;
        // Targeted reward to ensure growth logic triggers correctly
        runtime.tick_with_reward_targeted(&inputs, Some(500), None);
    }

    // After 10 ticks (interval=10), structural plasticity should have run
    assert!(runtime.engine.model.synapses.len() > 0, "Structural plasticity should have grown new synapses");
}
