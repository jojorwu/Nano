use genesis_core::{BakedModel, NeuronsSoA, SynapsesSoA, NetworkConfig, Compartment};
use genesis_node::{Runtime, SimulationSettings, Observer, Telemetry};
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
        titan_memory: Some(genesis_core::titan::TitanMemory::new(10, 500)),
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
            let mut mm = genesis_core::ModuleManager::new();
            #[cfg(feature = "titan")]
            mm.register_factory("titan", || Box::new(genesis_core::titan::TitanMemory::new(10, 500)));
            mm
        },
        settings: SimulationSettings {
            night_phase_interval: 1, // Learning every tick for test
            ..SimulationSettings::default()
        },
        backend: Box::new(CpuBackend::default()),
        previous_spikes: vec![false; 10],
        tick_counter: 0,
        spikes_history: Vec::new(),
        rolling_spike_count: 0.0,
        network_manager: None,
        observer: Observer::new(10),
        remote_spike_queue: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
        telemetry: Telemetry::default(),
    };

    // 1. Manually instantiate titan (since it wasn't in module_states initially)
    runtime.modules.instantiate("titan");

    // 2. Perform some learning (Day/Night cycle)
    let input = [1024; 10];
    runtime.tick_with_reward(&input, Some(1000)); // High positive reward

    // 3. Force state synchronization
    for module in &runtime.modules.modules {
        runtime.model.module_states.insert(module.name().to_string(), module.get_state());
    }

    let saved_state = runtime.model.module_states.get("titan").expect("Titan state should be saved");
    let saved_data = saved_state.clone();

    // 4. Create new runtime and restore
    let model2 = runtime.model.clone();
    let mut runtime2 = Runtime {
        model: model2,
        modules: {
            let mut mm = genesis_core::ModuleManager::new();
            #[cfg(feature = "titan")]
            mm.register_factory("titan", || Box::new(genesis_core::titan::TitanMemory::new(10, 500)));
            mm
        },
        settings: SimulationSettings::default(),
        backend: Box::new(CpuBackend::default()),
        previous_spikes: vec![false; 10],
        tick_counter: 0,
        spikes_history: Vec::new(),
        rolling_spike_count: 0.0,
        network_manager: None,
        observer: Observer::new(10),
        remote_spike_queue: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
        telemetry: Telemetry::default(),
    };

    // Restoration logic (usually in load_with_settings, testing manually here)
    for (name, state) in &runtime2.model.module_states {
        if runtime2.modules.instantiate(name) {
            runtime2.modules.modules.last_mut().unwrap().set_state(state);
        }
    }

    let restored_state = runtime2.modules.modules[0].get_state();
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
            s.push_to_compartment(0, 2, 2000, Compartment::Proximal);
            s.push_to_compartment(1, 2, 2000, Compartment::Distal);
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
    let spikes = backend.day_phase(&mut model, &[0, 0, 0], &prev_spikes, 1);
    // Should NOT spike because distal is attenuated (2000 / 4 = 500 < 1024 threshold)
    assert!(!spikes[2], "Neuron 2 should not spike with only distal input");

    // Case 2: ONLY Proximal input (Direct)
    // Neuron 0 spikes -> Proximal input to Neuron 2
    model.neurons.potential[2] = 0;
    model.neurons.proximal_potential[2] = 0;
    model.neurons.distal_potential[2] = 0;
    model.neurons.refractory_timer[2] = 0;
    prev_spikes = vec![true, false, false];
    let spikes = backend.day_phase(&mut model, &[0, 0, 0], &prev_spikes, 2);
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
    let spikes = backend.day_phase(&mut model, &[0, 0, 600], &prev_spikes, 3);
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
        model,
        modules: genesis_core::ModuleManager::new(),
        settings: SimulationSettings {
            night_phase_interval: 10,
            ..SimulationSettings::default()
        },
        backend: Box::new(CpuBackend::default()),
        previous_spikes: vec![false; 100],
        tick_counter: 0,
        spikes_history: Vec::new(),
        rolling_spike_count: 0.0,
        network_manager: None,
        observer: Observer::new(100),
        remote_spike_queue: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
        telemetry: Telemetry::default(),
    };

    // Simulate high reward and some activity to trigger growth
    for _ in 0..10 {
        let mut inputs = vec![0; 100];
        inputs[0] = 5000;
        inputs[1] = 5000;
        runtime.tick_with_reward(&inputs, Some(500));
    }

    // After 10 ticks (interval=10), structural plasticity should have run
    assert!(runtime.model.synapses.len() > 0, "Structural plasticity should have grown new synapses");
}
