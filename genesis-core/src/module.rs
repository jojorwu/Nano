use crate::{IValue, bus::InputBus, model::{NeuronsSoA, SynapsesSoA}};
use serde::{Serialize, Deserialize};
use thiserror::Error;
use std::collections::HashMap;

#[derive(Error, Debug)]
pub enum ModuleError {
    #[error("Initialization failed: {0}")]
    InitFailed(String),
    #[error("Validation failed: {0}")]
    ValidationFailed(String),
    #[error("State serialization error: {0}")]
    SerializationError(#[from] bincode::Error),
    #[error("Input handling error: {0}")]
    InputError(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModuleInput {
    Text(String),
    Image(Vec<u8>),
    Audio(Vec<f32>),
    Control(String, IValue),
}

/// Defines the behavior and lifecycle of a functional unit (Module) in the SNN.
///
/// Modules can handle external inputs, inject signals into the `InputBus`,
/// and manage their own internal learning and structural plasticity rules.
pub trait NanoModule: Send + Sync {
    /// Unique identifier for the module type.
    fn name(&self) -> &str;

    /// Execution priority: lower tiers run first.
    fn tier(&self) -> u32 { 0 }

    /// Declares which channels this module writes to.
    fn outputs(&self) -> Vec<String> { Vec::new() }

    /// Declares which channels this module reads from.
    fn inputs(&self) -> Vec<String> { Vec::new() }

    /// Optional downcast to concrete type
    fn as_any(&self) -> &dyn std::any::Any { &() }

    /// Declares global signal modulations this module wants to apply.
    /// Returns a list of (signal_id, delta_value).
    fn get_global_modulations(&self) -> Vec<(usize, i32)> { Vec::new() }

    /// Handles external input directly without serialization overhead.
    fn handle_input(&mut self, _input: &ModuleInput) {}

    /// Called once when the module is added to the network or during model bootstrap.
    fn on_init(&mut self, _neurons: &mut NeuronsSoA) -> Result<(), ModuleError> { Ok(()) }

    /// Called to provide global network configuration to the module.
    fn on_config_sync(&mut self, _config: &crate::config::NetworkConfig) {}

    /// Called every simulation tick. Use this to inject external signals into the InputBus.
    /// The InputBus uses atomic integers to allow thread-safe signal injection from multiple modules.
    fn on_tick(&mut self, bus: &InputBus, previous_spikes: &[bool], tick: u32);

    /// Called during the learning phase to update module-specific internal weights or states.
    fn on_update_weights(&mut self, neurons: &mut NeuronsSoA, previous_spikes: &[bool], current_spikes: &[bool], tick: u32, reward: Option<IValue>);

    /// Called during the structural plasticity phase (Night Phase). Use this for periodic maintenance or consolidation.
    fn on_night_phase(&mut self, neurons: &mut NeuronsSoA, synapses: &mut SynapsesSoA, reward: Option<IValue>);

    // Serialization for persistence
    fn get_state(&self) -> Vec<u8> { Vec::new() }
    fn set_state(&mut self, _state: &[u8]) {}

    /// Validates the module's internal state against the current network topology.
    /// This is called during initialization to ensure all neuron indices and configurations are valid.
    fn validate_state(&self, _neurons: &NeuronsSoA) -> Result<(), ModuleError> { Ok(()) }

    // Factory registration
    fn box_clone(&self) -> Box<dyn NanoModule>;
}

/// FFI callback type for external modules (C/C++/Python bridge).
pub type ForeignTickFn = unsafe extern "C" fn(bus_ptr: *mut i32, bus_size: usize, tick: u32);

/// A module that executes logic in a foreign language (C++, CUDA, or Python callback).
#[derive(Clone)]
pub struct ForeignModule {
    pub name: String,
    pub tick_fn: Option<ForeignTickFn>,
}

impl NanoModule for ForeignModule {
    fn name(&self) -> &str { &self.name }
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn on_tick(&mut self, bus: &InputBus, _previous_spikes: &[bool], tick: u32) {
        if let Some(f) = self.tick_fn {
            // Simplified bus pointer pass for Zero-Copy FFI.
            let bus_ptr = bus.proximal().as_ptr() as *mut i32;
            unsafe { f(bus_ptr, bus.size, tick); }
        }
    }
    fn on_update_weights(&mut self, _: &mut NeuronsSoA, _: &[bool], _: &[bool], _: u32, _: Option<IValue>) {}
    fn on_night_phase(&mut self, _: &mut NeuronsSoA, _: &mut SynapsesSoA, _: Option<IValue>) {}
    fn box_clone(&self) -> Box<dyn NanoModule> { Box::new(self.clone()) }
}

/// Dynamic Plugin System for loading external shared libraries (.so, .dll)
pub struct DynamicPluginModule {
    pub name: String,
    _lib: std::sync::Arc<libloading::Library>,
    tick_fn: ForeignTickFn,
}

impl DynamicPluginModule {
    pub fn load(path: &str, symbol: &str) -> Result<Self, String> {
        unsafe {
            let lib = libloading::Library::new(path).map_err(|e| e.to_string())?;
            let tick_fn: ForeignTickFn = *lib.get(symbol.as_bytes()).map_err(|e| e.to_string())?;

            Ok(Self {
                name: format!("plugin:{}", symbol),
                _lib: std::sync::Arc::new(lib),
                tick_fn,
            })
        }
    }
}

impl NanoModule for DynamicPluginModule {
    fn name(&self) -> &str { &self.name }
    fn box_clone(&self) -> Box<dyn NanoModule> {
        Box::new(Self {
            name: self.name.clone(),
            _lib: self._lib.clone(),
            tick_fn: self.tick_fn,
        })
    }
    fn on_tick(&mut self, bus: &InputBus, _previous_spikes: &[bool], tick: u32) {
        let bus_ptr = bus.proximal().as_ptr() as *mut i32;
        unsafe { (self.tick_fn)(bus_ptr, bus.size, tick); }
    }
    fn on_update_weights(&mut self, _: &mut NeuronsSoA, _: &[bool], _: &[bool], _: u32, _: Option<IValue>) {}
    fn on_night_phase(&mut self, _: &mut NeuronsSoA, _: &mut SynapsesSoA, _: Option<IValue>) {}
}

impl Clone for Box<dyn NanoModule> {
    fn clone(&self) -> Box<dyn NanoModule> {
        self.box_clone()
    }
}

pub struct ModuleRegistry {
    pub factories: HashMap<String, Box<dyn Fn() -> Box<dyn NanoModule> + Send + Sync>>,
}

impl ModuleRegistry {
    pub fn new() -> Self {
        let mut registry = Self { factories: HashMap::new() };
        registry.register_defaults();
        registry
    }

    pub fn register_defaults(&mut self) {
        #[cfg(feature = "titan")]
        self.register_factory("titan", || Box::new(crate::titan::BitWiseTitan::new(100)));
        #[cfg(feature = "text")]
        self.register_factory("text_processor", || Box::new(crate::text::TextProcessorModule::new(64)));
        #[cfg(feature = "vision")]
        self.register_factory("vision", || Box::new(crate::vision::VisionModule::new(32, 32)));
        #[cfg(feature = "fusion")]
        self.register_factory("fusion", || Box::new(crate::fusion::SpikingFusionModule::new(Vec::new())));
        self.register_factory("graph_engine", || Box::new(crate::graph::SpikingGraphModule::new(crate::graph::TopologyType::SmallWorld)));
        self.register_factory("adaptive_lr", || Box::new(crate::plasticity::AdaptiveLearningRateModule::new(10)));
        self.register_factory("think", || Box::new(crate::ThinkModule::new(5)));
        self.register_factory("workspace", || Box::new(crate::workspace::WorkspaceModule::new()));
        self.register_factory("hierarchical", || Box::new(crate::hierarchical::HierarchicalModule::new()));
        self.register_factory("episodic", || Box::new(crate::episodic::EpisodicModule::new()));
        self.register_factory("cerebellum", || Box::new(crate::robotics::SpikingCerebellumModule::new(Vec::new(), Vec::new())));
    }

    pub fn register_factory<F>(&mut self, name: &str, factory: F)
    where F: Fn() -> Box<dyn NanoModule> + Send + Sync + 'static {
        self.factories.insert(name.to_string(), Box::new(factory));
    }

    pub fn create(&self, name: &str) -> Option<Box<dyn NanoModule>> {
        self.factories.get(name).map(|f| f())
    }
}

pub struct ModuleManager {
    pub modules: Vec<Box<dyn NanoModule>>,
    pub registry: ModuleRegistry,
    pub tiered_indices: Vec<Vec<usize>>,
}

impl Default for ModuleManager {
    fn default() -> Self {
        let mut registry = ModuleRegistry::new();
        registry.register_defaults();
        Self {
            modules: Vec::new(),
            registry,
            tiered_indices: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockModule {
        name: String,
        inputs: Vec<String>,
        outputs: Vec<String>,
    }

    impl NanoModule for MockModule {
        fn name(&self) -> &str { &self.name }
        fn inputs(&self) -> Vec<String> { self.inputs.clone() }
        fn outputs(&self) -> Vec<String> { self.outputs.clone() }
        fn on_tick(&mut self, _: &InputBus, _: &[bool], _: u32) {}
        fn on_update_weights(&mut self, _: &mut NeuronsSoA, _: &[bool], _: &[bool], _: u32, _: Option<IValue>) {}
        fn on_night_phase(&mut self, _: &mut NeuronsSoA, _: &mut SynapsesSoA, _: Option<IValue>) {}
        fn box_clone(&self) -> Box<dyn NanoModule> {
            Box::new(Self { name: self.name.clone(), inputs: self.inputs.clone(), outputs: self.outputs.clone() })
        }
    }

    #[test]
    fn test_module_dependency_sorting() {
        let mut manager = ModuleManager::new();
        manager.add_module(Box::new(MockModule {
            name: "fusion".into(),
            inputs: vec!["modality:text".into()],
            outputs: vec!["fused".into()],
        }));
        manager.add_module(Box::new(MockModule {
            name: "text".into(),
            inputs: vec![],
            outputs: vec!["modality:text".into()],
        }));

        manager.rebuild_tiers();

        // "text" must be in an earlier tier than "fusion"
        let text_idx = manager.modules.iter().position(|m| m.name() == "text").unwrap();
        let fusion_idx = manager.modules.iter().position(|m| m.name() == "fusion").unwrap();

        let text_tier = manager.tiered_indices.iter().position(|t| t.contains(&text_idx)).unwrap();
        let fusion_tier = manager.tiered_indices.iter().position(|t| t.contains(&fusion_idx)).unwrap();

        assert!(text_tier < fusion_tier);
    }

    #[test]
    fn test_circular_dependency_fallback() {
        let mut manager = ModuleManager::new();
        manager.add_module(Box::new(MockModule {
            name: "A".into(),
            inputs: vec!["B_out".into()],
            outputs: vec!["A_out".into()],
        }));
        manager.add_module(Box::new(MockModule {
            name: "B".into(),
            inputs: vec!["A_out".into()],
            outputs: vec!["B_out".into()],
        }));

        // Should not panic, but fallback to manual tiers (usually 0)
        manager.rebuild_tiers();
        assert!(!manager.tiered_indices.is_empty());
    }
}

impl ModuleManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_factory<F>(&mut self, name: &str, factory: F)
    where F: Fn() -> Box<dyn NanoModule> + Send + Sync + 'static {
        self.registry.register_factory(name, factory);
    }

    pub fn add_module(&mut self, module: Box<dyn NanoModule>) {
        self.modules.push(module);
        self.rebuild_tiers();
    }

    pub fn instantiate(&mut self, name: &str) -> bool {
        if let Some(m) = self.registry.create(name) {
            self.modules.push(m);
            self.rebuild_tiers();
            true
        } else {
            false
        }
    }

    pub fn rebuild_tiers(&mut self) {
        // Dependency analysis based on inputs/outputs
        let n = self.modules.len();
        let mut adj = vec![Vec::new(); n];
        let mut in_degree = vec![0; n];

        for i in 0..n {
            let outputs = self.modules[i].outputs();
            for j in 0..n {
                if i == j { continue; }
                let inputs = self.modules[j].inputs();
                if outputs.iter().any(|o| inputs.contains(o)) {
                    adj[i].push(j);
                    in_degree[j] += 1;
                }
            }
        }

        // BFS for topological sort (Kahn's algorithm)
        let mut tiers = Vec::new();
        let mut current_tier = Vec::new();

        for i in 0..n {
            if in_degree[i] == 0 {
                current_tier.push(i);
            }
        }

        while !current_tier.is_empty() {
            let mut next_tier = Vec::new();
            let mut tier_indices = Vec::new();

            for &u in &current_tier {
                tier_indices.push(u);
                for &v in &adj[u] {
                    in_degree[v] -= 1;
                    if in_degree[v] == 0 {
                        next_tier.push(v);
                    }
                }
            }
            tiers.push(tier_indices);
            current_tier = next_tier;
        }

        // If not all modules are covered, there's a cycle.
        // Fallback to manual tiers if cycle detected or simple dependency is missing.
        if tiers.iter().map(|t| t.len()).sum::<usize>() < n {
             let mut manual_tiers: Vec<u32> = self.modules.iter().map(|m| m.tier()).collect();
             manual_tiers.sort_unstable();
             manual_tiers.dedup();
             self.tiered_indices = manual_tiers.into_iter().map(|t| {
                 self.modules.iter().enumerate()
                     .filter(|(_, m)| m.tier() == t)
                     .map(|(i, _)| i)
                     .collect()
             }).collect();
        } else {
            self.tiered_indices = tiers;
        }
    }

    pub fn on_init(&mut self, neurons: &mut NeuronsSoA) -> Result<(), ModuleError> {
        for module in &mut self.modules {
            module.on_init(neurons)?;
            module.validate_state(neurons)?;
        }
        Ok(())
    }

    pub fn on_config_sync(&mut self, config: &crate::config::NetworkConfig) {
        for module in &mut self.modules {
            module.on_config_sync(config);
        }
    }

    /// Executes all modules, respecting their tier-based execution order.
    /// Modules within the same tier are executed in parallel using Rayon.
    /// Tier 0 is typically for raw input modules, while higher tiers are for fusion and reasoning.
    pub fn on_tick(&mut self, bus: &InputBus, previous_spikes: &[bool], tick: u32) {
        use rayon::prelude::*;

        for tier_indices in &self.tiered_indices {
            self.modules.par_iter_mut().enumerate()
                .filter(|(idx, _)| tier_indices.contains(idx))
                .for_each(|(_, m)| {
                    m.on_tick(bus, previous_spikes, tick);

                    // Apply requested global signal modulations
                    for (sig_id, delta) in m.get_global_modulations() {
                        if sig_id < bus.global_signals.len() {
                            InputBus::atomic_saturating_add(&bus.global_signals[sig_id], delta);
                        }
                    }
                });
        }
    }

    pub fn on_update_weights(&mut self, neurons: &mut NeuronsSoA, previous_spikes: &[bool], current_spikes: &[bool], tick: u32, reward: Option<IValue>) {
        for module in &mut self.modules {
            module.on_update_weights(neurons, previous_spikes, current_spikes, tick, reward);
        }
    }

    pub fn on_night_phase(&mut self, neurons: &mut NeuronsSoA, synapses: &mut SynapsesSoA, reward: Option<IValue>) {
        for module in &mut self.modules {
            module.on_night_phase(neurons, synapses, reward);
        }
    }
}
