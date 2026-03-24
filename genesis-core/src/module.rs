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

    /// Handles external input directly without serialization overhead.
    fn handle_input(&mut self, _input: &ModuleInput) {}

    /// Called once when the module is added to the network or during model bootstrap.
    fn on_init(&mut self, _neurons: &mut NeuronsSoA) -> Result<(), ModuleError> { Ok(()) }

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
        self.register_factory("titan", || Box::new(crate::titan::TitanMemory::new(64, 100)));
        #[cfg(feature = "text")]
        self.register_factory("text_processor", || Box::new(crate::text::TextProcessorModule::new(64)));
        #[cfg(feature = "vision")]
        self.register_factory("vision", || Box::new(crate::vision::VisionModule::new(32, 32)));
        #[cfg(feature = "fusion")]
        self.register_factory("fusion", || Box::new(crate::fusion::SpikingFusionModule::new(Vec::new())));
        self.register_factory("graph_engine", || Box::new(crate::graph::SpikingGraphModule::new(crate::graph::TopologyType::SmallWorld)));
        self.register_factory("adaptive_lr", || Box::new(crate::plasticity::AdaptiveLearningRateModule::new(10)));
        self.register_factory("think", || Box::new(crate::ThinkModule::new(5)));
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

    /// Executes all modules, respecting their tier-based execution order.
    /// Modules within the same tier are executed in parallel using Rayon.
    /// Tier 0 is typically for raw input modules, while higher tiers are for fusion and reasoning.
    pub fn on_tick(&mut self, bus: &InputBus, previous_spikes: &[bool], tick: u32) {
        use rayon::prelude::*;

        for tier_indices in &self.tiered_indices {
            // Parallel execution within the tier.
            // We use par_iter() on indices and then access modules.
            // Since tiered_indices ensures each module belongs to exactly one tier
            // and we execute tiers sequentially, this is safe.
            self.modules.par_iter_mut().enumerate()
                .filter(|(idx, _)| tier_indices.contains(idx))
                .for_each(|(_, m)| {
                    m.on_tick(bus, previous_spikes, tick);
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
