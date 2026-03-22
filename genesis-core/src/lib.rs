#[cfg(feature = "titan")]
pub mod titan;
#[cfg(feature = "text")]
pub mod text;
#[cfg(feature = "vision")]
pub mod vision;
#[cfg(feature = "audio")]
pub mod audio;
#[cfg(feature = "rl")]
pub mod rl;
#[cfg(feature = "fusion")]
pub mod fusion;
pub mod graph;
pub mod robotics;

pub mod plasticity;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ThinkModule {
    pub extra_ticks: usize,
    pub active: bool,
}

impl ThinkModule {
    pub fn new(ticks: usize) -> Self {
        Self { extra_ticks: ticks, active: true }
    }
}

impl NanoModule for ThinkModule {
    fn name(&self) -> &str { "think" }
    fn on_tick(&mut self, _neurons: &mut NeuronsSoA, _previous_spikes: &[bool], _tick: u32) {
        // Core logic: The Runtime will check for 'think' module and perform extra backend calls
        // This module acts as a state carrier for that behavior
    }
    fn on_update_weights(&mut self, _neurons: &mut NeuronsSoA, _previous_spikes: &[bool], _current_spikes: &[bool], _tick: u32, _reward: Option<IValue>) {}
    fn on_night_phase(&mut self, _neurons: &mut NeuronsSoA, _synapses: &mut SynapsesSoA, _reward: Option<IValue>) {}
    fn box_clone(&self) -> Box<dyn NanoModule> { Box::new(self.clone()) }
    fn get_state(&self) -> Vec<u8> { bincode::serialize(self).unwrap_or_default() }
    fn set_state(&mut self, state: &[u8]) {
        if let Ok(new_self) = bincode::deserialize::<Self>(state) { *self = new_self; }
    }
}

pub trait NanoModule: Send + Sync {
    fn name(&self) -> &str;
    fn on_tick(&mut self, neurons: &mut NeuronsSoA, previous_spikes: &[bool], tick: u32);
    fn on_update_weights(&mut self, neurons: &mut NeuronsSoA, previous_spikes: &[bool], current_spikes: &[bool], tick: u32, reward: Option<IValue>);
    fn on_night_phase(&mut self, neurons: &mut NeuronsSoA, synapses: &mut SynapsesSoA, reward: Option<IValue>);

    // Serialization for persistence
    fn get_state(&self) -> Vec<u8> { Vec::new() }
    fn set_state(&mut self, _state: &[u8]) {}

    // Factory registration
    fn box_clone(&self) -> Box<dyn NanoModule>;
}

impl Clone for Box<dyn NanoModule> {
    fn clone(&self) -> Box<dyn NanoModule> {
        self.box_clone()
    }
}

pub struct ModuleManager {
    pub modules: Vec<Box<dyn NanoModule>>,
    pub factories: HashMap<String, Box<dyn Fn() -> Box<dyn NanoModule> + Send + Sync>>,
}

impl Default for ModuleManager {
    fn default() -> Self {
        let mut mm = Self { modules: Vec::new(), factories: HashMap::new() };
        mm.register_defaults();
        mm
    }
}

impl ModuleManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_defaults(&mut self) {
        #[cfg(feature = "titan")]
        self.register_factory("titan", || Box::new(titan::TitanMemory::new(64, 100)));
        #[cfg(feature = "text")]
        self.register_factory("text_processor", || Box::new(text::TextProcessorModule::new(64)));
        #[cfg(feature = "vision")]
        self.register_factory("vision", || Box::new(vision::VisionModule::new(32, 32)));
        #[cfg(feature = "fusion")]
        self.register_factory("fusion", || Box::new(fusion::SpikingFusionModule::new(Vec::new())));
        self.register_factory("graph_engine", || Box::new(graph::SpikingGraphModule::new(graph::TopologyType::SmallWorld)));
        self.register_factory("adaptive_lr", || Box::new(plasticity::AdaptiveLearningRateModule::new(10)));
        self.register_factory("think", || Box::new(ThinkModule::new(5)));
        self.register_factory("cerebellum", || Box::new(robotics::SpikingCerebellumModule::new(Vec::new(), Vec::new())));
    }

    pub fn register_factory<F>(&mut self, name: &str, factory: F)
    where F: Fn() -> Box<dyn NanoModule> + Send + Sync + 'static {
        self.factories.insert(name.to_string(), Box::new(factory));
    }

    pub fn add_module(&mut self, module: Box<dyn NanoModule>) {
        self.modules.push(module);
    }

    pub fn instantiate(&mut self, name: &str) -> bool {
        if let Some(factory) = self.factories.get(name) {
            self.modules.push(factory());
            true
        } else {
            false
        }
    }

    pub fn on_tick(&mut self, neurons: &mut NeuronsSoA, previous_spikes: &[bool], tick: u32) {
        for module in &mut self.modules {
            module.on_tick(neurons, previous_spikes, tick);
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

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::collections::HashMap;

pub type IValue = i32;
pub const SCALE: IValue = 1024; // 2^10 for bit-shift optimizations

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[repr(u8)]
pub enum Compartment {
    Proximal = 0,
    Distal = 1,
    Apical = 2,
    Basal = 3,
}

impl Default for Compartment {
    fn default() -> Self {
        Compartment::Proximal
    }
}

/// Context passed to plasticity rules to improve flexibility and reduce argument count.
pub struct PlasticityContext<'a> {
    pub pre_spiked: bool,
    pub post_spiked: bool,
    pub compartment: Compartment,
    pub reward: Option<IValue>,
    pub pre_last_spike: u32,
    pub post_last_spike: u32,
    pub current_tick: u32,
    pub neurons: &'a NeuronsSoA,
}

/// Trait for weight update rules (e.g., GSOP, STDP)
pub trait PlasticityRule {
    fn apply(&self, weight: &mut IValue, ctx: &PlasticityContext);

    fn update_contrastive(&self, weight: &mut IValue, layer_correlation: IValue) {
        if layer_correlation > 512 {
             *weight = (*weight as i64 * (1024 - (layer_correlation / 10)) as i64 >> 10) as i32;
        }
    }
}

pub struct GsopRule {
    pub learning_rate: IValue,
}

impl PlasticityRule for GsopRule {
    fn apply(&self, weight: &mut IValue, ctx: &PlasticityContext) {
        let lr = match ctx.compartment {
            Compartment::Proximal => self.learning_rate,
            Compartment::Distal => self.learning_rate * 8 / 10,
            _ => self.learning_rate / 2,
        };

        // If reward is negative, we can invert the learning or inhibit it
        let reward_mod = if let Some(r) = ctx.reward { if r < 0 { -1 } else { 1 } } else { 1 };
        let lr_mod = lr * reward_mod;

        if ctx.pre_spiked && ctx.post_spiked {
            *weight = weight.saturating_add(lr_mod);
        } else if ctx.pre_spiked && !ctx.post_spiked {
            *weight = weight.saturating_sub(lr_mod / 2);
        }
        if *weight > SCALE * 5 { *weight = SCALE * 5; }
        if *weight < -SCALE * 5 { *weight = -SCALE * 5; }
    }
}

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct NeuronsSoA {
    pub layer_id: Vec<u16>,
    pub potential: Vec<IValue>,
    pub distal_potential: Vec<IValue>, // For distal dendrites (coincidence detection)
    pub proximal_potential: Vec<IValue>, // For somatic inputs
    pub apical_potential: Vec<IValue>,  // For hierarchical feedback
    pub basal_potential: Vec<IValue>,   // For lateral signals
    pub backprop_signal: Vec<IValue>, // Signal from soma to dendrites (SMBP)
    pub threshold: Vec<IValue>,
    pub base_threshold: Vec<IValue>, // Intrinsic Plasticity
    pub decay: Vec<IValue>,
    pub liquid_current: Vec<IValue>, // For LLIF (Liquid Neurons)
    pub dendritic_gate: Vec<IValue>, // SCALE = 1.0 (open), 0 = closed
    pub refractory_timer: Vec<i32>,
    pub last_spike_tick: Vec<u32>,
    pub update_interval: Vec<u32>, // Sub-tick precision: 1 = every tick, 10 = every 10 ticks
    pub next_update_tick: Vec<u32>,
    pub x: Vec<i16>,
    pub y: Vec<i16>,
    pub gate_threshold: Vec<IValue>,
}

impl NeuronsSoA {
    pub fn new(size: usize) -> Self {
        Self {
            layer_id: vec![0; size],
            potential: vec![0; size],
            distal_potential: vec![0; size],
            proximal_potential: vec![0; size],
            apical_potential: vec![0; size],
            basal_potential: vec![0; size],
            backprop_signal: vec![0; size],
            threshold: vec![SCALE; size],
            base_threshold: vec![SCALE; size],
            decay: vec![50; size],
            liquid_current: vec![0; size],
            dendritic_gate: vec![SCALE; size],
            refractory_timer: vec![0; size],
            last_spike_tick: vec![0; size],
            update_interval: vec![1; size],
            next_update_tick: vec![0; size],
            x: vec![0; size],
            y: vec![0; size],
            gate_threshold: vec![512; size],
        }
    }
    pub fn len(&self) -> usize {
        self.potential.len()
    }

    pub fn grow(&mut self, additional: usize) {
        let new_size = self.potential.len() + additional;
        self.layer_id.resize(new_size, 0);
        self.potential.resize(new_size, 0);
        self.distal_potential.resize(new_size, 0);
        self.proximal_potential.resize(new_size, 0);
        self.apical_potential.resize(new_size, 0);
        self.basal_potential.resize(new_size, 0);
        self.backprop_signal.resize(new_size, 0);
        self.threshold.resize(new_size, SCALE);
        self.base_threshold.resize(new_size, SCALE);
        self.decay.resize(new_size, 50);
        self.liquid_current.resize(new_size, 0);
        self.dendritic_gate.resize(new_size, SCALE);
        self.refractory_timer.resize(new_size, 0);
        self.last_spike_tick.resize(new_size, 0);
        self.update_interval.resize(new_size, 1);
        self.next_update_tick.resize(new_size, 0);
        self.x.resize(new_size, 0);
        self.y.resize(new_size, 0);
        self.gate_threshold.resize(new_size, 512);
    }
}

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct SynapsesSoA {
    pub source_index: Vec<u32>,
    pub target_index: Vec<u32>,
    pub weight: Vec<IValue>,
    pub compartment: Vec<Compartment>,
    pub latent_matrix: Option<LatentSynapseMatrix>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct LatentSynapseMatrix {
    pub u: Vec<IValue>, // Low-rank U matrix
    pub v: Vec<IValue>, // Low-rank V matrix
    pub rank: usize,
}

impl SynapsesSoA {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            source_index: Vec::with_capacity(capacity),
            target_index: Vec::with_capacity(capacity),
            weight: Vec::with_capacity(capacity),
            compartment: Vec::with_capacity(capacity),
            latent_matrix: None,
        }
    }

    pub fn push(&mut self, source: u32, target: u32, weight: IValue) {
        self.push_to_compartment(source, target, weight, Compartment::Proximal);
    }

    pub fn push_to_compartment(&mut self, source: u32, target: u32, weight: IValue, compartment: Compartment) {
        self.source_index.push(source);
        self.target_index.push(target);
        self.weight.push(weight);
        self.compartment.push(compartment);
    }

    pub fn set_latent(&mut self, u: Vec<IValue>, v: Vec<IValue>, rank: usize) {
        self.latent_matrix = Some(LatentSynapseMatrix { u, v, rank });
    }

    pub fn len(&self) -> usize {
        self.source_index.len()
    }

    pub fn remove(&mut self, index: usize) {
        self.source_index.swap_remove(index);
        self.target_index.swap_remove(index);
        self.weight.swap_remove(index);
        self.compartment.swap_remove(index);
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct NetworkConfig {
    pub default_threshold: IValue,
    pub default_decay: IValue,
    pub learning_rate: IValue,

    // Advanced Structural Plasticity
    pub max_synapses: usize,
    pub neurogenesis_reward_threshold: IValue,
    pub pruning_threshold: IValue,

    // Dendritic Gating
    pub dendritic_coincidence_threshold: IValue,

    // Intrinsic Plasticity (Core Memory / Adaptive Thresholds)
    pub ip_increment: IValue,
    pub ip_decay: IValue,

    // STDP Parameters
    pub stdp_tau: u64,
    pub stdp_a_plus: IValue,
    pub stdp_a_minus: IValue,

    // Metaplasticity
    pub metaplasticity_enabled: bool,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            default_threshold: SCALE,
            default_decay: 50,
            learning_rate: 10,
            max_synapses: 1_000_000,
            neurogenesis_reward_threshold: 200,
            pruning_threshold: 10,
            dendritic_coincidence_threshold: 512, // 0.5 * SCALE
            ip_increment: 50,
            ip_decay: 1,
            stdp_tau: 20,
            stdp_a_plus: 100,
            stdp_a_minus: 100,
            metaplasticity_enabled: true,
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct BakedModel {
    pub version: String,
    pub config: NetworkConfig,
    pub node_id: u32,
    pub local_range: (usize, usize), // (start, end) indices of local neurons
    pub neurons: NeuronsSoA,
    pub synapses: SynapsesSoA,

    // Dynamic Module State Storage
    pub module_states: HashMap<String, Vec<u8>>,

    #[cfg(feature = "titan")]
    pub titan_memory: Option<titan::TitanMemory>,
    #[cfg(feature = "text")]
    pub has_text: bool,
    #[cfg(feature = "vision")]
    pub has_vision: bool,
    #[cfg(feature = "audio")]
    pub has_audio: bool,
    #[cfg(feature = "robotics")]
    pub has_robotics: bool,
    #[cfg(feature = "fusion")]
    pub has_fusion: bool,
    #[cfg(feature = "text")]
    pub vocabulary: HashMap<String, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neurons_init_and_grow() {
        let mut neurons = NeuronsSoA::new(10);
        assert_eq!(neurons.len(), 10);
        assert_eq!(neurons.dendritic_gate[0], SCALE);
        assert_eq!(neurons.layer_id[0], 0);

        neurons.grow(5);
        assert_eq!(neurons.len(), 15);
        assert_eq!(neurons.dendritic_gate[14], SCALE);
        assert_eq!(neurons.potential[14], 0);
    }
}

impl BakedModel {
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, self).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let file = File::open(path).map_err(|e| {
            log::error!("Failed to open model file at '{}': {}", path, e);
            e
        })?;
        let reader = BufReader::new(file);
        let model: BakedModel = bincode::deserialize_from(reader).map_err(|e| {
            log::error!("Error deserializing model from '{}': {:?}", path, e);
            std::io::Error::new(std::io::ErrorKind::Other, e)
        })?;

        if model.version != "4.0" {
             log::warn!("Loading model version {} into v4.0 engine. Physics scaling (1024) may differ from older versions.", model.version);
        }

        Ok(model)
    }
}
