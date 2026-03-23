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

/// Thinking Mode Module: Enables "Chain of Thought" reasoning by performing
/// extra simulation sub-ticks for each external input tick.
/// This allows the network to iterate internally without new modality data.
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
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn handle_input(&mut self, input: &ModuleInput) {
        if let ModuleInput::Control(name, val) = input {
            if name == "active" { self.active = *val != 0; }
            if name == "ticks" { self.extra_ticks = *val as usize; }
        }
    }
    fn on_tick(&mut self, _bus: &InputBus, _previous_spikes: &[bool], _tick: u32) {
        // Core logic: The Runtime will check for 'think' module and perform extra backend calls
        // This module acts as a state carrier for that behavior
    }
    fn on_update_weights(&mut self, _neurons: &mut NeuronsSoA, _previous_spikes: &[bool], _current_spikes: &[bool], _tick: u32, surprise: Option<IValue>) {
        if let Some(s) = surprise {
            // Noradrenaline-like modulation: deep think more when surprised
            // Base extra_ticks is modified by novelty
            if s > 512 {
                self.extra_ticks = (self.extra_ticks + 1).min(20);
            } else if s < 100 {
                self.extra_ticks = self.extra_ticks.saturating_sub(1);
            }
        }
    }
    fn on_night_phase(&mut self, _neurons: &mut NeuronsSoA, _synapses: &mut SynapsesSoA, _reward: Option<IValue>) {}
    fn box_clone(&self) -> Box<dyn NanoModule> { Box::new(self.clone()) }
    fn get_state(&self) -> Vec<u8> { bincode::serialize(self).unwrap_or_default() }
    fn set_state(&mut self, state: &[u8]) {
        if let Ok(new_self) = bincode::deserialize::<Self>(state) { *self = new_self; }
    }
    fn validate_state(&self, _neurons: &NeuronsSoA) -> Result<(), String> { Ok(()) }
}

/// Represents a modular functional unit within the Spiking Neural Network.
/// Modules can inject signals, observe activity, and manage their own internal plasticity rules.
pub struct InputBus {
    pub proximal: Vec<AtomicI32>,
    pub distal: Vec<AtomicI32>,
    pub apical: Vec<AtomicI32>,
    pub basal: Vec<AtomicI32>,

    // Modality-specific buffers for high-order fusion
    pub vision: Vec<AtomicI32>,
    pub text: Vec<AtomicI32>,
    pub audio: Vec<AtomicI32>,
}

impl InputBus {
    pub fn new(size: usize) -> Self {
        Self {
            proximal: (0..size).map(|_| AtomicI32::new(0)).collect(),
            distal: (0..size).map(|_| AtomicI32::new(0)).collect(),
            apical: (0..size).map(|_| AtomicI32::new(0)).collect(),
            basal: (0..size).map(|_| AtomicI32::new(0)).collect(),
            vision: (0..size).map(|_| AtomicI32::new(0)).collect(),
            text: (0..size).map(|_| AtomicI32::new(0)).collect(),
            audio: (0..size).map(|_| AtomicI32::new(0)).collect(),
        }
    }

    pub fn clear(&self) {
        use rayon::prelude::*;
        let iterators = [
            &self.proximal, &self.distal, &self.apical, &self.basal,
            &self.vision, &self.text, &self.audio
        ];

        iterators.par_iter().for_each(|&vec| {
            vec.par_iter().for_each(|v| v.store(0, Ordering::Relaxed));
        });
    }

    /// Faster clear when unique access is available, using raw memory fill.
    pub fn clear_mut(&mut self) {
        fn clear_vec(v: &mut [AtomicI32]) {
            let ptr = v.as_mut_ptr() as *mut i32;
            let len = v.len();
            unsafe {
                std::ptr::write_bytes(ptr, 0, len);
            }
        }
        clear_vec(&mut self.proximal);
        clear_vec(&mut self.distal);
        clear_vec(&mut self.apical);
        clear_vec(&mut self.basal);
        clear_vec(&mut self.vision);
        clear_vec(&mut self.text);
        clear_vec(&mut self.audio);
    }

    pub fn atomic_saturating_add(target: &AtomicI32, val: i32) {
        let mut current = target.load(Ordering::Relaxed);
        loop {
            let next = current.saturating_add(val);
            match target.compare_exchange_weak(current, next, Ordering::SeqCst, Ordering::Relaxed) {
                Ok(_) => break,
                Err(updated) => current = updated,
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModuleInput {
    Text(String),
    Image(Vec<u8>),
    Audio(Vec<f32>),
    Control(String, IValue),
}

pub trait NanoModule: Send + Sync {
    /// Unique identifier for the module type.
    fn name(&self) -> &str;

    /// Execution priority: lower tiers run first.
    fn tier(&self) -> u32 { 0 }

    /// Optional downcast to concrete type
    fn as_any(&self) -> &dyn std::any::Any { &() }

    /// Handles external input directly without serialization overhead.
    fn handle_input(&mut self, _input: &ModuleInput) {}

    /// Called once when the module is added to the network or during model bootstrap.
    fn on_init(&mut self, _neurons: &mut NeuronsSoA) {}

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
    fn validate_state(&self, _neurons: &NeuronsSoA) -> Result<(), String> { Ok(()) }

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

    pub fn on_init(&mut self, neurons: &mut NeuronsSoA) -> Result<(), String> {
        for module in &mut self.modules {
            module.on_init(neurons);
            module.validate_state(neurons)?;
        }
        Ok(())
    }

    /// Executes all modules, respecting their tier-based execution order.
    /// Modules within the same tier are executed in parallel using Rayon.
    /// Tier 0 is typically for raw input modules, while higher tiers are for fusion and reasoning.
    pub fn on_tick(&mut self, bus: &InputBus, previous_spikes: &[bool], tick: u32) {
        use rayon::prelude::*;

        // Group modules by tier
        let mut tiers: Vec<u32> = self.modules.iter().map(|m| m.tier()).collect();
        tiers.sort_unstable();
        tiers.dedup();

        for tier in tiers {
            // Execute all modules in the current tier in parallel
            self.modules.par_iter_mut()
                .filter(|m| m.tier() == tier)
                .for_each(|m: &mut Box<dyn NanoModule>| {
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

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::collections::HashMap;
use std::sync::atomic::{AtomicI32, Ordering};

pub type IValue = i32;
pub const SCALE: IValue = 1024; // 2^10 for bit-shift optimizations

// --- Physics Constants ---
pub const DEFAULT_REFRACTORY_TICKS: i32 = 4;
pub const SMBP_DECAY: i64 = 800; // Multiplier out of SCALE
pub const ACTIVITY_EMA_ALPHA: i64 = 99; // Alpha out of 100
pub const TARGET_ACTIVITY_LEVEL: IValue = 100; // 10% target firing rate
pub const WEIGHT_CLAMP_LIMIT: IValue = SCALE * 5;

// --- Dendritic Gating Constants ---
pub const GATE_OPEN: IValue = SCALE;
pub const GATE_HALF: IValue = SCALE / 2;
pub const GATE_QUARTER: IValue = SCALE / 4;
pub const GATE_THREE_QUARTERS: IValue = SCALE * 3 / 4;

/// Represents the global chemical state of the network.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default)]
pub struct NeuromodulationState {
    pub dopamine: IValue,       // Reward / Prediction Error
    pub noradrenaline: IValue,  // Surprise / Novelty / Arousal
    pub serotonin: IValue,      // Stability / Risk Mitigation
}

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
    pub backprop_signal: IValue, // SMBP: signal from soma to dendrites
    pub compartment: Compartment,
    pub reward: Option<IValue>,
    pub neuromodulation: NeuromodulationState,
    pub pre_last_spike: u32,
    pub post_last_spike: u32,
    pub current_tick: u32,
    pub post_index: usize,
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

        // SMBP Modulation: active backpropagation signal amplifies learning in distal dendrites
        let smbp_mod = if ctx.compartment != Compartment::Proximal {
            (SCALE + ctx.backprop_signal) >> 10
        } else {
            1
        };

        // Neuromodulation: Noradrenaline amplifies learning (Surprise), Dopamine scales reward
        let neuromod_gain = (SCALE + ctx.neuromodulation.noradrenaline) as i64;
        let dopamine_gain = (SCALE + ctx.neuromodulation.dopamine.abs()) as i64;

        let lr = (lr as i64 * neuromod_gain * dopamine_gain) >> 20;
        let lr = lr as i32;

        // If reward is negative, we can invert the learning or inhibit it
        let reward_mod = if let Some(r) = ctx.reward { if r < 0 { -1 } else { 1 } } else { 1 };
        let lr_mod = lr * reward_mod * smbp_mod;

        let old_weight = *weight;
        if ctx.pre_spiked && ctx.post_spiked {
            *weight = weight.saturating_add(lr_mod);
        } else if ctx.pre_spiked && !ctx.post_spiked {
            *weight = weight.saturating_sub(lr_mod / 2);
        }

        // Sign Preservation: Ensure weight never crosses zero (Dale's Law)
        if old_weight > 0 && *weight < 0 { *weight = 1; }
        if old_weight < 0 && *weight > 0 { *weight = -1; }
        if *weight > WEIGHT_CLAMP_LIMIT { *weight = WEIGHT_CLAMP_LIMIT; }
        if *weight < -WEIGHT_CLAMP_LIMIT { *weight = -WEIGHT_CLAMP_LIMIT; }
    }
}

/// Structure of Arrays (SoA) layout for neural state data.
/// Optimized for SIMD access and GPU memory alignment.
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct NeuronsSoA {
    /// Optional identifier for neural functional columns/layers.
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
    pub activity_ema: Vec<IValue>, // Long-term activity tracking (SCALE = 1.0)
    pub is_excitatory: Vec<bool>,
    pub adaptation_current: Vec<IValue>, // Spike-Frequency Adaptation (SFA)
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
            activity_ema: vec![0; size],
            is_excitatory: vec![true; size],
            adaptation_current: vec![0; size],
        }
    }
    pub fn len(&self) -> usize {
        self.potential.len()
    }

    pub fn validate(&self) -> Result<(), String> {
        let l = self.len();
        if self.layer_id.len() != l { return Err("layer_id length mismatch".into()); }
        if self.threshold.len() != l { return Err("threshold length mismatch".into()); }
        if self.base_threshold.len() != l { return Err("base_threshold length mismatch".into()); }
        if self.decay.len() != l { return Err("decay length mismatch".into()); }
        if self.refractory_timer.len() != l { return Err("refractory_timer length mismatch".into()); }
        if self.last_spike_tick.len() != l { return Err("last_spike_tick length mismatch".into()); }
        if self.update_interval.len() != l { return Err("update_interval length mismatch".into()); }
        if self.next_update_tick.len() != l { return Err("next_update_tick length mismatch".into()); }
        if self.activity_ema.len() != l { return Err("activity_ema length mismatch".into()); }
        if self.is_excitatory.len() != l { return Err("is_excitatory length mismatch".into()); }
        if self.adaptation_current.len() != l { return Err("adaptation_current length mismatch".into()); }
        Ok(())
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
        self.activity_ema.resize(new_size, 0);
        self.is_excitatory.resize(new_size, true);
        self.adaptation_current.resize(new_size, 0);
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum SpikeData {
    Sparse(Vec<usize>),
    Dense(Vec<u8>), // Bitmask
    Compressed(Vec<u8>), // Elias-Fano or similar bit-packed format
}

impl Default for SpikeData {
    fn default() -> Self {
        Self::Sparse(Vec::new())
    }
}

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct SynapsesSoA {
    pub source_index: Vec<u32>,
    pub target_index: Vec<u32>,
    pub weight: Vec<IValue>,
    pub delay: Vec<u8>, // Axonal delays (1-16 ticks)
    pub stp_resources: Vec<IValue>, // Short-Term Depression (SCALE = 1.0)
    pub stp_calcium: Vec<IValue>,   // Short-Term Facilitation (SCALE = 1.0)
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
            delay: Vec::with_capacity(capacity),
            stp_resources: Vec::with_capacity(capacity),
            stp_calcium: Vec::with_capacity(capacity),
            compartment: Vec::with_capacity(capacity),
            latent_matrix: None,
        }
    }

    pub fn push(&mut self, source: u32, target: u32, weight: IValue) {
        self.push_to_compartment(source, target, weight, 1, Compartment::Proximal);
    }

    pub fn push_delayed(&mut self, source: u32, target: u32, weight: IValue, delay: u8) {
        self.push_to_compartment(source, target, weight, delay, Compartment::Proximal);
    }

    pub fn push_to_compartment(&mut self, source: u32, target: u32, weight: IValue, delay: u8, compartment: Compartment) {
        self.source_index.push(source);
        self.target_index.push(target);
        self.weight.push(weight);
        self.delay.push(delay.max(1));
        self.stp_resources.push(SCALE); // Start fully charged
        self.stp_calcium.push(0);       // Start at baseline
        self.compartment.push(compartment);
    }

    pub fn push_polarized(&mut self, source: u32, target: u32, weight: IValue, delay: u8, compartment: Compartment, neurons: &NeuronsSoA) {
        let polarized_weight = if neurons.is_excitatory[source as usize] {
            weight.abs()
        } else {
            -weight.abs()
        };
        self.push_to_compartment(source, target, polarized_weight, delay, compartment);
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
        self.delay.swap_remove(index);
        self.stp_resources.swap_remove(index);
        self.stp_calcium.swap_remove(index);
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
    pub max_neurons: usize,
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

    // Stochastic Firing (Neural Noise)
    pub noise_amplitude: IValue, // SCALE = 1.0 (max noise)

    // Hardware Backend
    pub preferred_backend: String,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            default_threshold: SCALE,
            default_decay: 50,
            learning_rate: 10,
            max_synapses: 1_000_000,
            max_neurons: 100_000,
            neurogenesis_reward_threshold: 200,
            pruning_threshold: 10,
            dendritic_coincidence_threshold: 512, // 0.5 * SCALE
            ip_increment: 50,
            ip_decay: 1,
            stdp_tau: 20,
            stdp_a_plus: 100,
            stdp_a_minus: 100,
            metaplasticity_enabled: true,
            noise_amplitude: 50, // 5% noise by default
            preferred_backend: "cpu".to_string(),
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
    fn test_neurons_validation() {
        let mut neurons = NeuronsSoA::new(10);
        assert!(neurons.validate().is_ok());

        neurons.potential.push(0); // Break length consistency
        assert!(neurons.validate().is_err());
    }

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

        if model.version != "4.2" {
             log::warn!("Loading model version {} into v4.2 engine. Physics scaling (1024) may differ from older versions.", model.version);
        }

        model.neurons.validate().map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let n_count = model.neurons.len();
        if model.local_range.1 > n_count {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Local range out of bounds"));
        }

        Ok(model)
    }
}
