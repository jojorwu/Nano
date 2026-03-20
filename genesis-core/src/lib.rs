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

pub mod plasticity;

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::collections::HashMap;

pub type IValue = i32;
pub const SCALE: IValue = 1024; // 2^10 for bit-shift optimizations

/// Trait for weight update rules (e.g., GSOP, STDP)
pub trait PlasticityRule {
    fn update(&self, weight: &mut IValue, pre_spiked: bool, post_spiked: bool);
    fn update_contrastive(&self, weight: &mut IValue, layer_correlation: IValue) {
        // Default: Reduce weight if correlation in layer is too high (penalize redundancy)
        if layer_correlation > 512 {
             *weight = (*weight as i64 * (1024 - (layer_correlation / 10)) as i64 >> 10) as i32;
        }
    }
    fn update_rewarded(&self, weight: &mut IValue, pre_spiked: bool, post_spiked: bool, reward: IValue) {
        // Default: just do normal update if reward is positive, or nothing if negative?
        // Usually RL uses a third factor.
        if reward > 0 {
            self.update(weight, pre_spiked, post_spiked);
        }
    }
    fn update_temporal(&self, _weight: &mut IValue, _pre_tick: u64, _post_tick: u64, _current_tick: u64) {
        // Default implementation does nothing
    }
}

pub struct GsopRule {
    pub learning_rate: IValue,
}

impl PlasticityRule for GsopRule {
    fn update(&self, weight: &mut IValue, pre_spiked: bool, post_spiked: bool) {
        if pre_spiked && post_spiked {
            *weight = weight.saturating_add(self.learning_rate);
        } else if pre_spiked && !post_spiked {
            *weight = weight.saturating_sub(self.learning_rate / 2);
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
}

impl NeuronsSoA {
    pub fn new(size: usize) -> Self {
        Self {
            layer_id: vec![0; size],
            potential: vec![0; size],
            distal_potential: vec![0; size],
            proximal_potential: vec![0; size],
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
    }
}

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct SynapsesSoA {
    pub source_index: Vec<u32>,
    pub target_index: Vec<u32>,
    pub weight: Vec<IValue>,
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
            latent_matrix: None,
        }
    }

    pub fn push(&mut self, source: u32, target: u32, weight: IValue) {
        self.source_index.push(source);
        self.target_index.push(target);
        self.weight.push(weight);
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

    // Intrinsic Plasticity
    pub intrinsic_plasticity_increment: IValue,
    pub intrinsic_plasticity_decay: IValue,

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
            intrinsic_plasticity_increment: 50,
            intrinsic_plasticity_decay: 1,
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

        if model.version != "3.7" {
             log::warn!("Loading model version {} into v3.7 engine. Stability not guaranteed.", model.version);
        }

        Ok(model)
    }
}
