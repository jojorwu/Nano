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
pub const SCALE: IValue = 1000;

/// Trait for weight update rules (e.g., GSOP, STDP)
pub trait PlasticityRule {
    fn update(&self, weight: &mut IValue, pre_spiked: bool, post_spiked: bool);
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
    pub potential: Vec<IValue>,
    pub threshold: Vec<IValue>,
    pub base_threshold: Vec<IValue>, // Intrinsic Plasticity
    pub decay: Vec<IValue>,
    pub liquid_current: Vec<IValue>, // For LLIF (Liquid Neurons)
    pub refractory_timer: Vec<i32>,
    pub last_spike_tick: Vec<u32>,
    pub update_interval: Vec<u32>, // Sub-tick precision: 1 = every tick, 10 = every 10 ticks
    pub next_update_tick: Vec<u32>,
}

impl NeuronsSoA {
    pub fn new(size: usize) -> Self {
        Self {
            potential: vec![0; size],
            threshold: vec![1000; size],
            base_threshold: vec![1000; size],
            decay: vec![50; size],
            liquid_current: vec![0; size],
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
        self.potential.resize(new_size, 0);
        self.threshold.resize(new_size, 1000);
        self.base_threshold.resize(new_size, 1000);
        self.decay.resize(new_size, 50);
        self.liquid_current.resize(new_size, 0);
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

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct NetworkConfig {
    pub default_threshold: IValue,
    pub default_decay: IValue,
    pub learning_rate: IValue,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct BakedModel {
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
        bincode::deserialize_from(reader).map_err(|e| {
            log::error!("Error deserializing model from '{}': {:?}", path, e);
            std::io::Error::new(std::io::ErrorKind::Other, e)
        })
    }
}
