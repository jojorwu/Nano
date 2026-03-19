#[cfg(feature = "titan")]
pub mod titan;
#[cfg(feature = "text")]
pub mod text;
#[cfg(feature = "vision")]
pub mod vision;

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
    pub decay: Vec<IValue>,
    pub refractory_timer: Vec<i32>,
}

impl NeuronsSoA {
    pub fn new(size: usize) -> Self {
        Self {
            potential: vec![0; size],
            threshold: vec![1000; size],
            decay: vec![50; size],
            refractory_timer: vec![0; size],
        }
    }
    pub fn len(&self) -> usize {
        self.potential.len()
    }
}

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct SynapsesSoA {
    pub source_index: Vec<u32>,
    pub target_index: Vec<u32>,
    pub weight: Vec<IValue>,
}

impl SynapsesSoA {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            source_index: Vec::with_capacity(capacity),
            target_index: Vec::with_capacity(capacity),
            weight: Vec::with_capacity(capacity),
        }
    }

    pub fn push(&mut self, source: u32, target: u32, weight: IValue) {
        self.source_index.push(source);
        self.target_index.push(target);
        self.weight.push(weight);
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

#[derive(Serialize, Deserialize, Clone)]
pub struct BakedModel {
    pub neurons: NeuronsSoA,
    pub synapses: SynapsesSoA,
    #[cfg(feature = "titan")]
    pub titan_memory: Option<titan::TitanMemory>,
    pub has_text: bool,
    pub has_vision: bool,
    pub vocabulary: HashMap<String, usize>,
}

impl BakedModel {
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, self).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        bincode::deserialize_from(reader).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
}
