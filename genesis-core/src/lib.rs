#[cfg(feature = "titan")]
pub mod titan;
#[cfg(feature = "text")]
pub mod text;

pub mod plasticity;

use serde::{Deserialize, Serialize};
use bytemuck::{Pod, Zeroable};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::collections::HashMap;

pub type IValue = i32;
pub const SCALE: IValue = 1000;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Pod, Zeroable)]
#[repr(C)]
pub struct NeuronState {
    pub potential: IValue,
    pub threshold: IValue,
    pub decay: IValue,
    pub refractory_timer: i32,
}

impl Default for NeuronState {
    fn default() -> Self {
        Self {
            potential: 0,
            threshold: 1000, // 1.0
            decay: 50,       // 0.05
            refractory_timer: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Pod, Zeroable, PartialEq)]
#[repr(C)]
pub struct Synapse {
    pub source_index: u32,
    pub target_index: u32,
    pub weight: IValue,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct BakedModel {
    pub neurons: Vec<NeuronState>,
    pub synapses: Vec<Synapse>,
    #[cfg(feature = "titan")]
    pub titan_memory: Option<titan::TitanMemory>,
    pub has_text: bool,
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

pub trait SpikingNeuron {
    fn tick(&mut self, input: IValue) -> bool;
}

impl SpikingNeuron for NeuronState {
    fn tick(&mut self, input: IValue) -> bool {
        if self.refractory_timer > 0 {
            self.refractory_timer -= 1;
            self.potential = 0;
            return false;
        }

        // Potential update with overflow protection
        self.potential = self.potential.saturating_add(input);

        // Decay
        self.potential = (self.potential * (SCALE - self.decay)) / SCALE;

        if self.potential >= self.threshold {
            self.potential = 0;
            self.refractory_timer = 2;
            return true;
        }
        false
    }
}

pub fn update_weight_gsop(weight: &mut IValue, pre_spiked: bool, post_spiked: bool, learning_rate: IValue) {
    if pre_spiked && post_spiked {
        *weight = weight.saturating_add(learning_rate);
    } else if pre_spiked && !post_spiked {
        *weight = weight.saturating_sub(learning_rate / 2);
    }
    if *weight > SCALE * 5 { *weight = SCALE * 5; }
    if *weight < -SCALE * 5 { *weight = -SCALE * 5; }
}
