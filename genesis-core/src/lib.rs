use serde::{Deserialize, Serialize};
use bytemuck::{Pod, Zeroable};
use std::fs::File;
use std::io::{BufReader, BufWriter};

pub type IValue = i32;
pub const SCALE: IValue = 1000;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Pod, Zeroable, Default)]
#[repr(C)]
pub struct NeuronState {
    pub potential: IValue,
    pub threshold: IValue,
    pub decay: IValue,
    pub refractory_timer: i32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Pod, Zeroable)]
#[repr(C)]
pub struct Synapse {
    pub source_index: u32,
    pub target_index: u32,
    pub weight: IValue,
}

#[derive(Serialize, Deserialize)]
pub struct BakedModel {
    pub neurons: Vec<NeuronState>,
    pub synapses: Vec<Synapse>,
    #[cfg(feature = "titan")]
    pub titan_memory: Option<titan::TitanMemory>,
    pub has_text: bool,
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

        // Potential update
        self.potential += input;

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

/// GSOP: Global Spiking Optimization Plasticity
/// Updates weight based on pre- and post-synaptic activity.
/// This happens at each tick or night cycle.
pub fn update_weight_gsop(weight: &mut IValue, pre_spiked: bool, post_spiked: bool, learning_rate: IValue) {
    if pre_spiked && post_spiked {
        // Potentiation: neurons fire together, weights grow
        *weight += learning_rate;
    } else if pre_spiked && !post_spiked {
        // Depression: pre fired but post didn't
        *weight -= learning_rate / 2;
    }

    // Clamp weight for stability
    if *weight > SCALE * 5 { *weight = SCALE * 5; }
    if *weight < -SCALE * 5 { *weight = -SCALE * 5; }
}

#[cfg(feature = "titan")]
pub mod titan;
#[cfg(feature = "text")]
pub mod text;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gsop_potentiation() {
        let mut w = 500;
        update_weight_gsop(&mut w, true, true, 100);
        assert_eq!(w, 600);
    }

    #[test]
    fn test_gsop_depression() {
        let mut w = 500;
        update_weight_gsop(&mut w, true, false, 100);
        assert_eq!(w, 450);
    }
}
