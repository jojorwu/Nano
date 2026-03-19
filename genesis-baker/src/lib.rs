use serde::{Deserialize, Serialize};
use genesis_core::{NeuronState, Synapse, BakedModel};
#[cfg(feature = "titan")]
use genesis_core::titan::TitanMemory;
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelBlueprint {
    pub name: String,
    pub architecture: ArchitectureConfig,
    pub modules: Vec<ModuleConfig>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ArchitectureConfig {
    pub neuron_count: usize,
    pub synapse_count: usize,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ModuleConfig {
    TitanMemory { learning_rate: i32, size: usize },
    TextProcessor { vocab_size: usize },
}

impl ModelBlueprint {
    pub fn bake(&self) -> BakedModel {
        let neurons = vec![NeuronState::default(); self.architecture.neuron_count];
        let mut synapses = Vec::with_capacity(self.architecture.synapse_count);

        for i in 0..self.architecture.synapse_count {
            synapses.push(Synapse {
                source_index: (i % self.architecture.neuron_count) as u32,
                target_index: ((i + 1) % self.architecture.neuron_count) as u32,
                weight: 500,
            });
        }

        let mut titan_memory = None;
        let mut has_text = false;

        for module in &self.modules {
            match module {
                ModuleConfig::TitanMemory { learning_rate, size } => {
                    #[cfg(feature = "titan")]
                    { titan_memory = Some(TitanMemory::new(*size, *learning_rate)); }
                },
                ModuleConfig::TextProcessor { .. } => has_text = true,
            }
        }

        BakedModel { neurons, synapses, titan_memory, has_text, vocabulary: HashMap::new() }
    }
}
