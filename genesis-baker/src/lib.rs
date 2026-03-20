use serde::{Deserialize, Serialize};
use genesis_core::{NeuronsSoA, SynapsesSoA, BakedModel};
#[cfg(feature = "titan")]
use genesis_core::titan::TitanMemory;
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelBlueprint {
    pub name: String,
    pub config: Option<genesis_core::NetworkConfig>,
    pub architecture: ArchitectureConfig,
    pub modules: Vec<ModuleConfig>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ArchitectureConfig {
    pub neuron_count: usize,
    pub synapse_count: usize,
    pub layers: Option<Vec<LayerConfig>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LayerConfig {
    pub id: u16,
    pub range: (usize, usize),
    pub name: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ModuleConfig {
    TitanMemory { learning_rate: i32, size: usize, decay_rate: Option<i32> },
    TextProcessor { vocab_size: usize },
    Vision { resolution: (u32, u32) },
    AudioProcessor { sample_rate: u32 },
    RobotControl { num_motors: usize },
}

impl ModelBlueprint {
    pub fn bake(&self) -> BakedModel {
        let mut neurons = NeuronsSoA::new(self.architecture.neuron_count);
        if let Some(ref layers) = self.architecture.layers {
            for layer in layers {
                for i in layer.range.0..layer.range.1 {
                    if i < neurons.len() { neurons.layer_id[i] = layer.id; }
                }
            }
        }
        let mut synapses = SynapsesSoA::with_capacity(self.architecture.synapse_count);

        for i in 0..self.architecture.synapse_count {
            synapses.push(
                (i % self.architecture.neuron_count) as u32,
                ((i + 1) % self.architecture.neuron_count) as u32,
                500,
            );
        }

        #[cfg(feature = "titan")]
        let mut titan_memory = None;
        let mut has_text = false;
        let mut has_vision = false;
        let mut has_audio = false;
        let mut has_robotics = false;
        let mut has_fusion = false;

        for module in &self.modules {
            match module {
                ModuleConfig::TitanMemory { learning_rate, size, decay_rate } => {
                    #[cfg(feature = "titan")]
                    {
                        let mut tm = TitanMemory::new(*size, *learning_rate);
                        if let Some(dr) = decay_rate { tm.decay_rate = *dr; }
                        titan_memory = Some(tm);
                    }
                },
                ModuleConfig::TextProcessor { .. } => has_text = true,
                ModuleConfig::Vision { .. } => has_vision = true,
                ModuleConfig::AudioProcessor { .. } => has_audio = true,
                ModuleConfig::RobotControl { .. } => has_robotics = true,
            }
        }

        // Logic for auto-enabling fusion if multiple modalities are present
        if (has_text as usize + has_vision as usize + has_audio as usize) > 1 {
            has_fusion = true;
        }

        BakedModel {
            version: "3.7".to_string(),
            config: self.config.clone().unwrap_or_default(),
            node_id: 0,
            local_range: (0, self.architecture.neuron_count),
            neurons,
            synapses,
            #[cfg(feature = "titan")]
            titan_memory,
            has_text,
            has_vision,
            has_audio,
            has_robotics,
            has_fusion,
            vocabulary: HashMap::new()
        }
    }
}
