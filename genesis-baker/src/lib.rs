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

        // Balanced E-I: 80% Excitatory, 20% Inhibitory by default
        for i in 0..neurons.len() {
            neurons.is_excitatory[i] = if (i % 5) != 0 { 1 } else { 0 };
        }

        if let Some(ref layers) = self.architecture.layers {
            for layer in layers {
                for i in layer.range.0..layer.range.1 {
                    if i < neurons.len() { neurons.layer_id[i] = layer.id; }
                }
            }
        }
        let mut synapses = SynapsesSoA::with_capacity(self.architecture.synapse_count);

        use rand::Rng;
        let mut rng = rand::thread_rng();
        for _ in 0..self.architecture.synapse_count {
            let src = rng.gen_range(0..self.architecture.neuron_count) as u32;
            let tgt = rng.gen_range(0..self.architecture.neuron_count) as u32;
            if src == tgt { continue; }

            let delay = rng.gen_range(1..9);

            // Inhibitory neurons target Basal (lateral inhibition) or Proximal
            let comp = if neurons.is_excitatory[src as usize] == 0 {
                if rng.gen_bool(0.7) { genesis_core::Compartment::Basal } else { genesis_core::Compartment::Proximal }
            } else {
                if rng.gen_bool(0.8) { genesis_core::Compartment::Proximal } else { genesis_core::Compartment::Distal }
            };

            synapses.push_polarized(src, tgt, 500, delay, comp, &neurons);
        }

        #[cfg(feature = "titan")]
        let mut titan_memory = None;
        let mut has_text = false;
        let mut has_vision = false;
        let mut has_audio = false;
        let mut has_robotics = false;
        let mut has_fusion = false;

        let mut module_states = HashMap::new();

        for module in &self.modules {
            match module {
                ModuleConfig::TitanMemory { learning_rate, size, decay_rate } => {
                    #[cfg(feature = "titan")]
                    {
                        let mut tm = TitanMemory::new(*size, *learning_rate);
                        if let Some(dr) = decay_rate { tm.decay_rate = *dr; }
                        module_states.insert("titan".to_string(), bincode::serialize(&tm).unwrap());
                        titan_memory = Some(tm);
                    }
                },
                ModuleConfig::TextProcessor { .. } => {
                    has_text = true;
                    let m = genesis_core::text::TextProcessorModule::new(64);
                    module_states.insert("text_processor".to_string(), bincode::serialize(&m).unwrap());
                },
                ModuleConfig::Vision { resolution } => {
                    has_vision = true;
                    let m = genesis_core::vision::VisionModule::new(resolution.0, resolution.1);
                    module_states.insert("vision".to_string(), bincode::serialize(&m).unwrap());
                },
                ModuleConfig::AudioProcessor { .. } => has_audio = true,
                ModuleConfig::RobotControl { num_motors } => {
                    has_robotics = true;
                    // Map motor neurons to the end of the population
                    let start = self.architecture.neuron_count.saturating_sub(*num_motors);
                    let m = genesis_core::robotics::RobotControlModule::new((start..self.architecture.neuron_count).collect());
                    module_states.insert("robot_control".to_string(), bincode::serialize(&m).unwrap());
                }
            }
        }

        // Logic for auto-enabling fusion if multiple modalities are present
        if (has_text as usize + has_vision as usize + has_audio as usize) > 1 {
            has_fusion = true;
        }

        BakedModel {
            version: "4.0".to_string(),
            config: self.config.clone().unwrap_or_default(),
            node_id: 0,
            local_range: (0, self.architecture.neuron_count),
            neurons,
            synapses,
            module_states,
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
