use crate::{NanoModule, NeuronsSoA, SynapsesSoA, InputBus, IValue, SCALE};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AttnResModule {
    /// Learned pseudo-query vectors for each layer.
    /// In this SNN implementation, we use them to modulate compartment gains.
    pub layer_queries: Vec<[IValue; 4]>, // [Proximal, Distal, Apical, Basal] weights per layer
    pub learning_rate: IValue,
}

impl AttnResModule {
    pub fn new(num_layers: usize) -> Self {
        Self {
            layer_queries: vec![[SCALE; 4]; num_layers],
            learning_rate: 10,
        }
    }
}

impl NanoModule for AttnResModule {
    fn name(&self) -> &str { "attn_res" }

    fn on_tick(&mut self, _bus: &InputBus, _previous_spikes: &[bool], _tick: u32) {
        // This module applies attention weights to the input bus signals
        // However, the bus is atomic and shared.
        // Real-time modulation might be better handled in the backend or by pre-scaling.
        // For now, we'll let the backend handle the actual gating based on `dendritic_gate`.
    }

    fn on_update_weights(&mut self, neurons: &mut NeuronsSoA, _previous_spikes: &[bool], _current_spikes: &[bool], _tick: u32, surprise: Option<IValue>) {
        let n_count = neurons.len();
        if n_count == 0 { return; }

        // Dynamic Attention Update:
        // If surprise is high, we adjust the layer queries to favor compartments that might
        // reduce surprise in the future (e.g., more attention to Context/Distal).
        if let Some(s) = surprise {
            for (_layer_id, queries) in self.layer_queries.iter_mut().enumerate() {
                // Simple heuristic: high surprise increases attention to Apical (feedback) and Distal (context)
                if s > 512 {
                    queries[1] = (queries[1] * 105) / 100; // Distal
                    queries[2] = (queries[2] * 105) / 100; // Apical
                } else if s < 100 {
                    queries[1] = (queries[1] * 95) / 100;
                    queries[2] = (queries[2] * 95) / 100;
                }

                for q in queries.iter_mut() {
                    *q = (*q).clamp(SCALE / 4, SCALE * 4);
                }
            }
        }

        // Apply these queries to the neurons' dendritic gates
        for i in 0..n_count {
            let lid = neurons.layer_id[i] as usize;
            if lid < self.layer_queries.len() {
                // In a true AttnRes, this would be input-dependent.
                // Here we use the layer-wide pseudo-query as a first approximation.
                let q = self.layer_queries[lid];

                // We use the Proximal query to set the base dendritic gate
                neurons.dendritic_gate[i] = q[0];
                neurons.distal_gate[i] = q[1];
                neurons.apical_gate[i] = q[2];
                neurons.basal_gate[i] = q[3];
            }
        }
    }

    fn on_night_phase(&mut self, _neurons: &mut NeuronsSoA, _synapses: &mut SynapsesSoA, _reward: Option<IValue>) {
        // Consolidation of attention weights
    }

    fn box_clone(&self) -> Box<dyn NanoModule> {
        Box::new(self.clone())
    }

    fn get_state(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap_or_default()
    }

    fn set_state(&mut self, state: &[u8]) {
        if let Ok(new_self) = bincode::deserialize::<Self>(state) {
            *self = new_self;
        }
    }
}
