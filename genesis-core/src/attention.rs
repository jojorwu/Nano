use crate::{NanoModule, NeuronsSoA, SynapsesSoA, InputBus, IValue, SCALE};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AttnResModule {
    /// Learned pseudo-query vectors for each functional block (mini-column).
    /// Groups of neurons (block_id) share these weights for efficiency.
    pub block_queries: Vec<[IValue; 4]>, // [Proximal, Distal, Apical, Basal] weights per block
    /// Context Fingerprint: weighted moving average of block-level activity.
    pub context_fingerprint: Vec<f32>,
    pub learning_rate: IValue,
}

impl AttnResModule {
    pub fn new(num_blocks: usize) -> Self {
        Self {
            block_queries: vec![[SCALE; 4]; num_blocks],
            context_fingerprint: vec![0.0; num_blocks],
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

    fn on_update_weights(&mut self, neurons: &mut NeuronsSoA, _previous_spikes: &[bool], current_spikes: &[bool], _tick: u32, surprise: Option<IValue>) {
        let n_count = neurons.len();
        if n_count == 0 { return; }

        // Update Context Fingerprint
        for i in 0..n_count {
            let bid = neurons.block_id[i] as usize;
            if bid < self.context_fingerprint.len() && current_spikes[i] {
                self.context_fingerprint[bid] = self.context_fingerprint[bid] * 0.9 + 0.1;
            }
        }
        for f in &mut self.context_fingerprint { *f *= 0.99; } // Slow decay

        // Dynamic Attention Update:
        // If surprise is high, we adjust the block queries to favor compartments that might
        // reduce surprise in the future (e.g., more attention to Context/Distal).
        if let Some(s) = surprise {
            for (_block_id, queries) in self.block_queries.iter_mut().enumerate() {
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

        // Apply these queries to the neurons' dendritic gates based on block_id and context
        for i in 0..n_count {
            let bid = neurons.block_id[i] as usize;
            if bid < self.block_queries.len() {
                // Context-Aware Selection:
                // if the block is already very active in current context,
                // prioritize memory (Distal) over new input (Proximal).
                let mut q = self.block_queries[bid];
                let context_impact = self.context_fingerprint[bid];

                if context_impact > 0.5 {
                    q[0] = (q[0] * 800) >> 10; // Lower Proximal
                    q[1] = (q[1] * 1200) >> 10; // Boost Distal (Memory)
                }

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
