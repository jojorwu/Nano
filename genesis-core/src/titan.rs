use serde::{Deserialize, Serialize};
use crate::{IValue, NanoModule, NeuronsSoA, SynapsesSoA};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BitWiseTitan {
    /// Sparse Associative Memory: maps (input_block_id) -> Vec<(target_neuron_id, weight_counter)>
    /// This drastically reduces memory for sparse networks.
    pub sparse_associations: std::collections::HashMap<u32, Vec<Association>>,
    pub learning_rate: IValue,
    pub surprise_threshold: IValue,
    pub decay_rate: IValue,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Association {
    pub target: u32,
    pub weight: i16, // Using 16-bit counters for efficiency
}

impl BitWiseTitan {
    pub fn new(lr: IValue) -> Self {
        Self {
            sparse_associations: std::collections::HashMap::new(),
            learning_rate: lr,
            surprise_threshold: 100,
            decay_rate: 1,
        }
    }

    /// Three-Factor Learning using BitPacked history for speed
    pub fn learn_from_history(&mut self, history: &[crate::SpikeData], neurons: &NeuronsSoA, surprise: IValue) {
        if surprise < self.surprise_threshold { return; }

        let n_count = neurons.len();
        if history.len() < 2 { return; }

        let now = history[0].to_bitpacked(n_count);
        let past = history[1].to_bitpacked(n_count);

        // Find co-active blocks
        for (i, &past_word) in past.iter().enumerate() {
            if past_word == 0 { continue; }
            for bit in 0..64 {
                if (past_word >> bit) & 1 == 1 {
                    let src_idx = i * 64 + bit;
                    if src_idx >= n_count { continue; }
                    let src_bid = neurons.block_id[src_idx];

                    let entries = self.sparse_associations.entry(src_bid).or_insert_with(Vec::new);

                    // Update connections to neurons that are firing NOW
                    for (j, &now_word) in now.iter().enumerate() {
                        if now_word == 0 { continue; }
                        for now_bit in 0..64 {
                            if (now_word >> now_bit) & 1 == 1 {
                                let tgt_idx = (j * 64 + now_bit) as u32;
                                if let Some(assoc) = entries.iter_mut().find(|a| a.target == tgt_idx) {
                                    assoc.weight = assoc.weight.saturating_add(1);
                                } else if entries.len() < 100 { // Limit fan-out per block for stability
                                    entries.push(Association { target: tgt_idx, weight: 1 });
                                }
                            }
                        }
                    }
                }
            }
        }

        // Periodic Decay
        if surprise > 500 {
            for associations in self.sparse_associations.values_mut() {
                associations.retain_mut(|a| {
                    a.weight = a.weight.saturating_sub(1);
                    a.weight > 0
                });
            }
        }
    }
}

impl NanoModule for BitWiseTitan {
    fn name(&self) -> &str { "titan" }
    fn tier(&self) -> u32 { 0 }
    fn outputs(&self) -> Vec<String> { vec!["distal".to_string()] }

    fn on_tick(&mut self, _bus: &crate::InputBus, _previous_spikes: &[bool], _tick: u32) {
        // Retrieval is now predominantly handled by the backend or specialized calls
        // to maintain sparse efficiency.
    }

    fn on_update_weights(&mut self, _neurons: &mut NeuronsSoA, _previous_spikes: &[bool], _current_spikes: &[bool], _tick: u32, _surprise: Option<IValue>) {
        // We now use the specialized history-based learning in the pipeline.
    }

    fn on_night_phase(&mut self, _neurons: &mut NeuronsSoA, _synapses: &mut SynapsesSoA, reward: Option<IValue>) {
        if let Some(r) = reward {
             // Reward-modulated consolidation or pruning could go here
             log::debug!("Titan Night Phase with reward: {}", r);
        }
    }

    fn get_state(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap_or_default()
    }

    fn set_state(&mut self, state: &[u8]) {
        if let Ok(new_self) = bincode::deserialize::<Self>(state) {
            *self = new_self;
        }
    }

    fn box_clone(&self) -> Box<dyn NanoModule> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitwise_titan_init() {
        let titan = BitWiseTitan::new(100);
        assert_eq!(titan.sparse_associations.len(), 0);
    }
}
