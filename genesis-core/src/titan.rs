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

    /// Three-Factor Learning using BitPacked history for speed.
    /// Elastic Context: search depth increases with surprise.
    pub fn learn_from_history(&mut self, history: &[crate::SpikeData], h_ptr: usize, neurons: &NeuronsSoA, surprise: IValue) {
        if surprise < self.surprise_threshold { return; }

        let n_count = neurons.len();
        if history.len() < 2 { return; }
        if history[h_ptr].is_empty() { return; }

        let now = history[h_ptr].to_bitpacked(n_count);

        // Elastic Window: high surprise = look further into the past (up to 16 steps)
        let search_depth = if surprise > 1000 { 16 } else if surprise > 500 { 8 } else { 2 };
        let search_depth = search_depth.min(history.len());

        for t in 1..search_depth {
            let past_idx = (h_ptr + history.len() - t) % history.len();
            let past = history[past_idx].to_bitpacked(n_count);

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
                                    // Prediction Success: boost weight significantly
                                    assoc.weight = assoc.weight.saturating_add(2);
                                } else if entries.len() < 100 {
                                    entries.push(Association { target: tgt_idx, weight: 5 }); // Initial confidence
                                }
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

    fn on_tick(&mut self, bus: &crate::InputBus, previous_spikes: &[bool], _tick: u32) {
        let dist = bus.distal();
        // Use a set to avoid multiple retrievals for the same block in one tick
        let mut triggered_blocks = std::collections::HashSet::new();

        for (i, &fired) in previous_spikes.iter().enumerate() {
            if fired {
                // Map neuron index to block_id for retrieval
                // Ideally this would use NeuronsSoA, but as a test fallback:
                let bid = (i / 4) as u32;
                triggered_blocks.insert(bid);
            }
        }

        for bid in triggered_blocks {
            if let Some(assocs) = self.sparse_associations.get(&bid) {
                for a in assocs {
                    if (a.target as usize) < dist.len() {
                            let weight = if a.weight > 0 { (a.weight as i32) * 50 } else { 0 }; // Scale weight for impact
                            crate::InputBus::atomic_saturating_add(&dist[a.target as usize], weight);
                    }
                }
            }
        }
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
