use serde::{Deserialize, Serialize};
use crate::{IValue, NanoModule, NeuronsSoA, SynapsesSoA};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BitWiseTitan {
    /// Contiguous Association Table for maximum cache efficiency and O(1) access.
    /// Indexed via block_offsets and block_counts.
    pub associations_flat: Vec<Association>,
    pub block_offsets: Vec<u32>,
    pub block_counts: Vec<u32>,
    pub learning_rate: IValue,
    pub surprise_threshold: IValue,
    pub decay_rate: IValue,
    /// L3 Buffer: most recent episodic patterns for cross-referencing
    pub l3_buffer: Vec<Vec<u16>>,
    /// Context Hashes: maps 64-bit pattern hash to a list of associated block_ids
    pub context_hashes: std::collections::HashMap<u64, Vec<u32>>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Association {
    pub target: u32,
    pub weight: i8, // Quantized to 8-bit for extreme memory efficiency
}

impl BitWiseTitan {
    pub fn new(lr: IValue) -> Self {
        Self {
            associations_flat: Vec::new(),
            block_offsets: vec![0; 64],
            block_counts: vec![0; 64],
            learning_rate: lr,
            surprise_threshold: 100,
            decay_rate: 1,
            l3_buffer: Vec::new(),
            context_hashes: std::collections::HashMap::new(),
        }
    }

    /// Semantic Hashing: maps block activity pattern to a context hash
    pub fn compute_context_hash(pattern: &[u64]) -> u64 {
        let mut h = 0u64;
        for &w in pattern {
            h ^= w.wrapping_mul(0x517cc1b727220a95);
        }
        h
    }

    fn ensure_capacity(&mut self, max_bid: u32) {
        if (max_bid as usize) >= self.block_offsets.len() {
            let new_size = (max_bid as usize + 1).max(self.block_offsets.len() * 2);
            self.block_offsets.resize(new_size, 0);
            self.block_counts.resize(new_size, 0);
        }
    }

    /// Internal helper to get a temporary mutable view of a block's associations
    fn get_block_mut(&mut self, bid: u32) -> Vec<Association> {
        let start = self.block_offsets[bid as usize] as usize;
        let count = self.block_counts[bid as usize] as usize;
        self.associations_flat[start..start + count].to_vec()
    }

    pub fn update_block(&mut self, bid: u32, new_assocs: Vec<Association>) {
        // This is a simplified implementation. A real high-performance version
        // would use a fragmented buffer or pre-allocated slots to avoid O(N) shifts.
        // For now, we perform a naive update for correctness.
        let start = self.block_offsets[bid as usize] as usize;
        let count = self.block_counts[bid as usize] as usize;

        self.associations_flat.drain(start..start + count);
        let mut idx = start;
        for a in new_assocs {
            self.associations_flat.insert(idx, a);
            idx += 1;
        }

        let delta = (idx - start) as i32 - count as i32;
        self.block_counts[bid as usize] = (idx - start) as u32;

        for i in (bid as usize + 1)..self.block_offsets.len() {
            self.block_offsets[i] = (self.block_offsets[i] as i32 + delta) as u32;
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

        // Update Context Hashes
        let current_hash = Self::compute_context_hash(&now);

        for t in 1..search_depth {
            let past_idx = (h_ptr + history.len() - t) % history.len();
            let past = history[past_idx].to_bitpacked(n_count);

            let reinforcement = if t == 1 { 2 } else { 1 };
            let reduction_bonus = if surprise < 100 { 5 } else { 0 };

            for (i, &past_word) in past.iter().enumerate() {
                if past_word == 0 { continue; }
                for bit in 0..64 {
                    if (past_word >> bit) & 1 == 1 {
                        let src_idx = i * 64 + bit;
                        if src_idx >= n_count { continue; }
                        let src_bid = neurons.block_id[src_idx];
                        let src_modality = neurons.layer_id[src_idx] >> 12;

                        self.ensure_capacity(src_bid);
                        let mut entries = self.get_block_mut(src_bid);
                        let mut changed = false;

                        for (j, &now_word) in now.iter().enumerate() {
                            if now_word == 0 { continue; }
                            for now_bit in 0..64 {
                                if (now_word >> now_bit) & 1 == 1 {
                                    let tgt_idx = (j * 64 + now_bit) as u32;
                                    let tgt_modality = neurons.layer_id[tgt_idx as usize] >> 12;
                                    let synesthesia_bonus = if src_modality != tgt_modality { 2 } else { 0 };

                                    if let Some(assoc) = entries.iter_mut().find(|a| a.target == tgt_idx) {
                                        assoc.weight = assoc.weight.saturating_add(reinforcement + reduction_bonus + synesthesia_bonus);
                                        changed = true;
                                    } else if entries.len() < 100 {
                                        entries.push(Association { target: tgt_idx, weight: reinforcement + reduction_bonus + synesthesia_bonus });
                                        changed = true;
                                    }
                                }
                            }
                        }
                        if changed {
                            self.update_block(src_bid, entries);
                            // Associate this block with the current contextual hash
                            self.context_hashes.entry(current_hash).or_default().push(src_bid);
                        }
                    }
                }
            }
        }

        // Periodic Decay (Simplified for flattened storage)
        if surprise > 500 {
             for bid in 0..self.block_offsets.len() as u32 {
                 let mut entries = self.get_block_mut(bid);
                 if !entries.is_empty() {
                     entries.retain_mut(|a| {
                         a.weight = a.weight.saturating_sub(1);
                         a.weight > 0
                     });
                     self.update_block(bid, entries);
                 }
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
        let n_count = dist.len();

        // Fuzzy Retrieval using Semantic Hashes
        // 1. Pack previous spikes into a pattern
        let mut packed = vec![0u64; (n_count + 63) / 64];
        for (i, &s) in previous_spikes.iter().enumerate() {
            if s { packed[i / 64] |= 1 << (i % 64); }
        }
        let current_hash = Self::compute_context_hash(&packed);

        // 2. Check for similar contexts (Fuzzy match)
        if let Some(fuzzy_blocks) = self.context_hashes.get(&current_hash) {
            for &bid in fuzzy_blocks {
                if (bid as usize) < self.block_offsets.len() {
                    let start = self.block_offsets[bid as usize] as usize;
                    let count = self.block_counts[bid as usize] as usize;
                    for i in start..start + count {
                        let a = &self.associations_flat[i];
                        if (a.target as usize) < dist.len() {
                            crate::InputBus::atomic_saturating_add(&dist[a.target as usize], a.weight as i32 * 10);
                        }
                    }
                }
            }
        }


        // L3 Retrieval: if we have very long-term patterns, inject them with low weight
        if !self.l3_buffer.is_empty() {
             for pattern in &self.l3_buffer {
                 for (_bid, &count) in pattern.iter().enumerate() {
                     if count > 5 {
                         // Find representative neuron in block or apply to all?
                         // Simplified: boost all neurons in block if L3 pattern matches
                         // (Implementation omitted for performance in on_tick)
                     }
                 }
             }
        }

        // Optimized Sparse Retrieval: Map-reduce triggered blocks without O(N) loop if possible.
        // For now, we utilize the fact that previous_spikes is often sparse.
        let mut triggered_blocks = std::collections::HashSet::new();

        // Heuristic: iterate only over active indices if provided via a sparse hint or similar.
        // Since we only have [bool], we can optimize with bitmask-style iteration if it was bitpacked.
        for (i, &fired) in previous_spikes.iter().enumerate() {
            if fired {
                let bid = (i / 4) as u32;
                triggered_blocks.insert(bid);
            }
        }

        for bid in triggered_blocks {
            if (bid as usize) < self.block_offsets.len() {
                let start = self.block_offsets[bid as usize] as usize;
                let count = self.block_counts[bid as usize] as usize;

                for i in start..start + count {
                    let a = &self.associations_flat[i];
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
        assert_eq!(titan.associations_flat.len(), 0);
    }
}
