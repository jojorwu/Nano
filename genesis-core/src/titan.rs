use serde::{Deserialize, Serialize};
use crate::{IValue, NanoModule, NeuronsSoA, SynapsesSoA, SCALE};

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
    /// LSH (Locality Sensitive Hashing) tables for fuzzy retrieval
    pub lsh_tables: Vec<std::collections::HashMap<u32, Vec<u32>>>,
    /// Byte-Addressable Memory Buffer
    pub byte_memory: Vec<u8>,
    /// Address range of neurons that map to byte_memory (start_idx, end_idx)
    pub memory_mapped_range: Option<(usize, usize)>,
    /// Scripting sequences stored in byte_memory
    pub script_sequences: Vec<ScriptSequence>,
    /// Memory Utility Tracking: tracks how often each block is successfully retrieved
    pub block_utility: Vec<f32>,
    /// Last access tick per block for age-based decay
    pub last_access: Vec<u32>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ScriptSequence {
    pub start_addr: usize,
    pub length: usize,
    pub trigger_pattern: Vec<u64>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Association {
    pub target: u32,
    pub weight: i8, // Quantized to 8-bit for extreme memory efficiency
}

impl BitWiseTitan {
    pub fn new(lr: IValue) -> Self {
        let mut lsh_tables = Vec::with_capacity(4);
        for _ in 0..4 {
            lsh_tables.push(std::collections::HashMap::new());
        }
        Self {
            associations_flat: Vec::with_capacity(1000000), // Pre-allocate for scale
            block_offsets: vec![0; 64],
            block_counts: vec![0; 64],
            learning_rate: lr,
            surprise_threshold: 100,
            decay_rate: 1,
            l3_buffer: Vec::new(),
            context_hashes: std::collections::HashMap::new(),
            lsh_tables,
            byte_memory: vec![0; 1024 * 1024], // 1MB initial byte memory
            memory_mapped_range: None,
            block_utility: vec![0.0; 64],
            last_access: vec![0; 64],
            script_sequences: Vec::new(),
        }
    }

    /// Semantic Hashing: maps block activity pattern to a context hash
    pub fn compute_context_hash(pattern: &[u64]) -> u64 {
        let mut h = 0u64;
        for (i, &w) in pattern.iter().enumerate() {
            h ^= w.wrapping_mul(0x517cc1b727220a95 ^ (i as u64));
            h = h.rotate_left(7);
        }
        h
    }

    /// Compute LSH signatures for fuzzy matching
    pub fn compute_lsh_signatures(pattern: &[u64]) -> Vec<u32> {
        let mut sigs = vec![0u32; 4];
        for (i, sig) in sigs.iter_mut().enumerate() {
            let mut h = 0u64;
            for (j, &w) in pattern.iter().enumerate() {
                // Different projection for each table
                h ^= w.wrapping_mul(0xbf58476d1ce4e5b9 ^ (i as u64) ^ (j as u64));
            }
            *sig = (h ^ (h >> 32)) as u32;
        }
        sigs
    }

    fn ensure_capacity(&mut self, max_bid: u32) {
        if (max_bid as usize) >= self.block_offsets.len() {
            let new_size = (max_bid as usize + 1).max(self.block_offsets.len() * 2);
            self.block_offsets.resize(new_size, 0);
            self.block_counts.resize(new_size, 0);
            self.block_utility.resize(new_size, 0.0);
            self.last_access.resize(new_size, 0);
        }
    }

    /// Internal helper to get a temporary mutable view of a block's associations
    fn get_block_mut(&mut self, bid: u32) -> Vec<Association> {
        let start = self.block_offsets[bid as usize] as usize;
        let count = self.block_counts[bid as usize] as usize;
        self.associations_flat[start..start + count].to_vec()
    }

    pub fn update_block(&mut self, bid: u32, new_assocs: Vec<Association>) {
        // High-efficiency update: use a simple append-only strategy with fragmentation
        // Defragmentation happens during night_phase.
        let current_count = self.block_counts[bid as usize] as usize;
        let new_count = new_assocs.len();
        let start = self.block_offsets[bid as usize] as usize;

        if new_count <= current_count {
            // Update in-place
            for (i, a) in new_assocs.into_iter().enumerate() {
                self.associations_flat[start + i] = a;
            }
            self.block_counts[bid as usize] = new_count as u32;
        } else {
            // Grow: move to end of buffer
            let new_start = self.associations_flat.len();
            self.block_offsets[bid as usize] = new_start as u32;
            self.block_counts[bid as usize] = new_count as u32;
            self.associations_flat.extend(new_assocs);
        }
    }

    /// Three-Factor Learning using BitPacked history for speed.
    /// Elastic Context: search depth increases with surprise.
    pub fn learn_from_history(&mut self, history: &[crate::SpikeData], h_ptr: usize, neurons: &NeuronsSoA, surprise: IValue) {
        if surprise > 1500 {
            // Store highly surprising patterns in L3 episodic archive for night replay
            let now = history[h_ptr].to_bitpacked(neurons.len());
            let mut active_blocks = Vec::new();
            for (i, &word) in now.iter().enumerate() {
                if word == 0 { continue; }
                for bit in 0..64 {
                    if (word >> bit) & 1 == 1 {
                        let idx = i * 64 + bit;
                        if idx < neurons.len() {
                            let bid = neurons.block_id[idx];
                            if !active_blocks.contains(&bid) { active_blocks.push(bid); }
                        }
                    }
                }
            }
            if !active_blocks.is_empty() {
                self.l3_buffer.push(active_blocks.iter().map(|&b| b as u16).collect());
            }
        }

        // Test-Time Adaptation: Learn more aggressively if surprise is extreme
        let effective_threshold = if surprise > 2000 { self.surprise_threshold / 2 } else { self.surprise_threshold };
        if surprise < effective_threshold { return; }

        let n_count = neurons.len();
        if history.len() < 2 { return; }
        if history[h_ptr].is_empty() { return; }

        let now = history[h_ptr].to_bitpacked(n_count);

        // Elastic Window: high surprise = look further into the past (up to 16 steps)
        let search_depth = if surprise > 1000 { 16 } else if surprise > 500 { 8 } else { 2 };
        let search_depth = search_depth.min(history.len());

        // Update Context Hashes and LSH
        let current_hash = Self::compute_context_hash(&now);
        let current_sigs = Self::compute_lsh_signatures(&now);

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
                                        let boost = if surprise > 1500 { 5 } else { 0 };
                                        assoc.weight = assoc.weight.saturating_add(reinforcement + reduction_bonus + synesthesia_bonus + boost);
                                        changed = true;
                                    } else if entries.len() < 256 { // Increased capacity per block
                                        let boost = if surprise > 1500 { 10 } else { 0 };
                                        entries.push(Association { target: tgt_idx, weight: reinforcement + reduction_bonus + synesthesia_bonus + boost });
                                        changed = true;
                                    }
                                }
                            }
                        }
                        if changed {
                            self.update_block(src_bid, entries);
                            // Associate this block with the current contextual hash
                            self.context_hashes.entry(current_hash).or_default().push(src_bid);
                            for (table_idx, &sig) in current_sigs.iter().enumerate() {
                                let table = &mut self.lsh_tables[table_idx];
                                let entry = table.entry(sig).or_default();
                                if !entry.contains(&src_bid) {
                                    entry.push(src_bid);
                                }
                            }
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
        // Script Execution (Hierarchical Planning)
        // Check if any script is triggered by the previous spikes
        let n_count_script = previous_spikes.len();
        let packed_len = (n_count_script + 63) / 64;
        let mut current_packed = vec![0u64; packed_len];
        for (i, &s) in previous_spikes.iter().enumerate() {
            if s { current_packed[i / 64] |= 1 << (i % 64); }
        }

        for script in &self.script_sequences {
            let mut matched = true;
            for (i, &pattern_word) in script.trigger_pattern.iter().enumerate() {
                if i < packed_len && (current_packed[i] & pattern_word) != pattern_word {
                    matched = false;
                    break;
                }
            }

            if matched {
                // Execute script: read bytes and inject as proximal signals over time
                if script.length > 0 && script.start_addr < self.byte_memory.len() {
                    let first_byte = self.byte_memory[script.start_addr];
                    let prox = bus.proximal();
                    for i in 0..8 {
                        if (i as usize) < prox.len() {
                            if (first_byte >> i) & 1 == 1 {
                                crate::InputBus::atomic_saturating_add(&prox[i], SCALE);
                            }
                        }
                    }
                }
            }
        }

        // Memory-Mapped Interface: Read/Write from byte_memory
        if let Some((start, end)) = self.memory_mapped_range {
            let mut addr = 0usize;
            let mut val = 0u8;
            let addr_bits = 16; // Fixed 16 bits for address

            for i in 0..addr_bits {
                if start + i < previous_spikes.len() && previous_spikes[start + i] {
                    addr |= 1 << i;
                }
            }
            for i in 0..8 {
                if start + addr_bits + i < previous_spikes.len() && previous_spikes[start + addr_bits + i] {
                    val |= 1 << i;
                }
            }

            addr %= self.byte_memory.len();

            // Writing to byte memory if a specific "Write Enable" neuron is active
            let write_enable_idx = start + addr_bits + 8;
            if write_enable_idx < end && write_enable_idx < previous_spikes.len() && previous_spikes[write_enable_idx] {
                self.byte_memory[addr] = val;
            }

            // Reading from byte memory: inject byte value into potentials of memory neurons
            let read_byte = self.byte_memory[addr];
            let prox = bus.proximal();
            for i in 0..8 {
                let output_idx = start + addr_bits + 9 + i;
                if output_idx < end && (read_byte >> i) & 1 == 1 {
                    crate::InputBus::atomic_saturating_add(&prox[output_idx], SCALE / 2);
                }
            }
        }

        let dist = bus.distal();
        let n_count = dist.len();

        // 1. Pack previous spikes into a pattern
        let mut packed = vec![0u64; (n_count + 63) / 64];
        for (i, &s) in previous_spikes.iter().enumerate() {
            if s { packed[i / 64] |= 1 << (i % 64); }
        }

        // Fuzzy Retrieval using LSH and Semantic Hashes
        let mut candidate_blocks = std::collections::HashSet::new();

        // 2a. Exact Match
        let current_hash = Self::compute_context_hash(&packed);
        if let Some(blocks) = self.context_hashes.get(&current_hash) {
            for &bid in blocks { candidate_blocks.insert(bid); }
        }

        // 2b. LSH Fuzzy Match (Only if exact match didn't yield enough or always for robustness?)
        // If we want "everything", we check LSH too.
        let sigs = Self::compute_lsh_signatures(&packed);
        for (i, &sig) in sigs.iter().enumerate() {
            if let Some(blocks) = self.lsh_tables[i].get(&sig) {
                for &bid in blocks {
                    candidate_blocks.insert(bid);
                }
            }
        }

        // 3. Inject candidates into distal potential
        for bid in candidate_blocks {
            if (bid as usize) < self.block_offsets.len() {
                // Update Utility and Last Access
                self.block_utility[bid as usize] = self.block_utility[bid as usize] * 0.99 + 0.1;
                self.last_access[bid as usize] = _tick;

                let start = self.block_offsets[bid as usize] as usize;
                let count = self.block_counts[bid as usize] as usize;
                for i in start..start + count {
                    let a = &self.associations_flat[i];
                    if (a.target as usize) < dist.len() {
                        // Fuzzy matches get slightly less weight than exact (if we could distinguish)
                        // For now, simple boost.
                        crate::InputBus::atomic_saturating_add(&dist[a.target as usize], a.weight as i32 * 15);
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
                // Update Utility and Last Access
                self.block_utility[bid as usize] = self.block_utility[bid as usize] * 0.99 + 0.05;
                self.last_access[bid as usize] = _tick;

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

    fn on_update_weights(&mut self, neurons: &mut NeuronsSoA, _previous_spikes: &[bool], current_spikes: &[bool], _tick: u32, surprise: Option<IValue>) {
        // Metaplasticity: Adjust plasticity_gate based on surprise and activity
        if let Some(s) = surprise {
            for i in 0..neurons.len() {
                if current_spikes[i] {
                    // High surprise increases plasticity (learning)
                    if s > 800 {
                        neurons.plasticity_gate[i] = neurons.plasticity_gate[i].saturating_add(50).min(SCALE);
                    } else if s < 100 {
                        // Low surprise / stability reduces plasticity (freezing weights)
                        neurons.plasticity_gate[i] = neurons.plasticity_gate[i].saturating_sub(5);
                    }
                }
            }
        }
    }

    fn on_night_phase(&mut self, _neurons: &mut NeuronsSoA, _synapses: &mut SynapsesSoA, reward: Option<IValue>) {
        let r_val = reward.unwrap_or(0);
        log::debug!("Titan Night Phase with reward: {}", r_val);

        // Human-like Forgetting: Decay weights based on utility and age
        // Memories with low utility or those that haven't been accessed in a long time decay faster.
        for bid in 0..self.block_offsets.len() as u32 {
            let utility = self.block_utility[bid as usize];
            let mut entries = self.get_block_mut(bid);
            if entries.is_empty() { continue; }

            // Base decay + Utility-based protection
            // If utility is high (frequently used), decay is slower.
            let decay_amount: i32 = if utility > 0.5 { 0 } else if utility > 0.1 { 1 } else { 2 };

            // Protection for "important" memories (associated with high reward)
            let protection: i32 = if r_val > 500 { 1 } else { 0 };

            // Surprise-Modulated Protection: if the block was recently updated (likely due to high surprise), protect it.
            let surprise_protection = if self.last_access[bid as usize] > 0 { 1 } else { 0 };

            let final_decay = decay_amount.saturating_sub(protection + surprise_protection);

            if final_decay > 0 {
                entries.retain_mut(|a| {
                    a.weight = a.weight.saturating_sub(final_decay as i8);
                    a.weight > 0
                });
                self.update_block(bid, entries);
            }

            // Global Utility Decay (Forgetfulness over time)
            self.block_utility[bid as usize] *= 0.95;
            // Reset last_access for next cycle to ensure protection is "recent"
            self.last_access[bid as usize] = 0;
        }

        // Deep Replay: Re-process L3 Episodic Archive during sleep
        // This simulates the consolidation of memories from the hippocampus to the cortex.
        if !self.l3_buffer.is_empty() {
             log::debug!("Titan Deep Replay: consolidating {} episodic patterns", self.l3_buffer.len());
             for pattern in self.l3_buffer.clone() {
                 // If the replayed pattern exists in memory, boost all its associations
                 for &bid in &pattern {
                     let mut entries = self.get_block_mut(bid as u32);
                     for a in &mut entries {
                         a.weight = a.weight.saturating_add(1); // Consolidate
                     }
                     self.update_block(bid as u32, entries);
                 }
             }
             // L3 buffer clears partially after replay (gradual transfer to long-term)
             if self.l3_buffer.len() > 10 {
                 self.l3_buffer.drain(0..5);
             }
        }

        // Defragmentation: Rebuild associations_flat to remove gaps
        let mut new_flat = Vec::with_capacity(self.associations_flat.len());
        for bid in 0..self.block_offsets.len() {
            let start = self.block_offsets[bid] as usize;
            let count = self.block_counts[bid] as usize;
            let new_start = new_flat.len();

            if start + count <= self.associations_flat.len() {
                new_flat.extend_from_slice(&self.associations_flat[start..start+count]);
                self.block_offsets[bid] = new_start as u32;
            } else {
                self.block_counts[bid] = 0;
                self.block_offsets[bid] = new_start as u32;
            }
        }
        self.associations_flat = new_flat;
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

    #[test]
    fn test_memory_mapped_neurons() {
        println!("Starting test_memory_mapped_neurons");
        let mut titan = BitWiseTitan::new(100);
        titan.byte_memory = vec![0; 256];
        titan.memory_mapped_range = Some((0, 48));

        // 0..4: address bits (4 bits = 16 addresses)
        // 4..12: value bits (8 bits)
        // 12: write enable
        // 13..21: read output bits

        let bus = crate::InputBus::new(100);
        let mut spikes = vec![false; 100];

        // Write value 0xAA to address 5
        // Address 5 = 1010 binary (bits 0 and 2 set)
        spikes[0] = true;
        spikes[2] = true;

        // Value 0xAA = 10101010 binary
        spikes[16 + 1] = true;
        spikes[16 + 3] = true;
        spikes[16 + 5] = true;
        spikes[16 + 7] = true;

        // Write enable (addr_bits + 8 = 16 + 8 = 24)
        spikes[24] = true;

        titan.on_tick(&bus, &spikes, 0);
        assert_eq!(titan.byte_memory[5], 0xAA);

        // Read from address 5 (without write enable)
        spikes[24] = false;
        bus.clear();
        titan.on_tick(&bus, &spikes, 1);

        // Check if output bits (addr_bits + 9 .. = 16 + 9 = 25) got boosted
        let prox = bus.proximal();
        use std::sync::atomic::Ordering;
        assert!(prox[25 + 1].load(Ordering::Relaxed) > 0);
        assert!(prox[25 + 3].load(Ordering::Relaxed) > 0);
        assert!(prox[25 + 5].load(Ordering::Relaxed) > 0);
        assert!(prox[25 + 7].load(Ordering::Relaxed) > 0);
        assert_eq!(prox[25 + 0].load(Ordering::Relaxed), 0);
    }
}
