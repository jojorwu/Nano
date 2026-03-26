use serde::{Deserialize, Serialize};
use crate::{IValue, NanoModule, NeuronsSoA, SynapsesSoA, SCALE};

pub type TitanMemory = BitWiseTitan;

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
    /// L3 Buffer: most recent episodic patterns for consolidation
    pub l3_buffer: Vec<EpisodicEvent>,
    /// Context Hashes: maps 64-bit pattern hash to a list of associated block_ids
    pub context_hashes: std::collections::HashMap<u64, Vec<u32>>,
    /// LSH (Locality Sensitive Hashing) tables for fuzzy retrieval (L1: Broad)
    pub lsh_tables: Vec<std::collections::HashMap<u32, Vec<u32>>>,
    /// LSH (Locality Sensitive Hashing) tables for detailed fuzzy retrieval (L2: Detailed)
    pub lsh_tables_l2: Vec<std::collections::HashMap<u32, Vec<u32>>>,
    /// Byte-Addressable Memory Buffer
    pub byte_memory: Vec<u8>,
    /// Address range of neurons that map to byte_memory (start_idx, end_idx)
    pub memory_mapped_range: Option<(usize, usize)>,
    /// Scripting sequences stored in byte_memory
    pub script_sequences: Vec<ScriptSequence>,
    /// Active script playback state: (script_idx, current_step)
    pub active_scripts: Vec<(usize, usize)>,
    /// Memory Utility Tracking: tracks how often each block is successfully retrieved
    pub block_utility: Vec<f32>,
    /// Last access tick per block for age-based decay
    pub last_access: Vec<u32>,
    /// Cached mapping from neuron index to block_id
    #[serde(default)]
    pub neuron_to_block: Vec<u32>,
    // Configurable Limits
    pub max_associations: usize,
    pub max_blocks: u32,
    pub max_entries_per_block: usize,
    pub deep_replay_threshold: IValue,
    pub elastic_window_max: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EpisodicEvent {
    pub context_hash: u64,
    pub active_blocks: Vec<u32>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ScriptSequence {
    pub start_addr: usize,
    pub length: usize,
    pub trigger_pattern: Vec<u64>,
    /// Ticks to wait between bytes
    pub interval: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Association {
    pub target: u32,
    pub weight: i8, // Quantized to 8-bit for extreme memory efficiency
}

impl BitWiseTitan {
    pub fn new_with_size(size: usize, lr: IValue) -> Self {
        let mut lsh_tables = Vec::with_capacity(4);
        let mut lsh_tables_l2 = Vec::with_capacity(4);
        for _ in 0..4 {
            lsh_tables.push(std::collections::HashMap::new());
            lsh_tables_l2.push(std::collections::HashMap::new());
        }
        Self {
            associations_flat: Vec::with_capacity(1_000_000), // Pre-allocate for scale
            block_offsets: vec![0; size.max(64)],
            block_counts: vec![0; size.max(64)],
            learning_rate: lr,
            surprise_threshold: 100,
            decay_rate: 1,
            l3_buffer: Vec::new(),
            context_hashes: std::collections::HashMap::new(),
            lsh_tables,
            lsh_tables_l2,
            byte_memory: vec![0; 1024 * 1024], // 1MB initial byte memory
            memory_mapped_range: None,
            block_utility: vec![0.0; size.max(64)],
            last_access: vec![0; size.max(64)],
            neuron_to_block: Vec::new(),
            script_sequences: Vec::new(),
            active_scripts: Vec::new(),
            max_associations: 1_000_000,
            max_blocks: 1_000_000,
            max_entries_per_block: 256,
            deep_replay_threshold: 1500,
            elastic_window_max: 16,
        }
    }

    pub fn new(lr: IValue) -> Self {
        Self::new_with_size(64, lr)
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

    /// VSA: Circular Convolution / XOR-binding for structured relationships
    pub fn vsa_bind(a: &[u64], b: &[u64]) -> Vec<u64> {
        let len = a.len().min(b.len());
        let mut result = vec![0u64; len];
        for i in 0..len {
            // XOR-binding is the standard for Binary Spatter Codes (BSC)
            result[i] = a[i] ^ b[i].rotate_left(1);
        }
        result
    }

    /// VSA: Superposition of patterns (bundling)
    pub fn vsa_bundle(patterns: &[Vec<u64>], n_count: usize) -> Vec<u64> {
        let packed_len = (n_count + 63) / 64;
        let mut result = vec![0u64; packed_len];
        if patterns.is_empty() { return result; }

        for i in 0..packed_len {
            let mut bit_counts = [0u16; 64];
            for p in patterns {
                if i < p.len() {
                    for bit in 0..64 {
                        if (p[i] >> bit) & 1 == 1 { bit_counts[bit] += 1; }
                    }
                }
            }
            // Majority rule for bundling
            let threshold = (patterns.len() / 2) as u16;
            for bit in 0..64 {
                if bit_counts[bit] > threshold {
                    result[i] |= 1 << bit;
                }
            }
        }
        result
    }

    /// Compute LSH signatures for fuzzy matching (Hierarchical)
    /// L1: Broad projection for coarse matching
    /// L2: Detailed projection for fine-grained matching
    pub fn compute_lsh_signatures_hierarchical(pattern: &[u64]) -> (Vec<u32>, Vec<u32>) {
        let mut l1 = vec![0u32; 4];
        let mut l2 = vec![0u32; 4];
        for (i, sig) in l1.iter_mut().enumerate() {
            let mut h = 0u64;
            for (j, &w) in pattern.iter().enumerate() {
                // Coarse projection: use bitCount for similarity-based hashing
                let pop = w.count_ones() as u64;
                h ^= pop.wrapping_mul(0xbf58476d1ce4e5b9 ^ (i as u64) ^ (j as u64));
                h = h.rotate_left(5);
            }
            *sig = (h ^ (h >> 32)) as u32;
        }
        for (i, sig) in l2.iter_mut().enumerate() {
            let mut h = 0u64;
            for (j, &w) in pattern.iter().enumerate() {
                // Detailed projection
                h ^= w.wrapping_mul(0x94d049bb133111eb ^ (i as u64) ^ (j as u64));
            }
            *sig = (h ^ (h >> 32)) as u32;
        }
        (l1, l2)
    }

    /// Backwards compatibility or default signature
    pub fn compute_lsh_signatures(pattern: &[u64]) -> Vec<u32> {
        Self::compute_lsh_signatures_hierarchical(pattern).0
    }

    fn ensure_capacity(&mut self, max_bid: u32) {
        // Limit max_bid to prevent excessive memory allocation (e.g. 1M blocks)
        let max_bid = max_bid.min(self.max_blocks);
        if (max_bid as usize) >= self.block_offsets.len() {
            let new_size = (max_bid as usize + 1).max(self.block_offsets.len() * 2).min(self.max_blocks as usize);
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
        if (bid as usize) >= self.block_offsets.len() { return; }
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
            // Respect total association limit
            if self.associations_flat.len() + new_count > self.max_associations {
                return;
            }
            // Grow: move to end of buffer
            let new_start = self.associations_flat.len();
            self.block_offsets[bid as usize] = new_start as u32;
            self.block_counts[bid as usize] = new_count as u32;
            self.associations_flat.extend(new_assocs);
        }
    }

    /// Merges state from another Titan instance (used for distributed synchronization)
    pub fn merge_state(&mut self, mut other: BitWiseTitan) {
        // Simple merge: append missing associations and update LSH tables
        for bid in 0..other.block_offsets.len() as u32 {
            let other_entries = other.get_block_const(bid);
            if other_entries.is_empty() { continue; }

            self.ensure_capacity(bid);
            let mut my_entries = self.get_block_mut(bid);
            let mut changed = false;

            for oa in other_entries {
                if let Some(ma) = my_entries.iter_mut().find(|a| a.target == oa.target) {
                    ma.weight = ma.weight.max(oa.weight);
                } else if my_entries.len() < self.max_entries_per_block {
                    my_entries.push(oa);
                    changed = true;
                }
            }
            if changed {
                self.update_block(bid, my_entries);
            }
        }

        // Merge LSH Tables
        let other_lsh = std::mem::take(&mut other.lsh_tables);
        for (i, table) in other_lsh.into_iter().enumerate() {
            if i < self.lsh_tables.len() {
                for (sig, blocks) in table {
                    let my_entry = self.lsh_tables[i].entry(sig).or_default();
                    for b in blocks {
                        if !my_entry.contains(&b) { my_entry.push(b); }
                    }
                }
            }
        }
        let other_lsh_l2 = std::mem::take(&mut other.lsh_tables_l2);
        for (i, table) in other_lsh_l2.into_iter().enumerate() {
            if i < self.lsh_tables_l2.len() {
                for (sig, blocks) in table {
                    let my_entry = self.lsh_tables_l2[i].entry(sig).or_default();
                    for b in blocks {
                        if !my_entry.contains(&b) { my_entry.push(b); }
                    }
                }
            }
        }
        // Merge Context Hashes
        for (hash, blocks) in other.context_hashes {
            let my_entry = self.context_hashes.entry(hash).or_default();
            for b in blocks {
                if !my_entry.contains(&b) { my_entry.push(b); }
            }
        }
    }

    /// Constant version of get_block_mut
    fn get_block_const(&self, bid: u32) -> Vec<Association> {
        if (bid as usize) >= self.block_offsets.len() { return Vec::new(); }
        let start = self.block_offsets[bid as usize] as usize;
        let count = self.block_counts[bid as usize] as usize;
        if start + count > self.associations_flat.len() { return Vec::new(); }
        self.associations_flat[start..start + count].to_vec()
    }

    /// Three-Factor Learning using BitPacked history for speed.
    /// Elastic Context: search depth increases with surprise.
    pub fn learn_from_history(&mut self, history: &[crate::SpikeData], h_ptr: usize, neurons: &NeuronsSoA, surprise: IValue) {
        // 1. Pattern Archiving
        self.archive_surprising_patterns(history, h_ptr, neurons, surprise);

        // 2. Learning Gate
        let effective_threshold = if surprise > 2000 { self.surprise_threshold / 2 } else { self.surprise_threshold };
        if surprise < effective_threshold || history.len() < 2 || history[h_ptr].is_empty() { return; }

        let n_count = neurons.len();
        let now_packed = history[h_ptr].to_bitpacked(n_count);

        // Pre-extract active indices for "now" to avoid scanning in nested loops
        let mut now_active = Vec::new();
        for (i, &word) in now_packed.iter().enumerate() {
            if word == 0 { continue; }
            for bit in 0..64 {
                if (word >> bit) & 1 == 1 {
                    let idx = (i * 64 + bit) as u32;
                    if (idx as usize) < n_count { now_active.push(idx); }
                }
            }
        }
        if now_active.is_empty() { return; }

        // 3. Elastic Temporal Window
        let search_depth = if surprise > 1000 { self.elastic_window_max } else if surprise > 500 { self.elastic_window_max / 2 } else { 2 };
        let search_depth = search_depth.min(history.len());

        // 4. Update Associations
        self.perform_temporal_learning_optimized(history, h_ptr, &now_packed, &now_active, search_depth, neurons, surprise);

        // 5. Global Memory Maintenance
        if surprise > 500 { self.decay_associations(); }
    }

    fn archive_surprising_patterns(&mut self, history: &[crate::SpikeData], h_ptr: usize, neurons: &NeuronsSoA, surprise: IValue) {
        if surprise > self.deep_replay_threshold {
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
                let context_hash = Self::compute_context_hash(&now);
                self.l3_buffer.push(EpisodicEvent { context_hash, active_blocks });
            }
        }
    }

    fn perform_temporal_learning_optimized(&mut self, history: &[crate::SpikeData], h_ptr: usize, now_packed: &[u64], now_active: &[u32], depth: usize, neurons: &NeuronsSoA, surprise: IValue) {
        let n_count = neurons.len();
        // Index by the 'now' signatures for retrieval when this pattern appears as 'previous'
        let (sigs_l1, sigs_l2) = Self::compute_lsh_signatures_hierarchical(now_packed);
        let now_hash = Self::compute_context_hash(now_packed);

        for t in 1..depth {
            let past_idx = (h_ptr + history.len() - t) % history.len();
            let past_packed = history[past_idx].to_bitpacked(n_count);
            let past_hash = Self::compute_context_hash(&past_packed);

            let bound = Self::vsa_bind(now_packed, &past_packed);
            let bound_hash = Self::compute_context_hash(&bound);

            let reinforcement = if t == 1 { 2 } else { 1 };
            let reduction_bonus = if surprise < 100 { 5 } else { 0 };

            for (i, &past_word) in past_packed.iter().enumerate() {
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

                        for &tgt_idx in now_active {
                            let tgt_modality = neurons.layer_id[tgt_idx as usize] >> 12;
                            let synesthesia_bonus = if src_modality != tgt_modality { 2 } else { 0 };

                            if let Some(assoc) = entries.iter_mut().find(|a| a.target == tgt_idx) {
                                let boost = if surprise > 1500 { 5 } else { 0 };
                                assoc.weight = assoc.weight.saturating_add(reinforcement + reduction_bonus + synesthesia_bonus + boost);
                                changed = true;
                            } else if entries.len() < self.max_entries_per_block {
                                let boost = if surprise > 1500 { 10 } else { 0 };
                                entries.push(Association { target: tgt_idx, weight: reinforcement + reduction_bonus + synesthesia_bonus + boost });
                                changed = true;
                            }
                        }
                        // Always return the block to memory
                        self.update_block(src_bid, entries);

                        if changed {
                            // Store mapping: Past Pattern Hash -> Src Block
                            // This allows retrieval when 'past' pattern repeats as 'previous_spikes'
                            self.context_hashes.entry(past_hash).or_default().push(src_bid);
                            self.context_hashes.entry(now_hash).or_default().push(src_bid);
                            self.context_hashes.entry(bound_hash).or_default().push(src_bid);
                            for (table_idx, &sig) in sigs_l1.iter().enumerate() { self.lsh_tables[table_idx].entry(sig).or_default().push(src_bid); }
                            for (table_idx, &sig) in sigs_l2.iter().enumerate() { self.lsh_tables_l2[table_idx].entry(sig).or_default().push(src_bid); }
                        }
                    }
                }
            }
        }
    }

    fn decay_associations(&mut self) {
        for bid in 0..self.block_offsets.len() as u32 {
            let mut entries = self.get_block_mut(bid);
            if !entries.is_empty() {
                entries.retain_mut(|a| { a.weight = a.weight.saturating_sub(1); a.weight > 0 });
                self.update_block(bid, entries);
            }
        }
    }

    /// Specialized learning from an episodic sequence replay
    pub fn learn_from_sequence(&mut self, sequence: &crate::episodic::EpisodicSequence, neurons: &NeuronsSoA) {
        if sequence.ticks.is_empty() { return; }
        let n_count = neurons.len();

        // Convert sequence ticks to SpikeData and perform history-based learning
        let history: Vec<crate::SpikeData> = sequence.ticks.iter().map(|t| {
            crate::SpikeData::Sparse(t.clone())
        }).collect();

        // Replay sequence: treat each step as "now" and learn associations with the past.
        // Optimization: avoid temporal wrap-around for linear episodic sequences.
        for i in 1..history.len() {
             // 1. Pattern Archiving (Skip if already archived during original event)
             // 2. Learning Gate
             let effective_threshold = if sequence.surprise_at_start > 2000 { self.surprise_threshold / 2 } else { self.surprise_threshold };
             if sequence.surprise_at_start < effective_threshold { continue; }

             let now_packed = history[i].to_bitpacked(n_count);
             let mut now_active = Vec::new();
             for (idx, &word) in now_packed.iter().enumerate() {
                 if word == 0 { continue; }
                 for bit in 0..64 {
                     if (word >> bit) & 1 == 1 {
                         let n_idx = (idx * 64 + bit) as u32;
                         if (n_idx as usize) < n_count { now_active.push(n_idx); }
                     }
                 }
             }
             if now_active.is_empty() { continue; }

             // 3. Elastic Temporal Window (Linear, no wrap)
             let search_depth = if sequence.surprise_at_start > 1000 { self.elastic_window_max } else if sequence.surprise_at_start > 500 { self.elastic_window_max / 2 } else { 2 };
             let actual_depth = search_depth.min(i); // Limit depth to available history in linear sequence

             // 4. Update Associations (using linear history slice)
             self.perform_temporal_learning_linear(&history[0..i+1], i, &now_packed, &now_active, actual_depth, neurons, sequence.surprise_at_start);
        }
    }

    fn perform_temporal_learning_linear(&mut self, history: &[crate::SpikeData], h_ptr: usize, now_packed: &[u64], now_active: &[u32], depth: usize, neurons: &NeuronsSoA, surprise: IValue) {
        let n_count = neurons.len();
        let (sigs_l1, sigs_l2) = Self::compute_lsh_signatures_hierarchical(now_packed);

        for t in 1..=depth {
            if t > h_ptr { break; }
            let past_idx = h_ptr - t;
            let past_packed = history[past_idx].to_bitpacked(n_count);
            let past_hash = Self::compute_context_hash(&past_packed);

            let bound = Self::vsa_bind(now_packed, &past_packed);
            let bound_hash = Self::compute_context_hash(&bound);

            let reinforcement = if t == 1 { 2 } else { 1 };
            let reduction_bonus = if surprise < 100 { 5 } else { 0 };

            for (i, &past_word) in past_packed.iter().enumerate() {
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

                        for &tgt_idx in now_active {
                            let tgt_modality = neurons.layer_id[tgt_idx as usize] >> 12;
                            let synesthesia_bonus = if src_modality != tgt_modality { 2 } else { 0 };

                            if let Some(assoc) = entries.iter_mut().find(|a| a.target == tgt_idx) {
                                let boost = if surprise > 1500 { 5 } else { 0 };
                                assoc.weight = assoc.weight.saturating_add(reinforcement + reduction_bonus + synesthesia_bonus + boost);
                                changed = true;
                            } else if entries.len() < self.max_entries_per_block {
                                let boost = if surprise > 1500 { 10 } else { 0 };
                                entries.push(Association { target: tgt_idx, weight: reinforcement + reduction_bonus + synesthesia_bonus + boost });
                                changed = true;
                            }
                        }
                        // Always return the block to memory
                        self.update_block(src_bid, entries);

                        if changed {
                            self.context_hashes.entry(past_hash).or_default().push(src_bid);
                            self.context_hashes.entry(bound_hash).or_default().push(src_bid);
                            for (table_idx, &sig) in sigs_l1.iter().enumerate() { self.lsh_tables[table_idx].entry(sig).or_default().push(src_bid); }
                            for (table_idx, &sig) in sigs_l2.iter().enumerate() { self.lsh_tables_l2[table_idx].entry(sig).or_default().push(src_bid); }
                        }
                    }
                }
            }
        }
    }
}

impl NanoModule for BitWiseTitan {
    fn name(&self) -> &str { "titan" }
    fn tier(&self) -> u32 { 0 }
    fn outputs(&self) -> Vec<String> { vec!["distal".to_string()] }
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }

    fn on_event(&mut self, event: &crate::event::GlobalEvent) {
        match event {
            crate::event::GlobalEvent::ConsolidationTriggered => {
                // Boost learning rate during consolidation replays
                self.learning_rate = self.learning_rate.saturating_add(5);
            }
            crate::event::GlobalEvent::RewardSignal(r) => {
                if *r > 500 {
                    // Halve surprise threshold for high reward events to capture associations more easily
                    self.surprise_threshold = self.surprise_threshold / 2;
                }
            }
            _ => {}
        }
    }

    fn on_init(&mut self, neurons: &mut NeuronsSoA) -> Result<(), crate::ModuleError> {
        self.neuron_to_block = neurons.block_id.clone();
        Ok(())
    }

    fn on_config_sync(&mut self, config: &crate::NetworkConfig) {
        self.max_associations = config.titan.max_associations;
        self.max_blocks = config.titan.max_blocks;
        self.max_entries_per_block = config.titan.max_entries_per_block;
        self.surprise_threshold = config.titan.surprise_threshold;
        self.deep_replay_threshold = config.titan.deep_replay_threshold;
        self.elastic_window_max = config.titan.elastic_window_max;
        self.decay_rate = config.titan.decay_rate;
    }

    fn on_tick(&mut self, bus: &crate::InputBus, previous_spikes: &[bool], _tick: u32) {
        // Script Execution (Hierarchical Planning)
        let n_count_script = previous_spikes.len();
        let packed_len = (n_count_script + 63) / 64;
        let mut current_packed = vec![0u64; packed_len];
        for (i, &s) in previous_spikes.iter().enumerate() {
            if s { current_packed[i / 64] |= 1 << (i % 64); }
        }

        // 1. Check for new triggers
        for (idx, script) in self.script_sequences.iter().enumerate() {
            let mut matched = true;
            if script.trigger_pattern.is_empty() || script.trigger_pattern.len() > packed_len {
                 matched = false;
            } else {
                for (i, &pattern_word) in script.trigger_pattern.iter().enumerate() {
                    if (current_packed[i] & pattern_word) != pattern_word {
                        matched = false;
                        break;
                    }
                }
            }

            if matched {
                if !self.active_scripts.iter().any(|(s_idx, _)| *s_idx == idx) {
                    self.active_scripts.push((idx, 0));
                }
            }
        }

        // 2. Execute active scripts
        let mut finished_scripts = Vec::new();
        for i in 0..self.active_scripts.len() {
            let (script_idx, step) = self.active_scripts[i];
            let script = &self.script_sequences[script_idx];

            // Interval gating
            if _tick % script.interval.max(1) != 0 { continue; }

            if step < script.length {
                let addr = script.start_addr + step;
                if addr < self.byte_memory.len() {
                    let byte = self.byte_memory[addr];
                    let prox = bus.proximal();
                    for b in 0..8 {
                        if (b as usize) < prox.len() {
                            if (byte >> b) & 1 == 1 {
                                crate::InputBus::atomic_saturating_add(&prox[b], SCALE);
                            }
                        }
                    }
                }
                self.active_scripts[i].1 += 1;
            } else {
                finished_scripts.push(i);
            }
        }
        for idx in finished_scripts.into_iter().rev() {
            self.active_scripts.remove(idx);
        }

        // Memory-Mapped Interface: Read/Write from byte_memory
        if let Some((start, end)) = self.memory_mapped_range {
            if start >= previous_spikes.len() || end > previous_spikes.len() || start >= end {
                return;
            }
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

        log::debug!("Titan on_tick: previous_spikes sum = {}", previous_spikes.iter().filter(|&&s| s).count());

        // 1. Pack previous spikes into a pattern
        let mut packed = vec![0u64; (n_count + 63) / 64];
        for (i, &s) in previous_spikes.iter().enumerate() {
            if s { packed[i / 64] |= 1 << (i % 64); }
        }

        // Hierarchical Fuzzy Retrieval using LSH and Semantic Hashes
        let mut candidate_blocks = std::collections::HashSet::new();

        // 2a. Exact Match
        let current_hash = Self::compute_context_hash(&packed);
        if let Some(blocks) = self.context_hashes.get(&current_hash) {
            for &bid in blocks { candidate_blocks.insert(bid); }
        }

        // 2b. Hierarchical LSH Fuzzy Match
        let (sigs_l1, sigs_l2) = Self::compute_lsh_signatures_hierarchical(&packed);

        // Detailed search (L2) - High precision
        for (i, &sig) in sigs_l2.iter().enumerate() {
        if i < self.lsh_tables_l2.len() {
            if let Some(blocks) = self.lsh_tables_l2[i].get(&sig) {
                for &bid in blocks { candidate_blocks.insert(bid); }
            }
            }
        }

        // Broad search (L1) - High recall (only if candidates are sparse)
        if candidate_blocks.len() < 5 {
            for (i, &sig) in sigs_l1.iter().enumerate() {
                if i < self.lsh_tables.len() {
                    if let Some(blocks) = self.lsh_tables[i].get(&sig) {
                        for &bid in blocks { candidate_blocks.insert(bid); }
                    }
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

        // Optimized Sparse Retrieval: Map-reduce triggered blocks without O(N) loop if possible.
        // For now, we utilize the fact that previous_spikes is often sparse.
        let mut triggered_blocks = std::collections::HashSet::new();

        // Optimized: trigger blocks based on individual neuron firing using cached mapping.
        for (i, &fired) in previous_spikes.iter().enumerate() {
            if fired {
                if i < self.neuron_to_block.len() {
                    triggered_blocks.insert(self.neuron_to_block[i]);
                } else {
                    // Fallback heuristic if mapping is missing
                    triggered_blocks.insert((i / 16) as u32);
                }
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
                            // Scale weight for impact. Boosted to 100x to ensure it's detectable and above thresholds.
                            let weight = if a.weight > 0 { (a.weight as i32) * 100 } else { 0 };
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
             for event in self.l3_buffer.clone() {
                 let replay_hash = event.context_hash;

                 // Strengthen associations for blocks triggered by this hash
                 let blocks = self.context_hashes.get(&replay_hash).cloned();
                 if let Some(blocks) = blocks {
                     for &bid in &blocks {
                         let mut entries = self.get_block_mut(bid);
                         for a in &mut entries {
                             a.weight = a.weight.saturating_add(5); // Consolidation boost
                         }
                         self.update_block(bid, entries);
                     }
                 }

                 // Boost associations within the replayed active blocks
                 for &bid in &event.active_blocks {
                     let mut entries = self.get_block_mut(bid);
                     for a in &mut entries {
                         a.weight = a.weight.saturating_add(1); // Small auxiliary boost
                     }
                     self.update_block(bid, entries);
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
        self.associations_flat.shrink_to_fit();
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
        let mut titan = BitWiseTitan::new(100);
        titan.byte_memory = vec![0; 256];
        titan.memory_mapped_range = Some((0, 48));

        let bus = crate::InputBus::new(100);
        let mut spikes = vec![false; 100];

        // Write value 0xAA to address 5
        spikes[0] = true;
        spikes[2] = true;
        spikes[16 + 1] = true;
        spikes[16 + 3] = true;
        spikes[16 + 5] = true;
        spikes[16 + 7] = true;
        spikes[24] = true;

        titan.on_tick(&bus, &spikes, 0);
        assert_eq!(titan.byte_memory[5], 0xAA);

        // Read from address 5
        spikes[24] = false;
        bus.clear();
        titan.on_tick(&bus, &spikes, 1);

        let prox = bus.proximal();
        use std::sync::atomic::Ordering;
        assert!(prox[25 + 1].load(Ordering::Relaxed) > 0);
        assert!(prox[25 + 3].load(Ordering::Relaxed) > 0);
        assert!(prox[25 + 5].load(Ordering::Relaxed) > 0);
        assert!(prox[25 + 7].load(Ordering::Relaxed) > 0);
        assert_eq!(prox[25 + 0].load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_titan_deep_replay() {
        let mut titan = BitWiseTitan::new(100);
        let mut neurons = NeuronsSoA::new(100);
        for i in 0..100 { neurons.block_id[i] = (i / 10) as u32; }

        titan.ensure_capacity(9);
        titan.update_block(0, vec![Association { target: 50, weight: 5 }]);

        let h1 = crate::SpikeData::Sparse((0..10).collect());
        let h2 = crate::SpikeData::Sparse((50..60).collect());

        let history = vec![h2, h1];
        titan.learn_from_history(&history, 0, &neurons, 2000);

        assert!(!titan.l3_buffer.is_empty());
        let initial_weight = titan.get_block_mut(0).iter().find(|a| a.target == 50).unwrap().weight;

        let mut synapses = SynapsesSoA::with_capacity(0);
        titan.on_night_phase(&mut neurons, &mut synapses, None);

        let consolidated_weight = titan.get_block_mut(0).iter().find(|a| a.target == 50).unwrap().weight;
        assert!(consolidated_weight > initial_weight);
    }
}
