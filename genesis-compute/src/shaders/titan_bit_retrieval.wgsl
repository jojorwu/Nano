struct Association {
    target: u32,
    weight: i32,
}

@group(0) @binding(0) var<storage, read> associations: array<Association>;
@group(0) @binding(1) var<storage, read> block_offsets: array<u32>;
@group(0) @binding(2) var<storage, read> previous_spikes: array<u32>; // Bitmask u32
@group(0) @binding(3) var<storage, read> block_id: array<u32>;
@group(0) @binding(4) var<storage, read_write> distal_potentials: array<atomic<i32>>;
@group(0) @binding(5) var<storage, read> context_hashes: array<u64>; // Archive of past contextual hashes

// VSA Context Matching
fn vsa_context_similarity(a: u64, b: u64) -> u32 {
    // Hamming similarity on Binary Spatter Codes
    return 64u - bitCount(a ^ b);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let neuron_idx = id.x;
    let n_count = arrayLength(&block_id);
    if (neuron_idx >= n_count) { return; }

    let bid = block_id[neuron_idx];
    let word_idx = neuron_idx / 32u;
    let bit_idx = neuron_idx % 32u;
    let fired = (previous_spikes[word_idx] >> bit_idx) & 1u;

    if (fired != 0u) {
        // Bit-Parallel Context Search
        let current_hash = context_hashes[0]; // Simplified
        var relevance = 1024u;

        // If context hashes are available, match against them
        if (arrayLength(&context_hashes) > 1u) {
             let best_match = context_hashes[1];
             let sim = vsa_context_similarity(current_hash, best_match);
             if (sim < 32u) { relevance = 512u; }
        }

        let start = block_offsets[bid];
        let end = block_offsets[bid + 1u];

        for (var i = start; i < end; i = i + 1u) {
            let assoc = associations[i];
            let weighted = (assoc.weight * i32(relevance)) >> 10;
            atomicAdd(&distal_potentials[assoc.target], weighted);
        }
    }
}
