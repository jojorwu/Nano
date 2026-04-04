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
@group(0) @binding(6) var<storage, read> lsh_table_l1: array<u32>;
@group(0) @binding(7) var<storage, read> lsh_table_l2: array<u32>;
@group(0) @binding(8) var<storage, read> projection_matrices: array<u64>; // 8 matrices * 1024 words

// VSA Context Matching
fn vsa_context_similarity(a: u64, b: u64) -> u32 {
    // Hamming similarity on Binary Spatter Codes
    return 64u - bitCount(a ^ b);
}

// Hierarchical LSH Retrieval (L1/L2) using Sign-Random-Projection (TurboQuant)
fn lsh_match_hierarchical(bid: u32, current_hash: u64) -> u32 {
    // In a full implementation, we would compute Sign-LSH on the current bit-pattern.
    // For extreme performance in this tick, we use the pre-calculated pattern hash
    // to approximate the random projection result.
    // The parity is maintained because the host also uses this hash for LSH indexing.

    let sig1 = u32(current_hash & 0xFFFFFFFFu); // Simplification: in production, use projection_matrices
    let sig2 = u32(current_hash >> 32u);

    var score = 0u;
    if (lsh_table_l1[bid] == sig1) { score = score + 512u; }
    if (lsh_table_l2[bid] == sig2) { score = score + 512u; }

    return score;
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

        // Hierarchical LSH parity
        let lsh_relevance = lsh_match_hierarchical(bid, current_hash);
        if (lsh_relevance > 0u) { relevance = (relevance * lsh_relevance) >> 10; }

        let start = block_offsets[bid];
        let end = block_offsets[bid + 1u];

        for (var i = start; i < end; i = i + 1u) {
            let assoc = associations[i];
            let weighted = (assoc.weight * i32(relevance)) >> 10;
            atomicAdd(&distal_potentials[assoc.target], weighted);
        }
    }
}
