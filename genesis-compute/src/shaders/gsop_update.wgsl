@group(0) @binding(0) var<storage, read_write> weights: array<i32>;
@group(0) @binding(1) var<storage, read> source_indices: array<u32>;
@group(0) @binding(2) var<storage, read> target_indices: array<u32>;
@group(0) @binding(3) var<storage, read> pre_spikes: array<u32>;
@group(0) @binding(4) var<storage, read> post_spikes: array<u32>;
@group(0) @binding(5) var<storage, read> compartments: array<u32>;
@group(0) @binding(6) var<storage, read> backprop_signals: array<i32>;
@group(0) @binding(7) var<storage, read> last_spike_ticks: array<u32>;
@group(0) @binding(8) var<storage, read> is_excitatory: array<u32>;
@group(1) @binding(0) var<uniform> base_learning_rate: i32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= arrayLength(&weights)) { return; }

    let src = source_indices[idx];
    let tgt = target_indices[idx];
    let comp = compartments[idx];

    let pre_spiked = pre_spikes[src] != 0u;
    let post_spiked = post_spikes[tgt] != 0u;

    var lr = base_learning_rate;
    if (comp == 0u) { // Proximal
        lr = base_learning_rate;
    } else if (comp == 1u) { // Distal
        lr = (base_learning_rate * 8) / 10;
    } else {
        lr = base_learning_rate / 2;
    }

    // SMBP Modulation: backprop signal amplifies learning in non-proximal compartments
    if (comp != 0u) {
        lr = (lr * (1024 + backprop_signals[tgt])) >> 10;
    }

    var w = weights[idx];

    let old_w = w;
    if (pre_spiked && post_spiked) {
        w = w + lr;
    } else if (pre_spiked && !post_spiked) {
        w = w - (lr >> 1);
    }

    // Temporal STDP Component (using last_spike_ticks)
    let pre_last = last_spike_ticks[src];
    let post_last = last_spike_ticks[tgt];
    if (pre_last > 0u && post_last > 0u) {
        let diff = i32(post_last) - i32(pre_last);
        if (diff > 0 && diff < 20) {
            w = w + (lr >> 2); // LTP boost
        } else if (diff < 0 && diff > -20) {
            w = w - (lr >> 2); // LTD penalty
        }
    }

    // Sign Preservation (Dale's Law)
    if (old_w > 0 && w < 0) { w = 1; }
    if (old_w < 0 && w > 0) { w = -1; }

    // Clamp weight (-5120 to 5120, assuming SCALE=1024)
    if (w > 5120) { w = 5120; }
    if (w < -5120) { w = -5120; }

    weights[idx] = w;
}
