struct Neuron {
    potential: i32,
    threshold: i32,
    decay: i32,
    refractory: i32,
}

@group(0) @binding(0) var<storage, read_write> potentials: array<i32>;
@group(0) @binding(9) var<storage, read> distal_potentials: array<i32>;
@group(0) @binding(10) var<storage, read> proximal_potentials: array<i32>;
@group(0) @binding(1) var<storage, read_write> thresholds: array<i32>;
@group(0) @binding(11) var<storage, read> base_thresholds: array<i32>;
@group(0) @binding(14) var<storage, read> layer_ids: array<u32>;
@group(0) @binding(15) var<storage, read_write> backprop_signals: array<i32>;
struct Config {
    ip_increment: i32,
    ip_decay: i32,
}
@group(0) @binding(16) var<storage, read> config: Config;
@group(0) @binding(2) var<storage, read_write> decays: array<i32>;
@group(0) @binding(3) var<storage, read_write> refractory: array<i32>;
@group(0) @binding(4) var<storage, read_write> spikes: array<u32>;
@group(0) @binding(12) var<storage, read_write> sparse_spikes: array<u32>;
@group(0) @binding(13) var<storage, read_write> spike_counter: atomic<u32>;
@group(0) @binding(5) var<storage, read> inputs: array<i32>;
@group(0) @binding(6) var<storage, read_write> next_update: array<u32>;
@group(0) @binding(7) var<storage, read> intervals: array<u32>;
@group(0) @binding(8) var<storage, read> dendritic_gate: array<i32>;
@group(1) @binding(0) var<uniform> current_tick: u32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&potentials)) { return; }

    if (current_tick < next_update[i]) { return; }

    if (refractory[i] > 0) {
        refractory[i] = refractory[i] - 1;
        potentials[i] = 0;
        spikes[i] = 0;
        next_update[i] = current_tick + intervals[i];
        return;
    }

    let proximal = proximal_potentials[i];
    let distal = distal_potentials[i];

    var dend_factor = distal / 4;
    if (proximal > 500) { dend_factor = distal; }

    let gated_input = (inputs[i] * dendritic_gate[i]) / 1024;
    var pot = potentials[i] + gated_input + proximal + dend_factor;

    // LLIF: Liquid Decay
    let base_decay = decays[i];
    let liquid_mod = (abs(inputs[i]) * 10) / 1024;
    let final_decay = max(1, base_decay - liquid_mod);

    pot = (pot * (1024 - final_decay)) / 1024;

    if (pot >= thresholds[i]) {
        potentials[i] = 0;
        refractory[i] = 2;
        spikes[i] = 1;

        // SMBP: Active backpropagation signal
        backprop_signals[i] = 1024;

        let count = atomicAdd(&spike_counter, 1u);
        if (count < arrayLength(&sparse_spikes)) {
            // Simple index storage for now, bit-packing can be done in a separate kernel if needed.
            sparse_spikes[count] = i;
        }

        // Intrinsic Plasticity
        thresholds[i] = thresholds[i] + config.ip_increment;
    } else {
        potentials[i] = pot;
        spikes[i] = 0;
        // Threshold decay back to base (Intrinsic Plasticity)
        if (thresholds[i] > base_thresholds[i]) {
            thresholds[i] = thresholds[i] - config.ip_decay;
        }
        // Decaying backprop signal
        backprop_signals[i] = (backprop_signals[i] * 800) / 1024;
    }

    next_update[i] = current_tick + intervals[i];
}
