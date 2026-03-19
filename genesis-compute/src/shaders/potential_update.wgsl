struct Neuron {
    potential: i32,
    threshold: i32,
    decay: i32,
    refractory: i32,
}

@group(0) @binding(0) var<storage, read_write> potentials: array<i32>;
@group(0) @binding(1) var<storage, read_write> thresholds: array<i32>;
@group(0) @binding(2) var<storage, read_write> decays: array<i32>;
@group(0) @binding(3) var<storage, read_write> refractory: array<i32>;
@group(0) @binding(4) var<storage, read_write> spikes: array<u32>;
@group(0) @binding(5) var<storage, read> inputs: array<i32>;
@group(0) @binding(6) var<storage, read_write> next_update: array<u32>;
@group(0) @binding(7) var<storage, read> intervals: array<u32>;
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

    var pot = potentials[i] + inputs[i];

    // LLIF: Liquid Decay
    let base_decay = decays[i];
    let liquid_mod = (abs(inputs[i]) * 10) / 1000;
    let final_decay = max(1, base_decay - liquid_mod);

    pot = (pot * (1000 - final_decay)) / 1000;

    if (pot >= thresholds[i]) {
        potentials[i] = 0;
        refractory[i] = 2;
        spikes[i] = 1;
        // Intrinsic Plasticity
        thresholds[i] = thresholds[i] + 50;
    } else {
        potentials[i] = pot;
        spikes[i] = 0;
        // Threshold decay back to base
        // (Requires base_thresholds binding)
    }

    next_update[i] = current_tick + intervals[i];
}
