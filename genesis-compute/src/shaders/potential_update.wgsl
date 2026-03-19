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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&potentials)) { return; }

    if (refractory[i] > 0) {
        refractory[i] = refractory[i] - 1;
        potentials[i] = 0;
        spikes[i] = 0;
        return;
    }

    var pot = potentials[i] + inputs[i];
    let dec = decays[i];
    pot = (pot * (1000 - dec)) / 1000;

    if (pot >= thresholds[i]) {
        potentials[i] = 0;
        refractory[i] = 2;
        spikes[i] = 1;
    } else {
        potentials[i] = pot;
        spikes[i] = 0;
    }
}
