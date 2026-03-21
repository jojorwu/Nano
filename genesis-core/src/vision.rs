use crate::{IValue, SCALE, NanoModule, NeuronsSoA, SynapsesSoA};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct VisionModule {
    pub resolution: (u32, u32),
    pub input_buffer: Vec<u8>,
}

impl VisionModule {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            resolution: (width, height),
            input_buffer: Vec::new(),
        }
    }

    pub fn set_input(&mut self, pixels: &[u8]) {
        self.input_buffer = pixels.to_vec();
    }
}

impl NanoModule for VisionModule {
    fn name(&self) -> &str { "vision" }

    fn on_tick(&mut self, neurons: &mut NeuronsSoA, _previous_spikes: &[bool], _tick: u32) {
        if !self.input_buffer.is_empty() {
            // Convert pixel values (0-255) to spike potentials.
            for (i, &p) in self.input_buffer.iter().enumerate() {
                if i < neurons.len() {
                    let val = (p as IValue * SCALE) / 255;
                    neurons.proximal_potential[i] = neurons.proximal_potential[i].saturating_add(val);
                }
            }
            // Image input is usually transient or managed by higher level loop
            self.input_buffer.clear();
        }
    }

    fn on_update_weights(&mut self, _neurons: &mut NeuronsSoA, _previous_spikes: &[bool], _current_spikes: &[bool], _tick: u32, _reward: Option<IValue>) {}
    fn on_night_phase(&mut self, _synapses: &mut SynapsesSoA, _reward: Option<IValue>) {}

    fn box_clone(&self) -> Box<dyn NanoModule> {
        Box::new(self.clone())
    }

    fn get_state(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap_or_default()
    }

    fn set_state(&mut self, state: &[u8]) {
        if let Ok(new_self) = bincode::deserialize::<Self>(state) {
            *self = new_self;
        }
    }
}

pub struct SpikingVisionModule {
    pub resolution: (u32, u32),
}

impl SpikingVisionModule {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            resolution: (width, height),
        }
    }

    /// Convert pixel values (0-255) to spike potentials.
    /// Higher intensity results in higher initial potential.
    pub fn rate_encode(&self, pixels: &[u8]) -> Vec<IValue> {
        let mut potentials = Vec::with_capacity(pixels.len());
        for &p in pixels {
            // Map 0..255 to 0..SCALE (1.0)
            let val = (p as IValue * SCALE) / 255;
            potentials.push(val);
        }
        potentials
    }

    /// Map 2D pixel index (x, y) to a 1D neuron index.
    pub fn get_neuron_index(&self, x: u32, y: u32, offset: usize) -> usize {
        offset + (y * self.resolution.0 + x) as usize
    }
}
