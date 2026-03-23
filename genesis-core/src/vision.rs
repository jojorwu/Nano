use crate::{IValue, SCALE, NanoModule, NeuronsSoA, SynapsesSoA};
use serde::{Serialize, Deserialize};
use rand::Rng;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct VisionModule {
    pub resolution: (u32, u32),
    pub input_buffer: Vec<u8>,
    pub poisson_mode: bool,
}

impl VisionModule {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            resolution: (width, height),
            input_buffer: Vec::new(),
            poisson_mode: true,
        }
    }

    pub fn set_input(&mut self, pixels: &[u8]) {
        self.input_buffer = pixels.to_vec();
    }
}

impl NanoModule for VisionModule {
    fn name(&self) -> &str { "vision" }
    fn tier(&self) -> u32 { 0 }

    fn handle_input(&mut self, input: &crate::ModuleInput) {
        if let crate::ModuleInput::Image(pixels) = input {
            self.set_input(pixels);
        }
    }

    fn on_init(&mut self, neurons: &mut NeuronsSoA) {
        if self.resolution.0 > 0 {
            for y in 0..self.resolution.1 {
                for x in 0..self.resolution.0 {
                    let i = (y * self.resolution.0 + x) as usize;
                    if i < neurons.len() {
                        neurons.x[i] = x as i16;
                        neurons.y[i] = y as i16;
                    }
                }
            }
        }
    }

    fn on_tick(&mut self, bus: &crate::InputBus, _previous_spikes: &[bool], _tick: u32) {
        if !self.input_buffer.is_empty() {
            let mut rng = rand::thread_rng();
            for (i, &p) in self.input_buffer.iter().enumerate() {
                if i < bus.proximal.len() {
                    let rate = (p as f32) / 255.0;

                    let spiked = if self.poisson_mode {
                        rng.gen::<f32>() < rate
                    } else {
                        true
                    };

                    if spiked {
                        let val = if self.poisson_mode { SCALE } else { (p as IValue * SCALE) / 255 };
                        bus.set_modality("vision", i, val);
                        crate::InputBus::atomic_saturating_add(&bus.proximal[i], val);
                    }
                }
            }
            // Image input is usually transient or managed by higher level loop
            // In poisson mode, we might want to keep the buffer for multiple ticks
            if !self.poisson_mode {
                self.input_buffer.clear();
            }
        }
    }

    fn on_update_weights(&mut self, _neurons: &mut NeuronsSoA, _previous_spikes: &[bool], _current_spikes: &[bool], _tick: u32, _reward: Option<IValue>) {}
    fn on_night_phase(&mut self, _neurons: &mut NeuronsSoA, _synapses: &mut SynapsesSoA, _reward: Option<IValue>) {}

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

    pub fn rate_encode(&self, pixels: &[u8]) -> Vec<IValue> {
        let mut potentials = Vec::with_capacity(pixels.len());
        for &p in pixels {
            let val = (p as IValue * SCALE) / 255;
            potentials.push(val);
        }
        potentials
    }

    pub fn get_neuron_index(&self, x: u32, y: u32, offset: usize) -> usize {
        offset + (y * self.resolution.0 + x) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vision_poisson() {
        let mut vision = VisionModule::new(10, 10);
        vision.set_input(&[255; 100]); // Max intensity
        let mut bus = crate::InputBus::new(100);

        vision.on_tick(&mut bus, &[], 1);

        // With intensity 255, Poisson should almost always spike (rate=1.0)
        use std::sync::atomic::Ordering;
        let total_potential: i32 = bus.proximal.iter().map(|v| v.load(Ordering::Relaxed)).sum();
        assert!(total_potential > 0);
    }
}
