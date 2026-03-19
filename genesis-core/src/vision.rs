use crate::{IValue, SCALE};

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
