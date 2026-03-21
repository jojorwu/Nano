use crate::{IValue, SCALE, NanoModule, NeuronsSoA, SynapsesSoA};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SpikingCerebellumModule {
    pub delay_line: Vec<Vec<bool>>,
    pub max_delay: usize,
    pub mossy_fiber_indices: Vec<usize>,
    pub purkinje_indices: Vec<usize>,
}

impl SpikingCerebellumModule {
    pub fn new(mossy: Vec<usize>, purkinje: Vec<usize>) -> Self {
        Self {
            delay_line: Vec::new(),
            max_delay: 10,
            mossy_fiber_indices: mossy,
            purkinje_indices: purkinje,
        }
    }
}

impl NanoModule for SpikingCerebellumModule {
    fn name(&self) -> &str { "cerebellum" }

    fn on_tick(&mut self, neurons: &mut NeuronsSoA, previous_spikes: &[bool], _tick: u32) {
        // 1. Maintain delay line of mossy fiber activity
        let mut current_mossy = vec![false; self.mossy_fiber_indices.len()];
        for (i, &idx) in self.mossy_fiber_indices.iter().enumerate() {
            if idx < previous_spikes.len() {
                current_mossy[i] = previous_spikes[idx];
            }
        }
        self.delay_line.insert(0, current_mossy);
        if self.delay_line.len() > self.max_delay { self.delay_line.pop(); }

        // 2. Predictive feedback: delayed activity is injected into Apical compartments
        // simulating the role of the cerebellum in temporal coordination.
        if self.delay_line.len() >= self.max_delay {
            let delayed = &self.delay_line[self.max_delay - 1];
            for (i, &spiked) in delayed.iter().enumerate() {
                if spiked {
                    // Inject into corresponding Purkinje (output) neurons' apical dendrites
                    if i < self.purkinje_indices.len() {
                        let target = self.purkinje_indices[i];
                        if target < neurons.len() {
                            neurons.apical_potential[target] = neurons.apical_potential[target].saturating_add(SCALE / 2);
                        }
                    }
                }
            }
        }
    }

    fn on_update_weights(&mut self, _neurons: &mut NeuronsSoA, _previous_spikes: &[bool], _current_spikes: &[bool], _tick: u32, _reward: Option<IValue>) {}
    fn on_night_phase(&mut self, _synapses: &mut SynapsesSoA, _reward: Option<IValue>) {}

    fn box_clone(&self) -> Box<dyn NanoModule> { Box::new(self.clone()) }
    fn get_state(&self) -> Vec<u8> { bincode::serialize(self).unwrap_or_default() }
    fn set_state(&mut self, state: &[u8]) {
        if let Ok(new_self) = bincode::deserialize::<Self>(state) { *self = new_self; }
    }
}
