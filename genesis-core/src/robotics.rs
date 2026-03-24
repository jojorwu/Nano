use crate::{IValue, SCALE, NanoModule, NeuronsSoA, SynapsesSoA};
use serde::{Serialize, Deserialize};
use std::collections::VecDeque;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SpikingCerebellumModule {
    pub delay_line: VecDeque<Vec<bool>>,
    pub max_delay: usize,
    pub mossy_fiber_indices: Vec<usize>,
    pub purkinje_indices: Vec<usize>,
}

impl SpikingCerebellumModule {
    pub fn new(mossy: Vec<usize>, purkinje: Vec<usize>) -> Self {
        Self {
            delay_line: VecDeque::new(),
            max_delay: 10,
            mossy_fiber_indices: mossy,
            purkinje_indices: purkinje,
        }
    }
}

impl NanoModule for SpikingCerebellumModule {
    fn name(&self) -> &str { "cerebellum" }
    fn outputs(&self) -> Vec<String> { vec!["apical".to_string()] }
    fn inputs(&self) -> Vec<String> { vec!["proximal".to_string()] }

    fn on_tick(&mut self, bus: &crate::InputBus, previous_spikes: &[bool], _tick: u32) {
        // 1. Maintain delay line of mossy fiber activity
        let mut current_mossy = vec![false; self.mossy_fiber_indices.len()];
        for (i, &idx) in self.mossy_fiber_indices.iter().enumerate() {
            if idx < previous_spikes.len() {
                current_mossy[i] = previous_spikes[idx];
            }
        }
        self.delay_line.push_front(current_mossy);
        if self.delay_line.len() > self.max_delay { self.delay_line.pop_back(); }

        // 2. Predictive feedback: delayed activity is injected into Apical compartments
        // simulating the role of the cerebellum in temporal coordination.
        if self.delay_line.len() >= self.max_delay {
            let delayed = &self.delay_line[self.max_delay - 1];
            for (i, &spiked) in delayed.iter().enumerate() {
                if spiked {
                    // Inject into corresponding Purkinje (output) neurons' apical dendrites
                    if i < self.purkinje_indices.len() {
                        let target = self.purkinje_indices[i];
                        if target < bus.apical.len() {
                            crate::InputBus::atomic_saturating_add(&bus.apical[target], SCALE / 2);
                        }
                    }
                }
            }
        }
    }

    fn on_update_weights(&mut self, _neurons: &mut NeuronsSoA, _previous_spikes: &[bool], _current_spikes: &[bool], _tick: u32, _reward: Option<IValue>) {}
    fn on_night_phase(&mut self, _neurons: &mut NeuronsSoA, _synapses: &mut SynapsesSoA, _reward: Option<IValue>) {}

    fn box_clone(&self) -> Box<dyn NanoModule> { Box::new(self.clone()) }
    fn get_state(&self) -> Vec<u8> { bincode::serialize(self).unwrap_or_default() }
    fn set_state(&mut self, state: &[u8]) {
        if let Ok(new_self) = bincode::deserialize::<Self>(state) { *self = new_self; }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RobotControlModule {
    pub motor_neuron_indices: Vec<usize>,
    pub spike_counters: Vec<u32>,
    pub window_size: u32,
    pub current_motor_outputs: Vec<f32>,
}

impl RobotControlModule {
    pub fn new(neurons: Vec<usize>) -> Self {
        let count = neurons.len();
        Self {
            motor_neuron_indices: neurons,
            spike_counters: vec![0; count],
            window_size: 10,
            current_motor_outputs: vec![0.0; count],
        }
    }

    pub fn decode_motor_outputs(&mut self) -> Vec<f32> {
        for (i, counter) in self.spike_counters.iter_mut().enumerate() {
            // Rate encoding: density of spikes over the window
            self.current_motor_outputs[i] = (*counter as f32) / (self.window_size as f32);
            *counter = 0; // Reset for next window
        }
        self.current_motor_outputs.clone()
    }
}

impl NanoModule for RobotControlModule {
    fn name(&self) -> &str { "robot_control" }
    fn outputs(&self) -> Vec<String> { vec!["proximal".to_string()] }
    fn inputs(&self) -> Vec<String> { vec!["proximal".to_string()] }

    fn on_tick(&mut self, bus: &crate::InputBus, previous_spikes: &[bool], _tick: u32) {
        // Intrinsic Motivation (Surprise-driven exploration):
        // If the network is stagnant (low activity/surprise), inject exploratory noise
        // specifically into motor-assigned neurons to trigger trial-and-error behavior.
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Heuristic: Inject noise if total activity is low (derived from previous spikes)
        let total_active = previous_spikes.iter().filter(|&&s| s).count();
        if total_active < self.motor_neuron_indices.len() / 2 {
            for &idx in &self.motor_neuron_indices {
                if idx < bus.proximal.len() {
                    // Stochastic boost to proximal potential
                    if rng.gen::<f32>() < 0.1 {
                        crate::InputBus::atomic_saturating_add(&bus.proximal[idx], 500);
                    }
                }
            }
        }

        for (i, &idx) in self.motor_neuron_indices.iter().enumerate() {
            if idx < previous_spikes.len() && previous_spikes[idx] {
                self.spike_counters[i] += 1;
            }
        }
    }

    fn on_update_weights(&mut self, _neurons: &mut NeuronsSoA, _previous_spikes: &[bool], _current_spikes: &[bool], _tick: u32, _reward: Option<IValue>) {}
    fn on_night_phase(&mut self, _neurons: &mut NeuronsSoA, _synapses: &mut SynapsesSoA, _reward: Option<IValue>) {}

    fn box_clone(&self) -> Box<dyn NanoModule> { Box::new(self.clone()) }
    fn get_state(&self) -> Vec<u8> { bincode::serialize(self).unwrap_or_default() }
    fn set_state(&mut self, state: &[u8]) {
        if let Ok(new_self) = bincode::deserialize::<Self>(state) { *self = new_self; }
    }
}
