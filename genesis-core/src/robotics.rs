use crate::{IValue, SCALE, NanoModule, NeuronsSoA, SynapsesSoA};
use serde::{Serialize, Deserialize};
use std::collections::VecDeque;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SpikingCerebellumModule {
    pub delay_line: VecDeque<Vec<bool>>,
    pub max_delay: usize,
    pub mossy_fiber_indices: Vec<usize>,
    pub purkinje_indices: Vec<usize>,
    /// Internal weight matrix: Mossy x Purkinje
    #[serde(default)]
    pub weights: Vec<Vec<IValue>>,
}

impl SpikingCerebellumModule {
    pub fn new(mossy: Vec<usize>, purkinje: Vec<usize>) -> Self {
        let m_count = mossy.len();
        let p_count = purkinje.len();
        Self {
            delay_line: VecDeque::new(),
            max_delay: 10,
            mossy_fiber_indices: mossy,
            purkinje_indices: purkinje,
            weights: vec![vec![SCALE / 2; p_count]; m_count],
        }
    }
}

impl NanoModule for SpikingCerebellumModule {
    fn name(&self) -> &str { "cerebellum" }
    fn outputs(&self) -> Vec<String> { vec!["apical".to_string()] }
    fn inputs(&self) -> Vec<String> { vec!["proximal".to_string()] }
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }

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
            let apical = bus.apical();

            for (m_idx, &spiked) in delayed.iter().enumerate() {
                if spiked {
                    for (p_idx, &target) in self.purkinje_indices.iter().enumerate() {
                        if target < apical.len() {
                            let w = self.weights[m_idx][p_idx];
                            crate::InputBus::atomic_saturating_add(&apical[target], w);
                        }
                    }
                }
            }
        }
    }

    fn on_update_weights(&mut self, _neurons: &mut NeuronsSoA, _previous_spikes: &[bool], current_spikes: &[bool], _tick: u32, reward: Option<IValue>) {
        // Cerebellar Plasticity (LTD-driven learning)
        // If Purkinje cells fire coincident with high surprise/error, weaken the active mossy connections.
        let r_val = reward.unwrap_or(0);
        if r_val < -100 && self.delay_line.len() >= self.max_delay {
             let delayed = &self.delay_line[self.max_delay - 1];
             for (m_idx, &m_spiked) in delayed.iter().enumerate() {
                 if m_spiked {
                     for (p_idx, &p_target) in self.purkinje_indices.iter().enumerate() {
                         if p_target < current_spikes.len() && current_spikes[p_target] {
                             // Cerebellar LTD: reduce weight for connections that cause error
                             self.weights[m_idx][p_idx] = (self.weights[m_idx][p_idx] - 5).max(0);
                         }
                     }
                 }
             }
        } else if r_val > 100 && self.delay_line.len() >= self.max_delay {
             // LTP-like recovery
             let delayed = &self.delay_line[self.max_delay - 1];
             for (m_idx, &m_spiked) in delayed.iter().enumerate() {
                 if m_spiked {
                     for (p_idx, &p_target) in self.purkinje_indices.iter().enumerate() {
                         if p_target < current_spikes.len() && current_spikes[p_target] {
                             self.weights[m_idx][p_idx] = (self.weights[m_idx][p_idx] + 2).min(SCALE);
                         }
                     }
                 }
             }
        }
    }
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
    /// Individual gains for motor neurons, adjusted by reward
    #[serde(default)]
    pub motor_gains: Vec<IValue>,
    /// Tracks recent activity of motor neurons for reward assignment
    #[serde(default)]
    pub activity_trace: Vec<f32>,
}

impl RobotControlModule {
    pub fn new(neurons: Vec<usize>) -> Self {
        let count = neurons.len();
        Self {
            motor_neuron_indices: neurons,
            spike_counters: vec![0; count],
            window_size: 10,
            current_motor_outputs: vec![0.0; count],
            motor_gains: vec![SCALE; count],
            activity_trace: vec![0.0; count],
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
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }

    fn on_tick(&mut self, bus: &crate::InputBus, previous_spikes: &[bool], _tick: u32) {
        // Intrinsic Motivation (Surprise-driven exploration):
        // If the network is stagnant (low activity/surprise), inject exploratory noise
        // specifically into motor-assigned neurons to trigger trial-and-error behavior.
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let prox = bus.proximal();
        // Heuristic: Inject noise if total activity is low (derived from previous spikes)
        let total_active = previous_spikes.iter().filter(|&&s| s).count();
        if total_active < self.motor_neuron_indices.len() / 2 {
            for (i, &idx) in self.motor_neuron_indices.iter().enumerate() {
                if idx < prox.len() {
                    // Stochastic boost to proximal potential, scaled by current gain
                    if rng.gen::<f32>() < 0.1 {
                        let gain_boost = (500i64 * self.motor_gains[i] as i64 >> 10) as i32;
                        crate::InputBus::atomic_saturating_add(&prox[idx], gain_boost);
                    }
                }
            }
        }

        for (i, &idx) in self.motor_neuron_indices.iter().enumerate() {
            if idx < previous_spikes.len() && previous_spikes[idx] {
                self.spike_counters[i] += 1;
                self.activity_trace[i] = self.activity_trace[i] * 0.9 + 0.1;
            } else {
                self.activity_trace[i] *= 0.95;
            }
        }
    }

    fn on_update_weights(&mut self, _neurons: &mut NeuronsSoA, _previous_spikes: &[bool], _current_spikes: &[bool], _tick: u32, _reward: Option<IValue>) {}
    fn on_night_phase(&mut self, _neurons: &mut NeuronsSoA, _synapses: &mut SynapsesSoA, reward: Option<IValue>) {
        if let Some(r) = reward {
             // Reward-Modulated Gain: boost gain for motor neurons that were active
             // during high positive reward, or reduce it if reward was negative.
             for (i, &trace) in self.activity_trace.iter().enumerate() {
                 if trace > 0.05 {
                      let adjustment = (r as f32 * trace * 0.1) as i32;
                      self.motor_gains[i] = (self.motor_gains[i] + adjustment).clamp(SCALE / 4, SCALE * 4);
                 }
             }
        }
    }

    fn box_clone(&self) -> Box<dyn NanoModule> { Box::new(self.clone()) }
    fn get_state(&self) -> Vec<u8> { bincode::serialize(self).unwrap_or_default() }
    fn set_state(&mut self, state: &[u8]) {
        if let Ok(new_self) = bincode::deserialize::<Self>(state) { *self = new_self; }
    }
}
