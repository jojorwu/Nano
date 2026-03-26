use crate::{NanoModule, NeuronsSoA, SynapsesSoA, InputBus, IValue};
use serde::{Serialize, Deserialize};

/// LoadBalancer Module: Dynamically manages block-to-hardware mapping based on
/// block activity and specialization.
/// It monitors computational load and moves "hot" blocks to the high-performance
/// backend (e.g. GPU) while keeping "cold" or "sparse" blocks on the CPU.

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LoadBalancerModule {
    pub block_activity: Vec<f32>,
    pub gpu_threshold: f32,
    pub rebalance_interval: u32,
    pub last_rebalance_tick: u32,
    /// Mapping of block_id to preferred device (0 = GPU, 1 = CPU)
    pub preferred_device: Vec<u8>,
}

impl LoadBalancerModule {
    pub fn new(num_blocks: usize) -> Self {
        Self {
            block_activity: vec![0.0; num_blocks],
            gpu_threshold: 0.1, // Move to GPU if activity > 10%
            rebalance_interval: 100,
            last_rebalance_tick: 0,
            preferred_device: vec![0; num_blocks], // Default to GPU
        }
    }

    pub fn rebalance(&mut self, neurons: &mut NeuronsSoA) -> usize {
        let mut changes = 0;
        for (bid, &activity) in self.block_activity.iter().enumerate() {
            let new_device = if activity > self.gpu_threshold { 0 } else { 1 };
            if new_device != self.preferred_device[bid] {
                self.preferred_device[bid] = new_device;
                changes += 1;
            }
        }

        if changes > 0 {
            // Apply new mapping to neurons
            for i in 0..neurons.len() {
                let bid = neurons.block_id[i] as usize;
                if bid < self.preferred_device.len() {
                    // Update is_remote: 1 means this neuron should be handled by the secondary backend
                    neurons.is_remote[i] = self.preferred_device[bid];
                }
            }
        }
        changes
    }
}

impl NanoModule for LoadBalancerModule {
    fn name(&self) -> &str { "load_balancer" }

    fn on_tick(&mut self, _bus: &InputBus, _previous_spikes: &[bool], tick: u32) {
        self.last_rebalance_tick = tick;
    }

    fn on_update_weights(&mut self, neurons: &mut NeuronsSoA, _previous_spikes: &[bool], current_spikes: &[bool], tick: u32, _surprise: Option<IValue>) {
        let n_count = neurons.len();
        if n_count == 0 { return; }

        // Update block activity moving average
        let mut counts = vec![0u32; self.block_activity.len()];
        let mut block_sizes = vec![0u32; self.block_activity.len()];

        for i in 0..n_count {
            let bid = neurons.block_id[i] as usize;
            if bid < self.block_activity.len() {
                block_sizes[bid] += 1;
                if current_spikes[i] { counts[bid] += 1; }
            }
        }

        for bid in 0..self.block_activity.len() {
            if block_sizes[bid] > 0 {
                let current_activity = counts[bid] as f32 / block_sizes[bid] as f32;
                self.block_activity[bid] = self.block_activity[bid] * 0.95 + current_activity * 0.05;
            }
        }

        if tick > self.last_rebalance_tick + self.rebalance_interval {
            self.rebalance(neurons);
            self.last_rebalance_tick = tick;
        }
    }

    fn on_night_phase(&mut self, _neurons: &mut NeuronsSoA, _synapses: &mut SynapsesSoA, _reward: Option<IValue>) {
        // Long-term rebalancing logic
    }

    fn box_clone(&self) -> Box<dyn NanoModule> { Box::new(self.clone()) }
    fn get_state(&self) -> Vec<u8> { bincode::serialize(self).unwrap_or_default() }
    fn set_state(&mut self, state: &[u8]) {
        if let Ok(new_self) = bincode::deserialize::<Self>(state) { *self = new_self; }
    }
}
