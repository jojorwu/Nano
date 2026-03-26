use crate::{IValue, SCALE, NanoModule, bus::InputBus, model::{NeuronsSoA, SynapsesSoA}};
use serde::{Serialize, Deserialize};

/// Global Neuronal Workspace (GNW) Implementation
/// Based on Dehaene's GNW theory: "Competition, Ignition, and Broadcast".
/// This module selects the most active/salient neural block and broadcasts its
/// pattern across the entire network, enabling a form of "conscious" attention.

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct WorkspaceModule {
    /// Currently 'ignited' block ID
    pub active_block: Option<u32>,
    /// Persistence of ignition (ticks remaining)
    pub ignition_timer: u32,
    /// Last activity per block for competition
    pub block_activity: Vec<f32>,
    /// Broadcasting strength
    pub broadcast_intensity: IValue,
    /// Minimum activity to trigger ignition
    pub ignition_threshold: f32,
    /// Per-block lists of neuron indices for efficient broadcasting
    pub block_to_neurons: Vec<Vec<u32>>,
}

impl WorkspaceModule {
    pub fn new() -> Self {
        Self {
            active_block: None,
            ignition_timer: 0,
            block_activity: Vec::new(),
            broadcast_intensity: SCALE / 2, // 0.5 intensity
            ignition_threshold: 10.0,
            block_to_neurons: Vec::new(),
        }
    }
}

impl NanoModule for WorkspaceModule {
    fn name(&self) -> &str { "workspace" }
    fn tier(&self) -> u32 { 10 } // High tier: runs after sensory processing
    fn outputs(&self) -> Vec<String> { vec!["broadcast".to_string()] }
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }

    fn on_init(&mut self, neurons: &mut NeuronsSoA) -> Result<(), crate::module::ModuleError> {
        let max_bid = neurons.block_id.iter().max().copied().unwrap_or(0) as usize;
        self.block_to_neurons = vec![Vec::new(); max_bid + 1];
        self.block_activity = vec![0.0; max_bid + 1];

        for (i, &bid) in neurons.block_id.iter().enumerate() {
            if (bid as usize) < self.block_to_neurons.len() {
                self.block_to_neurons[bid as usize].push(i as u32);
            }
        }
        Ok(())
    }

    fn on_tick(&mut self, bus: &InputBus, _previous_spikes: &[bool], _tick: u32) {
        // GNW Ignition Competition: selects winner based on activity and surprise
        let surprise = bus.global_signals[1].load(std::sync::atomic::Ordering::Relaxed);

        if self.ignition_timer > 0 {
            self.ignition_timer -= 1;

            // 2. Broadcast: Inject active block's pattern (Targeted neurons only)
            if let Some(bid) = self.active_block {
                if (bid as usize) < self.block_to_neurons.len() {
                    let neurons = &self.block_to_neurons[bid as usize];
                    let prox = bus.proximal();
                    for &idx in neurons {
                        if (idx as usize) < prox.len() {
                            InputBus::atomic_saturating_add(&prox[idx as usize], self.broadcast_intensity);
                        }
                    }
                }
            }
        } else {
            self.active_block = None;

            // Find new winner
            let mut winner = None;
            let mut max_score = self.ignition_threshold;

            for (bid, &act) in self.block_activity.iter().enumerate() {
                // Heuristic: surprise-driven ignition
                // Active blocks that coincide with global surprise get a boost in the competition.
                let surprise_boost = if surprise > 500 { act * 0.5 } else { 0.0 };
                let score = act + surprise_boost;

                if score > max_score {
                    max_score = score;
                    winner = Some(bid as u32);
                }
            }

            if let Some(bid) = winner {
                log::info!("GNW Ignition: Block {} won competition", bid);
                self.active_block = Some(bid);
                self.ignition_timer = 5; // Sustain for 5 ticks
            }
        }

        // Decay block activity
        for act in &mut self.block_activity {
            *act *= 0.9;
        }
    }

    fn on_update_weights(&mut self, neurons: &mut NeuronsSoA, _previous_spikes: &[bool], current_spikes: &[bool], _tick: u32, _reward: Option<IValue>) {
        // Monitor current activity to update competition state
        if self.block_activity.is_empty() { return; }
        for i in 0..neurons.len() {
            if current_spikes[i] {
                let bid = neurons.block_id[i] as usize;
                if bid < self.block_activity.len() {
                    self.block_activity[bid] += 1.0;
                }
            }
        }
    }

    fn on_night_phase(&mut self, _neurons: &mut NeuronsSoA, _synapses: &mut SynapsesSoA, _reward: Option<IValue>) {
        self.block_activity.clear();
    }

    fn box_clone(&self) -> Box<dyn NanoModule> { Box::new(self.clone()) }
    fn get_state(&self) -> Vec<u8> { bincode::serialize(self).unwrap_or_default() }
    fn set_state(&mut self, state: &[u8]) {
        if let Ok(new_self) = bincode::deserialize::<Self>(state) { *self = new_self; }
    }

    fn on_config_sync(&mut self, config: &crate::config::NetworkConfig) {
        self.broadcast_intensity = config.modules.workspace_broadcast_intensity;
        self.ignition_threshold = config.modules.workspace_ignition_threshold;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::NeuronsSoA;
    use crate::bus::InputBus;

    #[test]
    fn test_workspace_ignition() {
        let mut workspace = WorkspaceModule::new();
        let mut neurons = NeuronsSoA::new(32);
        for i in 0..32 { neurons.block_id[i] = (i / 16) as u32; }

        workspace.on_init(&mut neurons).unwrap();
        let bus = InputBus::new(32);

        // Simulate activity in block 1
        let mut spikes = vec![false; 32];
        for i in 16..32 { spikes[i] = true; }

        workspace.on_update_weights(&mut neurons, &[], &spikes, 0, None);

        // Tick to trigger ignition
        workspace.on_tick(&bus, &spikes, 1);

        assert_eq!(workspace.active_block, Some(1));
        assert!(workspace.ignition_timer > 0);
    }
}
