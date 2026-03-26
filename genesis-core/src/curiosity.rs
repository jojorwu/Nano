use crate::{IValue, NanoModule, bus::InputBus, model::{NeuronsSoA, SynapsesSoA}};
use serde::{Serialize, Deserialize};

/// Curiosity Module: Intrinsic Motivation
/// Rewards the network for encountering "novel but learnable" patterns.
/// High surprise that is consistently decreasing indicates progress in learning,
/// which triggers an internal dopamine boost.

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CuriosityModule {
    pub surprise_history: Vec<i32>,
    pub curiosity_reward_strength: IValue,
    pub last_internal_reward: IValue,
}

impl CuriosityModule {
    pub fn new() -> Self {
        Self {
            surprise_history: Vec::new(),
            curiosity_reward_strength: 50,
            last_internal_reward: 0,
        }
    }
}

impl NanoModule for CuriosityModule {
    fn name(&self) -> &str { "curiosity" }
    fn tier(&self) -> u32 { 30 } // High tier

    fn on_tick(&mut self, _bus: &InputBus, _previous_spikes: &[bool], _tick: u32) {
        // Core curiosity logic handled in on_update_weights based on surprise
    }

    fn on_update_weights(&mut self, _neurons: &mut NeuronsSoA, _prev: &[bool], _current: &[bool], _tick: u32, surprise: Option<IValue>) {
        if let Some(s) = surprise {
            self.surprise_history.push(s);
            if self.surprise_history.len() > 10 { self.surprise_history.remove(0); }

            // Calculate internal reward based on the "derivative of surprise"
            if self.surprise_history.len() >= 5 {
                let prev_avg: i32 = self.surprise_history.iter().take(3).sum::<i32>() / 3;
                let curr_avg: i32 = self.surprise_history.iter().skip(5).sum::<i32>() / 5;

                // If surprise is high but dropping, the system is actively learning -> REWARD
                if s > 500 && curr_avg < prev_avg {
                     self.last_internal_reward = self.curiosity_reward_strength;
                } else {
                     self.last_internal_reward = 0;
                }
            }
        }
    }

    fn get_global_modulations(&self) -> Vec<(usize, i32)> {
        if self.last_internal_reward > 0 {
             // Inject internal reward into global dopamine signal (signal 0)
             vec![(0, self.last_internal_reward)]
        } else {
             Vec::new()
        }
    }

    fn on_night_phase(&mut self, _neurons: &mut NeuronsSoA, _synapses: &mut SynapsesSoA, _reward: Option<IValue>) {
        self.surprise_history.clear();
    }

    fn box_clone(&self) -> Box<dyn NanoModule> { Box::new(self.clone()) }
    fn get_state(&self) -> Vec<u8> { bincode::serialize(self).unwrap_or_default() }
    fn set_state(&mut self, state: &[u8]) {
        if let Ok(new_self) = bincode::deserialize::<Self>(state) { *self = new_self; }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_curiosity_reward() {
        let mut curiosity = CuriosityModule::new();

        // Simulate decreasing surprise (learning)
        curiosity.on_update_weights(&mut NeuronsSoA::new(0), &[], &[], 0, Some(1000));
        curiosity.on_update_weights(&mut NeuronsSoA::new(0), &[], &[], 0, Some(900));
        curiosity.on_update_weights(&mut NeuronsSoA::new(0), &[], &[], 0, Some(800));
        curiosity.on_update_weights(&mut NeuronsSoA::new(0), &[], &[], 0, Some(700));
        curiosity.on_update_weights(&mut NeuronsSoA::new(0), &[], &[], 0, Some(600));

        assert!(curiosity.last_internal_reward > 0);
        let mods = curiosity.get_global_modulations();
        assert_eq!(mods[0], (0, curiosity.curiosity_reward_strength));
    }
}
