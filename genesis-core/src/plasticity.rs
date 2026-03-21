use crate::{SynapsesSoA, IValue};

pub struct StdpRule {
    pub tau: u64,
    pub a_plus: IValue,
    pub a_minus: IValue,
    pub reward_scale: IValue, // R-STDP factor
}

impl crate::PlasticityRule for StdpRule {
    fn update(&self, weight: &mut IValue, pre_spiked: bool, post_spiked: bool, _compartment: crate::Compartment) {
        if pre_spiked && post_spiked {
            *weight = weight.saturating_add(self.a_plus / 2);
        }
    }

    fn update_rewarded(&self, weight: &mut IValue, pre_spiked: bool, post_spiked: bool, reward: IValue, _compartment: crate::Compartment) {
        // R-STDP: Reward modulates the base temporal update
        if pre_spiked && post_spiked {
            // (a * reward * scale) >> 20
            let delta = ((self.a_plus as i64 * reward as i64 * self.reward_scale as i64) >> 20) as i32;
            *weight = weight.saturating_add(delta);
        }
    }

    fn update_temporal(&self, weight: &mut IValue, pre_tick: u64, post_tick: u64, _current_tick: u64) {
        if pre_tick == 0 || post_tick == 0 { return; }
        let diff = (post_tick as i64) - (pre_tick as i64);

        if diff > 0 && diff < self.tau as i64 {
            // Long-Term Potentiation (LTP)
            let delta = (self.a_plus as i64 * (self.tau as i64 - diff) / self.tau as i64) as i32;
            *weight = weight.saturating_add(delta);
        } else if diff < 0 && diff > -(self.tau as i64) {
            // Long-Term Depression (LTD)
            let delta = (self.a_minus as i64 * (self.tau as i64 - diff.abs()) / self.tau as i64) as i32;
            *weight = weight.saturating_sub(delta);
        }

        // Clamp weights
        if *weight > 5000 { *weight = 5000; }
        if *weight < -5000 { *weight = -5000; }
    }
}

pub struct StructuralPlasticityConfig {
    pub prune_threshold: IValue,
    pub grow_threshold: usize,
    pub max_synapses: usize,
}

impl Default for StructuralPlasticityConfig {
    fn default() -> Self {
        Self {
            prune_threshold: 10,
            grow_threshold: 5,
            max_synapses: 1000000,
        }
    }
}

pub fn prune_synapses(synapses: &mut SynapsesSoA, threshold: IValue) -> usize {
    let mut pruned = 0;
    let mut i = 0;
    while i < synapses.len() {
        if synapses.weight[i].abs() < threshold {
            synapses.remove(i);
            pruned += 1;
        } else {
            i += 1;
        }
    }
    pruned
}

pub struct EvolutionaryOptimizer {
    pub mutation_rate: f32,
}

impl EvolutionaryOptimizer {
    pub fn new(mutation_rate: f32) -> Self {
        Self { mutation_rate }
    }

    /// Perform structural mutations based on reward and activity.
    pub fn mutate(&self, synapses: &mut SynapsesSoA, neuron_count: usize, reward: IValue) {
        self.mutate_with_activity(synapses, neuron_count, reward, &[]);
    }

    /// Perform structural mutations with activity correlation.
    pub fn mutate_with_activity(&self, synapses: &mut SynapsesSoA, neuron_count: usize, reward: IValue, activity_history: &[Vec<bool>]) {
        if reward < -100 {
            // High negative reward -> Prune weak synapses more aggressively
            let prune_count = (synapses.len() as f32 * self.mutation_rate).max(1.0) as usize;
            use rand::Rng;
            let mut rng = rand::thread_rng();
            for _ in 0..prune_count {
                if synapses.len() > 0 {
                    // Evolutionary Pruning: Remove synapses that are weak or randomly for exploration.
                    let idx = rng.gen_range(0..synapses.len());
                    synapses.remove(idx);
                }
            }
        } else if reward > 100 {
            // High positive reward -> Grow synapses based on activity correlation if available
            let grow_count = (neuron_count as f32 * self.mutation_rate).max(1.0) as usize;

            use crate::Compartment;

            if !activity_history.is_empty() {
                // Correlational Growth: find neurons that fire together
                let mut grown = 0;
                // Simple heuristic: check last few steps for coincidences
                for step in activity_history.iter().rev().take(5) {
                    let active: Vec<usize> = step.iter().enumerate().filter(|&(_, &s)| s).map(|(i, _)| i).collect();
                    if active.len() >= 2 {
                        for &i in active.iter().take(3) {
                            for &j in active.iter().take(3) {
                                if i != j && grown < grow_count {
                                    let comp = if (i + j) % 2 == 0 { Compartment::Proximal } else { Compartment::Distal };
                                    if grow_synapse_in_compartment(synapses, i as u32, j as u32, 100, comp, &StructuralPlasticityConfig::default()) {
                                        grown += 1;
                                    }
                                }
                            }
                        }
                    }
                    if grown >= grow_count { break; }
                }
            } else {
                // Fallback to exploratory growth
                let mut offset = synapses.len() as u32;
                for _ in 0..grow_count {
                    let src = (reward as u32 + offset) % neuron_count as u32;
                    let target = (reward as u32 * 31 + offset + 7) % neuron_count as u32;
                    if src != target {
                        let comp = if (src + target) % 2 == 0 { Compartment::Proximal } else { Compartment::Distal };
                        grow_synapse_in_compartment(synapses, src, target, 100, comp, &StructuralPlasticityConfig::default());
                    }
                    offset = offset.wrapping_add(1);
                }
            }
        }
    }
}

pub fn grow_synapse(
    synapses: &mut SynapsesSoA,
    source: u32,
    target: u32,
    initial_weight: IValue,
    config: &StructuralPlasticityConfig
) -> bool {
    grow_synapse_in_compartment(synapses, source, target, initial_weight, crate::Compartment::Proximal, config)
}

pub fn grow_synapse_in_compartment(
    synapses: &mut SynapsesSoA,
    source: u32,
    target: u32,
    initial_weight: IValue,
    compartment: crate::Compartment,
    config: &StructuralPlasticityConfig
) -> bool {
    if synapses.len() >= config.max_synapses {
        return false;
    }

    // Check if already exists
    for i in 0..synapses.len() {
        if synapses.source_index[i] == source && synapses.target_index[i] == target {
            return false;
        }
    }

    synapses.push_to_compartment(source, target, initial_weight, compartment);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pruning() {
        let mut synapses = SynapsesSoA::with_capacity(10);
        synapses.push(0, 1, 100);
        synapses.push(1, 2, 5);
        let pruned = prune_synapses(&mut synapses, 10);
        assert_eq!(pruned, 1);
        assert_eq!(synapses.len(), 1);
    }

    #[test]
    fn test_stdp() {
        use crate::PlasticityRule;
        let stdp = StdpRule { tau: 10, a_plus: 100, a_minus: 100, reward_scale: 1024 };
        let mut weight = 1024;

        // LTP: pre=5, post=8 (diff=3)
        stdp.update_temporal(&mut weight, 5, 8, 10);
        assert!(weight > 1024);

        // LTD: pre=8, post=5 (diff=-3)
        let mut weight2 = 1024;
        stdp.update_temporal(&mut weight2, 8, 5, 10);
        assert!(weight2 < 1024);
    }

    #[test]
    fn test_contrastive_learning() {
        use crate::PlasticityRule;
        let gsop = crate::GsopRule { learning_rate: 10 };
        let mut weight = 1000;

        // High correlation should reduce weight
        gsop.update_contrastive(&mut weight, 600);
        assert!(weight < 1000);

        let mut weight2 = 1000;
        // Low correlation should not change weight much
        gsop.update_contrastive(&mut weight2, 100);
        assert_eq!(weight2, 1000);
    }
}

use crate::{NanoModule, NeuronsSoA};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AdaptiveLearningRateModule {
    pub base_lr: IValue,
    pub current_lr: IValue,
    pub window_reward: Vec<IValue>,
}

impl AdaptiveLearningRateModule {
    pub fn new(base: IValue) -> Self {
        Self {
            base_lr: base,
            current_lr: base,
            window_reward: Vec::new(),
        }
    }
}

impl NanoModule for AdaptiveLearningRateModule {
    fn name(&self) -> &str { "adaptive_lr" }

    fn on_tick(&mut self, _neurons: &mut NeuronsSoA, _previous_spikes: &[bool], _tick: u32) {}

    fn on_update_weights(&mut self, _neurons: &mut NeuronsSoA, _previous_spikes: &[bool], _current_spikes: &[bool], _tick: u32, surprise: Option<IValue>) {
        if let Some(s) = surprise {
            if s > 1500 {
                self.current_lr = (self.current_lr * 11) / 10;
            } else if s < 200 {
                self.current_lr = (self.current_lr * 9) / 10;
            }
            self.current_lr = self.current_lr.clamp(self.base_lr / 2, self.base_lr * 4);
        }
    }

    fn on_night_phase(&mut self, _synapses: &mut SynapsesSoA, reward: Option<IValue>) {
        if let Some(r) = reward {
            self.window_reward.push(r);
            if self.window_reward.len() > 10 { self.window_reward.remove(0); }

            let avg: i32 = self.window_reward.iter().sum::<i32>() / self.window_reward.len().max(1) as i32;
            if avg < 0 {
                self.current_lr = self.current_lr.saturating_add(1);
            }
        }
    }

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
