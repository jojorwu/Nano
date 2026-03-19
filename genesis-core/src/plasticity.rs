use crate::{SynapsesSoA, IValue};

pub struct StdpRule {
    pub tau: u64,
    pub a_plus: IValue,
    pub a_minus: IValue,
    pub reward_scale: IValue, // R-STDP factor
}

impl crate::PlasticityRule for StdpRule {
    fn update(&self, weight: &mut IValue, pre_spiked: bool, post_spiked: bool) {
        if pre_spiked && post_spiked {
            *weight = weight.saturating_add(self.a_plus / 2);
        }
    }

    fn update_rewarded(&self, weight: &mut IValue, pre_spiked: bool, post_spiked: bool, reward: IValue) {
        // R-STDP: Reward modulates the base temporal update
        if pre_spiked && post_spiked {
            let delta = (self.a_plus * reward * self.reward_scale) / 1000000;
            *weight = weight.saturating_add(delta);
        }
    }

    fn update_temporal(&self, weight: &mut IValue, pre_tick: u64, post_tick: u64, _current_tick: u64) {
        if pre_tick == 0 || post_tick == 0 { return; }
        let diff = (post_tick as i64) - (pre_tick as i64);

        if diff > 0 && diff < self.tau as i64 {
            // Long-Term Potentiation (LTP)
            let delta = (self.a_plus * (self.tau as i64 - diff) as i32) / self.tau as i32;
            *weight = weight.saturating_add(delta);
        } else if diff < 0 && diff > -(self.tau as i64) {
            // Long-Term Depression (LTD)
            let delta = (self.a_minus * (self.tau as i64 - diff.abs()) as i32) / self.tau as i32;
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
        if reward < -100 {
            // High negative reward -> Prune weak synapses more aggressively
            let prune_count = (synapses.len() as f32 * self.mutation_rate).max(1.0) as usize;
            for _ in 0..prune_count {
                if synapses.len() > 0 {
                    // Simple mutation: remove a random (first) connection for exploration
                    synapses.remove(0);
                }
            }
        } else if reward > 100 {
            // High positive reward -> Grow exploratory synapses between random neurons
            let grow_count = (neuron_count as f32 * self.mutation_rate).max(1.0) as usize;

            // Deterministic but non-redundant growth logic using current reward and synapse length as entropy
            let mut offset = synapses.len() as u32;
            for _ in 0..grow_count {
                let src = (reward as u32 + offset) % neuron_count as u32;
                let target = (reward as u32 * 31 + offset + 7) % neuron_count as u32;

                if src != target {
                    // Use helper to avoid duplicates
                    grow_synapse(synapses, src, target, 100, &StructuralPlasticityConfig::default());
                }
                offset = offset.wrapping_add(1);
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
    if synapses.len() >= config.max_synapses {
        return false;
    }

    // Check if already exists
    for i in 0..synapses.len() {
        if synapses.source_index[i] == source && synapses.target_index[i] == target {
            return false;
        }
    }

    synapses.push(source, target, initial_weight);
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
        let stdp = StdpRule { tau: 10, a_plus: 100, a_minus: 100, reward_scale: 1000 };
        let mut weight = 1000;

        // LTP: pre=5, post=8 (diff=3)
        stdp.update_temporal(&mut weight, 5, 8, 10);
        assert!(weight > 1000);

        // LTD: pre=8, post=5 (diff=-3)
        let mut weight2 = 1000;
        stdp.update_temporal(&mut weight2, 8, 5, 10);
        assert!(weight2 < 1000);
    }
}
