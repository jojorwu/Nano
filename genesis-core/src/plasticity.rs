use crate::{SynapsesSoA, IValue, SCALE};

pub struct StdpRule {
    pub tau: u64,
    pub a_plus: IValue,
    pub a_minus: IValue,
    pub reward_scale: IValue, // R-STDP factor
    pub metaplasticity_enabled: bool,
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
            let delta = ((self.a_plus as i64 * reward as i64 * self.reward_scale as i64) >> 20) as i32;
            *weight = weight.saturating_add(delta);
        }
    }

    fn update_temporal(&self, weight: &mut IValue, pre_tick: u64, post_tick: u64, _current_tick: u64) {
        if pre_tick == 0 || post_tick == 0 { return; }
        let diff = (post_tick as i64) - (pre_tick as i64);

        // Metaplasticity: Sliding threshold for LTP/LTD
        // In this implementation, we adjust the effective A+ and A- based on weight magnitude
        // to prevent saturation and encourage homeostatic stability.
        let mut local_a_plus = self.a_plus;
        let mut local_a_minus = self.a_minus;

        if self.metaplasticity_enabled {
            // BCM-like behavior: high weight -> harder to increase, easier to decrease
            if *weight > SCALE * 2 {
                local_a_plus = (local_a_plus * 8) / 10;
                local_a_minus = (local_a_minus * 12) / 10;
            } else if *weight < -SCALE * 2 {
                local_a_plus = (local_a_plus * 12) / 10;
                local_a_minus = (local_a_minus * 8) / 10;
            }
        }

        if diff > 0 && diff < self.tau as i64 {
            // Long-Term Potentiation (LTP)
            let delta = (local_a_plus as i64 * (self.tau as i64 - diff) / self.tau as i64) as i32;
            *weight = weight.saturating_add(delta);
        } else if diff < 0 && diff > -(self.tau as i64) {
            // Long-Term Depression (LTD)
            let delta = (local_a_minus as i64 * (self.tau as i64 - diff.abs()) / self.tau as i64) as i32;
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

    pub fn mutate(&self, synapses: &mut SynapsesSoA, neuron_count: usize, reward: IValue) {
        self.mutate_with_activity(synapses, neuron_count, reward, &[]);
    }

    pub fn mutate_with_activity(&self, synapses: &mut SynapsesSoA, neuron_count: usize, reward: IValue, activity_history: &[Vec<bool>]) {
        if reward < -100 {
            let prune_count = (synapses.len() as f32 * self.mutation_rate).max(1.0) as usize;
            use rand::Rng;
            let mut rng = rand::thread_rng();
            for _ in 0..prune_count {
                if synapses.len() > 0 {
                    let idx = rng.gen_range(0..synapses.len());
                    synapses.remove(idx);
                }
            }
        } else if reward > 100 {
            let grow_count = (neuron_count as f32 * self.mutation_rate).max(1.0) as usize;
            use crate::Compartment;

            if !activity_history.is_empty() {
                let mut grown = 0;
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
        let stdp = StdpRule { tau: 10, a_plus: 100, a_minus: 100, reward_scale: 1024, metaplasticity_enabled: false };
        let mut weight = 1024;

        stdp.update_temporal(&mut weight, 5, 8, 10);
        assert!(weight > 1024);

        let mut weight2 = 1024;
        stdp.update_temporal(&mut weight2, 8, 5, 10);
        assert!(weight2 < 1024);
    }

    #[test]
    fn test_stdp_metaplasticity() {
        use crate::PlasticityRule;
        let stdp = StdpRule { tau: 10, a_plus: 100, a_minus: 100, reward_scale: 1024, metaplasticity_enabled: true };

        // High weight: should be harder to increase
        let mut high_weight = 3000; // > SCALE * 2
        stdp.update_temporal(&mut high_weight, 5, 8, 10);
        let delta_high = high_weight - 3000;

        let mut low_weight = 1000;
        stdp.update_temporal(&mut low_weight, 5, 8, 10);
        let delta_low = low_weight - 1000;

        assert!(delta_high < delta_low, "High weight should have smaller increment due to metaplasticity");
    }

    #[test]
    fn test_contrastive_learning() {
        use crate::PlasticityRule;
        let gsop = crate::GsopRule { learning_rate: 10 };
        let mut weight = 1000;

        gsop.update_contrastive(&mut weight, 600);
        assert!(weight < 1000);

        let mut weight2 = 1000;
        gsop.update_contrastive(&mut weight2, 100);
        assert_eq!(weight2, 1000);
    }
}
