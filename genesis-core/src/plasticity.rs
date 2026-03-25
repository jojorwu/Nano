use crate::{SynapsesSoA, IValue, WEIGHT_CLAMP_LIMIT};

pub fn clamp_and_preserve_sign(weight: &mut IValue, old_weight: IValue) {
    if old_weight > 0 && *weight < 0 { *weight = 1; }
    if old_weight < 0 && *weight > 0 { *weight = -1; }
    if *weight > WEIGHT_CLAMP_LIMIT { *weight = WEIGHT_CLAMP_LIMIT; }
    if *weight < -WEIGHT_CLAMP_LIMIT { *weight = -WEIGHT_CLAMP_LIMIT; }
}

pub struct StdpRule {
    pub tau: u64,
    pub a_plus: IValue,
    pub a_minus: IValue,
    pub reward_scale: IValue, // R-STDP factor
}

/// Highly efficient STDP implementation using BitPacked history
pub struct SparseStdpRule {
    pub tau: u64,
    pub a_plus: IValue,
    pub a_minus: IValue,
}

impl SparseStdpRule {
    pub fn apply_bitpacked(&self, weight: &mut IValue, src: usize, tgt: usize, history: &[crate::SpikeData]) {
        let n_count = (src.max(tgt) + 64) / 64 * 64; // Approximate
        let mut ltp_count = 0;
        let mut ltd_count = 0;

        // Optimized bitwise coincidence detection across time window
        for t in 1..self.tau as usize {
            if t >= history.len() { break; }
            let now = &history[0].to_bitpacked(n_count);
            let past = &history[t].to_bitpacked(n_count);

            let src_word = src / 64;
            let src_bit = src % 64;
            let tgt_word = tgt / 64;
            let tgt_bit = tgt % 64;

            // LTP: Pre (past) -> Post (now)
            if ((past[src_word] >> src_bit) & 1 == 1) && ((now[tgt_word] >> tgt_bit) & 1 == 1) {
                ltp_count += 1;
            }
            // LTD: Post (past) -> Pre (now)
            if ((past[tgt_word] >> tgt_bit) & 1 == 1) && ((now[src_word] >> src_bit) & 1 == 1) {
                ltd_count += 1;
            }
        }

        let delta = (self.a_plus * ltp_count as i32) - (self.a_minus * ltd_count as i32);
        *weight = weight.saturating_add(delta / 10);
    }

    /// Update weights based on long-term temporal context (L2 History)
    pub fn apply_l2_context(&self, weight: &mut IValue, src_bid: usize, tgt_bid: usize, l2_history: &[Vec<u16>]) {
        if l2_history.is_empty() { return; }

        // Sum activity over L2 window
        let mut src_activity = 0u32;
        let mut tgt_activity = 0u32;

        for step in l2_history {
            if src_bid < step.len() { src_activity += step[src_bid] as u32; }
            if tgt_bid < step.len() { tgt_activity += step[tgt_bid] as u32; }
        }

        // Correlational learning on long timescales:
        // if both blocks are consistently active, strengthen connection.
        if src_activity > 10 && tgt_activity > 10 {
            let correlation = (src_activity * tgt_activity) / 100;
            let delta = (self.a_plus * correlation as i32) / 100;
            *weight = weight.saturating_add(delta);
        }
    }
}

impl crate::PlasticityRule for StdpRule {
    fn apply(&self, weight: &mut IValue, ctx: &crate::PlasticityContext) {
        let weight_before = *weight;
        let pre_spiked = ctx.pre_spiked;
        let post_spiked = ctx.post_spiked;

        let modulation_gain = ctx.get_modulation_gain();

        // 1. Reward-modulated update (R-STDP component)
        if let Some(reward) = ctx.reward {
             if pre_spiked && post_spiked {
                let delta = ((self.a_plus as i64 * reward as i64 * self.reward_scale as i64 * modulation_gain) >> 40) as i32;
                *weight = weight.saturating_add(delta);
            }
        } else {
             if pre_spiked && post_spiked {
                let delta = (self.a_plus as i64 * modulation_gain >> 20) as i32;
                *weight = weight.saturating_add(delta);
            }
        }

        // 2. Temporal update (Metaplastic STDP / BCM-lite)
        if ctx.pre_last_spike == 0 || ctx.post_last_spike == 0 {
            clamp_and_preserve_sign(weight, weight_before);
            return;
        }
        let diff = (ctx.post_last_spike as i64) - (ctx.pre_last_spike as i64);

        // BCM Logic: Use activity_ema to shift the LTP/LTD threshold
        // If the neuron is highly active, it becomes harder to strengthen connections (LTP)
        // and easier to weaken them (LTD), maintaining homeostatic stability.
        let activity = ctx.neurons.activity_ema[ctx.post_index];
        let bcm_threshold = 100; // Target activity

        // Exponential integer approximation for temporal decay
        // exp(-x/tau) approx (tau - x) / tau (linear) -> we want something better
        // LUT-like approach or bit-shift approximation
        let temporal_factor = |d: i64, tau: i64| -> i64 {
            let ratio = (d.abs() * 1024) / tau;
            if ratio > 2048 { return 0; } // ~2*tau
            // Piecewise linear approximation of exp(-x)
            if ratio < 512 { 1024 - ratio } // 0 to 0.5
            else if ratio < 1024 { 512 - (ratio - 512) / 2 } // 0.5 to 1.0
            else { 256 - (ratio - 1024) / 4 } // 1.0 to 2.0
        };

        if diff > 0 {
            let factor = temporal_factor(diff, self.tau as i64);
            let mut ltp_scale = 1024i64;
            if activity > bcm_threshold {
                ltp_scale = (1024 * bcm_threshold as i64) / activity.max(1) as i64;
            }
            let delta = (self.a_plus as i64 * factor * ltp_scale >> 20) as i32;
            *weight = weight.saturating_add(delta);
        } else if diff < 0 {
            let factor = temporal_factor(diff, self.tau as i64);
            let mut ltd_scale = 1024i64;
            if activity > bcm_threshold {
                ltd_scale = (1024 * activity as i64) / bcm_threshold as i64;
            }
            let delta = (self.a_minus as i64 * factor * ltd_scale >> 20) as i32;
            *weight = weight.saturating_sub(delta);
        }

        clamp_and_preserve_sign(weight, weight_before);
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
    pub fn mutate(&self, synapses: &mut SynapsesSoA, neurons: &mut crate::NeuronsSoA, reward: IValue) {
        self.mutate_with_activity(synapses, neurons, reward, &[], 1000000);
    }

    /// Perform structural mutations with activity correlation and topographic constraints.
    pub fn mutate_with_activity(
        &self,
        synapses: &mut SynapsesSoA,
        neurons: &mut crate::NeuronsSoA,
        reward: IValue,
        activity_history: &[Vec<bool>],
        max_synapses: usize
    ) {
        self.mutate_with_surprise(synapses, neurons, reward, activity_history, max_synapses, &[]);
    }

    pub fn mutate_with_surprise(
        &self,
        synapses: &mut SynapsesSoA,
        neurons: &mut crate::NeuronsSoA,
        reward: IValue,
        activity_history: &[Vec<bool>],
        max_synapses: usize,
        block_surprise: &[f32],
    ) {
        let neuron_count = neurons.len();
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
                                    // Distance-aware compartment targeting
                                    let dx = (neurons.x[i] - neurons.x[j]) as i32;
                                    let dy = (neurons.y[i] - neurons.y[j]) as i32;
                                    let dist_sq = dx*dx + dy*dy;

                                    let comp = if dist_sq < 100 {
                                        Compartment::Proximal
                                    } else if dist_sq < 400 {
                                        Compartment::Distal
                                    } else {
                                        Compartment::Apical
                                    };

                                    let config = StructuralPlasticityConfig { max_synapses, ..Default::default() };

                    // Surprise-Targeted Neurogenesis:
                    // If surprise in target block is very high, grow NEW neurons first.
                    if !block_surprise.is_empty() {
                        let bid = neurons.block_id[j];
                        if (bid as usize) < block_surprise.len() && block_surprise[bid as usize] > 0.8 {
                            grow_neurons_in_block(neurons, bid, 1, neurons.layer_id[j as usize]);
                        }
                    }

                    // Surprise-Targeted Growth: increase initial weight for neurons in surprised blocks
                    let initial_weight = if !block_surprise.is_empty() {
                        let bid = neurons.block_id[j] as usize;
                        if bid < block_surprise.len() && block_surprise[bid] > 0.5 { 200 } else { 100 }
                    } else { 100 };

                    if grow_synapse_in_compartment(synapses, i as u32, j as u32, initial_weight, comp, &config) {
                                        grown += 1;
                                    }
                                }
                            }
                        }
                    }
                    if grown >= grow_count { break; }
                }
            } else {
                // Topographic Exploratory Growth: encourage local connections
                use rand::Rng;
                let mut rng = rand::thread_rng();
                for _ in 0..grow_count {
                    let src = rng.gen_range(0..neuron_count) as u32;
                    let sx = neurons.x[src as usize];
                    let sy = neurons.y[src as usize];

                    // Spatial Optimization: Search in a local neighborhood first
                    let mut best_target = (src + 1) % neuron_count as u32;
                    let mut max_score = -1000000i32;

                    // Instead of global sampling, we sample indices near the source index
                    // assuming similar indices are spatially closer (standard for SoA layouts)
                    let search_radius = (neuron_count / 20).max(50);

                    for _ in 0..20 {
                        let offset = rng.gen_range(0..search_radius * 2) as i32 - search_radius as i32;
                        let cand = ((src as i32 + offset).rem_euclid(neuron_count as i32)) as u32;
                        if cand == src { continue; }

                        let dx = (neurons.x[cand as usize] - sx) as i32;
                        let dy = (neurons.y[cand as usize] - sy) as i32;
                        let dist_sq = dx*dx + dy*dy;

                        // Score: prefer nearby neurons AND those with high long-term activity (activity_ema)
                        // but not too much activity (avoid hyper-hubs)
                        let activity = neurons.activity_ema[cand as usize];

                        // Higher score is better
                        let score = (activity as i32 * 10) - (dist_sq / 10);

                        if score > max_score {
                            max_score = score;
                            best_target = cand;
                        }
                    }

                    // Targeted compartment selection based on distance
                    let dx = (neurons.x[best_target as usize] - sx) as i32;
                    let dy = (neurons.y[best_target as usize] - sy) as i32;
                    let dist = ((dx*dx + dy*dy) as f32).sqrt();

                    let comp = if dist < 10.0 {
                        Compartment::Proximal
                    } else if dist < 50.0 {
                        Compartment::Distal
                    } else {
                        Compartment::Apical
                    };

                    let config = StructuralPlasticityConfig { max_synapses, ..Default::default() };
                    grow_synapse_in_compartment(synapses, src, best_target, 150, comp, &config);
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

pub fn grow_neurons_in_block(
    neurons: &mut crate::NeuronsSoA,
    block_id: u32,
    count: usize,
    layer_id: u16
) {
    let start_idx = neurons.len();
    neurons.grow(count);
    for i in 0..count {
        let idx = start_idx + i;
        neurons.block_id[idx] = block_id;
        neurons.layer_id[idx] = layer_id;
        // Inherit spatial coordinates from an existing neuron in the same block if possible
        if let Some(ref_idx) = neurons.block_id.iter().take(start_idx).position(|&b| b == block_id) {
            neurons.x[idx] = neurons.x[ref_idx];
            neurons.y[idx] = neurons.y[ref_idx];
        }
    }
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

    synapses.push_to_compartment(source, target, initial_weight, 1, compartment);
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
        use crate::{PlasticityRule, PlasticityContext, NeuronsSoA, Compartment};
        let stdp = StdpRule { tau: 10, a_plus: 100, a_minus: 100, reward_scale: 1024 };
        let mut weight = 1024;
        let neurons = NeuronsSoA::new(1);

        // LTP: pre=5, post=8 (diff=3)
        let ctx = PlasticityContext {
            pre_spiked: true, post_spiked: true, backprop_signal: 0,
            compartment: Compartment::Proximal,
            reward: None,
            neuromodulation: crate::NeuromodulationState::default(),
            pre_last_spike: 5, post_last_spike: 8, current_tick: 10,
            post_index: 0,
            neurons: &neurons,
        };
        stdp.apply(&mut weight, &ctx);
        assert!(weight > 1024);

        // LTD: pre=8, post=5 (diff=-3)
        let mut weight2 = 1024;
        let ctx2 = PlasticityContext {
            pre_spiked: true, post_spiked: true, backprop_signal: 0,
            compartment: Compartment::Proximal,
            reward: None,
            neuromodulation: crate::NeuromodulationState::default(),
            pre_last_spike: 8, post_last_spike: 5, current_tick: 10,
            post_index: 0,
            neurons: &neurons,
        };
        stdp.apply(&mut weight2, &ctx2);
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

    fn on_tick(&mut self, _bus: &crate::InputBus, _previous_spikes: &[bool], _tick: u32) {}

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

    fn on_night_phase(&mut self, _neurons: &mut NeuronsSoA, _synapses: &mut SynapsesSoA, reward: Option<IValue>) {
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
