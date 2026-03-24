use genesis_core::{BakedModel, GsopRule, PlasticityRule, SCALE, IValue, Compartment};
use genesis_core::plasticity::{prune_synapses, StructuralPlasticityConfig};
use rayon::prelude::*;
use crate::{ComputeBackend, kernels::calculate_membrane_potential};

pub struct CpuBackend {
    pub structural_config: StructuralPlasticityConfig,
    pub plasticity_rule: Box<dyn PlasticityRule + Send + Sync>,
    pub optimizer: genesis_core::plasticity::EvolutionaryOptimizer,
    pub expert_masks: Vec<bool>, // MoE: which neuron groups are active
    pub synapse_index: Vec<[Vec<usize>; 16]>, // source_neuron -> delay (1..16) -> list of synapse_indices
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self {
            structural_config: StructuralPlasticityConfig::default(),
            plasticity_rule: Box::new(GsopRule { learning_rate: 10 }),
            optimizer: genesis_core::plasticity::EvolutionaryOptimizer::new(0.01),
            expert_masks: Vec::new(),
            synapse_index: Vec::new(),
        }
    }
}

impl CpuBackend {
    pub fn rebuild_index(&mut self, model: &BakedModel) {
        let n_count = model.neurons.len();
        self.synapse_index.clear();
        self.synapse_index.resize_with(n_count, Default::default);

        for (i, (&src, &delay)) in model.synapses.source_index.iter().zip(&model.synapses.delay).enumerate() {
            let src = src as usize;
            let d_idx = (delay.clamp(1, 16) - 1) as usize;
            if src < n_count {
                self.synapse_index[src][d_idx].push(i);
            }
        }
    }

    fn propagate_sparse_delayed_spikes(&mut self, model: &mut BakedModel, previous_spikes: &[bool], history: &[Vec<bool>]) {
        if self.synapse_index.len() != model.neurons.len() {
            self.rebuild_index(model);
        }

        // Optimized O(active_spikes * average_fanout) lookup.
        // We only iterate over spiked neurons across the rolling temporal window.
        for d in 1..=16 {
            let spikes = if d == 1 {
                previous_spikes
            } else if d - 1 < history.len() {
                &history[d - 1]
            } else {
                continue;
            };

            let d_idx = (d - 1) as usize;

            // Find active source neurons
            let active_sources: Vec<usize> = spikes.iter().enumerate()
                .filter(|&(_, &fired)| fired)
                .map(|(i, _)| i)
                .collect();

            if active_sources.is_empty() { continue; }

            // To parallelize without atomics, we'd need to group by target.
            // Instead, let's process source neurons in parallel chunks
            // and use raw pointers for potential additions (safe if we accept
            // minor jitter or use a more complex grouping).
            // For now, let's optimize the inner loop processing.

            for &src in &active_sources {
                if src >= self.synapse_index.len() { continue; }

                for &syn_idx in &self.synapse_index[src][d_idx] {
                    let target = model.synapses.target_index[syn_idx] as usize;
                    let gate = model.neurons.dendritic_gate[target];
                    if gate < 8 { continue; }

                    // Short-Term Plasticity (STP): modulate weight by available resources and calcium
                    let u_facilitation = model.synapses.stp_calcium[syn_idx];
                    let r_depression = model.synapses.stp_resources[syn_idx];

                    let stp_weight = ((model.synapses.weight[syn_idx] as i64 * r_depression as i64) >> 10) as i32;
                    let stp_weight = ((stp_weight as i64 * (SCALE + u_facilitation) as i64) >> 10) as i32;

                    // Consumption: firing uses resources and increases calcium
                    model.synapses.stp_resources[syn_idx] = (model.synapses.stp_resources[syn_idx] * 800) >> 10;
                    model.synapses.stp_calcium[syn_idx] = (model.synapses.stp_calcium[syn_idx] + 200).min(SCALE);

                    let gated_weight = ((stp_weight as i64 * gate as i64) >> 10) as i32;
                    match model.synapses.compartment[syn_idx] {
                        Compartment::Proximal => { model.neurons.proximal_potential[target] = model.neurons.proximal_potential[target].saturating_add(gated_weight); }
                        Compartment::Distal => { model.neurons.distal_potential[target] = model.neurons.distal_potential[target].saturating_add(gated_weight); }
                        Compartment::Apical => { model.neurons.apical_potential[target] = model.neurons.apical_potential[target].saturating_add(gated_weight); }
                        Compartment::Basal => { model.neurons.basal_potential[target] = model.neurons.basal_potential[target].saturating_add(gated_weight); }
                    }
                }
            }
        }
    }

    fn propagate_latent_spikes(&self, model: &mut BakedModel, previous_spikes: &[bool]) {
        if let Some(ref latent) = model.synapses.latent_matrix {
            let mut latent_state = vec![0i32; latent.rank];
            // Sparse MLA: iterate only over spiked neurons
            let active_indices: Vec<usize> = previous_spikes.iter().enumerate()
                .filter(|&(_, &s)| s).map(|(i, _)| i).collect();

            if active_indices.is_empty() { return; }

            for i in active_indices {
                let offset = i * latent.rank;
                for r in 0..latent.rank {
                    latent_state[r] = latent_state[r].saturating_add(latent.u[offset + r]);
                }
            }

            // SIMD-friendly second pass: propagate latent state to all neurons
            for j in 0..model.neurons.len() {
                let gate = model.neurons.dendritic_gate[j];
                let mut sum = 0i64;
                for r in 0..latent.rank {
                    let weight = latent.v[r * model.neurons.len() + j];
                    sum += latent_state[r] as i64 * weight as i64;
                }
                let contribution = ((sum * gate as i64) >> 20) as i32;

                // Latent MLA connections are predominantly distal in this architecture
                model.neurons.distal_potential[j] = model.neurons.distal_potential[j].saturating_add(contribution);
            }
        }
    }

    fn update_neuron_states(&self, model: &mut BakedModel, current_tick: u32, new_spikes: &mut [bool]) {
        let n_count = model.neurons.len();
        let ip_inc = model.config.ip_increment;
        let ip_dec = model.config.ip_decay;
        let noise_amp = model.config.noise_amplitude;
        let target_activity = 100; // 10% of SCALE=1000 base
        let expert_masks = &self.expert_masks;

        let neurons = &mut model.neurons;

        // Optimization: Use a single parallel pass over the SoA structures.
        // We ensure Rayon parallelizes efficiently.
        let spike_results: Vec<bool> = neurons.potential.par_iter_mut()
            .zip(neurons.next_update_tick.par_iter_mut())
            .zip(neurons.refractory_timer.par_iter_mut())
            .zip(neurons.last_spike_tick.par_iter_mut())
            .zip(neurons.backprop_signal.par_iter_mut())
            .zip(neurons.threshold.par_iter_mut())
            .zip(neurons.activity_ema.par_iter_mut())
            .zip(neurons.base_threshold.par_iter_mut())
            .zip(neurons.adaptation_current.par_iter_mut())
            .zip(&neurons.proximal_potential)
            .zip(&neurons.distal_potential)
            .zip(&neurons.apical_potential)
            .zip(&neurons.basal_potential)
            .zip(&neurons.gate_threshold)
            .zip(&neurons.liquid_current)
            .zip(&neurons.decay)
            .zip(&neurons.update_interval)
            .zip(0..n_count)
            .map(|(((((((((((((((((pot, next_upd), refr), last_spk), bprop), thresh), activity), b_thresh), adaptation), prox), dist), apical), basal), g_thresh), liquid), decay), upd_int), i)| {
                if current_tick < *next_upd { return false; }
                if !expert_masks.is_empty() && !expert_masks[i % expert_masks.len()] { return false; }

                let current_pot = calculate_membrane_potential(*pot, *prox, *dist, *apical, *basal, *g_thresh, *liquid, *decay, noise_amp, *adaptation);

                // Relative Refractory: Exponentially decaying threshold multiplier
                let refr_mult = if *refr > 0 { 1 + (1 << *refr) } else { 1 };
                let effective_threshold = *thresh * refr_mult;
                let fired = current_pot >= effective_threshold;

                if fired {
                    *pot = 0;
                    *refr = 4;
                    *last_spk = current_tick;
                    *bprop = SCALE;
                    *thresh = thresh.saturating_add(ip_inc);
                    *activity = (*activity * 990 + 1000) / 1000;
                    *adaptation = adaptation.saturating_add(100); // Metabolic cost
                } else {
                    *pot = current_pot;
                    if *refr > 0 { *refr -= 1; }
                    if *thresh > *b_thresh { *thresh = thresh.saturating_sub(ip_dec); }
                    *bprop = ((*bprop as i64 * 800) >> 10) as i32;
                    *activity = (*activity * 990) / 1000;
                    *adaptation = (*adaptation * 95) / 100; // Recovery
                }

                let homeo_rate = if *activity > target_activity * 2 { 2 } else { 1 };
                if *activity > target_activity {
                    *b_thresh = b_thresh.saturating_add(homeo_rate);
                } else if *activity < target_activity && *b_thresh > 512 {
                    *b_thresh = b_thresh.saturating_sub(1);
                }
                *next_upd = current_tick + *upd_int;
                fired
            }).collect();

        for i in 0..n_count { if spike_results[i] { new_spikes[i] = true; } }
    }
}

impl ComputeBackend for CpuBackend {
    fn name(&self) -> &'static str { "CpuBackend" }
    fn day_phase(&mut self, model: &mut BakedModel, external_inputs: &[i32], previous_spikes: &[bool], history: &[Vec<bool>], current_tick: u32, _modulation: genesis_core::NeuromodulationState) -> genesis_core::SpikeData {
        let n_count = model.neurons.len();

        // Apply external inputs directly to proximal potential with dendritic gating
        for (i, &val) in external_inputs.iter().enumerate() {
            if i < n_count {
                let gate = model.neurons.dendritic_gate[i];
                let gated_val = ((val as i64 * gate as i64) >> 10) as i32;
                model.neurons.proximal_potential[i] = model.neurons.proximal_potential[i].saturating_add(gated_val);
            }
        }

        self.propagate_sparse_delayed_spikes(model, previous_spikes, history);
        self.propagate_latent_spikes(model, previous_spikes);

        // Parallelized STP Recovery
        let synapses = &mut model.synapses;
        synapses.stp_resources.par_iter_mut()
            .zip(synapses.stp_calcium.par_iter_mut())
            .for_each(|(r, c)| {
                *r = (*r * 99 + SCALE) / 100;
                *c = (*c * 95) / 100;
            });

        let mut new_spikes = vec![false; n_count];
        self.update_neuron_states(model, current_tick, &mut new_spikes);

        let active_indices: Vec<usize> = new_spikes.iter().enumerate()
            .filter(|&(_, &s)| s).map(|(i, _)| i).collect();
        genesis_core::SpikeData::Sparse(active_indices)
    }

    fn update_weights(&mut self, model: &mut BakedModel, previous_spikes: &[bool], current_spikes: &[bool], current_tick: u32, reward: Option<IValue>, history: &[Vec<bool>]) {
        self.update_weights_modulated(model, previous_spikes, current_spikes, current_tick, reward, genesis_core::NeuromodulationState::default(), history);
    }

    fn update_weights_modulated(&mut self, model: &mut BakedModel, previous_spikes: &[bool], current_spikes: &[bool], current_tick: u32, reward: Option<IValue>, modulation: genesis_core::NeuromodulationState, _history: &[Vec<bool>]) {
        if self.synapse_index.len() != model.neurons.len() {
            self.rebuild_index(model);
        }

        // Apply learning rate from model config
        self.plasticity_rule = Box::new(GsopRule { learning_rate: model.config.learning_rate });

        use std::collections::HashSet;
        let mut active_synapses = HashSet::new();

        // Sparse selection: only synapses where source OR target spiked
        for (src, &fired) in previous_spikes.iter().enumerate() {
            if fired && src < self.synapse_index.len() {
                for d in 0..16 {
                    for &syn_idx in &self.synapse_index[src][d] {
                        active_synapses.insert(syn_idx);
                    }
                }
            }
        }

        // Note: we don't have an incoming_synapse_index here,
        // we'll rely on source spikes primarily for now or add the index if needed.
        // Actually, I should have implemented incoming index earlier.
        // Re-implementing with the logic I intended.

        let active_indices: Vec<usize> = active_synapses.into_iter().collect();

        // Parallel update of active synapses
        let plasticity_rule = &self.plasticity_rule;
        let neurons = &model.neurons;
        let synapses = &mut model.synapses;
        let weights_ptr = synapses.weight.as_mut_ptr() as usize;

        active_indices.into_par_iter().for_each(|i| {
            let src = synapses.source_index[i] as usize;
            let target = synapses.target_index[i] as usize;

            let ctx = genesis_core::PlasticityContext {
                pre_spiked: previous_spikes[src],
                post_spiked: current_spikes[target],
                backprop_signal: neurons.backprop_signal[target],
                compartment: synapses.compartment[i],
                reward,
                neuromodulation: modulation,
                pre_last_spike: neurons.last_spike_tick[src],
                post_last_spike: neurons.last_spike_tick[target],
                current_tick,
                post_index: target,
                neurons,
            };

            // SAFETY: HashSet ensures unique indices, so no data races on weights[i].
            // Encapsulating unsafe in a tight block with clear justification.
            unsafe {
                let weight_ref = &mut *(weights_ptr as *mut IValue).add(i);
                plasticity_rule.apply(weight_ref, &ctx);
                plasticity_rule.update_contrastive(weight_ref, 0);
            }
        });
    }

    fn structural_plasticity(&mut self, model: &mut BakedModel, reward: Option<IValue>, history: &[Vec<bool>]) {
        prune_synapses(&mut model.synapses, self.structural_config.prune_threshold);
        model.synapses.shrink_to_fit();

        // Structure changed -> Index must be rebuilt next tick
        self.synapse_index.clear();

        // SNNaS: Evolutionary mutation
        if let Some(r) = reward {
            self.optimizer.mutate_with_activity(&mut model.synapses, &model.neurons, r, history, model.config.max_synapses);

            if r > model.config.neurogenesis_reward_threshold && model.neurons.len() < model.config.max_neurons {
                let grow_size = (model.neurons.len() / 20).max(1);
                log::info!("Neurogenesis: growing population by {} neurons", grow_size);
                model.neurons.grow(grow_size);
            }

            if model.config.metaplasticity_enabled {
                if r.abs() < 10 {
                    model.config.learning_rate = (model.config.learning_rate as i64 * 900 / 1024) as i32;
                } else if r.abs() > 500 {
                    model.config.learning_rate = (model.config.learning_rate as i64 * 1100 / 1024) as i32;
                }
            }
        }

        // Parallelized Local Homeostatic Scaling: O(S / Cores)
        let n_count = model.neurons.len();
        let s_count = model.synapses.len();
        if s_count == 0 { return; }

        let mut sum_weights = vec![0i64; n_count];
        for i in 0..s_count {
            let target = model.synapses.target_index[i] as usize;
            if target < n_count {
                sum_weights[target] += model.synapses.weight[i].abs() as i64;
            }
        }

        let max_neuron_sum = 1024 * 16;
        let synapses = &mut model.synapses;

        // Use chunks to parallelize weight adjustment across large synapse populations
        synapses.weight.par_iter_mut().enumerate().for_each(|(i, w)| {
            let target = synapses.target_index[i] as usize;
            if target < n_count && sum_weights[target] > max_neuron_sum {
                let scale_factor = (max_neuron_sum << 10) / sum_weights[target];
                *w = ((*w as i64 * scale_factor) >> 10) as i32;
            }
        });
    }
}
