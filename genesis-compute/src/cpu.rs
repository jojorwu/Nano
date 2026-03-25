use genesis_core::{BakedModel, GsopRule, PlasticityRule, SCALE, IValue, Compartment};
use genesis_core::plasticity::{prune_synapses, StructuralPlasticityConfig};
use rayon::prelude::*;
use crate::{ComputeBackend, kernels::calculate_membrane_potential};

pub struct CpuBackend {
    pub structural_config: StructuralPlasticityConfig,
    pub plasticity_rule: Box<dyn PlasticityRule + Send + Sync>,
    pub optimizer: genesis_core::plasticity::EvolutionaryOptimizer,
    pub expert_masks: Vec<bool>, // MoE: which neuron groups are active

    // Flat CSR representation for synapse indices
    pub synapse_offsets: Vec<u32>,      // size: (n_count * 16) + 1
    pub synapse_indices_flat: Vec<usize>,

    pub incoming_synapse_offsets: Vec<u32>, // size: n_count + 1
    pub incoming_synapse_indices_flat: Vec<usize>,
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self {
            structural_config: StructuralPlasticityConfig::default(),
            plasticity_rule: Box::new(GsopRule { learning_rate: 10 }),
            optimizer: genesis_core::plasticity::EvolutionaryOptimizer::new(0.01),
            expert_masks: Vec::new(),
            synapse_offsets: Vec::new(),
            synapse_indices_flat: Vec::new(),
            incoming_synapse_offsets: Vec::new(),
            incoming_synapse_indices_flat: Vec::new(),
        }
    }
}

impl CpuBackend {
    fn update_single_neuron(&self, i: usize, neurons: &mut genesis_core::NeuronsSoA, model: &BakedModel, current_tick: u32, noise_amp: i32, ip_inc: i32, ip_dec: i32, target_activity: i32) -> bool {
        let attn_dist = ((neurons.distal_potential[i] as i64 * neurons.distal_gate[i] as i64) >> 10) as i32;
        let attn_apical = ((neurons.apical_potential[i] as i64 * neurons.apical_gate[i] as i64) >> 10) as i32;
        let attn_basal = ((neurons.basal_potential[i] as i64 * neurons.basal_gate[i] as i64) >> 10) as i32;

        let astro_mod = ((neurons.astro_calcium[i] as i64 * SCALE as i64) >> 12) as i32;
        let current_pot = calculate_membrane_potential(
            neurons.potential[i], neurons.proximal_potential[i], attn_dist, attn_apical, attn_basal,
            neurons.gate_threshold[i], neurons.liquid_current[i], neurons.decay[i], noise_amp,
            neurons.adaptation_current[i] + astro_mod
        );

        let refr_mult = if neurons.refractory_timer[i] > 0 { 1 + (1 << neurons.refractory_timer[i]) } else { 1 };
        let effective_threshold = neurons.threshold[i] * refr_mult;
        let fired = current_pot >= effective_threshold;

        if fired {
            neurons.potential[i] = 0;
            neurons.refractory_timer[i] = model.config.default_refractory_ticks;
            neurons.last_spike_tick[i] = current_tick;
            neurons.backprop_signal[i] = SCALE;
            neurons.action_potential[i] = SCALE;
            neurons.threshold[i] = neurons.threshold[i].saturating_add(ip_inc);
            let alpha = model.config.activity_ema_alpha as i32;
            neurons.activity_ema[i] = ((neurons.activity_ema[i] as i64 * alpha as i64 + (1000 - alpha) as i64 * 100) / 1000) as i32;
            neurons.adaptation_current[i] = neurons.adaptation_current[i].saturating_add(100);
            neurons.astro_calcium[i] = neurons.astro_calcium[i].saturating_add(model.config.astro_increment);
        } else {
            neurons.potential[i] = current_pot;
            if neurons.refractory_timer[i] > 0 { neurons.refractory_timer[i] -= 1; }
            if neurons.threshold[i] > neurons.base_threshold[i] { neurons.threshold[i] = neurons.threshold[i].saturating_sub(ip_dec); }
            neurons.backprop_signal[i] = ((neurons.backprop_signal[i] as i64 * model.config.smbp_decay) >> 10) as i32;
            neurons.action_potential[i] = ((neurons.action_potential[i] as i64 * model.config.smbp_decay) >> 10) as i32;
            let alpha = model.config.activity_ema_alpha as i32;
            neurons.activity_ema[i] = ((neurons.activity_ema[i] as i64 * alpha as i64) / 1000) as i32;
            neurons.adaptation_current[i] = (neurons.adaptation_current[i] * 95) / 100;
            neurons.astro_calcium[i] = ((neurons.astro_calcium[i] as i64 * model.config.astro_decay_rate) / 1000) as i32;
        }

        let error = neurons.activity_ema[i] - target_activity;
        let homeo_rate = if error.abs() > target_activity { 2 } else { 1 };
        if error > 0 {
            neurons.base_threshold[i] = neurons.base_threshold[i].saturating_add(homeo_rate);
        } else if error < 0 && neurons.base_threshold[i] > model.config.default_threshold / 2 {
            neurons.base_threshold[i] = neurons.base_threshold[i].saturating_sub(1);
        }
        neurons.next_update_tick[i] = current_tick + neurons.update_interval[i];
        fired
    }

    pub fn rebuild_index_internal(&mut self, model: &BakedModel) {
        let n_count = model.neurons.len();
        let s_count = model.synapses.len();

        // 1. Build Forward Index (CSR)
        let mut forward_counts = vec![0u32; n_count * 16];
        for (&src, &delay) in model.synapses.source_index.iter().zip(&model.synapses.delay) {
            let src = src as usize;
            if src < n_count {
                let d_idx = (delay.clamp(1, 16) - 1) as usize;
                forward_counts[src * 16 + d_idx] += 1;
            }
        }

        self.synapse_offsets = vec![0u32; n_count * 16 + 1];
        for i in 0..(n_count * 16) {
            self.synapse_offsets[i + 1] = self.synapse_offsets[i] + forward_counts[i];
        }

        self.synapse_indices_flat = vec![0; s_count];
        let mut current_forward_offsets = self.synapse_offsets.clone();
        for (i, (&src, &delay)) in model.synapses.source_index.iter().zip(&model.synapses.delay).enumerate() {
            let src = src as usize;
            if src < n_count {
                let d_idx = (delay.clamp(1, 16) - 1) as usize;
                let pos = &mut current_forward_offsets[src * 16 + d_idx];
                self.synapse_indices_flat[*pos as usize] = i;
                *pos += 1;
            }
        }

        // 2. Build Incoming Index (CSR)
        let mut incoming_counts = vec![0u32; n_count];
        for &tgt in &model.synapses.target_index {
            if (tgt as usize) < n_count {
                incoming_counts[tgt as usize] += 1;
            }
        }

        self.incoming_synapse_offsets = vec![0u32; n_count + 1];
        for i in 0..n_count {
            self.incoming_synapse_offsets[i + 1] = self.incoming_synapse_offsets[i] + incoming_counts[i];
        }

        self.incoming_synapse_indices_flat = vec![0; s_count];
        let mut current_incoming_offsets = self.incoming_synapse_offsets.clone();
        for (i, &tgt) in model.synapses.target_index.iter().enumerate() {
            if (tgt as usize) < n_count {
                let pos = &mut current_incoming_offsets[tgt as usize];
                self.incoming_synapse_indices_flat[*pos as usize] = i;
                *pos += 1;
            }
        }
    }

    fn propagate_sparse_delayed_spikes(&mut self, model: &mut BakedModel, previous_spikes: &[bool], history: &[Vec<bool>]) {
        // Pack spikes for SIMD-like processing if possible
        // (Note: full SIMD requires specialized crates, but we can optimize the loops)
        if self.synapse_offsets.is_empty() {
            self.rebuild_index_internal(model);
        }

        let n_count = model.neurons.len();

        // Optimized CSR traversal for spike propagation
        for d in 1..=16 {
            let spikes = if d == 1 {
                previous_spikes
            } else if d - 1 < history.len() {
                &history[d - 1]
            } else {
                continue;
            };

            let d_idx = (d - 1) as usize;

            for (src, &fired) in spikes.iter().enumerate() {
                if !fired || src >= n_count { continue; }

                let start = self.synapse_offsets[src * 16 + d_idx] as usize;
                let end = self.synapse_offsets[src * 16 + d_idx + 1] as usize;

                for i in start..end {
                    let syn_idx = self.synapse_indices_flat[i];
                    let target = model.synapses.target_index[syn_idx] as usize;
                    let gate = model.neurons.dendritic_gate[target];
                    if gate < 8 { continue; }

                    // Short-Term Plasticity (STP): modulate weight by available resources and calcium
                    let u_facilitation = model.synapses.stp_calcium[syn_idx];
                    let r_depression = model.synapses.stp_resources[syn_idx];

                    let stp_weight = ((model.synapses.weight[syn_idx] as i64 * r_depression as i64) >> 10) as i32;
                    let stp_weight = ((stp_weight as i64 * (SCALE as i64 + u_facilitation as i64)) >> 10) as i32;

                    // Consumption: firing uses resources and increases calcium
                    model.synapses.stp_resources[syn_idx] = (model.synapses.stp_resources[syn_idx] as i64 * model.config.stp_resource_decay >> 10) as i32;
                    model.synapses.stp_calcium[syn_idx] = (model.synapses.stp_calcium[syn_idx] as i64 + model.config.stp_calcium_recovery as i64).min(SCALE as i64) as i32;

                    let gated_weight = ((stp_weight as i64 * gate as i64) >> 10) as i32;
                    match model.synapses.compartment[syn_idx] {
                        Compartment::Proximal => {
                            model.neurons.proximal_potential[target] = model.neurons.proximal_potential[target].saturating_add(gated_weight);
                        }
                        Compartment::Distal => {
                            let attn_gated = ((gated_weight as i64 * model.neurons.distal_gate[target] as i64) >> 10) as i32;
                            model.neurons.distal_potential[target] = model.neurons.distal_potential[target].saturating_add(attn_gated);
                        }
                        Compartment::Apical => {
                            let attn_gated = ((gated_weight as i64 * model.neurons.apical_gate[target] as i64) >> 10) as i32;
                            model.neurons.apical_potential[target] = model.neurons.apical_potential[target].saturating_add(attn_gated);
                        }
                        Compartment::Basal => {
                            let attn_gated = ((gated_weight as i64 * model.neurons.basal_gate[target] as i64) >> 10) as i32;
                            model.neurons.basal_potential[target] = model.neurons.basal_potential[target].saturating_add(attn_gated);
                        }
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
        let target_activity = model.config.target_activity_level;
        let expert_masks = &self.expert_masks;

        let neurons = &mut model.neurons;

        // Parallelizing potential updates with a custom worker to avoid deep zip-chain nesting
        let mut spike_results = vec![false; n_count];
        let chunk_size = (n_count / rayon::current_num_threads()).max(64);

        let expert_masks_ptr = expert_masks.as_ptr() as usize;
        let neurons_ptr = neurons as *mut _ as usize;

        spike_results.par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let start_idx = chunk_idx * chunk_size;
                let end_idx = (start_idx + chunk_size).min(n_count);

                // SAFETY: We are accessing disjoint chunks of the SoA.
                // The update_single_neuron method needs &mut NeuronsSoA.
                // We use a local unsafe pointer to bypass borrow checker for parallel mutation of disjoint elements.
                unsafe {
                    let n_mut = &mut *(neurons_ptr as *mut genesis_core::NeuronsSoA);
                    let e_masks = if expert_masks_ptr == 0 { &[] } else {
                        std::slice::from_raw_parts(expert_masks_ptr as *const bool, expert_masks.len())
                    };

                    for i in start_idx..end_idx {
                        if current_tick < n_mut.next_update_tick[i] { continue; }
                        if !e_masks.is_empty() && !e_masks[i % e_masks.len()] { continue; }

                        let fired = self.update_single_neuron(i, n_mut, model, current_tick, noise_amp, ip_inc, ip_dec, target_activity);
                        chunk[i - start_idx] = fired;
                    }
                }
            });

        for i in 0..n_count { if spike_results[i] { new_spikes[i] = true; } }
    }
}

impl ComputeBackend for CpuBackend {
    fn name(&self) -> &'static str { "CpuBackend" }

    fn rebuild_index(&mut self, model: &BakedModel) {
        self.rebuild_index_internal(model);
    }

    fn execute_kernel(&mut self, kernel: crate::SimulationKernel, model: &mut BakedModel, ctx: &crate::KernelContext) -> Option<genesis_core::SpikeData> {
        let n_count = model.neurons.len();
        match kernel {
            crate::SimulationKernel::PropagateSynapses => {
                // Apply external inputs directly to proximal potential with dendritic gating
                for (i, &val) in ctx.external_inputs.iter().enumerate() {
                    if i < n_count {
                        let gate = model.neurons.dendritic_gate[i];
                        let gated_val = ((val as i64 * gate as i64) >> 10) as i32;
                        model.neurons.proximal_potential[i] = model.neurons.proximal_potential[i].saturating_add(gated_val);
                    }
                }
                self.propagate_sparse_delayed_spikes(model, ctx.previous_spikes, ctx.history);
                self.propagate_latent_spikes(model, ctx.previous_spikes);

                // Parallelized STP Recovery
                let synapses = &mut model.synapses;
                let r_rec = model.config.stp_resource_recovery;
                let c_dec = model.config.stp_calcium_decay;
                synapses.stp_resources.par_iter_mut()
                    .zip(synapses.stp_calcium.par_iter_mut())
                    .for_each(|(r, c)| {
                        *r = ((*r as i64 * r_rec + SCALE as i64) / 100) as i32;
                        *c = ((*c as i64 * c_dec) / 100) as i32;
                    });
                None
            }
            crate::SimulationKernel::UpdateMembranePotentials => {
                // Potential updates are handled within GenerateSpikes for CPU for efficiency
                None
            }
            crate::SimulationKernel::GenerateSpikes => {
                let mut new_spikes = vec![false; n_count];
                self.update_neuron_states(model, ctx.current_tick, &mut new_spikes);

                let active_indices: Vec<usize> = new_spikes.iter().enumerate()
                    .filter(|&(_, &s)| s).map(|(i, _)| i).collect();
                Some(genesis_core::SpikeData::Sparse(active_indices))
            }
            crate::SimulationKernel::ApplyModulation(_modulation) => {
                // Modulators are currently handled during weight updates
                None
            }
        }
    }

    fn day_phase(&mut self, model: &mut BakedModel, external_inputs: &[i32], previous_spikes: &[bool], history: &[Vec<bool>], current_tick: u32, modulation: genesis_core::NeuromodulationState) -> genesis_core::SpikeData {
        let ctx = crate::KernelContext { external_inputs, previous_spikes, history, current_tick, modulation };
        self.execute_kernel(crate::SimulationKernel::PropagateSynapses, model, &ctx);
        self.execute_kernel(crate::SimulationKernel::GenerateSpikes, model, &ctx).unwrap_or_default()
    }

    fn update_weights(&mut self, model: &mut BakedModel, previous_spikes: &[bool], current_spikes: &[bool], current_tick: u32, reward: Option<IValue>, history: &[Vec<bool>]) {
        self.update_weights_modulated(model, previous_spikes, current_spikes, current_tick, reward, genesis_core::NeuromodulationState::default(), history);
    }

    fn update_weights_modulated(&mut self, model: &mut BakedModel, previous_spikes: &[bool], current_spikes: &[bool], current_tick: u32, reward: Option<IValue>, modulation: genesis_core::NeuromodulationState, _history: &[Vec<bool>]) {
        if self.synapse_offsets.is_empty() {
            self.rebuild_index_internal(model);
        }

        let n_count = model.neurons.len();

        // Apply learning rate from model config
        self.plasticity_rule = Box::new(GsopRule { learning_rate: model.config.learning_rate });

        use std::collections::HashSet;
        let mut active_synapses = HashSet::new();

        // Sparse selection using CSR Forward Index
        for (src, &fired) in previous_spikes.iter().enumerate() {
            if fired && src < n_count {
                for d in 0..16 {
                    let start = self.synapse_offsets[src * 16 + d] as usize;
                    let end = self.synapse_offsets[src * 16 + d + 1] as usize;
                    for i in start..end {
                        active_synapses.insert(self.synapse_indices_flat[i]);
                    }
                }
            }
        }

        // Sparse selection using CSR Incoming Index
        for (tgt, &fired) in current_spikes.iter().enumerate() {
            if fired && tgt < n_count {
                let start = self.incoming_synapse_offsets[tgt] as usize;
                let end = self.incoming_synapse_offsets[tgt + 1] as usize;
                for i in start..end {
                    active_synapses.insert(self.incoming_synapse_indices_flat[i]);
                }
            }
        }

        let active_indices: Vec<usize> = active_synapses.into_iter().collect();

        // Parallel update of active synapses (Tagging Phase)
        let plasticity_rule = &self.plasticity_rule;
        let neurons = &model.neurons;
        let synapses = &mut model.synapses;
        let tags_ptr = synapses.tag.as_mut_ptr() as usize;
        let timers_ptr = synapses.tag_timer.as_mut_ptr() as usize;
        let volatility_ptr = synapses.volatility.as_mut_ptr() as usize;

        active_indices.into_par_iter().for_each(|i| {
            let src = synapses.source_index[i] as usize;
            let target = synapses.target_index[i] as usize;

            // Predictive Coding Error calculation for this synapse's target
            let prediction_error = neurons.proximal_potential[target] - neurons.distal_potential[target];

            let ctx = genesis_core::PlasticityContext {
                pre_spiked: previous_spikes[src],
                post_spiked: current_spikes[target],
                backprop_signal: neurons.backprop_signal[target],
                prediction_error,
                compartment: synapses.compartment[i],
                reward,
                neuromodulation: modulation,
                pre_last_spike: neurons.last_spike_tick[src],
                post_last_spike: neurons.last_spike_tick[target],
                current_tick,
                post_index: target,
                neurons,
                config: &model.config,
            };

            // SAFETY: HashSet ensures unique indices, so no data races.
            unsafe {
                let tag_ref = &mut *(tags_ptr as *mut IValue).add(i);
                let timer_ref = &mut *(timers_ptr as *mut u16).add(i);
                let volatility_ref = &mut *(volatility_ptr as *mut u8).add(i);
                plasticity_rule.tag(tag_ref, timer_ref, volatility_ref, &ctx);
            }
        });

        // Synaptic Tagging and Capture (STC): Capture Phase
        // Convert tags to weights if global PRPs (Plasticity-Related Proteins) are present.
        // PRPs are triggered by reward or high surprise.
        let prp_present = reward.is_some() || (modulation.noradrenaline > 500);

        if prp_present {
            synapses.weight.par_iter_mut()
                .zip(synapses.tag.par_iter_mut())
                .zip(synapses.tag_timer.par_iter_mut())
                .for_each(|((w, t), timer)| {
                    if *timer > 0 {
                        let capture_strength = if reward.is_some() { 2 } else { 1 };
                        let delta = *t * capture_strength;
                        let old_w = *w;
                        *w = w.saturating_add(delta);
                        genesis_core::plasticity::clamp_and_preserve_sign_with_limit(w, old_w, model.config.weight_clamp_limit);

                        // Consolidation: tag is used up
                        *t = 0;
                        *timer = 0;
                    }
                });
        } else {
            // Tag Decay
            synapses.tag_timer.par_iter_mut()
                .zip(synapses.tag.par_iter_mut())
                .for_each(|(timer, t)| {
                    if *timer > 0 {
                        *timer -= 1;
                        if *timer == 0 { *t = 0; }
                    }
                });
        }
    }

    fn structural_plasticity(&mut self, model: &mut BakedModel, reward: Option<IValue>, history: &[Vec<bool>]) {
        self.structural_plasticity_with_surprise(model, reward, history, &[])
    }

    fn structural_plasticity_with_surprise(&mut self, model: &mut BakedModel, reward: Option<IValue>, history: &[Vec<bool>], block_surprise: &[f32]) {
        prune_synapses(&mut model.synapses, self.structural_config.prune_threshold);
        model.synapses.shrink_to_fit();

        // Structure changed -> Index must be rebuilt next tick
        self.synapse_offsets.clear();

        // SNNaS: Evolutionary mutation
        if let Some(r) = reward {
            self.optimizer.mutate_with_surprise(&mut model.synapses, &mut model.neurons, r, history, model.config.max_synapses, block_surprise);

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

        let max_neuron_sum = model.config.homeostatic_scaling_limit;
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
