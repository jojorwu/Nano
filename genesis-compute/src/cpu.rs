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
    /// Integrated Single Neuron State Update (Thread-Safe using raw pointers)
    fn update_single_neuron_ffi(&self, i: usize, n: &genesis_core::NeuronsFFI, model: &BakedModel, current_tick: u32, noise_amp: i32, ip_inc: i32, ip_dec: i32, target_activity: i32) -> bool {
        unsafe {
            let attn_dist = ((*n.distal_potential.add(i) as i64 * *n.distal_gate.add(i) as i64) >> 10) as i32;
            let attn_apical = ((*n.apical_potential.add(i) as i64 * *n.apical_gate.add(i) as i64) >> 10) as i32;
            let attn_basal = ((*n.basal_potential.add(i) as i64 * *n.basal_gate.add(i) as i64) >> 10) as i32;

            // Multi-Segment Dendritic Integration
            let mut total_distal = attn_dist;
            let seg_offset = i * 4;
            for s in 0..4 {
                 let seg_pot = *n.segment_potentials.add(seg_offset + s);
                 let seg_gate = *n.segment_gates.add(seg_offset + s);
                 let gated = (seg_pot as i64 * seg_gate as i64 >> 10) as i32;
                 if gated > 500 { // Segment threshold
                      total_distal = total_distal.saturating_add(SCALE); // Segmental Ignition
                 }
            }

            let current_pot = calculate_membrane_potential(
                *n.potential.add(i), *n.proximal_potential.add(i), total_distal, attn_apical, attn_basal,
                *n.gate_threshold.add(i), *n.liquid_current.add(i), *n.decay.add(i), noise_amp,
                *n.adaptation_current.add(i)
            );

            // Refractory Multiplier: exponential threshold increase during refractory period
            let refr_timer = *n.refractory_timer.add(i);
            let refr_mult = if refr_timer > 0 {
                 // Clamp timer to avoid extreme shifts (max 16x threshold)
                 1 + (1 << refr_timer.min(4))
            } else {
                 1
            };
            let mut effective_threshold = (*n.threshold.add(i)).saturating_mul(refr_mult);

            let theta = if model.config.physics.theta_rhythm {
                 ((current_tick as f32 * model.config.physics.theta_frequency).sin() * 200.0) as i32
            } else { 0 };
            effective_threshold = effective_threshold.saturating_add(theta);

            let fired = current_pot >= effective_threshold;

            if fired {
                *n.potential.add(i) = 0;
                *n.refractory_timer.add(i) = model.config.physics.default_refractory_ticks;
                *n.last_spike_tick.add(i) = current_tick;
                *n.backprop_signal.add(i) = SCALE;
                *n.action_potential.add(i) = SCALE;

                // IP (Intrinsic Plasticity): Increment threshold upon firing
                *n.threshold.add(i) = (*n.threshold.add(i)).saturating_add(ip_inc);

                // EMA Activity tracking
                let alpha = model.config.physics.activity_ema_alpha as i32;
                *n.activity_ema.add(i) = ((*n.activity_ema.add(i) as i64 * alpha as i64 + (1000 - alpha) as i64 * 100) / 1000) as i32;

                *n.adaptation_current.add(i) = (*n.adaptation_current.add(i)).saturating_add(100);
            } else {
                *n.potential.add(i) = current_pot;

                if *n.refractory_timer.add(i) > 0 {
                     *n.refractory_timer.add(i) -= 1;
                }

                // IP Decay: threshold slowly returns to baseline
                if *n.threshold.add(i) > *n.base_threshold.add(i) {
                     *n.threshold.add(i) = (*n.threshold.add(i)).saturating_sub(ip_dec);
                }

                let smbp_decay = model.config.physics.smbp_decay;
                *n.backprop_signal.add(i) = ((*n.backprop_signal.add(i) as i64 * smbp_decay) >> 10) as i32;
                *n.action_potential.add(i) = ((*n.action_potential.add(i) as i64 * smbp_decay) >> 10) as i32;

                let alpha = model.config.physics.activity_ema_alpha as i32;
                *n.activity_ema.add(i) = ((*n.activity_ema.add(i) as i64 * alpha as i64) / 1000) as i32;

                *n.adaptation_current.add(i) = (*n.adaptation_current.add(i) * 95) / 100;

                let seg_offset = i * 4;
                for s in 0..4 {
                     *n.segment_potentials.add(seg_offset + s) = (*n.segment_potentials.add(seg_offset + s) * 90) / 100;
                }
            }

            if fired {
                *n.specialization_score.add(i) = *n.specialization_score.add(i) * model.config.physics.specialization_decay + 0.1;
            }

            let activity_ema = *n.activity_ema.add(i);
            let error = activity_ema - target_activity;

            // 1. Threshold-based Homeostasis (Intrinsic Plasticity)
            let homeo_rate = if error.abs() > target_activity { 2 } else { 1 };
            if error > 0 {
                *n.base_threshold.add(i) = (*n.base_threshold.add(i)).saturating_add(homeo_rate);
            } else if error < 0 && *n.base_threshold.add(i) > model.config.physics.default_threshold / 2 {
                *n.base_threshold.add(i) = (*n.base_threshold.add(i)).saturating_sub(1);
            }

            // 2. Metaplasticity-based Homeostasis (BCM Rule)
            // Adjust plasticity_gate based on activity error to maintain stable firing.
            // If the neuron is hyper-active, reduce learning capacity to prevent runaway LTP.
            // If the neuron is under-active, increase learning capacity to allow reorganization.
            let p_gate = *n.plasticity_gate.add(i);
            if error > target_activity {
                // High activity -> Reduce plasticity
                *n.plasticity_gate.add(i) = p_gate.saturating_sub(2);
            } else if error < -(target_activity / 2) {
                // Low activity -> Increase plasticity
                *n.plasticity_gate.add(i) = p_gate.saturating_add(5).min(SCALE);
            }
            // next_update_tick and update_interval are not in NeuronsFFI yet, let's fix that if needed.
            // For now assume sequential update
            fired
        }
    }

    pub fn rebuild_index_internal(&mut self, model: &BakedModel) {
        let n_count = model.neurons.len();
        let s_count = model.synapses.len();

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

    fn apply_plasticity_tagging(&self, model: &mut BakedModel, active_indices: &[usize], previous_spikes: &[bool], current_spikes: &[bool], current_tick: u32, reward: Option<IValue>, modulation: genesis_core::NeuromodulationState) {
        let plasticity_rule = &self.plasticity_rule;
        let neurons = &model.neurons;
        let synapses = &mut model.synapses;
        let tags_ptr = synapses.tag.as_mut_ptr() as usize;
        let timers_ptr = synapses.tag_timer.as_mut_ptr() as usize;
        let volatility_ptr = synapses.volatility.as_mut_ptr() as usize;
        let causality_ptr = synapses.causality_index.as_mut_ptr() as usize;

        active_indices.into_par_iter().for_each(|&i| {
            let src = synapses.source_index[i] as usize;
            let target = synapses.target_index[i] as usize;
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
            unsafe {
                let tag_ref = &mut *(tags_ptr as *mut IValue).add(i);
                let timer_ref = &mut *(timers_ptr as *mut u16).add(i);
                let volatility_ref = &mut *(volatility_ptr as *mut u8).add(i);
                let causality_ref = &mut *(causality_ptr as *mut u8).add(i);
                plasticity_rule.tag(tag_ref, timer_ref, volatility_ref, causality_ref, &ctx);
            }
        });
    }

    fn apply_plasticity_capture(&self, model: &mut BakedModel, reward: Option<IValue>, modulation: genesis_core::NeuromodulationState) {
        let synapses = &mut model.synapses;
        let prp_present = reward.is_some() || (modulation.noradrenaline > 500);

        if prp_present {
            synapses.weight.par_iter_mut()
                .zip(synapses.tag.par_iter_mut())
                .zip(synapses.tag_timer.par_iter_mut())
                .zip(synapses.causality_index.par_iter_mut())
                .for_each(|(((w, t), timer), causality)| {
                    if *timer > 0 {
                        let causal_boost = if *causality > 128 { 2 } else { 1 };
                        let capture_strength = if reward.is_some() { 2 * causal_boost } else { 1 * causal_boost };
                        let delta = *t * capture_strength;
                        let old_w = *w;
                        *w = w.saturating_add(delta);
                        genesis_core::plasticity::clamp_and_preserve_sign_with_limit(w, old_w, model.config.plasticity.weight_clamp_limit);
                        *t = 0;
                        *timer = 0;
                    }
                });
        } else {
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

    fn apply_morphogenesis(&self, model: &mut BakedModel, reward: Option<IValue>) {
        let n_count_morpho = model.neurons.len();
        if reward.unwrap_or(0) > 500 {
             use rand::Rng;
             let mut rng = rand::thread_rng();
             let mut successful_blocks = Vec::new();
             for &bid in &model.neurons.block_id {
                  if !successful_blocks.contains(&bid) && rng.gen_bool(0.1) {
                       successful_blocks.push(bid);
                  }
             }
             if !successful_blocks.is_empty() {
                  for i in 0..n_count_morpho {
                       if model.neurons.specialization_score[i] < 0.01 && rng.gen_bool(0.05) {
                            let new_bid = successful_blocks[rng.gen_range(0..successful_blocks.len())];
                            model.neurons.block_id[i] = new_bid;
                            model.neurons.specialization_score[i] = 0.1;
                       }
                  }
             }
        }
    }

    fn apply_evolutionary_mutations(&self, model: &mut BakedModel, reward: Option<IValue>, history: &[Vec<bool>], block_surprise: &[f32]) {
        if let Some(r) = reward {
            self.optimizer.mutate_with_surprise(&mut model.synapses, &mut model.neurons, r, history, model.config.plasticity.max_synapses, block_surprise);

            if r > model.config.plasticity.neurogenesis_reward_threshold && model.neurons.len() < model.config.plasticity.max_neurons {
                let grow_size = (model.neurons.len() / 20).max(1);
                model.neurons.grow(grow_size);
            }

            if model.config.plasticity.metaplasticity_enabled {
                if r.abs() < 10 {
                    model.config.plasticity.learning_rate = (model.config.plasticity.learning_rate as i64 * 900 / 1024) as i32;
                } else if r.abs() > 500 {
                    model.config.plasticity.learning_rate = (model.config.plasticity.learning_rate as i64 * 1100 / 1024) as i32;
                }
            }
        }
    }

    fn apply_homeostatic_weight_scaling(&self, model: &mut BakedModel) {
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

        let max_neuron_sum = model.config.plasticity.homeostatic_scaling_limit;
        let synapses = &mut model.synapses;

        synapses.weight.par_iter_mut().enumerate().for_each(|(i, w)| {
            let target = synapses.target_index[i] as usize;
            if target < n_count && sum_weights[target] > max_neuron_sum {
                let scale_factor = (max_neuron_sum << 10) / sum_weights[target];
                *w = ((*w as i64 * scale_factor) >> 10) as i32;
            }
        });
    }

    fn propagate_sparse_delayed_spikes(&mut self, model: &mut BakedModel, previous_spikes: &[bool], history: &[Vec<bool>]) {
        if self.synapse_offsets.is_empty() {
            self.rebuild_index_internal(model);
        }
        let n_count = model.neurons.len();

        // Efficiency: Bitset-based iteration to reduce branch mispredictions
        for d in 1..=16 {
            let spikes = if d == 1 { previous_spikes } else if d - 1 < history.len() { &history[d - 1] } else { continue; };
            let d_idx = (d - 1) as usize;

            // Pack spikes into u64s for faster traversal
            let packed_len = (spikes.len() + 63) / 64;
            for block_idx in 0..packed_len {
                 let mut mask = 0u64;
                 let base = block_idx * 64;
                 for bit in 0..64 {
                      if base + bit < spikes.len() && spikes[base + bit] {
                           mask |= 1 << bit;
                      }
                 }
                 if mask == 0 { continue; }

                 // Process set bits using trailing zero count
                 let mut temp_mask = mask;
                 while temp_mask != 0 {
                      let bit_idx = temp_mask.trailing_zeros() as usize;
                      let src = base + bit_idx;
                      temp_mask &= !(1 << bit_idx);

                      if src >= n_count { continue; }
                      let start = self.synapse_offsets[src * 16 + d_idx] as usize;
                      let end = self.synapse_offsets[src * 16 + d_idx + 1] as usize;
                      for i in start..end {
                          let syn_idx = self.synapse_indices_flat[i];
                          let target = model.synapses.target_index[syn_idx] as usize;
                          let gate = model.neurons.dendritic_gate[target];
                          if gate < 8 { continue; }
                          let u_facilitation = model.synapses.stp_calcium[syn_idx];
                          let r_depression = model.synapses.stp_resources[syn_idx];
                          let stp_weight = ((model.synapses.weight[syn_idx] as i64 * r_depression as i64) >> 10) as i32;
                          let stp_weight = ((stp_weight as i64 * (SCALE as i64 + u_facilitation as i64)) >> 10) as i32;
                          model.synapses.stp_resources[syn_idx] = (model.synapses.stp_resources[syn_idx] as i64 * model.config.plasticity.stp_resource_decay >> 10) as i32;
                          model.synapses.stp_calcium[syn_idx] = (model.synapses.stp_calcium[syn_idx] as i64 + model.config.plasticity.stp_calcium_recovery as i64).min(SCALE as i64) as i32;
                          let gated_weight = ((stp_weight as i64 * gate as i64) >> 10) as i32;
                          match model.synapses.compartment[syn_idx] {
                              Compartment::Proximal => { model.neurons.proximal_potential[target] = model.neurons.proximal_potential[target].saturating_add(gated_weight); }
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
                              Compartment::Custom(_) => { model.neurons.proximal_potential[target] = model.neurons.proximal_potential[target].saturating_add(gated_weight); }
                          }
                      }
                 }
            }
        }
    }

    fn propagate_latent_spikes(&self, model: &mut BakedModel, previous_spikes: &[bool]) {
        if let Some(ref latent) = model.synapses.latent_matrix {
            let mut latent_state = vec![0i32; latent.rank];
            let active_indices: Vec<usize> = previous_spikes.iter().enumerate().filter(|&(_, &s)| s).map(|(i, _)| i).collect();
            if active_indices.is_empty() { return; }
            for i in active_indices {
                let offset = i * latent.rank;
                for r in 0..latent.rank { latent_state[r] = latent_state[r].saturating_add(latent.u[offset + r]); }
            }
            for j in 0..model.neurons.len() {
                let gate = model.neurons.dendritic_gate[j];
                let mut sum = 0i64;
                for r in 0..latent.rank {
                    let weight = latent.v[r * model.neurons.len() + j];
                    sum += latent_state[r] as i64 * weight as i64;
                }
                let contribution = ((sum * gate as i64) >> 20) as i32;
                model.neurons.distal_potential[j] = model.neurons.distal_potential[j].saturating_add(contribution);
            }
        }
    }


    fn update_neuron_states_range(&self, model: &mut BakedModel, current_tick: u32, new_spikes: &mut [bool], range: std::ops::Range<usize>) {
        let n_count = model.neurons.len();
        let range_len = range.end - range.start;
        let ip_inc = model.config.plasticity.ip_increment;
        let ip_dec = model.config.plasticity.ip_decay;
        let noise_amp = model.config.physics.noise_amplitude;
        let target_activity = model.config.physics.target_activity_level;
        let expert_masks = &self.expert_masks;

        // Use FFI view to safely share pointers across threads
        let n_ffi = model.neurons.as_ffi();
        let next_update_ptr = model.neurons.next_update_tick.as_mut_ptr() as usize;
        let is_remote_ptr = model.neurons.is_remote.as_ptr() as usize;

        let mut spike_results = vec![false; n_count];
        let chunk_size = (range_len / rayon::current_num_threads()).max(64);
        let expert_masks_ptr = expert_masks.as_ptr() as usize;

        spike_results.par_chunks_mut(chunk_size).enumerate().for_each(|(chunk_idx, chunk)| {
            let start_idx = range.start + chunk_idx * chunk_size;
            let end_idx = (start_idx + chunk_size).min(range.end);
            unsafe {
                let e_masks = if expert_masks_ptr == 0 { &[] } else { std::slice::from_raw_parts(expert_masks_ptr as *const bool, expert_masks.len()) };
                let next_up = next_update_ptr as *mut u32;
                let is_rem = is_remote_ptr as *const u8;
                for i in start_idx..end_idx {
                    if current_tick < *next_up.add(i) { continue; }
                    if *is_rem.add(i) != 0 { continue; }
                    if !e_masks.is_empty() && !e_masks[i % e_masks.len()] { continue; }

                    let fired = self.update_single_neuron_ffi(i, &n_ffi, model, current_tick, noise_amp, ip_inc, ip_dec, target_activity);
                    chunk[i - start_idx] = fired;

                    // Assumptions about update interval - for now sequential
                    *next_up.add(i) = current_tick + 1;
                }
            }
        });
        for i in range { if spike_results[i] { new_spikes[i] = true; } }
    }
}

impl ComputeBackend for CpuBackend {
    fn name(&self) -> &'static str { "CpuBackend" }
    fn rebuild_index(&mut self, model: &BakedModel) { self.rebuild_index_internal(model); }

    fn execute_kernel(&mut self, kernel: crate::SimulationKernel, model: &mut BakedModel, ctx: &crate::KernelContext) -> Option<genesis_core::SpikeData> {
        let n_count = model.neurons.len();
        self.execute_kernel_range(kernel, model, ctx, 0..n_count)
    }

    fn execute_kernel_range(&mut self, kernel: crate::SimulationKernel, model: &mut BakedModel, ctx: &crate::KernelContext, range: std::ops::Range<usize>) -> Option<genesis_core::SpikeData> {
        let n_count = model.neurons.len();
        match kernel {
            crate::SimulationKernel::PropagateSynapses => {
                for (i, &val) in ctx.external_inputs.iter().enumerate() {
                    if i >= range.start && i < range.end && i < n_count {
                        let gate = model.neurons.dendritic_gate[i];
                        let gated_val = ((val as i64 * gate as i64) >> 10) as i32;
                        model.neurons.proximal_potential[i] = model.neurons.proximal_potential[i].saturating_add(gated_val);
                    }
                }
                if let Some(td) = ctx.top_down_modulation {
                    for (i, &boost) in td.iter().enumerate() {
                        if i >= range.start && i < range.end && i < n_count {
                            model.neurons.proximal_potential[i] = model.neurons.proximal_potential[i].saturating_add(boost);
                        }
                    }
                }
                self.propagate_sparse_delayed_spikes(model, ctx.previous_spikes, ctx.history);
                self.propagate_latent_spikes(model, ctx.previous_spikes);
                let synapses = &mut model.synapses;
                let r_rec = model.config.plasticity.stp_resource_recovery;
                let c_dec = model.config.plasticity.stp_calcium_decay;
                synapses.stp_resources.par_iter_mut().zip(synapses.stp_calcium.par_iter_mut()).for_each(|(r, c)| {
                    *r = ((*r as i64 * r_rec + SCALE as i64) / 100) as i32;
                    *c = ((*c as i64 * c_dec) / 100) as i32;
                });
                None
            }
            crate::SimulationKernel::UpdateMembranePotentials => { None }
            crate::SimulationKernel::GenerateSpikes => {
                let mut new_spikes = vec![false; n_count];
                self.update_neuron_states_range(model, ctx.current_tick, &mut new_spikes, range.clone());
                let active_indices: Vec<usize> = new_spikes.iter().enumerate()
                    .filter(|&(i, &s)| s && i >= range.start && i < range.end)
                    .map(|(i, _)| i).collect();
                Some(genesis_core::SpikeData::Sparse(active_indices))
            }
            crate::SimulationKernel::ApplyModulation(_modulation) => { None }
        }
    }

    fn day_phase(&mut self, model: &mut BakedModel, external_inputs: &[i32], previous_spikes: &[bool], history: &[Vec<bool>], current_tick: u32, modulation: genesis_core::NeuromodulationState) -> genesis_core::SpikeData {
        let ctx = crate::KernelContext { external_inputs, previous_spikes, history, current_tick, modulation, top_down_modulation: None };
        self.execute_kernel(crate::SimulationKernel::PropagateSynapses, model, &ctx);
        self.execute_kernel(crate::SimulationKernel::GenerateSpikes, model, &ctx).unwrap_or_default()
    }
    fn update_weights(&mut self, model: &mut BakedModel, previous_spikes: &[bool], current_spikes: &[bool], current_tick: u32, reward: Option<IValue>, history: &[Vec<bool>]) {
        self.update_weights_modulated(model, previous_spikes, current_spikes, current_tick, reward, genesis_core::NeuromodulationState::default(), history);
    }
    fn update_weights_modulated(&mut self, model: &mut BakedModel, previous_spikes: &[bool], current_spikes: &[bool], current_tick: u32, reward: Option<IValue>, modulation: genesis_core::NeuromodulationState, _history: &[Vec<bool>]) {
        if self.synapse_offsets.is_empty() { self.rebuild_index_internal(model); }
        let n_count = model.neurons.len();
        self.plasticity_rule = Box::new(GsopRule { learning_rate: model.config.plasticity.learning_rate });
        use std::collections::HashSet;
        let mut active_synapses = HashSet::new();
        for (src, &fired) in previous_spikes.iter().enumerate() {
            if fired && src < n_count {
                for d in 0..16 {
                    let start = self.synapse_offsets[src * 16 + d] as usize;
                    let end = self.synapse_offsets[src * 16 + d + 1] as usize;
                    for i in start..end { active_synapses.insert(self.synapse_indices_flat[i]); }
                }
            }
        }
        for (tgt, &fired) in current_spikes.iter().enumerate() {
            if fired && tgt < n_count {
                let start = self.incoming_synapse_offsets[tgt] as usize;
                let end = self.incoming_synapse_offsets[tgt + 1] as usize;
                for i in start..end { active_synapses.insert(self.incoming_synapse_indices_flat[i]); }
            }
        }
        let active_indices: Vec<usize> = active_synapses.into_iter().collect();
        self.apply_plasticity_tagging(model, &active_indices, previous_spikes, current_spikes, current_tick, reward, modulation);
        self.apply_plasticity_capture(model, reward, modulation);
    }
    fn structural_plasticity(&mut self, model: &mut BakedModel, reward: Option<IValue>, history: &[Vec<bool>]) {
        self.structural_plasticity_with_surprise(model, reward, history, &[])
    }
    fn structural_plasticity_with_surprise(&mut self, model: &mut BakedModel, reward: Option<IValue>, history: &[Vec<bool>], block_surprise: &[f32]) {
        self.apply_morphogenesis(model, reward);
        prune_synapses(&mut model.synapses, &model.neurons, self.structural_config.prune_threshold);

        // Metabolic Culling: Reset "dead" neurons
        let culled = genesis_core::plasticity::cull_inactive_neurons(&mut model.neurons, 5);
        if !culled.is_empty() {
             log::debug!("Metabolic Culling: {} neurons reset", culled.len());
        }

        model.synapses.shrink_to_fit();
        self.synapse_offsets.clear();
        self.apply_evolutionary_mutations(model, reward, history, block_surprise);
        self.apply_homeostatic_weight_scaling(model);
    }
}
