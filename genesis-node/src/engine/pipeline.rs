use crate::{SimulationEngine};
use genesis_core::{SpikeData};
use std::time::Instant;

pub trait PipelineStage: Send + Sync {
    fn name(&self) -> &str;
    fn execute(&mut self, engine: &mut SimulationEngine, context: &mut PipelineContext);
}

pub struct PipelineContext {
    pub tick: u32,
    pub external_inputs: Vec<i32>,
    pub reward: Option<i32>,
    pub normalized_reward: Option<i32>,
    pub surprise: i32,
    pub start_time: Instant,
    pub events: Vec<genesis_core::GlobalEvent>,
    pub top_down_data: Option<Vec<i32>>,
    /// Blackboard for inter-stage communication
    pub blackboard: std::collections::HashMap<String, f32>,
}

pub struct SimulationPipeline {
    pub stages: Vec<Box<dyn PipelineStage>>,
}

impl SimulationPipeline {
    pub fn new(settings: &crate::SimulationSettings) -> Self {
        Self {
            stages: vec![
                Box::new(InputStage),
                Box::new(PropagationStage),
                Box::new(ThinkingStage),
                Box::new(ObservationStage),
                Box::new(NeuromodulationStage),
                Box::new(NormalizationStage),
                Box::new(StructuralPlasticityStage::new(settings.night_phase_interval)),
                Box::new(AnomalyDetectionStage),
            ],
        }
    }

    pub fn from_config(active_stages: &[String], settings: &crate::SimulationSettings) -> Self {
        let mut stages: Vec<Box<dyn PipelineStage>> = Vec::new();
        for name in active_stages {
            match name.as_str() {
                "input" => stages.push(Box::new(InputStage)),
                "propagation" => stages.push(Box::new(PropagationStage)),
                "thinking" => stages.push(Box::new(ThinkingStage)),
                "observation" => stages.push(Box::new(ObservationStage)),
                "neuromodulation" => stages.push(Box::new(NeuromodulationStage)),
                "normalization" => stages.push(Box::new(NormalizationStage)),
                "load_balancing" => stages.push(Box::new(LoadBalancingStage)),
                "structural_plasticity" => stages.push(Box::new(StructuralPlasticityStage::new(settings.night_phase_interval))),
                "anomaly_detection" => stages.push(Box::new(AnomalyDetectionStage)),
                _ => log::warn!("Unknown pipeline stage: {}", name),
            }
        }
        Self { stages }
    }

    pub fn execute(&mut self, engine: &mut SimulationEngine, context: &mut PipelineContext) {
        let mut i = 0;
        while i < self.stages.len() {
            let stage = &mut self.stages[i];
            stage.execute(engine, context);

            // Dynamic Gating Logic
            // If high surprise is detected, certain stages might be repeated or skipped.
            if stage.name() == "propagation" && context.surprise > 1800 {
                 // High surprise during propagation: re-run propagation once to stabilize
                 if !context.blackboard.contains_key("prop_retry") {
                      context.blackboard.insert("prop_retry".to_string(), 1.0);
                      continue; // Execute the same stage index again
                 }
            }
            i += 1;
        }
    }
}

pub struct InputStage;
impl PipelineStage for InputStage {
    fn name(&self) -> &str { "input" }
    fn execute(&mut self, engine: &mut SimulationEngine, context: &mut PipelineContext) {
        engine.reset_potential_buffers();

        // Prepare merged inputs logic
        let n_count = engine.model.neurons.len();
        engine.state.merged_inputs_buffer.fill(0);
        let merge_len = context.external_inputs.len().min(n_count);
        engine.state.merged_inputs_buffer[..merge_len].copy_from_slice(&context.external_inputs[..merge_len]);

        {
            let mut remote_spikes = engine.remote_spike_queue.lock().unwrap();
            for &idx in remote_spikes.iter() {
                if idx < n_count {
                    engine.state.merged_inputs_buffer[idx] = engine.state.merged_inputs_buffer[idx].saturating_add(genesis_core::SCALE);
                }
            }
            remote_spikes.clear();
        }

        engine.input_bus.clear_mut();
        engine.modules.on_tick(&engine.input_bus, &engine.state.previous_spikes, context.tick);

        // Pre-calculate Top-Down Modulation for WGPU efficiency
        let mut top_down_data = vec![0i32; n_count];
        let mut has_top_down = false;
        for m in &engine.modules.modules {
            if let Some(hier) = m.as_any().downcast_ref::<genesis_core::HierarchicalModule>() {
                if hier.enabled {
                    for (&high_layer, low_layers) in &hier.hierarchy_map {
                        let high_act = hier.layer_activity.get(&high_layer).cloned().unwrap_or(0.0);
                        if high_act > 0.1 {
                            has_top_down = true;
                            let boost = (high_act * hier.top_down_gain as f32) as i32;
                            for &low_layer in low_layers {
                                if let Some(target_neurons) = hier.layer_to_neurons.get(&low_layer) {
                                    for &idx in target_neurons {
                                        if (idx as usize) < n_count {
                                            top_down_data[idx as usize] = top_down_data[idx as usize].saturating_add(boost);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if has_top_down {
            context.top_down_data = Some(top_down_data);
        }

        engine.finalize_potentials_from_bus();
    }
}

pub struct PropagationStage;
impl PipelineStage for PropagationStage {
    fn name(&self) -> &str { "propagation" }
    fn execute(&mut self, engine: &mut SimulationEngine, context: &mut PipelineContext) {
        let n_count = engine.model.neurons.len();
        let full_history = engine.reconstruct_history(16);


        let spike_data = if let Some(ref mut secondary) = engine.secondary_backend {
            // Heterogeneous Compute: split the neural blocks between backends
            // Primary backend (typically WGPU/GPU) handles the bulk of neurons
            // Secondary backend (typically CPU) handles a specific range (e.g. the last 20%)
            let split_point = (n_count * 80) / 100;

            let kernel_ctx = genesis_compute::KernelContext {
                external_inputs: &engine.state.merged_inputs_buffer,
                previous_spikes: &engine.state.previous_spikes,
                history: &full_history,
                current_tick: context.tick,
                modulation: engine.state.global_modulators,
                top_down_modulation: context.top_down_data.as_deref(),
            };

            // 1. Concurrent Synapse Propagation (Currently simplified to full on primary)
            engine.backend.execute_kernel(genesis_compute::SimulationKernel::PropagateSynapses, &mut engine.model, &kernel_ctx);

            // 2. Parallel Membrane Potential & Spike Generation
            // In a production environment, we would use the is_remote flags to filter neurons per-backend.
            // For this implementation, we use a simple range-based split.
            let primary_range = 0..split_point;
            let secondary_range = split_point..n_count;

            let primary_spikes = engine.backend.execute_kernel_range(
                genesis_compute::SimulationKernel::GenerateSpikes,
                &mut engine.model,
                &kernel_ctx,
                primary_range
            );

            let secondary_spikes = secondary.execute_kernel_range(
                genesis_compute::SimulationKernel::GenerateSpikes,
                &mut engine.model,
                &kernel_ctx,
                secondary_range
            );

            // 3. Merge results
            let mut merged_indices = Vec::new();
            if let Some(genesis_core::SpikeData::Sparse(indices)) = primary_spikes {
                merged_indices.extend(indices);
            }
            if let Some(genesis_core::SpikeData::Sparse(indices)) = secondary_spikes {
                merged_indices.extend(indices);
            }
            genesis_core::SpikeData::Sparse(merged_indices)
        } else {
            let kernel_ctx = genesis_compute::KernelContext {
                external_inputs: &engine.state.merged_inputs_buffer,
                previous_spikes: &engine.state.previous_spikes,
                history: &full_history,
                current_tick: context.tick,
                modulation: engine.state.global_modulators,
                top_down_modulation: context.top_down_data.as_deref(),
            };
            engine.backend.execute_kernel(genesis_compute::SimulationKernel::UpdateMembranePotentials, &mut engine.model, &kernel_ctx);
            engine.backend.execute_kernel(genesis_compute::SimulationKernel::PropagateSynapses, &mut engine.model, &kernel_ctx);
            engine.backend.execute_kernel(genesis_compute::SimulationKernel::GenerateSpikes, &mut engine.model, &kernel_ctx).unwrap_or(SpikeData::Sparse(vec![]))
        };

        engine.state.current_spikes_buffer.fill(false);
        match &spike_data {
            SpikeData::Sparse(indices) => {
                for &idx in indices { if idx < n_count { engine.state.current_spikes_buffer[idx] = true; } }
            }
            SpikeData::Dense(mask) => {
                for i in 0..n_count {
                    if (mask[i / 8] >> (i % 8)) & 1 == 1 { engine.state.current_spikes_buffer[i] = true; }
                }
            }
            SpikeData::BitPacked(packed) => {
                for i in 0..n_count {
                    if (packed[i / 64] >> (i % 64)) & 1 == 1 { engine.state.current_spikes_buffer[i] = true; }
                }
            }
            _ => {}
        }

        // Ensure BitPacked history is always available for subsequent stages (Titan learning)
        let current_bitpacked = if let SpikeData::BitPacked(_) = spike_data {
            spike_data
        } else {
            let mut packed = vec![0u64; (n_count + 63) / 64];
            for (i, &s) in engine.state.current_spikes_buffer.iter().enumerate() {
                if s { packed[i / 64] |= 1 << (i % 64); }
            }
            SpikeData::BitPacked(packed)
        };

        engine.state.spikes_history[engine.state.history_ptr] = current_bitpacked;

        // Continuous Synaptic Pruning: every 50 ticks, prune extremely weak synapses
        if context.tick % 50 == 0 {
             let mut i = 0;
             while i < engine.model.synapses.len() {
                 if engine.model.synapses.weight[i].abs() < 5 {
                     engine.model.synapses.remove(i);
                     // Clear index to force rebuild
                     engine.backend.rebuild_index(&engine.model);
                 } else {
                     i += 1;
                 }
             }
        }
    }
}

pub struct ThinkingStage;
impl PipelineStage for ThinkingStage {
    fn name(&self) -> &str { "thinking" }
    fn execute(&mut self, engine: &mut SimulationEngine, context: &mut PipelineContext) {
        let initial_spike_data = engine.state.spikes_history[engine.state.history_ptr].clone();

        let n_count = engine.model.neurons.len();
        let mut think_ticks = 0;
        for m in &engine.modules.modules {
            if m.name() == "think" {
                if let Some(tm) = m.as_any().downcast_ref::<genesis_core::ThinkModule>() {
                    if tm.active { think_ticks = tm.extra_ticks; }
                } else if let Ok(state) = bincode::deserialize::<genesis_core::ThinkModule>(&m.get_state()) {
                     if state.active { think_ticks = state.extra_ticks; }
                }
                break;
            }
        }

        let final_spike_data = if think_ticks > 0 {
            let mut current_dense = vec![false; n_count];
            match &initial_spike_data {
                SpikeData::Sparse(indices) => { for &i in indices { if i < n_count { current_dense[i] = true; } } }
                SpikeData::Dense(mask) => { for i in 0..n_count { if (mask[i/8] >> (i%8)) & 1 == 1 { current_dense[i] = true; } } }
                _ => {}
            }
            engine.backend.think_cycles(&mut engine.model, &current_dense, think_ticks, context.tick, engine.state.global_modulators)
        } else {
            initial_spike_data
        };

        // Update dense buffer if thinking changed things
        engine.state.current_spikes_buffer.fill(false);
        match &final_spike_data {
            SpikeData::Sparse(indices) => {
                for &idx in indices { if idx < n_count { engine.state.current_spikes_buffer[idx] = true; } }
            }
            SpikeData::Dense(mask) => {
                for i in 0..n_count {
                    if (mask[i / 8] >> (i % 8)) & 1 == 1 { engine.state.current_spikes_buffer[i] = true; }
                }
            }
            SpikeData::BitPacked(packed) => {
                for i in 0..n_count {
                    if (packed[i / 64] >> (i % 64)) & 1 == 1 { engine.state.current_spikes_buffer[i] = true; }
                }
            }
            _ => {}
        }
        engine.state.spikes_history[engine.state.history_ptr] = final_spike_data;
    }
}

pub struct ObservationStage;
impl PipelineStage for ObservationStage {
    fn name(&self) -> &str { "observation" }
    fn execute(&mut self, engine: &mut SimulationEngine, context: &mut PipelineContext) {
        let n_count = engine.model.neurons.len();
        let spike_count = engine.state.current_spikes_buffer.iter().filter(|&&s| s).count();

        // Hierarchical L2 History Update (every 16 ticks for long-term context)
        if true { // Run always for tests/now, optimize later
            let mut block_activity = std::collections::HashMap::new();
            for i in 0..n_count {
                if engine.state.current_spikes_buffer[i] {
                    let bid = engine.model.neurons.block_id[i];
                    *block_activity.entry(bid).or_insert(0u16) += 1;
                }
            }

            let max_bid = engine.model.neurons.block_id.iter().max().copied().unwrap_or(0);
            let mut summary = vec![0u16; (max_bid + 1) as usize];
            for (bid, count) in block_activity {
                summary[bid as usize] = count;
            }

            let l2_len = engine.state.l2_history.len();
            engine.state.l2_history[engine.state.l2_ptr] = summary.clone();
            engine.state.l2_ptr = (engine.state.l2_ptr + 1) % l2_len;

            // L3 Episodic Archive Trigger: store if surprise is very high (Rare event)
            if context.surprise > 1500 {
                engine.state.l3_archive.push(summary);
                if engine.state.l3_archive.len() > 1000 { engine.state.l3_archive.remove(0); }
                log::debug!("L3: Episodic memory stored (Surprise: {})", context.surprise);
            }

            // Update BitWise Titan Memory if present using zero-copy downcasting
            for m in engine.modules.modules.iter_mut() {
                if let Some(titan) = m.as_any_mut().downcast_mut::<genesis_core::titan::BitWiseTitan>() {
                    titan.learn_from_history(&engine.state.spikes_history, engine.state.history_ptr, &engine.model.neurons, context.surprise);
                }
            }
        }

        // Calculate surprise logic
        let current = spike_count as f32;
        let surprise = (current - engine.state.rolling_spike_count).abs();

        // Update Block Surprise Map
        let max_bid = engine.model.neurons.block_id.iter().max().copied().unwrap_or(0) as usize;
        if engine.state.block_surprise.len() <= max_bid {
            engine.state.block_surprise.resize(max_bid + 1, 0.0);
        }

        // Simple heuristic: surprise is higher for blocks whose activity deviated from history
        let l2_ptr = if engine.state.l2_ptr == 0 { engine.state.l2_history.len() - 1 } else { engine.state.l2_ptr - 1 };
        let prev_summary = &engine.state.l2_history[l2_ptr];

        for i in 0..n_count {
            let bid = engine.model.neurons.block_id[i] as usize;
            if bid < prev_summary.len() {
                let fired = engine.state.current_spikes_buffer[i];
                let block_avg = prev_summary[bid] as f32;

                // If a neuron fires in a block that was expected to be quiet, or vice versa
                if fired && block_avg < 1.0 {
                    engine.state.block_surprise[bid] = engine.state.block_surprise[bid] * 0.95 + 1.0 * 0.05;
                } else {
                    engine.state.block_surprise[bid] *= 0.99;
                }
            }
        }

        engine.state.rolling_spike_count = engine.state.rolling_spike_count * 0.9 + current * 0.1;
        let scaled_surprise = (surprise * 1024.0 / (engine.state.rolling_spike_count + 1.0)) as i32;
        context.surprise = scaled_surprise.min(2048);

        context.blackboard.insert("avg_activity".to_string(), engine.state.rolling_spike_count / engine.model.neurons.len() as f32);
    }
}

pub struct NeuromodulationStage;
impl PipelineStage for NeuromodulationStage {
    fn name(&self) -> &str { "neuromodulation" }
    fn execute(&mut self, engine: &mut SimulationEngine, context: &mut PipelineContext) {
        crate::engine::neuromodulation::NeuromodulationEngine::update_with_events(engine, context.surprise, context.normalized_reward, &context.events);
    }
}

pub struct NormalizationStage;
impl PipelineStage for NormalizationStage {
    fn name(&self) -> &str { "normalization" }
    fn execute(&mut self, engine: &mut SimulationEngine, context: &mut PipelineContext) {
        if context.tick % 100 != 0 { return; }

        // Metabolic Homeostasis: Normalize synaptic weights to prevent metabolic explosion
        use rayon::prelude::*;
        let n_count = engine.model.neurons.len();
        let s_count = engine.model.synapses.len();
        if s_count == 0 { return; }

        let mut sum_weights = vec![0i64; n_count];
        for i in 0..s_count {
            let target = engine.model.synapses.target_index[i] as usize;
            if target < n_count {
                sum_weights[target] += engine.model.synapses.weight[i].abs() as i64;
            }
        }

        let max_neuron_sum = 1024 * 32; // Limit total incoming weight to 32.0 units
        let synapses = &mut engine.model.synapses;

        synapses.weight.par_iter_mut().enumerate().for_each(|(i, w)| {
            let target = synapses.target_index[i] as usize;
            if target < n_count && sum_weights[target] > max_neuron_sum {
                let scale_factor = (max_neuron_sum << 10) / sum_weights[target];
                *w = ((*w as i64 * scale_factor) >> 10) as i32;
            }
        });
    }
}

pub struct StructuralPlasticityStage {
    pub interval: u32,
}

impl StructuralPlasticityStage {
    pub fn new(interval: u32) -> Self {
        Self { interval }
    }
}

impl PipelineStage for StructuralPlasticityStage {
    fn name(&self) -> &str { "structural_plasticity" }
    fn execute(&mut self, engine: &mut SimulationEngine, context: &mut PipelineContext) {
        if context.tick % self.interval != 0 { return; }

        let reward_val = context.reward.map(|r| r as i32);
        let history = engine.reconstruct_history(16);
        engine.backend.structural_plasticity_with_surprise(&mut engine.model, reward_val, &history, &engine.state.block_surprise);

        // DBS: Neuron Migration between blocks
        for m in engine.modules.modules.iter_mut() {
            if m.name() == "attn_res" {
                if let Ok(mut attn) = bincode::deserialize::<genesis_core::AttnResModule>(&m.get_state()) {
                    let count = attn.migrate_neurons(&mut engine.model.neurons);
                    if count > 0 { log::debug!("DBS: {} neurons migrated between blocks", count); }
                    m.set_state(&bincode::serialize(&attn).unwrap());
                }
            }
        }

        engine.modules.on_night_phase(&mut engine.model.neurons, &mut engine.model.synapses, reward_val);
    }
}

pub struct LoadBalancingStage;
impl PipelineStage for LoadBalancingStage {
    fn name(&self) -> &str { "load_balancing" }
    fn execute(&mut self, engine: &mut SimulationEngine, _context: &mut PipelineContext) {
         for m in engine.modules.modules.iter_mut() {
             if m.name() == "load_balancer" {
                 if let Ok(mut lb) = bincode::deserialize::<genesis_core::LoadBalancerModule>(&m.get_state()) {
                     let changes = lb.rebalance(&mut engine.model.neurons);
                     if changes > 0 { log::debug!("LoadBalancer: {} blocks remapped", changes); }
                     m.set_state(&bincode::serialize(&lb).unwrap());
                 }
             }
         }
    }
}

pub struct AnomalyDetectionStage;
impl PipelineStage for AnomalyDetectionStage {
    fn name(&self) -> &str { "anomaly_detection" }
    fn execute(&mut self, engine: &mut SimulationEngine, context: &mut PipelineContext) {
        let n_count = engine.model.neurons.len();
        if n_count == 0 { return; }

        let spike_count = engine.state.current_spikes_buffer.iter().filter(|&&s| s).count();
        let activity_ratio = context.blackboard.get("avg_activity").cloned()
            .unwrap_or_else(|| (spike_count as f32) / (n_count as f32));

        // Detect "Activity Storms": more than 80% neurons firing at once
        if activity_ratio > 0.8 {
            log::warn!("Anomaly detected: Activity Storm ({}%)! Triggering emergency dampening.", (activity_ratio * 100.0) as u32);
            // Emergency Dampening: Massive boost to Serotonin (stability)
            engine.state.global_modulators.serotonin = (engine.state.global_modulators.serotonin + 1024).min(2048);

            // Temporary global inhibition by raising all base thresholds
            for b_thresh in &mut engine.model.neurons.base_threshold {
                *b_thresh = b_thresh.saturating_add(512);
            }
        }

        // Detect "Dead Network": zero activity for long period
        if activity_ratio < 0.001 {
            engine.state.dead_ticks = engine.state.dead_ticks.saturating_add(1);
            if engine.state.dead_ticks > 100 {
                log::warn!("Anomaly detected: Stagnant Network! Injecting metabolic noise.");
                engine.state.dead_ticks = 0;
                // Metabolic Reset: lowering thresholds to encourage firing
                for b_thresh in &mut engine.model.neurons.base_threshold {
                    if *b_thresh > 512 { *b_thresh -= 256; }
                }
            }
        } else {
            engine.state.dead_ticks = 0;
        }
    }
}
