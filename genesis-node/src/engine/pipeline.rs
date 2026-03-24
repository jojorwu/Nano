use crate::{SimulationEngine, SimulationEvent};
use genesis_core::{SpikeData, SCALE};
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
}

pub struct SimulationPipeline {
    pub stages: Vec<Box<dyn PipelineStage>>,
}

impl SimulationPipeline {
    pub fn new() -> Self {
        Self {
            stages: vec![
                Box::new(InputStage),
                Box::new(PropagationStage),
                Box::new(ThinkingStage),
                Box::new(ObservationStage),
                Box::new(NeuromodulationStage),
                Box::new(NormalizationStage),
                Box::new(AnomalyDetectionStage),
            ],
        }
    }

    pub fn from_config(active_stages: &[String]) -> Self {
        let mut stages: Vec<Box<dyn PipelineStage>> = Vec::new();
        for name in active_stages {
            match name.as_str() {
                "input" => stages.push(Box::new(InputStage)),
                "propagation" => stages.push(Box::new(PropagationStage)),
                "thinking" => stages.push(Box::new(ThinkingStage)),
                "observation" => stages.push(Box::new(ObservationStage)),
                "neuromodulation" => stages.push(Box::new(NeuromodulationStage)),
                "normalization" => stages.push(Box::new(NormalizationStage)),
                "anomaly_detection" => stages.push(Box::new(AnomalyDetectionStage)),
                _ => log::warn!("Unknown pipeline stage: {}", name),
            }
        }
        Self { stages }
    }

    pub fn execute(&mut self, engine: &mut SimulationEngine, context: &mut PipelineContext) {
        for stage in &mut self.stages {
            stage.execute(engine, context);
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
        engine.finalize_potentials_from_bus();
    }
}

pub struct PropagationStage;
impl PipelineStage for PropagationStage {
    fn name(&self) -> &str { "propagation" }
    fn execute(&mut self, engine: &mut SimulationEngine, context: &mut PipelineContext) {
        let full_history = engine.reconstruct_history(16);
        let spike_data = engine.backend.day_phase(
            &mut engine.model,
            &engine.state.merged_inputs_buffer,
            &engine.state.previous_spikes,
            &full_history,
            context.tick,
            engine.state.global_modulators,
        );
        engine.state.current_spikes_buffer.fill(false);
        let n_count = engine.model.neurons.len();
        match &spike_data {
            SpikeData::Sparse(indices) => {
                for &idx in indices { if idx < n_count { engine.state.current_spikes_buffer[idx] = true; } }
            }
            SpikeData::Dense(mask) => {
                for i in 0..n_count {
                    if (mask[i / 8] >> (i % 8)) & 1 == 1 { engine.state.current_spikes_buffer[i] = true; }
                }
            }
            _ => {}
        }
        engine.state.spikes_history[engine.state.history_ptr] = spike_data;
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
            _ => {}
        }
        engine.state.spikes_history[engine.state.history_ptr] = final_spike_data;
    }
}

pub struct ObservationStage;
impl PipelineStage for ObservationStage {
    fn name(&self) -> &str { "observation" }
    fn execute(&mut self, engine: &mut SimulationEngine, context: &mut PipelineContext) {
        let spike_count = engine.state.current_spikes_buffer.iter().filter(|&&s| s).count();

        // Calculate surprise logic
        let current = spike_count as f32;
        let surprise = (current - engine.state.rolling_spike_count).abs();
        engine.state.rolling_spike_count = engine.state.rolling_spike_count * 0.9 + current * 0.1;
        let scaled_surprise = (surprise * 1024.0 / (engine.state.rolling_spike_count + 1.0)) as i32;
        context.surprise = scaled_surprise.min(2048);
    }
}

pub struct NeuromodulationStage;
impl PipelineStage for NeuromodulationStage {
    fn name(&self) -> &str { "neuromodulation" }
    fn execute(&mut self, engine: &mut SimulationEngine, context: &mut PipelineContext) {
        crate::engine::neuromodulation::NeuromodulationEngine::update(engine, context.surprise, context.normalized_reward);
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

pub struct AnomalyDetectionStage;
impl PipelineStage for AnomalyDetectionStage {
    fn name(&self) -> &str { "anomaly_detection" }
    fn execute(&mut self, engine: &mut SimulationEngine, _context: &mut PipelineContext) {
        let n_count = engine.model.neurons.len();
        if n_count == 0 { return; }

        let spike_count = engine.state.current_spikes_buffer.iter().filter(|&&s| s).count();
        let activity_ratio = (spike_count as f32) / (n_count as f32);

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
