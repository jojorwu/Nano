use crate::SimulationEngine;
use genesis_core::SpikeData;
use std::time::Instant;

pub trait PipelineStage {
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
            ],
        }
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
