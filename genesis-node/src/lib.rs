use genesis_core::{SpikeData, ModuleError};
use thiserror::Error;

/// Represents high-level errors that can occur during simulation setup or execution.
#[derive(Error, Debug)]
pub enum RuntimeError {
    #[error("IO error at {0}: {1}")]
    IoWithPath(String, std::io::Error),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error during {0}: {1}")]
    SerializationContext(String, bincode::Error),
    #[error("Serialization error: {0}")]
    Serialization(#[from] bincode::Error),
    #[error("Configuration error: {0}")]
    Config(#[from] serde_json::Error),
    #[error("Model load failure: {0}")]
    ModelLoad(String),
    #[error("Backend creation failed for '{0}': {1}")]
    BackendCreationFailed(String, String),
    #[error("Network error on node {0}: {1}")]
    NetworkError(String, String),
    #[error("Internal state error in {0}: {1}")]
    StateError(String, String),
    #[error("Module error in '{0}': {1}")]
    ModuleContext(String, ModuleError),
    #[error("Module error: {0}")]
    Module(#[from] ModuleError),
}

pub mod examples_rl;
pub mod observer;
pub mod telemetry;
pub mod engine;
pub mod commands;
pub mod persistence;
pub mod events;
pub mod settings;
pub mod ffi;
pub mod network;
pub mod builder;

pub use observer::Observer;
pub use telemetry::Telemetry;
pub use engine::SimulationEngine;
pub use events::{SimulationEvent, SimulationObserver};
pub use settings::SimulationSettings;
pub use network::{NetworkManager, SpikePacket};
pub use builder::RuntimeBuilder;

pub struct Runtime {
    pub engine: SimulationEngine,
    pub settings: SimulationSettings,
    pub episode_reward_history: Vec<i32>, // GRPO-lite: for reward normalization
    pub network_manager: Option<std::sync::Arc<NetworkManager>>,
    pub observers: Vec<Box<dyn SimulationObserver>>,
    pub last_surprise: i32,
    pub surprise_history: Vec<i32>,
}

impl Runtime {
    /// Bootstraps a simulation session from a model file on disk.
    pub fn load(path: &str) -> Result<Self, RuntimeError> {
        RuntimeBuilder::new().from_model(path).build()
    }

    /// Loads a model and configures the simulation with specific hardware and timing parameters.
    pub fn load_with_settings(path: &str, settings: SimulationSettings) -> Result<Self, RuntimeError> {
        RuntimeBuilder::new().from_model(path).with_settings(settings).build()
    }

    pub fn post_init(&mut self) -> Result<(), genesis_core::ModuleError> {
        self.engine.modules.on_init(&mut self.engine.model.neurons)
    }

    /// Executes a single simulation tick with optional reward modulation and layer targeting.
    /// This is the primary entry point for model interaction.
    pub fn emit_event(&mut self, event: SimulationEvent) {
        for obs in &mut self.observers {
            obs.on_event(&event, &mut self.engine);
        }
    }

    pub fn tick_with_reward_targeted(&mut self, external_inputs: &[i32], reward: Option<i32>, layer_mask: Option<u16>) -> Vec<bool> {
        use crate::engine::pipeline::{SimulationPipeline, PipelineContext};

        let start_time = std::time::Instant::now();
        let normalized_reward = self.prepare_reward(reward);
        self.engine.state.tick_counter = self.engine.state.tick_counter.wrapping_add(1);
        let tick = self.engine.state.tick_counter;

        self.emit_event(SimulationEvent::TickStarted(tick));

        let mut context = PipelineContext {
            tick,
            external_inputs: external_inputs.to_vec(),
            reward,
            normalized_reward,
            surprise: 0,
            start_time,
            blackboard: std::collections::HashMap::new(),
        };

        let mut pipeline = if let Some(ref stages) = self.settings.active_pipeline_stages {
            SimulationPipeline::from_config(stages, &self.settings)
        } else {
            SimulationPipeline::new(&self.settings)
        };
        pipeline.execute(&mut self.engine, &mut context);

        // History population is now handled by the Propagation stage (or sub-ticks)
        // to ensure it's available for Neuromodulation/Observation stages.

        let final_spike_data = self.engine.state.spikes_history[self.engine.state.history_ptr].clone();
        let spike_count = self.engine.state.current_spikes_buffer.iter().filter(|&&s| s).count();

        self.emit_event(SimulationEvent::TickComplete {
            tick,
            spike_count,
            data: final_spike_data,
            execution_time: start_time.elapsed(),
        });

        self.emit_event(SimulationEvent::SurpriseDetected(context.surprise));
        self.last_surprise = context.surprise;
        self.surprise_history.push(context.surprise);
        if self.surprise_history.len() > 1000 { self.surprise_history.remove(0); }

        // Critical sync path for safety
        for obs in &mut self.observers {
            if let Some(o) = obs.as_any_mut().downcast_mut::<Observer>() {
                o.process_spikes(&mut self.engine.state.current_spikes_buffer, &mut self.engine.model);
            }
        }

        self.engine.state.history_ptr = (self.engine.state.history_ptr + 1) % self.engine.state.spikes_history.len();

        // High resolution sync for testing/real-time
        self.engine.finalize_potentials_from_bus();

        if tick > 0 && tick % self.settings.night_phase_interval == 0 {
            self.emit_event(SimulationEvent::NightPhaseStarted(tick));
            self.perform_night_phase(reward, normalized_reward, layer_mask, context.surprise);
            self.emit_event(SimulationEvent::NightPhaseComplete(tick));
        }

        self.engine.state.previous_spikes.copy_from_slice(&self.engine.state.current_spikes_buffer);
        self.broadcast_ghost_spikes(&self.engine.state.previous_spikes);
        self.engine.state.previous_spikes.clone()
    }

    fn prepare_reward(&mut self, reward: Option<i32>) -> Option<i32> {
        reward.map(|r| {
            self.emit_event(SimulationEvent::RewardReceived(r));
            self.episode_reward_history.push(r);
            if self.episode_reward_history.len() > 100 { self.episode_reward_history.remove(0); }
            let mean = (self.episode_reward_history.iter().sum::<i32>() as f32) / (self.episode_reward_history.len() as f32);
            r - (mean as i32)
        })
    }


    fn perform_night_phase(&mut self, raw_reward: Option<i32>, normalized_reward: Option<i32>, layer_mask: Option<u16>, surprise: i32) {
        let modulators = self.engine.state.global_modulators;
        let mut prev = self.engine.state.previous_spikes.clone();
        let reconstructed = self.engine.reconstruct_history(self.settings.night_phase_interval as usize);

        for (i, current) in reconstructed.iter().rev().enumerate() {
            let hist_len = reconstructed.len() as u32;
            let tick = self.engine.state.tick_counter.saturating_sub(hist_len).saturating_add(i as u32).saturating_add(1);
            for (n_idx, &spiked) in current.iter().enumerate() {
                if spiked { self.engine.model.neurons.last_spike_tick[n_idx] = tick; }
            }

            let effective_reward = if let Some(m) = layer_mask {
                let in_mask = current.iter().enumerate().any(|(idx, &s)| s && self.engine.model.neurons.layer_id[idx] == m);
                if in_mask { normalized_reward } else { None }
            } else {
                normalized_reward
            };

            // Pass global chemical state to backend for modulated plasticity
            self.engine.backend.update_weights_modulated(&mut self.engine.model, &prev, current, tick, effective_reward, modulators, &reconstructed);
            self.engine.modules.on_update_weights(&mut self.engine.model.neurons, &prev, current, tick, Some(surprise));
            prev = current.clone();
        }

        // Structural plasticity and Global Module updates use raw reward
        self.engine.backend.structural_plasticity(&mut self.engine.model, raw_reward, &reconstructed);
        self.engine.modules.on_night_phase(&mut self.engine.model.neurons, &mut self.engine.model.synapses, raw_reward);
        self.sync_modules_to_model();
        // NOTE: history_ptr is NOT reset here to maintain circular buffer continuity.
    }

    pub fn tick_with_reward(&mut self, external_inputs: &[i32], reward: Option<i32>) -> Vec<bool> {
        self.tick_with_reward_targeted(external_inputs, reward, None)
    }

    fn broadcast_ghost_spikes(&self, current_spikes: &[bool]) {
        if let Some(ref nm) = self.network_manager {
            let active_indices: Vec<usize> = current_spikes.iter().enumerate()
                .filter(|&(_, &s)| s)
                .map(|(i, _)| i)
                .collect();

            if !active_indices.is_empty() {
                let data = if active_indices.len() < current_spikes.len() / 16 {
                    SpikeData::Compressed(SpikePacket::compress_indices(&active_indices, current_spikes.len()))
                } else if active_indices.len() < current_spikes.len() / 8 {
                    SpikeData::Sparse(active_indices)
                } else {
                    let mut mask = vec![0u8; (current_spikes.len() + 7) / 8];
                    for (i, &s) in current_spikes.iter().enumerate() {
                        if s { mask[i / 8] |= 1 << (i % 8); }
                    }
                    SpikeData::Dense(mask)
                };
                let packet = SpikePacket { tick: self.engine.state.tick_counter as u64, data };
                let nm_clone = nm.clone();
                tokio::spawn(async move {
                    nm_clone.broadcast_spikes(packet).await;
                });
            }
        }
    }

    pub fn tick(&mut self, external_inputs: &[i32]) -> Vec<bool> {
        self.tick_with_reward(external_inputs, None)
    }

    pub fn inject_text(&mut self, text: &str) {
        let input = genesis_core::ModuleInput::Text(text.to_string());
        for m in &mut self.engine.modules.modules {
            if m.name() == "text_processor" {
                m.handle_input(&input);
            }
        }
    }

    #[cfg(feature = "vision")]
    pub fn inject_image(&mut self, pixels: &[u8]) {
        let input = genesis_core::ModuleInput::Image(pixels.to_vec());
        for m in &mut self.engine.modules.modules {
            if m.name() == "vision" {
                m.handle_input(&input);
            }
        }
    }

    pub fn sync_state(&mut self) {
        self.engine.backend.sync_state(&mut self.engine.model);
        self.sync_modules_to_model();
    }

    pub fn sync_modules_to_model(&mut self) {
        for module in &self.engine.modules.modules {
            self.engine.model.module_states.insert(module.name().to_string(), module.get_state());
        }
    }

    pub fn reload_settings(&mut self, path: &str) -> Result<(), RuntimeError> {
        let content = std::fs::read_to_string(path)?;
        let new_settings: SimulationSettings = serde_json::from_str(&content)?;
        self.settings = new_settings;
        log::info!("Simulation settings reloaded from {}", path);
        Ok(())
    }

    pub fn handle_command(&mut self, cmd: &str) -> String {
        commands::CommandProcessor::handle(&mut self.engine, cmd)
    }

    /// Processes an input burst (event-driven mode).
    /// Executes ticks automatically until activity stabilizes or a limit is reached.
    pub fn process_burst(&mut self, external_inputs: &[i32], max_ticks: u32) -> Vec<bool> {
        let mut total_spikes = vec![false; self.engine.model.neurons.len()];
        let mut ticks_done = 0;
        let mut last_spike_count = 0;

        while ticks_done < max_ticks {
            let current_spikes = self.tick(if ticks_done == 0 { external_inputs } else { &[] });
            let current_count = current_spikes.iter().filter(|&&s| s).count();

            for (i, &s) in current_spikes.iter().enumerate() {
                if s { total_spikes[i] = true; }
            }

            // Stop if activity has settled (no spikes for 2 ticks or very low activity)
            if current_count == 0 && last_spike_count == 0 {
                break;
            }

            last_spike_count = current_count;
            ticks_done += 1;
        }

        log::debug!("Burst completed in {} ticks", ticks_done);
        total_spikes
    }

    /// High-level API: Process text without worrying about ticks.
    pub fn process_text(&mut self, text: &str) -> Vec<bool> {
        self.inject_text(text);
        self.process_burst(&[], 32) // Allow up to 32 internal ticks for "thinking"
    }

    /// High-level API: Process image without worrying about ticks.
    #[cfg(feature = "vision")]
    pub fn process_image(&mut self, pixels: &[u8]) -> Vec<bool> {
        self.inject_image(pixels);
        self.process_burst(&[], 64)
    }

    /// Enters a memory consolidation phase (Replay Mode).
    /// The network processes its own history to strengthen permanent associations.
    pub fn consolidate_memory(&mut self, iterations: u32) {
        log::info!("Starting memory consolidation phase ({} iterations)...", iterations);

        for _ in 0..iterations {
            // Memory Replay: Fetch past bitpacked patterns
            let history = &self.engine.state.spikes_history;
            if history.len() < 2 { break; }

            // Trigger Titan learning specifically from its own internal history
            for m in &mut self.engine.modules.modules {
                if m.name() == "titan" {
                    if let Ok(mut titan) = bincode::deserialize::<genesis_core::titan::BitWiseTitan>(&m.get_state()) {
                        // Memory Replay: iterate through history and treat each step as "now"
                        let h_len = history.len();
                        for i in 0..h_len {
                            titan.learn_from_history(history, i, &self.engine.model.neurons, 1000);
                        }
                        m.set_state(&bincode::serialize(&titan).unwrap());
                    }
                }
            }

            // Run a few "thinking" ticks to propagate these internal patterns
            self.process_burst(&[], 5);
        }
        log::info!("Consolidation complete.");
    }
}

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_advanced;
