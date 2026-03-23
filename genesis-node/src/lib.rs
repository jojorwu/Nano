use genesis_core::{ModuleManager, SpikeData};
use serde::{Serialize, Deserialize};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum RuntimeError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serialization(#[from] bincode::Error),
    #[error("Configuration error: {0}")]
    Config(#[from] serde_json::Error),
    #[error("Model load failure: {0}")]
    ModelLoad(String),
    #[error("Backend creation failed: {0}")]
    BackendCreationFailed(String),
    #[error("Network error: {0}")]
    NetworkError(String),
    #[error("Internal state error: {0}")]
    StateError(String),
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SpikePacket {
    pub tick: u64,
    pub data: SpikeData,
}

impl SpikePacket {
    pub fn compress_indices(indices: &[usize], universe: usize) -> Vec<u8> {
        let mut bits = Vec::new();
        let count = indices.len() as u32;
        bits.extend_from_slice(&count.to_le_bytes());

        let low_bits = if count > 0 { (universe as u32 / count).ilog2().max(1) } else { 1 };
        bits.push(low_bits as u8);

        let mut bit_buf = 0u8;
        let mut bit_count = 0;
        let mut last_high = 0u32;

        let mut sorted = indices.to_vec();
        sorted.sort_unstable();

        for &idx in &sorted {
            let high = (idx as u32) >> low_bits;
            let low = (idx as u32) & ((1 << low_bits) - 1);

            // Encode high part (unary)
            for _ in 0..(high - last_high) {
                if bit_count == 8 { bits.push(bit_buf); bit_buf = 0; bit_count = 0; }
                bit_count += 1;
            }
            bit_buf |= 1 << bit_count;
            bit_count += 1;
            if bit_count == 8 { bits.push(bit_buf); bit_buf = 0; bit_count = 0; }
            last_high = high;

            // Encode low part
            for i in 0..low_bits {
                if (low >> i) & 1 == 1 { bit_buf |= 1 << bit_count; }
                bit_count += 1;
                if bit_count == 8 { bits.push(bit_buf); bit_buf = 0; bit_count = 0; }
            }
        }
        if bit_count > 0 { bits.push(bit_buf); }
        bits
    }
}


#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SimulationSettings {
    pub tick_rate_hz: Option<u32>,
    pub checkpoint_interval: u32,
    pub night_phase_interval: u32,
    pub distributed_port: u16,
    pub save_on_exit: bool,
    pub telemetry_enabled: bool,
    pub preferred_backend: Option<String>,
}

impl Default for SimulationSettings {
    fn default() -> Self {
        Self {
            tick_rate_hz: None, // As fast as possible
            checkpoint_interval: 1000,
            night_phase_interval: 100,
            distributed_port: 8080,
            save_on_exit: true,
            telemetry_enabled: true,
            preferred_backend: None,
        }
    }
}

pub mod examples_rl;
pub mod observer;
pub mod telemetry;
pub mod engine;
pub mod commands;
pub mod persistence;
pub mod events;

pub use observer::Observer;
pub use telemetry::Telemetry;
pub use engine::SimulationEngine;
pub use events::{SimulationEvent, SimulationObserver};

pub struct Runtime {
    pub engine: SimulationEngine,
    pub settings: SimulationSettings,
    pub episode_reward_history: Vec<i32>, // GRPO-lite: for reward normalization
    pub network_manager: Option<std::sync::Arc<NetworkManager>>,
    pub observers: Vec<Box<dyn SimulationObserver>>,
    pub telemetry: Telemetry, // Still here for now, but will be an observer soon
}

pub struct NetworkManager {
    pub node_id: String,
    pub peers: Vec<String>,
    pub socket: tokio::net::UdpSocket,
}

impl NetworkManager {
    pub async fn new(node_id: String, peers: Vec<String>, port: u16) -> std::io::Result<Self> {
        let socket = tokio::net::UdpSocket::bind(format!("0.0.0.0:{}", port)).await?;
        Ok(Self { node_id, peers, socket })
    }

    pub async fn run(&self, queue: std::sync::Arc<std::sync::Mutex<Vec<usize>>>) {
        let mut buf = [0u8; 65535];
        loop {
            if let Ok((len, _)) = self.socket.recv_from(&mut buf).await {
                if let Ok(packet) = bincode::deserialize::<SpikePacket>(&buf[..len]) {
                    let mut q = queue.lock().unwrap();
                    match packet.data {
                        SpikeData::Sparse(indices) => q.extend(indices),
                        SpikeData::Dense(mask) => {
                            for i in 0..mask.len() * 8 {
                                if (mask[i / 8] >> (i % 8)) & 1 == 1 {
                                    q.push(i);
                                }
                            }
                        }
                        SpikeData::Compressed(data) => {
                            // Elias-Fano Bit-Packing: Decompress sorted sparse indices
                            if data.len() < 8 { return; }
                            let count = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
                            let low_bits = data[4] as u32;
                            let mut bit_ptr = 40usize; // Start after count(32) + low_bits(8)

                            let mut current_high = 0u32;
                            for _ in 0..count {
                                // Find next 1 in high bits (unary)
                                while bit_ptr / 8 < data.len() && (data[bit_ptr / 8] >> (bit_ptr % 8)) & 1 == 0 {
                                    current_high += 1;
                                    bit_ptr += 1;
                                }
                                bit_ptr += 1; // Skip the 1

                                // Read low bits
                                let mut low = 0u32;
                                for i in 0..low_bits {
                                    if bit_ptr / 8 < data.len() && (data[bit_ptr / 8] >> (bit_ptr % 8)) & 1 == 1 {
                                        low |= 1 << i;
                                    }
                                    bit_ptr += 1;
                                }
                                q.push(((current_high << low_bits) | low) as usize);
                            }
                        }
                    }
                }
            }
        }
    }

    pub async fn broadcast_spikes(&self, packet: SpikePacket) {
        let data = bincode::serialize(&packet).unwrap();
        for peer in &self.peers {
            let _ = self.socket.send_to(&data, peer).await;
        }
    }
}

impl Runtime {
    /// Bootstraps a simulation session from a model file on disk.
    pub fn load(path: &str) -> Result<Self, RuntimeError> {
        Self::load_with_settings(path, SimulationSettings::default())
    }

    /// Loads a model and configures the simulation with specific hardware and timing parameters.
    pub fn load_with_settings(path: &str, settings: SimulationSettings) -> Result<Self, RuntimeError> {
        let model = persistence::PersistenceManager::load(path)?;
        let n_count = model.neurons.len();

        let backend_name = settings.preferred_backend.as_deref()
            .unwrap_or(&model.config.preferred_backend);

        let registry = genesis_compute::BackendRegistry::new();
        let backend = registry.create(backend_name)
            .or_else(|| {
                log::warn!("Backend '{}' not found, falling back to CPU", backend_name);
                registry.create("cpu")
            })
            .ok_or_else(|| RuntimeError::BackendCreationFailed("Could not instantiate CPU fallback".into()))?;

        let mut modules = ModuleManager::new();

        // Register model-specific factories (e.g. pre-initialized Titan from BakedModel)
        #[cfg(feature = "titan")]
        if let Some(ref titan) = model.titan_memory {
            let t = titan.clone();
            modules.register_factory("titan", move || Box::new(t.clone()));
        }

        // Instantiate modules based on model state
        for name in model.module_states.keys() {
            if modules.instantiate(name) {
                let state = model.module_states.get(name).unwrap();
                if let Some(m) = modules.modules.last_mut() {
                    m.set_state(state);
                }
            }
        }

        // Fallback for titan if not in module_states but in titan_memory (migration/legacy)
        #[cfg(feature = "titan")]
        if model.titan_memory.is_some() && !model.module_states.contains_key("titan") {
            modules.instantiate("titan");
        }

        let mut rt = Self {
            engine: SimulationEngine::new(model, modules, backend, &settings),
            settings: settings.clone(),
            episode_reward_history: Vec::new(),
            network_manager: None,
            observers: vec![Box::new(Observer::new(n_count))],
            telemetry: Telemetry::default(),
        };
        rt.post_init().map_err(RuntimeError::StateError)?;
        Ok(rt)
    }

    pub fn post_init(&mut self) -> Result<(), String> {
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
        let normalized_reward = self.prepare_reward(reward);
        self.engine.tick_counter = self.engine.tick_counter.wrapping_add(1);
        let n_count = self.engine.model.neurons.len();

        self.emit_event(SimulationEvent::TickStarted(self.engine.tick_counter));

        self.engine.reset_potential_buffers();
        self.engine.prepare_merged_inputs(external_inputs);

        self.engine.input_bus.clear_mut();
        self.engine.modules.on_tick(&self.engine.input_bus, &self.engine.previous_spikes, self.engine.tick_counter);

        self.engine.finalize_potentials_from_bus();

        let full_history = self.engine.reconstruct_history(16);
        let spike_data = self.engine.backend.day_phase(&mut self.engine.model, &self.engine.merged_inputs_buffer, &self.engine.previous_spikes, &full_history, self.engine.tick_counter, self.engine.global_modulators);

        let final_spike_data = self.engine.execute_thinking_cycles(spike_data);

        // Convert final spike data to dense buffer for subsequent logic
        self.engine.current_spikes_buffer.fill(false);
        let mut spike_count = 0;
        match &final_spike_data {
            genesis_core::SpikeData::Sparse(indices) => {
                spike_count = indices.len();
                for &idx in indices { if idx < n_count { self.engine.current_spikes_buffer[idx] = true; } }
            }
            genesis_core::SpikeData::Dense(mask) => {
                for i in 0..n_count {
                    if (mask[i / 8] >> (i % 8)) & 1 == 1 {
                        self.engine.current_spikes_buffer[i] = true;
                        spike_count += 1;
                    }
                }
            }
            _ => {}
        }

        self.emit_event(SimulationEvent::TickComplete {
            tick: self.engine.tick_counter,
            spike_count,
            data: final_spike_data.clone()
        });

        let surprise = self.engine.calculate_surprise(spike_count);
        self.emit_event(SimulationEvent::SurpriseDetected(surprise));

        self.apply_neuromodulation(surprise, normalized_reward);

        // Critical sync path for safety
        for obs in &mut self.observers {
            if let Some(o) = obs.as_any_mut().downcast_mut::<Observer>() {
                o.process_spikes(&mut self.engine.current_spikes_buffer, &mut self.engine.model);
            }
        }

        self.telemetry.spike_counts.push(spike_count);

        self.engine.spikes_history[self.engine.history_ptr] = final_spike_data;
        self.engine.history_ptr = (self.engine.history_ptr + 1) % self.engine.spikes_history.len();

        if self.engine.tick_counter > 0 && self.engine.tick_counter % self.settings.night_phase_interval == 0 {
            self.emit_event(SimulationEvent::NightPhaseStarted(self.engine.tick_counter));
            self.perform_night_phase(reward, normalized_reward, layer_mask, surprise);
            self.emit_event(SimulationEvent::NightPhaseComplete(self.engine.tick_counter));
        }

        self.engine.previous_spikes.copy_from_slice(&self.engine.current_spikes_buffer);
        self.broadcast_ghost_spikes(&self.engine.previous_spikes);
        self.engine.previous_spikes.clone()
    }

    fn prepare_reward(&mut self, reward: Option<i32>) -> Option<i32> {
        reward.map(|r| {
            self.emit_event(SimulationEvent::RewardReceived(r));
            self.telemetry.episode_rewards.push(r);
            self.episode_reward_history.push(r);
            if self.episode_reward_history.len() > 100 { self.episode_reward_history.remove(0); }
            let mean = (self.episode_reward_history.iter().sum::<i32>() as f32) / (self.episode_reward_history.len() as f32);
            r - (mean as i32)
        })
    }

    fn apply_neuromodulation(&mut self, surprise: i32, normalized_reward: Option<i32>) {
        self.engine.global_modulators.noradrenaline = surprise;
        if let Some(r) = normalized_reward {
            self.engine.global_modulators.dopamine = r;
        } else {
            self.engine.global_modulators.dopamine = (self.engine.global_modulators.dopamine * 9) / 10;
        }
        // Serotonin tracks long-term stability
        self.engine.global_modulators.serotonin = (self.engine.global_modulators.serotonin * 99 + (1024 - surprise).max(0)) / 100;
    }

    fn perform_night_phase(&mut self, raw_reward: Option<i32>, normalized_reward: Option<i32>, layer_mask: Option<u16>, surprise: i32) {
        let modulators = self.engine.global_modulators;
        let mut prev = self.engine.previous_spikes.clone();
        let reconstructed = self.engine.reconstruct_history(self.settings.night_phase_interval as usize);

        for (i, current) in reconstructed.iter().rev().enumerate() {
            let hist_len = reconstructed.len() as u32;
            let tick = self.engine.tick_counter.saturating_sub(hist_len).saturating_add(i as u32).saturating_add(1);
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
                let packet = SpikePacket { tick: self.engine.tick_counter as u64, data };
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
}

#[cfg(test)]
mod tests;
