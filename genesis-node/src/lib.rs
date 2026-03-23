use genesis_core::{BakedModel, ModuleManager};
use genesis_compute::ComputeBackend;
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

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum SpikeData {
    Sparse(Vec<usize>),
    Dense(Vec<u8>), // Bitmask
    Compressed(Vec<u8>), // Elias-Fano or similar bit-packed format
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

pub use observer::Observer;
pub use telemetry::Telemetry;

pub struct Runtime {
    pub model: BakedModel,
    pub modules: ModuleManager,
    pub settings: SimulationSettings,
    pub backend: Box<dyn ComputeBackend + Send + Sync>,
    pub previous_spikes: Vec<bool>,
    pub current_spikes_buffer: Vec<bool>,
    pub merged_inputs_buffer: Vec<i32>,
    pub tick_counter: u32,
    pub spikes_history: Vec<SpikeData>, // Optimized history storage
    pub history_ptr: usize,             // Ring-buffer pointer
    pub episode_reward_history: Vec<i32>, // GRPO-lite: for reward normalization
    pub global_modulators: genesis_core::NeuromodulationState,
    pub rolling_spike_count: f32, // For surprise calculation
    pub network_manager: Option<std::sync::Arc<NetworkManager>>,
    pub observer: Observer,
    pub remote_spike_queue: std::sync::Arc<std::sync::Mutex<Vec<usize>>>,
    pub input_bus: genesis_core::InputBus,
    pub telemetry: Telemetry,
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
        let model = BakedModel::load(path).map_err(|e| RuntimeError::ModelLoad(e.to_string()))?;
        let n_count = model.neurons.len();

        let backend_name = settings.preferred_backend.as_deref()
            .unwrap_or(&model.config.preferred_backend);

        let registry = genesis_compute::BackendRegistry::new();
        let backend = registry.create(backend_name)
            .or_else(|| {
                log::warn!("Backend '{}' not found, falling back to CPU", backend_name);
                registry.create("cpu")
            })
            .expect("Failed to create any compute backend");

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
            model,
            modules,
            settings: settings.clone(),
            backend,
            previous_spikes: vec![false; n_count],
            current_spikes_buffer: vec![false; n_count],
            merged_inputs_buffer: vec![0; n_count],
            tick_counter: 0,
            spikes_history: vec![SpikeData::Sparse(Vec::new()); settings.night_phase_interval.max(16) as usize],
            history_ptr: 0,
            episode_reward_history: Vec::new(),
            global_modulators: genesis_core::NeuromodulationState::default(),
            rolling_spike_count: 0.0,
            network_manager: None,
            observer: Observer::new(n_count),
            remote_spike_queue: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            input_bus: genesis_core::InputBus::new(n_count),
            telemetry: Telemetry::default(),
        };
        rt.post_init();
        Ok(rt)
    }

    pub fn post_init(&mut self) {
        self.modules.on_init(&mut self.model.neurons);
    }

    pub fn calculate_surprise(&mut self, current_spike_count: usize) -> i32 {
        let current = current_spike_count as f32;
        let surprise = (current - self.rolling_spike_count).abs();

        // Update rolling average (EMA)
        self.rolling_spike_count = self.rolling_spike_count * 0.9 + current * 0.1;

        // Scale to IValue (SCALE=1024)
        // If current count is double the average, surprise is high.
        let scaled_surprise = (surprise * 1024.0 / (self.rolling_spike_count + 1.0)) as i32;
        scaled_surprise.min(2048) // Clamp
    }

    fn reset_potential_buffers(&mut self) {
        self.model.neurons.proximal_potential.fill(0);
        self.model.neurons.distal_potential.fill(0);
        self.model.neurons.apical_potential.fill(0);
        self.model.neurons.basal_potential.fill(0);
    }

    fn reconstruct_history(&self, window: usize) -> Vec<Vec<bool>> {
        let n_count = self.model.neurons.len();
        let hist_len = self.spikes_history.len();

        (0..window.min(hist_len)).map(|i| {
            // Circular access: start from (history_ptr - 1 - i) % hist_len
            let idx = (self.history_ptr + hist_len - 1 - i) % hist_len;
            let data = &self.spikes_history[idx];

            let mut vec = vec![false; n_count];
            match data {
                SpikeData::Sparse(indices) => { for &idx in indices { if idx < n_count { vec[idx] = true; } } }
                SpikeData::Dense(mask) => {
                    for i in 0..n_count { if (mask[i / 8] >> (i % 8)) & 1 == 1 { vec[i] = true; } }
                }
                _ => {}
            }
            vec
        }).collect()
    }

    /// Executes a single simulation tick with optional reward modulation and layer targeting.
    /// This is the primary entry point for model interaction.
    pub fn tick_with_reward_targeted(&mut self, external_inputs: &[i32], reward: Option<i32>, layer_mask: Option<u16>) -> Vec<bool> {
        let normalized_reward = self.prepare_reward(reward);
        self.tick_counter = self.tick_counter.wrapping_add(1);
        let n_count = self.model.neurons.len();

        self.reset_potential_buffers();
        self.prepare_merged_inputs(external_inputs, n_count);

        self.input_bus.clear();
        self.modules.on_tick(&mut self.input_bus, &self.previous_spikes, self.tick_counter);

        // Finalize potentials by zipping module inputs into the model
        self.model.neurons.proximal_potential.iter_mut()
            .zip(&self.input_bus.proximal)
            .for_each(|(p, b)| *p = p.saturating_add(*b));
        self.model.neurons.distal_potential.iter_mut()
            .zip(&self.input_bus.distal)
            .for_each(|(p, b)| *p = p.saturating_add(*b));
        self.model.neurons.apical_potential.iter_mut()
            .zip(&self.input_bus.apical)
            .for_each(|(p, b)| *p = p.saturating_add(*b));
        self.model.neurons.basal_potential.iter_mut()
            .zip(&self.input_bus.basal)
            .for_each(|(p, b)| *p = p.saturating_add(*b));

        let full_history = self.reconstruct_history(16);
        self.current_spikes_buffer = self.backend.day_phase(&mut self.model, &self.merged_inputs_buffer, &self.previous_spikes, &full_history, self.tick_counter, self.global_modulators);

        // Execute thinking cycles if think module is active
        if let Some(think_config) = self.get_think_config() {
            if think_config.active {
                self.current_spikes_buffer = self.backend.think_cycles(&mut self.model, &self.current_spikes_buffer, think_config.extra_ticks, self.tick_counter, self.global_modulators);
            }
        }

        let spike_count = self.current_spikes_buffer.iter().filter(|&&s| s).count();
        let surprise = self.calculate_surprise(spike_count);

        self.apply_neuromodulation(surprise, normalized_reward);
        self.observer.process_spikes(&mut self.current_spikes_buffer, &mut self.model);
        self.telemetry.spike_counts.push(spike_count);

        self.manage_spike_history(n_count);

        if self.tick_counter > 0 && self.tick_counter % self.settings.night_phase_interval == 0 {
            self.perform_night_phase(reward, normalized_reward, layer_mask, surprise);
        }

        self.previous_spikes.copy_from_slice(&self.current_spikes_buffer);
        self.broadcast_ghost_spikes(&self.previous_spikes);
        self.previous_spikes.clone()
    }

    fn prepare_reward(&mut self, reward: Option<i32>) -> Option<i32> {
        reward.map(|r| {
            self.telemetry.episode_rewards.push(r);
            self.episode_reward_history.push(r);
            if self.episode_reward_history.len() > 100 { self.episode_reward_history.remove(0); }
            let mean = (self.episode_reward_history.iter().sum::<i32>() as f32) / (self.episode_reward_history.len() as f32);
            r - (mean as i32)
        })
    }

    fn prepare_merged_inputs(&mut self, external_inputs: &[i32], n_count: usize) {
        self.merged_inputs_buffer.fill(0);

        // Merge external inputs
        let merge_len = external_inputs.len().min(n_count);
        self.merged_inputs_buffer[..merge_len].copy_from_slice(&external_inputs[..merge_len]);

        // Merge remote spikes
        let mut remote_spikes = self.remote_spike_queue.lock().unwrap();
        for &idx in remote_spikes.iter() {
            if idx < n_count {
                self.merged_inputs_buffer[idx] = self.merged_inputs_buffer[idx].saturating_add(genesis_core::SCALE);
            }
        }
        remote_spikes.clear();
    }

    fn apply_neuromodulation(&mut self, surprise: i32, normalized_reward: Option<i32>) {
        self.global_modulators.noradrenaline = surprise;
        if let Some(r) = normalized_reward {
            self.global_modulators.dopamine = r;
        } else {
            self.global_modulators.dopamine = (self.global_modulators.dopamine * 9) / 10;
        }
        // Serotonin tracks long-term stability
        self.global_modulators.serotonin = (self.global_modulators.serotonin * 99 + (1024 - surprise).max(0)) / 100;
    }

    fn manage_spike_history(&mut self, n_count: usize) {
        let active_indices: Vec<usize> = self.current_spikes_buffer.iter().enumerate()
            .filter(|&(_, &s)| s).map(|(i, _)| i).collect();

        let data = if active_indices.len() < n_count / 32 {
            SpikeData::Sparse(active_indices)
        } else {
            let mut mask = vec![0u8; (n_count + 7) / 8];
            for &idx in &active_indices { mask[idx / 8] |= 1 << (idx % 8); }
            SpikeData::Dense(mask)
        };

        self.spikes_history[self.history_ptr] = data;
        self.history_ptr = (self.history_ptr + 1) % self.spikes_history.len();
    }

    fn perform_night_phase(&mut self, raw_reward: Option<i32>, normalized_reward: Option<i32>, layer_mask: Option<u16>, surprise: i32) {
        let modulators = self.global_modulators;
        let mut prev = self.previous_spikes.clone();
        let reconstructed = self.reconstruct_history(self.settings.night_phase_interval as usize);

        for (i, current) in reconstructed.iter().rev().enumerate() {
            let hist_len = reconstructed.len() as u32;
            let tick = self.tick_counter.saturating_sub(hist_len).saturating_add(i as u32).saturating_add(1);
            for (n_idx, &spiked) in current.iter().enumerate() {
                if spiked { self.model.neurons.last_spike_tick[n_idx] = tick; }
            }

            let effective_reward = if let Some(m) = layer_mask {
                let in_mask = current.iter().enumerate().any(|(idx, &s)| s && self.model.neurons.layer_id[idx] == m);
                if in_mask { normalized_reward } else { None }
            } else {
                normalized_reward
            };

            // Pass global chemical state to backend for modulated plasticity
            self.backend.update_weights_modulated(&mut self.model, &prev, current, tick, effective_reward, modulators, &reconstructed);
            self.modules.on_update_weights(&mut self.model.neurons, &prev, current, tick, Some(surprise));
            prev = current.clone();
        }

        // Structural plasticity and Global Module updates use raw reward
        self.backend.structural_plasticity(&mut self.model, raw_reward, &reconstructed);
        self.modules.on_night_phase(&mut self.model.neurons, &mut self.model.synapses, raw_reward);
        self.sync_modules_to_model();
        // Reset history pointer after consolidation if desired,
        // though strictly not needed as pointers are tick-based.
        self.history_ptr = 0;
    }

    pub fn tick_with_reward(&mut self, external_inputs: &[i32], reward: Option<i32>) -> Vec<bool> {
        self.tick_with_reward_targeted(external_inputs, reward, None)
    }

    fn get_think_config(&self) -> Option<genesis_core::ThinkModule> {
        self.modules.modules.iter()
            .find(|m| m.name() == "think")
            .and_then(|m| bincode::deserialize(&m.get_state()).ok())
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
                let packet = SpikePacket { tick: self.tick_counter as u64, data };
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
        for m in &mut self.modules.modules {
            if m.name() == "text_processor" {
                m.handle_input(&input);
            }
        }
    }

    #[cfg(feature = "vision")]
    pub fn inject_image(&mut self, pixels: &[u8]) {
        let input = genesis_core::ModuleInput::Image(pixels.to_vec());
        for m in &mut self.modules.modules {
            if m.name() == "vision" {
                m.handle_input(&input);
            }
        }
    }

    pub fn sync_state(&mut self) {
        self.backend.sync_state(&mut self.model);
        self.sync_modules_to_model();
    }

    pub fn sync_modules_to_model(&mut self) {
        for module in &self.modules.modules {
            self.model.module_states.insert(module.name().to_string(), module.get_state());
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
        let parts: Vec<&str> = cmd.trim().split_whitespace().collect();
        if parts.is_empty() { return "No command provided".to_string(); }

        match parts[0] {
            "help" => "Commands: set_lr <val>, set_think <active|ticks>, status, save, exit".to_string(),
            "set_lr" => {
                if parts.len() < 2 { return "Usage: set_lr <val>".to_string(); }
                if let Ok(lr) = parts[1].parse::<i32>() {
                    self.model.config.learning_rate = lr;
                    format!("Learning rate set to {}", lr)
                } else { "Invalid value".to_string() }
            },
            "set_think" => {
                if parts.len() < 2 { return "Usage: set_think <active|ticks> <val>".to_string(); }
                let mut found = false;
                if parts.len() > 2 {
                    let val = if parts[1] == "active" {
                        if parts[2] == "true" || parts[2] == "1" { 1 } else { 0 }
                    } else {
                        parts[2].parse().unwrap_or(5)
                    };
                    let input = genesis_core::ModuleInput::Control(parts[1].to_string(), val);
                    for m in &mut self.modules.modules {
                        if m.name() == "think" {
                            m.handle_input(&input);
                            found = true;
                        }
                    }
                }
                if found { "Think settings updated".to_string() } else { "Think module not found".to_string() }
            },
            "status" => {
                format!("Tick: {}, Synapses: {}, Neurons: {}", self.tick_counter, self.model.synapses.len(), self.model.neurons.len())
            },
            _ => format!("Unknown command: {}", parts[0]),
        }
    }
}

#[cfg(test)]
mod tests;
