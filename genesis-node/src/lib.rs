use genesis_core::{BakedModel, ModuleManager};
use genesis_compute::{ComputeBackend, CpuBackend};
use serde::{Serialize, Deserialize};

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

#[derive(Serialize, Deserialize, Debug)]
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
    pub tick_counter: u32,
    pub spikes_history: Vec<Vec<u32>>, // Sparse: only active indices
    pub network_manager: Option<std::sync::Arc<NetworkManager>>,
    pub observer: Observer,
    pub remote_spike_queue: std::sync::Arc<std::sync::Mutex<Vec<usize>>>,
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
    pub fn load(path: &str) -> std::io::Result<Self> {
        Self::load_with_settings(path, SimulationSettings::default())
    }

    pub fn load_with_settings(path: &str, settings: SimulationSettings) -> std::io::Result<Self> {
        let model = BakedModel::load(path)?;
        let n_count = model.neurons.len();

        let mut modules = ModuleManager::new();

        // Register available factories
        #[cfg(feature = "titan")]
        if let Some(ref titan) = model.titan_memory {
            let t = titan.clone();
            modules.register_factory("titan", move || Box::new(t.clone()));
        }
        modules.register_factory("think", || Box::new(genesis_core::ThinkModule::new(5)));

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

        Ok(Self {
            model,
            modules,
            settings,
            backend: Box::new(CpuBackend::default()),
            previous_spikes: vec![false; n_count],
            tick_counter: 0,
            spikes_history: Vec::new(),
            network_manager: None,
            observer: Observer::new(n_count),
            remote_spike_queue: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            telemetry: Telemetry::default(),
        })
    }

    pub fn tick_with_reward(&mut self, external_inputs: &[i32], reward: Option<i32>) -> Vec<bool> {
        if let Some(r) = reward { self.telemetry.episode_rewards.push(r); }
        self.tick_counter = self.tick_counter.wrapping_add(1);

        let n_count = self.model.neurons.len();

        // 0. Reset somatic input buffers each tick before module injection
        for i in 0..n_count {
            self.model.neurons.proximal_potential[i] = 0;
            self.model.neurons.distal_potential[i] = 0;
        }

        let mut merged_inputs = vec![0; n_count];
        for (i, &val) in external_inputs.iter().enumerate() {
            if i < n_count { merged_inputs[i] = val; }
        }

        {
            let mut remote_spikes = self.remote_spike_queue.lock().unwrap();
            for &idx in remote_spikes.iter() {
                if idx < n_count {
                    merged_inputs[idx] = merged_inputs[idx].saturating_add(1024);
                }
            }
            remote_spikes.clear();
        }

        // Module Pre-tick logic (e.g. Signal Injection)
        self.modules.on_tick(&mut self.model.neurons, &self.previous_spikes, self.tick_counter);

        // 1. Day Phase: Inference
        let mut current_spikes = self.backend.day_phase(&mut self.model, &merged_inputs, &self.previous_spikes, self.tick_counter);

        // Chain-of-Thought Reasoning (Thinking Mode)
        for module in &self.modules.modules {
            if module.name() == "think" {
                let state = module.get_state();
                if let Ok(think) = bincode::deserialize::<genesis_core::ThinkModule>(&state) {
                    if think.active {
                        for _ in 0..think.extra_ticks {
                            // Internal cycles: no external input, feed back spikes
                            current_spikes = self.backend.day_phase(&mut self.model, &vec![0; n_count], &current_spikes, self.tick_counter);
                        }
                    }
                }
            }
        }

        let spike_count = current_spikes.iter().filter(|&&s| s).count();
        self.observer.process_spikes(&mut current_spikes, &mut self.model);
        self.telemetry.spike_counts.push(spike_count);

        let sparse_spikes: Vec<u32> = current_spikes.iter().enumerate()
            .filter(|&(_, &s)| s)
            .map(|(i, _)| i as u32)
            .collect();
        self.spikes_history.push(sparse_spikes);

        // 2. Night Phase: Learning
        if self.tick_counter % self.settings.night_phase_interval == 0 {
            // Replay history for learning
            let mut prev = self.previous_spikes.clone();
            let n_count = self.model.neurons.len();

            // Reconstruct full boolean history for Correlational Growth
            let full_history: Vec<Vec<bool>> = self.spikes_history.iter().map(|sparse| {
                let mut vec = vec![false; n_count];
                for &idx in sparse { if (idx as usize) < n_count { vec[idx as usize] = true; } }
                vec
            }).collect();

            for (i, current) in full_history.iter().enumerate() {
                let tick = self.tick_counter - (full_history.len() as u32) + (i as u32) + 1;

                // Update neuron last_spike_tick for the replayed tick
                for (n_idx, &spiked) in current.iter().enumerate() {
                    if spiked { self.model.neurons.last_spike_tick[n_idx] = tick; }
                }

                self.backend.update_weights(&mut self.model, &prev, current, tick, reward, &full_history);
                self.modules.on_update_weights(&mut self.model.neurons, &prev, current, tick, reward);
                prev = current.clone();
            }
            self.backend.structural_plasticity(&mut self.model, reward, &full_history);
            self.modules.on_night_phase(&mut self.model.synapses, reward);

            // Sync module states back to model for persistence
            self.sync_modules_to_model();

            self.spikes_history.clear();
        }

        self.previous_spikes = current_spikes.clone();

        // Ghost Axons: Transmit spikes to remote nodes
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

        current_spikes
    }

    pub fn tick(&mut self, external_inputs: &[i32]) -> Vec<bool> {
        self.tick_with_reward(external_inputs, None)
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

    pub fn reload_settings(&mut self, path: &str) -> std::io::Result<()> {
        let content = std::fs::read_to_string(path)?;
        if let Ok(new_settings) = serde_json::from_str::<SimulationSettings>(&content) {
            self.settings = new_settings;
            log::info!("Simulation settings reloaded from {}", path);
        }
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
                // Implementation to find think module and update it
                let mut found = false;
                for m in &mut self.modules.modules {
                    if m.name() == "think" {
                        let mut state: genesis_core::ThinkModule = bincode::deserialize(&m.get_state()).unwrap();
                        if parts[1] == "active" && parts.len() > 2 {
                             state.active = parts[2] == "true" || parts[2] == "1";
                        } else if parts[1] == "ticks" && parts.len() > 2 {
                             state.extra_ticks = parts[2].parse().unwrap_or(5);
                        }
                        m.set_state(&bincode::serialize(&state).unwrap());
                        found = true;
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
