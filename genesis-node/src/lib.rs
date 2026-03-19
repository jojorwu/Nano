use genesis_core::BakedModel;
use genesis_compute::{ComputeBackend, CpuBackend};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct SpikePacket {
    pub tick: u64,
    pub data: SpikeData,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum SpikeData {
    Sparse(Vec<usize>),
    Dense(Vec<u8>), // Bitmask
}

pub struct Runtime {
    pub model: BakedModel,
    pub backend: Box<dyn ComputeBackend + Send + Sync>,
    pub previous_spikes: Vec<bool>,
    pub tick_counter: u32,
    pub spikes_history: Vec<Vec<bool>>,
    pub network_manager: Option<std::sync::Arc<NetworkManager>>,
    pub observer: Observer,
    pub remote_spike_queue: std::sync::Arc<std::sync::Mutex<Vec<usize>>>,
    pub telemetry: Telemetry,
}

pub struct Telemetry {
    pub spike_counts: Vec<usize>,
    pub episode_rewards: Vec<i32>,
}

pub struct Observer {
    pub max_spikes_per_tick: usize,
    pub total_energy_consumed: u64,
}

impl Observer {
    pub fn process_spikes(&mut self, spikes: &mut [bool]) {
        let spike_count = spikes.iter().filter(|&&s| s).count();
        if spike_count > self.max_spikes_per_tick {
            // Activity capping: Spike Storm Protection
            for i in 0..spikes.len() { spikes[i] = false; }
        }
        self.total_energy_consumed += spike_count as u64;
    }
}

pub mod examples_rl;

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
        let model = BakedModel::load(path)?;
        let n_count = model.neurons.len();
        Ok(Self {
            model,
            backend: Box::new(CpuBackend::default()),
            previous_spikes: vec![false; n_count],
            tick_counter: 0,
            spikes_history: Vec::new(),
            network_manager: None,
            observer: Observer { max_spikes_per_tick: n_count / 2, total_energy_consumed: 0 },
            remote_spike_queue: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            telemetry: Telemetry { spike_counts: Vec::new(), episode_rewards: Vec::new() },
        })
    }

    pub fn tick_with_reward(&mut self, external_inputs: &[i32], reward: Option<i32>) -> Vec<bool> {
        if let Some(r) = reward { self.telemetry.episode_rewards.push(r); }
        self.tick_counter = self.tick_counter.wrapping_add(1);

        let mut merged_inputs = external_inputs.to_vec();
        {
            let mut remote_spikes = self.remote_spike_queue.lock().unwrap();
            for &idx in remote_spikes.iter() {
                if idx < merged_inputs.len() {
                    merged_inputs[idx] = merged_inputs[idx].saturating_add(1000);
                }
            }
            remote_spikes.clear();
        }

        // 1. Day Phase: Inference
        let mut current_spikes = self.backend.day_phase(&mut self.model, &merged_inputs, &self.previous_spikes, self.tick_counter);

        let spike_count = current_spikes.iter().filter(|&&s| s).count();
        self.observer.process_spikes(&mut current_spikes);
        self.telemetry.spike_counts.push(spike_count);

        self.spikes_history.push(current_spikes.clone());

        // 2. Night Phase: Learning (every 100 ticks)
        if self.tick_counter % 100 == 0 {
            // Replay history for learning
            let mut prev = self.previous_spikes.clone(); // Correct batch boundary
            for (i, current) in self.spikes_history.iter().enumerate() {
                let tick = self.tick_counter - (self.spikes_history.len() as u32) + (i as u32) + 1;

                // Update neuron last_spike_tick for the replayed tick
                for (n_idx, &spiked) in current.iter().enumerate() {
                    if spiked { self.model.neurons.last_spike_tick[n_idx] = tick; }
                }

                self.backend.night_phase(&mut self.model, &prev, current, tick, reward, &self.spikes_history);
                prev = current.clone();
            }
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
                let data = if active_indices.len() < current_spikes.len() / 8 {
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
}

#[cfg(test)]
mod tests;
