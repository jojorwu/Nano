use genesis_core::BakedModel;
use genesis_compute::{ComputeBackend, CpuBackend};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct SpikePacket {
    pub tick: u64,
    pub active_indices: Vec<usize>,
}

pub struct Runtime {
    pub model: BakedModel,
    pub backend: Box<dyn ComputeBackend + Send + Sync>,
    pub previous_spikes: Vec<bool>,
    pub tick_counter: u64,
    pub spikes_history: Vec<Vec<bool>>,
    pub network_manager: Option<NetworkManager>,
    pub observer: Observer,
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

pub struct NetworkManager {
    pub node_id: String,
    pub peers: Vec<String>,
}

impl NetworkManager {
    pub async fn broadcast_spikes(&self, packet: SpikePacket) {
        use tokio::net::UdpSocket;

        let data = serde_json::to_vec(&packet).unwrap();
        let socket = UdpSocket::bind("0.0.0.0:0").await.unwrap();
        for peer in &self.peers {
            let _ = socket.send_to(&data, peer).await;
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
        })
    }

    pub fn tick_with_reward(&mut self, external_inputs: &[i32], reward: Option<i32>) -> Vec<bool> {
        self.tick_counter += 1;
        // 1. Day Phase: Inference
        let mut current_spikes = self.backend.day_phase(&mut self.model, external_inputs, &self.previous_spikes, self.tick_counter);

        self.observer.process_spikes(&mut current_spikes);

        self.spikes_history.push(current_spikes.clone());

        // 2. Night Phase: Learning (every 100 ticks)
        if self.tick_counter % 100 == 0 {
            // Replay history for learning
            let mut prev = self.previous_spikes.clone(); // Correct batch boundary
            for (i, current) in self.spikes_history.iter().enumerate() {
                let tick = self.tick_counter - (self.spikes_history.len() as u64) + (i as u64) + 1;

                // Update neuron last_spike_tick for the replayed tick
                for (n_idx, &spiked) in current.iter().enumerate() {
                    if spiked { self.model.neurons.last_spike_tick[n_idx] = tick; }
                }

                self.backend.night_phase(&mut self.model, &prev, current, tick, reward);
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
                let packet = SpikePacket { tick: self.tick_counter, active_indices };
                // Using non-blocking send or spawn
                let peers = nm.peers.clone();
                tokio::spawn(async move {
                    use tokio::net::UdpSocket;
                    let data = serde_json::to_vec(&packet).unwrap();
                    if let Ok(socket) = UdpSocket::bind("0.0.0.0:0").await {
                        for peer in peers {
                            let _ = socket.send_to(&data, peer).await;
                        }
                    }
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
