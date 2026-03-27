use genesis_core::{SpikeData};
use serde::{Serialize, Deserialize};
use std::sync::{Arc, Mutex};
use tokio::net::{TcpListener};
use tokio_tungstenite::{accept_async, connect_async, tungstenite::protocol::Message};
use futures_util::{StreamExt, SinkExt};

#[derive(Serialize, Deserialize, Debug)]
pub struct SpikePacket {
    pub tick: u64,
    pub data: SpikeData,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum NetworkPacket {
    Spikes(SpikePacket),
    TitanState(genesis_core::titan::BitWiseTitan),
    Control(String),
}


pub struct NetworkManager {
    pub node_id: String,
    pub peers: Vec<String>,
    pub spike_sender: tokio::sync::broadcast::Sender<Vec<u8>>,
}

impl NetworkManager {
    pub async fn new(node_id: String, peers: Vec<String>, port: u16, queue: Arc<Mutex<Vec<usize>>>, titan_tx: Option<tokio::sync::mpsc::Sender<genesis_core::titan::BitWiseTitan>>) -> std::io::Result<Self> {
        let (tx, _) = tokio::sync::broadcast::channel::<Vec<u8>>(100);
        let tx_clone = tx.clone();

        // Server for receiving spikes
        let listener = TcpListener::bind(format!("0.0.0.0:{}", port)).await?;
        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                let queue = queue.clone();
                let titan_tx = titan_tx.clone();
                tokio::spawn(async move {
                    if let Ok(mut ws_stream) = accept_async(stream).await {
                        while let Some(msg) = ws_stream.next().await {
                            if let Ok(Message::Binary(data)) = msg {
                                if let Ok(net_packet) = bincode::deserialize::<NetworkPacket>(&data) {
                                    match net_packet {
                                        NetworkPacket::Spikes(packet) => {
                                            let mut q = queue.lock().unwrap();
                                            match packet.data {
                                                SpikeData::Sparse(indices) => q.extend(indices),
                                                SpikeData::BitPacked(packed) => {
                                                    for (i, &word) in packed.iter().enumerate() {
                                                        if word == 0 { continue; }
                                                        for bit in 0..64 {
                                                            if (word >> bit) & 1 == 1 {
                                                                q.push(i * 64 + bit);
                                                            }
                                                        }
                                                    }
                                                }
                                                _ => {}
                                            }
                                        }
                                        NetworkPacket::TitanState(state) => {
                                            if let Some(ref tx) = titan_tx {
                                                let _ = tx.send(state).await;
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }
                    }
                });
            }
        });

        // Background broadcaster
        let peers_clone = peers.clone();
        let mut rx = tx.subscribe();
        tokio::spawn(async move {
            while let Ok(data) = rx.recv().await {
                for peer in &peers_clone {
                    if let Ok((mut ws_stream, _)) = connect_async(format!("ws://{}", peer)).await {
                        let _ = ws_stream.send(Message::Binary(data.clone())).await;
                    }
                }
            }
        });

        Ok(Self { node_id, peers, spike_sender: tx_clone })
    }

    pub async fn broadcast_spikes(&self, packet: SpikePacket) {
        let net_packet = NetworkPacket::Spikes(packet);
        if let Ok(data) = bincode::serialize(&net_packet) {
            let _ = self.spike_sender.send(data);
        }
    }

    pub async fn broadcast_titan(&self, state: genesis_core::titan::BitWiseTitan) {
        let net_packet = NetworkPacket::TitanState(state);
        if let Ok(data) = bincode::serialize(&net_packet) {
            let _ = self.spike_sender.send(data);
        }
    }
}
