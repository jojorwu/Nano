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

            for _ in 0..(high - last_high) {
                if bit_count == 8 { bits.push(bit_buf); bit_buf = 0; bit_count = 0; }
                bit_count += 1;
            }
            bit_buf |= 1 << bit_count;
            bit_count += 1;
            if bit_count == 8 { bits.push(bit_buf); bit_buf = 0; bit_count = 0; }
            last_high = high;

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

pub struct NetworkManager {
    pub node_id: String,
    pub peers: Vec<String>,
    pub spike_sender: tokio::sync::broadcast::Sender<Vec<u8>>,
}

impl NetworkManager {
    pub async fn new(node_id: String, peers: Vec<String>, port: u16, queue: Arc<Mutex<Vec<usize>>>) -> std::io::Result<Self> {
        let (tx, _) = tokio::sync::broadcast::channel::<Vec<u8>>(100);
        let tx_clone = tx.clone();

        // Server for receiving spikes
        let listener = TcpListener::bind(format!("0.0.0.0:{}", port)).await?;
        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                let queue = queue.clone();
                tokio::spawn(async move {
                    if let Ok(mut ws_stream) = accept_async(stream).await {
                        while let Some(msg) = ws_stream.next().await {
                            if let Ok(Message::Binary(data)) = msg {
                                if let Ok(packet) = bincode::deserialize::<SpikePacket>(&data) {
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
        if let Ok(data) = bincode::serialize(&packet) {
            let _ = self.spike_sender.send(data);
        }
    }
}
