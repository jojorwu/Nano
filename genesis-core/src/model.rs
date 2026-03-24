use crate::{IValue, SCALE};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum SpikeData {
    Sparse(Vec<usize>),
    Dense(Vec<u8>), // Bitmask (byte-aligned)
    BitPacked(Vec<u64>), // 64-bit optimized bitmask for SIMD/GPU
    Compressed(Vec<u8>), // Elias-Fano or similar bit-packed format
}

impl Default for SpikeData {
    fn default() -> Self {
        Self::Sparse(Vec::new())
    }
}

impl SpikeData {
    pub fn is_empty(&self) -> bool {
        match self {
            SpikeData::Sparse(v) => v.is_empty(),
            SpikeData::Dense(v) => v.iter().all(|&b| b == 0),
            SpikeData::BitPacked(v) => v.iter().all(|&w| w == 0),
            SpikeData::Compressed(v) => v.is_empty(),
        }
    }

    pub fn to_bitpacked(&self, n_count: usize) -> Vec<u64> {
        let packed_len = (n_count + 63) / 64;
        let mut packed = vec![0u64; packed_len];
        match self {
            SpikeData::BitPacked(p) => return p.clone(),
            SpikeData::Sparse(indices) => {
                for &idx in indices {
                    if idx < n_count {
                        packed[idx / 64] |= 1 << (idx % 64);
                    }
                }
            }
            SpikeData::Dense(mask) => {
                for i in 0..n_count {
                    if (mask[i / 8] >> (i % 8)) & 1 == 1 {
                        packed[i / 64] |= 1 << (i % 64);
                    }
                }
            }
            SpikeData::Compressed(_) => {
                // Not implemented for now, fallback to empty
            }
        }
        packed
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[repr(u8)]
pub enum Compartment {
    Proximal = 0,
    Distal = 1,
    Apical = 2,
    Basal = 3,
}

impl Default for Compartment {
    fn default() -> Self {
        Compartment::Proximal
    }
}

/// Structure of Arrays (SoA) layout for neural state data.
/// Optimized for SIMD access and GPU memory alignment.
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct NeuronsSoA {
    /// Optional identifier for neural functional columns/layers.
    pub layer_id: Vec<u16>,
    /// Identifier for grouping neurons into blocks (mini-columns) for shared attention.
    pub block_id: Vec<u32>,
    pub potential: Vec<IValue>,
    pub distal_potential: Vec<IValue>, // For distal dendrites (coincidence detection)
    pub proximal_potential: Vec<IValue>, // For somatic inputs
    pub apical_potential: Vec<IValue>,  // For hierarchical feedback
    pub basal_potential: Vec<IValue>,   // For lateral signals
    pub backprop_signal: Vec<IValue>, // Signal from soma to dendrites (SMBP)
    pub threshold: Vec<IValue>,
    pub base_threshold: Vec<IValue>, // Intrinsic Plasticity
    pub decay: Vec<IValue>,
    pub liquid_current: Vec<IValue>, // For LLIF (Liquid Neurons)
    pub dendritic_gate: Vec<IValue>, // SCALE = 1.0 (open), 0 = closed
    pub refractory_timer: Vec<i32>,
    pub last_spike_tick: Vec<u32>,
    pub update_interval: Vec<u32>, // Sub-tick precision: 1 = every tick, 10 = every 10 ticks
    pub next_update_tick: Vec<u32>,
    pub x: Vec<i16>,
    pub y: Vec<i16>,
    pub gate_threshold: Vec<IValue>,
    pub distal_gate: Vec<IValue>, // Gating scalars per compartment
    pub apical_gate: Vec<IValue>,
    pub basal_gate: Vec<IValue>,
    pub activity_ema: Vec<IValue>, // Long-term activity tracking (SCALE = 1.0)
    pub is_excitatory: Vec<u8>,    // 1 = true, 0 = false (FFI compatible)
    pub adaptation_current: Vec<IValue>, // Spike-Frequency Adaptation (SFA)
}

/// FFI-safe view of the NeuronsSoA for Zero-Copy access from C++ and Python.
#[repr(C)]
pub struct NeuronsFFI {
    pub potential: *mut IValue,
    pub proximal_potential: *mut IValue,
    pub distal_potential: *mut IValue,
    pub apical_potential: *mut IValue,
    pub basal_potential: *mut IValue,
    pub threshold: *mut IValue,
    pub decay: *mut IValue,
    pub dendritic_gate: *mut IValue,
    pub distal_gate: *mut IValue,
    pub apical_gate: *mut IValue,
    pub basal_gate: *mut IValue,
    pub gate_threshold: *mut IValue,
    pub adaptation_current: *mut IValue,
    pub refractory_timer: *mut i32,
    pub last_spike_tick: *mut u32,
    pub backprop_signal: *mut IValue,
    pub activity_ema: *mut IValue,
    pub base_threshold: *mut IValue,
    pub liquid_current: *mut IValue,
    pub is_excitatory: *mut u8,
    pub block_id: *mut u32,
    pub len: u32,
}

impl NeuronsSoA {
    pub fn as_ffi(&mut self) -> NeuronsFFI {
        NeuronsFFI {
            potential: self.potential.as_mut_ptr(),
            proximal_potential: self.proximal_potential.as_mut_ptr(),
            distal_potential: self.distal_potential.as_mut_ptr(),
            apical_potential: self.apical_potential.as_mut_ptr(),
            basal_potential: self.basal_potential.as_mut_ptr(),
            threshold: self.threshold.as_mut_ptr(),
            decay: self.decay.as_mut_ptr(),
            dendritic_gate: self.dendritic_gate.as_mut_ptr(),
            distal_gate: self.distal_gate.as_mut_ptr(),
            apical_gate: self.apical_gate.as_mut_ptr(),
            basal_gate: self.basal_gate.as_mut_ptr(),
            gate_threshold: self.gate_threshold.as_mut_ptr(),
            adaptation_current: self.adaptation_current.as_mut_ptr(),
            refractory_timer: self.refractory_timer.as_mut_ptr(),
            last_spike_tick: self.last_spike_tick.as_mut_ptr(),
            backprop_signal: self.backprop_signal.as_mut_ptr(),
            activity_ema: self.activity_ema.as_mut_ptr(),
            base_threshold: self.base_threshold.as_mut_ptr(),
            liquid_current: self.liquid_current.as_mut_ptr(),
            is_excitatory: self.is_excitatory.as_mut_ptr(),
            block_id: self.block_id.as_mut_ptr(),
            len: self.len() as u32,
        }
    }
}

impl NeuronsSoA {
    pub fn new(size: usize) -> Self {
        Self::with_capacity(size, size)
    }

    pub fn with_capacity(size: usize, capacity: usize) -> Self {
        let mut neurons = Self {
            layer_id: Vec::with_capacity(capacity),
            block_id: Vec::with_capacity(capacity),
            potential: Vec::with_capacity(capacity),
            distal_potential: Vec::with_capacity(capacity),
            proximal_potential: Vec::with_capacity(capacity),
            apical_potential: Vec::with_capacity(capacity),
            basal_potential: Vec::with_capacity(capacity),
            backprop_signal: Vec::with_capacity(capacity),
            threshold: Vec::with_capacity(capacity),
            base_threshold: Vec::with_capacity(capacity),
            decay: Vec::with_capacity(capacity),
            liquid_current: Vec::with_capacity(capacity),
            dendritic_gate: Vec::with_capacity(capacity),
            refractory_timer: Vec::with_capacity(capacity),
            last_spike_tick: Vec::with_capacity(capacity),
            update_interval: Vec::with_capacity(capacity),
            next_update_tick: Vec::with_capacity(capacity),
            x: Vec::with_capacity(capacity),
            y: Vec::with_capacity(capacity),
            gate_threshold: Vec::with_capacity(capacity),
            distal_gate: Vec::with_capacity(capacity),
            apical_gate: Vec::with_capacity(capacity),
            basal_gate: Vec::with_capacity(capacity),
            activity_ema: Vec::with_capacity(capacity),
            is_excitatory: Vec::with_capacity(capacity),
            adaptation_current: Vec::with_capacity(capacity),
        };
        neurons.grow(size);
        neurons
    }
    pub fn len(&self) -> usize {
        self.potential.len()
    }

    pub fn validate(&self) -> Result<(), String> {
        let l = self.len();
        if self.layer_id.len() != l { return Err("layer_id length mismatch".into()); }
        if self.block_id.len() != l { return Err("block_id length mismatch".into()); }
        if self.threshold.len() != l { return Err("threshold length mismatch".into()); }
        if self.base_threshold.len() != l { return Err("base_threshold length mismatch".into()); }
        if self.decay.len() != l { return Err("decay length mismatch".into()); }
        if self.refractory_timer.len() != l { return Err("refractory_timer length mismatch".into()); }
        if self.last_spike_tick.len() != l { return Err("last_spike_tick length mismatch".into()); }
        if self.update_interval.len() != l { return Err("update_interval length mismatch".into()); }
        if self.next_update_tick.len() != l { return Err("next_update_tick length mismatch".into()); }
        if self.activity_ema.len() != l { return Err("activity_ema length mismatch".into()); }
        if self.is_excitatory.len() != l { return Err("is_excitatory length mismatch".into()); }
        if self.adaptation_current.len() != l { return Err("adaptation_current length mismatch".into()); }
        Ok(())
    }

    pub fn grow(&mut self, additional: usize) {
        let new_size = self.len() + additional;
        self.layer_id.resize(new_size, 0);
        self.block_id.resize(new_size, 0);
        self.potential.resize(new_size, 0);
        self.distal_potential.resize(new_size, 0);
        self.proximal_potential.resize(new_size, 0);
        self.apical_potential.resize(new_size, 0);
        self.basal_potential.resize(new_size, 0);
        self.backprop_signal.resize(new_size, 0);
        self.threshold.resize(new_size, SCALE);
        self.base_threshold.resize(new_size, SCALE);
        self.decay.resize(new_size, 50);
        self.liquid_current.resize(new_size, 0);
        self.dendritic_gate.resize(new_size, SCALE);
        self.refractory_timer.resize(new_size, 0);
        self.last_spike_tick.resize(new_size, 0);
        self.update_interval.resize(new_size, 1);
        self.next_update_tick.resize(new_size, 0);
        self.x.resize(new_size, 0);
        self.y.resize(new_size, 0);
        self.gate_threshold.resize(new_size, 512);
        self.distal_gate.resize(new_size, SCALE);
        self.apical_gate.resize(new_size, SCALE);
        self.basal_gate.resize(new_size, SCALE);
        self.activity_ema.resize(new_size, 0);
        self.is_excitatory.resize(new_size, 1);
        self.adaptation_current.resize(new_size, 0);
    }

    pub fn shrink_to_fit(&mut self) {
        self.layer_id.shrink_to_fit();
        self.block_id.shrink_to_fit();
        self.potential.shrink_to_fit();
        self.distal_potential.shrink_to_fit();
        self.proximal_potential.shrink_to_fit();
        self.apical_potential.shrink_to_fit();
        self.basal_potential.shrink_to_fit();
        self.backprop_signal.shrink_to_fit();
        self.threshold.shrink_to_fit();
        self.base_threshold.shrink_to_fit();
        self.decay.shrink_to_fit();
        self.liquid_current.shrink_to_fit();
        self.dendritic_gate.shrink_to_fit();
        self.refractory_timer.shrink_to_fit();
        self.last_spike_tick.shrink_to_fit();
        self.update_interval.shrink_to_fit();
        self.next_update_tick.shrink_to_fit();
        self.x.shrink_to_fit();
        self.y.shrink_to_fit();
        self.gate_threshold.shrink_to_fit();
        self.distal_gate.shrink_to_fit();
        self.apical_gate.shrink_to_fit();
        self.basal_gate.shrink_to_fit();
        self.activity_ema.shrink_to_fit();
        self.is_excitatory.shrink_to_fit();
        self.adaptation_current.shrink_to_fit();
    }
}

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct SynapsesSoA {
    pub source_index: Vec<u32>,
    pub target_index: Vec<u32>,
    pub weight: Vec<IValue>,
    pub delay: Vec<u8>, // Axonal delays (1-16 ticks)
    pub stp_resources: Vec<IValue>, // Short-Term Depression (SCALE = 1.0)
    pub stp_calcium: Vec<IValue>,   // Short-Term Facilitation (SCALE = 1.0)
    pub compartment: Vec<Compartment>,
    pub latent_matrix: Option<LatentSynapseMatrix>,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SynapsesFFI {
    pub source_index: *const u32,
    pub target_index: *const u32,
    pub weight: *mut IValue,
    pub delay: *const u8,
    pub stp_resources: *mut IValue,
    pub stp_calcium: *mut IValue,
    pub compartment: *const u8,
    pub len: u32,

    // CSR Index support
    pub offsets: *const u32,       // size: (n_count * 16) + 1
    pub indices_flat: *const usize, // size: len
}

impl SynapsesSoA {
    pub fn as_ffi(&mut self, offsets: *const u32, indices: *const usize) -> SynapsesFFI {
        SynapsesFFI {
            source_index: self.source_index.as_ptr(),
            target_index: self.target_index.as_ptr(),
            weight: self.weight.as_mut_ptr(),
            delay: self.delay.as_ptr(),
            stp_resources: self.stp_resources.as_mut_ptr(),
            stp_calcium: self.stp_calcium.as_mut_ptr(),
            compartment: self.compartment.as_ptr() as *const u8,
            len: self.len() as u32,
            offsets,
            indices_flat: indices,
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct LatentSynapseMatrix {
    pub u: Vec<IValue>, // Low-rank U matrix
    pub v: Vec<IValue>, // Low-rank V matrix
    pub rank: usize,
}

impl SynapsesSoA {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            source_index: Vec::with_capacity(capacity),
            target_index: Vec::with_capacity(capacity),
            weight: Vec::with_capacity(capacity),
            delay: Vec::with_capacity(capacity),
            stp_resources: Vec::with_capacity(capacity),
            stp_calcium: Vec::with_capacity(capacity),
            compartment: Vec::with_capacity(capacity),
            latent_matrix: None,
        }
    }

    pub fn push(&mut self, source: u32, target: u32, weight: IValue) {
        self.push_to_compartment(source, target, weight, 1, Compartment::Proximal);
    }

    pub fn push_delayed(&mut self, source: u32, target: u32, weight: IValue, delay: u8) {
        self.push_to_compartment(source, target, weight, delay, Compartment::Proximal);
    }

    pub fn push_to_compartment(&mut self, source: u32, target: u32, weight: IValue, delay: u8, compartment: Compartment) {
        self.source_index.push(source);
        self.target_index.push(target);
        self.weight.push(weight);
        self.delay.push(delay.max(1));
        self.stp_resources.push(SCALE); // Start fully charged
        self.stp_calcium.push(0);       // Start at baseline
        self.compartment.push(compartment);
    }

    pub fn push_polarized(&mut self, source: u32, target: u32, weight: IValue, delay: u8, compartment: Compartment, neurons: &NeuronsSoA) {
        let polarized_weight = if neurons.is_excitatory[source as usize] != 0 {
            weight.abs()
        } else {
            -weight.abs()
        };
        self.push_to_compartment(source, target, polarized_weight, delay, compartment);
    }

    pub fn set_latent(&mut self, u: Vec<IValue>, v: Vec<IValue>, rank: usize) {
        self.latent_matrix = Some(LatentSynapseMatrix { u, v, rank });
    }

    pub fn len(&self) -> usize {
        self.source_index.len()
    }

    pub fn remove(&mut self, index: usize) {
        self.source_index.swap_remove(index);
        self.target_index.swap_remove(index);
        self.weight.swap_remove(index);
        self.delay.swap_remove(index);
        self.stp_resources.swap_remove(index);
        self.stp_calcium.swap_remove(index);
        self.compartment.swap_remove(index);
    }

    pub fn shrink_to_fit(&mut self) {
        self.source_index.shrink_to_fit();
        self.target_index.shrink_to_fit();
        self.weight.shrink_to_fit();
        self.delay.shrink_to_fit();
        self.stp_resources.shrink_to_fit();
        self.stp_calcium.shrink_to_fit();
        self.compartment.shrink_to_fit();
    }
}

/// Represents the transient, ephemeral state of a simulation session.
#[derive(Serialize, Deserialize, Clone)]
pub struct SimulationState {
    pub previous_spikes: Vec<bool>,
    pub current_spikes_buffer: Vec<bool>,
    pub merged_inputs_buffer: Vec<IValue>,
    pub spikes_history: Vec<SpikeData>,
    pub history_ptr: usize,
    pub tick_counter: u32,
    pub global_modulators: crate::NeuromodulationState,
    pub rolling_spike_count: f32,
    pub dead_ticks: u32,
    /// L2 Hierarchical History: stores block-level activity summaries over long windows.
    /// Each entry is a Vec of activity counts (one per block_id).
    pub l2_history: Vec<Vec<u16>>,
    pub l2_ptr: usize,
    /// Per-block surprise rolling average for targeted neurogenesis
    pub block_surprise: Vec<f32>,
}

impl SimulationState {
    pub fn new(n_count: usize, history_len: usize) -> Self {
        let bitpacked_len = (n_count + 63) / 64;
        Self {
            previous_spikes: vec![false; n_count],
            current_spikes_buffer: vec![false; n_count],
            merged_inputs_buffer: vec![0; n_count],
            spikes_history: vec![SpikeData::BitPacked(vec![0; bitpacked_len]); history_len.max(16)],
            history_ptr: 0,
            tick_counter: 0,
            global_modulators: crate::NeuromodulationState::default(),
            rolling_spike_count: 0.0,
            dead_ticks: 0,
            l2_history: vec![Vec::new(); 16], // L2 window
            l2_ptr: 0,
            block_surprise: Vec::new(),
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct BakedModel {
    pub version: String,
    pub config: crate::config::NetworkConfig,
    pub node_id: u32,
    pub local_range: (usize, usize), // (start, end) indices of local neurons
    pub neurons: NeuronsSoA,
    pub synapses: SynapsesSoA,

    // Dynamic Module State Storage
    pub module_states: HashMap<String, Vec<u8>>,

    #[cfg(feature = "titan")]
    pub titan_memory: Option<crate::titan::BitWiseTitan>,
    #[cfg(feature = "text")]
    pub has_text: bool,
    #[cfg(feature = "vision")]
    pub has_vision: bool,
    #[cfg(feature = "audio")]
    pub has_audio: bool,
    #[cfg(feature = "robotics")]
    pub has_robotics: bool,
    #[cfg(feature = "fusion")]
    pub has_fusion: bool,
    #[cfg(feature = "text")]
    pub vocabulary: HashMap<String, usize>,
}

impl BakedModel {
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, self).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let file = File::open(path).map_err(|e| {
            log::error!("Failed to open model file at '{}': {}", path, e);
            e
        })?;
        let reader = BufReader::new(file);
        let model: BakedModel = bincode::deserialize_from(reader).map_err(|e| {
            log::error!("Error deserializing model from '{}': {:?}", path, e);
            std::io::Error::new(std::io::ErrorKind::Other, e)
        })?;

        if model.version != "4.2" {
             log::warn!("Loading model version {} into v4.2 engine. Physics scaling (1024) may differ from older versions.", model.version);
        }

        model.neurons.validate().map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let n_count = model.neurons.len();
        if model.local_range.1 > n_count {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Local range out of bounds"));
        }

        Ok(model)
    }
}
