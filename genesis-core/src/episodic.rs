use crate::{IValue, NanoModule, bus::InputBus, model::{NeuronsSoA, SynapsesSoA}};
use serde::{Serialize, Deserialize};

/// Episodic Memory Module (Hippocampal-Cortical Pathway)
/// Records raw spike sequences during high-surprise events and replays them
/// during the Night Phase to consolidate long-term associations.

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct EpisodicModule {
    /// Buffer of recorded episodic sequences
    pub sequences: Vec<EpisodicSequence>,
    /// Active recording state
    pub recording_sequence: Option<EpisodicSequence>,
    /// Minimum surprise to trigger recording
    pub surprise_trigger: IValue,
    /// Maximum length of a recorded sequence
    pub max_seq_len: usize,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct EpisodicSequence {
    pub ticks: Vec<Vec<usize>>, // List of active neuron indices per tick
    pub surprise_at_start: IValue,
}

impl EpisodicModule {
    pub fn new() -> Self {
        Self {
            sequences: Vec::new(),
            recording_sequence: None,
            surprise_trigger: 1200,
            max_seq_len: 50,
        }
    }
}

impl NanoModule for EpisodicModule {
    fn name(&self) -> &str { "episodic" }
    fn tier(&self) -> u32 { 20 }

    fn on_tick(&mut self, _bus: &InputBus, _previous_spikes: &[bool], _tick: u32) {
        // Core replay logic is handled in on_night_phase
    }

    fn on_update_weights(&mut self, _neurons: &mut NeuronsSoA, _prev: &[bool], current: &[bool], _tick: u32, surprise: Option<IValue>) {
        if let Some(s) = surprise {
            // 1. Trigger new recording if surprise is high
            if s > self.surprise_trigger && self.recording_sequence.is_none() {
                self.recording_sequence = Some(EpisodicSequence {
                    ticks: Vec::new(),
                    surprise_at_start: s,
                });
            }

            // 2. Record current activity into active sequence
            if let Some(ref mut seq) = self.recording_sequence {
                let active: Vec<usize> = current.iter().enumerate()
                    .filter(|&(_, &fired)| fired).map(|(i, _)| i).collect();
                seq.ticks.push(active);

                // 3. Close sequence if max length reached or surprise dropped significantly
                if seq.ticks.len() >= self.max_seq_len || (s < 500 && seq.ticks.len() > 10) {
                    let finished = self.recording_sequence.take().unwrap();
                    if !finished.ticks.is_empty() {
                         self.sequences.push(finished);
                    }
                }
            }
        }
    }

    fn on_night_phase(&mut self, _neurons: &mut NeuronsSoA, _synapses: &mut SynapsesSoA, _reward: Option<IValue>) {
        // In the night phase, sequences could be consolidated into Titan.
        // For now, we clear them to simulate successful transfer to cortical memory
        // OR we could keep them for future replay cycles.
        if self.sequences.len() > 10 {
             self.sequences.drain(0..5);
        }
    }

    fn box_clone(&self) -> Box<dyn NanoModule> { Box::new(self.clone()) }
    fn get_state(&self) -> Vec<u8> { bincode::serialize(self).unwrap_or_default() }
    fn set_state(&mut self, state: &[u8]) {
        if let Ok(new_self) = bincode::deserialize::<Self>(state) { *self = new_self; }
    }
}
