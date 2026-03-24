use serde::{Serialize, Deserialize};
use std::sync::atomic::{AtomicI32, Ordering};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Modality {
    Vision = 0,
    Text = 1,
    Audio = 2,
    Other = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChannelId(pub usize);

/// Represents the central signal exchange bus of the network.
///
/// The `InputBus` facilitates asynchronous communication between modules and the core
/// simulation engine. It uses atomic integers to allow multiple threads (modules)
/// to inject potentials concurrently without locking.
pub struct InputBus {
    pub size: usize,
    /// All channels in a flat vector for fast access via index.
    pub channels: Vec<Vec<AtomicI32>>,
    /// Name-to-Index mapping for human-readable access and initialization.
    pub name_map: HashMap<String, usize>,

    // Fixed indices for core compartments
    pub proximal_idx: usize,
    pub distal_idx: usize,
    pub apical_idx: usize,
    pub basal_idx: usize,
    /// Global broadcast signals (Hormones/Neuromodulators)
    pub global_signals: Vec<AtomicI32>,
}

impl InputBus {
    /// Creates a new InputBus with the specified number of neurons.
    /// All buffers (somatic, dendritic, and modality-specific) are initialized to zero.
    pub fn new(size: usize) -> Self {
        let mut name_map = HashMap::new();
        let mut channels = Vec::new();

        let mut add_chan = |name: &str| {
            let idx = channels.len();
            channels.push((0..size).map(|_| AtomicI32::new(0)).collect::<Vec<_>>());
            name_map.insert(name.to_string(), idx);
            idx
        };

        let proximal_idx = add_chan("proximal");
        let distal_idx = add_chan("distal");
        let apical_idx = add_chan("apical");
        let basal_idx = add_chan("basal");

        add_chan("modality:vision");
        add_chan("modality:text");
        add_chan("modality:audio");
        add_chan("modality:other");

        let global_signals = (0..16).map(|_| AtomicI32::new(0)).collect();

        Self {
            size,
            channels,
            name_map,
            global_signals,
            proximal_idx,
            distal_idx,
            apical_idx,
            basal_idx,
        }
    }

    /// Registers a new custom channel by name. Returns its ID.
    pub fn register_custom_channel(&mut self, name: &str) -> ChannelId {
        if let Some(&idx) = self.name_map.get(name) {
            return ChannelId(idx);
        }
        let idx = self.channels.len();
        self.channels.push((0..self.size).map(|_| AtomicI32::new(0)).collect::<Vec<_>>());
        self.name_map.insert(name.to_string(), idx);
        ChannelId(idx)
    }

    pub fn get_channel_id(&self, name: &str) -> Option<ChannelId> {
        self.name_map.get(name).map(|&idx| ChannelId(idx))
    }

    /// Fast access to a channel by ID.
    #[inline]
    pub fn channel(&self, id: ChannelId) -> &Vec<AtomicI32> {
        &self.channels[id.0]
    }

    /// Fast access to a channel by ID (mutable).
    #[inline]
    pub fn channel_mut(&mut self, id: ChannelId) -> &mut Vec<AtomicI32> {
        &mut self.channels[id.0]
    }

    /// Helper for core compartments.
    pub fn proximal(&self) -> &Vec<AtomicI32> { &self.channels[self.proximal_idx] }
    pub fn distal(&self) -> &Vec<AtomicI32> { &self.channels[self.distal_idx] }
    pub fn apical(&self) -> &Vec<AtomicI32> { &self.channels[self.apical_idx] }
    pub fn basal(&self) -> &Vec<AtomicI32> { &self.channels[self.basal_idx] }

    /// Thread-safe parallel clear of all InputBus buffers.
    pub fn clear(&self) {
        use rayon::prelude::*;
        self.channels.par_iter().for_each(|chan| {
            chan.par_iter().for_each(|v| v.store(0, Ordering::Relaxed));
        });
        self.global_signals.iter().for_each(|v| v.store(0, Ordering::Relaxed));
    }

    /// Faster clear when unique access is available, using raw memory fill.
    pub fn clear_mut(&mut self) {
        for chan in &mut self.channels {
            let ptr = chan.as_mut_ptr() as *mut i32;
            let len = chan.len();
            unsafe {
                std::ptr::write_bytes(ptr, 0, len);
            }
        }
        for v in &mut self.global_signals {
            v.store(0, Ordering::Relaxed);
        }
    }

    pub fn set_modality(&self, modality: Modality, index: usize, val: i32) {
        let name = match modality {
            Modality::Vision => "modality:vision",
            Modality::Text => "modality:text",
            Modality::Audio => "modality:audio",
            Modality::Other => "modality:other",
        };
        if let Some(id) = self.get_channel_id(name) {
            let chan = self.channel(id);
            if index < chan.len() {
                Self::atomic_saturating_add(&chan[index], val);
            } else {
                log::warn!("Modality {} index {} out of bounds (len: {})", name, index, chan.len());
            }
        }
    }

    pub fn get_modality(&self, modality: Modality, index: usize) -> i32 {
        let name = match modality {
            Modality::Vision => "modality:vision",
            Modality::Text => "modality:text",
            Modality::Audio => "modality:audio",
            Modality::Other => "modality:other",
        };
        if let Some(id) = self.get_channel_id(name) {
            let chan = self.channel(id);
            if index < chan.len() {
                return chan[index].load(Ordering::Relaxed);
            }
        }
        0
    }

    pub fn atomic_saturating_add(target: &AtomicI32, val: i32) {
        let mut current = target.load(Ordering::Relaxed);
        loop {
            let next = current.saturating_add(val);
            match target.compare_exchange_weak(current, next, Ordering::SeqCst, Ordering::Relaxed) {
                Ok(_) => break,
                Err(updated) => current = updated,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_bus_atomic_saturating_add(initial in -100000i32..100000, add in -100000i32..100000) {
            let atomic = AtomicI32::new(initial);
            InputBus::atomic_saturating_add(&atomic, add);
            let expected = initial.saturating_add(add);
            assert_eq!(atomic.load(Ordering::Relaxed), expected);
        }

        #[test]
        fn test_bus_bounds_safety(idx in 0usize..200, val in -1000i32..1000) {
            let bus = InputBus::new(100);
            // This should not panic even if idx >= 100
            bus.set_modality(Modality::Vision, idx, val);
            if idx < 100 {
                assert_eq!(bus.get_modality(Modality::Vision, idx), val);
            }
        }
    }
}
