use serde::{Serialize, Deserialize};
use std::sync::atomic::{AtomicI32, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Modality {
    Vision = 0,
    Text = 1,
    Audio = 2,
    Other = 3,
}

/// Represents a modular functional unit within the Spiking Neural Network.
/// Modules can inject signals, observe activity, and manage their own internal plasticity rules.
pub struct InputBus {
    pub proximal: Vec<AtomicI32>,
    pub distal: Vec<AtomicI32>,
    pub apical: Vec<AtomicI32>,
    pub basal: Vec<AtomicI32>,

    // Modality-specific buffers for high-order fusion
    pub modalities: [Vec<AtomicI32>; 4],
}

impl InputBus {
    /// Creates a new InputBus with the specified number of neurons.
    /// All buffers (somatic, dendritic, and modality-specific) are initialized to zero.
    pub fn new(size: usize) -> Self {
        let make_vec = || (0..size).map(|_| AtomicI32::new(0)).collect::<Vec<_>>();
        Self {
            proximal: make_vec(),
            distal: make_vec(),
            apical: make_vec(),
            basal: make_vec(),
            modalities: [
                make_vec(), // Vision
                make_vec(), // Text
                make_vec(), // Audio
                make_vec(), // Other
            ],
        }
    }

    /// Thread-safe parallel clear of all InputBus buffers.
    pub fn clear(&self) {
        use rayon::prelude::*;

        self.proximal.par_iter().for_each(|v| v.store(0, Ordering::Relaxed));
        self.distal.par_iter().for_each(|v| v.store(0, Ordering::Relaxed));
        self.apical.par_iter().for_each(|v| v.store(0, Ordering::Relaxed));
        self.basal.par_iter().for_each(|v| v.store(0, Ordering::Relaxed));

        for vec in &self.modalities {
            vec.par_iter().for_each(|v| v.store(0, Ordering::Relaxed));
        }
    }

    /// Faster clear when unique access is available, using raw memory fill.
    pub fn clear_mut(&mut self) {
        fn clear_vec(v: &mut [AtomicI32]) {
            let ptr = v.as_mut_ptr() as *mut i32;
            let len = v.len();
            unsafe {
                std::ptr::write_bytes(ptr, 0, len);
            }
        }
        clear_vec(&mut self.proximal);
        clear_vec(&mut self.distal);
        clear_vec(&mut self.apical);
        clear_vec(&mut self.basal);

        for vec in &mut self.modalities {
            clear_vec(vec);
        }
    }

    pub fn set_modality(&self, modality: Modality, index: usize, val: i32) {
        let vec = &self.modalities[modality as usize];
        if index < vec.len() {
            Self::atomic_saturating_add(&vec[index], val);
        }
    }

    pub fn get_modality(&self, modality: Modality, index: usize) -> i32 {
        let vec = &self.modalities[modality as usize];
        if index < vec.len() {
            return vec[index].load(Ordering::Relaxed);
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
