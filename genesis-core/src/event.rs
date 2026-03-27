use serde::{Serialize, Deserialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GlobalEvent {
    ObjectRecognized(String, f32), // label, confidence
    LowEnergy(u32),               // block_id
    HighSurprise(i32),            // surprise level
    RewardSignal(i32),            // reward value
    ConsolidationTriggered,       // Start of Night Phase / Replay
    BlockTriggered(u32),          // block_id was activated (e.g. by Titan)
    ModulationShift(usize, i32),  // signal_id, delta
    StructuralUpdate(String),     // e.g. "Neurogenesis"
    Custom(String, Vec<u8>),
}

/// Global Event Bus for high-level asynchronous communication between modules.
pub struct EventBus {
    pub events: Arc<crossbeam_queue::SegQueue<GlobalEvent>>,
}

impl EventBus {
    pub fn new() -> Self {
        Self {
            events: Arc::new(crossbeam_queue::SegQueue::new()),
        }
    }

    pub fn publish(&self, event: GlobalEvent) {
        self.events.push(event);
    }

    pub fn drain_all(&self) -> Vec<GlobalEvent> {
        let mut results = Vec::new();
        while let Some(ev) = self.events.pop() {
            results.push(ev);
        }
        results
    }

    pub fn clear(&self) {
        while self.events.pop().is_some() {}
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}
