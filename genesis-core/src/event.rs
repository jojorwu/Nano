use serde::{Serialize, Deserialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GlobalEvent {
    ObjectRecognized(String, f32), // label, confidence
    LowEnergy(u32),               // block_id
    HighSurprise(i32),            // surprise level
    RewardSignal(i32),            // reward value
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

    pub fn poll_all(&self) -> Vec<GlobalEvent> {
        let mut results = Vec::new();
        while let Some(ev) = self.events.pop() {
            results.push(ev);
        }
        results
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}
