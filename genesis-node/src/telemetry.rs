use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct Telemetry {
    pub spike_counts: Vec<usize>,
    pub episode_rewards: Vec<i32>,
}
