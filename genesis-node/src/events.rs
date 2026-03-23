use genesis_core::SpikeData;

#[derive(Clone, Debug)]
pub enum SimulationEvent {
    TickStarted(u32),
    TickComplete {
        tick: u32,
        spike_count: usize,
        data: SpikeData,
    },
    RewardReceived(i32),
    NightPhaseStarted(u32),
    NightPhaseComplete(u32),
    SurpriseDetected(i32),
}

pub trait SimulationObserver: Send + Sync {
    fn on_event(&mut self, event: &SimulationEvent, engine: &mut crate::engine::SimulationEngine);
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}
