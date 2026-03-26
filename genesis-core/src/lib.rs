use serde::{Serialize, Deserialize};

pub mod bus;
pub mod module;
pub mod model;
pub mod physics;
pub mod config;

#[cfg(feature = "titan")]
pub mod titan;
#[cfg(feature = "text")]
pub mod text;
#[cfg(feature = "vision")]
pub mod vision;
#[cfg(feature = "audio")]
pub mod audio;
#[cfg(feature = "rl")]
pub mod rl;
#[cfg(feature = "fusion")]
pub mod fusion;
pub mod graph;
pub mod robotics;
pub mod plasticity;
pub mod plasticity_rules;
pub mod attention;
pub mod workspace;
pub mod hierarchical;
pub mod episodic;
pub mod curiosity;
pub mod think;

// Re-exports for convenience and compatibility
pub use bus::{InputBus, Modality};
pub use module::{NanoModule, ModuleManager, ModuleRegistry, ModuleError, ModuleInput, ForeignModule, ForeignTickFn};
pub use model::{NeuronsSoA, SynapsesSoA, BakedModel, SpikeData, Compartment, LatentSynapseMatrix, NeuronsFFI, SynapsesFFI};
pub use attention::AttnResModule;
pub use workspace::WorkspaceModule;
pub use hierarchical::HierarchicalModule;
pub use episodic::EpisodicModule;
pub use curiosity::CuriosityModule;
pub use think::ThinkModule;
pub use plasticity_rules::{PlasticityContext, PlasticityRule, GsopRule};
pub use config::{NetworkConfig, PhysicsConfig, PlasticityConfig, TitanConfig, ModuleConfig, AstroConfig, HardwareConfig};

pub type IValue = i32;
pub const SCALE: IValue = 1024; // 2^10 for bit-shift optimizations

// --- Physics Constants (Legacy/Defaults - prefer config fields) ---
pub const DEFAULT_REFRACTORY_TICKS: i32 = 4;
pub const SMBP_DECAY: i64 = 800; // Multiplier out of SCALE
pub const ACTIVITY_EMA_ALPHA: i64 = 99; // Alpha out of 100
pub const TARGET_ACTIVITY_LEVEL: IValue = 100; // 10% target firing rate
pub const WEIGHT_CLAMP_LIMIT: IValue = SCALE * 5;

// --- Dendritic Gating Constants ---
pub const GATE_OPEN: IValue = SCALE;
pub const GATE_HALF: IValue = SCALE / 2;
pub const GATE_QUARTER: IValue = SCALE / 4;
pub const GATE_THREE_QUARTERS: IValue = SCALE * 3 / 4;

/// Represents the global chemical state of the network.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default)]
pub struct NeuromodulationState {
    pub dopamine: IValue,       // Reward / Prediction Error
    pub noradrenaline: IValue,  // Surprise / Novelty / Arousal
    pub serotonin: IValue,      // Stability / Risk Mitigation
}



#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    #[test]
    fn test_neurons_validation() {
        let mut neurons = NeuronsSoA::new(10);
        assert!(neurons.validate().is_ok());

        neurons.potential.push(0); // Break length consistency
        assert!(neurons.validate().is_err());
    }

    #[test]
    fn test_input_bus_parallel_injection() {
        use std::sync::Arc;
        let bus = Arc::new(InputBus::new(100));
        let threads: Vec<_> = (0..10).map(|_| {
            let b = Arc::clone(&bus);
            std::thread::spawn(move || {
                for i in 0..100 {
                    InputBus::atomic_saturating_add(&b.proximal()[i], 10);
                }
            })
        }).collect();

        for t in threads { t.join().unwrap(); }
        for i in 0..100 {
            assert_eq!(bus.proximal()[i].load(Ordering::Relaxed), 100);
        }

        bus.clear();
        assert_eq!(bus.proximal()[0].load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_input_bus_clear_mut() {
        let mut bus = InputBus::new(100);
        bus.proximal()[50].store(500, Ordering::Relaxed);
        bus.clear_mut();
        assert_eq!(bus.proximal()[50].load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_neurons_init_and_grow() {
        let mut neurons = NeuronsSoA::new(10);
        assert_eq!(neurons.len(), 10);
        assert_eq!(neurons.dendritic_gate[0], SCALE);
        assert_eq!(neurons.layer_id[0], 0);

        neurons.grow(5);
        assert_eq!(neurons.len(), 15);
        assert_eq!(neurons.dendritic_gate[14], SCALE);
        assert_eq!(neurons.potential[14], 0);

        neurons.shrink_to_fit();
        assert_eq!(neurons.len(), 15);
    }
}
