use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use genesis_node::{RuntimeBuilder, SimulationSettings};
use genesis_core::{BakedModel, NeuronsSoA, SynapsesSoA};
use std::collections::HashMap;

fn bench_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("Simulation Scaling");

    for &size in [10_000, 100_000, 1_000_000].iter() {
        // Benchmark Tick
        group.bench_with_input(BenchmarkId::new("Tick", size), &size, |b, &size| {
            let mut model = BakedModel {
                version: "5.0".to_string(),
                config: Default::default(),
                node_id: 0,
                local_range: (0, size),
                neurons: NeuronsSoA::new(size),
                synapses: SynapsesSoA::with_capacity(size),
                module_states: HashMap::new(),
                has_text: false,
                has_vision: false,
                has_audio: false,
                has_robotics: false,
                has_fusion: false,
                vocabulary: HashMap::new(),
                #[cfg(feature = "titan")]
                titan_memory: None,
            };

            for i in 0..size {
                model.neurons.block_id[i] = (i / 16) as u32;
            }

            let settings = SimulationSettings {
                night_phase_interval: 100,
                preferred_backend: Some("cpu".to_string()),
                ..Default::default()
            };

            let mut rt = RuntimeBuilder::new()
                .from_model_direct(model)
                .with_settings(settings)
                .build()
                .unwrap();

            let inputs = vec![0i32; size];
            b.iter(|| {
                rt.tick(&inputs);
            });
        });

        // Benchmark Night Phase
        group.bench_with_input(BenchmarkId::new("NightPhase", size), &size, |b, &size| {
            let mut model = BakedModel {
                version: "5.0".to_string(),
                config: Default::default(),
                node_id: 0,
                local_range: (0, size),
                neurons: NeuronsSoA::new(size),
                synapses: SynapsesSoA::with_capacity(size),
                module_states: HashMap::new(),
                has_text: false,
                has_vision: false,
                has_audio: false,
                has_robotics: false,
                has_fusion: false,
                vocabulary: HashMap::new(),
                #[cfg(feature = "titan")]
                titan_memory: None,
            };

            for i in 0..size {
                model.neurons.block_id[i] = (i / 16) as u32;
            }

            // Add Titan module to test consolidation
            model.module_states.insert("titan".to_string(), Vec::new());

            let settings = SimulationSettings {
                night_phase_interval: 10, // More frequent for bench
                preferred_backend: Some("cpu".to_string()),
                ..Default::default()
            };

            let mut rt = RuntimeBuilder::new()
                .from_model_direct(model)
                .with_settings(settings)
                .build()
                .unwrap();

            let inputs = vec![0i32; size];
            // Run enough ticks to trigger a night phase in the next tick
            for _ in 0..9 {
                rt.tick(&inputs);
            }

            b.iter(|| {
                // This tick will trigger night phase
                rt.tick(&inputs);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_scaling);
criterion_main!(benches);
