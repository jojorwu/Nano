use clap::{Parser, Subcommand};
use std::fs;
use genesis_baker::ModelBlueprint;
use genesis_node::Runtime;
#[cfg(feature = "text")]
use genesis_core::text::SpikingTextModule;

#[derive(Parser)]
struct Cli {
    #[command(subcommand)] command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Bake { #[arg(short, long)] blueprint: String, #[arg(short, long, default_value = "model.state")] output: String },
    Run { #[arg(short, long)] model: String, #[arg(short, long)] input: Option<String> },
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    match &cli.command {
        Commands::Bake { blueprint, output } => {
            let content = fs::read_to_string(blueprint).expect("Failed to read");
            let bp: ModelBlueprint = toml::from_str(&content).expect("Invalid");
            let baked = bp.bake();
            baked.save(output).expect("Failed to save");
            println!("✅ Model '{}' baked to {}.", bp.name, output);
            println!("   Neurons: {}, Synapses: {}", baked.neurons.len(), baked.synapses.len());
        }
        Commands::Run { model, input } => {
            println!("🚀 Loading model: {}", model);
            let mut runtime = Runtime::load(model).expect("Failed to load model");
            let initial_synapses = runtime.model.synapses.len();

            if let Some(text) = input {
                println!("📝 Input: '{}'", text);

                #[cfg(feature = "text")]
                {
                    // Encapsulate everything text-related to avoid borrow conflicts
                    let tokens = {
                        let mut text_mod = SpikingTextModule::new(&mut runtime.model.vocabulary);
                        text_mod.tokenize(text)
                    };

                    for token in tokens {
                        let mut inputs = vec![0; runtime.model.neurons.len()];
                        {
                            let text_mod = SpikingTextModule::new(&mut runtime.model.vocabulary);
                            // Adjust encoding to use a significant portion of the network
                            let pattern_len = (runtime.model.neurons.len() / 4).min(256).max(10);
                            let pattern = text_mod.encode(token, pattern_len);
                            for (i, &spiked) in pattern.iter().enumerate() {
                                if spiked { inputs[i] = 1000; }
                            }
                        }

                        let spikes = runtime.tick(&inputs);
                        println!("   Token {}: Generated {} spikes", token, spikes.iter().filter(|&&s| s).count());
                    }
                }
                #[cfg(not(feature = "text"))]
                {
                    println!("❌ Text processing is disabled. Compile with 'text' feature to use it.");
                }
            } else {
                println!("🕒 Idle run (100 ticks)...");
                for _ in 0..100 { runtime.tick(&[]); }
            }

            let final_synapses = runtime.model.synapses.len();
            println!("✨ Simulation finished.");
            println!("   Structural Evolution: {} -> {} synapses", initial_synapses, final_synapses);

            runtime.model.save(model).expect("Failed to auto-save model after run");
            println!("💾 Model state and vocabulary saved to {}.", model);
        }
    }
}
