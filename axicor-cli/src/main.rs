use clap::{Parser, Subcommand};
use std::fs;
use genesis_baker::ModelBlueprint;
use genesis_node::Runtime;
use genesis_core::text::SpikingTextModule;

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
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
            println!("Model '{}' baked to {}. Neurons: {}", bp.name, output, baked.neurons.len());
        }
        Commands::Run { model, input } => {
            println!("Loading model: {}", model);
            let mut runtime = Runtime::load(model).expect("Failed to load model");

            if let Some(text) = input {
                println!("Input: '{}'", text);
                let mut text_mod = SpikingTextModule::new(5000, 256);
                let tokens = text_mod.tokenize(text);
                for token in tokens {
                    let pattern = text_mod.encode(token, 10); // use first 10 neurons
                    let mut inputs = vec![0; runtime.model.neurons.len()];
                    for (i, &spiked) in pattern.iter().enumerate() {
                        if spiked { inputs[i] = 1000; }
                    }
                    let spikes = runtime.tick(&inputs);
                    let spike_count = spikes.iter().filter(|&&s| s).count();
                    println!("Token {}: Spikes generated: {}", token, spike_count);
                }
            } else {
                println!("No input provided. Ticking 10 times...");
                for _ in 0..10 {
                    runtime.tick(&[]);
                }
            }
        }
    }
}
