use clap::{Parser, Subcommand};
use std::fs;
use genesis_baker::ModelBlueprint;
use genesis_node::Runtime;
#[cfg(feature = "text")]
use genesis_core::text::SpikingTextModule;
#[cfg(feature = "vision")]
use genesis_core::vision::SpikingVisionModule;
#[cfg(feature = "vision")]
use image::GenericImageView;

#[derive(Parser)]
struct Cli {
    #[command(subcommand)] command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Bake { #[arg(short, long)] blueprint: String, #[arg(short, long, default_value = "model.state")] output: String },
    Run {
        #[arg(short, long)] model: String,
        #[arg(short, long)] input: Option<String>,
        #[cfg(feature = "vision")]
        #[arg(short = 'g', long)] image: Option<String>,
    },
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
        Commands::Run { model, input, #[cfg(feature = "vision")] image } => {
            println!("🚀 Loading model: {}", model);
            let mut runtime = Runtime::load(model).expect("Failed to load model");
            let initial_synapses = runtime.model.synapses.len();

            if let Some(text) = input {
                println!("📝 Text Input: '{}'", text);
                #[cfg(feature = "text")]
                {
                    let tokens = {
                        let mut text_mod = SpikingTextModule::new(&mut runtime.model.vocabulary);
                        text_mod.tokenize(text)
                    };
                    for token in tokens {
                        let mut inputs = vec![0; runtime.model.neurons.len()];
                        {
                            let text_mod = SpikingTextModule::new(&mut runtime.model.vocabulary);
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
                    println!("❌ Text processing is disabled in this build.");
                }
            }

            #[cfg(feature = "vision")]
            if let Some(img_path) = image {
                println!("🖼️ Image Input: '{}'", img_path);
                let img = image::open(img_path).expect("Failed to open image");
                let (w, h) = img.dimensions();
                println!("   Resolution: {}x{}", w, h);
                let vision_mod = SpikingVisionModule::new(w, h);
                let gray = img.to_luma8();
                let pixel_potentials = vision_mod.rate_encode(gray.as_raw());
                let mut inputs = vec![0; runtime.model.neurons.len()];
                for (i, &pot) in pixel_potentials.iter().enumerate() {
                    if i < inputs.len() { inputs[i] = pot; }
                }
                let spikes = runtime.tick(&inputs);
                println!("   Generated {} spikes from image", spikes.iter().filter(|&&s| s).count());
            }

            let final_synapses = runtime.model.synapses.len();
            println!("✨ Simulation finished.");
            println!("   Structural Evolution: {} -> {} synapses", initial_synapses, final_synapses);

            runtime.model.save(model).expect("Failed to auto-save model");
            println!("💾 Model state and vocabulary saved.");
        }
    }
}
