use genesis_node::Runtime;

fn main() {
    let mut runtime = Runtime::load("model.state").expect("Failed to load");
    println!("Runtime started with {} neurons", runtime.model.neurons.len());
}
