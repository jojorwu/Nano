import genesis_client
import numpy as np
import matplotlib.pyplot as plt

def run_demo():
    print("=== Genesis SDK Programmatic Builder & NumPy Demo ===")

    # 1. Design a simple network
    builder = genesis_client.BrainBuilder("DemoNetwork")
    builder.add_layer("Input", 100) \
           .add_layer("Processing", 200) \
           .connect("Input", "Processing", weight=800, density=0.1) \
           .add_module("think", ticks=5) \
           .set_config(learning_rate=20)

    print(f"Created blueprint for {builder.name}")
    builder.build_blueprint("demo.toml")

    # 2. Simulation and Data Analysis (Assuming model is baked)
    # For this demo, we simulate the analysis workflow
    try:
        # rt = genesis_client.NanoRuntime("demo.model")
        print("\n[Analysis Workflow Simulation]")

        # Simulated data (in reality, these come from rt.get_potentials())
        n_neurons = 300
        potentials = np.random.normal(0, 500, n_neurons).astype(np.int32)

        print(f"Captured potentials for {n_neurons} neurons via Zero-Copy NumPy view")
        print(f"Mean potential: {np.mean(potentials):.2f}")
        print(f"Max activity: {np.max(potentials)}")

        # High-speed analysis using NumPy vector operations
        active_mask = potentials > 1024
        n_active = np.sum(active_mask)
        print(f"Neurons above firing threshold: {n_active} ({n_active/n_neurons*100:.1f}%)")

        # 3. Visualization
        plt.figure(figsize=(10, 4))
        plt.hist(potentials, bins=50, color='skyblue', edgecolor='black')
        plt.title("Neural Potential Distribution (via NumPy SDK)")
        plt.xlabel("Potential (IValue)")
        plt.ylabel("Neuron Count")
        plt.grid(alpha=0.3)
        print("\nPlot generated. Close window to continue.")
        # plt.show() # Disabled in sandbox
        plt.savefig("potentials_distribution.png")
        print("Visualization saved to potentials_distribution.png")

    except Exception as e:
        print(f"Workflow notice: {e}")

if __name__ == "__main__":
    run_demo()
