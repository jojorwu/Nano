try:
    from .genesis_python import PyRuntime
except ImportError:
    # If the shared library is not in the same directory, try loading from typical build locations
    import sys
    import os
    # Placeholder for dynamic loading or user instructions
    pass

class NanoRuntime:
    def __init__(self, model_path: str, backend: str = "cpu"):
        self._inner = PyRuntime(model_path, backend)

    def tick(self, inputs: list[int]) -> list[bool]:
        return self._inner.tick(inputs)

    def inject_text(self, text: str):
        self._inner.inject_text(text)

    def handle_command(self, cmd: str) -> str:
        return self._inner.handle_command(cmd)

    def save(self, path: str):
        self._inner.save(path)

    @property
    def neuron_count(self) -> int:
        return self._inner.neuron_count()

class BrainBuilder:
    """Helper class for building TOML blueprints from Python."""
    def __init__(self, name: str):
        self.name = name
        self.neurons = 1000
        self.synapses = 5000
        self.modules = []

    def add_module(self, module_type: str, **kwargs):
        self.modules.append({"type": module_type, **kwargs})

    def to_toml(self) -> str:
        import toml
        data = {
            "name": self.name,
            "architecture": {
                "neuron_count": self.neurons,
                "synapse_count": self.synapses
            },
            "modules": self.modules
        }
        return toml.dumps(data)

    def build(self, output_path: str):
        with open("temp_blueprint.toml", "w") as f:
            f.write(self.to_toml())
        # Here we would call genesis-baker (potentially via FFI too)
        print(f"Blueprint saved to temp_blueprint.toml. Run 'nano-cli bake' to create the model.")
