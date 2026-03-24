import sys
import os

# Attempt to find the Rust extension library in common build locations
_current_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_current_dir, "..", ".."))

_search_paths = [
    _current_dir, # Packaged location
    os.path.join(_repo_root, "target", "debug"),
    os.path.join(_repo_root, "target", "release"),
]

for _path in _search_paths:
    if os.path.exists(_path):
        sys.path.append(_path)

try:
    from genesis_python import PyRuntime
except ImportError:
    # Fallback: check if it's already in path or installed
    try:
        from genesis_python import PyRuntime
    except ImportError:
        PyRuntime = None

class NanoRuntime:
    def __init__(self, model_path: str, backend: str = "cpu"):
        if PyRuntime is None:
            raise ImportError("Could not find genesis_python shared library. Please build with 'cargo build -p genesis-python'.")
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

    def get_potentials(self, copy: bool = True):
        """Returns neural potentials as a NumPy array."""
        if copy:
            return self._inner.get_potentials()
        else:
            return self._inner.get_potentials_view()

    def set_potential(self, index: int, val: int):
        self._inner.set_potential(index, val)

    def get_thresholds(self, copy: bool = True):
        """Returns neural thresholds as a NumPy array."""
        if copy:
            return self._inner.get_thresholds()
        else:
            return self._inner.get_thresholds_view()

    def add_module(self, name: str, callback: callable):
        self._inner.add_python_module(name, callback)

class BrainBuilder:
    """Helper class for building TOML blueprints and model configurations from Python."""
    def __init__(self, name: str):
        self.name = name
        self.layers = {}
        self.connections = []
        self.modules = []
        self.global_config = {
            "default_threshold": 1024,
            "learning_rate": 10
        }

    def add_layer(self, name: str, size: int, excitatory: bool = True):
        self.layers[name] = {
            "size": size,
            "is_excitatory": excitatory,
            "start_index": sum(l["size"] for l in self.layers.values())
        }
        return self

    def connect(self, source: str, target: str, weight: int = 500, pattern: str = "all-to-all", density: float = 1.0):
        self.connections.append({
            "source": source,
            "target": target,
            "weight": weight,
            "pattern": pattern,
            "density": density
        })
        return self

    def add_module(self, module_type: str, **kwargs):
        self.modules.append({"type": module_type, **kwargs})
        return self

    def set_config(self, **kwargs):
        self.global_config.update(kwargs)
        return self

    def to_toml(self) -> str:
        import toml
        total_neurons = sum(l["size"] for l in self.layers.values())
        data = {
            "name": self.name,
            "config": self.global_config,
            "architecture": {
                "neuron_count": total_neurons,
                "layers": self.layers,
                "connections": self.connections
            },
            "modules": self.modules
        }
        return toml.dumps(data)

    def build_blueprint(self, output_path: str):
        with open(output_path, "w") as f:
            f.write(self.to_toml())
        print(f"Blueprint saved to {output_path}. Use 'nano-cli bake' to generate the binary model.")
