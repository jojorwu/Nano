use pyo3::prelude::*;
use genesis_node::{Runtime, SimulationSettings};
use genesis_core::{InputBus, Modality};

#[pyclass]
struct PyRuntime {
    inner: Runtime,
}

#[pymethods]
impl PyRuntime {
    #[new]
    fn new(model_path: &str, backend: Option<String>) -> PyResult<Self> {
        let mut settings = SimulationSettings::default();
        if let Some(b) = backend {
            settings.preferred_backend = Some(b);
        }
        let runtime = Runtime::load_with_settings(model_path, settings)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        Ok(PyRuntime { inner: runtime })
    }

    fn tick(&mut self, inputs: Vec<i32>) -> PyResult<Vec<bool>> {
        let spikes = self.inner.tick(&inputs);
        Ok(spikes)
    }

    fn inject_text(&mut self, text: &str) {
        self.inner.inject_text(text);
    }

    fn handle_command(&mut self, cmd: &str) -> String {
        self.inner.handle_command(cmd)
    }

    fn save(&mut self, path: &str) -> PyResult<()> {
        self.inner.sync_state();
        self.inner.engine.model.save(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))
    }

    fn neuron_count(&self) -> usize {
        self.inner.engine.model.neurons.len()
    }
}

#[pymodule]
fn genesis_python(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyRuntime>()?;
    Ok(())
}
