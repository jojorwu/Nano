use pyo3::prelude::*;
use numpy::{PyArray1, ToPyArray};
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

    fn add_python_module(&mut self, name: String, callback: PyObject) {
        let n_count = self.inner.engine.model.neurons.len();
        let module = PythonModuleProxy {
            name,
            callback,
            n_count,
        };
        self.inner.engine.modules.add_module(Box::new(module));
    }

    fn get_potentials<'py>(&self, py: Python<'py>) -> &'py PyArray1<i32> {
        self.inner.engine.model.neurons.potential.to_pyarray(py)
    }

    /// Returns a zero-copy view of the potentials.
    /// WARNING: The returned array is a view into Rust memory.
    /// Ticking the simulation or growing the network may invalidate this view or change its contents.
    fn get_potentials_view<'py>(&self, py: Python<'py>) -> &'py PyArray1<i32> {
        let pot = &self.inner.engine.model.neurons.potential;
        unsafe {
            numpy::ndarray::ArrayView1::from_shape_ptr(pot.len(), pot.as_ptr()).to_pyarray(py)
        }
    }

    fn set_potential(&mut self, index: usize, val: i32) {
        if index < self.inner.engine.model.neurons.len() {
            self.inner.engine.model.neurons.potential[index] = val;
        }
    }

    fn get_thresholds<'py>(&self, py: Python<'py>) -> &'py PyArray1<i32> {
        self.inner.engine.model.neurons.threshold.to_pyarray(py)
    }

    fn get_thresholds_view<'py>(&self, py: Python<'py>) -> &'py PyArray1<i32> {
        let thr = &self.inner.engine.model.neurons.threshold;
        unsafe {
            numpy::ndarray::ArrayView1::from_shape_ptr(thr.len(), thr.as_ptr()).to_pyarray(py)
        }
    }
}

#[derive(Clone)]
struct PythonModuleProxy {
    name: String,
    callback: PyObject,
    n_count: usize,
}

impl genesis_core::NanoModule for PythonModuleProxy {
    fn name(&self) -> &str { &self.name }
    fn box_clone(&self) -> Box<dyn genesis_core::NanoModule> { Box::new(self.clone()) }
    fn on_tick(&mut self, _bus: &genesis_core::InputBus, _previous_spikes: &[bool], tick: u32) {
        Python::with_gil(|py| {
            let _ = self.callback.call1(py, (tick,));
            // In a real implementation, we would pass the bus to the callback
            // but that requires wrapping InputBus in a PyClass
        });
    }
    fn on_update_weights(&mut self, _: &mut genesis_core::NeuronsSoA, _: &[bool], _: &[bool], _: u32, _: Option<genesis_core::IValue>) {}
    fn on_night_phase(&mut self, _: &mut genesis_core::NeuronsSoA, _: &mut genesis_core::SynapsesSoA, _: Option<genesis_core::IValue>) {}
}

#[pymodule]
fn genesis_python(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyRuntime>()?;
    Ok(())
}
