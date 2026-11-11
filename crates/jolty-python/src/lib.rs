//! The Python extension module `jolty._core`.
//!
//! It exposes one class, [`Simulator`], that elaborates a circuit from its JSON text and runs
//! analyses described in JSON, returning arrays as `NumPy` arrays. Everything an agent or a
//! person reads, from units to reports, is built on top of it in Python.

use std::sync::Mutex;

use jolty_engine::{Analysis, OperatingPoint, Options, Outcome, Statistics};
use jolty_ir::Circuit;
use jolty_models::{Library, Quantity};
use numpy::{PyArray1, PyArray2};
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

create_exception!(
    _core,
    ElaborationError,
    PyException,
    "A circuit cannot be elaborated. Its only argument lists every problem as a string."
);
create_exception!(_core, SimulationError, PyException, "An analysis failed.");

/// A circuit elaborated for simulation.
#[pyclass(module = "jolty._core", frozen)]
struct Simulator {
    inner: Mutex<jolty_engine::Simulator>,
    unknowns: Vec<(String, &'static str)>,
    devices: usize,
}

#[pymethods]
impl Simulator {
    /// Elaborates the circuit whose JSON text is `circuit`.
    #[new]
    fn new(circuit: &str) -> PyResult<Self> {
        let circuit = Circuit::from_json(circuit).map_err(|error| unreadable(&error))?;
        let inner =
            jolty_engine::Simulator::new(&circuit, &Library::standard()).map_err(|error| {
                let problems: Vec<String> =
                    error.problems().iter().map(ToString::to_string).collect();
                ElaborationError::new_err((problems,))
            })?;
        let unknowns = inner
            .unknowns()
            .iter()
            .map(|unknown| {
                let quantity = match unknown.quantity {
                    Quantity::Voltage => "voltage",
                    Quantity::Current => "current",
                };
                (unknown.name.clone(), quantity)
            })
            .collect();
        let devices = inner.device_count();
        Ok(Self {
            inner: Mutex::new(inner),
            unknowns,
            devices,
        })
    }

    /// Each unknown's name and quantity, `"voltage"` or `"current"`, in the order of every
    /// solution's values.
    #[getter]
    fn unknowns(&self) -> Vec<(String, &'static str)> {
        self.unknowns.clone()
    }

    /// The number of devices, with every component instance unfolded.
    #[getter]
    fn devices(&self) -> usize {
        self.devices
    }

    /// Runs the analysis whose JSON text is `analysis` with the options whose JSON text is
    /// `options`, releasing the interpreter while it runs.
    fn run<'py>(
        &self,
        py: Python<'py>,
        analysis: &str,
        options: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let analysis: Analysis =
            serde_json::from_str(analysis).map_err(|error| unreadable(&error))?;
        let options: Options = serde_json::from_str(options).map_err(|error| unreadable(&error))?;
        let outcome = py.detach(|| {
            let mut simulator = self
                .inner
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            simulator.run(&analysis, &options)
        });
        outcome_to_python(
            py,
            outcome.map_err(|error| SimulationError::new_err(error.to_string()))?,
        )
    }
}

/// A `ValueError` for input the engine cannot read.
fn unreadable(error: &impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// The outcome of an analysis as a dictionary whose `kind` names the analysis.
fn outcome_to_python(py: Python<'_>, outcome: Outcome) -> PyResult<Bound<'_, PyDict>> {
    let result = PyDict::new(py);
    match outcome {
        Outcome::OperatingPoint(operating_point) => {
            result.set_item("kind", "operating_point")?;
            add_operating_point(&result, operating_point)?;
        }
        Outcome::DcSweep(sweep) => {
            result.set_item("kind", "dc_sweep")?;
            result.set_item("levels", PyArray1::from_vec(py, sweep.levels))?;
            result.set_item("values", PyArray2::from_vec2(py, &sweep.values)?)?;
        }
        Outcome::Ac(response) => {
            result.set_item("kind", "ac")?;
            result.set_item("frequencies", PyArray1::from_vec(py, response.frequencies))?;
            result.set_item("values", PyArray2::from_vec2(py, &response.values)?)?;
            let operating_point = PyDict::new(py);
            add_operating_point(&operating_point, response.operating_point)?;
            result.set_item("operating_point", operating_point)?;
        }
        Outcome::Transient(waveforms) => {
            result.set_item("kind", "transient")?;
            result.set_item("times", PyArray1::from_vec(py, waveforms.times))?;
            result.set_item("values", PyArray2::from_vec2(py, &waveforms.values)?)?;
            result.set_item(
                "statistics",
                statistics_to_python(py, waveforms.statistics)?,
            )?;
        }
    }
    Ok(result)
}

fn add_operating_point(
    result: &Bound<'_, PyDict>,
    operating_point: OperatingPoint,
) -> PyResult<()> {
    result.set_item(
        "values",
        PyArray1::from_vec(result.py(), operating_point.values),
    )?;
    result.set_item("strategy", operating_point.strategy.name())
}

fn statistics_to_python(py: Python<'_>, statistics: Statistics) -> PyResult<Bound<'_, PyDict>> {
    let result = PyDict::new(py);
    result.set_item("initial", statistics.initial.name())?;
    result.set_item("accepted", statistics.accepted)?;
    result.set_item("rejected", statistics.rejected)?;
    result.set_item("iterations", statistics.iterations)?;
    Ok(result)
}

/// The Rust core of jolty.
#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add_class::<Simulator>()?;
    module.add("ElaborationError", py.get_type::<ElaborationError>())?;
    module.add("SimulationError", py.get_type::<SimulationError>())?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
