//! The Python extension module `jolty._core`.
//!
//! It exposes one class, [`Simulator`], that elaborates a circuit from its JSON text, checks
//! its structure, and runs analyses described in JSON, returning arrays as `NumPy` arrays,
//! and the table of the structural checks' [`rules`]. Everything an agent or a person reads,
//! from units to reports, is built on top of it in Python.

use std::sync::Mutex;

use jolty_engine::{Analysis, OperatingPoint, Options, Outcome, Statistics, Terminals};
use jolty_ir::Circuit;
use jolty_models::{Level, Library, Quantity};
use numpy::{Element, PyArray1, PyArray2, PyArrayMethods};
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

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
    nets: Vec<String>,
    terminals: Vec<String>,
    devices: usize,
}

#[pymethods]
impl Simulator {
    /// Elaborates the circuit whose JSON text is `circuit`, its devices at `temperature`,
    /// in kelvins, with the parameters of the JSON object `overrides`, by instance path,
    /// replacing the circuit's own, and the models of the JSON object `substitutes`, each a
    /// `model` and its `parameters` by component instance path, standing in for them.
    #[new]
    fn new(circuit: &str, temperature: f64, overrides: &str, substitutes: &str) -> PyResult<Self> {
        let circuit = Circuit::from_json(circuit).map_err(|error| unreadable(&error))?;
        let conditions = jolty_engine::Conditions {
            temperature,
            overrides: serde_json::from_str(overrides).map_err(|error| unreadable(&error))?,
            substitutes: serde_json::from_str(substitutes).map_err(|error| unreadable(&error))?,
        };
        let inner = jolty_engine::Simulator::new(&circuit, &Library::standard(), &conditions)
            .map_err(|error| {
                let problems: Vec<String> =
                    error.problems().iter().map(ToString::to_string).collect();
                ElaborationError::new_err((problems,))
            })?;
        let unknowns = inner
            .unknowns()
            .iter()
            .map(|unknown| (unknown.name.clone(), quantity_name(unknown.quantity)))
            .collect();
        let nets = inner
            .logic_nets()
            .iter()
            .map(|net| net.name.clone())
            .collect();
        let devices = inner.devices().len() - inner.elements() + inner.logic_devices().len();
        let terminals = inner.terminals().to_vec();
        Ok(Self {
            inner: Mutex::new(inner),
            unknowns,
            nets,
            terminals,
            devices,
        })
    }

    /// Each unknown's name and quantity, `"voltage"` or `"current"`, in the order of every
    /// solution's values.
    #[getter]
    fn unknowns(&self) -> Vec<(String, &'static str)> {
        self.unknowns.clone()
    }

    /// Each logic net's name, in the order of every result's levels.
    #[getter]
    fn nets(&self) -> Vec<String> {
        self.nets.clone()
    }

    /// Every terminal of every device, such as `X1/R1.a`, in the order of every result's
    /// currents and charges.
    #[getter]
    fn terminals(&self) -> Vec<String> {
        self.terminals.clone()
    }

    /// The number of devices, analog and logic, with every component instance unfolded.
    #[getter]
    fn devices(&self) -> usize {
        self.devices
    }

    /// Every diagnostic of the structural checks, each a dictionary of its `rule`, its
    /// `message`, its `instances` and its `nets`; each instance is a dictionary of its `path`
    /// and its `source`, the script line that placed it or `None`.
    fn check<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let simulator = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let diagnostics = PyList::empty(py);
        for diagnostic in jolty_checks::check(&simulator) {
            let item = PyDict::new(py);
            item.set_item("rule", diagnostic.rule)?;
            item.set_item("message", diagnostic.message)?;
            let instances = PyList::empty(py);
            for location in diagnostic.instances {
                let instance = PyDict::new(py);
                instance.set_item("path", location.path)?;
                let source = match location.source {
                    Some(source) => {
                        let line = PyDict::new(py);
                        line.set_item("file", source.file)?;
                        line.set_item("line", source.line)?;
                        line.set_item("function", source.function)?;
                        Some(line)
                    }
                    None => None,
                };
                instance.set_item("source", source)?;
                instances.append(instance)?;
            }
            item.set_item("instances", instances)?;
            item.set_item("nets", diagnostic.nets)?;
            diagnostics.append(item)?;
        }
        Ok(diagnostics)
    }

    /// Where the last analysis used logic devices outside their domains, each a dictionary
    /// of the device's `path`, the `bound` it left, such as `"load"`, the domain's `low`
    /// and `high`, the `value` furthest outside, the `time` it first left, and the `count`
    /// of times it was read outside.
    fn excursions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let simulator = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let found = PyList::empty(py);
        for excursion in simulator.excursions() {
            let item = PyDict::new(py);
            item.set_item("path", simulator.logic_devices()[excursion.device].path())?;
            item.set_item("bound", excursion.bound.name())?;
            item.set_item("low", excursion.range.0)?;
            item.set_item("high", excursion.range.1)?;
            item.set_item("value", excursion.value)?;
            item.set_item("time", excursion.time)?;
            item.set_item("count", excursion.count)?;
            found.append(item)?;
        }
        Ok(found)
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

/// The name of a quantity: `"voltage"` or `"current"`.
fn quantity_name(quantity: Quantity) -> &'static str {
    if quantity.is_current() {
        "current"
    } else {
        "voltage"
    }
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
            result.set_item("source", sweep.source)?;
            result.set_item("quantity", quantity_name(sweep.quantity))?;
            result.set_item("levels", PyArray1::from_vec(py, sweep.levels))?;
            result.set_item("values", PyArray2::from_vec2(py, &sweep.values)?)?;
            add_terminals(&result, &sweep.terminals)?;
            let logic: Vec<Vec<f64>> = sweep.logic.iter().map(|levels| numbers(levels)).collect();
            result.set_item("logic", matrix(py, &logic)?)?;
        }
        Outcome::Ac(response) => {
            result.set_item("kind", "ac")?;
            result.set_item("source", response.source)?;
            result.set_item("frequencies", PyArray1::from_vec(py, response.frequencies))?;
            result.set_item("values", PyArray2::from_vec2(py, &response.values)?)?;
            result.set_item("currents", matrix(py, &response.currents)?)?;
            let operating_point = PyDict::new(py);
            add_operating_point(&operating_point, response.operating_point)?;
            result.set_item("operating_point", operating_point)?;
        }
        Outcome::Transient(waveforms) => {
            result.set_item("kind", "transient")?;
            result.set_item("times", PyArray1::from_vec(py, waveforms.times))?;
            result.set_item("values", PyArray2::from_vec2(py, &waveforms.values)?)?;
            add_terminals(&result, &waveforms.terminals)?;
            let traces = PyList::empty(py);
            for trace in waveforms.logic {
                let (times, levels): (Vec<f64>, Vec<Level>) = trace.into_iter().unzip();
                traces.append((
                    PyArray1::from_vec(py, times),
                    PyArray1::from_vec(py, numbers(&levels)),
                ))?;
            }
            result.set_item("logic", traces)?;
            result.set_item(
                "statistics",
                statistics_to_python(py, waveforms.statistics)?,
            )?;
        }
        Outcome::Sensitivity(sensitivities) => {
            result.set_item("kind", "sensitivity")?;
            result.set_item("outputs", sensitivities.outputs)?;
            let (parameters, values): (Vec<(String, String)>, Vec<f64>) = sensitivities
                .parameters
                .into_iter()
                .map(|parameter| ((parameter.device, parameter.name), parameter.value))
                .unzip();
            result.set_item("parameters", parameters)?;
            result.set_item("values", PyArray1::from_vec(py, values))?;
            result.set_item("derivatives", matrix(py, &sensitivities.derivatives)?)?;
            let operating_point = PyDict::new(py);
            add_operating_point(&operating_point, sensitivities.operating_point)?;
            result.set_item("operating_point", operating_point)?;
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
    result.set_item(
        "logic",
        PyArray1::from_vec(result.py(), numbers(&operating_point.logic)),
    )?;
    let Terminals { currents, charges } = operating_point.terminals;
    result.set_item("currents", PyArray1::from_vec(result.py(), currents))?;
    result.set_item("charges", PyArray1::from_vec(result.py(), charges))?;
    result.set_item("strategy", operating_point.strategy.name())
}

/// Adds the terminals' `currents` and `charges` at each point to `result`, one row per
/// point.
fn add_terminals(result: &Bound<'_, PyDict>, points: &[Terminals]) -> PyResult<()> {
    let currents: Vec<Vec<f64>> = points.iter().map(|point| point.currents.clone()).collect();
    let charges: Vec<Vec<f64>> = points.iter().map(|point| point.charges.clone()).collect();
    result.set_item("currents", matrix(result.py(), &currents)?)?;
    result.set_item("charges", matrix(result.py(), &charges)?)
}

/// Logic levels as numbers: 0, 1, and NaN for an unknown level.
fn numbers(levels: &[Level]) -> Vec<f64> {
    levels
        .iter()
        .map(|level| match level {
            Level::Low => 0.0,
            Level::High => 1.0,
            Level::Unknown => f64::NAN,
        })
        .collect()
}

/// `rows` as a two-dimensional array, with as many rows even when they are empty.
fn matrix<'py, T: Element + Copy>(
    py: Python<'py>,
    rows: &[Vec<T>],
) -> PyResult<Bound<'py, PyArray2<T>>> {
    let columns = rows.first().map_or(0, Vec::len);
    let flat: Vec<T> = rows.iter().flatten().copied().collect();
    PyArray1::from_vec(py, flat).reshape([rows.len(), columns])
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
/// The terminals of a device of `model` with `parameters`, a JSON object, in the order
/// the model declares them.
#[pyfunction]
fn terminals(model: &str, parameters: &str) -> PyResult<Vec<String>> {
    let library = Library::standard();
    let parameters = serde_json::from_str(parameters).map_err(|error| unreadable(&error))?;
    if let Some(logic) = library.logic(model) {
        let pins = logic
            .pins(&parameters)
            .map_err(|error| PyValueError::new_err(error.0))?;
        return Ok(pins.names());
    }
    let model = library
        .get(model)
        .ok_or_else(|| PyValueError::new_err(format!("no model is named `{model}`")))?;
    let interface = model
        .interface(&parameters)
        .map_err(|error| PyValueError::new_err(error.0))?;
    Ok(interface.terminals)
}

/// The value of `expression`, an expression in jolty's language that reads no variable.
#[pyfunction]
fn constant(expression: &str) -> PyResult<f64> {
    let parsed: jolty_expressions::Expression = expression
        .parse()
        .map_err(|error| PyValueError::new_err(format!("`{expression}`: {error}")))?;
    if !parsed.variables().is_empty() {
        return Err(PyValueError::new_err(format!(
            "`{expression}` reads a voltage, a current or time"
        )));
    }
    let mut workspace = jolty_expressions::Workspace::new(&parsed);
    Ok(parsed.evaluate(&[], &mut workspace, &mut []))
}

/// Every rule of the structural checks, in the order a check reports them: its name, its
/// default level and its condition.
#[pyfunction]
fn rules() -> Vec<(&'static str, &'static str, &'static str)> {
    jolty_checks::RULES
        .iter()
        .map(|rule| (rule.name, rule.level.name(), rule.condition))
        .collect()
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add_class::<Simulator>()?;
    module.add_function(wrap_pyfunction!(rules, module)?)?;
    module.add_function(wrap_pyfunction!(terminals, module)?)?;
    module.add_function(wrap_pyfunction!(constant, module)?)?;
    module.add("EXPRESSIONS", jolty_expressions::LANGUAGE)?;
    module.add("BOLTZMANN", jolty_models::BOLTZMANN)?;
    module.add("ELEMENTARY_CHARGE", jolty_models::ELEMENTARY_CHARGE)?;
    module.add("REFERENCE_TEMPERATURE", jolty_models::REFERENCE_TEMPERATURE)?;
    module.add("ElaborationError", py.get_type::<ElaborationError>())?;
    module.add("SimulationError", py.get_type::<SimulationError>())?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
