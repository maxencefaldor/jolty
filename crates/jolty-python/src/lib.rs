//! Python extension module `jolty._core`.
//!
//! The module exposes `Simulator` for elaborating JSON circuits, running structural checks,
//! and executing JSON-configured analyses returning `NumPy` arrays. Additional exports
//! include `Elaborated` (shared across simulators built at different `Values`), dedicated
//! `Pool` thread pools isolated from rayon's global pool, Monte Carlo `sample` draws, and
//! structural check `rules`. Every engine call releases the Python interpreter while
//! executing, and all higher-level formatting and unit handling lives in Python.

mod batch;
mod convert;
mod helpers;
mod measure;
mod variants;

use std::sync::{Arc, Mutex, MutexGuard};

use jolty_engine::{
    Analysis, Cotangents, Differentiated, Options, ParameterCotangents, Record, Sensitivity,
    SimulationError as EngineError,
};
use jolty_ir::Circuit;
use jolty_models::{Library, Model, ParameterError, ParameterKind};
use numpy::{Complex64, PyArray1, PyReadonlyArray2};
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::convert::{
    diagnostics_to_python, excursions_to_python, jvp_to_python, quantity_name, result_to_python,
};
use crate::helpers::{
    cotangent, elaboration_result, lock_unpoisoned, parse_json, simulation_result, value_result,
};

/// Global allocator using mimalloc per-thread heaps for concurrent batch allocations.
///
/// Benchmarks on eight threads showed the shared system allocator added about one-third
/// wall-time overhead per run, compared with roughly one-tenth for eight separate
/// processes.
#[global_allocator]
static ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

create_exception!(
    _core,
    ElaborationError,
    PyException,
    "A circuit cannot be elaborated; the single argument lists every problem as a dictionary \
     of `message` and `instance`."
);
create_exception!(_core, SimulationError, PyException, "An analysis failed.");

/// Dedicated rayon thread pool for engine execution, isolated from rayon's global pool.
///
/// Simulation outputs remain bit-identical regardless of thread count.
#[pyclass(module = "jolty._core", frozen)]
struct Pool {
    pub(crate) inner: rayon::ThreadPool,
}

#[pymethods]
impl Pool {
    /// Creates a dedicated thread pool with `threads` workers, or one per processor when
    /// `threads` is `None`.
    ///
    /// Raises `ValueError` when `threads` is zero or when worker threads cannot start.
    #[new]
    #[pyo3(signature = (threads = None))]
    fn new(threads: Option<usize>) -> PyResult<Self> {
        if threads == Some(0) {
            return Err(PyValueError::new_err(
                "a pool holds a thread or more, not 0",
            ));
        }
        let inner = value_result(
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads.unwrap_or(0))
                .build(),
        )?;
        Ok(Self { inner })
    }

    /// Returns the number of worker threads in the pool.
    #[getter]
    fn threads(&self) -> usize {
        self.inner.current_num_threads()
    }
}

impl Pool {
    fn run<T: Send>(
        &self,
        (options, spreads): (&Options, bool),
        work: impl FnOnce(&Options) -> T + Send,
    ) -> T {
        if spreads && self.inner.current_num_threads() >= 2 {
            return self.inner.install(|| work(options));
        }
        let mut alone = options.clone();
        alone.parallelism.max_threads = Some(1);
        work(&alone)
    }
}

/// Circuit elaborated once and shared across simulators instantiated at different parameter
/// values.
#[pyclass(module = "jolty._core", frozen)]
struct Elaborated {
    pub(crate) inner: Arc<jolty_engine::Elaborated>,
}

/// Parameter values and operating temperature applied to a simulator built from an
/// elaborated circuit.
#[pyclass(module = "jolty._core", frozen)]
struct Values {
    inner: jolty_engine::Values,
}

/// Structural digest of parameter overrides relative to the base elaborated circuit;
/// equal values share an identical matrix structure.
#[pyclass(module = "jolty._core", frozen, eq, hash)]
#[derive(PartialEq, Eq, Hash)]
struct StructureKey {
    inner: jolty_engine::StructureKey,
}

#[pymethods]
impl Elaborated {
    /// Elaborates the circuit from JSON `circuit` with model `substitutes` mapping component
    /// instance paths to `{model, parameters}`.
    ///
    /// Raises `ElaborationError` on invalid circuits, matching [`Simulator`].
    #[new]
    fn new(py: Python<'_>, circuit: &str, substitutes: &str) -> PyResult<Self> {
        let circuit = value_result(Circuit::from_json(circuit))?;
        let structure = jolty_engine::Structure {
            substitutes: parse_json!(substitutes)?,
        };
        let inner = elaboration_result(
            py,
            py.detach(|| jolty_engine::Elaborated::new(&circuit, &Library::standard(), &structure)),
        )?;
        Ok(Self { inner })
    }

    /// Builds [`Values`] from JSON `overrides` (nested by instance path and parameter path)
    /// at `temperature` in kelvins.
    ///
    /// Raises `ElaborationError` naming every unrecognized parameter path.
    fn values(&self, py: Python<'_>, overrides: &str, temperature: f64) -> PyResult<Values> {
        let overrides = parse_json!(overrides)?;
        let inner = elaboration_result(py, self.inner.values(&overrides, Some(temperature)))?;
        Ok(Values { inner })
    }

    /// Computes the [`StructureKey`] induced by `values` relative to the base elaborated
    /// circuit.
    ///
    /// Raises `ElaborationError` naming every device with out-of-range parameters.
    fn structure_of(&self, py: Python<'_>, values: &Values) -> PyResult<StructureKey> {
        let inner = elaboration_result(py, py.detach(|| self.inner.structure_of(&values.inner)))?;
        Ok(StructureKey { inner })
    }

    /// Re-elaborates the circuit for the structural changes induced by `values`, preserving
    /// all unchanged parameters.
    ///
    /// Raises `ElaborationError` under the same conditions as [`Elaborated::structure_of`].
    fn restructured(&self, py: Python<'_>, values: &Values) -> PyResult<Self> {
        let inner = elaboration_result(py, py.detach(|| self.inner.restructured(&values.inner)))?;
        Ok(Self { inner })
    }
}

#[pymethods]
impl StructureKey {
    /// Returns whether the parameter values preserve the base circuit structure.
    #[getter]
    fn unchanged(&self) -> bool {
        self.inner.is_empty()
    }
}

/// Circuit elaborated for simulation.
#[pyclass(module = "jolty._core", frozen)]
struct Simulator {
    inner: Mutex<jolty_engine::Simulator>,
    unknowns: Vec<(String, &'static str)>,
    nets: Vec<String>,
    devices: usize,
    foundry_devices: Vec<String>,
}

/// Reverse-mode pullback from a VJP analysis, mapping output cotangents back to parameters.
#[pyclass(module = "jolty._core", frozen)]
struct Pullback {
    inner: jolty_engine::Pullback,
}

type RecordedDevice = (
    String,
    Vec<String>,
    Vec<bool>,
    usize,
    usize,
    Vec<&'static str>,
    Vec<&'static str>,
    Vec<&'static str>,
);

type Recording = (Vec<RecordedDevice>, Vec<(String, bool)>);

impl Simulator {
    /// Wraps an engine simulator and caches metadata read once from Python.
    pub(crate) fn wrapping(inner: jolty_engine::Simulator) -> Self {
        let unknowns = inner
            .unknowns()
            .iter()
            .map(|unknown| (unknown.name.clone(), quantity_name(unknown.nature)))
            .collect();
        let nets = inner
            .logic_nets()
            .iter()
            .map(|net| net.name.clone())
            .collect();
        let devices = inner.devices().len() - inner.elements() + inner.logic_devices().len();
        let foundry_devices = inner
            .foundry_devices()
            .into_iter()
            .map(str::to_owned)
            .collect();
        Self {
            inner: Mutex::new(inner),
            unknowns,
            nets,
            devices,
            foundry_devices,
        }
    }
}

#[pymethods]
impl Simulator {
    /// Elaborates JSON `circuit` at `temperature` in kelvins, applying JSON `overrides`
    /// by instance path and JSON `substitutes` by component instance path.
    ///
    /// Raises `ElaborationError` with a list of problem dictionaries (`message` and
    /// `instance`, or `None`) when elaboration fails.
    #[new]
    fn new(
        py: Python<'_>,
        circuit: &str,
        temperature: f64,
        overrides: &str,
        substitutes: &str,
    ) -> PyResult<Self> {
        let circuit = value_result(Circuit::from_json(circuit))?;
        let conditions = jolty_engine::Conditions {
            temperature,
            overrides: parse_json!(overrides)?,
            substitutes: parse_json!(substitutes)?,
        };
        let inner = elaboration_result(
            py,
            py.detach(|| jolty_engine::Simulator::new(&circuit, &Library::standard(), &conditions)),
        )?;
        Ok(Self::wrapping(inner))
    }

    /// Creates a simulator from `elaborated` at `values`, restructuring the circuit when
    /// parameter values change device topology.
    ///
    /// Raises `ElaborationError` naming every device with out-of-range parameters.
    #[staticmethod]
    fn from_elaborated(py: Python<'_>, elaborated: &Elaborated, values: &Values) -> PyResult<Self> {
        let shared = Arc::clone(&elaborated.inner);
        let inner = elaboration_result(
            py,
            py.detach(|| jolty_engine::Simulator::from_elaborated(shared, values.inner.clone())),
        )?;
        Ok(Self::wrapping(inner))
    }

    /// Returns the shared elaborated circuit backing this simulator.
    #[getter]
    fn elaborated(&self) -> Elaborated {
        Elaborated {
            inner: Arc::clone(self.lock().elaborated()),
        }
    }

    /// Executes the half-open range `runs` over `variants` on `pool`, releasing the Python
    /// interpreter while running.
    ///
    /// Each run applies variant overrides on top of the simulator's base values as a pure
    /// function of run index. Every run executes `analyses` (pairs of analysis and options
    /// JSON), evaluates `measures` (`(analysis_index, measure)`), and records `traces`
    /// (`(analysis_index, unknown_indices)`), linearly interpolating transient waveforms.
    ///
    /// Returns a dictionary of `NumPy` arrays indexed from row 0 at `runs.0`:
    /// - `values`: varied parameter values per run.
    /// - `measured`: measure results (`NaN` on failure).
    /// - `causes` and `reasons`: `-1` on success, otherwise index into `reasons`
    ///   (`("failed", [], "")` or `(tag, numbers, message)`).
    /// - `ended`: boolean completion matrix per run and analysis.
    /// - `traces`: list of `(runs, points, unknowns)` arrays.
    /// - `problems` and `errors`: per-row elaboration problems and analysis failure strings.
    ///
    /// When `each` is provided, the callback runs under the interpreter lock as runs finish
    /// (in completion order) with `(run_index, restructured_simulator, problems, results)`.
    /// The first callback exception propagates after all runs finish. Raises `ValueError`
    /// for invalid JSON inputs or out-of-range run bounds.
    #[pyo3(signature = (analyses, variants, runs, measures, traces, pool, each = None))]
    #[expect(
        clippy::too_many_arguments,
        reason = "a batch is its runs, its analyses, its measures, what it traces, its \
                  threads and the function that reads each run, as Python names them"
    )]
    fn batch<'py>(
        &self,
        py: Python<'py>,
        analyses: Vec<(String, String)>,
        variants: &variants::Variants,
        runs: (u64, u64),
        measures: Vec<(usize, Py<measure::Measure>)>,
        traces: Vec<(usize, Vec<usize>)>,
        pool: &Pool,
        each: Option<Py<PyAny>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let (elaborated, base) = self.snapshot();
        let request = batch::Request {
            analyses,
            runs: runs.0..runs.1,
            measures,
            traces,
            each,
        };
        batch::run(py, (&elaborated, &base), variants, request, pool)
    }

    /// Computes batch reverse-mode VJPs of JSON `analysis` and `options` with respect to
    /// `parameters` across `runs` of `variants` on `pool`.
    ///
    /// Each entry in `parameters` is `(instance_path, parameter_name)`, and `unknowns`
    /// lists traced unknown indices. Work executes with the interpreter released and returns
    /// `(result_dict, batch_pullback)`, where `result_dict` holds `traces`
    /// (`runs x points x unknowns`), oscillator `periods`, `problems`, and `errors`.
    ///
    /// Raises `ValueError` for invalid JSON inputs or out-of-range run bounds.
    #[expect(
        clippy::too_many_arguments,
        reason = "a batch's VJP is its analysis, its options, its parameters, its runs, what \
                  it traces and its threads, as Python names them"
    )]
    fn batch_vjp<'py>(
        &self,
        py: Python<'py>,
        analysis: String,
        options: String,
        parameters: Vec<(String, String)>,
        variants: &variants::Variants,
        runs: (u64, u64),
        unknowns: Vec<usize>,
        pool: &Pool,
    ) -> PyResult<(Bound<'py, PyDict>, batch::BatchPullback)> {
        let (elaborated, base) = self.snapshot();
        let request = batch::Differentiating {
            analysis,
            options,
            parameters,
            runs: runs.0..runs.1,
            unknowns,
        };
        batch::vjp(py, (&elaborated, &base), variants, request, pool)
    }

    /// Returns `(name, quantity)` for each unknown in solution vector order (`"voltage"`,
    /// `"current"`, `"temperature"`, or `"state"`).
    #[getter]
    fn unknowns(&self) -> Vec<(String, &'static str)> {
        self.unknowns.clone()
    }

    /// Returns logic net names in result logic-level order.
    #[getter]
    fn nets(&self) -> Vec<String> {
        self.nets.clone()
    }

    /// Returns all recordable instance paths in order: device paths (such as `X1/R1`)
    /// followed by component instances and logic devices with summed terminals (such as
    /// `X1`).
    #[getter]
    fn instances(&self) -> Vec<String> {
        let simulator = self.lock();
        let mut paths: Vec<String> = simulator
            .devices()
            .iter()
            .map(|device| device.path().to_owned())
            .collect();
        for summed_terminal in simulator.summed_terminals() {
            if paths
                .last()
                .is_none_or(|last| last != summed_terminal.owner())
            {
                paths.push(summed_terminal.owner().to_owned());
            }
        }
        paths
    }

    /// Resolves the device and summed-terminal layout recorded by JSON `record` and the
    /// `regions` flag, matching result array order.
    ///
    /// Returns `(devices, summed_terminals)`. Each full device tuple gives path, terminal
    /// names, thermal flags, first internal unknown index, internal unknown count, variable
    /// names, region names, and margin names.
    ///
    /// Transistors recorded for regions only follow with empty terminals, zero internal
    /// unknowns, and region variables only. Each summed terminal entry gives `(name,
    /// is_thermal)` for a component instance or logic device pin (such as `X1.vdd`).
    fn recording(&self, record: &str, regions: bool) -> PyResult<Recording> {
        let record: Record = parse_json!(record)?;
        let simulator = self.lock();
        let recorded = simulator.recorded(&record, regions);
        let in_full = recorded.devices.iter().map(|&index| {
            let device = &simulator.devices()[index];
            let thermal = (0..device.terminals().len())
                .map(|terminal| device.thermal().contains(&terminal))
                .collect();
            let internals = device.internals();
            let regions = device.regions();
            (
                device.path().to_owned(),
                device.terminals().to_vec(),
                thermal,
                internals.start,
                internals.len(),
                device.variable_names().to_vec(),
                regions.names.to_vec(),
                regions.margins.to_vec(),
            )
        });
        let regions_alone = recorded.regions.iter().map(|&index| {
            let device = &simulator.devices()[index];
            let names = device.variable_names();
            let regions = device.regions();
            (
                device.path().to_owned(),
                Vec::new(),
                Vec::new(),
                0,
                0,
                simulator
                    .region_variables(index)
                    .into_iter()
                    .map(|variable| names[variable])
                    .collect(),
                regions.names.to_vec(),
                regions.margins.to_vec(),
            )
        });
        let devices = in_full.chain(regions_alone).collect();
        let summed_terminals = recorded
            .summed_terminals
            .iter()
            .map(|&index| {
                let summed_terminal = &simulator.summed_terminals()[index];
                (summed_terminal.name().to_owned(), summed_terminal.thermal())
            })
            .collect();
        Ok((devices, summed_terminals))
    }

    /// Resolves `terminal` of `instance` to `(unknown_name_or_none, is_thermal)`, where
    /// `None` denotes ground.
    ///
    /// Returns `None` when the terminal does not exist on an electrical or thermal net.
    fn terminal(&self, instance: &str, terminal: &str) -> Option<(Option<String>, bool)> {
        let simulator = self.lock();
        let (node, thermal) = simulator.terminal(instance, terminal)?;
        let name = node.map(|unknown| simulator.unknowns()[unknown.index()].name.clone());
        Some((name, thermal))
    }

    /// Returns paths of foundry devices delegated to ngspice, in placement order.
    #[getter]
    fn foundry_devices(&self) -> Vec<String> {
        self.foundry_devices.clone()
    }

    /// Returns the total count of unfolded analog and logic devices.
    #[getter]
    fn devices(&self) -> usize {
        self.devices
    }

    /// Runs structural checks and returns a list of diagnostic dictionaries (`rule`,
    /// `message`, `instances`, and `nets`).
    ///
    /// Each instance dictionary contains `path` and `source_location` (or `None`).
    fn check<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let found = py.detach(|| jolty_checks::check(&self.lock()));
        diagnostics_to_python(py, found)
    }

    /// Runs operating-point linearized checks under JSON `options` on `pool`, returning
    /// diagnostic dictionaries matching `check`.
    ///
    /// Returns an empty list when the circuit exceeds the linearization size limit.
    /// Execution releases the interpreter inside `pool` when parallelism applies, and
    /// raises `SimulationError` if no operating point is found.
    fn check_operating_point<'py>(
        &self,
        py: Python<'py>,
        options: &str,
        pool: &Pool,
    ) -> PyResult<Bound<'py, PyList>> {
        let options: Options = parse_json!(options)?;
        let found = self.simulate(
            py,
            pool,
            &options,
            |simulator| simulator.spreads(&Analysis::OperatingPoint, &options, pool.threads()),
            jolty_checks::check_operating_point,
        )?;
        diagnostics_to_python(py, found)
    }

    /// Returns domain excursions recorded for logic devices during the last analysis.
    ///
    /// Each dictionary contains `instance`, `bound` (such as `"load"`), `minimum`,
    /// `maximum`, extreme `value`, first excursion `time`, and `count`.
    fn excursions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        excursions_to_python(py, &self.lock())
    }

    /// Computes forward-mode directional derivatives (JVP) of JSON `analysis` under JSON
    /// `options` along `tangents` with respect to `parameters`.
    ///
    /// Each parameter is `(instance_path, parameter_name)` and each tangent vector supplies
    /// one value per parameter. Returns a dictionary with `result`, `tangents` (indexed by
    /// point, output, and tangent), and optional oscillator `period` derivatives.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 extracts owned Vec arguments from Python sequences"
    )]
    fn jvp<'py>(
        &self,
        py: Python<'py>,
        analysis: &str,
        parameters: Vec<(String, String)>,
        tangents: Vec<Vec<f64>>,
        options: &str,
        pool: &Pool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let (analysis, options) = differentiated(analysis, options)?;
        let found = self.simulate(
            py,
            pool,
            &options,
            |simulator| spreads_differentiating(simulator, &analysis, &options, pool),
            |simulator, options| simulator.jvp(&analysis, &parameters, &tangents, options),
        )?;
        jvp_to_python(py, found)
    }

    /// Computes reverse-mode VJP state for JSON `analysis` under JSON `options` with
    /// respect to `parameters`, returning `(result, pullback)`.
    ///
    /// Each entry in `parameters` is `(instance_path, parameter_name)`.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 extracts owned Vec arguments from Python sequences"
    )]
    fn vjp<'py>(
        &self,
        py: Python<'py>,
        analysis: &str,
        parameters: Vec<(String, String)>,
        options: &str,
        pool: &Pool,
    ) -> PyResult<(Bound<'py, PyDict>, Pullback)> {
        let (analysis, options) = differentiated(analysis, options)?;
        let inner = self.simulate(
            py,
            pool,
            &options,
            |simulator| spreads_differentiating(simulator, &analysis, &options, pool),
            |simulator, options| simulator.vjp(&analysis, &parameters, options),
        )?;
        let result = result_to_python(py, inner.result.clone())?;
        Ok((result, Pullback { inner }))
    }

    /// Pulls output cotangents `values` (indexed by point and output, real or complex) and
    /// oscillator `period` cotangent back to parameters via `pullback` under JSON `options`.
    fn pull_back<'py>(
        &self,
        py: Python<'py>,
        pullback: &Pullback,
        values: &Bound<'py, PyAny>,
        period: f64,
        options: &str,
        pool: &Pool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let options: Options = parse_json!(options)?;
        let cotangents = if let Ok(real) = values.extract::<PyReadonlyArray2<'py, f64>>() {
            Cotangents::Real(vec![cotangent(&real, period)])
        } else {
            let phasors = values.extract::<PyReadonlyArray2<'py, Complex64>>()?;
            Cotangents::Complex(vec![cotangent(&phasors, period)])
        };
        let found = self.simulate(
            py,
            pool,
            &options,
            |_| true,
            |simulator, options| simulator.pull_back(&pullback.inner, &cotangents, options),
        )?;
        match found {
            ParameterCotangents::Real(mut rows) => {
                Ok(PyArray1::from_vec(py, rows.pop().unwrap_or_default()).into_any())
            }
            ParameterCotangents::Complex(mut rows) => {
                Ok(PyArray1::from_vec(py, rows.pop().unwrap_or_default()).into_any())
            }
        }
    }

    /// Runs JSON `analysis` under JSON `options`, releasing the interpreter inside `pool`
    /// when options permit parallel execution.
    fn run<'py>(
        &self,
        py: Python<'py>,
        analysis: &str,
        options: &str,
        pool: &Pool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let analysis: Analysis = parse_json!(analysis)?;
        let options: Options = parse_json!(options)?;
        let result = self.simulate(
            py,
            pool,
            &options,
            |simulator| simulator.spreads(&analysis, &options, pool.threads()),
            |simulator, options| simulator.run(&analysis, options),
        )?;
        result_to_python(py, result)
    }
}

impl Simulator {
    fn snapshot(&self) -> (Arc<jolty_engine::Elaborated>, jolty_engine::Values) {
        let simulator = self.lock();
        (
            Arc::clone(simulator.elaborated()),
            simulator.values().clone(),
        )
    }

    fn simulate<T, S, W>(
        &self,
        py: Python<'_>,
        pool: &Pool,
        options: &Options,
        spreads: S,
        work: W,
    ) -> PyResult<T>
    where
        T: Send,
        S: FnOnce(&jolty_engine::Simulator) -> bool + Send,
        W: FnOnce(&mut jolty_engine::Simulator, &Options) -> Result<T, EngineError> + Send,
    {
        simulation_result(py.detach(|| {
            let spreads = spreads(&self.lock());
            pool.run((options, spreads), |options| {
                let mut simulator = self.lock();
                work(&mut simulator, options)
            })
        }))
    }

    fn lock(&self) -> MutexGuard<'_, jolty_engine::Simulator> {
        lock_unpoisoned(&self.inner)
    }
}

fn spreads_differentiating(
    simulator: &jolty_engine::Simulator,
    analysis: &Differentiated,
    options: &Options,
    pool: &Pool,
) -> bool {
    let sensitivity = Analysis::Sensitivity(Sensitivity {
        outputs: Vec::new(),
        of: analysis.clone(),
    });
    simulator.spreads(&sensitivity, options, pool.threads())
}

fn differentiated(analysis: &str, options: &str) -> PyResult<(Differentiated, Options)> {
    Ok((parse_json!(analysis)?, parse_json!(options)?))
}

/// Returns the terminal names of `model` configured with JSON `parameters`, in declaration
/// order.
#[pyfunction]
fn terminals(model: &str, parameters: &str) -> PyResult<Vec<String>> {
    let library = Library::standard();
    let parameters = parse_json!(parameters)?;
    let invalid = |error: ParameterError| PyValueError::new_err(error.0);
    if let Some(logic) = library.logic(model) {
        let device = logic.instantiate(&parameters).map_err(invalid)?;
        return Ok(device.pins().names());
    }
    let specification = analog(&library, model)?
        .read(&parameters)
        .map_err(invalid)?;
    Ok(specification.interface().terminals)
}

fn analog<'a>(library: &'a Library, name: &str) -> PyResult<&'a Arc<dyn Model>> {
    library
        .get(name)
        .ok_or_else(|| PyValueError::new_err(format!("no model is named `{name}`")))
}

/// Returns the engine's SPICE-compatible default options as JSON, serving as the single
/// source of defaults for `jolty.plan.Options`.
#[pyfunction]
fn default_options() -> PyResult<String> {
    value_result(serde_json::to_string(&Options::default()))
}

/// Returns the measured parallelism break-even thresholds by thread count as JSON, using
/// `null` where no measured size benefited from threading.
#[pyfunction]
fn break_evens() -> PyResult<String> {
    value_result(serde_json::to_string(&jolty_engine::BREAK_EVENS))
}

/// Returns all analog and logic model names in alphabetical order.
#[pyfunction]
fn models() -> Vec<&'static str> {
    Library::standard().names().collect()
}

/// Returns the inlined JSON Schema describing all parameters of `model`, including types,
/// defaults, and field documentation.
#[pyfunction]
fn schema(model: &str) -> PyResult<String> {
    let library = Library::standard();
    let schema = match library.logic(model) {
        Some(logic) => logic.schema(),
        None => analog(&library, model)?.schema(),
    };
    Ok(schema.as_value().to_string())
}

/// Returns the physical dimension or kind of each parameter of `model` in SI base units.
///
/// Entries map parameter names to a 5-tuple of doubled SI exponents `(kg, m, s, A, K)`,
/// `None` for dimensionless numbers, `"choice"` for discrete options, or a tag string for
/// conditional dimensions. Raises `ValueError` when `model` is unknown.
#[pyfunction]
fn parameter_dimensions<'py>(py: Python<'py>, model: &str) -> PyResult<Bound<'py, PyDict>> {
    let library = Library::standard();
    let kinds = if library.logic(model).is_some() {
        &[]
    } else {
        analog(&library, model)?.parameter_kinds()
    };
    let result = PyDict::new(py);
    for &(name, kind) in kinds {
        match kind {
            ParameterKind::Quantity(dimension) => {
                let [kilogram, meter, second, ampere, kelvin] = dimension.exponents();
                result.set_item(name, (kilogram, meter, second, ampere, kelvin))?;
            }
            ParameterKind::Number => result.set_item(name, None::<&str>)?,
            ParameterKind::Choice => result.set_item(name, "choice")?,
            ParameterKind::Conditional(tag) => result.set_item(name, tag)?,
        }
    }
    Ok(result)
}

/// Returns the base names of parameters of `model` that support length, width, and area
/// binning.
///
/// Raises `ValueError` when `model` is unknown.
#[pyfunction]
fn binned_parameters(model: &str) -> PyResult<Vec<&'static str>> {
    let library = Library::standard();
    let binned = if library.logic(model).is_some() {
        &[]
    } else {
        analog(&library, model)?.binned_parameters()
    };
    Ok(binned.to_vec())
}

/// Evaluates a constant `expression` in jolty's expression language with no circuit
/// variables.
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

/// Parses `expression` and serializes its [`jolty_expressions::Term`] AST to JSON.
///
/// Node forms are numbers, `"time"`, `{"voltage": [a, b]}` (`b` null for ground),
/// `{"current": p}`, `{"operation": name, "operands": [...]}`, and
/// `{"table": input, "abscissae": [...], "ordinates": [...], "rounding": r}`.
#[pyfunction]
fn expression_tree(expression: &str) -> PyResult<String> {
    let parsed: jolty_expressions::Expression = expression
        .parse()
        .map_err(|error| PyValueError::new_err(format!("`{expression}`: {error}")))?;
    value_result(term_to_json_string(&parsed.tree()))
}

/// Evaluates a JSON source `waveform` at `time` in seconds (`time = 0` gives the DC value).
#[pyfunction]
fn waveform_value(waveform: &str, time: f64) -> PyResult<f64> {
    let waveform: jolty_models::waveform::Waveform = parse_json!(waveform)?;
    Ok(waveform.value(time))
}

/// Returns JSON `parameters` for `model` with all omitted fields filled with engine
/// defaults so exported SPICE netlists match engine behavior.
#[pyfunction]
fn defaulted(model: &str, parameters: &str) -> PyResult<String> {
    let library = Library::standard();
    let parameters = parse_json!(parameters)?;
    let defaulted = match library.logic(model) {
        Some(logic) => logic.defaulted(&parameters),
        None => analog(&library, model)?.defaulted(&parameters),
    };
    let defaulted = defaulted
        .map_err(|error| PyValueError::new_err(format!("a {model}'s parameters: {}", error.0)))?;
    Ok(serde_json::Value::Object(defaulted).to_string())
}

/// Serializes a term of an expression's tree to JSON iteratively, as [`expression_tree`]
/// writes it, avoiding call-stack recursion on deep left-leaning binary chains.
fn term_to_json_string(root: &jolty_expressions::Term) -> serde_json::Result<String> {
    use jolty_expressions::{Term, Variable};

    enum Segment<'a> {
        Term(&'a Term),
        Literal(&'static str),
        OperationTail(&'static str),
    }

    let mut output = String::new();
    let mut stack = vec![Segment::Term(root)];
    while let Some(segment) = stack.pop() {
        match segment {
            Segment::Literal(text) => output.push_str(text),
            Segment::OperationTail(name) => {
                output.push_str("],\"operation\":");
                output.push_str(&serde_json::to_string(name)?);
                output.push('}');
            }
            Segment::Term(Term::Constant(value)) => {
                output.push_str(&serde_json::to_string(value)?);
            }
            Segment::Term(Term::Variable(Variable::Time)) => output.push_str("\"time\""),
            Segment::Term(Term::Variable(Variable::Voltage(positive, negative))) => {
                output.push_str("{\"voltage\":");
                output.push_str(&serde_json::to_string(&(positive, negative))?);
                output.push('}');
            }
            Segment::Term(Term::Variable(Variable::Current(port))) => {
                output.push_str("{\"current\":");
                output.push_str(&serde_json::to_string(port)?);
                output.push('}');
            }
            Segment::Term(Term::Operation(name, operands)) => {
                output.push_str("{\"operands\":[");
                stack.push(Segment::OperationTail(name));
                for (index, operand) in operands.iter().enumerate().rev() {
                    stack.push(Segment::Term(operand));
                    if index > 0 {
                        stack.push(Segment::Literal(","));
                    }
                }
            }
            Segment::Term(Term::Table {
                input,
                abscissae,
                ordinates,
                rounding,
            }) => {
                output.push_str("{\"abscissae\":");
                output.push_str(&serde_json::to_string(abscissae)?);
                output.push_str(",\"ordinates\":");
                output.push_str(&serde_json::to_string(ordinates)?);
                output.push_str(",\"rounding\":");
                output.push_str(&serde_json::to_string(rounding)?);
                output.push_str(",\"table\":");
                stack.push(Segment::Literal("}"));
                stack.push(Segment::Term(input));
            }
        }
    }
    Ok(output)
}

/// Returns all structural check rules in reporting order as `(name, default_level,
/// condition)` tuples.
#[pyfunction]
fn rules() -> Vec<(&'static str, &'static str, &'static str)> {
    jolty_checks::RULES
        .iter()
        .map(|rule| (rule.name, rule.level.name(), rule.condition))
        .collect()
}

/// Returns the decade mantissas in `[1, 10)` for the IEC E-series `series` (such as
/// `"E24"`).
#[pyfunction]
fn mantissas(series: &str) -> PyResult<Vec<f64>> {
    jolty_checks::preferred::Series::named(series)
        .map(jolty_checks::preferred::Series::mantissas)
        .ok_or_else(|| PyValueError::new_err(format!("no E series is named `{series}`")))
}

/// Draws Monte Carlo parameter values for sample `index` from `seed` for each entry in
/// JSON `toleranced`.
///
/// `toleranced` is a JSON list of objects with `instance`, `parameter`, `nominal`, and
/// `tolerance` fields; each value depends only on `seed`, `index`, and its parameter path,
/// matching batch sampling. Raises `ValueError` for invalid JSON input.
#[pyfunction]
fn sample(toleranced: &str, seed: u64, index: u64) -> PyResult<Vec<f64>> {
    let toleranced: Vec<jolty_engine::variants::Toleranced> = parse_json!(toleranced)?;
    Ok(toleranced
        .iter()
        .map(|parameter| parameter.drawn(seed, index))
        .collect())
}

/// Draws a standard `variate` (`"normal"` or `"uniform"`) for sample `index` from `seed`
/// at `call` within `scope`, matching engine tolerance sampling.
///
/// Raises `ValueError` for an unrecognized variate name.
#[pyfunction]
fn variate(variate: &str, seed: u64, index: u64, scope: &str, call: &str) -> PyResult<f64> {
    let kind = match variate {
        "normal" => jolty_engine::variants::Variate::Normal,
        "uniform" => jolty_engine::variants::Variate::Uniform,
        _ => {
            return Err(PyValueError::new_err(format!(
                "no variate is named `{variate}`: `normal` or `uniform`"
            )));
        }
    };
    Ok(jolty_engine::variants::variate(
        kind, seed, index, scope, call,
    ))
}

/// Lists all toleranced device parameters in JSON `circuit` in hierarchical order as
/// `(instance_path, parameter_path, nominal, tolerance_json)` tuples.
///
/// Raises `ValueError` when a tolerance targets a non-numeric parameter or when component
/// definitions form a cycle.
#[pyfunction]
fn toleranced(circuit: &str) -> PyResult<Vec<(String, String, f64, String)>> {
    let circuit = value_result(Circuit::from_json(circuit))?;
    let found = value_result(jolty_engine::variants::toleranced(&circuit))?;
    found
        .into_iter()
        .map(|parameter| {
            let tolerance = value_result(serde_json::to_string(&parameter.tolerance))?;
            Ok((
                parameter.instance,
                parameter.parameter,
                parameter.nominal,
                tolerance,
            ))
        })
        .collect()
}

#[pymodule(name = "_core", gil_used = false)]
fn extension(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add_class::<Simulator>()?;
    module.add_class::<Pullback>()?;
    module.add_class::<Pool>()?;
    module.add_function(wrap_pyfunction!(sample, module)?)?;
    module.add_function(wrap_pyfunction!(toleranced, module)?)?;
    module.add_function(wrap_pyfunction!(variate, module)?)?;
    measure::register(module)?;
    module.add_class::<Elaborated>()?;
    module.add_class::<Values>()?;
    module.add_class::<StructureKey>()?;
    module.add_class::<variants::Variants>()?;
    module.add_class::<batch::BatchPullback>()?;
    module.add_function(wrap_pyfunction!(rules, module)?)?;
    module.add_function(wrap_pyfunction!(terminals, module)?)?;
    module.add_function(wrap_pyfunction!(constant, module)?)?;
    module.add_function(wrap_pyfunction!(expression_tree, module)?)?;
    module.add_function(wrap_pyfunction!(defaulted, module)?)?;
    module.add_function(wrap_pyfunction!(default_options, module)?)?;
    module.add_function(wrap_pyfunction!(break_evens, module)?)?;
    module.add_function(wrap_pyfunction!(waveform_value, module)?)?;
    module.add_function(wrap_pyfunction!(models, module)?)?;
    module.add_function(wrap_pyfunction!(schema, module)?)?;
    module.add_function(wrap_pyfunction!(parameter_dimensions, module)?)?;
    module.add_function(wrap_pyfunction!(binned_parameters, module)?)?;
    module.add_function(wrap_pyfunction!(mantissas, module)?)?;
    module.add("EXPRESSIONS", jolty_expressions::LANGUAGE)?;
    module.add("BOLTZMANN", jolty_models::BOLTZMANN)?;
    module.add("ELEMENTARY_CHARGE", jolty_models::ELEMENTARY_CHARGE)?;
    module.add("REFERENCE_TEMPERATURE", jolty_models::REFERENCE_TEMPERATURE)?;
    module.add("REGION", jolty_models::REGION)?;
    module.add("LOOP_GAIN", jolty_engine::LOOP_GAIN)?;
    module.add("ElaborationError", py.get_type::<ElaborationError>())?;
    module.add("SimulationError", py.get_type::<SimulationError>())?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use jolty_expressions::{Term, Variable};

    use super::term_to_json_string;

    #[test]
    fn term_to_json_string_serializes_all_variants() {
        let constant = Term::Constant(1.5);
        assert_eq!(term_to_json_string(&constant).unwrap(), "1.5");

        let time = Term::Variable(Variable::Time);
        assert_eq!(term_to_json_string(&time).unwrap(), "\"time\"");

        let single_voltage = Term::Variable(Variable::Voltage("in".to_owned(), None));
        assert_eq!(
            term_to_json_string(&single_voltage).unwrap(),
            r#"{"voltage":["in",null]}"#
        );

        let diff_voltage = Term::Variable(Variable::Voltage("a".to_owned(), Some("b".to_owned())));
        assert_eq!(
            term_to_json_string(&diff_voltage).unwrap(),
            r#"{"voltage":["a","b"]}"#
        );

        let current = Term::Variable(Variable::Current("V1".to_owned()));
        assert_eq!(
            term_to_json_string(&current).unwrap(),
            r#"{"current":"V1"}"#
        );

        let sum = Term::Operation("+", vec![constant, time]);
        assert_eq!(
            term_to_json_string(&sum).unwrap(),
            r#"{"operands":[1.5,"time"],"operation":"+"}"#
        );

        let table = Term::Table {
            input: Box::new(Term::Variable(Variable::Voltage("n".to_owned(), None))),
            abscissae: vec![0.0, 1.0, 2.0],
            ordinates: vec![0.0, 0.5, 1.0],
            rounding: 0.1,
        };
        let expected = concat!(
            r#"{"abscissae":[0.0,1.0,2.0],"ordinates":[0.0,0.5,1.0],"#,
            r#""rounding":0.1,"table":{"voltage":["n",null]}}"#,
        );
        assert_eq!(term_to_json_string(&table).unwrap(), expected);
    }
}
