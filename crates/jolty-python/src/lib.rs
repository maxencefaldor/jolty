//! The Python extension module `jolty._core`.
//!
//! It exposes a class, `Simulator`, that elaborates a circuit from its JSON text, checks
//! its structure, and runs analyses described in JSON, returning arrays as `NumPy` arrays;
//! a circuit elaborated once, `Elaborated`, that simulators made at the `Values` of
//! some parameters share; the `Pool` of threads the engine runs on, never rayon's global
//! one; the draws of Monte Carlo samples, `sample`; and the table of the structural
//! checks' `rules`. Every call that runs the engine releases the interpreter while it
//! runs. Everything an agent or a person reads, from units to reports, is built on top of
//! it in Python.

mod batch;
mod measure;

use std::sync::{Arc, Mutex};

use jolty_engine::{
    Analysis, AnalysisResult, DcSweepResult, Derivatives, Location, NoiseResult,
    OperatingPointResult, Options, PoleZeroResult, Record, RecordedSeries, RecordedValues,
    Statistics, TransientResult,
};
use jolty_ir::Circuit;
use jolty_models::{Library, LogicLevel, Model, Nature, ParameterError};
use numpy::{Element, PyArray1, PyArray2, PyArray3, PyArrayMethods};
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// Every allocation of the engine's runs, which many threads make at once in a batch,
/// goes to mimalloc and its heaps of each thread: the system's allocator, shared by the
/// threads, made each run of a batch on eight threads take about a third more time than
/// on one, where eight processes each running one take a tenth more.
#[global_allocator]
static ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

create_exception!(
    _core,
    ElaborationError,
    PyException,
    "A circuit cannot be elaborated. Its only argument lists every problem as a string."
);
create_exception!(_core, SimulationError, PyException, "An analysis failed.");

/// The threads the engine runs on: a rayon pool of its own, which every call that runs
/// the engine runs inside, never rayon's global pool. No result depends on its threads.
#[pyclass(module = "jolty._core", frozen)]
struct Pool {
    pub(crate) inner: rayon::ThreadPool,
}

#[pymethods]
impl Pool {
    /// A pool of `threads` threads, or of one per processor with `None`.
    ///
    /// Raises `ValueError` for no thread, or when the threads cannot be started.
    #[new]
    #[pyo3(signature = (threads = None))]
    fn new(threads: Option<usize>) -> PyResult<Self> {
        if threads == Some(0) {
            return Err(PyValueError::new_err(
                "a pool holds a thread or more, not 0",
            ));
        }
        let inner = rayon::ThreadPoolBuilder::new()
            .num_threads(threads.unwrap_or(0))
            .build()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self { inner })
    }

    /// The number of threads.
    #[getter]
    fn threads(&self) -> usize {
        self.inner.current_num_threads()
    }
}

impl Pool {
    /// `work`, a run under `options` that `spreads` its work over threads or not, as
    /// `jolty_engine::Simulator::spreads` says: inside the pool, on its threads, when it
    /// spreads and the pool has two or more, and else on the calling thread, which a run
    /// that spreads nothing never leaves, so that it waits on no thread of the pool, under
    /// `options` capped at that one thread, as its parallelism's defaults then read.
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

/// A circuit elaborated once, which simulators of it at the values of any parameters share.
#[pyclass(module = "jolty._core", frozen)]
struct Elaborated {
    pub(crate) inner: Arc<jolty_engine::Elaborated>,
}

/// The values of some parameters of an elaborated circuit, and the temperature, that a
/// simulator of it runs at.
#[pyclass(module = "jolty._core", frozen)]
struct Values {
    inner: jolty_engine::Values,
}

/// The structure the values of some parameters give an elaborated circuit, against its
/// own: equal for values that share a structure.
#[pyclass(module = "jolty._core", frozen, eq, hash)]
#[derive(PartialEq, Eq, Hash)]
struct StructureKey {
    inner: jolty_engine::StructureKey,
}

#[pymethods]
impl Elaborated {
    /// Elaborates the circuit whose JSON text is `circuit`, with the models of the JSON
    /// object `substitutes`, each a `model` and its `parameters` by component instance
    /// path, standing in for them.
    ///
    /// A circuit that does not elaborate raises `ElaborationError` as
    /// [`Simulator`]'s constructor does.
    #[new]
    fn new(py: Python<'_>, circuit: &str, substitutes: &str) -> PyResult<Self> {
        let circuit = Circuit::from_json(circuit).map_err(|error| unreadable(&error))?;
        let structure = jolty_engine::Structure {
            substitutes: serde_json::from_str(substitutes).map_err(|error| unreadable(&error))?,
        };
        let inner = py
            .detach(|| jolty_engine::Elaborated::new(&circuit, &Library::standard(), &structure))
            .map_err(|error| elaboration_error(py, &error))?;
        Ok(Self { inner })
    }

    /// The values that replace the numbers of the JSON object `overrides`, by instance path
    /// then by the number's path among its parameters, at `temperature`, in kelvins.
    ///
    /// Raises `ElaborationError` naming every number the circuit's devices do not write.
    fn values(&self, py: Python<'_>, overrides: &str, temperature: f64) -> PyResult<Values> {
        let overrides = serde_json::from_str(overrides).map_err(|error| unreadable(&error))?;
        let inner = self
            .inner
            .values(&overrides, Some(temperature))
            .map_err(|error| elaboration_error(py, &error))?;
        Ok(Values { inner })
    }

    /// The structure `values` give the circuit, against its own.
    ///
    /// Raises `ElaborationError` naming every device whose numbers they put out of range.
    fn structure_of(&self, py: Python<'_>, values: &Values) -> PyResult<StructureKey> {
        let inner = py
            .detach(|| self.inner.structure_of(&values.inner))
            .map_err(|error| elaboration_error(py, &error))?;
        Ok(StructureKey { inner })
    }

    /// The circuit elaborated for the structure `values` give it, its parameters and
    /// everything else this one's.
    ///
    /// Raises `ElaborationError` as [`Elaborated::structure_of`] does.
    fn restructured(&self, py: Python<'_>, values: &Values) -> PyResult<Self> {
        let inner = py
            .detach(|| self.inner.restructured(&values.inner))
            .map_err(|error| elaboration_error(py, &error))?;
        Ok(Self { inner })
    }
}

#[pymethods]
impl StructureKey {
    /// Whether the values keep the structure of the circuit that found the key.
    #[getter]
    fn unchanged(&self) -> bool {
        self.inner.is_empty()
    }
}

/// A circuit elaborated for simulation.
#[pyclass(module = "jolty._core", frozen)]
struct Simulator {
    inner: Mutex<jolty_engine::Simulator>,
    unknowns: Vec<(String, &'static str)>,
    nets: Vec<String>,
    devices: usize,
    foundry_devices: Vec<String>,
}

/// A device a recording keeps: its path, its terminals' names, whether each is thermal, the
/// index of its first internal unknown and their number, its variables' names, and the
/// names of the regions its variable `region` numbers and of its margins.
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

/// What a recording keeps: each device, then each terminal of a component instance or a
/// logic device, its name and whether it is thermal.
type Recording = (Vec<RecordedDevice>, Vec<(String, bool)>);

impl Simulator {
    /// The simulator of `inner`, with what Python reads of it once.
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
    /// Elaborates the circuit whose JSON text is `circuit`, its devices at `temperature`,
    /// in kelvins, with the parameters of the JSON object `overrides`, by instance path,
    /// replacing the circuit's own, and the models of the JSON object `substitutes`, each a
    /// `model` and its `parameters` by component instance path, standing in for them.
    ///
    /// A circuit that does not elaborate raises `ElaborationError` with a list of its
    /// problems, each a dictionary of its `message` and its `instance`, as a diagnostic's,
    /// or `None`.
    #[new]
    fn new(
        py: Python<'_>,
        circuit: &str,
        temperature: f64,
        overrides: &str,
        substitutes: &str,
    ) -> PyResult<Self> {
        let circuit = Circuit::from_json(circuit).map_err(|error| unreadable(&error))?;
        let conditions = jolty_engine::Conditions {
            temperature,
            overrides: serde_json::from_str(overrides).map_err(|error| unreadable(&error))?,
            substitutes: serde_json::from_str(substitutes).map_err(|error| unreadable(&error))?,
        };
        let inner = py
            .detach(|| jolty_engine::Simulator::new(&circuit, &Library::standard(), &conditions))
            .map_err(|error| elaboration_error(py, &error))?;
        Ok(Self::wrapping(inner))
    }

    /// A simulator of `elaborated` at `values`, its devices made from the numbers they give,
    /// on the circuit restructured for them where they give some device another structure.
    ///
    /// Raises `ElaborationError` naming every device whose numbers they put out of range.
    #[staticmethod]
    fn from_elaborated(py: Python<'_>, elaborated: &Elaborated, values: &Values) -> PyResult<Self> {
        let shared = Arc::clone(&elaborated.inner);
        let inner = py
            .detach(|| jolty_engine::Simulator::from_elaborated(shared, values.inner.clone()))
            .map_err(|error| elaboration_error(py, &error))?;
        Ok(Self::wrapping(inner))
    }

    /// The circuit elaborated once that the simulator runs, which variants of its runs
    /// vary.
    #[getter]
    fn elaborated(&self) -> Elaborated {
        Elaborated {
            inner: Arc::clone(self.lock().elaborated()),
        }
    }

    /// Runs the runs `runs` of the circuit, from its start up to its stop, each at the
    /// simulator's values with those of the run's variant of `variants` in their place,
    /// releasing the interpreter while they run on the threads of `pool`, each run a pure
    /// function of its index.
    ///
    /// Each run runs `analyses` in order, each the JSON texts of an analysis and of its
    /// options; reads `measures`, each the index of the analysis it reads and the measure;
    /// and traces `traces`, each the index of an analysis and the indices of the unknowns
    /// whose values it keeps at each of its points: an operating point's one, a DC sweep's
    /// levels, a response's frequencies, or a transient's times, interpolated linearly
    /// between the times it accepted. Returns a dictionary of `NumPy` arrays, one row per
    /// run: `values`, of each number the variants vary; `measured`, what each measure read,
    /// or not a number; `causes`, -1 where it read a number and else the index among
    /// `reasons` of why it read none, each `("failed", [], "")` where the analysis did not
    /// end, or a `MeasureError`'s reason, numbers and message; `ended`, whether each
    /// analysis ended; and each traced array, runs × points × unknowns, among `traces`.
    /// `problems` holds, by row, the problems its run's values made, as
    /// `ElaborationError`'s, and `errors`, by row, the index of each analysis that failed
    /// and why. The run at `runs`' start is row 0.
    ///
    /// `each`, if given, is called under the interpreter's lock as each run ends, in no
    /// particular order, with the run's index, the simulator of its structure when its
    /// values gave the circuit another, its problems, if any, and each analysis's outcome,
    /// as `run` returns it, or why it failed; the first exception it raises is raised once
    /// every run has ended. Raises `ValueError` for input the engine cannot read, or runs
    /// that do not lie within the variants'.
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
        variants: &batch::Variants,
        runs: (u64, u64),
        measures: Vec<(usize, Py<measure::Measure>)>,
        traces: Vec<(usize, Vec<usize>)>,
        pool: &Pool,
        each: Option<Py<PyAny>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let (elaborated, base) = {
            let simulator = self.lock();
            (
                Arc::clone(simulator.elaborated()),
                simulator.values().clone(),
            )
        };
        let request = batch::Request {
            analyses,
            runs: runs.0..runs.1,
            measures,
            traces,
            each,
        };
        batch::run(py, (&elaborated, &base), variants, request, pool)
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

    /// Every path a recording can name, in order: each device's, such as `X1/R1`, then each
    /// component instance's and logic device's whose terminals the engine sums, such as
    /// `X1`.
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

    /// What the record whose JSON text is `record` keeps, and every other transistor's
    /// region when `regions`, in the order of every result's arrays: each recorded device,
    /// as its path, its terminals' names, whether each is thermal, the index among the
    /// unknowns of its first internal unknown and their number, its variables' names, and
    /// the names of its regions and of its margins, none for a device that is no
    /// transistor; then each transistor whose region alone is recorded, alike, with no
    /// terminal, no internal unknown and its region variables alone; then each recorded
    /// terminal of a component instance or a logic device, as its name, such as `X1.vdd`,
    /// and whether it is thermal.
    fn recording(&self, record: &str, regions: bool) -> PyResult<Recording> {
        let record: Record = serde_json::from_str(record).map_err(|error| unreadable(&error))?;
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

    /// Where the terminal `terminal` of the device, component instance or logic device at
    /// `instance` joins the circuit: the unknown that is its voltage, by name, or `None` on
    /// the reference, and whether it is thermal; `None` when there is no such terminal on
    /// an electrical or a thermal net.
    fn terminal(&self, instance: &str, terminal: &str) -> Option<(Option<String>, bool)> {
        let simulator = self.lock();
        let (node, thermal) = simulator.terminal(instance, terminal)?;
        let name = node.map(|unknown| simulator.unknowns()[unknown.index()].name.clone());
        Some((name, thermal))
    }

    /// The paths of the foundry's devices, which ngspice simulates in the engine's place,
    /// in the order they were placed.
    #[getter]
    fn foundry_devices(&self) -> Vec<String> {
        self.foundry_devices.clone()
    }

    /// The number of devices, analog and logic, with every component instance unfolded.
    #[getter]
    fn devices(&self) -> usize {
        self.devices
    }

    /// Every diagnostic of the structural checks, each a dictionary of its `rule`, its
    /// `message`, its `instances` and its `nets`; each instance is a dictionary of its `path`
    /// and its `source_location`, the script line that placed it or `None`.
    fn check<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let found = py.detach(|| jolty_checks::check(&self.lock()));
        diagnostics_to_python(py, found)
    }

    /// Every diagnostic of the rules on the circuit linearized at its operating point under
    /// `options`, a JSON object, as `check` gives them; none for a circuit of more unknowns
    /// than the checks linearize. It runs releasing the interpreter, inside `pool` when its
    /// options let it spread its work over threads. A circuit with no operating point
    /// raises `SimulationError`.
    fn check_operating_point<'py>(
        &self,
        py: Python<'py>,
        options: &str,
        pool: &Pool,
    ) -> PyResult<Bound<'py, PyList>> {
        let options: Options = serde_json::from_str(options).map_err(|error| unreadable(&error))?;
        let found = py.detach(|| {
            let spreads = self
                .lock()
                .spreads(&Analysis::OperatingPoint, &options, pool.threads());
            pool.run((&options, spreads), |options| {
                let mut simulator = self.lock();
                jolty_checks::check_operating_point(&mut simulator, options)
            })
        });
        diagnostics_to_python(
            py,
            found.map_err(|error| SimulationError::new_err(error.to_string()))?,
        )
    }

    /// Where the last analysis used logic devices outside their domains, each a dictionary
    /// of the device's `path`, the `bound` it left, such as `"load"`, the domain's `low`
    /// and `high`, the `value` furthest outside, the `time` it first left, and the `count`
    /// of times it was read outside.
    fn excursions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let simulator = self.lock();
        let found = PyList::empty(py);
        for excursion in simulator.excursions() {
            let item = PyDict::new(py);
            item.set_item(
                "instance",
                simulator.logic_devices()[excursion.device].path(),
            )?;
            item.set_item("bound", excursion.bound.name())?;
            item.set_item("minimum", excursion.minimum)?;
            item.set_item("maximum", excursion.maximum)?;
            item.set_item("value", excursion.value)?;
            item.set_item("time", excursion.time)?;
            item.set_item("count", excursion.count)?;
            found.append(item)?;
        }
        Ok(found)
    }

    /// Runs the analysis whose JSON text is `analysis` with the options whose JSON text is
    /// `options`, releasing the interpreter while it runs, inside `pool` when the options
    /// let it spread its work over threads.
    fn run<'py>(
        &self,
        py: Python<'py>,
        analysis: &str,
        options: &str,
        pool: &Pool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let analysis: Analysis =
            serde_json::from_str(analysis).map_err(|error| unreadable(&error))?;
        let options: Options = serde_json::from_str(options).map_err(|error| unreadable(&error))?;
        let outcome = py.detach(|| {
            let spreads = self.lock().spreads(&analysis, &options, pool.threads());
            pool.run((&options, spreads), |options| {
                let mut simulator = self.lock();
                simulator.run(&analysis, options)
            })
        });
        outcome_to_python(
            py,
            outcome.map_err(|error| SimulationError::new_err(error.to_string()))?,
        )
    }
}

impl Simulator {
    /// The engine's simulator, which an analysis on another thread may hold.
    fn lock(&self) -> std::sync::MutexGuard<'_, jolty_engine::Simulator> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// A `ValueError` for input the engine cannot read.
pub(crate) fn unreadable(error: &impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// The name of a quantity: `"voltage"`, `"current"`, `"temperature"` or `"state"`.
fn quantity_name(nature: Nature) -> &'static str {
    match nature {
        Nature::Temperature => "temperature",
        Nature::State => "state",
        _ if nature.is_current() => "current",
        _ => "voltage",
    }
}

/// The outcome of an analysis as a dictionary whose `kind` names the analysis.
pub(crate) fn outcome_to_python(
    py: Python<'_>,
    outcome: AnalysisResult,
) -> PyResult<Bound<'_, PyDict>> {
    let result = PyDict::new(py);
    match outcome {
        AnalysisResult::OperatingPoint(operating_point) => {
            result.set_item("kind", "operating_point")?;
            add_operating_point(&result, operating_point)?;
        }
        AnalysisResult::DcSweep(sweep) => {
            result.set_item("kind", "dc_sweep")?;
            add_dc_sweep(&result, sweep)?;
        }
        AnalysisResult::Ac(response) => {
            result.set_item("kind", "ac")?;
            result.set_item("source", response.source)?;
            result.set_item("frequencies", PyArray1::from_vec(py, response.frequencies))?;
            result.set_item("values", PyArray2::from_vec2(py, &response.values)?)?;
            result.set_item("currents", matrix(py, &response.currents)?)?;
            let operating_point = PyDict::new(py);
            add_operating_point(&operating_point, response.operating_point)?;
            result.set_item("operating_point", operating_point)?;
        }
        AnalysisResult::Transient(waveforms) => {
            result.set_item("kind", "transient")?;
            add_waveforms(&result, waveforms)?;
        }
        AnalysisResult::PeriodicSteadyState(found) => {
            result.set_item("kind", "periodic_steady_state")?;
            result.set_item("period", found.period)?;
            result.set_item("periods", found.periods)?;
            add_waveforms(&result, found.waveforms)?;
        }
        AnalysisResult::Sensitivity(sensitivities) => {
            result.set_item("kind", "sensitivity")?;
            result.set_item("outputs", sensitivities.outputs)?;
            let (parameters, values): (Vec<(String, String)>, Vec<f64>) = sensitivities
                .parameters
                .into_iter()
                .map(|parameter| ((parameter.instance, parameter.name), parameter.value))
                .unzip();
            result.set_item("parameters", parameters)?;
            result.set_item("values", PyArray1::from_vec(py, values))?;
            match &sensitivities.derivatives {
                Derivatives::Real(points) => result.set_item("derivatives", cube(py, points)?)?,
                Derivatives::Complex(points) => {
                    result.set_item("derivatives", cube(py, points)?)?;
                }
            }
            result.set_item("result", outcome_to_python(py, *sensitivities.result)?)?;
        }
        AnalysisResult::Stability(loop_gain) => {
            result.set_item("kind", "stability")?;
            result.set_item("probe", loop_gain.probe)?;
            result.set_item("frequencies", PyArray1::from_vec(py, loop_gain.frequencies))?;
            result.set_item("gains", PyArray1::from_vec(py, loop_gain.gains))?;
            let operating_point = PyDict::new(py);
            add_operating_point(&operating_point, loop_gain.operating_point)?;
            result.set_item("operating_point", operating_point)?;
        }
        AnalysisResult::Noise(spectrum) => {
            result.set_item("kind", "noise")?;
            add_noise(&result, spectrum)?;
        }
        AnalysisResult::OperatingPoints(points) => {
            result.set_item("kind", "operating_points")?;
            add_operating_points(&result, points)?;
        }
        AnalysisResult::PoleZero(found) => {
            result.set_item("kind", "pole_zero")?;
            add_poles_zeros(&result, found)?;
        }
        AnalysisResult::TransferFunction(transfer) => {
            result.set_item("kind", "transfer_function")?;
            result.set_item("source", transfer.source)?;
            result.set_item("input", quantity_name(transfer.input))?;
            result.set_item("output", transfer.output)?;
            result.set_item("reference", transfer.reference)?;
            result.set_item("gain", transfer.gain)?;
            result.set_item("input_resistance", transfer.input_resistance)?;
            result.set_item("output_resistance", transfer.output_resistance)?;
            let operating_point = PyDict::new(py);
            add_operating_point(&operating_point, transfer.operating_point)?;
            result.set_item("operating_point", operating_point)?;
        }
    }
    Ok(result)
}

/// The waveforms of a transient, or of a periodic steady state's period: the times, the
/// values one row per time, what it recorded, each logic net's changes and how it went.
fn add_waveforms(result: &Bound<'_, PyDict>, waveforms: TransientResult) -> PyResult<()> {
    let py = result.py();
    result.set_item("times", PyArray1::from_vec(py, waveforms.times))?;
    result.set_item("values", stacked(py, waveforms.values)?)?;
    add_observations(result, waveforms.observations)?;
    let traces = PyList::empty(py);
    for trace in waveforms.logic {
        let (times, levels): (Vec<f64>, Vec<LogicLevel>) = trace.into_iter().unzip();
        traces.append((
            PyArray1::from_vec(py, times),
            PyArray1::from_vec(py, numbers(&levels)),
        ))?;
    }
    result.set_item("logic", traces)?;
    result.set_item(
        "statistics",
        statistics_to_python(py, waveforms.statistics)?,
    )
}

/// The operating points of a DC sweep, one row per level, and how each was found.
fn add_dc_sweep(result: &Bound<'_, PyDict>, sweep: DcSweepResult) -> PyResult<()> {
    let py = result.py();
    result.set_item("source", sweep.source)?;
    result.set_item("parameter", sweep.parameter)?;
    result.set_item("nature", sweep.nature.map(quantity_name))?;
    result.set_item("swept", PyArray1::from_vec(py, sweep.swept))?;
    result.set_item("values", stacked(py, sweep.values)?)?;
    add_observations(result, sweep.observations)?;
    let logic: Vec<Vec<f64>> = sweep.logic.iter().map(|levels| numbers(levels)).collect();
    result.set_item("logic", matrix(py, &logic)?)?;
    let strategies: Vec<&str> = sweep.strategies.iter().map(|found| found.name()).collect();
    result.set_item("strategies", strategies)?;
    result.set_item("iterations", sweep.iterations)
}

/// The noise at an output, at each frequency, with each device noise source's share: the
/// sources as `(device, name)` pairs, and their contributions one row per source; and the
/// source it is referred to, if any, with the gains from it.
fn add_noise(result: &Bound<'_, PyDict>, spectrum: NoiseResult) -> PyResult<()> {
    let py = result.py();
    result.set_item("output", spectrum.output)?;
    result.set_item("reference", spectrum.reference)?;
    result.set_item("nature", quantity_name(spectrum.nature))?;
    let count = spectrum.frequencies.len();
    result.set_item("frequencies", PyArray1::from_vec(py, spectrum.frequencies))?;
    result.set_item("densities", PyArray1::from_vec(py, spectrum.densities))?;
    let input = match spectrum.input {
        Some(input) => {
            let found = PyDict::new(py);
            found.set_item("source", input.source)?;
            found.set_item("nature", quantity_name(input.nature))?;
            found.set_item("gains", PyArray1::from_vec(py, input.gains))?;
            Some(found)
        }
        None => None,
    };
    result.set_item("input", input)?;
    let sources: Vec<(String, &str)> = spectrum
        .contributions
        .iter()
        .map(|contribution| (contribution.instance.clone(), contribution.name))
        .collect();
    let rows = sources.len();
    let flat: Vec<f64> = spectrum
        .contributions
        .into_iter()
        .flat_map(|contribution| contribution.densities)
        .collect();
    result.set_item("sources", sources)?;
    result.set_item(
        "contributions",
        PyArray1::from_vec(py, flat).reshape([rows, count])?,
    )?;
    let operating_point = PyDict::new(py);
    add_operating_point(&operating_point, spectrum.operating_point)?;
    result.set_item("operating_point", operating_point)
}

/// The DC solutions, each a dictionary of an operating point's own, under `points`.
fn add_operating_points(
    result: &Bound<'_, PyDict>,
    points: Vec<OperatingPointResult>,
) -> PyResult<()> {
    let solutions = PyList::empty(result.py());
    for point in points {
        let solution = PyDict::new(result.py());
        add_operating_point(&solution, point)?;
        solutions.append(solution)?;
    }
    result.set_item("points", solutions)
}

/// The poles, and the zeros of the transfer function between the source and the output
/// echoed, if any, with the operating point the circuit was linearized at.
fn add_poles_zeros(result: &Bound<'_, PyDict>, found: PoleZeroResult) -> PyResult<()> {
    let py = result.py();
    result.set_item("source", found.source)?;
    result.set_item("output", found.output)?;
    result.set_item("reference", found.reference)?;
    result.set_item("poles", PyArray1::from_vec(py, found.poles))?;
    result.set_item(
        "zeros",
        found.zeros.map(|zeros| PyArray1::from_vec(py, zeros)),
    )?;
    let operating_point = PyDict::new(py);
    add_operating_point(&operating_point, found.operating_point)?;
    result.set_item("operating_point", operating_point)
}

fn add_operating_point(
    result: &Bound<'_, PyDict>,
    operating_point: OperatingPointResult,
) -> PyResult<()> {
    result.set_item(
        "values",
        PyArray1::from_vec(result.py(), operating_point.values),
    )?;
    result.set_item(
        "logic",
        PyArray1::from_vec(result.py(), numbers(&operating_point.logic)),
    )?;
    let RecordedValues {
        currents,
        charges,
        conductances,
        capacitances,
        variables,
    } = operating_point.observation;
    let py = result.py();
    result.set_item("currents", PyArray1::from_vec(py, currents))?;
    result.set_item("charges", PyArray1::from_vec(py, charges))?;
    result.set_item("conductances", PyArray1::from_vec(py, conductances))?;
    result.set_item("capacitances", PyArray1::from_vec(py, capacitances))?;
    result.set_item("variables", PyArray1::from_vec(py, variables))?;
    result.set_item("strategy", operating_point.strategy.name())?;
    result.set_item("iterations", operating_point.iterations)
}

/// Adds what the recorded devices did at each point to `result`: the terminals' `currents`
/// and `charges`, the derivatives' `conductances` and `capacitances`, and the `variables`,
/// one row per point.
fn add_observations(result: &Bound<'_, PyDict>, observations: RecordedSeries) -> PyResult<()> {
    let py = result.py();
    let points = observations.points;
    // Each quantity's rows already lie end to end, so the arrays take them without a copy.
    let rows = |values: Vec<f64>| {
        let width = values.len().checked_div(points).unwrap_or(0);
        PyArray1::from_vec(py, values).reshape([points, width])
    };
    result.set_item("currents", rows(observations.currents)?)?;
    result.set_item("charges", rows(observations.charges)?)?;
    result.set_item("conductances", rows(observations.conductances)?)?;
    result.set_item("capacitances", rows(observations.capacitances)?)?;
    result.set_item("variables", rows(observations.variables)?)
}

/// `rows`, all of one length, as a two-dimensional array with as many rows even when
/// they are empty; each row is freed once copied, so the values are held about once.
fn stacked(py: Python<'_>, rows: Vec<Vec<f64>>) -> PyResult<Bound<'_, PyArray2<f64>>> {
    let (count, columns) = (rows.len(), rows.first().map_or(0, Vec::len));
    let mut flat = Vec::with_capacity(count * columns);
    for row in rows {
        flat.extend_from_slice(&row);
    }
    PyArray1::from_vec(py, flat).reshape([count, columns])
}

/// Logic levels as numbers: 0, 1, and NaN for an unknown level.
fn numbers(levels: &[LogicLevel]) -> Vec<f64> {
    levels
        .iter()
        .map(|level| match level {
            LogicLevel::Low => 0.0,
            LogicLevel::High => 1.0,
            LogicLevel::Unknown => f64::NAN,
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

/// `points` as a three-dimensional array, with as many points even when they are empty.
fn cube<'py, T: Element + Copy>(
    py: Python<'py>,
    points: &[Vec<Vec<T>>],
) -> PyResult<Bound<'py, PyArray3<T>>> {
    let rows = points.first().map_or(0, Vec::len);
    let columns = points
        .first()
        .and_then(|first| first.first())
        .map_or(0, Vec::len);
    let flat: Vec<T> = points.iter().flatten().flatten().copied().collect();
    PyArray1::from_vec(py, flat).reshape([points.len(), rows, columns])
}

/// Each of `found` as a dictionary of its `rule`, its `message`, its `instances` and its
/// `nets`.
fn diagnostics_to_python(
    py: Python<'_>,
    found: Vec<jolty_checks::Diagnostic>,
) -> PyResult<Bound<'_, PyList>> {
    let diagnostics = PyList::empty(py);
    for diagnostic in found {
        let item = PyDict::new(py);
        item.set_item("rule", diagnostic.rule)?;
        item.set_item("message", diagnostic.message)?;
        let instances = PyList::empty(py);
        for location in &diagnostic.instances {
            instances.append(location_to_python(py, location)?)?;
        }
        item.set_item("instances", instances)?;
        item.set_item("nets", diagnostic.nets)?;
        diagnostics.append(item)?;
    }
    Ok(diagnostics)
}

/// Each problem of `error`, a dictionary of its `message` and its `instance` or `None`.
pub(crate) fn problems_to_python<'py>(
    py: Python<'py>,
    error: &jolty_engine::ElaborationError,
) -> PyResult<Bound<'py, PyList>> {
    let problems = PyList::empty(py);
    for problem in error.problems() {
        let item = PyDict::new(py);
        item.set_item("message", problem.message())?;
        let instance = problem
            .location()
            .map(|location| location_to_python(py, location))
            .transpose()?;
        item.set_item("instance", instance)?;
        problems.append(item)?;
    }
    Ok(problems)
}

/// An instance as a dictionary of its `path` and its `source_location`, the script line that placed
/// it, as a dictionary of its `file`, `line` and `function`, or `None`.
fn location_to_python<'py>(py: Python<'py>, location: &Location) -> PyResult<Bound<'py, PyDict>> {
    let instance = PyDict::new(py);
    instance.set_item("path", &location.path)?;
    let source = match &location.source_location {
        Some(source) => {
            let line = PyDict::new(py);
            line.set_item("file", &source.file)?;
            line.set_item("line", source.line)?;
            line.set_item("function", &source.function)?;
            Some(line)
        }
        None => None,
    };
    instance.set_item("source_location", source)?;
    Ok(instance)
}

fn statistics_to_python(py: Python<'_>, statistics: Statistics) -> PyResult<Bound<'_, PyDict>> {
    let result = PyDict::new(py);
    result.set_item("initial", statistics.initial.name())?;
    result.set_item("accepted", statistics.accepted)?;
    result.set_item("rejected", statistics.rejected)?;
    result.set_item("landed", statistics.landed)?;
    result.set_item("iterations", statistics.iterations)?;
    let work = statistics.work;
    let counts = PyDict::new(py);
    counts.set_item("iterations", work.iterations)?;
    counts.set_item("factorizations", work.factorizations)?;
    counts.set_item("refactorizations", work.refactorizations)?;
    counts.set_item("fallbacks", work.fallbacks)?;
    counts.set_item("reuses", work.reuses)?;
    result.set_item("work", counts)?;
    let profile = match statistics.profile {
        Some(profile) => {
            let seconds = PyDict::new(py);
            for (name, spent) in [
                ("preparing", profile.preparing),
                ("evaluating", profile.evaluating),
                ("assembling", profile.assembling),
                ("factoring", profile.factoring),
                ("solving", profile.solving),
                ("step_control", profile.step_control),
                ("recording", profile.recording),
                ("other", profile.other),
            ] {
                seconds.set_item(name, spent.as_secs_f64())?;
            }
            Some(seconds)
        }
        None => None,
    };
    result.set_item("profile", profile)?;
    Ok(result)
}

/// The error of a circuit that does not elaborate: `ElaborationError` with a list of its
/// problems, each a dictionary of its `message` and its `instance`.
pub(crate) fn elaboration_error(py: Python<'_>, error: &jolty_engine::ElaborationError) -> PyErr {
    match problems_to_python(py, error) {
        Ok(problems) => ElaborationError::new_err((problems.unbind(),)),
        Err(error) => error,
    }
}

/// The Rust core of jolty.
/// The terminals of a device of `model` with `parameters`, a JSON object, in the order
/// the model declares them.
#[pyfunction]
fn terminals(model: &str, parameters: &str) -> PyResult<Vec<String>> {
    let library = Library::standard();
    let parameters = serde_json::from_str(parameters).map_err(|error| unreadable(&error))?;
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

/// The analog model of `library` named `name`.
fn analog<'a>(library: &'a Library, name: &str) -> PyResult<&'a Arc<dyn Model>> {
    library
        .get(name)
        .ok_or_else(|| PyValueError::new_err(format!("no model is named `{name}`")))
}

/// The engine's default options, SPICE's, as the JSON object its runs read their options
/// from: every option jolty.plan.Options sets has its default here alone.
#[pyfunction]
fn default_options() -> PyResult<String> {
    serde_json::to_string(&Options::default())
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// The break-evens each threshold of the engine's parallelism left unset takes, on each
/// number of threads they were measured on, as a JSON object: the threads, then each
/// threshold's break-evens, none where no size measured paid.
#[pyfunction]
fn break_evens() -> PyResult<String> {
    serde_json::to_string(&jolty_engine::BREAK_EVENS)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// The name of every model, analog and logic, in alphabetical order.
#[pyfunction]
fn models() -> Vec<&'static str> {
    Library::standard().names().collect()
}

/// The parameters of the model named `model`, as a JSON schema of an object, every type
/// within it written in place: each parameter's type, its default, unless it is
/// required, and its description, the documentation of its field in the engine.
#[pyfunction]
fn schema(model: &str) -> PyResult<String> {
    let library = Library::standard();
    let schema = match library.logic(model) {
        Some(logic) => logic.schema(),
        None => analog(&library, model)?.schema(),
    };
    Ok(schema.as_value().to_string())
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

/// The tree of `expression`, an expression in jolty's language, as JSON: a number, the
/// text `"time"`, `{"voltage": [a, b]}` with `b` null for the ground, `{"current": p}`,
/// `{"operation": name, "operands": [...]}`, or `{"table": input, "abscissae": [...],
/// "ordinates": [...], "rounding": r}`, as [`jolty_expressions::Term`] describes them.
#[pyfunction]
fn expression_tree(expression: &str) -> PyResult<String> {
    let parsed: jolty_expressions::Expression = expression
        .parse()
        .map_err(|error| PyValueError::new_err(format!("`{expression}`: {error}")))?;
    Ok(term_to_json(&parsed.tree()).to_string())
}

/// The level of `waveform`, a source's waveform as JSON, at `time` in seconds: at time
/// zero, the level every analysis but a transient one reads.
#[pyfunction]
fn level(waveform: &str, time: f64) -> PyResult<f64> {
    let waveform: jolty_models::waveform::Waveform =
        serde_json::from_str(waveform).map_err(|error| unreadable(&error))?;
    Ok(waveform.value(time))
}

/// The parameters `parameters`, a JSON object, of a device of `model`, as JSON, each one
/// left out given the value the engine takes, so a SPICE netlist computes what the engine
/// does.
#[pyfunction]
fn defaulted(model: &str, parameters: &str) -> PyResult<String> {
    let library = Library::standard();
    let parameters = serde_json::from_str(parameters).map_err(|error| unreadable(&error))?;
    let defaulted = match library.logic(model) {
        Some(logic) => logic.defaulted(&parameters),
        None => analog(&library, model)?.defaulted(&parameters),
    };
    let defaulted = defaulted
        .map_err(|error| PyValueError::new_err(format!("a {model}'s parameters: {}", error.0)))?;
    Ok(serde_json::Value::Object(defaulted).to_string())
}

/// A term of an expression's tree as JSON, as [`expression_tree`] writes it.
fn term_to_json(term: &jolty_expressions::Term) -> serde_json::Value {
    use jolty_expressions::{Term, Variable};
    use serde_json::json;
    match term {
        Term::Constant(value) => json!(value),
        Term::Variable(Variable::Time) => json!("time"),
        Term::Variable(Variable::Voltage(positive, negative)) => {
            json!({"voltage": [positive, negative]})
        }
        Term::Variable(Variable::Current(port)) => json!({"current": port}),
        Term::Operation(name, operands) => json!({
            "operation": name,
            "operands": operands.iter().map(term_to_json).collect::<Vec<_>>(),
        }),
        Term::Table {
            input,
            abscissae,
            ordinates,
            rounding,
        } => json!({
            "table": term_to_json(input),
            "abscissae": abscissae,
            "ordinates": ordinates,
            "rounding": rounding,
        }),
    }
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

/// The values of the E series named `series`, such as `"E24"`, in the decade from 1 to 10.
#[pyfunction]
fn mantissas(series: &str) -> PyResult<Vec<f64>> {
    jolty_checks::preferred::Series::named(series)
        .map(jolty_checks::preferred::Series::mantissas)
        .ok_or_else(|| PyValueError::new_err(format!("no E series is named `{series}`")))
}

/// The value of each number of `varied`, a JSON list of objects of an `instance`, its path, a
/// number's `parameter` path, its `nominal` value and its `tolerance`, in the sample `index`
/// of a Monte Carlo analysis seeded with `seed`, in order: each drawn from the seed, the
/// index and its own name alone, as the engine draws it in a batch.
///
/// Raises `ValueError` for a list the engine cannot read.
#[pyfunction]
fn sample(varied: &str, seed: u64, index: u64) -> PyResult<Vec<f64>> {
    let varied: Vec<jolty_engine::variants::Toleranced> =
        serde_json::from_str(varied).map_err(|error| unreadable(&error))?;
    Ok(varied
        .iter()
        .map(|varied| varied.drawn(seed, index))
        .collect())
}

/// Every number of every device instance of the circuit whose JSON text is `circuit` that
/// has a tolerance, in the order of the hierarchy, each instance's numbers by path: each
/// as its instance's path, its own path, its nominal value and its tolerance's JSON text.
///
/// Raises `ValueError` for a tolerance of a parameter given no number.
#[pyfunction]
fn toleranced(circuit: &str) -> PyResult<Vec<(String, String, f64, String)>> {
    let circuit = Circuit::from_json(circuit).map_err(|error| unreadable(&error))?;
    let found = jolty_engine::variants::toleranced(&circuit).map_err(|error| unreadable(&error))?;
    found
        .into_iter()
        .map(|varied| {
            let tolerance =
                serde_json::to_string(&varied.tolerance).map_err(|error| unreadable(&error))?;
            Ok((varied.instance, varied.parameter, varied.nominal, tolerance))
        })
        .collect()
}

#[pymodule(name = "_core", gil_used = false)]
fn extension(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add_class::<Simulator>()?;
    module.add_class::<Pool>()?;
    module.add_function(wrap_pyfunction!(sample, module)?)?;
    module.add_function(wrap_pyfunction!(toleranced, module)?)?;
    measure::register(module)?;
    module.add_class::<Elaborated>()?;
    module.add_class::<Values>()?;
    module.add_class::<StructureKey>()?;
    module.add_class::<batch::Variants>()?;
    module.add_function(wrap_pyfunction!(rules, module)?)?;
    module.add_function(wrap_pyfunction!(terminals, module)?)?;
    module.add_function(wrap_pyfunction!(constant, module)?)?;
    module.add_function(wrap_pyfunction!(expression_tree, module)?)?;
    module.add_function(wrap_pyfunction!(defaulted, module)?)?;
    module.add_function(wrap_pyfunction!(default_options, module)?)?;
    module.add_function(wrap_pyfunction!(break_evens, module)?)?;
    module.add_function(wrap_pyfunction!(level, module)?)?;
    module.add_function(wrap_pyfunction!(models, module)?)?;
    module.add_function(wrap_pyfunction!(schema, module)?)?;
    module.add_function(wrap_pyfunction!(mantissas, module)?)?;
    module.add("EXPRESSIONS", jolty_expressions::LANGUAGE)?;
    module.add("BOLTZMANN", jolty_models::BOLTZMANN)?;
    module.add("ELEMENTARY_CHARGE", jolty_models::ELEMENTARY_CHARGE)?;
    module.add("REFERENCE_TEMPERATURE", jolty_models::REFERENCE_TEMPERATURE)?;
    module.add("REGION", jolty_models::REGION)?;
    module.add("ElaborationError", py.get_type::<ElaborationError>())?;
    module.add("SimulationError", py.get_type::<SimulationError>())?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
