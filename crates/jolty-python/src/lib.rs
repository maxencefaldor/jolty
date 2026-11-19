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
use jolty_models::{Library, Model, ParameterError};
use numpy::{Complex64, PyArray1, PyReadonlyArray2};
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::convert::{
    diagnostics_to_python, excursions_to_python, jvp_to_python, outcome_to_python, quantity_name,
};
use crate::helpers::{
    cotangent, elaboration_result, lock_unpoisoned, parse_json, simulation_result, value_result,
};

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
    "A circuit cannot be elaborated; the single argument lists every problem as a dictionary \
     of `message` and `instance`."
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
        let inner = value_result(
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads.unwrap_or(0))
                .build(),
        )?;
        Ok(Self { inner })
    }

    /// The number of threads.
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

    /// The values that replace the numbers of the JSON object `overrides`, by instance path
    /// then by the number's path among its parameters, at `temperature`, in kelvins.
    ///
    /// Raises `ElaborationError` naming every number the circuit's devices do not write.
    fn values(&self, py: Python<'_>, overrides: &str, temperature: f64) -> PyResult<Values> {
        let overrides = parse_json!(overrides)?;
        let inner = elaboration_result(py, self.inner.values(&overrides, Some(temperature)))?;
        Ok(Values { inner })
    }

    /// The structure `values` give the circuit, against its own.
    ///
    /// Raises `ElaborationError` naming every device whose numbers they put out of range.
    fn structure_of(&self, py: Python<'_>, values: &Values) -> PyResult<StructureKey> {
        let inner = elaboration_result(py, py.detach(|| self.inner.structure_of(&values.inner)))?;
        Ok(StructureKey { inner })
    }

    /// The circuit elaborated for the structure `values` give it, its parameters and
    /// everything else this one's.
    ///
    /// Raises `ElaborationError` as [`Elaborated::structure_of`] does.
    fn restructured(&self, py: Python<'_>, values: &Values) -> PyResult<Self> {
        let inner = elaboration_result(py, py.detach(|| self.inner.restructured(&values.inner)))?;
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

/// A VJP's pullback, as the engine found it: what carries a cotangent on an analysis's
/// outputs back to the parameters it names.
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

    /// A simulator of `elaborated` at `values`, its devices made from the numbers they give,
    /// on the circuit restructured for them where they give some device another structure.
    ///
    /// Raises `ElaborationError` naming every device whose numbers they put out of range.
    #[staticmethod]
    fn from_elaborated(py: Python<'_>, elaborated: &Elaborated, values: &Values) -> PyResult<Self> {
        let shared = Arc::clone(&elaborated.inner);
        let inner = elaboration_result(
            py,
            py.detach(|| jolty_engine::Simulator::from_elaborated(shared, values.inner.clone())),
        )?;
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

    /// The VJP of the analysis whose JSON text is `analysis`, as a sensitivity's `of` names
    /// it, with the options whose JSON text is `options`, with respect to `parameters`,
    /// each its instance's path and its name, on each of the runs `runs` of the circuit at
    /// the simulator's values with those of the run's variant of `variants` in their place,
    /// each a pure function of its index, on the threads of `pool`, the interpreter
    /// released, as `batch` runs them: a dictionary of the `traces` of the unknowns at
    /// `unknowns`, by their indices, runs × points × unknowns, as `batch` traces them, each
    /// oscillator's `periods`, the `problems` by row and the `errors` by row; and the
    /// pullbacks, a run each. Raises `ValueError` for input the engine cannot read, or runs
    /// that do not lie within the variants'.
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

    /// Each unknown's name and quantity, `"voltage"`, `"current"`, `"temperature"` or
    /// `"state"`, in the order of every solution's values.
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

    /// Where the last analysis used logic devices outside their domains, each a dictionary
    /// of the device's `instance`, the `bound` it left, such as `"load"`, the domain's
    /// `minimum` and `maximum`, the `value` furthest outside, the `time` it first left, and
    /// the `count` of times it was read outside.
    fn excursions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        excursions_to_python(py, &self.lock())
    }

    /// The JVP of the analysis whose JSON text is `analysis`, as a sensitivity's `of` names
    /// it, with the options whose JSON text is `options`, along each of `tangents`, a
    /// number per parameter of `parameters`, each its instance's path and its name:
    /// a dictionary of the analysis's `result`, the `tangents`, by point, unknown and
    /// tangent, and an oscillator's `period`'s along each tangent, or `None`.
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

    /// The VJP of the analysis whose JSON text is `analysis`, as a sensitivity's `of` names
    /// it, with the options whose JSON text is `options`, with respect to `parameters`,
    /// each its instance's path and its name: the analysis's result, and its pullback.
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
        let result = outcome_to_python(py, inner.result.clone())?;
        Ok((result, Pullback { inner }))
    }

    /// The value on each parameter of `pullback` of the cotangent `values`, on every
    /// unknown at each point of its analysis, real or complex, by point then unknown, and
    /// `period` on an oscillator's period, with the options whose JSON text is `options`.
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
        let analysis: Analysis = parse_json!(analysis)?;
        let options: Options = parse_json!(options)?;
        let outcome = self.simulate(
            py,
            pool,
            &options,
            |simulator| simulator.spreads(&analysis, &options, pool.threads()),
            |simulator, options| simulator.run(&analysis, options),
        )?;
        outcome_to_python(py, outcome)
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

/// The terminals of a device of `model` with `parameters`, a JSON object, in the order
/// the model declares them.
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

/// The engine's default options, SPICE's, as the JSON object its runs read their options
/// from: every option `jolty.plan.Options` sets has its default here alone.
#[pyfunction]
fn default_options() -> PyResult<String> {
    value_result(serde_json::to_string(&Options::default()))
}

/// The break-evens each threshold of the engine's parallelism left unset takes, on each
/// number of threads they were measured on, as a JSON object: the threads, then each
/// threshold's break-evens, none where no size measured paid.
#[pyfunction]
fn break_evens() -> PyResult<String> {
    value_result(serde_json::to_string(&jolty_engine::BREAK_EVENS))
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
    value_result(term_to_json_string(&parsed.tree()))
}

/// The level of `waveform`, a source's waveform as JSON, at `time` in seconds: at time
/// zero, the level every analysis but a transient one reads.
#[pyfunction]
fn level(waveform: &str, time: f64) -> PyResult<f64> {
    let waveform: jolty_models::waveform::Waveform = parse_json!(waveform)?;
    Ok(waveform.value(time))
}

/// The parameters `parameters`, a JSON object, of a device of `model`, as JSON, each one
/// left out given the value the engine takes, so a SPICE netlist computes what the engine
/// does.
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
    let varied: Vec<jolty_engine::variants::Toleranced> = parse_json!(varied)?;
    Ok(varied
        .iter()
        .map(|varied| varied.drawn(seed, index))
        .collect())
}

/// The standard `variate`, `"normal"` or `"uniform"`, a process kit's statistical function
/// draws in the sample `index` of a Monte Carlo analysis seeded with `seed`, at the call
/// `call` within `scope`, as the engine draws its tolerances.
///
/// Raises `ValueError` for another variate.
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

/// Every number of every device instance of the circuit whose JSON text is `circuit` that
/// has a tolerance, in the order of the hierarchy, each instance's numbers by path: each
/// as its instance's path, its own path, its nominal value and its tolerance's JSON text.
///
/// Raises `ValueError` for a tolerance of a parameter given no number, or for a
/// component cycle.
#[pyfunction]
fn toleranced(circuit: &str) -> PyResult<Vec<(String, String, f64, String)>> {
    let circuit = value_result(Circuit::from_json(circuit))?;
    let found = value_result(jolty_engine::variants::toleranced(&circuit))?;
    found
        .into_iter()
        .map(|varied| {
            let tolerance = value_result(serde_json::to_string(&varied.tolerance))?;
            Ok((varied.instance, varied.parameter, varied.nominal, tolerance))
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
