//! Many runs of one circuit in one call: the engine's variants, as `jolty._core.Variants`,
//! and its batch executor, which `jolty._core.Simulator.batch` runs through a sink that
//! writes each run's numbers straight into `NumPy` arrays, at the run's row.
//!
//! The arrays hold, one row per run of the range of runs asked for: the values of the
//! numbers the variants vary, what each measure read or why it read nothing, whether each
//! analysis ended, and the values of the unknowns traced at each point of their analyses.
//! A run's row is its index less the range's start, so the order the runs end in never
//! shows. A function Python gives sees each run's results too, under
//! the interpreter's lock, as each run ends.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ops::Range;
use std::sync::{Arc, Mutex, PoisonError};

use jolty_engine::batch::{Batch, Mapped, Measured, Run, Sink, Unmeasured, Variants as _};
use jolty_engine::measure::MeasureError;
use jolty_engine::variants::{Axis, Explicit, Grid, MonteCarlo, Product, Toleranced};
use jolty_engine::{
    Analysis, Cotangent, Cotangents, Differentiated, ElaborationError, Options,
    ParameterCotangents, ParameterId, Values,
};
use numpy::{Complex64, PyArray1, PyArrayMethods, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::{Elaborated, Pool, Simulator, elaboration_error, measure, unreadable};

/// The variants of a batch as Python composes them.
#[derive(Debug, Clone)]
enum Varied {
    Explicit(Arc<Explicit>),
    Grid(Arc<Grid>),
    /// The samples, and their seed.
    MonteCarlo(Arc<MonteCarlo>, u64),
    Product(Arc<Product<Self, Self>>),
}

impl jolty_engine::batch::Variants for Varied {
    fn len(&self) -> u64 {
        match self {
            Self::Explicit(explicit) => explicit.len(),
            Self::Grid(grid) => grid.len(),
            Self::MonteCarlo(samples, _) => samples.len(),
            Self::Product(product) => product.len(),
        }
    }

    fn values(&self, index: u64) -> Values {
        match self {
            Self::Explicit(explicit) => explicit.values(index),
            Self::Grid(grid) => grid.values(index),
            Self::MonteCarlo(samples, _) => samples.values(index),
            Self::Product(product) => product.values(index),
        }
    }

    fn chain(&self) -> u64 {
        match self {
            Self::Explicit(explicit) => explicit.chain(),
            Self::Grid(grid) => grid.chain(),
            Self::MonteCarlo(samples, _) => samples.chain(),
            Self::Product(product) => product.chain(),
        }
    }
}

impl Varied {
    /// The Monte Carlo sample the run at `index` draws, as its seed and its index among the
    /// samples: of the second variants of a product first; none when it draws none.
    fn sample(&self, index: u64) -> Option<(u64, u64)> {
        match self {
            Self::Explicit(_) | Self::Grid(_) => None,
            Self::MonteCarlo(_, seed) => Some((*seed, index)),
            Self::Product(product) => {
                let (first, second) = product.parts();
                let count = second.len().max(1);
                second
                    .sample(index % count)
                    .or_else(|| first.sample(index / count))
            }
        }
    }
}

/// A number a batch's runs vary: a parameter, by its instance's path and its own, or the
/// temperature.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum Column {
    Parameter(ParameterId, String, String),
    Temperature,
}

/// The values of every run of a batch, each a pure function of the run's index, of one
/// elaborated circuit: listed, on a grid, drawn as Monte Carlo samples, or every one of one
/// at every one of another.
#[pyclass(module = "jolty._core", frozen)]
pub(crate) struct Variants {
    varied: Varied,
    elaborated: Arc<jolty_engine::Elaborated>,
    /// The numbers the runs vary, in the order of a batch's columns of values.
    columns: Vec<Column>,
}

/// A parameter, by its instance's path and its own.
type Named = (String, String);

/// Numbers by instance path, then by parameter path.
type Replaced = BTreeMap<String, BTreeMap<String, f64>>;

/// `overrides`, a JSON object by instance path then by parameter path, as a map.
fn overrides(text: &str) -> PyResult<Replaced> {
    serde_json::from_str(text).map_err(|error| unreadable(&error))
}

#[pymethods]
impl Variants {
    /// Runs at listed values: each a JSON object of the numbers it replaces, by instance
    /// path then by parameter path, and a temperature in kelvins, or none for the
    /// circuit's.
    ///
    /// Raises `ElaborationError` naming every number the circuit's devices do not write.
    #[staticmethod]
    fn explicit(
        py: Python<'_>,
        elaborated: &Elaborated,
        runs: Vec<(String, Option<f64>)>,
    ) -> PyResult<Self> {
        let circuit = &elaborated.inner;
        let mut columns = BTreeSet::new();
        let mut listed = Vec::with_capacity(runs.len());
        let mut heated = false;
        for (text, temperature) in runs {
            let replaced = overrides(&text)?;
            let values = circuit
                .values(&replaced, temperature)
                .map_err(|error| elaboration_error(py, &error))?;
            for (path, numbers) in replaced {
                for name in numbers.into_keys() {
                    if let Some(parameter) = circuit.parameter(&path, &name) {
                        columns.insert(Column::Parameter(parameter, path.clone(), name));
                    }
                }
            }
            heated |= temperature.is_some();
            listed.push(values);
        }
        let mut columns: Vec<Column> = columns.into_iter().collect();
        if heated {
            columns.push(Column::Temperature);
        }
        Ok(Self {
            varied: Varied::Explicit(Arc::new(Explicit(listed))),
            elaborated: Arc::clone(circuit),
            columns,
        })
    }

    /// Runs on a grid of `axes`, each a parameter, by its instance's path and its own, or
    /// none for the temperature, and its values in order, in kelvins for the temperature;
    /// the last axis fastest, its lines chains of warm runs when `warm`.
    ///
    /// Raises `ElaborationError` naming every number the circuit's devices do not write,
    /// and `ValueError` for an axis of no values.
    #[staticmethod]
    fn grid(
        py: Python<'_>,
        elaborated: &Elaborated,
        axes: Vec<(Option<Named>, Vec<f64>)>,
        warm: bool,
    ) -> PyResult<Self> {
        let circuit = &elaborated.inner;
        let mut named = Replaced::new();
        for ((path, name), _) in axes
            .iter()
            .filter_map(|(axis, values)| axis.as_ref().map(|axis| (axis, values)))
        {
            named
                .entry(path.clone())
                .or_default()
                .insert(name.clone(), 0.0);
        }
        // Every number an axis names, checked as replaced values are.
        circuit
            .values(&named, None)
            .map_err(|error| elaboration_error(py, &error))?;
        let mut stepped = Vec::with_capacity(axes.len());
        let mut columns = Vec::with_capacity(axes.len());
        for (axis, values) in axes {
            if let Some((path, name)) = axis {
                let parameter = circuit.parameter(&path, &name).ok_or_else(|| {
                    PyValueError::new_err(format!("{path} writes no number at {name}"))
                })?;
                stepped.push((Axis::Parameter(parameter), values));
                columns.push(Column::Parameter(parameter, path, name));
            } else {
                stepped.push((Axis::Temperature, values));
                columns.push(Column::Temperature);
            }
        }
        let grid = Grid::new(stepped, warm).map_err(|error| unreadable(&error))?;
        Ok(Self {
            varied: Varied::Grid(Arc::new(grid)),
            elaborated: Arc::clone(circuit),
            columns,
        })
    }

    /// `samples` Monte Carlo samples of the numbers `varied`, a JSON list of objects of an
    /// `instance`, its path, a number's `parameter` path, its `nominal` value and its
    /// `tolerance`, drawn from `seed` as [`crate::sample`] draws them.
    ///
    /// Raises `ElaborationError` naming every number the circuit's devices do not write.
    #[staticmethod]
    fn monte_carlo(
        py: Python<'_>,
        elaborated: &Elaborated,
        varied: &str,
        seed: u64,
        samples: u64,
    ) -> PyResult<Self> {
        let circuit = &elaborated.inner;
        let toleranced: Vec<Toleranced> =
            serde_json::from_str(varied).map_err(|error| unreadable(&error))?;
        let columns = toleranced
            .iter()
            .filter_map(|toleranced| {
                let parameter = circuit.parameter(&toleranced.instance, &toleranced.parameter)?;
                Some(Column::Parameter(
                    parameter,
                    toleranced.instance.clone(),
                    toleranced.parameter.clone(),
                ))
            })
            .collect();
        let drawn = MonteCarlo::new(circuit, seed, samples, toleranced)
            .map_err(|error| elaboration_error(py, &error))?;
        Ok(Self {
            varied: Varied::MonteCarlo(Arc::new(drawn), seed),
            elaborated: Arc::clone(circuit),
            columns,
        })
    }

    /// Every variant of `first` at every variant of `second`: run `i` is variant
    /// `i // len(second)` of `first`, then variant `i % len(second)` of `second`, whose
    /// values replace `first`'s where both give one.
    ///
    /// Raises `ValueError` when they vary different circuits, or there are more runs than
    /// 2**64 - 1.
    #[staticmethod]
    fn product(first: &Self, second: &Self) -> PyResult<Self> {
        if !Arc::ptr_eq(&first.elaborated, &second.elaborated) {
            return Err(PyValueError::new_err(
                "a product's variants vary one elaborated circuit",
            ));
        }
        let product = Product::new(first.varied.clone(), second.varied.clone())
            .map_err(|error| unreadable(&error))?;
        let mut columns = first.columns.clone();
        for column in &second.columns {
            if !columns.contains(column) {
                columns.push(column.clone());
            }
        }
        Ok(Self {
            varied: Varied::Product(Arc::new(product)),
            elaborated: Arc::clone(&first.elaborated),
            columns,
        })
    }

    /// How many runs there are.
    fn __len__(&self) -> PyResult<usize> {
        usize::try_from(self.varied.len())
            .map_err(|_| PyValueError::new_err("more runs than an index counts"))
    }

    /// The numbers the runs vary, in the order of a batch's columns of values: each a
    /// parameter, as its instance's path and its own, or `None` for the temperature.
    #[getter]
    fn parameters(&self) -> Vec<Option<(String, String)>> {
        self.columns
            .iter()
            .map(|column| match column {
                Column::Parameter(_, path, name) => Some((path.clone(), name.clone())),
                Column::Temperature => None,
            })
            .collect()
    }

    /// The numbers the run at `index` replaces, by instance path then by parameter path,
    /// and its temperature in kelvins, if it gives one.
    ///
    /// Raises `ValueError` for an index past the last run.
    fn overrides(&self, index: u64) -> PyResult<(Replaced, Option<f64>)> {
        if index >= self.varied.len() {
            return Err(PyValueError::new_err(format!(
                "run {index} lies past the {} runs",
                self.varied.len()
            )));
        }
        let values = self.varied.values(index);
        let mut replaced = Replaced::new();
        for column in &self.columns {
            if let Column::Parameter(parameter, path, name) = column
                && let Some(value) = values.get(*parameter)
            {
                replaced
                    .entry(path.clone())
                    .or_default()
                    .insert(name.clone(), value);
            }
        }
        Ok((replaced, values.temperature))
    }

    /// The Monte Carlo sample the run at `index` draws, as its seed and its index among the
    /// samples; `None` when it draws none.
    fn sample(&self, index: u64) -> Option<(u64, u64)> {
        (index < self.varied.len())
            .then(|| self.varied.sample(index))
            .flatten()
    }
}

/// The variants a batch runs: each run at the values of the simulator the batch is of,
/// with those of its variant in their place where it gives them.
struct Based<'a> {
    base: &'a Values,
    varied: &'a Varied,
}

impl jolty_engine::batch::Variants for Based<'_> {
    fn len(&self) -> u64 {
        self.varied.len()
    }

    fn values(&self, index: u64) -> Values {
        let mut values = self.base.clone();
        values.extend(&self.varied.values(index));
        values
    }

    fn chain(&self) -> u64 {
        self.varied.chain()
    }
}

/// Where an analysis's traced unknowns lie along it.
#[derive(Debug, Clone)]
enum Points {
    /// An operating point's one point.
    Point,
    /// A DC sweep's levels, of which there are so many.
    Swept(usize),
    /// A small-signal response's frequencies, of which there are so many.
    Frequencies(usize),
    /// The times of a transient or of a periodic steady state's period, its values
    /// interpolated linearly between the times it accepted, and so exact at a time it
    /// stepped onto.
    Times(Vec<f64>),
}

impl Points {
    /// The points of `analysis`, which a batch traces unknowns along.
    ///
    /// Raises `ValueError` for an analysis of no such points.
    fn from_analysis(analysis: &Analysis) -> PyResult<Self> {
        Ok(match analysis {
            Analysis::OperatingPoint => Self::Point,
            Analysis::DcSweep(sweep) => Self::Swept(sweep.values.len()),
            Analysis::Ac(response) => Self::Frequencies(response.frequencies.len()),
            Analysis::Transient(transient) => Self::Times(transient.breakpoints.clone()),
            Analysis::PeriodicSteadyState(steady) => Self::Times(steady.breakpoints.clone()),
            _ => {
                return Err(PyValueError::new_err(
                    "a batch traces unknowns along an operating point, a DC sweep, a \
                     small-signal response, or a transient's or a periodic steady state's times",
                ));
            }
        })
    }

    fn len(&self) -> usize {
        match self {
            Self::Point => 1,
            Self::Swept(count) | Self::Frequencies(count) => *count,
            Self::Times(times) => times.len(),
        }
    }
}

/// Some unknowns a batch traces along an analysis: the analysis, by its index, the unknowns,
/// by their indices among the batch circuit's, and its points.
struct Traced {
    analysis: usize,
    unknowns: Vec<usize>,
    points: Points,
}

impl Traced {
    /// How many values it traces of a run: its unknowns at each of its points.
    fn len(&self) -> usize {
        self.points.len() * self.unknowns.len()
    }

    /// The values it traces of a run that found none: not a number, each.
    fn missing(&self) -> Vec<Complex64> {
        vec![Complex64::new(f64::NAN, f64::NAN); self.len()]
    }
}

/// The buffer of a traced analysis: runs × points × unknowns, real or complex.
enum Buffer<'a> {
    Real(&'a mut [f64]),
    Complex(&'a mut [Complex64]),
}

/// Why a measure of a run read nothing.
#[derive(Debug, Clone, Copy)]
enum Cause {
    /// The run's values made problems, or the analysis it reads failed.
    Failed,
    /// The measure could not read the analysis's result.
    Measure(MeasureError),
}

/// What the sink has written, behind its lock: the arrays, one row per run, the causes
/// measures read nothing for, and why each run that failed failed, by its row.
struct Written<'a> {
    values: &'a mut [f64],
    measured: &'a mut [f64],
    causes: &'a mut [i32],
    ended: &'a mut [bool],
    traces: Vec<Buffer<'a>>,
    /// Each cause, in the order the runs that met it first ended.
    reasons: Vec<Cause>,
    /// Each cause's place among `reasons`, by its text.
    known: HashMap<String, usize>,
    problems: BTreeMap<u64, ElaborationError>,
    errors: BTreeMap<u64, Vec<(usize, String)>>,
}

/// A sink that writes each run's numbers into the arrays of a batch at the run's row, its
/// index less the `first` run's, and hands each run's results to a Python function, if
/// given, under the interpreter's lock.
struct Columns<'a> {
    first: u64,
    varied: &'a [Column],
    nominal: &'a [f64],
    temperature: f64,
    analyses: usize,
    measures: usize,
    traces: &'a [Traced],
    written: Mutex<Written<'a>>,
    each: Option<Each>,
}

/// A Python function a batch hands each run's results to, the simulators of the other
/// structures its runs met, and the first exception it raised.
struct Each {
    function: Py<PyAny>,
    views: Mutex<HashMap<usize, Py<Simulator>>>,
    raised: Mutex<Option<PyErr>>,
}

impl Columns<'_> {
    /// The value of `column` at `values`.
    fn value(&self, column: &Column, values: &Values) -> f64 {
        match column {
            Column::Parameter(parameter, _, _) => values
                .get(*parameter)
                .or_else(|| self.nominal.get(parameter.index()).copied())
                .unwrap_or(f64::NAN),
            Column::Temperature => values.temperature.unwrap_or(self.temperature),
        }
    }

    /// Hands `outcome`, of the run at `index`, to the Python function, if any, unless it
    /// raised already; keeps the first exception it raises.
    fn hand(&self, index: u64, outcome: Run) {
        let Some(each) = &self.each else {
            return;
        };
        if each
            .raised
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .is_some()
        {
            return;
        }
        Python::attach(|py| {
            if let Err(error) = each.call(py, index, outcome) {
                each.raised
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .get_or_insert(error);
            }
        });
    }
}

impl Each {
    /// Calls the function with the run's index, the simulator of its structure when it is
    /// not the batch's, the problems its values made, or each analysis's outcome or why it
    /// failed.
    fn call(&self, py: Python<'_>, index: u64, outcome: Run) -> PyResult<()> {
        let structure = match &outcome.structure {
            None => None,
            Some(structure) => {
                let key = Arc::as_ptr(structure.elaborated()) as usize;
                let mut views = self.views.lock().unwrap_or_else(PoisonError::into_inner);
                if let Some(view) = views.get(&key) {
                    Some(view.clone_ref(py))
                } else {
                    let made = jolty_engine::Simulator::from_elaborated(
                        Arc::clone(structure.elaborated()),
                        outcome.values.clone(),
                    )
                    .map_err(|error| elaboration_error(py, &error))?;
                    let view = Py::new(py, Simulator::wrapping(made))?;
                    views.insert(key, view.clone_ref(py));
                    Some(view)
                }
            }
        };
        let problems = outcome
            .problems
            .as_ref()
            .map(|problems| crate::problems_to_python(py, problems))
            .transpose()?;
        let found = PyList::empty(py);
        for result in outcome.results {
            match result {
                Ok(result) => found.append(crate::outcome_to_python(py, result)?)?,
                Err(error) => found.append(error.to_string())?,
            }
        }
        self.function
            .call1(py, (index, structure, problems, found))
            .map(drop)
    }
}

impl Sink for Columns<'_> {
    fn accept(&self, index: u64, outcome: Run) {
        let Ok(row) = usize::try_from(index - self.first) else {
            return;
        };
        let values: Vec<f64> = self
            .varied
            .iter()
            .map(|column| self.value(column, &outcome.values))
            .collect();
        let traces: Vec<Vec<Complex64>> = self
            .traces
            .iter()
            .map(|traces| trace(&outcome, traces))
            .collect();
        let ended: Vec<bool> = if outcome.problems.is_some() {
            vec![false; self.analyses]
        } else {
            outcome.results.iter().map(Result::is_ok).collect()
        };
        let errors: Vec<(usize, String)> = outcome
            .results
            .iter()
            .enumerate()
            .filter_map(|(analysis, result)| {
                result
                    .as_ref()
                    .err()
                    .map(|error| (analysis, error.to_string()))
            })
            .collect();
        {
            let mut written = self.written.lock().unwrap_or_else(PoisonError::into_inner);
            let written = &mut *written;
            let width = self.varied.len();
            written.values[row * width..(row + 1) * width].copy_from_slice(&values);
            written.ended[row * self.analyses..(row + 1) * self.analyses].copy_from_slice(&ended);
            for (place, measured) in outcome.measured.iter().enumerate() {
                let cell = row * self.measures + place;
                match measured {
                    Ok(value) => written.measured[cell] = *value,
                    Err(unmeasured) => {
                        let cause = match unmeasured {
                            Unmeasured::Failed => Cause::Failed,
                            Unmeasured::Measure(error) => Cause::Measure(*error),
                        };
                        let text = format!("{cause:?}");
                        let next = written.reasons.len();
                        let known = *written.known.entry(text).or_insert(next);
                        if known == next {
                            written.reasons.push(cause);
                        }
                        written.causes[cell] = i32::try_from(known).unwrap_or(i32::MAX);
                    }
                }
            }
            for ((buffer, traces), found) in written.traces.iter_mut().zip(self.traces).zip(traces)
            {
                let stride = traces.len();
                let range = row * stride..(row + 1) * stride;
                match buffer {
                    Buffer::Real(buffer) => {
                        for (slot, value) in buffer[range].iter_mut().zip(found) {
                            *slot = value.re;
                        }
                    }
                    Buffer::Complex(buffer) => buffer[range].copy_from_slice(&found),
                }
            }
            if let Some(problems) = &outcome.problems {
                written
                    .problems
                    .insert(index - self.first, problems.clone());
            } else if !errors.is_empty() {
                written.errors.insert(index - self.first, errors);
            }
        }
        self.hand(index, outcome);
    }
}

/// The values of the unknowns `traces` names at each of its points in `outcome`, points
/// first, as complex numbers: not a number where the run did not end its analysis or has
/// no such unknown.
fn trace(outcome: &Run, traces: &Traced) -> Vec<Complex64> {
    match outcome.results.get(traces.analysis) {
        Some(Ok(result)) => trace_result(result, |unknown| outcome.column(unknown), traces),
        _ => traces.missing(),
    }
}

/// The values of the unknowns `traces` names at each of its points in `result`, points
/// first, as complex numbers, each read at `column` of the batch circuit's unknown: not a
/// number where the run has no such unknown.
fn trace_result(
    result: &jolty_engine::AnalysisResult,
    column: impl Fn(usize) -> Option<usize>,
    traces: &Traced,
) -> Vec<Complex64> {
    let mut found = traces.missing();
    let columns: Vec<Option<usize>> = traces
        .unknowns
        .iter()
        .map(|&unknown| column(unknown))
        .collect();
    let mut put = |point: usize, row: &dyn Fn(usize) -> Option<Complex64>| {
        for (place, column) in columns.iter().enumerate() {
            if let Some(value) = column.and_then(row) {
                found[point * columns.len() + place] = value;
            }
        }
    };
    let real = |values: &[f64], column: usize| values.get(column).map(|&value| value.into());
    match (result, &traces.points) {
        (jolty_engine::AnalysisResult::OperatingPoint(point), Points::Point) => {
            put(0, &|column| real(&point.values, column));
        }
        (jolty_engine::AnalysisResult::DcSweep(sweep), Points::Swept(_)) => {
            for (level, values) in sweep.values.iter().enumerate() {
                put(level, &|column| real(values, column));
            }
        }
        (jolty_engine::AnalysisResult::Ac(response), Points::Frequencies(_)) => {
            for (frequency, values) in response.values.iter().enumerate() {
                put(frequency, &|column| values.get(column).copied());
            }
        }
        (
            jolty_engine::AnalysisResult::Transient(waveforms)
            | jolty_engine::AnalysisResult::PeriodicSteadyState(
                jolty_engine::PeriodicSteadyStateResult { waveforms, .. },
            ),
            Points::Times(times),
        ) => {
            for (point, &time) in times.iter().enumerate() {
                put(point, &|column| {
                    at(&waveforms.times, &waveforms.values, column, time).map(Complex64::from)
                });
            }
        }
        _ => {}
    }
    found
}

/// Where `time` lies among the accepted `times`: at one of them, or between two, the
/// fraction of the way from the first to the second; none outside them.
enum Located {
    At(usize),
    Between(usize, usize, f64),
}

/// Where `time` lies among the accepted `times`, as [`Located`] says.
fn located(times: &[f64], time: f64) -> Option<Located> {
    match times.binary_search_by(|accepted| accepted.total_cmp(&time)) {
        Ok(found) => Some(Located::At(found)),
        Err(after) => {
            let before = after.checked_sub(1)?;
            let (start, stop) = (*times.get(before)?, *times.get(after)?);
            Some(Located::Between(
                before,
                after,
                (time - start) / (stop - start),
            ))
        }
    }
}

/// The value of the unknown at `column` at `time`: its value at a time accepted there, and
/// else interpolated linearly between the accepted times on either side; none outside
/// them.
fn at(times: &[f64], values: &[Vec<f64>], column: usize, time: f64) -> Option<f64> {
    match located(times, time)? {
        Located::At(found) => values.get(found)?.get(column).copied(),
        Located::Between(before, after, fraction) => {
            let low = *values.get(before)?.get(column)?;
            let high = *values.get(after)?.get(column)?;
            Some(low + (high - low) * fraction)
        }
    }
}

/// A new array of `len` elements, each `value`, for a sink to write into.
fn filled<T: numpy::Element + Copy>(
    py: Python<'_>,
    len: usize,
    value: T,
) -> Bound<'_, PyArray1<T>> {
    PyArray1::from_vec(py, vec![value; len])
}

/// The slice of every element of `array`, written in place.
fn slice<'a, T: numpy::Element>(array: &'a mut PyReadwriteArray1<'_, T>) -> PyResult<&'a mut [T]> {
    array
        .as_slice_mut()
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// A batch as Python asks for it: the analyses each run runs, in order, each the JSON
/// texts of an analysis and its options; the runs it runs, by their indices among the
/// variants'; the measures it reads, each the index of the analysis it reads and the
/// measure; what it traces, each the index of an analysis and the indices of the unknowns
/// it traces; and the function each run's results are handed to.
pub(crate) struct Request {
    pub(crate) analyses: Vec<(String, String)>,
    pub(crate) runs: Range<u64>,
    pub(crate) measures: Vec<(usize, Py<measure::Measure>)>,
    pub(crate) traces: Vec<(usize, Vec<usize>)>,
    pub(crate) each: Option<Py<PyAny>>,
}

/// The arrays a batch writes into, one row per run, before it runs.
struct Arrays<'py> {
    values: Bound<'py, PyArray1<f64>>,
    measured: Bound<'py, PyArray1<f64>>,
    causes: Bound<'py, PyArray1<i32>>,
    ended: Bound<'py, PyArray1<bool>>,
    /// Each traced analysis's values, real or complex, in the order of the traced.
    traces: Vec<Traces<'py>>,
}

/// A traced analysis's array: complex for a small-signal response's phasors.
enum Traces<'py> {
    Real(Bound<'py, PyArray1<f64>>),
    Complex(Bound<'py, PyArray1<Complex64>>),
}

/// What a batch runs, read from what Python asked for: the batch, the runs it runs, what
/// it traces, and how many analyses and measures it has.
struct Prepared {
    batch: Batch,
    runs: Range<u64>,
    traces: Vec<Traced>,
    analyses: usize,
    measures: usize,
}

/// The batch of runs of `elaborated` that `request` asks for.
fn prepare(elaborated: &Arc<jolty_engine::Elaborated>, request: &Request) -> PyResult<Prepared> {
    let analyses: Vec<(Analysis, Options)> = request
        .analyses
        .iter()
        .map(|(analysis, options)| {
            let read = serde_json::from_str(analysis).map_err(|error| unreadable(&error))?;
            let options = serde_json::from_str(options).map_err(|error| unreadable(&error))?;
            Ok((read, options))
        })
        .collect::<PyResult<_>>()?;
    let traces: Vec<Traced> = request
        .traces
        .iter()
        .map(|(analysis, unknowns)| {
            let (found, _) = analyses
                .get(*analysis)
                .ok_or_else(|| PyValueError::new_err(format!("no analysis {analysis}")))?;
            Ok(Traced {
                analysis: *analysis,
                unknowns: unknowns.clone(),
                points: Points::from_analysis(found)?,
            })
        })
        .collect::<PyResult<_>>()?;
    let reads: Vec<Measured> = request
        .measures
        .iter()
        .map(|(analysis, measure)| Measured {
            analysis: *analysis,
            measure: measure.get().0,
        })
        .collect();
    let (count, width) = (analyses.len(), reads.len());
    let batch =
        Batch::new(Arc::clone(elaborated), analyses, reads).map_err(|error| unreadable(&error))?;
    Ok(Prepared {
        batch,
        runs: request.runs.clone(),
        traces,
        analyses: count,
        measures: width,
    })
}

impl<'py> Arrays<'py> {
    /// The arrays of `runs` runs of `prepared`, varying `columns` numbers, each element not
    /// a number, or no cause, or an analysis not ended, until a run writes it.
    fn new(py: Python<'py>, runs: usize, columns: usize, prepared: &Prepared) -> Self {
        let traces = prepared
            .traces
            .iter()
            .map(|traces| {
                let len = runs * traces.len();
                if matches!(traces.points, Points::Frequencies(_)) {
                    Traces::Complex(filled(py, len, Complex64::new(f64::NAN, f64::NAN)))
                } else {
                    Traces::Real(filled(py, len, f64::NAN))
                }
            })
            .collect();
        Self {
            values: filled(py, runs * columns, f64::NAN),
            measured: filled(py, runs * prepared.measures, f64::NAN),
            causes: filled(py, runs * prepared.measures, -1),
            ended: filled(py, runs * prepared.analyses, false),
            traces,
        }
    }

    /// The arrays as `jolty._core.Simulator.batch` returns them, shaped, in `found`.
    fn shaped(
        self,
        found: &Bound<'py, PyDict>,
        runs: usize,
        columns: usize,
        prepared: &Prepared,
    ) -> PyResult<()> {
        found.set_item("values", self.values.reshape([runs, columns])?)?;
        found.set_item(
            "measured",
            self.measured.reshape([runs, prepared.measures])?,
        )?;
        found.set_item("causes", self.causes.reshape([runs, prepared.measures])?)?;
        found.set_item("ended", self.ended.reshape([runs, prepared.analyses])?)?;
        let traces = PyList::empty(found.py());
        for (array, each) in self.traces.into_iter().zip(&prepared.traces) {
            let shape = [runs, each.points.len(), each.unknowns.len()];
            match array {
                Traces::Real(array) => traces.append(array.reshape(shape)?)?,
                Traces::Complex(array) => traces.append(array.reshape(shape)?)?,
            }
        }
        found.set_item("traces", traces)
    }
}

/// What a batch of runs of `elaborated` at `base` found, as `jolty._core.Simulator.batch`
/// returns it.
pub(crate) fn run<'py>(
    py: Python<'py>,
    (elaborated, base): (&Arc<jolty_engine::Elaborated>, &Values),
    variants: &Variants,
    request: Request,
    pool: &Pool,
) -> PyResult<Bound<'py, PyDict>> {
    if !Arc::ptr_eq(elaborated, &variants.elaborated) {
        return Err(PyValueError::new_err(
            "the variants vary another elaborated circuit than the simulator's",
        ));
    }
    let prepared = prepare(elaborated, &request)?;
    let based = Based {
        base,
        varied: &variants.varied,
    };
    let Range { start, end } = prepared.runs;
    if start > end || end > based.len() {
        return Err(PyValueError::new_err(format!(
            "the runs from {start} up to {end} do not lie within the {} runs of the variants",
            based.len()
        )));
    }
    let runs = usize::try_from(end - start)
        .map_err(|_| PyValueError::new_err("more runs than an array holds"))?;
    let columns = &variants.columns;
    let arrays = Arrays::new(py, runs, columns.len(), &prepared);
    let written = write(
        py,
        elaborated,
        (&based, columns),
        &prepared,
        &arrays,
        request.each,
        pool,
    )?;
    let found = PyDict::new(py);
    arrays.shaped(&found, runs, columns.len(), &prepared)?;
    written.into_python(&found)?;
    Ok(found)
}

/// What a batch found besides its arrays: why measures read nothing, and why runs failed.
struct Found {
    reasons: Vec<Cause>,
    problems: BTreeMap<u64, ElaborationError>,
    errors: BTreeMap<u64, Vec<(usize, String)>>,
}

impl Found {
    /// Adds what it holds to `found`: each reason as `(name, numbers, message)`, the
    /// problems by row, and each failed analysis's index and error by row.
    fn into_python(self, found: &Bound<'_, PyDict>) -> PyResult<()> {
        let py = found.py();
        let told = PyList::empty(py);
        for cause in self.reasons {
            match cause {
                Cause::Failed => told.append(("failed", Vec::<f64>::new(), String::new()))?,
                Cause::Measure(error) => {
                    let (reason, numbers) = measure::reason(error);
                    told.append((reason, numbers, error.to_string()))?;
                }
            }
        }
        found.set_item("reasons", told)?;
        let problems = PyDict::new(py);
        for (index, error) in &self.problems {
            problems.set_item(index, crate::problems_to_python(py, error)?)?;
        }
        found.set_item("problems", problems)?;
        let errors = PyDict::new(py);
        for (index, failed) in self.errors {
            errors.set_item(index, failed)?;
        }
        found.set_item("errors", errors)
    }
}

/// Runs `prepared` over the runs of `variants`, on the threads of `pool`, the interpreter
/// released, writing each run into `arrays` and handing it to `each`, if given; raises
/// what `each` raised first.
fn write(
    py: Python<'_>,
    elaborated: &jolty_engine::Elaborated,
    (variants, columns): (&Based<'_>, &[Column]),
    prepared: &Prepared,
    arrays: &Arrays<'_>,
    each: Option<Py<PyAny>>,
    pool: &Pool,
) -> PyResult<Found> {
    let mut values = arrays.values.readwrite();
    let mut measured = arrays.measured.readwrite();
    let mut causes = arrays.causes.readwrite();
    let mut ended = arrays.ended.readwrite();
    let mut real = Vec::new();
    let mut complex = Vec::new();
    for traces in &arrays.traces {
        match traces {
            Traces::Real(array) => real.push(array.readwrite()),
            Traces::Complex(array) => complex.push(array.readwrite()),
        }
    }
    let mut buffers = Vec::with_capacity(arrays.traces.len());
    let (mut reals, mut complexes) = (real.iter_mut(), complex.iter_mut());
    for traces in &arrays.traces {
        let buffer = match traces {
            Traces::Real(_) => reals.next().map(|array| slice(array).map(Buffer::Real)),
            Traces::Complex(_) => complexes
                .next()
                .map(|array| slice(array).map(Buffer::Complex)),
        };
        if let Some(buffer) = buffer {
            buffers.push(buffer?);
        }
    }
    let sink = Columns {
        first: prepared.runs.start,
        varied: columns,
        nominal: elaborated.nominal(),
        temperature: elaborated.temperature(),
        analyses: prepared.analyses,
        measures: prepared.measures,
        traces: &prepared.traces,
        written: Mutex::new(Written {
            values: slice(&mut values)?,
            measured: slice(&mut measured)?,
            causes: slice(&mut causes)?,
            ended: slice(&mut ended)?,
            traces: buffers,
            reasons: Vec::new(),
            known: HashMap::new(),
            problems: BTreeMap::new(),
            errors: BTreeMap::new(),
        }),
        each: each.map(|function| Each {
            function,
            views: Mutex::new(HashMap::new()),
            raised: Mutex::new(None),
        }),
    };
    let ran = py.detach(|| {
        pool.inner
            .install(|| prepared.batch.run(variants, prepared.runs.clone(), &sink))
    });
    ran.map_err(|error| unreadable(&error))?;
    let Columns { written, each, .. } = sink;
    if let Some(error) = each.and_then(|each| {
        each.raised
            .into_inner()
            .unwrap_or_else(PoisonError::into_inner)
    }) {
        return Err(error);
    }
    let Written {
        causes: cells,
        reasons,
        problems,
        errors,
        ..
    } = written.into_inner().unwrap_or_else(PoisonError::into_inner);
    Ok(Found {
        reasons: canonical(cells, &reasons),
        problems,
        errors,
    })
}

/// The `reasons` renumbered in the order of the cells of `causes` that first name each, row
/// by row, so that the numbers do not depend on the order the runs ended in; `causes`
/// renumbered with them.
fn canonical(causes: &mut [i32], reasons: &[Cause]) -> Vec<Cause> {
    let mut renumbered: Vec<Option<i32>> = vec![None; reasons.len()];
    let mut ordered = Vec::with_capacity(reasons.len());
    for cell in causes.iter_mut() {
        let Ok(old) = usize::try_from(*cell) else {
            continue;
        };
        let new = *renumbered[old].get_or_insert_with(|| {
            ordered.push(reasons[old]);
            i32::try_from(ordered.len() - 1).unwrap_or(i32::MAX)
        });
        *cell = new;
    }
    ordered
}

/// A batch's VJP as Python asks for it: the analysis differentiated and its options, as
/// JSON texts, the parameters it differentiates with respect to, each its instance's path
/// and its name, the runs it runs, by their indices among the variants', and the unknowns
/// it traces, by their indices among the batch circuit's.
pub(crate) struct Differentiating {
    pub(crate) analysis: String,
    pub(crate) options: String,
    pub(crate) parameters: Vec<(String, String)>,
    pub(crate) runs: Range<u64>,
    pub(crate) unknowns: Vec<usize>,
}

/// What a run of a batch's VJP found: its pullback, or why its analysis failed.
type Derived = Mapped<Result<jolty_engine::Pullback, jolty_engine::SimulationError>>;

/// The pullbacks of a batch's VJP, a run each, as `jolty._core.Simulator.batch_vjp` finds
/// them: what carries a cotangent on each run's traced unknowns back to the parameters.
#[pyclass(module = "jolty._core", frozen)]
pub(crate) struct BatchPullback {
    batch: Batch,
    traced: Traced,
    /// The batch circuit's unknowns, by name, which a cotangent's messages name.
    names: Vec<String>,
    parameters: usize,
    runs: Vec<Derived>,
}

/// The VJP of every run `request` asks for of `elaborated` at `base`, as
/// `jolty._core.Simulator.batch_vjp` returns it: what the runs found, as [`outcomes`]
/// gives it, and their pullbacks.
pub(crate) fn vjp<'py>(
    py: Python<'py>,
    (elaborated, base): (&Arc<jolty_engine::Elaborated>, &Values),
    variants: &Variants,
    request: Differentiating,
    pool: &Pool,
) -> PyResult<(Bound<'py, PyDict>, BatchPullback)> {
    if !Arc::ptr_eq(elaborated, &variants.elaborated) {
        return Err(PyValueError::new_err(
            "the variants vary another elaborated circuit than the simulator's",
        ));
    }
    let analysis: Analysis =
        serde_json::from_str(&request.analysis).map_err(|error| unreadable(&error))?;
    let differentiated: Differentiated =
        serde_json::from_str(&request.analysis).map_err(|error| unreadable(&error))?;
    let options: Options =
        serde_json::from_str(&request.options).map_err(|error| unreadable(&error))?;
    let traced = Traced {
        analysis: 0,
        unknowns: request.unknowns,
        points: Points::from_analysis(&analysis)?,
    };
    let names: Vec<String> = elaborated
        .unknowns()
        .iter()
        .map(|unknown| unknown.name.clone())
        .collect();
    if let Some(unknown) = traced
        .unknowns
        .iter()
        .find(|&&unknown| unknown >= names.len())
    {
        return Err(PyValueError::new_err(format!(
            "the circuit has {} unknowns, not one at {unknown}",
            names.len()
        )));
    }
    let based = Based {
        base,
        varied: &variants.varied,
    };
    let batch = Batch::new(Arc::clone(elaborated), Vec::new(), Vec::new())
        .map_err(|error| unreadable(&error))?;
    let parameters = request.parameters;
    let runs = py
        .detach(|| {
            pool.inner.install(|| {
                batch.map(
                    &based,
                    request.runs.clone(),
                    &options,
                    &|_, simulator, options| simulator.vjp(&differentiated, &parameters, options),
                )
            })
        })
        .map_err(|error| unreadable(&error))?;
    let found = outcomes(py, &runs, &traced)?;
    let pullback = BatchPullback {
        batch,
        traced,
        names,
        parameters: parameters.len(),
        runs,
    };
    Ok((found, pullback))
}

/// What the runs `runs` of a batch's VJP found, as `jolty._core.Simulator.batch_vjp`
/// returns it: the `traces` `traced` names, runs × points × unknowns, each oscillator's
/// `periods`, not a number for another analysis or a failed run, the `problems` by row
/// and the `errors` by row.
fn outcomes<'py>(
    py: Python<'py>,
    runs: &[Derived],
    traced: &Traced,
) -> PyResult<Bound<'py, PyDict>> {
    let found = PyDict::new(py);
    let complex = matches!(traced.points, Points::Frequencies(_));
    let mut real_traces = Vec::new();
    let mut complex_traces = Vec::new();
    let mut periods = vec![f64::NAN; runs.len()];
    let problems = PyDict::new(py);
    let errors = PyDict::new(py);
    for (row, run) in runs.iter().enumerate() {
        let values = match &run.found {
            Ok(Ok(pullback)) => {
                if let jolty_engine::AnalysisResult::PeriodicSteadyState(steady) = &pullback.result
                {
                    periods[row] = steady.period;
                }
                trace_result(&pullback.result, |unknown| run.column(unknown), traced)
            }
            Ok(Err(error)) => {
                errors.set_item(row, error.to_string())?;
                traced.missing()
            }
            Err(error) => {
                problems.set_item(row, crate::problems_to_python(py, error)?)?;
                traced.missing()
            }
        };
        if complex {
            complex_traces.extend(values);
        } else {
            real_traces.extend(values.iter().map(|value| value.re));
        }
    }
    let shape = [runs.len(), traced.points.len(), traced.unknowns.len()];
    if complex {
        found.set_item(
            "traces",
            PyArray1::from_vec(py, complex_traces).reshape(shape)?,
        )?;
    } else {
        found.set_item(
            "traces",
            PyArray1::from_vec(py, real_traces).reshape(shape)?,
        )?;
    }
    found.set_item("periods", PyArray1::from_vec(py, periods))?;
    found.set_item("problems", problems)?;
    found.set_item("errors", errors)?;
    Ok(found)
}

/// A number a cotangent is made of: a real number on values, a complex one on phasors.
trait Cotangential:
    Copy
    + Default
    + PartialEq
    + Send
    + Sync
    + std::ops::AddAssign
    + std::ops::Mul<f64, Output = Self>
    + numpy::Element
{
    /// The cotangents of `each`, of this kind of number.
    fn cotangents(each: Vec<Cotangent<Self>>) -> Cotangents;

    /// Whether the number is not a number, which moves a loss as much as any other.
    fn is_nan(self) -> bool;
}

impl Cotangential for f64 {
    fn cotangents(each: Vec<Cotangent<Self>>) -> Cotangents {
        Cotangents::Real(each)
    }

    fn is_nan(self) -> bool {
        f64::is_nan(self)
    }
}

impl Cotangential for Complex64 {
    fn cotangents(each: Vec<Cotangent<Self>>) -> Cotangents {
        Cotangents::Complex(each)
    }

    fn is_nan(self) -> bool {
        Complex64::is_nan(self)
    }
}

/// A batch's pull back as Python asks for it: the runs, a row each, any run named by any
/// number of rows, and each row's cotangents, on the traced unknowns, by point then
/// unknown, and on the run's period.
struct Pulling<'py, T: Cotangential> {
    runs: Vec<usize>,
    values: numpy::PyReadonlyArray3<'py, T>,
    periods: Vec<f64>,
}

/// What a run's pullback finds of a cotangent: its value on each parameter.
type Pulled = Result<Vec<Complex64>, String>;

#[pymethods]
impl BatchPullback {
    /// How many runs the batch ran, a pullback each.
    #[getter]
    fn runs(&self) -> usize {
        self.runs.len()
    }

    /// The value on each parameter of each cotangent of `values`, runs × points × unknowns,
    /// on the unknowns traced at each point of the run at `runs` of the same row, as the
    /// traces hold them, real or complex, and of the row of `periods` on an oscillator's
    /// period, with the options whose JSON text is `options`: rows × parameters, complex
    /// over frequency, each row pulled back on the threads of `pool`, the interpreter
    /// released, as a batch runs its runs. A row may name a run more than once.
    ///
    /// Raises `ValueError` for cotangents of another shape, and `SimulationError` for a
    /// run that failed, a cotangent on an unknown its run does not have, or a pullback
    /// that fails.
    fn pull_back<'py>(
        &self,
        py: Python<'py>,
        runs: Vec<usize>,
        values: &Bound<'py, PyAny>,
        periods: Vec<f64>,
        options: &str,
        pool: &Pool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let options: Options = serde_json::from_str(options).map_err(|error| unreadable(&error))?;
        if let Ok(real) = values.extract::<numpy::PyReadonlyArray3<'py, f64>>() {
            let pulling = Pulling {
                runs,
                values: real,
                periods,
            };
            self.pulled(py, &pulling, &options, pool)
        } else {
            let pulling = Pulling {
                runs,
                values: values.extract::<numpy::PyReadonlyArray3<'py, Complex64>>()?,
                periods,
            };
            self.pulled(py, &pulling, &options, pool)
        }
    }
}

impl BatchPullback {
    /// The rows `pull_back` finds of `pulling`.
    fn pulled<'py, T: Cotangential>(
        &self,
        py: Python<'py>,
        pulling: &Pulling<'py, T>,
        options: &Options,
        pool: &Pool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let Pulling {
            runs,
            values,
            periods,
        } = pulling;
        let given = values.as_array();
        let expected = [
            runs.len(),
            self.traced.points.len(),
            self.traced.unknowns.len(),
        ];
        if given.shape() != expected || periods.len() != runs.len() {
            return Err(PyValueError::new_err(format!(
                "cotangents are runs × points × unknowns, {expected:?}, with a period each, \
                 not {:?} with {} periods",
                given.shape(),
                periods.len()
            )));
        }
        for &run in runs {
            let Some(derived) = self.runs.get(run) else {
                return Err(PyValueError::new_err(format!(
                    "the batch ran {} runs, not one at {run}",
                    self.runs.len()
                )));
            };
            succeeded(run, derived.found.as_ref().map(Result::as_ref))?;
        }
        let rows: Vec<Vec<T>> = given
            .outer_iter()
            .map(|row| row.iter().copied().collect())
            .collect();
        let selected = Selected {
            values: runs
                .iter()
                .map(|&run| self.runs[run].values.clone())
                .collect(),
        };
        let found = py.detach(|| {
            pool.inner.install(|| {
                self.batch.map(
                    &selected,
                    0..selected.values.len() as u64,
                    options,
                    &|index, simulator, options| -> Pulled {
                        let row = usize::try_from(index).map_err(|error| error.to_string())?;
                        let derived = &self.runs[runs[row]];
                        let Ok(Ok(pullback)) = &derived.found else {
                            return Err(format!("batch run {} has no pullback", runs[row]));
                        };
                        let width = simulator.unknowns().len();
                        let cotangent =
                            self.cotangent(derived, pullback, &rows[row], periods[row], width)?;
                        let pulled = simulator
                            .pull_back(pullback, &T::cotangents(vec![cotangent]), options)
                            .map_err(|error| error.to_string())?;
                        Ok(match pulled {
                            ParameterCotangents::Real(mut found) => found
                                .pop()
                                .unwrap_or_default()
                                .into_iter()
                                .map(Complex64::from)
                                .collect(),
                            ParameterCotangents::Complex(mut found) => {
                                found.pop().unwrap_or_default()
                            }
                        })
                    },
                )
            })
        });
        let found = found.map_err(|error| unreadable(&error))?;
        let mut pulled = Vec::with_capacity(runs.len() * self.parameters);
        for (&run, mapped) in runs.iter().zip(found) {
            pulled.extend(succeeded(run, mapped.found)?);
        }
        let shape = [runs.len(), self.parameters];
        if matches!(self.traced.points, Points::Frequencies(_)) {
            Ok(PyArray1::from_vec(py, pulled).reshape(shape)?.into_any())
        } else {
            let real: Vec<f64> = pulled.iter().map(|value| value.re).collect();
            Ok(PyArray1::from_vec(py, real).reshape(shape)?.into_any())
        }
    }

    /// The cotangent on the run `derived`, whose `pullback` it pulls back, of `values`, on
    /// the traced unknowns at each traced point, by point then unknown, and `period`: on
    /// each of the run's `width` unknowns at each of its own points, a transient's values
    /// at each traced time carried back onto the times either side of it as `at` reads
    /// them, the transpose of its interpolation.
    fn cotangent<T: Cotangential>(
        &self,
        derived: &Derived,
        pullback: &jolty_engine::Pullback,
        values: &[T],
        period: f64,
        width: usize,
    ) -> Result<Cotangent<T>, String> {
        let unknowns = &self.traced.unknowns;
        let mut points: BTreeMap<usize, Vec<T>> = BTreeMap::new();
        let mut add = |point: usize, column: usize, value: T| {
            points
                .entry(point)
                .or_insert_with(|| vec![T::default(); width])[column] += value;
        };
        let times = match &pullback.result {
            jolty_engine::AnalysisResult::Transient(waveforms)
            | jolty_engine::AnalysisResult::PeriodicSteadyState(
                jolty_engine::PeriodicSteadyStateResult { waveforms, .. },
            ) => Some(waveforms.times.as_slice()),
            _ => None,
        };
        for (point, row) in values.chunks(unknowns.len().max(1)).enumerate() {
            for (&unknown, &value) in unknowns.iter().zip(row) {
                if value == T::default() && !value.is_nan() {
                    continue;
                }
                let Some(column) = derived.column(unknown) else {
                    return Err(format!(
                        "the run has no unknown `{}`, which a cotangent moves",
                        self.names[unknown]
                    ));
                };
                match (&self.traced.points, times) {
                    (Points::Times(traced), Some(times)) => match located(times, traced[point]) {
                        Some(Located::At(found)) => add(found, column, value),
                        Some(Located::Between(before, after, fraction)) => {
                            add(before, column, value * (1.0 - fraction));
                            add(after, column, value * fraction);
                        }
                        None => {
                            return Err(format!(
                                "the time {} lies outside the run's times",
                                traced[point]
                            ));
                        }
                    },
                    _ => add(point, column, value),
                }
            }
        }
        points.retain(|_, row| row.iter().any(|value| *value != T::default()));
        Ok(Cotangent { points, period })
    }
}

/// What the batch's run at `run` found, or, as a `SimulationError` naming it, why it
/// failed: its numbers' problems or its analysis's error.
fn succeeded<T, E: std::fmt::Display>(
    run: usize,
    found: Result<Result<T, E>, impl std::fmt::Display>,
) -> PyResult<T> {
    let error = match found {
        Ok(Ok(value)) => return Ok(value),
        Ok(Err(error)) => error.to_string(),
        Err(error) => error.to_string(),
    };
    Err(crate::SimulationError::new_err(format!(
        "batch run {run}: {error}"
    )))
}

/// The values of runs a pullback carries cotangents back on, a run each, in the order of
/// the rows.
struct Selected {
    values: Vec<Values>,
}

impl jolty_engine::batch::Variants for Selected {
    fn len(&self) -> u64 {
        self.values.len() as u64
    }

    fn values(&self, index: u64) -> Values {
        usize::try_from(index)
            .ok()
            .and_then(|index| self.values.get(index))
            .cloned()
            .unwrap_or_default()
    }
}
