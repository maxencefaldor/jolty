//! Batch executor backing `jolty._core.Simulator.batch`, writing each run directly into
//! its row of preallocated `NumPy` arrays.
//!
//! Output arrays store one row per requested run: varied parameter values, measure values
//! and failure causes, per-analysis completion flags, and traced unknowns along each
//! analysis axis. Each row index equals `run_index - range.start`, independent of worker
//! completion order. An optional Python callback receives each completed run under the
//! interpreter lock.

mod vjp;

use std::collections::{BTreeMap, HashMap};
use std::ops::Range;
use std::sync::{Arc, Mutex};

use jolty_engine::batch::{Batch, Measured, Run, Sink, Unmeasured, Variants as _};
use jolty_engine::measure::{MeasureError, Point, SumError};
use jolty_engine::{
    Analysis, AnalysisResult, Differentiated, ElaborationError, Options, PeriodicSteadyStateResult,
    Values,
};
use numpy::{Complex64, PyArray1, PyArrayMethods, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::convert::{problems_to_python, py_dict, result_to_python};
use crate::helpers::{
    elaboration_result, filled, into_inner_unpoisoned, lock_unpoisoned, parse_json, slice_mut,
    value_result,
};
use crate::variants::{Based, Column, Variants};
use crate::{Pool, Simulator, measure};

pub(crate) use vjp::{BatchPullback, Differentiating, vjp};

/// Traced values for a single run, allocating `f64` for real analyses and `Complex64` only
/// for AC analyses.
pub(crate) enum TracedValues {
    Real(Vec<f64>),
    Complex(Vec<Complex64>),
}

/// Mutable slice buffer of a traced analysis: runs x points x unknowns.
enum Buffer<'a> {
    Real(&'a mut [f64]),
    Complex(&'a mut [Complex64]),
}

/// Exclusive read-write borrow of a traced `NumPy` array during batch execution.
enum TraceGuard<'py> {
    Real(PyReadwriteArray1<'py, f64>),
    Complex(PyReadwriteArray1<'py, Complex64>),
}

/// A traced analysis's `NumPy` array: complex for a small-signal response's phasors.
enum Traces<'py> {
    Real(Bound<'py, PyArray1<f64>>),
    Complex(Bound<'py, PyArray1<Complex64>>),
}

impl TraceGuard<'_> {
    fn as_buffer(&mut self) -> PyResult<Buffer<'_>> {
        match self {
            Self::Real(array) => slice_mut(array).map(Buffer::Real),
            Self::Complex(array) => slice_mut(array).map(Buffer::Complex),
        }
    }
}

/// Sample axis points along which a batch traces unknowns or noise outputs.
#[derive(Debug, Clone)]
pub(crate) enum Points {
    Point,
    Swept(usize),
    Frequencies(usize),
    Noise(usize),
    Times(Vec<f64>),
}

impl Points {
    /// Extracts the traceable sample points of `analysis`, raising `ValueError` for
    /// unsupported analyses.
    pub(crate) fn from_analysis(analysis: &Analysis) -> PyResult<Self> {
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

    /// Extracts the traceable sample points of a differentiated `analysis` for batch VJP.
    pub(crate) fn from_differentiated(analysis: &Differentiated) -> Self {
        match analysis {
            Differentiated::OperatingPoint => Self::Point,
            Differentiated::DcSweep(sweep) => Self::Swept(sweep.values.len()),
            Differentiated::Ac(response) => Self::Frequencies(response.frequencies.len()),
            Differentiated::Stability(stability) => Self::Frequencies(stability.frequencies.len()),
            Differentiated::Noise(noise) => Self::Noise(noise.frequencies.len()),
            Differentiated::Transient(transient) => Self::Times(transient.breakpoints.clone()),
            Differentiated::PeriodicSteadyState(steady) => Self::Times(steady.breakpoints.clone()),
        }
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Point => 1,
            Self::Swept(count) | Self::Frequencies(count) | Self::Noise(count) => *count,
            Self::Times(times) => times.len(),
        }
    }

    pub(crate) const fn is_complex(&self) -> bool {
        matches!(self, Self::Frequencies(_))
    }
}

/// Some unknowns a batch traces along an analysis: the analysis index, the unknown indices
/// in the batch circuit, and its points.
pub(crate) struct Traced {
    pub(crate) analysis: usize,
    pub(crate) unknowns: Vec<usize>,
    pub(crate) points: Points,
}

impl Traced {
    pub(crate) fn len(&self) -> usize {
        self.points.len() * self.unknowns.len()
    }

    pub(crate) const fn is_complex(&self) -> bool {
        self.points.is_complex()
    }

    pub(crate) fn missing(&self) -> TracedValues {
        if self.is_complex() {
            TracedValues::Complex(vec![Complex64::new(f64::NAN, f64::NAN); self.len()])
        } else {
            TracedValues::Real(vec![f64::NAN; self.len()])
        }
    }
}

/// Reason a batch measure produced no value for a run.
#[derive(Debug, Clone, Copy)]
enum Cause {
    Failed,
    Measure(MeasureError),
}

/// Allocation-free `Eq`/`Hash` key for [`Cause`], keying floating-point payloads by their
/// IEEE-754 bit patterns (`f64::to_bits`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum CauseKey {
    Failed,
    Kind,
    Unknown(usize),
    LengthMismatch,
    Empty,
    Outside(u8, u64, u64),
    EmptyWindow,
    Flat(u64),
    Uncrossed,
    Stays(u64),
    Rises(u64),
    Vanishes,
    Unpositive(u64),
    Short,
    Silent,
    Memory,
    EmptyBand,
    Band(u64, u64, u64, u64),
    SumOverflow,
    SumIndeterminate,
    NoPoles,
    Unreferred,
    NotDifferentiable,
}

impl From<Cause> for CauseKey {
    fn from(cause: Cause) -> Self {
        match cause {
            Cause::Failed => Self::Failed,
            Cause::Measure(error) => match error {
                MeasureError::Kind => Self::Kind,
                MeasureError::Unknown(index) => Self::Unknown(index),
                MeasureError::LengthMismatch => Self::LengthMismatch,
                MeasureError::Empty => Self::Empty,
                MeasureError::Outside { point, first, last } => Self::Outside(
                    match point {
                        Point::At => 0,
                        Point::Start => 1,
                        Point::Stop => 2,
                    },
                    first.to_bits(),
                    last.to_bits(),
                ),
                MeasureError::EmptyWindow => Self::EmptyWindow,
                MeasureError::Flat(value) => Self::Flat(value.to_bits()),
                MeasureError::Uncrossed => Self::Uncrossed,
                MeasureError::Stays(value) => Self::Stays(value.to_bits()),
                MeasureError::Rises(value) => Self::Rises(value.to_bits()),
                MeasureError::Vanishes => Self::Vanishes,
                MeasureError::Unpositive(value) => Self::Unpositive(value.to_bits()),
                MeasureError::Short => Self::Short,
                MeasureError::Silent => Self::Silent,
                MeasureError::Memory => Self::Memory,
                MeasureError::EmptyBand => Self::EmptyBand,
                MeasureError::Band {
                    start,
                    stop,
                    first,
                    last,
                } => Self::Band(
                    start.to_bits(),
                    stop.to_bits(),
                    first.to_bits(),
                    last.to_bits(),
                ),
                MeasureError::Sum(SumError::Overflow) => Self::SumOverflow,
                MeasureError::Sum(SumError::Indeterminate) => Self::SumIndeterminate,
                MeasureError::NoPoles => Self::NoPoles,
                MeasureError::Unreferred => Self::Unreferred,
                MeasureError::NotDifferentiable => Self::NotDifferentiable,
            },
        }
    }
}

/// Sink state behind its lock: output slices, deduplicated measure causes, and per-row
/// elaboration problems and simulation errors.
struct Written<'a> {
    values: &'a mut [f64],
    measured: &'a mut [f64],
    causes: &'a mut [i32],
    ended: &'a mut [bool],
    traces: Vec<Buffer<'a>>,
    reasons: Vec<Cause>,
    known: HashMap<CauseKey, usize>,
    problems: BTreeMap<u64, ElaborationError>,
    errors: BTreeMap<u64, Vec<(usize, String)>>,
}

/// Sink writing each run into its row of the batch's `NumPy` arrays and optionally invoking
/// a Python callback per completed run.
struct Columns<'a> {
    first: u64,
    toleranced: &'a [Column],
    nominal: &'a [f64],
    temperature: f64,
    analyses: usize,
    measures: usize,
    traces: &'a [Traced],
    written: Mutex<Written<'a>>,
    each: Option<Each>,
}

/// Python per-run callback, cached simulators for restructured circuits, and first error.
struct Each {
    function: Py<PyAny>,
    views: Mutex<HashMap<usize, Py<Simulator>>>,
    raised: Mutex<Option<PyErr>>,
}

impl Columns<'_> {
    fn value(&self, column: &Column, values: &Values) -> f64 {
        match column {
            Column::Parameter(parameter, _, _) => values
                .get(*parameter)
                .or_else(|| self.nominal.get(parameter.index()).copied())
                .unwrap_or(f64::NAN),
            Column::Temperature => values.temperature.unwrap_or(self.temperature),
        }
    }

    fn hand(&self, index: u64, outcome: Run) {
        let Some(each) = &self.each else {
            return;
        };
        if lock_unpoisoned(&each.raised).is_some() {
            return;
        }
        Python::attach(|py| {
            if let Err(error) = each.call(py, index, outcome) {
                lock_unpoisoned(&each.raised).get_or_insert(error);
            }
        });
    }
}

impl Each {
    fn call(&self, py: Python<'_>, index: u64, outcome: Run) -> PyResult<()> {
        let structure = match &outcome.structure {
            None => None,
            Some(structure) => {
                let key = Arc::as_ptr(structure.elaborated()) as usize;
                let mut views = lock_unpoisoned(&self.views);
                if let Some(view) = views.get(&key) {
                    Some(view.clone_ref(py))
                } else {
                    let made = elaboration_result(
                        py,
                        jolty_engine::Simulator::from_elaborated(
                            Arc::clone(structure.elaborated()),
                            outcome.values.clone(),
                        ),
                    )?;
                    let view = Py::new(py, Simulator::wrapping(made))?;
                    views.insert(key, view.clone_ref(py));
                    Some(view)
                }
            }
        };
        let problems = outcome
            .problems
            .as_ref()
            .map(|problems| problems_to_python(py, problems))
            .transpose()?;
        let found = PyList::empty(py);
        for result in outcome.results {
            match result {
                Ok(result) => found.append(result_to_python(py, result)?)?,
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
            .toleranced
            .iter()
            .map(|column| self.value(column, &outcome.values))
            .collect();
        let traces: Vec<TracedValues> = self
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
            let mut written = lock_unpoisoned(&self.written);
            let written = &mut *written;
            let width = self.toleranced.len();
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
                        let next = written.reasons.len();
                        let known = *written.known.entry(CauseKey::from(cause)).or_insert(next);
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
                match (buffer, found) {
                    (Buffer::Real(buffer), TracedValues::Real(values)) => {
                        buffer[range].copy_from_slice(&values);
                    }
                    (Buffer::Complex(buffer), TracedValues::Complex(values)) => {
                        buffer[range].copy_from_slice(&values);
                    }
                    _ => {}
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

fn trace(outcome: &Run, traces: &Traced) -> TracedValues {
    match outcome.results.get(traces.analysis) {
        Some(Ok(result)) => trace_result(result, |unknown| outcome.column(unknown), traces),
        _ => traces.missing(),
    }
}

fn fill_trace<T: Copy>(
    target: &mut [T],
    columns: &[Option<usize>],
    point: usize,
    row: impl Fn(usize) -> Option<T>,
) {
    let width = columns.len();
    for (place, column) in columns.iter().enumerate() {
        if let Some(value) = column.and_then(&row) {
            target[point * width + place] = value;
        }
    }
}

/// Extracts the traced unknowns of `traces` at each point of `result`, real or complex
/// according to the analysis, filling missing unknowns with NaN.
pub(crate) fn trace_result(
    result: &AnalysisResult,
    column: impl Fn(usize) -> Option<usize>,
    traces: &Traced,
) -> TracedValues {
    let mut found = traces.missing();
    let columns: Vec<Option<usize>> = traces
        .unknowns
        .iter()
        .map(|&unknown| column(unknown))
        .collect();
    match (result, &traces.points, &mut found) {
        (AnalysisResult::OperatingPoint(point), Points::Point, TracedValues::Real(buffer)) => {
            fill_trace(buffer, &columns, 0, |column| {
                point.values.get(column).copied()
            });
        }
        (AnalysisResult::DcSweep(sweep), Points::Swept(_), TracedValues::Real(buffer)) => {
            for (point, values) in sweep.values.iter().enumerate() {
                fill_trace(buffer, &columns, point, |column| {
                    values.get(column).copied()
                });
            }
        }
        (AnalysisResult::Ac(response), Points::Frequencies(_), TracedValues::Complex(buffer)) => {
            for (frequency, values) in response.values.iter().enumerate() {
                fill_trace(buffer, &columns, frequency, |column| {
                    values.get(column).copied()
                });
            }
        }
        (AnalysisResult::Noise(spectrum), Points::Noise(_), TracedValues::Real(buffer)) => {
            let referred = spectrum.referred();
            let width = traces.unknowns.len();
            for point in 0..spectrum.frequencies.len() {
                for (place, &index) in traces.unknowns.iter().enumerate() {
                    let value = match index {
                        0 => spectrum.densities.get(point).copied(),
                        1 => referred
                            .as_ref()
                            .and_then(|values| values.get(point).copied()),
                        _ => None,
                    };
                    if let Some(value) = value {
                        buffer[point * width + place] = value;
                    }
                }
            }
        }
        (
            AnalysisResult::Stability(loop_gain),
            Points::Frequencies(_),
            TracedValues::Complex(buffer),
        ) => {
            let width = traces.unknowns.len();
            for (frequency, &gain) in loop_gain.gains.iter().enumerate() {
                for (place, &output) in traces.unknowns.iter().enumerate() {
                    if output == 0 {
                        buffer[frequency * width + place] = gain;
                    }
                }
            }
        }
        (
            AnalysisResult::Transient(transient)
            | AnalysisResult::PeriodicSteadyState(PeriodicSteadyStateResult { transient, .. }),
            Points::Times(times),
            TracedValues::Real(buffer),
        ) => {
            for (point, &time) in times.iter().enumerate() {
                fill_trace(buffer, &columns, point, |column| {
                    at(&transient.times, &transient.values, column, time)
                });
            }
        }
        _ => {}
    }
    found
}

/// Position of a query `time` relative to accepted sample `times`: exact index or
/// interpolation bracket with fraction.
pub(crate) enum Located {
    At(usize),
    Between(usize, usize, f64),
}

/// Locates `time` within sorted accepted `times`, returning an exact index, a linear
/// interpolation bracket, or `None` outside the span.
pub(crate) fn located(times: &[f64], time: f64) -> Option<Located> {
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

/// A batch as Python asks for it: analyses, run range, measures, traced unknowns, and
/// optional per-run callback.
pub(crate) struct Request {
    pub(crate) analyses: Vec<(String, String)>,
    pub(crate) runs: Range<u64>,
    pub(crate) measures: Vec<(usize, Py<measure::Measure>)>,
    pub(crate) traces: Vec<(usize, Vec<usize>)>,
    pub(crate) each: Option<Py<PyAny>>,
}

struct Arrays<'py> {
    values: Bound<'py, PyArray1<f64>>,
    measured: Bound<'py, PyArray1<f64>>,
    causes: Bound<'py, PyArray1<i32>>,
    ended: Bound<'py, PyArray1<bool>>,
    traces: Vec<Traces<'py>>,
}

struct Prepared {
    batch: Batch,
    runs: Range<u64>,
    traces: Vec<Traced>,
    analyses: usize,
    measures: usize,
}

fn prepare(elaborated: &Arc<jolty_engine::Elaborated>, request: &Request) -> PyResult<Prepared> {
    let analyses: Vec<(Analysis, Options)> = request
        .analyses
        .iter()
        .map(|(analysis, options)| Ok((parse_json!(analysis)?, parse_json!(options)?)))
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
    let batch = value_result(Batch::new(Arc::clone(elaborated), analyses, reads))?;
    Ok(Prepared {
        batch,
        runs: request.runs.clone(),
        traces,
        analyses: count,
        measures: width,
    })
}

impl<'py> Arrays<'py> {
    fn new(py: Python<'py>, runs: usize, columns: usize, prepared: &Prepared) -> Self {
        let traces = prepared
            .traces
            .iter()
            .map(|traces| {
                let len = runs * traces.len();
                if traces.is_complex() {
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

    fn shaped(
        self,
        found: &Bound<'py, PyDict>,
        runs: usize,
        columns: usize,
        prepared: &Prepared,
    ) -> PyResult<()> {
        let traces = PyList::empty(found.py());
        for (array, each) in self.traces.into_iter().zip(&prepared.traces) {
            let shape = [runs, each.points.len(), each.unknowns.len()];
            match array {
                Traces::Real(array) => traces.append(array.reshape(shape)?)?,
                Traces::Complex(array) => traces.append(array.reshape(shape)?)?,
            }
        }
        py_dict!(in found;
            "values" => self.values.reshape([runs, columns])?,
            "measured" => self.measured.reshape([runs, prepared.measures])?,
            "causes" => self.causes.reshape([runs, prepared.measures])?,
            "ended" => self.ended.reshape([runs, prepared.analyses])?,
            "traces" => traces,
        );
        Ok(())
    }
}

/// Runs a batch of `elaborated` at `base` and returns the dictionary for
/// `jolty._core.Simulator.batch`.
pub(crate) fn run<'py>(
    py: Python<'py>,
    (elaborated, base): (&Arc<jolty_engine::Elaborated>, &Values),
    variants: &Variants,
    request: Request,
    pool: &Pool,
) -> PyResult<Bound<'py, PyDict>> {
    variants.ensure_circuit(elaborated)?;
    let prepared = prepare(elaborated, &request)?;
    let based = variants.based(base);
    let Range { start, end } = prepared.runs;
    if start > end || end > based.len() {
        return Err(PyValueError::new_err(format!(
            "the runs from {start} up to {end} do not lie within the {} runs of the variants",
            based.len()
        )));
    }
    let runs = usize::try_from(end - start)
        .map_err(|_| PyValueError::new_err("more runs than an array holds"))?;
    let columns = variants.columns();
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
    found.set_item("parameters", variants.parameters())?;
    written.into_python(&found)?;
    Ok(found)
}

struct Found {
    reasons: Vec<Cause>,
    problems: BTreeMap<u64, ElaborationError>,
    errors: BTreeMap<u64, Vec<(usize, String)>>,
}

impl Found {
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
        let problems = PyDict::new(py);
        for (index, error) in &self.problems {
            problems.set_item(index, problems_to_python(py, error)?)?;
        }
        let errors = PyDict::new(py);
        for (index, failed) in self.errors {
            errors.set_item(index, failed)?;
        }
        py_dict!(in found; "reasons" => told, "problems" => problems, "errors" => errors);
        Ok(())
    }
}

fn write(
    py: Python<'_>,
    elaborated: &jolty_engine::Elaborated,
    (variants, columns): (&Based<'_>, &[Column]),
    prepared: &Prepared,
    arrays: &Arrays<'_>,
    each: Option<Py<PyAny>>,
    pool: &Pool,
) -> PyResult<Found> {
    let (mut values, mut measured, mut causes, mut ended) = (
        arrays.values.readwrite(),
        arrays.measured.readwrite(),
        arrays.causes.readwrite(),
        arrays.ended.readwrite(),
    );
    let mut guards: Vec<TraceGuard<'_>> = arrays
        .traces
        .iter()
        .map(|traces| match traces {
            Traces::Real(array) => TraceGuard::Real(array.readwrite()),
            Traces::Complex(array) => TraceGuard::Complex(array.readwrite()),
        })
        .collect();
    let buffers: Vec<Buffer<'_>> = guards
        .iter_mut()
        .map(TraceGuard::as_buffer)
        .collect::<PyResult<_>>()?;
    let sink = Columns {
        first: prepared.runs.start,
        toleranced: columns,
        nominal: elaborated.nominal(),
        temperature: elaborated.temperature(),
        analyses: prepared.analyses,
        measures: prepared.measures,
        traces: &prepared.traces,
        written: Mutex::new(Written {
            values: slice_mut(&mut values)?,
            measured: slice_mut(&mut measured)?,
            causes: slice_mut(&mut causes)?,
            ended: slice_mut(&mut ended)?,
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
    value_result(ran)?;
    let Columns { written, each, .. } = sink;
    if let Some(error) = each.and_then(|each| into_inner_unpoisoned(each.raised)) {
        return Err(error);
    }
    let Written {
        causes: cells,
        reasons,
        problems,
        errors,
        ..
    } = into_inner_unpoisoned(written);
    Ok(Found {
        reasons: canonical(cells, &reasons),
        problems,
        errors,
    })
}

/// Renumbers `reasons` in row-major first-seen order across `causes` so output order never
/// depends on thread completion order.
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

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use jolty_engine::measure::{MeasureError, Point, SumError};

    use super::{Cause, CauseKey};

    #[test]
    fn cause_key_deduplicates_variants_and_float_bit_patterns() {
        let nan_a = f64::from_bits(0x7ff8_0000_0000_0001);
        let nan_b = f64::from_bits(0x7ff8_0000_0000_0002);
        let samples = [
            Cause::Failed,
            Cause::Measure(MeasureError::Kind),
            Cause::Measure(MeasureError::Unknown(0)),
            Cause::Measure(MeasureError::Unknown(1)),
            Cause::Measure(MeasureError::LengthMismatch),
            Cause::Measure(MeasureError::Empty),
            Cause::Measure(MeasureError::Outside {
                point: Point::At,
                first: 0.0,
                last: 1.0,
            }),
            Cause::Measure(MeasureError::Outside {
                point: Point::Start,
                first: 0.0,
                last: 1.0,
            }),
            Cause::Measure(MeasureError::Outside {
                point: Point::Stop,
                first: 0.0,
                last: 1.0,
            }),
            Cause::Measure(MeasureError::EmptyWindow),
            Cause::Measure(MeasureError::Flat(0.0)),
            Cause::Measure(MeasureError::Flat(-0.0)),
            Cause::Measure(MeasureError::Flat(f64::INFINITY)),
            Cause::Measure(MeasureError::Flat(f64::NEG_INFINITY)),
            Cause::Measure(MeasureError::Flat(nan_a)),
            Cause::Measure(MeasureError::Flat(nan_b)),
            Cause::Measure(MeasureError::Uncrossed),
            Cause::Measure(MeasureError::Stays(100.0)),
            Cause::Measure(MeasureError::Rises(0.5)),
            Cause::Measure(MeasureError::Vanishes),
            Cause::Measure(MeasureError::Unpositive(0.0)),
            Cause::Measure(MeasureError::Short),
            Cause::Measure(MeasureError::Silent),
            Cause::Measure(MeasureError::Memory),
            Cause::Measure(MeasureError::EmptyBand),
            Cause::Measure(MeasureError::Band {
                start: 10.0,
                stop: 100.0,
                first: 1.0,
                last: 50.0,
            }),
            Cause::Measure(MeasureError::Sum(SumError::Overflow)),
            Cause::Measure(MeasureError::Sum(SumError::Indeterminate)),
            Cause::Measure(MeasureError::NoPoles),
            Cause::Measure(MeasureError::Unreferred),
        ];

        let mut unique = HashSet::new();
        let mut table = HashMap::new();
        for (index, cause) in samples.into_iter().enumerate() {
            let key = CauseKey::from(cause);
            assert!(unique.insert(key), "duplicate key for {cause:?}");
            table.insert(key, index);
        }
        // Repeated lookups find the exact same index, including NaN bit patterns.
        for (index, cause) in samples.into_iter().enumerate() {
            let key = CauseKey::from(cause);
            assert_eq!(table.get(&key), Some(&index), "missing key for {cause:?}");
        }
    }
}
