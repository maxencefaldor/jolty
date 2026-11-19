//! Batch vector-Jacobian products and pullbacks exposed as `jolty._core.BatchPullback`.

use std::collections::BTreeMap;
use std::fmt;
use std::ops::{AddAssign, Mul, Range};
use std::sync::Arc;

use jolty_engine::batch::{Batch, Mapped, Variants as _};
use jolty_engine::variants::Explicit;
use jolty_engine::{
    AnalysisResult, Cotangent, Cotangents, Differentiated, Options, ParameterCotangents, Pullback,
    SimulationError, Values,
};
use numpy::{Complex64, Element, PyArray1, PyArrayMethods, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::Pool;
use crate::convert::{problems_to_python, py_dict};
use crate::helpers::{parse_json, value_result};
use crate::variants::Variants;

use super::{Located, Points, Traced, TracedValues, located, trace_result};

/// A batch's VJP request: analysis JSON, options JSON, parameters, run range, and traced
/// unknowns.
pub(crate) struct Differentiating {
    pub(crate) analysis: String,
    pub(crate) options: String,
    pub(crate) parameters: Vec<(String, String)>,
    pub(crate) runs: Range<u64>,
    pub(crate) unknowns: Vec<usize>,
}

type Derived = Mapped<Result<Pullback, SimulationError>>;

/// Per-run pullbacks returned by `jolty._core.Simulator.batch_vjp`, mapping output
/// cotangents back to parameters.
#[pyclass(module = "jolty._core", frozen)]
pub(crate) struct BatchPullback {
    batch: Batch,
    traced: Traced,
    names: Vec<String>,
    parameters: usize,
    runs: Vec<Derived>,
}

/// Runs the batch VJP for `request` on `elaborated` at `base`, returning [`results`] and
/// the corresponding [`BatchPullback`].
pub(crate) fn vjp<'py>(
    py: Python<'py>,
    (elaborated, base): (&Arc<jolty_engine::Elaborated>, &Values),
    variants: &Variants,
    request: Differentiating,
    pool: &Pool,
) -> PyResult<(Bound<'py, PyDict>, BatchPullback)> {
    variants.ensure_circuit(elaborated)?;
    let differentiated: Differentiated = parse_json!(&request.analysis)?;
    let options: Options = parse_json!(&request.options)?;
    let traced = Traced {
        analysis: 0,
        unknowns: request.unknowns,
        points: Points::from_differentiated(&differentiated),
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
    let based = variants.based(base);
    let batch = value_result(Batch::new(Arc::clone(elaborated), Vec::new(), Vec::new()))?;
    let parameters = request.parameters;
    let runs = value_result(py.detach(|| {
        pool.inner.install(|| {
            batch.map(
                &based,
                request.runs.clone(),
                &options,
                &|_, simulator, options| simulator.vjp(&differentiated, &parameters, options),
            )
        })
    }))?;
    let found = results(py, &runs, &traced)?;
    let pullback = BatchPullback {
        batch,
        traced,
        names,
        parameters: parameters.len(),
        runs,
    };
    Ok((found, pullback))
}

/// Builds the result dictionary for `batch_vjp`: `traces`, `periods`, `problems`, `errors`.
fn results<'py>(
    py: Python<'py>,
    runs: &[Derived],
    traced: &Traced,
) -> PyResult<Bound<'py, PyDict>> {
    let total = runs.len() * traced.len();
    let mut collected = if traced.is_complex() {
        TracedValues::Complex(Vec::with_capacity(total))
    } else {
        TracedValues::Real(Vec::with_capacity(total))
    };
    let mut periods = vec![f64::NAN; runs.len()];
    let (problems, errors) = (PyDict::new(py), PyDict::new(py));
    for (row, run) in runs.iter().enumerate() {
        let values = match &run.found {
            Ok(Ok(pullback)) => {
                if let AnalysisResult::PeriodicSteadyState(steady) = &pullback.result {
                    periods[row] = steady.period;
                }
                trace_result(&pullback.result, |unknown| run.column(unknown), traced)
            }
            Ok(Err(error)) => {
                errors.set_item(row, error.to_string())?;
                traced.missing()
            }
            Err(error) => {
                problems.set_item(row, problems_to_python(py, error)?)?;
                traced.missing()
            }
        };
        match (&mut collected, values) {
            (TracedValues::Real(all), TracedValues::Real(row_values)) => all.extend(row_values),
            (TracedValues::Complex(all), TracedValues::Complex(row_values)) => {
                all.extend(row_values);
            }
            _ => {}
        }
    }
    let shape = [runs.len(), traced.points.len(), traced.unknowns.len()];
    let array = match collected {
        TracedValues::Real(values) => PyArray1::from_vec(py, values).reshape(shape)?.into_any(),
        TracedValues::Complex(values) => PyArray1::from_vec(py, values).reshape(shape)?.into_any(),
    };
    Ok(py_dict!(py;
        "traces" => array, "periods" => PyArray1::from_vec(py, periods),
        "problems" => problems, "errors" => errors,
    ))
}

trait Cotangential:
    Copy + Default + PartialEq + Send + Sync + AddAssign + Mul<f64, Output = Self> + Element
{
    fn cotangents(each: Vec<Cotangent<Self>>) -> Cotangents;
    fn unpack(pulled: ParameterCotangents) -> Result<Vec<Self>, String>;
    fn is_nan(self) -> bool;
}

impl Cotangential for f64 {
    fn cotangents(each: Vec<Cotangent<Self>>) -> Cotangents {
        Cotangents::Real(each)
    }

    fn unpack(pulled: ParameterCotangents) -> Result<Vec<Self>, String> {
        match pulled {
            ParameterCotangents::Real(mut found) => Ok(found.pop().unwrap_or_default()),
            ParameterCotangents::Complex(_) => Err("expected real parameter cotangents".to_owned()),
        }
    }

    fn is_nan(self) -> bool {
        f64::is_nan(self)
    }
}

impl Cotangential for Complex64 {
    fn cotangents(each: Vec<Cotangent<Self>>) -> Cotangents {
        Cotangents::Complex(each)
    }

    fn unpack(pulled: ParameterCotangents) -> Result<Vec<Self>, String> {
        match pulled {
            ParameterCotangents::Complex(mut found) => Ok(found.pop().unwrap_or_default()),
            ParameterCotangents::Real(_) => Err("expected complex parameter cotangents".to_owned()),
        }
    }

    fn is_nan(self) -> bool {
        Complex64::is_nan(self)
    }
}

#[pymethods]
impl BatchPullback {
    /// Returns the number of runs recorded in this batch pullback.
    #[getter]
    fn runs(&self) -> usize {
        self.runs.len()
    }

    /// Pulls output cotangents `values` and oscillator period cotangents `periods` back to
    /// parameters for each run index in `runs`.
    ///
    /// `values` has shape `(len(runs), points, unknowns)` (real, or complex for AC
    /// analyses), and `options` is the JSON options text. Work executes across `pool` with
    /// the interpreter released, returning a `(len(runs), parameters)` array; `runs` may
    /// repeat a run index.
    ///
    /// Raises `ValueError` for invalid shapes or out-of-range run indices, and
    /// `SimulationError` for failed runs, missing unknowns, or pullback failures.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 extracts owned Vec arguments from Python sequences"
    )]
    fn pull_back<'py>(
        &self,
        py: Python<'py>,
        runs: Vec<usize>,
        values: &Bound<'py, PyAny>,
        periods: Vec<f64>,
        options: &str,
        pool: &Pool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let options: Options = parse_json!(options)?;
        if let Ok(real) = values.extract::<PyReadonlyArray3<'py, f64>>() {
            self.pulled(py, (&runs, &real, &periods), &options, pool)
        } else {
            let phasors = values.extract::<PyReadonlyArray3<'py, Complex64>>()?;
            self.pulled(py, (&runs, &phasors, &periods), &options, pool)
        }
    }
}

impl BatchPullback {
    fn pulled<'py, T: Cotangential>(
        &self,
        py: Python<'py>,
        (runs, values, periods): (&[usize], &PyReadonlyArray3<'py, T>, &[f64]),
        options: &Options,
        pool: &Pool,
    ) -> PyResult<Bound<'py, PyAny>> {
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
        let shape = [runs.len(), self.parameters];
        if runs.is_empty() {
            let empty = if self.traced.is_complex() {
                PyArray1::from_vec(py, Vec::<Complex64>::new())
                    .reshape(shape)?
                    .into_any()
            } else {
                PyArray1::from_vec(py, Vec::<f64>::new())
                    .reshape(shape)?
                    .into_any()
            };
            return Ok(empty);
        }
        let rows: Vec<Vec<T>> = given
            .outer_iter()
            .map(|row| row.iter().copied().collect())
            .collect();
        let selected = Explicit(
            runs.iter()
                .map(|&run| self.runs[run].values.clone())
                .collect(),
        );
        let found = value_result(py.detach(|| {
            pool.inner.install(|| {
                self.batch
                    .map(
                        &selected,
                        0..selected.len(),
                        options,
                        &|index, simulator, options| -> Result<Vec<T>, String> {
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
                            T::unpack(pulled)
                        },
                    )
            })
        }))?;
        let mut pulled: Vec<T> = Vec::with_capacity(runs.len() * self.parameters);
        for (&run, mapped) in runs.iter().zip(found) {
            pulled.extend(succeeded(run, mapped.found)?);
        }
        Ok(PyArray1::from_vec(py, pulled).reshape(shape)?.into_any())
    }

    fn cotangent<T: Cotangential>(
        &self,
        derived: &Derived,
        pullback: &Pullback,
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
            AnalysisResult::Transient(waveforms) => Some(waveforms.times.as_slice()),
            AnalysisResult::PeriodicSteadyState(steady) => Some(steady.transient.times.as_slice()),
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

fn succeeded<T, E: fmt::Display>(
    run: usize,
    found: Result<Result<T, E>, impl fmt::Display>,
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
