//! Conversions between engine results, diagnostics, and Python dictionaries and `NumPy`
//! arrays.

use jolty_engine::{
    AnalysisResult, ChargeLeak, DcSweepResult, Derivatives, Jvp, Location, MismatchResult,
    NoiseResult, OperatingPointResult, RecordedSeries, SensitivityResult, Statistics, Strategy,
    TransientResult,
};
use jolty_models::{LogicLevel, Nature};
use numpy::{Element, PyArray1, PyArray2, PyArray3, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// Builds a `PyDict` from key-value pairs, or inserts entries into an existing dictionary.
macro_rules! py_dict {
    ($py:expr; $($key:expr => $value:expr),+ $(,)?) => {{
        let dict = PyDict::new($py);
        $(dict.set_item($key, $value)?;)+
        dict
    }};
    (in $dict:ident; $($key:expr => $value:expr),+ $(,)?) => {{
        $($dict.set_item($key, $value)?;)+
    }};
}

pub(crate) use py_dict;

/// Converts `values` into a one-dimensional `NumPy` array without copying.
fn array1<T: Element>(py: Python<'_>, values: Vec<T>) -> Bound<'_, PyArray1<T>> {
    PyArray1::from_vec(py, values)
}

/// The name of a quantity: `"voltage"`, `"current"`, `"temperature"` or `"state"`.
pub(crate) fn quantity_name(nature: Nature) -> &'static str {
    match nature {
        Nature::Temperature => "temperature",
        Nature::State => "state",
        _ if nature.is_current() => "current",
        _ => "voltage",
    }
}

/// Converts real or complex derivative cubes into a three-dimensional `NumPy` array.
fn derivatives_to_py<'py>(
    py: Python<'py>,
    derivatives: &Derivatives,
) -> PyResult<Bound<'py, PyAny>> {
    match derivatives {
        Derivatives::Real(points) => Ok(cube(py, points)?.into_any()),
        Derivatives::Complex(points) => Ok(cube(py, points)?.into_any()),
    }
}

/// Converts a JVP result into a Python dictionary of `tangents`, `period`, and `result`.
pub(crate) fn jvp_to_python(py: Python<'_>, found: Jvp) -> PyResult<Bound<'_, PyDict>> {
    Ok(py_dict!(py;
        "tangents" => derivatives_to_py(py, &found.tangents)?,
        "period" => found.period.map(|period| array1(py, period)),
        "result" => outcome_to_python(py, found.result)?,
    ))
}

/// Converts logic device domain excursions from `simulator` into a Python list of
/// dictionaries.
pub(crate) fn excursions_to_python<'py>(
    py: Python<'py>,
    simulator: &jolty_engine::Simulator,
) -> PyResult<Bound<'py, PyList>> {
    let found = PyList::empty(py);
    for excursion in simulator.excursions() {
        found.append(py_dict!(py;
            "instance" => simulator.logic_devices()[excursion.device].path(),
            "bound" => excursion.bound.name(), "minimum" => excursion.minimum,
            "maximum" => excursion.maximum, "value" => excursion.value,
            "time" => excursion.time, "count" => excursion.count,
        ))?;
    }
    Ok(found)
}

/// The outcome of an analysis as a dictionary whose `kind` names the analysis.
pub(crate) fn outcome_to_python(
    py: Python<'_>,
    outcome: AnalysisResult,
) -> PyResult<Bound<'_, PyDict>> {
    Ok(match outcome {
        AnalysisResult::OperatingPoint(point) => {
            let result = operating_point_dict(py, point)?;
            result.set_item("kind", "operating_point")?;
            result
        }
        AnalysisResult::DcSweep(sweep) => dc_sweep_dict(py, sweep)?,
        AnalysisResult::Ac(response) => py_dict!(py;
            "kind" => "ac", "source" => response.source,
            "frequencies" => array1(py, response.frequencies),
            "values" => PyArray2::from_vec2(py, &response.values)?,
            "currents" => matrix(py, &response.currents)?,
            "operating_point" => operating_point_dict(py, response.operating_point)?,
        ),
        AnalysisResult::Transient(waveforms) => {
            let result = py_dict!(py; "kind" => "transient");
            add_waveforms(&result, waveforms)?;
            result
        }
        AnalysisResult::PeriodicSteadyState(found) => {
            let result = py_dict!(py;
                "kind" => "periodic_steady_state",
                "period" => found.period, "periods" => found.periods,
            );
            add_waveforms(&result, found.waveforms)?;
            result
        }
        AnalysisResult::Sensitivity(sensitivities) => sensitivity_dict(py, sensitivities)?,
        AnalysisResult::Mismatch(mismatch) => mismatch_dict(py, mismatch)?,
        AnalysisResult::Stability(loop_gain) => py_dict!(py;
            "kind" => "stability", "probe" => loop_gain.probe,
            "frequencies" => array1(py, loop_gain.frequencies),
            "gains" => array1(py, loop_gain.gains),
            "operating_point" => operating_point_dict(py, loop_gain.operating_point)?,
        ),
        AnalysisResult::Noise(spectrum) => noise_dict(py, spectrum)?,
        AnalysisResult::OperatingPoints(points) => {
            let solutions = PyList::empty(py);
            for point in points {
                solutions.append(operating_point_dict(py, point)?)?;
            }
            py_dict!(py; "kind" => "operating_points", "points" => solutions)
        }
        AnalysisResult::PoleZero(found) => py_dict!(py;
            "kind" => "pole_zero", "source" => found.source,
            "output" => found.output, "reference" => found.reference,
            "poles" => array1(py, found.poles),
            "zeros" => found.zeros.map(|zeros| array1(py, zeros)),
            "operating_point" => operating_point_dict(py, found.operating_point)?,
        ),
        AnalysisResult::TransferFunction(transfer) => py_dict!(py;
            "kind" => "transfer_function", "source" => transfer.source,
            "input" => quantity_name(transfer.input), "output" => transfer.output,
            "reference" => transfer.reference, "gain" => transfer.gain,
            "input_resistance" => transfer.input_resistance,
            "output_resistance" => transfer.output_resistance,
            "operating_point" => operating_point_dict(py, transfer.operating_point)?,
        ),
    })
}

/// Converts a sensitivity analysis result into a `PyDict`.
fn sensitivity_dict(
    py: Python<'_>,
    sensitivities: SensitivityResult,
) -> PyResult<Bound<'_, PyDict>> {
    let (parameters, values): (Vec<(String, String)>, Vec<f64>) = sensitivities
        .parameters
        .into_iter()
        .map(|parameter| ((parameter.instance, parameter.name), parameter.value))
        .unzip();
    let result = py_dict!(py;
        "kind" => "sensitivity", "parameters" => parameters, "values" => array1(py, values),
    );
    add_differentiated(
        &result,
        sensitivities.outputs,
        &sensitivities.derivatives,
        sensitivities.period_derivatives,
        *sensitivities.result,
    )?;
    Ok(result)
}

/// Converts a mismatch analysis result into a `PyDict`.
fn mismatch_dict(py: Python<'_>, mismatch: MismatchResult) -> PyResult<Bound<'_, PyDict>> {
    let random_variables: Vec<_> = mismatch
        .random_variables
        .into_iter()
        .map(|variable| {
            (
                variable.instance,
                variable.lot,
                variable.name,
                variable.mean,
                variable.standard_deviation,
                variable.derivatives,
            )
        })
        .collect();
    let result = py_dict!(py; "kind" => "mismatch", "random_variables" => random_variables);
    add_differentiated(
        &result,
        mismatch.outputs,
        &mismatch.derivatives,
        mismatch.period_derivatives,
        *mismatch.result,
    )?;
    Ok(result)
}

/// Adds the common tail of a sensitivity or mismatch result: outputs, derivatives, optional
/// oscillator period derivatives, and the underlying analysis result.
fn add_differentiated(
    result: &Bound<'_, PyDict>,
    outputs: Vec<String>,
    derivatives: &Derivatives,
    period_derivatives: Option<Vec<f64>>,
    inner: AnalysisResult,
) -> PyResult<()> {
    let py = result.py();
    py_dict!(in result;
        "outputs" => outputs,
        "derivatives" => derivatives_to_py(py, derivatives)?,
        "period_derivatives" => period_derivatives.map(|values| array1(py, values)),
        "result" => outcome_to_python(py, inner)?,
    );
    Ok(())
}

/// Adds transient or periodic steady state waveforms, logic traces, observations, and
/// statistics to `result`.
fn add_waveforms(result: &Bound<'_, PyDict>, waveforms: TransientResult) -> PyResult<()> {
    let py = result.py();
    let traces = PyList::empty(py);
    for trace in waveforms.logic {
        let (times, levels): (Vec<f64>, Vec<LogicLevel>) = trace.into_iter().unzip();
        traces.append((array1(py, times), array1(py, numbers(&levels))))?;
    }
    py_dict!(in result;
        "times" => array1(py, waveforms.times), "values" => stacked(py, waveforms.values)?,
        "logic" => traces, "statistics" => statistics_to_python(py, waveforms.statistics)?,
    );
    add_observations(result, waveforms.observations)
}

/// Converts a DC sweep result into a `PyDict`.
fn dc_sweep_dict(py: Python<'_>, sweep: DcSweepResult) -> PyResult<Bound<'_, PyDict>> {
    let logic: Vec<Vec<f64>> = sweep.logic.iter().map(|levels| numbers(levels)).collect();
    let strategies: Vec<&str> = sweep
        .strategies
        .iter()
        .copied()
        .map(Strategy::name)
        .collect();
    let result = py_dict!(py;
        "kind" => "dc_sweep", "source" => sweep.source, "parameter" => sweep.parameter,
        "nature" => sweep.nature.map(quantity_name), "swept" => array1(py, sweep.swept),
        "values" => stacked(py, sweep.values)?, "logic" => matrix(py, &logic)?,
        "strategies" => strategies, "iterations" => sweep.iterations,
    );
    add_observations(&result, sweep.observations)?;
    Ok(result)
}

/// Converts a noise spectrum result into a `PyDict`.
fn noise_dict(py: Python<'_>, spectrum: NoiseResult) -> PyResult<Bound<'_, PyDict>> {
    let count = spectrum.frequencies.len();
    let input = match spectrum.input {
        Some(input) => Some(py_dict!(py;
            "source" => input.source, "nature" => quantity_name(input.nature),
            "gains" => array1(py, input.gains),
        )),
        None => None,
    };
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
    Ok(py_dict!(py;
        "kind" => "noise", "output" => spectrum.output, "reference" => spectrum.reference,
        "nature" => quantity_name(spectrum.nature),
        "frequencies" => array1(py, spectrum.frequencies),
        "densities" => array1(py, spectrum.densities), "input" => input,
        "sources" => sources, "contributions" => array1(py, flat).reshape([rows, count])?,
        "operating_point" => operating_point_dict(py, spectrum.operating_point)?,
    ))
}

/// Converts an operating point into a `PyDict`.
fn operating_point_dict(
    py: Python<'_>,
    operating_point: OperatingPointResult,
) -> PyResult<Bound<'_, PyDict>> {
    let recorded = operating_point.observation;
    Ok(py_dict!(py;
        "values" => array1(py, operating_point.values),
        "logic" => array1(py, numbers(&operating_point.logic)),
        "currents" => array1(py, recorded.currents), "charges" => array1(py, recorded.charges),
        "conductances" => array1(py, recorded.conductances),
        "capacitances" => array1(py, recorded.capacitances),
        "variables" => array1(py, recorded.variables),
        "strategy" => operating_point.strategy.name(),
        "iterations" => operating_point.iterations,
    ))
}

/// Adds recorded terminal and variable series to `result`, one row per point.
fn add_observations(result: &Bound<'_, PyDict>, observations: RecordedSeries) -> PyResult<()> {
    let py = result.py();
    let points = observations.points;
    // Each quantity's rows already lie end to end, so the arrays take them without a copy.
    let rows = |values: Vec<f64>| {
        let width = values.len().checked_div(points).unwrap_or(0);
        array1(py, values).reshape([points, width])
    };
    py_dict!(in result;
        "currents" => rows(observations.currents)?, "charges" => rows(observations.charges)?,
        "conductances" => rows(observations.conductances)?,
        "capacitances" => rows(observations.capacitances)?,
        "variables" => rows(observations.variables)?,
    );
    Ok(())
}

/// `rows`, all of one length, as a two-dimensional array with as many rows even when
/// they are empty; each row is freed once copied, so the values are held about once.
fn stacked(py: Python<'_>, rows: Vec<Vec<f64>>) -> PyResult<Bound<'_, PyArray2<f64>>> {
    let (count, columns) = (rows.len(), rows.first().map_or(0, Vec::len));
    let mut flat = Vec::with_capacity(count * columns);
    for row in rows {
        flat.extend_from_slice(&row);
    }
    array1(py, flat).reshape([count, columns])
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
    array1(py, flat).reshape([rows.len(), columns])
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
    array1(py, flat).reshape([points.len(), rows, columns])
}

/// Each of `found` as a dictionary of its `rule`, its `message`, its `instances` and its
/// `nets`.
pub(crate) fn diagnostics_to_python(
    py: Python<'_>,
    found: Vec<jolty_checks::Diagnostic>,
) -> PyResult<Bound<'_, PyList>> {
    let diagnostics = PyList::empty(py);
    for diagnostic in found {
        let instances = PyList::empty(py);
        for location in &diagnostic.instances {
            instances.append(location_to_python(py, location)?)?;
        }
        diagnostics.append(py_dict!(py;
            "rule" => diagnostic.rule, "message" => diagnostic.message,
            "instances" => instances, "nets" => diagnostic.nets,
        ))?;
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
        let instance = problem
            .location()
            .map(|location| location_to_python(py, location))
            .transpose()?;
        problems.append(py_dict!(py; "message" => problem.message(), "instance" => instance))?;
    }
    Ok(problems)
}

/// An instance as a dictionary of its `path` and its `source_location`, or `None`.
fn location_to_python<'py>(py: Python<'py>, location: &Location) -> PyResult<Bound<'py, PyDict>> {
    let source = match &location.source_location {
        Some(source) => Some(py_dict!(py;
            "file" => &source.file, "line" => source.line, "function" => &source.function,
        )),
        None => None,
    };
    Ok(py_dict!(py; "path" => &location.path, "source_location" => source))
}

/// Converts transient or periodic steady state run statistics into a `PyDict`.
fn statistics_to_python(py: Python<'_>, statistics: Statistics) -> PyResult<Bound<'_, PyDict>> {
    let work = statistics.work;
    let profile = match statistics.profile {
        Some(profile) => Some(py_dict!(py;
            "preparing" => profile.preparing.as_secs_f64(),
            "evaluating" => profile.evaluating.as_secs_f64(),
            "assembling" => profile.assembling.as_secs_f64(),
            "factoring" => profile.factoring.as_secs_f64(),
            "solving" => profile.solving.as_secs_f64(),
            "step_control" => profile.step_control.as_secs_f64(),
            "recording" => profile.recording.as_secs_f64(),
            "other" => profile.other.as_secs_f64(),
        )),
        None => None,
    };
    let ChargeLeak { signed, absolute } = statistics.charge_leak;
    Ok(py_dict!(py;
        "initial" => statistics.initial.map(Strategy::name),
        "accepted" => statistics.accepted, "rejected" => statistics.rejected,
        "landed" => statistics.landed, "iterations" => statistics.iterations,
        "work" => py_dict!(py;
            "iterations" => work.iterations, "factorizations" => work.factorizations,
            "refactorizations" => work.refactorizations, "fallbacks" => work.fallbacks,
            "reuses" => work.reuses,
        ),
        "profile" => profile,
        "charge_leak" => py_dict!(py;
            "signed" => array1(py, signed), "absolute" => array1(py, absolute),
        ),
    ))
}
