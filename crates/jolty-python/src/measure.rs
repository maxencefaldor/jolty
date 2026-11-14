//! The engine's measures, as `jolty._core.Measure`, and the arithmetic of noise and poles that
//! results share with them.
//!
//! A measure is made with its signals resolved to the indices of a result's unknowns and its
//! times, levels and frequencies in SI base units, then reads a result from its arrays. A
//! measure that cannot read a result raises `MeasureError`, whose arguments are its reason, a
//! name such as `"outside"`, then the numbers it states; `jolty.specs` says why in words.

use std::borrow::Cow;

use jolty_engine::measure::{
    self, Edge, MeasureError as Failure, Results, Signal, SumError, Table, Transfer, Window,
};
use numpy::ndarray::ArrayView2;
use numpy::{Complex64, Element, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyMemoryError, PyOverflowError, PyValueError};
use pyo3::prelude::*;

create_exception!(
    _core,
    MeasureError,
    PyException,
    "A measure cannot read a result. Its arguments are the reason, such as \"outside\", then \
     the numbers it states."
);

/// A signal as Python gives it: the index of its unknown, and of the unknown it is taken
/// relative to, if any.
type PySignal = (usize, Option<usize>);

/// A transfer as Python gives it: none for a loop gain, or an output and what it is divided
/// by, if anything.
type PyTransfer = Option<(PySignal, Option<PySignal>)>;

/// A window as Python gives it: its start and its stop, each none for the axis's own end.
type PyWindow = (Option<f64>, Option<f64>);

/// One number read from the result of one analysis, as the engine reads it.
#[pyclass(module = "jolty._core", frozen)]
pub(crate) struct Measure(pub(crate) measure::Measure);

fn to_signal((unknown, reference): PySignal) -> Signal {
    Signal { unknown, reference }
}

fn to_transfer(transfer: PyTransfer) -> Transfer {
    match transfer {
        None => Transfer::Loop,
        Some((output, over)) => Transfer::Output {
            output: to_signal(output),
            over: over.map(to_signal),
        },
    }
}

fn to_window((start, stop): PyWindow) -> Window {
    Window { start, stop }
}

#[pymethods]
impl Measure {
    /// The value of `signal` at an operating point, or interpolated `at` a time or a level.
    #[staticmethod]
    fn value(signal: PySignal, at: Option<f64>) -> Self {
        Self(measure::Measure::Value {
            signal: to_signal(signal),
            at,
        })
    }

    /// The gain of `transfer` at the frequency `at`, in decibels.
    #[staticmethod]
    fn gain(transfer: PyTransfer, at: f64) -> Self {
        Self(measure::Measure::Gain {
            transfer: to_transfer(transfer),
            at,
        })
    }

    /// The phase of `transfer` at the frequency `at`, in degrees.
    #[staticmethod]
    fn phase(transfer: PyTransfer, at: f64) -> Self {
        Self(measure::Measure::Phase {
            transfer: to_transfer(transfer),
            at,
        })
    }

    /// The frequency where the gain of `transfer` first falls `drop` decibels.
    #[staticmethod]
    fn bandwidth(transfer: PyTransfer, drop: f64) -> Self {
        Self(measure::Measure::Bandwidth {
            transfer: to_transfer(transfer),
            drop,
        })
    }

    /// The frequency where the gain of `transfer` first falls through 0 dB.
    #[staticmethod]
    fn crossover(transfer: PyTransfer) -> Self {
        Self(measure::Measure::Crossover {
            transfer: to_transfer(transfer),
        })
    }

    /// 180° plus the phase of `transfer` where its gain first falls through 0 dB.
    #[staticmethod]
    fn phase_margin(transfer: PyTransfer) -> Self {
        Self(measure::Measure::PhaseMargin {
            transfer: to_transfer(transfer),
        })
    }

    /// Minus the gain of `transfer` where its phase first falls through -180°, in decibels.
    #[staticmethod]
    fn gain_margin(transfer: PyTransfer) -> Self {
        Self(measure::Measure::GainMargin {
            transfer: to_transfer(transfer),
        })
    }

    /// The highest value of `signal` in `window`.
    #[staticmethod]
    fn maximum(signal: PySignal, window: PyWindow) -> Self {
        Self(measure::Measure::Maximum {
            signal: to_signal(signal),
            window: to_window(window),
        })
    }

    /// The lowest value of `signal` in `window`.
    #[staticmethod]
    fn minimum(signal: PySignal, window: PyWindow) -> Self {
        Self(measure::Measure::Minimum {
            signal: to_signal(signal),
            window: to_window(window),
        })
    }

    /// The highest less the lowest value of `signal` in `window`.
    #[staticmethod]
    fn peak_to_peak(signal: PySignal, window: PyWindow) -> Self {
        Self(measure::Measure::PeakToPeak {
            signal: to_signal(signal),
            window: to_window(window),
        })
    }

    /// The mean of `signal` in `window`, by the trapezoidal rule.
    #[staticmethod]
    fn average(signal: PySignal, window: PyWindow) -> Self {
        Self(measure::Measure::Average {
            signal: to_signal(signal),
            window: to_window(window),
        })
    }

    /// The root mean square of `signal` in `window`.
    #[staticmethod]
    fn rms(signal: PySignal, window: PyWindow) -> Self {
        Self(measure::Measure::Rms {
            signal: to_signal(signal),
            window: to_window(window),
        })
    }

    /// The time `signal` takes to pass from `low` to `high` of its step across `window`.
    #[staticmethod]
    fn rise_time(signal: PySignal, window: PyWindow, low: f64, high: f64) -> Self {
        Self(measure::Measure::RiseTime {
            signal: to_signal(signal),
            window: to_window(window),
            low,
            high,
        })
    }

    /// How far `signal` passes its final value, in percent of its step across `window`.
    #[staticmethod]
    fn overshoot(signal: PySignal, window: PyWindow) -> Self {
        Self(measure::Measure::Overshoot {
            signal: to_signal(signal),
            window: to_window(window),
        })
    }

    /// The time until `signal` stays within `band` of its step across `window`.
    #[staticmethod]
    fn settling_time(signal: PySignal, window: PyWindow, band: f64) -> Self {
        Self(measure::Measure::SettlingTime {
            signal: to_signal(signal),
            window: to_window(window),
            band,
        })
    }

    /// The first time `signal` crosses `level` in `window` the way `edge` says: `"rising"`,
    /// `"falling"` or `"either"`.
    #[staticmethod]
    fn crossing(signal: PySignal, window: PyWindow, level: f64, edge: &str) -> PyResult<Self> {
        let edge = match edge {
            "rising" => Edge::Rising,
            "falling" => Edge::Falling,
            "either" => Edge::Either,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "an edge is rising, falling or either, not {edge}"
                )));
            }
        };
        Ok(Self(measure::Measure::Crossing {
            signal: to_signal(signal),
            window: to_window(window),
            level,
            edge,
        }))
    }

    /// How often `signal` rises through `level` in `window`, by default its midpoint.
    #[staticmethod]
    fn frequency(signal: PySignal, window: PyWindow, level: Option<f64>) -> Self {
        Self(measure::Measure::Frequency {
            signal: to_signal(signal),
            window: to_window(window),
            level,
        })
    }

    /// The total harmonic distortion of `signal` at `fundamental`, harmonics 2 to
    /// `harmonics`, in decibels.
    #[staticmethod]
    fn distortion(signal: PySignal, window: PyWindow, fundamental: f64, harmonics: usize) -> Self {
        Self(measure::Measure::Distortion {
            signal: to_signal(signal),
            window: to_window(window),
            fundamental,
            harmonics,
        })
    }

    /// The noise density at the frequency `at`, referred to the input if `referred`.
    #[staticmethod]
    fn noise_density(at: f64, referred: bool) -> Self {
        Self(measure::Measure::NoiseDensity { at, referred })
    }

    /// The RMS noise from `start` to `stop`, referred to the input if `referred`.
    #[staticmethod]
    fn integrated_noise(start: Option<f64>, stop: Option<f64>, referred: bool) -> Self {
        Self(measure::Measure::IntegratedNoise {
            start,
            stop,
            referred,
        })
    }

    /// How many DC solutions a search found.
    #[staticmethod]
    fn solutions() -> Self {
        Self(measure::Measure::Solutions)
    }

    /// The least damping ratio of the poles.
    #[staticmethod]
    fn damping() -> Self {
        Self(measure::Measure::Damping)
    }

    /// The least natural frequency of the poles, in hertz.
    #[staticmethod]
    fn dominant_pole() -> Self {
        Self(measure::Measure::DominantPole)
    }

    /// What the measure reads of an operating point's `values`.
    fn read_point(&self, values: &Bound<'_, PyArray1<f64>>) -> PyResult<f64> {
        let values = values.readonly();
        self.read(&Results::Point(&slice(&values)))
    }

    /// What the measure reads of a DC sweep's `levels` and `values`, one row per level.
    fn read_sweep(
        &self,
        levels: &Bound<'_, PyArray1<f64>>,
        values: &Bound<'_, PyArray2<f64>>,
    ) -> PyResult<f64> {
        let levels = levels.readonly();
        let values = values.readonly();
        self.read(&Results::Sweep {
            levels: &slice(&levels),
            values: &Rows(values.as_array()),
        })
    }

    /// What the measure reads of a transient's `times` and `values`, one row per time.
    fn read_transient(
        &self,
        times: &Bound<'_, PyArray1<f64>>,
        values: &Bound<'_, PyArray2<f64>>,
    ) -> PyResult<f64> {
        let times = times.readonly();
        let values = values.readonly();
        self.read(&Results::Transient {
            times: &slice(&times),
            values: &Rows(values.as_array()),
        })
    }

    /// What the measure reads of a small-signal response's `frequencies` and phasors,
    /// `values`, one row per frequency.
    fn read_response(
        &self,
        frequencies: &Bound<'_, PyArray1<f64>>,
        values: &Bound<'_, PyArray2<Complex64>>,
    ) -> PyResult<f64> {
        let frequencies = frequencies.readonly();
        let values = values.readonly();
        self.read(&Results::Response {
            frequencies: &slice(&frequencies),
            values: &Rows(values.as_array()),
        })
    }

    /// What the measure reads of a loop gain's `frequencies` and `gains`.
    fn read_loop(
        &self,
        frequencies: &Bound<'_, PyArray1<f64>>,
        gains: &Bound<'_, PyArray1<Complex64>>,
    ) -> PyResult<f64> {
        let frequencies = frequencies.readonly();
        let gains = gains.readonly();
        self.read(&Results::Loop {
            frequencies: &slice(&frequencies),
            gains: &slice(&gains),
        })
    }

    /// What the measure reads of a noise analysis's `frequencies`, `densities`, and `gains`
    /// from its source, if it names one.
    fn read_noise(
        &self,
        frequencies: &Bound<'_, PyArray1<f64>>,
        densities: &Bound<'_, PyArray1<f64>>,
        gains: Option<&Bound<'_, PyArray1<Complex64>>>,
    ) -> PyResult<f64> {
        let frequencies = frequencies.readonly();
        let densities = densities.readonly();
        let gains = gains.map(PyArrayMethods::readonly);
        let gains = gains.as_ref().map(slice);
        self.read(&Results::Noise {
            frequencies: &slice(&frequencies),
            densities: &slice(&densities),
            gains: gains.as_deref(),
        })
    }

    /// What the measure reads of a pole-zero analysis's `poles`.
    fn read_poles(&self, poles: &Bound<'_, PyArray1<Complex64>>) -> PyResult<f64> {
        let poles = poles.readonly();
        self.read(&Results::Poles(&slice(&poles)))
    }

    /// What the measure reads of a search that found `count` DC solutions.
    fn read_solutions(&self, count: usize) -> PyResult<f64> {
        self.read(&Results::Points(count))
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

impl Measure {
    fn read(&self, results: &Results<'_>) -> PyResult<f64> {
        self.0.measure(results).map_err(failure)
    }
}

/// The rows of a two-dimensional array, as a table of a result's unknowns.
struct Rows<'a, T>(ArrayView2<'a, T>);

impl<T: Copy> Table<T> for Rows<'_, T> {
    fn rows(&self) -> usize {
        self.0.nrows()
    }

    fn value(&self, row: usize, column: usize) -> Option<T> {
        self.0.get((row, column)).copied()
    }
}

/// The values of `array`, borrowed when they lie in order in memory, else copied.
fn slice<'a, T: Element + Copy>(array: &'a PyReadonlyArray1<'_, T>) -> Cow<'a, [T]> {
    match array.as_slice() {
        Ok(values) => Cow::Borrowed(values),
        Err(_) => Cow::Owned(array.as_array().iter().copied().collect()),
    }
}

/// `error` as a Python exception: the one Python's `math.fsum` or the allocation of a
/// distortion's samples would raise, or a `MeasureError` of its reason and numbers.
pub(crate) fn failure(error: Failure) -> PyErr {
    match reason(error) {
        ("overflow", _) => PyOverflowError::new_err(error.to_string()),
        ("indeterminate", _) => PyValueError::new_err(error.to_string()),
        ("memory", _) => PyMemoryError::new_err(error.to_string()),
        (reason, numbers) => MeasureError::new_err((reason, numbers)),
    }
}

/// Why `error` read nothing, as Python is told: the reason a `MeasureError` names, such as
/// `"outside_at"`, and the numbers it states; or `"overflow"`, `"indeterminate"` or
/// `"memory"` where Python raises `OverflowError`, `ValueError` or `MemoryError`, as its
/// `math.fsum` or its allocation of a distortion's samples would.
pub(crate) fn reason(error: Failure) -> (&'static str, Vec<f64>) {
    match error {
        Failure::Sum(SumError::Overflow) => ("overflow", vec![]),
        Failure::Sum(SumError::Indeterminate) => ("indeterminate", vec![]),
        Failure::Memory => ("memory", vec![]),
        Failure::Kind => ("kind", vec![]),
        Failure::Unknown(_) => ("unknown", vec![]),
        Failure::Mismatch => ("mismatch", vec![]),
        Failure::Empty => ("empty", vec![]),
        Failure::Outside { point, first, last } => {
            let named = match point {
                measure::Point::At => "outside_at",
                measure::Point::Start => "outside_start",
                measure::Point::Stop => "outside_stop",
            };
            (named, vec![first, last])
        }
        Failure::EmptyWindow => ("empty_window", vec![]),
        Failure::Flat(level) => ("flat", vec![level]),
        Failure::Uncrossed => ("uncrossed", vec![]),
        Failure::Stays(lowest) => ("stays", vec![lowest]),
        Failure::Rises(level) => ("rises", vec![level]),
        Failure::Vanishes => ("vanishes", vec![]),
        Failure::Unpositive(lowest) => ("unpositive", vec![lowest]),
        Failure::Short => ("short", vec![]),
        Failure::Silent => ("silent", vec![]),
        Failure::EmptyBand => ("empty_band", vec![]),
        Failure::Band {
            low,
            high,
            first,
            last,
        } => ("band", vec![low, high, first, last]),
        Failure::NoPoles => ("no_poles", vec![]),
        Failure::Unreferred => ("unreferred", vec![]),
    }
}

/// The integral of `densities` over `frequencies` from `start` to `stop`, by default the
/// lowest and the highest frequency, each density a power law between frequencies.
#[pyfunction]
#[pyo3(signature = (frequencies, densities, start=None, stop=None))]
fn integrate(
    frequencies: &Bound<'_, PyArray1<f64>>,
    densities: &Bound<'_, PyArray1<f64>>,
    start: Option<f64>,
    stop: Option<f64>,
) -> PyResult<f64> {
    let frequencies = frequencies.readonly();
    let densities = densities.readonly();
    measure::integrate(&slice(&frequencies), &slice(&densities), start, stop).map_err(failure)
}

/// The density at the frequency `at`, a power law between the frequencies around it.
#[pyfunction]
fn density(
    frequencies: &Bound<'_, PyArray1<f64>>,
    densities: &Bound<'_, PyArray1<f64>>,
    at: f64,
) -> PyResult<f64> {
    let frequencies = frequencies.readonly();
    let densities = densities.readonly();
    measure::density(&slice(&frequencies), &slice(&densities), at).map_err(failure)
}

/// The noise `densities` referred to an input through `gains`, each over the gain's
/// magnitude squared.
#[pyfunction]
fn referred<'py>(
    py: Python<'py>,
    densities: &Bound<'py, PyArray1<f64>>,
    gains: &Bound<'py, PyArray1<Complex64>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let densities = densities.readonly();
    let gains = gains.readonly();
    let (densities, gains) = (slice(&densities), slice(&gains));
    if densities.len() != gains.len() {
        return Err(failure(Failure::Mismatch));
    }
    Ok(PyArray1::from_vec(
        py,
        measure::referred(&densities, &gains),
    ))
}

/// The damping ratio of each of `roots`, minus its real part over its magnitude.
#[pyfunction]
fn dampings<'py>(
    py: Python<'py>,
    roots: &Bound<'py, PyArray1<Complex64>>,
) -> Bound<'py, PyArray1<f64>> {
    let roots = roots.readonly();
    PyArray1::from_vec(py, measure::dampings(&slice(&roots)))
}

/// The natural frequency of each of `roots`, its magnitude over 2π, in hertz.
#[pyfunction]
fn natural_frequencies<'py>(
    py: Python<'py>,
    roots: &Bound<'py, PyArray1<Complex64>>,
) -> Bound<'py, PyArray1<f64>> {
    let roots = roots.readonly();
    PyArray1::from_vec(py, measure::natural_frequencies(&slice(&roots)))
}

/// Adds the measures, their error and the arithmetic results share with them to `module`.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add_class::<Measure>()?;
    module.add("MeasureError", py.get_type::<MeasureError>())?;
    module.add("SAMPLES_PER_PERIOD", measure::SAMPLES_PER_PERIOD)?;
    module.add_function(wrap_pyfunction!(integrate, module)?)?;
    module.add_function(wrap_pyfunction!(density, module)?)?;
    module.add_function(wrap_pyfunction!(referred, module)?)?;
    module.add_function(wrap_pyfunction!(dampings, module)?)?;
    module.add_function(wrap_pyfunction!(natural_frequencies, module)?)?;
    Ok(())
}
