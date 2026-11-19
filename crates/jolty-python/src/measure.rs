//! The engine's measures, as `jolty._core.Measure`, and the arithmetic of noise and poles that
//! results share with them.
//!
//! A measure is made with its signals resolved to the indices of a result's unknowns and its
//! times, values and frequencies in SI base units, then reads a result from its arrays. A
//! measure that cannot read a result raises `MeasureError`, whose arguments are its reason, a
//! name such as `"outside"`, then the numbers it states; `jolty.specs` says why in words.

use std::borrow::Cow;

use jolty_engine::measure::{
    self, AnalysisResult, Edge, MeasureError as Failure, Signal, SumError, Table, Transfer, Window,
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

/// A transfer as Python gives it: none for a loop gain, or a signal and what it is divided
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
        Some((signal, over)) => Transfer::Signal {
            signal: to_signal(signal),
            over: over.map(to_signal),
        },
    }
}

fn to_window((start, stop): PyWindow) -> Window {
    Window { start, stop }
}

/// Generates `#[pymethods] impl Measure` with uniform `#[staticmethod]` constructors and
/// array readers preserving each method's docstrings, followed by custom methods.
macro_rules! measure {
    (@val signal $value:ident) => {
        to_signal($value)
    };
    (@val transfer $value:ident) => {
        to_transfer($value)
    };
    (@val window $value:ident) => {
        to_window($value)
    };
    (@val $_:ident $value:ident) => {
        $value
    };
    (@body $variant:ident) => {
        Self(measure::Measure::$variant)
    };
    (@body $variant:ident $($param:ident)+) => {
        Self(measure::Measure::$variant {
            $($param: measure!(@val $param $param)),+
        })
    };
    (
        constructors {
            $(
                $(#[$doc:meta])*
                $name:ident($($param:ident: $ty:ty),* $(,)?) => $variant:ident;
            )*
        }
        table_readers {
            $(
                $(#[$reader_doc:meta])*
                $reader:ident($axis:ident, $values:ident: $element:ty) => $kind:ident;
            )*
        }
        slice_readers {
            $(
                $(#[$single_doc:meta])*
                $single:ident($arg:ident: $arg_ty:ty) => $single_kind:ident;
            )*
        }
        methods {
            $($custom:tt)*
        }
    ) => {
        #[pymethods]
        impl Measure {
            $(
                $(#[$doc])*
                #[staticmethod]
                fn $name($($param: $ty),*) -> Self {
                    measure!(@body $variant $($param)*)
                }
            )*
            $(
                $(#[$reader_doc])*
                fn $reader(
                    &self,
                    $axis: &Bound<'_, PyArray1<f64>>,
                    $values: &Bound<'_, PyArray2<$element>>,
                ) -> PyResult<f64> {
                    self.read_table($axis, $values, |$axis, $values| {
                        AnalysisResult::$kind { $axis, $values }
                    })
                }
            )*
            $(
                $(#[$single_doc])*
                fn $single(&self, $arg: &Bound<'_, PyArray1<$arg_ty>>) -> PyResult<f64> {
                    with_slice($arg, |$arg| self.read(&AnalysisResult::$single_kind($arg)))
                }
            )*
            $($custom)*
        }
    };
}

measure! {
    constructors {
        /// The value of `signal` at an operating point, or interpolated `at` a time or a
        /// swept value.
        value(signal: PySignal, at: Option<f64>) => Value;
        /// The gain of `transfer` at the frequency `at`, in decibels.
        gain(transfer: PyTransfer, at: f64) => Gain;
        /// The phase of `transfer` at the frequency `at`, in degrees.
        phase(transfer: PyTransfer, at: f64) => Phase;
        /// The frequency where the gain of `transfer` first falls `drop` decibels.
        bandwidth(transfer: PyTransfer, drop: f64) => Bandwidth;
        /// The frequency where the gain of `transfer` first falls through 0 dB.
        crossover(transfer: PyTransfer) => Crossover;
        /// 180° plus the phase of `transfer` where its gain first falls through 0 dB.
        phase_margin(transfer: PyTransfer) => PhaseMargin;
        /// Minus the gain of `transfer` where its phase first falls through -180°, in decibels.
        gain_margin(transfer: PyTransfer) => GainMargin;
        /// The highest value of `signal` in `window`.
        maximum(signal: PySignal, window: PyWindow) => Maximum;
        /// The lowest value of `signal` in `window`.
        minimum(signal: PySignal, window: PyWindow) => Minimum;
        /// The highest less the lowest value of `signal` in `window`.
        peak_to_peak(signal: PySignal, window: PyWindow) => PeakToPeak;
        /// The mean of `signal` in `window`, by the trapezoidal rule.
        mean(signal: PySignal, window: PyWindow) => Mean;
        /// The root mean square of `signal` in `window`.
        rms(signal: PySignal, window: PyWindow) => Rms;
        /// The time `signal` takes to pass from `lower` to `upper` of its step across `window`.
        rise_time(signal: PySignal, window: PyWindow, lower: f64, upper: f64) => RiseTime;
        /// How far `signal` passes its final value, in percent of its step across `window`.
        overshoot(signal: PySignal, window: PyWindow) => Overshoot;
        /// The time until `signal` stays within `band` of its step across `window`.
        settling_time(signal: PySignal, window: PyWindow, band: f64) => SettlingTime;
        /// How often `signal` rises through `threshold` in `window`, by default its midpoint.
        frequency(signal: PySignal, window: PyWindow, threshold: Option<f64>) => Frequency;
        /// The total harmonic distortion of `signal` at `fundamental`, harmonics 2 to
        /// `harmonics`, in decibels.
        distortion(
            signal: PySignal,
            window: PyWindow,
            fundamental: f64,
            harmonics: usize,
        ) => Distortion;
        /// The noise density at the frequency `at`, referred to the input if `referred`.
        noise_density(at: f64, referred: bool) => NoiseDensity;
        /// The RMS noise from `start` to `stop`, referred to the input if `referred`.
        integrated_noise(start: Option<f64>, stop: Option<f64>, referred: bool) => IntegratedNoise;
        /// How many DC solutions a search found.
        solutions() => Solutions;
        /// The least damping ratio of the poles.
        damping() => Damping;
        /// The least natural frequency of the poles, in hertz.
        dominant_pole() => DominantPole;
    }
    table_readers {
        /// What the measure reads of a DC sweep's `swept` values and `values`, one row per point.
        read_dc_sweep(swept, values: f64) => DcSweep;
        /// What the measure reads of a transient's `times` and `values`, one row per time.
        read_transient(times, values: f64) => Transient;
        /// What the measure reads of a small-signal response's `frequencies` and phasors,
        /// `values`, one row per frequency.
        read_ac(frequencies, values: Complex64) => Ac;
    }
    slice_readers {
        /// What the measure reads of an operating point's `values`.
        read_operating_point(values: f64) => OperatingPoint;
        /// What the measure reads of a pole-zero analysis's `poles`.
        read_pole_zero(poles: Complex64) => PoleZero;
    }
    methods {
        /// The first time `signal` crosses `threshold` in `window` the way `edge` says:
        /// `"rising"`, `"falling"` or `"either"`.
        #[staticmethod]
        fn crossing(
            signal: PySignal,
            window: PyWindow,
            threshold: f64,
            edge: &str,
        ) -> PyResult<Self> {
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
                threshold,
                edge,
            }))
        }

        /// What the measure reads of a loop gain's `frequencies` and `gains`.
        fn read_stability(
            &self,
            frequencies: &Bound<'_, PyArray1<f64>>,
            gains: &Bound<'_, PyArray1<Complex64>>,
        ) -> PyResult<f64> {
            with_slices(frequencies, gains, |frequencies, gains| {
                self.read(&AnalysisResult::Stability { frequencies, gains })
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
            with_slices(frequencies, densities, |frequencies, densities| {
                let gains = gains.map(PyArrayMethods::readonly);
                let gains = gains.as_ref().map(slice);
                self.read(&AnalysisResult::Noise {
                    frequencies,
                    densities,
                    gains: gains.as_deref(),
                })
            })
        }

        /// What the measure reads of a search that found `count` DC solutions.
        fn read_operating_points(&self, count: usize) -> PyResult<f64> {
            self.read(&AnalysisResult::OperatingPoints(count))
        }

        fn __repr__(&self) -> String {
            format!("{:?}", self.0)
        }
    }
}

impl Measure {
    fn read(&self, results: &AnalysisResult<'_>) -> PyResult<f64> {
        self.0.measure(results).map_err(failure)
    }

    fn read_table<T: Element + Copy>(
        &self,
        axis: &Bound<'_, PyArray1<f64>>,
        values: &Bound<'_, PyArray2<T>>,
        build: impl for<'a> FnOnce(&'a [f64], &'a dyn Table<T>) -> AnalysisResult<'a>,
    ) -> PyResult<f64> {
        let (axis, values) = (axis.readonly(), values.readonly());
        let rows = Rows(values.as_array());
        let table: &dyn Table<T> = &rows;
        self.read(&build(&slice(&axis), table))
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

/// Runs `work` on a contiguous slice of `array`, borrowing in place when contiguous.
fn with_slice<T: Element + Copy, R>(
    array: &Bound<'_, PyArray1<T>>,
    work: impl FnOnce(&[T]) -> R,
) -> R {
    let guard = array.readonly();
    work(&slice(&guard))
}

/// Runs `work` on contiguous slices of `first` and `second`.
fn with_slices<A: Element + Copy, B: Element + Copy, R>(
    first: &Bound<'_, PyArray1<A>>,
    second: &Bound<'_, PyArray1<B>>,
    work: impl FnOnce(&[A], &[B]) -> R,
) -> R {
    let (first, second) = (first.readonly(), second.readonly());
    work(&slice(&first), &slice(&second))
}

/// `error` as a Python exception: the one Python's `math.fsum` or the allocation of a
/// distortion's samples would raise, or a `MeasureError` of its reason and numbers.
pub(crate) fn failure(error: Failure) -> PyErr {
    match error {
        Failure::Sum(SumError::Overflow) => PyOverflowError::new_err(error.to_string()),
        Failure::Sum(SumError::Indeterminate) => PyValueError::new_err(error.to_string()),
        Failure::Memory => PyMemoryError::new_err(error.to_string()),
        other => MeasureError::new_err(reason(other)),
    }
}

/// Why `error` read nothing, as Python is told: the reason a `MeasureError` names, such as
/// `"outside_at"`, and the numbers it states; or `"overflow"`, `"indeterminate"` or
/// `"memory"` where Python raises `OverflowError`, `ValueError` or `MemoryError`, as its
/// `math.fsum` or its allocation of a distortion's samples would.
pub(crate) fn reason(error: Failure) -> (&'static str, Vec<f64>) {
    let name = match error {
        Failure::Outside { point, first, last } => {
            let named = match point {
                measure::Point::At => "outside_at",
                measure::Point::Start => "outside_start",
                measure::Point::Stop => "outside_stop",
            };
            return (named, vec![first, last]);
        }
        Failure::Flat(value) => return ("flat", vec![value]),
        Failure::Stays(lowest) => return ("stays", vec![lowest]),
        Failure::Rises(value) => return ("rises", vec![value]),
        Failure::Unpositive(lowest) => return ("unpositive", vec![lowest]),
        Failure::Band {
            start,
            stop,
            first,
            last,
        } => return ("band", vec![start, stop, first, last]),
        Failure::Sum(SumError::Overflow) => "overflow",
        Failure::Sum(SumError::Indeterminate) => "indeterminate",
        Failure::Memory => "memory",
        Failure::Kind => "kind",
        Failure::Unknown(_) => "unknown",
        Failure::LengthMismatch => "mismatch",
        Failure::Empty => "empty",
        Failure::EmptyWindow => "empty_window",
        Failure::Uncrossed => "uncrossed",
        Failure::Vanishes => "vanishes",
        Failure::Short => "short",
        Failure::Silent => "silent",
        Failure::EmptyBand => "empty_band",
        Failure::NoPoles => "no_poles",
        Failure::Unreferred => "unreferred",
    };
    (name, Vec::new())
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
    with_slices(frequencies, densities, |frequencies, densities| {
        measure::integrate(frequencies, densities, start, stop).map_err(failure)
    })
}

/// The density at the frequency `at`, a power law between the frequencies around it.
#[pyfunction]
fn density(
    frequencies: &Bound<'_, PyArray1<f64>>,
    densities: &Bound<'_, PyArray1<f64>>,
    at: f64,
) -> PyResult<f64> {
    with_slices(frequencies, densities, |frequencies, densities| {
        measure::density(frequencies, densities, at).map_err(failure)
    })
}

/// The noise `densities` referred to an input through `gains`, each over the gain's
/// magnitude squared.
#[pyfunction]
fn referred<'py>(
    py: Python<'py>,
    densities: &Bound<'py, PyArray1<f64>>,
    gains: &Bound<'py, PyArray1<Complex64>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    with_slices(densities, gains, |densities, gains| {
        if densities.len() != gains.len() {
            return Err(failure(Failure::LengthMismatch));
        }
        Ok(PyArray1::from_vec(py, measure::referred(densities, gains)))
    })
}

/// The damping ratio of each of `roots`, minus its real part over its magnitude.
#[pyfunction]
fn dampings<'py>(
    py: Python<'py>,
    roots: &Bound<'py, PyArray1<Complex64>>,
) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_vec(py, with_slice(roots, measure::dampings))
}

/// The natural frequency of each of `roots`, its magnitude over 2π, in hertz.
#[pyfunction]
fn natural_frequencies<'py>(
    py: Python<'py>,
    roots: &Bound<'py, PyArray1<Complex64>>,
) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_vec(py, with_slice(roots, measure::natural_frequencies))
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
