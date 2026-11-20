//! Engine measures exposed as `jolty._core.Measure`, together with shared noise and pole
//! arithmetic.
//!
//! Each measure resolves signals to unknown indices with times, values, and frequencies in
//! SI base units, then evaluates directly on result arrays. When evaluation fails, the
//! measure raises `MeasureError` carrying a reason tag (such as `"outside"`) and numeric
//! payload; `jolty.specs` formats the message.

use std::borrow::Cow;

use jolty_engine::measure::{
    self, AnalysisResult, Edge, MeasureCotangent, MeasureError as Failure, Signal, SumError, Table,
    Transfer, Window,
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
    "A measure cannot evaluate a result. Exception arguments are the reason tag (such as \
     \"outside\") followed by the associated numeric values."
);

/// Signal passed from Python: `(unknown_index, optional_reference_index)`.
type PySignal = (usize, Option<usize>);

/// Transfer passed from Python: `None` for loop gain, or `(signal, optional_divisor)`.
type PyTransfer = Option<(PySignal, Option<PySignal>)>;

/// Time window passed from Python: `(start, stop)`, where `None` selects the axis endpoint.
type PyWindow = (Option<f64>, Option<f64>);

/// Sparse real cotangent returned to Python: `(value, [(point, unknown, weight)], period)`.
type PyRealCotangent = (f64, Vec<(usize, usize, f64)>, f64);

/// Sparse complex table cotangent returned to Python: `(value, [(point, unknown, real, imag)])`.
type PyComplexCotangent = (f64, Vec<(usize, usize, f64, f64)>);

/// Sparse complex vector cotangent returned to Python: `(value, [(point, real, imag)])`.
type PyPointComplexCotangent = (f64, Vec<(usize, f64, f64)>);

/// Sparse noise cotangent returned to Python:
/// `(value, [(point, density_weight)], [(point, gain_real, gain_imag)])`.
type PyNoiseCotangent = (f64, Vec<(usize, f64)>, Vec<(usize, f64, f64)>);

/// Scalar measurement evaluated directly on engine analysis results.
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
        /// Creates a measure returning `signal` at an operating point or interpolated `at` a
        /// time or swept value.
        value(signal: PySignal, at: Option<f64>) => Value;
        /// Creates a measure returning the gain of `transfer` at frequency `at` in decibels.
        gain(transfer: PyTransfer, at: f64) => Gain;
        /// Creates a measure returning the phase of `transfer` at frequency `at` in degrees.
        phase(transfer: PyTransfer, at: f64) => Phase;
        /// Creates a measure returning the frequency where `transfer` gain first drops by
        /// `drop` decibels.
        bandwidth(transfer: PyTransfer, drop: f64) => Bandwidth;
        /// Creates a measure returning the frequency where `transfer` gain first crosses 0 dB.
        crossover(transfer: PyTransfer) => Crossover;
        /// Creates a measure returning 180 degrees plus the phase of `transfer` at 0 dB gain
        /// crossover.
        phase_margin(transfer: PyTransfer) => PhaseMargin;
        /// Creates a measure returning negated gain of `transfer` in decibels at -180 degree
        /// phase crossover.
        gain_margin(transfer: PyTransfer) => GainMargin;
        /// Creates a measure returning the maximum value of `signal` within `window`.
        maximum(signal: PySignal, window: PyWindow) => Maximum;
        /// Creates a measure returning the minimum value of `signal` within `window`.
        minimum(signal: PySignal, window: PyWindow) => Minimum;
        /// Creates a measure returning the peak-to-peak amplitude of `signal` within `window`.
        peak_to_peak(signal: PySignal, window: PyWindow) => PeakToPeak;
        /// Creates a measure returning the trapezoidal mean of `signal` within `window`.
        mean(signal: PySignal, window: PyWindow) => Mean;
        /// Creates a measure returning the root mean square of `signal` within `window`.
        rms(signal: PySignal, window: PyWindow) => Rms;
        /// Creates a measure returning the time for `signal` to traverse from `lower` to
        /// `upper` of its step across `window`.
        rise_time(signal: PySignal, window: PyWindow, lower: f64, upper: f64) => RiseTime;
        /// Creates a measure returning the percentage overshoot of `signal` relative to its
        /// step across `window`.
        overshoot(signal: PySignal, window: PyWindow) => Overshoot;
        /// Creates a measure returning the settling time until `signal` stays within `band`
        /// of its step across `window`.
        settling_time(signal: PySignal, window: PyWindow, band: f64) => SettlingTime;
        /// Creates a measure returning the rising-edge crossing frequency of `signal` across
        /// `threshold` (midpoint by default) in `window`.
        frequency(signal: PySignal, window: PyWindow, threshold: Option<f64>) => Frequency;
        /// Creates a measure returning total harmonic distortion of `signal` at `fundamental`
        /// over harmonics 2 through `harmonics`, in decibels.
        distortion(
            signal: PySignal,
            window: PyWindow,
            fundamental: f64,
            harmonics: usize,
        ) => Distortion;
        /// Creates a measure returning spot noise density at frequency `at`, input-referred
        /// when `referred` is true.
        noise_density(at: f64, referred: bool) => NoiseDensity;
        /// Creates a measure returning integrated RMS noise from `start` to `stop`,
        /// input-referred when `referred` is true.
        integrated_noise(start: Option<f64>, stop: Option<f64>, referred: bool) => IntegratedNoise;
        /// Creates a measure returning the steady-state period of a periodic steady-state
        /// analysis in seconds.
        period() => Period;
        /// Creates a measure returning the number of DC operating-point solutions found.
        solutions() => Solutions;
        /// Creates a measure returning the minimum damping ratio across all poles.
        damping() => Damping;
        /// Creates a measure returning the minimum pole natural frequency in hertz.
        dominant_pole() => DominantPole;
    }
    table_readers {
        /// Evaluates the measure on a DC sweep's `swept` axis and `values` matrix (one row
        /// per point).
        read_dc_sweep(swept, values: f64) => DcSweep;
        /// Evaluates the measure on a transient's `times` axis and `values` matrix (one row
        /// per time).
        read_transient(times, values: f64) => Transient;
        /// Evaluates the measure on an AC response's `frequencies` axis and complex `values`
        /// matrix (one row per frequency).
        read_ac(frequencies, values: Complex64) => Ac;
    }
    slice_readers {
        /// Evaluates the measure on an operating point's `values` vector.
        read_operating_point(values: f64) => OperatingPoint;
        /// Evaluates the measure on a pole-zero analysis's `poles` vector.
        read_pole_zero(poles: Complex64) => PoleZero;
    }
    methods {
        /// Creates a measure returning the first time `signal` crosses `threshold` in
        /// `window` on the specified `edge` (`"rising"`, `"falling"`, or `"either"`).
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

        /// Evaluates the measure on a stability loop gain's `frequencies` and complex `gains`.
        fn read_stability(
            &self,
            frequencies: &Bound<'_, PyArray1<f64>>,
            gains: &Bound<'_, PyArray1<Complex64>>,
        ) -> PyResult<f64> {
            with_slices(frequencies, gains, |frequencies, gains| {
                self.read(&AnalysisResult::Stability { frequencies, gains })
            })
        }

        /// Evaluates the measure on a noise analysis's `frequencies`, `densities`, and
        /// optional input-referred `gains`.
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

        /// Evaluates the measure on a periodic steady state's `times`, `values` matrix,
        /// and `period` in seconds.
        fn read_periodic_steady_state(
            &self,
            times: &Bound<'_, PyArray1<f64>>,
            values: &Bound<'_, PyArray2<f64>>,
            period: f64,
        ) -> PyResult<f64> {
            self.read_table(times, values, |times, values| {
                AnalysisResult::PeriodicSteadyState {
                    times,
                    values,
                    period,
                }
            })
        }

        /// Evaluates the measure on an operating-points search returning `count` DC
        /// solutions.
        fn read_operating_points(&self, count: usize) -> PyResult<f64> {
            self.read(&AnalysisResult::OperatingPoints(count))
        }

        /// Evaluates the measure and its sparse real cotangent on an operating point's
        /// `values` vector.
        fn cotangent_operating_point(
            &self,
            values: &Bound<'_, PyArray1<f64>>,
        ) -> PyResult<PyRealCotangent> {
            let outcome = with_slice(values, |values| {
                self.cotangent(&AnalysisResult::OperatingPoint(values))
            })?;
            unpack_real(outcome)
        }

        /// Evaluates the measure and its sparse real cotangent on a DC sweep's `swept` axis
        /// and `values` matrix.
        fn cotangent_dc_sweep(
            &self,
            swept: &Bound<'_, PyArray1<f64>>,
            values: &Bound<'_, PyArray2<f64>>,
        ) -> PyResult<PyRealCotangent> {
            let outcome = self.cotangent_table(swept, values, |swept, values| {
                AnalysisResult::DcSweep { swept, values }
            })?;
            unpack_real(outcome)
        }

        /// Evaluates the measure and its sparse real cotangent on a transient's `times` axis
        /// and `values` matrix.
        fn cotangent_transient(
            &self,
            times: &Bound<'_, PyArray1<f64>>,
            values: &Bound<'_, PyArray2<f64>>,
        ) -> PyResult<PyRealCotangent> {
            let outcome = self.cotangent_table(times, values, |times, values| {
                AnalysisResult::Transient { times, values }
            })?;
            unpack_real(outcome)
        }

        /// Evaluates the measure and its sparse real cotangent (including autonomous period
        /// sensitivity) on a periodic steady state's `times`, `values`, and `period`.
        fn cotangent_periodic_steady_state(
            &self,
            times: &Bound<'_, PyArray1<f64>>,
            values: &Bound<'_, PyArray2<f64>>,
            period: f64,
        ) -> PyResult<PyRealCotangent> {
            let outcome = self.cotangent_table(times, values, |times, values| {
                AnalysisResult::PeriodicSteadyState {
                    times,
                    values,
                    period,
                }
            })?;
            unpack_real(outcome)
        }

        /// Evaluates the measure and its sparse bilinear complex cotangent on an AC
        /// response's `frequencies` axis and complex `values` matrix.
        fn cotangent_ac(
            &self,
            frequencies: &Bound<'_, PyArray1<f64>>,
            values: &Bound<'_, PyArray2<Complex64>>,
        ) -> PyResult<PyComplexCotangent> {
            let outcome = self.cotangent_table(frequencies, values, |frequencies, values| {
                AnalysisResult::Ac {
                    frequencies,
                    values,
                }
            })?;
            unpack_complex(outcome)
        }

        /// Evaluates the measure and its sparse bilinear complex cotangent on a stability
        /// loop gain's `frequencies` and complex `gains`.
        fn cotangent_stability(
            &self,
            frequencies: &Bound<'_, PyArray1<f64>>,
            gains: &Bound<'_, PyArray1<Complex64>>,
        ) -> PyResult<PyPointComplexCotangent> {
            let outcome = with_slices(frequencies, gains, |frequencies, gains| {
                self.cotangent(&AnalysisResult::Stability { frequencies, gains })
            })?;
            unpack_point_complex(outcome)
        }

        /// Evaluates the measure and its sparse cotangent on a noise analysis's
        /// `frequencies`, `densities`, and optional input-referred `gains`.
        fn cotangent_noise(
            &self,
            frequencies: &Bound<'_, PyArray1<f64>>,
            densities: &Bound<'_, PyArray1<f64>>,
            gains: Option<&Bound<'_, PyArray1<Complex64>>>,
        ) -> PyResult<PyNoiseCotangent> {
            let outcome = with_slices(frequencies, densities, |frequencies, densities| {
                let gains = gains.map(PyArrayMethods::readonly);
                let gains = gains.as_ref().map(slice);
                self.cotangent(&AnalysisResult::Noise {
                    frequencies,
                    densities,
                    gains: gains.as_deref(),
                })
            })?;
            unpack_noise(outcome)
        }

        /// Evaluates the measure and its sparse bilinear complex cotangent on a pole-zero
        /// analysis's `poles` vector.
        fn cotangent_pole_zero(
            &self,
            poles: &Bound<'_, PyArray1<Complex64>>,
        ) -> PyResult<PyPointComplexCotangent> {
            let outcome = with_slice(poles, |poles| {
                self.cotangent(&AnalysisResult::PoleZero(poles))
            })?;
            unpack_point_complex(outcome)
        }

        /// Evaluates the measure's cotangent on an operating-points search (raises
        /// `MeasureError` because operating-point counts are discrete).
        fn cotangent_operating_points(&self, count: usize) -> PyResult<PyRealCotangent> {
            let outcome = self.cotangent(&AnalysisResult::OperatingPoints(count))?;
            unpack_real(outcome)
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

    fn cotangent(&self, results: &AnalysisResult<'_>) -> PyResult<(f64, MeasureCotangent)> {
        self.0.cotangent(results).map_err(failure)
    }

    fn cotangent_table<T: Element + Copy>(
        &self,
        axis: &Bound<'_, PyArray1<f64>>,
        values: &Bound<'_, PyArray2<T>>,
        build: impl for<'a> FnOnce(&'a [f64], &'a dyn Table<T>) -> AnalysisResult<'a>,
    ) -> PyResult<(f64, MeasureCotangent)> {
        let (axis, values) = (axis.readonly(), values.readonly());
        let rows = Rows(values.as_array());
        let table: &dyn Table<T> = &rows;
        self.cotangent(&build(&slice(&axis), table))
    }
}

fn unpack_real((value, cotangent): (f64, MeasureCotangent)) -> PyResult<PyRealCotangent> {
    match cotangent {
        MeasureCotangent::Real { entries, period } => Ok((value, entries, period)),
        _ => Err(failure(Failure::Kind)),
    }
}

fn unpack_complex((value, cotangent): (f64, MeasureCotangent)) -> PyResult<PyComplexCotangent> {
    match cotangent {
        MeasureCotangent::Complex { entries } => Ok((
            value,
            entries
                .into_iter()
                .map(|(point, unknown, weight)| (point, unknown, weight.re, weight.im))
                .collect(),
        )),
        _ => Err(failure(Failure::Kind)),
    }
}

fn unpack_point_complex(
    (value, cotangent): (f64, MeasureCotangent),
) -> PyResult<PyPointComplexCotangent> {
    match cotangent {
        MeasureCotangent::Loop { entries } | MeasureCotangent::Poles { entries } => Ok((
            value,
            entries
                .into_iter()
                .map(|(point, weight)| (point, weight.re, weight.im))
                .collect(),
        )),
        _ => Err(failure(Failure::Kind)),
    }
}

fn unpack_noise((value, cotangent): (f64, MeasureCotangent)) -> PyResult<PyNoiseCotangent> {
    match cotangent {
        MeasureCotangent::Noise { densities, gains } => Ok((
            value,
            densities,
            gains
                .into_iter()
                .map(|(point, weight)| (point, weight.re, weight.im))
                .collect(),
        )),
        _ => Err(failure(Failure::Kind)),
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

/// Returns a contiguous slice of `array`, borrowing in place when contiguous and copying
/// otherwise.
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

/// Converts `error` into a Python exception (`OverflowError`, `ValueError`, `MemoryError`,
/// or `MeasureError`).
pub(crate) fn failure(error: Failure) -> PyErr {
    match error {
        Failure::Sum(SumError::Overflow) => PyOverflowError::new_err(error.to_string()),
        Failure::Sum(SumError::Indeterminate) => PyValueError::new_err(error.to_string()),
        Failure::Memory => PyMemoryError::new_err(error.to_string()),
        other => MeasureError::new_err(reason(other)),
    }
}

/// Maps `error` to its Python `MeasureError` reason tag and numeric payload.
///
/// Overflow, indeterminate sum, and allocation failures map to `"overflow"`,
/// `"indeterminate"`, and `"memory"` to match `math.fsum` and `MemoryError`.
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
        Failure::NotDifferentiable => "not_differentiable",
    };
    (name, Vec::new())
}

/// Integrates `densities` over `frequencies` from `start` to `stop` (defaulting to the
/// axis endpoints) using piecewise power-law interpolation.
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

/// Interpolates spot noise density at frequency `at` using a power law between neighboring
/// `frequencies`.
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

/// Refers output noise `densities` to an input by dividing each entry by the squared
/// magnitude of the corresponding complex gain.
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

/// Computes the damping ratio `-Re(s) / |s|` for each complex root in `roots`.
#[pyfunction]
fn dampings<'py>(
    py: Python<'py>,
    roots: &Bound<'py, PyArray1<Complex64>>,
) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_vec(py, with_slice(roots, measure::dampings))
}

/// Computes the natural frequency `|s| / (2 * pi)` in hertz for each complex root in
/// `roots`.
#[pyfunction]
fn natural_frequencies<'py>(
    py: Python<'py>,
    roots: &Bound<'py, PyArray1<Complex64>>,
) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_vec(py, with_slice(roots, measure::natural_frequencies))
}

/// Registers `Measure`, `MeasureError`, `SAMPLES_PER_PERIOD`, and shared arithmetic
/// functions on `module`.
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
