//! Scalar measurements extracted from simulation results for specification checks.
//!
//! Each [`Measure`] evaluates one scalar quantity from an [`AnalysisResult`], including
//! signal values, windowed statistics, edge timings, gains, stability margins, noise
//! densities, and pole damping ratios. Signal indices and SI base units are pre-resolved so
//! evaluation requires no string lookups or unit conversion; the Python layer resolves names
//! and formats [`MeasureError`] diagnostics.
//!
//! Arithmetic matches `NumPy` operation for operation on platforms with fused multiply-add,
//! including interpolation and pairwise summation order. Distortion computes harmonic
//! amplitudes via direct discrete Fourier sums rather than an FFT.
//!
//! ```
//! use jolty_engine::measure::{Measure, AnalysisResult, Signal, Window};
//!
//! // Ramp from 0 to 1 over 1 s on the single unknown.
//! let times = [0.0, 0.5, 1.0];
//! let values = vec![vec![0.0], vec![0.5], vec![1.0]];
//! let results = AnalysisResult::Transient { times: &times, values: &values };
//! let signal = Signal { unknown: 0, reference: None };
//! let window = Window { start: Some(0.25), stop: None };
//! assert_eq!(Measure::Mean { signal, window }.measure(&results), Ok(0.625));
//! ```

mod arithmetic;
mod cotangent;

use std::borrow::Cow;
use std::collections::BTreeMap;

use jolty_numerics::Complex64;

use crate::{
    AcResult, Cotangent, Cotangents, DcSweepResult, NoiseResult, OperatingPointResult,
    PeriodicSteadyStateResult, PoleZeroResult, StabilityResult, TransientResult,
};

pub(crate) use arithmetic::trapezoid;
pub use arithmetic::{dampings, density, integrate, natural_frequencies, referred};

/// Relative endpoint tolerance (`1e-9` of axis span) within which a query snaps onto the
/// axis bounds.
///
/// Derivation: endpoint rounding margin (`~4.5e6 * f64::EPSILON`) absorbing decimal-unit and
/// log-grid rounding (such as `stop = N / fundamental`) while rejecting out-of-range queries.
pub const ON_AXIS: f64 = 1e-9;

/// Uniformly interpolated samples per fundamental period used for harmonic distortion.
///
/// Derivation: power-of-two grid (`2^10 = 1024`; compare ngspice 47 `DEF_FOURGRIDSIZE = 200`
/// in `src/frontend/fourier.c`, line 32) keeping quadrant axes exact and resolving harmonics
/// 1 through 511 below Nyquist.
pub const SAMPLES_PER_PERIOD: usize = 1024;

/// Minimum harmonic-to-fundamental ratio reported by distortion, yielding -6000 dB instead
/// of negative infinity for pure tones.
///
/// Derivation: round power of ten (`20 * log10(1e-300) = -6000 dB`) safely above IEEE 754
/// binary64 `f64::MIN_POSITIVE` (`~2.225e-308`) to avoid subnormals and `-inf`.
const MIN_DISTORTION: f64 = 1e-300;

/// Single unknown or differential pair of unknowns indexed into an analysis result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Signal {
    /// Index of the primary unknown.
    pub unknown: usize,
    /// Optional index of a reference unknown subtracted from `unknown`.
    pub reference: Option<usize>,
}

/// Optional `[start, stop]` interval in SI base units over a transient or sweep axis,
/// defaulting to the full axis span.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Window {
    /// Inclusive window start in SI base units, or axis start if `None`.
    pub start: Option<f64>,
    /// Inclusive window stop in SI base units, or axis stop if `None`.
    pub stop: Option<f64>,
}

/// Direction of a threshold crossing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Edge {
    /// Crossing from below the threshold to at or above the threshold.
    Rising,
    /// Crossing from above the threshold to at or below the threshold.
    Falling,
    /// Crossing in either direction.
    Either,
}

/// Target transfer quantity for small-signal measures: a response signal (optionally divided
/// by a denominator signal) or a stability loop gain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transfer {
    /// Loop gain from a stability analysis.
    Loop,
    /// Signal phasor from an AC response, optionally divided by `over`.
    Signal {
        /// Numerator signal.
        signal: Signal,
        /// Optional denominator signal.
        over: Option<Signal>,
    },
}

/// Specification of one scalar measurement over an [`AnalysisResult`].
///
/// Results are returned in SI base units unless documented otherwise. Small-signal measures
/// interpolate linearly in log frequency from the lowest frequency; time and sweep measures
/// interpolate linearly along their native axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Measure {
    /// Value of `signal` at an operating point or interpolated `at` a transient time or DC
    /// sweep coordinate.
    Value {
        /// Target signal.
        signal: Signal,
        /// Axis coordinate in SI base units, or `None` for an operating point.
        at: Option<f64>,
    },
    /// Small-signal gain at frequency `at` in decibels.
    Gain {
        /// Transfer function being measured.
        transfer: Transfer,
        /// Query frequency in hertz.
        at: f64,
    },
    /// Unwrapped small-signal phase at frequency `at` in degrees, starting from the lowest
    /// frequency.
    Phase {
        /// Transfer function being measured.
        transfer: Transfer,
        /// Query frequency in hertz.
        at: f64,
    },
    /// First frequency where gain drops `drop` decibels below its lowest-frequency value.
    Bandwidth {
        /// Transfer function being measured.
        transfer: Transfer,
        /// Gain drop in decibels.
        drop: f64,
    },
    /// First frequency where gain falls through 0 dB.
    Crossover {
        /// Transfer function being measured.
        transfer: Transfer,
    },
    /// Phase margin (`180° + phase`) in degrees at the first 0 dB unity-gain crossing.
    PhaseMargin {
        /// Transfer function being measured.
        transfer: Transfer,
    },
    /// Gain margin (`-gain`) in decibels at the first -180° phase crossing.
    GainMargin {
        /// Transfer function being measured.
        transfer: Transfer,
    },
    /// Maximum value of `signal` across `window` in a transient or DC sweep.
    Maximum {
        /// Target signal.
        signal: Signal,
        /// Evaluation window.
        window: Window,
    },
    /// Minimum value of `signal` across `window` in a transient or DC sweep.
    Minimum {
        /// Target signal.
        signal: Signal,
        /// Evaluation window.
        window: Window,
    },
    /// Peak-to-peak amplitude (`max - min`) of `signal` across `window` in a transient or DC
    /// sweep.
    PeakToPeak {
        /// Target signal.
        signal: Signal,
        /// Evaluation window.
        window: Window,
    },
    /// Trapezoidal mean of `signal` across `window` in a transient or DC sweep.
    Mean {
        /// Target signal.
        signal: Signal,
        /// Evaluation window.
        window: Window,
    },
    /// Root-mean-square value of `signal` across `window` via trapezoidal integration.
    Rms {
        /// Target signal.
        signal: Signal,
        /// Evaluation window.
        window: Window,
    },
    /// Transition duration of `signal` between fractions `lower` and `upper` of its step
    /// across `window`.
    RiseTime {
        /// Target signal.
        signal: Signal,
        /// Evaluation window.
        window: Window,
        /// Lower step fraction.
        lower: f64,
        /// Upper step fraction.
        upper: f64,
    },
    /// Percentage overshoot of `signal` past its final value across `window`.
    Overshoot {
        /// Target signal.
        signal: Signal,
        /// Evaluation window.
        window: Window,
    },
    /// Elapsed time from window start until `signal` remains within `band` of its step around
    /// its final value.
    SettlingTime {
        /// Target signal.
        signal: Signal,
        /// Evaluation window.
        window: Window,
        /// Relative settling band as a fraction of step height.
        band: f64,
    },
    /// First time `signal` crosses `threshold` in direction `edge` within `window`.
    Crossing {
        /// Target signal.
        signal: Signal,
        /// Evaluation window.
        window: Window,
        /// Crossing threshold in SI base units.
        threshold: f64,
        /// Crossing direction.
        edge: Edge,
    },
    /// Average rising-edge frequency of `signal` across `threshold` within `window`.
    Frequency {
        /// Target signal.
        signal: Signal,
        /// Evaluation window.
        window: Window,
        /// Crossing threshold, defaulting to the midpoint of the signal's range.
        threshold: Option<f64>,
    },
    /// Total harmonic distortion of `signal` in decibels across harmonics `2..=harmonics` of
    /// `fundamental`, evaluated over the final period or all whole periods in `window`.
    Distortion {
        /// Target signal.
        signal: Signal,
        /// Evaluation window.
        window: Window,
        /// Positive fundamental frequency in hertz.
        fundamental: f64,
        /// Highest harmonic index included.
        harmonics: usize,
    },
    /// Square-root noise density at frequency `at` interpolated by power law, optionally
    /// input-referred.
    NoiseDensity {
        /// Query frequency in hertz.
        at: f64,
        /// Flag selecting input-referred noise over output noise.
        referred: bool,
    },
    /// Root-mean-square noise integrated over `[start, stop]` using piecewise power-law
    /// segments.
    IntegratedNoise {
        /// Lower frequency bound in hertz, defaulting to the lowest frequency.
        start: Option<f64>,
        /// Upper frequency bound in hertz, defaulting to the highest frequency.
        stop: Option<f64>,
        /// Flag selecting input-referred noise over output noise.
        referred: bool,
    },
    /// Steady-state period of a periodic steady-state analysis, in seconds.
    Period,
    /// Count of distinct DC solutions returned by an operating-points search.
    Solutions,
    /// Minimum damping ratio across all poles of a pole-zero analysis.
    Damping,
    /// Lowest natural frequency across all poles of a pole-zero analysis, in hertz.
    DominantPole,
}

/// Row-indexed view of unknown values across analysis points.
pub trait Table<T> {
    /// Returns the number of points (rows) in the table.
    fn rows(&self) -> usize;
    /// Returns the entry at `(row, column)`, or `None` if either index is out of bounds.
    fn value(&self, row: usize, column: usize) -> Option<T>;
}

impl<T: Copy> Table<T> for Vec<Vec<T>> {
    fn rows(&self) -> usize {
        self.len()
    }

    fn value(&self, row: usize, column: usize) -> Option<T> {
        self.get(row)?.get(column).copied()
    }
}

/// The numbers of an analysis's result that a measure reads.
#[derive(Clone, Copy)]
pub enum AnalysisResult<'a> {
    /// An operating point: each unknown's value.
    OperatingPoint(&'a [f64]),
    /// A search for operating points: how many it found.
    OperatingPoints(usize),
    /// A DC sweep: its swept values, and the unknowns' values at each.
    DcSweep {
        /// The values swept.
        swept: &'a [f64],
        /// The values, one row per swept value.
        values: &'a dyn Table<f64>,
    },
    /// A transient: its times, and the unknowns' values at each.
    Transient {
        /// The times.
        times: &'a [f64],
        /// The values, one row per time.
        values: &'a dyn Table<f64>,
    },
    /// One period of a periodic steady state: its times, unknown values, and steady-state
    /// period in seconds.
    PeriodicSteadyState {
        /// The times across one period.
        times: &'a [f64],
        /// The values, one row per time.
        values: &'a dyn Table<f64>,
        /// The steady-state period in seconds.
        period: f64,
    },
    /// A small-signal response: its frequencies, and the unknowns' phasors at each.
    Ac {
        /// The frequencies.
        frequencies: &'a [f64],
        /// The phasors, one row per frequency.
        values: &'a dyn Table<Complex64>,
    },
    /// A loop gain: its frequencies, and the gain at each.
    Stability {
        /// The frequencies.
        frequencies: &'a [f64],
        /// The gains.
        gains: &'a [Complex64],
    },
    /// A noise analysis: its frequencies, the output's power density at each, and the gain
    /// from its source, if it names one.
    Noise {
        /// The frequencies.
        frequencies: &'a [f64],
        /// The densities.
        densities: &'a [f64],
        /// The gains from the source.
        gains: Option<&'a [Complex64]>,
    },
    /// A pole-zero analysis: its poles.
    PoleZero(&'a [Complex64]),
}

impl<'a> AnalysisResult<'a> {
    /// Extracts a measurable view of `result`, returning `None` for sensitivity, mismatch,
    /// and transfer-function results.
    #[must_use]
    pub fn from_result(result: &'a crate::AnalysisResult) -> Option<Self> {
        Some(match result {
            crate::AnalysisResult::OperatingPoint(point) => point.into(),
            crate::AnalysisResult::DcSweep(sweep) => sweep.into(),
            crate::AnalysisResult::Ac(response) => response.into(),
            crate::AnalysisResult::Transient(waveforms) => waveforms.into(),
            crate::AnalysisResult::PeriodicSteadyState(found) => found.into(),
            crate::AnalysisResult::Sensitivity(_)
            | crate::AnalysisResult::Mismatch(_)
            | crate::AnalysisResult::TransferFunction(_) => {
                return None;
            }
            crate::AnalysisResult::Stability(gain) => gain.into(),
            crate::AnalysisResult::Noise(spectrum) => spectrum.into(),
            crate::AnalysisResult::PoleZero(found) => found.into(),
            crate::AnalysisResult::OperatingPoints(points) => Self::OperatingPoints(points.len()),
        })
    }
}

impl<'a> From<&'a OperatingPointResult> for AnalysisResult<'a> {
    fn from(point: &'a OperatingPointResult) -> Self {
        Self::OperatingPoint(&point.values)
    }
}

impl<'a> From<&'a DcSweepResult> for AnalysisResult<'a> {
    fn from(sweep: &'a DcSweepResult) -> Self {
        Self::DcSweep {
            swept: &sweep.swept,
            values: &sweep.values,
        }
    }
}

impl<'a> From<&'a TransientResult> for AnalysisResult<'a> {
    fn from(waveforms: &'a TransientResult) -> Self {
        Self::Transient {
            times: &waveforms.times,
            values: &waveforms.values,
        }
    }
}

impl<'a> From<&'a PeriodicSteadyStateResult> for AnalysisResult<'a> {
    fn from(found: &'a PeriodicSteadyStateResult) -> Self {
        Self::PeriodicSteadyState {
            times: &found.transient.times,
            values: &found.transient.values,
            period: found.period,
        }
    }
}

impl<'a> From<&'a AcResult> for AnalysisResult<'a> {
    fn from(response: &'a AcResult) -> Self {
        Self::Ac {
            frequencies: &response.frequencies,
            values: &response.values,
        }
    }
}

impl<'a> From<&'a StabilityResult> for AnalysisResult<'a> {
    fn from(gain: &'a StabilityResult) -> Self {
        Self::Stability {
            frequencies: &gain.frequencies,
            gains: &gain.gains,
        }
    }
}

impl<'a> From<&'a NoiseResult> for AnalysisResult<'a> {
    fn from(spectrum: &'a NoiseResult) -> Self {
        Self::Noise {
            frequencies: &spectrum.frequencies,
            densities: &spectrum.densities,
            gains: spectrum.input.as_ref().map(|input| input.gains.as_slice()),
        }
    }
}

impl<'a> From<&'a PoleZeroResult> for AnalysisResult<'a> {
    fn from(found: &'a PoleZeroResult) -> Self {
        Self::PoleZero(&found.poles)
    }
}

/// Role of an axis boundary or query point reported in [`MeasureError::Outside`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Point {
    /// Query coordinate for a point evaluation (value, gain, phase, or noise density).
    At,
    /// Lower bound of a window or frequency band.
    Start,
    /// Upper bound of a window or frequency band.
    Stop,
}

impl std::fmt::Display for Point {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::At => "point read at",
            Self::Start => "start",
            Self::Stop => "stop",
        })
    }
}

/// Failure condition matching Python's `math.fsum` exceptions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SumError {
    /// Partial sum of finite values overflowed `f64`.
    #[error("intermediate overflow in fsum")]
    Overflow,
    /// Input contained both positive and negative infinity.
    #[error("-inf + inf in fsum")]
    Indeterminate,
}

/// Reason a [`Measure`] could not be evaluated on an [`AnalysisResult`].
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum MeasureError {
    /// Analysis result variant incompatible with this measure.
    #[error("the measure does not read this kind of result")]
    Kind,
    /// Unknown index missing from the analysis result.
    #[error("the result has no unknown at index {0}")]
    Unknown(usize),
    /// Mismatch between axis length and row count.
    #[error("the result's values are not as many as its points")]
    LengthMismatch,
    /// Analysis result containing zero data points.
    #[error("the result holds no points")]
    Empty,
    /// Query point outside `[first, last]` beyond [`ON_AXIS`] tolerance.
    #[error("the {point}, outside the axis from {first:?} to {last:?}")]
    Outside {
        /// Role of the out-of-range point.
        point: Point,
        /// Lower axis bound.
        first: f64,
        /// Upper axis bound.
        last: f64,
    },
    /// Window whose start is not strictly less than its stop.
    #[error("the window is empty")]
    EmptyWindow,
    /// Signal whose starting and ending window values are equal, giving a zero step.
    #[error("the signal ends the window where it starts it, at {0:?}")]
    Flat(f64),
    /// Signal failing to cross the target threshold (edge level, 0 dB, -180°, or step level).
    #[error("the signal never crosses the threshold the measure looks for")]
    Uncrossed,
    /// Gain remaining within `drop` decibels of its lowest-frequency value across the sweep.
    #[error("the gain stays within the drop of its gain at {0:?} Hz")]
    Stays(f64),
    /// Signal rising through the frequency threshold fewer than twice.
    #[error("the signal rises through {0:?} fewer than twice")]
    Rises(f64),
    /// Zero denominator phasor encountered while evaluating a signal ratio transfer.
    #[error("the divisor vanishes in the response")]
    Vanishes,
    /// Non-positive frequency encountered in a log-frequency small-signal transfer.
    #[error("a response is read over log frequency, so not at {0:?} Hz")]
    Unpositive(f64),
    /// Window shorter than one full period of the distortion fundamental.
    #[error("the window spans less than a period of the fundamental")]
    Short,
    /// Zero amplitude at the distortion fundamental frequency.
    #[error("the signal has no component at the fundamental")]
    Silent,
    /// Distortion resampling grid exceeding available memory or index range.
    #[error("the distortion's samples would not fit in memory")]
    Memory,
    /// Noise integration band with start not strictly below stop.
    #[error("the band is empty")]
    EmptyBand,
    /// Noise integration band `[start, stop]` empty or outside `[first, last]`.
    #[error(
        "a band from {start:?} Hz to {stop:?} Hz does not lie within the frequencies, {first:?} Hz to {last:?} Hz"
    )]
    Band {
        /// Band lower frequency in hertz.
        start: f64,
        /// Band upper frequency in hertz.
        stop: f64,
        /// Lowest available frequency in hertz.
        first: f64,
        /// Highest available frequency in hertz.
        last: f64,
    },
    /// Exact summation error matching Python's `math.fsum` while integrating noise segments.
    #[error(transparent)]
    Sum(SumError),
    /// Pole-zero result containing zero poles.
    #[error("the circuit has no poles")]
    NoPoles,
    /// Input-referred noise requested without a source gain vector.
    #[error("the noise analysis names no source to refer its noise to")]
    Unreferred,
    /// Measure or evaluation point is discrete or singular and has no finite derivative.
    #[error("the measure is not differentiable at this result")]
    NotDifferentiable,
}

/// Sparse reverse-mode derivative of a scalar [`Measure`] with respect to its
/// [`AnalysisResult`] inputs.
///
/// Complex entries follow the engine's bilinear convention `v = dL/da - j dL/db` for a real
/// scalar `L` of complex input `z = a + jb`, so `Re(v * dz) = dL`.
#[derive(Debug, Clone, PartialEq)]
pub enum MeasureCotangent {
    /// Cotangent on a real unknown table (`OperatingPoint`, `DcSweep`, `Transient`, or
    /// `PeriodicSteadyState`).
    Real {
        /// Nonzero `(point, unknown, weight)` entries in ascending `(point, unknown)` order.
        entries: Vec<(usize, usize, f64)>,
        /// Derivative with respect to an autonomous oscillator's period in seconds.
        period: f64,
    },
    /// Bilinear complex cotangent on an AC response's unknown phasor table.
    Complex {
        /// Nonzero `(point, unknown, weight)` entries in ascending `(point, unknown)` order.
        entries: Vec<(usize, usize, Complex64)>,
    },
    /// Bilinear complex cotangent on a stability analysis's loop-gain vector.
    Loop {
        /// Nonzero `(point, weight)` entries in ascending `point` order.
        entries: Vec<(usize, Complex64)>,
    },
    /// Cotangent on a noise analysis's output and input-referred power spectral densities.
    Noise {
        /// Nonzero `(point, weight)` entries with respect to output power densities
        /// (`NOISE_DENSITY`, output column `0`).
        densities: Vec<(usize, f64)>,
        /// Nonzero `(point, weight)` entries with respect to input-referred power densities
        /// (`REFERRED_NOISE`, output column `1`).
        referred: Vec<(usize, f64)>,
    },
    /// Bilinear complex cotangent on a pole-zero analysis's poles vector.
    Poles {
        /// Nonzero `(pole_index, weight)` entries in ascending `pole_index` order.
        entries: Vec<(usize, Complex64)>,
    },
}

impl MeasureCotangent {
    /// Returns a copy of this cotangent with every weight scaled by `factor`.
    #[must_use]
    pub fn scale(&self, factor: f64) -> Self {
        if factor == 0.0 {
            return match self {
                Self::Real { .. } => Self::Real {
                    entries: Vec::new(),
                    period: 0.0,
                },
                Self::Complex { .. } => Self::Complex {
                    entries: Vec::new(),
                },
                Self::Loop { .. } => Self::Loop {
                    entries: Vec::new(),
                },
                Self::Noise { .. } => Self::Noise {
                    densities: Vec::new(),
                    referred: Vec::new(),
                },
                Self::Poles { .. } => Self::Poles {
                    entries: Vec::new(),
                },
            };
        }
        match self {
            Self::Real { entries, period } => Self::Real {
                entries: entries
                    .iter()
                    .map(|&(point, unknown, weight)| (point, unknown, weight * factor))
                    .collect(),
                period: period * factor,
            },
            Self::Complex { entries } => Self::Complex {
                entries: entries
                    .iter()
                    .map(|&(point, unknown, weight)| (point, unknown, weight * factor))
                    .collect(),
            },
            Self::Loop { entries } => Self::Loop {
                entries: entries
                    .iter()
                    .map(|&(point, weight)| (point, weight * factor))
                    .collect(),
            },
            Self::Noise {
                densities,
                referred,
            } => Self::Noise {
                densities: densities
                    .iter()
                    .map(|&(point, weight)| (point, weight * factor))
                    .collect(),
                referred: referred
                    .iter()
                    .map(|&(point, weight)| (point, weight * factor))
                    .collect(),
            },
            Self::Poles { entries } => Self::Poles {
                entries: entries
                    .iter()
                    .map(|&(index, weight)| (index, weight * factor))
                    .collect(),
            },
        }
    }

    /// Expands a [`Self::Real`] or [`Self::Noise`] cotangent into an engine real [`Cotangent`]
    /// with rows of length `width`, returning `None` for other variants.
    #[must_use]
    pub fn to_real_cotangent(&self, width: usize) -> Option<Cotangent<f64>> {
        let mut points: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
        let period = match self {
            Self::Real { entries, period } => {
                for &(point, unknown, weight) in entries {
                    if weight == 0.0 {
                        continue;
                    }
                    let row = points.entry(point).or_insert_with(|| vec![0.0; width]);
                    if let Some(slot) = row.get_mut(unknown) {
                        *slot += weight;
                    }
                }
                *period
            }
            Self::Noise {
                densities,
                referred,
            } => {
                if width == 0 || (!referred.is_empty() && width < 2) {
                    return None;
                }
                for &(point, weight) in densities {
                    if weight != 0.0 {
                        let row = points.entry(point).or_insert_with(|| vec![0.0; width]);
                        row[0] += weight;
                    }
                }
                for &(point, weight) in referred {
                    if weight != 0.0 {
                        let row = points.entry(point).or_insert_with(|| vec![0.0; width]);
                        row[1] += weight;
                    }
                }
                0.0
            }
            Self::Complex { .. } | Self::Loop { .. } | Self::Poles { .. } => return None,
        };
        points.retain(|_, row| row.iter().any(|&value| value != 0.0));
        Some(Cotangent { points, period })
    }

    /// Expands a [`Self::Complex`] or [`Self::Loop`] cotangent into an engine complex
    /// [`Cotangent`] with rows of length `width`, returning `None` for other variants.
    #[must_use]
    pub fn to_complex_cotangent(&self, width: usize) -> Option<Cotangent<Complex64>> {
        let zero = Complex64::default();
        let mut points: BTreeMap<usize, Vec<Complex64>> = BTreeMap::new();
        match self {
            Self::Complex { entries } => {
                for &(point, unknown, weight) in entries {
                    if weight == zero {
                        continue;
                    }
                    let row = points.entry(point).or_insert_with(|| vec![zero; width]);
                    if let Some(slot) = row.get_mut(unknown) {
                        *slot += weight;
                    }
                }
            }
            Self::Loop { entries } => {
                if width == 0 {
                    return None;
                }
                for &(point, weight) in entries {
                    if weight != zero {
                        let row = points.entry(point).or_insert_with(|| vec![zero; width]);
                        row[0] += weight;
                    }
                }
            }
            Self::Real { .. } | Self::Noise { .. } | Self::Poles { .. } => return None,
        }
        points.retain(|_, row| row.iter().any(|&value| value != zero));
        Some(Cotangent {
            points,
            period: 0.0,
        })
    }

    /// Expands a differentiable cotangent (`Real`, `Complex`, `Loop`, or `Noise`) into
    /// single-entry [`Cotangents`] suitable for
    /// [`Simulator::pull_back`](crate::Simulator::pull_back), returning `None` for variants
    /// without a reverse-mode VJP (`Poles`).
    #[must_use]
    pub fn to_cotangents(&self, width: usize) -> Option<Cotangents> {
        match self {
            Self::Real { .. } | Self::Noise { .. } => self
                .to_real_cotangent(width)
                .map(|cotangent| Cotangents::Real(vec![cotangent])),
            Self::Complex { .. } | Self::Loop { .. } => self
                .to_complex_cotangent(width)
                .map(|cotangent| Cotangents::Complex(vec![cotangent])),
            Self::Poles { .. } => None,
        }
    }
}

impl Measure {
    /// Remaps all unknown indices in this measure using `column` for a restructured circuit.
    ///
    /// # Errors
    ///
    /// Returns [`MeasureError::Unknown`] with the first unknown index for which `column`
    /// returns `None`.
    pub fn moved(&self, column: impl Fn(usize) -> Option<usize>) -> Result<Self, MeasureError> {
        let at = |signal: Signal| -> Result<Signal, MeasureError> {
            let moved = |unknown| column(unknown).ok_or(MeasureError::Unknown(unknown));
            Ok(Signal {
                unknown: moved(signal.unknown)?,
                reference: signal.reference.map(moved).transpose()?,
            })
        };
        let through = |transfer: Transfer| -> Result<Transfer, MeasureError> {
            Ok(match transfer {
                Transfer::Loop => Transfer::Loop,
                Transfer::Signal { signal, over } => Transfer::Signal {
                    signal: at(signal)?,
                    over: over.map(at).transpose()?,
                },
            })
        };
        let mut moved = *self;
        match &mut moved {
            Self::Value { signal, .. }
            | Self::Maximum { signal, .. }
            | Self::Minimum { signal, .. }
            | Self::PeakToPeak { signal, .. }
            | Self::Mean { signal, .. }
            | Self::Rms { signal, .. }
            | Self::RiseTime { signal, .. }
            | Self::Overshoot { signal, .. }
            | Self::SettlingTime { signal, .. }
            | Self::Crossing { signal, .. }
            | Self::Frequency { signal, .. }
            | Self::Distortion { signal, .. } => *signal = at(*signal)?,
            Self::Gain { transfer, .. }
            | Self::Phase { transfer, .. }
            | Self::Bandwidth { transfer, .. }
            | Self::Crossover { transfer }
            | Self::PhaseMargin { transfer }
            | Self::GainMargin { transfer } => *transfer = through(*transfer)?,
            Self::NoiseDensity { .. }
            | Self::IntegratedNoise { .. }
            | Self::Period
            | Self::Solutions
            | Self::Damping
            | Self::DominantPole => {}
        }
        Ok(moved)
    }

    /// Evaluates this measure on `results` in the unit documented on its variant.
    ///
    /// # Errors
    ///
    /// Returns the [`MeasureError`] describing why `results` cannot satisfy this measure.
    pub fn measure(&self, results: &AnalysisResult<'_>) -> Result<f64, MeasureError> {
        match *self {
            Self::Value { signal, at } => value(signal, at, results),
            Self::Gain { transfer, at } => gain(transfer, at, results),
            Self::Phase { transfer, at } => phase(transfer, at, results),
            Self::Bandwidth { transfer, drop } => bandwidth(transfer, drop, results),
            Self::Crossover { transfer } => {
                let (frequencies, transfer) = read_transfer(transfer, results)?;
                Ok(unity(&logarithms(&frequencies), &transfer)?.exp())
            }
            Self::PhaseMargin { transfer } => phase_margin(transfer, results),
            Self::GainMargin { transfer } => gain_margin(transfer, results),
            Self::Maximum { signal, window } => {
                let (_, values) = windowed(signal, window, results, Scope::Any)?;
                Ok(arithmetic::maximum(&values))
            }
            Self::Minimum { signal, window } => {
                let (_, values) = windowed(signal, window, results, Scope::Any)?;
                Ok(arithmetic::minimum(&values))
            }
            Self::PeakToPeak { signal, window } => {
                let (_, values) = windowed(signal, window, results, Scope::Any)?;
                Ok(arithmetic::maximum(&values) - arithmetic::minimum(&values))
            }
            Self::Mean { signal, window } => {
                let (points, values) = windowed(signal, window, results, Scope::Any)?;
                Ok(mean(&points, &values))
            }
            Self::Rms { signal, window } => {
                let (points, values) = windowed(signal, window, results, Scope::Any)?;
                let squares: Vec<f64> = values.iter().map(|value| value * value).collect();
                Ok(mean(&points, &squares).sqrt())
            }
            Self::RiseTime {
                signal,
                window,
                lower,
                upper,
            } => rise_time(signal, window, lower, upper, results),
            Self::Overshoot { signal, window } => overshoot(signal, window, results),
            Self::SettlingTime {
                signal,
                window,
                band,
            } => settling_time(signal, window, band, results),
            Self::Crossing {
                signal,
                window,
                threshold,
                edge,
            } => {
                let (times, values) = windowed(signal, window, results, Scope::Transient)?;
                arithmetic::crossings(&times, &values, threshold, edge)
                    .next()
                    .ok_or(MeasureError::Uncrossed)
            }
            Self::Frequency {
                signal,
                window,
                threshold,
            } => frequency(signal, window, threshold, results),
            Self::Distortion {
                signal,
                window,
                fundamental,
                harmonics,
            } => {
                let (times, values) = windowed(signal, window, results, Scope::Transient)?;
                let span = if window.start.is_some() {
                    PeriodSpan::Whole
                } else {
                    PeriodSpan::Last
                };
                distortion(&times, &values, fundamental, harmonics, span)
            }
            Self::NoiseDensity { at, referred } => {
                let (frequencies, densities) = noise(results, referred)?;
                let located = Axis(frequencies).locate(at, Point::At)?;
                Ok(density(frequencies, &densities, located)?.sqrt())
            }
            Self::IntegratedNoise {
                start,
                stop,
                referred,
            } => integrated_noise(start, stop, referred, results),
            Self::Period => match *results {
                AnalysisResult::PeriodicSteadyState { period, .. } => Ok(period),
                _ => Err(MeasureError::Kind),
            },
            Self::Solutions => match results {
                AnalysisResult::OperatingPoints(found) => Ok(count(*found)),
                _ => Err(MeasureError::Kind),
            },
            Self::Damping => Ok(arithmetic::minimum(&dampings(poles(results)?))),
            Self::DominantPole => Ok(arithmetic::minimum(&natural_frequencies(poles(results)?))),
        }
    }

    /// Evaluates this measure on `results` alongside its exact sparse reverse-mode
    /// [`MeasureCotangent`].
    ///
    /// The returned scalar matches [`Measure::measure`] on the same `results`.
    ///
    /// # Errors
    ///
    /// Returns [`MeasureError::NotDifferentiable`] for discrete measures or singular points
    /// (such as zero RMS, zero transfer phasor, or zero pole magnitude), and any evaluation
    /// error returned by [`Measure::measure`].
    pub fn cotangent(
        &self,
        results: &AnalysisResult<'_>,
    ) -> Result<(f64, MeasureCotangent), MeasureError> {
        cotangent::evaluate(*self, results)
    }
}

/// Evaluates the gain of `transfer` at frequency `at` in decibels.
fn gain(transfer: Transfer, at: f64, results: &AnalysisResult<'_>) -> Result<f64, MeasureError> {
    let (frequencies, transfer) = read_transfer(transfer, results)?;
    let point = Axis(&frequencies).locate(at, Point::At)?.ln();
    Ok(interpolated(
        point,
        &logarithms(&frequencies),
        &decibels(&transfer),
    ))
}

/// Evaluates the unwrapped phase of `transfer` at frequency `at` in degrees.
fn phase(transfer: Transfer, at: f64, results: &AnalysisResult<'_>) -> Result<f64, MeasureError> {
    let (frequencies, transfer) = read_transfer(transfer, results)?;
    let point = Axis(&frequencies).locate(at, Point::At)?.ln();
    let phases = arithmetic::phases(&transfer);
    Ok(interpolated(point, &logarithms(&frequencies), &phases))
}

/// Finds the first frequency where the gain of `transfer` drops `drop` decibels below its
/// lowest-frequency value.
fn bandwidth(
    transfer: Transfer,
    drop: f64,
    results: &AnalysisResult<'_>,
) -> Result<f64, MeasureError> {
    let (frequencies, transfer) = read_transfer(transfer, results)?;
    let gains = decibels(&transfer);
    let logarithms = logarithms(&frequencies);
    arithmetic::crossings(&logarithms, &gains, gains[0] - drop, Edge::Falling)
        .next()
        .map(f64::exp)
        .ok_or(MeasureError::Stays(frequencies[0]))
}

/// Computes phase margin (`180° + phase`) at the first 0 dB gain crossing of `transfer`.
fn phase_margin(transfer: Transfer, results: &AnalysisResult<'_>) -> Result<f64, MeasureError> {
    let (frequencies, transfer) = read_transfer(transfer, results)?;
    let logarithms = logarithms(&frequencies);
    let unity = unity(&logarithms, &transfer)?;
    let phases = arithmetic::phases(&transfer);
    Ok(180.0 + interpolated(unity, &logarithms, &phases))
}

/// Computes gain margin (`-gain`) in decibels at the first -180° phase crossing of `transfer`.
fn gain_margin(transfer: Transfer, results: &AnalysisResult<'_>) -> Result<f64, MeasureError> {
    let (frequencies, transfer) = read_transfer(transfer, results)?;
    let logarithms = logarithms(&frequencies);
    let phases = arithmetic::phases(&transfer);
    let inverted = arithmetic::crossings(&logarithms, &phases, -180.0, Edge::Falling)
        .next()
        .ok_or(MeasureError::Uncrossed)?;
    Ok(-interpolated(inverted, &logarithms, &decibels(&transfer)))
}

/// Computes the transition duration of `signal` between fractions `lower` and `upper` of its
/// step across `window`.
fn rise_time(
    signal: Signal,
    window: Window,
    lower: f64,
    upper: f64,
    results: &AnalysisResult<'_>,
) -> Result<f64, MeasureError> {
    let (times, values, step) = stepped(signal, window, results)?;
    let edge = if step > 0.0 {
        Edge::Rising
    } else {
        Edge::Falling
    };
    let first = arithmetic::crossings(&times, &values, values[0] + lower * step, edge)
        .next()
        .ok_or(MeasureError::Uncrossed)?;
    let second = arithmetic::crossings(&times, &values, values[0] + upper * step, edge)
        .find(|&crossed| crossed >= first)
        .ok_or(MeasureError::Uncrossed)?;
    Ok(second - first)
}

/// Computes percentage overshoot of `signal` past its final value across `window`.
fn overshoot(
    signal: Signal,
    window: Window,
    results: &AnalysisResult<'_>,
) -> Result<f64, MeasureError> {
    let (_, values, step) = stepped(signal, window, results)?;
    let peak = if step > 0.0 {
        arithmetic::maximum(&values)
    } else {
        arithmetic::minimum(&values)
    };
    let passed = (peak - values[values.len() - 1]) / step;
    // Python's `max(0.0, passed)` keeps zero unless `passed` exceeds it.
    Ok(100.0 * if passed > 0.0 { passed } else { 0.0 })
}

/// Computes the settling time from window start until `signal` stays within `band` of its
/// step around its final value.
fn settling_time(
    signal: Signal,
    window: Window,
    band: f64,
    results: &AnalysisResult<'_>,
) -> Result<f64, MeasureError> {
    let (times, values, step) = stepped(signal, window, results)?;
    let last = values[values.len() - 1];
    let within = band * step.abs();
    let excess: Vec<f64> = values
        .iter()
        .map(|value| (value - last).abs() - within)
        .collect();
    let Some(outside) = excess.iter().rposition(|&excess| excess > 0.0) else {
        return Ok(0.0);
    };
    // The last sample lies inside the band, so `outside + 1` exists.
    let (Some(&next), Some(&after)) = (excess.get(outside + 1), times.get(outside + 1)) else {
        return Err(MeasureError::LengthMismatch);
    };
    let fraction = excess[outside] / (excess[outside] - next);
    let settled = times[outside] + fraction * (after - times[outside]);
    Ok(settled - times[0])
}

/// Computes average rising-edge frequency of `signal` across `threshold` (or signal midpoint
/// when `None`).
fn frequency(
    signal: Signal,
    window: Window,
    threshold: Option<f64>,
    results: &AnalysisResult<'_>,
) -> Result<f64, MeasureError> {
    let (times, values) = windowed(signal, window, results, Scope::Transient)?;
    #[expect(
        clippy::manual_midpoint,
        reason = "numpy's `(max + min) / 2`, which overflows where `f64::midpoint` would not"
    )]
    let middle = threshold
        .unwrap_or_else(|| (arithmetic::maximum(&values) + arithmetic::minimum(&values)) / 2.0);
    let rises: Vec<f64> = arithmetic::crossings(&times, &values, middle, Edge::Rising).collect();
    match (rises.first(), rises.last()) {
        (Some(&first), Some(&last)) if rises.len() >= 2 => {
            Ok(count(rises.len() - 1) / (last - first))
        }
        _ => Err(MeasureError::Rises(middle)),
    }
}

/// Computes integrated RMS noise over `[start, stop]`, optionally input-referred.
fn integrated_noise(
    start: Option<f64>,
    stop: Option<f64>,
    referred: bool,
    results: &AnalysisResult<'_>,
) -> Result<f64, MeasureError> {
    let (frequencies, densities) = noise(results, referred)?;
    let order = arithmetic::order(frequencies);
    let sorted: Vec<f64> = order.iter().map(|&index| frequencies[index]).collect();
    let axis = Axis(&sorted);
    let low = start
        .map(|start| axis.locate(start, Point::Start))
        .transpose()?;
    let high = stop
        .map(|stop| axis.locate(stop, Point::Stop))
        .transpose()?;
    if let (Some(low), Some(high)) = (low, high)
        && low >= high
    {
        return Err(MeasureError::EmptyBand);
    }
    Ok(integrate(frequencies, &densities, low, high)?.sqrt())
}

/// Converts every complex entry of `transfer` to decibels.
fn decibels(transfer: &[Complex64]) -> Vec<f64> {
    transfer.iter().map(|&z| arithmetic::decibels(z)).collect()
}

/// Converts `count` to `f64` (exact for counts below `2⁵³`).
#[expect(
    clippy::cast_precision_loss,
    reason = "a count of points or solutions is far below 2⁵³, where floats hold every integer"
)]
fn count(count: usize) -> f64 {
    count as f64
}

impl Signal {
    /// Extracts this signal's per-row values from `table`, subtracting `self.reference` if set.
    fn trace<T: Copy + std::ops::Sub<Output = T>>(
        self,
        table: &dyn Table<T>,
    ) -> Result<Vec<T>, MeasureError> {
        (0..table.rows())
            .map(|row| {
                let value = table
                    .value(row, self.unknown)
                    .ok_or(MeasureError::Unknown(self.unknown))?;
                match self.reference {
                    None => Ok(value),
                    Some(reference) => table
                        .value(row, reference)
                        .map(|base| value - base)
                        .ok_or(MeasureError::Unknown(reference)),
                }
            })
            .collect()
    }
}

/// Ordered axis coordinates validated against endpoint rounding tolerance [`ON_AXIS`].
struct Axis<'a>(&'a [f64]);

impl Axis<'_> {
    /// Validates `point` against this axis within [`ON_AXIS`] of its span and clamps boundary
    /// rounding onto the endpoints.
    fn locate(&self, point: f64, which: Point) -> Result<f64, MeasureError> {
        let (Some(&first), Some(&last)) = (self.0.first(), self.0.last()) else {
            return Err(MeasureError::Empty);
        };
        let within = ON_AXIS * (last - first);
        if !(first - within <= point && point <= last + within) {
            return Err(MeasureError::Outside {
                point: which,
                first,
                last,
            });
        }
        // Python's `min(max(point, first), last)` keeps its first argument on ties.
        let point = if first > point { first } else { point };
        Ok(if last < point { last } else { point })
    }
}

/// Interpolates `(points, values)` linearly at scalar coordinate `at` matching `NumPy`'s
/// `interp`.
pub(crate) fn interpolated(at: f64, points: &[f64], values: &[f64]) -> f64 {
    arithmetic::interpolate(&[at], points, values)
        .first()
        .copied()
        .unwrap_or(f64::NAN)
}

/// Computes the natural logarithm of every frequency in `frequencies`.
fn logarithms(frequencies: &[f64]) -> Vec<f64> {
    frequencies.iter().map(|frequency| frequency.ln()).collect()
}

/// Computes the trapezoidal mean of `values` over `points`.
fn mean(points: &[f64], values: &[f64]) -> f64 {
    trapezoid(points, values) / (points[points.len() - 1] - points[0])
}

/// Evaluates `signal` at an operating point or interpolates it at axis coordinate `at`.
fn value(
    signal: Signal,
    at: Option<f64>,
    results: &AnalysisResult<'_>,
) -> Result<f64, MeasureError> {
    match (results, at) {
        (AnalysisResult::OperatingPoint(values), None) => {
            let read = |index: usize| {
                values
                    .get(index)
                    .copied()
                    .ok_or(MeasureError::Unknown(index))
            };
            let value = read(signal.unknown)?;
            match signal.reference {
                None => Ok(value),
                Some(reference) => Ok(value - read(reference)?),
            }
        }
        (
            AnalysisResult::DcSweep { .. }
            | AnalysisResult::Transient { .. }
            | AnalysisResult::PeriodicSteadyState { .. },
            Some(at),
        ) => {
            let (points, values) = traced(signal, results, Scope::Any)?;
            let point = Axis(&points).locate(at, Point::At)?;
            Ok(interpolated(point, &points, &values))
        }
        _ => Err(MeasureError::Kind),
    }
}

/// Accepted sweep-axis categories for trace-based measures.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Scope {
    Any,
    Transient,
}

/// Extracts an ascending `(axis, values)` trace for `signal` from `results` under `scope`,
/// reversing descending DC sweeps.
fn traced<'a>(
    signal: Signal,
    results: &AnalysisResult<'a>,
    scope: Scope,
) -> Result<(Cow<'a, [f64]>, Vec<f64>), MeasureError> {
    let (points, table) = match *results {
        AnalysisResult::Transient { times, values }
        | AnalysisResult::PeriodicSteadyState { times, values, .. } => (times, values),
        AnalysisResult::DcSweep { swept, values } if scope == Scope::Any => (swept, values),
        _ => return Err(MeasureError::Kind),
    };
    let mut values = signal.trace(table)?;
    if values.len() != points.len() {
        return Err(MeasureError::LengthMismatch);
    }
    let (Some(&first), Some(&last)) = (points.first(), points.last()) else {
        return Err(MeasureError::Empty);
    };
    if first > last {
        values.reverse();
        return Ok((points.iter().rev().copied().collect(), values));
    }
    Ok((Cow::Borrowed(points), values))
}

/// Slices and endpoint-interpolates `signal` onto `window` within `results` under `scope`.
fn windowed(
    signal: Signal,
    window: Window,
    results: &AnalysisResult<'_>,
    scope: Scope,
) -> Result<(Vec<f64>, Vec<f64>), MeasureError> {
    let (points, values) = traced(signal, results, scope)?;
    let axis = Axis(&points);
    let mut first = points[0];
    let mut last = points[points.len() - 1];
    if let Some(start) = window.start {
        first = axis.locate(start, Point::Start)?;
    }
    if let Some(stop) = window.stop {
        last = axis.locate(stop, Point::Stop)?;
    }
    if first.partial_cmp(&last) != Some(std::cmp::Ordering::Less) {
        return Err(MeasureError::EmptyWindow);
    }
    let mut inside = vec![first];
    inside.extend(
        points
            .iter()
            .filter(|&&point| point > first && point < last),
    );
    inside.push(last);
    let interpolated = arithmetic::interpolate(&inside, &points, &values);
    Ok((inside, interpolated))
}

/// Extracts windowed transient `(times, values, step)` for `signal`, rejecting zero steps.
fn stepped(
    signal: Signal,
    window: Window,
    results: &AnalysisResult<'_>,
) -> Result<(Vec<f64>, Vec<f64>, f64), MeasureError> {
    let (times, values) = windowed(signal, window, results, Scope::Transient)?;
    let step = values[values.len() - 1] - values[0];
    if step == 0.0 {
        return Err(MeasureError::Flat(values[0]));
    }
    Ok((times, values, step))
}

/// Extracts ascending positive frequencies and complex values for `transfer` from `results`.
fn read_transfer(
    transfer: Transfer,
    results: &AnalysisResult<'_>,
) -> Result<(Vec<f64>, Vec<Complex64>), MeasureError> {
    let (frequencies, values) = match (transfer, *results) {
        (Transfer::Loop, AnalysisResult::Stability { frequencies, gains }) => {
            (frequencies, gains.to_vec())
        }
        (
            Transfer::Signal { signal, over },
            AnalysisResult::Ac {
                frequencies,
                values,
            },
        ) => {
            let phasors = signal.trace(values)?;
            let values = match over {
                None => phasors,
                Some(over) => {
                    let divisors = over.trace(values)?;
                    if divisors
                        .iter()
                        .any(|&divisor| divisor == Complex64::new(0.0, 0.0))
                    {
                        return Err(MeasureError::Vanishes);
                    }
                    phasors
                        .iter()
                        .zip(&divisors)
                        .map(|(&phasor, &divisor)| arithmetic::divide(phasor, divisor))
                        .collect()
                }
            };
            (frequencies, values)
        }
        _ => return Err(MeasureError::Kind),
    };
    if values.len() != frequencies.len() {
        return Err(MeasureError::LengthMismatch);
    }
    if frequencies.is_empty() {
        return Err(MeasureError::Empty);
    }
    let order = arithmetic::order(frequencies);
    let sorted: Vec<f64> = order.iter().map(|&index| frequencies[index]).collect();
    if sorted[0] <= 0.0 {
        return Err(MeasureError::Unpositive(sorted[0]));
    }
    Ok((sorted, order.iter().map(|&index| values[index]).collect()))
}

/// Finds the first log frequency where the gain of `transfer` falls through 0 dB.
fn unity(logarithms: &[f64], transfer: &[Complex64]) -> Result<f64, MeasureError> {
    arithmetic::crossings(logarithms, &decibels(transfer), 0.0, Edge::Falling)
        .next()
        .ok_or(MeasureError::Uncrossed)
}

/// Extracts noise frequencies and densities from `results`, referring densities to the input
/// source when `referred` is true.
fn noise<'a>(
    results: &AnalysisResult<'a>,
    referred: bool,
) -> Result<(&'a [f64], Vec<f64>), MeasureError> {
    let AnalysisResult::Noise {
        frequencies,
        densities,
        gains,
    } = *results
    else {
        return Err(MeasureError::Kind);
    };
    let densities = match (referred, gains) {
        (false, _) => densities.to_vec(),
        (true, Some(gains)) if gains.len() == densities.len() => {
            arithmetic::referred(densities, gains)
        }
        (true, Some(_)) => return Err(MeasureError::LengthMismatch),
        (true, None) => return Err(MeasureError::Unreferred),
    };
    Ok((frequencies, densities))
}

/// Extracts a non-empty slice of poles from a pole-zero `results` value.
fn poles<'a>(results: &AnalysisResult<'a>) -> Result<&'a [Complex64], MeasureError> {
    match *results {
        AnalysisResult::PoleZero([]) => Err(MeasureError::NoPoles),
        AnalysisResult::PoleZero(poles) => Ok(poles),
        _ => Err(MeasureError::Kind),
    }
}

/// Period-selection mode for total harmonic distortion evaluation.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PeriodSpan {
    Last,
    Whole,
}

/// Computes total harmonic distortion in decibels over harmonics `2..=harmonics` of
/// `fundamental` across the final period or all whole periods in `(times, values)`.
///
/// Resamples each period at [`SAMPLES_PER_PERIOD`] uniform points (like SPICE `.four`) and
/// sums each harmonic DFT bin in `NumPy`'s pairwise order.
fn distortion(
    times: &[f64],
    values: &[f64],
    fundamental: f64,
    harmonics: usize,
    span_kind: PeriodSpan,
) -> Result<f64, MeasureError> {
    let span = (times[times.len() - 1] - times[0]) * fundamental;
    let stretched = span * (1.0 + ON_AXIS);
    let periods = match span_kind {
        PeriodSpan::Whole => stretched.floor(),
        PeriodSpan::Last => 1.0,
    };
    if !(periods >= 1.0 && stretched >= periods) {
        return Err(MeasureError::Short);
    }
    let samples = periods * count(SAMPLES_PER_PERIOD);
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a whole number of periods, 1 or more, times the samples of each, checked below \
                  to fit as a count"
    )]
    let length = samples as usize;
    if !arithmetic::same(count(length), samples) {
        return Err(MeasureError::Memory);
    }
    let mut at = Vec::new();
    at.try_reserve_exact(length)
        .map_err(|_| MeasureError::Memory)?;
    let last = times[times.len() - 1];
    let duration = periods / fundamental;
    at.extend((0..length).map(|index| last - duration * (1.0 - count(index) / samples)));
    let waveform = arithmetic::interpolate(&at, times, values);
    // Sample `n` advances harmonic `h` by `(h * n) mod SAMPLES_PER_PERIOD` steps.
    let turns: Vec<(f64, f64)> = (0..SAMPLES_PER_PERIOD).map(turn).collect();
    let amplitude = |harmonic: usize| {
        let phase = |index: usize| {
            turns[(harmonic % SAMPLES_PER_PERIOD) * (index % SAMPLES_PER_PERIOD)
                % SAMPLES_PER_PERIOD]
        };
        let real = arithmetic::sum(length, |index| waveform[index] * phase(index).0);
        let imaginary = arithmetic::sum(length, |index| -waveform[index] * phase(index).1);
        arithmetic::magnitude(Complex64::new(real, imaginary))
    };
    let fundamental = amplitude(1);
    if fundamental == 0.0 {
        return Err(MeasureError::Silent);
    }
    let higher: Vec<f64> = (2..=harmonics).map(amplitude).collect();
    let total = arithmetic::sum(higher.len(), |index| higher[index] * higher[index]).sqrt();
    let ratio = total / fundamental;
    // Python's `max(ratio, 1e-300)` keeps `ratio` unless the floor exceeds it.
    let ratio = if MIN_DISTORTION > ratio {
        MIN_DISTORTION
    } else {
        ratio
    };
    Ok(20.0 * ratio.log10())
}

/// Computes `(cos, sin)` at discrete angle index `step` of [`SAMPLES_PER_PERIOD`] via
/// first-quadrant symmetry so axis points remain exact.
fn turn(step: usize) -> (f64, f64) {
    let quarter = SAMPLES_PER_PERIOD / 4;
    let angle = std::f64::consts::FRAC_PI_2 * count(step % quarter) / count(quarter);
    let (sine, cosine) = angle.sin_cos();
    match step / quarter {
        0 => (cosine, sine),
        1 => (-sine, cosine),
        2 => (-cosine, -sine),
        _ => (sine, -cosine),
    }
}
