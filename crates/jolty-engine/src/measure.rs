//! Measures of results: the numbers specs read from an analysis's result.
//!
//! A [`Measure`] reads one number from the [`AnalysisResult`] of one analysis: a signal's value, a
//! statistic of it over a window, the time of an edge, a gain, a margin, a noise density, a
//! damping ratio. It holds its signals already resolved, each the index of an unknown and of
//! the unknown it is taken relative to, and its times, levels and frequencies in SI base
//! units, so it reads a run's results with no names or units; the Python package resolves
//! them, and says why a measure could not read a result from its [`MeasureError`].
//!
//! Every measure reads the same bits jolty's measures read when numpy computed them, on a
//! platform whose numpy rounds a product and a sum once, as arm64's does: its arithmetic
//! follows numpy's operation for operation, from interpolation to the order of its sums. Only
//! a distortion reads amplitudes numpy took from a fast Fourier transform, which it takes from
//! the discrete Fourier transform's sum at each harmonic instead.
//!
//! ```
//! use jolty_engine::measure::{Measure, AnalysisResult, Signal, Window};
//!
//! // A ramp from 0 to 1 over a second, the first and only unknown.
//! let times = [0.0, 0.5, 1.0];
//! let values = vec![vec![0.0], vec![0.5], vec![1.0]];
//! let results = AnalysisResult::Transient { times: &times, values: &values };
//! let signal = Signal { unknown: 0, reference: None };
//! let window = Window { start: Some(0.25), stop: None };
//! assert_eq!(Measure::Mean { signal, window }.measure(&results), Ok(0.625));
//! ```

mod arithmetic;

use std::borrow::Cow;

use jolty_numerics::Complex64;

use crate::{
    AcResult, DcSweepResult, NoiseResult, OperatingPointResult, PoleZeroResult, StabilityResult,
    TransientResult,
};

pub(crate) use arithmetic::trapezoid;
pub use arithmetic::{dampings, density, integrate, natural_frequencies, referred};

/// How far past an end of an axis, relative to its span, a point still lies on it.
///
/// Origin: jolty's own endpoint rounding margin (`1e-9` of span, about `4.5e6 * f64::EPSILON`),
/// wide enough to absorb binary-decimal unit scaling and log-grid rounding at axis endpoints
/// (such as `stop = N / fundamental` in period counting) while rejecting genuinely out-of-range
/// requests.
pub const ON_AXIS: f64 = 1e-9;

/// The points a distortion samples each period of its fundamental at, interpolated.
///
/// Origin: jolty's own power-of-two sampling grid (`2^10 = 1024`; compare ngspice 47's
/// `DEF_FOURGRIDSIZE = 200` in `src/frontend/fourier.c`, line 32). Divisibility by 4 keeps
/// quarter-turn axes exact (`256` steps per quadrant), and `N / 2 = 512` resolves harmonics
/// 1 through 511 below Nyquist without aliasing.
pub const SAMPLES_PER_PERIOD: usize = 1024;

/// The least ratio of its harmonics to its fundamental a distortion reads, so that a signal
/// without harmonics reads -6000 dB rather than minus infinity.
///
/// Derivation: round power of ten (`20 * log10(1e-300) = -6000 dB`) safely above the smallest
/// positive IEEE 754 binary64 normal `f64::MIN_POSITIVE` (`~2.225e-308`, margin `~4.5e7`),
/// avoiding subnormals and `-inf` in `log10`.
const LEAST_DISTORTION: f64 = 1e-300;

/// An unknown, or a voltage relative to another: their indices among a result's unknowns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Signal {
    /// The unknown's index.
    pub unknown: usize,
    /// The index of the unknown whose value is subtracted from the unknown's, such as the
    /// voltage of a reference net; none for the unknown itself.
    pub reference: Option<usize>,
}

/// From where to where a measure reads a transient's times or a sweep's levels: each end a
/// time or a level in SI base units, by default the axis's own end.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Window {
    /// Where the window starts.
    pub start: Option<f64>,
    /// Where the window stops.
    pub stop: Option<f64>,
}

/// Which way a signal crosses a level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Edge {
    /// From below the level to the level or above.
    Rising,
    /// From above the level to the level or below.
    Falling,
    /// Either way.
    Either,
}

/// What a small-signal measure reads: an output of a small-signal response, over another
/// signal or per unit excitation, or the loop gain of a stability analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transfer {
    /// The loop gain itself.
    Loop,
    /// An output of a response.
    Output {
        /// The output.
        output: Signal,
        /// The signal the output is divided by, if any.
        over: Option<Signal>,
    },
}

/// One number read from the result of one analysis.
///
/// Each reads the kinds of results its documentation names, in SI base units unless it says
/// otherwise. A small-signal measure reads its transfer over log frequency, from the lowest
/// frequency, and interpolates between frequencies linearly in log frequency; any other
/// interpolates linearly between the points of its axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Measure {
    /// The value of `signal` at an operating point, or interpolated `at` a time of a transient
    /// or a level of a DC sweep.
    Value {
        /// The signal.
        signal: Signal,
        /// The time or the level; none at an operating point.
        at: Option<f64>,
    },
    /// The gain at the frequency `at`, in decibels.
    Gain {
        /// What the gain is of.
        transfer: Transfer,
        /// The frequency.
        at: f64,
    },
    /// The phase at the frequency `at`, in degrees, unwrapped from the lowest frequency.
    Phase {
        /// What the phase is of.
        transfer: Transfer,
        /// The frequency.
        at: f64,
    },
    /// The frequency where the gain first falls `drop` decibels below its gain at the lowest
    /// frequency.
    Bandwidth {
        /// What the gain is of.
        transfer: Transfer,
        /// The drop, in decibels.
        drop: f64,
    },
    /// The frequency where the gain first falls through 0 dB.
    Crossover {
        /// What the gain is of.
        transfer: Transfer,
    },
    /// 180° plus the phase where the gain first falls through 0 dB, in degrees.
    PhaseMargin {
        /// What the gain and the phase are of.
        transfer: Transfer,
    },
    /// Minus the gain where the phase first falls through -180°, in decibels.
    GainMargin {
        /// What the gain and the phase are of.
        transfer: Transfer,
    },
    /// The highest value of `signal` in `window`, of a transient or a DC sweep.
    Maximum {
        /// The signal.
        signal: Signal,
        /// The window.
        window: Window,
    },
    /// The lowest value of `signal` in `window`, of a transient or a DC sweep.
    Minimum {
        /// The signal.
        signal: Signal,
        /// The window.
        window: Window,
    },
    /// The highest less the lowest value of `signal` in `window`, of a transient or a DC
    /// sweep.
    PeakToPeak {
        /// The signal.
        signal: Signal,
        /// The window.
        window: Window,
    },
    /// The mean of `signal` in `window`, of a transient or a DC sweep, by the trapezoidal
    /// rule.
    Mean {
        /// The signal.
        signal: Signal,
        /// The window.
        window: Window,
    },
    /// The root mean square of `signal` in `window`, of a transient or a DC sweep, its square
    /// integrated by the trapezoidal rule.
    Rms {
        /// The signal.
        signal: Signal,
        /// The window.
        window: Window,
    },
    /// The time a transient's `signal` takes to rise or fall from `low` to `high` of its step,
    /// from its value at the window's start to its value at the window's end.
    RiseTime {
        /// The signal.
        signal: Signal,
        /// The window.
        window: Window,
        /// The fraction of the step the time starts at.
        low: f64,
        /// The fraction of the step the time ends at.
        high: f64,
    },
    /// How far a transient's `signal` passes its final value, at the window's end, in percent
    /// of its step from its value at the window's start.
    Overshoot {
        /// The signal.
        signal: Signal,
        /// The window.
        window: Window,
    },
    /// The time from the window's start until a transient's `signal` stays within `band` of
    /// its step from its final value, at the window's end.
    SettlingTime {
        /// The signal.
        signal: Signal,
        /// The window.
        window: Window,
        /// The band, a fraction of the step.
        band: f64,
    },
    /// The first time a transient's `signal` crosses `level` the way `edge` says.
    Crossing {
        /// The signal.
        signal: Signal,
        /// The window.
        window: Window,
        /// The level.
        level: f64,
        /// Which way.
        edge: Edge,
    },
    /// How often a transient's `signal` rises through `level`: the rises less one over the
    /// time from the first rise to the last.
    Frequency {
        /// The signal.
        signal: Signal,
        /// The window.
        window: Window,
        /// The level; by default the midpoint of the signal's lowest and highest values.
        level: Option<f64>,
    },
    /// The total harmonic distortion of a transient's `signal`, in decibels: the root sum of
    /// the squares of the amplitudes of harmonics 2 to `harmonics` of the `fundamental` over
    /// the fundamental's amplitude, read over the last period of the fundamental before the
    /// window's stop, or over every whole period of a window with a start.
    Distortion {
        /// The signal.
        signal: Signal,
        /// The window.
        window: Window,
        /// The fundamental frequency, positive.
        fundamental: f64,
        /// The last harmonic counted.
        harmonics: usize,
    },
    /// The square root of a noise analysis's power density at the frequency `at`, a power law
    /// between frequencies; referred, the density at the output over the gain's square.
    NoiseDensity {
        /// The frequency.
        at: f64,
        /// Whether the noise is referred to the analysis's source.
        referred: bool,
    },
    /// The square root of a noise analysis's power density integrated from `start` to
    /// `stop`, by default the lowest and the highest frequency, a power law between
    /// frequencies.
    IntegratedNoise {
        /// Where the band starts.
        start: Option<f64>,
        /// Where the band stops.
        stop: Option<f64>,
        /// Whether the noise is referred to the analysis's source.
        referred: bool,
    },
    /// How many DC solutions a search for operating points found.
    Solutions,
    /// The least damping ratio of a pole-zero analysis's poles.
    Damping,
    /// The least natural frequency of a pole-zero analysis's poles, in hertz.
    DominantPole,
}

/// The values of a result's unknowns at each of its points, one row per point.
pub trait Table<T> {
    /// The number of rows.
    fn rows(&self) -> usize;
    /// The value at `row` of the unknown at `column`; none past either end.
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
    /// A DC sweep: its source's levels, and the unknowns' values at each.
    DcSweep {
        /// The values swept.
        swept: &'a [f64],
        /// The values, one row per level.
        values: &'a dyn Table<f64>,
    },
    /// A transient, or one period of a periodic steady state: its times, and the unknowns'
    /// values at each.
    Transient {
        /// The times.
        times: &'a [f64],
        /// The values, one row per time.
        values: &'a dyn Table<f64>,
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
    /// The numbers of `outcome`; none for sensitivities and transfer functions, which no
    /// measure reads.
    #[must_use]
    pub fn from_result(outcome: &'a crate::AnalysisResult) -> Option<Self> {
        Some(match outcome {
            crate::AnalysisResult::OperatingPoint(point) => point.into(),
            crate::AnalysisResult::DcSweep(sweep) => sweep.into(),
            crate::AnalysisResult::Ac(response) => response.into(),
            crate::AnalysisResult::Transient(waveforms) => waveforms.into(),
            crate::AnalysisResult::PeriodicSteadyState(found) => (&found.waveforms).into(),
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

/// A point a measure places on an axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Point {
    /// Where a value, a gain, a phase or a noise density is read.
    At,
    /// Where a window or a band starts.
    Start,
    /// Where a window or a band stops.
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

/// Why Python's `math.fsum` would raise on a sum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SumError {
    /// A partial sum of finite values overflows.
    #[error("intermediate overflow in fsum")]
    Overflow,
    /// The values hold both infinities.
    #[error("-inf + inf in fsum")]
    Indeterminate,
}

/// Why a measure could not read a result.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum MeasureError {
    /// The result is not of a kind the measure reads.
    #[error("the measure does not read this kind of result")]
    Kind,
    /// The result has no unknown at an index the measure reads.
    #[error("the result has no unknown at index {0}")]
    Unknown(usize),
    /// The result's values are not as many as its points.
    #[error("the result's values are not as many as its points")]
    Mismatch,
    /// The result holds no points.
    #[error("the result holds no points")]
    Empty,
    /// A point lies outside the axis, from `first` to `last`.
    #[error("the {point}, outside the axis from {first:?} to {last:?}")]
    Outside {
        /// Which point.
        point: Point,
        /// The axis's first point.
        first: f64,
        /// The axis's last point.
        last: f64,
    },
    /// The window does not start before it stops.
    #[error("the window is empty")]
    EmptyWindow,
    /// The signal ends the window at the value it starts it at.
    #[error("the signal ends the window where it starts it, at {0:?}")]
    Flat(f64),
    /// The signal never crosses the level the measure looks for, the way it looks for: a
    /// crossing's level, 0 dB, -180°, or a step's levels.
    #[error("the signal never crosses the level the measure looks for")]
    Uncrossed,
    /// The gain stays within a bandwidth's drop of its gain at the lowest frequency.
    #[error("the gain stays within the drop of its gain at {0:?} Hz")]
    Stays(f64),
    /// The signal rises through the level fewer than twice.
    #[error("the signal rises through {0:?} fewer than twice")]
    Rises(f64),
    /// The signal a transfer is divided by vanishes.
    #[error("the divisor vanishes in the response")]
    Vanishes,
    /// A transfer is read over log frequency, and its lowest frequency is not positive.
    #[error("a response is read over log frequency, so not at {0:?} Hz")]
    Unpositive(f64),
    /// The window spans less than a period of the fundamental.
    #[error("the window spans less than a period of the fundamental")]
    Short,
    /// The signal has no component at the fundamental.
    #[error("the signal has no component at the fundamental")]
    Silent,
    /// The distortion's samples would not fit in memory.
    #[error("the distortion's samples would not fit in memory")]
    Memory,
    /// A band does not start before it stops.
    #[error("the band is empty")]
    EmptyBand,
    /// A band from `low` to `high` does not lie within the frequencies, from `first` to
    /// `last`, or is empty.
    #[error(
        "a band from {low:?} Hz to {high:?} Hz does not lie within the frequencies, {first:?} Hz to {last:?} Hz"
    )]
    Band {
        /// Where the band starts.
        low: f64,
        /// Where the band stops.
        high: f64,
        /// The lowest frequency.
        first: f64,
        /// The highest frequency.
        last: f64,
    },
    /// Summing a noise integral's segments fails, as Python's `math.fsum` does.
    #[error(transparent)]
    Sum(SumError),
    /// The circuit has no poles.
    #[error("the circuit has no poles")]
    NoPoles,
    /// The noise analysis names no source to refer its noise to.
    #[error("the noise analysis names no source to refer its noise to")]
    Unreferred,
}

impl Measure {
    /// The measure reading the same signals of a circuit whose unknowns lie elsewhere: each
    /// signal's unknown, and its reference's, at the index `column` gives it, as a run on
    /// another structure of the circuit holds them.
    ///
    /// # Errors
    ///
    /// Returns [`MeasureError::Unknown`] with the index of the first unknown `column` gives
    /// no index for.
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
                Transfer::Output { output, over } => Transfer::Output {
                    output: at(output)?,
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
            | Self::Solutions
            | Self::Damping
            | Self::DominantPole => {}
        }
        Ok(moved)
    }

    /// What the measure reads of `results`, in the unit its variant says.
    ///
    /// # Errors
    ///
    /// Returns the [`MeasureError`] that says why the measure cannot read `results`.
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
                low,
                high,
            } => rise_time(signal, window, low, high, results),
            Self::Overshoot { signal, window } => overshoot(signal, window, results),
            Self::SettlingTime {
                signal,
                window,
                band,
            } => settling_time(signal, window, band, results),
            Self::Crossing {
                signal,
                window,
                level,
                edge,
            } => {
                let (times, values) = windowed(signal, window, results, Scope::Transient)?;
                arithmetic::crossings(&times, &values, level, edge)
                    .next()
                    .ok_or(MeasureError::Uncrossed)
            }
            Self::Frequency {
                signal,
                window,
                level,
            } => frequency(signal, window, level, results),
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
            Self::Solutions => match results {
                AnalysisResult::OperatingPoints(found) => Ok(count(*found)),
                _ => Err(MeasureError::Kind),
            },
            Self::Damping => Ok(arithmetic::minimum(&dampings(poles(results)?))),
            Self::DominantPole => Ok(arithmetic::minimum(&natural_frequencies(poles(results)?))),
        }
    }
}

/// The gain of `transfer` at the frequency `at`, in decibels.
fn gain(transfer: Transfer, at: f64, results: &AnalysisResult<'_>) -> Result<f64, MeasureError> {
    let (frequencies, transfer) = read_transfer(transfer, results)?;
    let point = Axis(&frequencies).locate(at, Point::At)?.ln();
    Ok(interpolated(
        point,
        &logarithms(&frequencies),
        &decibels(&transfer),
    ))
}

/// The phase of `transfer` at the frequency `at`, in degrees.
fn phase(transfer: Transfer, at: f64, results: &AnalysisResult<'_>) -> Result<f64, MeasureError> {
    let (frequencies, transfer) = read_transfer(transfer, results)?;
    let point = Axis(&frequencies).locate(at, Point::At)?.ln();
    let phases = arithmetic::phases(&transfer);
    Ok(interpolated(point, &logarithms(&frequencies), &phases))
}

/// The frequency where the gain of `transfer` first falls `drop` decibels below its gain at
/// the lowest frequency.
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

/// 180° plus the phase of `transfer` where its gain first falls through 0 dB.
fn phase_margin(transfer: Transfer, results: &AnalysisResult<'_>) -> Result<f64, MeasureError> {
    let (frequencies, transfer) = read_transfer(transfer, results)?;
    let logarithms = logarithms(&frequencies);
    let unity = unity(&logarithms, &transfer)?;
    let phases = arithmetic::phases(&transfer);
    Ok(180.0 + interpolated(unity, &logarithms, &phases))
}

/// Minus the gain of `transfer` where its phase first falls through -180°.
fn gain_margin(transfer: Transfer, results: &AnalysisResult<'_>) -> Result<f64, MeasureError> {
    let (frequencies, transfer) = read_transfer(transfer, results)?;
    let logarithms = logarithms(&frequencies);
    let phases = arithmetic::phases(&transfer);
    let inverted = arithmetic::crossings(&logarithms, &phases, -180.0, Edge::Falling)
        .next()
        .ok_or(MeasureError::Uncrossed)?;
    Ok(-interpolated(inverted, &logarithms, &decibels(&transfer)))
}

/// The time `signal` takes to pass from `low` to `high` of its step.
fn rise_time(
    signal: Signal,
    window: Window,
    low: f64,
    high: f64,
    results: &AnalysisResult<'_>,
) -> Result<f64, MeasureError> {
    let (times, values, step) = stepped(signal, window, results)?;
    let edge = if step > 0.0 {
        Edge::Rising
    } else {
        Edge::Falling
    };
    let lower = arithmetic::crossings(&times, &values, values[0] + low * step, edge)
        .next()
        .ok_or(MeasureError::Uncrossed)?;
    let upper = arithmetic::crossings(&times, &values, values[0] + high * step, edge)
        .find(|&upper| upper >= lower)
        .ok_or(MeasureError::Uncrossed)?;
    Ok(upper - lower)
}

/// How far `signal` passes its final value, in percent of its step.
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
    // Python's `max(0.0, passed)`, which keeps zero unless `passed` exceeds it.
    Ok(100.0 * if passed > 0.0 { passed } else { 0.0 })
}

/// The time from the window's start until `signal` stays within `band` of its step from its
/// final value.
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
    // The last value lies within the band, so the last outside it has one after it.
    let (Some(&next), Some(&after)) = (excess.get(outside + 1), times.get(outside + 1)) else {
        return Err(MeasureError::Mismatch);
    };
    let fraction = excess[outside] / (excess[outside] - next);
    let settled = times[outside] + fraction * (after - times[outside]);
    Ok(settled - times[0])
}

/// How often `signal` rises through `level`, by default the midpoint of its range.
fn frequency(
    signal: Signal,
    window: Window,
    level: Option<f64>,
    results: &AnalysisResult<'_>,
) -> Result<f64, MeasureError> {
    let (times, values) = windowed(signal, window, results, Scope::Transient)?;
    #[expect(
        clippy::manual_midpoint,
        reason = "numpy's `(max + min) / 2`, which overflows where `f64::midpoint` would not"
    )]
    let middle = level
        .unwrap_or_else(|| (arithmetic::maximum(&values) + arithmetic::minimum(&values)) / 2.0);
    let rises: Vec<f64> = arithmetic::crossings(&times, &values, middle, Edge::Rising).collect();
    match (rises.first(), rises.last()) {
        (Some(&first), Some(&last)) if rises.len() >= 2 => {
            Ok(count(rises.len() - 1) / (last - first))
        }
        _ => Err(MeasureError::Rises(middle)),
    }
}

/// The RMS noise from `start` to `stop`, at the output or referred to the input.
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

/// The gain of each of `transfer`, in decibels.
fn decibels(transfer: &[Complex64]) -> Vec<f64> {
    transfer.iter().map(|&z| arithmetic::decibels(z)).collect()
}

/// `count` as a float, exact below 2⁵³.
#[expect(
    clippy::cast_precision_loss,
    reason = "a count of points or solutions is far below 2⁵³, where floats hold every integer"
)]
fn count(count: usize) -> f64 {
    count as f64
}

impl Signal {
    /// The signal's values in `table`, one per row: the unknown's, less the reference's.
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

/// The points of an axis, which a measure places its points on.
struct Axis<'a>(&'a [f64]);

impl Axis<'_> {
    /// `point`, after checking it lies on the axis, from its first point to its last, within
    /// [`ON_AXIS`] of its span past either, and moved onto it if it lies past.
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
        // Python's `min(max(point, first), last)`, each keeping its first argument on a tie.
        let point = if first > point { first } else { point };
        Ok(if last < point { last } else { point })
    }
}

/// numpy's interpolation of `values`, given at `points`, at the one point `at`.
pub(crate) fn interpolated(at: f64, points: &[f64], values: &[f64]) -> f64 {
    arithmetic::interpolate(&[at], points, values)
        .first()
        .copied()
        .unwrap_or(f64::NAN)
}

/// The natural logarithm of each of `frequencies`.
fn logarithms(frequencies: &[f64]) -> Vec<f64> {
    frequencies.iter().map(|frequency| frequency.ln()).collect()
}

/// The mean of `values` over `points`, by the trapezoidal rule.
fn mean(points: &[f64], values: &[f64]) -> f64 {
    trapezoid(points, values) / (points[points.len() - 1] - points[0])
}

/// The value of `signal` at an operating point, or `at` a time or a level of the axis of a
/// transient or a DC sweep.
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
        (AnalysisResult::DcSweep { .. } | AnalysisResult::Transient { .. }, Some(at)) => {
            let (points, values) = traced(signal, results, Scope::Any)?;
            let point = Axis(&points).locate(at, Point::At)?;
            Ok(interpolated(point, &points, &values))
        }
        _ => Err(MeasureError::Kind),
    }
}

/// Which sweep axes a trace-based measure accepts.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Scope {
    Any,
    Transient,
}

/// The points of a transient's or a sweep's axis, increasing from its first to its last if
/// they fall, and the signal's values at each; a transient's alone under [`Scope::Transient`].
fn traced<'a>(
    signal: Signal,
    results: &AnalysisResult<'a>,
    scope: Scope,
) -> Result<(Cow<'a, [f64]>, Vec<f64>), MeasureError> {
    let (points, table) = match *results {
        AnalysisResult::Transient { times, values } => (times, values),
        AnalysisResult::DcSweep { swept, values } if scope == Scope::Any => (swept, values),
        _ => return Err(MeasureError::Kind),
    };
    let mut values = signal.trace(table)?;
    if values.len() != points.len() {
        return Err(MeasureError::Mismatch);
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

/// The points of `window` on a transient's or a sweep's axis, a transient's alone under
/// [`Scope::Transient`], and the signal's values at each: its ends, interpolated, and every
/// point of the axis between them.
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

/// The times and values of a transient's window, and the signal's step across it.
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

/// The frequencies of a response or a loop gain, increasing, and its transfer at each: the
/// loop gain, or the output over the signal it is divided by, if any.
fn read_transfer(
    transfer: Transfer,
    results: &AnalysisResult<'_>,
) -> Result<(Vec<f64>, Vec<Complex64>), MeasureError> {
    let (frequencies, values) = match (transfer, *results) {
        (Transfer::Loop, AnalysisResult::Stability { frequencies, gains }) => {
            (frequencies, gains.to_vec())
        }
        (
            Transfer::Output { output, over },
            AnalysisResult::Ac {
                frequencies,
                values,
            },
        ) => {
            let phasors = output.trace(values)?;
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
        return Err(MeasureError::Mismatch);
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

/// The log frequency where the gain of `transfer` first falls through 0 dB.
fn unity(logarithms: &[f64], transfer: &[Complex64]) -> Result<f64, MeasureError> {
    arithmetic::crossings(logarithms, &decibels(transfer), 0.0, Edge::Falling)
        .next()
        .ok_or(MeasureError::Uncrossed)
}

/// A noise analysis's frequencies, and its output's densities, or its densities referred to
/// its source if `referred`.
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
        (true, Some(_)) => return Err(MeasureError::Mismatch),
        (true, None) => return Err(MeasureError::Unreferred),
    };
    Ok((frequencies, densities))
}

/// A pole-zero analysis's poles, at least one.
fn poles<'a>(results: &AnalysisResult<'a>) -> Result<&'a [Complex64], MeasureError> {
    match *results {
        AnalysisResult::PoleZero([]) => Err(MeasureError::NoPoles),
        AnalysisResult::PoleZero(poles) => Ok(poles),
        _ => Err(MeasureError::Kind),
    }
}

/// How many periods of the fundamental frequency a distortion measure integrates over.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PeriodSpan {
    Last,
    Whole,
}

/// The total harmonic distortion, in decibels, of `values`, given at `times`, at the
/// `fundamental` frequency, harmonics 2 to `harmonics`: over the last period before the last
/// time, or over every whole period of the times under [`PeriodSpan::Whole`].
///
/// The signal is sampled [`SAMPLES_PER_PERIOD`] times a period, interpolated, as SPICE's
/// `.four` samples it, and each harmonic's amplitude is the magnitude of the discrete Fourier
/// transform's sum at its frequency, each sum in numpy's pairwise order.
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
    // The sample `n` of a period turns harmonic `h` by `h · n` of its 1024 steps around.
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
    // Python's `max(ratio, 1e-300)`, which keeps the ratio unless the floor exceeds it.
    let ratio = if LEAST_DISTORTION > ratio {
        LEAST_DISTORTION
    } else {
        ratio
    };
    Ok(20.0 * ratio.log10())
}

/// The cosine and the sine of `step` of [`SAMPLES_PER_PERIOD`] steps around a circle, each
/// quarter turn's from the first quarter's, so the steps on the axes are exact.
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
