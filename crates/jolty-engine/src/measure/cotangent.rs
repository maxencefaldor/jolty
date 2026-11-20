//! Exact sparse reverse-mode cotangents for [`Measure`] evaluation.
//!
//! Each routine computes the scalar measurement identically to [`Measure::measure`] and
//! differentiates it in reverse mode with respect to the underlying [`AnalysisResult`]
//! arrays (and autonomous period when evaluating a [`AnalysisResult::PeriodicSteadyState`]).
//! Complex cotangents follow the engine's bilinear convention `v = dL/da - j dL/db`, so
//! `Re(v * dz) = dL`.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::f64::consts::{LN_10, PI, TAU};

use jolty_numerics::Complex64;

use super::arithmetic::{self, InterpolatedPoint, PowerLawPoint};
use super::{
    AnalysisResult, Axis, Edge, MIN_DISTORTION, Measure, MeasureCotangent, MeasureError, ON_AXIS,
    PeriodSpan, Point, SAMPLES_PER_PERIOD, Scope, Signal, Transfer, Window, count, dampings,
    decibels, logarithms, mean, natural_frequencies, poles, turn,
};

/// Evaluates `measure` on `results` together with its exact sparse [`MeasureCotangent`].
pub(super) fn evaluate(
    measure: Measure,
    results: &AnalysisResult<'_>,
) -> Result<(f64, MeasureCotangent), MeasureError> {
    let outcome = match measure {
        Measure::Value { signal, at } => value_cotangent(signal, at, results)?,
        Measure::Gain { transfer, at } => gain_cotangent(transfer, at, results)?,
        Measure::Phase { transfer, at } => phase_cotangent(transfer, at, results)?,
        Measure::Bandwidth { transfer, drop } => bandwidth_cotangent(transfer, drop, results)?,
        Measure::Crossover { transfer } => crossover_cotangent(transfer, results)?,
        Measure::PhaseMargin { transfer } => phase_margin_cotangent(transfer, results)?,
        Measure::GainMargin { transfer } => gain_margin_cotangent(transfer, results)?,
        Measure::Maximum { signal, window } => extremum_cotangent(signal, window, results, true)?,
        Measure::Minimum { signal, window } => extremum_cotangent(signal, window, results, false)?,
        Measure::PeakToPeak { signal, window } => peak_to_peak_cotangent(signal, window, results)?,
        Measure::Mean { signal, window } => mean_cotangent(signal, window, results)?,
        Measure::Rms { signal, window } => rms_cotangent(signal, window, results)?,
        Measure::RiseTime {
            signal,
            window,
            lower,
            upper,
        } => rise_time_cotangent(signal, window, lower, upper, results)?,
        Measure::Overshoot { signal, window } => overshoot_cotangent(signal, window, results)?,
        Measure::SettlingTime {
            signal,
            window,
            band,
        } => settling_time_cotangent(signal, window, band, results)?,
        Measure::Crossing {
            signal,
            window,
            threshold,
            edge,
        } => crossing_cotangent(signal, window, threshold, edge, results)?,
        Measure::Frequency {
            signal,
            window,
            threshold,
        } => frequency_cotangent(signal, window, threshold, results)?,
        Measure::Distortion {
            signal,
            window,
            fundamental,
            harmonics,
        } => distortion_cotangent(signal, window, fundamental, harmonics, results)?,
        Measure::NoiseDensity { at, referred } => noise_density_cotangent(at, referred, results)?,
        Measure::IntegratedNoise {
            start,
            stop,
            referred,
        } => integrated_noise_cotangent(start, stop, referred, results)?,
        Measure::Period => period_cotangent(results)?,
        Measure::Solutions => {
            return match results {
                AnalysisResult::OperatingPoints(_) => Err(MeasureError::NotDifferentiable),
                _ => Err(MeasureError::Kind),
            };
        }
        Measure::Damping => damping_cotangent(results)?,
        Measure::DominantPole => dominant_pole_cotangent(results)?,
    };
    if !outcome.0.is_finite() {
        return Err(MeasureError::NotDifferentiable);
    }
    Ok(outcome)
}

/// Evaluates `Measure::Period` and its unit period cotangent.
fn period_cotangent(results: &AnalysisResult<'_>) -> Result<(f64, MeasureCotangent), MeasureError> {
    match *results {
        AnalysisResult::PeriodicSteadyState { period, .. } => Ok((
            period,
            MeasureCotangent::Real {
                entries: Vec::new(),
                period: 1.0,
            },
        )),
        _ => Err(MeasureError::Kind),
    }
}

/// Evaluates `Measure::Value` and its exact sparse real cotangent.
fn value_cotangent(
    signal: Signal,
    at: Option<f64>,
    results: &AnalysisResult<'_>,
) -> Result<(f64, MeasureCotangent), MeasureError> {
    match (*results, at) {
        (AnalysisResult::OperatingPoint(values), None) => {
            let read = |index: usize| {
                values
                    .get(index)
                    .copied()
                    .ok_or(MeasureError::Unknown(index))
            };
            let primary = read(signal.unknown)?;
            let value = match signal.reference {
                None => primary,
                Some(reference) => primary - read(reference)?,
            };
            let entries = scatter_real(signal, &[(0, 1.0)])?;
            Ok((
                value,
                MeasureCotangent::Real {
                    entries,
                    period: 0.0,
                },
            ))
        }
        (
            AnalysisResult::DcSweep { .. }
            | AnalysisResult::Transient { .. }
            | AnalysisResult::PeriodicSteadyState { .. },
            Some(at),
        ) => {
            let trace = OrderedTrace::extract(signal, results, Scope::Any)?;
            let point = Axis(&trace.points).locate(at, Point::At)?;
            let sample = arithmetic::interpolate_weights(point, &trace.points, &trace.values);
            let mut point_weights = vec![(trace.original(sample.first.0), sample.first.1)];
            if let Some((second_index, second_weight)) = sample.second {
                point_weights.push((trace.original(second_index), second_weight));
            }
            let period_weight = match trace.stretch {
                Some(period) if period != 0.0 && trace.points.len() >= 2 => {
                    -sample.slope * (point / period)
                }
                _ => 0.0,
            };
            if !period_weight.is_finite() {
                return Err(MeasureError::NotDifferentiable);
            }
            let entries = scatter_real(signal, &point_weights)?;
            Ok((
                sample.value,
                MeasureCotangent::Real {
                    entries,
                    period: period_weight,
                },
            ))
        }
        _ => Err(MeasureError::Kind),
    }
}

/// Ascending real trace extracted from a DC sweep, transient, or periodic steady state,
/// recording reversal and optional period-scaling metadata for reverse-mode pullback.
struct OrderedTrace<'a> {
    points: Cow<'a, [f64]>,
    values: Vec<f64>,
    reversed: bool,
    stretch: Option<f64>,
}

impl<'a> OrderedTrace<'a> {
    /// Extracts an ascending trace for `signal` under `scope`.
    fn extract(
        signal: Signal,
        results: &AnalysisResult<'a>,
        scope: Scope,
    ) -> Result<Self, MeasureError> {
        let (points, table, stretch) = match *results {
            AnalysisResult::Transient { times, values } => (times, values, None),
            AnalysisResult::PeriodicSteadyState {
                times,
                values,
                period,
            } => (times, values, Some(period)),
            AnalysisResult::DcSweep { swept, values } if scope == Scope::Any => {
                (swept, values, None)
            }
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
            return Ok(Self {
                points: points.iter().rev().copied().collect(),
                values,
                reversed: true,
                stretch,
            });
        }
        Ok(Self {
            points: Cow::Borrowed(points),
            values,
            reversed: false,
            stretch,
        })
    }

    /// Maps ascending index `index` back to the original row index in `AnalysisResult`.
    const fn original(&self, index: usize) -> usize {
        if self.reversed {
            self.values.len() - 1 - index
        } else {
            index
        }
    }
}

/// Windowed real trace holding endpoint interpolation metadata for exact reverse pullback.
struct WindowedTrace {
    times: Vec<f64>,
    values: Vec<f64>,
    samples: Vec<InterpolatedPoint>,
    fixed: Vec<bool>,
    raw_len: usize,
    reversed: bool,
    stretch: Option<f64>,
}

impl WindowedTrace {
    /// Slices and endpoint-interpolates `signal` onto `window` within `results` under `scope`.
    fn extract(
        signal: Signal,
        window: Window,
        results: &AnalysisResult<'_>,
        scope: Scope,
    ) -> Result<Self, MeasureError> {
        let trace = OrderedTrace::extract(signal, results, scope)?;
        let axis = Axis(&trace.points);
        let mut first = trace.points[0];
        let mut last = trace.points[trace.points.len() - 1];
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
        let mut fixed = vec![window.start.is_some()];
        for &point in trace.points.iter() {
            if point > first && point < last {
                inside.push(point);
                fixed.push(false);
            }
        }
        inside.push(last);
        fixed.push(window.stop.is_some());
        let samples: Vec<InterpolatedPoint> = inside
            .iter()
            .map(|&point| arithmetic::interpolate_weights(point, &trace.points, &trace.values))
            .collect();
        let values: Vec<f64> = samples.iter().map(|sample| sample.value).collect();
        Ok(Self {
            times: inside,
            values,
            samples,
            fixed,
            raw_len: trace.values.len(),
            reversed: trace.reversed,
            stretch: trace.stretch,
        })
    }

    /// Pulls windowed `(value_weights, time_weights)` back to a real [`MeasureCotangent`] on
    /// `signal`.
    fn pull_back(
        &self,
        signal: Signal,
        value_weights: &[f64],
        time_weights: &[f64],
    ) -> Result<MeasureCotangent, MeasureError> {
        let mut raw_weights = vec![0.0; self.raw_len];
        let mut period_weight = 0.0;
        let stretch = self
            .stretch
            .filter(|&period| period != 0.0 && self.raw_len >= 2);
        for (index, ((&value_weight, &time_weight), (&sample, &is_fixed))) in value_weights
            .iter()
            .zip(time_weights)
            .zip(self.samples.iter().zip(&self.fixed))
            .enumerate()
        {
            if value_weight != 0.0 {
                raw_weights[sample.first.0] += value_weight * sample.first.1;
                if let Some((second_index, second_weight)) = sample.second {
                    raw_weights[second_index] += value_weight * second_weight;
                }
                if is_fixed && let Some(period) = stretch {
                    period_weight -= value_weight * sample.slope * (self.times[index] / period);
                }
            }
            if time_weight != 0.0
                && !is_fixed
                && let Some(period) = stretch
            {
                period_weight += time_weight * (self.times[index] / period);
            }
        }
        if !period_weight.is_finite() {
            return Err(MeasureError::NotDifferentiable);
        }
        let mut point_weights = Vec::new();
        for (index, &weight) in raw_weights.iter().enumerate() {
            if weight != 0.0 {
                let original = if self.reversed {
                    self.raw_len - 1 - index
                } else {
                    index
                };
                point_weights.push((original, weight));
            }
        }
        let entries = scatter_real(signal, &point_weights)?;
        Ok(MeasureCotangent::Real {
            entries,
            period: period_weight,
        })
    }
}

/// Evaluates `Measure::Maximum` (`is_maximum == true`) or `Measure::Minimum` (`false`).
fn extremum_cotangent(
    signal: Signal,
    window: Window,
    results: &AnalysisResult<'_>,
    is_maximum: bool,
) -> Result<(f64, MeasureCotangent), MeasureError> {
    let trace = WindowedTrace::extract(signal, window, results, Scope::Any)?;
    let found = if is_maximum {
        arithmetic::maximum_index(&trace.values)
    } else {
        arithmetic::minimum_index(&trace.values)
    };
    let (index, value) = found.ok_or(MeasureError::NotDifferentiable)?;
    let mut value_weights = vec![0.0; trace.values.len()];
    let time_weights = vec![0.0; trace.times.len()];
    value_weights[index] = 1.0;
    let cotangent = trace.pull_back(signal, &value_weights, &time_weights)?;
    Ok((value, cotangent))
}

/// Evaluates `Measure::PeakToPeak` and its exact sparse real cotangent.
fn peak_to_peak_cotangent(
    signal: Signal,
    window: Window,
    results: &AnalysisResult<'_>,
) -> Result<(f64, MeasureCotangent), MeasureError> {
    let trace = WindowedTrace::extract(signal, window, results, Scope::Any)?;
    let (highest_index, highest_value) =
        arithmetic::maximum_index(&trace.values).ok_or(MeasureError::NotDifferentiable)?;
    let (lowest_index, lowest_value) =
        arithmetic::minimum_index(&trace.values).ok_or(MeasureError::NotDifferentiable)?;
    let mut value_weights = vec![0.0; trace.values.len()];
    let time_weights = vec![0.0; trace.times.len()];
    value_weights[highest_index] += 1.0;
    value_weights[lowest_index] -= 1.0;
    let cotangent = trace.pull_back(signal, &value_weights, &time_weights)?;
    Ok((highest_value - lowest_value, cotangent))
}

/// Computes trapezoidal mean weights `(value_weights, time_weights)` scaled by `upstream`.
fn accumulate_mean_weights(
    times: &[f64],
    values: &[f64],
    mean_value: f64,
    upstream: f64,
    value_weights: &mut [f64],
    time_weights: &mut [f64],
) {
    let last = times.len() - 1;
    let span = times[last] - times[0];
    let area_weight = upstream / span;
    let span_weight = -upstream * mean_value / span;
    for index in 0..last {
        let delta_time = times[index + 1] - times[index];
        let half_step = area_weight * 0.5 * delta_time;
        value_weights[index] += half_step;
        value_weights[index + 1] += half_step;
        let average = f64::midpoint(values[index], values[index + 1]);
        let time_term = area_weight * average;
        time_weights[index + 1] += time_term;
        time_weights[index] -= time_term;
    }
    time_weights[last] += span_weight;
    time_weights[0] -= span_weight;
}

/// Evaluates `Measure::Mean` and its exact sparse real cotangent.
fn mean_cotangent(
    signal: Signal,
    window: Window,
    results: &AnalysisResult<'_>,
) -> Result<(f64, MeasureCotangent), MeasureError> {
    let trace = WindowedTrace::extract(signal, window, results, Scope::Any)?;
    let value = mean(&trace.times, &trace.values);
    let mut value_weights = vec![0.0; trace.values.len()];
    let mut time_weights = vec![0.0; trace.times.len()];
    accumulate_mean_weights(
        &trace.times,
        &trace.values,
        value,
        1.0,
        &mut value_weights,
        &mut time_weights,
    );
    let cotangent = trace.pull_back(signal, &value_weights, &time_weights)?;
    Ok((value, cotangent))
}

/// Evaluates `Measure::Rms` and its exact sparse real cotangent.
fn rms_cotangent(
    signal: Signal,
    window: Window,
    results: &AnalysisResult<'_>,
) -> Result<(f64, MeasureCotangent), MeasureError> {
    let trace = WindowedTrace::extract(signal, window, results, Scope::Any)?;
    let squares: Vec<f64> = trace.values.iter().map(|value| value * value).collect();
    let mean_square = mean(&trace.times, &squares);
    let value = mean_square.sqrt();
    if value == 0.0 || !value.is_finite() {
        return Err(MeasureError::NotDifferentiable);
    }
    let mut square_weights = vec![0.0; trace.values.len()];
    let mut time_weights = vec![0.0; trace.times.len()];
    accumulate_mean_weights(
        &trace.times,
        &squares,
        mean_square,
        0.5 / value,
        &mut square_weights,
        &mut time_weights,
    );
    let value_weights: Vec<f64> = square_weights
        .iter()
        .zip(&trace.values)
        .map(|(&weight, &sample)| weight * 2.0 * sample)
        .collect();
    let cotangent = trace.pull_back(signal, &value_weights, &time_weights)?;
    Ok((value, cotangent))
}

/// Extracts a windowed transient trace and its nonzero step `last - first`.
fn stepped_trace(
    signal: Signal,
    window: Window,
    results: &AnalysisResult<'_>,
) -> Result<(WindowedTrace, f64), MeasureError> {
    let trace = WindowedTrace::extract(signal, window, results, Scope::Transient)?;
    let step = trace.values[trace.values.len() - 1] - trace.values[0];
    if step == 0.0 {
        return Err(MeasureError::Flat(trace.values[0]));
    }
    Ok((trace, step))
}

/// Pulls cotangent `point_weight` on linear threshold crossing `crossing` back to the
/// bracketing values, optional axis coordinates, and threshold cotangent.
fn pull_crossing(
    crossing: arithmetic::Crossing,
    points: &[f64],
    values: &[f64],
    point_weight: f64,
    value_weights: &mut [f64],
    time_weights: Option<&mut [f64]>,
) -> f64 {
    let index = crossing.index;
    let delta_point = points[index + 1] - points[index];
    let delta_value = values[index + 1] - values[index];
    let threshold_weight = point_weight * delta_point / delta_value;
    value_weights[index] -= (1.0 - crossing.fraction) * threshold_weight;
    value_weights[index + 1] -= crossing.fraction * threshold_weight;
    if let Some(time_weights) = time_weights {
        time_weights[index] += point_weight * (1.0 - crossing.fraction);
        time_weights[index + 1] += point_weight * crossing.fraction;
    }
    threshold_weight
}

/// Evaluates `Measure::RiseTime` and its exact sparse real cotangent.
fn rise_time_cotangent(
    signal: Signal,
    window: Window,
    lower: f64,
    upper: f64,
    results: &AnalysisResult<'_>,
) -> Result<(f64, MeasureCotangent), MeasureError> {
    let (trace, step) = stepped_trace(signal, window, results)?;
    let edge = if step > 0.0 {
        Edge::Rising
    } else {
        Edge::Falling
    };
    let lower_level = trace.values[0] + lower * step;
    let upper_level = trace.values[0] + upper * step;
    let first = arithmetic::indexed_crossings(&trace.times, &trace.values, lower_level, edge)
        .next()
        .ok_or(MeasureError::Uncrossed)?;
    let second = arithmetic::indexed_crossings(&trace.times, &trace.values, upper_level, edge)
        .find(|crossing| crossing.point >= first.point)
        .ok_or(MeasureError::Uncrossed)?;
    let mut value_weights = vec![0.0; trace.values.len()];
    let mut time_weights = vec![0.0; trace.times.len()];
    let upper_weight = pull_crossing(
        second,
        &trace.times,
        &trace.values,
        1.0,
        &mut value_weights,
        Some(&mut time_weights),
    );
    let lower_weight = pull_crossing(
        first,
        &trace.times,
        &trace.values,
        -1.0,
        &mut value_weights,
        Some(&mut time_weights),
    );
    let last = trace.values.len() - 1;
    value_weights[0] += lower_weight * (1.0 - lower) + upper_weight * (1.0 - upper);
    value_weights[last] += lower_weight * lower + upper_weight * upper;
    let cotangent = trace.pull_back(signal, &value_weights, &time_weights)?;
    Ok((second.point - first.point, cotangent))
}

/// Evaluates `Measure::Overshoot` and its exact sparse real cotangent.
fn overshoot_cotangent(
    signal: Signal,
    window: Window,
    results: &AnalysisResult<'_>,
) -> Result<(f64, MeasureCotangent), MeasureError> {
    let (trace, step) = stepped_trace(signal, window, results)?;
    let found = if step > 0.0 {
        arithmetic::maximum_index(&trace.values)
    } else {
        arithmetic::minimum_index(&trace.values)
    };
    let (peak_index, peak) = found.ok_or(MeasureError::NotDifferentiable)?;
    let last = trace.values.len() - 1;
    let passed = (peak - trace.values[last]) / step;
    let mut value_weights = vec![0.0; trace.values.len()];
    let time_weights = vec![0.0; trace.times.len()];
    let value = if passed > 0.0 {
        let scale = 100.0 / step;
        value_weights[peak_index] += scale;
        value_weights[last] -= scale * (1.0 + passed);
        value_weights[0] += scale * passed;
        100.0 * passed
    } else {
        0.0
    };
    let cotangent = trace.pull_back(signal, &value_weights, &time_weights)?;
    Ok((value, cotangent))
}

/// Evaluates `Measure::SettlingTime` and its exact sparse real cotangent.
fn settling_time_cotangent(
    signal: Signal,
    window: Window,
    band: f64,
    results: &AnalysisResult<'_>,
) -> Result<(f64, MeasureCotangent), MeasureError> {
    let (trace, step) = stepped_trace(signal, window, results)?;
    let last_index = trace.values.len() - 1;
    let last_value = trace.values[last_index];
    let within = band * step.abs();
    let excess: Vec<f64> = trace
        .values
        .iter()
        .map(|value| (value - last_value).abs() - within)
        .collect();
    let mut value_weights = vec![0.0; trace.values.len()];
    let mut time_weights = vec![0.0; trace.times.len()];
    let Some(outside) = excess.iter().rposition(|&entry| entry > 0.0) else {
        let cotangent = trace.pull_back(signal, &value_weights, &time_weights)?;
        return Ok((0.0, cotangent));
    };
    let (Some(&next_excess), Some(&next_time)) =
        (excess.get(outside + 1), trace.times.get(outside + 1))
    else {
        return Err(MeasureError::LengthMismatch);
    };
    let denominator = excess[outside] - next_excess;
    let fraction = excess[outside] / denominator;
    let delta_time = next_time - trace.times[outside];
    let settled = trace.times[outside] + fraction * delta_time;

    time_weights[0] -= 1.0;
    time_weights[outside] += 1.0 - fraction;
    time_weights[outside + 1] += fraction;
    let outside_weight = delta_time * (1.0 - fraction) / denominator;
    let next_weight = delta_time * fraction / denominator;

    let outside_sign = (trace.values[outside] - last_value).signum();
    value_weights[outside] += outside_weight * outside_sign;
    value_weights[last_index] -= outside_weight * outside_sign;

    if outside + 1 < last_index {
        let next_delta = trace.values[outside + 1] - last_value;
        if next_delta != 0.0 {
            let next_sign = next_delta.signum();
            value_weights[outside + 1] += next_weight * next_sign;
            value_weights[last_index] -= next_weight * next_sign;
        }
    }

    let within_weight = -(outside_weight + next_weight);
    let step_term = within_weight * band * step.signum();
    value_weights[last_index] += step_term;
    value_weights[0] -= step_term;

    let cotangent = trace.pull_back(signal, &value_weights, &time_weights)?;
    Ok((settled - trace.times[0], cotangent))
}

/// Evaluates `Measure::Crossing` and its exact sparse real cotangent.
fn crossing_cotangent(
    signal: Signal,
    window: Window,
    threshold: f64,
    edge: Edge,
    results: &AnalysisResult<'_>,
) -> Result<(f64, MeasureCotangent), MeasureError> {
    let trace = WindowedTrace::extract(signal, window, results, Scope::Transient)?;
    let crossing = arithmetic::indexed_crossings(&trace.times, &trace.values, threshold, edge)
        .next()
        .ok_or(MeasureError::Uncrossed)?;
    let mut value_weights = vec![0.0; trace.values.len()];
    let mut time_weights = vec![0.0; trace.times.len()];
    pull_crossing(
        crossing,
        &trace.times,
        &trace.values,
        1.0,
        &mut value_weights,
        Some(&mut time_weights),
    );
    let cotangent = trace.pull_back(signal, &value_weights, &time_weights)?;
    Ok((crossing.point, cotangent))
}

/// Evaluates `Measure::Frequency` and its exact sparse real cotangent.
fn frequency_cotangent(
    signal: Signal,
    window: Window,
    threshold: Option<f64>,
    results: &AnalysisResult<'_>,
) -> Result<(f64, MeasureCotangent), MeasureError> {
    let trace = WindowedTrace::extract(signal, window, results, Scope::Transient)?;
    let (middle, midpoint_indices) = if let Some(given) = threshold {
        (given, None)
    } else {
        let (highest_index, highest_value) =
            arithmetic::maximum_index(&trace.values).ok_or(MeasureError::NotDifferentiable)?;
        let (lowest_index, lowest_value) =
            arithmetic::minimum_index(&trace.values).ok_or(MeasureError::NotDifferentiable)?;
        #[expect(
            clippy::manual_midpoint,
            reason = "numpy's `(max + min) / 2` overflows where `f64::midpoint` does not"
        )]
        let center = (highest_value + lowest_value) / 2.0;
        (center, Some((highest_index, lowest_index)))
    };
    let rises: Vec<arithmetic::Crossing> =
        arithmetic::indexed_crossings(&trace.times, &trace.values, middle, Edge::Rising).collect();
    let (Some(&first), Some(&last)) = (rises.first(), rises.last()) else {
        return Err(MeasureError::Rises(middle));
    };
    if rises.len() < 2 {
        return Err(MeasureError::Rises(middle));
    }
    let span = last.point - first.point;
    if span == 0.0 || !span.is_finite() {
        return Err(MeasureError::NotDifferentiable);
    }
    let cycles = count(rises.len() - 1);
    let value = cycles / span;
    let span_scale = value / span;
    let mut value_weights = vec![0.0; trace.values.len()];
    let mut time_weights = vec![0.0; trace.times.len()];
    let mut threshold_weight = pull_crossing(
        last,
        &trace.times,
        &trace.values,
        -span_scale,
        &mut value_weights,
        Some(&mut time_weights),
    );
    threshold_weight += pull_crossing(
        first,
        &trace.times,
        &trace.values,
        span_scale,
        &mut value_weights,
        Some(&mut time_weights),
    );
    if let Some((highest_index, lowest_index)) = midpoint_indices {
        value_weights[highest_index] += 0.5 * threshold_weight;
        value_weights[lowest_index] += 0.5 * threshold_weight;
    }
    let cotangent = trace.pull_back(signal, &value_weights, &time_weights)?;
    Ok((value, cotangent))
}

/// Resamples `trace` onto `length` uniform points spanning `duration` ending at the window
/// stop and returns the interpolation metadata for each sample.
fn resample_distortion(
    trace: &WindowedTrace,
    length: usize,
    samples: f64,
    duration: f64,
) -> Result<Vec<InterpolatedPoint>, MeasureError> {
    let mut query_times = Vec::new();
    query_times
        .try_reserve_exact(length)
        .map_err(|_| MeasureError::Memory)?;
    let last_time = trace.times[trace.times.len() - 1];
    query_times
        .extend((0..length).map(|index| last_time - duration * (1.0 - count(index) / samples)));
    Ok(query_times
        .iter()
        .map(|&point| arithmetic::interpolate_weights(point, &trace.times, &trace.values))
        .collect())
}

/// Evaluates `Measure::Distortion` and its exact sparse real cotangent.
fn distortion_cotangent(
    signal: Signal,
    window: Window,
    fundamental_frequency: f64,
    harmonics: usize,
    results: &AnalysisResult<'_>,
) -> Result<(f64, MeasureCotangent), MeasureError> {
    let trace = WindowedTrace::extract(signal, window, results, Scope::Transient)?;
    let span_kind = if window.start.is_some() {
        PeriodSpan::Whole
    } else {
        PeriodSpan::Last
    };
    let span = (trace.times[trace.times.len() - 1] - trace.times[0]) * fundamental_frequency;
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
    let resampled = resample_distortion(&trace, length, samples, periods / fundamental_frequency)?;
    let waveform: Vec<f64> = resampled.iter().map(|sample| sample.value).collect();
    let turns: Vec<(f64, f64)> = (0..SAMPLES_PER_PERIOD).map(turn).collect();
    let bin = |harmonic: usize| -> (f64, f64, f64) {
        let phase = |index: usize| {
            turns[(harmonic % SAMPLES_PER_PERIOD) * (index % SAMPLES_PER_PERIOD)
                % SAMPLES_PER_PERIOD]
        };
        let real = arithmetic::sum(length, |index| waveform[index] * phase(index).0);
        let imaginary = arithmetic::sum(length, |index| -waveform[index] * phase(index).1);
        let amplitude = arithmetic::magnitude(Complex64::new(real, imaginary));
        (real, imaginary, amplitude)
    };
    let (fundamental_real, fundamental_imaginary, fundamental_amplitude) = bin(1);
    if fundamental_amplitude == 0.0 {
        return Err(MeasureError::Silent);
    }
    let higher_bins: Vec<(f64, f64, f64)> = (2..=harmonics).map(bin).collect();
    let harmonic_power = arithmetic::sum(higher_bins.len(), |index| {
        higher_bins[index].2 * higher_bins[index].2
    });
    let raw_ratio = harmonic_power.sqrt() / fundamental_amplitude;
    let mut value_weights = vec![0.0; trace.values.len()];
    let mut time_weights = vec![0.0; trace.times.len()];
    if MIN_DISTORTION > raw_ratio || harmonic_power == 0.0 {
        let clamped = if MIN_DISTORTION > raw_ratio {
            MIN_DISTORTION
        } else {
            raw_ratio
        };
        let cotangent = trace.pull_back(signal, &value_weights, &time_weights)?;
        return Ok((20.0 * clamped.log10(), cotangent));
    }
    let decibel_scale = 20.0 / LN_10;
    let fundamental_power = fundamental_amplitude * fundamental_amplitude;
    let mut bin_weights = Vec::with_capacity(1 + higher_bins.len());
    bin_weights.push((
        -decibel_scale * fundamental_real / fundamental_power,
        -decibel_scale * fundamental_imaginary / fundamental_power,
    ));
    for &(real, imaginary, _) in &higher_bins {
        bin_weights.push((
            decibel_scale * real / harmonic_power,
            decibel_scale * imaginary / harmonic_power,
        ));
    }
    accumulate_distortion_weights(
        &resampled,
        &turns,
        &bin_weights,
        &mut value_weights,
        &mut time_weights,
    );
    let cotangent = trace.pull_back(signal, &value_weights, &time_weights)?;
    Ok((20.0 * raw_ratio.log10(), cotangent))
}

/// Pulls harmonic DFT bin cotangents back through uniform resampling into windowed
/// `(value_weights, time_weights)`.
fn accumulate_distortion_weights(
    resampled: &[InterpolatedPoint],
    turns: &[(f64, f64)],
    bin_weights: &[(f64, f64)],
    value_weights: &mut [f64],
    time_weights: &mut [f64],
) {
    let last_index = time_weights.len() - 1;
    for (sample_index, &sample) in resampled.iter().enumerate() {
        let mut waveform_weight = 0.0;
        for (offset, &(real_weight, imaginary_weight)) in bin_weights.iter().enumerate() {
            let harmonic = offset + 1;
            let (cosine, sine) = turns[(harmonic % SAMPLES_PER_PERIOD)
                * (sample_index % SAMPLES_PER_PERIOD)
                % SAMPLES_PER_PERIOD];
            waveform_weight += real_weight * cosine - imaginary_weight * sine;
        }
        value_weights[sample.first.0] += waveform_weight * sample.first.1;
        if let Some((second_index, second_weight)) = sample.second {
            value_weights[second_index] += waveform_weight * second_weight;
        }
        if sample.slope != 0.0 {
            let slope_weight = waveform_weight * sample.slope;
            time_weights[last_index] += slope_weight;
            time_weights[sample.first.0] -= slope_weight * sample.first.1;
            if let Some((second_index, second_weight)) = sample.second {
                time_weights[second_index] -= slope_weight * second_weight;
            }
        }
    }
}

/// Sorted positive-frequency transfer data retaining original indices and optional AC
/// numerator/denominator phasors for quotient-rule pullback.
struct OrderedTransfer {
    order: Vec<usize>,
    frequencies: Vec<f64>,
    values: Vec<Complex64>,
    divisors: Option<Vec<Complex64>>,
    transfer: Transfer,
}

impl OrderedTransfer {
    /// Extracts ascending positive frequencies and complex transfer values from `results`.
    fn extract(transfer: Transfer, results: &AnalysisResult<'_>) -> Result<Self, MeasureError> {
        let (frequencies, values, divisors) = match (transfer, *results) {
            (Transfer::Loop, AnalysisResult::Stability { frequencies, gains }) => {
                (frequencies, gains.to_vec(), None)
            }
            (
                Transfer::Signal { signal, over },
                AnalysisResult::Ac {
                    frequencies,
                    values,
                },
            ) => {
                let phasors = signal.trace(values)?;
                let (ratios, divisors) = match over {
                    None => (phasors, None),
                    Some(over) => {
                        let divisors = over.trace(values)?;
                        if divisors
                            .iter()
                            .any(|&divisor| divisor == Complex64::new(0.0, 0.0))
                        {
                            return Err(MeasureError::Vanishes);
                        }
                        let ratios = phasors
                            .iter()
                            .zip(&divisors)
                            .map(|(&phasor, &divisor)| arithmetic::divide(phasor, divisor))
                            .collect();
                        (ratios, Some(divisors))
                    }
                };
                (frequencies, ratios, divisors)
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
        let sorted_frequencies: Vec<f64> = order.iter().map(|&index| frequencies[index]).collect();
        if sorted_frequencies[0] <= 0.0 {
            return Err(MeasureError::Unpositive(sorted_frequencies[0]));
        }
        let sorted_values: Vec<Complex64> = order.iter().map(|&index| values[index]).collect();
        let sorted_divisors =
            divisors.map(|denominators| order.iter().map(|&index| denominators[index]).collect());
        Ok(Self {
            order,
            frequencies: sorted_frequencies,
            values: sorted_values,
            divisors: sorted_divisors,
            transfer,
        })
    }

    /// Pulls sorted `(gain_weights, phase_weights)` back into a complex [`MeasureCotangent`].
    fn pull_back(
        &self,
        gain_weights: &[f64],
        phase_weights: &[f64],
    ) -> Result<MeasureCotangent, MeasureError> {
        match self.transfer {
            Transfer::Loop => {
                let mut map: BTreeMap<usize, Complex64> = BTreeMap::new();
                for (index, (&gain_weight, &phase_weight)) in
                    gain_weights.iter().zip(phase_weights).enumerate()
                {
                    let weight =
                        complex_transfer_weight(self.values[index], gain_weight, phase_weight)?;
                    if weight != Complex64::default() {
                        *map.entry(self.order[index]).or_default() += weight;
                    }
                }
                let entries = collect_point_complex(map)?;
                Ok(MeasureCotangent::Loop { entries })
            }
            Transfer::Signal { signal, over } => {
                let mut map: BTreeMap<(usize, usize), Complex64> = BTreeMap::new();
                for (index, (&gain_weight, &phase_weight)) in
                    gain_weights.iter().zip(phase_weights).enumerate()
                {
                    let transfer_weight =
                        complex_transfer_weight(self.values[index], gain_weight, phase_weight)?;
                    if transfer_weight == Complex64::default() {
                        continue;
                    }
                    let original = self.order[index];
                    match (over, &self.divisors) {
                        (Some(over_signal), Some(divisors)) => {
                            let divisor = divisors[index];
                            let numerator_weight = arithmetic::divide(transfer_weight, divisor);
                            let denominator_weight =
                                -arithmetic::divide(transfer_weight * self.values[index], divisor);
                            add_signal_complex(&mut map, original, signal, numerator_weight)?;
                            add_signal_complex(
                                &mut map,
                                original,
                                over_signal,
                                denominator_weight,
                            )?;
                        }
                        _ => add_signal_complex(&mut map, original, signal, transfer_weight)?,
                    }
                }
                let entries = collect_table_complex(map)?;
                Ok(MeasureCotangent::Complex { entries })
            }
        }
    }
}

/// Converts real dB-gain and degree-phase cotangents on `phasor` to its bilinear complex
/// cotangent `dL/da - j dL/db`.
fn complex_transfer_weight(
    phasor: Complex64,
    gain_weight: f64,
    phase_weight: f64,
) -> Result<Complex64, MeasureError> {
    if gain_weight == 0.0 && phase_weight == 0.0 {
        return Ok(Complex64::default());
    }
    if arithmetic::magnitude(phasor) == 0.0 || !phasor.re.is_finite() || !phasor.im.is_finite() {
        return Err(MeasureError::NotDifferentiable);
    }
    let numerator = Complex64::new(gain_weight * (20.0 / LN_10), -phase_weight * (180.0 / PI));
    let weight = arithmetic::divide(numerator, phasor);
    if !weight.re.is_finite() || !weight.im.is_finite() {
        return Err(MeasureError::NotDifferentiable);
    }
    Ok(weight)
}

/// Evaluates `Measure::Gain` and its exact sparse complex cotangent.
fn gain_cotangent(
    transfer: Transfer,
    at: f64,
    results: &AnalysisResult<'_>,
) -> Result<(f64, MeasureCotangent), MeasureError> {
    let ordered = OrderedTransfer::extract(transfer, results)?;
    let point = Axis(&ordered.frequencies).locate(at, Point::At)?.ln();
    let logs = logarithms(&ordered.frequencies);
    let gains = decibels(&ordered.values);
    let sample = arithmetic::interpolate_weights(point, &logs, &gains);
    let mut gain_weights = vec![0.0; ordered.values.len()];
    let phase_weights = vec![0.0; ordered.values.len()];
    gain_weights[sample.first.0] += sample.first.1;
    if let Some((second_index, second_weight)) = sample.second {
        gain_weights[second_index] += second_weight;
    }
    let cotangent = ordered.pull_back(&gain_weights, &phase_weights)?;
    Ok((sample.value, cotangent))
}

/// Evaluates `Measure::Phase` and its exact sparse complex cotangent.
fn phase_cotangent(
    transfer: Transfer,
    at: f64,
    results: &AnalysisResult<'_>,
) -> Result<(f64, MeasureCotangent), MeasureError> {
    let ordered = OrderedTransfer::extract(transfer, results)?;
    let point = Axis(&ordered.frequencies).locate(at, Point::At)?.ln();
    let logs = logarithms(&ordered.frequencies);
    let phases = arithmetic::phases(&ordered.values);
    let sample = arithmetic::interpolate_weights(point, &logs, &phases);
    let gain_weights = vec![0.0; ordered.values.len()];
    let mut phase_weights = vec![0.0; ordered.values.len()];
    phase_weights[sample.first.0] += sample.first.1;
    if let Some((second_index, second_weight)) = sample.second {
        phase_weights[second_index] += second_weight;
    }
    let cotangent = ordered.pull_back(&gain_weights, &phase_weights)?;
    Ok((sample.value, cotangent))
}

/// Evaluates `Measure::Bandwidth` and its exact sparse complex cotangent.
fn bandwidth_cotangent(
    transfer: Transfer,
    drop: f64,
    results: &AnalysisResult<'_>,
) -> Result<(f64, MeasureCotangent), MeasureError> {
    let ordered = OrderedTransfer::extract(transfer, results)?;
    let gains = decibels(&ordered.values);
    let logs = logarithms(&ordered.frequencies);
    let crossing = arithmetic::indexed_crossings(&logs, &gains, gains[0] - drop, Edge::Falling)
        .next()
        .ok_or(MeasureError::Stays(ordered.frequencies[0]))?;
    let value = crossing.point.exp();
    let mut gain_weights = vec![0.0; ordered.values.len()];
    let phase_weights = vec![0.0; ordered.values.len()];
    let threshold_weight = pull_crossing(crossing, &logs, &gains, value, &mut gain_weights, None);
    gain_weights[0] += threshold_weight;
    let cotangent = ordered.pull_back(&gain_weights, &phase_weights)?;
    Ok((value, cotangent))
}

/// Evaluates `Measure::Crossover` and its exact sparse complex cotangent.
fn crossover_cotangent(
    transfer: Transfer,
    results: &AnalysisResult<'_>,
) -> Result<(f64, MeasureCotangent), MeasureError> {
    let ordered = OrderedTransfer::extract(transfer, results)?;
    let gains = decibels(&ordered.values);
    let logs = logarithms(&ordered.frequencies);
    let crossing = arithmetic::indexed_crossings(&logs, &gains, 0.0, Edge::Falling)
        .next()
        .ok_or(MeasureError::Uncrossed)?;
    let value = crossing.point.exp();
    let mut gain_weights = vec![0.0; ordered.values.len()];
    let phase_weights = vec![0.0; ordered.values.len()];
    pull_crossing(crossing, &logs, &gains, value, &mut gain_weights, None);
    let cotangent = ordered.pull_back(&gain_weights, &phase_weights)?;
    Ok((value, cotangent))
}

/// Evaluates `Measure::PhaseMargin` and its exact sparse complex cotangent.
fn phase_margin_cotangent(
    transfer: Transfer,
    results: &AnalysisResult<'_>,
) -> Result<(f64, MeasureCotangent), MeasureError> {
    let ordered = OrderedTransfer::extract(transfer, results)?;
    let gains = decibels(&ordered.values);
    let logs = logarithms(&ordered.frequencies);
    let crossing = arithmetic::indexed_crossings(&logs, &gains, 0.0, Edge::Falling)
        .next()
        .ok_or(MeasureError::Uncrossed)?;
    let phases = arithmetic::phases(&ordered.values);
    let phase_sample = arithmetic::interpolate_weights(crossing.point, &logs, &phases);
    let mut gain_weights = vec![0.0; ordered.values.len()];
    let mut phase_weights = vec![0.0; ordered.values.len()];
    phase_weights[phase_sample.first.0] += phase_sample.first.1;
    if let Some((second_index, second_weight)) = phase_sample.second {
        phase_weights[second_index] += second_weight;
    }
    pull_crossing(
        crossing,
        &logs,
        &gains,
        phase_sample.slope,
        &mut gain_weights,
        None,
    );
    let cotangent = ordered.pull_back(&gain_weights, &phase_weights)?;
    Ok((180.0 + phase_sample.value, cotangent))
}

/// Evaluates `Measure::GainMargin` and its exact sparse complex cotangent.
fn gain_margin_cotangent(
    transfer: Transfer,
    results: &AnalysisResult<'_>,
) -> Result<(f64, MeasureCotangent), MeasureError> {
    let ordered = OrderedTransfer::extract(transfer, results)?;
    let logs = logarithms(&ordered.frequencies);
    let phases = arithmetic::phases(&ordered.values);
    let crossing = arithmetic::indexed_crossings(&logs, &phases, -180.0, Edge::Falling)
        .next()
        .ok_or(MeasureError::Uncrossed)?;
    let gains = decibels(&ordered.values);
    let gain_sample = arithmetic::interpolate_weights(crossing.point, &logs, &gains);
    let mut gain_weights = vec![0.0; ordered.values.len()];
    let mut phase_weights = vec![0.0; ordered.values.len()];
    gain_weights[gain_sample.first.0] -= gain_sample.first.1;
    if let Some((second_index, second_weight)) = gain_sample.second {
        gain_weights[second_index] -= second_weight;
    }
    pull_crossing(
        crossing,
        &logs,
        &phases,
        -gain_sample.slope,
        &mut phase_weights,
        None,
    );
    let cotangent = ordered.pull_back(&gain_weights, &phase_weights)?;
    Ok((-gain_sample.value, cotangent))
}

/// Sorted noise spectrum retaining original indices and optional input gains for pullback.
struct OrderedNoise {
    order: Vec<usize>,
    frequencies: Vec<f64>,
    effective_densities: Vec<f64>,
    gains: Option<Vec<Complex64>>,
}

impl OrderedNoise {
    /// Extracts ascending frequencies and effective (possibly input-referred) noise densities.
    fn extract(results: &AnalysisResult<'_>, referred: bool) -> Result<Self, MeasureError> {
        let AnalysisResult::Noise {
            frequencies,
            densities,
            gains,
        } = *results
        else {
            return Err(MeasureError::Kind);
        };
        if frequencies.len() != densities.len() {
            return Err(MeasureError::LengthMismatch);
        }
        if frequencies.is_empty() {
            return Err(MeasureError::Empty);
        }
        let (effective, raw_gains) = match (referred, gains) {
            (false, _) => (densities.to_vec(), None),
            (true, Some(gains)) if gains.len() == densities.len() => {
                (arithmetic::referred(densities, gains), Some(gains))
            }
            (true, Some(_)) => return Err(MeasureError::LengthMismatch),
            (true, None) => return Err(MeasureError::Unreferred),
        };
        let order = arithmetic::order(frequencies);
        let sorted_frequencies = order.iter().map(|&index| frequencies[index]).collect();
        let sorted_effective = order.iter().map(|&index| effective[index]).collect();
        let sorted_gains = raw_gains.map(|gains| order.iter().map(|&index| gains[index]).collect());
        Ok(Self {
            order,
            frequencies: sorted_frequencies,
            effective_densities: sorted_effective,
            gains: sorted_gains,
        })
    }

    /// Pulls sorted effective-density cotangents back to output or input-referred densities.
    fn pull_back(&self, effective_weights: &[f64]) -> Result<MeasureCotangent, MeasureError> {
        let mut density_map: BTreeMap<usize, f64> = BTreeMap::new();
        let mut referred_map: BTreeMap<usize, f64> = BTreeMap::new();
        for (index, &weight) in effective_weights.iter().enumerate() {
            if weight == 0.0 {
                continue;
            }
            if !weight.is_finite() {
                return Err(MeasureError::NotDifferentiable);
            }
            let original = self.order[index];
            match &self.gains {
                None => {
                    *density_map.entry(original).or_default() += weight;
                }
                Some(gains) => {
                    let magnitude = arithmetic::magnitude(gains[index]);
                    if magnitude == 0.0 || !magnitude.is_finite() {
                        return Err(MeasureError::NotDifferentiable);
                    }
                    *referred_map.entry(original).or_default() += weight;
                }
            }
        }
        let densities = collect_point_real(density_map)?;
        let referred = collect_point_real(referred_map)?;
        Ok(MeasureCotangent::Noise {
            densities,
            referred,
        })
    }
}

/// Evaluates `Measure::NoiseDensity` and its exact sparse noise cotangent.
fn noise_density_cotangent(
    at: f64,
    referred: bool,
    results: &AnalysisResult<'_>,
) -> Result<(f64, MeasureCotangent), MeasureError> {
    let AnalysisResult::Noise {
        frequencies: raw_frequencies,
        ..
    } = *results
    else {
        return Err(MeasureError::Kind);
    };
    let ordered = OrderedNoise::extract(results, referred)?;
    let located = Axis(raw_frequencies).locate(at, Point::At)?;
    let (first, last) = (
        ordered.frequencies[0],
        ordered.frequencies[ordered.frequencies.len() - 1],
    );
    if !(first <= located && located <= last) {
        return Err(MeasureError::Outside {
            point: Point::At,
            first,
            last,
        });
    }
    let sample =
        arithmetic::power_law_weights(&ordered.frequencies, &ordered.effective_densities, located);
    let value = sample.value.sqrt();
    if value == 0.0 || !value.is_finite() {
        return Err(MeasureError::NotDifferentiable);
    }
    let scale = 0.5 / value;
    let mut effective_weights = vec![0.0; ordered.effective_densities.len()];
    accumulate_power_law(&mut effective_weights, sample, scale);
    let cotangent = ordered.pull_back(&effective_weights)?;
    Ok((value, cotangent))
}

/// Evaluates `Measure::IntegratedNoise` and its exact sparse noise cotangent.
fn integrated_noise_cotangent(
    start: Option<f64>,
    stop: Option<f64>,
    referred: bool,
    results: &AnalysisResult<'_>,
) -> Result<(f64, MeasureCotangent), MeasureError> {
    let ordered = OrderedNoise::extract(results, referred)?;
    let axis = Axis(&ordered.frequencies);
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
    let first = ordered.frequencies[0];
    let last = ordered.frequencies[ordered.frequencies.len() - 1];
    let band_start = low.unwrap_or(first);
    let band_stop = high.unwrap_or(last);
    if !(first <= band_start && band_start < band_stop && band_stop <= last) {
        return Err(MeasureError::Band {
            start: band_start,
            stop: band_stop,
            first,
            last,
        });
    }
    let start_sample = arithmetic::power_law_weights(
        &ordered.frequencies,
        &ordered.effective_densities,
        band_start,
    );
    let stop_sample = arithmetic::power_law_weights(
        &ordered.frequencies,
        &ordered.effective_densities,
        band_stop,
    );
    let mut band_frequencies = vec![band_start];
    let mut band_densities = vec![start_sample.value];
    let mut interior_indices = Vec::new();
    for (index, &frequency) in ordered.frequencies.iter().enumerate() {
        if frequency > band_start && frequency < band_stop {
            band_frequencies.push(frequency);
            band_densities.push(ordered.effective_densities[index]);
            interior_indices.push(index);
        }
    }
    band_frequencies.push(band_stop);
    band_densities.push(stop_sample.value);

    let mut areas = Vec::with_capacity(band_frequencies.len() - 1);
    let mut band_weights = vec![0.0; band_frequencies.len()];
    for index in 0..band_frequencies.len() - 1 {
        let (area, derivative_low, derivative_high) = arithmetic::segment_weights(
            band_frequencies[index],
            band_frequencies[index + 1],
            band_densities[index],
            band_densities[index + 1],
        );
        areas.push(area);
        band_weights[index] += derivative_low;
        band_weights[index + 1] += derivative_high;
    }
    let total_power = arithmetic::exact_sum(areas).map_err(MeasureError::Sum)?;
    let value = total_power.sqrt();
    if value == 0.0 || !value.is_finite() {
        return Err(MeasureError::NotDifferentiable);
    }
    let scale = 0.5 / value;
    let mut effective_weights = vec![0.0; ordered.effective_densities.len()];
    accumulate_power_law(
        &mut effective_weights,
        start_sample,
        band_weights[0] * scale,
    );
    for (offset, &source_index) in interior_indices.iter().enumerate() {
        effective_weights[source_index] += band_weights[offset + 1] * scale;
    }
    accumulate_power_law(
        &mut effective_weights,
        stop_sample,
        band_weights[band_weights.len() - 1] * scale,
    );
    let cotangent = ordered.pull_back(&effective_weights)?;
    Ok((value, cotangent))
}

/// Accumulates scaled power-law interpolation weights into `weights`.
fn accumulate_power_law(weights: &mut [f64], sample: PowerLawPoint, scale: f64) {
    weights[sample.first.0] += scale * sample.first.1;
    if let Some((second_index, second_weight)) = sample.second {
        weights[second_index] += scale * second_weight;
    }
}

/// Evaluates `Measure::Damping` and its exact bilinear complex pole cotangent.
fn damping_cotangent(
    results: &AnalysisResult<'_>,
) -> Result<(f64, MeasureCotangent), MeasureError> {
    let roots = poles(results)?;
    let values = dampings(roots);
    let (index, value) =
        arithmetic::minimum_index(&values).ok_or(MeasureError::NotDifferentiable)?;
    let root = roots[index];
    let radius = arithmetic::magnitude(root);
    if radius == 0.0 || !radius.is_finite() {
        return Err(MeasureError::NotDifferentiable);
    }
    let cubed = radius * radius * radius;
    let weight = Complex64::new(-(root.im * root.im) / cubed, -(root.re * root.im) / cubed);
    if !weight.re.is_finite() || !weight.im.is_finite() {
        return Err(MeasureError::NotDifferentiable);
    }
    let entries = if weight == Complex64::default() {
        Vec::new()
    } else {
        vec![(index, weight)]
    };
    Ok((value, MeasureCotangent::Poles { entries }))
}

/// Evaluates `Measure::DominantPole` and its exact bilinear complex pole cotangent.
fn dominant_pole_cotangent(
    results: &AnalysisResult<'_>,
) -> Result<(f64, MeasureCotangent), MeasureError> {
    let roots = poles(results)?;
    let values = natural_frequencies(roots);
    let (index, value) =
        arithmetic::minimum_index(&values).ok_or(MeasureError::NotDifferentiable)?;
    let root = roots[index];
    let radius = arithmetic::magnitude(root);
    if radius == 0.0 || !radius.is_finite() {
        return Err(MeasureError::NotDifferentiable);
    }
    let denominator = TAU * radius;
    let weight = Complex64::new(root.re / denominator, -root.im / denominator);
    if !weight.re.is_finite() || !weight.im.is_finite() {
        return Err(MeasureError::NotDifferentiable);
    }
    let entries = if weight == Complex64::default() {
        Vec::new()
    } else {
        vec![(index, weight)]
    };
    Ok((value, MeasureCotangent::Poles { entries }))
}

/// Scatters real per-point weights on `signal` into sorted, finite, nonzero
/// `(point, unknown, weight)` entries.
fn scatter_real(
    signal: Signal,
    point_weights: &[(usize, f64)],
) -> Result<Vec<(usize, usize, f64)>, MeasureError> {
    let mut map: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    for &(point, weight) in point_weights {
        if weight == 0.0 {
            continue;
        }
        if !weight.is_finite() {
            return Err(MeasureError::NotDifferentiable);
        }
        *map.entry((point, signal.unknown)).or_default() += weight;
        if let Some(reference) = signal.reference {
            *map.entry((point, reference)).or_default() -= weight;
        }
    }
    let mut entries = Vec::with_capacity(map.len());
    for ((point, unknown), weight) in map {
        if !weight.is_finite() {
            return Err(MeasureError::NotDifferentiable);
        }
        if weight != 0.0 {
            entries.push((point, unknown, weight));
        }
    }
    Ok(entries)
}

/// Accumulates a complex weight on `signal` at `point` into `map`.
fn add_signal_complex(
    map: &mut BTreeMap<(usize, usize), Complex64>,
    point: usize,
    signal: Signal,
    weight: Complex64,
) -> Result<(), MeasureError> {
    if !weight.re.is_finite() || !weight.im.is_finite() {
        return Err(MeasureError::NotDifferentiable);
    }
    if weight == Complex64::default() {
        return Ok(());
    }
    *map.entry((point, signal.unknown)).or_default() += weight;
    if let Some(reference) = signal.reference {
        *map.entry((point, reference)).or_default() -= weight;
    }
    Ok(())
}

/// Filters `map` into sorted, finite, nonzero `(point, weight)` real entries.
fn collect_point_real(map: BTreeMap<usize, f64>) -> Result<Vec<(usize, f64)>, MeasureError> {
    let mut entries = Vec::with_capacity(map.len());
    for (point, weight) in map {
        if !weight.is_finite() {
            return Err(MeasureError::NotDifferentiable);
        }
        if weight != 0.0 {
            entries.push((point, weight));
        }
    }
    Ok(entries)
}

/// Filters `map` into sorted, finite, nonzero `(point, weight)` complex entries.
fn collect_point_complex(
    map: BTreeMap<usize, Complex64>,
) -> Result<Vec<(usize, Complex64)>, MeasureError> {
    let zero = Complex64::default();
    let mut entries = Vec::with_capacity(map.len());
    for (point, weight) in map {
        if !weight.re.is_finite() || !weight.im.is_finite() {
            return Err(MeasureError::NotDifferentiable);
        }
        if weight != zero {
            entries.push((point, weight));
        }
    }
    Ok(entries)
}

/// Filters `map` into sorted, finite, nonzero `(point, unknown, weight)` complex entries.
fn collect_table_complex(
    map: BTreeMap<(usize, usize), Complex64>,
) -> Result<Vec<(usize, usize, Complex64)>, MeasureError> {
    let zero = Complex64::default();
    let mut entries = Vec::with_capacity(map.len());
    for ((point, unknown), weight) in map {
        if !weight.re.is_finite() || !weight.im.is_finite() {
            return Err(MeasureError::NotDifferentiable);
        }
        if weight != zero {
            entries.push((point, unknown, weight));
        }
    }
    Ok(entries)
}

#[cfg(test)]
mod tests {
    //! Central finite-difference verification of every [`Measure`] cotangent variant.

    use super::*;

    /// Central finite-difference relative tolerance (`1e-6`).
    ///
    /// Derivation: central differences at step `1e-6` carry `O(step^2) = 1e-12` truncation
    /// error plus `O(f64::EPSILON / step) ~ 2e-10` cancellation error; `1e-6` catches any
    /// missing term or wrong sign while accommodating nonlinear curvature in gain, phase,
    /// and crossing ratios.
    const FINITE_DIFFERENCE_TOLERANCE: f64 = 1e-6;

    /// Step size used for central finite differences on result values (`1e-6`).
    ///
    /// Derivation: balances `O(step^2)` truncation against `O(eps / step)` rounding near
    /// `eps^(1/3) ~ 6e-6` (Press et al., *Numerical Recipes*, 3rd ed., §5.7).
    const STEP: f64 = 1e-6;

    fn assert_close(exact: f64, finite: f64, scale: f64, label: &str) {
        let bound = FINITE_DIFFERENCE_TOLERANCE * scale.max(1.0);
        assert!(
            (exact - finite).abs() <= bound,
            "{label}: exact {exact:.12e} vs finite {finite:.12e} (diff {:.3e}, bound {bound:.3e})",
            (exact - finite).abs()
        );
    }

    fn verify_real_table(
        measure: Measure,
        axis: &[f64],
        base: &[Vec<f64>],
        build: impl for<'a> Fn(&'a [f64], &'a Vec<Vec<f64>>) -> AnalysisResult<'a>,
        label: &str,
    ) {
        let mut values = base.to_vec();
        let primal = measure.measure(&build(axis, &values)).unwrap();
        let (value, cotangent) = measure.cotangent(&build(axis, &values)).unwrap();
        assert_eq!(
            value.to_bits(),
            primal.to_bits(),
            "{label}: primal mismatch"
        );
        let MeasureCotangent::Real { entries, .. } = cotangent else {
            panic!("{label}: expected real cotangent");
        };
        let lookup: BTreeMap<(usize, usize), f64> = entries
            .into_iter()
            .map(|(point, unknown, weight)| ((point, unknown), weight))
            .collect();
        let scale = lookup
            .values()
            .fold(0.0_f64, |largest, &weight| largest.max(weight.abs()));
        for point in 0..values.len() {
            for unknown in 0..values[point].len() {
                let original = values[point][unknown];
                let step = STEP * original.abs().max(1.0);
                values[point][unknown] = original + step;
                let above = measure.measure(&build(axis, &values)).unwrap();
                values[point][unknown] = original - step;
                let below = measure.measure(&build(axis, &values)).unwrap();
                values[point][unknown] = original;
                let finite = (above - below) / (2.0 * step);
                let exact = lookup.get(&(point, unknown)).copied().unwrap_or(0.0);
                assert_close(exact, finite, scale, &format!("{label}[{point},{unknown}]"));
            }
        }
    }

    fn verify_complex_vector(
        values: &[Complex64],
        entries: &[(usize, Complex64)],
        eval: impl Fn(&[Complex64]) -> f64,
        label: &str,
    ) {
        let mut perturbed = values.to_vec();
        let lookup: BTreeMap<usize, Complex64> = entries.iter().copied().collect();
        let scale = lookup
            .values()
            .fold(0.0_f64, |largest, weight| largest.max(weight.norm()));
        for index in 0..perturbed.len() {
            let original = perturbed[index];
            let step = STEP * original.norm().max(1.0);
            perturbed[index] = Complex64::new(original.re + step, original.im);
            let real_plus = eval(&perturbed);
            perturbed[index] = Complex64::new(original.re - step, original.im);
            let real_minus = eval(&perturbed);
            perturbed[index] = Complex64::new(original.re, original.im + step);
            let imag_plus = eval(&perturbed);
            perturbed[index] = Complex64::new(original.re, original.im - step);
            let imag_minus = eval(&perturbed);
            perturbed[index] = original;
            let exact = lookup.get(&index).copied().unwrap_or_default();
            // Bilinear convention: `exact = dL/da - j dL/db`.
            assert_close(
                exact.re,
                (real_plus - real_minus) / (2.0 * step),
                scale,
                &format!("{label}[{index}].re"),
            );
            assert_close(
                -exact.im,
                (imag_plus - imag_minus) / (2.0 * step),
                scale,
                &format!("{label}[{index}].im"),
            );
        }
    }

    fn verify_complex_table(
        measure: Measure,
        frequencies: &[f64],
        base: &[Vec<Complex64>],
        label: &str,
    ) {
        let mut values = base.to_vec();
        let eval = |table: &Vec<Vec<Complex64>>| {
            measure
                .measure(&AnalysisResult::Ac {
                    frequencies,
                    values: table,
                })
                .unwrap()
        };
        let primal = eval(&values);
        let (value, cotangent) = measure
            .cotangent(&AnalysisResult::Ac {
                frequencies,
                values: &values,
            })
            .unwrap();
        assert_eq!(
            value.to_bits(),
            primal.to_bits(),
            "{label}: primal mismatch"
        );
        let MeasureCotangent::Complex { entries } = cotangent else {
            panic!("{label}: expected complex cotangent");
        };
        let lookup: BTreeMap<(usize, usize), Complex64> = entries
            .into_iter()
            .map(|(point, unknown, weight)| ((point, unknown), weight))
            .collect();
        let scale = lookup
            .values()
            .fold(0.0_f64, |largest, weight| largest.max(weight.norm()));
        for point in 0..values.len() {
            for unknown in 0..values[point].len() {
                let original = values[point][unknown];
                let step = STEP * original.norm().max(1.0);
                values[point][unknown] = Complex64::new(original.re + step, original.im);
                let real_plus = eval(&values);
                values[point][unknown] = Complex64::new(original.re - step, original.im);
                let real_minus = eval(&values);
                values[point][unknown] = Complex64::new(original.re, original.im + step);
                let imag_plus = eval(&values);
                values[point][unknown] = Complex64::new(original.re, original.im - step);
                let imag_minus = eval(&values);
                values[point][unknown] = original;
                let exact = lookup.get(&(point, unknown)).copied().unwrap_or_default();
                assert_close(
                    exact.re,
                    (real_plus - real_minus) / (2.0 * step),
                    scale,
                    &format!("{label}[{point},{unknown}].re"),
                );
                assert_close(
                    -exact.im,
                    (imag_plus - imag_minus) / (2.0 * step),
                    scale,
                    &format!("{label}[{point},{unknown}].im"),
                );
            }
        }
    }

    #[test]
    fn operating_point_cotangent_matches_finite_differences() {
        let mut point = vec![2.5, 0.7, -1.2];
        let differential = Signal {
            unknown: 0,
            reference: Some(1),
        };
        let measure = Measure::Value {
            signal: differential,
            at: None,
        };
        let (value, cotangent) = measure
            .cotangent(&AnalysisResult::OperatingPoint(&point))
            .unwrap();
        assert_eq!(value.to_bits(), (2.5_f64 - 0.7_f64).to_bits());
        assert_eq!(
            cotangent,
            MeasureCotangent::Real {
                entries: vec![(0, 0, 1.0), (0, 1, -1.0)],
                period: 0.0,
            }
        );
        let MeasureCotangent::Real { entries, .. } = cotangent else {
            panic!("expected real cotangent");
        };
        for &(row, unknown, weight) in &entries {
            assert_eq!(row, 0);
            let original = point[unknown];
            point[unknown] = original + STEP;
            let above = measure
                .measure(&AnalysisResult::OperatingPoint(&point))
                .unwrap();
            point[unknown] = original - STEP;
            let below = measure
                .measure(&AnalysisResult::OperatingPoint(&point))
                .unwrap();
            point[unknown] = original;
            assert_close(weight, (above - below) / (2.0 * STEP), 1.0, "op");
        }
    }

    #[test]
    fn dc_sweep_cotangents_match_finite_differences() {
        let differential = Signal {
            unknown: 0,
            reference: Some(1),
        };
        // Descending DC sweep exercising axis reversal and window endpoint interpolation.
        let swept = [5.0, 3.5, 2.0, 1.0, 0.0];
        let rows = vec![
            vec![4.2, 0.5],
            vec![2.9, 0.4],
            vec![1.7, 0.2],
            vec![0.8, 0.1],
            vec![-0.3, 0.0],
        ];
        let window = Window {
            start: Some(0.5),
            stop: Some(4.25),
        };
        for (label, dc_measure) in [
            (
                "dc_value",
                Measure::Value {
                    signal: differential,
                    at: Some(2.75),
                },
            ),
            (
                "dc_max",
                Measure::Maximum {
                    signal: differential,
                    window,
                },
            ),
            (
                "dc_min",
                Measure::Minimum {
                    signal: differential,
                    window,
                },
            ),
            (
                "dc_p2p",
                Measure::PeakToPeak {
                    signal: differential,
                    window,
                },
            ),
            (
                "dc_mean",
                Measure::Mean {
                    signal: differential,
                    window,
                },
            ),
            (
                "dc_rms",
                Measure::Rms {
                    signal: differential,
                    window,
                },
            ),
        ] {
            verify_real_table(
                dc_measure,
                &swept,
                &rows,
                |axis, table| AnalysisResult::DcSweep {
                    swept: axis,
                    values: table,
                },
                label,
            );
        }
    }

    #[test]
    fn transient_step_measure_cotangents_match_finite_differences() {
        let signal = Signal {
            unknown: 0,
            reference: Some(1),
        };
        // Underdamped step response with overshoot and settling outside a 5% band.
        let times: Vec<f64> = (0..41).map(|step| f64::from(step) * 0.25e-6).collect();
        let step_values: Vec<Vec<f64>> = times
            .iter()
            .map(|&time| {
                let damp = (-0.5e6 * time).exp();
                let response = 1.0 - damp * ((2.0e6 * time).cos() + 0.25 * (2.0e6 * time).sin());
                vec![response + 0.3, 0.3]
            })
            .collect();
        let window = Window {
            start: Some(0.1e-6),
            stop: Some(9.6e-6),
        };
        for (label, measure) in [
            (
                "tr_value",
                Measure::Value {
                    signal,
                    at: Some(1.37e-6),
                },
            ),
            ("tr_max", Measure::Maximum { signal, window }),
            ("tr_min", Measure::Minimum { signal, window }),
            ("tr_p2p", Measure::PeakToPeak { signal, window }),
            ("tr_mean", Measure::Mean { signal, window }),
            ("tr_rms", Measure::Rms { signal, window }),
            (
                "tr_rise",
                Measure::RiseTime {
                    signal,
                    window,
                    lower: 0.1,
                    upper: 0.9,
                },
            ),
            ("tr_overshoot", Measure::Overshoot { signal, window }),
            (
                "tr_settling",
                Measure::SettlingTime {
                    signal,
                    window,
                    band: 0.05,
                },
            ),
            (
                "tr_crossing",
                Measure::Crossing {
                    signal,
                    window,
                    threshold: 0.55,
                    edge: Edge::Rising,
                },
            ),
        ] {
            verify_real_table(
                measure,
                &times,
                &step_values,
                |axis, table| AnalysisResult::Transient {
                    times: axis,
                    values: table,
                },
                label,
            );
        }
    }

    #[test]
    fn transient_periodic_measure_cotangents_match_finite_differences() {
        let signal = Signal {
            unknown: 0,
            reference: Some(1),
        };
        let periodic_times: Vec<f64> = (0..81).map(|step| f64::from(step) * 0.05e-3).collect();
        let distorted: Vec<Vec<f64>> = periodic_times
            .iter()
            .map(|&time| {
                let angle = TAU * 1e3 * time;
                let wave = 1.5 * angle.sin()
                    + 0.12 * (2.0 * angle + 0.3).sin()
                    + 0.05 * (3.0 * angle - 0.4).cos()
                    + 0.2;
                vec![wave, 0.2]
            })
            .collect();
        let full = Window::default();
        let partial = Window {
            start: Some(0.12e-3),
            stop: Some(3.87e-3),
        };
        for (label, measure) in [
            (
                "freq_explicit",
                Measure::Frequency {
                    signal,
                    window: partial,
                    threshold: Some(0.1),
                },
            ),
            (
                "freq_midpoint",
                Measure::Frequency {
                    signal,
                    window: partial,
                    threshold: None,
                },
            ),
            (
                "thd_last",
                Measure::Distortion {
                    signal,
                    window: full,
                    fundamental: 1e3,
                    harmonics: 5,
                },
            ),
            (
                "thd_whole",
                Measure::Distortion {
                    signal,
                    window: partial,
                    fundamental: 1e3,
                    harmonics: 5,
                },
            ),
        ] {
            verify_real_table(
                measure,
                &periodic_times,
                &distorted,
                |axis, table| AnalysisResult::Transient {
                    times: axis,
                    values: table,
                },
                label,
            );
        }
    }

    #[test]
    fn ac_transfer_cotangents_match_finite_differences() {
        // Unsorted three-pole response with gain > 0 dB at DC crossing 0 dB and -180 deg.
        let frequencies = [1e1, 1e4, 1e2, 3e2, 1e3, 3e3, 3e4, 1e5];
        let values: Vec<Vec<Complex64>> = frequencies
            .iter()
            .map(|&frequency| {
                let ratio = Complex64::new(1.0, frequency / 500.0);
                let divisor = Complex64::new(1.2, 0.15 * (frequency / 1e4));
                let loop_response = Complex64::new(12.0, 0.0) / (ratio * ratio * ratio);
                vec![loop_response * divisor, divisor, Complex64::new(0.1, -0.05)]
            })
            .collect();
        let output = Signal {
            unknown: 0,
            reference: None,
        };
        let input = Signal {
            unknown: 1,
            reference: None,
        };
        let differential = Signal {
            unknown: 0,
            reference: Some(2),
        };
        for (label, transfer) in [
            (
                "ac_direct",
                Transfer::Signal {
                    signal: differential,
                    over: None,
                },
            ),
            (
                "ac_ratio",
                Transfer::Signal {
                    signal: output,
                    over: Some(input),
                },
            ),
        ] {
            for (kind, measure) in [
                (
                    "gain",
                    Measure::Gain {
                        transfer,
                        at: 700.0,
                    },
                ),
                (
                    "phase",
                    Measure::Phase {
                        transfer,
                        at: 700.0,
                    },
                ),
                (
                    "bw",
                    Measure::Bandwidth {
                        transfer,
                        drop: 3.0,
                    },
                ),
                ("crossover", Measure::Crossover { transfer }),
                ("pm", Measure::PhaseMargin { transfer }),
                ("gm", Measure::GainMargin { transfer }),
            ] {
                verify_complex_table(measure, &frequencies, &values, &format!("{label}_{kind}"));
            }
        }
    }

    #[test]
    fn stability_loop_gain_cotangents_match_finite_differences() {
        let frequencies = [1e1, 1e4, 1e2, 3e2, 1e3, 3e3, 3e4, 1e5];
        let gains: Vec<Complex64> = frequencies
            .iter()
            .map(|&frequency| {
                let ratio = Complex64::new(1.0, frequency / 500.0);
                Complex64::new(12.0, 0.0) / (ratio * ratio * ratio)
            })
            .collect();
        for (kind, measure) in [
            (
                "loop_gain",
                Measure::Gain {
                    transfer: Transfer::Loop,
                    at: 700.0,
                },
            ),
            (
                "loop_phase",
                Measure::Phase {
                    transfer: Transfer::Loop,
                    at: 700.0,
                },
            ),
            (
                "loop_bw",
                Measure::Bandwidth {
                    transfer: Transfer::Loop,
                    drop: 3.0,
                },
            ),
            (
                "loop_crossover",
                Measure::Crossover {
                    transfer: Transfer::Loop,
                },
            ),
            (
                "loop_pm",
                Measure::PhaseMargin {
                    transfer: Transfer::Loop,
                },
            ),
            (
                "loop_gm",
                Measure::GainMargin {
                    transfer: Transfer::Loop,
                },
            ),
        ] {
            let result = AnalysisResult::Stability {
                frequencies: &frequencies,
                gains: &gains,
            };
            let (value, cotangent) = measure.cotangent(&result).unwrap();
            assert_eq!(value.to_bits(), measure.measure(&result).unwrap().to_bits());
            let MeasureCotangent::Loop { entries } = cotangent else {
                panic!("{kind}: expected loop cotangent");
            };
            verify_complex_vector(
                &gains,
                &entries,
                |perturbed| {
                    measure
                        .measure(&AnalysisResult::Stability {
                            frequencies: &frequencies,
                            gains: perturbed,
                        })
                        .unwrap()
                },
                kind,
            );
        }
    }

    fn verify_noise_measure(
        label: &str,
        measure: Measure,
        frequencies: &[f64],
        densities: &mut [f64],
        gains: &[Complex64],
    ) {
        let result = AnalysisResult::Noise {
            frequencies,
            densities,
            gains: Some(gains),
        };
        let (value, cotangent) = measure.cotangent(&result).unwrap();
        assert_eq!(value.to_bits(), measure.measure(&result).unwrap().to_bits());
        let MeasureCotangent::Noise {
            densities: density_entries,
            referred: referred_entries,
        } = cotangent
        else {
            panic!("{label}: expected noise cotangent");
        };
        let density_lookup: BTreeMap<usize, f64> = density_entries.into_iter().collect();
        let referred_lookup: BTreeMap<usize, f64> = referred_entries.into_iter().collect();
        let expected_densities: BTreeMap<usize, f64> = (0..densities.len())
            .map(|index| {
                let magnitude = arithmetic::magnitude(gains[index]);
                let from_density = density_lookup.get(&index).copied().unwrap_or(0.0);
                let from_referred =
                    referred_lookup.get(&index).copied().unwrap_or(0.0) / (magnitude * magnitude);
                (index, from_density + from_referred)
            })
            .collect();
        let expected_gains: Vec<(usize, Complex64)> = referred_lookup
            .iter()
            .map(|(&index, &weight)| {
                let magnitude = arithmetic::magnitude(gains[index]);
                let referred_value = densities[index] / (magnitude * magnitude);
                let gain_weight = arithmetic::divide(
                    Complex64::new(-2.0 * weight * referred_value, 0.0),
                    gains[index],
                );
                (index, gain_weight)
            })
            .collect();
        let density_scale = expected_densities
            .values()
            .fold(0.0_f64, |largest, &weight| largest.max(weight.abs()));
        for index in 0..densities.len() {
            let original = densities[index];
            let step = STEP * original;
            densities[index] = original + step;
            let above = measure
                .measure(&AnalysisResult::Noise {
                    frequencies,
                    densities,
                    gains: Some(gains),
                })
                .unwrap();
            densities[index] = original - step;
            let below = measure
                .measure(&AnalysisResult::Noise {
                    frequencies,
                    densities,
                    gains: Some(gains),
                })
                .unwrap();
            densities[index] = original;
            let exact = expected_densities.get(&index).copied().unwrap_or(0.0);
            assert_close(exact, (above - below) / (2.0 * step), density_scale, label);
        }
        verify_complex_vector(
            gains,
            &expected_gains,
            |perturbed| {
                measure
                    .measure(&AnalysisResult::Noise {
                        frequencies,
                        densities,
                        gains: Some(perturbed),
                    })
                    .unwrap()
            },
            label,
        );
    }

    #[test]
    fn noise_cotangents_match_finite_differences() {
        // Include a 1/f segment (growth == 0) alongside general power-law segments.
        let frequencies = [1e1, 1e3, 1e2, 1e4, 1e5];
        let mut densities = vec![1e-14, 1e-16, 1e-15, 2e-17, 1.5e-17];
        let gains = vec![
            Complex64::new(2.0, 0.3),
            Complex64::new(1.5, -0.7),
            Complex64::new(1.9, -0.1),
            Complex64::new(0.8, -1.1),
            Complex64::new(0.3, -0.6),
        ];
        for (label, measure) in [
            (
                "spot_out",
                Measure::NoiseDensity {
                    at: 300.0,
                    referred: false,
                },
            ),
            (
                "spot_in",
                Measure::NoiseDensity {
                    at: 300.0,
                    referred: true,
                },
            ),
            (
                "int_out",
                Measure::IntegratedNoise {
                    start: Some(30.0),
                    stop: Some(3e4),
                    referred: false,
                },
            ),
            (
                "int_in",
                Measure::IntegratedNoise {
                    start: Some(30.0),
                    stop: Some(3e4),
                    referred: true,
                },
            ),
        ] {
            verify_noise_measure(label, measure, &frequencies, &mut densities, &gains);
        }
    }

    #[test]
    fn pole_zero_and_discrete_cotangents_match_finite_differences() {
        let roots = vec![
            Complex64::new(-3e3, 4e3),
            Complex64::new(-1e4, 0.0),
            Complex64::new(-2e3, 6e3),
        ];
        for (label, measure) in [
            ("damping", Measure::Damping),
            ("dominant_pole", Measure::DominantPole),
        ] {
            let (value, cotangent) = measure
                .cotangent(&AnalysisResult::PoleZero(&roots))
                .unwrap();
            assert_eq!(
                value.to_bits(),
                measure
                    .measure(&AnalysisResult::PoleZero(&roots))
                    .unwrap()
                    .to_bits()
            );
            let MeasureCotangent::Poles { entries } = cotangent else {
                panic!("{label}: expected pole cotangent");
            };
            verify_complex_vector(
                &roots,
                &entries,
                |perturbed| {
                    measure
                        .measure(&AnalysisResult::PoleZero(perturbed))
                        .unwrap()
                },
                label,
            );
        }
        assert_eq!(
            Measure::Solutions.cotangent(&AnalysisResult::OperatingPoints(3)),
            Err(MeasureError::NotDifferentiable)
        );
        assert_eq!(
            Measure::Solutions.cotangent(&AnalysisResult::OperatingPoint(&[1.0])),
            Err(MeasureError::Kind)
        );
    }

    #[test]
    fn periodic_steady_state_period_cotangents_match_finite_differences() {
        let period = 1e-6;
        let fractions: Vec<f64> = (0..21).map(|step| f64::from(step) / 20.0).collect();
        let values: Vec<Vec<f64>> = fractions
            .iter()
            .map(|&fraction| vec![(TAU * fraction).sin() + 0.2 * (2.0 * TAU * fraction).cos()])
            .collect();
        let times: Vec<f64> = fractions
            .iter()
            .map(|&fraction| fraction * period)
            .collect();
        let signal = Signal {
            unknown: 0,
            reference: None,
        };
        let pss = AnalysisResult::PeriodicSteadyState {
            times: &times,
            values: &values,
            period,
        };
        assert_eq!(
            Measure::Period.cotangent(&pss),
            Ok((
                period,
                MeasureCotangent::Real {
                    entries: Vec::new(),
                    period: 1.0,
                }
            ))
        );
        for (label, measure, unit_scale) in [
            (
                "pss_value",
                Measure::Value {
                    signal,
                    at: Some(0.37e-6),
                },
                1.0 / period,
            ),
            (
                "pss_crossing",
                Measure::Crossing {
                    signal,
                    window: Window::default(),
                    threshold: 0.3,
                    edge: Edge::Rising,
                },
                1.0,
            ),
            (
                "pss_mean_window",
                Measure::Mean {
                    signal,
                    window: Window {
                        start: Some(0.13e-6),
                        stop: Some(0.83e-6),
                    },
                },
                1.0 / period,
            ),
            (
                "pss_mean_full",
                Measure::Mean {
                    signal,
                    window: Window::default(),
                },
                1.0 / period,
            ),
        ] {
            let (_, cotangent) = measure.cotangent(&pss).unwrap();
            let MeasureCotangent::Real {
                period: period_weight,
                ..
            } = cotangent
            else {
                panic!("{label}");
            };
            let delta = STEP * period;
            let measure_at = |shifted: f64| {
                let shifted_times: Vec<f64> = fractions
                    .iter()
                    .map(|&fraction| fraction * shifted)
                    .collect();
                measure
                    .measure(&AnalysisResult::PeriodicSteadyState {
                        times: &shifted_times,
                        values: &values,
                        period: shifted,
                    })
                    .unwrap()
            };
            let above = measure_at(period + delta);
            let below = measure_at(period - delta);
            assert_close(
                period_weight,
                (above - below) / (2.0 * delta),
                period_weight.abs().max(unit_scale),
                label,
            );
        }
    }
}
