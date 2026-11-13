//! Waveforms: the level of an independent source over time.
//!
//! A waveform is written as a JSON object with one key, its shape:
//!
//! - `dc`: the level.
//! - `pulse`: `initial`, `pulsed`, `delay` (default 0), `rise`, `fall`, `width`, and `period`
//!   (by default, a single pulse).
//! - `sine`: `offset`, `amplitude`, `frequency`, `delay` (default 0), `damping` (default 0) and
//!   `phase` in radians (default 0).
//! - `piecewise`: `points`, a list of `[time, level]` pairs in increasing time.
//!
//! Times are in seconds, frequencies in hertz and damping in inverse seconds. A sine is
//! `offset + amplitude · e^{-damping·s} · sin(2π·frequency·s + phase)`, where `s` is the time since
//! its delay, and holds its value at `s = 0` before the delay. A piecewise waveform draws
//! straight lines between its points and holds its first and last levels outside them.

use std::f64::consts::TAU;

use serde::{Deserialize, Serialize};

use crate::{ParameterError, Real};

/// The level of an independent source over time, in volts or amperes.
///
/// Every analysis but a transient one reads the level at time zero. A waveform is checked as
/// it is read, so every value of this type is a valid waveform. Its numbers are plain, or
/// carry their derivatives with respect to a parameter.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(try_from = "Shape<R>", bound = "")]
pub struct Waveform<R: Real = f64>(Shape<R>);

impl<R: Real> Serialize for Waveform<R> {
    /// The waveform as it is written: its shape.
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(serializer)
    }
}

/// A waveform's shape, an object with one key, as it is written.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields, bound = "")]
enum Shape<R: Real> {
    /// A constant level.
    Dc(R),
    /// A trapezoidal pulse, repeated each period, if it has one.
    Pulse(Pulse<R>),
    /// A sine, damped from its delay on.
    Sine(Sine<R>),
    /// Straight lines between points, holding the first and last levels outside them.
    Piecewise {
        /// Each point's time and level, in strictly increasing time; at least one.
        points: Vec<(R, R)>,
    },
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields, bound = "")]
struct Pulse<R: Real> {
    /// The level before the delay and between pulses.
    initial: R,
    /// The level at the top of each pulse.
    pulsed: R,
    /// The time the first pulse starts its rise, in seconds.
    #[serde(default)]
    delay: R,
    /// The time each pulse takes to rise, in seconds; positive.
    rise: R,
    /// The time each pulse takes to fall, in seconds; positive.
    fall: R,
    /// The time each pulse holds its top, in seconds.
    width: R,
    /// The time from one pulse's start to the next's, in seconds; without it, a single
    /// pulse.
    #[serde(default)]
    period: Option<R>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields, bound = "")]
struct Sine<R: Real> {
    /// The level the sine swings about.
    offset: R,
    /// The sine's amplitude at its delay.
    amplitude: R,
    /// The sine's frequency, in hertz.
    frequency: R,
    /// The time the sine starts, in seconds, before which it holds its value then.
    #[serde(default)]
    delay: R,
    /// The rate at which the amplitude decays from the delay on, per second.
    #[serde(default)]
    damping: R,
    /// The phase at the delay, in radians.
    #[serde(default)]
    phase: R,
}

impl<R: Real> TryFrom<Shape<R>> for Waveform<R> {
    type Error = ParameterError;

    fn try_from(shape: Shape<R>) -> Result<Self, Self::Error> {
        match &shape {
            Shape::Dc(_) => {}
            Shape::Pulse(pulse) => {
                ParameterError::non_negative("pulse delay", pulse.delay)?;
                ParameterError::positive("pulse rise", pulse.rise)?;
                ParameterError::positive("pulse fall", pulse.fall)?;
                ParameterError::non_negative("pulse width", pulse.width)?;
                let duration = (pulse.rise + pulse.width + pulse.fall).value();
                if let Some(period) = pulse.period.map(Real::value)
                    && period < duration
                {
                    return Err(ParameterError(format!(
                        "pulse period {period:?} is shorter than one pulse, {duration:?}"
                    )));
                }
            }
            Shape::Sine(sine) => {
                ParameterError::non_negative("sine frequency", sine.frequency)?;
                ParameterError::non_negative("sine delay", sine.delay)?;
            }
            Shape::Piecewise { points } => {
                if points.is_empty() {
                    return Err(ParameterError(
                        "a piecewise waveform needs a point".to_owned(),
                    ));
                }
                if points
                    .windows(2)
                    .any(|pair| pair[1].0.value() <= pair[0].0.value())
                {
                    return Err(ParameterError(
                        "the times of a piecewise waveform must increase".to_owned(),
                    ));
                }
            }
        }
        Ok(Self(shape))
    }
}

impl Waveform {
    /// A constant `level`.
    #[must_use]
    pub fn dc(level: f64) -> Self {
        Self(Shape::Dc(level))
    }
}

impl<R: Real> Waveform<R> {
    /// The level at `time`.
    #[must_use]
    pub fn value(&self, time: f64) -> R {
        match &self.0 {
            Shape::Dc(level) => *level,
            Shape::Pulse(pulse) => pulse.value(time),
            Shape::Sine(sine) => sine.value(time),
            Shape::Piecewise { points } => piecewise(points, time),
        }
    }

    /// Appends the times in `[start, stop]` where the level has a corner.
    pub fn breakpoints(&self, start: f64, stop: f64, times: &mut Vec<f64>) {
        let mut push = |time: f64| {
            if (start..=stop).contains(&time) {
                times.push(time);
            }
        };
        match &self.0 {
            Shape::Dc(_) => {}
            Shape::Pulse(pulse) => {
                let (rise, width, fall) =
                    (pulse.rise.value(), pulse.width.value(), pulse.fall.value());
                let top = rise + width;
                let corners = [0.0, rise, top, top + fall];
                let mut origin = pulse.delay.value();
                while origin <= stop {
                    for corner in corners {
                        push(origin + corner);
                    }
                    let Some(period) = pulse.period else { break };
                    origin += period.value();
                }
            }
            Shape::Sine(sine) => {
                if sine.delay.value() > 0.0 {
                    push(sine.delay.value());
                }
            }
            Shape::Piecewise { points } => {
                for &(time, _) in points {
                    push(time.value());
                }
            }
        }
    }
}

impl<R: Real> Pulse<R> {
    fn value(&self, time: f64) -> R {
        if time < self.delay.value() {
            return self.initial;
        }
        let elapsed = -(self.delay - time);
        let local = self
            .period
            .map_or(elapsed, |period| elapsed.rem_euclid(period));
        let top = self.rise + self.width;
        if local.value() < self.rise.value() {
            self.initial + (self.pulsed - self.initial) * local / self.rise
        } else if local.value() < top.value() {
            self.pulsed
        } else if local.value() < (top + self.fall).value() {
            self.pulsed + (self.initial - self.pulsed) * (local - top) / self.fall
        } else {
            self.initial
        }
    }
}

impl<R: Real> Sine<R> {
    fn value(&self, time: f64) -> R {
        let since = (-(self.delay - time)).max(R::constant(0.0));
        let envelope = (-self.damping * since).exp();
        self.offset + self.amplitude * envelope * (self.frequency * TAU * since + self.phase).sin()
    }
}

/// Interpolates between `points`, which hold at least one point in increasing time.
fn piecewise<R: Real>(points: &[(R, R)], time: f64) -> R {
    let after = points.partition_point(|&(t, _)| t.value() <= time);
    if after == 0 {
        return points[0].1;
    }
    if after == points.len() {
        return points[after - 1].1;
    }
    let ((t0, v0), (t1, v1)) = (points[after - 1], points[after]);
    v0 + (v1 - v0) * (-(t0 - time)) / (t1 - t0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(json: &str) -> Result<Waveform, serde_json::Error> {
        serde_json::from_str(json)
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    const PULSE: &str =
        r#"{"pulse": {"initial": 0, "pulsed": 2, "delay": 1, "rise": 1, "fall": 2, "width": 3}}"#;

    #[test]
    fn a_pulse_rises_holds_and_falls() {
        let wave = read(PULSE).unwrap();
        let samples = [0.5, 1.5, 2.0, 4.0, 6.0, 7.0, 10.0].map(|t| wave.value(t));
        let expected = [0.0, 1.0, 2.0, 2.0, 1.0, 0.0, 0.0];
        assert!(
            samples.iter().zip(expected).all(|(&a, b)| close(a, b)),
            "{samples:?}"
        );
    }

    #[test]
    fn a_periodic_pulse_repeats_its_corners() {
        let wave = read(&PULSE.replace(r#""width": 3"#, r#""width": 3, "period": 10"#)).unwrap();
        assert!(close(wave.value(11.5), wave.value(1.5)));
        let mut times = Vec::new();
        wave.breakpoints(0.0, 15.0, &mut times);
        assert_eq!(times, [1.0, 2.0, 5.0, 7.0, 11.0, 12.0, 15.0]);
    }

    #[test]
    fn a_sine_holds_before_its_delay_and_decays_after() {
        let sine =
            r#"{"offset": 1, "amplitude": 2, "frequency": 0.25, "delay": 1, "damping": 0.5}"#;
        let wave = read(&format!(r#"{{"sine": {sine}}}"#)).unwrap();
        assert!(close(wave.value(0.0), 1.0));
        assert!(close(wave.value(2.0), 1.0 + 2.0 * (-0.5f64).exp()));
    }

    #[test]
    fn a_piecewise_waveform_interpolates_and_holds_its_ends() {
        let wave = read(r#"{"piecewise": {"points": [[1, 0], [3, 4]]}}"#).unwrap();
        let samples = [0.0, 1.0, 2.0, 3.0, 5.0].map(|t| wave.value(t));
        assert!(
            samples
                .iter()
                .zip([0.0, 0.0, 2.0, 4.0, 4.0])
                .all(|(&a, b)| close(a, b))
        );
    }

    #[test]
    fn rejects_impossible_waveforms_as_they_are_read() {
        let short_period = PULSE.replace(r#""width": 3"#, r#""width": 3, "period": 5"#);
        let error = read(&short_period).unwrap_err().to_string();
        assert!(
            error.starts_with("pulse period 5.0 is shorter than one pulse, 6.0"),
            "{error}"
        );
        assert!(read(r#"{"piecewise": {"points": [[1, 0], [1, 2]]}}"#).is_err());
        assert!(read(r#"{"piecewise": {"points": []}}"#).is_err());
        assert!(read(r#"{"square": 1}"#).is_err());
    }
}
