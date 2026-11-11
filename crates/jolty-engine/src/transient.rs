//! The transient analysis: the circuit's response over time.

use std::collections::VecDeque;

use jolty_models::Quantity;
use jolty_numerics::integration::{Formula, Method, divided_difference};
use serde::Deserialize;

use crate::newton::Mode;
use crate::{Options, SimulationError, Simulator, Strategy};

/// The first step after the start or a breakpoint, as a fraction of the shorter of the step
/// before and the gap to the next breakpoint, so a short waveform segment takes several steps.
const RESTART_FRACTION: f64 = 0.1;
/// The smallest step allowed, as a fraction of the longest.
const SMALLEST_STEP_FRACTION: f64 = 1e-9;
/// The factor a step shrinks by when Newton iteration fails.
const FAILURE_SHRINK: f64 = 0.125;
/// The bounds on the factor the truncation error changes the step by.
const FACTOR_BOUNDS: (f64, f64) = (0.1, 2.0);
/// The safety factor on the step the truncation error predicts.
const SAFETY: f64 = 0.9;
/// The most accepted points the error estimate reads: enough for second-order formulas.
const HISTORY: usize = 3;

/// A transient analysis from time zero, starting at the operating point.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Transient {
    /// The time to simulate until, in seconds.
    pub stop: f64,
    /// The longest step allowed, in seconds; by default a fiftieth of `stop`.
    #[serde(default)]
    pub max_step: Option<f64>,
    /// The integration method between breakpoints; trapezoidal by default.
    #[serde(default)]
    pub method: Method,
}

/// The waveforms of a transient analysis, at every accepted time point.
#[derive(Debug, Clone, PartialEq)]
pub struct Waveforms {
    /// The accepted times, in seconds, from zero to the stop time.
    pub times: Vec<f64>,
    /// The value of every unknown at each time.
    pub values: Vec<Vec<f64>>,
    /// How the analysis went.
    pub statistics: Statistics,
}

/// How a transient analysis went.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Statistics {
    /// How the initial operating point was found.
    pub initial: Strategy,
    /// The steps accepted.
    pub accepted: usize,
    /// The steps rejected, for failing to converge or for too large an error.
    pub rejected: usize,
    /// The Newton iterations of the accepted steps.
    pub iterations: usize,
}

/// The charges at the points accepted since the last breakpoint, and their derivative at the
/// last one.
struct History {
    /// The points before the last, oldest first.
    earlier: VecDeque<(f64, Vec<f64>)>,
    /// The last accepted point.
    last: (f64, Vec<f64>),
    /// The charges' derivative at the last accepted point.
    derivative: Vec<f64>,
}

impl History {
    /// Starts afresh from the point at `time`, forgetting the points before a breakpoint.
    fn restart(&mut self, time: f64, charge: Vec<f64>, derivative: Vec<f64>) {
        self.earlier.clear();
        self.last = (time, charge);
        self.derivative = derivative;
    }

    /// Appends the point at `time`, keeping the last [`HISTORY`] points.
    fn accept(&mut self, time: f64, charge: Vec<f64>, derivative: Vec<f64>) {
        let previous = std::mem::replace(&mut self.last, (time, charge));
        if self.earlier.len() == HISTORY - 1 {
            self.earlier.pop_front();
        }
        self.earlier.push_back(previous);
        self.derivative = derivative;
    }

    /// The number of points kept.
    fn len(&self) -> usize {
        self.earlier.len() + 1
    }

    /// The points kept, oldest first.
    fn points(&self) -> impl Iterator<Item = &(f64, Vec<f64>)> {
        self.earlier.iter().chain(std::iter::once(&self.last))
    }

    /// The charges at the last point and at the one before it, or the last one twice after a
    /// restart.
    fn last_two(&self) -> (&[f64], &[f64]) {
        let last = self.last.1.as_slice();
        (
            last,
            self.earlier
                .back()
                .map_or(last, |(_, charge)| charge.as_slice()),
        )
    }
}

/// The state of a transient analysis between steps.
struct Stepper {
    /// The method between breakpoints.
    method: Method,
    /// The longest step allowed.
    longest: f64,
    /// The shortest step allowed.
    shortest: f64,
    /// The time of the last accepted point.
    time: f64,
    /// The solution at the last accepted point.
    x: Vec<f64>,
    history: History,
    /// The length of the next step to try.
    next: f64,
    /// The length of the last accepted step.
    previous: f64,
    /// Whether the next step follows a breakpoint, and so uses backward Euler.
    restart: bool,
    /// The history terms of the step being tried, one per unknown.
    terms: Vec<f64>,
    statistics: Statistics,
}

impl Stepper {
    /// Records a rejected step and shortens the next one to `next`.
    fn reject(&mut self, next: f64) -> Result<bool, SimulationError> {
        self.statistics.rejected += 1;
        self.next = next;
        if next < self.shortest {
            return Err(SimulationError::StepTooSmall {
                time: self.time,
                minimum: self.shortest,
            });
        }
        Ok(false)
    }
}

impl Simulator {
    /// Simulates from the operating point at time zero until `transient.stop`.
    ///
    /// Each step is solved by Newton iteration and accepted when its estimated truncation error
    /// is within tolerance, and the next step grows or shrinks to keep it so. Steps land on
    /// every breakpoint of the sources, and restart with backward Euler after each, as a
    /// corner in a waveform would make the trapezoidal formula ring.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::InvalidAnalysis`] for a stop time or longest step that is
    /// not positive, [`SimulationError::StepTooSmall`] when no step short enough converges,
    /// and the errors of [`Simulator::operating_point`].
    pub fn transient(
        &mut self,
        transient: &Transient,
        options: &Options,
    ) -> Result<Waveforms, SimulationError> {
        options.check()?;
        let longest = transient.max_step.unwrap_or(transient.stop / 50.0);
        if transient.stop <= 0.0 || longest <= 0.0 {
            return Err(SimulationError::InvalidAnalysis(format!(
                "the stop time and longest step must be positive, not {:?} and {longest:?}",
                transient.stop
            )));
        }
        let shortest = longest * SMALLEST_STEP_FRACTION;
        let initial = self.operating_point(options)?;
        self.evaluate(&initial.values, 0.0, 1.0, options.gmin);
        let unknowns = initial.values.len();
        let breakpoints = self.breakpoints(transient.stop, shortest);
        let first = RESTART_FRACTION * longest.min(breakpoints[0]);
        let mut stepper = Stepper {
            method: transient.method,
            longest,
            shortest,
            time: 0.0,
            history: History {
                earlier: VecDeque::with_capacity(HISTORY - 1),
                last: (0.0, self.workspace.charge.clone()),
                derivative: vec![0.0; unknowns],
            },
            next: first,
            previous: first,
            restart: true,
            terms: vec![0.0; unknowns],
            statistics: Statistics {
                initial: initial.strategy,
                accepted: 0,
                rejected: 0,
                iterations: 0,
            },
            x: initial.values,
        };
        let (mut times, mut values) = (vec![0.0], vec![stepper.x.clone()]);
        for (index, &breakpoint) in breakpoints.iter().enumerate() {
            let gap = breakpoints
                .get(index + 1)
                .map_or(f64::INFINITY, |next| next - breakpoint);
            while stepper.time < breakpoint {
                if self.step_toward(&mut stepper, breakpoint, gap, options)? {
                    times.push(stepper.time);
                    values.push(stepper.x.clone());
                }
            }
        }
        Ok(Waveforms {
            times,
            values,
            statistics: stepper.statistics,
        })
    }

    /// Tries one step toward `breakpoint`, landing on it when it is within reach; `gap` is the
    /// time from it to the breakpoint after. Returns whether the step was accepted.
    fn step_toward(
        &mut self,
        stepper: &mut Stepper,
        breakpoint: f64,
        gap: f64,
        options: &Options,
    ) -> Result<bool, SimulationError> {
        let remaining = breakpoint - stepper.time;
        let landing = stepper.next >= remaining - stepper.shortest;
        let length = if landing {
            remaining
        } else {
            stepper.next.min(stepper.longest)
        };
        let method = if stepper.restart {
            Method::BackwardEuler
        } else {
            stepper.method
        };
        let formula = Formula::new(method, length, stepper.previous);
        let (last, before) = stepper.history.last_two();
        for (index, term) in stepper.terms.iter_mut().enumerate() {
            *term = formula.history(
                [last[index], before[index]],
                stepper.history.derivative[index],
            );
        }

        let mut candidate = stepper.x.clone();
        let mode = Mode::Step {
            time: stepper.time + length,
            alpha: formula.alpha,
            history: &stepper.terms,
        };
        let Ok(iterations) =
            self.newton(&mut candidate, mode, options, options.transient_iterations)
        else {
            return stepper.reject(length * FAILURE_SHRINK);
        };
        let charge = self.workspace.charge.clone();
        let derivative: Vec<f64> = charge
            .iter()
            .zip(&stepper.terms)
            .map(|(&q, &term)| formula.alpha * q + term)
            .collect();
        let ratio = if stepper.restart {
            None
        } else {
            let new = (stepper.time + length, charge.as_slice());
            self.error_ratio(method, length, &stepper.history, new, &derivative, options)
        };
        let factor = ratio.map_or(FACTOR_BOUNDS.1, |ratio| {
            (SAFETY * method.step_factor(ratio)).clamp(FACTOR_BOUNDS.0, FACTOR_BOUNDS.1)
        });
        if ratio.is_some_and(|ratio| ratio > 1.0) {
            return stepper.reject(length * factor);
        }

        stepper.statistics.accepted += 1;
        stepper.statistics.iterations += iterations;
        stepper.x = candidate;
        stepper.previous = length;
        if landing {
            stepper.time = breakpoint;
            stepper.history.restart(breakpoint, charge, derivative);
            stepper.restart = true;
            stepper.next = RESTART_FRACTION * length.min(gap);
        } else {
            stepper.time += length;
            stepper.history.accept(stepper.time, charge, derivative);
            stepper.restart = false;
            stepper.next = length * factor;
        }
        Ok(true)
    }

    /// The breakpoints of every device in `(0, stop]`, sorted, merged when closer than
    /// `minimum`, and ending at `stop`, so never empty.
    fn breakpoints(&self, stop: f64, minimum: f64) -> Vec<f64> {
        let mut times = vec![stop];
        for placed in &self.system.devices {
            placed.device.breakpoints(0.0, stop, &mut times);
        }
        times.retain(|&time| time > minimum);
        times.sort_by(f64::total_cmp);
        times.dedup_by(|later, earlier| *later - *earlier < minimum);
        if let Some(last) = times.last_mut() {
            *last = stop;
        }
        times
    }

    /// The largest ratio of a charge's estimated truncation error to its tolerance, for the
    /// step to `new`; `None` while too few points follow the last breakpoint to estimate it.
    ///
    /// A charge's tolerance is the larger of a relative tolerance on the charge itself and one
    /// on the current it carries over the step, scaled by the truncation tolerance.
    fn error_ratio(
        &self,
        method: Method,
        length: f64,
        history: &History,
        new: (f64, &[f64]),
        derivative: &[f64],
        options: &Options,
    ) -> Option<f64> {
        let needed = method.order() + 2;
        let available = history.len() + 1;
        if available < needed {
            return None;
        }
        let used: Vec<&(f64, Vec<f64>)> = history.points().skip(available - needed).collect();
        let mut times = [0.0; HISTORY + 1];
        for (slot, (time, _)) in times.iter_mut().zip(&used) {
            *slot = *time;
        }
        times[needed - 1] = new.0;
        let mut samples = [0.0; HISTORY + 1];
        let mut worst: f64 = 0.0;
        for (index, unknown) in self.system.unknowns.iter().enumerate() {
            for (slot, (_, charge)) in samples.iter_mut().zip(&used) {
                *slot = charge[index];
            }
            samples[needed - 1] = new.1[index];
            let error = method.truncation_error(
                length,
                divided_difference(&times[..needed], &samples[..needed]),
            );
            let previous_charge = history.last.1[index];
            let absolute_current = match unknown.quantity {
                Quantity::Voltage => options.current_tolerance,
                Quantity::Current => options.voltage_tolerance,
            };
            let charge_tolerance = options.relative_tolerance
                * new.1[index].abs().max(previous_charge.abs())
                + options.charge_tolerance;
            let current_tolerance = length
                * (options.relative_tolerance
                    * derivative[index].abs().max(history.derivative[index].abs())
                    + absolute_current);
            let tolerance = options.truncation_tolerance * charge_tolerance.max(current_tolerance);
            worst = worst.max(error / tolerance);
        }
        Some(worst)
    }
}
