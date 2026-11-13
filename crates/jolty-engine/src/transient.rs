//! The transient analysis: the circuit's response over time.

use std::collections::VecDeque;

use jolty_models::{Choice, Input, Level};
use jolty_numerics::integration::{Formula, Method, divided_difference};
use serde::Deserialize;

use crate::logic::{Command, Crossing, crossing};
use crate::newton::{Failure, Mode, NAMED_UNKNOWNS};
use crate::periodic::Monodromy;
use crate::sensitivity::Forward;
use crate::{Observation, Observations, Options, Rejection, SimulationError, Simulator, Strategy};

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
/// The earlier derivatives kept to tell the trapezoidal formula's ringing, besides the last.
const RINGING: usize = 2;

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
    /// Times, in seconds, the analysis steps onto exactly besides its own, without
    /// restarting its integration there; none by default.
    #[serde(default)]
    pub times: Vec<f64>,
    /// Whether to find the periodic steady state instead, `stop` the period of every
    /// source: one period from the state the circuit repeats, which shooting finds.
    #[serde(default)]
    pub periodic: bool,
}

/// The waveforms of a transient analysis, at every accepted time point.
#[derive(Debug, Clone, PartialEq)]
pub struct Waveforms {
    /// The accepted times, in seconds, from zero to the stop time.
    pub times: Vec<f64>,
    /// The value of every unknown at each time.
    pub values: Vec<Vec<f64>>,
    /// What every device did at each time: the current into each terminal and the charge it
    /// holds, its derivatives and its variables. A current includes its charge's
    /// derivative, by the step's integration formula.
    pub observations: Observations,
    /// Each logic net's changes: its level at time zero, then each time it changes, in
    /// seconds, and its new level.
    pub logic: Vec<Vec<(f64, Level)>>,
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
    /// For a periodic steady state, the periods shooting simulated to find it, the last
    /// the one returned.
    pub periods: Option<usize>,
}

/// What a transient carries along its accepted steps besides its solution.
pub(crate) enum Carried<'a> {
    /// Nothing.
    Nothing,
    /// Each parameter's derivatives, for a sensitivity analysis.
    Forward(&'a mut Forward),
    /// The derivatives with respect to the initial state, for a periodic steady state.
    Monodromy(&'a mut Monodromy),
}

/// The charges' derivatives with respect to one variable at the points a transient accepted
/// last, which its integration formula reads: what the direct method carries from step to
/// step for each variable it differentiates by.
pub(crate) struct Tangent {
    /// `dq/dp` at the last point.
    last: Vec<f64>,
    /// `dq/dp` at the point before, unless the integration restarted at the last.
    before: Option<Vec<f64>>,
    /// `d(dq/dt)/dp` at the last point.
    rate: Vec<f64>,
}

impl Tangent {
    /// Starts at a point where the charges' derivatives are `charge`, at rest.
    pub(crate) fn new(charge: Vec<f64>) -> Self {
        let rate = vec![0.0; charge.len()];
        Self {
            last: charge,
            before: None,
            rate,
        }
    }

    /// The derivative of the part of `formula`'s `dq/dt` that the earlier points
    /// contribute.
    pub(crate) fn history(&self, formula: &Formula) -> Vec<f64> {
        let before = self.before.as_ref().unwrap_or(&self.last);
        self.last
            .iter()
            .zip(before)
            .zip(&self.rate)
            .map(|((&last, &before), &rate)| formula.history([last, before], rate))
            .collect()
    }

    /// Moves on to the point `formula` stepped to, where the charges' derivatives are
    /// `charge` and `history` is [`Tangent::history`]'s; `restarted` says whether the
    /// integration restarts there.
    pub(crate) fn advance(
        &mut self,
        charge: Vec<f64>,
        history: &[f64],
        formula: &Formula,
        restarted: bool,
    ) {
        self.rate = charge
            .iter()
            .zip(history)
            .map(|(value, term)| formula.alpha * value + term)
            .collect();
        let last = std::mem::replace(&mut self.last, charge);
        self.before = (!restarted).then_some(last);
    }
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
    /// The derivatives at the points before the last, oldest first, while the trapezoidal
    /// formula took them all: the run that shows whether it rings.
    ringing: VecDeque<Vec<f64>>,
}

impl History {
    /// Starts afresh from the point at `time`, forgetting the points before a breakpoint.
    fn restart(&mut self, time: f64, charge: Vec<f64>, derivative: Vec<f64>) {
        self.earlier.clear();
        self.ringing.clear();
        self.last = (time, charge);
        self.derivative = derivative;
    }

    /// Appends the point at `time`, keeping the last [`HISTORY`] points; `trapezoidal` says
    /// whether the trapezoidal formula took its derivative, which a run of keeps.
    fn accept(&mut self, time: f64, charge: Vec<f64>, derivative: Vec<f64>, trapezoidal: bool) {
        let previous = std::mem::replace(&mut self.last, (time, charge));
        if self.earlier.len() == HISTORY - 1 {
            self.earlier.pop_front();
        }
        self.earlier.push_back(previous);
        let before = std::mem::replace(&mut self.derivative, derivative);
        if trapezoidal {
            if self.ringing.len() == RINGING {
                self.ringing.pop_front();
            }
            self.ringing.push_back(before);
        } else {
            self.ringing.clear();
        }
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

/// The times a transient analysis steps onto exactly, after the current time.
struct Breakpoints {
    /// The breakpoints, sorted, the last the stop time.
    times: Vec<f64>,
    /// How close two breakpoints may be before one is dropped.
    minimum: f64,
}

impl Breakpoints {
    /// Adds `time`, unless it lies past the stop or within the minimum of another.
    fn insert(&mut self, time: f64) {
        let stop = self.times.last().copied().unwrap_or(f64::INFINITY);
        if time >= stop {
            return;
        }
        let index = self.times.partition_point(|&other| other < time);
        let near =
            |other: Option<&f64>| other.is_some_and(|&other| (other - time).abs() < self.minimum);
        if !near(self.times.get(index))
            && !near(
                index
                    .checked_sub(1)
                    .and_then(|before| self.times.get(before)),
            )
        {
            self.times.insert(index, time);
        }
    }

    /// The first breakpoint after `time`, and the gap from it to the one after.
    fn after(&mut self, time: f64) -> (f64, f64) {
        let passed = self.times.partition_point(|&other| other <= time);
        let kept = passed.min(self.times.len() - 1);
        self.times.drain(..kept);
        let next = self.times[0];
        let gap = self
            .times
            .get(1)
            .map_or(f64::INFINITY, |after| after - next);
        (next, gap)
    }
}

/// What came of trying a step.
enum Step {
    /// The step was accepted, with what every device did at its end, and logic drivers
    /// began ramps that settle at these times.
    Accepted(Observation, Vec<f64>),
    /// The step was rejected, to be tried again shorter.
    Rejected,
    /// The step was rejected because a detector crosses, a logic driver changes or a
    /// device's contribution jumps within it, at this time, which the step must land on.
    Land(f64),
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
    /// The last iterate of a step whose Newton iteration ran out of iterations, which the
    /// shorter step after it starts from: an iteration converging slowly, as at a
    /// transistor turning off with nothing else to hold its source, goes on where it
    /// stopped instead of starting over.
    resume: Option<Vec<f64>>,
    history: History,
    /// The length of the next step to try.
    next: f64,
    /// The length of the last accepted step.
    previous: f64,
    /// Whether the next step follows a breakpoint, and so uses backward Euler.
    restart: bool,
    /// Whether the next step uses backward Euler, its error still controlled, to end the
    /// trapezoidal formula's ringing.
    damp: bool,
    /// The history terms of the step being tried, one per unknown.
    terms: Vec<f64>,
    /// The integration formula of the last accepted step.
    formula: Formula,
    statistics: Statistics,
}

impl Stepper {
    /// Records a rejected step, and shortens the next one to `next`.
    /// Whether an event at `time`, within a step from the last point to `end`, comes
    /// before `end` by more than the relative tolerance of the step, and after the last
    /// point by the shortest step at least, so the step should land on it instead: an
    /// event located by interpolating across a step is accurate only within a short one.
    fn lands_before(&self, time: f64, end: f64, options: &Options) -> bool {
        let tolerance = self
            .shortest
            .max(options.relative_tolerance * (end - self.time));
        time < end - tolerance && time - self.time >= self.shortest
    }

    fn shorten(&mut self, next: f64) {
        self.statistics.rejected += 1;
        self.next = next;
    }

    /// Records a step rejected for the reason `why` gives, and shortens the next one to
    /// `next`, failing when it falls below the shortest step allowed.
    fn reject(
        &mut self,
        next: f64,
        why: impl FnOnce() -> Rejection,
    ) -> Result<Step, SimulationError> {
        self.shorten(next);
        if next < self.shortest {
            return Err(SimulationError::StepTooSmall {
                time: self.time,
                minimum: self.shortest,
                rejection: why(),
            });
        }
        Ok(Step::Rejected)
    }
}

impl Simulator {
    /// Simulates from the operating point at time zero until `transient.stop`.
    ///
    /// Each step is solved by Newton iteration and accepted when its estimated truncation error
    /// is within tolerance, and the next step grows or shrinks to keep it so. Steps land on
    /// every breakpoint of the sources, and restart with backward Euler after each, as a
    /// corner in a waveform would make the trapezoidal formula ring. A periodic transient
    /// simulates instead one period of the periodic steady state, found by shooting from
    /// the operating point.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::InvalidAnalysis`] for a stop time or longest step that is
    /// not positive, or a periodic transient of a circuit with logic,
    /// [`SimulationError::StepTooSmall`] when no step short enough converges,
    /// [`SimulationError::NoSteadyState`] when shooting does not converge, and the errors of
    /// [`Simulator::operating_point`].
    pub fn transient(
        &mut self,
        transient: &Transient,
        options: &Options,
    ) -> Result<Waveforms, SimulationError> {
        self.solvable()?;
        if transient.periodic {
            return self.periodic_steady_state(transient, options);
        }
        self.transient_with(transient, options, None, Carried::Nothing)
    }

    /// The longest step `transient` allows, after checking its times.
    fn longest_step(&self, transient: &Transient) -> Result<f64, SimulationError> {
        // Without unknowns, nothing analog needs steps, and the logic moves on at once.
        let longest = if self.system.unknowns.is_empty() {
            transient.stop
        } else {
            transient.max_step.unwrap_or(transient.stop / 50.0)
        };
        if transient.stop <= 0.0 || longest <= 0.0 {
            return Err(SimulationError::InvalidAnalysis(format!(
                "the stop time and longest step must be positive, not {:?} and {longest:?}",
                transient.stop
            )));
        }
        if let Some(time) = transient
            .times
            .iter()
            .find(|&&time| !(time > 0.0 && time <= transient.stop))
        {
            return Err(SimulationError::InvalidAnalysis(format!(
                "the times to step onto lie after 0 and at most the stop time, not {time:?}"
            )));
        }
        Ok(longest)
    }

    /// Starts carrying `carried` from the state `x` at time zero.
    fn start_carried(
        &mut self,
        carried: &mut Carried<'_>,
        x: &[f64],
        options: &Options,
    ) -> Result<(), SimulationError> {
        match carried {
            Carried::Nothing => {}
            Carried::Forward(forward) => self.start_forward(forward, x, options)?,
            Carried::Monodromy(monodromy) => self.start_monodromy(monodromy, x, options),
        }
        Ok(())
    }

    /// Carries `carried` over the step accepted at `time`, its solution `x`, by `formula`;
    /// `restarted` says whether the integration restarts after it.
    fn advance_carried(
        &mut self,
        carried: &mut Carried<'_>,
        (time, x): (f64, &[f64]),
        (formula, restarted): (&Formula, bool),
        options: &Options,
    ) -> Result<(), SimulationError> {
        let (at, formula) = ((time, x), (formula, restarted));
        match carried {
            Carried::Nothing => Ok(()),
            Carried::Forward(forward) => self.advance_forward(forward, at, formula, options),
            Carried::Monodromy(monodromy) => {
                self.advance_monodromy(monodromy, at, formula, options)
            }
        }
    }

    /// Runs `transient` from `start`, a state of a circuit without logic and how the
    /// operating point it came from was found, or else from the operating point, carrying
    /// `carried` along its accepted steps.
    pub(crate) fn transient_with(
        &mut self,
        transient: &Transient,
        options: &Options,
        start: Option<(Vec<f64>, Strategy)>,
        mut carried: Carried<'_>,
    ) -> Result<Waveforms, SimulationError> {
        options.check()?;
        let longest = self.longest_step(transient)?;
        let shortest = longest * SMALLEST_STEP_FRACTION;
        let mut observations = transient.times.clone();
        observations.sort_by(f64::total_cmp);
        let (initial, strategy) = if let Some(start) = start {
            start
        } else {
            let point = self.operating_point(options)?;
            (point.values, point.strategy)
        };
        self.start_carried(&mut carried, &initial, options)?;
        self.evaluate(&initial, Some(0.0), (1.0, options.gmin), true);
        self.events.start(&self.system.logic, transient.stop);
        let unknowns = initial.len();
        let mut breakpoints = Breakpoints {
            times: self.breakpoints(transient.stop, shortest),
            minimum: shortest,
        };
        let first = RESTART_FRACTION * longest.min(breakpoints.times[0]);
        let mut stepper = Stepper {
            method: transient.method,
            longest,
            shortest,
            time: 0.0,
            resume: None,
            history: History {
                earlier: VecDeque::with_capacity(HISTORY - 1),
                last: (0.0, self.workspace.contributions.charge.clone()),
                derivative: vec![0.0; unknowns],
                ringing: VecDeque::with_capacity(RINGING),
            },
            next: first,
            previous: first,
            restart: true,
            damp: false,
            terms: vec![0.0; unknowns],
            formula: Formula::new(Method::BackwardEuler, first, first),
            statistics: Statistics {
                initial: strategy,
                accepted: 0,
                rejected: 0,
                iterations: 0,
                periods: None,
            },
            x: initial,
        };
        let (mut times, mut values) = (vec![0.0], vec![stepper.x.clone()]);
        let mut observed = Observations::default();
        observed.push(&self.observer.start(&stepper.x, options.gmin));
        while stepper.time < transient.stop {
            if let Some(change) = self.events.next_analog_change(&self.system.logic) {
                breakpoints.insert(change);
            }
            let (breakpoint, gap) = breakpoints.after(stepper.time);
            let passed = observations.partition_point(|&time| time < stepper.time + shortest);
            let observation = observations.get(passed).copied().unwrap_or(f64::INFINITY);
            match self.step_toward(&mut stepper, (breakpoint, gap), observation, options)? {
                Step::Accepted(found, settles) => {
                    let at = (stepper.time, stepper.x.as_slice());
                    let formula = (&stepper.formula, stepper.restart);
                    self.advance_carried(&mut carried, at, formula, options)?;
                    self.watch_supplies(&stepper.x, stepper.time, options);
                    times.push(stepper.time);
                    values.push(stepper.x.clone());
                    observed.push(&found);
                    for time in settles {
                        breakpoints.insert(time);
                    }
                }
                Step::Rejected => {}
                Step::Land(time) => breakpoints.insert(time),
            }
        }
        Ok(Waveforms {
            times,
            values,
            observations: observed,
            logic: self.events.traces().to_vec(),
            statistics: stepper.statistics,
        })
    }

    /// Tries one step toward `breakpoint`, landing on it when it is within reach; `gap` is the
    /// time from it to the breakpoint after. A step lands on `observation` too, when it
    /// comes first, without restarting the integration there.
    ///
    /// An accepted step moves the logic on to its end, with the detectors' crossings within
    /// it. A step in which a detector crosses, a logic driver changes or a device's
    /// contribution jumps, before its end, is rejected to land there; the integration
    /// restarts after a step in which a device's contribution jumps.
    fn step_toward(
        &mut self,
        stepper: &mut Stepper,
        (breakpoint, gap): (f64, f64),
        observation: f64,
        options: &Options,
    ) -> Result<Step, SimulationError> {
        let target = breakpoint.min(observation);
        let remaining = target - stepper.time;
        let landing = stepper.next >= remaining - stepper.shortest;
        let length = if landing {
            remaining
        } else {
            stepper.next.min(stepper.longest)
        };
        let end = if landing {
            target
        } else {
            stepper.time + length
        };
        // Only a breakpoint is a corner in the waveforms, after which the integration
        // restarts; an observation is a time like any other.
        let corner = landing && target >= breakpoint;
        let method = if stepper.restart || stepper.damp {
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

        let mut candidate = stepper.resume.take().unwrap_or_else(|| stepper.x.clone());
        let mode = Mode::Step {
            time: end,
            alpha: formula.alpha,
            history: &stepper.terms,
        };
        let solved = self.newton(&mut candidate, mode, options, options.transient_iterations);
        let Ok(iterations) = solved else {
            if solved == Err(Failure::Diverged) && candidate.iter().all(|value| value.is_finite()) {
                stepper.resume = Some(candidate.clone());
            }
            return stepper.reject(length * FAILURE_SHRINK, || Rejection::Unconverged {
                worst: self.furthest_from_converging(&candidate, options, NAMED_UNKNOWNS),
            });
        };
        let charge = self.workspace.contributions.charge.clone();
        let derivative: Vec<f64> = charge
            .iter()
            .zip(&stepper.terms)
            .map(|(&q, &term)| formula.alpha * q + term)
            .collect();
        let ratio = if stepper.restart {
            None
        } else {
            let new = (end, charge.as_slice());
            self.error_ratio(method, length, &stepper.history, new, &derivative, options)
        };
        let factor = ratio.map_or(FACTOR_BOUNDS.1, |(ratio, _)| {
            (SAFETY * method.step_factor(ratio)).clamp(FACTOR_BOUNDS.0, FACTOR_BOUNDS.1)
        });
        if let Some((ratio, worst)) = ratio
            && ratio > 1.0
        {
            let name = &self.system.unknowns[worst].name;
            return stepper.reject(length * factor, || Rejection::Inaccurate(name.clone()));
        }
        let (commands, jumped) = match self.events_within(stepper, (end, &candidate), options) {
            Ok(found) => found,
            Err(early) => {
                // An event lands no closer than the shortest step allowed.
                stepper.shorten(early - stepper.time);
                return Ok(Step::Land(early));
            }
        };

        stepper.statistics.accepted += 1;
        stepper.statistics.iterations += iterations;
        stepper.formula = formula;
        stepper.x = candidate;
        stepper.previous = length;
        let settles = self.follow(&commands);
        let restart = corner || !commands.is_empty() || jumped;
        let found = self
            .observer
            .step(&stepper.x, end, options.gmin, (&formula, restart));
        if restart {
            // A breakpoint, a driver's ramp starting or a device's jump is a corner in the
            // waveforms.
            let gap = if corner { gap } else { breakpoint - end };
            stepper.time = end;
            stepper.history.restart(end, charge, derivative);
            stepper.restart = true;
            stepper.damp = false;
            stepper.next = RESTART_FRACTION * length.min(gap);
        } else {
            let trapezoidal = method == Method::Trapezoidal;
            stepper.damp = trapezoidal && self.rings(&stepper.history, &derivative, options);
            stepper.time = end;
            stepper.history.accept(end, charge, derivative, trapezoidal);
            stepper.restart = false;
            stepper.next = length * factor;
        }
        Ok(Step::Accepted(found, settles))
    }

    /// Whether the trapezoidal formula rings: whether some charge's derivative, over the last
    /// four points it took, turned back and forth twice by more than its tolerance.
    ///
    /// The formula carries an error in a derivative from step to step undamped, flipping its
    /// sign each time; a smooth waveform turns at most once in so few steps. A derivative no
    /// equation reads, such as a Meyer displacement while its capacitance is zero, would
    /// gather that error without bound. One step of backward Euler, needing no derivative,
    /// starts it afresh.
    fn rings(&self, history: &History, newest: &[f64], options: &Options) -> bool {
        if history.ringing.len() < RINGING {
            return false;
        }
        let (oldest, older) = (&history.ringing[0], &history.ringing[1]);
        self.system
            .unknowns
            .iter()
            .enumerate()
            .any(|(index, unknown)| {
                let run = [
                    oldest[index],
                    older[index],
                    history.derivative[index],
                    newest[index],
                ];
                let turns = [run[1] - run[0], run[2] - run[1], run[3] - run[2]];
                let absolute = options.equation_tolerance(unknown.quantity);
                let largest = run
                    .iter()
                    .fold(0.0_f64, |most, value| most.max(value.abs()));
                let tolerance = options.truncation_tolerance
                    * (options.relative_tolerance * largest + absolute);
                turns[0] * turns[1] < 0.0
                    && turns[1] * turns[2] < 0.0
                    && turns.iter().all(|turn| turn.abs() > tolerance)
            })
    }

    /// The events of a step from the stepper's last point to `end`, where the solution is
    /// `after`: the changes the logic drivers must follow, and whether a device's
    /// contribution jumps. An event early enough that the step should land on it returns
    /// its time instead, leaving the logic where it was.
    fn events_within(
        &mut self,
        stepper: &Stepper,
        (end, after): (f64, &[f64]),
        options: &Options,
    ) -> Result<(Vec<Command>, bool), f64> {
        let jump = self.first_jump(stepper, (end, after));
        if let Some(time) = jump
            && stepper.lands_before(time, end, options)
        {
            return Err(time);
        }
        let commands = self.advance_logic(stepper, (end, after), options)?;
        Ok((commands, jump.is_some()))
    }

    /// The first time between the stepper's last point and `end`, where the solution is
    /// `after`, at which a device's contribution jumps, if one does: where the earliest of
    /// its choices that changed between the two points changes, located by linear
    /// interpolation.
    fn first_jump(&mut self, stepper: &Stepper, (end, after): (f64, &[f64])) -> Option<f64> {
        let (mut before, mut later) = (Vec::new(), Vec::new());
        let start = stepper.time;
        let mut first: Option<f64> = None;
        for placed in &mut self.system.devices {
            before.clear();
            placed
                .device
                .choices(&stepping(&stepper.x, start), &mut before);
            if before.is_empty() {
                continue;
            }
            later.clear();
            placed.device.choices(&stepping(after, end), &mut later);
            if let Some(fraction) = Choice::first_jump(&before, &later) {
                let time = start + (end - start) * fraction;
                first = Some(first.map_or(time, |earlier| earlier.min(time)));
            }
        }
        first
    }

    /// Moves the logic from the stepper's last point to `end`, where the solution is
    /// `after`, and returns the changes the logic drivers must follow.
    ///
    /// A detector's crossing earlier than the end, so that the step should land on it,
    /// returns its time instead; so does a change of a logic driver that early, leaving
    /// the logic where it was.
    fn advance_logic(
        &mut self,
        stepper: &Stepper,
        (end, after): (f64, &[f64]),
        options: &Options,
    ) -> Result<Vec<Command>, f64> {
        let Self { system, events, .. } = self;
        let logic = &system.logic;
        let early = |time: f64| stepper.lands_before(time, end, options);
        let held = events.detector_levels(logic);
        let crossings: Vec<Crossing> = logic
            .detectors
            .iter()
            .enumerate()
            .filter_map(|(detector, &(element, _))| {
                crossing(
                    &system.devices[element],
                    detector,
                    held[detector],
                    (stepper.time, &stepper.x),
                    (end, after),
                )
            })
            .collect();
        let first = crossings
            .iter()
            .map(|crossing| crossing.time)
            .filter(|&time| early(time))
            .min_by(f64::total_cmp);
        if let Some(first) = first {
            return Err(first);
        }
        if crossings.is_empty() && !events.due(end) {
            return Ok(Vec::new());
        }
        // Without drivers, no change reaches back into the analog circuit, so the logic can
        // move on in place.
        let mut tentative = (!logic.drivers.is_empty()).then(|| events.clone());
        let state = tentative.as_mut().unwrap_or(events);
        state.inject(logic, &crossings);
        let commands = state.advance(logic, end);
        let first = commands
            .iter()
            .map(|command| command.time)
            .filter(|&time| early(time))
            .min_by(f64::total_cmp);
        if let Some(first) = first {
            return Err(first);
        }
        if let Some(tentative) = tentative {
            *events = tentative;
        }
        Ok(commands)
    }

    /// Starts each logic driver of `commands` on its ramp, and returns when each ramp ends.
    fn follow(&mut self, commands: &[Command]) -> Vec<f64> {
        let mut settles = Vec::with_capacity(commands.len());
        for command in commands {
            let element = self.system.logic.drivers[command.driver].0;
            let ramp = (command.time, command.level, command.transition);
            settles.extend(self.drive(element, ramp));
        }
        settles
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
    /// step to `new`, and the index of that charge's equation; `None` while too few points
    /// follow the last breakpoint to estimate it.
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
    ) -> Option<(f64, usize)> {
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
        let mut worst = (0.0, 0);
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
            let absolute_current = options.equation_tolerance(unknown.quantity);
            let charge_tolerance = options.relative_tolerance
                * new.1[index].abs().max(previous_charge.abs())
                + options.charge_tolerance;
            let current_tolerance = length
                * (options.relative_tolerance
                    * derivative[index].abs().max(history.derivative[index].abs())
                    + absolute_current);
            let tolerance = options.truncation_tolerance * charge_tolerance.max(current_tolerance);
            if error / tolerance > worst.0 {
                worst = (error / tolerance, index);
            }
        }
        Some(worst)
    }
}

/// The input of the state `x` at `time` of a transient analysis, with no gmin.
fn stepping(x: &[f64], time: f64) -> Input<'_> {
    let mut input = Input::new(x, time, 0.0);
    input.transient = true;
    input
}
