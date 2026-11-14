//! The transient analysis: the circuit's response over time.

use std::collections::{BTreeSet, VecDeque};

use jolty_models::{Input, Level};
use jolty_numerics::integration::{Formula, Method, divided_difference};
use serde::Deserialize;

use crate::logic::{Checkpoint, Command, Crossing, crossing};
use crate::newton::{Failure, Mode, NAMED_UNKNOWNS};
use crate::observe::{Observer, Read};
use crate::options::NUMBERS_PER_GIB;
use crate::periodic::Monodromy;
use crate::profile::Activity;
use crate::sensitivity::Forward;
use crate::{
    Observation, Observations, Options, Profile, Rejection, SimulationError, Simulator, Strategy,
    Work,
};

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
    /// What the devices its options record did at each time: the current into each
    /// recorded terminal and the charge it holds, and the recorded devices' derivatives and
    /// variables. A current includes its charge's derivative, by the step's integration
    /// formula.
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
    /// The steps tried again shorter to land on an event within them: a detector's
    /// crossing, a logic driver's change or a device's jump.
    pub landed: usize,
    /// The Newton iterations of the accepted steps.
    pub iterations: usize,
    /// For a periodic steady state, the periods shooting simulated to find it, the last
    /// the one returned.
    pub periods: Option<usize>,
    /// The work the analysis did, its operating point's included, and for a periodic
    /// steady state every period's. Unlike the results, it depends on what the simulator
    /// did before: on an operating point it remembers, and on the pivots its sparse
    /// solver kept.
    pub work: Work,
    /// Where its time went, when its options' `profile` is on.
    pub profile: Option<Profile>,
}

impl Statistics {
    /// The statistics of a transient before its first step, from an operating point found
    /// by `initial`.
    fn starting(initial: Strategy) -> Self {
        Self {
            initial,
            accepted: 0,
            rejected: 0,
            landed: 0,
            iterations: 0,
            periods: None,
            work: Work::default(),
            profile: None,
        }
    }
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
    /// The breakpoints, the last the stop time, each by the bits of its time: times
    /// are zero or more, whose bits order as the times do, so the set finds the next
    /// breakpoint, and the neighbors of a new one, in logarithmic time.
    times: BTreeSet<u64>,
    /// How close two breakpoints may be before one is dropped.
    minimum: f64,
}

impl Breakpoints {
    /// The breakpoints at `times`, sorted, the last the stop time.
    fn new(times: &[f64], minimum: f64) -> Self {
        Self {
            times: times.iter().map(|&time| Self::key(time)).collect(),
            minimum,
        }
    }

    /// The key of `time`, zero or more: the bits of positive zero for a negative zero.
    fn key(time: f64) -> u64 {
        (time + 0.0).to_bits()
    }

    /// The earliest breakpoint.
    fn first(&self) -> f64 {
        self.times
            .first()
            .map_or(f64::INFINITY, |&key| f64::from_bits(key))
    }

    /// Adds `time`, unless it lies past the stop or within the minimum of another.
    fn insert(&mut self, time: f64) {
        let stop = self
            .times
            .last()
            .map_or(f64::INFINITY, |&key| f64::from_bits(key));
        if time >= stop {
            return;
        }
        let key = Self::key(time);
        let near = |other: Option<&u64>| {
            other.is_some_and(|&other| (f64::from_bits(other) - time).abs() < self.minimum)
        };
        if !near(self.times.range(..key).next_back()) && !near(self.times.range(key..).next()) {
            self.times.insert(key);
        }
    }

    /// The first breakpoint after `time`, and the gap from it to the one after.
    fn after(&mut self, time: f64) -> (f64, f64) {
        let passed = Self::key(time);
        while self.times.len() > 1 && self.times.first().is_some_and(|&key| key <= passed) {
            self.times.pop_first();
        }
        let mut ahead = self.times.iter().map(|&key| f64::from_bits(key));
        let next = ahead.next().unwrap_or(f64::INFINITY);
        let gap = ahead.next().map_or(f64::INFINITY, |after| after - next);
        (next, gap)
    }
}

/// What came of trying a step.
enum Step {
    /// The step was accepted, with the first step after the last corner before it if that
    /// was pending, oldest first; logic drivers began ramps that settle at these times.
    Accepted(Vec<Point>, Vec<f64>),
    /// The step is the first after a corner, solved but accepted only once the next has
    /// estimated its error.
    Pending,
    /// The step was rejected, to be tried again shorter.
    Rejected,
    /// A detector crosses, a logic driver changes or a device's contribution jumps within
    /// the step, at this time, which the step is tried again to land on.
    Land(f64),
}

/// A point a transient accepted.
struct Point {
    time: f64,
    /// The solution there.
    x: Vec<f64>,
    /// The formula of the step that reached it.
    formula: Formula,
    /// The length of that step.
    length: f64,
    /// Whether the integration restarts after it.
    restarted: bool,
    /// What every recorded device did there.
    found: Observation,
}

/// The first step after a corner, solved and through the logic but not yet accepted.
///
/// Its error needs the curvature of the charges after the corner, which takes a third point
/// to estimate: the end of the step after it. That step accepts it along with itself, or,
/// when the estimate shows its error too large, goes back to the corner to take it shorter.
struct Pending {
    /// The step's formula.
    formula: Formula,
    /// The step's length.
    length: f64,
    /// The Newton iterations that solved it.
    iterations: usize,
    /// What the recorded devices did at its end.
    read: Read,
    /// The corner it started from.
    corner: Corner,
}

/// The state at a corner that a pending first step goes back to.
struct Corner {
    time: f64,
    /// The solution, the charges and their derivatives there.
    x: Vec<f64>,
    charge: Vec<f64>,
    derivative: Vec<f64>,
    /// The length of the step that reached it.
    previous: f64,
    /// The logic there, before the pending step moved it on.
    logic: Checkpoint,
}

/// A step judged by its truncation error: its length, and the charges and their
/// derivatives at its end and at its start.
struct Span<'a> {
    length: f64,
    /// The coefficient of the new charge in the step's formula.
    alpha: f64,
    charge: (&'a [f64], &'a [f64]),
    derivative: (&'a [f64], &'a [f64]),
}

/// A step solved by Newton iteration, on its way to being taken.
struct Tried {
    end: f64,
    length: f64,
    method: Method,
    formula: Formula,
    /// The Newton iterations that solved it.
    iterations: usize,
    /// The solution at its end, the charges there and their derivatives.
    x: Vec<f64>,
    charge: Vec<f64>,
    derivative: Vec<f64>,
    /// What the recorded devices did there, as the evaluation that solved it left them.
    read: Read,
}

/// How a step's estimated truncation error compares with its tolerance.
#[derive(Clone, Copy)]
struct Judgement {
    /// The largest ratio of a charge's error to its tolerance.
    ratio: f64,
    /// The index of that charge's equation.
    worst: usize,
    /// The longest step every charge's error, by the same estimate, allows.
    longest: f64,
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
    /// The accepted point before the last, its time and solution, through which each
    /// step's Newton iteration starts from a prediction; none after a corner in the
    /// waveforms, a breakpoint, a driver's ramp starting or a device's jump, where the last
    /// point alone is the start.
    before: Option<(f64, Vec<f64>)>,
    history: History,
    /// The length of the next step to try: the longest the error allows, which a step
    /// landing on a breakpoint or an event cuts short without changing.
    next: f64,
    /// The length of the last step taken.
    previous: f64,
    /// Whether the next step uses backward Euler, its error still controlled, to end the
    /// trapezoidal formula's ringing.
    damp: bool,
    /// The history terms of the step being tried, one per unknown.
    terms: Vec<f64>,
    /// The first step after the last corner, while it waits for the next to estimate its
    /// error; the stepper stands at its end.
    pending: Option<Pending>,
    statistics: Statistics,
}

impl Stepper {
    /// Where Newton iteration starts a step ending at `end`, as SPICE's does: the line
    /// through the last two accepted points carried on to `end`, or the last point after a
    /// corner.
    fn predicted(&self, end: f64) -> Vec<f64> {
        let Some((time, before)) = &self.before else {
            return self.x.clone();
        };
        let ratio = (end - self.time) / (self.time - time);
        self.x
            .iter()
            .zip(before)
            .map(|(&last, &earlier)| last + (last - earlier) * ratio)
            .collect()
    }

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

    /// Records a step rejected for the reason `why` gives, and shortens the next one to
    /// `next`, failing when it falls below the shortest step allowed.
    fn reject(
        &mut self,
        next: f64,
        why: impl FnOnce() -> Rejection,
    ) -> Result<Step, SimulationError> {
        self.statistics.rejected += 1;
        self.next = next;
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
        let longest = if self.elaborated.unknowns.is_empty() {
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

    /// The numbers each point of a transient's results holds: the unknowns, and the
    /// recorded terminals' currents and charges and the recorded devices' derivatives and
    /// variables.
    pub(crate) fn point_width(&self) -> f64 {
        let width = self.elaborated.unknowns.len() + self.state.observer.width();
        f64::from(u32::try_from(width).unwrap_or(u32::MAX))
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
        let reading = self.read_meter(options.profile);
        self.prepare(options)?;
        let longest = self.longest_step(transient)?;
        let shortest = longest * SMALLEST_STEP_FRACTION;
        let width = self.point_width();
        let most_held = options.memory * NUMBERS_PER_GIB;
        if transient.stop / longest * width > most_held {
            return Err(SimulationError::TooManyPoints {
                time: 0.0,
                step: longest,
                memory: options.memory,
            });
        }
        let mut held = width;
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
        let start = self.read_recorded();
        self.state
            .events
            .start(&self.elaborated.logic, transient.stop);
        let unknowns = initial.len();
        let mut breakpoints =
            Breakpoints::new(&self.breakpoints(transient.stop, shortest), shortest);
        let first = RESTART_FRACTION * longest.min(breakpoints.first());
        let mut stepper = Stepper {
            method: transient.method,
            longest,
            shortest,
            time: 0.0,
            resume: None,
            before: None,
            history: History {
                earlier: VecDeque::with_capacity(HISTORY - 1),
                last: (0.0, self.state.workspace.contributions.charge.clone()),
                derivative: vec![0.0; unknowns],
                ringing: VecDeque::with_capacity(RINGING),
            },
            next: first,
            previous: first,
            damp: false,
            terms: vec![0.0; unknowns],
            pending: None,
            statistics: Statistics::starting(strategy),
            x: initial,
        };
        let (mut times, mut values) = (vec![0.0], vec![stepper.x.clone()]);
        let mut observed = Observations::default();
        observed.push(&self.record(|observer| observer.start(&start)));
        while stepper.time < transient.stop {
            if let Some(change) = self.state.events.next_analog_change(&self.elaborated.logic) {
                breakpoints.insert(change);
            }
            let (breakpoint, gap) = breakpoints.after(stepper.time);
            let passed = observations.partition_point(|&time| time < stepper.time + shortest);
            let observation = observations.get(passed).copied().unwrap_or(f64::INFINITY);
            match self.step_toward(&mut stepper, (breakpoint, gap), observation, options)? {
                Step::Accepted(points, settles) => {
                    for point in points {
                        let at = (point.time, point.x.as_slice());
                        let formula = (&point.formula, point.restarted);
                        self.advance_carried(&mut carried, at, formula, options)?;
                        self.watch_supplies(&point.x, point.time, options);
                        held += width;
                        if held > most_held {
                            return Err(SimulationError::TooManyPoints {
                                time: point.time,
                                step: point.length,
                                memory: options.memory,
                            });
                        }
                        times.push(point.time);
                        values.push(point.x);
                        observed.push(&point.found);
                    }
                    for time in settles {
                        breakpoints.insert(time);
                    }
                }
                Step::Pending | Step::Rejected => {}
                Step::Land(time) => breakpoints.insert(time),
            }
        }
        let mut statistics = stepper.statistics;
        (statistics.work, statistics.profile) = self.metered_since(&reading);
        Ok(Waveforms {
            times,
            values,
            observations: observed,
            logic: self.state.events.traces().to_vec(),
            statistics,
        })
    }

    /// What `read` reads of the recorded devices, its time counted as recording.
    fn record(&mut self, read: impl FnOnce(&mut Observer) -> Observation) -> Observation {
        let previous = self.state.workspace.meter.enter(Activity::Recording);
        let found = read(&mut self.state.observer);
        self.state.workspace.meter.leave(previous);
        found
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
        // The step grows from the last by its error, but never past the longest allowed,
        // whether it lands on its target or not.
        let reach = stepper.next.min(stepper.longest);
        let landing = reach >= remaining - stepper.shortest;
        let length = if landing { remaining } else { reach };
        let end = if landing {
            target
        } else {
            stepper.time + length
        };
        // Only a breakpoint is a corner in the waveforms, after which the integration
        // restarts; an observation is a time like any other.
        let corner = landing && target >= breakpoint;
        // Backward Euler until the points since the last corner are enough to estimate the
        // error of the transient's own method, so that every step but the first after a
        // corner is judged by its own estimate, and the first by the second's.
        let available = stepper.history.len() + 1;
        let method = if stepper.damp || available < stepper.method.order() + 2 {
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

        let mut candidate = stepper
            .resume
            .take()
            .unwrap_or_else(|| stepper.predicted(end));
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
        // The recorded devices are read as the evaluation that solved the step left them,
        // before any other evaluation.
        let read = self.read_recorded();
        let charge = self.state.workspace.contributions.charge.clone();
        let derivative: Vec<f64> = charge
            .iter()
            .zip(&stepper.terms)
            .map(|(&q, &term)| formula.alpha * q + term)
            .collect();
        let tried = Tried {
            end,
            length,
            method,
            formula,
            iterations,
            x: candidate,
            charge,
            derivative,
            read,
        };
        let previous = self.state.workspace.meter.enter(Activity::StepControl);
        let judged = self.judged(stepper, &tried, options);
        self.state.workspace.meter.leave(previous);
        let Some(next) = judged? else {
            return Ok(Step::Rejected);
        };
        // The logic before a first step moves it on, for going back to the corner.
        let first = stepper.history.len() == 1;
        let logic = first.then(|| self.state.events.checkpoint());
        let previous = self.state.workspace.meter.enter(Activity::StepControl);
        let events = self.events_within(stepper, (end, &tried.x), options);
        self.state.workspace.meter.leave(previous);
        let (commands, jumped) = match events {
            Ok(found) => found,
            Err(early) => {
                // An event lands no closer than the shortest step allowed; the step tried
                // next lands on it, as it is shorter than this one.
                stepper.statistics.landed += 1;
                return Ok(Step::Land(early));
            }
        };
        // A breakpoint, a driver's ramp starting or a device's jump is a corner in the
        // waveforms. The step after it is a tenth of the step the error allowed before it,
        // as SPICE's saved step, or of the gap to the next breakpoint if shorter.
        let restart = (corner || !commands.is_empty() || jumped).then(|| {
            let gap = if corner { gap } else { breakpoint - end };
            RESTART_FRACTION * reach.min(gap)
        });
        Ok(self.take(stepper, tried, (next, restart), (commands, logic), options))
    }

    /// Judges the step `tried` by its error, and the pending first step after the last
    /// corner with it; returns the length of the step after it, or `None` after rejecting
    /// the step, or both, to take it again shorter.
    fn judged(
        &mut self,
        stepper: &mut Stepper,
        tried: &Tried,
        options: &Options,
    ) -> Result<Option<f64>, SimulationError> {
        let new = (tried.end, tried.charge.as_slice());
        if let Some(pending) = &stepper.pending {
            // The first step after the corner, by the curvature it shares with this one.
            let span = Span {
                length: pending.length,
                alpha: pending.formula.alpha,
                charge: (&stepper.history.last.1, &pending.corner.charge),
                derivative: (&stepper.history.derivative, &pending.corner.derivative),
            };
            let judged = self.judge(Method::BackwardEuler, &stepper.history, new, &span, options);
            if let Some(judged) = judged
                && judged.ratio > 1.0
            {
                let next = next_length(pending.length, Some(judged));
                self.back_to_corner(stepper, next, judged.worst)?;
                return Ok(None);
            }
        }
        // The first step after a corner has no estimate of its own: the next judges it.
        let own = if stepper.history.len() == 1 {
            None
        } else {
            let span = Span {
                length: tried.length,
                alpha: tried.formula.alpha,
                charge: (&tried.charge, &stepper.history.last.1),
                derivative: (&tried.derivative, &stepper.history.derivative),
            };
            self.judge(tried.method, &stepper.history, new, &span, options)
        };
        let next = next_length(tried.length, own);
        if let Some(judged) = own
            && judged.ratio > 1.0
        {
            let name = &self.elaborated.unknowns[judged.worst].name;
            stepper.reject(next, || Rejection::Inaccurate(name.clone()))?;
            return Ok(None);
        }
        Ok(Some(next))
    }

    /// Takes the step `tried`, judged to allow a next step `next` long, and through the logic,
    /// which drivers must follow `commands`: accepts it, with the pending first step after
    /// the last corner, or holds it as that first step, the logic before it `logic`.
    /// `restart`, the length of the step after it, says the integration restarts at its end.
    fn take(
        &mut self,
        stepper: &mut Stepper,
        tried: Tried,
        (next, restart): (f64, Option<f64>),
        (commands, logic): (Vec<Command>, Option<Checkpoint>),
        options: &Options,
    ) -> Step {
        let Tried {
            end,
            length,
            method,
            formula,
            iterations,
            x,
            charge,
            derivative,
            read,
        } = tried;
        let mut points = Vec::with_capacity(2);
        if let Some(pending) = stepper.pending.take() {
            // The first step after the corner, its error now estimated within tolerance.
            let at = (stepper.time, stepper.x.clone());
            let step = (pending.formula, pending.length, pending.iterations);
            points.push(self.accept(stepper, at, step, (false, &pending.read)));
        }
        let last = (stepper.time, std::mem::replace(&mut stepper.x, x));
        let settles = self.follow(&commands);
        if restart.is_none()
            && let Some(logic) = logic
        {
            stepper.pending = Some(Pending {
                formula,
                length,
                iterations,
                read,
                corner: Corner {
                    time: last.0,
                    x: last.1.clone(),
                    charge: stepper.history.last.1.clone(),
                    derivative: stepper.history.derivative.clone(),
                    previous: stepper.previous,
                    logic,
                },
            });
        } else {
            // A first step that ends at a corner has no point after it to estimate its
            // error from, and is accepted as SPICE accepts it.
            let at = (end, stepper.x.clone());
            let step = (formula, length, iterations);
            points.push(self.accept(stepper, at, step, (restart.is_some(), &read)));
        }
        stepper.previous = length;
        stepper.time = end;
        if let Some(after_corner) = restart {
            stepper.before = None;
            stepper.history.restart(end, charge, derivative);
            stepper.damp = false;
            stepper.next = after_corner;
        } else {
            let trapezoidal = method == Method::Trapezoidal;
            let previous = self.state.workspace.meter.enter(Activity::StepControl);
            stepper.damp = trapezoidal && self.rings(&stepper.history, &derivative, options);
            self.state.workspace.meter.leave(previous);
            stepper.before = Some(last);
            stepper.history.accept(end, charge, derivative, trapezoidal);
            stepper.next = next;
        }
        if points.is_empty() {
            Step::Pending
        } else {
            Step::Accepted(points, settles)
        }
    }

    /// Accepts the point `(time, x)` that a step reached by `formula` over its length in so
    /// many Newton iterations, where the recorded devices did what `read` read; `restarted`
    /// when the integration begins afresh from it.
    fn accept(
        &mut self,
        stepper: &mut Stepper,
        (time, x): (f64, Vec<f64>),
        (formula, length, iterations): (Formula, f64, usize),
        (restarted, read): (bool, &Read),
    ) -> Point {
        stepper.statistics.accepted += 1;
        stepper.statistics.iterations += iterations;
        let found = self.record(|observer| observer.step(read, (&formula, restarted)));
        Point {
            time,
            x,
            formula,
            length,
            restarted,
            found,
        }
    }

    /// Goes back to the corner the pending first step started from, its error, estimated by
    /// the step after it, too large in the equation `worst`, and rejects both steps, to take
    /// the first again `next` long.
    fn back_to_corner(
        &mut self,
        stepper: &mut Stepper,
        next: f64,
        worst: usize,
    ) -> Result<Step, SimulationError> {
        if let Some(Pending { corner, .. }) = stepper.pending.take() {
            self.state.events.restore(corner.logic);
            stepper.time = corner.time;
            stepper.x = corner.x;
            stepper.before = None;
            stepper.resume = None;
            stepper
                .history
                .restart(corner.time, corner.charge, corner.derivative);
            stepper.previous = corner.previous;
            stepper.damp = false;
            stepper.statistics.rejected += 1;
        }
        let name = &self.elaborated.unknowns[worst].name;
        stepper.reject(next, || Rejection::Inaccurate(name.clone()))
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
        self.elaborated
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
        // A jump closer to the step's start than the shortest step is the one the step
        // before landed on, where a strict comparison had not yet turned: the integration
        // restarted there, and the step already follows the jumped contribution.
        let jumped = jump.is_some_and(|time| time - stepper.time >= stepper.shortest);
        Ok((commands, jumped))
    }

    /// The first time between the stepper's last point and `end`, where the solution is
    /// `after`, at which a device's contribution jumps, if one does: where the earliest of
    /// its choices that changed between the two points changes, located by linear
    /// interpolation.
    fn first_jump(&mut self, stepper: &Stepper, (end, after): (f64, &[f64])) -> Option<f64> {
        let start = stepper.time;
        let before = Input::new(&stepper.x, Some(start), 0.0);
        let later = Input::new(after, Some(end), 0.0);
        let groups = &self.elaborated.groups;
        let fraction = self.state.runs.first_jump(groups, (&before, &later))?;
        Some(start + (end - start) * fraction)
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
        let Self {
            elaborated, state, ..
        } = self;
        let (logic, events) = (&elaborated.logic, &mut state.events);
        let early = |time: f64| stepper.lands_before(time, end, options);
        let held = events.detector_levels(logic);
        let crossings: Vec<Crossing> = logic
            .detectors
            .iter()
            .enumerate()
            .filter_map(|(detector, &(element, _))| {
                crossing(
                    |input| state.runs.sense(&elaborated.groups, element, input),
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
        // move on for good; with them, it moves on tentatively, and goes back if a driver
        // must change before the step ends.
        let checkpoint = (!logic.drivers.is_empty()).then(|| events.checkpoint());
        events.inject(logic, &crossings);
        let commands = events.advance(logic, end);
        let first = commands
            .iter()
            .map(|command| command.time)
            .filter(|&time| early(time))
            .min_by(f64::total_cmp);
        if let Some(first) = first {
            if let Some(checkpoint) = checkpoint {
                events.restore(checkpoint);
            }
            return Err(first);
        }
        Ok(commands)
    }

    /// Starts each logic driver of `commands` on its ramp, and returns when each ramp ends.
    fn follow(&mut self, commands: &[Command]) -> Vec<f64> {
        let mut settles = Vec::with_capacity(commands.len());
        for command in commands {
            let element = self.elaborated.logic.drivers[command.driver].0;
            let ramp = (command.time, command.level, command.transition);
            settles.extend(self.drive(element, ramp));
        }
        settles
    }

    /// The breakpoints of every device in `(0, stop]`, sorted, merged when closer than
    /// `minimum`, and ending at `stop`, so never empty; their time is counted as preparing.
    fn breakpoints(&mut self, stop: f64, minimum: f64) -> Vec<f64> {
        let previous = self.state.workspace.meter.enter(Activity::Preparing);
        let mut times = vec![stop];
        self.state.runs.breakpoints((0.0, stop), &mut times);
        times.retain(|&time| time > minimum);
        times.sort_by(f64::total_cmp);
        times.dedup_by(|later, earlier| *later - *earlier < minimum);
        if let Some(last) = times.last_mut() {
            *last = stop;
        }
        self.state.workspace.meter.leave(previous);
        times
    }

    /// Judges `span`, a step by `method`, by its truncation error, estimated from the
    /// divided difference of the charges at the last of `history`'s points and at `new`, as
    /// many as the method's order and two, filtered through the step's own equations;
    /// `None` while too few points follow the last corner to estimate it.
    ///
    /// A charge's tolerance is the larger of a relative tolerance on the charge itself and one
    /// on the current it carries over the step, scaled by the truncation tolerance. That
    /// tolerance bounds the error as SPICE's `TRTOL` does, as SPICE estimates it: the error
    /// constant times the divided difference, without the factorial that makes the
    /// divided difference a derivative, so the error itself may be `(p+1)!` times larger.
    fn judge(
        &mut self,
        method: Method,
        history: &History,
        new: (f64, &[f64]),
        span: &Span<'_>,
        options: &Options,
    ) -> Option<Judgement> {
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
        let differences: Vec<f64> = (0..new.1.len())
            .map(|index| {
                for (slot, (_, charge)) in samples.iter_mut().zip(&used) {
                    *slot = charge[index];
                }
                samples[needed - 1] = new.1[index];
                divided_difference(&times[..needed], &samples[..needed])
            })
            .collect();
        let differences = self.filtered(differences, span.alpha);
        let scale = options.truncation_tolerance * method.factorial();
        let mut worst = Judgement {
            ratio: 0.0,
            worst: 0,
            longest: f64::INFINITY,
        };
        for ((index, unknown), &difference) in self
            .elaborated
            .unknowns
            .iter()
            .enumerate()
            .zip(&differences)
        {
            let error = method.truncation_error(span.length, difference);
            let absolute_current = options.equation_tolerance(unknown.quantity);
            let charge = span.charge.0[index].abs().max(span.charge.1[index].abs());
            let charge_tolerance = options.relative_tolerance * charge + options.charge_tolerance;
            let current = span.derivative.0[index]
                .abs()
                .max(span.derivative.1[index].abs());
            let current_tolerance =
                span.length * (options.relative_tolerance * current + absolute_current);
            let tolerance = scale * charge_tolerance.max(current_tolerance);
            if error / tolerance > worst.ratio {
                worst.ratio = error / tolerance;
                worst.worst = index;
            }
            let longest = method.longest_step(
                difference,
                scale * charge_tolerance,
                scale * current_tolerance / span.length,
            );
            worst.longest = worst.longest.min(longest);
        }
        Some(worst)
    }

    /// The charges' divided `differences` filtered through the step's equations at its end,
    /// whose formula's coefficient of the new charge is `alpha`: `C (G + αC)⁻¹ α d`.
    ///
    /// An error `e` in the formula's derivative of the charges moves the step's solution by
    /// `−(G + αC)⁻¹ α e`, so where the capacitances dominate the step's equations, as along
    /// a smooth waveform, the charges move by `e` itself, while a stiff component, one that
    /// settles within the step, barely moves: the divided differences read how far it
    /// settled, which the formula damps, not the formula's error (Shampine, "Implementation
    /// of implicit formulas for the solution of ODEs", SIAM Journal on Scientific and
    /// Statistical Computing 1(1), 1980; Hairer and Wanner, "Solving Ordinary Differential
    /// Equations II", section IV.8). Where the matrix cannot be factored, the differences are
    /// taken as they are.
    fn filtered(&mut self, mut differences: Vec<f64>, alpha: f64) -> Vec<f64> {
        let contributions = &self.state.workspace.contributions;
        let matrix: Vec<f64> = contributions
            .conductance
            .iter()
            .zip(&contributions.capacitance)
            .map(|(conductance, capacitance)| conductance + alpha * capacitance)
            .collect();
        let mut moved: Vec<f64> = differences
            .iter()
            .map(|difference| alpha * difference)
            .collect();
        let meter = &mut self.state.workspace.meter;
        let previous = meter.enter(Activity::Factoring);
        let solved = self.state.solver.factor(&matrix).and_then(|mut factors| {
            meter.enter(Activity::Solving);
            factors.solve(&mut moved)
        });
        meter.leave(previous);
        if solved.is_ok() {
            differences.fill(0.0);
            self.elaborated.pattern.add_product(
                &self.state.workspace.contributions.capacitance,
                &moved,
                &mut differences,
            );
        }
        differences
    }
}

/// The length of the step after one `length` long judged `judged`: the longest its error
/// allows, with a safety factor, within bounds on the change, or the most it may grow to
/// when its error was not estimated.
fn next_length(length: f64, judged: Option<Judgement>) -> f64 {
    let (shortest, longest) = (FACTOR_BOUNDS.0 * length, FACTOR_BOUNDS.1 * length);
    judged.map_or(longest, |judged| {
        (SAFETY * judged.longest).clamp(shortest, longest)
    })
}

#[cfg(test)]
mod tests {
    use super::Breakpoints;

    /// The bits of a breakpoint and of the gap after it, to compare them exactly.
    fn bits((next, gap): (f64, f64)) -> (u64, u64) {
        (next.to_bits(), gap.to_bits())
    }

    #[test]
    fn breakpoints_come_in_order_without_near_twins_and_keep_the_stop() {
        let mut breakpoints = Breakpoints::new(&[1.0, 3.0, 10.0], 0.1);
        breakpoints.insert(2.0);
        // Within the minimum of 2 and of 3, so dropped; past the stop, so dropped.
        breakpoints.insert(2.05);
        breakpoints.insert(2.95);
        breakpoints.insert(11.0);
        assert_eq!(bits(breakpoints.after(0.5)), bits((1.0, 1.0)));
        assert_eq!(bits(breakpoints.after(1.0)), bits((2.0, 1.0)));
        assert_eq!(bits(breakpoints.after(2.5)), bits((3.0, 7.0)));
        // The stop stays, however far past it the time asked for.
        assert_eq!(bits(breakpoints.after(20.0)), bits((10.0, f64::INFINITY)));
        // A negative zero orders as zero, before every positive time.
        breakpoints.insert(-0.0);
        assert_eq!(breakpoints.first().to_bits(), 0.0_f64.to_bits());
    }
}
