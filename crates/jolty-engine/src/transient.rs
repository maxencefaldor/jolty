//! The transient analysis: the circuit's response over time.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use jolty_models::{Input, LogicLevel, Past};
use jolty_numerics::integration::{
    CONSTANT_STEP, ErrorBound, Formula, Method, divided_difference, longest_at,
};
use serde::Deserialize;

use crate::derivative::Forward;
use crate::logic::{Checkpoint, Command, Crossing, crossing};
use crate::newton::{
    Failure, Mode, NAMED_UNKNOWNS, Workspace, add_product, each_chunk, pieces, step_matrix,
};
use crate::options::{Inside, NUMBERS_PER_GIB};
use crate::periodic::Monodromy;
use crate::profile::Activity;
use crate::record::{Observer, Read};
use crate::ringing;
use crate::{
    Options, Profile, RecordedSeries, RecordedValues, Rejection, SimulationError, Simulator,
    Strategy, Work,
};

/// The steps of a run its longest step is by default a fraction of: SPICE's longest step,
/// a fiftieth of the run (ngspice 47's `traninit.c`, lines 26 to 34).
pub(crate) const STEPS_PER_RUN: f64 = 50.0;
/// The first step after the start or a breakpoint, as a fraction of the shorter of the step
/// before and the gap to the next breakpoint, so a short waveform segment takes several steps.
///
/// Source: ngspice 47 (`src/spicelib/analysis/dctran.c`, lines 507 to 508:
/// `MIN(ckt->CKTdelta, .1 * MIN(ckt->CKTsaveDelta, ckt->CKTbreaks[1] - ckt->CKTbreaks[0]))`).
const RESTART_FRACTION: f64 = 0.1;
/// The smallest step allowed, as a fraction of the longest: SPICE's `delmin` (ngspice 47's
/// `traninit.c`, line 36). A charge behind a small resistance settles within picoseconds
/// after a corner, such as the benchmarks' buck's switches' internal sources, 10 mΩ to 1 nF,
/// in 10 ps, and resolving that to SPICE's charge tolerance at a relative tolerance of 1e-5
/// takes steps of 2e-14 s, below 1e-9 of the buck's longest.
const SMALLEST_STEP_FRACTION: f64 = 1e-11;
/// The factor a step shrinks by when Newton iteration fails (`1 / 8 = 0.125`, exact in binary
/// floating point).
///
/// Source: ngspice 47 (`src/spicelib/analysis/dctran.c`, line 733:
/// `ckt->CKTdelta = ckt->CKTdelta/8;`).
const FAILURE_SHRINK: f64 = 0.125;
/// The bounds on the factor the truncation error changes the step by.
///
/// Source: upper bound `2.0` matches SPICE / ngspice 47 (`src/spicelib/analysis/ckttrunc.c`,
/// line 53: `*timeStep = MIN(2 * *timeStep, timetemp);`); lower bound `0.1` follows standard
/// ODE step-size clamping (Hairer, Nørsett and Wanner, *Solving Ordinary Differential
/// Equations I*, 2nd ed., 1993, II.4).
const FACTOR_BOUNDS: (f64, f64) = (0.1, 2.0);
/// The safety factor on the step the truncation error predicts.
///
/// Source: standard ODE step-size safety factor `fac = 0.9` (Hairer, Nørsett and Wanner,
/// *Solving Ordinary Differential Equations I*, 2nd ed., 1993, II.4; Hairer and Wanner,
/// *Solving Ordinary Differential Equations II*, 1996, IV.8), matching ngspice 47's step
/// acceptance threshold `newdelta > .9 * ckt->CKTdelta` (`dctran.c`, line 801).
const SAFETY: f64 = 0.9;
/// The most accepted points the error estimate reads: enough for second-order formulas.
///
/// Derivation: the local truncation error of a second-order formula (`p = 2`, trapezoidal or
/// Gear2) uses the `(p + 1) = 3`rd divided difference over `p + 2 = 4` points, i.e. 3 prior
/// accepted points plus the candidate step.
const HISTORY: usize = 3;
/// The earlier derivatives kept to tell the trapezoidal formula's ringing, besides the last.
///
/// Derivation: detecting two consecutive slope sign reversals (`turns[0] * turns[1] < 0` and
/// `turns[1] * turns[2] < 0`) requires 4 consecutive derivatives, so 2 earlier ones are held
/// beside `history.derivative` and `newest`.
const RINGING: usize = 2;

/// A transient analysis from time zero, starting at the operating point, or from its
/// initial conditions alone.
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
    pub breakpoints: Vec<f64>,
    /// The voltage, in volts, each net named, such as `amplifier/mid`, starts from, as
    /// SPICE's `.ic`: forced there while the operating point the transient starts from is
    /// solved, then left free; or, without that operating point, set there. A net that
    /// voltage sources and inductors join to the reference, or to a net named before it,
    /// is left to them, and its voltage at the operating point must be the one given,
    /// within the tolerances. None by default.
    #[serde(default)]
    pub initial_conditions: BTreeMap<String, f64>,
    /// Whether the transient starts from the operating point; or else from its initial
    /// conditions alone, every other unknown zero, as SPICE's `UIC`, so that its first
    /// step leaves a state that need not solve the circuit's equations. From the
    /// operating point by default.
    #[serde(default = "from_operating_point")]
    pub operating_point: bool,
}

/// That a transient starts from the operating point, unless it says otherwise.
const fn from_operating_point() -> bool {
    true
}

/// The waveforms of a transient analysis, at every accepted time point.
#[derive(Debug, Clone, PartialEq)]
pub struct TransientResult {
    /// The accepted times, in seconds, from zero to the stop time.
    pub times: Vec<f64>,
    /// The value of every unknown at each time.
    pub values: Vec<Vec<f64>>,
    /// What the devices its options record did at each time: the current into each
    /// recorded terminal and the charge it holds, and the recorded devices' derivatives and
    /// variables. A current includes its charge's derivative, by the step's integration
    /// formula.
    pub observations: RecordedSeries,
    /// Each logic net's changes: its level at time zero, then each time it changes, in
    /// seconds, and its new level.
    pub logic: Vec<Vec<(f64, LogicLevel)>>,
    /// How the analysis went.
    pub statistics: Statistics,
}

/// Each equation's converged residual integrated over a transient's accepted steps, in the
/// order of the circuit's unknowns (Kundert and Clifford, "Achieving Accurate Results with
/// a Circuit Simulator", IEEE CICC 1993).
///
/// For an electrical node (and a displacement equation), the unit is coulombs (`A·s`):
/// the charge the solver's nonzero KCL residual added to the node. For a branch current
/// equation, the unit is webers (`V·s`, magnetic flux linkage); for a thermal net, joules
/// (`W·s`); for a device's state equation, the state's own unit.
///
/// At every accepted step `n`, Newton iteration stops with residual
/// `r_n = f(x_n, t_n) + q̇_n`. Integrating `r_n` by the step's own
/// [`Formula`](jolty_numerics::integration::Formula),
/// `Δℓ_n = (r_n − γ r_{n-1} + β₁ Δℓ_{n-1}) / α`, splits each step's discrete charge change
/// `Δq_n` into the integral of the static currents `−f` plus `Δℓ_n`. On any closed node or
/// cutset whose static currents sum to zero, the change in stored charge from start to end
/// equals the sum of `signed` to floating-point rounding, and the charge excursion at every
/// intermediate accepted time stays within the sum of `absolute`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ChargeLeak {
    /// The signed integral `∑ Δℓ_n` for each unknown: the net charge, flux or energy the
    /// solver's nonzero residual added over the transient.
    pub signed: Vec<f64>,
    /// The cumulative absolute integral `∑ |Δℓ_n|` for each unknown: the sum of step leak
    /// magnitudes without cancellation, bounding the excursion at every intermediate
    /// accepted time.
    pub absolute: Vec<f64>,
}

/// How a transient analysis went.
#[derive(Debug, Clone, PartialEq)]
pub struct Statistics {
    /// How the initial operating point was found, or `None` for a transient that starts
    /// from its initial conditions alone.
    pub initial: Option<Strategy>,
    /// The steps accepted.
    pub accepted: usize,
    /// The steps rejected, for failing to converge or for too large an error.
    pub rejected: usize,
    /// The steps tried again shorter to land on an event within them: a detector's
    /// crossing, a logic driver's change or a device's jump.
    pub landed: usize,
    /// The Newton iterations of the accepted steps.
    pub iterations: usize,
    /// The work the analysis did, its operating point's included, and for a periodic
    /// steady state every period's. Unlike the results, it depends on what the simulator
    /// did before: on an operating point it remembers, and on the pivots its sparse
    /// solver kept.
    pub work: Work,
    /// Where its time went, when its options' `profile` is on.
    pub profile: Option<Profile>,
    /// Each equation's converged residual integrated over the transient's accepted steps.
    pub charge_leak: ChargeLeak,
}

impl Statistics {
    /// The statistics of a transient before its first step, from an operating point found
    /// by `initial`, or from its initial conditions with `None`.
    fn starting(initial: Option<Strategy>) -> Self {
        Self {
            initial,
            accepted: 0,
            rejected: 0,
            landed: 0,
            iterations: 0,
            work: Work::default(),
            profile: None,
            charge_leak: ChargeLeak::default(),
        }
    }
}

/// Adds `term` into `(total, compensation)` using Neumaier's compensated summation
/// (Neumaier, *ZAMM* 54, 1974, pages 39 to 51), preserving accuracy both when small terms add
/// into a larger running sum and when terms of opposite sign cancel back toward zero.
#[inline]
fn neumaier_add(total: &mut f64, compensation: &mut f64, term: f64) {
    let next = *total + term;
    if total.abs() >= term.abs() {
        *compensation += (*total - next) + term;
    } else {
        *compensation += (term - next) + *total;
    }
    *total = next;
}

/// Running residual-integral state for a single unknown across accepted transient steps.
#[derive(Debug, Clone, Copy, Default)]
struct UnknownLeak {
    signed: f64,
    signed_compensation: f64,
    absolute: f64,
    absolute_compensation: f64,
    previous_residual: f64,
    previous_increment: f64,
}

/// Accumulates each equation's residual integrated over accepted steps in a single cache
/// line per unknown, updated sequentially in time and independently per unknown so that
/// splitting across threads is bit-for-bit identical to running serially.
struct LeakAccumulator {
    states: Vec<UnknownLeak>,
}

impl LeakAccumulator {
    /// Starts at zero for `unknowns` equations.
    fn new(unknowns: usize) -> Self {
        Self {
            states: vec![UnknownLeak::default(); unknowns],
        }
    }

    /// Accumulates one accepted step solved with `formula` whose converged residual is
    /// `residual`; `restarted` clears the previous-step history when integration restarts
    /// after a corner. Work across unknowns splits as `inside` says.
    fn accumulate(
        &mut self,
        formula: &Formula,
        residual: &[f64],
        restarted: bool,
        inside: Option<Inside>,
    ) {
        each_chunk(&mut self.states, inside, |start, chunk| {
            for (state, &current_residual) in chunk.iter_mut().zip(&residual[start..]) {
                let increment = formula.integral_increment(
                    current_residual,
                    state.previous_residual,
                    state.previous_increment,
                );
                neumaier_add(&mut state.signed, &mut state.signed_compensation, increment);
                neumaier_add(
                    &mut state.absolute,
                    &mut state.absolute_compensation,
                    increment.abs(),
                );
                if restarted {
                    state.previous_residual = 0.0;
                    state.previous_increment = 0.0;
                } else {
                    state.previous_residual = current_residual;
                    state.previous_increment = increment;
                }
            }
        });
    }

    /// Completes the compensated sums and yields the final [`ChargeLeak`].
    fn finish(self) -> ChargeLeak {
        let mut signed = Vec::with_capacity(self.states.len());
        let mut absolute = Vec::with_capacity(self.states.len());
        for state in self.states {
            signed.push(state.signed + state.signed_compensation);
            absolute.push(state.absolute + state.absolute_compensation);
        }
        ChargeLeak { signed, absolute }
    }
}

/// What a transient carries along its accepted steps besides its solution.
pub(crate) enum Carried<'a> {
    /// Nothing.
    Nothing,
    /// Each parameter's derivatives, for a sensitivity analysis.
    Forward(&'a mut Forward),
    /// The derivatives with respect to the initial state, and an oscillator's with respect
    /// to its period, for a periodic steady state.
    Monodromy(&'a mut Monodromy),
    /// Each accepted step's formula, and whether the integration restarts after it, in
    /// their order, for the adjoint method to go back through.
    Steps(&'a mut Vec<(Formula, bool)>),
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

    /// Of the charges themselves, carried as their own tangent, the part of `formula`'s
    /// `dq/dt` at the point where they are `charge` that the steps' lengths divide:
    /// `α q_{n+1} + β₀ q_n + β₁ q_{n-1}`, without the earlier derivative's `γ q̇_n`.
    pub(crate) fn divided(&self, charge: &[f64], formula: &Formula) -> Vec<f64> {
        let before = self.before.as_ref().unwrap_or(&self.last);
        charge
            .iter()
            .zip(&self.last)
            .zip(before)
            .map(|((&new, &last), &before)| {
                formula.alpha * new + formula.history([last, before], 0.0)
            })
            .collect()
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
    found: RecordedValues,
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
    /// The converged residual vector at its end, kept until the next step accepts or rejects it.
    residual: Vec<f64>,
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
    /// The length of the step before it, which a multistep formula's error reads.
    previous: f64,
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
    /// The longest step every charge's error, by the same estimate, allows next: after the
    /// step, from its end, where it is accepted, or in its place, from its start, where it is
    /// rejected, which differ for a multistep formula, whose error reads the step before.
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
    /// The running residual integral of every accepted step so far.
    leak: LeakAccumulator,
    statistics: Statistics,
}

impl Stepper {
    /// Starts a transient stepper at time zero with initial solution `x` and `charge`.
    fn starting(
        method: Method,
        (longest, shortest, first): (f64, f64, f64),
        (x, charge): (Vec<f64>, Vec<f64>),
        strategy: Option<Strategy>,
    ) -> Self {
        let unknowns = x.len();
        Self {
            method,
            longest,
            shortest,
            time: 0.0,
            resume: None,
            before: None,
            history: History {
                earlier: VecDeque::with_capacity(HISTORY - 1),
                last: (0.0, charge),
                derivative: vec![0.0; unknowns],
                ringing: VecDeque::with_capacity(RINGING),
            },
            next: first,
            previous: first,
            damp: false,
            terms: vec![0.0; unknowns],
            pending: None,
            leak: LeakAccumulator::new(unknowns),
            statistics: Statistics::starting(strategy),
            x,
        }
    }

    /// Where Newton iteration starts a step ending at `end`, as SPICE's does: the line
    /// through the last two accepted points carried on to `end`, or the last point after a
    /// corner.
    ///
    /// The work on each unknown is split as `inside` says.
    fn predicted(&self, end: f64, inside: Option<Inside>) -> Vec<f64> {
        let Some((time, before)) = &self.before else {
            return self.x.clone();
        };
        let ratio = (end - self.time) / (self.time - time);
        let mut predicted = vec![0.0; self.x.len()];
        each_chunk(&mut predicted, inside, |start, chunk| {
            let points = self.x[start..].iter().zip(&before[start..]);
            for (value, (&last, &earlier)) in chunk.iter_mut().zip(points) {
                *value = last + (last - earlier) * ratio;
            }
        });
        predicted
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
    ) -> Result<TransientResult, SimulationError> {
        options.parallelism.confined(|| {
            self.solvable()?;
            self.transient_with(transient, options, None, Carried::Nothing)
        })
    }

    /// The state `transient` starts from, and how its operating point was found: the
    /// operating point, solved with its initial conditions forced if it has any, or else
    /// its initial conditions alone, every other unknown zero.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::InvalidAnalysis`] for initial conditions that name no
    /// net, or that the circuit's sources contradict, and for a transient from its
    /// initial conditions alone of a circuit with logic, whose levels only an operating
    /// point settles; and the errors of [`Simulator::operating_point`].
    fn initial_state(
        &mut self,
        transient: &Transient,
        options: &Options,
    ) -> Result<(Vec<f64>, Option<Strategy>), SimulationError> {
        let initial = &transient.initial_conditions;
        if transient.operating_point {
            let point = if initial.is_empty() {
                self.operating_point(options)?
            } else {
                self.forced_operating_point(initial, options)?
            };
            return Ok((point.values, Some(point.strategy)));
        }
        if !self.elaborated.logic.devices.is_empty() {
            return Err(SimulationError::InvalidAnalysis(
                "a transient from its initial conditions alone of a circuit with logic, \
                 whose levels only an operating point settles"
                    .to_owned(),
            ));
        }
        let [given] = self.forcings([(initial, "initial condition")])?;
        let mut state = vec![0.0; self.elaborated.unknowns.len()];
        for (unknown, voltage) in given.forced.into_iter().chain(given.fixed) {
            state[unknown.index()] = voltage;
        }
        Ok((state, None))
    }

    /// The longest step `transient` allows, after checking its times.
    pub(crate) fn longest_step(&self, transient: &Transient) -> Result<f64, SimulationError> {
        // Without unknowns, nothing analog needs steps, and the logic moves on at once.
        let longest = if self.elaborated.unknowns.is_empty() {
            transient.stop
        } else {
            transient.max_step.unwrap_or(transient.stop / STEPS_PER_RUN)
        };
        if transient.stop <= 0.0 || longest <= 0.0 {
            return Err(SimulationError::InvalidAnalysis(format!(
                "the stop time and longest step must be positive, not {:?} and {longest:?}",
                transient.stop
            )));
        }
        if let Some(time) = transient
            .breakpoints
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
            Carried::Steps(steps) => steps.clear(),
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
            Carried::Steps(steps) => {
                steps.push((*formula.0, formula.1));
                Ok(())
            }
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
    /// operating point it came from was found, or else from its own initial state,
    /// carrying `carried` along its accepted steps.
    pub(crate) fn transient_with(
        &mut self,
        transient: &Transient,
        options: &Options,
        start: Option<(Vec<f64>, Option<Strategy>)>,
        mut carried: Carried<'_>,
    ) -> Result<TransientResult, SimulationError> {
        let reading = self.read_meter(options.profile);
        self.prepare(options)?;
        let longest = self.longest_step(transient)?;
        let shortest = longest * SMALLEST_STEP_FRACTION;
        let width = self.point_width();
        let most_held = options.max_memory * NUMBERS_PER_GIB;
        if transient.stop / longest * width > most_held {
            return Err(SimulationError::TooManyPoints {
                time: 0.0,
                step: longest,
                max_memory: options.max_memory,
            });
        }
        let mut held = width;
        let mut observations = transient.breakpoints.clone();
        observations.sort_by(f64::total_cmp);
        let (initial, strategy) =
            start.map_or_else(|| self.initial_state(transient, options), Ok)?;
        self.start_carried(&mut carried, &initial, options)?;
        let delays = self.start_past(&initial, transient.method);
        self.evaluate(&initial, Some(0.0), (1.0, options.gmin), true);
        let start = self.read_recorded();
        self.state
            .events
            .start(&self.elaborated.logic, transient.stop);
        let mut breakpoints = Breakpoints::new(
            &self.breakpoints(transient.stop, shortest, &delays),
            shortest,
        );
        let first = RESTART_FRACTION * longest.min(breakpoints.first());
        let charge = self.state.workspace.contributions.charge.clone();
        let mut stepper = Stepper::starting(
            transient.method,
            (longest, shortest, first),
            (initial, charge),
            strategy,
        );
        let (mut times, mut values) = (vec![0.0], vec![stepper.x.clone()]);
        let mut observed = RecordedSeries::default();
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
                                max_memory: options.max_memory,
                            });
                        }
                        self.passed(&mut breakpoints, (point.time, point.restarted), &delays);
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
        self.state.past = None;
        let mut statistics = stepper.statistics;
        statistics.charge_leak = stepper.leak.finish();
        (statistics.work, statistics.profile) = self.metered_since(&reading);
        Ok(TransientResult {
            times,
            values,
            observations: observed,
            logic: self.state.events.traces().to_vec(),
            statistics,
        })
    }

    /// Starts the past the circuit's delay lines read from the state `x` at time zero,
    /// a corner, read by interpolation of `method`'s order, keeping the unknowns they
    /// read; none for a circuit without one. Returns each distinct delay, in increasing
    /// order.
    fn start_past(&mut self, x: &[f64], method: Method) -> Vec<f64> {
        let delayed = self.state.runs.delayed(&self.elaborated.groups);
        let mut delays: Vec<f64> = delayed.iter().map(|part| part.delay).collect();
        delays.sort_by(f64::total_cmp);
        delays.dedup();
        self.state.past = (!delayed.is_empty()).then(|| {
            let kept: Vec<usize> = delayed.iter().map(|part| part.column as usize).collect();
            let mut past = Past::new(x.len(), &kept, method.order());
            past.push(0.0, x, true);
            past
        });
        delays
    }

    /// Moves the past the delay lines read on past the point accepted at `time`, and
    /// adds the breakpoint each corner makes, where `restarted` says the integration
    /// restarts: each of `delays` after it, where the corner arrives at a line's other
    /// port.
    ///
    /// The past keeps the points of the longest delay before `time`, which an
    /// interpolation at any later time reads.
    fn passed(
        &mut self,
        breakpoints: &mut Breakpoints,
        (time, restarted): (f64, bool),
        delays: &[f64],
    ) {
        let (Some(past), Some(&longest)) = (&mut self.state.past, delays.last()) else {
            return;
        };
        past.forget_before(time - longest);
        if restarted {
            for &delay in delays {
                breakpoints.insert(time + delay);
            }
        }
    }

    /// What `read` reads of the recorded devices, its time counted as recording.
    fn record(&mut self, read: impl FnOnce(&mut Observer) -> RecordedValues) -> RecordedValues {
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
        // Each unknown's work is its own, spread as Newton iteration spreads its work on
        // each unknown.
        let inside = options.parallelism.inside(stepper.x.len());
        let (last, before) = stepper.history.last_two();
        let derivative = &stepper.history.derivative;
        each_chunk(&mut stepper.terms, inside, |start, chunk| {
            for (offset, term) in chunk.iter_mut().enumerate() {
                let index = start + offset;
                *term = formula.history([last[index], before[index]], derivative[index]);
            }
        });

        let mut candidate = stepper
            .resume
            .take()
            .unwrap_or_else(|| stepper.predicted(end, inside));
        let mode = Mode::Step {
            time: end,
            alpha: formula.alpha,
            history: &stepper.terms,
        };
        let solved = self.newton(
            &mut candidate,
            mode,
            options,
            options.max_transient_iterations,
        );
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
        let mut derivative = vec![0.0; charge.len()];
        each_chunk(&mut derivative, inside, |start, chunk| {
            let terms = charge[start..].iter().zip(&stepper.terms[start..]);
            for (derivative, (&q, &term)) in chunk.iter_mut().zip(terms) {
                *derivative = formula.alpha * q + term;
            }
        });
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
                // A restart's step follows none, which backward Euler does not read.
                previous: pending.length,
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
                previous: stepper.previous,
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
        let inside = options.parallelism.inside(stepper.x.len());
        let mut points = Vec::with_capacity(2);
        if let Some(pending) = stepper.pending.take() {
            // The first step after the corner, its error now estimated within tolerance.
            let at = (stepper.time, stepper.x.clone());
            let step = (pending.formula, pending.length, pending.iterations);
            stepper.leak.accumulate(&pending.formula, &pending.residual, false, inside);
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
                residual: self.state.workspace.contributions.residual.clone(),
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
            stepper.leak.accumulate(
                &formula,
                &self.state.workspace.contributions.residual,
                restart.is_some(),
                inside,
            );
            points.push(self.accept(stepper, at, step, (restart.is_some(), &read)));
        }
        stepper.previous = length;
        stepper.time = end;
        if let Some(past) = &mut self.state.past {
            past.push(end, &stepper.x, restart.is_some());
        }
        if let Some(after_corner) = restart {
            stepper.before = None;
            stepper.history.restart(end, charge, derivative);
            stepper.damp = false;
            stepper.next = after_corner;
        } else {
            let trapezoidal = method == Method::Trapezoidal;
            let previous = self.state.workspace.meter.enter(Activity::StepControl);
            stepper.damp = trapezoidal
                && self.rings(
                    &stepper.history,
                    (end, &charge, &derivative),
                    formula.alpha,
                    options,
                );
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
            if let Some(past) = &mut self.state.past {
                past.back_to(corner.time);
            }
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

    /// Whether the trapezoidal formula is numerically ringing at `(end, charge, derivative)`:
    /// distinguishes genuine `(-1)^n` trapezoidal ringing (stiff real decaying modes with
    /// `σ ≥ α = 2/h`, algebraic capacitor-voltage-source loop modes, and zero-capacitance
    /// Meyer displacement unknowns) from real physical oscillations (single- or multi-mode
    /// lossless LC tanks, coupled transformers, gyrators, and active harmonic oscillators).
    ///
    /// One step of backward Euler, needing no derivative, restarts a ringing derivative
    /// afresh while leaving physical lossless oscillations undamped.
    fn rings(
        &mut self,
        history: &History,
        (end, charge, derivative): (f64, &[f64], &[f64]),
        alpha: f64,
        options: &Options,
    ) -> bool {
        if history.ringing.len() < RINGING || history.earlier.len() < RINGING {
            return false;
        }
        let run = ringing::Run {
            times: [
                history.earlier[0].0,
                history.earlier[1].0,
                history.last.0,
                end,
            ],
            charges: [
                history.earlier[0].1.as_slice(),
                history.earlier[1].1.as_slice(),
                history.last.1.as_slice(),
                charge,
            ],
            derivatives: [
                history.ringing[0].as_slice(),
                history.ringing[1].as_slice(),
                history.derivative.as_slice(),
                derivative,
            ],
            alpha,
        };
        ringing::rings(
            &self.elaborated.pattern,
            &self.elaborated.natures,
            &mut self.state.solver,
            &mut self.state.workspace,
            &run,
            options,
        )
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

    /// The breakpoints of every device in `(0, stop]`, and each of `delays`, where the
    /// start, a corner where the waveforms leave the operating point, arrives at a delay
    /// line's other port; sorted, merged when closer than `minimum`, and ending at `stop`,
    /// so never empty; their time is counted as preparing.
    fn breakpoints(&mut self, stop: f64, minimum: f64, delays: &[f64]) -> Vec<f64> {
        let previous = self.state.workspace.meter.enter(Activity::Preparing);
        let mut times = vec![stop];
        self.state.runs.breakpoints((0.0, stop), &mut times);
        times.extend(delays.iter().filter(|&&delay| delay < stop));
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
    /// A charge's tolerance is the larger of a relative tolerance on the charge itself, or on
    /// the charge tolerance where the charge is smaller, and one on the current it carries
    /// over the step, scaled by the truncation tolerance. That tolerance bounds the error as
    /// SPICE's `TRTOL` does, as SPICE estimates it: the error constant times the divided
    /// difference, without the factorial that makes the divided difference a derivative, so
    /// the error itself may be `(p+1)!` times larger.
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
        // Each unknown's work is its own, spread as Newton iteration spreads its work on each
        // unknown.
        let inside = options.parallelism.inside(new.1.len());
        let mut differences = std::mem::take(&mut self.state.workspace.differences);
        divided_differences((history, new), needed, inside, &mut differences);
        self.filter(&mut differences, span.alpha, inside);
        let scale = options.truncation_tolerance * method.factorial();
        let natures = &self.elaborated.natures;
        // Each charge's tolerances, on the charge and on the current it carries over the step,
        // times the truncation tolerance and the factorial SPICE's estimate leaves out. The
        // one on the charge is SPICE's, `reltol · max(|q|, chgtol)` (ngspice 47's
        // `cktterr.c`, lines 40 and 41): the charge tolerance is the least charge the
        // relative tolerance is taken of, not a tolerance added to it.
        let tolerances = |index: usize| {
            let charge = span.charge.0[index].abs().max(span.charge.1[index].abs());
            let current = span.derivative.0[index]
                .abs()
                .max(span.derivative.1[index].abs());
            let absolute_current = options.equation_tolerance(natures[index]);
            let charge_tolerance =
                options.relative_tolerance * charge.max(options.charge_tolerance);
            let current_tolerance =
                span.length * (options.relative_tolerance * current + absolute_current);
            (scale * charge_tolerance, scale * current_tolerance)
        };
        let none = (
            Judgement {
                ratio: 0.0,
                worst: 0,
                longest: f64::INFINITY,
            },
            (f64::INFINITY, ErrorBound::Charge),
        );
        // The largest ratio, at the first unknown that reaches it, and the shortest of the
        // charges' longest steps at a constant step's error constant, found in each piece and
        // then over the pieces in their order.
        // A multistep formula's longest step is found at its own error constant, from each
        // charge's unit steps, which the pass keeps for it.
        let keep = method.is_multistep();
        let per_difference = method.error_per_difference(span.length, span.previous);
        let found = pieces(differences.len(), inside, |range| {
            let (mut worst, mut found) = none;
            let mut units = Vec::with_capacity(if keep { range.len() } else { 0 });
            for index in range {
                let difference = differences[index];
                // The step's truncation error, as `Method::truncation_error` gives it.
                let error = per_difference * difference.abs();
                let (charge, current) = tolerances(index);
                let ratio = error / charge.max(current);
                if ratio > worst.ratio {
                    worst.ratio = ratio;
                    worst.worst = index;
                }
                let unit = method.unit_steps(difference, charge, current / span.length);
                found = least(found, longest_at(unit, CONSTANT_STEP));
                if keep {
                    units.push(unit);
                }
            }
            (worst, found, units)
        });
        let (worst, shortest_at_constant) =
            found
                .iter()
                .fold(none, |(worst, found), (piece, piece_found, _)| {
                    let worst = if piece.ratio > worst.ratio {
                        *piece
                    } else {
                        worst
                    };
                    (worst, least(found, *piece_found))
                });
        let mut found = found;
        let units: Vec<(f64, f64)> = match found.len() {
            1 => found.pop().map_or_else(Vec::new, |(_, _, units)| units),
            _ => found.into_iter().flat_map(|(_, _, units)| units).collect(),
        };
        // The longest step they all allow: after this one where it is accepted, in its place
        // where it is rejected.
        let previous = if worst.ratio > 1.0 {
            span.previous
        } else {
            span.length
        };
        let longest = method.longest_common_step(previous, shortest_at_constant, |scales| {
            shortest(&units, scales, inside)
        });
        self.state.workspace.differences = differences;
        Some(Judgement { longest, ..worst })
    }

    /// Filters the charges' divided `differences` through the step's equations at its end,
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
    /// left as they are.
    ///
    /// The work on each unknown and each stored entry is split as `inside` says, the same
    /// bits however it is.
    fn filter(&mut self, differences: &mut [f64], alpha: f64, inside: Option<Inside>) {
        let Workspace {
            contributions,
            jacobian,
            moved,
            meter,
            ..
        } = &mut self.state.workspace;
        step_matrix(jacobian, contributions, alpha, inside);
        moved.clear();
        moved.resize(differences.len(), 0.0);
        each_chunk(moved, inside, |start, chunk| {
            for (moved, &difference) in chunk.iter_mut().zip(&differences[start..]) {
                *moved = alpha * difference;
            }
        });
        let previous = meter.enter(Activity::Factoring);
        let solved = self.state.solver.factor(jacobian).and_then(|mut factors| {
            meter.enter(Activity::Solving);
            factors.solve(moved)
        });
        meter.leave(previous);
        if solved.is_err() {
            return;
        }
        differences.fill(0.0);
        let capacitance = &contributions.capacitance;
        add_product(
            &self.elaborated.pattern,
            (capacitance, moved),
            differences,
            inside,
        );
    }
}

/// Writes into `differences` each charge's divided difference over the last `needed`
/// points: those of `history`, then `new`, its work on each unknown split as `inside` says.
fn divided_differences(
    (history, new): (&History, (f64, &[f64])),
    needed: usize,
    inside: Option<Inside>,
    differences: &mut Vec<f64>,
) {
    // The times and the charges of the points used, oldest first, then the new one's.
    let mut times = [0.0; HISTORY + 1];
    let mut charges: [&[f64]; HISTORY + 1] = [&[]; HISTORY + 1];
    let used = history
        .points()
        .skip(history.len() + 1 - needed)
        .map(|(time, charge)| (*time, charge.as_slice()));
    for ((time, charge), (at, held)) in times.iter_mut().zip(&mut charges).zip(used.chain([new])) {
        *time = at;
        *charge = held;
    }
    differences.clear();
    differences.resize(new.1.len(), 0.0);
    each_chunk(differences, inside, |start, chunk| {
        let mut samples = [0.0; HISTORY + 1];
        for (offset, difference) in chunk.iter_mut().enumerate() {
            let index = start + offset;
            for (slot, charge) in samples.iter_mut().zip(&charges[..needed]) {
                *slot = charge[index];
            }
            *difference = divided_difference(&times[..needed], &mut samples[..needed]);
        }
    });
}

/// The length of the step after one `length` long judged `judged`, or of the step in its
/// place where its error rejects it: the longest its error allows, with a safety factor,
/// within bounds on the change, or the most it may grow to when its error was not
/// estimated.
fn next_length(length: f64, judged: Option<Judgement>) -> f64 {
    let (shortest, longest) = (FACTOR_BOUNDS.0 * length, FACTOR_BOUNDS.1 * length);
    judged.map_or(longest, |judged| {
        (SAFETY * judged.longest).clamp(shortest, longest)
    })
}

/// The shortest of the charges' longest steps at the scales `scales`, from their unit steps
/// `units`, and the tolerance that holds it: on one thread, one charge after another; spread,
/// by pieces, which find the same, as `least` is a total order.
fn shortest(units: &[(f64, f64)], scales: (f64, f64), inside: Option<Inside>) -> (f64, ErrorBound) {
    let none = (f64::INFINITY, ErrorBound::Charge);
    if inside.is_none() {
        return units
            .iter()
            .map(|&unit| longest_at(unit, scales))
            .fold(none, least);
    }
    pieces(units.len(), inside, |range| {
        units[range]
            .iter()
            .map(|&unit| longest_at(unit, scales))
            .fold(none, least)
    })
    .into_iter()
    .fold(none, least)
}

/// The shorter of two charges' longest steps, and the tolerance that holds it: the least by
/// length, then by tolerance, so that the same is found however the pieces split the charges.
fn least(a: (f64, ErrorBound), b: (f64, ErrorBound)) -> (f64, ErrorBound) {
    if b.0.total_cmp(&a.0).then(b.1.cmp(&a.1)).is_lt() {
        b
    } else {
        a
    }
}

#[cfg(test)]
mod tests {
    use jolty_numerics::integration::{Formula, Method};
    use jolty_numerics::reduce;
    use rayon::ThreadPoolBuilder;

    use super::{Breakpoints, LeakAccumulator, neumaier_add};
    use crate::options::Inside;

    /// The bits of a breakpoint and of the gap after it, to compare them exactly.
    fn bits((next, gap): (f64, f64)) -> (u64, u64) {
        (next.to_bits(), gap.to_bits())
    }

    /// The raw IEEE-754 bits of every value in `values`.
    fn slice_bits(values: &[f64]) -> Vec<u64> {
        values.iter().copied().map(f64::to_bits).collect()
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

    #[test]
    fn neumaier_preserves_small_terms_across_large_cancellation() {
        let (mut total, mut compensation) = (0.0, 0.0);
        for term in [1.0, 1e-16, -1.0, 2e-16] {
            neumaier_add(&mut total, &mut compensation, term);
        }
        let sum = total + compensation;
        assert!((sum - 3e-16).abs() <= 1e-30, "got {sum:e}");
    }

    #[test]
    fn leak_accumulator_matches_serial_and_parallel_bit_for_bit() {
        let unknowns = 2 * reduce::CHUNK + 17;
        let run = |inside: Option<Inside>| {
            let mut accumulator = LeakAccumulator::new(unknowns);
            let steps = [
                (Method::BackwardEuler, 1e-6, 1e-6, false),
                (Method::BackwardEuler, 1.5e-6, 1e-6, false),
                (Method::Trapezoidal, 2e-6, 1.5e-6, false),
                (Method::Trapezoidal, 0.5e-6, 2e-6, true),
                (Method::BackwardEuler, 0.2e-6, 0.5e-6, false),
                (Method::BackwardEuler, 0.4e-6, 0.2e-6, false),
                (Method::Gear2, 0.8e-6, 0.4e-6, false),
            ];
            for (step_index, &(method, length, previous, restarted)) in steps.iter().enumerate() {
                let formula = Formula::new(method, length, previous);
                let sign = if step_index % 2 == 0 { 1.0 } else { -1.0 };
                let residuals: Vec<f64> = (0..unknowns)
                    .map(|index| {
                        let scale = f64::from(u32::try_from(index + 1).expect("small index"));
                        sign * 1e-12 * scale
                    })
                    .collect();
                accumulator.accumulate(&formula, &residuals, restarted, inside);
            }
            accumulator.finish()
        };
        let serial = run(None);
        assert_eq!(serial.signed.len(), unknowns);
        assert_eq!(serial.absolute.len(), unknowns);
        for (&signed, &absolute) in serial.signed.iter().zip(&serial.absolute) {
            assert!(absolute >= 0.0);
            assert!(signed.abs() <= absolute);
        }
        let pool = ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("a pool");
        let parallel = pool.install(|| run(Some(Inside { chunks: 1 })));
        assert_eq!(slice_bits(&parallel.signed), slice_bits(&serial.signed));
        assert_eq!(slice_bits(&parallel.absolute), slice_bits(&serial.absolute));
    }
}
