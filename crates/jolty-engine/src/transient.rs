//! The transient analysis: the circuit's response over time.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use jolty_models::{Input, LogicLevel, Past};
use jolty_numerics::Pattern;
use jolty_numerics::integration::{
    CONSTANT_STEP, ErrorBound, Formula, Method, divided_difference, longest_at,
};
use serde::Deserialize;

use crate::derivative::Forward;
use crate::logic::{Checkpoint, Command, Crossing, crossing};
use crate::newton::{
    Failure, Mode, NAMED_UNKNOWNS, Workspace, add_product, each_chunk, pieces, step_matrix,
};
use crate::options::{UnknownGrain, float_count};
use crate::periodic::Monodromy;
use crate::profile::Activity;
use crate::record::{Read, Recorder};
use crate::ringing;
use crate::solver::Factors;
use crate::{
    Options, Profile, RecordedSeries, RecordedValues, Rejection, SimulationError, Simulator,
    Strategy, Work,
};

/// Default step-count divisor setting `max_step = stop / 50`, matching SPICE's default
/// longest step (source: ngspice 47 `traninit.c`, lines 26 to 34).
pub(crate) const STEPS_PER_RUN: f64 = 50.0;
/// Initial step size after time zero or a breakpoint as a fraction of the shorter of the
/// prior step and the gap to the next breakpoint.
///
/// Source: ngspice 47 (`src/spicelib/analysis/dctran.c`, lines 507 to 508:
/// `MIN(ckt->CKTdelta, .1 * MIN(ckt->CKTsaveDelta, ckt->CKTbreaks[1] - ckt->CKTbreaks[0]))`).
const RESTART_FRACTION: f64 = 0.1;
/// Minimum distinguishable time step in IEEE-754 binary64 machine epsilons (`16 ε ≈ 3.55e-15`),
/// scaled by `max(|time|, scale)` via [`minimum_step`].
///
/// Derivation: advancing from `time` requires `time + step > time` (more than 1 ULP, i.e.
/// `f64::EPSILON * |time|`), and divided-difference denominators in truncation-error estimation
/// lose precision as `step` approaches rounding. Four guard bits (`2⁴ = 16`) above 1 ULP
/// guarantee distinct IEEE-754 timestamps while retaining at least 48 bits of precision in
/// step differences.
///
/// Source: floating-point step-size floor proportional to `ε * |t|` in Shampine and Reichelt
/// (*SIAM Journal on Scientific Computing* 18(1), 1997, §2) and Hairer, Nørsett, and Wanner
/// (*Solving Ordinary Differential Equations I*, 2nd ed., 1993, §II.4).
const MIN_STEP_ULPS: f64 = 16.0;
/// Factor reducing the step size after a Newton convergence failure (`1 / 8 = 0.125`, exact
/// in binary floating point).
///
/// Source: ngspice 47 (`src/spicelib/analysis/dctran.c`, line 733:
/// `ckt->CKTdelta = ckt->CKTdelta/8;`).
const FAILURE_SHRINK: f64 = 0.125;
/// Lower and upper clamping ratios applied to truncation-error step-size updates.
///
/// Source: upper bound `2.0` matches ngspice 47 (`src/spicelib/analysis/ckttrunc.c`, line 53);
/// lower bound `0.1` follows standard ODE step clamping (Hairer, Nørsett, and Wanner,
/// *Solving Ordinary Differential Equations I*, 2nd ed., 1993, II.4).
const FACTOR_BOUNDS: (f64, f64) = (0.1, 2.0);
/// Safety multiplier applied to predicted truncation-error step lengths.
///
/// Source: standard ODE safety factor `fac = 0.9` (Hairer, Nørsett, and Wanner, 1993, II.4;
/// Hairer and Wanner, 1996, IV.8), matching ngspice 47's acceptance threshold in `dctran.c`
/// (line 801).
const SAFETY: f64 = 0.9;
/// Maximum accepted points retained for local truncation error estimation of second-order
/// formulas.
///
/// Derivation: a second-order formula (`p = 2`, trapezoidal or Gear2) evaluates the
/// `(p + 1) = 3`rd divided difference across `p + 2 = 4` points (3 prior points plus the
/// candidate step).
const HISTORY: usize = 3;
/// Number of prior charge derivatives retained alongside the latest two to detect
/// trapezoidal ringing.
///
/// Derivation: detecting two consecutive slope sign reversals (`turns[0] * turns[1] < 0` and
/// `turns[1] * turns[2] < 0`) requires 4 consecutive derivatives, storing 2 older values
/// alongside `history.derivative` and `newest`.
const RINGING: usize = 2;

/// Configuration for a transient simulation from time zero, starting either from the DC
/// operating point or directly from initial conditions.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Transient {
    /// End time of the simulation in seconds.
    pub stop: f64,
    /// Maximum allowed step length in seconds; defaults to `stop / 50`.
    #[serde(default)]
    pub max_step: Option<f64>,
    /// Integration method used between breakpoints; defaults to trapezoidal.
    #[serde(default, with = "crate::MethodSerde")]
    pub method: Method,
    /// Additional times in seconds that the solver hits exactly without restarting
    /// integration; empty by default.
    #[serde(default)]
    pub breakpoints: Vec<f64>,
    /// Per-net starting voltages in volts (such as `amplifier/mid`), analogous to SPICE `.ic`.
    ///
    /// Forced during the initial operating-point solve and released afterward, or applied
    /// directly when `operating_point` is false. Nets fixed by voltage sources or inductors
    /// remain unconstrained during the solve and are verified against the given values within
    /// tolerance.
    #[serde(default)]
    pub initial_conditions: BTreeMap<String, f64>,
    /// Flag selecting whether to compute an initial DC operating point (`true` by default).
    ///
    /// When false, the transient begins directly from `initial_conditions` with all other
    /// unknowns zeroed (like SPICE `UIC`), so the starting state need not satisfy DC KCL.
    #[serde(default = "from_operating_point")]
    pub operating_point: bool,
}

/// Returns the default `operating_point` flag (`true`) for deserializing [`Transient`].
const fn from_operating_point() -> bool {
    true
}

/// Time-domain waveforms and diagnostics produced by a transient analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct TransientResult {
    /// Accepted simulation time points in seconds from zero to `stop`.
    pub times: Vec<f64>,
    /// Unknown vectors at each accepted time point.
    pub values: Vec<Vec<f64>>,
    /// Terminal currents (including capacitive `dq/dt` from the step formula), charges,
    /// device derivatives, and operating variables requested by options.
    pub recorded: RecordedSeries,
    /// Event trace `(time_seconds, level)` for each logic net, beginning with its time-zero
    /// level.
    pub logic: Vec<Vec<(f64, LogicLevel)>>,
    /// Step counts, solver work, optional profiling timings, and integrated residual leak.
    pub statistics: Statistics,
}

/// Converged equation residuals integrated over accepted transient steps, indexed in unknown
/// order (Kundert and Clifford, "Achieving Accurate Results with a Circuit Simulator", IEEE
/// CICC 1993).
///
/// Electrical nodes and displacements carry coulombs (`A·s`, spurious charge from nonzero
/// KCL residuals), branch current equations carry webers (`V·s`), thermal nets carry joules
/// (`W·s`), and state equations carry their native units.
///
/// Integrating `r_n = f(x_n, t_n) + q̇_n` with the step's
/// [`Formula`](jolty_numerics::integration::Formula),
/// `Δℓ_n = (r_n - γ r_{n-1} + β₁ Δℓ_{n-1}) / α`, splits each step's discrete charge change
/// into integrated static currents plus `Δℓ_n`. On any closed node or cutset with zero net
/// static current, total charge drift equals the sum of `signed` to floating-point rounding,
/// and intermediate charge excursions stay bounded by the sum of `absolute`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ChargeLeak {
    /// Signed residual integral `∑ Δℓ_n` per unknown, capturing net spurious charge, flux,
    /// or energy injected over the transient.
    pub signed: Vec<f64>,
    /// Cumulative absolute integral `∑ |Δℓ_n|` per unknown, bounding peak intermediate
    /// excursion without cancellation.
    pub absolute: Vec<f64>,
}

/// Summary metrics for a transient simulation.
#[derive(Debug, Clone, PartialEq)]
pub struct Statistics {
    /// Strategy that found the initial operating point, or `None` when starting directly from
    /// initial conditions.
    pub initial: Option<Strategy>,
    /// Count of accepted time steps.
    pub accepted: usize,
    /// Count of steps rejected due to Newton failure or excessive truncation error.
    pub rejected: usize,
    /// Count of steps shortened to land accurately on a threshold crossing, logic driver
    /// transition, or device jump.
    pub landed: usize,
    /// Total Newton iterations across accepted steps.
    pub iterations: usize,
    /// Cumulative solver work across the run, including the initial operating point and all
    /// shooting periods for periodic steady state.
    ///
    /// Unlike simulation waveforms, solver work reflects prior cache reuse and retained sparse
    /// pivots.
    pub work: Work,
    /// Phase timing breakdown populated when `options.profile` is enabled.
    pub profile: Option<Profile>,
    /// Converged equation residuals integrated over all accepted transient steps.
    pub charge_leak: ChargeLeak,
}

impl Statistics {
    /// Initializes zeroed transient statistics with the operating-point `initial` strategy.
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

/// Per-unknown residual-integral accumulator packed into a single cache line per unknown
/// for bit-identical serial and parallel updates.
struct LeakAccumulator {
    states: Vec<UnknownLeak>,
}

impl LeakAccumulator {
    /// Initializes zeroed accumulators for `unknowns` equations.
    fn new(unknowns: usize) -> Self {
        Self {
            states: vec![UnknownLeak::default(); unknowns],
        }
    }

    /// Accumulates one accepted step's `residual` using `step.formula`, clearing prior
    /// history when `step.restarted` marks a corner and splitting work by `grain`.
    fn accumulate(&mut self, step: StepFormula, residual: &[f64], grain: Option<UnknownGrain>) {
        each_chunk(&mut self.states, grain, |start, chunk| {
            for (state, &current_residual) in chunk.iter_mut().zip(&residual[start..]) {
                let increment = step.formula.integral_increment(
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
                if step.restarted {
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

/// A time point and its borrowed slice of unknown values or charges.
#[derive(Clone, Copy)]
pub(crate) struct StepPoint<'a> {
    pub(crate) time: f64,
    pub(crate) values: &'a [f64],
}

/// One accepted transient step's integration formula and whether integration restarts after it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct StepFormula {
    pub(crate) formula: Formula,
    pub(crate) restarted: bool,
}

/// Linearized capacitance matrix and step context for advancing a [`Tangent`].
#[derive(Clone, Copy)]
pub(crate) struct TangentLinear<'a> {
    pub(crate) pattern: &'a Pattern,
    pub(crate) capacitance: &'a [f64],
    pub(crate) step: StepFormula,
    pub(crate) grain: Option<UnknownGrain>,
}

/// Auxiliary state propagated alongside accepted transient steps.
pub(crate) enum Carried<'a> {
    /// No auxiliary state.
    Nothing,
    /// Per-parameter forward tangents for transient sensitivity analysis.
    Forward(&'a mut Forward),
    /// State-transition and optional oscillator-period tangents for periodic steady state.
    Monodromy(&'a mut Monodromy),
    /// Ordered step formulas and restart flags recorded for a reverse adjoint pass.
    Steps(&'a mut Vec<StepFormula>),
}

/// Charge-derivative history carried from step to step for a single differentiated variable
/// during direct-method sensitivity or monodromy integration.
pub(crate) struct Tangent {
    /// Charge derivative `dq/dp` at the most recently accepted point.
    last: Vec<f64>,
    /// Charge derivative `dq/dp` at the preceding point, or `None` after an integration restart.
    before: Option<Vec<f64>>,
    /// Time derivative `d(dq/dt)/dp` at the most recently accepted point.
    rate: Vec<f64>,
}

impl Tangent {
    /// Initializes a tangent at rest from initial charge derivatives `charge`.
    pub(crate) fn new(charge: Vec<f64>) -> Self {
        let rate = vec![0.0; charge.len()];
        Self {
            last: charge,
            before: None,
            rate,
        }
    }

    /// Computes the step-scaled charge term `α q_{n+1} + β₀ q_n + β₁ q_{n-1}` of `formula`
    /// (excluding the prior-rate term `γ q̇_n`) when tracking raw charges as a tangent.
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

    /// Computes the historical contribution to `d(dq/dt)/dp` under `formula`.
    pub(crate) fn history(&self, formula: &Formula) -> Vec<f64> {
        let before = self.before.as_ref().unwrap_or(&self.last);
        self.last
            .iter()
            .zip(before)
            .zip(&self.rate)
            .map(|((&last, &before), &rate)| formula.history([last, before], rate))
            .collect()
    }

    /// Advances the tangent to the next accepted point with charge derivatives `charge` and
    /// precomputed `history`, clearing two-step history when `step.restarted` is true.
    pub(crate) fn advance(&mut self, charge: Vec<f64>, history: &[f64], step: StepFormula) {
        self.rate = charge
            .iter()
            .zip(history)
            .map(|(value, term)| step.formula.alpha * value + term)
            .collect();
        let last = std::mem::replace(&mut self.last, charge);
        self.before = (!step.restarted).then_some(last);
    }

    /// Solves `(G + αC) dx = rhs` with `factors`, adds `C dx` into `charge`, advances this
    /// tangent with `history` and `linear.step`, and returns `dx`.
    pub(crate) fn solve_step(
        &mut self,
        factors: &mut Factors<'_, f64>,
        linear: TangentLinear<'_>,
        mut rhs: Vec<f64>,
        history: &[f64],
        mut charge: Vec<f64>,
    ) -> Result<Vec<f64>, SimulationError> {
        factors.solve(&mut rhs)?;
        add_product(
            linear.pattern,
            linear.capacitance,
            &rhs,
            &mut charge,
            linear.grain,
        );
        self.advance(charge, history, linear.step);
        Ok(rhs)
    }

    /// Advances a plain charge-history tangent using its formula history.
    pub(crate) fn advance_plain(&mut self, charge: Vec<f64>, step: StepFormula) {
        let history = self.history(&step.formula);
        self.advance(charge, &history, step);
    }
}

/// Indicator for whether an accepted step extends or resets the trapezoidal ringing window.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RingingTrack {
    Trapezoidal,
    Reset,
}

/// Accepted charge samples since the most recent breakpoint together with the latest charge
/// derivative and ringing-detection window.
struct History {
    /// Accepted points prior to `last`, ordered oldest first.
    earlier: VecDeque<(f64, Vec<f64>)>,
    /// Most recently accepted `(time, charge)` pair.
    last: (f64, Vec<f64>),
    /// Charge derivative `dq/dt` at the most recently accepted point.
    derivative: Vec<f64>,
    /// Consecutive prior trapezoidal derivatives ordered oldest first for ringing detection.
    ringing: VecDeque<Vec<f64>>,
}

impl History {
    /// Clears prior history and restarts from `(time, charge, derivative)` after a breakpoint.
    fn restart(&mut self, time: f64, charge: Vec<f64>, derivative: Vec<f64>) {
        self.earlier.clear();
        self.ringing.clear();
        self.last = (time, charge);
        self.derivative = derivative;
    }

    /// Appends an accepted point at `time` while retaining at most [`HISTORY`] points and
    /// updating the ringing window according to `tracking`.
    fn accept(
        &mut self,
        time: f64,
        charge: Vec<f64>,
        derivative: Vec<f64>,
        tracking: RingingTrack,
    ) {
        let previous = std::mem::replace(&mut self.last, (time, charge));
        if self.earlier.len() == HISTORY - 1 {
            self.earlier.pop_front();
        }
        self.earlier.push_back(previous);
        let before = std::mem::replace(&mut self.derivative, derivative);
        match tracking {
            RingingTrack::Trapezoidal => {
                if self.ringing.len() == RINGING {
                    self.ringing.pop_front();
                }
                self.ringing.push_back(before);
            }
            RingingTrack::Reset => self.ringing.clear(),
        }
    }

    /// Returns the number of stored charge points.
    fn len(&self) -> usize {
        self.earlier.len() + 1
    }

    /// Iterates over stored `(time, charge)` points from oldest to newest.
    fn points(&self) -> impl Iterator<Item = &(f64, Vec<f64>)> {
        self.earlier.iter().chain(std::iter::once(&self.last))
    }

    /// Returns the charge vectors at the last two accepted points, duplicating the latest
    /// after a restart.
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

/// Returns the smallest distinguishable time step at `time` given initial time scale `scale`.
///
/// Scales [`MIN_STEP_ULPS`] machine epsilons by `max(|time|, scale)` so `time + step > time`
/// holds with four guard bits at every timestamp while remaining strictly positive at time zero.
fn minimum_step(time: f64, scale: f64) -> f64 {
    MIN_STEP_ULPS * f64::EPSILON * time.abs().max(scale)
}

/// Ordered queue of upcoming simulation breakpoints ending at the transient stop time.
struct Breakpoints {
    /// Non-negative breakpoint times encoded as IEEE-754 bits for `O(log n)` neighbor lookup
    /// and ordering, with `stop` always last.
    times: BTreeSet<u64>,
    /// Initial time scale anchoring [`minimum_step`] for merging near-duplicate breakpoints.
    scale: f64,
}

impl Breakpoints {
    /// Constructs a breakpoint set from sorted `times` with initial time scale `scale`.
    fn new(times: &[f64], scale: f64) -> Self {
        Self {
            times: times.iter().map(|&time| Self::key(time)).collect(),
            scale,
        }
    }

    /// Encodes a non-negative `time` as canonical IEEE-754 bits, normalizing `-0.0` to `+0.0`.
    fn key(time: f64) -> u64 {
        (time + 0.0).to_bits()
    }

    /// Returns the earliest remaining breakpoint time, or infinity if empty.
    fn first(&self) -> f64 {
        self.times
            .first()
            .map_or(f64::INFINITY, |&key| f64::from_bits(key))
    }

    /// Inserts `time` unless it exceeds the stop time or falls within
    /// `minimum_step(time, self.scale)` of an existing breakpoint.
    fn insert(&mut self, time: f64) {
        let stop = self
            .times
            .last()
            .map_or(f64::INFINITY, |&key| f64::from_bits(key));
        if time >= stop {
            return;
        }
        let key = Self::key(time);
        let minimum = minimum_step(time, self.scale);
        let near = |other: Option<&u64>| {
            other.is_some_and(|&other| (f64::from_bits(other) - time).abs() < minimum)
        };
        if !near(self.times.range(..key).next_back()) && !near(self.times.range(key..).next()) {
            self.times.insert(key);
        }
    }

    /// Discards breakpoints at or before `time` and returns the next breakpoint and gap to
    /// the one following it.
    fn after(&mut self, time: f64) -> NextBreakpoint {
        let passed = Self::key(time);
        while self.times.len() > 1 && self.times.first().is_some_and(|&key| key <= passed) {
            self.times.pop_first();
        }
        let mut ahead = self.times.iter().map(|&key| f64::from_bits(key));
        let time = ahead.next().unwrap_or(f64::INFINITY);
        let gap = ahead.next().map_or(f64::INFINITY, |after| after - time);
        NextBreakpoint { time, gap }
    }
}

/// Next upcoming breakpoint and the time interval to the following breakpoint.
#[derive(Clone, Copy)]
struct NextBreakpoint {
    time: f64,
    gap: f64,
}

/// Outcome of attempting one transient step.
enum Step {
    /// Accepted point(s) ordered oldest first (including any previously pending corner step),
    /// paired with logic driver ramp settle times.
    Accepted(Vec<Point>, Vec<f64>),
    /// First step after a corner, deferred until the next step provides a curvature estimate.
    Pending,
    /// Rejected step to be retried with a shorter length.
    Rejected,
    /// Exact event time (detector crossing, driver change, or device jump) inside the step
    /// requiring a shortened step to land on it.
    Land(f64),
}

/// Accepted transient time point ready for output recording.
struct Point {
    time: f64,
    /// Converged unknown vector at `time`.
    x: Vec<f64>,
    /// Integration formula and restart flag of the step reaching `time`.
    step: StepFormula,
    /// Duration of the step reaching `time`.
    length: f64,
    /// Recorded device currents, charges, derivatives, and variables at `time`.
    found: RecordedValues,
}

/// Deferred first step after a corner waiting for the second step to validate its error.
///
/// Estimating local truncation error after a corner requires charge curvature across three
/// points. The following step either commits the pending step alongside itself or rewinds to
/// `corner` to retry shorter.
struct Pending {
    /// Integration formula used for the pending step.
    formula: Formula,
    /// Duration of the pending step.
    length: f64,
    /// Newton iterations required to solve the pending step.
    iterations: usize,
    /// Converged residual vector retained until the next step commits or discards the step.
    residual: Vec<f64>,
    /// Snapshot of recorded device data at the pending step endpoint.
    read: Read,
    /// Saved state at the preceding corner for rollback if the pending step is rejected.
    corner: Corner,
}

/// Saved simulator state at a corner enabling rollback of a pending first step.
struct Corner {
    time: f64,
    /// Unknown vector, charges, and charge derivatives at the corner.
    x: Vec<f64>,
    charge: Vec<f64>,
    derivative: Vec<f64>,
    /// Duration of the step that landed on the corner.
    previous: f64,
    /// Logic event checkpoint captured before advancing across the pending step.
    logic: Checkpoint,
}

/// Step geometry and endpoint charge states passed to truncation-error estimation.
struct Span<'a> {
    length: f64,
    /// Preceding step length used by multistep error constants.
    previous: f64,
    /// Leading charge coefficient `α` of the step's integration formula.
    alpha: f64,
    charge: (&'a [f64], &'a [f64]),
    derivative: (&'a [f64], &'a [f64]),
}

/// Solved candidate step awaiting truncation-error evaluation and event checks.
struct Tried {
    end: f64,
    length: f64,
    method: Method,
    formula: Formula,
    /// Newton iterations used to converge the candidate step.
    iterations: usize,
    /// Converged unknown vector, charges, and charge derivatives at `end`.
    x: Vec<f64>,
    charge: Vec<f64>,
    derivative: Vec<f64>,
    /// Snapshot of recorded device data from the final Newton evaluation.
    read: Read,
}

/// Truncation-error comparison against step tolerances.
#[derive(Clone, Copy)]
struct Judgement {
    /// Maximum ratio of estimated charge truncation error to tolerance.
    ratio: f64,
    /// Unknown index realizing `ratio`.
    worst: usize,
    /// Maximum next step length permitted by the error estimate, measured from the step end
    /// when accepted or from the step start when rejected.
    longest: f64,
}

/// Initial step-size bounds and time scale for a transient stepper.
#[derive(Clone, Copy)]
struct StepLimits {
    longest: f64,
    scale: f64,
    first: f64,
}

/// Control-flow and logic outputs of a judged step handed to [`Simulator::take`].
struct StepTransition {
    next: f64,
    restart: Option<f64>,
    commands: Vec<Command>,
    logic: Option<Checkpoint>,
}

/// Accepted step data recorded into a [`Point`].
struct AcceptedStep<'a> {
    time: f64,
    x: Vec<f64>,
    step: StepFormula,
    length: f64,
    iterations: usize,
    read: &'a Read,
}

/// Candidate end-of-step charges and derivatives tested for trapezoidal ringing.
#[derive(Clone, Copy)]
struct CandidatePoint<'a> {
    time: f64,
    charge: &'a [f64],
    derivative: &'a [f64],
}

/// Mutable state maintained by the transient time stepper between steps.
struct Stepper {
    /// Integration method used between breakpoints.
    method: Method,
    /// Maximum allowed step duration.
    longest: f64,
    /// Initial time scale anchoring [`minimum_step`] at time zero.
    scale: f64,
    /// Timestamp of the last accepted point.
    time: f64,
    /// Solution vector at the last accepted point.
    x: Vec<f64>,
    /// Final finite iterate from a diverged Newton attempt reused as the initial guess for
    /// the shortened retry step, preserving progress on slowly converging turn-off nodes.
    resume: Option<Vec<f64>>,
    /// Preceding accepted `(time, solution)` pair used for linear step prediction; `None`
    /// immediately after a breakpoint, driver ramp start, or device jump.
    before: Option<(f64, Vec<f64>)>,
    history: History,
    /// Error-controlled length of the next step, preserved across shortened breakpoint or
    /// event landings.
    next: f64,
    /// Duration of the previous step.
    previous: f64,
    /// Flag requesting a single backward Euler step to damp detected trapezoidal ringing.
    damp: bool,
    /// Per-unknown integration history terms for the active step.
    terms: Vec<f64>,
    /// Deferred first step after a corner while awaiting validation from the next step.
    pending: Option<Pending>,
    /// Stateful trapezoidal ringing detector caching ideal sources and saddle LU factors.
    detector: ringing::Detector,
    /// Running residual integral over all accepted steps.
    leak: LeakAccumulator,
    statistics: Statistics,
}

impl Stepper {
    /// Starts a transient stepper at time zero with initial solution `x` and `charge`.
    fn starting(
        method: Method,
        limits: StepLimits,
        x: Vec<f64>,
        charge: Vec<f64>,
        strategy: Option<Strategy>,
    ) -> Self {
        let unknowns = x.len();
        Self {
            method,
            longest: limits.longest,
            scale: limits.scale,
            time: 0.0,
            resume: None,
            before: None,
            history: History {
                earlier: VecDeque::with_capacity(HISTORY - 1),
                last: (0.0, charge),
                derivative: vec![0.0; unknowns],
                ringing: VecDeque::with_capacity(RINGING),
            },
            next: limits.first,
            previous: limits.first,
            damp: false,
            terms: vec![0.0; unknowns],
            pending: None,
            detector: ringing::Detector::default(),
            leak: LeakAccumulator::new(unknowns),
            statistics: Statistics::starting(strategy),
            x,
        }
    }

    /// Computes the SPICE-style linear predictor at `end` from the last two accepted points,
    /// falling back to `self.x` after a corner, parallelized by `grain`.
    fn predicted(&self, end: f64, grain: Option<UnknownGrain>) -> Vec<f64> {
        let Some((time, before)) = &self.before else {
            return self.x.clone();
        };
        let ratio = (end - self.time) / (self.time - time);
        let mut predicted = vec![0.0; self.x.len()];
        each_chunk(&mut predicted, grain, |start, chunk| {
            let points = self.x[start..].iter().zip(&before[start..]);
            for (value, (&last, &earlier)) in chunk.iter_mut().zip(points) {
                *value = last + (last - earlier) * ratio;
            }
        });
        predicted
    }

    /// Chooses the integration formula for a step of `length` and writes each unknown's
    /// history term into `self.terms`, parallelized by `grain`.
    fn prepare_formula(&mut self, length: f64, grain: Option<UnknownGrain>) -> (Method, Formula) {
        // Use backward Euler until enough post-corner points exist to evaluate the target
        // method's error estimate.
        let available = self.history.len() + 1;
        let method = if self.damp || available < self.method.order() + 2 {
            Method::BackwardEuler
        } else {
            self.method
        };
        let formula = Formula::new(method, length, self.previous);
        let (last, before) = self.history.last_two();
        let derivative = &self.history.derivative;
        each_chunk(&mut self.terms, grain, |start, chunk| {
            for (offset, term) in chunk.iter_mut().enumerate() {
                let index = start + offset;
                *term = formula.history([last[index], before[index]], derivative[index]);
            }
        });
        (method, formula)
    }

    /// Returns the smallest distinguishable time step at `time`.
    fn minimum_step_at(&self, time: f64) -> f64 {
        minimum_step(time, self.scale)
    }

    /// Returns the smallest distinguishable time step at the current simulation time.
    fn shortest(&self) -> f64 {
        self.minimum_step_at(self.time)
    }

    /// Checks whether an event at `time` falls early enough inside `(self.time, end)` to
    /// warrant shortening the step to land on it accurately.
    fn lands_before(&self, time: f64, end: f64, options: &Options) -> bool {
        let tolerance = self
            .minimum_step_at(end)
            .max(options.relative_tolerance * (end - self.time));
        time < end - tolerance && time - self.time >= self.shortest()
    }

    /// Records a step rejection with reason `why` and updates `self.next` to `next`, returning
    /// [`SimulationError::StepTooSmall`] if `next` drops below [`Self::shortest`].
    fn reject(
        &mut self,
        next: f64,
        why: impl FnOnce() -> Rejection,
    ) -> Result<Step, SimulationError> {
        self.statistics.rejected += 1;
        self.next = next;
        let shortest = self.shortest();
        if next < shortest {
            return Err(SimulationError::StepTooSmall {
                time: self.time,
                minimum: shortest,
                rejection: why(),
            });
        }
        Ok(Step::Rejected)
    }
}

impl Simulator {
    /// Simulates the circuit in the time domain from time zero until `transient.stop`.
    ///
    /// Adaptive steps are solved with Newton iteration and controlled by local truncation
    /// error. Integration lands on source breakpoints and restarts with backward Euler after
    /// each corner to prevent trapezoidal ringing.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::InvalidAnalysis`] for non-positive or non-finite stop times
    /// or step limits, [`SimulationError::StepTooSmall`] if step control hits the minimum step
    /// size, and any error from [`Simulator::operating_point`].
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

    /// Computes the initial state vector and optional operating-point strategy for `transient`.
    ///
    /// Solves the operating point (forcing `initial_conditions` if present) when
    /// `transient.operating_point` is true, or populates `initial_conditions` over zeros
    /// otherwise.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::InvalidAnalysis`] for invalid or conflicting initial
    /// conditions or when skipping the operating point on a mixed-signal circuit, plus any
    /// error from [`Simulator::operating_point`].
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

    /// Validates `transient` timing settings and returns the effective maximum step duration.
    pub(crate) fn longest_step(&self, transient: &Transient) -> Result<f64, SimulationError> {
        // Pure-logic circuits need no analog stepping and advance directly to `stop`.
        let longest = if self.elaborated.unknowns.is_empty() {
            transient.stop
        } else {
            transient.max_step.unwrap_or(transient.stop / STEPS_PER_RUN)
        };
        if !(transient.stop > 0.0
            && transient.stop.is_finite()
            && longest > 0.0
            && longest.is_finite())
        {
            return Err(SimulationError::InvalidAnalysis(format!(
                "stop time and longest step require positive finite values; got {:?} and \
                 {longest:?}",
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

    /// Initializes `carried` auxiliary state from the time-zero solution `x`.
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

    /// Advances `carried` auxiliary state across an accepted step ending at `point` with `step`.
    fn advance_carried(
        &mut self,
        carried: &mut Carried<'_>,
        point: StepPoint<'_>,
        step: StepFormula,
        options: &Options,
    ) -> Result<(), SimulationError> {
        match carried {
            Carried::Nothing => Ok(()),
            Carried::Steps(steps) => {
                steps.push(step);
                Ok(())
            }
            Carried::Forward(forward) => self.advance_forward(forward, point, step, options),
            Carried::Monodromy(monodromy) => {
                self.advance_monodromy(monodromy, point, step, options)
            }
        }
    }

    /// Returns the number of `f64` values stored per transient time point across unknowns and
    /// recorded terminal and device quantities.
    pub(crate) fn point_width(&self) -> f64 {
        float_count(self.elaborated.unknowns.len() + self.state.recorder.width())
    }

    /// Runs `transient` from an optional precomputed `start` state (or computes the initial
    /// state), propagating `carried` across accepted steps and clearing delay-line history on
    /// exit.
    pub(crate) fn transient_with(
        &mut self,
        transient: &Transient,
        options: &Options,
        start: Option<(Vec<f64>, Option<Strategy>)>,
        carried: Carried<'_>,
    ) -> Result<TransientResult, SimulationError> {
        let result = self.run_transient(transient, options, start, carried);
        self.state.past = None;
        result
    }

    /// Executes the transient integration for [`Simulator::transient_with`]; the caller
    /// clears `self.state.past` on both success and early error returns.
    fn run_transient(
        &mut self,
        transient: &Transient,
        options: &Options,
        start: Option<(Vec<f64>, Option<Strategy>)>,
        mut carried: Carried<'_>,
    ) -> Result<TransientResult, SimulationError> {
        let reading = self.read_meter(options.profile);
        self.prepare(options)?;
        let longest = self.longest_step(transient)?;
        let width = self.point_width();
        options.hold_points(transient.stop / longest * width, 0.0, longest)?;
        let mut held = width;
        let mut requested = transient.breakpoints.clone();
        requested.sort_by(f64::total_cmp);
        let (initial, strategy) =
            start.map_or_else(|| self.initial_state(transient, options), Ok)?;
        self.start_carried(&mut carried, &initial, options)?;
        let delays = self.start_past(&initial, transient.method);
        self.evaluate(&initial, Some(0.0), 1.0, options.gmin, true);
        let start = self.read_recorded();
        self.state
            .events
            .start(&self.elaborated.logic, transient.stop);
        let (breakpoint_times, scale) = self.breakpoints(transient.stop, longest, &delays);
        let mut breakpoints = Breakpoints::new(&breakpoint_times, scale);
        let first = RESTART_FRACTION * longest.min(breakpoints.first());
        let charge = self.state.workspace.contributions.charge.clone();
        let mut stepper = Stepper::starting(
            transient.method,
            StepLimits {
                longest,
                scale,
                first,
            },
            initial,
            charge,
            strategy,
        );
        let (mut times, mut values) = (vec![0.0], vec![stepper.x.clone()]);
        let mut recorded = RecordedSeries::default();
        recorded.push(&self.record(|recorder| recorder.start(&start)));
        while stepper.time < transient.stop {
            if let Some(change) = self.state.events.next_analog_change(&self.elaborated.logic) {
                breakpoints.insert(change);
            }
            let breakpoint = breakpoints.after(stepper.time);
            let next_requested = stepper.time + stepper.shortest();
            let passed = requested.partition_point(|&time| time < next_requested);
            let requested_time = requested.get(passed).copied().unwrap_or(f64::INFINITY);
            match self.step_toward(&mut stepper, breakpoint, requested_time, options)? {
                Step::Accepted(points, settles) => {
                    for point in points {
                        let at = StepPoint {
                            time: point.time,
                            values: &point.x,
                        };
                        self.advance_carried(&mut carried, at, point.step, options)?;
                        self.watch_supplies(&point.x, point.time, options);
                        held += width;
                        options.hold_points(held, point.time, point.length)?;
                        self.passed(&mut breakpoints, point.time, point.step, &delays);
                        times.push(point.time);
                        values.push(point.x);
                        recorded.push(&point.found);
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
        statistics.charge_leak = stepper.leak.finish();
        (statistics.work, statistics.profile) = self.metered_since(&reading);
        Ok(TransientResult {
            times,
            values,
            recorded,
            logic: self.state.events.traces().to_vec(),
            statistics,
        })
    }

    /// Initializes transmission-line history from state `x` at time zero with `method`'s
    /// interpolation order, returning sorted distinct line delays.
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

    /// Prunes delay-line history older than the maximum delay before `time` and schedules
    /// delayed corner breakpoints when `step.restarted` is true.
    fn passed(
        &mut self,
        breakpoints: &mut Breakpoints,
        time: f64,
        step: StepFormula,
        delays: &[f64],
    ) {
        let (Some(past), Some(&longest)) = (&mut self.state.past, delays.last()) else {
            return;
        };
        past.forget_before(time - longest);
        if step.restarted {
            for &delay in delays {
                breakpoints.insert(time + delay);
            }
        }
    }

    /// Executes `read` against the simulator's [`Recorder`] while charging elapsed time to
    /// [`Activity::Recording`].
    fn record(&mut self, read: impl FnOnce(&mut Recorder) -> RecordedValues) -> RecordedValues {
        let previous = self.state.workspace.meter.enter(Activity::Recording);
        let found = read(&mut self.state.recorder);
        self.state.workspace.meter.leave(previous);
        found
    }

    /// Attempts a single step toward `breakpoint` or earlier `requested_time`, landing on
    /// whichever comes first.
    ///
    /// Accepted steps advance logic state through the step interval. Early detector crossings,
    /// logic driver transitions, or device jumps reject the step to land on the event time,
    /// restarting integration after any waveform corner.
    fn step_toward(
        &mut self,
        stepper: &mut Stepper,
        breakpoint: NextBreakpoint,
        requested_time: f64,
        options: &Options,
    ) -> Result<Step, SimulationError> {
        let target = breakpoint.time.min(requested_time);
        let remaining = target - stepper.time;
        // Clamp next step length to `stepper.longest` regardless of target landing.
        let reach = stepper.next.min(stepper.longest);
        let landing = reach >= remaining - stepper.minimum_step_at(target);
        let length = if landing { remaining } else { reach };
        let end = if landing {
            target
        } else {
            stepper.time + length
        };
        // Only source/event breakpoints create waveform corners; user-requested times do not.
        let corner = landing && target >= breakpoint.time;
        // Partition per-unknown loops identically to Newton iteration.
        let grain = options.parallelism.unknown_grain(stepper.x.len());
        let (method, formula) = stepper.prepare_formula(length, grain);

        let mut candidate = stepper
            .resume
            .take()
            .unwrap_or_else(|| stepper.predicted(end, grain));
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
        each_chunk(&mut derivative, grain, |start, chunk| {
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
        let end_point = StepPoint {
            time: end,
            values: &tried.x,
        };
        let events = self.events_within(stepper, end_point, options);
        self.state.workspace.meter.leave(previous);
        let (commands, jumped) = match events {
            Ok(found) => found,
            Err(early) => {
                // Events land at least `shortest` ahead, so the next shortened step hits `early`.
                stepper.statistics.landed += 1;
                return Ok(Step::Land(early));
            }
        };
        // A breakpoint, a driver's ramp starting or a device's jump is a corner in the
        // waveforms. The step after it is a tenth of the step the error allowed before it,
        // as SPICE's saved step, or of the gap to the next breakpoint if shorter.
        let restart = (corner || !commands.is_empty() || jumped).then(|| {
            let gap = if corner {
                breakpoint.gap
            } else {
                breakpoint.time - end
            };
            (RESTART_FRACTION * reach.min(gap)).max(stepper.minimum_step_at(end))
        });
        let transition = StepTransition {
            next,
            restart,
            commands,
            logic,
        };
        Ok(self.take(stepper, tried, transition, options))
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
        let new = StepPoint {
            time: tried.end,
            values: &tried.charge,
        };
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
        // Defer error estimation on the first post-corner step until the second step runs.
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

    /// Takes the step `tried` along `transition`: accepts it, with the pending first step
    /// after the last corner, or holds it as that first step.
    fn take(
        &mut self,
        stepper: &mut Stepper,
        tried: Tried,
        transition: StepTransition,
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
        let grain = options.parallelism.unknown_grain(stepper.x.len());
        let mut points = Vec::with_capacity(2);
        if let Some(pending) = stepper.pending.take() {
            // The first step after the corner, its error now estimated within tolerance.
            points.push(self.accept_pending(stepper, &pending, grain));
        }
        let last = (stepper.time, std::mem::replace(&mut stepper.x, x));
        let settles = self.follow(&transition.commands);
        if transition.restart.is_none()
            && let Some(logic) = transition.logic
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
            let step = StepFormula {
                formula,
                restarted: transition.restart.is_some(),
            };
            stepper
                .leak
                .accumulate(step, &self.state.workspace.contributions.residual, grain);
            points.push(self.accept(
                stepper,
                AcceptedStep {
                    time: end,
                    x: stepper.x.clone(),
                    step,
                    length,
                    iterations,
                    read: &read,
                },
            ));
        }
        stepper.previous = length;
        stepper.time = end;
        if let Some(past) = &mut self.state.past {
            past.push(end, &stepper.x, transition.restart.is_some());
        }
        if let Some(after_corner) = transition.restart {
            stepper.before = None;
            stepper.history.restart(end, charge, derivative);
            stepper.damp = false;
            stepper.next = after_corner;
        } else {
            let candidate = CandidatePoint {
                time: end,
                charge: &charge,
                derivative: &derivative,
            };
            let (damp, tracking) =
                self.check_ringing(stepper, method, candidate, formula.alpha, options);
            stepper.damp = damp;
            stepper.before = Some(last);
            stepper.history.accept(end, charge, derivative, tracking);
            stepper.next = transition.next;
        }
        if points.is_empty() {
            Step::Pending
        } else {
            Step::Accepted(points, settles)
        }
    }

    /// Accepts `pending`, the first step after a corner, once the following step validates it.
    fn accept_pending(
        &mut self,
        stepper: &mut Stepper,
        pending: &Pending,
        grain: Option<UnknownGrain>,
    ) -> Point {
        let step = StepFormula {
            formula: pending.formula,
            restarted: false,
        };
        stepper.leak.accumulate(step, &pending.residual, grain);
        self.accept(
            stepper,
            AcceptedStep {
                time: stepper.time,
                x: stepper.x.clone(),
                step,
                length: pending.length,
                iterations: pending.iterations,
                read: &pending.read,
            },
        )
    }

    /// Evaluates trapezoidal ringing at `candidate` and returns `(damp, tracking)`.
    fn check_ringing(
        &mut self,
        stepper: &mut Stepper,
        method: Method,
        candidate: CandidatePoint<'_>,
        alpha: f64,
        options: &Options,
    ) -> (bool, RingingTrack) {
        let trapezoidal = method == Method::Trapezoidal;
        let previous = self.state.workspace.meter.enter(Activity::StepControl);
        let damp = trapezoidal
            && self.rings(
                &mut stepper.detector,
                &stepper.history,
                candidate,
                alpha,
                options,
            );
        self.state.workspace.meter.leave(previous);
        let tracking = if trapezoidal {
            RingingTrack::Trapezoidal
        } else {
            RingingTrack::Reset
        };
        (damp, tracking)
    }

    /// Accepts `accepted` into `stepper`'s statistics and records its device values.
    fn accept(&mut self, stepper: &mut Stepper, accepted: AcceptedStep<'_>) -> Point {
        stepper.statistics.accepted += 1;
        stepper.statistics.iterations += accepted.iterations;
        let found = self.record(|recorder| recorder.step(accepted.read, accepted.step));
        Point {
            time: accepted.time,
            x: accepted.x,
            step: accepted.step,
            length: accepted.length,
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

    /// Checks whether the trapezoidal rule is numerically ringing at `candidate`.
    ///
    /// Distinguishes numerical `(-1)^n` trapezoidal ringing (stiff real modes with
    /// `σ ≥ α = 2/h`, algebraic capacitor-voltage-source loops, and zero-capacitance Meyer
    /// displacements) from physical oscillations such as lossless LC tanks and active
    /// oscillators. A single backward Euler step resets a ringing derivative without damping
    /// physical oscillations.
    fn rings(
        &mut self,
        detector: &mut ringing::Detector,
        history: &History,
        candidate: CandidatePoint<'_>,
        alpha: f64,
        options: &Options,
    ) -> bool {
        if history.ringing.len() < RINGING || history.earlier.len() < RINGING {
            return false;
        }
        let window = ringing::StepWindow {
            times: [
                history.earlier[0].0,
                history.earlier[1].0,
                history.last.0,
                candidate.time,
            ],
            charges: [
                history.earlier[0].1.as_slice(),
                history.earlier[1].1.as_slice(),
                history.last.1.as_slice(),
                candidate.charge,
            ],
            derivatives: [
                history.ringing[0].as_slice(),
                history.ringing[1].as_slice(),
                history.derivative.as_slice(),
                candidate.derivative,
            ],
            alpha,
        };
        detector.rings(
            &self.elaborated.pattern,
            &self.elaborated.natures,
            &mut self.state.solver,
            &mut self.state.workspace,
            &window,
            options,
        )
    }

    /// Evaluates device jumps and mixed-signal logic events across the step ending at `end`,
    /// returning `(driver_commands, device_jumped)` or `Err(early_time)` if the step needs to
    /// shorten and land on an interior event.
    fn events_within(
        &mut self,
        stepper: &Stepper,
        end: StepPoint<'_>,
        options: &Options,
    ) -> Result<(Vec<Command>, bool), f64> {
        let jump = self.first_jump(stepper, end);
        if let Some(time) = jump
            && stepper.lands_before(time, end.time, options)
        {
            return Err(time);
        }
        let commands = self.advance_logic(stepper, end, options)?;
        // A jump within `stepper.shortest()` of `stepper.time` belongs to the corner landed on
        // by the prior step, where integration already restarted.
        let jumped = jump.is_some_and(|time| time - stepper.time >= stepper.shortest());
        Ok((commands, jumped))
    }

    /// Finds the earliest linearly interpolated discontinuity time in `(stepper.time, end]`
    /// among devices whose piecewise branch changed across the step, if any.
    fn first_jump(&mut self, stepper: &Stepper, end: StepPoint<'_>) -> Option<f64> {
        let start = stepper.time;
        let before = Input::new(&stepper.x, Some(start), 0.0);
        let later = Input::new(end.values, Some(end.time), 0.0);
        let groups = &self.elaborated.groups;
        let fraction = self.state.runs.first_jump(groups, (&before, &later))?;
        Some(start + (end.time - start) * fraction)
    }

    /// Advances the logic partition to `end` and returns driver commands scheduled at the step
    /// endpoint, or `Err(early_time)` if an interior crossing or driver change occurs earlier.
    fn advance_logic(
        &mut self,
        stepper: &Stepper,
        end: StepPoint<'_>,
        options: &Options,
    ) -> Result<Vec<Command>, f64> {
        let Self {
            elaborated, state, ..
        } = self;
        let (logic, events) = (&elaborated.logic, &mut state.events);
        let early = |time: f64| stepper.lands_before(time, end.time, options);
        let held = events.detector_levels(logic);
        let start = StepPoint {
            time: stepper.time,
            values: &stepper.x,
        };
        let crossings: Vec<Crossing> = logic
            .detectors
            .iter()
            .enumerate()
            .filter_map(|(detector, &(element, _))| {
                crossing(
                    |input| state.runs.sense(&elaborated.groups, element, input),
                    detector,
                    held[detector],
                    start,
                    end,
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
        if crossings.is_empty() && !events.due(end.time) {
            return Ok(Vec::new());
        }
        // Without drivers, logic cannot feed back into the analog circuit; with drivers,
        // checkpoint first so an interior driver change can roll logic back cleanly.
        let checkpoint = (!logic.drivers.is_empty()).then(|| events.checkpoint());
        events.inject(logic, &crossings);
        let commands = events.advance(logic, end.time);
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

    /// Starts each logic driver ramp in `commands` and returns the ramp completion times.
    fn follow(&mut self, commands: &[Command]) -> Vec<f64> {
        let mut settles = Vec::with_capacity(commands.len());
        for command in commands {
            let element = self.elaborated.logic.drivers[command.driver].0;
            settles.extend(self.drive(element, command.time, command.level, command.transition));
        }
        settles
    }

    /// Collects sorted, deduplicated device breakpoints and initial delay-line arrival times
    /// in `(0, stop]`, always ending at `stop`, returning them alongside the initial time
    /// scale `min(longest, first_breakpoint)` metered under [`Activity::Preparing`].
    fn breakpoints(&mut self, stop: f64, longest: f64, delays: &[f64]) -> (Vec<f64>, f64) {
        let previous = self.state.workspace.meter.enter(Activity::Preparing);
        let mut times = vec![stop];
        self.state.runs.breakpoints((0.0, stop), &mut times);
        times.extend(delays.iter().filter(|&&delay| delay < stop));
        times.retain(|&time| time > 0.0);
        times.sort_by(f64::total_cmp);
        let scale = longest.min(times.first().copied().unwrap_or(stop));
        times.retain(|&time| time > minimum_step(0.0, scale));
        times.dedup_by(|later, earlier| *later - *earlier < minimum_step(*earlier, scale));
        if let Some(last) = times.last_mut() {
            *last = stop;
        }
        self.state.workspace.meter.leave(previous);
        (times, scale)
    }

    /// Evaluates the local truncation error of `span` under `method` using filtered charge
    /// divided differences from `history` and `new`, returning `None` until enough post-corner
    /// points exist.
    ///
    /// Charge tolerances follow SPICE `TRTOL` scaling (`reltol * max(|q|, chgtol)` and
    /// current tolerance over the step), multiplying by `(p + 1)!` to match SPICE's factorial
    /// omission.
    fn judge(
        &mut self,
        method: Method,
        history: &History,
        new: StepPoint<'_>,
        span: &Span<'_>,
        options: &Options,
    ) -> Option<Judgement> {
        let needed = method.order() + 2;
        let available = history.len() + 1;
        if available < needed {
            return None;
        }
        // Partition per-unknown loops identically to Newton iteration.
        let grain = options.parallelism.unknown_grain(new.values.len());
        let mut differences = std::mem::take(&mut self.state.workspace.differences);
        divided_differences(history, new, needed, grain, &mut differences);
        self.filter(&mut differences, span.alpha, grain);
        let scale = options.truncation_tolerance * method.factorial();
        let natures = &self.elaborated.natures;
        // Scale charge and current tolerances by `truncation_tolerance` and the `(p + 1)!`
        // factor omitted by SPICE. Charge tolerance uses `reltol * max(|q|, chgtol)` per
        // ngspice 47 `cktterr.c` (lines 40 to 41).
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
        // Compute per-chunk worst error ratio and constant-step shortest next length, retaining
        // per-charge unit steps for multistep variable-ratio adjustment.
        let keep = method.is_multistep();
        let per_difference = method.error_per_difference(span.length, span.previous);
        let found = pieces(differences.len(), grain, |range| {
            let (mut worst, mut found) = none;
            let mut units = Vec::with_capacity(if keep { range.len() } else { 0 });
            for index in range {
                let difference = differences[index];
                // Step truncation error matching `Method::truncation_error`.
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
        // Use the accepted step as `previous` for the next step, or keep `span.previous` on
        // rejection.
        let previous = if worst.ratio > 1.0 {
            span.previous
        } else {
            span.length
        };
        let longest = method.longest_common_step(previous, shortest_at_constant, |scales| {
            shortest(&units, scales.0, scales.1, grain)
        });
        self.state.workspace.differences = differences;
        Some(Judgement { longest, ..worst })
    }

    /// Filters charge divided `differences` through the step equations via
    /// `C (G + αC)⁻¹ α d`, parallelized identically across thread counts by `grain`.
    ///
    /// Filtering preserves smooth capacitive error while attenuating stiff decaying modes
    /// already damped by the implicit step (Shampine, *SIAM Journal on Scientific and
    /// Statistical Computing* 1(1), 1980; Hairer and Wanner, *Solving Ordinary Differential
    /// Equations II*, §IV.8). Leaves `differences` unchanged if factorization fails.
    fn filter(&mut self, differences: &mut [f64], alpha: f64, grain: Option<UnknownGrain>) {
        let Workspace {
            contributions,
            jacobian,
            moved,
            meter,
            ..
        } = &mut self.state.workspace;
        step_matrix(jacobian, contributions, alpha, grain);
        moved.clear();
        moved.resize(differences.len(), 0.0);
        each_chunk(moved, grain, |start, chunk| {
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
            capacitance,
            moved,
            differences,
            grain,
        );
    }
}

/// Computes per-unknown charge divided differences over the trailing `needed` points of
/// `history` plus `new`, writing into `differences` according to `grain`.
fn divided_differences(
    history: &History,
    new: StepPoint<'_>,
    needed: usize,
    grain: Option<UnknownGrain>,
    differences: &mut Vec<f64>,
) {
    // Collect timestamps and charge slices from oldest to newest, appending `new`.
    let mut times = [0.0; HISTORY + 1];
    let mut charges: [&[f64]; HISTORY + 1] = [&[]; HISTORY + 1];
    let used = history
        .points()
        .skip(history.len() + 1 - needed)
        .map(|(time, charge)| (*time, charge.as_slice()));
    for ((time, charge), (at, held)) in times
        .iter_mut()
        .zip(&mut charges)
        .zip(used.chain([(new.time, new.values)]))
    {
        *time = at;
        *charge = held;
    }
    differences.clear();
    differences.resize(new.values.len(), 0.0);
    each_chunk(differences, grain, |start, chunk| {
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

/// Computes the clamped next step length from `length` and optional `judged`, scaling by
/// [`SAFETY`] within [`FACTOR_BOUNDS`] or growing to the upper bound when unjudged.
fn next_length(length: f64, judged: Option<Judgement>) -> f64 {
    let (shortest, longest) = (FACTOR_BOUNDS.0 * length, FACTOR_BOUNDS.1 * length);
    judged.map_or(longest, |judged| {
        (SAFETY * judged.longest).clamp(shortest, longest)
    })
}

/// Step length paired with the active [`ErrorBound`] returned by [`longest_at`].
type StepBound = (f64, ErrorBound);

/// Finds the shortest step length across all charge `units` at `(charge_scale, current_scale)`,
/// reducing deterministically via [`least`] across optional Rayon `grain`.
fn shortest(
    units: &[(f64, f64)],
    charge_scale: f64,
    current_scale: f64,
    grain: Option<UnknownGrain>,
) -> StepBound {
    let scales = (charge_scale, current_scale);
    let none = (f64::INFINITY, ErrorBound::Charge);
    if grain.is_none() {
        return units
            .iter()
            .map(|&unit| longest_at(unit, scales))
            .fold(none, least);
    }
    pieces(units.len(), grain, |range| {
        units[range]
            .iter()
            .map(|&unit| longest_at(unit, scales))
            .fold(none, least)
    })
    .into_iter()
    .fold(none, least)
}

/// Returns the smaller of two [`StepBound`] values ordered by step length then error bound so
/// parallel reductions remain deterministic.
fn least(first: StepBound, second: StepBound) -> StepBound {
    let order = second.0.total_cmp(&first.0).then(second.1.cmp(&first.1));
    if order.is_lt() { second } else { first }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use jolty_numerics::integration::{Formula, Method};
    use jolty_numerics::reduce;
    use rayon::ThreadPoolBuilder;
    use serde_json::json;

    use super::{
        Breakpoints, LeakAccumulator, MIN_STEP_ULPS, NextBreakpoint, StepFormula, minimum_step,
        neumaier_add,
    };
    use crate::options::{NUMBERS_PER_GIB, UnknownGrain};
    use crate::test_helpers::{grounded_simulator, source_resistor_simulator};
    use crate::{Options, SimulationError, Transient};

    /// Converts a [`NextBreakpoint`]'s time and gap to raw IEEE-754 bits for exact comparison.
    fn bits(breakpoint: NextBreakpoint) -> (u64, u64) {
        (breakpoint.time.to_bits(), breakpoint.gap.to_bits())
    }

    /// Converts every value in `values` to its raw IEEE-754 bit pattern.
    fn slice_bits(values: &[f64]) -> Vec<u64> {
        values.iter().copied().map(f64::to_bits).collect()
    }

    #[test]
    fn minimum_step_scales_with_time_and_always_advances_the_clock() {
        for scale in [1e-9, 0.8] {
            let at_zero = minimum_step(0.0, scale);
            assert_eq!(
                at_zero.to_bits(),
                (MIN_STEP_ULPS * f64::EPSILON * scale).to_bits()
            );
            assert!(at_zero > 0.0);
            let mut previous = at_zero;
            for time in [0.0, 1e-12, 1e-9, 1e-3, 1.0, 40.0, 1e6] {
                let step = minimum_step(time, scale);
                assert!(step >= previous, "minimum_step shrank at {time:e}");
                assert!(
                    time + step > time,
                    "step {step:e} failed to advance clock at {time:e}"
                );
                if time <= scale {
                    assert_eq!(step.to_bits(), at_zero.to_bits());
                } else {
                    assert_eq!(
                        step.to_bits(),
                        (MIN_STEP_ULPS * f64::EPSILON * time).to_bits()
                    );
                }
                previous = step;
            }
        }
    }

    #[test]
    fn breakpoints_come_in_order_without_near_twins_and_keep_the_stop() {
        let scale = 0.1;
        let mut breakpoints = Breakpoints::new(&[1.0, 3.0, 10.0], scale);
        breakpoints.insert(2.0);
        // Drop entries within `minimum_step(time, scale)` of 2 or 3, and entries beyond `stop`.
        breakpoints.insert(2.0 + 0.5 * minimum_step(2.0, scale));
        breakpoints.insert(3.0 - 0.5 * minimum_step(3.0, scale));
        breakpoints.insert(11.0);
        // Keep entries separated by more than `minimum_step(time, scale)`, including at small
        // times anchored to `scale` where spacing is finer than at later times.
        let early = 2.0 * minimum_step(0.0, scale);
        breakpoints.insert(early);
        assert_eq!(
            bits(breakpoints.after(0.0)),
            (early.to_bits(), (1.0 - early).to_bits())
        );
        assert_eq!(
            bits(breakpoints.after(0.5)),
            (1.0_f64.to_bits(), 1.0_f64.to_bits())
        );
        assert_eq!(
            bits(breakpoints.after(1.0)),
            (2.0_f64.to_bits(), 1.0_f64.to_bits())
        );
        assert_eq!(
            bits(breakpoints.after(2.5)),
            (3.0_f64.to_bits(), 7.0_f64.to_bits())
        );
        // The stop breakpoint persists regardless of how far past it `after` queries.
        assert_eq!(
            bits(breakpoints.after(20.0)),
            (10.0_f64.to_bits(), f64::INFINITY.to_bits())
        );
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
        let run = |grain: Option<UnknownGrain>| {
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
                let step = StepFormula { formula, restarted };
                accumulator.accumulate(step, &residuals, grain);
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
        let parallel = pool.install(|| run(Some(UnknownGrain { chunks: 1 })));
        assert_eq!(slice_bits(&parallel.signed), slice_bits(&serial.signed));
        assert_eq!(slice_bits(&parallel.absolute), slice_bits(&serial.absolute));
    }

    #[test]
    fn transient_rejects_non_finite_stop_and_max_step() {
        let mut simulator = source_resistor_simulator();
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -1e-6] {
            let stop_result = simulator.transient(
                &Transient {
                    stop: bad,
                    max_step: None,
                    method: Method::Trapezoidal,
                    breakpoints: Vec::new(),
                    initial_conditions: BTreeMap::new(),
                    operating_point: true,
                },
                &Options::default(),
            );
            assert!(
                matches!(stop_result, Err(SimulationError::InvalidAnalysis(_))),
                "expected InvalidAnalysis for stop={bad:?}, got {stop_result:?}"
            );
            let max_step_result = simulator.transient(
                &Transient {
                    stop: 1e-6,
                    max_step: Some(bad),
                    method: Method::Trapezoidal,
                    breakpoints: Vec::new(),
                    initial_conditions: BTreeMap::new(),
                    operating_point: true,
                },
                &Options::default(),
            );
            assert!(
                matches!(max_step_result, Err(SimulationError::InvalidAnalysis(_))),
                "expected InvalidAnalysis for max_step={bad:?}, got {max_step_result:?}"
            );
        }
    }

    #[test]
    fn transient_clears_past_after_early_error() {
        let mut simulator = grounded_simulator(
            "test",
            "0",
            &json!([
                {"name": "V1", "of": {"model": "voltage_source"},
                 "parameters": {"waveform": {"pulse": {
                     "initial": 0.0, "pulsed": 1.0, "delay": 0.5e-9,
                     "rise": 0.2e-9, "fall": 0.2e-9, "width": 1.0}}},
                 "terminals": {"positive": "in", "negative": "0"}},
                {"name": "R1", "of": {"model": "resistor"},
                 "parameters": {"resistance": 50.0},
                 "terminals": {"a": "in", "b": "a"}},
                {"name": "T1", "of": {"model": "transmission_line"},
                 "parameters": {"characteristic_impedance": 50.0, "delay": 1e-9},
                 "terminals": {"positive1": "a", "negative1": "0",
                               "positive2": "b", "negative2": "0"}},
                {"name": "R2", "of": {"model": "resistor"},
                 "parameters": {"resistance": 50.0},
                 "terminals": {"a": "b", "b": "0"}}
            ]),
        );
        let width = simulator.point_width();
        let options = Options {
            // Allow the pre-check (`stop / longest * width == width`) to pass so `start_past`
            // installs `state.past`, then fail inside the loop on the first accepted point.
            max_memory: 1.5 * width / NUMBERS_PER_GIB,
            ..Options::default()
        };
        let result = simulator.transient(
            &Transient {
                stop: 3e-9,
                max_step: Some(3e-9),
                method: Method::Trapezoidal,
                breakpoints: Vec::new(),
                initial_conditions: BTreeMap::new(),
                operating_point: true,
            },
            &options,
        );
        assert!(
            matches!(result, Err(SimulationError::TooManyPoints { .. })),
            "expected TooManyPoints, got {result:?}"
        );
        assert!(simulator.state.past.is_none());
    }
}
