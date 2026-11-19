//! Derivatives of every analysis with respect to the parameters named: its JVP, which
//! pushes tangents on the parameters forward to every unknown at every point of the
//! analysis, and its VJP, whose pullback carries cotangents on those outputs back to the
//! parameters.
//!
//! Each point of an analysis solves the circuit's equations `F(x, p) = 0`, and its solution
//! moves with the parameters by `dx/dp = −J⁻¹ ∂F/∂p`, where `J = ∂F/∂x` (the implicit
//! function theorem). Both factors are exact: `J` from the devices' dual numbers, `∂F/∂p`
//! from each parameter's twin, its instance prepared again computing with the derivative
//! of every number with respect to the parameter and evaluated where the instance is, into
//! a block of its own, so that a parameter costs what its instance does. The JVP evaluates
//! the product from the right, one solve with `J` per tangent `t`, `ẋ = −J⁻¹ (∂F/∂p) t`;
//! the VJP from the left, one solve with `Jᵀ` per cotangent `v`, `p̄ = −(J⁻ᵀ v)ᵀ ∂F/∂p`,
//! which serves every parameter however many (the adjoint method). At each point one
//! factorization serves every tangent and every cotangent. The two are transposes of one
//! linear map, so `⟨v, ẋ⟩ = ⟨p̄, t⟩` to the rounding of their solves.
//!
//! At the operating point and at each level of a DC sweep, `F` is the DC equations; a
//! number the sweep holds moves nothing there. Over frequency, the response to an
//! excitation `b` solves `(G + jωC) x = b`, so it moves by `−(G + jωC)⁻¹ (dG + jω dC) x`,
//! where `G` and `C` move with each parameter directly and through the operating point,
//! both exactly: the JVP pushes each tangent's operating-point shift `s = −G⁻¹ ∂F/∂p t`
//! through twins of the nonlinear devices that seed the unknowns along `s`, and the VJP
//! contracts the frequency adjoints and phasors with unknown-seeded twins of those devices
//! into `g = ∂/∂x Σ_k λ_kᵀ (G + jω_k C) x_k`, then solves `Gᵀ μ = g` once at the operating
//! point whatever the number of parameters. A phasor's cotangent pulls back to a complex
//! number, `vᵀ dx/dp`, bilinear; a real loss of the phasors `x = a + jb` takes the real
//! part, with `v = ∂L/∂a − j ∂L/∂b`.
//!
//! Over time, each step solves `f(x) + α q(x) + h = 0`, where the integration formula's
//! `α` and history `h` stand for the charges' derivative. The JVP is the direct method: it
//! carries each tangent's derivatives of `x`, of the charges and of their rate from step
//! to step, `J ẋ = −(∂f/∂p t + α ∂q/∂p t + ḣ)`, each step's Jacobian evaluated again at its
//! solution (D. E. Hocevar, P. Yang, T. N. Trick and B. D. Epler, "Transient sensitivity
//! computation for MOSFET circuits", *IEEE Transactions on Computer-Aided Design* 4(4),
//! 1985). The VJP is its transpose step for step, the discrete adjoint, one backward pass
//! from the last point to the first for every cotangent at once, one transposed solve a
//! step and cotangent whatever the number of parameters. A derivative moves no step and no
//! logic event. The start's derivatives are those of the state the transient starts from:
//! the operating point's, through the DC Jacobian, its rows of the nets the initial
//! conditions hold while it is solved replaced by theirs, `v − v₀ = 0`, which no parameter
//! moves; and none for a transient from its initial conditions alone, whose state is given.
//!
//! Over a periodic steady state, a period from the start `x₀` it repeats carries each
//! tangent's derivatives the same way, the start held, to its end `∂φ/∂p t`. The steady
//! state moves so that it still repeats: `(Φ − I) ẋ₀ = −∂φ/∂p t`, by the same monodromy
//! matrix and in the same units as shooting, or for an oscillator, whose period `T` moves
//! too, the bordered system `[[Φ − I, ∂φ/∂T], [∂φ/∂Tᵀ, 0]] [ẋ₀, Ṫ] = [−∂φ/∂p t, 0]`, its
//! start still orthogonal to the flow. A second pass over the period then carries the
//! tangent from `ẋ₀`, each step's charge terms also scaled by `−Ṫ / T`, every point at its
//! fraction of the period. Where `Φ − I` is singular, a neutral direction, `ẋ₀` leaves
//! every quantity a period conserves as it was, as shooting does, no singular value
//! counting as zero but within the rounding. The VJP takes the same three steps backward:
//! the adjoint over the period, the start held, gives the cotangent's own terms and its
//! derivatives with respect to the start and the period; the transpose of the bordered
//! solve carries those to the period's end; and the adjoint of the period from there gives
//! the rest. Both passes read the steps the converged period took, evaluating the circuit
//! and factoring its Jacobian again at each point.

use std::collections::BTreeMap;

use jolty_models::UnknownIndex;
use jolty_numerics::{Complex64, Pattern, Scalar, solve_each_in_range};

use crate::ac::over_frequencies;
use crate::newton::{add_product, factor_step_jacobian, force_row};
use crate::options::{Parallelism, float_count};
use crate::periodic::{DENSE_MATRICES, ShootingSystem};
use crate::sensitivity::{Derivatives, Differentiated, Parameter};
use crate::solver::Factors;
use crate::transient::{Carried, StepFormula, StepPoint, Tangent, TangentLinear};
use crate::twins::{Part, Twins};
use crate::{
    AcResult, AnalysisResult, DcSweep, DcSweepResult, Options, PeriodicSteadyStateResult,
    SimulationError, Simulator, Transient, TransientResult,
};

/// Analysis label for frequency-domain differentiation in delay-line refusals.
/// Origin: shared jolty diagnostic text across `jvp` and `vjp` (no numeric parameter).
const OVER_FREQUENCY: &str = "derivative over frequency";
/// Explanation for refusing delay lines in frequency-domain differentiation.
/// Origin: shared jolty diagnostic text across `jvp` and `vjp` (no numeric parameter).
const LEFT_OUT_OVER_FREQUENCY: &str = "it leaves out those of the terms its lines delay";
/// Analysis label for time-domain differentiation in delay-line refusals.
/// Origin: shared jolty diagnostic text across `jvp` and `vjp` (no numeric parameter).
const OVER_TIME: &str = "derivative over time";
/// Explanation for refusing delay lines in time-domain differentiation.
/// Origin: shared jolty diagnostic text across `jvp` and `vjp` (no numeric parameter).
const LEFT_OUT_OVER_TIME: &str = "it leaves out those of the past its lines read";
/// A periodic steady state's sensitivity, as [`SimulationError::TooMuchMemory`] names it.
/// Origin: shared jolty diagnostic text across `derivative` and `sensitivity` (no numeric parameter).
pub(crate) const PERIODIC_SENSITIVITY: &str = "periodic steady state sensitivity";

/// Message text for cotangent kind mismatches in the pullback (`vjp`).
/// Origin: shared jolty diagnostic text derived from real vs complex analysis returns.
const KINDS: &str =
    "cotangents on a response's phasors are complex, and on any other analysis's values real";

/// What [`Simulator::jvp`] finds: the analysis's result, and the derivative of every
/// unknown at each of its points along each tangent.
#[derive(Debug, Clone, PartialEq)]
pub struct JvpResult {
    /// The parameters the tangents move, in their order, each at its value.
    pub parameters: Vec<Parameter>,
    /// The derivatives, by point, then unknown, then tangent: of values, or over
    /// frequency of phasors.
    pub tangents: Derivatives,
    /// For an oscillator's periodic steady state, its period's derivative along each
    /// tangent.
    pub period: Option<Vec<f64>>,
    /// The result of the analysis differentiated.
    pub result: AnalysisResult,
}

/// A cotangent on an analysis's outputs, every unknown at every point: a linear function
/// of them, such as a loss's derivative with respect to each.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Cotangent<T> {
    /// By the index of each point where it is not zero, its value on every unknown there,
    /// in their order.
    pub points: BTreeMap<usize, Vec<T>>,
    /// Its value on an oscillator's period; zero for any other analysis.
    pub period: f64,
}

/// Cotangents on an analysis's outputs: on values, or over frequency on phasors.
#[derive(Debug, Clone, PartialEq)]
pub enum Cotangents {
    /// On values: at an operating point, along a sweep or a period, or over time.
    Real(Vec<Cotangent<f64>>),
    /// On phasors, over frequency.
    Complex(Vec<Cotangent<Complex64>>),
}

/// What a pullback finds of cotangents: each one's value on every parameter, by cotangent
/// then parameter.
#[derive(Debug, Clone, PartialEq)]
pub enum ParameterCotangents {
    /// Of cotangents on values.
    Real(Vec<Vec<f64>>),
    /// Of cotangents on phasors.
    Complex(Vec<Vec<Complex64>>),
}

/// What [`Simulator::vjp`] finds: the analysis's result, and what
/// [`Simulator::pull_back`] reads to carry cotangents on it back to the parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct Pullback {
    /// The parameters the cotangents pull back to, in their order, each at its value.
    pub parameters: Vec<Parameter>,
    /// The result of the analysis differentiated.
    pub result: AnalysisResult,
    /// What else of the analysis the pullback reads.
    saved: Saved,
}

/// What a pullback reads of its analysis besides its result.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Saved {
    /// Nothing, at an operating point or over frequency.
    Nothing,
    /// The sweep, whose values it holds again.
    DcSweep(DcSweep),
    /// The transient's start, and each step's formula and whether the integration
    /// restarts after it.
    Transient(Start, Vec<StepFormula>),
    /// The converged period's steps, and shooting's bordered system.
    Period(Vec<StepFormula>, ShootingSystem),
}

/// A transient's waveforms together with the step formulas that produced them.
#[derive(Clone, Copy)]
pub(crate) struct Trajectory<'a> {
    pub(crate) waveforms: &'a TransientResult,
    pub(crate) steps: &'a [StepFormula],
}

/// A periodic steady state's period, the steps it took and its shooting's bordered system.
#[derive(Clone, Copy)]
pub(crate) struct Period<'a> {
    pub(crate) found: &'a PeriodicSteadyStateResult,
    pub(crate) steps: &'a [StepFormula],
    pub(crate) system: &'a ShootingSystem,
}

/// Inputs shared across a transient backward adjoint pass and its start step.
#[derive(Clone, Copy)]
pub(crate) struct BackwardPass<'a> {
    pub(crate) trajectory: Trajectory<'a>,
    pub(crate) seeds: &'a [&'a BTreeMap<usize, Vec<f64>>],
    pub(crate) start: &'a Start,
    pub(crate) stretch: Option<f64>,
}

/// Whether a periodic steady state holds a fixed period or solves for a free oscillator period.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PeriodKind {
    Forced,
    Free,
}

impl PeriodKind {
    /// Returns [`Self::Free`] when shooting bordered the system with a period scale, else
    /// [`Self::Forced`].
    pub(crate) const fn from_scale(scale: Option<f64>) -> Self {
        match scale {
            Some(_) => Self::Free,
            None => Self::Forced,
        }
    }
}

/// Where a transient's derivatives start.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Start {
    /// At the operating point, solved with the unknowns given forced, as a transient's
    /// initial conditions force theirs.
    Solved(Vec<UnknownIndex>),
    /// At a state held, given: its derivatives along each direction, or none, zero.
    Held(Option<Vec<Vec<f64>>>),
}

/// The total `f64` numbers held inside a transient's waveforms and recorded values.
pub(crate) fn waveforms_numbers(waveforms: &TransientResult) -> f64 {
    let values = float_count(waveforms.values.len())
        * float_count(waveforms.values.first().map_or(0, Vec::len));
    let recorded = &waveforms.recorded;
    let recorded_count = float_count(recorded.currents.len())
        + float_count(recorded.charges.len())
        + float_count(recorded.conductances.len())
        + float_count(recorded.capacitances.len())
        + float_count(recorded.variables.len());
    values + recorded_count
}

/// Size parameters for [`forward_period_numbers`].
#[derive(Clone, Copy)]
struct ForwardPeriodBudget {
    unknowns: usize,
    rows: usize,
    points: usize,
    waveforms: f64,
    directions: usize,
    kept: usize,
    kind: PeriodKind,
}

/// Size parameters for [`PullPeriodBudget::check`].
#[derive(Clone, Copy)]
pub(crate) struct PullPeriodBudget {
    pub(crate) unknowns: usize,
    pub(crate) rows: usize,
    pub(crate) waveforms: f64,
    pub(crate) cotangents: f64,
    pub(crate) parameters: usize,
    pub(crate) kind: PeriodKind,
    pub(crate) extra: f64,
}

impl PullPeriodBudget {
    /// Checks that [`Simulator::pull_period`] together with caller buffers of `self.extra`
    /// numbers fits within `options.max_memory`.
    pub(crate) fn check(self, options: &Options) -> Result<(), SimulationError> {
        options.hold(PERIODIC_SENSITIVITY, self.rows, pull_period_numbers(self))
    }
}

/// The peak `f64` numbers held during [`Simulator::along_period`], across the bordered solve
/// (`DENSE_MATRICES` square matrices plus `directions` columns) and the replay passes (the
/// saved waveforms, the bordered matrix, each direction's three charge-history vectors and
/// step solution, an oscillator's charge tangent, and the recorded output derivatives).
fn forward_period_numbers(budget: ForwardPeriodBudget) -> f64 {
    let ForwardPeriodBudget {
        unknowns,
        rows,
        points,
        waveforms,
        directions,
        kept,
        kind,
    } = budget;
    let rows_float = float_count(rows);
    let unknowns_float = float_count(unknowns);
    let directions_float = float_count(directions);
    let solve_phase =
        DENSE_MATRICES * rows_float * rows_float + directions_float * rows_float + waveforms;
    let charges = match kind {
        PeriodKind::Free => 3.0 * unknowns_float,
        PeriodKind::Forced => 0.0,
    };
    let replay_phase = rows_float * rows_float
        + 4.0 * directions_float * unknowns_float
        + charges
        + float_count(points) * float_count(kept) * directions_float
        + waveforms;
    solve_phase.max(replay_phase)
}

/// The peak `f64` numbers held during [`Simulator::pull_period`] together with caller buffers
/// of `budget.extra` numbers (`unit_rows` and `pulled` in `Simulator::by_rows`, or `found` in
/// [`Simulator::pull_back`]).
fn pull_period_numbers(budget: PullPeriodBudget) -> f64 {
    let PullPeriodBudget {
        unknowns,
        rows,
        waveforms,
        cotangents,
        parameters,
        kind,
        extra,
    } = budget;
    let rows_float = float_count(rows);
    let unknowns_float = float_count(unknowns);
    // During `solve_each_in_range(matrix, &mut map)`, `map` adds one `rows × rows` matrix
    // beside `DENSE_MATRICES`, and `ends` holds one vector per cotangent.
    let solve_phase =
        (DENSE_MATRICES + 1.0) * rows_float * rows_float + cotangents * unknowns_float;
    // In `backward`, each cotangent keeps 3 vectors (`charge`, `before`, `rate`, plus 2 for
    // an oscillator's stretch charges) and builds up to 4 temporary vectors per step (`held`,
    // `onward`, `solution`, `shifted`), with `ended` adding 1 vector in the second pass;
    // `twins.dots` and its transpose hold up to `3 * cotangents * parameters` beside `6` (or
    // `7`) vectors of unknowns.
    let (step_vectors, dot_vectors) = match kind {
        PeriodKind::Free => (9.0, 7.0),
        PeriodKind::Forced => (8.0, 6.0),
    };
    let step_peak = step_vectors * cotangents * unknowns_float;
    let dot_peak =
        dot_vectors * cotangents * unknowns_float + 3.0 * cotangents * float_count(parameters);
    let backward_phase = rows_float * rows_float + step_peak.max(dot_peak);
    waveforms + extra + solve_phase.max(backward_phase)
}

/// The unit directions, one along each of `count` parameters.
pub(crate) fn units(count: usize) -> Vec<Vec<f64>> {
    (0..count)
        .map(|index| {
            let mut unit = vec![0.0; count];
            unit[index] = 1.0;
            unit
        })
        .collect()
}

/// Adds `scale` times `x` to `into`.
fn add(into: &mut [f64], scale: f64, x: &[f64]) {
    for (value, term) in into.iter_mut().zip(x) {
        *value += scale * term;
    }
}

/// Adds to `into` the product of the transpose of the matrix of `pattern` whose stored
/// values are `values` and the vector `x`, the matrix's rows in order.
fn add_transposed_product(pattern: &Pattern, values: &[f64], x: &[f64], into: &mut [f64]) {
    for (row, &factor) in x.iter().enumerate() {
        for &(slot, column) in pattern.row(row) {
            into[column] += values[slot] * factor;
        }
    }
}

/// `rows` by `columns` turned to `columns` by `rows`.
fn transposed<T: Copy>(rows: Vec<Vec<T>>, columns: usize) -> Vec<Vec<T>> {
    let mut turned = vec![Vec::with_capacity(rows.len()); columns];
    for row in rows {
        for (column, value) in turned.iter_mut().zip(row) {
            column.push(value);
        }
    }
    turned
}

/// The direct method's state along a transient: each direction's derivatives of the
/// charges, and those of the unknowns kept at every accepted point.
pub(crate) struct Forward {
    pub(crate) twins: Twins,
    /// The directions, by direction then twin.
    directions: Vec<Vec<f64>>,
    /// The unknowns whose derivatives it keeps, by index.
    kept: Vec<usize>,
    start: Start,
    tangents: Vec<Tangent>,
    /// By point, then unknown kept, then direction.
    pub(crate) found: Vec<Vec<Vec<f64>>>,
    /// An oscillator's period, and its derivative along each direction, which scales
    /// every step.
    pub(crate) stretch: Option<(f64, Vec<f64>)>,
    /// The charges themselves, while the period stretches, whose formula's terms the
    /// steps' lengths divide.
    charges: Option<Tangent>,
    /// Each direction's derivatives of the unknowns at the last point, kept for a period
    /// whose end they are.
    ends: Option<Vec<Vec<f64>>>,
}

impl Forward {
    /// The direct method along `directions` of `twins`, keeping the derivatives of the
    /// unknowns `kept`, from `start`.
    pub(crate) fn new(
        twins: Twins,
        directions: Vec<Vec<f64>>,
        kept: Vec<usize>,
        start: Start,
    ) -> Self {
        Self {
            twins,
            directions,
            kept,
            start,
            tangents: Vec::new(),
            found: Vec::new(),
            stretch: None,
            charges: None,
            ends: None,
        }
    }

    /// The derivatives of the unknowns kept in `moved`, by direction then unknown, by
    /// unknown kept then direction.
    fn keep(&self, moved: &[Vec<f64>]) -> Vec<Vec<f64>> {
        kept_of(&self.kept, moved)
    }
}

/// What a JVP finds: the twins, as they ended the analysis, the derivatives of the
/// unknowns kept, by point, then unknown kept, then direction, an oscillator's period's
/// along each direction, and the analysis's result.
pub(crate) struct Pushed {
    pub(crate) twins: Twins,
    pub(crate) tangents: Derivatives,
    pub(crate) period: Option<Vec<f64>>,
    pub(crate) result: AnalysisResult,
}

/// Each cotangent's derivatives, as the adjoint method carries them backward along a
/// transient, with respect to the charges at the last point, at the one before it and to
/// their rate, and while a period stretches the same of the stretch's terms, and the
/// period's derivative found so far.
struct Adjoint {
    charge: Vec<f64>,
    before: Vec<f64>,
    rate: Vec<f64>,
    stretch: Option<(Vec<f64>, Vec<f64>)>,
    period: f64,
    /// The last point its cotangent is not zero at, before which it carries nothing.
    last: Option<usize>,
}

/// Per-cotangent vectors prepared for one backward adjoint step.
struct StepAdjoint {
    held: Vec<f64>,
    onward: Vec<f64>,
    solution: Vec<f64>,
}

/// Unknown-seeded twins of nonlinear devices and the sorted matrix slots they write to.
struct NonlinearTwins<'a> {
    twins: &'a mut [jolty_models::Twin],
    slots: &'a [usize],
}

impl Adjoint {
    /// Carries the derivatives back over `step` to a point where the charges are `charge`,
    /// from `vectors` (`held` with respect to the point's charges, `onward` going on to the
    /// point before, and `solution` as the step's `λ`); while a period of length `stretch`
    /// stretches, its terms too. Returns `μ − α λ`, which the charges' partial derivatives
    /// take.
    fn step_back(
        &mut self,
        vectors: &StepAdjoint,
        step: StepFormula,
        charge: &[f64],
        stretch: Option<f64>,
    ) -> Vec<f64> {
        let formula = step.formula;
        let mut through = std::mem::take(&mut self.rate);
        add(&mut through, -1.0, &vectors.solution);
        if let (Some((stretch_charge, stretch_before)), Some(period)) = (&mut self.stretch, stretch)
        {
            let mut read = std::mem::take(stretch_charge);
            let onward = if step.restarted {
                add(&mut read, 1.0, stretch_before);
                vec![0.0; read.len()]
            } else {
                std::mem::take(stretch_before)
            };
            add(&mut read, formula.alpha, &through);
            let dot: f64 = read.iter().zip(charge).map(|(a, q)| a * q).sum();
            self.period -= dot / period;
            *stretch_charge = onward;
            add(stretch_charge, formula.beta[0], &through);
            *stretch_before = through
                .iter()
                .map(|value| formula.beta[1] * value)
                .collect();
        }
        self.charge.clear();
        self.charge.extend_from_slice(&vectors.onward);
        add(&mut self.charge, formula.beta[0], &through);
        self.before = through
            .iter()
            .map(|value| formula.beta[1] * value)
            .collect();
        self.rate = through.iter().map(|value| formula.gamma * value).collect();
        let mut moved = vectors.held.clone();
        add(&mut moved, -formula.alpha, &vectors.solution);
        moved
    }
}

/// What the adjoint method leaves of each cotangent at a transient's start: its derivative
/// with respect to the state held there, and with respect to an oscillator's period.
type Ends = Vec<(Vec<f64>, f64)>;

/// What takes each term of a cotangent's value on every parameter, by its index, every
/// parameter's at once; the terms sum to the values.
pub(crate) type Fold<'a> = &'a mut dyn FnMut(usize, &[f64]);

impl Simulator {
    /// The derivative of every unknown at each point of `analysis` along each of
    /// `tangents`, each a number per parameter of `parameters`, every parameter by its
    /// device's path and its name, such as `R1` and `resistance` or `V1` and
    /// `waveform.dc`: the JVP of the analysis, by the direct method, one solve per tangent
    /// at each point, exact.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::UnknownParameters`] for every parameter no device holds
    /// as a real number, [`SimulationError::InvalidAnalysis`] for a tangent that is not a
    /// finite number per parameter, [`SimulationError::TooMuchMemory`] when a periodic
    /// steady state's bordered system and tangent columns would hold more than the options'
    /// `max_memory`, [`SimulationError::Singular`] when a Jacobian is, and the errors of the
    /// analysis.
    pub fn jvp(
        &mut self,
        analysis: &Differentiated,
        parameters: &[(String, String)],
        tangents: &[Vec<f64>],
        options: &Options,
    ) -> Result<JvpResult, SimulationError> {
        options.parallelism.confined(|| {
            self.solvable()?;
            let named = self.named_parameters(parameters)?;
            if let Some(tangent) = tangents.iter().find(|tangent| {
                tangent.len() != parameters.len() || !tangent.iter().all(|value| value.is_finite())
            }) {
                return Err(SimulationError::InvalidAnalysis(format!(
                    "a tangent is a finite number for each of the {} parameters, not {tangent:?}",
                    parameters.len()
                )));
            }
            let (twins, directions) = self
                .tangent_twins(named, tangents)
                .map_err(|unread| SimulationError::UnknownParameters(named_of(&unread)))?;
            let kept: Vec<usize> = (0..self.elaborated.unknowns.len()).collect();
            let pushed = self.push_forward(analysis, twins, directions, &kept, options)?;
            Ok(JvpResult {
                parameters: pushed.twins.parameters,
                tangents: pushed.tangents,
                period: pushed.period,
                result: pushed.result,
            })
        })
    }

    /// The result of `analysis`, and its pullback: what carries a cotangent on every
    /// unknown at each of its points back to each parameter of `parameters`, each by its
    /// device's path and its name, through [`Simulator::pull_back`]: the VJP of the
    /// analysis, by the adjoint method, one transposed solve per cotangent at each point
    /// whatever the number of parameters.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::UnknownParameters`] for every parameter no device holds
    /// as a real number, [`SimulationError::Delayed`] over frequency or time for a circuit
    /// with delay lines, whose terms the derivatives would leave out,
    /// [`SimulationError::InvalidAnalysis`] over time for a circuit with logic, whose
    /// drivers a point evaluated again after the run does not see as they were, and the
    /// errors of the analysis.
    pub fn vjp(
        &mut self,
        analysis: &Differentiated,
        parameters: &[(String, String)],
        options: &Options,
    ) -> Result<Pullback, SimulationError> {
        options.parallelism.confined(|| {
            self.solvable()?;
            let twins = self.named_twins(parameters)?;
            let (result, saved) = self.run_saving(analysis, options)?;
            Ok(Pullback {
                parameters: twins.parameters,
                result,
                saved,
            })
        })
    }

    /// The value of each of `cotangents` on every parameter of `pullback`, by cotangent
    /// then parameter: its pullback, by the adjoint method.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::InvalidAnalysis`] for cotangents of another kind than
    /// the analysis's outputs, or at a point it does not have, or of another number of
    /// unknowns, or not finite; [`SimulationError::TooMuchMemory`] when a periodic steady
    /// state's pullback would hold more than the options' `max_memory`;
    /// [`SimulationError::Singular`] when a Jacobian is.
    pub fn pull_back(
        &mut self,
        pullback: &Pullback,
        cotangents: &Cotangents,
        options: &Options,
    ) -> Result<ParameterCotangents, SimulationError> {
        options.parallelism.confined(|| {
            self.solvable()?;
            self.prepare(options)?;
            let names = named_of(&pullback.parameters);
            let mut twins = self.named_twins(&names)?;
            self.checked(&pullback.result, cotangents)?;
            let parameter_count = twins.len();
            match (&pullback.saved, &pullback.result, cotangents) {
                (_, AnalysisResult::Ac(response), Cotangents::Complex(cotangents)) => {
                    Ok(ParameterCotangents::Complex(self.pull_frequencies(
                        &mut twins, response, cotangents, options,
                    )?))
                }
                (saved, result, Cotangents::Real(cotangents)) => {
                    if let Saved::Period(_, system) = saved
                        && let AnalysisResult::PeriodicSteadyState(steady) = result
                        && !cotangents.is_empty()
                    {
                        let unknowns = system.tolerances.len();
                        let rows = unknowns + usize::from(system.scale.is_some());
                        let cotangent_count = float_count(cotangents.len());
                        let extra = cotangent_count * float_count(parameter_count);
                        PullPeriodBudget {
                            unknowns,
                            rows,
                            waveforms: waveforms_numbers(&steady.transient),
                            cotangents: cotangent_count,
                            parameters: parameter_count,
                            kind: PeriodKind::from_scale(system.scale),
                            extra,
                        }
                        .check(options)?;
                    }
                    let found =
                        self.pull_collected(&mut twins, saved, result, cotangents, options)?;
                    Ok(ParameterCotangents::Real(found))
                }
                _ => Err(SimulationError::InvalidAnalysis(KINDS.to_owned())),
            }
        })
    }

    /// The twins of `parameters`, each by its device's path and its name, a column each,
    /// in their order.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::UnknownParameters`] naming every one no device holds as
    /// a real number.
    pub(crate) fn named_twins(
        &self,
        parameters: &[(String, String)],
    ) -> Result<Twins, SimulationError> {
        let (twins, unread) = self.unit_twins(self.named_parameters(parameters)?);
        if unread.is_empty() {
            Ok(twins)
        } else {
            Err(SimulationError::UnknownParameters(named_of(&unread)))
        }
    }

    /// Each of `parameters`, by its device's path and its name, its device's index and its
    /// value as the device is made now.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::UnknownParameters`] naming every one no device holds as
    /// a number.
    pub(crate) fn named_parameters(
        &self,
        parameters: &[(String, String)],
    ) -> Result<Vec<(usize, Parameter)>, SimulationError> {
        let mut found = Vec::with_capacity(parameters.len());
        let mut missing = Vec::new();
        for (path, name) in parameters {
            match self.parameter_at(path, name) {
                Some(parameter) => found.push(parameter),
                None => missing.push((path.clone(), name.clone())),
            }
        }
        if missing.is_empty() {
            Ok(found)
        } else {
            Err(SimulationError::UnknownParameters(missing))
        }
    }

    /// Checks that `cotangents` have the points and unknowns of `result` and are finite.
    fn checked(
        &self,
        result: &AnalysisResult,
        cotangents: &Cotangents,
    ) -> Result<(), SimulationError> {
        let points = points(result);
        let unknowns = self.elaborated.unknowns.len();
        let fit = match cotangents {
            Cotangents::Real(each) => cotangents_fit(each, points, unknowns),
            Cotangents::Complex(each) => cotangents_fit(each, points, unknowns),
        };
        if fit {
            Ok(())
        } else {
            Err(SimulationError::InvalidAnalysis(format!(
                "a cotangent is finite, at points from 0 to {} of {unknowns} unknowns each",
                points.saturating_sub(1)
            )))
        }
    }

    /// The JVP of `analysis` along `directions`, each a number per twin of `twins`,
    /// keeping the derivatives of the unknowns `kept`.
    ///
    /// # Errors
    ///
    /// As [`Simulator::jvp`].
    pub(crate) fn push_forward(
        &mut self,
        analysis: &Differentiated,
        mut twins: Twins,
        directions: Vec<Vec<f64>>,
        kept: &[usize],
        options: &Options,
    ) -> Result<Pushed, SimulationError> {
        let mut period = None;
        let (tangents, result) = match analysis {
            Differentiated::OperatingPoint => {
                let point = self.operating_point(options)?;
                let moved = self.dc_shifts(&mut twins, &point.values, &directions, &[], options)?;
                let at = kept_of(kept, &moved);
                (
                    Derivatives::Real(vec![at]),
                    AnalysisResult::OperatingPoint(point),
                )
            }
            Differentiated::DcSweep(sweep) => {
                let result = self.dc_sweep(sweep, options)?;
                let found =
                    self.along_values(sweep, &result, &mut twins, |simulator, twins, x| {
                        let moved = simulator.dc_shifts(twins, x, &directions, &[], options)?;
                        Ok(kept_of(kept, &moved))
                    })?;
                (Derivatives::Real(found), AnalysisResult::DcSweep(result))
            }
            Differentiated::Ac(ac) => {
                self.undelayed(OVER_FREQUENCY, LEFT_OUT_OVER_FREQUENCY)?;
                let response = self.ac(ac, options)?;
                let found =
                    self.frequency_tangents(&mut twins, &response, &directions, kept, options)?;
                (Derivatives::Complex(found), AnalysisResult::Ac(response))
            }
            Differentiated::Transient(transient) => {
                self.undelayed(OVER_TIME, LEFT_OUT_OVER_TIME)?;
                let start = self.start_of(transient)?;
                let mut forward = Forward::new(twins, directions, kept.to_vec(), start);
                let carried = Carried::Forward(&mut forward);
                let waveforms = self.transient_with(transient, options, None, carried)?;
                twins = forward.twins;
                (
                    Derivatives::Real(forward.found),
                    AnalysisResult::Transient(waveforms),
                )
            }
            Differentiated::PeriodicSteadyState(analysis) => {
                self.check_periodic(analysis)?;
                options.check()?;
                if !directions.is_empty() {
                    let rows =
                        self.elaborated.unknowns.len() + usize::from(analysis.period.is_none());
                    options.hold_square(
                        PERIODIC_SENSITIVITY,
                        rows,
                        DENSE_MATRICES,
                        directions.len(),
                    )?;
                }
                let (found, mut shot) = self.shot_steady_state(analysis, options)?;
                let steps = shot.take_steps();
                let system = shot.system();
                let period_data = Period {
                    found: &found,
                    steps: &steps,
                    system: &system,
                };
                let forward = Forward::new(twins, directions, kept.to_vec(), Start::Held(None));
                let forward = self.along_period(period_data, forward, options)?;
                twins = forward.twins;
                period = forward.stretch.map(|(_, derivatives)| derivatives);
                (
                    Derivatives::Real(forward.found),
                    AnalysisResult::PeriodicSteadyState(found),
                )
            }
        };
        Ok(Pushed {
            twins,
            tangents,
            period,
            result,
        })
    }

    /// The result of `analysis`, and what its pullback reads of it besides.
    ///
    /// # Errors
    ///
    /// As [`Simulator::vjp`].
    pub(crate) fn run_saving(
        &mut self,
        analysis: &Differentiated,
        options: &Options,
    ) -> Result<(AnalysisResult, Saved), SimulationError> {
        Ok(match analysis {
            Differentiated::OperatingPoint => (
                AnalysisResult::OperatingPoint(self.operating_point(options)?),
                Saved::Nothing,
            ),
            Differentiated::DcSweep(sweep) => (
                AnalysisResult::DcSweep(self.dc_sweep(sweep, options)?),
                Saved::DcSweep(sweep.clone()),
            ),
            Differentiated::Ac(ac) => {
                self.undelayed(OVER_FREQUENCY, LEFT_OUT_OVER_FREQUENCY)?;
                (AnalysisResult::Ac(self.ac(ac, options)?), Saved::Nothing)
            }
            Differentiated::Transient(transient) => {
                self.replayable()?;
                let start = self.start_of(transient)?;
                let mut steps = Vec::new();
                let waveforms =
                    self.transient_with(transient, options, None, Carried::Steps(&mut steps))?;
                (
                    AnalysisResult::Transient(waveforms),
                    Saved::Transient(start, steps),
                )
            }
            Differentiated::PeriodicSteadyState(analysis) => {
                let (found, mut shot) = self.shot_steady_state(analysis, options)?;
                let steps = shot.take_steps();
                (
                    AnalysisResult::PeriodicSteadyState(found),
                    Saved::Period(steps, shot.system()),
                )
            }
        })
    }

    /// Fails for a circuit whose points the adjoint method cannot evaluate again after its
    /// run as they were: one with delay lines, whose past the derivatives would leave out,
    /// or with logic, whose drivers it does not see as they were.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::Delayed`] and [`SimulationError::InvalidAnalysis`].
    pub(crate) fn replayable(&self) -> Result<(), SimulationError> {
        self.undelayed(OVER_TIME, LEFT_OUT_OVER_TIME)?;
        if self.elaborated.logic.nets.is_empty() {
            return Ok(());
        }
        Err(SimulationError::InvalidAnalysis(
            "a derivative over time by the adjoint method is found for circuits without \
             logic, whose drivers a point evaluated again after the run does not see as \
             they were"
                .to_owned(),
        ))
    }

    /// Where the derivatives of `transient` start: at its operating point, with the nets
    /// its initial conditions force there, or at its initial conditions alone, held.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::InvalidAnalysis`] for initial conditions that name no
    /// net, or a voltage that is not finite.
    pub(crate) fn start_of(&self, transient: &Transient) -> Result<Start, SimulationError> {
        if !transient.operating_point {
            return Ok(Start::Held(None));
        }
        let [initial] = self.forcings([(&transient.initial_conditions, "initial condition")])?;
        Ok(Start::Solved(
            initial
                .forced
                .into_iter()
                .map(|(unknown, _)| unknown)
                .collect(),
        ))
    }

    /// Pulls back `cotangents` on `result` (which left `saved`) to every twin, returning
    /// accumulated sums by cotangent then twin.
    pub(crate) fn pull_collected(
        &mut self,
        twins: &mut Twins,
        saved: &Saved,
        result: &AnalysisResult,
        cotangents: &[Cotangent<f64>],
        options: &Options,
    ) -> Result<Vec<Vec<f64>>, SimulationError> {
        let mut found = vec![vec![0.0; twins.len()]; cotangents.len()];
        let mut fold = |index: usize, terms: &[f64]| {
            for (total, term) in found[index].iter_mut().zip(terms) {
                *total += term;
            }
        };
        self.pull(twins, saved, result, cotangents, &mut fold, options)?;
        Ok(found)
    }

    /// The real cotangents `cotangents` of the analysis whose result is `result`, which
    /// left `saved`, pulled back to the twins: each term of each cotangent's values, by
    /// its index, to `fold`.
    fn pull(
        &mut self,
        twins: &mut Twins,
        saved: &Saved,
        result: &AnalysisResult,
        cotangents: &[Cotangent<f64>],
        fold: Fold<'_>,
        options: &Options,
    ) -> Result<(), SimulationError> {
        let seeds = cotangent_seeds(cotangents);
        match (saved, result) {
            (Saved::Nothing, AnalysisResult::OperatingPoint(point)) => {
                let at = seeds_at(&seeds, 0);
                self.dc_cotangents(twins, &point.values, &at, fold, options)
            }
            (Saved::DcSweep(sweep), AnalysisResult::DcSweep(found)) => {
                let mut point_index = 0;
                self.along_values(sweep, found, twins, |simulator, twins, x| {
                    let at = seeds_at(&seeds, point_index);
                    point_index += 1;
                    simulator.dc_cotangents(twins, x, &at, &mut *fold, options)
                })?;
                Ok(())
            }
            (Saved::Transient(start, steps), AnalysisResult::Transient(waveforms)) => {
                let pass = BackwardPass {
                    trajectory: Trajectory { waveforms, steps },
                    seeds: &seeds,
                    start,
                    stretch: None,
                };
                self.backward(twins, pass, fold, options)?;
                Ok(())
            }
            (Saved::Period(steps, system), AnalysisResult::PeriodicSteadyState(found)) => {
                let period = Period {
                    found,
                    steps,
                    system,
                };
                self.pull_period(twins, period, cotangents, fold, options)
            }
            _ => Err(SimulationError::InvalidAnalysis(KINDS.to_owned())),
        }
    }

    /// Factors the DC conductance matrix currently in the workspace, replacing the rows of
    /// `forced` with `v − v₀ = 0` when non-empty.
    fn factor_dc_forced(
        &mut self,
        forced: &[UnknownIndex],
    ) -> Result<Factors<'_, f64>, SimulationError> {
        if forced.is_empty() {
            return self
                .state
                .solver
                .factor(&self.state.workspace.contributions.conductance);
        }
        let mut matrix = self.state.workspace.contributions.conductance.clone();
        for unknown in forced {
            force_row(&self.elaborated, &mut matrix, unknown.index());
        }
        self.state.solver.factor(&matrix)
    }

    /// Each direction's derivative of the DC solution `x`, `−J⁻¹ Σ t_j ∂F/∂p_j`, by
    /// direction then unknown, the rows of the unknowns `forced` held at their voltages,
    /// `v − v₀ = 0`, which no parameter moves; the twins left evaluated at `x`.
    fn dc_shifts(
        &mut self,
        twins: &mut Twins,
        x: &[f64],
        directions: &[Vec<f64>],
        forced: &[UnknownIndex],
        options: &Options,
    ) -> Result<Vec<Vec<f64>>, SimulationError> {
        self.evaluate(x, None, 1.0, options.gmin, false);
        let factors = self.factor_dc_forced(forced)?;
        let shared = factors.shared();
        let parallelism = &options.parallelism;
        twins.evaluate(x, None, options.gmin, parallelism);
        let dimension = x.len();
        let twins = &*twins;
        let mut directions: Vec<&Vec<f64>> = directions.iter().collect();
        let moved = parallelism.spread(
            &mut directions,
            || shared.workspace(),
            |workspace, direction| {
                let mut moved: Vec<f64> = twins
                    .combined(direction, dimension, Part::Residual)
                    .iter()
                    .map(|value| -value)
                    .collect();
                for unknown in forced {
                    moved[unknown.index()] = 0.0;
                }
                shared.solve(&mut moved, workspace)?;
                Ok(moved)
            },
        );
        moved.into_iter().collect()
    }

    /// Pulls back the cotangents `at`, each its index and its value on every unknown, at
    /// the DC solution `x`: `−λᵀ ∂F/∂p` for each twin, where `Jᵀ λ = v`, to `fold`.
    fn dc_cotangents(
        &mut self,
        twins: &mut Twins,
        x: &[f64],
        at: &[(usize, &[f64])],
        fold: Fold<'_>,
        options: &Options,
    ) -> Result<(), SimulationError> {
        if at.is_empty() {
            return Ok(());
        }
        self.evaluate(x, None, 1.0, options.gmin, false);
        let factors = self
            .state
            .solver
            .factor(&self.state.workspace.contributions.conductance)?;
        let shared = factors.shared();
        let parallelism = &options.parallelism;
        let mut given: Vec<&[f64]> = at.iter().map(|&(_, values)| values).collect();
        let adjoints = parallelism.spread(
            &mut given,
            || shared.workspace(),
            |workspace, values| {
                let mut adjoint = values.to_vec();
                shared.solve_transpose(&mut adjoint, workspace)?;
                Ok(adjoint)
            },
        );
        let adjoints = adjoints
            .into_iter()
            .collect::<Result<Vec<Vec<f64>>, SimulationError>>()?;
        twins.evaluate(x, None, options.gmin, parallelism);
        let by_twin = twins.dots(Part::Residual, -1.0, &adjoints, parallelism);
        for (&(index, _), terms) in at.iter().zip(transposed(by_twin, at.len())) {
            fold(index, &terms);
        }
        Ok(())
    }

    /// What `each` finds at each swept value of the sweep `result`, of `sweep`, given the
    /// solution there, its source held there, and the twins of its device prepared again
    /// there, from what its instance is made of: a number the sweep holds moves nothing.
    fn along_values<R>(
        &mut self,
        sweep: &DcSweep,
        result: &DcSweepResult,
        twins: &mut Twins,
        mut each: impl FnMut(&mut Self, &mut Twins, &[f64]) -> Result<R, SimulationError>,
    ) -> Result<Vec<R>, SimulationError> {
        let swept = self.swept(sweep)?;
        let mut found = Vec::with_capacity(result.swept.len());
        let mut run = || {
            for (&value, x) in result.swept.iter().zip(&result.values) {
                self.hold(swept, Some(value))?;
                twins.hold(self, &swept, sweep.parameter.as_deref());
                found.push(each(self, twins, x)?);
            }
            Ok(())
        };
        let ran = run();
        self.hold(swept, None)?;
        ran.map(|()| found)
    }

    /// The derivatives of the phasors of the unknowns `kept` in `response` along each of
    /// `directions`, by frequency, then unknown kept, then direction.
    fn frequency_tangents(
        &mut self,
        twins: &mut Twins,
        response: &AcResult,
        directions: &[Vec<f64>],
        kept: &[usize],
        options: &Options,
    ) -> Result<Vec<Vec<Vec<Complex64>>>, SimulationError> {
        let operating = &response.operating_point.values;
        let nonlinear = self.nonlinear_devices();
        let biases = if nonlinear.is_empty() {
            self.evaluate(operating, None, 1.0, options.gmin, false);
            twins.evaluate(operating, None, options.gmin, &options.parallelism);
            vec![Vec::new(); directions.len()]
        } else {
            let shifts = self.dc_shifts(twins, operating, directions, &[], options)?;
            self.entries_along(
                &nonlinear,
                operating,
                &shifts,
                options.gmin,
                &options.parallelism,
            )
        };
        let positions: Vec<(usize, usize)> = self.elaborated.pattern.positions().collect();
        let twins = &*twins;
        let dimension = self.elaborated.unknowns.len();
        over_frequencies(
            self.linearization(),
            &self.state.workspace.contributions,
            &response.frequencies,
            &options.parallelism,
            &|factors, index, frequency| {
                let phasors = &response.values[index];
                let omega = std::f64::consts::TAU * frequency;
                let mut at = Vec::with_capacity(directions.len());
                for (direction, entries) in directions.iter().zip(&biases) {
                    let mut moved = vec![Complex64::default(); dimension];
                    for (column, &along) in direction.iter().enumerate() {
                        if along != 0.0 {
                            twins
                                .add_product(column, &positions, along, omega, phasors, &mut moved);
                        }
                    }
                    for &(slot, conductance_rate, capacitance_rate) in entries {
                        let (row, column) = positions[slot];
                        moved[row] += Complex64::new(conductance_rate, omega * capacitance_rate)
                            * phasors[column];
                    }
                    for value in &mut moved {
                        *value = -*value;
                    }
                    factors.solve(&mut moved)?;
                    at.push(kept.iter().map(|&unknown| moved[unknown]).collect());
                }
                Ok(transposed(at, kept.len()))
            },
        )
    }

    /// The values of `cotangents` on the phasors of `response` on every twin, by
    /// cotangent then twin, summed over the frequencies in order and through the
    /// operating point by one transposed DC solve.
    pub(crate) fn pull_frequencies(
        &mut self,
        twins: &mut Twins,
        response: &AcResult,
        cotangents: &[Cotangent<Complex64>],
        options: &Options,
    ) -> Result<Vec<Vec<Complex64>>, SimulationError> {
        if cotangents.is_empty() {
            return Ok(Vec::new());
        }
        if twins.len() == 0 {
            return Ok(vec![Vec::new(); cotangents.len()]);
        }
        let operating = &response.operating_point.values;
        let parallelism = &options.parallelism;
        self.evaluate(operating, None, 1.0, options.gmin, false);
        twins.evaluate(operating, None, options.gmin, parallelism);
        let positions: Vec<(usize, usize)> = self.elaborated.pattern.positions().collect();
        let found = {
            let twins = &*twins;
            over_frequencies(
                self.linearization(),
                &self.state.workspace.contributions,
                &response.frequencies,
                parallelism,
                &|factors, index, frequency| {
                    let phasors = &response.values[index];
                    let omega = std::f64::consts::TAU * frequency;
                    let mut at = Vec::new();
                    for (cotangent, given) in cotangents.iter().enumerate() {
                        let Some(values) = given.points.get(&index) else {
                            continue;
                        };
                        let mut adjoint = values.clone();
                        factors.solve_transpose(&mut adjoint)?;
                        let by_twin: Vec<Complex64> = (0..twins.len())
                            .map(|column| {
                                -twins.product(column, &positions, &adjoint, omega, phasors)
                            })
                            .collect();
                        at.push((cotangent, omega, adjoint, by_twin));
                    }
                    Ok(at)
                },
            )?
        };
        let nonlinear = self.nonlinear_devices();
        let (mut unknown_twins, nonlinear_slots) = if nonlinear.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            self.unknown_twins(&nonlinear)
        };
        let mut sums = vec![vec![Complex64::default(); twins.len()]; cotangents.len()];
        let mut weights: Vec<Option<(Vec<Complex64>, Vec<Complex64>)>> =
            vec![None; cotangents.len()];
        for (index, entries) in found.into_iter().enumerate() {
            let phasors = &response.values[index];
            for (cotangent, omega, adjoint, terms) in entries {
                for (total, term) in sums[cotangent].iter_mut().zip(terms) {
                    *total += term;
                }
                if nonlinear_slots.is_empty() {
                    continue;
                }
                let (weight_conductance, weight_capacitance) = weights[cotangent]
                    .get_or_insert_with(|| {
                        (
                            vec![Complex64::default(); nonlinear_slots.len()],
                            vec![Complex64::default(); nonlinear_slots.len()],
                        )
                    });
                for (packed, &slot) in nonlinear_slots.iter().enumerate() {
                    let (row, column) = positions[slot];
                    let product = adjoint[row] * phasors[column];
                    weight_conductance[packed] += product;
                    weight_capacitance[packed] +=
                        Complex64::new(-omega * product.im, omega * product.re);
                }
            }
        }
        if !nonlinear_slots.is_empty() {
            let mut nonlinear_twins = NonlinearTwins {
                twins: &mut unknown_twins,
                slots: &nonlinear_slots,
            };
            self.add_operating_point_cotangents(
                twins,
                &mut nonlinear_twins,
                operating,
                &weights,
                &mut sums,
                options,
            )?;
        }
        Ok(sums)
    }

    /// Adds to `sums` the operating-point share `+(J₀⁻ᵀ g)ᵀ ∂F₀/∂p` of each active
    /// cotangent's `weights`, indexed by `nonlinear.slots`, in
    /// [`Simulator::pull_frequencies`].
    fn add_operating_point_cotangents(
        &mut self,
        twins: &mut Twins,
        nonlinear: &mut NonlinearTwins<'_>,
        operating: &[f64],
        weights: &[Option<(Vec<Complex64>, Vec<Complex64>)>],
        sums: &mut [Vec<Complex64>],
        options: &Options,
    ) -> Result<(), SimulationError> {
        let parallelism = &options.parallelism;
        let gradients = self.entries_cotangents(
            nonlinear.twins,
            nonlinear.slots,
            operating,
            weights,
            options.gmin,
            parallelism,
        );
        let active: Vec<(usize, &Vec<Complex64>)> = gradients
            .iter()
            .enumerate()
            .filter_map(|(cotangent, gradient)| {
                let gradient = gradient.as_ref()?;
                gradient
                    .iter()
                    .any(|value| *value != Complex64::default())
                    .then_some((cotangent, gradient))
            })
            .collect();
        if active.is_empty() {
            return Ok(());
        }
        let factors = self
            .state
            .solver
            .factor(&self.state.workspace.contributions.conductance)?;
        let shared = factors.shared();
        let mut sides: Vec<Vec<f64>> = Vec::with_capacity(active.len() * 2);
        for &(_, gradient) in &active {
            sides.push(gradient.iter().map(|value| value.re).collect());
            sides.push(gradient.iter().map(|value| value.im).collect());
        }
        let solved = parallelism.spread(
            &mut sides,
            || shared.workspace(),
            |workspace, side| {
                shared.solve_transpose(side, workspace)?;
                Ok(())
            },
        );
        solved
            .into_iter()
            .collect::<Result<Vec<()>, SimulationError>>()?;
        let by_column = twins.dots(Part::Residual, 1.0, &sides, parallelism);
        for (column, dots) in by_column.into_iter().enumerate() {
            for (&[real, imaginary], &(cotangent, _)) in dots.as_chunks::<2>().0.iter().zip(&active)
            {
                sums[cotangent][column] += Complex64::new(real, imaginary);
            }
        }
        Ok(())
    }

    /// Each direction's derivatives along `period`: `forward`, carried over the period's steps
    /// once from the start held to the period's end, then again from the start's own
    /// derivatives, and with an oscillator's period's.
    pub(crate) fn along_period(
        &mut self,
        period: Period<'_>,
        mut forward: Forward,
        options: &Options,
    ) -> Result<Forward, SimulationError> {
        let Period {
            found,
            steps,
            system,
        } = period;
        let ShootingSystem {
            matrix,
            tolerances,
            scale,
        } = system;
        if forward.directions.is_empty() {
            forward.found =
                vec![vec![Vec::new(); forward.kept.len()]; found.transient.values.len()];
            forward.stretch = scale.map(|_| (found.period, Vec::new()));
            return Ok(forward);
        }
        let trajectory = Trajectory {
            waveforms: &found.transient,
            steps,
        };
        let unknowns = self.elaborated.unknowns.len();
        let rows = unknowns + usize::from(scale.is_some());
        let needed = forward_period_numbers(ForwardPeriodBudget {
            unknowns,
            rows,
            points: found.transient.values.len(),
            waveforms: waveforms_numbers(&found.transient),
            directions: forward.directions.len(),
            kept: forward.kept.len(),
            kind: PeriodKind::from_scale(*scale),
        });
        options.hold(PERIODIC_SENSITIVITY, rows, needed)?;
        forward.ends = Some(Vec::new());
        self.replay(&mut forward, trajectory, options)?;
        forward.tangents.clear();
        forward.tangents.shrink_to_fit();
        // (Φ − I) ẋ₀ + ∂φ/∂T Ṫ = −∂φ/∂p t, in shooting's units, bordered for an
        // oscillator by the phase condition's row.
        let mut columns: Vec<Vec<f64>> = forward
            .ends
            .take()
            .unwrap_or_default()
            .into_iter()
            .map(|mut column| {
                for (derivative, tolerance) in column.iter_mut().zip(tolerances) {
                    *derivative = -*derivative / *tolerance;
                }
                if scale.is_some() {
                    column.push(0.0);
                }
                column
            })
            .collect();
        solve_each_in_range(matrix, &mut columns).map_err(|error| self.state.solver.name(error))?;
        forward.stretch = scale.map(|scale| {
            let derivatives = columns.iter().map(|column| column[unknowns] * scale);
            (found.period, derivatives.collect())
        });
        for column in &mut columns {
            column.truncate(unknowns);
            for (scaled, tolerance) in column.iter_mut().zip(tolerances) {
                *scaled *= *tolerance;
            }
        }
        forward.start = Start::Held(Some(columns));
        self.replay(&mut forward, trajectory, options)?;
        Ok(forward)
    }

    /// Carries `forward` over the points of `trajectory`, as the transient carried it when it
    /// took them.
    pub(crate) fn replay(
        &mut self,
        forward: &mut Forward,
        trajectory: Trajectory<'_>,
        options: &Options,
    ) -> Result<(), SimulationError> {
        let result = trajectory.waveforms;
        self.start_forward(forward, &result.values[0], options)?;
        for (point, &step) in trajectory.steps.iter().enumerate() {
            let at = StepPoint {
                time: result.times[point + 1],
                values: &result.values[point + 1],
            };
            self.advance_forward(forward, at, step, options)?;
        }
        Ok(())
    }

    /// The values on every twin of `cotangents` on `period`, each term to `fold`: the adjoint
    /// over the period, the start held, the transpose of the bordered solve, and the adjoint
    /// over the period again from its end.
    fn pull_period(
        &mut self,
        twins: &mut Twins,
        period: Period<'_>,
        cotangents: &[Cotangent<f64>],
        fold: Fold<'_>,
        options: &Options,
    ) -> Result<(), SimulationError> {
        if cotangents.is_empty() {
            return Ok(());
        }
        let Period {
            found,
            steps,
            system,
        } = period;
        let ShootingSystem {
            matrix,
            tolerances,
            scale,
        } = system;
        let unknowns = tolerances.len();
        let rows = unknowns + usize::from(scale.is_some());
        PullPeriodBudget {
            unknowns,
            rows,
            waveforms: waveforms_numbers(&found.transient),
            cotangents: float_count(cotangents.len()),
            parameters: twins.len(),
            kind: PeriodKind::from_scale(*scale),
            extra: 0.0,
        }
        .check(options)?;
        let trajectory = Trajectory {
            waveforms: &found.transient,
            steps,
        };
        let seeds = cotangent_seeds(cotangents);
        let held = Start::Held(None);
        let stretch = scale.map(|_| found.period);
        let first_pass = BackwardPass {
            trajectory,
            seeds: &seeds,
            start: &held,
            stretch,
        };
        let ends = self.backward(twins, first_pass, fold, options)?;
        // The bordered solve's map, column by column, whose transpose carries each
        // cotangent on the start and the period to the period's end.
        let mut map = units(rows);
        solve_each_in_range(matrix, &mut map).map_err(|error| self.state.solver.name(error))?;
        let last = found.transient.values.len() - 1;
        let ended: Vec<BTreeMap<usize, Vec<f64>>> = ends
            .into_iter()
            .zip(cotangents)
            .map(|((start, period), cotangent)| {
                let mut scaled: Vec<f64> = start
                    .iter()
                    .zip(tolerances)
                    .map(|(value, tolerance)| value * tolerance)
                    .collect();
                if let Some(scale) = scale {
                    scaled.push((period + cotangent.period) * scale);
                }
                let end = tolerances
                    .iter()
                    .zip(&map)
                    .map(|(tolerance, column)| {
                        let dot: f64 = column.iter().zip(&scaled).map(|(m, w)| m * w).sum();
                        -dot / tolerance
                    })
                    .collect();
                BTreeMap::from([(last, end)])
            })
            .collect();
        drop(map);
        let seeds: Vec<&BTreeMap<usize, Vec<f64>>> = ended.iter().collect();
        let second_pass = BackwardPass {
            trajectory,
            seeds: &seeds,
            start: &held,
            stretch: None,
        };
        self.backward(twins, second_pass, fold, options)?;
        Ok(())
    }

    /// Starts the direct method at the state `x` a transient starts from: each
    /// direction's derivative of it, by the DC Jacobian at the operating point, with the
    /// rows of the nets forced there held, or as given for a state held; and of the
    /// charges there.
    pub(crate) fn start_forward(
        &mut self,
        forward: &mut Forward,
        x: &[f64],
        options: &Options,
    ) -> Result<(), SimulationError> {
        let shifts = match &mut forward.start {
            Start::Held(shifts) => shifts
                .take()
                .unwrap_or_else(|| vec![vec![0.0; x.len()]; forward.directions.len()]),
            Start::Solved(forced) => {
                self.dc_shifts(&mut forward.twins, x, &forward.directions, forced, options)?
            }
        };
        let at = forward.ends.is_none().then(|| forward.keep(&shifts));
        // A state that only time moves holds at the operating point, and has a charge
        // only once time passes.
        self.evaluate(x, Some(0.0), 1.0, options.gmin, false);
        forward.charges = forward
            .stretch
            .as_ref()
            .map(|_| Tangent::new(self.state.workspace.contributions.charge.clone()));
        forward.tangents.clear();
        let grain = options.parallelism.unknown_grain(x.len());
        forward
            .twins
            .evaluate(x, Some(0.0), options.gmin, &options.parallelism);
        for (direction, moved) in forward.directions.iter().zip(&shifts) {
            let mut charge = forward.twins.combined(direction, x.len(), Part::Charge);
            let capacitances = &self.state.workspace.contributions.capacitance;
            add_product(
                &self.elaborated.pattern,
                capacitances,
                moved,
                &mut charge,
                grain,
            );
            forward.tangents.push(Tangent::new(charge));
        }
        if let Some(at) = at {
            forward.found = vec![at];
        } else {
            forward.found.clear();
        }
        Ok(())
    }

    /// Carries the direct method over the step a transient accepted at `point` by `step`.
    /// While an oscillator's period stretches, each direction's history also takes the charges'
    /// terms the steps divide, `α q_{n+1} + β₀ q_n + β₁ q_{n-1}`, times `−Ṫ / T`.
    pub(crate) fn advance_forward(
        &mut self,
        forward: &mut Forward,
        point: StepPoint<'_>,
        step: StepFormula,
        options: &Options,
    ) -> Result<(), SimulationError> {
        self.evaluate(point.values, Some(point.time), 1.0, options.gmin, false);
        let unknowns = point.values.len();
        let grain = options.parallelism.unknown_grain(unknowns);
        let mut factors = factor_step_jacobian(
            &mut self.state.solver,
            &mut self.state.workspace,
            step.formula.alpha,
            grain,
        )?;
        let contributions = &self.state.workspace.contributions;
        let stretched = match (&forward.stretch, &forward.charges) {
            (Some((period, derivatives)), Some(charges)) => {
                let divided = charges.divided(&contributions.charge, &step.formula);
                Some((divided, period, derivatives))
            }
            _ => None,
        };
        forward.twins.evaluate(
            point.values,
            Some(point.time),
            options.gmin,
            &options.parallelism,
        );
        if let Some(ends) = &mut forward.ends {
            ends.clear();
        }
        let linear = TangentLinear {
            pattern: &self.elaborated.pattern,
            capacitance: &contributions.capacitance,
            step,
            grain,
        };
        let mut solutions = Vec::with_capacity(forward.directions.len());
        for (index, (direction, tangent)) in forward
            .directions
            .iter()
            .zip(&mut forward.tangents)
            .enumerate()
        {
            let residual = forward.twins.combined(direction, unknowns, Part::Residual);
            let partial_charge = forward.twins.combined(direction, unknowns, Part::Charge);
            let mut history = tangent.history(&step.formula);
            if let Some((divided, period, derivatives)) = &stretched {
                let by = derivatives[index] / *period;
                for (term, divided) in history.iter_mut().zip(divided) {
                    *term -= by * divided;
                }
            }
            let rhs: Vec<f64> = residual
                .iter()
                .zip(&partial_charge)
                .zip(&history)
                .map(|((residual, charge), term)| -(residual + step.formula.alpha * charge + term))
                .collect();
            let moved = tangent.solve_step(&mut factors, linear, rhs, &history, partial_charge)?;
            solutions.push(moved);
        }
        if let Some(charges) = &mut forward.charges {
            charges.advance_plain(contributions.charge.clone(), step);
        }
        if forward.ends.is_some() {
            forward.ends = Some(solutions);
        } else {
            forward.found.push(forward.keep(&solutions));
        }
        Ok(())
    }

    /// The adjoint method over `pass.trajectory` from the last point back to the first, for
    /// every cotangent of `pass.seeds` at once, each a value on every unknown at the points
    /// where it is not zero: each term of each one's value on every twin to `fold`, by its
    /// index. The derivatives start at `pass.start`; while an oscillator's period
    /// `pass.stretch` stretches, each step's charge terms scale with it. Returns, for each
    /// cotangent, its derivatives with respect to the state held at the start, zero for a
    /// start solved, and to the period.
    ///
    /// At the point `k`, where the cotangent is `c_k`, with `μ = λ_q + α λ_r`, and `+ λ_q'`
    /// where the integration restarts after the point:
    ///
    /// ```text
    /// (G + α C)ᵀ λ = c_k + Cᵀ μ,          G and C evaluated at x_k
    /// p̄ += −λᵀ ∂f/∂p + (μ − α λ)ᵀ ∂q/∂p
    /// λ_q ← β₀ (λ_r − λ), and + λ_q' where it does not restart
    /// λ_q' ← β₁ (λ_r − λ),  λ_r ← γ (λ_r − λ)
    /// ```
    ///
    /// At the start, where the charges' derivative is zero and the point before is the
    /// point itself, `μ = λ_q + λ_q'` and `p̄ += μᵀ ∂q/∂p`; at an operating point, the DC
    /// Jacobian, its forced rows held, takes the step's place, `Jᵀ λ = c_0 + Cᵀ μ`, and
    /// `p̄ += −λᵀ ∂F/∂p`; a state held instead has the derivative `c_0 + Cᵀ μ`. A
    /// stretching period's terms `−(Ṫ / T) (α q_k + β₀ q_{k-1} + β₁ q'_{k-1})` in each
    /// step's history take `T̄ += −(λ_r − λ)ᵀ (…) / T`, carried back to the charges they
    /// read as `λ_q` and `λ_q'` are.
    pub(crate) fn backward(
        &mut self,
        twins: &mut Twins,
        pass: BackwardPass<'_>,
        fold: Fold<'_>,
        options: &Options,
    ) -> Result<Ends, SimulationError> {
        let dimension = self.elaborated.unknowns.len();
        let gmin = options.gmin;
        let grain = options.parallelism.unknown_grain(dimension);
        let zeros = || vec![0.0; dimension];
        let mut adjoints: Vec<Adjoint> = pass
            .seeds
            .iter()
            .map(|seed| Adjoint {
                charge: zeros(),
                before: zeros(),
                rate: zeros(),
                stretch: pass.stretch.map(|_| (zeros(), zeros())),
                period: 0.0,
                last: seed.keys().next_back().copied(),
            })
            .collect();
        for (point, &step) in pass.trajectory.steps.iter().enumerate().rev() {
            let point = point + 1;
            let active: Vec<usize> = (0..adjoints.len())
                .filter(|&index| adjoints[index].last.is_some_and(|last| last >= point))
                .collect();
            if active.is_empty() {
                continue;
            }
            let (time, x) = (
                pass.trajectory.waveforms.times[point],
                &pass.trajectory.waveforms.values[point],
            );
            self.evaluate(x, Some(time), 1.0, gmin, false);
            let capacitance = &self.state.workspace.contributions.capacitance;
            let mut solved = Vec::with_capacity(active.len());
            for &index in &active {
                let adjoint = &mut adjoints[index];
                // The derivative with respect to the charges at the point, all told, and
                // what goes on to those at the point before it.
                let mut held = std::mem::take(&mut adjoint.charge);
                let onward = if step.restarted {
                    add(&mut held, 1.0, &adjoint.before);
                    zeros()
                } else {
                    std::mem::replace(&mut adjoint.before, zeros())
                };
                add(&mut held, step.formula.alpha, &adjoint.rate);
                let mut solution = zeros();
                if let Some(values) = pass.seeds[index].get(&point) {
                    add(&mut solution, 1.0, values);
                }
                add_transposed_product(&self.elaborated.pattern, capacitance, &held, &mut solution);
                solved.push(StepAdjoint {
                    held,
                    onward,
                    solution,
                });
            }
            let mut factors = factor_step_jacobian(
                &mut self.state.solver,
                &mut self.state.workspace,
                step.formula.alpha,
                grain,
            )?;
            for vectors in &mut solved {
                factors.solve_transpose(&mut vectors.solution)?;
            }
            let charge = &self.state.workspace.contributions.charge;
            let mut shifted = Vec::with_capacity(active.len());
            for (&index, vectors) in active.iter().zip(&solved) {
                shifted.push(adjoints[index].step_back(vectors, step, charge, pass.stretch));
            }
            let solutions: Vec<Vec<f64>> = solved.into_iter().map(|item| item.solution).collect();
            twins.evaluate(x, Some(time), gmin, &options.parallelism);
            let charged = twins.dots(Part::Charge, 1.0, &shifted, &options.parallelism);
            let charged = transposed(charged, active.len());
            let resistive = twins.dots(Part::Residual, -1.0, &solutions, &options.parallelism);
            let resistive = transposed(resistive, active.len());
            for ((&index, charged), resistive) in active.iter().zip(&charged).zip(&resistive) {
                fold(index, charged);
                fold(index, resistive);
            }
        }
        self.backward_start(twins, pass, adjoints, fold, options)
    }

    /// The adjoint method at a transient's start, the point `0` of `pass.trajectory`, from
    /// what `adjoints` carried back to it, as [`Simulator::backward`] says.
    fn backward_start(
        &mut self,
        twins: &mut Twins,
        pass: BackwardPass<'_>,
        adjoints: Vec<Adjoint>,
        fold: Fold<'_>,
        options: &Options,
    ) -> Result<Ends, SimulationError> {
        let gmin = options.gmin;
        let x = &pass.trajectory.waveforms.values[0];
        self.evaluate(x, Some(0.0), 1.0, gmin, false);
        let capacitance = &self.state.workspace.contributions.capacitance;
        let charge = &self.state.workspace.contributions.charge;
        let mut held_all = Vec::with_capacity(adjoints.len());
        let mut solutions = Vec::with_capacity(adjoints.len());
        let mut periods = Vec::with_capacity(adjoints.len());
        for (seed, adjoint) in pass.seeds.iter().zip(adjoints) {
            let mut held = adjoint.charge;
            add(&mut held, 1.0, &adjoint.before);
            let mut solution = vec![0.0; x.len()];
            if let Some(values) = seed.get(&0) {
                add(&mut solution, 1.0, values);
            }
            add_transposed_product(&self.elaborated.pattern, capacitance, &held, &mut solution);
            let mut period = adjoint.period;
            if let (Some((mut read, before)), Some(stretched)) = (adjoint.stretch, pass.stretch) {
                add(&mut read, 1.0, &before);
                let dot: f64 = read.iter().zip(charge).map(|(a, q)| a * q).sum();
                period -= dot / stretched;
            }
            held_all.push(held);
            solutions.push(solution);
            periods.push(period);
        }
        twins.evaluate(x, Some(0.0), gmin, &options.parallelism);
        fold_dots(
            twins,
            Part::Charge,
            1.0,
            &held_all,
            &options.parallelism,
            fold,
        );
        let Start::Solved(forced) = pass.start else {
            return Ok(solutions.into_iter().zip(periods).collect());
        };
        self.evaluate(x, None, 1.0, gmin, false);
        let mut factors = self.factor_dc_forced(forced)?;
        for solution in &mut solutions {
            factors.solve_transpose(solution)?;
            for unknown in forced {
                solution[unknown.index()] = 0.0;
            }
        }
        twins.evaluate(x, None, gmin, &options.parallelism);
        fold_dots(
            twins,
            Part::Residual,
            -1.0,
            &solutions,
            &options.parallelism,
            fold,
        );
        let zeros = vec![0.0; x.len()];
        Ok(periods
            .into_iter()
            .map(|period| (zeros.clone(), period))
            .collect())
    }
}

/// Checks that every cotangent in `cotangents` has a finite period and finite values of length
/// `unknowns` at valid indices `< points`.
fn cotangents_fit<T: Scalar>(cotangents: &[Cotangent<T>], points: usize, unknowns: usize) -> bool {
    cotangents.iter().all(|cotangent| {
        cotangent.period.is_finite()
            && cotangent.points.iter().all(|(&at, values)| {
                at < points
                    && values.len() == unknowns
                    && values.iter().copied().all(Scalar::is_finite)
            })
    })
}

/// Borrows each cotangent's point-seed map in order.
fn cotangent_seeds(cotangents: &[Cotangent<f64>]) -> Vec<&BTreeMap<usize, Vec<f64>>> {
    cotangents
        .iter()
        .map(|cotangent| &cotangent.points)
        .collect()
}

/// Collects the active `(cotangent_index, values)` pairs present at `point`.
fn seeds_at<'a>(seeds: &[&'a BTreeMap<usize, Vec<f64>>], point: usize) -> Vec<(usize, &'a [f64])> {
    seeds
        .iter()
        .enumerate()
        .filter_map(|(index, seed)| Some((index, seed.get(&point)?.as_slice())))
        .collect()
}

/// Computes `sign * vectorsᵀ ∂(part)/∂p` across `twins` and folds each row into `fold`.
fn fold_dots(
    twins: &mut Twins,
    part: Part,
    sign: f64,
    vectors: &[Vec<f64>],
    parallelism: &Parallelism,
    fold: Fold<'_>,
) {
    let by_twin = twins.dots(part, sign, vectors, parallelism);
    for (index, terms) in transposed(by_twin, vectors.len()).iter().enumerate() {
        fold(index, terms);
    }
}

/// The number of points of `result`: one at an operating point, and a swept value's,
/// frequency's or time's each otherwise.
pub(crate) fn points(result: &AnalysisResult) -> usize {
    match result {
        AnalysisResult::DcSweep(sweep) => sweep.values.len(),
        AnalysisResult::Ac(response) => response.values.len(),
        AnalysisResult::Transient(waveforms) => waveforms.values.len(),
        AnalysisResult::PeriodicSteadyState(found) => found.transient.values.len(),
        _ => 1,
    }
}

/// The derivatives of the unknowns `kept` of `moved`, by direction then unknown, by
/// unknown kept then direction.
fn kept_of(kept: &[usize], moved: &[Vec<f64>]) -> Vec<Vec<f64>> {
    kept.iter()
        .map(|&unknown| moved.iter().map(|shift| shift[unknown]).collect())
        .collect()
}

/// The paths and names of `parameters`, as an error names them.
pub(crate) fn named_of(parameters: &[Parameter]) -> Vec<(String, String)> {
    parameters
        .iter()
        .map(|parameter| (parameter.instance.clone(), parameter.name.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use jolty_numerics::integration::Method;
    use serde_json::json;

    use super::{Cotangent, Cotangents, ParameterCotangents};
    use crate::test_helpers::grounded_simulator;
    use crate::{Ac, AnalysisResult, Differentiated, Options, Simulator, Transient};

    fn divider_simulator() -> Simulator {
        grounded_simulator(
            "test",
            "0",
            &json!([
                {"name": "V1", "of": {"model": "voltage_source"},
                 "parameters": {"waveform": {"dc": 2.0}},
                 "terminals": {"positive": "in", "negative": "0"}},
                {"name": "R1", "of": {"model": "resistor"},
                 "parameters": {"resistance": 1000.0},
                 "terminals": {"a": "in", "b": "out"}},
                {"name": "R2", "of": {"model": "resistor"},
                 "parameters": {"resistance": 1000.0},
                 "terminals": {"a": "out", "b": "0"}},
                {"name": "C1", "of": {"model": "capacitor"},
                 "parameters": {"capacitance": 1e-9},
                 "terminals": {"a": "out", "b": "0"}}
            ]),
        )
    }

    fn factor_calls(simulator: &Simulator) -> usize {
        let work = simulator.state.solver.statistics();
        work.factorizations + work.refactorizations + work.fallbacks + work.reuses
    }

    #[test]
    fn pull_frequencies_early_returns_when_cotangents_is_empty() {
        let mut simulator = divider_simulator();
        let parameters = vec![("R1".to_owned(), "resistance".to_owned())];
        let pullback = simulator
            .vjp(
                &Differentiated::Ac(Ac {
                    source: "V1".to_owned(),
                    frequencies: vec![1e3, 1e4, 1e5],
                }),
                &parameters,
                &Options::default(),
            )
            .expect("AC pullback succeeds");
        let pulled = simulator
            .pull_back(
                &pullback,
                &Cotangents::Complex(Vec::new()),
                &Options::default(),
            )
            .expect("empty AC pull_back succeeds");
        assert_eq!(pulled, ParameterCotangents::Complex(Vec::new()));
    }

    #[test]
    fn backward_skips_inactive_trailing_steps_and_avoids_forced_clone() {
        let mut simulator = divider_simulator();
        let parameters = vec![("R1".to_owned(), "resistance".to_owned())];
        let pullback = simulator
            .vjp(
                &Differentiated::Transient(Transient {
                    stop: 5e-6,
                    max_step: Some(1e-6),
                    method: Method::Trapezoidal,
                    breakpoints: Vec::new(),
                    initial_conditions: BTreeMap::new(),
                    operating_point: true,
                }),
                &parameters,
                &Options::default(),
            )
            .expect("transient pullback succeeds");
        let AnalysisResult::Transient(waveforms) = &pullback.result else {
            panic!("expected transient result");
        };
        assert!(waveforms.values.len() >= 6);
        let unknowns = simulator.elaborated.unknowns.len();
        let out = simulator.unknown("out").expect("out node").index();
        let mut unit = vec![0.0; unknowns];
        unit[out] = 1.0;

        // Seed only at point 0: all backward steps are skipped, leaving only the single
        // DC factorization in `backward_start` (with `forced` empty).
        let before_zero = factor_calls(&simulator);
        let at_zero = simulator
            .pull_back(
                &pullback,
                &Cotangents::Real(vec![Cotangent {
                    points: BTreeMap::from([(0, unit.clone())]),
                    period: 0.0,
                }]),
                &Options::default(),
            )
            .expect("pull_back at point 0 succeeds");
        assert_eq!(factor_calls(&simulator) - before_zero, 1);
        let ParameterCotangents::Real(rows_zero) = at_zero else {
            panic!("expected real cotangents");
        };
        assert!((rows_zero[0][0] + 5e-4).abs() < 1e-12);

        // Seed only at point 1: trailing steps past point 1 are skipped (1 step + 1 DC start).
        let before_one = factor_calls(&simulator);
        let at_one = simulator
            .pull_back(
                &pullback,
                &Cotangents::Real(vec![Cotangent {
                    points: BTreeMap::from([(1, unit)]),
                    period: 0.0,
                }]),
                &Options::default(),
            )
            .expect("pull_back at point 1 succeeds");
        assert_eq!(factor_calls(&simulator) - before_one, 2);
        let ParameterCotangents::Real(rows_one) = at_one else {
            panic!("expected real cotangents");
        };
        assert!((rows_one[0][0] + 5e-4).abs() < 1e-12);
    }
}
