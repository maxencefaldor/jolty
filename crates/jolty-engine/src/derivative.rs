//! Forward (JVP) and reverse (VJP) differentiation of circuit analyses with respect to
//! device parameters.
//!
//! At every analysis point, the circuit satisfies `F(x, p) = 0`, so the implicit function
//! theorem gives `dx/dp = -J⁻¹ ∂F/∂p` with `J = ∂F/∂x`. Dual-number device evaluations give
//! exact `J`, and parameter-seeded device twins evaluate `∂F/∂p` at per-instance cost. JVP
//! computes `ẋ = -J⁻¹ (∂F/∂p) t` in one solve per tangent `t`, whereas VJP computes
//! `p̄ = -(J⁻ᵀ v)ᵀ ∂F/∂p` in one transposed solve per cotangent `v` regardless of parameter
//! count. Both maps reuse one factorization per point and satisfy `⟨v, ẋ⟩ = ⟨p̄, t⟩` up to
//! solve rounding.
//!
//! Operating-point and DC-sweep analyses differentiate DC residuals directly, treating the
//! swept value as fixed. Frequency-domain AC solves `(G + jωC) x = b`, differentiating both
//! direct matrix terms and indirect bias shifts through nonlinear device twins. Reverse mode
//! accumulates `g = ∂/∂x Σ_k λ_kᵀ (G + jω_k C) x_k` and performs a single transposed DC
//! solve `Gᵀ μ = g`. Stability analysis differentiates Middlebrook's double-injection loop
//! gain `T = -u / (1 + u)` (`dT/du = -1 / (1 + u)²`, `u = x_s[entered] - x_i[carried]`) from
//! the same factored `G + jωC` matrix, propagating both injections through the shared
//! operating-point bias path. Phasor pullbacks are bilinear (`vᵀ dx/dp`), so real losses
//! over `x = a + jb` take the real part with `v = ∂L/∂a - j ∂L/∂b`.
//!
//! Transient steps solve `f(x) + α q(x) + h = 0`. JVP runs the forward direct method
//! (Hocevar, Yang, Trick, and Epler, *IEEE Trans. CAD* 4(4), 1985), solving
//! `J ẋ = -(∂f/∂p t + α ∂q/∂p t + ḣ)` at each accepted step. VJP runs the exact discrete
//! adjoint backward across recorded step formulas in one transposed solve per step and
//! cotangent, holding step times and logic events fixed. Initial-state derivatives propagate
//! through the DC Jacobian with forced initial-condition rows replaced by `v - v₀ = 0`, or
//! start at zero when skipping the operating point.
//!
//! Periodic steady-state JVP replays the converged period with a fixed start to obtain
//! `∂φ/∂p t` and solves `(Φ - I) ẋ₀ = -∂φ/∂p t` (bordered with `∂φ/∂T` and `Ṫ` for
//! oscillators). A second pass replays the period from `ẋ₀` with step-stretch terms `-Ṫ / T`.
//! Singular `Φ - I` preserves neutral invariants using the same minimum-norm solve as
//! shooting. VJP runs those three stages in reverse: fixed-start adjoint over the period,
//! transposed bordered solve mapping start and period adjoints to the period end, and a
//! second backward pass from the period end.

use std::collections::BTreeMap;

use jolty_models::UnknownIndex;
use jolty_numerics::{Complex64, Pattern, Scalar, solve_each_in_range};

use crate::ac::over_frequencies;
use crate::newton::{add_product, factor_step_jacobian, force_row};
use crate::options::{Parallelism, float_count};
use crate::periodic::{DENSE_MATRICES, ShootingSystem};
use crate::sensitivity::{Derivatives, Differentiated, Parameter};
use crate::solver::Factors;
use crate::stability::Probe;
use crate::transient::{Carried, StepFormula, StepPoint, Tangent, TangentLinear};
use crate::twins::{Part, Twins};
use crate::{
    AcResult, AnalysisResult, DcSweep, DcSweepResult, Options, PeriodicSteadyStateResult,
    SimulationError, Simulator, StabilityResult, Transient, TransientResult,
};

/// Analysis label for frequency-domain differentiation in delay-line refusals.
const OVER_FREQUENCY: &str = "derivative over frequency";
/// Explanation for refusing delay lines in frequency-domain differentiation.
const LEFT_OUT_OVER_FREQUENCY: &str =
    "frequency-domain differentiation omits derivatives of delayed transmission-line terms";
/// Analysis label for time-domain differentiation in delay-line refusals.
const OVER_TIME: &str = "derivative over time";
/// Explanation for refusing delay lines in time-domain differentiation.
const LEFT_OUT_OVER_TIME: &str =
    "time-domain differentiation omits derivatives of historical transmission-line samples";
/// Analysis label for periodic steady-state sensitivity in [`SimulationError::TooMuchMemory`].
pub(crate) const PERIODIC_SENSITIVITY: &str = "periodic steady state sensitivity";

/// Error text for cotangent domain mismatches in [`Simulator::pull_back`].
const KINDS: &str =
    "cotangents on an AC or stability analysis are complex, and on any other analysis real";

/// Result of [`Simulator::jvp`], pairing the primal analysis result with directional
/// derivatives across every output and analysis point.
#[derive(Debug, Clone, PartialEq)]
pub struct JvpResult {
    /// Differentiated parameters in tangent order along with their current values.
    pub parameters: Vec<Parameter>,
    /// Directional derivatives indexed by `[point][output][tangent]` (real values or complex
    /// AC and stability phasors).
    pub tangents: Derivatives,
    /// Period derivatives along each tangent for an autonomous periodic steady state.
    pub period: Option<Vec<f64>>,
    /// Primal result of the differentiated analysis.
    pub result: AnalysisResult,
}

/// Linear functional (cotangent) over all outputs at selected points of an analysis, such as
/// the gradient of a scalar loss with respect to simulation outputs.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Cotangent<T> {
    /// Nonzero per-output cotangent vectors keyed by analysis point index.
    pub points: BTreeMap<usize, Vec<T>>,
    /// Cotangent weight on an autonomous oscillator's period; zero for other analyses.
    pub period: f64,
}

/// Collection of real or complex cotangents matching an analysis output type.
#[derive(Debug, Clone, PartialEq)]
pub enum Cotangents {
    /// Cotangents for real-valued analyses (DC, sweep, transient, or periodic steady state).
    Real(Vec<Cotangent<f64>>),
    /// Cotangents for complex AC and stability analyses.
    Complex(Vec<Cotangent<Complex64>>),
}

/// Pulled-back parameter cotangents returned by [`Simulator::pull_back`], indexed by
/// `[cotangent][parameter]`.
#[derive(Debug, Clone, PartialEq)]
pub enum ParameterCotangents {
    /// Pullback results for real-valued analyses.
    Real(Vec<Vec<f64>>),
    /// Pullback results for complex AC and stability analyses.
    Complex(Vec<Vec<Complex64>>),
}

/// Captured forward state from [`Simulator::vjp`] enabling reverse-mode pullbacks via
/// [`Simulator::pull_back`].
#[derive(Debug, Clone, PartialEq)]
pub struct Pullback {
    /// Target parameters in pullback order along with their current values.
    pub parameters: Vec<Parameter>,
    /// Primal result of the differentiated analysis.
    pub result: AnalysisResult,
    /// Auxiliary analysis metadata retained for reverse replay.
    saved: Saved,
}

/// Auxiliary analysis state saved alongside primal outputs for [`Simulator::pull_back`].
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Saved {
    /// No auxiliary state (operating point, AC, or stability analysis).
    Nothing,
    /// DC sweep configuration needed to replay held sweep values.
    DcSweep(DcSweep),
    /// Initial-state boundary mode and accepted step formulas for transient replay.
    Transient(Start, Vec<StepFormula>),
    /// Accepted period step formulas and bordered shooting system for periodic replay.
    Period(Vec<StepFormula>, ShootingSystem),
}

/// Borrowed transient waveforms paired with the step formulas that produced them.
#[derive(Clone, Copy)]
pub(crate) struct Trajectory<'a> {
    pub(crate) waveforms: &'a TransientResult,
    pub(crate) steps: &'a [StepFormula],
}

/// Borrowed periodic steady-state result paired with its step formulas and shooting system.
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

/// Period mode distinguishing forced-period circuits from autonomous oscillators.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PeriodKind {
    Forced,
    Free,
}

impl PeriodKind {
    /// Returns [`Self::Free`] when shooting bordered the system with a period scale, or
    /// [`Self::Forced`] otherwise.
    pub(crate) const fn from_scale(scale: Option<f64>) -> Self {
        match scale {
            Some(_) => Self::Free,
            None => Self::Forced,
        }
    }
}

/// Boundary condition for transient sensitivities at time zero.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Start {
    /// Operating-point start listing node unknowns held fixed by initial conditions.
    Solved(Vec<UnknownIndex>),
    /// Fixed initial state with optional precomputed starting tangent shifts (zeros if `None`).
    Held(Option<Vec<Vec<f64>>>),
}

/// Counts the total `f64` values stored in `waveforms` across unknown vectors and recorded
/// series.
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

/// Estimates peak `f64` storage during [`Simulator::along_period`] across the bordered solve
/// (`DENSE_MATRICES` square matrices plus `directions` columns) and forward replay passes.
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

/// Estimates peak `f64` storage during [`Simulator::pull_period`] plus caller buffers of
/// `budget.extra` numbers (`unit_rows` and `pulled`, or `found` in [`Simulator::pull_back`]).
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
    // alongside `DENSE_MATRICES`, and `ends` holds one vector per cotangent.
    let solve_phase =
        (DENSE_MATRICES + 1.0) * rows_float * rows_float + cotangents * unknowns_float;
    // Each cotangent retains 3 state vectors (plus 2 for autonomous period stretch) and up to
    // 4 temporary vectors per step (`ended` adds 1 in pass 2). Contraction buffers hold up to
    // `3 * cotangents * parameters` alongside `6` (or `7`) unknown vectors.
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

/// Constructs `count` standard basis unit vectors of length `count`.
pub(crate) fn units(count: usize) -> Vec<Vec<f64>> {
    (0..count)
        .map(|index| {
            let mut unit = vec![0.0; count];
            unit[index] = 1.0;
            unit
        })
        .collect()
}

/// Accumulates `scale * x` into `into` elementwise.
fn add(into: &mut [f64], scale: f64, x: &[f64]) {
    for (value, term) in into.iter_mut().zip(x) {
        *value += scale * term;
    }
}

/// Accumulates the transposed sparse matrix-vector product `Aᵀ * x` into `into` in row order.
fn add_transposed_product(pattern: &Pattern, values: &[f64], x: &[f64], into: &mut [f64]) {
    for (row, &factor) in x.iter().enumerate() {
        for &(slot, column) in pattern.row(row) {
            into[column] += values[slot] * factor;
        }
    }
}

/// Transposes a `rows × columns` nested vector into `columns × rows`.
fn transposed<T: Copy>(rows: Vec<Vec<T>>, columns: usize) -> Vec<Vec<T>> {
    let mut turned = vec![Vec::with_capacity(rows.len()); columns];
    for row in rows {
        for (column, value) in turned.iter_mut().zip(row) {
            column.push(value);
        }
    }
    turned
}

/// Forward direct-method state tracking charge tangents and recorded output sensitivities
/// across a transient or periodic replay.
pub(crate) struct Forward {
    pub(crate) twins: Twins,
    /// Tangent directions indexed by `[direction][twin]`.
    directions: Vec<Vec<f64>>,
    /// Unknown indices whose sensitivities are recorded at each accepted point.
    kept: Vec<usize>,
    start: Start,
    tangents: Vec<Tangent>,
    /// Recorded directional derivatives indexed by `[point][kept_output][direction]`.
    pub(crate) found: Vec<Vec<Vec<f64>>>,
    /// Autonomous oscillator period paired with its per-direction derivatives scaling each
    /// step.
    pub(crate) stretch: Option<(f64, Vec<f64>)>,
    /// Raw charge trajectory used to compute step-scaled divided terms when stretching period.
    charges: Option<Tangent>,
    /// Endpoint unknown sensitivities retained per direction for periodic shooting replays.
    ends: Option<Vec<Vec<f64>>>,
}

impl Forward {
    /// Initializes forward sensitivity state for `twins` along `directions`, recording
    /// unknowns `kept` from boundary condition `start`.
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

    /// Extracts derivatives for `self.kept` unknowns from `moved`, transposing to
    /// `[kept_output][direction]`.
    fn keep(&self, moved: &[Vec<f64>]) -> Vec<Vec<f64>> {
        kept_of(&self.kept, moved)
    }
}

/// Internal JVP output pairing final twins, recorded derivatives, optional period
/// derivatives, and the primal analysis result.
pub(crate) struct Pushed {
    pub(crate) twins: Twins,
    pub(crate) tangents: Derivatives,
    pub(crate) period: Option<Vec<f64>>,
    pub(crate) result: AnalysisResult,
}

/// Reverse-mode adjoint state carried backward across transient steps for one cotangent.
struct Adjoint {
    charge: Vec<f64>,
    before: Vec<f64>,
    rate: Vec<f64>,
    stretch: Option<(Vec<f64>, Vec<f64>)>,
    period: f64,
    /// Latest point index with a nonzero cotangent seed, skipping inactive trailing steps.
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

/// Nonlinear operating-point Jacobian shift `(slot, ∂G/∂x · s, ∂C/∂x · s)`.
type EntryShift = (usize, f64, f64);

/// Shared sparse-matrix and twin views used to assemble `-(dG + jω dC) x` at one frequency.
#[derive(Clone, Copy)]
struct FrequencyTangentContext<'a> {
    twins: &'a Twins,
    positions: &'a [(usize, usize)],
    dimension: usize,
    omega: f64,
}

impl FrequencyTangentContext<'_> {
    /// Assembles `-(dG + jω dC) x` along `direction` with nonlinear operating-point shifts
    /// `entries`.
    fn right_hand_side(
        self,
        direction: &[f64],
        entries: &[EntryShift],
        phasors: &[Complex64],
    ) -> Vec<Complex64> {
        let mut moved = vec![Complex64::default(); self.dimension];
        for (column, &along) in direction.iter().enumerate() {
            if along != 0.0 {
                self.twins.add_product(
                    column,
                    self.positions,
                    along,
                    self.omega,
                    phasors,
                    &mut moved,
                );
            }
        }
        for &(slot, conductance_rate, capacitance_rate) in entries {
            let (row, column) = self.positions[slot];
            let entry = Complex64::new(conductance_rate, self.omega * capacitance_rate);
            moved[row] += entry * phasors[column];
        }
        for value in &mut moved {
            *value = -*value;
        }
        moved
    }
}

/// Per-frequency pullback terms accumulated across active cotangents in
/// [`Simulator::pull_stability`].
struct StabilityFrequencyPull {
    omega: f64,
    weighted: Vec<(usize, Complex64)>,
    by_twin: Vec<Complex64>,
    slot_products: Vec<Complex64>,
}

/// Shared views used to pull back active loop-gain cotangents at a single frequency.
#[derive(Clone, Copy)]
struct StabilityPullContext<'a> {
    probe: Probe,
    level: &'a [f64],
    twins: &'a Twins,
    positions: &'a [(usize, usize)],
    nonlinear_slots: &'a [usize],
    cotangents: &'a [Cotangent<Complex64>],
}

impl StabilityPullContext<'_> {
    /// Solves the double-injection primal and unit transposed systems at `index` and
    /// `frequency` when at least one cotangent is active there.
    fn at_frequency(
        self,
        factors: &mut Factors<'_, Complex64>,
        index: usize,
        frequency: f64,
    ) -> Result<Option<StabilityFrequencyPull>, SimulationError> {
        let active: Vec<(usize, Complex64)> = self
            .cotangents
            .iter()
            .enumerate()
            .filter_map(|(cotangent, given)| {
                let &value = given.points.get(&index)?.first()?;
                (value != Complex64::default()).then_some((cotangent, value))
            })
            .collect();
        if active.is_empty() {
            return Ok(None);
        }
        let injection = self.probe.inject(factors, self.level)?;
        let denominator = 1.0 + injection.ratio;
        let gain_rate = Complex64::new(-1.0, 0.0) / (denominator * denominator);
        let dimension = self.level.len();
        let mut adjoint_series = vec![Complex64::default(); dimension];
        adjoint_series[self.probe.entered.index()] = Complex64::new(1.0, 0.0);
        factors.solve_transpose(&mut adjoint_series)?;
        let mut adjoint_shunt = vec![Complex64::default(); dimension];
        adjoint_shunt[self.probe.carried.index()] = Complex64::new(1.0, 0.0);
        factors.solve_transpose(&mut adjoint_shunt)?;
        let omega = std::f64::consts::TAU * frequency;
        let by_twin: Vec<Complex64> = (0..self.twins.len())
            .map(|column| {
                let shunt = self.twins.product(
                    column,
                    self.positions,
                    &adjoint_shunt,
                    omega,
                    &injection.shunt,
                );
                let series = self.twins.product(
                    column,
                    self.positions,
                    &adjoint_series,
                    omega,
                    &injection.series,
                );
                shunt - series
            })
            .collect();
        let slot_products: Vec<Complex64> = self
            .nonlinear_slots
            .iter()
            .map(|&slot| {
                let (row, column) = self.positions[slot];
                adjoint_series[row] * injection.series[column]
                    - adjoint_shunt[row] * injection.shunt[column]
            })
            .collect();
        let weighted = active
            .into_iter()
            .map(|(cotangent, value)| (cotangent, value * gain_rate))
            .collect();
        Ok(Some(StabilityFrequencyPull {
            omega,
            weighted,
            by_twin,
            slot_products,
        }))
    }
}

impl Adjoint {
    /// Propagates adjoint state backward across `step` using `vectors` and `charge`, updating
    /// period-stretch terms when `stretch` is set and returning `μ - α λ`.
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

/// Boundary adjoints `(state_adjoint, period_adjoint)` produced at time zero for each
/// cotangent.
type Ends = Vec<(Vec<f64>, f64)>;

/// Callback receiving additive parameter-cotangent contributions `(cotangent_index, terms)`.
pub(crate) type Fold<'a> = &'a mut dyn FnMut(usize, &[f64]);

impl Simulator {
    /// Computes exact Jacobian-vector products (JVPs) of `analysis` along `tangents` with
    /// respect to `parameters` (`(device_path, parameter_name)` pairs).
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::UnknownParameters`] for missing or non-numeric parameters,
    /// [`SimulationError::InvalidAnalysis`] for malformed or non-finite tangents,
    /// [`SimulationError::TooMuchMemory`] if periodic sensitivity exceeds `max_memory`,
    /// [`SimulationError::Singular`] for singular Jacobians, and errors from `analysis`.
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
            let kept: Vec<usize> = if matches!(analysis, Differentiated::Stability(_)) {
                vec![0]
            } else {
                (0..self.elaborated.unknowns.len()).collect()
            };
            let pushed = self.push_forward(analysis, twins, directions, &kept, options)?;
            Ok(JvpResult {
                parameters: pushed.twins.parameters,
                tangents: pushed.tangents,
                period: pushed.period,
                result: pushed.result,
            })
        })
    }

    /// Runs `analysis` and returns a [`Pullback`] handle that maps output cotangents back to
    /// `parameters` via [`Simulator::pull_back`] in one transposed solve per cotangent.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::UnknownParameters`] for invalid parameters,
    /// [`SimulationError::Delayed`] for AC or transient circuits with delay lines,
    /// [`SimulationError::InvalidAnalysis`] for transient circuits with logic, and errors
    /// from `analysis`.
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

    /// Evaluates reverse-mode vector-Jacobian products for `cotangents` against `pullback`,
    /// returning parameter sensitivities indexed by `[cotangent][parameter]`.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::InvalidAnalysis`] for domain-mismatched, out-of-range,
    /// wrong-length, or non-finite cotangents, [`SimulationError::TooMuchMemory`] if periodic
    /// pullback exceeds `max_memory`, or [`SimulationError::Singular`] for singular Jacobians.
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
                (_, AnalysisResult::Stability(loop_gain), Cotangents::Complex(cotangents)) => {
                    Ok(ParameterCotangents::Complex(self.pull_stability(
                        &mut twins, loop_gain, cotangents, options,
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

    /// Builds unit-seeded parameter [`Twins`] for `parameters` in order.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::UnknownParameters`] listing every missing or non-numeric
    /// parameter.
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

    /// Resolves `(device_path, parameter_name)` pairs to `(device_index, Parameter)` entries
    /// reflecting current active device values.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::UnknownParameters`] listing every unresolved entry.
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

    /// Validates that `cotangents` have finite entries matching the point and output counts
    /// of `result`.
    fn checked(
        &self,
        result: &AnalysisResult,
        cotangents: &Cotangents,
    ) -> Result<(), SimulationError> {
        let points = points(result);
        let (outputs, label) = if matches!(result, AnalysisResult::Stability(_)) {
            (1, "output")
        } else {
            (self.elaborated.unknowns.len(), "unknowns")
        };
        let fit = match cotangents {
            Cotangents::Real(each) => cotangents_fit(each, points, outputs),
            Cotangents::Complex(each) => cotangents_fit(each, points, outputs),
        };
        if fit {
            Ok(())
        } else {
            Err(SimulationError::InvalidAnalysis(format!(
                "a cotangent is finite, at points from 0 to {} of {outputs} {label} each",
                points.saturating_sub(1)
            )))
        }
    }

    /// Computes the forward JVP of `analysis` along `directions` over `twins`, recording only
    /// `kept` unknowns.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Simulator::jvp`].
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
            Differentiated::Stability(stability) => {
                self.undelayed(OVER_FREQUENCY, LEFT_OUT_OVER_FREQUENCY)?;
                let loop_gain = self.stability(stability, options)?;
                let found =
                    self.stability_tangents(&mut twins, &loop_gain, &directions, kept, options)?;
                (
                    Derivatives::Complex(found),
                    AnalysisResult::Stability(loop_gain),
                )
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

    /// Runs `analysis` and captures the [`Saved`] replay state required for [`Simulator::pull_back`].
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Simulator::vjp`].
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
            Differentiated::Stability(stability) => {
                self.undelayed(OVER_FREQUENCY, LEFT_OUT_OVER_FREQUENCY)?;
                (
                    AnalysisResult::Stability(self.stability(stability, options)?),
                    Saved::Nothing,
                )
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

    /// Verifies that time-domain adjoint replay is supported, rejecting circuits with delay
    /// lines or logic nets.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::Delayed`] or [`SimulationError::InvalidAnalysis`].
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

    /// Determines the initial-state sensitivity boundary condition [`Start`] for `transient`.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::InvalidAnalysis`] for unknown initial-condition nets or
    /// non-finite voltages.
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

    /// Pulls back `cotangents` on `result` (with replay state `saved`) to every twin,
    /// returning accumulated sums by cotangent then twin.
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

    /// Pulls back real `cotangents` over `result` and `saved` to `twins`, streaming each
    /// additive contribution to `fold`.
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

    /// Factors the workspace DC conductance matrix, replacing `forced` rows with `v - v₀ = 0`
    /// when non-empty.
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

    /// Solves directional DC shifts `-J⁻¹ Σ t_j ∂F/∂p_j` at `x` with `forced` node rows held
    /// fixed, leaving `twins` evaluated at `x`.
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

    /// Pulls back active DC cotangents `at` at `x`, folding `-λᵀ ∂F/∂p` (where `Jᵀ λ = v`)
    /// into `fold`.
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

    /// Evaluates `each` across every point of `sweep` in `result`, holding the swept source
    /// or parameter fixed and rebuilding its affected twins at each step.
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

    /// Prepares device and twin evaluations at `operating` and returns per-direction nonlinear
    /// Jacobian shifts `(slot, ∂G/∂x · s, ∂C/∂x · s)` along `directions`.
    fn frequency_biases(
        &mut self,
        twins: &mut Twins,
        operating: &[f64],
        directions: &[Vec<f64>],
        options: &Options,
    ) -> Result<Vec<Vec<EntryShift>>, SimulationError> {
        let nonlinear = self.nonlinear_devices();
        if nonlinear.is_empty() {
            self.evaluate(operating, None, 1.0, options.gmin, false);
            twins.evaluate(operating, None, options.gmin, &options.parallelism);
            Ok(vec![Vec::new(); directions.len()])
        } else {
            let shifts = self.dc_shifts(twins, operating, directions, &[], options)?;
            Ok(self.entries_along(
                &nonlinear,
                operating,
                &shifts,
                options.gmin,
                &options.parallelism,
            ))
        }
    }

    /// Computes forward directional AC phasor sensitivities for `kept` unknowns along
    /// `directions`, indexed by `[frequency][kept_output][direction]`.
    fn frequency_tangents(
        &mut self,
        twins: &mut Twins,
        response: &AcResult,
        directions: &[Vec<f64>],
        kept: &[usize],
        options: &Options,
    ) -> Result<Vec<Vec<Vec<Complex64>>>, SimulationError> {
        let operating = &response.operating_point.values;
        let biases = self.frequency_biases(twins, operating, directions, options)?;
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
                let context = FrequencyTangentContext {
                    twins,
                    positions: &positions,
                    dimension,
                    omega: std::f64::consts::TAU * frequency,
                };
                let mut at = Vec::with_capacity(directions.len());
                for (direction, entries) in directions.iter().zip(&biases) {
                    let mut moved = context.right_hand_side(direction, entries, phasors);
                    factors.solve(&mut moved)?;
                    at.push(kept.iter().map(|&unknown| moved[unknown]).collect());
                }
                Ok(transposed(at, kept.len()))
            },
        )
    }

    /// Computes forward directional loop-gain sensitivities along `directions` for `kept`
    /// outputs, indexed by `[frequency][kept_output][direction]`.
    fn stability_tangents(
        &mut self,
        twins: &mut Twins,
        loop_gain: &StabilityResult,
        directions: &[Vec<f64>],
        kept: &[usize],
        options: &Options,
    ) -> Result<Vec<Vec<Vec<Complex64>>>, SimulationError> {
        if kept.is_empty() || directions.is_empty() {
            return Ok(vec![
                vec![Vec::new(); kept.len()];
                loop_gain.frequencies.len()
            ]);
        }
        let probe = self.probe(&loop_gain.probe)?;
        let level = self.source_excitation(probe.device);
        let operating = &loop_gain.operating_point.values;
        let biases = self.frequency_biases(twins, operating, directions, options)?;
        let positions: Vec<(usize, usize)> = self.elaborated.pattern.positions().collect();
        let twins = &*twins;
        let dimension = self.elaborated.unknowns.len();
        over_frequencies(
            self.linearization(),
            &self.state.workspace.contributions,
            &loop_gain.frequencies,
            &options.parallelism,
            &|factors, _, frequency| {
                let injection = probe.inject(factors, &level)?;
                let denominator = 1.0 + injection.ratio;
                let gain_rate = Complex64::new(-1.0, 0.0) / (denominator * denominator);
                let context = FrequencyTangentContext {
                    twins,
                    positions: &positions,
                    dimension,
                    omega: std::f64::consts::TAU * frequency,
                };
                let mut at = Vec::with_capacity(directions.len());
                for (direction, entries) in directions.iter().zip(&biases) {
                    let mut series = context.right_hand_side(direction, entries, &injection.series);
                    factors.solve(&mut series)?;
                    let mut shunt = context.right_hand_side(direction, entries, &injection.shunt);
                    factors.solve(&mut shunt)?;
                    let ratio_rate = series[probe.entered.index()] - shunt[probe.carried.index()];
                    at.push(gain_rate * ratio_rate);
                }
                Ok(vec![at; kept.len()])
            },
        )
    }

    /// Pulls back complex `cotangents` over `response` to `twins`, accumulating frequency
    /// contributions in order and folding indirect bias shifts through one transposed DC solve.
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

    /// Pulls back complex `cotangents` over `loop_gain` to `twins`, accumulating frequency
    /// contributions in order and folding indirect bias shifts through one transposed DC solve.
    pub(crate) fn pull_stability(
        &mut self,
        twins: &mut Twins,
        loop_gain: &StabilityResult,
        cotangents: &[Cotangent<Complex64>],
        options: &Options,
    ) -> Result<Vec<Vec<Complex64>>, SimulationError> {
        if cotangents.is_empty() {
            return Ok(Vec::new());
        }
        if twins.len() == 0 {
            return Ok(vec![Vec::new(); cotangents.len()]);
        }
        let probe = self.probe(&loop_gain.probe)?;
        let level = self.source_excitation(probe.device);
        let operating = &loop_gain.operating_point.values;
        let parallelism = &options.parallelism;
        self.evaluate(operating, None, 1.0, options.gmin, false);
        twins.evaluate(operating, None, options.gmin, parallelism);
        let positions: Vec<(usize, usize)> = self.elaborated.pattern.positions().collect();
        let nonlinear = self.nonlinear_devices();
        let (mut unknown_twins, nonlinear_slots) = if nonlinear.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            self.unknown_twins(&nonlinear)
        };
        let found = {
            let context = StabilityPullContext {
                probe,
                level: &level,
                twins,
                positions: &positions,
                nonlinear_slots: &nonlinear_slots,
                cotangents,
            };
            over_frequencies(
                self.linearization(),
                &self.state.workspace.contributions,
                &loop_gain.frequencies,
                parallelism,
                &|factors, index, frequency| context.at_frequency(factors, index, frequency),
            )?
        };
        let mut sums = vec![vec![Complex64::default(); twins.len()]; cotangents.len()];
        let mut weights: Vec<Option<(Vec<Complex64>, Vec<Complex64>)>> =
            vec![None; cotangents.len()];
        accumulate_stability_pulls(found, nonlinear_slots.len(), &mut sums, &mut weights);
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
    /// [`Simulator::pull_frequencies`] and [`Simulator::pull_stability`].
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

    /// Computes forward directional sensitivities across `period` by replaying once from a
    /// fixed start, solving the bordered shooting sensitivity system, and replaying from the
    /// shifted start and optional period derivative.
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
        // (Φ - I) ẋ₀ + ∂φ/∂T Ṫ = -∂φ/∂p t, in shooting's units, bordered for an
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

    /// Replays `forward` across every accepted point of `trajectory`, matching the forward
    /// transient steps.
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

    /// Pulls back `cotangents` across `period` to `twins` via `fold`, running a fixed-start
    /// adjoint pass, a transposed bordered solve, and a second adjoint pass from the period
    /// end.
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

    /// Initializes `forward` tangents at time-zero state `x`, solving initial DC shifts (with
    /// forced initial-condition rows held) or applying precomputed held shifts.
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
    /// terms the steps divide, `α q_{n+1} + β₀ q_n + β₁ q_{n-1}`, times `-Ṫ / T`.
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

    /// Runs the discrete adjoint backward across `pass.trajectory` for all `pass.seeds`
    /// simultaneously, streaming twin terms to `fold` and returning start-state and period
    /// boundary adjoints.
    ///
    /// At step `k` with cotangent seed `c_k` and `μ = λ_q + α λ_r` (plus `λ_q'` when
    /// restarting):
    ///
    /// ```text
    /// (G + α C)ᵀ λ = c_k + Cᵀ μ,          G and C evaluated at x_k
    /// p̄ += -λᵀ ∂f/∂p + (μ - α λ)ᵀ ∂q/∂p
    /// λ_q ← β₀ (λ_r - λ), plus λ_q' when not restarting
    /// λ_q' ← β₁ (λ_r - λ),  λ_r ← γ (λ_r - λ)
    /// ```
    ///
    /// At point `0`, `μ = λ_q + λ_q'` contributes `μᵀ ∂q/∂p`, and solved starts perform one
    /// transposed DC solve `Jᵀ λ = c_0 + Cᵀ μ` (with forced rows held) contributing
    /// `-λᵀ ∂F/∂p`. When `pass.stretch` is active, step-stretch terms accumulate into the
    /// period adjoint `T̄`.
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

    /// Completes the backward adjoint pass at point `0` of `pass.trajectory` from accumulated
    /// `adjoints`, folding initial charge and DC-operating-point terms into `fold`.
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

/// Accumulates per-frequency stability pullback terms into parameter cotangent `sums` and
/// nonlinear operating-point slot `weights`.
fn accumulate_stability_pulls(
    found: Vec<Option<StabilityFrequencyPull>>,
    slot_count: usize,
    sums: &mut [Vec<Complex64>],
    weights: &mut [Option<(Vec<Complex64>, Vec<Complex64>)>],
) {
    for pull in found.into_iter().flatten() {
        for (cotangent, weight) in pull.weighted {
            for (total, &term) in sums[cotangent].iter_mut().zip(&pull.by_twin) {
                *total += weight * term;
            }
            if slot_count == 0 {
                continue;
            }
            let (weight_conductance, weight_capacitance) =
                weights[cotangent].get_or_insert_with(|| {
                    (
                        vec![Complex64::default(); slot_count],
                        vec![Complex64::default(); slot_count],
                    )
                });
            for (packed, &slot_product) in pull.slot_products.iter().enumerate() {
                let product = weight * slot_product;
                weight_conductance[packed] += product;
                weight_capacitance[packed] +=
                    Complex64::new(-pull.omega * product.im, pull.omega * product.re);
            }
        }
    }
}

/// Checks that every cotangent in `cotangents` has a finite period and finite values of length
/// `outputs` at valid indices `< points`.
fn cotangents_fit<T: Scalar>(cotangents: &[Cotangent<T>], points: usize, outputs: usize) -> bool {
    cotangents.iter().all(|cotangent| {
        cotangent.period.is_finite()
            && cotangent.points.iter().all(|(&at, values)| {
                at < points
                    && values.len() == outputs
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

/// Returns the number of output points in `result` (1 for operating point, or the sweep,
/// frequency, or time sample count).
pub(crate) fn points(result: &AnalysisResult) -> usize {
    match result {
        AnalysisResult::DcSweep(sweep) => sweep.values.len(),
        AnalysisResult::Ac(response) => response.values.len(),
        AnalysisResult::Stability(loop_gain) => loop_gain.gains.len(),
        AnalysisResult::Transient(waveforms) => waveforms.values.len(),
        AnalysisResult::PeriodicSteadyState(found) => found.transient.values.len(),
        _ => 1,
    }
}

/// Extracts `kept` unknown entries from directional shifts `moved`, transposing to
/// `[kept_output][direction]`.
fn kept_of(kept: &[usize], moved: &[Vec<f64>]) -> Vec<Vec<f64>> {
    kept.iter()
        .map(|&unknown| moved.iter().map(|shift| shift[unknown]).collect())
        .collect()
}

/// Maps `parameters` to `(instance_path, parameter_name)` pairs for error reporting.
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
