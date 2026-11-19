//! Periodic steady-state analysis via shooting.
//!
//! A circuit driven by periodic sources of period `T` settles onto a repeating state `x₀`
//! satisfying `φ(x₀) = x₀`, where `φ` integrates the circuit across one period. Shooting
//! solves this boundary-value problem by Newton iteration (Aprille and Trick, *Proceedings
//! of the IEEE* 60(1), 1972). Each simulated period propagates sensitivities with respect to
//! `x₀` to form the monodromy matrix `Φ = ∂φ/∂x₀`, then updates `x₀` by solving
//! `(Φ - I) Δ = x₀ - φ(x₀)`. Starting from the operating point with a backward Euler first
//! step lets lightly damped circuits converge in a few periods rather than many transient
//! cycles.
//!
//! Circuits can contain neutral directions, such as an unperturbed memristor state or
//! floating capacitor charge, where `uᵀ Φ = uᵀ` makes `Φ - I` singular. Taking the
//! minimum-norm least-squares step `Δ` in the range of `Φ - I` preserves every neutral
//! invariant from the starting point and reduces to ordinary Newton iteration whenever
//! `Φ - I` is regular. Equations are scaled by per-unknown tolerances, treating singular
//! values within rounding of the largest as zero.
//!
//! Autonomous oscillators have an unknown period `T`, yielding `n` equations `φ(x₀, T) = x₀`
//! in `n + 1` unknowns. Time translation along the orbit produces a neutral direction along
//! the flow `ẋ = ∂φ/∂T`, removed by the orthogonal phase condition `ẋᵀ Δx = 0` in
//! tolerance-scaled units (R. Seydel, *Practical Bifurcation and Stability Analysis*, 3rd ed.,
//! Springer, 2010, §7.1). Newton iteration solves the bordered system
//! `[[Φ - I, ∂φ/∂T], [∂φ/∂Tᵀ, 0]] [Δx, ΔT] = [x₀ - φ(x₀, T), 0]`. Propagating `∂x/∂T`
//! treats every step length as a fixed fraction of `T`, scaling divided charge terms by
//! `∂(dq/dt)/∂T = -(α q_{n+1} + β₀ q_n + β₁ q_{n-1}) / T`.
//!
//! Autonomous shooting cannot start at the unstable DC operating point because every period
//! repeats equilibrium. Startup perturbs the operating point by one tolerance along the
//! fastest-growing natural mode and integrates doubling transient stretches until hyperplane
//! returns stop growing. Each stretch limits steps to `period / 50` using the latest return
//! interval and hands the final return state and period to shooting.

use std::collections::BTreeMap;
use std::f64::consts::TAU;

use jolty_numerics::integration::Method;
use jolty_numerics::{Complex64, solve_in_range};
use serde::Deserialize;

use crate::newton::{NAMED_UNKNOWNS, factor_step_jacobian};
use crate::options::NUMBERS_PER_GIB;
use crate::transient::{Carried, STEPS_PER_RUN, StepFormula, StepPoint, Tangent, TangentLinear};
use crate::{
    OperatingPointResult, Options, SimulationError, Simulator, Strategy, Transient, TransientResult,
};

/// Maximum shooting periods and oscillator settling stretches before aborting.
///
/// Derivation: doubling stretches reach `2³⁹` (`~5.5e11`) times the initial period, past the
/// `10¹¹` dynamic range (`MIN_STEP_FRACTION`) of a single transient.
const MAX_PERIODS: usize = 40;

/// Peak count of dense `n × n` (or `(n + 1) × (n + 1)`) matrices live simultaneously during
/// singular shooting solves.
///
/// Derivation: accounts for the system matrix, primary and reduced SVD factors, the reduced
/// system, and `faer` SVD workspace (`~7.3` matrices), dominating the 4 matrices held during
/// period integration (three tangent vectors per unknown plus `Φ`).
pub(crate) const DENSE_MATRICES: f64 = 14.0;

/// Configuration for periodic steady-state analysis found by shooting.
///
/// Serializes in JSON as `{"periodic_steady_state": {"period": 1e-6}}` for driven circuits or
/// `{"periodic_steady_state": {}}` for autonomous oscillators.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PeriodicSteadyState {
    /// Source period in seconds, or `None` for an autonomous oscillator with an unknown period.
    #[serde(default)]
    pub period: Option<f64>,
    /// Maximum allowed step length in seconds; defaults to `period / 50`.
    #[serde(default)]
    pub max_step: Option<f64>,
    /// Integration method used between breakpoints; defaults to trapezoidal.
    #[serde(default, with = "crate::MethodSerde")]
    pub method: Method,
    /// Additional times within a fixed period in seconds that the solver hits exactly without
    /// restarting integration; empty by default.
    #[serde(default)]
    pub breakpoints: Vec<f64>,
}

/// Converged periodic steady-state waveform spanning one full period.
#[derive(Debug, Clone, PartialEq)]
pub struct PeriodicSteadyStateResult {
    /// Period in seconds (configured for driven circuits, or solved for oscillators).
    pub period: f64,
    /// Number of shooting periods simulated to converge, including the final period in
    /// `transient`.
    pub periods: usize,
    /// Converged single-period waveform from time zero to `period`.
    ///
    /// Solver work in `transient.statistics` aggregates every shooting period, the operating
    /// point, and any oscillator settling transient.
    pub transient: TransientResult,
}

/// Forward sensitivities propagated across one transient period with respect to initial state
/// (and period for oscillators).
#[derive(Default)]
pub(crate) struct Monodromy {
    tangents: Vec<Tangent>,
    /// Columns of `∂x/∂x₀`, forming the monodromy matrix `Φ` at the end of the period.
    at: Vec<Vec<f64>>,
    /// Period-stretch sensitivity state for autonomous oscillators.
    stretch: Option<Stretch>,
    /// Accepted step formulas and restart flags recorded in order for sensitivity replay.
    steps: Vec<StepFormula>,
}

/// Period-derivative tracking for an autonomous oscillator with step lengths proportional to
/// period `T`.
struct Stretch {
    period: f64,
    /// Raw charge trajectory used to compute step-scaled divided charge terms.
    charges: Tangent,
    /// Charge sensitivity `∂q/∂T` propagated across steps.
    tangent: Tangent,
    /// State sensitivity `∂x/∂T` at the latest point, yielding orbit flow at period end.
    at: Vec<f64>,
}

impl Monodromy {
    /// Initializes monodromy tracking with autonomous period-derivative state for `period`.
    fn stretched(period: f64) -> Self {
        Self {
            stretch: Some(Stretch {
                period,
                charges: Tangent::new(Vec::new()),
                tangent: Tangent::new(Vec::new()),
                at: Vec::new(),
            }),
            ..Self::default()
        }
    }
}

/// Period-end monodromy state paired with unknown tolerances and period scaling.
pub(crate) struct Shot {
    monodromy: Monodromy,
    /// Per-unknown tolerances evaluated across period start and end.
    tolerances: Vec<f64>,
    /// Period scaling factor `relative_tolerance * period`.
    scale: f64,
}

/// Tolerance-scaled shooting Jacobian `S⁻¹ (Φ - I) S` (bordered for oscillators), with
/// unknown scales `S` and optional period scale `s_T`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ShootingSystem {
    pub(crate) matrix: Vec<f64>,
    pub(crate) tolerances: Vec<f64>,
    pub(crate) scale: Option<f64>,
}

/// Period mode distinguishing driven circuits from autonomous oscillators.
#[derive(Clone, Copy)]
enum PeriodMode {
    Forced(f64),
    Free(f64),
}

/// Initial state, operating-point strategy, and period mode for [`Simulator::shoot`].
struct ShootingStart {
    values: Vec<f64>,
    strategy: Strategy,
    period: PeriodMode,
}

impl Shot {
    /// Assembles the tolerance-scaled shooting system `S⁻¹ (Φ - I) S` in column-major order.
    ///
    /// For autonomous oscillators, borders the matrix with the scaled flow vector
    /// `c = S⁻¹ (∂φ/∂T) s_T` as both final column and phase-condition row, returning the
    /// unknown tolerances `S` and period scale `s_T`.
    pub(crate) fn system(self) -> ShootingSystem {
        let Self {
            monodromy,
            tolerances,
            scale,
        } = self;
        let Monodromy {
            tangents,
            at,
            stretch,
            steps,
        } = monodromy;
        drop((tangents, steps));
        let flow: Option<Vec<f64>> = stretch.map(|stretch| {
            stretch
                .at
                .iter()
                .zip(&tolerances)
                .map(|(derivative, tolerance)| derivative * scale / tolerance)
                .collect()
        });
        let rows = tolerances.len() + usize::from(flow.is_some());
        let mut matrix = Vec::with_capacity(rows * rows);
        for (column, (derivatives, scale)) in at.into_iter().zip(&tolerances).enumerate() {
            for (row, (derivative, tolerance)) in derivatives.iter().zip(&tolerances).enumerate() {
                let identity = if row == column { 1.0 } else { 0.0 };
                matrix.push((derivative - identity) * scale / tolerance);
            }
            if let Some(flow) = &flow {
                matrix.push(flow[column]);
            }
        }
        let scale = flow.as_ref().map(|flow| {
            matrix.extend(flow);
            matrix.push(0.0);
            scale
        });
        ShootingSystem {
            matrix,
            tolerances,
            scale,
        }
    }

    /// Takes and returns the recorded step formulas and restart flags in integration order.
    pub(crate) fn take_steps(&mut self) -> Vec<StepFormula> {
        std::mem::take(&mut self.monodromy.steps)
    }

    /// Checks whether an oscillator's orbit moves by more than an unknown tolerance under a
    /// period perturbation of `self.scale`, distinguishing true orbits from stationary points.
    fn moves(&self) -> bool {
        self.monodromy.stretch.as_ref().is_some_and(|stretch| {
            stretch
                .at
                .iter()
                .zip(&self.tolerances)
                .any(|(derivative, tolerance)| (derivative * self.scale).abs() > *tolerance)
        })
    }
}

impl Simulator {
    /// Computes one period of the periodic steady state by shooting.
    ///
    /// Driven circuits shoot directly from the operating point at the configured period;
    /// autonomous oscillators first integrate a perturbed transient until oscillation
    /// amplitude settles and then shoot for both state and period.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::InvalidAnalysis`] for unsupported mixed-signal, delayed, or
    /// misconfigured autonomous circuits; [`SimulationError::TooMuchMemory`] if dense shooting
    /// matrices exceed `max_memory`; [`SimulationError::NoGrowingMode`],
    /// [`SimulationError::Unsettled`], [`SimulationError::Stationary`], or
    /// [`SimulationError::NoSteadyState`] on convergence failures; and errors from
    /// [`Simulator::transient`] or [`Simulator::growing_modes`].
    pub fn periodic_steady_state(
        &mut self,
        analysis: &PeriodicSteadyState,
        options: &Options,
    ) -> Result<PeriodicSteadyStateResult, SimulationError> {
        options.parallelism.confined(|| {
            self.solvable()?;
            Ok(self.shot_steady_state(analysis, options)?.0)
        })
    }

    /// Validates circuit and configuration preconditions for shooting, rejecting logic nets,
    /// delay lines, autonomous breakpoints, and time-varying devices in autonomous runs.
    pub(crate) fn check_periodic(
        &self,
        analysis: &PeriodicSteadyState,
    ) -> Result<(), SimulationError> {
        if !self.elaborated.logic.nets.is_empty() {
            return Err(SimulationError::InvalidAnalysis(
                "a periodic steady state is found for circuits without logic".to_owned(),
            ));
        }
        self.undelayed(
            "periodic steady state",
            "the state a period starts from would hold the waves on its lines, which \
             shooting does not carry",
        )?;
        if analysis.period.is_none() {
            if !analysis.breakpoints.is_empty() {
                return Err(SimulationError::InvalidAnalysis(
                    "an oscillator's period is found, not given, so its periodic steady state \
                     has no times to step onto"
                        .to_owned(),
                ));
            }
            let varying = self.state.runs.time_varying(&self.elaborated.groups);
            if !varying.is_empty() {
                let names: Vec<String> = varying
                    .into_iter()
                    .map(|device| format!("`{}`", self.elaborated.devices[device].path()))
                    .collect();
                return Err(SimulationError::InvalidAnalysis(format!(
                    "an autonomous periodic steady state takes no source or behavioral device \
                     that varies with time: {}",
                    names.join(", ")
                )));
            }
        }
        Ok(())
    }

    /// Solves the periodic steady state as in [`Simulator::periodic_steady_state`], also
    /// returning the converged period's [`Shot`] monodromy data.
    pub(crate) fn shot_steady_state(
        &mut self,
        analysis: &PeriodicSteadyState,
        options: &Options,
    ) -> Result<(PeriodicSteadyStateResult, Shot), SimulationError> {
        self.check_periodic(analysis)?;
        options.check()?;
        let reading = self.read_meter(options.profile);
        let rows = self.elaborated.unknowns.len() + usize::from(analysis.period.is_none());
        let dense = options.hold_square("periodic steady state", rows, DENSE_MATRICES, 0)?;
        let period_options = Options {
            max_memory: options.max_memory - dense / NUMBERS_PER_GIB,
            ..options.clone()
        };
        let point = self.operating_point(options)?;
        let strategy = point.strategy;
        let (values, period) = if let Some(period) = analysis.period {
            (point.values, PeriodMode::Forced(period))
        } else {
            let (values, period) = self.settled(analysis, &point, &period_options)?;
            (values, PeriodMode::Free(period))
        };
        let initial = ShootingStart {
            values,
            strategy,
            period,
        };
        let (mut found, shot) = self.shoot(analysis, initial, &period_options)?;
        let statistics = &mut found.transient.statistics;
        (statistics.work, statistics.profile) = self.metered_since(&reading);
        Ok((found, shot))
    }

    /// Runs shooting iterations for `analysis` from `initial` until start and end states match
    /// within tolerance, returning the converged result and final [`Shot`].
    fn shoot(
        &mut self,
        analysis: &PeriodicSteadyState,
        initial: ShootingStart,
        options: &Options,
    ) -> Result<(PeriodicSteadyStateResult, Shot), SimulationError> {
        let mut start = initial.values;
        let strategy = initial.strategy;
        let (mut period, oscillator) = match initial.period {
            PeriodMode::Forced(period) => (period, false),
            PeriodMode::Free(period) => (period, true),
        };
        let unknowns = start.len();
        let mut mismatches = Vec::new();
        let mut simulated = 0;
        for periods in 1..=MAX_PERIODS {
            simulated = periods;
            let mut monodromy = if oscillator {
                Monodromy::stretched(period)
            } else {
                Monodromy::default()
            };
            let period_transient = Transient {
                stop: period,
                max_step: analysis.max_step,
                method: analysis.method,
                breakpoints: analysis.breakpoints.clone(),
                initial_conditions: BTreeMap::new(),
                operating_point: true,
            };
            let carried = Carried::Monodromy(&mut monodromy);
            let begun = Some((start.clone(), Some(strategy)));
            let transient = self.transient_with(&period_transient, options, begun, carried)?;
            let end = transient.values.last().unwrap_or(&start);
            let tolerances = self.tolerances(&start, end, options);
            mismatches = start
                .iter()
                .zip(end)
                .zip(&tolerances)
                .map(|((start, end), tolerance)| (start - end).abs() / tolerance)
                .collect();
            let mut step: Vec<f64> = start
                .iter()
                .zip(end)
                .zip(&tolerances)
                .map(|((start, end), tolerance)| (start - end) / tolerance)
                .collect();
            let shot = Shot {
                monodromy,
                tolerances,
                scale: options.relative_tolerance * period,
            };
            if mismatches.iter().all(|&mismatch| mismatch <= 1.0) {
                if oscillator && !shot.moves() {
                    return Err(SimulationError::Stationary);
                }
                let found = PeriodicSteadyStateResult {
                    period,
                    periods,
                    transient,
                };
                return Ok((found, shot));
            }
            if oscillator {
                step.push(0.0);
            }
            let system = shot.system();
            solve_in_range(&system.matrix, &mut step, (options.relative_tolerance, 1.0))
                .map_err(|error| self.state.solver.name(error))?;
            for ((value, step), tolerance) in start.iter_mut().zip(&step).zip(&system.tolerances) {
                *value += step * tolerance;
            }
            if let (Some(&stretched), Some(scale)) = (step.get(unknowns), system.scale) {
                period += stretched * scale;
                if period <= 0.0 {
                    break;
                }
            }
        }
        let mut worst: Vec<usize> = (0..unknowns).collect();
        worst.sort_by(|&a, &b| mismatches[b].total_cmp(&mismatches[a]));
        Err(SimulationError::NoSteadyState {
            periods: simulated,
            worst: worst
                .into_iter()
                .take(NAMED_UNKNOWNS)
                .map(|index| self.elaborated.unknowns[index].name.clone())
                .collect(),
        })
    }

    /// Integrates an autonomous oscillator from its perturbed operating point `point` in
    /// doubling transient stretches until oscillation amplitude settles, returning the final
    /// return state and estimated period.
    fn settled(
        &mut self,
        analysis: &PeriodicSteadyState,
        point: &OperatingPointResult,
        options: &Options,
    ) -> Result<(Vec<f64>, f64), SimulationError> {
        let modes = self.growing_modes(options)?;
        let mode = modes.first().ok_or(SimulationError::NoGrowingMode)?;
        let equilibrium = &point.values;
        let tolerances = self.tolerances(equilibrium, equilibrium, options);
        let kick = kick(&mode.shape, &tolerances);
        // Tolerance-scaled hyperplane through the operating point orthogonal to `kick`.
        let across = |x: &[f64]| -> f64 {
            x.iter()
                .zip(equilibrium)
                .zip(kick.iter().zip(&tolerances))
                .map(|((value, rest), (kick, tolerance))| {
                    kick * (value - rest) / (tolerance * tolerance)
                })
                .sum()
        };
        let mut state: Vec<f64> = equilibrium.iter().zip(&kick).map(|(x, k)| x + k).collect();
        let (mut time, mut length) = (0.0, TAU / mode.pole.norm());
        let mut returns = Returns::default();
        for _ in 0..MAX_PERIODS {
            // Limit stretch step size to `period / 50` once a return period is observed.
            let shown = returns.apart.map(|(period, _)| period / STEPS_PER_RUN);
            let stretch = Transient {
                stop: length,
                max_step: analysis.max_step.or(shown),
                method: analysis.method,
                breakpoints: Vec::new(),
                initial_conditions: BTreeMap::new(),
                operating_point: true,
            };
            returns.restart();
            let begun = Some((state, Some(point.strategy)));
            let waveforms = self.transient_with(&stretch, options, begun, Carried::Nothing)?;
            let points = waveforms.times.iter().zip(&waveforms.values);
            for ((before, from), (after, to)) in points.clone().zip(points.skip(1)) {
                let (below, above) = (across(from), across(to));
                if below <= 0.0 && above > 0.0 {
                    let fraction = -below / (above - below);
                    let crossed: Vec<f64> = from
                        .iter()
                        .zip(to)
                        .map(|(from, to)| from + fraction * (to - from))
                        .collect();
                    let at = time + before + fraction * (after - before);
                    let tolerances = self.tolerances(&crossed, &crossed, options);
                    returns.add(at, crossed, &tolerances);
                }
            }
            time += length;
            state = waveforms.values.last().cloned().unwrap_or_default();
            if let Some(settled) = returns.settled() {
                return Ok(settled);
            }
            length *= 2.0;
        }
        Err(SimulationError::Unsettled {
            time,
            returns: returns.count,
        })
    }

    /// Computes per-unknown tolerances between period `start` and `end` states under `options`.
    fn tolerances(&self, start: &[f64], end: &[f64], options: &Options) -> Vec<f64> {
        self.elaborated
            .unknowns
            .iter()
            .zip(start.iter().zip(end))
            .map(|(unknown, (start, end))| options.unknown_tolerance(unknown.nature, *start, *end))
            .collect()
    }

    /// Initializes `monodromy` at period start `x`: identity state sensitivities, initial
    /// capacitance columns for charge tangents, and zeroed period-stretch tangents.
    pub(crate) fn start_monodromy(
        &mut self,
        monodromy: &mut Monodromy,
        x: &[f64],
        options: &Options,
    ) {
        self.evaluate(x, Some(0.0), 1.0, options.gmin, false);
        let unknowns = x.len();
        let contributions = &self.state.workspace.contributions;
        let mut charges = vec![vec![0.0; unknowns]; unknowns];
        for ((row, column), &capacitance) in self
            .elaborated
            .pattern
            .positions()
            .zip(&contributions.capacitance)
        {
            charges[column][row] += capacitance;
        }
        monodromy.tangents = charges.into_iter().map(Tangent::new).collect();
        monodromy.steps.clear();
        monodromy.at = (0..unknowns)
            .map(|column| {
                let mut unit = vec![0.0; unknowns];
                unit[column] = 1.0;
                unit
            })
            .collect();
        if let Some(stretch) = &mut monodromy.stretch {
            stretch.charges = Tangent::new(contributions.charge.clone());
            stretch.tangent = Tangent::new(vec![0.0; unknowns]);
            stretch.at = vec![0.0; unknowns];
        }
    }

    /// Advances `monodromy` across an accepted step at `point` with `step`.
    ///
    /// Solves `(G + αC) ∂x/∂x₀ = -∂h/∂x₀` for each state-transition column, where `h` is the
    /// formula history term. For autonomous oscillators, also solves
    /// `(G + αC) ∂x/∂T = -∂h/∂T + (α q_{n+1} + β₀ q_n + β₁ q_{n-1}) / T`.
    pub(crate) fn advance_monodromy(
        &mut self,
        monodromy: &mut Monodromy,
        point: StepPoint<'_>,
        step: StepFormula,
        options: &Options,
    ) -> Result<(), SimulationError> {
        monodromy.steps.push(step);
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
        let linear = TangentLinear {
            pattern: &self.elaborated.pattern,
            capacitance: &contributions.capacitance,
            step,
            grain,
        };
        for (tangent, at) in monodromy.tangents.iter_mut().zip(&mut monodromy.at) {
            let history = tangent.history(&step.formula);
            let rhs = history.iter().map(|term| -term).collect();
            *at = tangent.solve_step(&mut factors, linear, rhs, &history, vec![0.0; unknowns])?;
        }
        if let Some(stretch) = &mut monodromy.stretch {
            let charge = contributions.charge.clone();
            let divided = stretch.charges.divided(&charge, &step.formula);
            let history: Vec<f64> = stretch
                .tangent
                .history(&step.formula)
                .iter()
                .zip(&divided)
                .map(|(term, divided)| term - divided / stretch.period)
                .collect();
            let rhs = history.iter().map(|term| -term).collect();
            stretch.at = stretch.tangent.solve_step(
                &mut factors,
                linear,
                rhs,
                &history,
                vec![0.0; unknowns],
            )?;
            stretch.charges.advance_plain(charge, step);
        }
        Ok(())
    }
}

/// Computes a real perturbation along eigenmode `shape`, rotated so the dominant
/// tolerance-scaled unknown is real and scaled to move that unknown by one tolerance.
fn kick(shape: &[Complex64], tolerances: &[f64]) -> Vec<f64> {
    let scaled = shape
        .iter()
        .zip(tolerances)
        .map(|(&value, &tolerance)| value / tolerance);
    let largest = scaled.fold(Complex64::default(), |largest, value| {
        if value.norm() > largest.norm() {
            value
        } else {
            largest
        }
    });
    let turn = largest.conj() / largest.norm();
    let along: Vec<f64> = shape.iter().map(|&value| (value * turn).re).collect();
    let reach = along
        .iter()
        .zip(tolerances)
        .map(|(value, tolerance)| value.abs() / tolerance)
        .fold(0.0, f64::max);
    along.iter().map(|value| value / reach).collect()
}

/// Tracker recording an oscillator's recent crossings of the Poincaré hyperplane.
#[derive(Default)]
struct Returns {
    /// Total crossings observed across all stretches.
    count: usize,
    /// Timestamp and state vector of the most recent crossing.
    last: Option<(f64, Vec<f64>)>,
    /// Elapsed period and maximum tolerance-scaled state distance between the last two
    /// crossings.
    apart: Option<(f64, f64)>,
    /// Flag indicating whether the last two crossings belong to the active stretch.
    compared: bool,
    /// Flag indicating that oscillation amplitude grew past tolerance and stopped increasing.
    ///
    /// Sub-tolerance crossing spacings are ignored because Newton noise can randomly shrink
    /// consecutive spacings while a small kick is still growing.
    settled: bool,
}

impl Returns {
    /// Records a hyperplane crossing at `time` and `state` using unknown `tolerances`.
    fn add(&mut self, time: f64, state: Vec<f64>, tolerances: &[f64]) {
        self.count += 1;
        if let Some((before, from)) = &self.last {
            let distance = state
                .iter()
                .zip(from)
                .zip(tolerances)
                .map(|((to, from), tolerance)| (to - from).abs() / tolerance)
                .fold(0.0, f64::max);
            let apart = self.apart.filter(|_| self.compared);
            self.settled |= apart.is_some_and(|(_, apart)| apart > 1.0 && distance <= apart);
            self.apart = Some((time - before, distance));
            self.compared = true;
        }
        self.last = Some((time, state));
    }

    /// Begins a new transient stretch, clearing per-stretch crossing comparisons while
    /// preserving the last observed return period for step-size control.
    fn restart(&mut self) {
        self.last = None;
        self.compared = false;
    }

    /// Returns the latest crossing `(state, period)` once oscillation growth has settled.
    fn settled(&self) -> Option<(Vec<f64>, f64)> {
        let (_, state) = self
            .last
            .as_ref()
            .filter(|_| self.settled && self.compared)?;
        let (period, _) = self.apart?;
        Some((state.clone(), period))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a unit-tolerance oscillator [`Shot`] with `T = 1 s` and period derivative `flow`.
    fn shot(flow: Vec<f64>) -> Shot {
        let unknowns = flow.len();
        let mut monodromy = Monodromy::stretched(1.0);
        if let Some(stretch) = &mut monodromy.stretch {
            stretch.at = flow;
        }
        Shot {
            monodromy,
            tolerances: vec![1.0; unknowns],
            scale: 1e-3,
        }
    }

    #[test]
    fn an_orbit_moves_when_its_period_moves_an_unknown_past_its_tolerance() {
        // A change of the period by its scale, 1e-3, moves the end by 1e-3 times the
        // flow: past a tolerance of 1 only where the flow passes 1e3.
        assert!(shot(vec![0.0, 2e3]).moves());
        assert!(!shot(vec![1e3, -1e3]).moves());
        assert!(!shot(vec![0.0, 0.0]).moves());
    }
}
