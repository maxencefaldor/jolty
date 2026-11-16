//! The periodic steady state, by shooting.
//!
//! A circuit driven by sources of period `T` settles, if it settles, into a state `x₀` that
//! one period of the transient carries back to itself: `φ(x₀) = x₀`, where `φ` integrates
//! the circuit's equations over a period from the state it is given. Shooting solves this by
//! Newton iteration (Aprille and Trick, "Steady-state analysis of nonlinear circuits with
//! periodic inputs", Proceedings of the IEEE 60(1), 1972): each period simulated from `x₀`
//! carries, besides the solution, its derivatives with respect to `x₀` along every step by
//! the direct method, which end as the monodromy matrix `Φ = ∂φ/∂x₀`, and the next start is
//! `x₀ + Δ`, where `(Φ − I) Δ = x₀ − φ(x₀)`. A period starts from its state by a backward
//! Euler step, which reads only its charges, so the unknowns that hold no charge follow the
//! rest. The search starts from the operating point; a circuit that settles slowly but
//! nearly linearly takes a few periods, where a transient would take many.
//!
//! A circuit can have a neutral direction, a state that a period neither grows nor decays,
//! such as a memristor's that its drive leaves alone, or a charge that nothing but a
//! capacitor holds: a quantity `uᵀ x` that one period carries back as it was, so that
//! `uᵀ Φ = uᵀ`. Then `Φ − I` is singular and the steady state is not unique: there is one
//! for each value of the quantity. The step `Δ` is the least-squares solution of least
//! norm among those in the range of `Φ − I`, which leaves every such quantity as it was:
//! shooting keeps the start's place along the neutral directions, and finds the steady
//! state a transient from it would settle into. Where `Φ − I` is regular, `Δ` is Newton's
//! step. The system is measured in units of each unknown's tolerance, as its mismatches
//! are, and its singular values within the rounding of the largest are zero.
//!
//! An oscillator, a circuit whose sources hold steady, has a period of its own, which
//! shooting finds with the state: `φ(x₀, T) = x₀`, `n` equations in `n + 1` unknowns. Any
//! state along the orbit starts a period as well as any other, a neutral direction of its
//! own, the flow `ẋ` that `Φ` carries back to itself; the phase condition takes it out, as
//! the least-norm step takes out the others: the start moves only across the orbit, its
//! step orthogonal to the flow there, `ẋᵀ Δx = 0`, in units of each unknown's tolerance.
//! The flow is `∂φ/∂T`, the end's derivative with respect to the period, which at the
//! steady state is the start's. Newton's step then solves the bordered system
//! `[[Φ − I, ∂φ/∂T], [∂φ/∂Tᵀ, 0]] [Δx, ΔT] = [x₀ − φ(x₀, T), 0]`, regular where the orbit
//! is isolated (the orthogonal phase condition: R. Seydel, *Practical Bifurcation and
//! Stability Analysis*, 3rd ed., Springer, 2010, §7.1). The derivative with respect to the
//! period is carried along the steps as the others are, every step's length a fixed
//! fraction of the period: each step's formula scales its charges' terms by `1 / T`,
//! `∂(dq/dt)/∂T = −(α q_{n+1} + β₀ q_n + β₁ q_{n-1}) / T`, which is exact for the steps
//! taken, and the circuit's equations, its sources holding steady, read no time.
//!
//! An oscillator's operating point is an equilibrium it leaves, and every period
//! repeats it, so shooting cannot start there. It starts on the orbit instead, which a
//! transient from the operating point finds: kicked by one tolerance along the natural mode
//! that grows fastest, the smallest step off the operating point the analysis resolves,
//! the transient runs until the oscillation stops growing, its returns to the hyperplane
//! through the operating point across the kick coming no further apart than the returns
//! before them, which were apart by more than their tolerance. It runs in stretches, each
//! twice as long as the one before, which restart its integration and so are compared
//! only within themselves, each step at most a fiftieth of the period the returns before
//! it show. Shooting starts from the last return, its period the time since the one
//! before.

use std::collections::BTreeMap;
use std::f64::consts::TAU;

use jolty_numerics::integration::{Formula, Method};
use jolty_numerics::{Complex64, solve_in_range};
use serde::Deserialize;

use crate::newton::{NAMED_UNKNOWNS, add_product, step_matrix};
use crate::options::NUMBERS_PER_GIB;
use crate::transient::{Carried, STEPS_PER_RUN, Tangent};
use crate::{
    OperatingPointResult, Options, SimulationError, Simulator, Strategy, Transient, TransientResult,
};

/// The most periods shooting simulates before it gives up, and the most stretches of
/// transient an oscillator's settling runs, each twice as long as the one before: the
/// last 2³⁹, some 5.5·10¹¹, times the first, past the 10¹¹ by which a transient's
/// longest step may exceed its shortest (`SMALLEST_STEP_FRACTION`), so that no orbit
/// slower than that shares a transient with the mode that starts it.
const MOST_PERIODS: usize = 40;

/// The most dense matrices of a row and a column per unknown, and one more for an
/// oscillator's period, that shooting holds at once, while it solves for the next start
/// where its system is singular: the system, the two factors of its singular value
/// decomposition, the reduced system, the two factors of the reduced system's own
/// decomposition, and that decomposition's workspace, up to about 7.3 such matrices in
/// faer's. Before, the first decomposition holds a copy of the system and a workspace of
/// its own instead of the reduced system and its factors, and a period four: its tangents,
/// three vectors per unknown, and `Φ`.
const DENSE_MATRICES: f64 = 14.0;

/// A periodic steady state: one period of the state a circuit repeats, found by shooting.
///
/// In JSON, it is `{"periodic_steady_state": {"period": 1e-6}}`, and an oscillator's
/// `{"periodic_steady_state": {}}`.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PeriodicSteadyState {
    /// The period of every source, in seconds; none for an oscillator, whose sources hold
    /// steady and whose period shooting finds.
    #[serde(default)]
    pub period: Option<f64>,
    /// The longest step allowed, in seconds; by default a fiftieth of the period.
    #[serde(default)]
    pub max_step: Option<f64>,
    /// The integration method between breakpoints; trapezoidal by default.
    #[serde(default)]
    pub method: Method,
    /// Times within a given period, in seconds, the analysis steps onto exactly besides
    /// its own, without restarting its integration there; none by default.
    #[serde(default)]
    pub breakpoints: Vec<f64>,
}

/// One period of a periodic steady state.
#[derive(Debug, Clone, PartialEq)]
pub struct PeriodicSteadyStateResult {
    /// The period, in seconds: the analysis's, or an oscillator's, which shooting found.
    pub period: f64,
    /// The periods shooting simulated to find the steady state, the last the one
    /// `waveforms` hold.
    pub periods: usize,
    /// The period from the state it repeats, its times from zero to the period. Its
    /// statistics' work counts every period's, the operating point's and an oscillator's
    /// settling.
    pub waveforms: TransientResult,
}

/// The derivatives with respect to the state a period starts from, at the last point a
/// transient accepted: of the charges, which the integration formula reads, and of the
/// unknowns, one of each per unknown of the start; and for an oscillator, with respect to
/// its period.
#[derive(Default)]
pub(crate) struct Monodromy {
    tangents: Vec<Tangent>,
    /// `∂x/∂x₀`, a column per unknown of the start: at the period's end, the monodromy
    /// matrix.
    at: Vec<Vec<f64>>,
    /// The derivatives with respect to an oscillator's period.
    stretch: Option<Stretch>,
    /// Each step's formula, and whether the integration restarts after it, in their order,
    /// for the derivatives with respect to the parameters to go through again.
    steps: Vec<(Formula, bool)>,
}

/// The derivatives with respect to an oscillator's period, every step's length a fixed
/// fraction of it.
struct Stretch {
    period: f64,
    /// The charges themselves, which the formula's terms the steps' lengths divide read.
    charges: Tangent,
    /// The charges' derivatives.
    tangent: Tangent,
    /// `∂x/∂T` at the last point: at the period's end, the flow there.
    at: Vec<f64>,
}

impl Monodromy {
    /// The derivatives an oscillator's period of length `period` carries.
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

/// What a period shooting simulated carried to its end, and the scales shooting measures it
/// in.
pub(crate) struct Shot {
    monodromy: Monodromy,
    /// Each unknown's tolerance at the period's start and end.
    tolerances: Vec<f64>,
    /// The period's own scale, its relative tolerance of it.
    scale: f64,
}

impl Shot {
    /// The system Newton's step solves, each unknown in units of its tolerance, by columns:
    /// `S⁻¹ (Φ − I) S`, bordered, for an oscillator, by the flow `c = S⁻¹ ∂φ/∂T s_T`, the
    /// period in units of its own scale `s_T`, as a column and as the phase condition's
    /// row; with the unknowns' tolerances `S` and an oscillator's period's scale.
    pub(crate) fn system(self) -> (Vec<f64>, Vec<f64>, Option<f64>) {
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
        (matrix, tolerances, scale)
    }

    /// The steps the period took, each its formula and whether the integration restarts
    /// after it, in their order, leaving none.
    pub(crate) fn take_steps(&mut self) -> Vec<(Formula, bool)> {
        std::mem::take(&mut self.monodromy.steps)
    }

    /// Whether an oscillator's orbit moves: whether a change of its period by its scale
    /// moves the period's end, by the flow there, by more than an unknown's tolerance. An
    /// orbit that does not is the operating point, which every period repeats.
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
    /// One period of the periodic steady state: of a circuit driven by sources of the
    /// analysis's period, found by shooting from the operating point, or of an oscillator,
    /// whose period it finds, by shooting from where a transient from its operating point
    /// settles onto its orbit.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::InvalidAnalysis`] for a circuit with logic, whose state
    /// shooting does not carry, or an oscillator's analysis with times to step onto;
    /// [`SimulationError::TooMuchMemory`] when its dense matrices would hold more than the
    /// options' `max_memory`; [`SimulationError::NoGrowingMode`] and
    /// [`SimulationError::Unsettled`] when an oscillator does not start or settle,
    /// [`SimulationError::Stationary`] when its shooting converges onto its operating point;
    /// [`SimulationError::NoSteadyState`] when the periods do not converge; and the errors
    /// of [`Simulator::transient`] and [`Simulator::growing_modes`], each period's results
    /// holding what the dense matrices leave of the memory.
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

    /// One period of the periodic steady state, as [`Simulator::periodic_steady_state`]
    /// finds it, and what that period carried: its derivatives with respect to its start
    /// and, for an oscillator, to its period.
    pub(crate) fn shot_steady_state(
        &mut self,
        analysis: &PeriodicSteadyState,
        options: &Options,
    ) -> Result<(PeriodicSteadyStateResult, Shot), SimulationError> {
        if !self.elaborated.logic.nets.is_empty() {
            return Err(SimulationError::InvalidAnalysis(
                "a periodic steady state is found for circuits without logic".to_owned(),
            ));
        }
        self.undelayed(
            "periodic steady state",
            "the state a period starts from would hold the waves on its lines, which              shooting does not carry",
        )?;
        if analysis.period.is_none() && !analysis.breakpoints.is_empty() {
            return Err(SimulationError::InvalidAnalysis(
                "an oscillator's period is found, not given, so its periodic steady state \
                 has no times to step onto"
                    .to_owned(),
            ));
        }
        options.check()?;
        let reading = self.read_meter(options.profile);
        let rows = self.elaborated.unknowns.len() + usize::from(analysis.period.is_none());
        let size = f64::from(u32::try_from(rows).unwrap_or(u32::MAX));
        let dense = DENSE_MATRICES * size * size;
        options.hold("periodic steady state", rows, dense)?;
        let period_options = Options {
            max_memory: options.max_memory - dense / NUMBERS_PER_GIB,
            ..options.clone()
        };
        let point = self.operating_point(options)?;
        let strategy = point.strategy;
        let (start, period) = match analysis.period {
            Some(period) => (point.values, period),
            None => self.settled(analysis, &point, &period_options)?,
        };
        let (mut found, shot) = self.shoot(
            analysis,
            (start, strategy),
            (period, analysis.period.is_none()),
            &period_options,
        )?;
        let statistics = &mut found.waveforms.statistics;
        (statistics.work, statistics.profile) = self.metered_since(&reading);
        Ok((found, shot))
    }

    /// The periods of `analysis` from `start` and how the operating point it came from
    /// was found, the first `period` long, until one repeats, and what that one carried;
    /// `oscillator` says whether the period is an unknown too.
    fn shoot(
        &mut self,
        analysis: &PeriodicSteadyState,
        (mut start, strategy): (Vec<f64>, Strategy),
        (mut period, oscillator): (f64, bool),
        options: &Options,
    ) -> Result<(PeriodicSteadyStateResult, Shot), SimulationError> {
        let unknowns = start.len();
        let mut mismatches = Vec::new();
        let mut simulated = 0;
        for periods in 1..=MOST_PERIODS {
            simulated = periods;
            let mut monodromy = if oscillator {
                Monodromy::stretched(period)
            } else {
                Monodromy::default()
            };
            let transient = Transient {
                stop: period,
                max_step: analysis.max_step,
                method: analysis.method,
                breakpoints: analysis.breakpoints.clone(),
                initial_conditions: BTreeMap::new(),
                operating_point: true,
            };
            let carried = Carried::Monodromy(&mut monodromy);
            let begun = Some((start.clone(), Some(strategy)));
            let waveforms = self.transient_with(&transient, options, begun, carried)?;
            let end = waveforms.values.last().unwrap_or(&start);
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
                    waveforms,
                };
                return Ok((found, shot));
            }
            if oscillator {
                step.push(0.0);
            }
            let (matrix, tolerances, scale) = shot.system();
            solve_in_range(&matrix, &mut step, (options.relative_tolerance, 1.0))
                .map_err(|error| self.state.solver.name(error))?;
            for ((value, step), tolerance) in start.iter_mut().zip(&step).zip(&tolerances) {
                *value += step * tolerance;
            }
            if let (Some(&stretched), Some(scale)) = (step.get(unknowns), scale) {
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

    /// Where an oscillator's transient from its operating point `point` settles onto its
    /// orbit, and the time it took to return there last: the transient, kicked by one
    /// tolerance along the natural mode that grows fastest, runs in stretches, the first
    /// the mode's natural period, each twice as long as the one before, until a stretch in
    /// which the oscillation stops growing.
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
        // The hyperplane through the operating point across the kick, in units of each
        // unknown's tolerance: a return crosses it the way the kick does.
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
        for _ in 0..MOST_PERIODS {
            // A stretch's steps are at most a fiftieth of the period the returns before it
            // show, as a period's own are, or of the stretch while none shows one.
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

    /// The tolerance of each unknown of a period started at `start` and ended at `end`,
    /// which it ends within of where it started if it repeats.
    fn tolerances(&self, start: &[f64], end: &[f64], options: &Options) -> Vec<f64> {
        self.elaborated
            .unknowns
            .iter()
            .zip(start.iter().zip(end))
            .map(|(unknown, (start, end))| {
                options.relative_tolerance * start.abs().max(end.abs())
                    + options.value_tolerance(unknown.nature)
            })
            .collect()
    }

    /// Starts carrying the derivatives with respect to the state `x` a period starts from:
    /// the unknowns' are the identity, and the charges' the capacitances there; an
    /// oscillator's with respect to its period are zero, its charges those at `x`.
    pub(crate) fn start_monodromy(
        &mut self,
        monodromy: &mut Monodromy,
        x: &[f64],
        options: &Options,
    ) {
        self.evaluate(x, Some(0.0), (1.0, options.gmin), false);
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

    /// Carries the derivatives with respect to the period's start over the step a
    /// transient accepted at `time`, its solution `x`, by `formula`; `restarted` says
    /// whether the integration restarts after it. Each column solves
    /// `(G + αC) ∂x/∂x₀ = −∂h/∂x₀`, where `h` is the part of the formula's `dq/dt` that the
    /// earlier points contribute, and an oscillator's derivative with respect to its period
    /// `(G + αC) ∂x/∂T = −∂h/∂T + (α q_{n+1} + β₀ q_n + β₁ q_{n-1}) / T`.
    pub(crate) fn advance_monodromy(
        &mut self,
        monodromy: &mut Monodromy,
        (time, x): (f64, &[f64]),
        (formula, restarted): (&Formula, bool),
        options: &Options,
    ) -> Result<(), SimulationError> {
        monodromy.steps.push((*formula, restarted));
        self.evaluate(x, Some(time), (1.0, options.gmin), false);
        let inside = options.parallelism.inside(x.len());
        let workspace = &mut self.state.workspace;
        let contributions = &workspace.contributions;
        step_matrix(
            &mut workspace.jacobian,
            contributions,
            formula.alpha,
            inside,
        );
        let mut factors = self.state.solver.factor(&self.state.workspace.jacobian)?;
        let contributions = &self.state.workspace.contributions;
        let pattern = &self.elaborated.pattern;
        for (tangent, at) in monodromy.tangents.iter_mut().zip(&mut monodromy.at) {
            let history = tangent.history(formula);
            let mut moved: Vec<f64> = history.iter().map(|term| -term).collect();
            factors.solve(&mut moved)?;
            let mut charge = vec![0.0; x.len()];
            add_product(
                pattern,
                (&contributions.capacitance, &moved),
                &mut charge,
                inside,
            );
            tangent.advance(charge, &history, formula, restarted);
            *at = moved;
        }
        if let Some(stretch) = &mut monodromy.stretch {
            let charge = contributions.charge.clone();
            let divided = stretch.charges.divided(&charge, formula);
            let history: Vec<f64> = stretch
                .tangent
                .history(formula)
                .iter()
                .zip(&divided)
                .map(|(term, divided)| term - divided / stretch.period)
                .collect();
            let mut moved: Vec<f64> = history.iter().map(|term| -term).collect();
            factors.solve(&mut moved)?;
            let mut moved_charge = vec![0.0; x.len()];
            add_product(
                pattern,
                (&contributions.capacitance, &moved),
                &mut moved_charge,
                inside,
            );
            stretch
                .tangent
                .advance(moved_charge, &history, formula, restarted);
            let charges_history = stretch.charges.history(formula);
            stretch
                .charges
                .advance(charge, &charges_history, formula, restarted);
            stretch.at = moved;
        }
        Ok(())
    }
}

/// The kick that starts an oscillation: along the mode of amplitudes `shape`, turned so
/// its largest unknown in units of its tolerance is real, its real part, scaled so that
/// unknown moves by its tolerance and no other by more than its own.
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

/// An oscillator's returns to the hyperplane across its kick, the last two kept.
#[derive(Default)]
struct Returns {
    /// The returns so far.
    count: usize,
    /// The last return's time and state.
    last: Option<(f64, Vec<f64>)>,
    /// The time between the last two returns, and the largest distance between their
    /// states, each unknown in units of its tolerance at the last.
    apart: Option<(f64, f64)>,
    /// Whether the last two returns are of the stretch running.
    compared: bool,
    /// Whether a return has come no further from the one before than that one from its
    /// own, which was beyond their tolerance: the oscillation grew past what the analysis
    /// resolves, and stopped growing. Returns within tolerance of each other show neither,
    /// as every return is while the oscillation grows from a kick of one tolerance, the
    /// error of each step's own Newton iteration making them come nearer or further by
    /// chance.
    settled: bool,
}

impl Returns {
    /// Adds the return at `time` to `state`, where the unknowns' tolerances are
    /// `tolerances`.
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

    /// Starts a stretch of transient, which restarts its integration with backward Euler's
    /// steps, damping the oscillation a little: a return is compared only with those of
    /// its own stretch, from the second on, and the time between the last two returns
    /// before stays until two of the new stretch replace it.
    fn restart(&mut self) {
        self.last = None;
        self.compared = false;
    }

    /// The last return's state and the time since the one before, once the oscillation
    /// stopped growing.
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

    /// A shot of an oscillator's period, `T` = 1 s, whose end moves with the period by
    /// `flow`, each unknown's tolerance 1.
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
