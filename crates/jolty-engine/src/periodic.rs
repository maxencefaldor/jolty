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

use jolty_numerics::integration::Formula;
use jolty_numerics::solve_in_range;

use crate::newton::NAMED_UNKNOWNS;
use crate::options::NUMBERS_PER_GIB;
use crate::transient::{Carried, Tangent};
use crate::{Options, SimulationError, Simulator, Transient, Waveforms};

/// The most periods shooting simulates before it gives up.
const MOST_PERIODS: usize = 40;

/// The most dense matrices of a row and a column per unknown that shooting holds at once,
/// while it solves for the next start where `Φ − I` is singular: `Φ − I`, the two factors
/// of its singular value decomposition, the reduced system, the two factors of the reduced
/// system's own decomposition, and that decomposition's workspace, up to about 7.3 such
/// matrices in faer's. Before, the first decomposition holds a copy of `Φ − I` and a
/// workspace of its own instead of the reduced system and its factors, and a period four:
/// its tangents, three vectors per unknown, and `Φ`.
const DENSE_MATRICES: f64 = 14.0;

/// The derivatives with respect to the state a period starts from, at the last point a
/// transient accepted: of the charges, which the integration formula reads, and of the
/// unknowns, one of each per unknown of the start.
#[derive(Default)]
pub(crate) struct Monodromy {
    tangents: Vec<Tangent>,
    /// `∂x/∂x₀`, a column per unknown of the start: at the period's end, the monodromy
    /// matrix.
    at: Vec<Vec<f64>>,
}

impl Simulator {
    /// One period of the periodic steady state that `transient` drives, its stop the
    /// period, found by shooting from the operating point.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::InvalidAnalysis`] for a circuit with logic, whose state
    /// shooting does not carry, [`SimulationError::TooMuchMemory`] when its dense matrices
    /// would hold more than the options' `memory`, [`SimulationError::NoSteadyState`] when
    /// the periods do not converge, and the errors of [`Simulator::transient`], each
    /// period's results holding what the dense matrices leave of the memory.
    pub(crate) fn periodic_steady_state(
        &mut self,
        transient: &Transient,
        options: &Options,
    ) -> Result<Waveforms, SimulationError> {
        if !self.elaborated.logic.nets.is_empty() {
            return Err(SimulationError::InvalidAnalysis(
                "a periodic steady state is found for circuits without logic".to_owned(),
            ));
        }
        options.check()?;
        let reading = self.read_meter(options.profile);
        let unknowns = self.elaborated.unknowns.len();
        let size = f64::from(u32::try_from(unknowns).unwrap_or(u32::MAX));
        let dense = DENSE_MATRICES * size * size;
        options.hold("periodic steady state", unknowns, dense)?;
        let period_options = Options {
            memory: options.memory - dense / NUMBERS_PER_GIB,
            ..options.clone()
        };
        let point = self.operating_point(options)?;
        let strategy = point.strategy;
        let mut start = point.values;
        let mut mismatches = Vec::new();
        for period in 1..=MOST_PERIODS {
            let mut monodromy = Monodromy::default();
            let carried = Carried::Monodromy(&mut monodromy);
            let begun = Some((start.clone(), strategy));
            let mut waveforms = self.transient_with(transient, &period_options, begun, carried)?;
            let end = waveforms.values.last().unwrap_or(&start);
            let tolerances = self.tolerances(&start, end, options);
            mismatches = start
                .iter()
                .zip(end)
                .zip(&tolerances)
                .map(|((start, end), tolerance)| (start - end).abs() / tolerance)
                .collect();
            if mismatches.iter().all(|&mismatch| mismatch <= 1.0) {
                let statistics = &mut waveforms.statistics;
                statistics.periods = Some(period);
                (statistics.work, statistics.profile) = self.metered_since(&reading);
                return Ok(waveforms);
            }
            // (Φ − I) Δ = x₀ − φ(x₀), each unknown in units of its tolerance:
            // S⁻¹ (Φ − I) S · S⁻¹ Δ = S⁻¹ (x₀ − φ(x₀)), Φ stored by columns.
            let Monodromy { tangents, at } = monodromy;
            drop(tangents);
            let mut matrix = Vec::with_capacity(unknowns * unknowns);
            for (column, (derivatives, scale)) in at.into_iter().zip(&tolerances).enumerate() {
                for (row, (derivative, tolerance)) in
                    derivatives.iter().zip(&tolerances).enumerate()
                {
                    let identity = if row == column { 1.0 } else { 0.0 };
                    matrix.push((derivative - identity) * scale / tolerance);
                }
            }
            let mut step: Vec<f64> = start
                .iter()
                .zip(end)
                .zip(&tolerances)
                .map(|((start, end), tolerance)| (start - end) / tolerance)
                .collect();
            solve_in_range(&matrix, &mut step, (options.relative_tolerance, 1.0))
                .map_err(|error| self.state.solver.name(error))?;
            for ((value, step), tolerance) in start.iter_mut().zip(&step).zip(&tolerances) {
                *value += step * tolerance;
            }
        }
        let mut worst: Vec<usize> = (0..unknowns).collect();
        worst.sort_by(|&a, &b| mismatches[b].total_cmp(&mismatches[a]));
        Err(SimulationError::NoSteadyState {
            periods: MOST_PERIODS,
            worst: worst
                .into_iter()
                .take(NAMED_UNKNOWNS)
                .map(|index| self.elaborated.unknowns[index].name.clone())
                .collect(),
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
                    + options.value_tolerance(unknown.quantity)
            })
            .collect()
    }

    /// Starts carrying the derivatives with respect to the state `x` a period starts from:
    /// the unknowns' are the identity, and the charges' the capacitances there.
    pub(crate) fn start_monodromy(
        &mut self,
        monodromy: &mut Monodromy,
        x: &[f64],
        options: &Options,
    ) {
        self.evaluate(x, Some(0.0), (1.0, options.gmin), false);
        let unknowns = x.len();
        let mut charges = vec![vec![0.0; unknowns]; unknowns];
        let capacitances = &self.state.workspace.contributions.capacitance;
        for ((row, column), &capacitance) in self.elaborated.pattern.positions().zip(capacitances) {
            charges[column][row] += capacitance;
        }
        monodromy.tangents = charges.into_iter().map(Tangent::new).collect();
        monodromy.at = (0..unknowns)
            .map(|column| {
                let mut unit = vec![0.0; unknowns];
                unit[column] = 1.0;
                unit
            })
            .collect();
    }

    /// Carries the derivatives with respect to the period's start over the step a
    /// transient accepted at `time`, its solution `x`, by `formula`; `restarted` says
    /// whether the integration restarts after it. Each column solves
    /// `(G + αC) ∂x/∂x₀ = −∂h/∂x₀`, where `h` is the part of the formula's `dq/dt` that the
    /// earlier points contribute.
    pub(crate) fn advance_monodromy(
        &mut self,
        monodromy: &mut Monodromy,
        (time, x): (f64, &[f64]),
        (formula, restarted): (&Formula, bool),
        options: &Options,
    ) -> Result<(), SimulationError> {
        self.evaluate(x, Some(time), (1.0, options.gmin), false);
        let contributions = &self.state.workspace.contributions;
        let jacobian: Vec<f64> = contributions
            .conductance
            .iter()
            .zip(&contributions.capacitance)
            .map(|(conductance, capacitance)| conductance + formula.alpha * capacitance)
            .collect();
        let mut factors = self.state.solver.factor(&jacobian)?;
        for (tangent, at) in monodromy.tangents.iter_mut().zip(&mut monodromy.at) {
            let history = tangent.history(formula);
            let mut moved: Vec<f64> = history.iter().map(|term| -term).collect();
            factors.solve(&mut moved)?;
            let mut charge = vec![0.0; x.len()];
            let capacitances = &self.state.workspace.contributions.capacitance;
            self.elaborated
                .pattern
                .add_product(capacitances, &moved, &mut charge);
            tangent.advance(charge, &history, formula, restarted);
            *at = moved;
        }
        Ok(())
    }
}
