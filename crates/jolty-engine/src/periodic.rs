//! The periodic steady state, by shooting.
//!
//! A circuit driven by sources of period `T` settles, if it settles, into a state `x₀` that
//! one period of the transient carries back to itself: `φ(x₀) = x₀`, where `φ` integrates
//! the circuit's equations over a period from the state it is given. Shooting solves this by
//! Newton iteration (Aprille and Trick, "Steady-state analysis of nonlinear circuits with
//! periodic inputs", Proceedings of the IEEE 60(1), 1972): each period simulated from `x₀`
//! carries, besides the solution, its derivatives with respect to `x₀` along every step by
//! the direct method, which end as the monodromy matrix `Φ = ∂φ/∂x₀`, and the next start is
//! `x₀ − (Φ − I)⁻¹ (φ(x₀) − x₀)`. A period starts from its state by a backward Euler step,
//! which reads only its charges, so the unknowns that hold no charge follow the rest. The
//! search starts from the operating point; a circuit that settles slowly but nearly
//! linearly takes a few periods, where a transient would take many.

use jolty_numerics::integration::Formula;
use jolty_numerics::solve_dense;

use crate::newton::NAMED_UNKNOWNS;
use crate::transient::{Carried, Tangent};
use crate::{Options, SimulationError, Simulator, Transient, Waveforms};

/// The most periods shooting simulates before it gives up.
const MOST_PERIODS: usize = 40;

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
    /// shooting does not carry, [`SimulationError::NoSteadyState`] when the periods do not
    /// converge, and the errors of [`Simulator::transient`].
    pub(crate) fn periodic_steady_state(
        &mut self,
        transient: &Transient,
        options: &Options,
    ) -> Result<Waveforms, SimulationError> {
        if !self.system.logic.nets.is_empty() {
            return Err(SimulationError::InvalidAnalysis(
                "a periodic steady state is found for circuits without logic".to_owned(),
            ));
        }
        let point = self.operating_point(options)?;
        let strategy = point.strategy;
        let mut start = point.values;
        let unknowns = start.len();
        let mut mismatches = Vec::new();
        for period in 1..=MOST_PERIODS {
            let mut monodromy = Monodromy::default();
            let carried = Carried::Monodromy(&mut monodromy);
            let begun = Some((start.clone(), strategy));
            let mut waveforms = self.transient_with(transient, options, begun, carried)?;
            let end = waveforms.values.last().unwrap_or(&start);
            mismatches = self.mismatches(&start, end, options);
            if mismatches.iter().all(|&mismatch| mismatch <= 1.0) {
                waveforms.statistics.periods = Some(period);
                return Ok(waveforms);
            }
            // (Φ − I) Δ = x₀ − φ(x₀), Φ stored by columns.
            let mut matrix = monodromy.at.concat();
            for index in 0..unknowns {
                matrix[index * unknowns + index] -= 1.0;
            }
            let mut step: Vec<f64> = start.iter().zip(end).map(|(x, y)| x - y).collect();
            solve_dense(&matrix, &mut step)?;
            for (value, step) in start.iter_mut().zip(&step) {
                *value += step;
            }
        }
        let mut worst: Vec<usize> = (0..unknowns).collect();
        worst.sort_by(|&a, &b| mismatches[b].total_cmp(&mismatches[a]));
        Err(SimulationError::NoSteadyState {
            periods: MOST_PERIODS,
            worst: worst
                .into_iter()
                .take(NAMED_UNKNOWNS)
                .map(|index| self.system.unknowns[index].name.clone())
                .collect(),
        })
    }

    /// How far each unknown ends a period, started at `start` and ended at `end`, from
    /// where it started, against its tolerance: at most one where it repeats.
    fn mismatches(&self, start: &[f64], end: &[f64], options: &Options) -> Vec<f64> {
        self.system
            .unknowns
            .iter()
            .zip(start.iter().zip(end))
            .map(|(unknown, (start, end))| {
                let tolerance = options.relative_tolerance * start.abs().max(end.abs())
                    + options.value_tolerance(unknown.quantity);
                (start - end).abs() / tolerance
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
        self.evaluate(x, 0.0, (1.0, options.gmin), false);
        let unknowns = x.len();
        let mut charges = vec![vec![0.0; unknowns]; unknowns];
        let capacitances = &self.workspace.contributions.capacitance;
        for ((row, column), &capacitance) in self.system.pattern.positions().zip(capacitances) {
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
        self.evaluate(x, time, (1.0, options.gmin), false);
        let contributions = &self.workspace.contributions;
        let jacobian: Vec<f64> = contributions
            .conductance
            .iter()
            .zip(&contributions.capacitance)
            .map(|(conductance, capacitance)| conductance + formula.alpha * capacitance)
            .collect();
        let mut factors = self.solver.factor(&jacobian)?;
        for (tangent, at) in monodromy.tangents.iter_mut().zip(&mut monodromy.at) {
            let history = tangent.history(formula);
            let mut moved: Vec<f64> = history.iter().map(|term| -term).collect();
            factors.solve(&mut moved)?;
            let mut charge = vec![0.0; x.len()];
            let capacitances = &self.workspace.contributions.capacitance;
            self.system
                .pattern
                .add_product(capacitances, &moved, &mut charge);
            tangent.advance(charge, &history, formula, restarted);
            *at = moved;
        }
        Ok(())
    }
}
