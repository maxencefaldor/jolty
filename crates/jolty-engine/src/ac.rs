//! The small-signal frequency response around the operating point.

use std::f64::consts::TAU;

use jolty_models::{Contributions, Quantity, Unknown};
use jolty_numerics::Complex64;
use serde::Deserialize;

use crate::solver::{Factors, Solver};
use crate::{OperatingPoint, Options, SimulationError, Simulator};

/// A small-signal analysis: the response to a unit excitation of one independent source.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Ac {
    /// The path of the excited source, such as `V1`.
    pub source: String,
    /// The frequencies, in hertz.
    pub frequencies: Vec<f64>,
}

/// The small-signal response of every unknown to a unit excitation, at each frequency.
#[derive(Debug, Clone, PartialEq)]
pub struct Response {
    /// The path of the excited source.
    pub source: String,
    /// The operating point the circuit was linearized at.
    pub operating_point: OperatingPoint,
    /// The frequencies, in hertz.
    pub frequencies: Vec<f64>,
    /// The phasor of every unknown at each frequency, per volt or ampere of excitation.
    pub values: Vec<Vec<Complex64>>,
    /// The phasor of the current into each terminal its options record at each frequency,
    /// per volt or ampere of excitation, in the order of an [`Observation`](crate::Observation)'s
    /// currents.
    pub currents: Vec<Vec<Complex64>>,
}

impl Simulator {
    /// Linearizes the circuit at its operating point and solves `(G + jωC) x = −∂f/∂s` at each
    /// frequency, where `s` is the excited source's level.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::UnknownSource`] or [`SimulationError::NotASource`] when the
    /// excited instance is not an independent source, [`SimulationError::InvalidAnalysis`]
    /// for no frequency or a negative one, [`SimulationError::Singular`] when the linearized
    /// circuit is singular at a frequency, and the errors of [`Simulator::operating_point`].
    pub fn ac(&mut self, ac: &Ac, options: &Options) -> Result<Response, SimulationError> {
        self.solvable()?;
        let index = self.source_index(&ac.source)?;
        let operating_point = self.linearize(&ac.frequencies, options)?;
        let mut excitation = vec![0.0; self.elaborated.unknowns.len()];
        if let Some(source) = self.state.devices[index].as_source() {
            source.level_derivative(&mut excitation);
        }
        let mut linearization = Linearization::new(&self.state.solver);
        self.state
            .observer
            .linearize(&operating_point.values, options.gmin);
        let mut values = Vec::with_capacity(ac.frequencies.len());
        let mut currents = Vec::with_capacity(ac.frequencies.len());
        for &frequency in &ac.frequencies {
            let mut phasors: Vec<Complex64> = excitation
                .iter()
                .map(|&d| Complex64::new(-d, 0.0))
                .collect();
            linearization
                .factor(&self.state.workspace.contributions, frequency)?
                .solve(&mut phasors)?;
            currents.push(
                self.state
                    .observer
                    .phasors(&phasors, TAU * frequency, index),
            );
            values.push(phasors);
        }
        Ok(Response {
            source: ac.source.clone(),
            operating_point,
            frequencies: ac.frequencies.clone(),
            values,
            currents,
        })
    }

    /// Checks a small-signal analysis's `frequencies`, solves the operating point and
    /// evaluates the devices there, so the workspace holds `G` and `C`.
    pub(crate) fn linearize(
        &mut self,
        frequencies: &[f64],
        options: &Options,
    ) -> Result<OperatingPoint, SimulationError> {
        options.check()?;
        if frequencies.is_empty() {
            return Err(SimulationError::InvalidAnalysis(
                "a small-signal analysis needs at least one frequency".to_owned(),
            ));
        }
        if let Some(frequency) = frequencies.iter().find(|&&frequency| frequency < 0.0) {
            return Err(SimulationError::InvalidAnalysis(format!(
                "frequencies must be at least 0, not {frequency:?}"
            )));
        }
        self.linearized(options)
    }

    /// Solves the operating point and evaluates the devices there, so the workspace holds
    /// `G` and `C`.
    pub(crate) fn linearized(
        &mut self,
        options: &Options,
    ) -> Result<OperatingPoint, SimulationError> {
        let operating_point = self.operating_point(options)?;
        self.evaluate(&operating_point.values, None, (1.0, options.gmin), true);
        Ok(operating_point)
    }

    /// The unknown named `output`, and the one named `reference` it is taken over, if
    /// any, with what the difference is: a voltage over a node, or any unknown alone. The
    /// reference node's name names no unknown.
    pub(crate) fn measured(
        &self,
        output: &str,
        reference: Option<&str>,
    ) -> Result<(Option<Unknown>, Option<Unknown>, Quantity), SimulationError> {
        let found = self.node(output)?;
        let base = reference.map(|name| self.node(name)).transpose()?.flatten();
        if found == base {
            return Err(SimulationError::InvalidAnalysis(format!(
                "the output `{output}` is taken over itself"
            )));
        }
        let quantity = |node: Option<Unknown>| {
            node.map_or(Quantity::Voltage, |unknown| {
                self.elaborated.unknowns[unknown.index()].quantity
            })
        };
        let quantity = match (quantity(found), quantity(base), reference) {
            (quantity, _, None) => quantity,
            (Quantity::Voltage, Quantity::Voltage, Some(_)) => Quantity::Voltage,
            (_, _, Some(name)) => {
                return Err(SimulationError::InvalidAnalysis(format!(
                    "an output over a node is a voltage, but `{output}` over `{name}` is not"
                )));
            }
        };
        Ok((found, base, quantity))
    }

    /// The node or unknown named `name`: none for the reference node.
    fn node(&self, name: &str) -> Result<Option<Unknown>, SimulationError> {
        if name == self.reference() {
            return Ok(None);
        }
        self.unknown(name)
            .map(Some)
            .ok_or_else(|| SimulationError::UnknownOutput(name.to_owned()))
    }
}

/// The matrix `G + jωC` of a circuit linearized at its operating point, factored at any
/// frequency.
pub(crate) struct Linearization {
    solver: Solver<Complex64>,
    matrix: Vec<Complex64>,
}

impl Linearization {
    /// A linearization over the matrices of `solver`, the simulator's, whose ordering it
    /// shares.
    pub(crate) fn new(solver: &Solver<f64>) -> Self {
        Self {
            solver: solver.sharing(),
            matrix: vec![Complex64::default(); solver.pattern().len()],
        }
    }

    /// `G + jωC` at `frequency`, from the `contributions` at the operating point, factored.
    pub(crate) fn factor(
        &mut self,
        contributions: &Contributions,
        frequency: f64,
    ) -> Result<Factors<'_, Complex64>, SimulationError> {
        self.factor_at(contributions, Complex64::new(0.0, TAU * frequency))
    }

    /// `G + sC` at the complex frequency `s`, from the `contributions` at the operating
    /// point, factored.
    pub(crate) fn factor_at(
        &mut self,
        contributions: &Contributions,
        s: Complex64,
    ) -> Result<Factors<'_, Complex64>, SimulationError> {
        for ((value, &conductance), &capacitance) in self
            .matrix
            .iter_mut()
            .zip(&contributions.conductance)
            .zip(&contributions.capacitance)
        {
            *value = conductance + s * capacitance;
        }
        self.solver.factor(&self.matrix)
    }
}
