//! The small-signal frequency response around the operating point.

use std::f64::consts::TAU;

use jolty_numerics::{Complex64, SparseLu};
use serde::Deserialize;

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
    /// The operating point the circuit was linearized at.
    pub operating_point: OperatingPoint,
    /// The frequencies, in hertz.
    pub frequencies: Vec<f64>,
    /// The phasor of every unknown at each frequency, per volt or ampere of excitation.
    pub values: Vec<Vec<Complex64>>,
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
        options.check()?;
        if ac.frequencies.is_empty() {
            return Err(SimulationError::InvalidAnalysis(
                "an AC analysis needs at least one frequency".to_owned(),
            ));
        }
        if let Some(frequency) = ac.frequencies.iter().find(|&&frequency| frequency < 0.0) {
            return Err(SimulationError::InvalidAnalysis(format!(
                "frequencies must not be negative, not {frequency}"
            )));
        }
        let index = self.source_index(&ac.source)?;
        let operating_point = self.operating_point(options)?;
        self.evaluate(&operating_point.values, 0.0, 1.0, options.gmin);

        let unknowns = self.system.unknowns.len();
        let mut excitation = vec![0.0; unknowns];
        if let Some(source) = self.system.devices[index].device.as_source() {
            source.level_derivative(&mut excitation);
        }
        let mut solver = SparseLu::<Complex64>::new(self.system.pattern.clone())?;
        let mut matrix = vec![Complex64::default(); self.system.pattern.len()];
        let mut values = Vec::with_capacity(ac.frequencies.len());
        for &frequency in &ac.frequencies {
            let omega = TAU * frequency;
            for ((value, &conductance), &capacitance) in matrix
                .iter_mut()
                .zip(&self.workspace.conductance)
                .zip(&self.workspace.capacitance)
            {
                *value = Complex64::new(conductance, omega * capacitance);
            }
            let mut phasors: Vec<Complex64> = excitation
                .iter()
                .map(|&d| Complex64::new(-d, 0.0))
                .collect();
            solver
                .factor(&matrix)
                .and_then(|mut factors| factors.solve(&mut phasors))?;
            values.push(phasors);
        }
        Ok(Response {
            operating_point,
            frequencies: ac.frequencies.clone(),
            values,
        })
    }
}
