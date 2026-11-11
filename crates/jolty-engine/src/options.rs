use serde::Deserialize;

use crate::SimulationError;

/// The numerical settings of a simulation: tolerances and iteration limits.
///
/// The defaults are SPICE's, so results compare with SPICE simulators run at their defaults.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Options {
    /// The tolerance on every unknown and residual, relative to its magnitude; default 1e-3.
    pub relative_tolerance: f64,
    /// The absolute tolerance on voltages, in volts; default 1e-6.
    pub voltage_tolerance: f64,
    /// The absolute tolerance on currents, in amperes; default 1e-12.
    pub current_tolerance: f64,
    /// The absolute tolerance on charges, in coulombs; default 1e-14.
    pub charge_tolerance: f64,
    /// The conductance placed across every nonlinear junction, in siemens; default 1e-12.
    pub gmin: f64,
    /// The Newton iterations allowed for one DC solution; default 100.
    pub dc_iterations: usize,
    /// The Newton iterations allowed for one time step before the step is shortened; default 10.
    pub transient_iterations: usize,
    /// How many times the estimated truncation error may exceed the tolerances before a step
    /// is rejected; default 7, which accounts for the estimate's pessimism.
    pub truncation_tolerance: f64,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            relative_tolerance: 1e-3,
            voltage_tolerance: 1e-6,
            current_tolerance: 1e-12,
            charge_tolerance: 1e-14,
            gmin: 1e-12,
            dc_iterations: 100,
            transient_iterations: 10,
            truncation_tolerance: 7.0,
        }
    }
}

impl Options {
    /// Checks that every tolerance is positive and every limit allows an iteration.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::InvalidOptions`] naming the first setting out of range.
    pub fn check(&self) -> Result<(), SimulationError> {
        let tolerances = [
            ("relative_tolerance", self.relative_tolerance),
            ("voltage_tolerance", self.voltage_tolerance),
            ("current_tolerance", self.current_tolerance),
            ("charge_tolerance", self.charge_tolerance),
            ("truncation_tolerance", self.truncation_tolerance),
        ];
        if let Some((name, value)) = tolerances.into_iter().find(|&(_, value)| value <= 0.0) {
            return Err(SimulationError::InvalidOptions(format!(
                "{name} must be positive, not {value}"
            )));
        }
        if self.gmin < 0.0 {
            return Err(SimulationError::InvalidOptions(format!(
                "gmin must not be negative, not {}",
                self.gmin
            )));
        }
        if self.dc_iterations == 0 || self.transient_iterations == 0 {
            return Err(SimulationError::InvalidOptions(
                "iteration limits must be at least 1".to_owned(),
            ));
        }
        Ok(())
    }
}
