//! The DC sweep: operating points while one source steps through levels.

use jolty_models::{BranchKind, Quantity};
use serde::Deserialize;

use crate::newton::Mode;
use crate::{Options, SimulationError, Simulator};

/// A DC sweep: the operating point at each level of one independent source.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DcSweep {
    /// The path of the swept source, such as `V1` or `bias/I1`.
    pub source: String,
    /// The levels to hold the source at, in volts or amperes, in sweep order.
    pub levels: Vec<f64>,
}

/// The operating points of a DC sweep.
#[derive(Debug, Clone, PartialEq)]
pub struct Sweep {
    /// The path of the swept source.
    pub source: String,
    /// Whether the levels are voltages or currents.
    pub quantity: Quantity,
    /// The levels of the swept source.
    pub levels: Vec<f64>,
    /// The value of every unknown at each level.
    pub values: Vec<Vec<f64>>,
}

impl Simulator {
    /// Runs `sweep`, starting each operating point from the previous one.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::UnknownSource`] or [`SimulationError::NotASource`] when the
    /// swept instance is not an independent source, [`SimulationError::InvalidAnalysis`] when
    /// no level is given, and the errors of [`Simulator::operating_point`] at any level.
    pub fn dc_sweep(
        &mut self,
        sweep: &DcSweep,
        options: &Options,
    ) -> Result<Sweep, SimulationError> {
        options.check()?;
        if sweep.levels.is_empty() {
            return Err(SimulationError::InvalidAnalysis(
                "a DC sweep needs at least one level".to_owned(),
            ));
        }
        let index = self.source_index(&sweep.source)?;
        let result = self.sweep_source(index, &sweep.levels, options);
        self.hold_source(index, None);
        Ok(Sweep {
            source: sweep.source.clone(),
            quantity: self.source_quantity(index),
            levels: sweep.levels.clone(),
            values: result?,
        })
    }

    fn sweep_source(
        &mut self,
        index: usize,
        levels: &[f64],
        options: &Options,
    ) -> Result<Vec<Vec<f64>>, SimulationError> {
        let mut values = vec![0.0; self.system.unknowns.len()];
        let mut solutions = Vec::with_capacity(levels.len());
        for (step, &level) in levels.iter().enumerate() {
            self.hold_source(index, Some(level));
            let mut continued = values.clone();
            let direct = Mode::Dc {
                source_scale: 1.0,
                shunt: 0.0,
            };
            values = if step > 0
                && self
                    .newton(&mut continued, direct, options, options.dc_iterations)
                    .is_ok()
            {
                continued
            } else {
                self.solve_dc(values, options)?.values
            };
            solutions.push(values.clone());
        }
        Ok(solutions)
    }

    /// The index of the device of the independent source at `path`.
    pub(crate) fn source_index(&mut self, path: &str) -> Result<usize, SimulationError> {
        let index = self
            .system
            .devices
            .iter()
            .position(|placed| placed.path() == path)
            .ok_or_else(|| SimulationError::UnknownSource(path.to_owned()))?;
        if self.system.devices[index].device.as_source().is_none() {
            return Err(SimulationError::NotASource(path.to_owned()));
        }
        Ok(index)
    }

    /// Whether the source of the device at `index` holds a voltage or drives a current.
    pub(crate) fn source_quantity(&self, index: usize) -> Quantity {
        let drives_current = self.system.devices[index]
            .branches()
            .iter()
            .any(|branch| branch.kind == BranchKind::CurrentSource);
        if drives_current {
            Quantity::Current
        } else {
            Quantity::Voltage
        }
    }

    fn hold_source(&mut self, index: usize, level: Option<f64>) {
        if let Some(source) = self.system.devices[index].device.as_source() {
            source.hold(level);
        }
    }
}
