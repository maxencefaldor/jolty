//! The DC sweep: operating points while one source steps through levels.

use jolty_models::{BranchKind, Level, Quantity};
use serde::Deserialize;

use crate::newton::Mode;
use crate::operating_point::Solved;
use crate::{Observations, OperatingPoint, Options, SimulationError, Simulator, Strategy};

/// The fraction of the way from one level of a sweep to the next below which continuation
/// gives up, as a divisor.
const SMALLEST_CONTINUATION: f64 = 1024.0;

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
    /// What every device did at each level: the current into each terminal and the charge
    /// it holds, its derivatives and its variables.
    pub observations: Observations,
    /// The level of every logic net at each level.
    pub logic: Vec<Vec<Level>>,
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
        self.solvable()?;
        options.check()?;
        if sweep.levels.is_empty() {
            return Err(SimulationError::InvalidAnalysis(
                "a DC sweep needs at least one level".to_owned(),
            ));
        }
        let index = self.source_index(&sweep.source)?;
        let result = self.sweep_source(index, &sweep.levels, options);
        self.hold_source(index, None);
        let points = result?;
        let mut values = Vec::with_capacity(points.len());
        let mut observations = Observations::default();
        let mut logic = Vec::with_capacity(points.len());
        for point in points {
            values.push(point.values);
            observations.push(&point.observation);
            logic.push(point.logic);
        }
        Ok(Sweep {
            source: sweep.source.clone(),
            quantity: self.source_quantity(index),
            levels: sweep.levels.clone(),
            values,
            observations,
            logic,
        })
    }

    /// The operating point at each of `levels` of the source at `index`. The logic settles at
    /// each from where it settled at the one before.
    fn sweep_source(
        &mut self,
        index: usize,
        levels: &[f64],
        options: &Options,
    ) -> Result<Vec<OperatingPoint>, SimulationError> {
        self.reset_logic();
        let mut values = vec![0.0; self.system.unknowns.len()];
        let mut points = Vec::with_capacity(levels.len());
        let mut previous: Option<f64> = None;
        for &level in levels {
            let continued = previous.is_some_and(|from| {
                self.continue_level(index, (from, level), &mut values, options)
            });
            let solved = if continued {
                Solved {
                    values: std::mem::take(&mut values),
                    strategy: Strategy::Direct,
                }
            } else {
                self.hold_source(index, Some(level));
                match self.solve_dc(values.clone(), options) {
                    Ok(solved) => solved,
                    // The first level may lie where no strategy finds the solution from
                    // scratch: there, continue to it from the circuit's own operating point.
                    Err(error) if previous.is_none() => Solved {
                        values: self.reach_first_level(index, level, options).ok_or(error)?,
                        strategy: Strategy::Direct,
                    },
                    Err(error) => return Err(error),
                }
            };
            let point = self.settle(solved, options)?;
            previous = Some(level);
            values.clone_from(&point.values);
            points.push(point);
        }
        Ok(points)
    }

    /// The solution with the source at `index` at `level`, continued from the operating
    /// point with the source at its own level, if both are found.
    fn reach_first_level(
        &mut self,
        index: usize,
        level: f64,
        options: &Options,
    ) -> Option<Vec<f64>> {
        let own = self.system.devices[index].device.as_source()?.level();
        self.hold_source(index, Some(own));
        let mut values = self
            .solve_dc(vec![0.0; self.system.unknowns.len()], options)
            .ok()?
            .values;
        self.continue_level(index, (own, level), &mut values, options)
            .then_some(values)
    }

    /// Moves the source at `index` from one level to the next, solving from each solution
    /// to the next by Newton iteration alone: in one step if it converges, else in steps
    /// halved after each failure and doubled after each success. Leaves the solution in
    /// `values` and returns true, or returns false, `values` untouched, once a step has
    /// shrunk to a thousandth of the way.
    fn continue_level(
        &mut self,
        index: usize,
        (from, to): (f64, f64),
        values: &mut Vec<f64>,
        options: &Options,
    ) -> bool {
        let direct = Mode::Dc {
            source_scale: 1.0,
            shunt: 0.0,
            deflated: &[],
        };
        let span = to - from;
        let (mut level, mut increment) = (from, span);
        let mut solution = values.clone();
        loop {
            let last = (to - level).abs() <= increment.abs();
            let next = if last { to } else { level + increment };
            self.hold_source(index, Some(next));
            let mut trial = solution.clone();
            if self
                .newton(&mut trial, direct, options, options.dc_iterations)
                .is_ok()
            {
                solution = trial;
                if last {
                    break;
                }
                level = next;
                increment *= 2.0;
            } else {
                increment /= 2.0;
                if increment.abs() < (span / SMALLEST_CONTINUATION).abs() {
                    return false;
                }
            }
        }
        *values = solution;
        true
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

    pub(crate) fn hold_source(&mut self, index: usize, level: Option<f64>) {
        if let Some(source) = self.system.devices[index].device.as_source() {
            source.hold(level);
        }
        self.observer.hold(index, level);
    }
}
