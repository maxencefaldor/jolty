//! The DC sweep: operating points while an independent source steps through levels, or a
//! number of a device's parameters through values.
//!
//! A sweep makes the swept device again at each level, from the parameters of a source of
//! that constant level or from its own with the number replaced, and leaves every other
//! device as the level before left it: each operating point starts from the one before,
//! as a chain of runs does from one run to the next.

use jolty_models::{BranchKind, Level, Quantity};
use serde::Deserialize;

use crate::newton::Mode;
use crate::operating_point::Solved;
use crate::state::Held;
use crate::{Observations, OperatingPointResult, Options, SimulationError, Simulator, Strategy};

/// The fraction of the way from one level of a sweep to the next below which continuation
/// gives up, as a divisor.
const SMALLEST_CONTINUATION: f64 = 1024.0;

/// A DC sweep: the operating point at each level of one independent source, or at each
/// value of one number of a device's parameters.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DcSweep {
    /// The path of the swept instance, such as `V1` or `bias/I1`: an independent source,
    /// whose level the sweep sets whatever its waveform, unless a parameter is named.
    pub source: String,
    /// The number swept, by its path among the instance's parameters, such as
    /// `resistance`; none for an independent source's level.
    #[serde(default)]
    pub parameter: Option<String>,
    /// The levels, in volts or amperes for a source, or the values of the parameter in SI
    /// base units, in sweep order.
    pub levels: Vec<f64>,
}

/// The operating points of a DC sweep.
#[derive(Debug, Clone, PartialEq)]
pub struct DcSweepResult {
    /// The path of the swept instance.
    pub source: String,
    /// The number swept, by its path among the instance's parameters; none for a source's
    /// level.
    pub parameter: Option<String>,
    /// Whether the levels of a source are voltages or currents; none for a parameter.
    pub quantity: Option<Quantity>,
    /// The levels, or the values of the parameter.
    pub levels: Vec<f64>,
    /// The value of every unknown at each level.
    pub values: Vec<Vec<f64>>,
    /// What the devices its options record did at each level: the current into each
    /// recorded terminal and the charge it holds, and the recorded devices' derivatives and
    /// variables.
    pub observations: Observations,
    /// The level of every logic net at each level.
    pub logic: Vec<Vec<Level>>,
}

/// What a sweep sweeps: what it holds, the device whose instance it prepares again, and
/// where the run's own values leave it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Swept {
    pub(crate) held: Held,
    pub(crate) device: usize,
    pub(crate) own: f64,
}

impl Simulator {
    /// Runs `sweep`, starting each operating point from the previous one.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::UnknownSource`] when there is no instance at the swept
    /// path, [`SimulationError::NotASource`] when a sweep of a level is not of an
    /// independent source, [`SimulationError::InvalidAnalysis`] when no level is given, a
    /// level is not finite, the instance writes no such number, or a level puts it out of
    /// range or changes its unknowns, and the errors of [`Simulator::operating_point`] at
    /// any level.
    pub fn dc_sweep(
        &mut self,
        sweep: &DcSweep,
        options: &Options,
    ) -> Result<DcSweepResult, SimulationError> {
        options.parallelism.confined(|| {
            self.solvable()?;
            self.prepare(options)?;
            if sweep.levels.is_empty() {
                return Err(SimulationError::InvalidAnalysis(
                    "a DC sweep needs at least one level".to_owned(),
                ));
            }
            if let Some(level) = sweep.levels.iter().find(|level| !level.is_finite()) {
                return Err(SimulationError::InvalidAnalysis(format!(
                    "a DC sweep's levels must be finite, not {level:?}"
                )));
            }
            let swept = self.swept(sweep)?;
            let result = self.sweep_levels(swept, &sweep.levels, options);
            self.hold(swept, None)?;
            let points = result?;
            let mut values = Vec::with_capacity(points.len());
            let mut observations = Observations::default();
            let mut logic = Vec::with_capacity(points.len());
            for point in points {
                values.push(point.values);
                observations.push(&point.observation);
                logic.push(point.logic);
            }
            Ok(DcSweepResult {
                source: sweep.source.clone(),
                parameter: sweep.parameter.clone(),
                quantity: sweep
                    .parameter
                    .is_none()
                    .then(|| self.source_quantity(swept.device)),
                levels: sweep.levels.clone(),
                values,
                observations,
                logic,
            })
        })
    }

    /// What `sweep` sweeps.
    pub(crate) fn swept(&mut self, sweep: &DcSweep) -> Result<Swept, SimulationError> {
        let Some(name) = &sweep.parameter else {
            let device = self.source_index(&sweep.source)?;
            let own = self
                .state
                .runs
                .level(&self.elaborated.groups, device)
                .unwrap_or(0.0);
            return Ok(Swept {
                held: Held::Level(device),
                device,
                own,
            });
        };
        let device = self
            .elaborated
            .device(&sweep.source)
            .ok_or_else(|| SimulationError::UnknownSource(sweep.source.clone()))?;
        let parameter = self
            .elaborated
            .parameter(&sweep.source, name)
            .ok_or_else(|| {
                SimulationError::InvalidAnalysis(format!(
                    "`{}` writes no number `{name}` to sweep",
                    sweep.source
                ))
            })?;
        let own = self
            .values
            .get(parameter)
            .unwrap_or(self.elaborated.nominal()[parameter.index()]);
        Ok(Swept {
            held: Held::Number(parameter),
            device,
            own,
        })
    }

    /// The operating point at each of `levels` of `swept`. The logic settles at each from
    /// where it settled at the one before.
    fn sweep_levels(
        &mut self,
        swept: Swept,
        levels: &[f64],
        options: &Options,
    ) -> Result<Vec<OperatingPointResult>, SimulationError> {
        self.reset_logic();
        let mut values = vec![0.0; self.elaborated.unknowns.len()];
        let mut points = Vec::with_capacity(levels.len());
        let mut previous: Option<f64> = None;
        for &level in levels {
            let continued = previous.is_some_and(|from| {
                let mut set =
                    |simulator: &mut Self, at: &[f64]| simulator.hold(swept, Some(at[0])).is_ok();
                self.continue_along(&[(from, level)], &mut set, &mut values, options)
            });
            let solved = if continued {
                Solved {
                    values: std::mem::take(&mut values),
                    strategy: Strategy::Direct,
                    iterations: None,
                }
            } else {
                self.hold(swept, Some(level))?;
                let found = if previous.is_none() {
                    self.solve_from_zero(options)
                } else {
                    self.solve_dc(values.clone(), options)
                };
                match found {
                    Ok(solved) => solved,
                    // The first level may lie where no strategy finds the solution from
                    // scratch: there, continue to it from the circuit's own operating point.
                    Err(error) if previous.is_none() => Solved {
                        values: self.reach_first_level(swept, level, options).ok_or(error)?,
                        strategy: Strategy::Direct,
                        iterations: None,
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

    /// The solution with `swept` at `level`, continued from the operating point with it at
    /// its own level, if both are found.
    fn reach_first_level(
        &mut self,
        swept: Swept,
        level: f64,
        options: &Options,
    ) -> Option<Vec<f64>> {
        self.hold(swept, Some(swept.own)).ok()?;
        let mut values = self.solve_from_zero(options).ok()?.values;
        let mut set = |simulator: &mut Self, at: &[f64]| simulator.hold(swept, Some(at[0])).is_ok();
        self.continue_along(&[(swept.own, level)], &mut set, &mut values, options)
            .then_some(values)
    }

    /// Moves the circuit along `spans`, each coordinate from its first value to its second,
    /// each step putting it at a point through `set`, solving from each solution to the
    /// next by Newton iteration alone: in one step if it converges, else in steps halved
    /// after each failure and doubled after each success, every coordinate moving the same
    /// fraction of its span. Leaves the solution in `values` and returns true, or returns
    /// false, `values` untouched, once a step has shrunk to a thousandth of the way or
    /// `set` cannot put the circuit at a point; the circuit is left at the last point set.
    pub(crate) fn continue_along(
        &mut self,
        spans: &[(f64, f64)],
        set: &mut dyn FnMut(&mut Self, &[f64]) -> bool,
        values: &mut Vec<f64>,
        options: &Options,
    ) -> bool {
        let direct = Mode::Dc {
            source_scale: 1.0,
            shunt: 0.0,
            anchor: None,
            deflated: &[],
        };
        let mut level: Vec<f64> = spans.iter().map(|&(from, _)| from).collect();
        // Each coordinate's increment is its span times the scale, a power of two, so as
        // exactly as doubling and halving the increment itself.
        let mut scale = 1.0;
        let mut next = level.clone();
        let mut solution = values.clone();
        loop {
            let last = spans
                .iter()
                .zip(&level)
                .all(|(&(from, to), &at)| (to - at).abs() <= ((to - from) * scale).abs());
            for ((point, &(from, to)), &at) in next.iter_mut().zip(spans).zip(&level) {
                *point = if last { to } else { at + (to - from) * scale };
            }
            if !set(self, &next) {
                return false;
            }
            let mut trial = solution.clone();
            if self
                .newton(&mut trial, direct, options, options.dc_iterations)
                .is_ok()
            {
                solution = trial;
                if last {
                    break;
                }
                level.clone_from(&next);
                scale *= 2.0;
            } else {
                scale /= 2.0;
                if scale < 1.0 / SMALLEST_CONTINUATION {
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
            .elaborated
            .devices
            .iter()
            .position(|placed| placed.path() == path)
            .ok_or_else(|| SimulationError::UnknownSource(path.to_owned()))?;
        if self
            .state
            .runs
            .level(&self.elaborated.groups, index)
            .is_none()
        {
            return Err(SimulationError::NotASource(path.to_owned()));
        }
        Ok(index)
    }

    /// Whether the source of the device at `index` holds a voltage or drives a current.
    pub(crate) fn source_quantity(&self, index: usize) -> Quantity {
        let drives_current = self.elaborated.devices[index]
            .branches()
            .iter()
            .any(|branch| branch.kind == BranchKind::CurrentSource);
        if drives_current {
            Quantity::Current
        } else {
            Quantity::Voltage
        }
    }

    /// Holds `swept` at `level`, or releases it to the run's own values with `None`:
    /// prepares its instance again from the parameters of a source of that constant level,
    /// or from the device's own with the number replaced. Every other instance stays as it
    /// is.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::InvalidAnalysis`] when the level puts the device's numbers
    /// out of range, or gives it other unknowns or entries; the device is left as it was.
    pub(crate) fn hold(&mut self, swept: Swept, level: Option<f64>) -> Result<(), SimulationError> {
        let index = swept.device;
        let refused = |message: String| {
            SimulationError::InvalidAnalysis(format!(
                "{} at {}: {message}",
                self.elaborated.devices[index].location(),
                level.unwrap_or(swept.own)
            ))
        };
        let origin = match (level, swept.held) {
            (None, _) => None,
            (Some(level), Held::Level(_)) => Some(
                self.elaborated
                    .held_at(index, level)
                    .map_err(|problem| refused(problem.to_string()))?,
            ),
            (Some(level), Held::Number(parameter)) => {
                let mut values = self.values.clone();
                values.set(parameter, level);
                let patches = self.elaborated.patches_of(&values, index);
                Some(
                    self.elaborated
                        .patched(index, patches)
                        .map_err(|problem| refused(problem.to_string()))?,
                )
            }
        };
        let previous = match origin {
            Some(origin) => self.state.swept.insert(index, origin),
            None => self.state.swept.remove(&index),
        };
        // Its instance is prepared again from what it is now made of.
        let kept = self
            .state
            .prepare_again(&self.elaborated, index)
            .map_err(|error| refused(error.to_string()));
        if !matches!(kept, Ok(true)) {
            match previous {
                Some(previous) => self.state.swept.insert(index, previous),
                None => self.state.swept.remove(&index),
            };
            return Err(kept.err().unwrap_or_else(|| {
                refused("it would claim other unknowns or entries, which a sweep cannot".to_owned())
            }));
        }
        // At its own level, a device held leaves the DC equations as they are.
        match level.filter(|level| level.total_cmp(&swept.own).is_ne()) {
            Some(level) => self.state.held.insert(swept.held, level),
            None => self.state.held.remove(&swept.held),
        };
        Ok(())
    }
}
