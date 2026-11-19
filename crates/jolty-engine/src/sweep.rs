//! DC sweep analysis computing operating points across values of an independent source or
//! device parameter.
//!
//! Each sweep step updates only the swept device (rebuilt as a constant source or with its
//! swept parameter replaced), leaving all other devices unchanged and warm-starting Newton
//! iteration from the previous solution.

use jolty_models::{BranchKind, LogicLevel, Nature};
use serde::Deserialize;

use crate::newton::Mode;
use crate::operating_point::Solved;
use crate::state::Held;
use crate::{OperatingPointResult, Options, RecordedSeries, SimulationError, Simulator, Strategy};

/// Reciprocal of the minimum natural-parameter continuation fraction (`2^10 = 1024`, ten
/// exact power-of-two step halvings on Newton failure); ngspice 47 instead retries failed
/// points from scratch (`dctrcurv.c`, lines 306 to 314).
const MIN_CONTINUATION: f64 = 1024.0;

/// DC sweep specification stepping one independent source or one numeric device parameter.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DcSweep {
    /// Instance path of the swept device (such as `V1` or `bias/I1`); sweeps the source's
    /// DC value regardless of waveform unless `parameter` is set.
    pub source: String,
    /// Parameter path within the instance (such as `resistance`), or `None` to sweep an
    /// independent source value.
    #[serde(default)]
    pub parameter: Option<String>,
    /// Ordered sweep values in volts, amperes, or SI base units of `parameter`.
    pub values: Vec<f64>,
}

/// Operating points collected across a DC sweep.
#[derive(Debug, Clone, PartialEq)]
pub struct DcSweepResult {
    /// Instance path of the swept device.
    pub source: String,
    /// Swept parameter path within the instance, or `None` for an independent source sweep.
    pub parameter: Option<String>,
    /// Physical quantity (`Voltage` or `Current`) for a source sweep, or `None` for a
    /// parameter sweep.
    pub nature: Option<Nature>,
    /// Ordered swept values.
    pub swept: Vec<f64>,
    /// Unknown values at each swept point.
    pub values: Vec<Vec<f64>>,
    /// Recorded device currents, charges, derivatives, and variables across swept points.
    pub recorded: RecordedSeries,
    /// Logic net levels at each swept point.
    pub logic: Vec<Vec<LogicLevel>>,
    /// Convergence strategy used at each swept point.
    pub strategies: Vec<Strategy>,
    /// Newton iteration counts at each swept point, matching
    /// [`OperatingPointResult::iterations`].
    pub iterations: Vec<usize>,
}

/// Resolved target of a DC sweep: held key, device index, and baseline value.
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
    /// Returns [`SimulationError::UnknownSource`] for a missing instance path,
    /// [`SimulationError::NotASource`] for a non-source value sweep,
    /// [`SimulationError::InvalidAnalysis`] for empty or non-finite values, unknown
    /// parameter paths, or topology-changing values, and any error from
    /// [`Simulator::operating_point`].
    pub fn dc_sweep(
        &mut self,
        sweep: &DcSweep,
        options: &Options,
    ) -> Result<DcSweepResult, SimulationError> {
        options.parallelism.confined(|| {
            self.solvable()?;
            self.prepare(options)?;
            if sweep.values.is_empty() {
                return Err(SimulationError::InvalidAnalysis(
                    "a DC sweep needs at least one value".to_owned(),
                ));
            }
            if let Some(value) = sweep.values.iter().find(|value| !value.is_finite()) {
                return Err(SimulationError::InvalidAnalysis(format!(
                    "DC sweep values require finite numbers; got {value:?}"
                )));
            }
            let swept = self.swept(sweep)?;
            let result = self.sweep_values(swept, &sweep.values, options);
            self.hold(swept, None)?;
            let points = result?;
            let mut values = Vec::with_capacity(points.len());
            let mut recorded = RecordedSeries::default();
            let mut logic = Vec::with_capacity(points.len());
            let mut strategies = Vec::with_capacity(points.len());
            let mut iterations = Vec::with_capacity(points.len());
            for point in points {
                values.push(point.values);
                recorded.push(&point.recorded);
                logic.push(point.logic);
                strategies.push(point.strategy);
                iterations.push(point.iterations);
            }
            Ok(DcSweepResult {
                source: sweep.source.clone(),
                parameter: sweep.parameter.clone(),
                nature: sweep
                    .parameter
                    .is_none()
                    .then(|| self.source_nature(swept.device)),
                swept: sweep.values.clone(),
                values,
                recorded,
                logic,
                strategies,
                iterations,
            })
        })
    }

    /// Resolves the target device, parameter handle, and baseline value for `sweep`.
    pub(crate) fn swept(&mut self, sweep: &DcSweep) -> Result<Swept, SimulationError> {
        let Some(name) = &sweep.parameter else {
            let device = self.source_index(&sweep.source)?;
            let own = self
                .state
                .runs
                .dc_value(&self.elaborated.groups, device)
                .unwrap_or(0.0);
            return Ok(Swept {
                held: Held::Value(device),
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

    /// Solves the operating point at each value in `swept_values`, carrying settled logic
    /// state from each point to the next.
    fn sweep_values(
        &mut self,
        swept: Swept,
        swept_values: &[f64],
        options: &Options,
    ) -> Result<Vec<OperatingPointResult>, SimulationError> {
        self.reset_logic();
        let mut values = vec![0.0; self.elaborated.unknowns.len()];
        let mut points = Vec::with_capacity(swept_values.len());
        let mut previous: Option<f64> = None;
        for &target in swept_values {
            let before = self.state.workspace.meter.iterations();
            let continued = previous.is_some_and(|from| {
                let mut set =
                    |simulator: &mut Self, at: &[f64]| simulator.hold(swept, Some(at[0])).is_ok();
                self.continue_along(&[(from, target)], &mut set, &mut values, options)
            });
            // Continuation from the value before, which runs no iteration at the first.
            let continuation = self.iterations_since(before);
            let solved = if continued {
                Solved {
                    values: std::mem::take(&mut values),
                    strategy: Strategy::Direct,
                    iterations: continuation,
                    from_guess: false,
                }
            } else {
                self.hold(swept, Some(target))?;
                let found = if previous.is_none() {
                    self.solve_from_zero(options)
                } else {
                    self.solve_dc(&values, options)
                };
                match found {
                    Ok(solved) => Solved {
                        iterations: continuation + solved.iterations,
                        from_guess: solved.from_guess && continuation == 0,
                        ..solved
                    },
                    // When the first point resists direct solve, continue from the nominal
                    // operating point instead.
                    Err(error) if previous.is_none() => Solved {
                        values: self
                            .reach_first_value(swept, target, options)
                            .ok_or(error)?,
                        strategy: Strategy::Direct,
                        iterations: self.iterations_since(before),
                        from_guess: false,
                    },
                    Err(error) => return Err(error),
                }
            };
            let point = self.settle(solved, options)?;
            previous = Some(target);
            values.clone_from(&point.values);
            points.push(point);
        }
        Ok(points)
    }

    /// Reaches `value` by continuing from the operating point solved at `swept`'s baseline
    /// value, returning `None` if either step fails.
    fn reach_first_value(
        &mut self,
        swept: Swept,
        value: f64,
        options: &Options,
    ) -> Option<Vec<f64>> {
        self.hold(swept, Some(swept.own)).ok()?;
        let mut values = self.solve_from_zero(options).ok()?.values;
        let mut set = |simulator: &mut Self, at: &[f64]| simulator.hold(swept, Some(at[0])).is_ok();
        self.continue_along(&[(swept.own, value)], &mut set, &mut values, options)
            .then_some(values)
    }

    /// Continues the circuit along `spans` via adaptive Newton stepping, updating coordinates
    /// through `set` at each step.
    ///
    /// Halves the step fraction after each Newton failure and doubles after each success.
    /// Writes the final solution into `values` and returns `true` on completion, or leaves
    /// `values` untouched and returns `false` if step size drops below `1 / MIN_CONTINUATION`
    /// or `set` fails.
    pub(crate) fn continue_along(
        &mut self,
        spans: &[(f64, f64)],
        set: &mut dyn FnMut(&mut Self, &[f64]) -> bool,
        values: &mut Vec<f64>,
        options: &Options,
    ) -> bool {
        let direct = Mode::direct(&[]);
        let mut current: Vec<f64> = spans.iter().map(|&(from, _)| from).collect();
        // Each coordinate's increment is its span times the scale, a power of two, so as
        // exactly as doubling and halving the increment itself.
        let mut scale = 1.0;
        let mut next = current.clone();
        let mut solution = values.clone();
        loop {
            let last = spans
                .iter()
                .zip(&current)
                .all(|(&(from, to), &at)| (to - at).abs() <= ((to - from) * scale).abs());
            for ((point, &(from, to)), &at) in next.iter_mut().zip(spans).zip(&current) {
                *point = if last { to } else { at + (to - from) * scale };
            }
            if !set(self, &next) {
                return false;
            }
            let mut trial = solution.clone();
            if self
                .newton(&mut trial, direct, options, options.max_dc_iterations)
                .is_ok()
            {
                solution = trial;
                if last {
                    break;
                }
                current.clone_from(&next);
                scale *= 2.0;
            } else {
                scale /= 2.0;
                if scale < 1.0 / MIN_CONTINUATION {
                    return false;
                }
            }
        }
        *values = solution;
        true
    }

    /// Returns the device index of the independent source at `path`.
    pub(crate) fn source_index(&self, path: &str) -> Result<usize, SimulationError> {
        let index = self
            .elaborated
            .device(path)
            .ok_or_else(|| SimulationError::UnknownSource(path.to_owned()))?;
        if self
            .state
            .runs
            .dc_value(&self.elaborated.groups, index)
            .is_none()
        {
            return Err(SimulationError::NotASource(path.to_owned()));
        }
        Ok(index)
    }

    /// Returns whether the independent source at `index` drives voltage or current.
    pub(crate) fn source_nature(&self, index: usize) -> Nature {
        let drives_current = self.elaborated.devices[index]
            .branches()
            .iter()
            .any(|branch| branch.kind == BranchKind::CurrentSource);
        if drives_current {
            Nature::Current
        } else {
            Nature::Voltage
        }
    }

    /// Holds `swept` at `value`, or restores the baseline value when `value` is `None`.
    ///
    /// Re-prepares only the swept device instance from a constant source or patched
    /// parameter set, leaving all other devices unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::InvalidAnalysis`] without modifying the device when
    /// `value` is out of range or alters the device's unknowns or matrix entries.
    pub(crate) fn hold(&mut self, swept: Swept, value: Option<f64>) -> Result<(), SimulationError> {
        let index = swept.device;
        let refused = |message: String| {
            SimulationError::InvalidAnalysis(format!(
                "{} at {}: {message}",
                self.elaborated.devices[index].location(),
                value.unwrap_or(swept.own)
            ))
        };
        let origin = match (value, swept.held) {
            (None, _) => None,
            (Some(value), Held::Value(_)) => Some(
                self.elaborated
                    .held_at(index, value)
                    .map_err(|problem| refused(problem.to_string()))?,
            ),
            (Some(value), Held::Number(parameter)) => {
                let mut values = self.values.clone();
                values.set(parameter, value);
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
        // Re-prepare the swept device instance from its updated specification.
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
        // Holding at the baseline value leaves the DC equations unchanged.
        match value.filter(|value| value.total_cmp(&swept.own).is_ne()) {
            Some(value) => self.state.held.insert(swept.held, value),
            None => self.state.held.remove(&swept.held),
        };
        Ok(())
    }
}
