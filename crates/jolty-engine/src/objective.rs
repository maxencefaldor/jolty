//! Scalar transient objectives and their gradients with respect to every numeric device
//! parameter, computed via [`Simulator::vjp`].
//!
//! Each supported functional is linear in the simulated waveform samples, so its output
//! derivatives form a cotangent pulled back to every parameter in a single backward adjoint
//! pass with one transposed solve per step. Gradients differentiate the discrete transient
//! solution at its accepted time steps and agree with the forward direct method to solver
//! rounding.
//!
//! Replay consumes the transient waveforms alongside only each step's four-number integration
//! formula, re-evaluating devices and re-factoring Jacobians once per point.

use std::collections::BTreeMap;

use serde::Deserialize;

use crate::derivative::{BackwardPass, Trajectory};
use crate::measure::{interpolated, trapezoid};
use crate::sensitivity::Parameter;
use crate::transient::Carried;
use crate::twins::Twins;
use crate::{Options, SimulationError, Simulator, Transient, TransientResult};

/// An objective: a scalar of a transient's waveforms, and its derivative with respect to
/// every number of every device's parameters.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Objective {
    /// The scalar.
    pub functional: Functional,
    /// The transient whose waveforms it reads.
    pub of: Transient,
}

/// A scalar of a transient's waveforms: of one unknown, by name, such as `out` or
/// `V1:current`.
///
/// In JSON, an object with one key, such as `{"integral": "out"}` or
/// `{"value": {"unknown": "out", "at": 1e-6}}`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Functional {
    /// The unknown's integral over the run, by the trapezoidal rule over the points the
    /// transient accepted.
    Integral(String),
    /// The unknown's value at a time within the run, interpolated linearly between the
    /// points the transient accepted on either side, as a measure reads it; a breakpoint of
    /// the transient there lands a point on it.
    Value {
        /// The unknown.
        unknown: String,
        /// The time, in seconds.
        at: f64,
    },
    /// The sum of the unknown's values at every point the transient accepted.
    Sum(String),
}

impl Functional {
    /// Returns the name of the target unknown read by this functional.
    fn unknown(&self) -> &str {
        match self {
            Self::Integral(unknown) | Self::Value { unknown, .. } | Self::Sum(unknown) => unknown,
        }
    }

    /// Evaluates the functional on `(times, values)` and returns `(value, sample_weights)`.
    fn read(&self, times: &[f64], values: &[f64]) -> (f64, Vec<f64>) {
        match *self {
            Self::Integral(_) => (trapezoid(times, values), trapezoid_weights(times)),
            Self::Value { at, .. } => (
                interpolated(at, times, values),
                interpolation_weights(at, times),
            ),
            Self::Sum(_) => (values.iter().sum(), vec![1.0; values.len()]),
        }
    }
}

/// Scalar value and full parameter gradient of an [`Objective`].
#[derive(Debug, Clone, PartialEq)]
pub struct ObjectiveResult {
    /// Objective value in SI base units.
    pub value: f64,
    /// Differentiated device parameters in gradient order.
    pub parameters: Vec<Parameter>,
    /// Gradient entries matching `parameters`.
    pub derivatives: Vec<f64>,
    /// Underlying transient waveforms whose statistics include the backward adjoint pass.
    pub result: TransientResult,
}

/// Accumulator callback receiving per-step gradient contributions across all parameters at once.
type Fold<'a> = &'a mut (dyn FnMut(&[f64]) + Send);

impl Simulator {
    /// Evaluates `objective` and computes its exact gradient with respect to every numeric
    /// device parameter in a single reverse adjoint pass.
    ///
    /// Discrete choice parameters (such as a MOSFET level) are excluded from differentiation.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::UnknownOutput`] if the target unknown does not exist,
    /// [`SimulationError::InvalidAnalysis`] if a sample time lies outside the transient span,
    /// and any error from [`Simulator::vjp`].
    pub fn objective(
        &mut self,
        objective: &Objective,
        options: &Options,
    ) -> Result<ObjectiveResult, SimulationError> {
        let mut derivatives = Vec::new();
        let (value, twins, result) = self.objective_with(objective, options, &mut |terms| {
            derivatives.resize(terms.len(), 0.0);
            for (total, term) in derivatives.iter_mut().zip(terms) {
                *total += term;
            }
        })?;
        Ok(ObjectiveResult {
            value,
            parameters: twins.parameters,
            derivatives,
            result,
        })
    }

    /// Evaluates `objective`, streaming per-step gradient terms into `fold` and returning
    /// `(value, twins, transient_result)` with backward-pass work included in statistics.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Simulator::objective`].
    pub(crate) fn objective_with(
        &mut self,
        objective: &Objective,
        options: &Options,
        fold: Fold<'_>,
    ) -> Result<(f64, Twins, TransientResult), SimulationError> {
        options.parallelism.confined(|| {
            self.solvable()?;
            let functional = &objective.functional;
            let name = functional.unknown();
            let unknown = self
                .unknown(name)
                .ok_or_else(|| SimulationError::UnknownOutput(name.to_owned()))?
                .index();
            self.replayable()?;
            let transient = &objective.of;
            if let Functional::Value { at, .. } = *functional
                && !(0.0..=transient.stop).contains(&at)
            {
                return Err(SimulationError::InvalidAnalysis(format!(
                    "an objective's time lies within the run, from 0 to {:?}, not {at:?}",
                    transient.stop
                )));
            }
            let mut twins = self.every_twin();
            let reading = self.read_meter(options.profile);
            let start = self.start_of(transient)?;
            let mut steps = Vec::new();
            let mut result =
                self.transient_with(transient, options, None, Carried::Steps(&mut steps))?;
            let values: Vec<f64> = result.values.iter().map(|x| x[unknown]).collect();
            let (value, weights) = functional.read(&result.times, &values);
            let dimension = self.elaborated.unknowns.len();
            let seed: BTreeMap<usize, Vec<f64>> = weights
                .iter()
                .enumerate()
                .filter(|&(_, &weight)| weight != 0.0)
                .map(|(point, &weight)| {
                    let mut values = vec![0.0; dimension];
                    values[unknown] = weight;
                    (point, values)
                })
                .collect();
            let pass = BackwardPass {
                trajectory: Trajectory {
                    waveforms: &result,
                    steps: &steps,
                },
                seeds: &[&seed],
                start: &start,
                stretch: None,
            };
            let mut each = |_: usize, terms: &[f64]| fold(terms);
            self.backward(&mut twins, pass, &mut each, options)?;
            (result.statistics.work, result.statistics.profile) = self.metered_since(&reading);
            Ok((value, twins, result))
        })
    }
}

/// Computes linear interpolation weights at `at` over sorted `times`, clamping outside the
/// endpoints to match `NumPy`'s `interp`.
fn interpolation_weights(at: f64, times: &[f64]) -> Vec<f64> {
    let mut weights = vec![0.0; times.len()];
    let after = times.partition_point(|&time| time < at);
    match (after.checked_sub(1), times.get(after)) {
        (Some(before), Some(&time)) if time > at => {
            let share = (at - times[before]) / (time - times[before]);
            weights[before] = 1.0 - share;
            weights[after] = share;
        }
        // Exact match or before the first sample: weight falls on `after`.
        (_, Some(_)) => weights[after] = 1.0,
        // Past the final sample: weight falls on the last sample.
        (Some(before), None) => weights[before] = 1.0,
        (None, None) => {}
    }
    weights
}

/// Computes trapezoidal integration weights over `times`, assigning each sample half the
/// span of its adjacent intervals.
fn trapezoid_weights(times: &[f64]) -> Vec<f64> {
    let last = times.len().saturating_sub(1);
    (0..times.len())
        .map(|point| (times[(point + 1).min(last)] - times[point.saturating_sub(1)]) / 2.0)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::trapezoid_weights;
    use crate::kernels::Recording;
    use crate::{
        Analysis, AnalysisResult, Derivatives, Differentiated, Functional, Objective, Options,
        Sensitivity, SimulationError, Simulator, Transient,
    };

    /// Maximum `max_step` multiples simulated per corpus transient check, long enough to cross
    /// corpus switching edges starting at 10 µs (50 maximum steps).
    const LONGEST_STEPS: f64 = 100.0;

    /// Maximum relative disagreement allowed between adjoint and direct gradients over the
    /// corpus (measured worst rounding difference 4.8e-13 on 16 November 2026; typical 1e-15).
    const TOLERANCE: f64 = 1e-11;

    /// Direct-method integral sensitivities across every unknown of a transient.
    struct Integrated {
        times: Vec<f64>,
        /// Integral derivatives indexed by `[unknown][parameter]`.
        derivatives: Vec<Vec<f64>>,
        /// Per-parameter maximum summed term magnitude across unknowns, scaling solve rounding.
        anywhere: Vec<f64>,
    }

    /// Computes trapezoidal integral sensitivities for every unknown via the forward direct
    /// method.
    fn direct(
        simulator: &mut Simulator,
        transient: &Transient,
        options: &Options,
    ) -> Result<Integrated, SimulationError> {
        let sensitivity = Sensitivity {
            outputs: simulator
                .unknowns()
                .iter()
                .map(|unknown| unknown.name.clone())
                .collect(),
            of: Differentiated::Transient(transient.clone()),
        };
        let found = simulator.sensitivity(&sensitivity, options)?;
        let AnalysisResult::Transient(waveforms) = *found.result else {
            panic!("a transient's waveforms");
        };
        let Derivatives::Real(points) = found.derivatives else {
            panic!("real derivatives");
        };
        let weights = trapezoid_weights(&waveforms.times);
        let zeros = vec![vec![0.0; found.parameters.len()]; sensitivity.outputs.len()];
        let (mut derivatives, mut magnitudes) = (zeros.clone(), zeros);
        for (weight, outputs) in weights.iter().zip(&points) {
            let by_output = derivatives.iter_mut().zip(&mut magnitudes).zip(outputs);
            for ((sums, magnitudes), found) in by_output {
                for ((sum, magnitude), derivative) in sums.iter_mut().zip(magnitudes).zip(found) {
                    *sum += weight * derivative;
                    *magnitude += (weight * derivative).abs();
                }
            }
        }
        let anywhere = (0..found.parameters.len())
            .map(|parameter| {
                magnitudes
                    .iter()
                    .map(|magnitudes| magnitudes[parameter])
                    .fold(0.0, f64::max)
            })
            .collect();
        Ok(Integrated {
            times: waveforms.times,
            derivatives,
            anywhere,
        })
    }

    /// Returns the worst relative disagreement across all parameters between adjoint and
    /// direct integral gradients for `unknowns` over `transient`.
    fn disagreement(
        simulator: &mut Simulator,
        transient: &Transient,
        unknowns: &[usize],
        options: &Options,
    ) -> Result<f64, SimulationError> {
        let integrated = direct(simulator, transient, options)?;
        let mut worst = 0.0_f64;
        for &unknown in unknowns {
            let name = simulator.unknowns()[unknown].name.clone();
            let objective = Objective {
                functional: Functional::Integral(name.clone()),
                of: transient.clone(),
            };
            let count = integrated.anywhere.len();
            let (mut sums, mut magnitudes) = (vec![0.0; count], vec![0.0; count]);
            let (_, twins, result) =
                simulator.objective_with(&objective, options, &mut |terms| {
                    for ((sum, magnitude), term) in sums.iter_mut().zip(&mut magnitudes).zip(terms)
                    {
                        *sum += term;
                        *magnitude += term.abs();
                    }
                })?;
            assert_eq!(result.times, integrated.times, "the same steps");
            for (index, parameter) in twins.parameters.iter().enumerate() {
                let (adjoint, direct) = (sums[index], integrated.derivatives[unknown][index]);
                let gap = (adjoint - direct).abs();
                let scale = magnitudes[index] + integrated.anywhere[index];
                let relative = if gap == 0.0 { 0.0 } else { gap / scale };
                assert!(
                    relative <= TOLERANCE,
                    "d∫{name}/d{}.{}: adjoint {adjoint:e}, direct {direct:e}, terms of \
                     {scale:e}",
                    parameter.instance,
                    parameter.name,
                );
                worst = worst.max(relative);
            }
        }
        Ok(worst)
    }

    /// Verifies that the adjoint matches the direct method on every transient run of shard
    /// `shard` (`run_index % shards == shard`) of the kernel corpus. Sharding by transient
    /// run rather than by circuit balances load evenly across shards (3 or 4 runs each).
    fn check_adjoint_corpus_shard(shard: usize, shards: usize) {
        let (mut checked, mut refused, mut worst) = (0_usize, 0_usize, 0.0_f64);
        let mut run_index = 0_usize;
        for record in Recording::corpus() {
            let transient_indices: Vec<usize> = record
                .runs
                .iter()
                .enumerate()
                .filter_map(|(position, (analysis, _))| {
                    matches!(analysis, Analysis::Transient(_)).then_some(position)
                })
                .collect();
            if transient_indices.is_empty() {
                continue;
            }
            let own: Vec<usize> = transient_indices
                .into_iter()
                .filter(|_| {
                    let keep = run_index % shards == shard;
                    run_index += 1;
                    keep
                })
                .collect();
            if own.is_empty() {
                continue;
            }
            let (circuit, mut simulator) = record.simulator();
            let unknowns = match simulator.unknowns().len() {
                0 => continue,
                1 => vec![0],
                count => vec![0, count - 1],
            };
            let replayable = simulator.replayable().is_ok();
            for position in own {
                let (Analysis::Transient(transient), options) = &record.runs[position] else {
                    continue;
                };
                let longest = transient.max_step.unwrap_or(transient.stop / 50.0);
                let transient = Transient {
                    stop: transient.stop.min(LONGEST_STEPS * longest),
                    ..transient.clone()
                };
                match disagreement(&mut simulator, &transient, &unknowns, options) {
                    Ok(relative) => {
                        assert!(
                            replayable,
                            "{}: simulator is not replayable yet disagreement returned Ok({relative:e})",
                            circuit.top
                        );
                        checked += 1;
                        worst = worst.max(relative);
                    }
                    Err(SimulationError::InvalidAnalysis(_) | SimulationError::Delayed { .. })
                        if !replayable =>
                    {
                        refused += 1;
                    }
                    Err(error) => panic!("{}: {error}", circuit.top),
                }
            }
        }
        assert!(
            checked + refused > 0,
            "shard {shard}/{shards} had no transient runs (worst {worst:e})"
        );
    }

    #[test]
    fn corpus_transient_count_matches_expected() {
        let (mut replayable, mut refused) = (0_usize, 0_usize);
        for record in Recording::corpus() {
            let count = record
                .runs
                .iter()
                .filter(|(analysis, _)| matches!(analysis, Analysis::Transient(_)))
                .count();
            if count == 0 {
                continue;
            }
            let (_, simulator) = record.simulator();
            if simulator.unknowns().is_empty() {
                continue;
            }
            if simulator.replayable().is_ok() {
                replayable += count;
            } else {
                refused += count;
            }
        }
        // 53 checked plus 6 refused (logic or delay lines).
        assert_eq!((replayable, refused), (53, 6));
    }

    crate::kernels::sharded_tests!(
        check_adjoint_corpus_shard;
        the_adjoint_agrees_with_the_direct_method_on_corpus_shard_00,
        the_adjoint_agrees_with_the_direct_method_on_corpus_shard_01,
        the_adjoint_agrees_with_the_direct_method_on_corpus_shard_02,
        the_adjoint_agrees_with_the_direct_method_on_corpus_shard_03,
        the_adjoint_agrees_with_the_direct_method_on_corpus_shard_04,
        the_adjoint_agrees_with_the_direct_method_on_corpus_shard_05,
        the_adjoint_agrees_with_the_direct_method_on_corpus_shard_06,
        the_adjoint_agrees_with_the_direct_method_on_corpus_shard_07,
        the_adjoint_agrees_with_the_direct_method_on_corpus_shard_08,
        the_adjoint_agrees_with_the_direct_method_on_corpus_shard_09,
        the_adjoint_agrees_with_the_direct_method_on_corpus_shard_10,
        the_adjoint_agrees_with_the_direct_method_on_corpus_shard_11,
        the_adjoint_agrees_with_the_direct_method_on_corpus_shard_12,
        the_adjoint_agrees_with_the_direct_method_on_corpus_shard_13,
        the_adjoint_agrees_with_the_direct_method_on_corpus_shard_14,
        the_adjoint_agrees_with_the_direct_method_on_corpus_shard_15,
    );
}
