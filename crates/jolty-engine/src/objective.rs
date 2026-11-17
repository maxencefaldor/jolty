//! Objectives: a scalar of a transient's waveforms, and its derivative with respect to every
//! number of every device's parameters, by the transient's VJP ([`Simulator::vjp`]).
//!
//! The scalar is a linear function of the waveforms, or interpolates them linearly, so its
//! derivative with respect to the unknown at each point is a cotangent on them, which the
//! discrete adjoint of the steps the transient took carries back to every parameter in one
//! backward pass, one transposed solve a step whatever the number of parameters. The
//! derivatives are those of the solution the transient computed, with its steps where they
//! fell, as a derivative moves no step, and agree with the direct method's to the rounding
//! of their solves.
//!
//! The backward pass reads the waveforms the transient keeps as its result, and keeps
//! besides only each step's formula, four numbers a step; it evaluates the circuit and
//! factors its Jacobian once more at each point, as the direct method does.

use std::collections::BTreeMap;

use serde::Deserialize;

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
    /// The unknown it reads, by name.
    fn unknown(&self) -> &str {
        match self {
            Self::Integral(unknown) | Self::Value { unknown, .. } | Self::Sum(unknown) => unknown,
        }
    }

    /// Its value on `values`, an unknown's at each of `times`, and its derivative with
    /// respect to the unknown at each.
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

/// The value of an [`Objective`], and its derivatives.
#[derive(Debug, Clone, PartialEq)]
pub struct ObjectiveResult {
    /// The objective's value, in SI base units.
    pub value: f64,
    /// Every parameter the derivatives are with respect to.
    pub parameters: Vec<Parameter>,
    /// The objective's derivative with respect to each parameter, in their order.
    pub derivatives: Vec<f64>,
    /// The transient's waveforms; its work counts the backward pass's.
    pub result: TransientResult,
}

/// What takes each term of an objective's derivatives, every parameter's at once.
type Fold<'a> = &'a mut (dyn FnMut(&[f64]) + Send);

impl Simulator {
    /// The value of `objective`, and its derivative with respect to every number of every
    /// device's parameters, by the adjoint method: one backward pass through the steps the
    /// transient took, whatever the number of parameters.
    ///
    /// The derivatives are exact, those of the solution the transient computed, with its
    /// steps where they fell. A number the device's model reads as a choice, such as a
    /// MOSFET's level, is no real parameter and has none.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::UnknownOutput`] for a functional of no unknown,
    /// [`SimulationError::InvalidAnalysis`] for a time outside the run, and the errors of
    /// [`Simulator::vjp`] over time.
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

    /// The value of `objective`, the twins of every parameter, in their order, and the
    /// transient's waveforms, whose work counts the backward pass's; `fold` takes each term
    /// of the derivatives with respect to the twins' numbers, all twins' at once, which
    /// sum to the derivatives.
    ///
    /// # Errors
    ///
    /// As [`Simulator::objective`].
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
            let waveforms = (&result, steps.as_slice());
            let mut each = |_: usize, terms: &[f64]| fold(terms);
            self.backward(
                &mut twins,
                waveforms,
                &[&seed],
                (&start, None),
                &mut each,
                options,
            )?;
            (result.statistics.work, result.statistics.profile) = self.metered_since(&reading);
            Ok((value, twins, result))
        })
    }
}

/// The derivative of the linear interpolation at `at` between `times` with respect to the
/// value at each time: of the two on either side, each its share of the gap, or one at a
/// time `at` is, and at the end it lies beyond, as numpy's `interp` reads them.
fn interpolation_weights(at: f64, times: &[f64]) -> Vec<f64> {
    let mut weights = vec![0.0; times.len()];
    let after = times.partition_point(|&time| time < at);
    match (after.checked_sub(1), times.get(after)) {
        (Some(before), Some(&time)) if time > at => {
            let share = (at - times[before]) / (time - times[before]);
            weights[before] = 1.0 - share;
            weights[after] = share;
        }
        // At a time, or before the first, which reads the first value.
        (_, Some(_)) => weights[after] = 1.0,
        // After the last, which reads the last value.
        (Some(before), None) => weights[before] = 1.0,
        (None, None) => {}
    }
    weights
}

/// The derivative of the trapezoidal rule's integral over `times` with respect to the value
/// at each time: half the span of the steps on either side of it.
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

    /// The longest steps of a corpus transient each check runs for: enough for the
    /// corpus's switches, whose sources start to move ten microseconds in, fifty of their
    /// longest steps.
    const LONGEST_STEPS: f64 = 100.0;

    /// The largest disagreement allowed between the two methods, relative to the
    /// magnitudes of their terms: they sum the same terms in other orders, from solves of
    /// the same matrices, so they differ by the rounding of those sums and solves, at
    /// worst 4.8e-13 over the corpus (elm, 16 November 2026), and most within 1e-15; a
    /// term that either leaves out shows by its own size.
    const TOLERANCE: f64 = 1e-11;

    /// What the direct method finds of the integral of every unknown over a transient.
    struct Integrated {
        times: Vec<f64>,
        /// By unknown, then parameter: the integral's derivative.
        derivatives: Vec<Vec<f64>>,
        /// By parameter: the largest, over the unknowns, of the magnitudes of the terms
        /// of its derivative summed; a solve's rounding is relative to the whole of the
        /// vector it solves for.
        anywhere: Vec<f64>,
    }

    /// The integral of every unknown over `transient`, by the direct method, each point's
    /// derivatives weighted by the trapezoidal rule's.
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

    /// The largest disagreement, over every parameter, between the adjoint's derivative
    /// of the integral of each of `unknowns` over `transient` and the direct method's,
    /// relative to the magnitudes of the adjoint's terms and the direct method's, summed;
    /// or the error both return.
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

    #[test]
    fn the_adjoint_agrees_with_the_direct_method_on_every_corpus_transient() {
        let (mut checked, mut refused, mut worst) = (0, 0, 0.0_f64);
        for record in &Recording::corpus() {
            let (circuit, mut simulator) = record.simulator();
            let unknowns = match simulator.unknowns().len() {
                0 => continue,
                1 => vec![0],
                count => vec![0, count - 1],
            };
            for (analysis, options) in &record.runs {
                let Analysis::Transient(transient) = analysis else {
                    continue;
                };
                let longest = transient.max_step.unwrap_or(transient.stop / 50.0);
                let transient = Transient {
                    stop: transient.stop.min(LONGEST_STEPS * longest),
                    ..transient.clone()
                };
                match disagreement(&mut simulator, &transient, &unknowns, options) {
                    Ok(relative) => {
                        checked += 1;
                        worst = worst.max(relative);
                    }
                    Err(SimulationError::InvalidAnalysis(_) | SimulationError::Delayed { .. }) => {
                        refused += 1;
                    }
                    Err(error) => panic!("{}: {error}", circuit.top),
                }
            }
        }
        // Every transient but those with logic or delay lines, those from initial
        // conditions among them.
        assert_eq!((checked, refused), (53, 6), "worst {worst:e}");
    }
}
