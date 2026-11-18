//! Mismatch: how far each output spreads, to first order, as the numbers a sample draws at
//! random move the devices' parameters, and which of those numbers it spreads with.
//!
//! Each number drawn at random is a random variable `z` of mean `μ` and standard deviation
//! `σ`, which moves some of the devices' parameters `p` by `∂p/∂z`. An output `y` moves with
//! it by `dy/dz = Σ (dy/dp) (∂p/∂z)`, where `dy/dp` is the sensitivity analysis's derivative,
//! exact, by the adjoint at an operating point or a frequency, so that one solve per output
//! serves every random variable. The random variables are drawn apart from one another, so
//! to first order `y` spreads with the standard deviation `√(Σ (dy/dz · σ)²)`, each random
//! variable's share of its variance `(dy/dz · σ)² / Σ (dy/dz · σ)²`: what a designer asks of
//! mismatch first, from one analysis rather than many samples, as Spectre's `dcmatch` and
//! `acmatch` give it.
//!
//! A circuit's tolerances are the random variables of [`Toleranced::random_variables`], at
//! the values the parameters take in the run; a process kit's statistical functions are
//! those the plan gives, with the derivatives its reader found.

use std::collections::BTreeMap;

use serde::Deserialize;

use crate::derivative::named_of;
use crate::sensitivity::Parameter;
use crate::variants::Toleranced;
use crate::{AnalysisResult, Derivatives, Differentiated, Options, SimulationError, Simulator};

/// A mismatch analysis: how far each unknown of `outputs` spreads, to first order, at each
/// point of the analysis `of`, with the circuit's tolerances and the random variables given.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Mismatch {
    /// The unknowns, by name, such as `out` or `V1:current`.
    pub outputs: Vec<String>,
    /// The analysis differentiated: the operating point by default.
    #[serde(default)]
    pub of: Differentiated,
    /// The circuit's toleranced numbers, as a Monte Carlo analysis draws them.
    #[serde(default)]
    pub toleranced: Vec<Toleranced>,
    /// The random variables besides the tolerances', such as a process kit's statistical
    /// functions.
    #[serde(default)]
    pub random_variables: Vec<RandomVariable>,
}

/// A number a sample draws at random, and how it moves the devices' parameters.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RandomVariable {
    /// The path of the instance that draws it: the part whose tolerance it is, or the
    /// placement whose definitions call a process kit's statistical function; none for a
    /// lot's.
    #[serde(default)]
    pub instance: Option<String>,
    /// The name of the lot whose shared spread it is, if it is one.
    #[serde(default)]
    pub lot: Option<String>,
    /// What it is within its instance or lot: the parameter a tolerance spreads, such as
    /// `resistance`, or which call of a statistical function it is, such as `vth0#0`.
    pub name: String,
    /// Its mean, which the circuit's parameters are at.
    pub mean: f64,
    /// Its standard deviation.
    pub standard_deviation: f64,
    /// The derivative of each number it moves with respect to it, by the device's path, then
    /// by the number's path among its parameters, in SI base units per unit of it.
    pub derivatives: BTreeMap<String, BTreeMap<String, f64>>,
}

/// The derivatives of a [`Mismatch`] analysis's outputs with respect to its random
/// variables.
#[derive(Debug, Clone, PartialEq)]
pub struct MismatchResult {
    /// The unknowns, by name, in the order of the derivatives' outputs.
    pub outputs: Vec<String>,
    /// Every random variable, the tolerances' first, each part's own then each lot's, in the
    /// order of the circuit's toleranced numbers, then those given.
    pub random_variables: Vec<RandomVariable>,
    /// The derivatives of the outputs with respect to the random variables, by point, then
    /// output, then random variable, as a sensitivity analysis's are by parameter.
    pub derivatives: Derivatives,
    /// For an oscillator's periodic steady state, the derivative of its period with respect
    /// to each random variable.
    pub period_derivatives: Option<Vec<f64>>,
    /// The result of the analysis differentiated.
    pub result: Box<AnalysisResult>,
}

impl Simulator {
    /// The derivative of each unknown of `mismatch` with respect to each of its random
    /// variables, at each point of the analysis it differentiates.
    ///
    /// The derivatives are a sensitivity analysis's, exact, each summed over the
    /// parameters a random variable moves.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::UnknownParameters`] for every number a random variable
    /// moves that no device's parameters hold as a real number,
    /// [`SimulationError::InvalidAnalysis`] for a random variable whose mean, standard
    /// deviation or derivative is not finite, and the errors of
    /// [`Simulator::sensitivity`].
    pub fn mismatch(
        &mut self,
        mismatch: &Mismatch,
        options: &Options,
    ) -> Result<MismatchResult, SimulationError> {
        options.parallelism.confined(|| {
            self.solvable()?;
            let mut moved = Moved::default();
            let mut random_variables = self.toleranced(&mismatch.toleranced, &mut moved)?;
            random_variables.extend(mismatch.random_variables.iter().cloned());
            for variable in &random_variables {
                finite(variable)?;
                for (path, numbers) in &variable.derivatives {
                    for name in numbers.keys() {
                        moved.column(self, path, name);
                    }
                }
            }
            let (twins, unread) = self.unit_twins(moved.parameters);
            moved.missing.extend(named_of(&unread));
            if !moved.missing.is_empty() {
                return Err(SimulationError::UnknownParameters(moved.missing));
            }
            let weights: Vec<Vec<(usize, f64)>> = random_variables
                .iter()
                .map(|variable| {
                    variable
                        .derivatives
                        .iter()
                        .flat_map(|(path, numbers)| {
                            numbers.iter().map(|(name, derivative)| {
                                (moved.columns[&(path.clone(), name.clone())], *derivative)
                            })
                        })
                        .collect()
                })
                .collect();
            let found = self.differentiate(&mismatch.outputs, &mismatch.of, twins, options)?;
            let derivatives = match found.derivatives {
                Derivatives::Real(points) => Derivatives::Real(combined(&points, &weights)),
                Derivatives::Complex(points) => Derivatives::Complex(combined(&points, &weights)),
            };
            let period_derivatives = found.period_derivatives.map(|period| {
                weights
                    .iter()
                    .map(|weighed| {
                        weighed
                            .iter()
                            .map(|&(at, weight)| weight * period[at])
                            .sum()
                    })
                    .collect()
            });
            Ok(MismatchResult {
                outputs: mismatch.outputs.clone(),
                random_variables,
                derivatives,
                period_derivatives,
                result: Box::new(found.result),
            })
        })
    }

    /// The random variables of the toleranced numbers `toleranced`, each part's own then each
    /// lot's, a lot's shared by the parts that name it, with the derivatives of the numbers
    /// at their values in the run; the numbers added to `moved`.
    fn toleranced(
        &self,
        toleranced: &[Toleranced],
        moved: &mut Moved,
    ) -> Result<Vec<RandomVariable>, SimulationError> {
        let mut own = Vec::with_capacity(toleranced.len());
        let mut lots: Vec<RandomVariable> = Vec::new();
        for number in toleranced {
            let Some(at) = moved.column(self, &number.instance, &number.parameter) else {
                continue;
            };
            let value = moved.parameters[at].1.value;
            let (spread, shared) = number.random_variables();
            own.push(RandomVariable {
                instance: Some(number.instance.clone()),
                lot: None,
                name: number.parameter.clone(),
                mean: 0.0,
                standard_deviation: spread.standard_deviation,
                derivatives: moving(&number.instance, &number.parameter, spread.slope * value),
            });
            let Some((lot, spread)) = shared else {
                continue;
            };
            let derivative = spread.slope * value;
            let found = lots.iter_mut().find(|variable| {
                variable.lot.as_deref() == Some(lot) && variable.name == number.parameter
            });
            if let Some(variable) = found {
                variable
                    .derivatives
                    .entry(number.instance.clone())
                    .or_default()
                    .insert(number.parameter.clone(), derivative);
            } else {
                lots.push(RandomVariable {
                    instance: None,
                    lot: Some(lot.to_owned()),
                    name: number.parameter.clone(),
                    mean: 0.0,
                    standard_deviation: spread.standard_deviation,
                    derivatives: moving(&number.instance, &number.parameter, derivative),
                });
            }
        }
        if !moved.missing.is_empty() {
            return Err(SimulationError::UnknownParameters(moved.missing.clone()));
        }
        own.extend(lots);
        Ok(own)
    }
}

/// The derivatives of the one number at `name` of the device at `path`, `derivative`.
fn moving(path: &str, name: &str, derivative: f64) -> BTreeMap<String, BTreeMap<String, f64>> {
    BTreeMap::from([(
        path.to_owned(),
        BTreeMap::from([(name.to_owned(), derivative)]),
    )])
}

/// The numbers random variables move, each once with its device's index, and the numbers
/// no device holds.
#[derive(Default)]
struct Moved {
    parameters: Vec<(usize, Parameter)>,
    /// The place of each number among them, by the device's path and the number's.
    columns: BTreeMap<(String, String), usize>,
    missing: Vec<(String, String)>,
}

impl Moved {
    /// The place of the number `name` of the device at `path`, added the first time it is
    /// asked for; none, the number noted missing, when no device holds it.
    fn column(&mut self, simulator: &Simulator, path: &str, name: &str) -> Option<usize> {
        let key = (path.to_owned(), name.to_owned());
        if let Some(&at) = self.columns.get(&key) {
            return Some(at);
        }
        let Some(parameter) = simulator.parameter_at(path, name) else {
            if !self.missing.contains(&key) {
                self.missing.push(key);
            }
            return None;
        };
        self.columns.insert(key, self.parameters.len());
        self.parameters.push(parameter);
        Some(self.parameters.len() - 1)
    }
}

/// Checks that `variable`'s mean, standard deviation and derivatives are finite, and its
/// standard deviation not negative.
fn finite(variable: &RandomVariable) -> Result<(), SimulationError> {
    let derivatives = variable.derivatives.values().flat_map(BTreeMap::values);
    let numbers = [variable.mean, variable.standard_deviation];
    if numbers
        .into_iter()
        .chain(derivatives.copied())
        .all(f64::is_finite)
        && variable.standard_deviation >= 0.0
    {
        return Ok(());
    }
    Err(SimulationError::InvalidAnalysis(format!(
        "the random variable `{}` has a mean, a standard deviation of 0 or more and \
         derivatives that are finite",
        variable.name
    )))
}

/// The derivatives `points`, by point, output and parameter, summed into each random
/// variable's by its `weights`: each parameter's place and the derivative of its number
/// with respect to the random variable.
fn combined<T>(points: &[Vec<Vec<T>>], weights: &[Vec<(usize, f64)>]) -> Vec<Vec<Vec<T>>>
where
    T: Copy + Default + std::ops::Add<Output = T> + std::ops::Mul<f64, Output = T>,
{
    points
        .iter()
        .map(|outputs| {
            outputs
                .iter()
                .map(|by_parameter| {
                    weights
                        .iter()
                        .map(|weighed| {
                            weighed.iter().fold(T::default(), |total, &(at, weight)| {
                                total + by_parameter[at] * weight
                            })
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}
