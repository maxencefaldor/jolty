//! First-order statistical mismatch analysis estimating output spread and per-variable
//! variance contributions across random parameter variations.
//!
//! Each random variable `z` with mean `μ` and standard deviation `σ` perturbs device
//! parameters `p` by `∂p/∂z`. By the chain rule, `dy/dz = Σ (dy/dp) (∂p/∂z)`, where
//! `dy/dp` uses exact adjoint sensitivity so one solve per output covers all variables.
//! Independent random variables yield a first-order standard deviation of
//! `√(Σ (dy/dz · σ)²)` with fractional variance shares `(dy/dz · σ)² / Σ (dy/dz · σ)²`,
//! matching Spectre `dcmatch` and `acmatch`.
//!
//! Circuit tolerances expand via [`Toleranced::random_variables`] at current parameter
//! values, while process-kit statistical functions enter via explicit random variables.

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
    /// Path of the instance drawing this variable (the toleranced part or the placement
    /// calling a process-kit statistical function), or `None` for a lot variable.
    #[serde(default)]
    pub instance: Option<String>,
    /// The name of the lot whose shared spread it is, if it is one.
    #[serde(default)]
    pub lot: Option<String>,
    /// Identifier within the instance or lot, such as `resistance` or `vth0#0`.
    pub name: String,
    /// Mean value of the random variable at the nominal operating parameters.
    pub mean: f64,
    /// Standard deviation of the random variable.
    pub standard_deviation: f64,
    /// Parameter partial derivatives `∂p/∂z` indexed by device path then parameter path, in
    /// SI base units per unit of `z`.
    pub derivatives: BTreeMap<String, BTreeMap<String, f64>>,
}

/// First-order derivatives of [`Mismatch`] outputs with respect to random variables.
#[derive(Debug, Clone, PartialEq)]
pub struct MismatchResult {
    /// Output unknown names matching the derivative output axis.
    pub outputs: Vec<String>,
    /// Random variables in column order: per-instance tolerance variables, shared lot
    /// tolerance variables, and explicit random variables.
    pub random_variables: Vec<RandomVariable>,
    /// Output derivatives with respect to random variables, indexed by point, output, and
    /// random variable.
    pub derivatives: Derivatives,
    /// Derivative of oscillator period with respect to each random variable for autonomous
    /// periodic steady state.
    pub period_derivatives: Option<Vec<f64>>,
    /// Underlying analysis result at nominal parameter values.
    pub result: Box<AnalysisResult>,
}

impl Simulator {
    /// Computes first-order derivatives of `mismatch` outputs with respect to all random
    /// variables across the target analysis.
    ///
    /// Combines exact sensitivity derivatives across all parameters affected by each random
    /// variable.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::UnknownParameters`] for unrecognized numeric parameter
    /// paths, [`SimulationError::InvalidAnalysis`] for non-finite statistics or negative
    /// standard deviations, and any error from [`Simulator::sensitivity`].
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

    /// Expands `toleranced` entries into per-instance and shared-lot [`RandomVariable`]
    /// entries evaluated at current run values, registering affected parameters in `moved`.
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

/// Builds a single-entry derivative map setting `∂(path.name)/∂z = derivative`.
fn moving(path: &str, name: &str, derivative: f64) -> BTreeMap<String, BTreeMap<String, f64>> {
    BTreeMap::from([(
        path.to_owned(),
        BTreeMap::from([(name.to_owned(), derivative)]),
    )])
}

/// Deduplicated table of device parameters perturbed by random variables, plus missing keys.
#[derive(Default)]
struct Moved {
    parameters: Vec<(usize, Parameter)>,
    /// Column index of each perturbed parameter, keyed by `(device_path, parameter_path)`.
    columns: BTreeMap<(String, String), usize>,
    missing: Vec<(String, String)>,
}

impl Moved {
    /// Returns the column index for `(path, name)`, registering it on first lookup or
    /// recording it as missing when no device defines that numeric parameter.
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

/// Validates that `variable` has finite mean, derivatives, and non-negative standard
/// deviation.
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

/// Combines parameter derivatives `points` (indexed by point, output, and parameter) into
/// random-variable derivatives using per-variable `(parameter_index, weight)` lists.
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
