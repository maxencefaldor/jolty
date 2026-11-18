//! Sensitivities: how the unknowns asked for move with every number of every device's
//! parameters, at the operating point, along a DC sweep, over frequency, over time and
//! along a periodic steady state's period.
//!
//! A sensitivity is the analysis's Jacobian, each output at each point against each
//! parameter, which its JVP ([`Simulator::jvp`]) finds a column at a time and its VJP
//! ([`Simulator::vjp`]) a row at a time: one tangent per parameter, or one cotangent per
//! output at each point, and for an oscillator's period one more. It takes whichever
//! costs the fewer solves, counted for the circuit at hand. At a DC point and over
//! frequency each point is solved apart, so the VJP's `k` outputs cost `k` transposed
//! solves a point and the JVP's `P` parameters `P` solves: the VJP unless there are fewer
//! parameters than outputs. Over time, a cotangent at a point pulls back over every step
//! before it, so `k` outputs at `N + 1` points cost `k (N + 1)(N + 2) / 2` transposed
//! solves, where the JVP costs `P (N + 1)`, and along a period the VJP solves twice for
//! each cotangent as the JVP does for each tangent, once from the end besides. A
//! transient takes at least a point per longest step, so the JVP is taken at once when it
//! costs no more at that least number, and for a circuit with logic, whose points the VJP
//! cannot evaluate again as they were; otherwise the transient records its steps, and the
//! cheaper goes through them once it has ended.

use std::collections::BTreeMap;

use jolty_ir::numbers;
use jolty_numerics::Complex64;
use serde::Deserialize;

use crate::derivative::{Cotangent, Forward, Saved, Start, points, units};
use crate::elaborated::{Origin, Origins};
use crate::twins::Twins;
use crate::{
    Ac, AnalysisResult, DcSweep, OperatingPointResult, Options, PeriodicSteadyState,
    SimulationError, Simulator, Transient,
};

/// A sensitivity analysis: how the unknowns `outputs` move with every device parameter
/// along the analysis `of`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sensitivity {
    /// The unknowns, by name, such as `out` or `V1:current`.
    pub outputs: Vec<String>,
    /// The analysis differentiated: the operating point by default.
    #[serde(default)]
    pub of: Differentiated,
}

/// An analysis differentiated: by a sensitivity or a mismatch analysis, or by a JVP or a
/// VJP.
///
/// In JSON, the operating point is the string `"operating_point"`, and the others are
/// objects with one key, such as `{"transient": {"stop": 1e-3}}`.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Differentiated {
    /// The DC operating point.
    #[default]
    OperatingPoint,
    /// A DC sweep of one source.
    DcSweep(DcSweep),
    /// A small-signal frequency response.
    Ac(Ac),
    /// A transient analysis.
    Transient(Transient),
    /// One period of a periodic steady state.
    PeriodicSteadyState(PeriodicSteadyState),
}

/// One number of a device's parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct Parameter {
    /// The device's path, such as `X1/R1`.
    pub instance: String,
    /// The parameter's name, its keys joined by dots within an object and a list's
    /// entries keyed by their index from 0, such as `resistance`, `waveform.dc` or
    /// `waveform.piecewise.1.1`.
    pub name: String,
    /// Its value, in SI base units.
    pub value: f64,
}

/// Derivatives at each point of an analysis, one at an operating point: of a sensitivity
/// analysis, each output's derivative with respect to each parameter, by point, then
/// output, then parameter, and of a JVP, each unknown's along each tangent, by point,
/// then unknown, then tangent.
#[derive(Debug, Clone, PartialEq)]
pub enum Derivatives {
    /// The derivatives of values, at an operating point, along a sweep or over time.
    Real(Vec<Vec<Vec<f64>>>),
    /// The derivatives of phasors, over frequency.
    Complex(Vec<Vec<Vec<Complex64>>>),
}

/// The sensitivities of a [`Sensitivity`] analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct SensitivityResult {
    /// The unknowns, by name, in the order of the derivatives' outputs.
    pub outputs: Vec<String>,
    /// Every parameter the derivatives are with respect to.
    pub parameters: Vec<Parameter>,
    /// The derivatives, by point, then output, then parameter; along a periodic steady
    /// state's period, each point at its fraction of the period.
    pub derivatives: Derivatives,
    /// For an oscillator's periodic steady state, the derivative of its period with
    /// respect to each parameter.
    pub period_derivatives: Option<Vec<f64>>,
    /// The result of the analysis differentiated.
    pub result: Box<AnalysisResult>,
}

/// What [`Simulator::differentiate`] finds: the derivatives of the outputs with respect to
/// the parameter of each column of the twins, in their order, and the result of the
/// analysis differentiated.
pub(crate) struct Differentiation {
    /// The twins, a column for each parameter, as they ended the analysis.
    pub(crate) twins: Twins,
    pub(crate) derivatives: Derivatives,
    /// For an oscillator's periodic steady state, its period's derivative with respect to
    /// the number of each twin.
    pub(crate) period_derivatives: Option<Vec<f64>>,
    pub(crate) result: AnalysisResult,
}

impl Simulator {
    /// The derivative of each unknown of `sensitivity` with respect to every number of every
    /// device's parameters, at each point of the analysis it differentiates.
    ///
    /// Every derivative is exact. A number the device's model reads as a choice, such as a
    /// MOSFET's level, is no real parameter and has none.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::UnknownOutput`] for an output that is no unknown,
    /// [`SimulationError::Singular`] when a Jacobian is, and the errors of the analysis
    /// differentiated.
    pub fn sensitivity(
        &mut self,
        sensitivity: &Sensitivity,
        options: &Options,
    ) -> Result<SensitivityResult, SimulationError> {
        options.parallelism.confined(|| {
            self.solvable()?;
            let twins = self.every_twin();
            let found =
                self.differentiate(&sensitivity.outputs, &sensitivity.of, twins, options)?;
            Ok(SensitivityResult {
                outputs: sensitivity.outputs.clone(),
                parameters: found.twins.parameters,
                derivatives: found.derivatives,
                period_derivatives: found.period_derivatives,
                result: Box::new(found.result),
            })
        })
    }

    /// The derivative of each unknown of `outputs` with respect to the parameter of each
    /// column of `twins`, at each point of the analysis `of`, and that analysis's result.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::UnknownOutput`] for an output that is no unknown,
    /// [`SimulationError::Singular`] when a Jacobian is, and the errors of the analysis
    /// differentiated.
    pub(crate) fn differentiate(
        &mut self,
        outputs: &[String],
        of: &Differentiated,
        twins: Twins,
        options: &Options,
    ) -> Result<Differentiation, SimulationError> {
        let kept = outputs
            .iter()
            .map(|name| {
                self.unknown(name)
                    .map(jolty_models::UnknownIndex::index)
                    .ok_or_else(|| SimulationError::UnknownOutput(name.clone()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let parameters = twins.len();
        let backward = match of {
            Differentiated::Transient(transient) => {
                if self.replayable().is_err() {
                    false
                } else {
                    let least = self.longest_step(transient)?;
                    let points = (transient.stop / least).ceil();
                    // The least number of points, a whole number below 2⁵³.
                    #[expect(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        reason = "a ceiling of a positive ratio, below 2⁵³ for any run that \
                                  fits in memory"
                    )]
                    let points = points as usize + 1;
                    Self::cheaper_backward(of, kept.len(), parameters, points)
                }
            }
            Differentiated::PeriodicSteadyState(_) => true,
            _ => kept.len() <= parameters,
        };
        if !backward {
            let pushed = self.push_forward(of, twins, units(parameters), &kept, options)?;
            return Ok(Differentiation {
                twins: pushed.twins,
                derivatives: pushed.tangents,
                period_derivatives: pushed.period,
                result: pushed.result,
            });
        }
        let (result, saved) = self.run_saving(of, options)?;
        let points = points(&result);
        if Self::cheaper_backward(of, kept.len(), parameters, points) {
            return self.by_rows(twins, (result, saved), (&kept, points), options);
        }
        // Over time or a period, more points than the least make the JVP the cheaper.
        let directions = units(parameters);
        let (forward, period_derivatives) = match (&saved, &result) {
            (Saved::Transient(start, steps), AnalysisResult::Transient(waveforms)) => {
                let mut forward = Forward::new(twins, directions, kept, start.clone());
                self.replay(&mut forward, (waveforms, steps), options)?;
                (forward, None)
            }
            (Saved::Period(steps, system), AnalysisResult::PeriodicSteadyState(found)) => {
                let forward = Forward::new(twins, directions, kept, Start::Held(None));
                let mut forward = self.along_period((found, steps, system), forward, options)?;
                let period = forward.stretch.take().map(|(_, derivatives)| derivatives);
                (forward, period)
            }
            _ => return self.by_rows(twins, (result, saved), (&kept, points), options),
        };
        Ok(Differentiation {
            twins: forward.twins,
            derivatives: Derivatives::Real(forward.found),
            period_derivatives,
            result,
        })
    }

    /// Whether the VJP of `of` costs fewer solves than its JVP, for the derivatives of
    /// `outputs` unknowns at each of `points` points with respect to `parameters`
    /// parameters, as the module's documentation counts them.
    fn cheaper_backward(
        of: &Differentiated,
        outputs: usize,
        parameters: usize,
        points: usize,
    ) -> bool {
        let rows = outputs * points;
        match of {
            Differentiated::Transient(_) => {
                outputs * points * (points + 1) / 2 < parameters * points
            }
            Differentiated::PeriodicSteadyState(analysis) => {
                let cotangents = rows + usize::from(analysis.period.is_none());
                outputs * points * (points + 1) / 2 + cotangents * points < 2 * parameters * points
            }
            _ => outputs <= parameters,
        }
    }

    /// The derivatives of the unknowns `kept` at each of the `points` of the result
    /// `result`, which left `saved`, with respect to the parameter of each column of
    /// `twins`, a row at a time: the VJP of a cotangent on each unknown kept at each point,
    /// and for an oscillator's period one more.
    fn by_rows(
        &mut self,
        mut twins: Twins,
        (result, saved): (AnalysisResult, Saved),
        (kept, points): (&[usize], usize),
        options: &Options,
    ) -> Result<Differentiation, SimulationError> {
        let unknowns = self.elaborated.unknowns.len();
        let oscillator = matches!(&saved, Saved::Period(_, (_, _, Some(_))));
        let parameters = twins.len();
        if let AnalysisResult::Ac(response) = &result {
            let one = Complex64::new(1.0, 0.0);
            let cotangents = unit_rows(kept, points, unknowns, one);
            let pulled = self.pull_frequencies(&mut twins, response, &cotangents, options)?;
            return Ok(Differentiation {
                twins,
                derivatives: Derivatives::Complex(by_point(pulled, kept.len())),
                period_derivatives: None,
                result,
            });
        }
        let mut cotangents = unit_rows(kept, points, unknowns, 1.0);
        if oscillator {
            cotangents.push(Cotangent {
                points: BTreeMap::new(),
                period: 1.0,
            });
        }
        let mut pulled = vec![vec![0.0; parameters]; cotangents.len()];
        let mut fold = |index: usize, terms: &[f64]| {
            for (total, term) in pulled[index].iter_mut().zip(terms) {
                *total += term;
            }
        };
        self.pull(
            &mut twins,
            (&saved, &result),
            &cotangents,
            &mut fold,
            options,
        )?;
        let period_derivatives = if oscillator { pulled.pop() } else { None };
        Ok(Differentiation {
            twins,
            derivatives: Derivatives::Real(by_point(pulled, kept.len())),
            period_derivatives,
            result,
        })
    }

    /// The operating point, and how far each of its unknowns moves, to first order, as
    /// gmin doubles: `gmin ∂x/∂gmin`, which is `−x` for a node only gmin holds.
    ///
    /// Every device's residual is linear in gmin, so doubling it changes the residuals by
    /// `gmin ∂f/∂gmin` exactly, and the solution by that change through the Jacobian.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::Singular`] when the Jacobian is, and the errors of
    /// [`Simulator::operating_point`].
    pub fn gmin_dependence(
        &mut self,
        options: &Options,
    ) -> Result<(OperatingPointResult, Vec<f64>), SimulationError> {
        options.parallelism.confined(|| {
            self.solvable()?;
            let point = self.operating_point(options)?;
            self.evaluate(&point.values, None, (1.0, 2.0 * options.gmin), false);
            let doubled = self.state.workspace.contributions.residual.clone();
            self.evaluate(&point.values, None, (1.0, options.gmin), false);
            let mut moves: Vec<f64> = self
                .state
                .workspace
                .contributions
                .residual
                .iter()
                .zip(&doubled)
                .map(|(residual, doubled)| residual - doubled)
                .collect();
            self.state
                .solver
                .factor(&self.state.workspace.contributions.conductance)?
                .solve(&mut moves)?;
            Ok((point, moves))
        })
    }

    /// The twins of every number of every device's parameters its model reads as a real
    /// number, a column each, in the devices' order then the numbers'.
    pub(crate) fn every_twin(&self) -> Twins {
        let mut parameters = Vec::new();
        let origins = self.origins();
        for (index, placed) in self.elaborated.devices.iter().enumerate() {
            let Origin::Model(made) = origins.of(index) else {
                continue;
            };
            for (name, value) in numbers(&made.parameters) {
                let parameter = Parameter {
                    instance: placed.path().to_owned(),
                    name,
                    value,
                };
                parameters.push((index, parameter));
            }
        }
        self.unit_twins(parameters).0
    }

    /// The number `name` of the device at `path`, and the device's index, its value as the
    /// device is made now; none when the circuit has no device there, or no number there.
    pub(crate) fn parameter_at(&self, path: &str, name: &str) -> Option<(usize, Parameter)> {
        let index = self.elaborated.device(path)?;
        let origins = self.origins();
        let Origin::Model(made) = origins.of(index) else {
            return None;
        };
        let value = jolty_ir::parameter(&made.parameters, name)?.as_f64()?;
        let parameter = Parameter {
            instance: path.to_owned(),
            name: name.to_owned(),
            value,
        };
        Some((index, parameter))
    }

    /// What the run's devices are made from.
    pub(crate) fn origins(&self) -> Origins<'_> {
        Origins {
            elaborated: &self.elaborated,
            patched: &self.state.patched,
            swept: &self.state.swept,
            temperature: self.state.temperature,
        }
    }
}

/// A cotangent of `one` on each unknown of `kept`, among `unknowns`, at each of `points`
/// points, by point then unknown kept.
fn unit_rows<T: Copy + Default>(
    kept: &[usize],
    points: usize,
    unknowns: usize,
    one: T,
) -> Vec<Cotangent<T>> {
    let mut cotangents = Vec::with_capacity(kept.len() * points + 1);
    for point in 0..points {
        for &unknown in kept {
            let mut values = vec![T::default(); unknowns];
            values[unknown] = one;
            cotangents.push(Cotangent {
                points: BTreeMap::from([(point, values)]),
                period: 0.0,
            });
        }
    }
    cotangents
}

/// The rows `pulled`, by point then output, `outputs` a point, grouped by point.
fn by_point<T>(pulled: Vec<Vec<T>>, outputs: usize) -> Vec<Vec<Vec<T>>> {
    let mut rows = pulled.into_iter();
    let mut found = Vec::new();
    loop {
        let point: Vec<Vec<T>> = rows.by_ref().take(outputs).collect();
        if point.is_empty() {
            return found;
        }
        found.push(point);
    }
}
