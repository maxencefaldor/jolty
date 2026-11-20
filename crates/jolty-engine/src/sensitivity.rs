//! Exact parameter sensitivities across operating-point, DC-sweep, AC, stability,
//! transient, and periodic steady-state analyses.
//!
//! A sensitivity table is the analysis Jacobian (outputs across points with respect to every
//! numeric parameter). Forward mode ([`Simulator::jvp`]) builds columns one parameter at a
//! time, while reverse mode ([`Simulator::vjp`]) builds rows one output point at a time
//! (plus one row for an oscillator's period). The driver picks whichever direction requires
//! fewer linear solves for the circuit and output set.
//!
//! For DC, AC, and stability analyses, points decouple, so reverse mode wins whenever
//! `outputs <= parameters`.
//! For transients over `N + 1` points, reverse mode requires `k (N + 1)(N + 2) / 2` transposed
//! solves versus `P (N + 1)` forward solves; periodic steady-state reverse mode adds a second
//! backward pass from the period end. Transients run forward immediately when the minimum
//! step count favors forward mode or when logic devices prevent replay; otherwise steps are
//! recorded so the cheaper mode replays them after integration.

use std::collections::BTreeMap;

use jolty_ir::numbers;
use jolty_numerics::Complex64;
use serde::Deserialize;

use crate::derivative::{
    Cotangent, Forward, PERIODIC_SENSITIVITY, Period, PeriodKind, PullPeriodBudget, Saved, Start,
    Trajectory, points, units, waveforms_numbers,
};
use crate::elaborated::{Origin, Origins};
use crate::options::float_count;
use crate::periodic::DENSE_MATRICES;
use crate::stability::LOOP_GAIN;
use crate::twins::Twins;
use crate::{
    Ac, AnalysisResult, DcSweep, OperatingPointResult, Options, PeriodicSteadyState,
    SimulationError, Simulator, Stability, Transient,
};

/// A sensitivity analysis: how the outputs in `outputs` move with every device parameter
/// along the analysis `of`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sensitivity {
    /// Output names, such as `out`, `V1:current`, or [`LOOP_GAIN`] for
    /// [`Differentiated::Stability`].
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
    /// A Middlebrook loop-gain stability analysis.
    Stability(Stability),
    /// A transient analysis.
    Transient(Transient),
    /// One period of a periodic steady state.
    PeriodicSteadyState(PeriodicSteadyState),
}

/// Numeric parameter belonging to a single device instance.
#[derive(Debug, Clone, PartialEq)]
pub struct Parameter {
    /// Device instance path, such as `X1/R1`.
    pub instance: String,
    /// Dot-separated parameter path with zero-based list indices, such as `resistance`,
    /// `waveform.dc`, or `waveform.piecewise.1.1`.
    pub name: String,
    /// Parameter value in SI base units.
    pub value: f64,
}

/// Analysis derivatives indexed by `[point][output][parameter]` for sensitivities or
/// `[point][unknown][tangent]` for JVPs (single point for operating-point analyses).
#[derive(Debug, Clone, PartialEq)]
pub enum Derivatives {
    /// Real derivatives for operating-point, DC-sweep, transient, or periodic analyses.
    Real(Vec<Vec<Vec<f64>>>),
    /// Complex phasor derivatives for AC and stability analyses.
    Complex(Vec<Vec<Vec<Complex64>>>),
}

/// Output of a [`Sensitivity`] analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct SensitivityResult {
    /// Output unknown names in the order of the derivative rows.
    pub outputs: Vec<String>,
    /// Differentiated parameters in column order.
    pub parameters: Vec<Parameter>,
    /// Sensitivity values indexed by `[point][output][parameter]`, evaluated at fixed period
    /// fractions for periodic steady states.
    pub derivatives: Derivatives,
    /// Period sensitivities with respect to each parameter for an autonomous oscillator.
    pub period_derivatives: Option<Vec<f64>>,
    /// Primal result of the differentiated analysis.
    pub result: Box<AnalysisResult>,
}

/// Internal output of [`Simulator::differentiate`] bundling final parameter twins,
/// derivatives, optional oscillator period derivatives, and primal analysis results.
pub(crate) struct Differentiation {
    /// Parameter twins in column order as left at the end of the analysis.
    pub(crate) twins: Twins,
    pub(crate) derivatives: Derivatives,
    /// Period derivatives with respect to each twin for an autonomous periodic steady state.
    pub(crate) period_derivatives: Option<Vec<f64>>,
    pub(crate) result: AnalysisResult,
}

impl Simulator {
    /// Computes exact derivatives of each unknown in `sensitivity.outputs` with respect to
    /// every numeric device parameter across `sensitivity.of`.
    ///
    /// Discrete choice parameters (such as a MOSFET level) are excluded from differentiation.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::UnknownOutput`] for unrecognized outputs,
    /// [`SimulationError::TooMuchMemory`] if periodic sensitivity exceeds `max_memory`,
    /// [`SimulationError::Singular`] for singular Jacobians, and errors from `sensitivity.of`.
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

    /// Computes derivatives of `outputs` with respect to every column of `twins` across `of`,
    /// automatically selecting forward JVP or reverse VJP based on solve count.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::UnknownOutput`] for unrecognized outputs,
    /// [`SimulationError::TooMuchMemory`] if periodic sensitivity exceeds `max_memory`,
    /// [`SimulationError::Singular`] for singular Jacobians, and errors from `of`.
    pub(crate) fn differentiate(
        &mut self,
        outputs: &[String],
        of: &Differentiated,
        twins: Twins,
        options: &Options,
    ) -> Result<Differentiation, SimulationError> {
        let kept = if matches!(of, Differentiated::Stability(_)) {
            outputs
                .iter()
                .map(|name| {
                    if name == LOOP_GAIN {
                        Ok(0)
                    } else {
                        Err(SimulationError::UnknownOutput(name.clone()))
                    }
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            outputs
                .iter()
                .map(|name| {
                    self.unknown(name)
                        .map(jolty_models::UnknownIndex::index)
                        .ok_or_else(|| SimulationError::UnknownOutput(name.clone()))
                })
                .collect::<Result<Vec<_>, _>>()?
        };
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
        if let Differentiated::PeriodicSteadyState(analysis) = of {
            self.check_periodic(analysis)?;
            options.check()?;
            if parameters > 0 && (!kept.is_empty() || analysis.period.is_none()) {
                let rows = self.elaborated.unknowns.len() + usize::from(analysis.period.is_none());
                // Whichever direction shooting's converged step count later selects, the
                // bordered solve holds at least `DENSE_MATRICES * rows^2` plus either
                // `parameters * rows` columns (forward) or `rows^2` inverse columns
                // (backward).
                options.hold_square(
                    PERIODIC_SENSITIVITY,
                    rows,
                    DENSE_MATRICES,
                    parameters.min(rows),
                )?;
            }
        }
        let (result, saved) = self.run_saving(of, options)?;
        let points = points(&result);
        if Self::cheaper_backward(of, kept.len(), parameters, points) {
            return self.by_rows(twins, result, &saved, &kept, points, options);
        }
        // Over time or a period, more points than the least make the JVP the cheaper.
        let directions = units(parameters);
        let (forward, period_derivatives) = match (&saved, &result) {
            (Saved::Transient(start, steps), AnalysisResult::Transient(waveforms)) => {
                let mut forward = Forward::new(twins, directions, kept, start.clone());
                let trajectory = Trajectory { waveforms, steps };
                self.replay(&mut forward, trajectory, options)?;
                (forward, None)
            }
            (Saved::Period(steps, system), AnalysisResult::PeriodicSteadyState(found)) => {
                let forward = Forward::new(twins, directions, kept, Start::Held(None));
                let period_data = Period {
                    found,
                    steps,
                    system,
                };
                let mut forward = self.along_period(period_data, forward, options)?;
                let period = forward.stretch.take().map(|(_, derivatives)| derivatives);
                (forward, period)
            }
            _ => return self.by_rows(twins, result, &saved, &kept, points, options),
        };
        Ok(Differentiation {
            twins: forward.twins,
            derivatives: Derivatives::Real(forward.found),
            period_derivatives,
            result,
        })
    }

    /// Checks whether reverse VJP requires fewer linear solves than forward JVP for `of` given
    /// `outputs`, `parameters`, and `points`.
    fn cheaper_backward(
        of: &Differentiated,
        outputs: usize,
        parameters: usize,
        points: usize,
    ) -> bool {
        let outputs_u128 = outputs as u128;
        let parameters_u128 = parameters as u128;
        let points_u128 = points as u128;
        match of {
            Differentiated::Transient(_) => {
                points > 0 && outputs_u128.saturating_mul(points_u128 + 1) < 2 * parameters_u128
            }
            Differentiated::PeriodicSteadyState(analysis) => {
                let free = u128::from(analysis.period.is_none());
                points > 0
                    && outputs_u128
                        .saturating_mul(3 * points_u128 + 1)
                        .saturating_add(2 * free)
                        < 4 * parameters_u128
            }
            _ => outputs <= parameters,
        }
    }

    /// Computes sensitivities of `kept` unknowns across `points` via reverse-mode VJP,
    /// seeding one unit cotangent per `(point, output)` pair plus one for an oscillator period.
    fn by_rows(
        &mut self,
        mut twins: Twins,
        result: AnalysisResult,
        saved: &Saved,
        kept: &[usize],
        points: usize,
        options: &Options,
    ) -> Result<Differentiation, SimulationError> {
        let unknowns = self.elaborated.unknowns.len();
        let oscillator = matches!(saved, Saved::Period(_, system) if system.scale.is_some());
        let parameters = twins.len();
        if let AnalysisResult::Ac(response) = &result {
            let one = Complex64::new(1.0, 0.0);
            let cotangents = unit_rows(kept, points, unknowns, one);
            let pulled = self.pull_frequencies(&mut twins, response, &cotangents, options)?;
            return Ok(Differentiation {
                twins,
                derivatives: Derivatives::Complex(by_point(pulled, points, kept.len())),
                period_derivatives: None,
                result,
            });
        }
        if let AnalysisResult::Stability(loop_gain) = &result {
            let one = Complex64::new(1.0, 0.0);
            let cotangents = unit_rows(kept, points, 1, one);
            let pulled = self.pull_stability(&mut twins, loop_gain, &cotangents, options)?;
            return Ok(Differentiation {
                twins,
                derivatives: Derivatives::Complex(by_point(pulled, points, kept.len())),
                period_derivatives: None,
                result,
            });
        }
        if let Saved::Period(_, system) = saved
            && let AnalysisResult::PeriodicSteadyState(found) = &result
            && ((!kept.is_empty() && points > 0) || oscillator)
        {
            let rows = system.tolerances.len() + usize::from(system.scale.is_some());
            let outputs_points = float_count(kept.len()) * float_count(points);
            let cotangent_count = outputs_points + if oscillator { 1.0 } else { 0.0 };
            let unit_rows_numbers = outputs_points * float_count(unknowns);
            let pulled_numbers = cotangent_count * float_count(parameters);
            PullPeriodBudget {
                unknowns,
                rows,
                waveforms: waveforms_numbers(&found.transient),
                cotangents: cotangent_count,
                parameters,
                kind: PeriodKind::from_scale(system.scale),
                extra: unit_rows_numbers + pulled_numbers,
            }
            .check(options)?;
        }
        let mut cotangents = unit_rows(kept, points, unknowns, 1.0);
        if oscillator {
            cotangents.push(Cotangent {
                points: BTreeMap::new(),
                period: 1.0,
            });
        }
        let mut pulled = self.pull_collected(&mut twins, saved, &result, &cotangents, options)?;
        let period_derivatives = if oscillator { pulled.pop() } else { None };
        Ok(Differentiation {
            twins,
            derivatives: Derivatives::Real(by_point(pulled, points, kept.len())),
            period_derivatives,
            result,
        })
    }

    /// Computes the DC operating point and the exact first-order shift `gmin * ∂x/∂gmin` when
    /// `gmin` doubles (yielding `-x` on nodes supported solely by `gmin`).
    ///
    /// Device residuals are affine in `gmin`, so evaluating at `2 * gmin` recovers
    /// `gmin * ∂f/∂gmin` exactly for a single linear solve.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::Singular`] if the DC Jacobian is singular, plus errors from
    /// [`Simulator::operating_point`].
    pub fn gmin_dependence(
        &mut self,
        options: &Options,
    ) -> Result<(OperatingPointResult, Vec<f64>), SimulationError> {
        options.parallelism.confined(|| {
            self.solvable()?;
            let point = self.operating_point(options)?;
            self.evaluate(&point.values, None, 1.0, 2.0 * options.gmin, false);
            let doubled = self.state.workspace.contributions.residual.clone();
            self.evaluate(&point.values, None, 1.0, options.gmin, false);
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

    /// Builds unit-seeded [`Twins`] for every differentiable real device parameter in device
    /// and parameter order.
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

    /// Looks up numeric parameter `name` on device `path`, returning `(device_index, Parameter)`
    /// with the active effective value, or `None` if absent.
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

    /// Returns the active [`Origins`] view combining elaboration, value patches, sweeps, and
    /// temperature.
    pub(crate) fn origins(&self) -> Origins<'_> {
        Origins {
            elaborated: &self.elaborated,
            patched: &self.state.patched,
            swept: &self.state.swept,
            temperature: self.state.temperature,
        }
    }
}

/// Builds unit cotangents with value `one` at each `(point, kept_unknown)` entry in
/// point-major order.
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

/// Reshapes flat point-major cotangent rows `pulled` into `points` groups of `outputs` rows.
fn by_point<T>(pulled: Vec<Vec<T>>, points: usize, outputs: usize) -> Vec<Vec<Vec<T>>> {
    let mut rows = pulled.into_iter();
    (0..points)
        .map(|_| rows.by_ref().take(outputs).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use jolty_numerics::integration::Method;

    use super::{Derivatives, Differentiated, Sensitivity, by_point};
    use crate::test_helpers::source_resistor_simulator;
    use crate::{Ac, Options, PeriodicSteadyState, Simulator, Transient};

    #[test]
    fn by_point_preserves_point_count_when_outputs_is_empty() {
        let empty: Vec<Vec<f64>> = Vec::new();
        let grouped = by_point(empty, 3, 0);
        assert_eq!(grouped.len(), 3);
        assert!(grouped.iter().all(Vec::is_empty));

        let non_empty = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        assert_eq!(
            by_point(non_empty, 2, 2),
            vec![vec![vec![1.0], vec![2.0]], vec![vec![3.0], vec![4.0]]]
        );

        let zero_points: Vec<Vec<f64>> = Vec::new();
        assert_eq!(by_point(zero_points, 0, 2), Vec::<Vec<Vec<f64>>>::new());
    }

    #[test]
    fn empty_outputs_ac_sensitivity_retains_one_empty_entry_per_frequency() {
        let mut simulator = source_resistor_simulator();
        let result = simulator
            .sensitivity(
                &Sensitivity {
                    outputs: Vec::new(),
                    of: Differentiated::Ac(Ac {
                        source: "V1".to_owned(),
                        frequencies: vec![10.0, 100.0, 1000.0],
                    }),
                },
                &Options::default(),
            )
            .expect("empty-output AC sensitivity succeeds");
        let Derivatives::Complex(points) = result.derivatives else {
            panic!("expected complex derivatives, got {:?}", result.derivatives);
        };
        assert_eq!(points.len(), 3);
        assert!(points.iter().all(Vec::is_empty));
    }

    #[test]
    fn cheaper_backward_does_not_overflow_usize_on_large_counts() {
        let transient = Differentiated::Transient(Transient {
            stop: 1.0,
            max_step: None,
            method: Method::Trapezoidal,
            breakpoints: Vec::new(),
            initial_conditions: BTreeMap::new(),
            operating_point: true,
        });
        let driven = Differentiated::PeriodicSteadyState(PeriodicSteadyState {
            period: Some(1e-3),
            ..PeriodicSteadyState::default()
        });
        let autonomous = Differentiated::PeriodicSteadyState(PeriodicSteadyState {
            period: None,
            ..PeriodicSteadyState::default()
        });

        // Values large enough that `points * (points + 1)` would overflow `usize`.
        let cheaper = Simulator::cheaper_backward;
        let big = 1_usize << (usize::BITS / 2);
        let max = usize::MAX;
        assert!(!cheaper(&transient, big, 100, big));
        assert!(!cheaper(&transient, max, max, max));
        assert!(!cheaper(&driven, max, max, max));
        assert!(!cheaper(&autonomous, max, max, max));
        // Zero outputs with positive points and parameters chooses backward without overflowing.
        assert!(cheaper(&transient, 0, 1, max));
        assert!(cheaper(&driven, 0, 1, max));
        assert!(cheaper(&autonomous, 0, 1, max));
        // Zero points never chooses backward.
        assert!(!cheaper(&transient, 0, 10, 0));
        assert!(!cheaper(&driven, 0, 10, 0));
        assert!(!cheaper(&autonomous, 0, 10, 0));
        // Exact boundary checks matching small-integer formula.
        assert!(cheaper(&transient, 1, 2, 2));
        assert!(!cheaper(&transient, 1, 2, 3));
        assert!(cheaper(&driven, 1, 2, 1));
        assert!(!cheaper(&driven, 1, 1, 1));
        assert!(cheaper(&autonomous, 1, 2, 1));
        assert!(!cheaper(&autonomous, 1, 1, 1));
    }
}
