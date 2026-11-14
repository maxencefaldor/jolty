//! Sensitivities: how the unknowns asked for move with every number of every device's
//! parameters, at the operating point, along a DC sweep, over frequency and over time.
//!
//! At a DC point `F(x, p) = 0`, an output `y = x_k` moves with a parameter `p` by
//! `dy/dp = −λᵀ ∂F/∂p`, where `Jᵀ λ = e_k` and `J = ∂F/∂x`: one transposed solve per output
//! serves every parameter, however many (the adjoint method). A DC sweep's levels are each
//! such a point.
//!
//! Over frequency, the response to an excitation `b` solves `(G + jωC) x = b`, so it moves
//! by `dx/dp = −(G + jωC)⁻¹ (dG/dp + jω dC/dp) x`, read through the adjoint at each
//! frequency. `G` and `C` move with `p` directly, and through the operating point, which
//! moves by `s = −G⁻¹ ∂F/∂p`: the first part is exact, and the second a central difference of
//! the devices' `G` and `C` a microvolt either way along `s`, since the devices give their
//! derivatives with respect to their voltages, not their second derivatives.
//!
//! Over time, each step solves `f(x) + α q(x) + h = 0`, where the integration formula's `α`
//! and history `h` stand for the charges' derivative. Its derivative with respect to `p`
//! carries `s = dx/dp` from step to step: `J s = −(∂f/∂p + α ∂q/∂p + h_p)`, where `h_p` is the
//! same formula's history of the charges' derivatives `dq/dp = C s + ∂q/∂p`, starting from the
//! operating point's (the direct method; D. E. Hocevar, P. Yang, T. N. Trick and B. D.
//! Epler, "Transient sensitivity computation for MOSFET circuits", *IEEE Transactions on
//! Computer-Aided Design* 4(4), 1985). Each step's Jacobian is evaluated again at its
//! solution. A derivative moves no logic event, and no step.
//!
//! Each device's `∂F/∂p` is exact: its instance prepared again computing with the
//! derivative of every number with respect to the parameter differentiated against, and
//! evaluated where the instance itself is, into a block of its own whose map places it
//! among the circuit's equations, so that a parameter costs what its instance does, not
//! what the circuit does.

use jolty_ir::numbers;
use jolty_models::{Gradient, Input, Unknown};
use jolty_numerics::integration::Formula;
use jolty_numerics::{Complex64, Workspace};
use serde::Deserialize;

use crate::ac::{Linearization, over_frequencies};
use crate::elaborated::{Origin, Origins};
use crate::state::Held;
use crate::sweep::Swept;
use crate::transient::{Carried, Tangent};
use crate::{Ac, DcSweep, OperatingPoint, Options, Outcome, SimulationError, Simulator, Transient};

/// How far the operating point moves, along its derivative, in the central difference
/// of `G` and `C`: its largest voltage moves this many volts either way.
const BIAS_STEP: f64 = 1e-6;

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

/// An analysis a sensitivity analysis differentiates.
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
}

/// One number of a device's parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct Parameter {
    /// The device's path, such as `X1/R1`.
    pub device: String,
    /// The parameter's name, its keys joined by dots within an object and a list's
    /// entries keyed by their index from 0, such as `resistance`, `waveform.dc` or
    /// `waveform.piecewise.1.1`.
    pub name: String,
    /// Its value, in SI base units.
    pub value: f64,
}

/// The derivatives of a sensitivity analysis: for each point of the analysis it
/// differentiates, one at an operating point, each output's derivative with respect to each
/// parameter.
#[derive(Debug, Clone, PartialEq)]
pub enum Derivatives {
    /// The derivatives of values, at an operating point, along a sweep or over time.
    Real(Vec<Vec<Vec<f64>>>),
    /// The derivatives of phasors, over frequency.
    Complex(Vec<Vec<Vec<Complex64>>>),
}

/// The sensitivities of a [`Sensitivity`] analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct Sensitivities {
    /// The unknowns, by name, in the order of the derivatives' outputs.
    pub outputs: Vec<String>,
    /// Every parameter the derivatives are with respect to.
    pub parameters: Vec<Parameter>,
    /// The derivatives, by point, then output, then parameter.
    pub derivatives: Derivatives,
    /// The result of the analysis differentiated.
    pub result: Box<Outcome>,
}

/// A device's instance computing with the derivative of every number with respect to one
/// number of its parameters.
struct Twin {
    /// The index of the device it is made from.
    device: usize,
    parameter: Parameter,
    made: jolty_models::Twin,
    /// Whether a sweep holds its number, which then moves nothing: the number swept, or
    /// any of a source the sweep holds at a level, which replaces its waveform.
    fixed: bool,
}

impl Twin {
    /// Evaluates the twin where the device is: at the solution `x`, at `time` in a
    /// transient analysis or, with `None`, at an operating point or about one.
    fn evaluate(&mut self, x: &[f64], time: Option<f64>, gmin: f64) {
        let mut input = Input::new(x, time, gmin);
        input.limiting = false;
        self.made.evaluate(&input);
    }

    /// `∂f/∂p`, by unknown, as last evaluated, among `dimension` unknowns.
    fn residual(&self, dimension: usize) -> Vec<f64> {
        dense(self.made.residual(), dimension)
    }

    /// `∂q/∂p`, by unknown, as last evaluated, among `dimension` unknowns.
    fn charge(&self, dimension: usize) -> Vec<f64> {
        dense(self.made.charge(), dimension)
    }

    /// `aᵀ ∂f/∂p` for the vector `a` of the unknowns, as last evaluated: the instance's own
    /// terms alone.
    fn dot(&self, a: &[f64]) -> f64 {
        self.made
            .residual()
            .map(|(unknown, residual)| a[unknown] * residual.derivative)
            .sum()
    }

    /// `aᵀ (∂G/∂p + jω ∂C/∂p) x` at `(ω, x)`, for the vector `a` of the equations, as last
    /// evaluated: the instance's own entries alone, each at its row and column of
    /// `positions`.
    fn product(
        &self,
        positions: &[(usize, usize)],
        a: &[Complex64],
        (omega, x): (f64, &[Complex64]),
    ) -> Complex64 {
        self.made
            .entries()
            .map(|(slot, conductance, capacitance)| {
                let (row, column) = positions[slot];
                let derivative =
                    Complex64::new(conductance.derivative, omega * capacitance.derivative);
                a[row] * derivative * x[column]
            })
            .sum()
    }
}

/// The derivatives `sparse` carries, by index, among `dimension`, the others zero; one
/// index's derivatives add.
fn dense(sparse: impl Iterator<Item = (usize, Gradient)>, dimension: usize) -> Vec<f64> {
    let mut values = vec![0.0; dimension];
    for (index, value) in sparse {
        values[index] += value.derivative;
    }
    values
}

/// The direct method's state along a transient: each parameter's derivatives of the
/// charges, and the outputs' derivatives at every accepted point.
pub(crate) struct Forward {
    outputs: Vec<Unknown>,
    twins: Vec<Twin>,
    tangents: Vec<Tangent>,
    found: Vec<Vec<Vec<f64>>>,
}

impl Simulator {
    /// The derivative of each unknown of `sensitivity` with respect to every number of every
    /// device's parameters, at each point of the analysis it differentiates.
    ///
    /// The derivatives are exact, but for a response's through its operating point, a
    /// central difference along the operating point's own derivative. A number the device's
    /// model reads as a choice, such as a MOSFET's level, is no real parameter and has none.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::UnknownOutput`] for an output that is no unknown,
    /// [`SimulationError::InvalidAnalysis`] for a periodic transient,
    /// [`SimulationError::Singular`] when a Jacobian is, and the errors of the analysis
    /// differentiated.
    pub fn sensitivity(
        &mut self,
        sensitivity: &Sensitivity,
        options: &Options,
    ) -> Result<Sensitivities, SimulationError> {
        self.solvable()?;
        let outputs = sensitivity
            .outputs
            .iter()
            .map(|name| {
                self.unknown(name)
                    .ok_or_else(|| SimulationError::UnknownOutput(name.clone()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut twins = self.twins();
        let (derivatives, result) = match &sensitivity.of {
            Differentiated::OperatingPoint => {
                let point = self.operating_point(options)?;
                let found = self.adjoint(&outputs, &mut twins, &point.values, options)?;
                (
                    Derivatives::Real(vec![found]),
                    Outcome::OperatingPoint(point),
                )
            }
            Differentiated::DcSweep(sweep) => {
                let result = self.dc_sweep(sweep, options)?;
                let swept = self.swept(sweep)?;
                let found =
                    self.along_sweep(&outputs, &mut twins, (swept, sweep), &result, options);
                self.hold(swept, None)?;
                (Derivatives::Real(found?), Outcome::DcSweep(result))
            }
            Differentiated::Ac(ac) => {
                let response = self.ac(ac, options)?;
                let found = self.over_frequency(&outputs, &mut twins, &response, options)?;
                (Derivatives::Complex(found), Outcome::Ac(response))
            }
            Differentiated::Transient(transient) => {
                if transient.periodic {
                    return Err(SimulationError::InvalidAnalysis(
                        "a sensitivity differentiates a transient from its operating point, \
                         not a periodic steady state"
                            .to_owned(),
                    ));
                }
                let mut forward = Forward {
                    outputs,
                    twins,
                    tangents: Vec::new(),
                    found: Vec::new(),
                };
                let carried = Carried::Forward(&mut forward);
                let waveforms = self.transient_with(transient, options, None, carried)?;
                twins = forward.twins;
                (
                    Derivatives::Real(forward.found),
                    Outcome::Transient(waveforms),
                )
            }
        };
        Ok(Sensitivities {
            outputs: sensitivity.outputs.clone(),
            parameters: twins.into_iter().map(|twin| twin.parameter).collect(),
            derivatives,
            result: Box::new(result),
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
    ) -> Result<(OperatingPoint, Vec<f64>), SimulationError> {
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
    }

    /// A twin of every device, for each number of its parameters its model reads as a
    /// real number, in the devices' order then the numbers'.
    fn twins(&self) -> Vec<Twin> {
        let mut twins = Vec::new();
        let origins = self.origins();
        for (index, placed) in self.elaborated.devices.iter().enumerate() {
            let Origin::Model(made) = origins.of(index) else {
                continue;
            };
            for (name, value) in numbers(&made.parameters) {
                let parameter = Parameter {
                    device: placed.path().to_owned(),
                    name,
                    value,
                };
                if let Some(twin) = self.twin(index, parameter) {
                    twins.push(twin);
                }
            }
        }
        twins
    }

    /// What the run's devices are made from.
    fn origins(&self) -> Origins<'_> {
        Origins {
            elaborated: &self.elaborated,
            patched: &self.state.patched,
            swept: &self.state.swept,
            temperature: self.state.temperature,
        }
    }

    /// The twin of the device at `index` that differentiates against its number
    /// `parameter`, made from what the device is made from now; none when its model reads
    /// no real number there.
    fn twin(&self, index: usize, parameter: Parameter) -> Option<Twin> {
        let origins = self.origins();
        let origin = origins.of(index);
        if !matches!(origin, Origin::Model(_)) {
            return None;
        }
        let instance = (&origin.kernel(), origins.temperature);
        let made =
            self.state
                .runs
                .twin(&self.elaborated.groups, index, instance, &parameter.name)?;
        Some(Twin {
            device: index,
            parameter,
            made,
            fixed: false,
        })
    }

    /// Each output's derivative with respect to each twin's parameter at the DC solution
    /// `x`, by the adjoint method.
    fn adjoint(
        &mut self,
        outputs: &[Unknown],
        twins: &mut [Twin],
        x: &[f64],
        options: &Options,
    ) -> Result<Vec<Vec<f64>>, SimulationError> {
        self.evaluate(x, None, (1.0, options.gmin), false);
        let factors = self
            .state
            .solver
            .factor(&self.state.workspace.contributions.conductance)?;
        let shared = factors.shared();
        let parallelism = &options.parallelism;
        let dimension = x.len();
        let adjoints = parallelism.spread(
            &mut outputs.to_vec(),
            || Workspace::new(dimension),
            |workspace, output| {
                let mut adjoint = vec![0.0; dimension];
                adjoint[output.index()] = 1.0;
                shared.solve_transpose(&mut adjoint, workspace)?;
                Ok(adjoint)
            },
        );
        let adjoints = adjoints
            .into_iter()
            .collect::<Result<Vec<Vec<f64>>, SimulationError>>()?;
        let by_twin = parallelism.spread(
            twins,
            || (),
            |(), twin| {
                if twin.fixed {
                    return vec![0.0; adjoints.len()];
                }
                twin.evaluate(x, None, options.gmin);
                adjoints
                    .iter()
                    .map(|adjoint| -twin.dot(adjoint))
                    .collect::<Vec<f64>>()
            },
        );
        let mut found = vec![Vec::with_capacity(by_twin.len()); outputs.len()];
        for derivatives in by_twin {
            for (of_output, derivative) in found.iter_mut().zip(derivatives) {
                of_output.push(derivative);
            }
        }
        Ok(found)
    }

    /// The adjoint derivatives at each level of the sweep `result`, of `swept` as `sweep`
    /// names it, held there, and the twins of its device prepared again there, from what
    /// its instance is made of: a number the sweep holds moves nothing.
    fn along_sweep(
        &mut self,
        outputs: &[Unknown],
        twins: &mut [Twin],
        (swept, sweep): (Swept, &DcSweep),
        result: &crate::Sweep,
        options: &Options,
    ) -> Result<Vec<Vec<Vec<f64>>>, SimulationError> {
        let mut found = Vec::with_capacity(result.levels.len());
        for (&level, x) in result.levels.iter().zip(&result.values) {
            self.hold(swept, Some(level))?;
            for twin in twins.iter_mut().filter(|twin| twin.device == swept.device) {
                twin.fixed = match swept.held {
                    Held::Level(_) => true,
                    Held::Number(_) => sweep.parameter.as_ref() == Some(&twin.parameter.name),
                };
                if !twin.fixed
                    && let Some(made) = self.twin(swept.device, twin.parameter.clone())
                {
                    *twin = made;
                }
            }
            found.push(self.adjoint(outputs, twins, x, options)?);
        }
        Ok(found)
    }

    /// The derivatives of the outputs' phasors in `response` at each of its frequencies.
    fn over_frequency(
        &mut self,
        outputs: &[Unknown],
        twins: &mut [Twin],
        response: &crate::Response,
        options: &Options,
    ) -> Result<Vec<Vec<Vec<Complex64>>>, SimulationError> {
        let x = &response.operating_point.values;
        // How the operating point moves with each parameter, and G and C with it.
        self.evaluate(x, None, (1.0, options.gmin), false);
        let factors = self
            .state
            .solver
            .factor(&self.state.workspace.contributions.conductance)?;
        let shared = factors.shared();
        let dimension = x.len();
        // Each twin stays evaluated at the operating point, which the response reads too.
        let moved = options.parallelism.spread(
            twins,
            || Workspace::new(dimension),
            |workspace, twin| {
                twin.evaluate(x, None, options.gmin);
                let mut shift: Vec<f64> = twin
                    .residual(dimension)
                    .iter()
                    .map(|value| -value)
                    .collect();
                shared.solve(&mut shift, workspace)?;
                Ok(shift)
            },
        );
        let moved = moved
            .into_iter()
            .collect::<Result<Vec<Vec<f64>>, SimulationError>>()?;
        let mut bias = Vec::with_capacity(twins.len());
        for shift in &moved {
            bias.push(self.bias_derivative(x, shift, options));
        }
        self.evaluate(x, None, (1.0, options.gmin), false);
        let positions: Vec<(usize, usize)> = self.elaborated.pattern.positions().collect();
        let twins = &*twins;
        over_frequencies(
            Linearization::new(&self.state.solver),
            &self.state.workspace.contributions,
            &response.frequencies,
            &options.parallelism,
            &|factors, index, frequency| {
                let phasors = &response.values[index];
                let omega = std::f64::consts::TAU * frequency;
                let mut adjoints = Vec::with_capacity(outputs.len());
                for output in outputs {
                    let mut adjoint = vec![Complex64::default(); dimension];
                    adjoint[output.index()] = Complex64::new(1.0, 0.0);
                    factors.solve_transpose(&mut adjoint)?;
                    adjoints.push(adjoint);
                }
                let mut at = vec![Vec::with_capacity(twins.len()); outputs.len()];
                for (twin, moving) in twins.iter().zip(&bias) {
                    for (adjoint, derivatives) in adjoints.iter().zip(&mut at) {
                        let mut total = twin.product(&positions, adjoint, (omega, phasors));
                        if let Some((conductance, capacitance)) = moving {
                            let entries = positions.iter().zip(conductance.iter().zip(capacitance));
                            total += entries
                                .map(|(&(row, column), (g, c))| {
                                    adjoint[row] * Complex64::new(*g, omega * c) * phasors[column]
                                })
                                .sum::<Complex64>();
                        }
                        derivatives.push(-total);
                    }
                }
                Ok(at)
            },
        )
    }

    /// The derivatives of `G` and `C`, by stored position, as the operating point `x` moves
    /// along `shift`: a central difference whose largest voltage moves [`BIAS_STEP`] either
    /// way. `None` when no voltage moves.
    fn bias_derivative(
        &mut self,
        x: &[f64],
        shift: &[f64],
        options: &Options,
    ) -> Option<(Vec<f64>, Vec<f64>)> {
        let largest = self
            .elaborated
            .unknowns
            .iter()
            .zip(shift)
            .filter(|(unknown, _)| !unknown.quantity.is_current())
            .fold(0.0_f64, |most, (_, value)| most.max(value.abs()));
        if largest == 0.0 || !largest.is_finite() {
            return None;
        }
        let step = BIAS_STEP / largest;
        let mut ends = [(); 2].map(|()| (Vec::new(), Vec::new()));
        for (end, sign) in ends.iter_mut().zip([1.0, -1.0]) {
            let moved: Vec<f64> = x
                .iter()
                .zip(shift)
                .map(|(value, along)| value + sign * step * along)
                .collect();
            self.evaluate(&moved, None, (1.0, options.gmin), false);
            let contributions = &self.state.workspace.contributions;
            *end = (
                contributions.conductance.clone(),
                contributions.capacitance.clone(),
            );
        }
        let [(g_up, c_up), (g_down, c_down)] = ends;
        let difference = |up: &[f64], down: &[f64]| -> Vec<f64> {
            up.iter()
                .zip(down)
                .map(|(above, below)| (above - below) / (2.0 * step))
                .collect()
        };
        Some((difference(&g_up, &g_down), difference(&c_up, &c_down)))
    }

    /// Starts the direct method at a transient's operating point `x`: each parameter's
    /// derivative of the solution, by the DC Jacobian, and of the charges the transient
    /// starts from.
    pub(crate) fn start_forward(
        &mut self,
        forward: &mut Forward,
        x: &[f64],
        options: &Options,
    ) -> Result<(), SimulationError> {
        self.evaluate(x, None, (1.0, options.gmin), false);
        let mut factors = self
            .state
            .solver
            .factor(&self.state.workspace.contributions.conductance)?;
        let mut at = vec![Vec::with_capacity(forward.twins.len()); forward.outputs.len()];
        let mut shifts = Vec::with_capacity(forward.twins.len());
        for twin in &mut forward.twins {
            twin.evaluate(x, None, options.gmin);
            let mut moved: Vec<f64> = twin.residual(x.len()).iter().map(|value| -value).collect();
            factors.solve(&mut moved)?;
            for (output, derivatives) in forward.outputs.iter().zip(&mut at) {
                derivatives.push(moved[output.index()]);
            }
            shifts.push(moved);
        }
        // A state that only time moves holds at the operating point, and has a charge
        // only once time passes.
        self.evaluate(x, Some(0.0), (1.0, options.gmin), false);
        forward.tangents.clear();
        for (twin, moved) in forward.twins.iter_mut().zip(&shifts) {
            twin.evaluate(x, Some(0.0), options.gmin);
            let mut charge = twin.charge(x.len());
            let capacitances = &self.state.workspace.contributions.capacitance;
            self.elaborated
                .pattern
                .add_product(capacitances, moved, &mut charge);
            forward.tangents.push(Tangent::new(charge));
        }
        forward.found = vec![at];
        Ok(())
    }

    /// Carries the direct method over the step a transient accepted at `time`, its solution
    /// `x`, by `formula`; `restarted` says whether the integration restarts after it.
    pub(crate) fn advance_forward(
        &mut self,
        forward: &mut Forward,
        (time, x): (f64, &[f64]),
        (formula, restarted): (&Formula, bool),
        options: &Options,
    ) -> Result<(), SimulationError> {
        self.evaluate(x, Some(time), (1.0, options.gmin), false);
        let jacobian: Vec<f64> = self
            .state
            .workspace
            .contributions
            .conductance
            .iter()
            .zip(&self.state.workspace.contributions.capacitance)
            .map(|(conductance, capacitance)| conductance + formula.alpha * capacitance)
            .collect();
        let mut factors = self.state.solver.factor(&jacobian)?;
        let mut at = vec![Vec::with_capacity(forward.twins.len()); forward.outputs.len()];
        for (twin, tangent) in forward.twins.iter_mut().zip(&mut forward.tangents) {
            twin.evaluate(x, Some(time), options.gmin);
            let partial_charge = twin.charge(x.len());
            let history = tangent.history(formula);
            let mut moved: Vec<f64> = twin
                .residual(x.len())
                .iter()
                .zip(&partial_charge)
                .zip(&history)
                .map(|((residual, charge), term)| -(residual + formula.alpha * charge + term))
                .collect();
            factors.solve(&mut moved)?;
            for (output, derivatives) in forward.outputs.iter().zip(&mut at) {
                derivatives.push(moved[output.index()]);
            }
            let mut charge = partial_charge;
            let capacitances = &self.state.workspace.contributions.capacitance;
            self.elaborated
                .pattern
                .add_product(capacitances, &moved, &mut charge);
            tangent.advance(charge, &history, formula, restarted);
        }
        forward.found.push(at);
        Ok(())
    }
}
