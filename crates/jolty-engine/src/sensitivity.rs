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
//! Each device's `∂F/∂p` is exact: the device made again computing with [`Gradient`], the
//! parameter differentiated against, and evaluated where the device itself is.

use jolty_ir::Parameters;
use jolty_models::{Binding, Contributions, Device, Gradient, Input, Unknown, UnknownInfo};
use jolty_numerics::integration::Formula;
use jolty_numerics::{Complex64, Pattern, PatternBuilder};
use serde::Deserialize;
use serde_json::Value;

use crate::ac::Linearization;
use crate::elaborate::Origin;
use crate::{Ac, DcSweep, Options, Outcome, SimulationError, Simulator, Transient};

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

/// A device made again computing with [`Gradient`], one number of its parameters
/// differentiated against.
struct Twin {
    /// The index of the device it is made from.
    device: usize,
    parameter: Parameter,
    made: Box<dyn Device<Gradient>>,
    pattern: Pattern,
    contributions: Contributions<Gradient>,
}

impl Twin {
    /// Evaluates the twin where the device is: at the solution `x`, at `time`.
    fn evaluate(&mut self, x: &[f64], time: f64, gmin: f64) {
        let mut input = Input::new(x, time, gmin);
        input.limiting = false;
        self.made
            .evaluate(&input, &mut self.contributions.output(&self.pattern));
    }

    /// `∂f/∂p`, by unknown, as last evaluated.
    fn residual(&self) -> Vec<f64> {
        derivatives(&self.contributions.residual)
    }

    /// `∂q/∂p`, by unknown, as last evaluated.
    fn charge(&self) -> Vec<f64> {
        derivatives(&self.contributions.charge)
    }

    /// Adds `(∂G/∂p + jω ∂C/∂p) x` to `product`, as last evaluated.
    fn add_product(&self, omega: f64, x: &[Complex64], product: &mut [Complex64]) {
        let entries = self.contributions.conductance.iter();
        let stored = entries.zip(&self.contributions.capacitance);
        for ((row, column), (conductance, capacitance)) in self.pattern.positions().zip(stored) {
            let value = Complex64::new(conductance.derivative, omega * capacitance.derivative);
            product[row] += value * x[column];
        }
    }
}

/// The derivatives the gradients `values` carry.
fn derivatives(values: &[Gradient]) -> Vec<f64> {
    values.iter().map(|value| value.derivative).collect()
}

/// The charges' derivatives with respect to one parameter at the points a transient
/// accepted last, which its integration formula reads.
struct Carried {
    /// `dq/dp` at the last point.
    last: Vec<f64>,
    /// `dq/dp` at the point before, unless the integration restarted at the last.
    before: Option<Vec<f64>>,
    /// `d(dq/dt)/dp` at the last point.
    rate: Vec<f64>,
}

/// The direct method's state along a transient: each parameter's derivatives of the
/// charges, and the outputs' derivatives at every accepted point.
pub(crate) struct Forward {
    outputs: Vec<Unknown>,
    twins: Vec<Twin>,
    carried: Vec<Carried>,
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
    /// [`SimulationError::Singular`] when a Jacobian is, and the errors of the analysis
    /// differentiated.
    pub fn sensitivity(
        &mut self,
        sensitivity: &Sensitivity,
        options: &Options,
    ) -> Result<Sensitivities, SimulationError> {
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
                let swept = self.dc_sweep(sweep, options)?;
                let index = self.source_index(&sweep.source)?;
                let found = self.along_sweep(&outputs, &mut twins, index, &swept, options);
                self.hold_source(index, None);
                for twin in twins.iter_mut().filter(|twin| twin.device == index) {
                    hold(twin, None);
                }
                (Derivatives::Real(found?), Outcome::DcSweep(swept))
            }
            Differentiated::Ac(ac) => {
                let response = self.ac(ac, options)?;
                let found = self.over_frequency(&outputs, &mut twins, &response, options)?;
                (Derivatives::Complex(found), Outcome::Ac(response))
            }
            Differentiated::Transient(transient) => {
                let mut forward = Forward {
                    outputs,
                    twins,
                    carried: Vec::new(),
                    found: Vec::new(),
                };
                let waveforms = self.transient_with(transient, options, Some(&mut forward))?;
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

    /// A twin of every device, for each number of its parameters its model reads as a
    /// real number, in the devices' order then the numbers'.
    fn twins(&self) -> Vec<Twin> {
        let mut twins = Vec::new();
        for (index, placed) in self.system.devices.iter().enumerate() {
            let Origin::Model(made) = &placed.origin else {
                continue;
            };
            for (name, value) in leaves(&made.parameters) {
                let mut seeded = made.parameters.clone();
                let derivative = serde_json::json!({"value": value, "derivative": 1.0});
                if set_leaf(&mut seeded, &name, derivative).is_none() {
                    continue;
                }
                let mut unknowns: Vec<UnknownInfo> =
                    self.system.unknowns[..placed.first_internal].to_vec();
                let mut builder = PatternBuilder::new();
                let mut binding = Binding::new(
                    placed.path(),
                    placed.nodes(),
                    &mut unknowns,
                    &mut builder,
                    self.system.temperature,
                );
                let Ok(device) = made.model.instantiate_gradient(&seeded, &mut binding) else {
                    continue;
                };
                let pattern = builder.finish(self.system.unknowns.len());
                twins.push(Twin {
                    device: index,
                    parameter: Parameter {
                        device: placed.path().to_owned(),
                        name,
                        value,
                    },
                    made: device,
                    contributions: Contributions::new(&pattern),
                    pattern,
                });
            }
        }
        twins
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
        self.evaluate(x, 0.0, (1.0, options.gmin), false);
        let mut factors = self
            .solver
            .factor(&self.workspace.contributions.conductance)?;
        let mut adjoints = Vec::with_capacity(outputs.len());
        for output in outputs {
            let mut adjoint = vec![0.0; x.len()];
            adjoint[output.index()] = 1.0;
            factors.solve_transpose(&mut adjoint)?;
            adjoints.push(adjoint);
        }
        let mut found = vec![Vec::with_capacity(twins.len()); outputs.len()];
        for twin in twins {
            twin.evaluate(x, 0.0, options.gmin);
            let residual = twin.residual();
            for (adjoint, derivatives) in adjoints.iter().zip(&mut found) {
                derivatives.push(-dot(adjoint, &residual));
            }
        }
        Ok(found)
    }

    /// The adjoint derivatives at each level of `swept`, the source at `index` held there,
    /// and its twins too.
    fn along_sweep(
        &mut self,
        outputs: &[Unknown],
        twins: &mut [Twin],
        index: usize,
        swept: &crate::Sweep,
        options: &Options,
    ) -> Result<Vec<Vec<Vec<f64>>>, SimulationError> {
        let mut found = Vec::with_capacity(swept.levels.len());
        for (&level, x) in swept.levels.iter().zip(&swept.values) {
            self.hold_source(index, Some(level));
            for twin in twins.iter_mut().filter(|twin| twin.device == index) {
                hold(twin, Some(level));
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
        self.evaluate(x, 0.0, (1.0, options.gmin), false);
        let mut factors = self
            .solver
            .factor(&self.workspace.contributions.conductance)?;
        let mut moved = Vec::with_capacity(twins.len());
        // Each twin stays evaluated at the operating point, which the response reads too.
        for twin in twins.iter_mut() {
            twin.evaluate(x, 0.0, options.gmin);
            let mut shift: Vec<f64> = twin.residual().iter().map(|value| -value).collect();
            factors.solve(&mut shift)?;
            moved.push(shift);
        }
        let mut bias = Vec::with_capacity(twins.len());
        for shift in &moved {
            bias.push(self.bias_derivative(x, shift, options));
        }
        self.evaluate(x, 0.0, (1.0, options.gmin), false);
        let mut linearization = Linearization::new(&self.system.pattern)?;
        let mut found = Vec::with_capacity(response.frequencies.len());
        for (&frequency, phasors) in response.frequencies.iter().zip(&response.values) {
            let omega = std::f64::consts::TAU * frequency;
            let mut factors = linearization.factor(&self.workspace.contributions, frequency)?;
            let mut adjoints = Vec::with_capacity(outputs.len());
            for output in outputs {
                let mut adjoint = vec![Complex64::default(); x.len()];
                adjoint[output.index()] = Complex64::new(1.0, 0.0);
                factors.solve_transpose(&mut adjoint)?;
                adjoints.push(adjoint);
            }
            let mut at = vec![Vec::with_capacity(twins.len()); outputs.len()];
            for (twin, moving) in twins.iter().zip(&bias) {
                let mut product = vec![Complex64::default(); x.len()];
                twin.add_product(omega, phasors, &mut product);
                if let Some((conductance, capacitance)) = moving {
                    for ((row, column), (g, c)) in self
                        .system
                        .pattern
                        .positions()
                        .zip(conductance.iter().zip(capacitance))
                    {
                        product[row] += Complex64::new(*g, omega * c) * phasors[column];
                    }
                }
                for (adjoint, derivatives) in adjoints.iter().zip(&mut at) {
                    let total: Complex64 = adjoint
                        .iter()
                        .zip(&product)
                        .map(|(lambda, value)| lambda * value)
                        .sum();
                    derivatives.push(-total);
                }
            }
            found.push(at);
        }
        Ok(found)
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
            .system
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
            self.evaluate(&moved, 0.0, (1.0, options.gmin), false);
            let contributions = &self.workspace.contributions;
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
    /// derivative of the solution, by the DC Jacobian, and of the charges.
    pub(crate) fn start_forward(
        &mut self,
        forward: &mut Forward,
        x: &[f64],
        options: &Options,
    ) -> Result<(), SimulationError> {
        self.evaluate(x, 0.0, (1.0, options.gmin), false);
        let mut factors = self
            .solver
            .factor(&self.workspace.contributions.conductance)?;
        let mut at = vec![Vec::with_capacity(forward.twins.len()); forward.outputs.len()];
        forward.carried.clear();
        for twin in &mut forward.twins {
            twin.evaluate(x, 0.0, options.gmin);
            let mut moved: Vec<f64> = twin.residual().iter().map(|value| -value).collect();
            factors.solve(&mut moved)?;
            for (output, derivatives) in forward.outputs.iter().zip(&mut at) {
                derivatives.push(moved[output.index()]);
            }
            let mut charge = twin.charge();
            let capacitances = &self.workspace.contributions.capacitance;
            add_product(&self.system.pattern, capacitances, &moved, &mut charge);
            forward.carried.push(Carried {
                last: charge,
                before: None,
                rate: vec![0.0; x.len()],
            });
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
        self.evaluate(x, time, (1.0, options.gmin), false);
        let jacobian: Vec<f64> = self
            .workspace
            .contributions
            .conductance
            .iter()
            .zip(&self.workspace.contributions.capacitance)
            .map(|(conductance, capacitance)| conductance + formula.alpha * capacitance)
            .collect();
        let mut factors = self.solver.factor(&jacobian)?;
        let mut at = vec![Vec::with_capacity(forward.twins.len()); forward.outputs.len()];
        for (twin, carried) in forward.twins.iter_mut().zip(&mut forward.carried) {
            twin.evaluate(x, time, options.gmin);
            let partial_charge = twin.charge();
            let before = carried.before.as_ref().unwrap_or(&carried.last);
            let history: Vec<f64> = (0..x.len())
                .map(|row| formula.history([carried.last[row], before[row]], carried.rate[row]))
                .collect();
            let mut moved: Vec<f64> = twin
                .residual()
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
            let capacitances = &self.workspace.contributions.capacitance;
            add_product(&self.system.pattern, capacitances, &moved, &mut charge);
            let rate = charge
                .iter()
                .zip(&history)
                .map(|(value, term)| formula.alpha * value + term)
                .collect();
            let last = std::mem::replace(&mut carried.last, charge);
            carried.before = (!restarted).then_some(last);
            carried.rate = rate;
        }
        forward.found.push(at);
        Ok(())
    }
}

/// Holds the twin's source at `level`, or releases it, if the twin is a source's.
fn hold(twin: &mut Twin, level: Option<f64>) {
    if let Some(source) = twin.made.as_source() {
        source.hold(level);
    }
}

/// Adds the product of the matrix of `values`, by stored position in `pattern`, and `s`
/// to `into`.
fn add_product(pattern: &Pattern, values: &[f64], s: &[f64], into: &mut [f64]) {
    for ((row, column), value) in pattern.positions().zip(values) {
        into[row] += value * s[column];
    }
}

/// The dot product of `a` and `b`.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Every number in `parameters`, with its name: its keys joined by dots, a list's
/// entries keyed by their index from 0.
fn leaves(parameters: &Parameters) -> Vec<(String, f64)> {
    fn walk(prefix: &str, value: &Value, found: &mut Vec<(String, f64)>) {
        let name = |key: &dyn std::fmt::Display| {
            if prefix.is_empty() {
                key.to_string()
            } else {
                format!("{prefix}.{key}")
            }
        };
        match value {
            Value::Number(number) => {
                if let Some(number) = number.as_f64() {
                    found.push((prefix.to_owned(), number));
                }
            }
            Value::Object(object) => {
                for (key, inner) in object {
                    walk(&name(key), inner, found);
                }
            }
            Value::Array(items) => {
                for (index, inner) in items.iter().enumerate() {
                    walk(&name(&index), inner, found);
                }
            }
            Value::Null | Value::Bool(_) | Value::String(_) => {}
        }
    }
    let mut found = Vec::new();
    for (key, value) in parameters {
        walk(key, value, &mut found);
    }
    found
}

/// The value named `key` within `value`, to change.
fn inner_mut<'a>(value: &'a mut Value, key: &str) -> Option<&'a mut Value> {
    match value {
        Value::Array(items) => items.get_mut(key.parse::<usize>().ok()?),
        _ => value.get_mut(key),
    }
}

/// Replaces the number named `name` in `parameters` by `replacement`; `None` when there
/// is none, or the replacement is not finite.
fn set_leaf(parameters: &mut Parameters, name: &str, replacement: Value) -> Option<()> {
    let mut keys = name.split('.');
    let mut value = parameters.get_mut(keys.next()?)?;
    for key in keys {
        value = inner_mut(value, key)?;
    }
    if !value.is_number() || replacement.is_null() {
        return None;
    }
    *value = replacement;
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn number(parameters: &Parameters, name: &str) -> Option<f64> {
        leaves(parameters)
            .into_iter()
            .find_map(|(found, value)| (found == name).then_some(value))
    }

    #[test]
    fn finds_and_sets_every_number_by_its_dotted_name() {
        let mut parameters = serde_json::json!({
            "resistance": 1e3, "waveform": {"pulse": {"initial": 0, "pulsed": 5}},
            "channel": "n", "points": [[0, 1]]
        })
        .as_object()
        .cloned()
        .unwrap();
        let names: Vec<String> = leaves(&parameters)
            .into_iter()
            .map(|(name, _)| name)
            .collect();
        assert_eq!(
            names,
            [
                "points.0.0",
                "points.0.1",
                "resistance",
                "waveform.pulse.initial",
                "waveform.pulse.pulsed"
            ]
        );
        assert_eq!(
            set_leaf(&mut parameters, "points.0.1", Value::from(2.0)),
            Some(())
        );
        assert_eq!(number(&parameters, "points.0.1"), Some(2.0));
        assert_eq!(
            set_leaf(&mut parameters, "points.1.0", Value::from(2.0)),
            None
        );
        assert_eq!(
            set_leaf(&mut parameters, "waveform.pulse.pulsed", Value::from(3.3)),
            Some(())
        );
        assert_eq!(
            set_leaf(&mut parameters, "waveform.pulse.width", Value::from(1.0)),
            None
        );
        assert_eq!(
            set_leaf(&mut parameters, "channel", Value::from(1.0)),
            None,
            "only a number is replaced"
        );
        assert_eq!(number(&parameters, "waveform.pulse.pulsed"), Some(3.3));
    }
}
