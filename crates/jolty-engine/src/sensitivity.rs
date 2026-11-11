//! DC sensitivities by the adjoint method: how every unknown asked for moves with every
//! parameter of every device, from one transposed solve per unknown.
//!
//! At the operating point `F(x, p) = 0`, an output `y = x_k` moves with a parameter `p` by
//! `dy/dp = −λᵀ ∂F/∂p`, where `Jᵀ λ = e_k` and `J = ∂F/∂x` is the Jacobian Newton iteration
//! converged with. One factorization of `J` and one transposed solve per output serve every
//! parameter, so the cost does not grow with their number. Each device's `∂F/∂p` comes from
//! the device alone, made again with the parameter a little above and below its value and
//! evaluated at the operating point: a central difference over one device, not a solve.

use jolty_ir::Parameters;
use jolty_models::{Binding, Input, Output, UnknownInfo};
use jolty_numerics::PatternBuilder;
use serde::Deserialize;
use serde_json::Value;

use crate::{OperatingPoint, Options, SimulationError, Simulator};

/// The relative step of a parameter in its central difference, or its absolute step, in
/// SI base units, when it is zero.
const STEP: f64 = 1e-6;

/// A sensitivity analysis: how the unknowns `outputs` move with every device parameter at
/// the operating point.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sensitivity {
    /// The unknowns, by name, such as `out` or `V1:current`.
    pub outputs: Vec<String>,
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

/// The sensitivities of a [`Sensitivity`] analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct Sensitivities {
    /// The unknowns, by name, in the order of `derivatives`.
    pub outputs: Vec<String>,
    /// Every parameter the derivatives are with respect to.
    pub parameters: Vec<Parameter>,
    /// For each output, its derivative with respect to each parameter.
    pub derivatives: Vec<Vec<f64>>,
    /// The operating point the derivatives are taken at.
    pub operating_point: OperatingPoint,
}

impl Simulator {
    /// The derivative of each unknown of `sensitivity` with respect to every number of every
    /// device's parameters, at the operating point.
    ///
    /// A number the device's model cannot take a little away from its value, such as a
    /// model's level, has no derivative and is left out.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::UnknownOutput`] for an output that is no unknown,
    /// [`SimulationError::Singular`] when the Jacobian is, and the errors of
    /// [`Simulator::operating_point`].
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
        let operating_point = self.operating_point(options)?;
        let x = &operating_point.values;
        self.evaluate(x, 0.0, (1.0, options.gmin), false);
        let mut factors = self.solver.factor(&self.workspace.conductance)?;
        let mut adjoints = Vec::with_capacity(outputs.len());
        for output in &outputs {
            let mut adjoint = vec![0.0; x.len()];
            adjoint[output.index()] = 1.0;
            factors.solve_transpose(&mut adjoint)?;
            adjoints.push(adjoint);
        }
        let mut parameters = Vec::new();
        let mut derivatives = vec![Vec::new(); outputs.len()];
        for index in 0..self.system.devices.len() {
            let Some(made) = &self.system.devices[index].made else {
                continue;
            };
            for (name, value) in leaves(&made.parameters) {
                let Some(residual) = self.residual_derivative(index, &name, x, options) else {
                    continue;
                };
                for (adjoint, found) in adjoints.iter().zip(&mut derivatives) {
                    let dot: f64 = adjoint
                        .iter()
                        .zip(&residual)
                        .map(|(lambda, partial)| lambda * partial)
                        .sum();
                    found.push(-dot);
                }
                parameters.push(Parameter {
                    device: self.system.devices[index].path().to_owned(),
                    name,
                    value,
                });
            }
        }
        Ok(Sensitivities {
            outputs: sensitivity.outputs.clone(),
            parameters,
            derivatives,
            operating_point,
        })
    }

    /// `∂F/∂p` of the device at `index` for its parameter `name` at the solution `x`: the
    /// central difference of its residual, made again with the parameter a little above
    /// and below its value. `None` when its model cannot take either value.
    fn residual_derivative(
        &self,
        index: usize,
        name: &str,
        x: &[f64],
        options: &Options,
    ) -> Option<Vec<f64>> {
        let placed = &self.system.devices[index];
        let made = placed.made.as_ref()?;
        let value = leaf(&made.parameters, name)?;
        let step = if value == 0.0 {
            STEP
        } else {
            STEP * value.abs()
        };
        let dimension = self.system.unknowns.len();
        let mut residuals = [vec![0.0; dimension], vec![0.0; dimension]];
        for (residual, shifted) in residuals.iter_mut().zip([value + step, value - step]) {
            let mut parameters = made.parameters.clone();
            set_leaf(&mut parameters, name, shifted)?;
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
            let mut device = made.model.instantiate(&parameters, &mut binding).ok()?;
            let pattern = builder.finish(dimension);
            let (mut scale, mut charge) = (vec![0.0; dimension], vec![0.0; dimension]);
            let mut conductance = vec![0.0; pattern.len()];
            let mut capacitance = vec![0.0; pattern.len()];
            let mut output = Output::new(
                &pattern,
                residual,
                &mut scale,
                &mut charge,
                &mut conductance,
                &mut capacitance,
            );
            let mut input = Input::new(x, 0.0, options.gmin);
            input.limiting = false;
            device.evaluate(&input, &mut output);
        }
        let [above, below] = residuals;
        Some(
            above
                .iter()
                .zip(&below)
                .map(|(above, below)| (above - below) / (2.0 * step))
                .collect(),
        )
    }
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

/// The value named `key` within `value`: an object's entry, or a list's by its index.
fn inner<'a>(value: &'a Value, key: &str) -> Option<&'a Value> {
    match value {
        Value::Array(items) => items.get(key.parse::<usize>().ok()?),
        _ => value.get(key),
    }
}

/// The value named `key` within `value`, to change.
fn inner_mut<'a>(value: &'a mut Value, key: &str) -> Option<&'a mut Value> {
    match value {
        Value::Array(items) => items.get_mut(key.parse::<usize>().ok()?),
        _ => value.get_mut(key),
    }
}

/// The number named `name` in `parameters`.
fn leaf(parameters: &Parameters, name: &str) -> Option<f64> {
    let mut keys = name.split('.');
    let mut value = parameters.get(keys.next()?)?;
    for key in keys {
        value = inner(value, key)?;
    }
    value.as_f64()
}

/// Sets the number named `name` in `parameters` to `number`; `None` when there is none
/// or `number` is not finite.
fn set_leaf(parameters: &mut Parameters, name: &str, number: f64) -> Option<()> {
    let mut keys = name.split('.');
    let mut value = parameters.get_mut(keys.next()?)?;
    for key in keys {
        value = inner_mut(value, key)?;
    }
    *value = Value::Number(serde_json::Number::from_f64(number)?);
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(set_leaf(&mut parameters, "points.0.1", 2.0), Some(()));
        assert_eq!(leaf(&parameters, "points.0.1"), Some(2.0));
        assert_eq!(set_leaf(&mut parameters, "points.1.0", 2.0), None);
        assert_eq!(
            set_leaf(&mut parameters, "waveform.pulse.pulsed", 3.3),
            Some(())
        );
        assert_eq!(set_leaf(&mut parameters, "waveform.pulse.width", 1.0), None);
        assert_eq!(leaf(&parameters, "waveform.pulse.pulsed"), Some(3.3));
    }
}
