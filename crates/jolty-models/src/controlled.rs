use jolty_numerics::Entry;
use serde::Deserialize;

use crate::{
    Binding, Branch, BranchKind, Device, Input, NativeDevice, Node, Output, ParameterError,
    Quantity, Unknown,
};

/// The terminals of a controlled source: an output pair and a control pair.
const CONTROLLED_TERMINALS: &[&str] = &[
    "positive",
    "negative",
    "control_positive",
    "control_negative",
];

/// A voltage-controlled voltage source.
///
/// It holds `v_positive − v_negative = gain · (v_control_positive − v_control_negative)`. Its
/// current, flowing from `positive` through the source to `negative`, is an unknown labelled
/// `current`. The control terminals draw no current.
#[derive(Debug)]
pub struct VoltageControlledVoltageSource {
    nodes: [Node; 4],
    current: Unknown,
    gain: f64,
    /// The entries at (positive, current), (negative, current), and (current, each terminal).
    entries: [Option<Entry>; 6],
}

/// The parameters of a [`VoltageControlledVoltageSource`].
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VoltageControlledVoltageSourceParameters {
    /// The ratio of the output voltage to the control voltage.
    pub gain: f64,
}

impl NativeDevice for VoltageControlledVoltageSource {
    const NAME: &'static str = "voltage_controlled_voltage_source";
    const TERMINALS: &'static [&'static str] = CONTROLLED_TERMINALS;
    const BRANCHES: &'static [Branch] = &[Branch::new(0, 1, BranchKind::VoltageSource)];
    type Parameters = VoltageControlledVoltageSourceParameters;

    fn new(
        parameters: VoltageControlledVoltageSourceParameters,
        binding: &mut Binding<'_>,
    ) -> Result<Self, ParameterError> {
        let nodes = [0, 1, 2, 3].map(|index| binding.terminal(index));
        let current = binding.internal("current", Quantity::Current);
        let branch = Some(current);
        let entries = [
            binding.entry(nodes[0], branch),
            binding.entry(nodes[1], branch),
            binding.entry(branch, nodes[0]),
            binding.entry(branch, nodes[1]),
            binding.entry(branch, nodes[2]),
            binding.entry(branch, nodes[3]),
        ];
        Ok(Self {
            nodes,
            current,
            gain: parameters.gain,
            entries,
        })
    }
}

impl Device for VoltageControlledVoltageSource {
    fn evaluate(&mut self, input: &Input<'_>, output: &mut Output<'_>) {
        let [positive, negative, control_positive, control_negative] = self.nodes;
        let [
            positive_current,
            negative_current,
            current_positive,
            current_negative,
            current_control_positive,
            current_control_negative,
        ] = self.entries;
        let branch = Some(self.current);
        let current = input.value(branch);
        output.add_residual(positive, current);
        output.add_residual(negative, -current);
        output.add_conductance(positive_current, 1.0);
        output.add_conductance(negative_current, -1.0);
        let control = input.value(control_positive) - input.value(control_negative);
        let voltage = input.value(positive) - input.value(negative);
        output.add_residual(branch, voltage - self.gain * control);
        output.add_conductance(current_positive, 1.0);
        output.add_conductance(current_negative, -1.0);
        output.add_conductance(current_control_positive, -self.gain);
        output.add_conductance(current_control_negative, self.gain);
    }
}

/// A voltage-controlled current source.
///
/// It drives `transconductance · (v_control_positive − v_control_negative)` from `positive`
/// through the source to `negative`. The control terminals draw no current.
#[derive(Debug)]
pub struct VoltageControlledCurrentSource {
    nodes: [Node; 4],
    transconductance: f64,
    /// The entries joining each output terminal to each control terminal.
    entries: [Option<Entry>; 4],
}

/// The parameters of a [`VoltageControlledCurrentSource`].
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VoltageControlledCurrentSourceParameters {
    /// The ratio of the output current to the control voltage, in siemens.
    pub transconductance: f64,
}

impl NativeDevice for VoltageControlledCurrentSource {
    const NAME: &'static str = "voltage_controlled_current_source";
    const TERMINALS: &'static [&'static str] = CONTROLLED_TERMINALS;
    const BRANCHES: &'static [Branch] = &[Branch::new(0, 1, BranchKind::CurrentSource)];
    type Parameters = VoltageControlledCurrentSourceParameters;

    fn new(
        parameters: VoltageControlledCurrentSourceParameters,
        binding: &mut Binding<'_>,
    ) -> Result<Self, ParameterError> {
        let nodes = [0, 1, 2, 3].map(|index| binding.terminal(index));
        let entries = [
            binding.entry(nodes[0], nodes[2]),
            binding.entry(nodes[0], nodes[3]),
            binding.entry(nodes[1], nodes[2]),
            binding.entry(nodes[1], nodes[3]),
        ];
        Ok(Self {
            nodes,
            transconductance: parameters.transconductance,
            entries,
        })
    }
}

impl Device for VoltageControlledCurrentSource {
    fn evaluate(&mut self, input: &Input<'_>, output: &mut Output<'_>) {
        let [positive, negative, control_positive, control_negative] = self.nodes;
        let [
            positive_control_positive,
            positive_control_negative,
            negative_control_positive,
            negative_control_negative,
        ] = self.entries;
        let transconductance = self.transconductance;
        let control = input.value(control_positive) - input.value(control_negative);
        output.add_residual(positive, transconductance * control);
        output.add_residual(negative, -transconductance * control);
        output.add_conductance(positive_control_positive, transconductance);
        output.add_conductance(positive_control_negative, -transconductance);
        output.add_conductance(negative_control_positive, -transconductance);
        output.add_conductance(negative_control_negative, transconductance);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::assert_jacobians_match;

    #[test]
    fn jacobians_match_finite_differences() {
        assert_jacobians_match::<VoltageControlledVoltageSource>(
            r#"{"gain": 10}"#,
            &[1.0, -0.5, 0.3, 0.1, 1e-3],
        );
        assert_jacobians_match::<VoltageControlledCurrentSource>(
            r#"{"transconductance": 0.02}"#,
            &[1.0, -0.5, 0.3, 0.1],
        );
    }
}
