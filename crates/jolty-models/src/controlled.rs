use jolty_numerics::Entry;
use serde::Deserialize;

use crate::{
    Binding, Branch, BranchKind, Device, Input, Interface, NativeDevice, Node, Output,
    ParameterError, Quantity, Unknown,
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
    type Parameters = VoltageControlledVoltageSourceParameters;

    fn interface(_: &VoltageControlledVoltageSourceParameters) -> Interface {
        Interface::new(
            CONTROLLED_TERMINALS,
            &[Branch::new(0, 1, BranchKind::VoltageSource)],
        )
    }

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
    type Parameters = VoltageControlledCurrentSourceParameters;

    fn interface(_: &VoltageControlledCurrentSourceParameters) -> Interface {
        Interface::new(
            CONTROLLED_TERMINALS,
            &[Branch::new(0, 1, BranchKind::CurrentSource)],
        )
    }

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

/// A current-controlled current source, after Modelica's: its control port is a short
/// circuit that senses the current through it.
///
/// It drives `gain` times the control current from `positive` through the source to
/// `negative`. The control current flows from `control_positive` through the port to
/// `control_negative`, which it holds at the same voltage; it is an unknown labelled
/// `control`.
#[derive(Debug)]
pub struct CurrentControlledCurrentSource {
    nodes: [Node; 4],
    control: Unknown,
    gain: f64,
    /// The entries at (each control terminal, control), (control, each control terminal)
    /// and (each output terminal, control).
    entries: [Option<Entry>; 6],
}

/// The parameters of a [`CurrentControlledCurrentSource`].
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CurrentControlledCurrentSourceParameters {
    /// The ratio of the output current to the control current.
    pub gain: f64,
}

/// The entries of a sensing port: the control current's into its terminals, and its
/// equation's on their voltages.
fn sense_entries(binding: &mut Binding<'_>, nodes: [Node; 4], control: Node) -> [Option<Entry>; 4] {
    [
        binding.entry(nodes[2], control),
        binding.entry(nodes[3], control),
        binding.entry(control, nodes[2]),
        binding.entry(control, nodes[3]),
    ]
}

/// Adds a sensing port: the control current through it, and its terminals held at one
/// voltage. Returns the control current.
fn add_sense(
    nodes: [Node; 4],
    control: Unknown,
    entries: [Option<Entry>; 4],
    input: &Input<'_>,
    output: &mut Output<'_>,
) -> f64 {
    let [.., control_positive, control_negative] = nodes;
    let [into_positive, into_negative, on_positive, on_negative] = entries;
    let branch = Some(control);
    let current = input.value(branch);
    output.add_residual(control_positive, current);
    output.add_residual(control_negative, -current);
    output.add_conductance(into_positive, 1.0);
    output.add_conductance(into_negative, -1.0);
    output.add_residual(
        branch,
        input.value(control_positive) - input.value(control_negative),
    );
    output.add_conductance(on_positive, 1.0);
    output.add_conductance(on_negative, -1.0);
    current
}

impl NativeDevice for CurrentControlledCurrentSource {
    const NAME: &'static str = "current_controlled_current_source";
    type Parameters = CurrentControlledCurrentSourceParameters;

    fn interface(_: &CurrentControlledCurrentSourceParameters) -> Interface {
        Interface::new(
            CONTROLLED_TERMINALS,
            &[
                Branch::new(0, 1, BranchKind::CurrentSource),
                Branch::new(2, 3, BranchKind::VoltageSource),
            ],
        )
    }

    fn new(
        parameters: CurrentControlledCurrentSourceParameters,
        binding: &mut Binding<'_>,
    ) -> Result<Self, ParameterError> {
        let nodes = [0, 1, 2, 3].map(|index| binding.terminal(index));
        let control = binding.internal("control", Quantity::Current);
        let branch = Some(control);
        let [a, b, c, d] = sense_entries(binding, nodes, branch);
        let entries = [
            a,
            b,
            c,
            d,
            binding.entry(nodes[0], branch),
            binding.entry(nodes[1], branch),
        ];
        Ok(Self {
            nodes,
            control,
            gain: parameters.gain,
            entries,
        })
    }
}

impl Device for CurrentControlledCurrentSource {
    fn evaluate(&mut self, input: &Input<'_>, output: &mut Output<'_>) {
        let [a, b, c, d, positive_control, negative_control] = self.entries;
        let current = add_sense(self.nodes, self.control, [a, b, c, d], input, output);
        let [positive, negative, ..] = self.nodes;
        output.add_residual(positive, self.gain * current);
        output.add_residual(negative, -self.gain * current);
        output.add_conductance(positive_control, self.gain);
        output.add_conductance(negative_control, -self.gain);
    }
}

/// A current-controlled voltage source, after Modelica's: its control port is a short
/// circuit that senses the current through it.
///
/// It holds `v_positive − v_negative = transresistance · i_control`. The control current
/// flows from `control_positive` through the port to `control_negative`, which it holds
/// at the same voltage; it is an unknown labelled `control`. The output current, from
/// `positive` through the source to `negative`, is an unknown labelled `current`.
#[derive(Debug)]
pub struct CurrentControlledVoltageSource {
    nodes: [Node; 4],
    control: Unknown,
    current: Unknown,
    transresistance: f64,
    /// The sensing port's entries, then those at (positive, current), (negative,
    /// current), (current, positive), (current, negative) and (current, control).
    entries: [Option<Entry>; 9],
}

/// The parameters of a [`CurrentControlledVoltageSource`].
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CurrentControlledVoltageSourceParameters {
    /// The ratio of the output voltage to the control current, in ohms.
    pub transresistance: f64,
}

impl NativeDevice for CurrentControlledVoltageSource {
    const NAME: &'static str = "current_controlled_voltage_source";
    type Parameters = CurrentControlledVoltageSourceParameters;

    fn interface(_: &CurrentControlledVoltageSourceParameters) -> Interface {
        Interface::new(
            CONTROLLED_TERMINALS,
            &[
                Branch::new(0, 1, BranchKind::VoltageSource),
                Branch::new(2, 3, BranchKind::VoltageSource),
            ],
        )
    }

    fn new(
        parameters: CurrentControlledVoltageSourceParameters,
        binding: &mut Binding<'_>,
    ) -> Result<Self, ParameterError> {
        let nodes = [0, 1, 2, 3].map(|index| binding.terminal(index));
        let control = binding.internal("control", Quantity::Current);
        let current = binding.internal("current", Quantity::Current);
        let [a, b, c, d] = sense_entries(binding, nodes, Some(control));
        let branch = Some(current);
        let entries = [
            a,
            b,
            c,
            d,
            binding.entry(nodes[0], branch),
            binding.entry(nodes[1], branch),
            binding.entry(branch, nodes[0]),
            binding.entry(branch, nodes[1]),
            binding.entry(branch, Some(control)),
        ];
        Ok(Self {
            nodes,
            control,
            current,
            transresistance: parameters.transresistance,
            entries,
        })
    }
}

impl Device for CurrentControlledVoltageSource {
    fn evaluate(&mut self, input: &Input<'_>, output: &mut Output<'_>) {
        let [
            a,
            b,
            c,
            d,
            positive_current,
            negative_current,
            current_positive,
            current_negative,
            current_control,
        ] = self.entries;
        let control = add_sense(self.nodes, self.control, [a, b, c, d], input, output);
        let [positive, negative, ..] = self.nodes;
        let branch = Some(self.current);
        let current = input.value(branch);
        output.add_residual(positive, current);
        output.add_residual(negative, -current);
        output.add_conductance(positive_current, 1.0);
        output.add_conductance(negative_current, -1.0);
        let voltage = input.value(positive) - input.value(negative);
        output.add_residual(branch, voltage - self.transresistance * control);
        output.add_conductance(current_positive, 1.0);
        output.add_conductance(current_negative, -1.0);
        output.add_conductance(current_control, -self.transresistance);
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
        assert_jacobians_match::<CurrentControlledCurrentSource>(
            r#"{"gain": 50}"#,
            &[1.0, -0.5, 0.3, 0.1, 2e-3],
        );
        assert_jacobians_match::<CurrentControlledVoltageSource>(
            r#"{"transresistance": 1000}"#,
            &[1.0, -0.5, 0.3, 0.1, 2e-3, -1e-3],
        );
    }

    #[test]
    fn a_control_port_senses_its_current_and_holds_no_voltage() {
        let mut bench =
            crate::testing::Bench::new::<CurrentControlledCurrentSource>(r#"{"gain": 50}"#);
        let evaluation = bench.evaluate(&[1.0, -0.5, 0.3, 0.3, 2e-3]);
        // 2 mA enters at control_positive and leaves at control_negative; 100 mA leaves
        // positive through the source; the port's equation holds its two voltages equal.
        crate::testing::assert_close(&evaluation.residual, &[0.1, -0.1, 2e-3, -2e-3, 0.0]);
        assert_eq!(bench.unknowns()[4], "device:control");
    }
}
