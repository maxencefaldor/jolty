use jolty_numerics::Entry;
use serde::Deserialize;

use crate::stamp::CurrentBranch;
use crate::{
    Binding, Branch, BranchKind, Device, Input, Interface, NativeDevice, NativeModel, Node, Output,
    ParameterError, Real,
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
pub struct VoltageControlledVoltageSource<R: Real = f64> {
    branch: CurrentBranch,
    control: [Node; 2],
    gain: R,
    /// The entries at (current, each control terminal).
    entries: [Option<Entry>; 2],
}

/// The parameters of a [`VoltageControlledVoltageSource`].
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, bound = "")]
pub struct VoltageControlledVoltageSourceParameters<R: Real = f64> {
    /// The ratio of the output voltage to the control voltage.
    pub gain: R,
}

impl NativeModel for VoltageControlledVoltageSource {
    type Device<R: Real> = VoltageControlledVoltageSource<R>;
}

impl<R: Real> NativeDevice<R> for VoltageControlledVoltageSource<R> {
    const NAME: &'static str = "voltage_controlled_voltage_source";
    type Parameters = VoltageControlledVoltageSourceParameters<R>;

    fn interface(_: &VoltageControlledVoltageSourceParameters<R>) -> Interface {
        Interface::new(
            CONTROLLED_TERMINALS,
            &[Branch::new(0, 1, BranchKind::VoltageSource)],
        )
    }

    fn new(
        parameters: VoltageControlledVoltageSourceParameters<R>,
        binding: &mut Binding<'_>,
    ) -> Result<Self, ParameterError> {
        let [positive, negative, control_positive, control_negative] =
            [0, 1, 2, 3].map(|index| binding.terminal(index));
        let branch = CurrentBranch::new(binding, "current", [positive, negative]);
        let row = Some(branch.unknown());
        let entries = [
            binding.entry(row, control_positive),
            binding.entry(row, control_negative),
        ];
        Ok(Self {
            branch,
            control: [control_positive, control_negative],
            gain: parameters.gain,
            entries,
        })
    }
}

impl<R: Real> Device<R> for VoltageControlledVoltageSource<R> {
    fn evaluate(&mut self, input: &Input<'_>, output: &mut Output<'_, R>) {
        let [control_positive, control_negative] = self.control;
        let control = input.value(control_positive) - input.value(control_negative);
        let voltage = self.branch.voltage(input);
        self.branch
            .add(input, output, -(self.gain * control - voltage));
        let [current_control_positive, current_control_negative] = self.entries;
        output.add_conductance(current_control_positive, -self.gain);
        output.add_conductance(current_control_negative, self.gain);
    }
}

/// A voltage-controlled current source.
///
/// It drives `transconductance · (v_control_positive − v_control_negative)` from `positive`
/// through the source to `negative`. The control terminals draw no current.
#[derive(Debug)]
pub struct VoltageControlledCurrentSource<R: Real = f64> {
    nodes: [Node; 4],
    transconductance: R,
    /// The entries joining each output terminal to each control terminal.
    entries: [Option<Entry>; 4],
}

/// The parameters of a [`VoltageControlledCurrentSource`].
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, bound = "")]
pub struct VoltageControlledCurrentSourceParameters<R: Real = f64> {
    /// The ratio of the output current to the control voltage, in siemens.
    pub transconductance: R,
}

impl NativeModel for VoltageControlledCurrentSource {
    type Device<R: Real> = VoltageControlledCurrentSource<R>;
}

impl<R: Real> NativeDevice<R> for VoltageControlledCurrentSource<R> {
    const NAME: &'static str = "voltage_controlled_current_source";
    type Parameters = VoltageControlledCurrentSourceParameters<R>;

    fn interface(_: &VoltageControlledCurrentSourceParameters<R>) -> Interface {
        Interface::new(
            CONTROLLED_TERMINALS,
            &[Branch::new(0, 1, BranchKind::CurrentSource)],
        )
    }

    fn new(
        parameters: VoltageControlledCurrentSourceParameters<R>,
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

impl<R: Real> Device<R> for VoltageControlledCurrentSource<R> {
    fn evaluate(&mut self, input: &Input<'_>, output: &mut Output<'_, R>) {
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
pub struct CurrentControlledCurrentSource<R: Real = f64> {
    output: [Node; 2],
    /// The sensing port, whose current is the control.
    sense: CurrentBranch,
    gain: R,
    /// The entries at (each output terminal, control).
    entries: [Option<Entry>; 2],
}

/// The parameters of a [`CurrentControlledCurrentSource`].
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, bound = "")]
pub struct CurrentControlledCurrentSourceParameters<R: Real = f64> {
    /// The ratio of the output current to the control current.
    pub gain: R,
}

/// Adds a sensing port, a short circuit, and returns the current through it.
fn add_sense<R: Real>(sense: &CurrentBranch, input: &Input<'_>, output: &mut Output<'_, R>) -> f64 {
    sense.add(input, output, R::constant(sense.voltage(input)));
    sense.current(input)
}

impl NativeModel for CurrentControlledCurrentSource {
    type Device<R: Real> = CurrentControlledCurrentSource<R>;
}

impl<R: Real> NativeDevice<R> for CurrentControlledCurrentSource<R> {
    const NAME: &'static str = "current_controlled_current_source";
    type Parameters = CurrentControlledCurrentSourceParameters<R>;

    fn interface(_: &CurrentControlledCurrentSourceParameters<R>) -> Interface {
        Interface::new(
            CONTROLLED_TERMINALS,
            &[
                Branch::new(0, 1, BranchKind::CurrentSource),
                Branch::new(2, 3, BranchKind::VoltageSource),
            ],
        )
    }

    fn new(
        parameters: CurrentControlledCurrentSourceParameters<R>,
        binding: &mut Binding<'_>,
    ) -> Result<Self, ParameterError> {
        let [positive, negative, control_positive, control_negative] =
            [0, 1, 2, 3].map(|index| binding.terminal(index));
        let sense = CurrentBranch::new(binding, "control", [control_positive, control_negative]);
        let control = Some(sense.unknown());
        let entries = [
            binding.entry(positive, control),
            binding.entry(negative, control),
        ];
        Ok(Self {
            output: [positive, negative],
            sense,
            gain: parameters.gain,
            entries,
        })
    }
}

impl<R: Real> Device<R> for CurrentControlledCurrentSource<R> {
    fn evaluate(&mut self, input: &Input<'_>, output: &mut Output<'_, R>) {
        let current = add_sense(&self.sense, input, output);
        let [positive, negative] = self.output;
        let [positive_control, negative_control] = self.entries;
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
pub struct CurrentControlledVoltageSource<R: Real = f64> {
    /// The sensing port, whose current is the control.
    sense: CurrentBranch,
    branch: CurrentBranch,
    transresistance: R,
    /// The entry at (current, control).
    entry: Option<Entry>,
}

/// The parameters of a [`CurrentControlledVoltageSource`].
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, bound = "")]
pub struct CurrentControlledVoltageSourceParameters<R: Real = f64> {
    /// The ratio of the output voltage to the control current, in ohms.
    pub transresistance: R,
}

impl NativeModel for CurrentControlledVoltageSource {
    type Device<R: Real> = CurrentControlledVoltageSource<R>;
}

impl<R: Real> NativeDevice<R> for CurrentControlledVoltageSource<R> {
    const NAME: &'static str = "current_controlled_voltage_source";
    type Parameters = CurrentControlledVoltageSourceParameters<R>;

    fn interface(_: &CurrentControlledVoltageSourceParameters<R>) -> Interface {
        Interface::new(
            CONTROLLED_TERMINALS,
            &[
                Branch::new(0, 1, BranchKind::VoltageSource),
                Branch::new(2, 3, BranchKind::VoltageSource),
            ],
        )
    }

    fn new(
        parameters: CurrentControlledVoltageSourceParameters<R>,
        binding: &mut Binding<'_>,
    ) -> Result<Self, ParameterError> {
        let [positive, negative, control_positive, control_negative] =
            [0, 1, 2, 3].map(|index| binding.terminal(index));
        let sense = CurrentBranch::new(binding, "control", [control_positive, control_negative]);
        let branch = CurrentBranch::new(binding, "current", [positive, negative]);
        let entry = binding.entry(Some(branch.unknown()), Some(sense.unknown()));
        Ok(Self {
            sense,
            branch,
            transresistance: parameters.transresistance,
            entry,
        })
    }
}

impl<R: Real> Device<R> for CurrentControlledVoltageSource<R> {
    fn evaluate(&mut self, input: &Input<'_>, output: &mut Output<'_, R>) {
        let control = add_sense(&self.sense, input, output);
        let voltage = self.branch.voltage(input);
        self.branch
            .add(input, output, -(self.transresistance * control - voltage));
        output.add_conductance(self.entry, -self.transresistance);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{assert_jacobians_match, assert_parameter_gradients_match};

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
    fn parameter_gradients_are_exact() {
        assert_parameter_gradients_match::<VoltageControlledVoltageSource>(
            r#"{"gain": 10}"#,
            &[1.0, -0.5, 0.3, 0.1, 1e-3],
        );
        assert_parameter_gradients_match::<VoltageControlledCurrentSource>(
            r#"{"transconductance": 0.02}"#,
            &[1.0, -0.5, 0.3, 0.1],
        );
        assert_parameter_gradients_match::<CurrentControlledCurrentSource>(
            r#"{"gain": 50}"#,
            &[1.0, -0.5, 0.3, 0.1, 2e-3],
        );
        assert_parameter_gradients_match::<CurrentControlledVoltageSource>(
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
