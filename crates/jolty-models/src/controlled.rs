#![cfg_attr(
    not(test),
    deny(
        clippy::indexing_slicing,
        clippy::panic,
        reason = "a kernel never panics, and reads fixed arrays through `get`"
    )
)]

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::kernel::parts::CurrentBranch;
use crate::{Branch, BranchKind, Interface, Kernel, Layout, Local, ParameterError, Real, Stamp};

/// The terminals of a controlled source: an output pair and a control pair.
const CONTROLLED_TERMINALS: &[&str] = &[
    "positive",
    "negative",
    "control_positive",
    "control_negative",
];

/// The local index of a controlled source's terminal `positive`.
const POSITIVE: usize = 0;
/// The local index of its terminal `negative`.
const NEGATIVE: usize = 1;
/// The local index of its terminal `control_positive`.
const CONTROL_POSITIVE: usize = 2;
/// The local index of its terminal `control_negative`.
const CONTROL_NEGATIVE: usize = 3;

/// A voltage-controlled voltage source.
///
/// It holds `v_positive − v_negative = gain · (v_control_positive − v_control_negative)`. Its
/// current, flowing from `positive` through the source to `negative`, is an unknown labelled
/// `current`. The control terminals draw no current.
#[derive(Debug, Clone, Copy)]
pub struct VoltageControlledVoltageSource;

/// Where a voltage-controlled voltage source's contributions go: through its branch, and
/// at the entries of its current's equation at each control terminal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VoltageControlledVoltageSourceShape {
    branch: CurrentBranch,
    entries: [usize; 2],
}

/// The parameters of a [`VoltageControlledVoltageSource`].
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields, bound = "")]
pub struct VoltageControlledVoltageSourceParameters<R: Real = f64> {
    /// The ratio of the output voltage to the control voltage.
    pub gain: R,
}

impl Kernel for VoltageControlledVoltageSource {
    const NAME: &'static str = "voltage_controlled_voltage_source";
    type Parameters<R: Real> = VoltageControlledVoltageSourceParameters<R>;
    type Shape = VoltageControlledVoltageSourceShape;
    type Prepared<R: Real> = R;
    type Memory = ();

    fn interface<R: Real>(_: &VoltageControlledVoltageSourceParameters<R>) -> Interface {
        Interface::new(
            CONTROLLED_TERMINALS,
            &[Branch::new(0, 1, BranchKind::VoltageSource)],
        )
    }

    fn prepare<R: Real>(
        parameters: &VoltageControlledVoltageSourceParameters<R>,
        _: R,
    ) -> Result<R, ParameterError> {
        Ok(parameters.gain)
    }

    fn layout<R: Real>(
        _: &VoltageControlledVoltageSourceParameters<R>,
        _: &R,
        layout: &mut Layout,
    ) -> Result<VoltageControlledVoltageSourceShape, ParameterError> {
        let branch = CurrentBranch::new(layout, "current", [POSITIVE, NEGATIVE]);
        let row = branch.unknown();
        let entries = [
            layout.entry(row, CONTROL_POSITIVE),
            layout.entry(row, CONTROL_NEGATIVE),
        ];
        Ok(VoltageControlledVoltageSourceShape { branch, entries })
    }

    fn linear(_: &VoltageControlledVoltageSourceShape, _: &f64) -> bool {
        true
    }

    fn evaluate<R: Real>(
        shape: &VoltageControlledVoltageSourceShape,
        &gain: &R,
        (): &mut (),
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        let control = at.value(CONTROL_POSITIVE) - at.value(CONTROL_NEGATIVE);
        let voltage = shape.branch.voltage(at);
        shape.branch.add(at, stamp, -(gain * control - voltage));
        let [current_control_positive, current_control_negative] = shape.entries;
        stamp.add_conductance(current_control_positive, -gain);
        stamp.add_conductance(current_control_negative, gain);
    }
}

/// A voltage-controlled current source.
///
/// It drives `transconductance · (v_control_positive − v_control_negative)` from `positive`
/// through the source to `negative`. The control terminals draw no current.
#[derive(Debug, Clone, Copy)]
pub struct VoltageControlledCurrentSource;

/// Where a voltage-controlled current source's contributions go: at the entries joining
/// each output terminal to each control terminal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ControlledCurrentShape([usize; 4]);

/// The parameters of a [`VoltageControlledCurrentSource`].
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields, bound = "")]
pub struct VoltageControlledCurrentSourceParameters<R: Real = f64> {
    /// The ratio of the output current to the control voltage, in siemens.
    pub transconductance: R,
}

impl Kernel for VoltageControlledCurrentSource {
    const NAME: &'static str = "voltage_controlled_current_source";
    type Parameters<R: Real> = VoltageControlledCurrentSourceParameters<R>;
    type Shape = ControlledCurrentShape;
    type Prepared<R: Real> = R;
    type Memory = ();

    fn interface<R: Real>(_: &VoltageControlledCurrentSourceParameters<R>) -> Interface {
        Interface::new(
            CONTROLLED_TERMINALS,
            &[Branch::new(0, 1, BranchKind::CurrentSource)],
        )
    }

    fn prepare<R: Real>(
        parameters: &VoltageControlledCurrentSourceParameters<R>,
        _: R,
    ) -> Result<R, ParameterError> {
        Ok(parameters.transconductance)
    }

    fn layout<R: Real>(
        _: &VoltageControlledCurrentSourceParameters<R>,
        _: &R,
        layout: &mut Layout,
    ) -> Result<ControlledCurrentShape, ParameterError> {
        Ok(ControlledCurrentShape([
            layout.entry(POSITIVE, CONTROL_POSITIVE),
            layout.entry(POSITIVE, CONTROL_NEGATIVE),
            layout.entry(NEGATIVE, CONTROL_POSITIVE),
            layout.entry(NEGATIVE, CONTROL_NEGATIVE),
        ]))
    }

    fn linear(_: &ControlledCurrentShape, _: &f64) -> bool {
        true
    }

    fn evaluate<R: Real>(
        ControlledCurrentShape(entries): &ControlledCurrentShape,
        &transconductance: &R,
        (): &mut (),
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        let [
            positive_control_positive,
            positive_control_negative,
            negative_control_positive,
            negative_control_negative,
        ] = *entries;
        let control = at.value(CONTROL_POSITIVE) - at.value(CONTROL_NEGATIVE);
        stamp.add_residual(POSITIVE, transconductance * control);
        stamp.add_residual(NEGATIVE, -transconductance * control);
        stamp.add_conductance(positive_control_positive, transconductance);
        stamp.add_conductance(positive_control_negative, -transconductance);
        stamp.add_conductance(negative_control_positive, -transconductance);
        stamp.add_conductance(negative_control_negative, transconductance);
    }
}

/// A current-controlled current source, after Modelica's: its control port is a short
/// circuit that senses the current through it.
///
/// It drives `gain` times the control current from `positive` through the source to
/// `negative`. The control current flows from `control_positive` through the port to
/// `control_negative`, which it holds at the same voltage; it is an unknown labelled
/// `control`.
#[derive(Debug, Clone, Copy)]
pub struct CurrentControlledCurrentSource;

/// Where a current-controlled current source's contributions go: through its sensing
/// port, and at the entries of each output terminal's equation at the control.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CurrentControlledCurrentSourceShape {
    sense: CurrentBranch,
    entries: [usize; 2],
}

/// The parameters of a [`CurrentControlledCurrentSource`].
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields, bound = "")]
pub struct CurrentControlledCurrentSourceParameters<R: Real = f64> {
    /// The ratio of the output current to the control current.
    pub gain: R,
}

/// Adds a sensing port, a short circuit, and returns the current through it.
fn add_sense<R: Real>(sense: &CurrentBranch, at: &Local<'_>, stamp: &mut Stamp<'_, R>) -> f64 {
    sense.add(at, stamp, R::constant(sense.voltage(at)));
    sense.current(at)
}

impl Kernel for CurrentControlledCurrentSource {
    const NAME: &'static str = "current_controlled_current_source";
    type Parameters<R: Real> = CurrentControlledCurrentSourceParameters<R>;
    type Shape = CurrentControlledCurrentSourceShape;
    type Prepared<R: Real> = R;
    type Memory = ();

    fn interface<R: Real>(_: &CurrentControlledCurrentSourceParameters<R>) -> Interface {
        Interface::new(
            CONTROLLED_TERMINALS,
            &[
                Branch::new(0, 1, BranchKind::CurrentSource),
                Branch::new(2, 3, BranchKind::VoltageSource),
            ],
        )
    }

    fn prepare<R: Real>(
        parameters: &CurrentControlledCurrentSourceParameters<R>,
        _: R,
    ) -> Result<R, ParameterError> {
        Ok(parameters.gain)
    }

    fn layout<R: Real>(
        _: &CurrentControlledCurrentSourceParameters<R>,
        _: &R,
        layout: &mut Layout,
    ) -> Result<CurrentControlledCurrentSourceShape, ParameterError> {
        let sense = CurrentBranch::new(layout, "control", [CONTROL_POSITIVE, CONTROL_NEGATIVE]);
        let control = sense.unknown();
        let entries = [
            layout.entry(POSITIVE, control),
            layout.entry(NEGATIVE, control),
        ];
        Ok(CurrentControlledCurrentSourceShape { sense, entries })
    }

    fn linear(_: &CurrentControlledCurrentSourceShape, _: &f64) -> bool {
        true
    }

    fn evaluate<R: Real>(
        shape: &CurrentControlledCurrentSourceShape,
        &gain: &R,
        (): &mut (),
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        let current = add_sense(&shape.sense, at, stamp);
        let [positive_control, negative_control] = shape.entries;
        stamp.add_residual(POSITIVE, gain * current);
        stamp.add_residual(NEGATIVE, -gain * current);
        stamp.add_conductance(positive_control, gain);
        stamp.add_conductance(negative_control, -gain);
    }
}

/// A current-controlled voltage source, after Modelica's: its control port is a short
/// circuit that senses the current through it.
///
/// It holds `v_positive − v_negative = transresistance · i_control`. The control current
/// flows from `control_positive` through the port to `control_negative`, which it holds
/// at the same voltage; it is an unknown labelled `control`. The output current, from
/// `positive` through the source to `negative`, is an unknown labelled `current`.
#[derive(Debug, Clone, Copy)]
pub struct CurrentControlledVoltageSource;

/// Where a current-controlled voltage source's contributions go: through its sensing
/// port and its output branch, and at the entry of its current's equation at the control.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SensedVoltageShape {
    sense: CurrentBranch,
    branch: CurrentBranch,
    entry: usize,
}

/// The parameters of a [`CurrentControlledVoltageSource`].
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields, bound = "")]
pub struct CurrentControlledVoltageSourceParameters<R: Real = f64> {
    /// The ratio of the output voltage to the control current, in ohms.
    pub transresistance: R,
}

impl Kernel for CurrentControlledVoltageSource {
    const NAME: &'static str = "current_controlled_voltage_source";
    type Parameters<R: Real> = CurrentControlledVoltageSourceParameters<R>;
    type Shape = SensedVoltageShape;
    type Prepared<R: Real> = R;
    type Memory = ();

    fn interface<R: Real>(_: &CurrentControlledVoltageSourceParameters<R>) -> Interface {
        Interface::new(
            CONTROLLED_TERMINALS,
            &[
                Branch::new(0, 1, BranchKind::VoltageSource),
                Branch::new(2, 3, BranchKind::VoltageSource),
            ],
        )
    }

    fn prepare<R: Real>(
        parameters: &CurrentControlledVoltageSourceParameters<R>,
        _: R,
    ) -> Result<R, ParameterError> {
        Ok(parameters.transresistance)
    }

    fn layout<R: Real>(
        _: &CurrentControlledVoltageSourceParameters<R>,
        _: &R,
        layout: &mut Layout,
    ) -> Result<SensedVoltageShape, ParameterError> {
        let sense = CurrentBranch::new(layout, "control", [CONTROL_POSITIVE, CONTROL_NEGATIVE]);
        let branch = CurrentBranch::new(layout, "current", [POSITIVE, NEGATIVE]);
        let entry = layout.entry(branch.unknown(), sense.unknown());
        Ok(SensedVoltageShape {
            sense,
            branch,
            entry,
        })
    }

    fn linear(_: &SensedVoltageShape, _: &f64) -> bool {
        true
    }

    fn evaluate<R: Real>(
        shape: &SensedVoltageShape,
        &transresistance: &R,
        (): &mut (),
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        let control = add_sense(&shape.sense, at, stamp);
        let voltage = shape.branch.voltage(at);
        shape
            .branch
            .add(at, stamp, -(transresistance * control - voltage));
        stamp.add_conductance(shape.entry, -transresistance);
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
