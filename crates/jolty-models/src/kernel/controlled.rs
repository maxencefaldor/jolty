use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::dimension::{CONDUCTANCE, RESISTANCE};
use crate::kernel::parts::CurrentBranch;
use crate::{
    Branch, BranchKind, Interface, Kernel, Layout, Lift, Local, ParameterError, ParameterKind,
    ParameterMetadata, Real, Stamp,
};

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

/// Expands the shared [`Kernel`] items of a linear controlled source inside its `impl Kernel`.
macro_rules! controlled_common {
    (
        name: $name:literal,
        parameters: $parameters:ident,
        field: $field:ident,
        shape: $shape:ty,
        branches: $branches:expr,
    ) => {
        const NAME: &'static str = $name;
        type Parameters<R: Real> = $parameters<R>;
        type Shape = $shape;
        type Prepared<R: Real> = R;
        type Memory = ();

        fn interface<R: Real>(_: &$parameters<R>) -> Interface {
            Interface::new(CONTROLLED_TERMINALS, $branches)
        }

        fn prepare<R: Real>(parameters: &$parameters<R>, _: R) -> Result<R, ParameterError> {
            Ok(parameters.$field)
        }

        fn linear(_: &$shape, _: &f64) -> bool {
            true
        }

        fn parameter_kinds() -> &'static [(&'static str, ParameterKind)] {
            $parameters::<f64>::PARAMETER_KINDS
        }
    };
}

/// Linear voltage-controlled voltage source.
///
/// Constrains `v_positive - v_negative = gain · (v_control_positive - v_control_negative)`.
/// Output current flowing from `positive` to `negative` is an unknown labeled `current`,
/// while control terminals draw zero current.
#[derive(Debug, Clone, Copy)]
pub struct VoltageControlledVoltageSource;

/// Layout shape for [`VoltageControlledVoltageSource`] holding the output branch and
/// control-voltage entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VoltageControlledVoltageSourceShape {
    branch: CurrentBranch,
    entries: [usize; 2],
}

/// The parameters of a [`VoltageControlledVoltageSource`].
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Lift, ParameterMetadata)]
#[serde(deny_unknown_fields)]
#[schemars(bound = "R: Serialize + JsonSchema")]
pub struct VoltageControlledVoltageSourceParameters<R: Real = f64> {
    /// The ratio of the output voltage to the control voltage.
    pub gain: R,
}

impl Kernel for VoltageControlledVoltageSource {
    controlled_common! {
        name: "voltage_controlled_voltage_source",
        parameters: VoltageControlledVoltageSourceParameters,
        field: gain,
        shape: VoltageControlledVoltageSourceShape,
        branches: &[Branch::new(0, 1, BranchKind::VoltageSource)],
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

    fn evaluate<R: Real>(
        shape: &VoltageControlledVoltageSourceShape,
        &gain: &R,
        (): &mut (),
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        let control = at.unknown::<R>(CONTROL_POSITIVE) - at.unknown::<R>(CONTROL_NEGATIVE);
        let voltage = shape.branch.voltage::<R>(at);
        shape.branch.add(at, stamp, -(gain * control - voltage));
        let [current_control_positive, current_control_negative] = shape.entries;
        stamp.add_conductance(current_control_positive, -gain);
        stamp.add_conductance(current_control_negative, gain);
    }
}

/// Linear voltage-controlled current source.
///
/// Drives `transconductance · (v_control_positive - v_control_negative)` from `positive`
/// through the source to `negative`, drawing zero current at the control terminals.
#[derive(Debug, Clone, Copy)]
pub struct VoltageControlledCurrentSource;

/// Layout shape for [`VoltageControlledCurrentSource`] indexing the four output-to-control
/// matrix entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VoltageControlledCurrentSourceShape([usize; 4]);

/// The parameters of a [`VoltageControlledCurrentSource`].
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Lift, ParameterMetadata)]
#[serde(deny_unknown_fields)]
#[schemars(bound = "R: Serialize + JsonSchema")]
pub struct VoltageControlledCurrentSourceParameters<R: Real = f64> {
    /// The ratio of the output current to the control voltage, in siemens.
    #[dimension(CONDUCTANCE)]
    pub transconductance: R,
}

impl Kernel for VoltageControlledCurrentSource {
    controlled_common! {
        name: "voltage_controlled_current_source",
        parameters: VoltageControlledCurrentSourceParameters,
        field: transconductance,
        shape: VoltageControlledCurrentSourceShape,
        branches: &[Branch::new(0, 1, BranchKind::CurrentSource)],
    }

    fn layout<R: Real>(
        _: &VoltageControlledCurrentSourceParameters<R>,
        _: &R,
        layout: &mut Layout,
    ) -> Result<VoltageControlledCurrentSourceShape, ParameterError> {
        Ok(VoltageControlledCurrentSourceShape([
            layout.entry(POSITIVE, CONTROL_POSITIVE),
            layout.entry(POSITIVE, CONTROL_NEGATIVE),
            layout.entry(NEGATIVE, CONTROL_POSITIVE),
            layout.entry(NEGATIVE, CONTROL_NEGATIVE),
        ]))
    }

    fn evaluate<R: Real>(
        VoltageControlledCurrentSourceShape(entries): &VoltageControlledCurrentSourceShape,
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
        let control = at.unknown::<R>(CONTROL_POSITIVE) - at.unknown::<R>(CONTROL_NEGATIVE);
        stamp.add_residual(POSITIVE, transconductance * control);
        stamp.add_residual(NEGATIVE, -transconductance * control);
        stamp.add_conductance(positive_control_positive, transconductance);
        stamp.add_conductance(positive_control_negative, -transconductance);
        stamp.add_conductance(negative_control_positive, -transconductance);
        stamp.add_conductance(negative_control_negative, transconductance);
    }
}

/// Linear current-controlled current source modeled with a zero-volt sensing port.
///
/// Drives `gain` times the control current from `positive` through the source to `negative`.
/// The control port shorts `control_positive` to `control_negative` with a branch-current
/// unknown labeled `control`.
#[derive(Debug, Clone, Copy)]
pub struct CurrentControlledCurrentSource;

/// Layout shape for [`CurrentControlledCurrentSource`] holding the sensing branch and
/// output-to-control matrix entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CurrentControlledCurrentSourceShape {
    sense: CurrentBranch,
    entries: [usize; 2],
}

/// The parameters of a [`CurrentControlledCurrentSource`].
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Lift, ParameterMetadata)]
#[serde(deny_unknown_fields)]
#[schemars(bound = "R: Serialize + JsonSchema")]
pub struct CurrentControlledCurrentSourceParameters<R: Real = f64> {
    /// The ratio of the output current to the control current.
    pub gain: R,
}

/// Adds a sensing port, a short circuit, and returns the current through it.
fn add_sense<R: Real>(sense: &CurrentBranch, at: &Local<'_>, stamp: &mut Stamp<'_, R>) -> R {
    sense.add(at, stamp, sense.voltage(at));
    sense.current(at)
}

impl Kernel for CurrentControlledCurrentSource {
    controlled_common! {
        name: "current_controlled_current_source",
        parameters: CurrentControlledCurrentSourceParameters,
        field: gain,
        shape: CurrentControlledCurrentSourceShape,
        branches: &[
            Branch::new(0, 1, BranchKind::CurrentSource),
            Branch::new(2, 3, BranchKind::VoltageSource),
        ],
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

/// Linear current-controlled voltage source modeled with a zero-volt sensing port.
///
/// Constrains `v_positive - v_negative = transresistance · i_control`. The control port
/// shorts `control_positive` to `control_negative` (unknown `control`), and output current
/// flowing from `positive` to `negative` is an unknown labeled `current`.
#[derive(Debug, Clone, Copy)]
pub struct CurrentControlledVoltageSource;

/// Layout shape for [`CurrentControlledVoltageSource`] holding the sensing port, output
/// branch, and transresistance matrix entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CurrentControlledVoltageSourceShape {
    sense: CurrentBranch,
    branch: CurrentBranch,
    entry: usize,
}

/// The parameters of a [`CurrentControlledVoltageSource`].
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Lift, ParameterMetadata)]
#[serde(deny_unknown_fields)]
#[schemars(bound = "R: Serialize + JsonSchema")]
pub struct CurrentControlledVoltageSourceParameters<R: Real = f64> {
    /// The ratio of the output voltage to the control current, in ohms.
    #[dimension(RESISTANCE)]
    pub transresistance: R,
}

impl Kernel for CurrentControlledVoltageSource {
    controlled_common! {
        name: "current_controlled_voltage_source",
        parameters: CurrentControlledVoltageSourceParameters,
        field: transresistance,
        shape: CurrentControlledVoltageSourceShape,
        branches: &[
            Branch::new(0, 1, BranchKind::VoltageSource),
            Branch::new(2, 3, BranchKind::VoltageSource),
        ],
    }

    fn layout<R: Real>(
        _: &CurrentControlledVoltageSourceParameters<R>,
        _: &R,
        layout: &mut Layout,
    ) -> Result<CurrentControlledVoltageSourceShape, ParameterError> {
        let sense = CurrentBranch::new(layout, "control", [CONTROL_POSITIVE, CONTROL_NEGATIVE]);
        let branch = CurrentBranch::new(layout, "current", [POSITIVE, NEGATIVE]);
        let entry = layout.entry(branch.unknown(), sense.unknown());
        Ok(CurrentControlledVoltageSourceShape {
            sense,
            branch,
            entry,
        })
    }

    fn evaluate<R: Real>(
        shape: &CurrentControlledVoltageSourceShape,
        &transresistance: &R,
        (): &mut (),
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        let control = add_sense(&shape.sense, at, stamp);
        let voltage = shape.branch.voltage::<R>(at);
        shape
            .branch
            .add(at, stamp, -(transresistance * control - voltage));
        stamp.add_conductance(shape.entry, -transresistance);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
