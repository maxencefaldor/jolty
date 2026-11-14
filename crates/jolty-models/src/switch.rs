use jolty_numerics::Entry;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::stamp::CurrentBranch;
use crate::{
    Binding, Branch, BranchKind, Device, Input, Interface, NativeDevice, NativeModel, Node,
    NoiseSource, Output, ParameterError, Real,
};

/// The law by which a switch's control moves its resistance, PSpice's, as the PSpice A/D
/// Reference Guide states it for its switches `S` and `W` in "Voltage-controlled switch
/// equations": with the control's fraction of the way from its off value to its on value
/// `s`, and `m = s − ½`, the resistance's logarithm is `ln √(r_on · r_off) + L · (3/2 · m −
/// 2 · m³)`, where `L = ln(r_on / r_off)`: a cubic that reaches each resistance with zero
/// slope at each end, beyond which the resistance holds.
#[derive(Debug)]
struct Law<R: Real> {
    off: R,
    /// The on value less the off value.
    span: R,
    on_conductance: R,
    off_conductance: R,
    /// `ln √(r_on · r_off)`.
    log_mean: R,
    /// `ln(r_on / r_off)`.
    log_ratio: R,
}

impl<R: Real> Law<R> {
    /// The law between `on_resistance` at `on` and `off_resistance` at `off`, the
    /// control named `control`, such as `voltage`, in errors.
    fn new(
        [on_resistance, off_resistance]: [R; 2],
        [on, off]: [R; 2],
        control: &str,
    ) -> Result<Self, ParameterError> {
        ParameterError::positive("on_resistance", on_resistance)?;
        ParameterError::positive("off_resistance", off_resistance)?;
        let span = on - off;
        if span.value() == 0.0 {
            let on = on.value();
            return Err(ParameterError(format!(
                "on_{control} and off_{control} must differ, not both be {on:?}"
            )));
        }
        Ok(Self {
            off,
            span,
            on_conductance: on_resistance.recip(),
            off_conductance: off_resistance.recip(),
            log_mean: (on_resistance * off_resistance).sqrt().ln(),
            log_ratio: (on_resistance / off_resistance).ln(),
        })
    }

    /// The conductance at `control`, and its derivative with respect to it.
    fn conductance(&self, control: f64) -> (R, R) {
        let fraction = -(self.off - control) / self.span;
        if fraction.value() >= 1.0 {
            return (self.on_conductance, R::constant(0.0));
        }
        if fraction.value() <= 0.0 {
            return (self.off_conductance, R::constant(0.0));
        }
        let m = fraction - 0.5;
        let log_resistance = self.log_mean + self.log_ratio * m * -(m * 2.0 * m - 1.5);
        let conductance = (-log_resistance).exp();
        let slope = self.log_ratio * -(m * 6.0 * m - 1.5) / self.span;
        (conductance, -conductance * slope)
    }
}

/// A voltage-controlled switch, in PSpice's model: a resistance between `positive` and
/// `negative` that the voltage of `control_positive` over `control_negative` moves smoothly
/// from the off resistance to the on one, by [PSpice's law](Law). The control terminals
/// draw no current.
#[derive(Debug)]
pub struct VoltageControlledSwitch<R: Real = f64> {
    nodes: [Node; 4],
    law: Law<R>,
    /// The entries joining each output terminal to every terminal.
    entries: [[Option<Entry>; 4]; 2],
    temperature: f64,
}

/// The parameters of a [`VoltageControlledSwitch`], with the defaults the PSpice A/D
/// Reference Guide gives its model `VSWITCH`.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[serde(default, deny_unknown_fields, bound = "")]
pub struct VoltageControlledSwitchParameters<R: Real = f64> {
    /// The resistance with the control at or past the on voltage, in ohms; default 1.
    pub on_resistance: R,
    /// The resistance with the control at or past the off voltage, in ohms; default 1e6.
    pub off_resistance: R,
    /// The control voltage at which the switch is on, in volts; default 1.
    pub on_voltage: R,
    /// The control voltage at which the switch is off, in volts; default 0.
    pub off_voltage: R,
}

impl<R: Real> Default for VoltageControlledSwitchParameters<R> {
    fn default() -> Self {
        Self {
            on_resistance: R::constant(1.0),
            off_resistance: R::constant(1e6),
            on_voltage: R::constant(1.0),
            off_voltage: R::constant(0.0),
        }
    }
}

/// The terminals of a switch: its output's, then its control's.
const SWITCH_TERMINALS: &[&str] = &[
    "positive",
    "negative",
    "control_positive",
    "control_negative",
];

impl NativeModel for VoltageControlledSwitch {
    type Device<R: Real> = VoltageControlledSwitch<R>;
}

impl<R: Real> NativeDevice<R> for VoltageControlledSwitch<R> {
    const NAME: &'static str = "voltage_controlled_switch";
    type Parameters = VoltageControlledSwitchParameters<R>;

    fn interface(_: &VoltageControlledSwitchParameters<R>) -> Interface {
        Interface::new(
            SWITCH_TERMINALS,
            &[Branch::new(0, 1, BranchKind::Conductive)],
        )
    }

    fn new(
        parameters: VoltageControlledSwitchParameters<R>,
        binding: &mut Binding<'_>,
    ) -> Result<Self, ParameterError> {
        let VoltageControlledSwitchParameters {
            on_resistance,
            off_resistance,
            on_voltage,
            off_voltage,
        } = parameters;
        let law = Law::new(
            [on_resistance, off_resistance],
            [on_voltage, off_voltage],
            "voltage",
        )?;
        let nodes = [0, 1, 2, 3].map(|index| binding.terminal(index));
        let entries =
            [nodes[0], nodes[1]].map(|row| nodes.map(|column| binding.entry(row, column)));
        Ok(Self {
            nodes,
            law,
            entries,
            temperature: binding.temperature(),
        })
    }
}

impl<R: Real> Device<R> for VoltageControlledSwitch<R> {
    fn evaluate(&mut self, input: &Input<'_>, output: &mut Output<'_, R>) {
        let [positive, negative, control_positive, control_negative] = self.nodes;
        let voltage = input.value(positive) - input.value(negative);
        let control = input.value(control_positive) - input.value(control_negative);
        let (conductance, slope) = self.law.conductance(control);
        let current = conductance * voltage;
        output.add_residual(positive, current);
        output.add_residual(negative, -current);
        // With respect to positive, negative, control_positive and control_negative.
        let derivatives = [conductance, -conductance, slope * voltage, -slope * voltage];
        for (row, sign) in self.entries.iter().zip([1.0, -1.0]) {
            for (&entry, derivative) in row.iter().zip(derivatives) {
                output.add_conductance(entry, derivative * sign);
            }
        }
    }

    /// The thermal noise of the resistance the control sets.
    fn noise(&self, input: &Input<'_>, sources: &mut Vec<NoiseSource>) {
        let [positive, negative, control_positive, control_negative] = self.nodes;
        let control = input.value(control_positive) - input.value(control_negative);
        let (conductance, _) = self.law.conductance(control);
        sources.push(NoiseSource::thermal(
            "thermal",
            [positive, negative],
            conductance.value(),
            self.temperature,
        ));
    }
}

/// A current-controlled switch, in PSpice's model: a resistance between `positive` and
/// `negative` that the current through its control port moves smoothly from the off
/// resistance to the on one, by [PSpice's law](Law).
///
/// The control port is a short circuit that senses the current through it, as a
/// current-controlled source's is: the current flows from `control_positive` through the
/// port to `control_negative`, which it holds at the same voltage, and is an unknown
/// labelled `control`.
#[derive(Debug)]
pub struct CurrentControlledSwitch<R: Real = f64> {
    output: [Node; 2],
    sense: CurrentBranch,
    law: Law<R>,
    /// The entries joining each output terminal to each output terminal and to the
    /// control.
    entries: [[Option<Entry>; 3]; 2],
    temperature: f64,
}

/// The parameters of a [`CurrentControlledSwitch`], with the defaults the PSpice A/D
/// Reference Guide gives its model `ISWITCH`.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[serde(default, deny_unknown_fields, bound = "")]
pub struct CurrentControlledSwitchParameters<R: Real = f64> {
    /// The resistance with the control at or past the on current, in ohms; default 1.
    pub on_resistance: R,
    /// The resistance with the control at or past the off current, in ohms; default 1e6.
    pub off_resistance: R,
    /// The control current at which the switch is on, in amperes; default 1 mA.
    pub on_current: R,
    /// The control current at which the switch is off, in amperes; default 0.
    pub off_current: R,
}

impl<R: Real> Default for CurrentControlledSwitchParameters<R> {
    fn default() -> Self {
        Self {
            on_resistance: R::constant(1.0),
            off_resistance: R::constant(1e6),
            on_current: R::constant(1e-3),
            off_current: R::constant(0.0),
        }
    }
}

impl NativeModel for CurrentControlledSwitch {
    type Device<R: Real> = CurrentControlledSwitch<R>;
}

impl<R: Real> NativeDevice<R> for CurrentControlledSwitch<R> {
    const NAME: &'static str = "current_controlled_switch";
    type Parameters = CurrentControlledSwitchParameters<R>;

    fn interface(_: &CurrentControlledSwitchParameters<R>) -> Interface {
        Interface::new(
            SWITCH_TERMINALS,
            &[
                Branch::new(0, 1, BranchKind::Conductive),
                Branch::new(2, 3, BranchKind::VoltageSource),
            ],
        )
    }

    fn new(
        parameters: CurrentControlledSwitchParameters<R>,
        binding: &mut Binding<'_>,
    ) -> Result<Self, ParameterError> {
        let CurrentControlledSwitchParameters {
            on_resistance,
            off_resistance,
            on_current,
            off_current,
        } = parameters;
        let law = Law::new(
            [on_resistance, off_resistance],
            [on_current, off_current],
            "current",
        )?;
        let [positive, negative, control_positive, control_negative] =
            [0, 1, 2, 3].map(|index| binding.terminal(index));
        let sense = CurrentBranch::new(binding, "control", [control_positive, control_negative]);
        let columns = [positive, negative, Some(sense.unknown())];
        let entries =
            [positive, negative].map(|row| columns.map(|column| binding.entry(row, column)));
        Ok(Self {
            output: [positive, negative],
            sense,
            law,
            entries,
            temperature: binding.temperature(),
        })
    }
}

impl<R: Real> Device<R> for CurrentControlledSwitch<R> {
    fn evaluate(&mut self, input: &Input<'_>, output: &mut Output<'_, R>) {
        // The port is a short circuit, its current the control.
        self.sense
            .add(input, output, R::constant(self.sense.voltage(input)));
        let control = self.sense.current(input);
        let [positive, negative] = self.output;
        let voltage = input.value(positive) - input.value(negative);
        let (conductance, slope) = self.law.conductance(control);
        let current = conductance * voltage;
        output.add_residual(positive, current);
        output.add_residual(negative, -current);
        // With respect to positive, negative and the control.
        let derivatives = [conductance, -conductance, slope * voltage];
        for (row, sign) in self.entries.iter().zip([1.0, -1.0]) {
            for (&entry, derivative) in row.iter().zip(derivatives) {
                output.add_conductance(entry, derivative * sign);
            }
        }
    }

    /// The thermal noise of the resistance the control sets.
    fn noise(&self, input: &Input<'_>, sources: &mut Vec<NoiseSource>) {
        let (conductance, _) = self.law.conductance(self.sense.current(input));
        sources.push(NoiseSource::thermal(
            "thermal",
            self.output,
            conductance.value(),
            self.temperature,
        ));
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::testing::{
        Bench, assert_close, assert_jacobians_match, assert_parameter_gradients_match, source,
        thermal,
    };

    const SWITCH: &str = r#"{"on_resistance": 10, "off_resistance": 1e6,
        "on_voltage": 2, "off_voltage": 1}"#;

    /// The switch's resistance with 1 V across it and the control at `control`.
    fn resistance(parameters: &str, control: f64) -> f64 {
        1.0 / Bench::new::<VoltageControlledSwitch>(parameters)
            .evaluate(&[1.0, 0.0, control, 0.0])
            .residual[0]
    }

    #[test]
    fn moves_from_off_to_on_through_the_geometric_mean() {
        let close = |a: f64, b: f64| (a - b).abs() < 1e-9 * b;
        assert!(close(resistance(SWITCH, 0.0), 1e6));
        assert!(close(resistance(SWITCH, 1.0), 1e6));
        assert!(close(resistance(SWITCH, 1.5), 1e7_f64.sqrt()));
        assert!(close(resistance(SWITCH, 2.0), 10.0));
        assert!(close(resistance(SWITCH, 3.0), 10.0));
        // Swapping the voltages reverses the switch.
        let reversed = r#"{"on_resistance": 10, "off_resistance": 1e6,
            "on_voltage": 1, "off_voltage": 2}"#;
        assert!(close(resistance(reversed, 0.0), 10.0));
        assert!(close(resistance(reversed, 1.25), resistance(SWITCH, 1.75)));
    }

    #[test]
    fn parameter_gradients_are_exact() {
        for control in [0.5, 1.3, 2.5] {
            assert_parameter_gradients_match::<VoltageControlledSwitch>(
                SWITCH,
                &[1.0, -0.5, control, 0.0],
            );
        }
    }

    #[test]
    fn rejects_equal_voltages() {
        let error =
            Bench::try_new::<VoltageControlledSwitch>(r#"{"on_voltage": 1, "off_voltage": 1}"#)
                .expect_err("equal voltages");
        assert_eq!(
            error.0,
            "on_voltage and off_voltage must differ, not both be 1.0"
        );
    }

    const CURRENT_SWITCH: &str = r#"{"on_resistance": 10, "off_resistance": 1e6,
        "on_current": 2e-3, "off_current": 1e-3}"#;

    #[test]
    fn a_current_moves_a_current_controlled_switch_as_a_voltage_moves_its_twin() {
        let mut bench = Bench::new::<CurrentControlledSwitch>(CURRENT_SWITCH);
        assert_eq!(bench.unknowns(), ["n0", "n1", "n2", "n3", "device:control"]);
        for control in [0.0, 1.25, 1.5, 1.9, 3.0] {
            let output = bench.evaluate(&[1.0, 0.0, 0.2, 0.1, control * 1e-3]);
            let twin = resistance(SWITCH, control);
            // The port carries its current, and holds its terminals together.
            assert_close(
                &output.residual,
                &[
                    1.0 / twin,
                    -1.0 / twin,
                    control * 1e-3,
                    -control * 1e-3,
                    0.1,
                ],
            );
            let x = [1.0, -0.5, 0.2, 0.1, control * 1e-3];
            assert_jacobians_match::<CurrentControlledSwitch>(CURRENT_SWITCH, &x);
            assert_parameter_gradients_match::<CurrentControlledSwitch>(CURRENT_SWITCH, &x);
        }
        let error = Bench::try_new::<CurrentControlledSwitch>(r#"{"on_current": 0}"#)
            .expect_err("equal currents");
        assert_eq!(
            error.0,
            "on_current and off_current must differ, not both be 0.0"
        );
        let sources = bench.noise(&[1.0, 0.0, 0.0, 0.0, 3e-3]);
        assert_close(&[source(&sources, "thermal").density(1.0)], &[thermal(0.1)]);
    }

    proptest! {
        #[test]
        fn jacobians_match_finite_differences(
            voltage in -5.0..5.0f64,
            control in 0.0..3.0f64,
        ) {
            assert_jacobians_match::<VoltageControlledSwitch>(SWITCH, &[voltage, 0.0, control, 0.0]);
        }
    }

    #[test]
    fn its_resistance_makes_thermal_noise() {
        let bench = Bench::new::<VoltageControlledSwitch>(SWITCH);
        for (control, conductance) in [(3.0, 0.1), (0.0, 1e-6)] {
            let sources = bench.noise(&[1.0, 0.0, control, 0.0]);
            assert_close(
                &[source(&sources, "thermal").density(1.0)],
                &[thermal(conductance)],
            );
        }
    }
}
