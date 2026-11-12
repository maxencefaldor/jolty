use jolty_numerics::Entry;
use serde::Deserialize;

use crate::{
    Binding, Branch, BranchKind, Device, Input, Interface, NativeDevice, NativeModel, Node,
    NoiseSource, Output, ParameterError, Real,
};

/// A voltage-controlled switch, in PSpice's model: a resistance between `positive` and
/// `negative` that the voltage of `control_positive` over `control_negative` moves smoothly
/// from the off resistance to the on one.
///
/// With the control's fraction of the way from the off voltage to the on voltage `s`, and
/// `m = s − ½`, the resistance's logarithm is `ln √(r_on · r_off) + L · (3/2 · m − 2 · m³)`,
/// where `L = ln(r_on / r_off)`: a cubic that reaches each resistance with zero slope at
/// each end, beyond which the resistance holds. The control terminals draw no current.
#[derive(Debug)]
pub struct VoltageControlledSwitch<R: Real = f64> {
    nodes: [Node; 4],
    off_voltage: R,
    /// The on voltage less the off voltage.
    span: R,
    on_conductance: R,
    off_conductance: R,
    /// `ln √(r_on · r_off)`.
    log_mean: R,
    /// `ln(r_on / r_off)`.
    log_ratio: R,
    /// The entries joining each output terminal to every terminal.
    entries: [[Option<Entry>; 4]; 2],
    temperature: f64,
}

/// The parameters of a [`VoltageControlledSwitch`], with PSpice's defaults.
#[derive(Debug, Clone, Deserialize)]
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

impl NativeModel for VoltageControlledSwitch {
    type Device<R: Real> = VoltageControlledSwitch<R>;
}

impl<R: Real> NativeDevice<R> for VoltageControlledSwitch<R> {
    const NAME: &'static str = "voltage_controlled_switch";
    type Parameters = VoltageControlledSwitchParameters<R>;

    fn interface(_: &VoltageControlledSwitchParameters<R>) -> Interface {
        Interface::new(
            &[
                "positive",
                "negative",
                "control_positive",
                "control_negative",
            ],
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
        ParameterError::positive("on_resistance", on_resistance)?;
        ParameterError::positive("off_resistance", off_resistance)?;
        let span = on_voltage - off_voltage;
        if span.value() == 0.0 {
            let on_voltage = on_voltage.value();
            return Err(ParameterError(format!(
                "on_voltage and off_voltage must differ, not both be {on_voltage:?}"
            )));
        }
        let nodes = [0, 1, 2, 3].map(|index| binding.terminal(index));
        let entries =
            [nodes[0], nodes[1]].map(|row| nodes.map(|column| binding.entry(row, column)));
        Ok(Self {
            nodes,
            off_voltage,
            span,
            on_conductance: on_resistance.recip(),
            off_conductance: off_resistance.recip(),
            log_mean: (on_resistance * off_resistance).sqrt().ln(),
            log_ratio: (on_resistance / off_resistance).ln(),
            entries,
            temperature: binding.temperature(),
        })
    }
}

impl<R: Real> VoltageControlledSwitch<R> {
    /// The conductance at `control`, and its derivative with respect to it.
    fn conductance(&self, control: f64) -> (R, R) {
        let fraction = -(self.off_voltage - control) / self.span;
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

impl<R: Real> Device<R> for VoltageControlledSwitch<R> {
    fn evaluate(&mut self, input: &Input<'_>, output: &mut Output<'_, R>) {
        let [positive, negative, control_positive, control_negative] = self.nodes;
        let voltage = input.value(positive) - input.value(negative);
        let control = input.value(control_positive) - input.value(control_negative);
        let (conductance, slope) = self.conductance(control);
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
        let (conductance, _) = self.conductance(control);
        sources.push(NoiseSource::thermal(
            "thermal",
            [positive, negative],
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
