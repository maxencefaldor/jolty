use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::kernel::dual::Dual;
use crate::kernel::parts::CurrentBranch;
use crate::{
    Branch, BranchKind, Interface, Kernel, Layout, Lift, Local, NoiseSource, ParameterError, Real,
    Stamp, UnknownIndex,
};

/// The law by which a switch's control moves its resistance, PSpice's, as the PSpice A/D
/// Reference Guide states it for its switches `S` and `W` in "Voltage-controlled switch
/// equations": with the control's fraction of the way from its off value to its on value
/// `s`, and `m = s − ½`, the resistance's logarithm is `ln √(r_on · r_off) + L · (3/2 · m −
/// 2 · m³)`, where `L = ln(r_on / r_off)`: a cubic that reaches each resistance with zero
/// slope at each end, beyond which the resistance holds.
#[derive(Debug, Clone, Copy)]
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

    /// The conductance at `control`.
    fn conductance<const N: usize>(&self, control: Dual<R, N>) -> Dual<R, N> {
        let fraction = control.plus(-self.off).divide(self.span);
        if fraction.value.value() >= 1.0 {
            return Dual::constant(self.on_conductance);
        }
        if fraction.value.value() <= 0.0 {
            return Dual::constant(self.off_conductance);
        }
        let m = fraction - 0.5;
        let log_resistance = (m * -(m * m * 2.0 - 1.5))
            .scale(self.log_ratio)
            .plus(self.log_mean);
        (-log_resistance).exp()
    }

    /// The current through the output at `voltage` across it, of `voltage` and
    /// `control`, which sets its conductance, a law of the control alone.
    fn current(&self, [voltage, control]: [f64; 2]) -> Dual<R, 2> {
        let [voltage, control] = Dual::variables([voltage, control]);
        control.apply(|control| self.conductance(control)) * voltage
    }
}

/// A voltage-controlled switch, in PSpice's model: a resistance between `positive` and
/// `negative` that the voltage of `control_positive` over `control_negative` moves smoothly
/// from the off resistance to the on one, by PSpice's law, its logarithm a cubic of the
/// control that reaches each resistance with zero slope. The control terminals
/// draw no current.
#[derive(Debug, Clone, Copy)]
pub struct VoltageControlledSwitch;

/// A switch's numbers at its temperature: its law, and the temperature its resistance's
/// thermal noise is at.
#[derive(Debug, Clone, Copy)]
pub struct PreparedSwitch<R: Real> {
    law: Law<R>,
    temperature: f64,
}

/// Where a voltage-controlled switch's contributions go: at the entries joining each output
/// terminal to every terminal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VoltageSwitchShape([[usize; 4]; 2]);

/// The parameters of a [`VoltageControlledSwitch`], with the defaults the PSpice A/D
/// Reference Guide gives its model `VSWITCH`.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Lift)]
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

/// The local indices of a switch's terminals, in the order of [`SWITCH_TERMINALS`].
const TERMINALS: [usize; 4] = [0, 1, 2, 3];

impl Kernel for VoltageControlledSwitch {
    const NAME: &'static str = "voltage_controlled_switch";
    type Parameters<R: Real> = VoltageControlledSwitchParameters<R>;
    type Shape = VoltageSwitchShape;
    type Prepared<R: Real> = PreparedSwitch<R>;
    type Memory = ();

    fn interface<R: Real>(_: &VoltageControlledSwitchParameters<R>) -> Interface {
        Interface::new(
            SWITCH_TERMINALS,
            &[Branch::new(0, 1, BranchKind::Conductive)],
        )
    }

    fn prepare<R: Real>(
        parameters: &VoltageControlledSwitchParameters<R>,
        temperature: R,
    ) -> Result<PreparedSwitch<R>, ParameterError> {
        let &VoltageControlledSwitchParameters {
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
        Ok(PreparedSwitch {
            law,
            temperature: temperature.value(),
        })
    }

    fn layout<R: Real>(
        _: &VoltageControlledSwitchParameters<R>,
        _: &PreparedSwitch<R>,
        layout: &mut Layout,
    ) -> Result<VoltageSwitchShape, ParameterError> {
        let [positive, negative, ..] = TERMINALS;
        Ok(VoltageSwitchShape([positive, negative].map(|row| {
            TERMINALS.map(|column| layout.entry(row, column))
        })))
    }

    fn evaluate<R: Real>(
        VoltageSwitchShape(entries): &VoltageSwitchShape,
        prepared: &PreparedSwitch<R>,
        (): &mut (),
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        let [positive, negative, control_positive, control_negative] = TERMINALS;
        let voltage = at.value(positive) - at.value(negative);
        let control = at.value(control_positive) - at.value(control_negative);
        let current = prepared.law.current([voltage, control]);
        stamp.add_residual(positive, current.value);
        stamp.add_residual(negative, -current.value);
        // With respect to positive, negative, control_positive and control_negative.
        let [by_voltage, by_control] = current.gradient;
        let derivatives = [by_voltage, -by_voltage, by_control, -by_control];
        for (row, sign) in entries.iter().zip([1.0, -1.0]) {
            for (&entry, derivative) in row.iter().zip(derivatives) {
                stamp.add_conductance(entry, derivative * sign);
            }
        }
    }

    /// The thermal noise of the resistance the control sets.
    fn noise<R: Real>(
        _: &VoltageSwitchShape,
        prepared: &PreparedSwitch<R>,
        at: &Local<'_>,
        sources: &mut Vec<NoiseSource>,
    ) {
        let [positive, negative, control_positive, control_negative] = TERMINALS;
        let control = at.value(control_positive) - at.value(control_negative);
        let conductance = prepared
            .law
            .conductance(Dual::<R, 0>::number(control))
            .value;
        sources.push(NoiseSource::thermal(
            "thermal",
            [
                Some(UnknownIndex::new(positive)),
                Some(UnknownIndex::new(negative)),
            ],
            conductance.value(),
            prepared.temperature,
        ));
    }
}

/// A current-controlled switch, in PSpice's model: a resistance between `positive` and
/// `negative` that the current through its control port moves smoothly from the off
/// resistance to the on one, by PSpice's law, its logarithm a cubic of the control that
/// reaches each resistance with zero slope.
///
/// The control port is a short circuit that senses the current through it, as a
/// current-controlled source's is: the current flows from `control_positive` through the
/// port to `control_negative`, which it holds at the same voltage, and is an unknown
/// labelled `control`.
#[derive(Debug, Clone, Copy)]
pub struct CurrentControlledSwitch;

/// Where a current-controlled switch's contributions go: through its sensing port, and at
/// the entries joining each output terminal to each output terminal and to the control.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CurrentSwitchShape {
    sense: CurrentBranch,
    entries: [[usize; 3]; 2],
}

/// The parameters of a [`CurrentControlledSwitch`], with the defaults the PSpice A/D
/// Reference Guide gives its model `ISWITCH`.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Lift)]
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

impl Kernel for CurrentControlledSwitch {
    const NAME: &'static str = "current_controlled_switch";
    type Parameters<R: Real> = CurrentControlledSwitchParameters<R>;
    type Shape = CurrentSwitchShape;
    type Prepared<R: Real> = PreparedSwitch<R>;
    type Memory = ();

    fn interface<R: Real>(_: &CurrentControlledSwitchParameters<R>) -> Interface {
        Interface::new(
            SWITCH_TERMINALS,
            &[
                Branch::new(0, 1, BranchKind::Conductive),
                Branch::new(2, 3, BranchKind::VoltageSource),
            ],
        )
    }

    fn prepare<R: Real>(
        parameters: &CurrentControlledSwitchParameters<R>,
        temperature: R,
    ) -> Result<PreparedSwitch<R>, ParameterError> {
        let &CurrentControlledSwitchParameters {
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
        Ok(PreparedSwitch {
            law,
            temperature: temperature.value(),
        })
    }

    fn layout<R: Real>(
        _: &CurrentControlledSwitchParameters<R>,
        _: &PreparedSwitch<R>,
        layout: &mut Layout,
    ) -> Result<CurrentSwitchShape, ParameterError> {
        let [positive, negative, control_positive, control_negative] = TERMINALS;
        let sense = CurrentBranch::new(layout, "control", [control_positive, control_negative]);
        let columns = [positive, negative, sense.unknown()];
        let entries =
            [positive, negative].map(|row| columns.map(|column| layout.entry(row, column)));
        Ok(CurrentSwitchShape { sense, entries })
    }

    fn evaluate<R: Real>(
        shape: &CurrentSwitchShape,
        prepared: &PreparedSwitch<R>,
        (): &mut (),
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        // The port is a short circuit, its current the control.
        let sense = &shape.sense;
        sense.add(at, stamp, R::constant(sense.voltage(at)));
        let control = sense.current(at);
        let [positive, negative, ..] = TERMINALS;
        let voltage = at.value(positive) - at.value(negative);
        let current = prepared.law.current([voltage, control]);
        stamp.add_residual(positive, current.value);
        stamp.add_residual(negative, -current.value);
        // With respect to positive, negative and the control.
        let [by_voltage, by_control] = current.gradient;
        let derivatives = [by_voltage, -by_voltage, by_control];
        for (row, sign) in shape.entries.iter().zip([1.0, -1.0]) {
            for (&entry, derivative) in row.iter().zip(derivatives) {
                stamp.add_conductance(entry, derivative * sign);
            }
        }
    }

    /// The thermal noise of the resistance the control sets.
    fn noise<R: Real>(
        shape: &CurrentSwitchShape,
        prepared: &PreparedSwitch<R>,
        at: &Local<'_>,
        sources: &mut Vec<NoiseSource>,
    ) {
        let control = Dual::<R, 0>::number(shape.sense.current(at));
        let conductance = prepared.law.conductance(control).value;
        let [positive, negative, ..] = TERMINALS;
        sources.push(NoiseSource::thermal(
            "thermal",
            [
                Some(UnknownIndex::new(positive)),
                Some(UnknownIndex::new(negative)),
            ],
            conductance.value(),
            prepared.temperature,
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
