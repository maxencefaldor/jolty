use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::dimension::{CURRENT, RESISTANCE, VOLTAGE};
use crate::kernel::dual::Dual;
use crate::kernel::parts::CurrentBranch;
use crate::{
    Branch, BranchKind, Interface, Kernel, Layout, Lift, Local, NoiseSource, ParameterError,
    ParameterKind, ParameterMetadata, Real, Stamp, UnknownIndex,
};

/// PSpice smooth resistance transition law for switches `S` and `W` (PSpice A/D Reference
/// Guide, "Voltage-controlled switch equations").
///
/// Let `s` be the normalized control fraction from off to on, `m = s - ½`, and
/// `L = ln(r_on / r_off)`. Between thresholds, log-resistance follows the cubic
/// `ln √(r_on · r_off) + L · (3/2 · m - 2 · m³)`, reaching both endpoints with zero slope
/// and clamping outside `[0, 1]`.
#[derive(Debug, Clone, Copy)]
struct Law<R: Real> {
    off: R,
    /// Control span `on - off`.
    span: R,
    on_conductance: R,
    off_conductance: R,
    /// Geometric mean log-resistance `ln √(r_on · r_off)`.
    log_mean: R,
    /// Log resistance ratio `ln(r_on / r_off)`.
    log_ratio: R,
}

impl<R: Real> Law<R> {
    /// Constructs a switch law transitioning from `off_resistance` at `off` to
    /// `on_resistance` at `on`, using `control` (such as `"voltage"`) in error messages.
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
                "on_{control} and off_{control} differ, not both {on:?}"
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

    /// Evaluates switch conductance at `control`.
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

    /// Evaluates output branch current and partial derivatives with respect to output
    /// `voltage` and `control`.
    fn current(&self, [voltage, control]: [R; 2]) -> Dual<R, 2> {
        let [voltage, control] = Dual::variables([voltage, control]);
        control.apply(|control| self.conductance(control)) * voltage
    }
}

/// Voltage-controlled switch implementing the PSpice `VSWITCH` model.
///
/// Control voltage `v(control_positive) - v(control_negative)` smoothly interpolates
/// log-resistance between `positive` and `negative` along a zero-slope cubic from off to
/// on. The control terminals draw zero current.
#[derive(Debug, Clone, Copy)]
pub struct VoltageControlledSwitch;

/// Prepared switch parameters pairing the resistance transition law with ambient noise
/// temperature.
#[derive(Debug, Clone, Copy)]
pub struct PreparedSwitch<R: Real> {
    law: Law<R>,
    temperature: f64,
}

impl<R: Real> PreparedSwitch<R> {
    /// Evaluates the switch's current at `control` and stamps its output residual and
    /// `derivatives` into `entries`.
    fn stamp<const N: usize>(
        &self,
        entries: &[[usize; N]; 2],
        control: R,
        at: &Local<'_>,
        derivatives: impl FnOnce(R, R) -> [R; N],
        stamp: &mut Stamp<'_, R>,
    ) {
        let [positive, negative, ..] = TERMINALS;
        let voltage = at.unknown::<R>(positive) - at.unknown::<R>(negative);
        let current = self.law.current([voltage, control]);
        let [by_voltage, by_control] = current.gradient;
        let derivatives = derivatives(by_voltage, by_control);
        stamp.add_residual(positive, current.value);
        stamp.add_residual(negative, -current.value);
        for (row, sign) in entries.iter().zip([1.0, -1.0]) {
            for (&entry, derivative) in row.iter().zip(derivatives) {
                stamp.add_conductance(entry, derivative * sign);
            }
        }
    }

    /// Appends the thermal noise of the switch's resistance at `control` to `sources`.
    fn thermal_noise(&self, control: f64, sources: &mut Vec<NoiseSource>) {
        let conductance = self.law.conductance(Dual::<R, 0>::number(control)).value;
        let [positive, negative, ..] = TERMINALS;
        sources.push(NoiseSource::thermal(
            "thermal",
            [
                Some(UnknownIndex::new(positive)),
                Some(UnknownIndex::new(negative)),
            ],
            conductance.value(),
            self.temperature,
        ));
    }
}

/// Layout shape for [`VoltageControlledSwitch`] storing the matrix entries joining each
/// output terminal to all four switch terminals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VoltageControlledSwitchShape([[usize; 4]; 2]);

/// Parameters for [`VoltageControlledSwitch`], matching the PSpice A/D Reference Guide
/// defaults for `VSWITCH`.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Lift, ParameterMetadata)]
#[serde(default, deny_unknown_fields)]
#[schemars(bound = "R: Serialize + JsonSchema")]
pub struct VoltageControlledSwitchParameters<R: Real = f64> {
    /// On-state resistance at or past `on_voltage`, in ohms; default 1.
    #[dimension(RESISTANCE)]
    pub on_resistance: R,
    /// Off-state resistance at or past `off_voltage`, in ohms; default 1e6.
    #[dimension(RESISTANCE)]
    pub off_resistance: R,
    /// Control voltage at which the switch reaches `on_resistance`, in volts; default 1.
    #[dimension(VOLTAGE)]
    pub on_voltage: R,
    /// Control voltage at which the switch reaches `off_resistance`, in volts; default 0.
    #[dimension(VOLTAGE)]
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
    type Shape = VoltageControlledSwitchShape;
    type Prepared<R: Real> = PreparedSwitch<R>;
    type Memory = ();

    fn parameter_kinds() -> &'static [(&'static str, ParameterKind)] {
        VoltageControlledSwitchParameters::<f64>::PARAMETER_KINDS
    }

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
        Ok(PreparedSwitch {
            law: Law::new(
                [on_resistance, off_resistance],
                [on_voltage, off_voltage],
                "voltage",
            )?,
            temperature: temperature.value(),
        })
    }

    fn layout<R: Real>(
        _: &VoltageControlledSwitchParameters<R>,
        _: &PreparedSwitch<R>,
        layout: &mut Layout,
    ) -> Result<VoltageControlledSwitchShape, ParameterError> {
        let [positive, negative, ..] = TERMINALS;
        Ok(VoltageControlledSwitchShape([positive, negative].map(
            |row| TERMINALS.map(|column| layout.entry(row, column)),
        )))
    }

    fn evaluate<R: Real>(
        VoltageControlledSwitchShape(entries): &VoltageControlledSwitchShape,
        prepared: &PreparedSwitch<R>,
        (): &mut (),
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        let [_, _, control_positive, control_negative] = TERMINALS;
        let control = at.unknown::<R>(control_positive) - at.unknown::<R>(control_negative);
        // With respect to positive, negative, control_positive and control_negative.
        prepared.stamp(
            entries,
            control,
            at,
            |by_voltage, by_control| [by_voltage, -by_voltage, by_control, -by_control],
            stamp,
        );
    }

    /// Appends thermal noise for the switch resistance determined by the control voltage.
    fn noise<R: Real>(
        _: &VoltageControlledSwitchShape,
        prepared: &PreparedSwitch<R>,
        at: &Local<'_>,
        sources: &mut Vec<NoiseSource>,
    ) {
        let [_, _, control_positive, control_negative] = TERMINALS;
        prepared.thermal_noise(
            at.value(control_positive) - at.value(control_negative),
            sources,
        );
    }
}

/// Current-controlled switch implementing the PSpice `ISWITCH` model.
///
/// Control port current smoothly interpolates log-resistance between `positive` and
/// `negative` along a zero-slope cubic from off to on. The control port acts as a zero-volt
/// short circuit from `control_positive` to `control_negative` with current unknown
/// `control`.
#[derive(Debug, Clone, Copy)]
pub struct CurrentControlledSwitch;

/// Layout shape for [`CurrentControlledSwitch`] storing the current-sensing branch and
/// output terminal coupling entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CurrentControlledSwitchShape {
    sense: CurrentBranch,
    entries: [[usize; 3]; 2],
}

/// Parameters for [`CurrentControlledSwitch`], matching the PSpice A/D Reference Guide
/// defaults for `ISWITCH`.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Lift, ParameterMetadata)]
#[serde(default, deny_unknown_fields)]
#[schemars(bound = "R: Serialize + JsonSchema")]
pub struct CurrentControlledSwitchParameters<R: Real = f64> {
    /// On-state resistance at or past `on_current`, in ohms; default 1.
    #[dimension(RESISTANCE)]
    pub on_resistance: R,
    /// Off-state resistance at or past `off_current`, in ohms; default 1e6.
    #[dimension(RESISTANCE)]
    pub off_resistance: R,
    /// Control current at which the switch reaches `on_resistance`, in amperes; default 1 mA.
    #[dimension(CURRENT)]
    pub on_current: R,
    /// Control current at which the switch reaches `off_resistance`, in amperes; default 0.
    #[dimension(CURRENT)]
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
    type Shape = CurrentControlledSwitchShape;
    type Prepared<R: Real> = PreparedSwitch<R>;
    type Memory = ();

    fn parameter_kinds() -> &'static [(&'static str, ParameterKind)] {
        CurrentControlledSwitchParameters::<f64>::PARAMETER_KINDS
    }

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
        Ok(PreparedSwitch {
            law: Law::new(
                [on_resistance, off_resistance],
                [on_current, off_current],
                "current",
            )?,
            temperature: temperature.value(),
        })
    }

    fn layout<R: Real>(
        _: &CurrentControlledSwitchParameters<R>,
        _: &PreparedSwitch<R>,
        layout: &mut Layout,
    ) -> Result<CurrentControlledSwitchShape, ParameterError> {
        let [positive, negative, control_positive, control_negative] = TERMINALS;
        let sense = CurrentBranch::new(layout, "control", [control_positive, control_negative]);
        let columns = [positive, negative, sense.unknown()];
        let entries =
            [positive, negative].map(|row| columns.map(|column| layout.entry(row, column)));
        Ok(CurrentControlledSwitchShape { sense, entries })
    }

    fn evaluate<R: Real>(
        shape: &CurrentControlledSwitchShape,
        prepared: &PreparedSwitch<R>,
        (): &mut (),
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        // The port is a short circuit, its current the control.
        let sense = &shape.sense;
        sense.add(at, stamp, sense.voltage(at));
        // With respect to positive, negative and the control.
        prepared.stamp(
            &shape.entries,
            sense.current(at),
            at,
            |by_voltage, by_control| [by_voltage, -by_voltage, by_control],
            stamp,
        );
    }

    /// Appends thermal noise for the switch resistance determined by the control current.
    fn noise<R: Real>(
        shape: &CurrentControlledSwitchShape,
        prepared: &PreparedSwitch<R>,
        at: &Local<'_>,
        sources: &mut Vec<NoiseSource>,
    ) {
        prepared.thermal_noise(shape.sense.current(at), sources);
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::testing::{Bench, assert_close, assert_jacobians_match, source, thermal};

    const SWITCH: &str = r#"{"on_resistance": 10, "off_resistance": 1e6,
        "on_voltage": 2, "off_voltage": 1}"#;

    /// Computes switch resistance at 1 V across the output and control voltage `control`.
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
    fn rejects_equal_voltages() {
        let error =
            Bench::try_new::<VoltageControlledSwitch>(r#"{"on_voltage": 1, "off_voltage": 1}"#)
                .expect_err("equal voltages");
        assert_eq!(error.0, "on_voltage and off_voltage differ, not both 1.0");
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
        }
        let error = Bench::try_new::<CurrentControlledSwitch>(r#"{"on_current": 0}"#)
            .expect_err("equal currents");
        assert_eq!(error.0, "on_current and off_current differ, not both 0.0");
        let sources = bench.noise(&[1.0, 0.0, 0.0, 0.0, 3e-3]);
        assert_close(&[source(&sources, "thermal").density(1.0)], &[thermal(0.1)]);
    }

    proptest! {
        #[test]
        fn jacobians_match_finite_differences(
            voltage in -5.0..5.0f64,
            control in 0.0..3.0f64,
        ) {
            assert_jacobians_match::<VoltageControlledSwitch>(
                SWITCH,
                &[voltage, 0.0, control, 0.0],
            );
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
