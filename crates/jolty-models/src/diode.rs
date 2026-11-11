use serde::Deserialize;

use crate::junction::{
    Depletion, THERMAL_VOLTAGE, below_one, critical_voltage, depletion_charge, junction_current,
    limit_junction,
};
use crate::{
    Binding, Branch, BranchKind, Device, Input, Interface, NativeDevice, Output, ParameterError,
    Quantity, TerminalPair,
};

/// A junction diode from terminal `anode` to terminal `cathode`, in the SPICE model.
///
/// The junction carries the [junction current](crate::junction) with exponential slope
/// `n·Vt`, plus `gmin · v`, and holds a depletion charge and a diffusion charge. With a
/// series resistance, the junction sits between an internal node, labelled `junction`, and
/// the cathode. Steps of the junction voltage are limited the way SPICE limits them, so
/// Newton iterations cannot overflow the exponential.
#[derive(Debug)]
pub struct Diode {
    series: Option<(TerminalPair, f64)>,
    junction: TerminalPair,
    saturation_current: f64,
    /// The emission coefficient times the thermal voltage.
    slope: f64,
    /// The voltage above which steps are limited: where the current's curvature takes over.
    critical_voltage: f64,
    depletion: Option<Depletion>,
    transit_time: f64,
    /// The junction voltage of the last evaluation, from which the next step is limited.
    last_voltage: f64,
}

/// The parameters of a [`Diode`], with SPICE's defaults.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DiodeParameters {
    /// The saturation current, in amperes; default 1e-14.
    pub saturation_current: f64,
    /// The emission coefficient; default 1.
    pub emission_coefficient: f64,
    /// The series resistance, in ohms; default 0.
    pub series_resistance: f64,
    /// The zero-bias junction capacitance, in farads; default 0.
    pub junction_capacitance: f64,
    /// The junction potential, in volts; default 1.
    pub junction_potential: f64,
    /// The grading coefficient of the junction, below 1; default 0.5.
    pub grading_coefficient: f64,
    /// The fraction of the junction potential above which the depletion capacitance is
    /// extended linearly, below 1; default 0.5.
    pub depletion_coefficient: f64,
    /// The transit time, in seconds; default 0.
    pub transit_time: f64,
}

impl Default for DiodeParameters {
    fn default() -> Self {
        Self {
            saturation_current: 1e-14,
            emission_coefficient: 1.0,
            series_resistance: 0.0,
            junction_capacitance: 0.0,
            junction_potential: 1.0,
            grading_coefficient: 0.5,
            depletion_coefficient: 0.5,
            transit_time: 0.0,
        }
    }
}

impl NativeDevice for Diode {
    const NAME: &'static str = "diode";
    type Parameters = DiodeParameters;

    fn interface(_: &DiodeParameters) -> Interface {
        Interface::new(
            &["anode", "cathode"],
            &[Branch::new(0, 1, BranchKind::Conductive)],
        )
    }

    fn new(parameters: DiodeParameters, binding: &mut Binding<'_>) -> Result<Self, ParameterError> {
        let DiodeParameters {
            saturation_current,
            emission_coefficient,
            series_resistance,
            junction_capacitance,
            junction_potential,
            grading_coefficient,
            depletion_coefficient,
            transit_time,
        } = parameters;
        ParameterError::positive("saturation_current", saturation_current)?;
        ParameterError::positive("emission_coefficient", emission_coefficient)?;
        ParameterError::non_negative("series_resistance", series_resistance)?;
        ParameterError::non_negative("junction_capacitance", junction_capacitance)?;
        ParameterError::positive("junction_potential", junction_potential)?;
        below_one("grading_coefficient", grading_coefficient)?;
        below_one("depletion_coefficient", depletion_coefficient)?;
        ParameterError::non_negative("transit_time", transit_time)?;

        let (anode, cathode) = (binding.terminal(0), binding.terminal(1));
        let (series, junction) = if series_resistance > 0.0 {
            let internal = Some(binding.internal("junction", Quantity::Voltage));
            let series = TerminalPair::new(binding, anode, internal);
            (
                Some((series, 1.0 / series_resistance)),
                TerminalPair::new(binding, internal, cathode),
            )
        } else {
            (None, TerminalPair::new(binding, anode, cathode))
        };
        let slope = emission_coefficient * THERMAL_VOLTAGE;
        Ok(Self {
            series,
            junction,
            saturation_current,
            slope,
            critical_voltage: critical_voltage(saturation_current, slope),
            depletion: Depletion::new(
                junction_capacitance,
                junction_potential,
                grading_coefficient,
                depletion_coefficient,
            ),
            transit_time,
            last_voltage: 0.0,
        })
    }
}

impl Device for Diode {
    fn evaluate(&mut self, input: &Input<'_>, output: &mut Output<'_>) {
        if let Some((pair, conductance)) = &self.series {
            pair.add_current(output, conductance * pair.voltage(input));
            pair.add_conductance(output, *conductance);
        }
        let applied = self.junction.voltage(input);
        let voltage = limit_junction(
            applied,
            self.last_voltage,
            self.slope,
            self.critical_voltage,
        )
        .map_or(applied, |limited| {
            output.mark_limited();
            limited
        });
        self.last_voltage = voltage;

        // Evaluate at the limited voltage and extend linearly to the applied one, so the
        // Newton step that follows solves the circuit linearized at the limited voltage.
        let (current, conductance) = junction_current(voltage, self.saturation_current, self.slope);
        let total_conductance = conductance + input.gmin;
        let total_current = current + input.gmin * voltage;
        self.junction.add_current(
            output,
            total_current + total_conductance * (applied - voltage),
        );
        self.junction.add_conductance(output, total_conductance);

        let (depletion_charge, depletion_capacitance) =
            depletion_charge(self.depletion.as_ref(), voltage);
        let charge = depletion_charge + self.transit_time * current;
        let capacitance = depletion_capacitance + self.transit_time * conductance;
        self.junction
            .add_charge(output, charge + capacitance * (applied - voltage));
        self.junction.add_capacitance(output, capacitance);
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::testing::{Bench, assert_jacobians_match};

    const CHARGED: &str =
        r#"{"junction_capacitance": 2e-12, "transit_time": 1e-9, "grading_coefficient": 0.4}"#;

    #[test]
    fn follows_the_shockley_equation() {
        let mut bench = Bench::new::<Diode>("{}");
        let current = bench.settle(&[0.6, 0.0]).residual[0];
        let expected = 1e-14 * ((0.6 / THERMAL_VOLTAGE).exp() - 1.0);
        assert!(
            (current / expected - 1.0).abs() < 1e-12,
            "{current} vs {expected}"
        );
    }

    #[test]
    fn limits_large_steps_until_they_settle() {
        let mut bench = Bench::new::<Diode>("{}");
        let evaluation = bench.evaluate(&[10.0, 0.0]);
        assert!(evaluation.limited);
        assert!(evaluation.residual[0].is_finite());
        assert!(bench.evaluate(&[-10.0, 0.0]).limited);
        assert!(!bench.settle(&[-10.0, 0.0]).limited);
    }

    #[test]
    fn a_series_resistance_adds_an_internal_node() {
        let bench = Bench::new::<Diode>(r#"{"series_resistance": 10}"#);
        assert_eq!(bench.unknowns(), ["n0", "n1", "device:junction"]);
    }

    #[test]
    fn rejects_out_of_range_parameters() {
        let error = Bench::try_new::<Diode>(r#"{"grading_coefficient": 1}"#).unwrap_err();
        assert_eq!(
            error.to_string(),
            "grading_coefficient must be at least 0 and below 1, not 1.0"
        );
    }

    proptest! {
        /// The junction's conductance and capacitance are the derivatives of its current and
        /// charge, on both sides of the depletion threshold.
        #[test]
        fn jacobians_match_finite_differences(voltage in -5.0..0.7f64) {
            assert_jacobians_match::<Diode>(CHARGED, &[voltage, 0.0]);
            let with_resistance = CHARGED.replace('{', r#"{"series_resistance": 5, "#);
            assert_jacobians_match::<Diode>(&with_resistance, &[voltage + 0.01, 0.0, voltage]);
        }
    }
}
