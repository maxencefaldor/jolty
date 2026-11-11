use serde::Deserialize;

use crate::junction::{
    Depletion, Temperatures, below_one, critical_voltage, depletion_charge, junction_current,
    limit_junction,
};
use crate::{
    Binding, Branch, BranchKind, Device, Input, Interface, NativeDevice, Output, ParameterError,
    Quantity, REFERENCE_TEMPERATURE, TerminalPair,
};

/// A junction diode from terminal `anode` to terminal `cathode`, in the SPICE model.
///
/// The junction carries the [junction current](crate::junction) with exponential slope
/// `n·Vt`, plus `gmin · v`, and holds a depletion charge and a diffusion charge. With a
/// series resistance, the junction sits between an internal node, labelled `junction`, and
/// the cathode. Given a breakdown voltage, the reverse current grows exponentially beyond
/// it, matched to the breakdown current there as ngspice matches it. Steps of the junction
/// voltage are limited the way SPICE limits them, so Newton iterations cannot overflow the
/// exponential.
///
/// Away from its nominal temperature, its saturation current scales with the energy gap,
/// its junction potential and capacitance with silicon's band gap, and its breakdown
/// voltage by its temperature coefficient, as ngspice scales them.
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
    breakdown: Option<Breakdown>,
    /// The junction voltage of the last evaluation, from which the next step is limited.
    last_voltage: f64,
}

/// A diode's reverse breakdown.
#[derive(Debug, Clone, Copy)]
struct Breakdown {
    /// The voltage below which the junction breaks down, matched to the breakdown current.
    voltage: f64,
    /// The breakdown emission coefficient times the thermal voltage.
    slope: f64,
}

impl Breakdown {
    /// The breakdown of a junction with `saturation` current, broken down at `voltage`
    /// with `current`, and exponential `slope`, at thermal voltage `vt`.
    ///
    /// ngspice moves the breakdown voltage so the current there is the breakdown current,
    /// by a fixed-point iteration, and leaves it where a breakdown current too small for
    /// the saturation current cannot be met.
    fn new(voltage: f64, current: f64, saturation: f64, slope: f64, vt: f64) -> Self {
        /// How closely the current at the matched voltage meets the breakdown current.
        const MATCH: f64 = 1e-12;
        if current < saturation * voltage / vt {
            return Self { voltage, slope };
        }
        let mut matched = voltage - slope * (1.0 + current / saturation).ln();
        for _ in 0..25 {
            matched = voltage - slope * (current / saturation + 1.0 - matched / vt).ln();
            let reached = saturation * (((voltage - matched) / slope).exp() - 1.0 + matched / vt);
            if (reached - current).abs() <= MATCH * current {
                break;
            }
        }
        Self {
            voltage: matched,
            slope,
        }
    }
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
    /// The reverse breakdown voltage, in volts, positive; without it, the junction never
    /// breaks down.
    pub breakdown_voltage: Option<f64>,
    /// The current at the breakdown voltage, in amperes; default 1e-3.
    pub breakdown_current: f64,
    /// The breakdown emission coefficient; the emission coefficient by default.
    pub breakdown_emission_coefficient: Option<f64>,
    /// How fast the breakdown voltage falls with temperature, in volts per kelvin;
    /// default 0.
    pub breakdown_temperature_coefficient: f64,
    /// The energy gap that scales the saturation current with temperature, in volts;
    /// default 1.11, silicon's.
    pub energy_gap: f64,
    /// The exponent of the temperature ratio in the saturation current's scaling;
    /// default 3.
    pub saturation_current_temperature_exponent: f64,
    /// The temperature the parameters were measured at, in kelvins; default 27 °C.
    pub nominal_temperature: f64,
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
            breakdown_voltage: None,
            breakdown_current: 1e-3,
            breakdown_emission_coefficient: None,
            breakdown_temperature_coefficient: 0.0,
            energy_gap: 1.11,
            saturation_current_temperature_exponent: 3.0,
            nominal_temperature: REFERENCE_TEMPERATURE,
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
            breakdown_voltage,
            breakdown_current,
            breakdown_emission_coefficient,
            breakdown_temperature_coefficient,
            energy_gap,
            saturation_current_temperature_exponent,
            nominal_temperature,
        } = parameters;
        ParameterError::positive("saturation_current", saturation_current)?;
        ParameterError::positive("emission_coefficient", emission_coefficient)?;
        ParameterError::non_negative("series_resistance", series_resistance)?;
        ParameterError::non_negative("junction_capacitance", junction_capacitance)?;
        ParameterError::positive("junction_potential", junction_potential)?;
        below_one("grading_coefficient", grading_coefficient)?;
        below_one("depletion_coefficient", depletion_coefficient)?;
        ParameterError::non_negative("transit_time", transit_time)?;
        ParameterError::positive("breakdown_current", breakdown_current)?;
        let breakdown_emission = breakdown_emission_coefficient.unwrap_or(emission_coefficient);
        ParameterError::positive("breakdown_emission_coefficient", breakdown_emission)?;
        ParameterError::non_negative("energy_gap", energy_gap)?;

        let temperatures = Temperatures::new(binding.temperature(), nominal_temperature)?;
        let vt = temperatures.thermal_voltage();
        let saturation_current = saturation_current
            * temperatures.saturation(
                energy_gap,
                saturation_current_temperature_exponent,
                emission_coefficient,
            );
        let (junction_potential, junction_capacitance) = temperatures.junction(
            junction_potential,
            junction_capacitance,
            grading_coefficient,
        );
        // ngspice holds the depletion threshold at or below the junction potential.
        let junction_potential = junction_potential.min(1.0 / depletion_coefficient);
        let breakdown = breakdown_voltage
            .map(|voltage| {
                ParameterError::positive("breakdown_voltage", voltage).map(|voltage| {
                    Breakdown::new(
                        voltage - breakdown_temperature_coefficient * temperatures.difference(),
                        breakdown_current,
                        saturation_current,
                        breakdown_emission * vt,
                        vt,
                    )
                })
            })
            .transpose()?;

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
        let slope = emission_coefficient * vt;
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
            breakdown,
            last_voltage: 0.0,
        })
    }
}

impl Diode {
    /// The junction's step to `applied` limited from the last evaluation's, as ngspice
    /// limits it: about the breakdown voltage when the junction nears it.
    fn limit(&self, applied: f64) -> Option<f64> {
        match self.breakdown {
            Some(breakdown) if applied < (10.0 * breakdown.slope - breakdown.voltage).min(0.0) => {
                limit_junction(
                    -(applied + breakdown.voltage),
                    -(self.last_voltage + breakdown.voltage),
                    breakdown.slope,
                    self.critical_voltage,
                )
                .map(|limited| -(limited + breakdown.voltage))
            }
            _ => limit_junction(
                applied,
                self.last_voltage,
                self.slope,
                self.critical_voltage,
            ),
        }
    }

    /// The junction's current at `voltage`, and its derivative.
    fn current(&self, voltage: f64) -> (f64, f64) {
        match self.breakdown {
            Some(breakdown) if voltage < -3.0 * self.slope && voltage < -breakdown.voltage => {
                let exponential = (-(breakdown.voltage + voltage) / breakdown.slope).exp();
                (
                    -self.saturation_current * exponential,
                    self.saturation_current * exponential / breakdown.slope,
                )
            }
            _ => junction_current(voltage, self.saturation_current, self.slope),
        }
    }
}

impl Device for Diode {
    fn evaluate(&mut self, input: &Input<'_>, output: &mut Output<'_>) {
        if let Some((pair, conductance)) = &self.series {
            pair.add_current(output, conductance * pair.voltage(input));
            pair.add_conductance(output, *conductance);
        }
        let applied = self.junction.voltage(input);
        let voltage = input
            .limiting
            .then(|| self.limit(applied))
            .flatten()
            .map_or(applied, |limited| {
                output.mark_limited();
                limited
            });
        self.last_voltage = voltage;

        // Evaluate at the limited voltage and extend linearly to the applied one, so the
        // Newton step that follows solves the circuit linearized at the limited voltage.
        let (current, conductance) = self.current(voltage);
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
    use crate::thermal_voltage;

    const THERMAL_VOLTAGE: f64 = thermal_voltage(REFERENCE_TEMPERATURE);

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
    fn the_saturation_current_scales_with_temperature() {
        let (hot, emission) = (350.0, 1.5);
        let mut bench = Bench::at::<Diode>(r#"{"emission_coefficient": 1.5}"#, hot);
        let current = bench.settle(&[0.6, 0.0]).residual[0];
        let vt = thermal_voltage(hot);
        let ratio = hot / REFERENCE_TEMPERATURE;
        let saturation = 1e-14 * (((ratio - 1.0) * 1.11 / vt + 3.0 * ratio.ln()) / emission).exp();
        let expected = saturation * ((0.6 / (emission * vt)).exp() - 1.0);
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
    fn breaks_down_at_its_breakdown_current() {
        let mut bench = Bench::new::<Diode>(
            r#"{"breakdown_voltage": 5.1, "breakdown_current": 1e-3, "emission_coefficient": 1.2}"#,
        );
        let current = bench.settle(&[-5.1, 0.0]).residual[0];
        assert!((current / -1e-3 - 1.0).abs() < 1e-3, "{current}");
        let deeper = bench.settle(&[-5.3, 0.0]).residual[0];
        let expected = current * (0.2 / (1.2 * THERMAL_VOLTAGE)).exp();
        assert!(
            (deeper / expected - 1.0).abs() < 1e-9,
            "{deeper} vs {expected}"
        );
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
            let breaking = CHARGED.replace('{', r#"{"breakdown_voltage": 3, "#);
            assert_jacobians_match::<Diode>(&breaking, &[voltage, 0.0]);
            let with_resistance = CHARGED.replace('{', r#"{"series_resistance": 5, "#);
            assert_jacobians_match::<Diode>(&with_resistance, &[voltage + 0.01, 0.0, voltage]);
        }
    }
}
