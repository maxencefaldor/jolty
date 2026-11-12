use serde::Deserialize;

use crate::junction::{
    Depletion, Temperatures, below_one, critical_voltage, depletion_charge, junction_current,
    limit_junction,
};
use crate::stamp::Series;
use crate::{
    Binding, Branch, BranchKind, Device, Input, Interface, NativeDevice, NativeModel, Output,
    ParameterError, REFERENCE_TEMPERATURE, Real, TerminalPair,
};

/// A junction diode from terminal `anode` to terminal `cathode`, in the SPICE model.
///
/// The junction carries the [junction current](crate::junction) with exponential slope
/// `n·Vt`, plus `gmin · v`, and holds a depletion charge and a diffusion charge. With a
/// series resistance, the junction sits between an internal node, labelled `junction`, and
/// the cathode; below [`SMALL_RESISTANCE`](crate::SMALL_RESISTANCE), the resistance also
/// carries its current as an unknown, labelled `junction_current`. Given a breakdown
/// voltage, the reverse current grows exponentially beyond it, matched to the breakdown
/// current there as ngspice matches it. Steps of the junction voltage are limited the way
/// SPICE limits them, so Newton iterations cannot overflow the exponential.
///
/// Away from its nominal temperature, its saturation current scales with the energy gap,
/// its junction potential and capacitance with silicon's band gap, and its breakdown
/// voltage by its temperature coefficient, as ngspice scales them.
#[derive(Debug)]
pub struct Diode<R: Real = f64> {
    series: Series<R>,
    junction: TerminalPair,
    saturation_current: R,
    /// The emission coefficient times the thermal voltage.
    slope: R,
    /// The voltage above which steps are limited: where the current's curvature takes over.
    critical_voltage: f64,
    depletion: Option<Depletion<R>>,
    transit_time: R,
    breakdown: Option<Breakdown<R>>,
    /// The junction voltage of the last evaluation, from which the next step is limited.
    last_voltage: f64,
}

/// A diode's reverse breakdown.
#[derive(Debug, Clone, Copy)]
struct Breakdown<R: Real> {
    /// The voltage below which the junction breaks down, matched to the breakdown current.
    voltage: R,
    /// The breakdown emission coefficient times the thermal voltage.
    slope: R,
}

impl<R: Real> Breakdown<R> {
    /// The breakdown of a junction with `saturation` current, broken down at `voltage`
    /// with `current`, and exponential `slope`, at thermal voltage `vt`.
    ///
    /// ngspice moves the breakdown voltage so the current there is the breakdown current,
    /// by a fixed-point iteration, and leaves it where a breakdown current too small for
    /// the saturation current cannot be met.
    fn new(voltage: R, current: R, saturation: R, slope: R, vt: f64) -> Self {
        /// How closely the current at the matched voltage meets the breakdown current.
        const MATCH: f64 = 1e-12;
        if current.value() < (saturation * voltage / vt).value() {
            return Self { voltage, slope };
        }
        let mut matched = voltage - slope * (current / saturation + 1.0).ln();
        for _ in 0..25 {
            matched = voltage - slope * (current / saturation + 1.0 - matched / vt).ln();
            let reached = saturation * (((voltage - matched) / slope).exp() - 1.0 + matched / vt);
            if (reached - current).abs().value() <= MATCH * current.value() {
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
#[serde(default, deny_unknown_fields, bound = "")]
pub struct DiodeParameters<R: Real = f64> {
    /// The saturation current, in amperes; default 1e-14.
    pub saturation_current: R,
    /// The emission coefficient; default 1.
    pub emission_coefficient: R,
    /// The series resistance, in ohms; default 0.
    pub series_resistance: R,
    /// The zero-bias junction capacitance, in farads; default 0.
    pub junction_capacitance: R,
    /// The junction potential, in volts; default 1.
    pub junction_potential: R,
    /// The grading coefficient of the junction, below 1; default 0.5.
    pub grading_coefficient: R,
    /// The fraction of the junction potential above which the depletion capacitance is
    /// extended linearly, below 1; default 0.5.
    pub depletion_coefficient: R,
    /// The transit time, in seconds; default 0.
    pub transit_time: R,
    /// The reverse breakdown voltage, in volts, positive; without it, the junction never
    /// breaks down.
    pub breakdown_voltage: Option<R>,
    /// The current at the breakdown voltage, in amperes; default 1e-3.
    pub breakdown_current: R,
    /// The breakdown emission coefficient; the emission coefficient by default.
    pub breakdown_emission_coefficient: Option<R>,
    /// How fast the breakdown voltage falls with temperature, in volts per kelvin;
    /// default 0.
    pub breakdown_temperature_coefficient: R,
    /// The energy gap that scales the saturation current with temperature, in volts;
    /// default 1.11, silicon's.
    pub energy_gap: R,
    /// The exponent of the temperature ratio in the saturation current's scaling;
    /// default 3.
    pub saturation_current_temperature_exponent: R,
    /// The temperature the parameters were measured at, in kelvins; default 27 °C.
    pub nominal_temperature: R,
}

impl<R: Real> Default for DiodeParameters<R> {
    fn default() -> Self {
        Self {
            saturation_current: R::constant(1e-14),
            emission_coefficient: R::constant(1.0),
            series_resistance: R::constant(0.0),
            junction_capacitance: R::constant(0.0),
            junction_potential: R::constant(1.0),
            grading_coefficient: R::constant(0.5),
            depletion_coefficient: R::constant(0.5),
            transit_time: R::constant(0.0),
            breakdown_voltage: None,
            breakdown_current: R::constant(1e-3),
            breakdown_emission_coefficient: None,
            breakdown_temperature_coefficient: R::constant(0.0),
            energy_gap: R::constant(1.11),
            saturation_current_temperature_exponent: R::constant(3.0),
            nominal_temperature: R::constant(REFERENCE_TEMPERATURE),
        }
    }
}

impl NativeModel for Diode {
    type Device<R: Real> = Diode<R>;
}

impl<R: Real> NativeDevice<R> for Diode<R> {
    const NAME: &'static str = "diode";
    type Parameters = DiodeParameters<R>;

    fn interface(_: &DiodeParameters<R>) -> Interface {
        Interface::new(
            &["anode", "cathode"],
            &[Branch::new(0, 1, BranchKind::Conductive)],
        )
    }

    fn new(
        parameters: DiodeParameters<R>,
        binding: &mut Binding<'_>,
    ) -> Result<Self, ParameterError> {
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
        let junction_potential = junction_potential.min(depletion_coefficient.recip());
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
        let (series, inner) = Series::behind(binding, anode, series_resistance, "junction");
        let junction = TerminalPair::new(binding, inner, cathode);
        let slope = emission_coefficient * vt;
        Ok(Self {
            series,
            junction,
            saturation_current,
            slope,
            critical_voltage: critical_voltage(saturation_current.value(), slope.value()),
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

impl<R: Real> Diode<R> {
    /// The junction's step to `applied` limited from the last evaluation's, as ngspice
    /// limits it: about the breakdown voltage when the junction nears it.
    fn limit(&self, applied: f64) -> Option<f64> {
        match self.breakdown {
            Some(breakdown)
                if applied
                    < (10.0 * breakdown.slope.value() - breakdown.voltage.value()).min(0.0) =>
            {
                let (voltage, slope) = (breakdown.voltage.value(), breakdown.slope.value());
                limit_junction(
                    -(applied + voltage),
                    -(self.last_voltage + voltage),
                    slope,
                    self.critical_voltage,
                )
                .map(|limited| -(limited + voltage))
            }
            _ => limit_junction(
                applied,
                self.last_voltage,
                self.slope.value(),
                self.critical_voltage,
            ),
        }
    }

    /// The junction's current at `voltage`, and its derivative.
    fn current(&self, voltage: f64) -> (R, R) {
        match self.breakdown {
            Some(breakdown)
                if voltage < -3.0 * self.slope.value() && voltage < -breakdown.voltage.value() =>
            {
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

impl<R: Real> Device<R> for Diode<R> {
    fn evaluate(&mut self, input: &Input<'_>, output: &mut Output<'_, R>) {
        self.series.add(input, output);
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
    use crate::testing::{Bench, assert_jacobians_match, assert_parameter_gradients_match};
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
    fn a_small_series_resistance_carries_its_current_as_an_unknown() {
        let small = r#"{"series_resistance": 0.5, "junction_capacitance": 1e-12}"#;
        let mut bench = Bench::new::<Diode>(small);
        assert_eq!(
            bench.unknowns(),
            ["n0", "n1", "device:junction", "device:junction_current"]
        );
        let x = [0.7, 0.0, 0.65, 0.1];
        let residual = bench.evaluate(&x).residual;
        assert!((residual[0] - 0.1).abs() < 1e-15, "{residual:?}");
        assert!(residual[3].abs() < 1e-15, "{residual:?}");
        assert_jacobians_match::<Diode>(small, &x);
        assert_parameter_gradients_match::<Diode>(small, &x);
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
    fn parameter_gradients_are_exact() {
        let every = r#"{"saturation_current": 2e-14, "emission_coefficient": 1.3,
            "junction_capacitance": 2e-12, "junction_potential": 0.8,
            "grading_coefficient": 0.4, "depletion_coefficient": 0.6, "transit_time": 1e-9,
            "breakdown_voltage": 5, "breakdown_current": 1e-3,
            "breakdown_emission_coefficient": 1.1, "breakdown_temperature_coefficient": 1e-3,
            "energy_gap": 1.2, "saturation_current_temperature_exponent": 2.5,
            "nominal_temperature": 290}"#;
        for voltage in [0.6, 0.55, -0.02, -1.0, -5.4] {
            assert_parameter_gradients_match::<Diode>(every, &[voltage, 0.0]);
        }
        let resistive = r#"{"series_resistance": 10, "junction_capacitance": 1e-12}"#;
        assert_parameter_gradients_match::<Diode>(resistive, &[0.7, 0.0, 0.65]);
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
