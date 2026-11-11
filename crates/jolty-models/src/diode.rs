use serde::Deserialize;

use crate::{Binding, Device, Input, NativeDevice, Output, ParameterError, Quantity, TerminalPair};

/// The Boltzmann constant, in joules per kelvin.
const BOLTZMANN: f64 = 1.380_649e-23;
/// The elementary charge, in coulombs.
const ELEMENTARY_CHARGE: f64 = 1.602_176_634e-19;
/// The temperature devices are evaluated at: 27 °C, in kelvins.
const TEMPERATURE: f64 = 300.15;

/// A junction diode from terminal `anode` to terminal `cathode`, in the SPICE model.
///
/// The junction carries `Is · (e^{v / (n·Vt)} − 1)` plus `gmin · v`, and holds a depletion
/// charge and a diffusion charge. With a series resistance, the junction sits between an
/// internal node, labelled `junction`, and the cathode. Steps of the junction voltage are
/// limited the way SPICE limits them, so Newton iterations cannot overflow the exponential.
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

/// The depletion charge of a junction: the classic power law below `threshold`, extended
/// linearly in capacitance above it, where the power law would diverge.
#[derive(Debug, Clone, Copy)]
struct Depletion {
    capacitance: f64,
    potential: f64,
    grading: f64,
    threshold: f64,
    /// The charge at the threshold, divided by the zero-bias capacitance.
    charge_at_threshold: f64,
    /// `(1 − fc)^{1 + m}`.
    divisor: f64,
    /// `1 − fc · (1 + m)`.
    offset: f64,
}

impl Depletion {
    fn new(capacitance: f64, potential: f64, grading: f64, coefficient: f64) -> Self {
        Self {
            capacitance,
            potential,
            grading,
            threshold: coefficient * potential,
            charge_at_threshold: potential * (1.0 - (1.0 - coefficient).powf(1.0 - grading))
                / (1.0 - grading),
            divisor: (1.0 - coefficient).powf(1.0 + grading),
            offset: 1.0 - coefficient * (1.0 + grading),
        }
    }

    /// The charge and capacitance at junction voltage `voltage`.
    fn evaluate(&self, voltage: f64) -> (f64, f64) {
        let Self {
            capacitance,
            potential,
            grading,
            threshold,
            ..
        } = *self;
        if voltage < threshold {
            let remaining = 1.0 - voltage / potential;
            let factor = remaining.powf(-grading);
            let charge = capacitance * potential * (1.0 - remaining * factor) / (1.0 - grading);
            (charge, capacitance * factor)
        } else {
            let above = voltage - threshold;
            let square = voltage * voltage - threshold * threshold;
            let charge = capacitance
                * (self.charge_at_threshold
                    + (self.offset * above + grading / (2.0 * potential) * square) / self.divisor);
            (
                charge,
                capacitance / self.divisor * (self.offset + grading * voltage / potential),
            )
        }
    }
}

impl NativeDevice for Diode {
    const NAME: &'static str = "diode";
    const TERMINALS: &'static [&'static str] = &["anode", "cathode"];
    type Parameters = DiodeParameters;

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
        let slope = emission_coefficient * BOLTZMANN * TEMPERATURE / ELEMENTARY_CHARGE;
        Ok(Self {
            series,
            junction,
            saturation_current,
            slope,
            critical_voltage: slope
                * (slope / (std::f64::consts::SQRT_2 * saturation_current)).ln(),
            depletion: (junction_capacitance > 0.0).then(|| {
                Depletion::new(
                    junction_capacitance,
                    junction_potential,
                    grading_coefficient,
                    depletion_coefficient,
                )
            }),
            transit_time,
            last_voltage: 0.0,
        })
    }
}

impl Diode {
    /// SPICE's junction voltage limiting: above the critical voltage, a step of more than two
    /// slopes is replaced by a logarithmic one, which the exponential follows safely. Returns
    /// the limited voltage, or `None` when `voltage` needs no limiting.
    fn limit(&self, voltage: f64) -> Option<f64> {
        let (old, slope) = (self.last_voltage, self.slope);
        if voltage <= self.critical_voltage || (voltage - old).abs() <= 2.0 * slope {
            return None;
        }
        Some(if old > 0.0 {
            let argument = 1.0 + (voltage - old) / slope;
            if argument > 0.0 {
                old + slope * argument.ln()
            } else {
                self.critical_voltage
            }
        } else {
            slope * (voltage / slope).ln()
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
        let voltage = self.limit(applied).map_or(applied, |limited| {
            output.mark_limited();
            limited
        });
        self.last_voltage = voltage;

        // Evaluate at the limited voltage and extend linearly to the applied one, so the
        // Newton step that follows solves the circuit linearized at the limited voltage.
        let exponential = (voltage / self.slope).exp();
        let current = self.saturation_current * (exponential - 1.0);
        let conductance = self.saturation_current * exponential / self.slope;
        let total_conductance = conductance + input.gmin;
        let total_current = current + input.gmin * voltage;
        self.junction.add_current(
            output,
            total_current + total_conductance * (applied - voltage),
        );
        self.junction.add_conductance(output, total_conductance);

        let (depletion_charge, depletion_capacitance) = self
            .depletion
            .map_or((0.0, 0.0), |depletion| depletion.evaluate(voltage));
        let charge = depletion_charge + self.transit_time * current;
        let capacitance = depletion_capacitance + self.transit_time * conductance;
        self.junction
            .add_charge(output, charge + capacitance * (applied - voltage));
        self.junction.add_capacitance(output, capacitance);
    }
}

fn below_one(parameter: &str, value: f64) -> Result<f64, ParameterError> {
    if (0.0..1.0).contains(&value) {
        Ok(value)
    } else {
        Err(ParameterError(format!(
            "{parameter} must be at least 0 and below 1, not {value:?}"
        )))
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
        let mut bench = Bench::new::<Diode>("{}", 2);
        let slope = BOLTZMANN * TEMPERATURE / ELEMENTARY_CHARGE;
        bench.evaluate(&[0.6, 0.0]);
        let current = bench.evaluate(&[0.6, 0.0]).residual[0];
        let expected = 1e-14 * ((0.6 / slope).exp() - 1.0);
        assert!(
            (current / expected - 1.0).abs() < 1e-12,
            "{current} vs {expected}"
        );
    }

    #[test]
    fn limits_large_forward_steps() {
        let mut bench = Bench::new::<Diode>("{}", 2);
        let evaluation = bench.evaluate(&[10.0, 0.0]);
        assert!(evaluation.limited);
        assert!(evaluation.residual[0].is_finite());
        assert!(!bench.evaluate(&[-10.0, 0.0]).limited);
    }

    #[test]
    fn a_series_resistance_adds_an_internal_node() {
        let bench = Bench::new::<Diode>(r#"{"series_resistance": 10}"#, 2);
        assert_eq!(bench.unknowns(), ["n0", "n1", "device:junction"]);
    }

    #[test]
    fn rejects_out_of_range_parameters() {
        let error = Bench::try_new::<Diode>(r#"{"grading_coefficient": 1}"#, 2).unwrap_err();
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
