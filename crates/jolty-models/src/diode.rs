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

use crate::junction::{
    Depletion, Temperatures, below_one, critical_voltage, depletion_charge, junction_current,
    limit_junction,
};
use crate::kernel::parts::{Pair, Series, series_value};
use crate::noise::Flicker;
use crate::{
    Branch, BranchKind, Interface, Kernel, Layout, Local, NoiseSource, ParameterError,
    REFERENCE_TEMPERATURE, Real, Stamp,
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
#[derive(Debug, Clone, Copy)]
pub struct Diode;

/// Where a diode's contributions go among its local unknowns: through its series
/// resistance, and across its junction, from the node behind the resistance to the cathode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiodeShape {
    series: Series,
    junction: Pair,
}

/// A diode's numbers at its temperature.
#[derive(Debug, Clone, Copy)]
pub struct PreparedDiode<R: Real> {
    /// The series resistance's conductance, or its resistance when its current is an
    /// unknown.
    series: R,
    junction: Junction<R>,
    transit_time: R,
    temperature: f64,
    flicker: Flicker,
}

/// What a diode remembers.
#[derive(Debug, Clone, Copy, Default)]
pub struct DiodeMemory {
    /// The junction voltage of the last evaluation, from which the next step is limited.
    last_voltage: f64,
}

/// A junction's numbers at its temperature: its current, its breakdown and its depletion
/// charge, and how its steps are limited, as ngspice's diode and the body diode of its
/// VDMOS share them.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Junction<R: Real> {
    saturation_current: R,
    /// The emission coefficient times the thermal voltage.
    slope: R,
    /// The voltage above which steps are limited: where the current's curvature takes over.
    critical_voltage: f64,
    depletion: Option<Depletion<R>>,
    breakdown: Option<Breakdown<R>>,
}

/// A junction's reverse breakdown.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Breakdown<R: Real> {
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
    pub(crate) fn new(voltage: R, current: R, saturation: R, slope: R, vt: R) -> Self {
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
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
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
    /// The flicker noise coefficient, `KF`; default 0, no flicker noise.
    pub flicker_coefficient: R,
    /// The power of the current in the flicker noise, `AF`; default 1.
    pub flicker_exponent: R,
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
            flicker_coefficient: R::constant(0.0),
            flicker_exponent: R::constant(1.0),
        }
    }
}

impl Kernel for Diode {
    const NAME: &'static str = "diode";
    type Parameters<R: Real> = DiodeParameters<R>;
    type Shape = DiodeShape;
    type Prepared<R: Real> = PreparedDiode<R>;
    type Memory = DiodeMemory;

    fn interface<R: Real>(_: &DiodeParameters<R>) -> Interface {
        Interface::new(
            &["anode", "cathode"],
            &[Branch::new(0, 1, BranchKind::Conductive)],
        )
    }

    fn prepare<R: Real>(
        parameters: &DiodeParameters<R>,
        temperature: R,
    ) -> Result<PreparedDiode<R>, ParameterError> {
        let flicker = Flicker::new(parameters.flicker_coefficient, parameters.flicker_exponent)?;
        let series_resistance =
            ParameterError::non_negative("series_resistance", parameters.series_resistance)?;
        let transit_time = ParameterError::non_negative("transit_time", parameters.transit_time)?;
        let junction = parameters.junction(temperature)?;
        Ok(PreparedDiode {
            series: series_value(series_resistance),
            junction,
            transit_time,
            temperature: temperature.value(),
            flicker,
        })
    }

    fn layout<R: Real>(
        parameters: &DiodeParameters<R>,
        _: &PreparedDiode<R>,
        layout: &mut Layout,
    ) -> Result<DiodeShape, ParameterError> {
        let series_resistance =
            ParameterError::non_negative("series_resistance", parameters.series_resistance)?;
        let (anode, cathode) = (0, 1);
        let (series, inner) = Series::behind(
            layout,
            anode,
            series_resistance,
            ("junction", "junction_current"),
        );
        let junction = Pair::new(layout, inner, cathode);
        Ok(DiodeShape { series, junction })
    }

    fn evaluate<R: Real>(
        shape: &DiodeShape,
        prepared: &PreparedDiode<R>,
        memory: &mut DiodeMemory,
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        shape.series.add(prepared.series, at, stamp);
        prepared.junction.add(
            &shape.junction,
            prepared.transit_time,
            &mut memory.last_voltage,
            at,
            stamp,
        );
    }

    /// The series resistance's thermal noise, and the junction current's shot and flicker
    /// noise.
    fn noise<R: Real>(
        shape: &DiodeShape,
        prepared: &PreparedDiode<R>,
        at: &Local<'_>,
        sources: &mut Vec<NoiseSource>,
    ) {
        let thermal = ("series thermal", prepared.temperature);
        shape.series.noise(prepared.series, thermal, sources);
        let (current, _) = prepared.junction.current(shape.junction.voltage(at));
        let nodes = shape.junction.equations();
        sources.push(NoiseSource::shot("shot", nodes, current.value()));
        prepared.flicker.add(nodes, current.value(), sources);
    }
}

impl<R: Real> DiodeParameters<R> {
    /// The diode's junction at its `temperature`, in kelvins, once each parameter it reads
    /// is checked.
    fn junction(&self, temperature: R) -> Result<Junction<R>, ParameterError> {
        let &Self {
            saturation_current,
            emission_coefficient,
            junction_capacitance,
            junction_potential,
            grading_coefficient,
            depletion_coefficient,
            breakdown_voltage,
            breakdown_current,
            breakdown_emission_coefficient,
            breakdown_temperature_coefficient,
            energy_gap,
            saturation_current_temperature_exponent,
            nominal_temperature,
            ..
        } = self;
        ParameterError::positive("saturation_current", saturation_current)?;
        ParameterError::positive("emission_coefficient", emission_coefficient)?;
        ParameterError::non_negative("junction_capacitance", junction_capacitance)?;
        ParameterError::positive("junction_potential", junction_potential)?;
        below_one("grading_coefficient", grading_coefficient)?;
        below_one("depletion_coefficient", depletion_coefficient)?;
        ParameterError::positive("breakdown_current", breakdown_current)?;
        let breakdown_emission = breakdown_emission_coefficient.unwrap_or(emission_coefficient);
        ParameterError::positive("breakdown_emission_coefficient", breakdown_emission)?;
        ParameterError::non_negative("energy_gap", energy_gap)?;

        let temperatures = Temperatures::of_junction(temperature, nominal_temperature)?;
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
        Ok(Junction::new(
            saturation_current,
            emission_coefficient * vt,
            Depletion::new(
                junction_capacitance,
                junction_potential,
                grading_coefficient,
                depletion_coefficient,
            ),
            breakdown,
        ))
    }
}

impl<R: Real> Junction<R> {
    /// A junction of `saturation_current` and exponential `slope`, the emission
    /// coefficient times the thermal voltage, which holds the `depletion` charge, if any,
    /// and breaks down as `breakdown` says, if at all.
    pub(crate) fn new(
        saturation_current: R,
        slope: R,
        depletion: Option<Depletion<R>>,
        breakdown: Option<Breakdown<R>>,
    ) -> Self {
        Self {
            saturation_current,
            slope,
            critical_voltage: critical_voltage(saturation_current.value(), slope.value()),
            depletion,
            breakdown,
        }
    }

    /// Adds the junction's current across `pair`, with gmin across it, and its depletion
    /// charge and the diffusion charge of `transit_time`, its voltage limited from `last`'s,
    /// which takes the voltage it is evaluated at, and returns the capacitance there.
    pub(crate) fn add(
        &self,
        pair: &Pair,
        transit_time: R,
        last: &mut f64,
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) -> R {
        let applied = pair.voltage(at);
        let voltage = at
            .limiting
            .then(|| self.limit(applied, *last))
            .flatten()
            .map_or(applied, |limited| {
                stamp.mark_limited();
                limited
            });
        *last = voltage;

        // Evaluate at the limited voltage and extend linearly to the applied one, so the
        // Newton step that follows solves the circuit linearized at the limited voltage.
        let (current, conductance) = self.current(voltage);
        let total_conductance = conductance + at.gmin;
        let total_current = current + at.gmin * voltage;
        pair.add_current(
            stamp,
            total_current + total_conductance * (applied - voltage),
        );
        pair.add_conductance(stamp, total_conductance);

        let (depletion_charge, depletion_capacitance) =
            depletion_charge(self.depletion.as_ref(), voltage);
        let charge = depletion_charge + transit_time * current;
        let capacitance = depletion_capacitance + transit_time * conductance;
        pair.add_charge(stamp, charge + capacitance * (applied - voltage));
        pair.add_capacitance(stamp, capacitance);
        capacitance
    }

    /// The junction's step to `applied` limited from the last evaluation's, `last`, as
    /// ngspice limits it: about the breakdown voltage when the junction nears it.
    fn limit(&self, applied: f64, last: f64) -> Option<f64> {
        match self.breakdown {
            Some(breakdown)
                if applied
                    < (10.0 * breakdown.slope.value() - breakdown.voltage.value()).min(0.0) =>
            {
                let (voltage, slope) = (breakdown.voltage.value(), breakdown.slope.value());
                limit_junction(
                    -(applied + voltage),
                    -(last + voltage),
                    slope,
                    self.critical_voltage,
                )
                .map(|limited| -(limited + voltage))
            }
            _ => limit_junction(applied, last, self.slope.value(), self.critical_voltage),
        }
    }

    /// The junction's current at `voltage`, and its derivative.
    pub(crate) fn current(&self, voltage: f64) -> (R, R) {
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

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::testing::{
        Bench, assert_close, assert_jacobians_match, assert_parameter_gradients_match, source,
        thermal,
    };
    use crate::{ELEMENTARY_CHARGE, thermal_voltage};

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

    #[test]
    fn its_current_makes_shot_and_flicker_noise() {
        let parameters = r#"{"saturation_current": 1e-14, "series_resistance": 10,
            "flicker_coefficient": 1e-12, "flicker_exponent": 1.5}"#;
        let mut bench = Bench::new::<Diode>(parameters);
        let x = [0.7, 0.0, 0.69];
        // The junction's current leaves through the cathode.
        let current = -bench.settle(&x).residual[1];
        let sources = bench.noise(&x);
        let shot = source(&sources, "shot");
        assert_close(&[shot.density(1.0)], &[2.0 * ELEMENTARY_CHARGE * current]);
        let flicker = source(&sources, "flicker");
        assert_eq!(flicker.equations, shot.equations);
        assert_close(
            &[flicker.density(100.0)],
            &[1e-12 * current.powf(1.5) / 100.0],
        );
        let series = source(&sources, "series thermal");
        assert_close(&[series.density(1.0)], &[thermal(0.1)]);
        assert_eq!(sources.len(), 3);
        let quiet = Bench::new::<Diode>(r#"{"saturation_current": 1e-14}"#).noise(&x[..2]);
        assert_eq!(quiet.len(), 1, "{quiet:?}");
    }
}
