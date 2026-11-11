use serde::Deserialize;

use crate::junction::{
    Depletion, Temperatures, critical_voltage, depletion_charge, junction_current, limit_gate,
    limit_junction,
};
use crate::stamp::Stamp;
use crate::{
    Binding, Branch, BranchKind, Device, Input, Interface, NativeDevice, NativeModel, Output,
    ParameterError, Quantity, REFERENCE_TEMPERATURE, Real,
};

/// The channel of a field-effect transistor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Channel {
    /// An n-channel, conducting with its gate above its threshold.
    N,
    /// A p-channel, the n-channel's mirror: every voltage and current reversed.
    P,
}

impl Channel {
    /// `1` for an n-channel, `−1` for a p-channel: the factor that orients voltages and
    /// currents as an n-channel's.
    #[must_use]
    pub const fn sign(self) -> f64 {
        match self {
            Self::N => 1.0,
            Self::P => -1.0,
        }
    }
}

/// A junction field-effect transistor, in the SPICE model with Sydney University's doping
/// tail, as ngspice implements it.
///
/// Two junctions join the gate to the channel, one at each end, each carrying a
/// [junction current](crate::junction) and holding a depletion charge of grading ½. The
/// drain current follows the square law, softened near the threshold by the doping tail
/// `b`: with `vgst = vgs − vto` and `β' = β · (1 + λ · vds)`, it is
/// `β' · vgst² · (b + (1 − b) · vgst / (pb − vto))` in saturation. The transistor is
/// symmetric: with the drain below the source, the two swap. A drain or source resistance
/// adds an internal node, labelled `drain` or `source`.
///
/// Away from its nominal temperature, its gate junctions' saturation current scales with
/// the energy gap and their potential and capacitances with silicon's band gap; its
/// threshold moves, and its transconductance parameter scales, by their own coefficients,
/// as ngspice scales them.
#[derive(Debug)]
pub struct Jfet<R: Real = f64> {
    stamp: Stamp<5>,
    sign: f64,
    /// The thermal voltage at the transistor's temperature.
    thermal_voltage: f64,
    drain_conductance: R,
    source_conductance: R,
    channel: DrainCurrent<R>,
    saturation_current: R,
    critical_voltage: f64,
    gate_source: Option<Depletion<R>>,
    gate_drain: Option<Depletion<R>>,
    /// The gate-source and gate-drain voltages of the last evaluation, from which the next
    /// steps are limited.
    last: [f64; 2],
}

/// Local node indices: the terminals, then the internal drain and source, which are the
/// terminals themselves without series resistance.
const GATE: usize = 0;
const DRAIN: usize = 1;
const SOURCE: usize = 2;
const INNER_DRAIN: usize = 3;
const INNER_SOURCE: usize = 4;

/// The parameters of a [`Jfet`], with SPICE's defaults.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields, bound = "")]
pub struct JfetParameters<R: Real = f64> {
    /// The channel; default `n`.
    pub channel: Channel,
    /// The threshold voltage, `vto`, in volts; default −2.
    pub threshold_voltage: R,
    /// The transconductance parameter, `β`, in amperes per square volt; default 1e-4.
    pub transconductance_parameter: R,
    /// The channel-length modulation, `λ`, per volt; default 0.
    pub channel_length_modulation: R,
    /// The drain resistance, in ohms; default 0.
    pub drain_resistance: R,
    /// The source resistance, in ohms; default 0.
    pub source_resistance: R,
    /// The zero-bias gate-source capacitance, in farads; default 0.
    pub gate_source_capacitance: R,
    /// The zero-bias gate-drain capacitance, in farads; default 0.
    pub gate_drain_capacitance: R,
    /// The gate junctions' potential, `pb`, in volts; default 1.
    pub junction_potential: R,
    /// The gate junctions' saturation current, in amperes; default 1e-14.
    pub saturation_current: R,
    /// The fraction of the junction potential above which the depletion capacitance is
    /// extended linearly, below 1; default 0.5.
    pub depletion_coefficient: R,
    /// The doping tail, `b`: 1 for the plain square law; default 1.
    pub doping_tail: R,
    /// How fast the threshold voltage rises with temperature, in volts per kelvin;
    /// default 0.
    pub threshold_temperature_coefficient: R,
    /// The exponent of the temperature ratio that scales the transconductance parameter;
    /// default 0.
    pub beta_temperature_exponent: R,
    /// The energy gap that scales the saturation current with temperature, in volts;
    /// default 1.11, silicon's.
    pub energy_gap: R,
    /// The exponent of the temperature ratio in the saturation current's scaling; default
    /// 0, as ngspice scales a JFET's by it only when it is given.
    pub saturation_current_temperature_exponent: R,
    /// The temperature the parameters were measured at, in kelvins; default 27 °C.
    pub nominal_temperature: R,
}

impl<R: Real> Default for JfetParameters<R> {
    fn default() -> Self {
        Self {
            channel: Channel::N,
            threshold_voltage: R::constant(-2.0),
            transconductance_parameter: R::constant(1e-4),
            channel_length_modulation: R::constant(0.0),
            drain_resistance: R::constant(0.0),
            source_resistance: R::constant(0.0),
            gate_source_capacitance: R::constant(0.0),
            gate_drain_capacitance: R::constant(0.0),
            junction_potential: R::constant(1.0),
            saturation_current: R::constant(1e-14),
            depletion_coefficient: R::constant(0.5),
            doping_tail: R::constant(1.0),
            threshold_temperature_coefficient: R::constant(0.0),
            beta_temperature_exponent: R::constant(0.0),
            energy_gap: R::constant(1.11),
            saturation_current_temperature_exponent: R::constant(0.0),
            nominal_temperature: R::constant(REFERENCE_TEMPERATURE),
        }
    }
}

/// The drain current of the channel, as a function of the gate-source and drain-source
/// voltages.
#[derive(Debug, Clone, Copy)]
struct DrainCurrent<R: Real> {
    beta: R,
    threshold: R,
    lambda: R,
    tail: R,
    /// `(1 − b) / (pb − vto)`.
    tail_factor: R,
}

impl<R: Real> DrainCurrent<R> {
    /// The current from drain to source, and its derivatives with respect to `vgs` at
    /// constant `vds`, and to `vds` at constant `vgs`.
    fn evaluate(&self, vgs: f64, vds: f64) -> (R, R, R) {
        let Self {
            beta,
            threshold,
            lambda,
            tail,
            tail_factor,
        } = *self;
        let zero = R::constant(0.0);
        if vds >= 0.0 {
            let vgst = -(threshold - vgs);
            if vgst.value() <= 0.0 {
                return (zero, zero, zero);
            }
            let betap = beta * (lambda * vds + 1.0);
            if vgst.value() >= vds {
                let a = tail * 2.0 + tail_factor * 3.0 * (vgst - vds);
                let c = ((tail_factor * vds - tail) * vds + vgst * a) * vds;
                let gm = betap * vds * (a + tail_factor * 3.0 * vgst);
                let gds = betap * (vgst - vds) * a + beta * lambda * c;
                (betap * c, gm, gds)
            } else {
                let factor = vgst * tail_factor;
                let c = vgst * vgst * (tail + factor);
                let gm = betap * vgst * (tail * 2.0 + factor * 3.0);
                (betap * c, gm, lambda * beta * c)
            }
        } else {
            // The drain and the source swap: the gate-drain voltage controls the channel.
            let vgdt = -(threshold - (vgs - vds));
            if vgdt.value() <= 0.0 {
                return (zero, zero, zero);
            }
            let betap = beta * -(lambda * vds - 1.0);
            if (vgdt + vds).value() >= 0.0 {
                let a = tail * 2.0 + tail_factor * 3.0 * (vgdt + vds);
                let c = ((-tail_factor * vds - tail) * -vds + vgdt * a) * vds;
                let gm = betap * vds * (a + tail_factor * 3.0 * vgdt);
                let gds = betap * (vgdt + vds) * a - beta * lambda * c - gm;
                (betap * c, gm, gds)
            } else {
                let factor = vgdt * tail_factor;
                let c = vgdt * vgdt * (tail + factor);
                let gm = -betap * vgdt * (tail * 2.0 + factor * 3.0);
                (-betap * c, gm, lambda * beta * c - gm)
            }
        }
    }
}

impl NativeModel for Jfet {
    type Device<R: Real> = Jfet<R>;
}

impl<R: Real> NativeDevice<R> for Jfet<R> {
    const NAME: &'static str = "jfet";
    type Parameters = JfetParameters<R>;

    fn interface(_: &JfetParameters<R>) -> Interface {
        Interface::new(
            &["drain", "gate", "source"],
            &[
                Branch::new(0, 2, BranchKind::Conductive),
                Branch::new(1, 0, BranchKind::Conductive),
                Branch::new(1, 2, BranchKind::Conductive),
            ],
        )
    }

    fn new(
        parameters: JfetParameters<R>,
        binding: &mut Binding<'_>,
    ) -> Result<Self, ParameterError> {
        let JfetParameters {
            channel,
            threshold_voltage,
            transconductance_parameter,
            channel_length_modulation,
            drain_resistance,
            source_resistance,
            gate_source_capacitance,
            gate_drain_capacitance,
            junction_potential,
            saturation_current,
            depletion_coefficient,
            doping_tail,
            threshold_temperature_coefficient,
            beta_temperature_exponent,
            energy_gap,
            saturation_current_temperature_exponent,
            nominal_temperature,
        } = parameters;
        ParameterError::non_negative("transconductance_parameter", transconductance_parameter)?;
        ParameterError::non_negative("drain_resistance", drain_resistance)?;
        ParameterError::non_negative("source_resistance", source_resistance)?;
        ParameterError::non_negative("gate_source_capacitance", gate_source_capacitance)?;
        ParameterError::non_negative("gate_drain_capacitance", gate_drain_capacitance)?;
        ParameterError::positive("junction_potential", junction_potential)?;
        ParameterError::positive("saturation_current", saturation_current)?;
        crate::junction::below_one("depletion_coefficient", depletion_coefficient)?;
        ParameterError::non_negative("energy_gap", energy_gap)?;
        // The doping tail's factor, `(1 − b) / (pb − vto)`, which vanishes at `b = 1`.
        let (tail_excess, gap) = (-(doping_tail - 1.0), junction_potential - threshold_voltage);
        if gap.value() == 0.0 && tail_excess.value() != 0.0 {
            return Err(ParameterError(
                "a doping tail other than 1 needs a junction potential other than the \
                 threshold voltage"
                    .to_owned(),
            ));
        }

        let (drain, gate, source) = (
            binding.terminal(0),
            binding.terminal(1),
            binding.terminal(2),
        );
        let inner_drain = if drain_resistance.value() > 0.0 {
            Some(binding.internal("drain", Quantity::Voltage))
        } else {
            drain
        };
        let inner_source = if source_resistance.value() > 0.0 {
            Some(binding.internal("source", Quantity::Voltage))
        } else {
            source
        };
        let stamp = Stamp::new(binding, [gate, drain, source, inner_drain, inner_source]);
        let temperatures = Temperatures::new(binding.temperature(), nominal_temperature)?;
        let vt = temperatures.thermal_voltage();
        let saturation_current = saturation_current
            * temperatures.saturation(
                energy_gap,
                saturation_current_temperature_exponent,
                R::constant(1.0),
            );
        let depletion = |capacitance| {
            let (potential, capacitance) =
                temperatures.junction(junction_potential, capacitance, R::constant(0.5));
            Depletion::new(
                capacitance,
                potential,
                R::constant(0.5),
                depletion_coefficient,
            )
        };
        Ok(Self {
            stamp,
            sign: channel.sign(),
            thermal_voltage: vt,
            drain_conductance: series_conductance(drain_resistance),
            source_conductance: series_conductance(source_resistance),
            channel: DrainCurrent {
                beta: transconductance_parameter
                    * temperatures.ratio().pow(beta_temperature_exponent),
                threshold: threshold_voltage
                    + threshold_temperature_coefficient * temperatures.difference(),
                lambda: channel_length_modulation,
                tail: doping_tail,
                tail_factor: if tail_excess.value() == 0.0 {
                    R::constant(0.0)
                } else {
                    tail_excess / gap
                },
            },
            saturation_current,
            critical_voltage: critical_voltage(saturation_current.value(), vt),
            gate_source: depletion(gate_source_capacitance),
            gate_drain: depletion(gate_drain_capacitance),
            last: [0.0; 2],
        })
    }
}

/// The conductance of a series `resistance`, or zero without one.
fn series_conductance<R: Real>(resistance: R) -> R {
    if resistance.value() > 0.0 {
        resistance.recip()
    } else {
        R::constant(0.0)
    }
}

impl<R: Real> Device<R> for Jfet<R> {
    fn evaluate(&mut self, input: &Input<'_>, output: &mut Output<'_, R>) {
        let stamp = &self.stamp;
        for (outer, inner, conductance) in [
            (DRAIN, INNER_DRAIN, self.drain_conductance),
            (SOURCE, INNER_SOURCE, self.source_conductance),
        ] {
            if conductance.value() > 0.0 {
                let voltage = stamp.voltage(input, outer) - stamp.voltage(input, inner);
                stamp.current(output, outer, inner, conductance * voltage);
                stamp.conductance(output, (outer, inner), (outer, inner), conductance);
            }
        }

        let sign = self.sign;
        let gate = stamp.voltage(input, GATE);
        let applied = [
            sign * (gate - stamp.voltage(input, INNER_SOURCE)),
            sign * (gate - stamp.voltage(input, INNER_DRAIN)),
        ];
        let limits = [0, 1].map(|index| {
            let (new, old) = (applied[index], self.last[index]);
            let junction = limit_junction(new, old, self.thermal_voltage, self.critical_voltage);
            let limited = limit_gate(junction.unwrap_or(new), old, self.channel.threshold.value());
            limited.or(junction).filter(|_| input.limiting)
        });
        if limits.iter().any(Option::is_some) {
            output.mark_limited();
        }
        let limited = [0, 1].map(|index| limits[index].unwrap_or(applied[index]));
        self.last = limited;
        let [vgs, vgd] = limited;
        // How far the applied voltages lie beyond the limited ones.
        let beyond = [applied[0] - vgs, applied[1] - vgd];

        // Each gate junction, evaluated at its limited voltage and extended linearly.
        for (inner, voltage, delta, depletion) in [
            (INNER_SOURCE, vgs, beyond[0], self.gate_source.as_ref()),
            (INNER_DRAIN, vgd, beyond[1], self.gate_drain.as_ref()),
        ] {
            let (current, conductance) = junction_current(
                voltage,
                self.saturation_current,
                R::constant(self.thermal_voltage),
            );
            let conductance = conductance + input.gmin;
            let current = current + input.gmin * voltage + conductance * delta;
            stamp.current(output, GATE, inner, current * sign);
            stamp.conductance(output, (GATE, inner), (GATE, inner), conductance);
            let (charge, capacitance) = depletion_charge(depletion, voltage);
            stamp.charge(output, GATE, inner, (charge + capacitance * delta) * sign);
            stamp.capacitance(output, (GATE, inner), (GATE, inner), capacitance);
        }

        let (current, gm, gds) = self.channel.evaluate(vgs, vgs - vgd);
        let current = current + gm * beyond[0] + gds * (beyond[0] - beyond[1]);
        let channel = (INNER_DRAIN, INNER_SOURCE);
        stamp.current(output, INNER_DRAIN, INNER_SOURCE, current * sign);
        stamp.conductance(output, channel, (GATE, INNER_SOURCE), gm);
        stamp.conductance(output, channel, channel, gds);
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::testing::{Bench, assert_jacobians_match, assert_parameter_gradients_match};

    const CHARGED: &str = r#"{"gate_source_capacitance": 2e-12, "gate_drain_capacitance": 1e-12,
        "channel_length_modulation": 0.02, "doping_tail": 0.8, "transconductance_parameter": 1e-3}"#;

    #[test]
    fn parameter_gradients_are_exact() {
        let every = r#"{"threshold_voltage": -1.8, "transconductance_parameter": 1e-3,
            "channel_length_modulation": 0.02, "drain_resistance": 10,
            "source_resistance": 5, "gate_source_capacitance": 2e-12,
            "gate_drain_capacitance": 1e-12, "junction_potential": 0.9,
            "saturation_current": 2e-14, "depletion_coefficient": 0.4, "doping_tail": 0.8,
            "threshold_temperature_coefficient": -2e-3, "beta_temperature_exponent": -1.5,
            "energy_gap": 1.2, "saturation_current_temperature_exponent": 2,
            "nominal_temperature": 290}"#;
        for [drain, gate] in [[3.0, 0.0], [0.5, -0.5], [-0.5, -0.2], [-3.0, 0.1]] {
            assert_parameter_gradients_match::<Jfet>(
                every,
                &[drain, gate, 0.0, drain - 0.01, 0.005],
            );
        }
    }

    #[test]
    fn follows_the_square_law_in_saturation() {
        let mut bench = Bench::new::<Jfet>(r#"{"transconductance_parameter": 1e-3}"#);
        let evaluation = bench.settle(&[5.0, 0.0, 0.0]);
        // vgs = 0, vto = −2: β · vgst² = 1e-3 · 4.
        let drain_current = evaluation.residual[0];
        assert!((drain_current - 4e-3).abs() < 1e-12, "{drain_current}");
    }

    #[test]
    fn a_p_channel_mirrors_an_n_channel() {
        let mut n = Bench::new::<Jfet>("{}");
        let mut p = Bench::new::<Jfet>(r#"{"channel": "p"}"#);
        let n_currents = n.settle(&[3.0, -0.5, 0.0]).residual;
        let p_currents = p.settle(&[-3.0, 0.5, 0.0]).residual;
        for (n_current, p_current) in n_currents.iter().zip(&p_currents) {
            assert!((n_current + p_current).abs() < 1e-15);
        }
    }

    #[test]
    fn series_resistances_add_internal_nodes() {
        let bench = Bench::new::<Jfet>(r#"{"drain_resistance": 10, "source_resistance": 5}"#);
        assert_eq!(
            bench.unknowns(),
            ["n0", "n1", "n2", "device:drain", "device:source"]
        );
    }

    proptest! {
        /// The conductances and capacitances are the derivatives of the currents and
        /// charges, in every region and in both modes.
        #[test]
        fn jacobians_match_finite_differences(
            drain in -4.0..4.0f64,
            gate in -3.0..0.6f64,
            p in proptest::bool::ANY,
        ) {
            let parameters = if p {
                CHARGED.replace('{', r#"{"channel": "p", "#)
            } else {
                CHARGED.to_owned()
            };
            let sign = if p { -1.0 } else { 1.0 };
            assert_jacobians_match::<Jfet>(&parameters, &[sign * drain, sign * gate, 0.0]);
            let resistive = parameters.replace('{', r#"{"drain_resistance": 50, "source_resistance": 20, "#);
            assert_jacobians_match::<Jfet>(
                &resistive,
                &[sign * drain, sign * gate, 0.0, sign * 0.9 * drain, sign * 0.01],
            );
        }
    }
}
